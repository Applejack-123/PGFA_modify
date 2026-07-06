# main2_causal_side_train.py
# -*- coding: utf-8 -*-
"""
Training script for the causal side-branch version of Light MA-Mamba Fusion.

Usage:
    1. Put this file under your project root, e.g.:
       /root/autodl-tmp/Github/PGFA_modify/main2_causal_side_train.py

    2. Make sure these two files are also replaced by the new versions:
       - Causal.py
       - align_mamba2_fusion.py

    3. Run:
       python main2_causal_side_train.py

Training strategy:
    - Load the old best fusion checkpoint, e.g. 0.8661 checkpoint.
    - Keep encoder / text projection / fusion backbone frozen.
    - Train only:
        self.fusion.causal
        self.fusion.causal_mix
    - Use a small AdamW learning rate.
    - Do not call the original large-lr warmup scheduler.
    - Add preserve loss to prevent the causal branch from damaging the old base feature.
"""

from config import *
from dataset import DataSet
from logger import Log

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import pi, cos
from tqdm import tqdm

from module.gcn.st_gcnV2 import Model
from module.shift_gcn import Model as ShiftGCN
from module.adapter import Adapter, Linear
from KLLoss import KLLoss, KDLoss
from tool import (
    gen_label,
    create_logits,
    get_acc,
    create_sim_matrix,
    gen_label_from_text_sim,
    get_m_theta,
    get_acc_v2,
)

from cross_mamba import MambaFusion
from module.skeleton_mamba_encoder import SkeletonMambaEncoder
from align_mamba2_fusion import AlignMamba2Fusion
from rgb_only_mlp import RGBOnlyModule as RGBModule


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)


class Processor:

    @ex.capture
    def load_data(
        self,
        train_list,
        train_label,
        test_list,
        test_label,
        train_rgb,
        test_rgb,
        batch_size,
        language_path,
    ):
        self.dataset = dict()
        self.data_loader = dict()
        self.best_epoch = -1
        self.best_acc = -1
        self.dim_loss = -1
        self.test_acc = -1
        self.test_aug_acc = -1
        self.best_aug_acc = -1
        self.best_aug_epoch = -1

        self.full_language = np.load(language_path)
        self.full_language = torch.Tensor(self.full_language).cuda()

        self.dataset["train"] = DataSet(train_list, train_label, train_rgb)
        self.dataset["test"] = DataSet(test_list, test_label, test_rgb)

        self.data_loader["train"] = torch.utils.data.DataLoader(
            dataset=self.dataset["train"],
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            drop_last=True,
        )

        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=self.dataset["test"],
            batch_size=64,
            num_workers=16,
            shuffle=False,
        )

    def load_weights(self, model=None, weight_path=None):
        checkpoint = torch.load(weight_path, map_location="cpu")

        if model is self.encoder and "encoder" in checkpoint:
            model.load_state_dict(checkpoint["encoder"])
            print("Loaded encoder.")

        elif model is self.adapter and "adapter" in checkpoint:
            model.load_state_dict(checkpoint["adapter"])
            print("Loaded adapter.")

        elif model is self.proj and "proj" in checkpoint:
            model.load_state_dict(checkpoint["proj"])
            print("Loaded text projection.")
            
        elif model is self.rgb_mlp and "rgb_mlp" in checkpoint:
            model.load_state_dict(checkpoint["rgb_mlp"])
            print("Loaded text projection.")
            
        elif model is self.fusion and "fusion" in checkpoint:
            # Old checkpoint does not contain causal side-branch parameters.
            # strict=False is required.
            missing, unexpected = model.load_state_dict(checkpoint["fusion"], strict=False)

            allowed_missing_prefix = (
                "causal.",
            )
            allowed_missing_exact = {
                "causal_mix",
            }

            bad_missing = [
                k for k in missing
                if not (k.startswith(allowed_missing_prefix) or k in allowed_missing_exact)
            ]

            if len(bad_missing) > 0:
                raise RuntimeError(
                    "Fusion checkpoint is missing non-causal keys. "
                    f"Bad missing keys: {bad_missing}"
                )

            if len(unexpected) > 0:
                raise RuntimeError(
                    f"Fusion checkpoint has unexpected keys: {unexpected}"
                )

            print("Loaded fusion with strict=False.")
            print("Newly initialized keys:", missing)

        else:
            raise Exception("cannot found the weight Error!")

    def adjust_learning_rate(
        self,
        optimizer,
        current_epoch,
        max_epoch,
        lr_min=0,
        lr_max=0.1,
        warmup_epoch=15,
        loss_mode="cos",
        step=[20, 30],
    ):
        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        elif loss_mode == "cos":
            lr = lr_min + (lr_max - lr_min) * (
                1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))
            ) / 2
        elif loss_mode == "step":
            lr = lr_max * (0.1 ** np.sum(current_epoch >= np.array(step)))
        else:
            raise Exception("Please check loss_mode!")

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def layernorm(self, feature):
        num = feature.shape[0]
        mean = torch.mean(feature, dim=1).reshape(num, -1)
        var = torch.var(feature, dim=1).reshape(num, -1)
        out = (feature - mean) / torch.sqrt(var)
        return out

    @ex.capture
    def load_model(
        self,
        in_channels,
        hidden_channels,
        hidden_dim,
        dropout,
        graph_args,
        edge_importance_weighting,
        visual_size,
        language_size,
        weight_path,
        loss_type,
        fix_encoder,
        finetune,
    ):
        self.encoder = SkeletonMambaEncoder(
            in_channels=3,
            num_point=25,
            num_person=2,
            embed_dim=hidden_dim,
            temporal_depth=2,
            dropout=0.1,
        ).cuda()

        self.adapter = Linear().cuda()

        self.proj = nn.Sequential(
            nn.LayerNorm(language_size),
            nn.Linear(language_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        ).cuda()

        self.fusion = AlignMamba2Fusion(
            skel_dim=512,
            rgb_dim=512,
            text_dim=512,
            dim=512,
            num_classes=None,
            unimodal_depth=1,
            fusion_depth=2,
            lambda_ot=0.001,
            lambda_mmd=0.01,
            use_text_in_fusion=False,
            use_causal=True,
            causal_mix_init=-6.0,
        ).cuda()
        
        self.rgb_mlp = RGBModule(
            dim=512,
            hidden_dim=1024,
            depth=2,
            temporal_depth=1,
            dropout=0.1,
            use_temporal=True,
            pool="mean",   # also "attn"
        ).cuda()

        # Fixed Python logic bug: `or "kl+margin"` is always True.
        if loss_type in ["kl", "klv2", "kl+cosface", "kl+sphereface", "kl+margin"]:
            self.loss = KLLoss().cuda()
        elif loss_type == "mse":
            self.loss = nn.MSELoss().cuda()
        elif loss_type == "kl+mse":
            self.loss_kl = KLLoss().cuda()
            self.loss_mse = nn.MSELoss().cuda()
        elif loss_type == "kl+kd":
            self.loss = KLLoss().cuda()
            self.kd_loss = KDLoss().cuda()
        else:
            raise Exception("loss_type Error!")

        self.logit_scale = self.adapter.get_logit_scale()
        self.logit_scale_v2 = self.adapter.get_logit_scale_v2()

        # For causal side-branch training, you should load old best fusion.
        # This is critical. Otherwise you freeze a random fusion backbone.
        if fix_encoder or finetune:
            self.load_weights(self.encoder, weight_path)
            self.load_weights(self.proj, weight_path)
            self.load_weights(self.rgb_mlp, weight_path)
            self.load_weights(self.fusion, weight_path)
            
        # Freeze encoder/proj permanently in this stage.
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.proj.parameters():
            p.requires_grad_(False)
        for p in self.adapter.parameters():
            p.requires_grad_(False)
        for p in self.rgb_mlp.parameters():
            p.requires_grad_(False)

    def freeze_fusion_except_causal_side(self):
        """
        Train only:
            - fusion.causal.*
            - fusion.causal_mix

        Keep all old fusion backbone parameters frozen.
        """
        for name, param in self.fusion.named_parameters():
            param.requires_grad_(False)

        if self.fusion.causal is not None:
            for name, param in self.fusion.causal.named_parameters():
                param.requires_grad_(True)

        if hasattr(self.fusion, "causal_mix"):
            self.fusion.causal_mix.requires_grad_(True)

        print("Trainable parameters:")
        total_trainable = 0
        for name, param in self.fusion.named_parameters():
            if param.requires_grad:
                n = param.numel()
                total_trainable += n
                print(f"  {name}: {tuple(param.shape)} / {n}")
        print("Total trainable fusion params:", total_trainable)

    @ex.capture
    def load_optim(
        self,
        lr,
        epoch_num,
        weight_decay,
        causal_lr=5e-5,
        mix_lr=5e-4,
    ):
        self.freeze_fusion_except_causal_side()

        causal_params = []
        mix_params = []

        for name, param in self.fusion.named_parameters():
            if not param.requires_grad:
                continue
            if name == "causal_mix":
                mix_params.append(param)
            elif name.startswith("causal."):
                causal_params.append(param)

        param_groups = []
        if len(causal_params) > 0:
            param_groups.append({"params": causal_params, "lr": causal_lr, "weight_decay": 1e-4})
        if len(mix_params) > 0:
            param_groups.append({"params": mix_params, "lr": mix_lr, "weight_decay": 0.0})

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
        )

        print(f"Optimizer: AdamW, causal_lr={causal_lr}, mix_lr={mix_lr}")

    @ex.capture
    def optimize(self, epoch_num, DA):
        self.log.info("causal side-branch train track")

        # First evaluate before training.
        # If loading is correct, this should be close to your old 0.8661 result.
        with torch.no_grad():
            self.test_epoch(epoch=-1)
        self.log.info("before training test acc: {}".format(self.test_acc))

        for epoch in range(epoch_num):
            self.train_epoch(epoch)

            with torch.no_grad():
                self.test_epoch(epoch=epoch)

            self.log.info("epoch [{}] train loss: {}".format(epoch, self.dim_loss))
            self.log.info("epoch [{}] test acc: {}".format(epoch, self.test_acc))
            self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch, self.best_acc))

            if hasattr(self.fusion, "causal_mix"):
                mix = torch.sigmoid(self.fusion.causal_mix).detach().cpu().item()
                self.log.info("epoch [{}] causal_mix sigmoid: {:.8f}".format(epoch, mix))

            if self.fusion.causal is not None:
                ca = torch.tanh(self.fusion.causal.alpha).detach().cpu().item()
                cs = torch.tanh(self.fusion.causal.conf_scale).detach().cpu().item()
                self.log.info("epoch [{}] causal alpha: {:.8f}, conf_scale: {:.8f}".format(epoch, ca, cs))

            if DA:
                self.log.info("epoch [{}] DA test acc: {}".format(epoch, self.test_aug_acc))
                self.log.info(
                    "epoch [{}] gets the best DA acc: {}".format(
                        self.best_aug_epoch,
                        self.best_aug_acc,
                    )
                )

    @ex.capture
    def train_epoch(
        self,
        epoch,
        lr,
        loss_mode,
        step,
        loss_type,
        alpha,
        beta,
        m,
        fix_encoder,
        preserve_weight=0.5,
    ):
        # Important:
        # Do not call self.fusion.train(), otherwise dropout in the frozen fusion backbone
        # will change the pretrained 0.8661 path. Keep the whole fusion in eval mode,
        # then switch only the causal module to train mode.
        self.encoder.eval()
        self.proj.eval()
        self.adapter.eval()
        self.rgb_mlp.eval()
        self.fusion.eval()
        if self.fusion.causal is not None:
            self.fusion.causal.train()

        # Important:
        # Do NOT call adjust_learning_rate here.
        # The original scheduler is for SGD with large lr and will overwrite AdamW small lr.
        # self.adjust_learning_rate(...)

        running_loss = []
        running_cls_loss = []
        running_preserve_loss = []

        loader = self.data_loader["train"]

        for data, label, rgb in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label_g = gen_label(label)
            label = label.type(torch.LongTensor).cuda()
            rgb_feat = rgb.type(torch.FloatTensor).cuda()

            # Encoder/proj are frozen. Use no_grad to save memory and keep stable features.
            with torch.no_grad():
                seen_language = self.full_language[label]
                seen_language_512 = self.proj(seen_language)
                rgb_feat = self.rgb_mlp(rgb_feat)
                skeleton_feat = self.encoder(data)

            out = self.fusion(
                skeleton_feat,
                rgb_feat,
                seen_language_512,
                labels=label,
            )

            fusion_feature = out["fusion_feature"]
            base_feature = out["base_feature"]

            if loss_type == "kl":
                feature = fusion_feature
                logits_per_skl, logits_per_text = create_logits(
                    feature,
                    seen_language_512,
                    self.logit_scale,
                    exp=True,
                )
                ground_truth = torch.tensor(label_g, dtype=feature.dtype).cuda()

                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                cls_loss = (loss_skls + loss_texts) / 2

                # Prevent the causal side branch from moving too far away from the old best feature.
                preserve_loss = 1.0 - F.cosine_similarity(
                    F.normalize(fusion_feature, dim=-1),
                    F.normalize(base_feature, dim=-1),
                    dim=-1,
                ).mean()

                loss = cls_loss + preserve_weight * preserve_loss
            else:
                raise NotImplementedError(
                    "This causal side-branch training script currently supports loss_type='kl'."
                )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.fusion.parameters(), max_norm=5.0)
            self.optimizer.step()

            running_loss.append(loss.detach())
            running_cls_loss.append(cls_loss.detach())
            running_preserve_loss.append(preserve_loss.detach())

        running_loss = torch.stack(running_loss)
        running_cls_loss = torch.stack(running_cls_loss)
        running_preserve_loss = torch.stack(running_preserve_loss)

        self.dim_loss = running_loss.mean().item()

        print(
            "Train loss detail | "
            f"total={running_loss.mean().item():.6f}, "
            f"cls={running_cls_loss.mean().item():.6f}, "
            f"preserve={running_preserve_loss.mean().item():.8f}"
        )

    @ex.capture
    def test_epoch(self, unseen_label, epoch, DA, support_factor):
        self.encoder.eval()
        self.proj.eval()
        self.adapter.eval()
        self.rgb_mlp.eval()
        self.fusion.eval()

        loader = self.data_loader["test"]
        acc_list = []

        unseen_language = self.full_language[unseen_label]
        unseen_language_512 = self.proj(unseen_language)

        for data, label, rgb in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            rgb_feat = rgb.type(torch.FloatTensor).cuda()
            rgb_feat = self.rgb_mlp(rgb_feat)
            skeleton_feat = self.encoder(data)

            out = self.fusion(
                skeleton_feat,
                rgb_feat,
                text_feat=None,
                labels=None,
                return_align_loss=False,
            )
            fusion_feature = out["fusion_feature"]

            acc_batch, pred = get_acc(
                fusion_feature,
                unseen_language_512,
                unseen_label,
                label,
            )
            acc_list.append(acc_batch)

        acc_list = torch.tensor(acc_list)
        acc = acc_list.mean()

        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            self.save_model()

        self.test_acc = acc

    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    @ex.capture
    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "proj": self.proj.state_dict(),
                "rgb_mlp": self.rgb_mlp.state_dict(),
                "fusion": self.fusion.state_dict(),

            },
            save_path,
        )

    def start(self):
        self.initialize()
        self.optimize()


@ex.automain
def main(track):
    p = Processor()
    p.start()
