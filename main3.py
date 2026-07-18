# main.py

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
from module.skeleton_guided_rgb_causal_debias import SkeletonGuidedRGBCausalDebias


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)


class CMoBBatchTextLoss(nn.Module):
    """
    CMoB-style modality contribution loss for your current zero-shot setting.

    Your training style:
        fusion_feature: [B,T,D] or [B,D]
        seen_language:  [B,D]
        label:          [B]

    Instead of using class-level classifier logits [B,C],
    this version uses batch-wise text matching logits [B,B].

    It computes:
        full logits      = fusion(skl, rgb) vs text
        skeleton logits  = fusion(skl, zero_rgb) vs text
        rgb logits       = fusion(zero_skl, rgb) vs text

    Then estimates sample-level Shapley-style contribution:
        contrib_skl = (value_skl + value_all - value_rgb) / 2
        contrib_rgb = (value_rgb + value_all - value_skl) / 2

    If skeleton is weaker, it strengthens skeleton-only branch.
    If RGB is weaker, it strengthens RGB-only branch.
    """

    def __init__(
        self,
        value_mode="prob",
        margin=0.0,
        base_aux_weight=0.10,
        weak_weight_scale=2.0,
        exp_logit_scale=True,
        max_logit_scale=100.0,
        eps=1e-6,
    ):
        super().__init__()
        assert value_mode in ["prob", "hard"]

        self.value_mode = value_mode
        self.margin = margin
        self.base_aux_weight = base_aux_weight
        self.weak_weight_scale = weak_weight_scale
        self.exp_logit_scale = exp_logit_scale
        self.max_logit_scale = max_logit_scale
        self.eps = eps

    def _pool_feature(self, x):
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 2:
            return x
        raise ValueError(
            "Expected feature shape [B,D] or [B,T,D], but got {}".format(x.shape)
        )

    def _get_scale(self, logit_scale):
        if isinstance(logit_scale, torch.Tensor):
            scale = logit_scale
            if self.exp_logit_scale:
                scale = scale.exp()
            scale = torch.clamp(scale, max=self.max_logit_scale)
            return scale

        scale = torch.tensor(float(logit_scale)).cuda()
        if self.exp_logit_scale:
            scale = scale.exp()
        scale = torch.clamp(scale, max=self.max_logit_scale)
        return scale

    def compute_pair_logits(self, feature, text_feature, logit_scale):
        """
        feature:      [B,D] or [B,T,D]
        text_feature: [B,D]

        return:
            logits: [B,B]
        """

        feature = self._pool_feature(feature)
        text_feature = self._pool_feature(text_feature)

        feature = F.normalize(feature, dim=-1)
        text_feature = F.normalize(text_feature, dim=-1)

        scale = self._get_scale(logit_scale)

        logits = scale * feature @ text_feature.t()
        return logits

    def _target_mask(self, labels):
        """
        labels: [B]

        return:
            raw_mask:  [B,B], same-class positions are 1
            soft_mask: [B,B], same-class positions are normalized
        """

        labels = labels.view(-1)
        raw_mask = (labels.view(-1, 1) == labels.view(1, -1)).float()
        soft_mask = raw_mask / raw_mask.sum(dim=1, keepdim=True).clamp_min(self.eps)

        return raw_mask, soft_mask

    def _soft_ce_each(self, logits, soft_target):
        """
        logits:      [B,B]
        soft_target: [B,B]

        return:
            loss_each: [B]
        """

        log_prob = F.log_softmax(logits, dim=1)
        loss_each = -(soft_target * log_prob).sum(dim=1)
        return loss_each

    def _benefit_value(self, logits, labels, raw_mask, modality_weight):
        """
        value_mode="prob":
            use probability mass assigned to all same-class text embeddings.

        value_mode="hard":
            use whether the top-1 predicted text belongs to the same class.
        """

        if self.value_mode == "hard":
            pred_index = torch.argmax(logits, dim=1)
            pred_label = labels[pred_index]
            correct = (pred_label == labels).float()
            return modality_weight * correct

        prob = F.softmax(logits, dim=1)
        same_class_prob = (prob * raw_mask).sum(dim=1)
        return modality_weight * same_class_prob

    def forward(
        self,
        fusion_feature_all,
        fusion_feature_skl_only,
        fusion_feature_rgb_only,
        text_feature,
        labels,
        logit_scale,
    ):
        labels = labels.long()

        logits_all = self.compute_pair_logits(
            fusion_feature_all,
            text_feature,
            logit_scale,
        )

        logits_skl = self.compute_pair_logits(
            fusion_feature_skl_only,
            text_feature,
            logit_scale,
        )

        logits_rgb = self.compute_pair_logits(
            fusion_feature_rgb_only,
            text_feature,
            logit_scale,
        )

        raw_mask, soft_mask = self._target_mask(labels)

        value_all = self._benefit_value(
            logits_all,
            labels,
            raw_mask,
            modality_weight=2.0,
        )

        value_skl = self._benefit_value(
            logits_skl,
            labels,
            raw_mask,
            modality_weight=1.0,
        )

        value_rgb = self._benefit_value(
            logits_rgb,
            labels,
            raw_mask,
            modality_weight=1.0,
        )

        contrib_skl = (value_skl + value_all - value_rgb) / 2.0
        contrib_rgb = (value_rgb + value_all - value_skl) / 2.0

        weak_skl_weight = F.relu(contrib_rgb - contrib_skl + self.margin)
        weak_rgb_weight = F.relu(contrib_skl - contrib_rgb + self.margin)

        weak_skl_weight = weak_skl_weight.detach()
        weak_rgb_weight = weak_rgb_weight.detach()

        loss_skl_each = self._soft_ce_each(logits_skl, soft_mask)
        loss_rgb_each = self._soft_ce_each(logits_rgb, soft_mask)

        weight_skl = self.base_aux_weight + self.weak_weight_scale * weak_skl_weight
        weight_rgb = self.base_aux_weight + self.weak_weight_scale * weak_rgb_weight

        loss_cmob = (
            weight_skl * loss_skl_each
            + weight_rgb * loss_rgb_each
        ).mean()

        return {
            "loss_cmob": loss_cmob,
            "logits_all": logits_all,
            "logits_skl": logits_skl,
            "logits_rgb": logits_rgb,
            "contrib_skl": contrib_skl.detach(),
            "contrib_rgb": contrib_rgb.detach(),
            "avg_contrib_skl": contrib_skl.detach().mean(),
            "avg_contrib_rgb": contrib_rgb.detach().mean(),
            "weak_skl_weight": weak_skl_weight.detach().mean(),
            "weak_rgb_weight": weak_rgb_weight.detach().mean(),
        }


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

        self.cmob_loss_value = 0.0
        self.cmob_contrib_skl = 0.0
        self.cmob_contrib_rgb = 0.0
        self.cmob_weak_skl = 0.0
        self.cmob_weak_rgb = 0.0

        self.full_language = np.load(language_path)
        self.full_language = torch.Tensor(self.full_language)
        self.full_language = self.full_language.cuda()

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
        checkpoint = torch.load(weight_path)

        if model is self.encoder and "encoder" in checkpoint:
            model.load_state_dict(checkpoint["encoder"])

        elif model is self.proj and "proj" in checkpoint:
            model.load_state_dict(checkpoint["proj"])

        elif model is self.rgb_mlp and "rgb_mlp" in checkpoint:
            model.load_state_dict(checkpoint["rgb_mlp"])

        elif model is self.fusion and "fusion" in checkpoint:
            missing, unexpected = model.load_state_dict(
                checkpoint["fusion"],
                strict=False,
            )
            print("fusion missing keys:", missing)
            print("fusion unexpected keys:", unexpected)

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
            lr = lr_max * (current_epoch+1) / warmup_epoch

        elif loss_mode == "cos":
            lr = lr_min + (lr_max - lr_min) * (
                1
                + cos(
                    pi
                    * (current_epoch - warmup_epoch)
                    / (max_epoch - warmup_epoch)
                )
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
    ):

        self.encoder = SkeletonMambaEncoder(
            in_channels=3,
            num_point=25,
            num_person=2,
            embed_dim=hidden_dim,
            temporal_depth=3,
            dropout=0.1,
        ).cuda()

        self.proj = nn.Sequential(
            nn.LayerNorm(language_size),
            nn.Linear(language_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        ).cuda()

        self.rgb_mlp = RGBModule(
            dim=512,
            hidden_dim=1024,
            depth=2,
            temporal_depth=1,
            dropout=0.1,
            use_temporal=True,
            pool="mean",
        ).cuda()

        self.rgb_causal_debias = SkeletonGuidedRGBCausalDebias(
            dim=512,
            hidden_dim=1024,
            dropout=0.1,
            use_temporal_conv=True,
            fusion_mode="gate",
            detach_skeleton_anchor=False,
        ).cuda()

        self.fusion = AlignMamba2Fusion(
            skel_dim=512,
            rgb_dim=512,
            text_dim=512,
            dim=512,
            num_classes=None,
            unimodal_depth=1,
            fusion_depth=3,
            lambda_ot=0.001,
            lambda_mmd=0.01,
            use_text_in_fusion=False,
            use_causal=False,
        ).cuda()

        self.cmob = CMoBBatchTextLoss(
            value_mode="prob",
            margin=0.0,
            base_aux_weight=0.10,
            weak_weight_scale=2.0,
            exp_logit_scale=True,
            max_logit_scale=100.0,
        ).cuda()

        self.use_cmob = True
        self.lambda_cmob = 0.05
        self.cmob_start_epoch = 0

        if loss_type in [
            "kl",
            "klv2",
            "kl+cosface",
            "kl+sphereface",
            "kl+margin",
        ]:
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

        self.logit_scale = self.fusion.logit_scale

        if fix_encoder:
            self.load_weights(self.encoder, weight_path)
            self.load_weights(self.proj, weight_path)
            self.load_weights(self.rgb_mlp, weight_path)
            self.load_weights(self.fusion, weight_path)

    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):

        if hasattr(self.fusion, "causal") and self.fusion.causal is not None:
            optim_params = [
                {
                    "params": self.fusion.causal.parameters(),
                    "lr": lr,
                    "weight_decay": 1e-4,
                }
            ]
            print("Optimizer: only fusion.causal parameters are trainable.")
        else:
            optim_params = [
                {
                    "params": self.fusion.parameters(),
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            ]
            print("Optimizer: fusion parameters are trainable.")

        self.optimizer = torch.optim.AdamW(
            optim_params,
            betas=(0.9, 0.999),
        )

    @ex.capture
    def optimize(self, epoch_num, DA):
        self.log.info("main track")

        with torch.no_grad():
            self.test_epoch(epoch=-1)

        self.log.info("before train test acc: {}".format(self.test_acc))

        for epoch in range(epoch_num):
            self.train_epoch(epoch)

            with torch.no_grad():
                self.test_epoch(epoch=epoch)

            self.log.info("epoch [{}] train loss: {}".format(epoch, self.dim_loss))
            self.log.info("epoch [{}] test acc: {}".format(epoch, self.test_acc))
            self.log.info(
                "epoch [{}] gets the best acc: {}".format(
                    self.best_epoch,
                    self.best_acc,
                )
            )

            self.log.info(
                "epoch [{}] CMoB loss: {:.6f}, contrib_skl: {:.6f}, contrib_rgb: {:.6f}, weak_skl: {:.6f}, weak_rgb: {:.6f}".format(
                    epoch,
                    self.cmob_loss_value,
                    self.cmob_contrib_skl,
                    self.cmob_contrib_rgb,
                    self.cmob_weak_skl,
                    self.cmob_weak_rgb,
                )
            )

            if hasattr(self.fusion, "causal") and self.fusion.causal is not None:
                if hasattr(self.fusion.causal, "alpha"):
                    self.log.info(
                        "causal alpha: {}".format(
                            torch.tanh(self.fusion.causal.alpha).item()
                        )
                    )

                if hasattr(self.fusion.causal, "conf_scale"):
                    self.log.info(
                        "conf scale: {}".format(
                            torch.tanh(self.fusion.causal.conf_scale).item()
                        )
                    )

            if DA:
                self.log.info(
                    "epoch [{}] DA test acc: {}".format(
                        epoch,
                        self.test_aug_acc,
                    )
                )
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
    ):
        self.encoder.train()
        self.proj.train()
        self.fusion.train()
        self.rgb_mlp.train()
        self.cmob.train()

        if fix_encoder:
            self.encoder.eval()
            self.proj.eval()
            self.rgb_mlp.eval()

        self.adjust_learning_rate(
            self.optimizer,
            current_epoch=epoch,
            max_epoch=50,
            lr_max=lr,
            warmup_epoch=5,
            loss_mode=loss_mode,
            step=step,
        )

        running_loss = []
        running_cmob = []
        running_contrib_skl = []
        running_contrib_rgb = []
        running_weak_skl = []
        running_weak_rgb = []

        loader = self.data_loader["train"]

        for data, label, rgb in tqdm(loader):

            data = data.type(torch.FloatTensor).cuda()
            label_g = gen_label(label)
            label = label.type(torch.LongTensor).cuda()

            if fix_encoder:
                with torch.no_grad():
                    seen_language = self.full_language[label]
                    seen_language = self.proj(seen_language)

                    skeleton_feat = self.encoder(data)

                    rgb_feat = rgb.type(torch.FloatTensor).cuda()
                    rgb_feat = self.rgb_mlp(rgb_feat)

                skeleton_feat = skeleton_feat.detach()
                rgb_feat = rgb_feat.detach()
                seen_language = seen_language.detach()

            else:
                seen_language = self.full_language[label]
                seen_language = self.proj(seen_language)

                skeleton_feat = self.encoder(data)

                rgb_feat = rgb.type(torch.FloatTensor).cuda()
                rgb_feat = self.rgb_mlp(rgb_feat)

            out = self.fusion(
                skeleton_feat,
                rgb_feat,
                seen_language,
                labels=label,
            )

            fusion_feature = out["fusion_feature"]

            if loss_type == "kl":
                feature = fusion_feature

                logits_per_skl, logits_per_text = create_logits(
                    feature,
                    seen_language,
                    self.logit_scale,
                    exp=True,
                )

                ground_truth = torch.tensor(
                    label_g,
                    dtype=feature.dtype,
                ).cuda()

                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)

                cls_loss = (loss_skls + loss_texts) / 2
                loss = cls_loss

            else:
                raise NotImplementedError(
                    "This modified main.py currently keeps your original KL training branch. "
                    "Please use loss_type='kl' first."
                )

            if self.use_cmob and epoch >= self.cmob_start_epoch:

                zero_rgb_feat = torch.zeros_like(rgb_feat)
                zero_skeleton_feat = torch.zeros_like(skeleton_feat)

                out_skl_only = self.fusion(
                    skeleton_feat,
                    zero_rgb_feat,
                    seen_language,
                    labels=label,
                    return_align_loss=False,
                )

                out_rgb_only = self.fusion(
                    zero_skeleton_feat,
                    rgb_feat,
                    seen_language,
                    labels=label,
                    return_align_loss=False,
                )

                fusion_feature_skl_only = out_skl_only["fusion_feature"]
                fusion_feature_rgb_only = out_rgb_only["fusion_feature"]

                cmob_out = self.cmob(
                    fusion_feature_all=fusion_feature,
                    fusion_feature_skl_only=fusion_feature_skl_only,
                    fusion_feature_rgb_only=fusion_feature_rgb_only,
                    text_feature=seen_language,
                    labels=label,
                    logit_scale=self.logit_scale,
                )

                loss_cmob = cmob_out["loss_cmob"]
                loss = loss + self.lambda_cmob * loss_cmob

                running_cmob.append(loss_cmob.detach())
                running_contrib_skl.append(cmob_out["avg_contrib_skl"].detach())
                running_contrib_rgb.append(cmob_out["avg_contrib_rgb"].detach())
                running_weak_skl.append(cmob_out["weak_skl_weight"].detach())
                running_weak_rgb.append(cmob_out["weak_rgb_weight"].detach())

            running_loss.append(loss.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        running_loss = torch.stack(running_loss)
        self.dim_loss = running_loss.mean().item()

        if len(running_cmob) > 0:
            self.cmob_loss_value = torch.stack(running_cmob).mean().item()
            self.cmob_contrib_skl = torch.stack(running_contrib_skl).mean().item()
            self.cmob_contrib_rgb = torch.stack(running_contrib_rgb).mean().item()
            self.cmob_weak_skl = torch.stack(running_weak_skl).mean().item()
            self.cmob_weak_rgb = torch.stack(running_weak_rgb).mean().item()

    @ex.capture
    def test_epoch(self, unseen_label, epoch, DA, support_factor):
        self.encoder.eval()
        self.proj.eval()
        self.fusion.eval()
        self.rgb_mlp.eval()

        loader = self.data_loader["test"]

        acc_list = []

        for data, label, rgb in tqdm(loader):

            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()

            rgb_feat = rgb.type(torch.FloatTensor).cuda()
            rgb_feat = self.rgb_mlp(rgb_feat)

            unseen_language = self.full_language[unseen_label]
            unseen_language_512 = self.proj(unseen_language)

            skeleton_feat = self.encoder(data)

            out = self.fusion(
                skeleton_feat,
                rgb_feat,
                text_feat=None,
                labels=None,
                return_align_loss=False,
            )

            fusion_feature = out["fusion_feature"]

            feat = fusion_feature

            acc_batch, pred = get_acc(
                feat,
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