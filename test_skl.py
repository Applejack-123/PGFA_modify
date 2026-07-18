# test.py

from config import *
from dataset import DataSet
from logger import Log

import os
import random
import textwrap
from math import pi, cos

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# from Causal import CausalIntervention as Causal

from rgb_only_mlp import RGBOnlyModule as RGBModule
from module.skeleton_guided_rgb_causal_debias import SkeletonGuidedRGBCausalDebias


NTU60_CLASS_NAMES = [
    "drink water",
    "eat meal/snack",
    "brushing teeth",
    "brushing hair",
    "drop",
    "pickup",
    "throw",
    "sitting down",
    "standing up",
    "clapping",
    "reading",
    "writing",
    "tear up paper",
    "wear jacket",
    "take off jacket",
    "wear a shoe",
    "take off a shoe",
    "wear on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping (one foot jumping)",
    "jump up",
    "make a phone call/answer phone",
    "playing with phone/tablet",
    "typing on a keyboard",
    "pointing to something with finger",
    "taking a selfie",
    "check time (from watch)",
    "rub two hands together",
    "nod head/bow",
    "shake head",
    "wipe face",
    "salute",
    "put the palms together",
    "cross hands in front (say stop)",
    "sneeze/cough",
    "staggering",
    "falling",
    "touch head (headache)",
    "touch chest (stomachache/heart pain)",
    "touch back (backache)",
    "touch neck (neckache)",
    "nausea or vomiting condition",
    "use a fan/feeling warm",
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "pat on back of other person",
    "point finger at the other person",
    "hugging other person",
    "giving something to other person",
    "touch other person's pocket",
    "handshaking",
    "walking towards each other",
    "walking apart from each other",
]

assert len(NTU60_CLASS_NAMES) == 60


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_ntu60_class_name(global_label):
    global_label = int(global_label)

    if 0 <= global_label < len(NTU60_CLASS_NAMES):
        return NTU60_CLASS_NAMES[global_label]

    return "unknown class"


def get_ntu60_class_description(global_label):
    global_label = int(global_label)
    return f"A{global_label + 1:03d} {get_ntu60_class_name(global_label)}"


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
        self.dataset = {}
        self.data_loader = {}

        self.best_epoch = -1
        self.best_acc = -1
        self.dim_loss = -1
        self.test_acc = -1
        self.test_aug_acc = -1
        self.best_aug_acc = -1
        self.best_aug_epoch = -1

        self.full_language = np.load(language_path)
        self.full_language = torch.tensor(self.full_language, dtype=torch.float32).cuda()

        self.dataset["train"] = DataSet(train_list, train_label, train_rgb)
        self.dataset["test"] = DataSet(test_list, test_label, test_rgb)

        self.data_loader["train"] = torch.utils.data.DataLoader(
            dataset=self.dataset["train"],
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=self.dataset["test"],
            batch_size=64,
            num_workers=16,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    def load_weights(self, model=None, weight_path=None):
        checkpoint = torch.load(weight_path, map_location="cuda")

        if model is self.encoder and "encoder" in checkpoint:
            model.load_state_dict(checkpoint["encoder"])

        elif model is self.proj and "proj" in checkpoint:
            model.load_state_dict(checkpoint["proj"])

        elif model is self.rgb_mlp and "rgb_mlp" in checkpoint:
            model.load_state_dict(checkpoint["rgb_mlp"])

        elif model is self.fusion and "fusion" in checkpoint:
            missing, unexpected = model.load_state_dict(checkpoint["fusion"], strict=False)

            if missing:
                print("Fusion missing keys:")
                for key in missing:
                    print("  ", key)

            if unexpected:
                print("Fusion unexpected keys:")
                for key in unexpected:
                    print("  ", key)

        else:
            raise RuntimeError(f"Cannot find weights for this model in checkpoint: {weight_path}")

    def adjust_learning_rate(
        self,
        optimizer,
        current_epoch,
        max_epoch,
        lr_min=0,
        lr_max=0.1,
        warmup_epoch=15,
        loss_mode="cos",
        step=(20, 30),
    ):
        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch

        elif loss_mode == "cos":
            progress = (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch)
            lr = lr_min + (lr_max - lr_min) * (1 + cos(pi * progress)) / 2

        elif loss_mode == "step":
            lr = lr_max * (0.1 ** np.sum(current_epoch >= np.array(step)))

        else:
            raise ValueError(f"Unsupported loss_mode: {loss_mode}")

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * param_group.get("lr_scale", 1.0)

    def layernorm(self, feature):
        mean = torch.mean(feature, dim=1, keepdim=True)
        var = torch.var(feature, dim=1, keepdim=True, unbiased=False)
        return (feature - mean) / torch.sqrt(var + 1e-6)

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

        """
        self.rgb_causal_debias = SkeletonGuidedRGBCausalDebias(
            dim=512,
            hidden_dim=1024,
            dropout=0.1,
            use_temporal_conv=True,
            fusion_mode="gate",
            detach_skeleton_anchor=False,
        ).cuda()
        """

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
            use_causal=False,
        ).cuda()

        kl_loss_types = {"kl", "klv2", "kl+cosface", "kl+sphereface", "kl+margin"}

        if loss_type in kl_loss_types:
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
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        self.logit_scale = self.fusion.logit_scale

        if fix_encoder:
            self.load_weights(self.encoder, weight_path)
            self.load_weights(self.proj, weight_path)
            self.load_weights(self.rgb_mlp, weight_path)
            self.load_weights(self.fusion, weight_path)
        else:
            print("NOT loading weights")

    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):
        self.optimizer = torch.optim.SGD(
            [
                {"params": self.encoder.parameters(), "lr": lr, "lr_scale": 1.0},
                {"params": self.proj.parameters(), "lr": lr, "lr_scale": 1.0},
                {"params": self.fusion.parameters(), "lr": lr, "lr_scale": 1.0},
                {"params": self.rgb_mlp.parameters(), "lr": lr, "lr_scale": 1.0},
            ],
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=False,
        )

    @ex.capture
    def optimize(self, epoch_num, DA, split, lr):
        print("split_{}".format(split))
        print("lr={}".format(lr))

        with torch.no_grad():
            self.test_epoch(epoch=-1)

        print("before train test acc: {}".format(self.test_acc))


    @staticmethod
    def normalize_unseen_label(unseen_label, device):
        if isinstance(unseen_label, torch.Tensor):
            unseen_label_tensor = unseen_label.to(device=device, dtype=torch.long)
        else:
            unseen_label_tensor = torch.tensor(list(unseen_label), dtype=torch.long, device=device)

        if unseen_label_tensor.ndim != 1:
            raise ValueError(f"unseen_label must be one-dimensional, current shape={tuple(unseen_label_tensor.shape)}")

        if unseen_label_tensor.numel() == 0:
            raise ValueError("unseen_label cannot be empty")

        return unseen_label_tensor

    @staticmethod
    def map_global_label_to_local(label, unseen_label_tensor):
        true_local = torch.full_like(label, fill_value=-1)

        for local_id, global_id in enumerate(unseen_label_tensor):
            true_local[label == global_id] = local_id

        invalid_mask = true_local < 0

        if invalid_mask.any():
            invalid_labels = torch.unique(label[invalid_mask]).detach().cpu().tolist()
            unseen_list = unseen_label_tensor.detach().cpu().tolist()
            raise ValueError(f"Test labels {invalid_labels} are not contained in unseen_label={unseen_list}")

        return true_local

    @staticmethod
    def calculate_test_logits(feature, text_feature, logit_scale):
        if feature.ndim == 3:
            feature = feature.mean(dim=1)

        if text_feature.ndim == 3:
            text_feature = text_feature.mean(dim=1)

        if feature.ndim != 2:
            raise ValueError(f"fusion_feature must be [B,D] or [B,T,D], current shape={tuple(feature.shape)}")

        if text_feature.ndim != 2:
            raise ValueError(f"text_feature must be [C,D], current shape={tuple(text_feature.shape)}")

        logits_per_feature, _ = create_logits(feature, text_feature, logit_scale, exp=True)
        return logits_per_feature

    @staticmethod
    def build_confusion_matrix(true_local, pred_local, num_classes):
        indices = true_local.long() * num_classes + pred_local.long()
        counts = torch.bincount(indices, minlength=num_classes * num_classes)
        return counts.reshape(num_classes, num_classes)

    @staticmethod
    def save_accuracy_figure(
        confusion_matrix,
        unseen_label,
        overall_accuracy,
        epoch,
        output_dir,
        split,
    ):
        os.makedirs(output_dir, exist_ok=True)

        unseen_label = [int(class_id) for class_id in unseen_label]
        confusion_matrix = confusion_matrix.detach().cpu().float()

        class_total = confusion_matrix.sum(dim=1)
        class_correct = confusion_matrix.diag()
        class_accuracy = torch.where(class_total > 0, class_correct / class_total, torch.zeros_like(class_total))

        macro_accuracy = class_accuracy.mean().item()
        row_sum = confusion_matrix.sum(dim=1, keepdim=True).clamp(min=1)
        normalized_confusion = confusion_matrix / row_sum

        class_descriptions = [get_ntu60_class_description(class_id) for class_id in unseen_label]
        bar_labels = [textwrap.fill(name, width=25) for name in class_descriptions]
        matrix_labels = [f"A{class_id + 1:03d}" for class_id in unseen_label]

        prefix = "before_train" if epoch < 0 else f"epoch_{epoch:03d}"
        save_path = os.path.join(output_dir, f"{prefix}_accuracy.png")

        figure_height = max(7, len(unseen_label) * 1.1)
        fig, axes = plt.subplots(1, 2, figsize=(20, figure_height), gridspec_kw={"width_ratios": [1.25, 1]})

        y_positions = np.arange(len(unseen_label))
        bars = axes[0].barh(y_positions, class_accuracy.numpy())

        axes[0].set_yticks(y_positions)
        axes[0].set_yticklabels(bar_labels, fontsize=10)
        axes[0].set_xlim(0.0, 1.08)
        axes[0].set_xlabel("Accuracy")
        axes[0].set_title(
            f"Split {split} Per-class Accuracy\n"
            f"Overall ACC = {overall_accuracy:.4f}    Macro ACC = {macro_accuracy:.4f}"
        )
        axes[0].invert_yaxis()
        axes[0].grid(axis="x", linestyle="--", alpha=0.4)

        for bar, accuracy, correct, total in zip(bars, class_accuracy, class_correct, class_total):
            axes[0].text(
                min(float(accuracy.item()) + 0.015, 1.01),
                bar.get_y() + bar.get_height() / 2,
                f"{accuracy.item() * 100:.2f}% ({int(correct.item())}/{int(total.item())})",
                va="center",
                fontsize=10,
            )

        image = axes[1].imshow(normalized_confusion.numpy(), vmin=0.0, vmax=1.0, aspect="auto")
        axes[1].set_xticks(np.arange(len(matrix_labels)))
        axes[1].set_yticks(np.arange(len(matrix_labels)))
        axes[1].set_xticklabels(matrix_labels, rotation=35, ha="right")
        axes[1].set_yticklabels(matrix_labels)
        axes[1].set_xlabel("Predicted class")
        axes[1].set_ylabel("True class")
        axes[1].set_title("Normalized Confusion Matrix")

        for row in range(len(unseen_label)):
            for column in range(len(unseen_label)):
                value = normalized_confusion[row, column].item()
                axes[1].text(column, row, f"{value * 100:.1f}%", ha="center", va="center", fontsize=9)

        colorbar = fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
        colorbar.set_label("Percentage")

        fig.suptitle(f"NTU RGB+D 60 Zero-shot Test Result - Split {split}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print("Accuracy figure saved to:", save_path)
        return save_path

    @ex.capture
    def test_epoch(
        self,
        unseen_label,
        epoch,
        DA,
        support_factor,
        split,
        sample_result_dir="./output/sample_results/skl_only",
    ):
        self.encoder.eval()
        self.proj.eval()
        self.fusion.eval()
        self.rgb_mlp.eval()

        loader = self.data_loader["test"]
        device = next(self.fusion.parameters()).device

        unseen_label_tensor = self.normalize_unseen_label(unseen_label, device)
        unseen_label_list = unseen_label_tensor.detach().cpu().tolist()
        num_unseen_classes = len(unseen_label_list)

        unseen_language = self.full_language[unseen_label_tensor]
        unseen_language_512 = self.proj(unseen_language)

        all_true_global = []
        all_true_local = []
        all_pred_global = []
        all_pred_local = []

        with torch.no_grad():
            for data, label, rgb in tqdm(loader, desc=f"Test epoch {epoch}"):
                data = data.float().cuda(non_blocking=True)
                label = label.long().cuda(non_blocking=True)

                skeleton_feat = self.encoder(data)

                logits = self.calculate_test_logits(
                    skeleton_feat,
                    unseen_language_512,
                    self.logit_scale,
                )

                if logits.shape[1] != num_unseen_classes:
                    raise RuntimeError(
                        f"logits class count does not match unseen class count: "
                        f"logits.shape={tuple(logits.shape)}, num_unseen_classes={num_unseen_classes}"
                    )

                pred_local = logits.argmax(dim=1)
                pred_global = unseen_label_tensor[pred_local]
                true_local = self.map_global_label_to_local(label, unseen_label_tensor)

                all_true_global.append(label.detach().cpu())
                all_true_local.append(true_local.detach().cpu())
                all_pred_global.append(pred_global.detach().cpu())
                all_pred_local.append(pred_local.detach().cpu())

        if not all_true_global:
            raise RuntimeError("Test DataLoader returned no samples")

        all_true_global = torch.cat(all_true_global, dim=0)
        all_true_local = torch.cat(all_true_local, dim=0)
        all_pred_global = torch.cat(all_pred_global, dim=0)
        all_pred_local = torch.cat(all_pred_local, dim=0)

        total_samples = all_true_global.numel()
        total_correct = all_pred_global.eq(all_true_global).sum().item()
        acc = total_correct / total_samples

        confusion_matrix = self.build_confusion_matrix(
            true_local=all_true_local,
            pred_local=all_pred_local,
            num_classes=num_unseen_classes,
        )

        unique_test_labels = sorted(torch.unique(all_true_global).tolist())
        invalid_test_labels = sorted(set(unique_test_labels) - set(unseen_label_list))

        print("\n" + "=" * 110)
        print("Test unique labels:", unique_test_labels)
        print("Unseen labels:", unseen_label_list)
        print("Test labels not in unseen labels:", invalid_test_labels)
        print("Total test samples:", total_samples)
        print("Correct samples:", total_correct)
        print("Overall test ACC: {:.6f}".format(acc))

        print("-" * 110)
        print("Per-class accuracy:")

        for local_id, global_id in enumerate(unseen_label_list):
            class_name = get_ntu60_class_name(global_id)
            action_id = f"A{global_id + 1:03d}"
            class_total = int(confusion_matrix[local_id].sum().item())
            class_correct = int(confusion_matrix[local_id, local_id].item())
            class_accuracy = class_correct / class_total if class_total > 0 else 0.0

            print(
                "local={:<2d} global={:<2d} {:<5s} {:<42s} ACC={:.6f} ({}/{})".format(
                    local_id,
                    global_id,
                    action_id,
                    class_name,
                    class_accuracy,
                    class_correct,
                    class_total,
                )
            )

        print("-" * 110)
        print("Confusion matrix class order:")

        for local_id, global_id in enumerate(unseen_label_list):
            print(
                "local={:<2d} -> global={:<2d} {:<5s} {}".format(
                    local_id,
                    global_id,
                    f"A{global_id + 1:03d}",
                    get_ntu60_class_name(global_id),
                )
            )

        print("Rows = true class, columns = predicted class")
        print(confusion_matrix)
        print("=" * 110)

        current_output_dir = os.path.join(sample_result_dir, f"split_{split}")

        self.save_accuracy_figure(
            confusion_matrix=confusion_matrix,
            unseen_label=unseen_label_list,
            overall_accuracy=acc,
            epoch=epoch,
            output_dir=current_output_dir,
            split=split,
        )

        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            # self.save_model()

        self.test_acc = acc
        return acc

    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    @ex.capture
    def save_model(self, save_path):
        save_dir = os.path.dirname(save_path)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

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
    processor = Processor()
    processor.start()