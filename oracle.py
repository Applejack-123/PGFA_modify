# main2_oracle.py
# 作用：只跑 Skeleton / RGB / Fusion 的 Oracle 诊断，不训练。
# 放到你的项目根目录 /root/autodl-tmp/Github/PGFA_modify/ 下运行：
#   python main2_oracle.py

from config import *
from dataset import DataSet
from logger import Log

import os
import csv
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from tool import get_acc
from module.skeleton_mamba_encoder import SkeletonMambaEncoder
from module.adapter import Linear
from align_mamba2_fusion import AlignMamba2Fusion
from rgb_only_mlp import RGBOnlyModule as RGBModule


def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(0)


class Processor:
    @ex.capture
    def load_data(self, test_list, test_label, test_rgb, batch_size, language_path):
        self.dataset = {}
        self.data_loader = {}

        self.full_language = np.load(language_path)
        self.full_language = torch.tensor(self.full_language, dtype=torch.float32).cuda()

        self.dataset["test"] = DataSet(test_list, test_label, test_rgb)
        self.data_loader["test"] = torch.utils.data.DataLoader(
            dataset=self.dataset["test"],
            batch_size=64,
            num_workers=16,
            shuffle=False,
            drop_last=False,
        )

    def _safe_load_state_dict(self, module, state_dict, name, strict=True, allow_missing_prefix=None):
        if strict:
            module.load_state_dict(state_dict)
            print(f"Loaded {name} with strict=True")
            return

        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        print(f"Loaded {name} with strict=False")
        print(f"  missing keys num: {len(missing)}")
        print(f"  unexpected keys num: {len(unexpected)}")

        if allow_missing_prefix is not None:
            bad_missing = [k for k in missing if not k.startswith(allow_missing_prefix)]
            if len(bad_missing) > 0:
                raise RuntimeError(
                    f"{name} missing non-allowed keys. First 20: {bad_missing[:20]}"
                )

    @ex.capture
    def load_model(
        self,
        hidden_dim,
        language_size,
        weight_path,
        # 如果你要测带 causal 的 fusion，把命令行或 config 里设 oracle_use_causal=True。
        # 默认 False：用于测你当前 0.866 的纯 Light MA-Mamba Fusion。
        oracle_use_causal=False,
    ):
        self.encoder = SkeletonMambaEncoder(
            in_channels=3,
            num_point=25,
            num_person=2,
            embed_dim=hidden_dim,
            temporal_depth=2,
            dropout=0.1,
        ).cuda()

        # 这里保留 adapter/logit_scale 兼容你的原工程，但 oracle 不直接用 adapter。
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
            use_causal=oracle_use_causal,
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

        if weight_path is None or str(weight_path).strip() == "":
            raise ValueError("必须提供 weight_path，否则 oracle 诊断没有意义。")

        checkpoint = torch.load(weight_path, map_location="cpu")
        print(f"Loaded checkpoint from: {weight_path}")
        print("Checkpoint keys:", list(checkpoint.keys()))

        if "encoder" not in checkpoint:
            raise KeyError("checkpoint 里没有 encoder 权重。")
        if "proj" not in checkpoint:
            raise KeyError("checkpoint 里没有 proj 权重。")
        if "fusion" not in checkpoint:
            raise KeyError("checkpoint 里没有 fusion 权重。")

        self._safe_load_state_dict(self.encoder, checkpoint["encoder"], "encoder", strict=True)
        self._safe_load_state_dict(self.proj, checkpoint["proj"], "proj", strict=True)
        self._safe_load_state_dict(self.rgb_mlp, checkpoint["rgb_mlp"], "rgb_mlp", strict=True)
        
        # fusion 用 strict=False：兼容 checkpoint 里有/没有 causal 的情况。
        self._safe_load_state_dict(
            self.fusion,
            checkpoint["fusion"],
            "fusion",
            strict=False,
        )

        self.encoder.eval()
        self.proj.eval()
        self.fusion.eval()

    @staticmethod
    def pool_feature(feat):
        """
        feat:
            [B, D]    -> [B, D]
            [B, T, D] -> [B, D]
        """
        if feat.dim() == 3:
            return feat.mean(dim=1)
        if feat.dim() == 2:
            return feat
        raise ValueError(f"Unsupported feature shape: {tuple(feat.shape)}")

    @staticmethod
    def _to_bool_cpu(x):
        return x.detach().bool().cpu()

    def _per_class_stats(self, labels, skl_correct, rgb_correct, fusion_correct, unseen_label):
        labels = labels.cpu().numpy()
        skl_correct = skl_correct.cpu().numpy().astype(bool)
        rgb_correct = rgb_correct.cpu().numpy().astype(bool)
        fusion_correct = fusion_correct.cpu().numpy().astype(bool)
        oracle_correct = skl_correct | rgb_correct

        rows = []
        for cls in list(unseen_label):
            cls = int(cls)
            mask = labels == cls
            n = int(mask.sum())
            if n == 0:
                continue
            rows.append({
                "class": cls,
                "num": n,
                "skl_acc": float(skl_correct[mask].mean()),
                "rgb_acc": float(rgb_correct[mask].mean()),
                "fusion_acc": float(fusion_correct[mask].mean()),
                "oracle_skl_rgb_acc": float(oracle_correct[mask].mean()),
                "only_skl": float((skl_correct[mask] & ~rgb_correct[mask]).mean()),
                "only_rgb": float((~skl_correct[mask] & rgb_correct[mask]).mean()),
                "both_wrong": float((~skl_correct[mask] & ~rgb_correct[mask]).mean()),
            })
        return rows

    @ex.capture
    def test_oracle_epoch(self, unseen_label, oracle_save_dir="oracle_outputs"):
        self.encoder.eval()
        self.proj.eval()
        self.fusion.eval()

        os.makedirs(oracle_save_dir, exist_ok=True)

        loader = self.data_loader["test"]

        all_labels = []
        all_skl_pred = []
        all_rgb_pred = []
        all_fusion_pred = []

        skl_correct_all = []
        rgb_correct_all = []
        fusion_correct_all = []

        gate_mean_list = []

        with torch.no_grad():
            unseen_language = self.full_language[unseen_label]
            unseen_language_512 = self.proj(unseen_language)

            for data, label, rgb in tqdm(loader, desc="Oracle Eval"):
                data = data.float().cuda()
                label = label.long().cuda()
                rgb_feat = rgb.float().cuda()
                rgb_feat = self.rgb_mlp(rgb_feat)
                
                skeleton_feat = self.encoder(data)

                # 单模态特征池化：[B,T,512] -> [B,512]
                skl_feature = self.pool_feature(skeleton_feat)
                rgb_feature = rgb_feat.mean(dim=1)          # [B, 512]

                # fusion 测试时不传 text/label，避免 zero-shot 标签泄漏
                out = self.fusion(
                    skeleton_feat,
                    rgb_feat,
                    text_feat=None,
                    labels=None,
                    return_align_loss=False,
                )
                fusion_feature = out["fusion_feature"]

                # 如果你的 fusion 返回了 fusion_gate，可以顺便看 gate；没有也不影响 oracle。
                if "fusion_gate" in out and out["fusion_gate"] is not None:
                    gate_mean_list.append(out["fusion_gate"].detach().mean().cpu())

                _, skl_pred = get_acc(skl_feature, unseen_language_512, unseen_label, label)
                _, rgb_pred = get_acc(rgb_feature, unseen_language_512, unseen_label, label)
                _, fusion_pred = get_acc(fusion_feature, unseen_language_512, unseen_label, label)

                skl_pred = skl_pred.to(label.device).long()
                rgb_pred = rgb_pred.to(label.device).long()
                fusion_pred = fusion_pred.to(label.device).long()

                skl_correct = skl_pred.eq(label)
                rgb_correct = rgb_pred.eq(label)
                fusion_correct = fusion_pred.eq(label)

                all_labels.append(label.detach().cpu())
                all_skl_pred.append(skl_pred.detach().cpu())
                all_rgb_pred.append(rgb_pred.detach().cpu())
                all_fusion_pred.append(fusion_pred.detach().cpu())

                skl_correct_all.append(self._to_bool_cpu(skl_correct))
                rgb_correct_all.append(self._to_bool_cpu(rgb_correct))
                fusion_correct_all.append(self._to_bool_cpu(fusion_correct))

        labels = torch.cat(all_labels, dim=0)
        skl_pred = torch.cat(all_skl_pred, dim=0)
        rgb_pred = torch.cat(all_rgb_pred, dim=0)
        fusion_pred = torch.cat(all_fusion_pred, dim=0)

        skl_correct = torch.cat(skl_correct_all, dim=0)
        rgb_correct = torch.cat(rgb_correct_all, dim=0)
        fusion_correct = torch.cat(fusion_correct_all, dim=0)

        oracle_skl_rgb = skl_correct | rgb_correct
        oracle_all = skl_correct | rgb_correct | fusion_correct

        both_correct = skl_correct & rgb_correct
        only_skl = skl_correct & (~rgb_correct)
        only_rgb = (~skl_correct) & rgb_correct
        both_wrong = (~skl_correct) & (~rgb_correct)

        result = {
            "num_samples": int(labels.numel()),
            "skl_acc": skl_correct.float().mean().item(),
            "rgb_acc": rgb_correct.float().mean().item(),
            "fusion_acc": fusion_correct.float().mean().item(),
            "oracle_skl_rgb_acc": oracle_skl_rgb.float().mean().item(),
            "oracle_all_acc": oracle_all.float().mean().item(),
            "oracle_gap_vs_fusion": (oracle_skl_rgb.float().mean() - fusion_correct.float().mean()).item(),
            "both_skl_rgb_correct": both_correct.float().mean().item(),
            "only_skl_correct": only_skl.float().mean().item(),
            "only_rgb_correct": only_rgb.float().mean().item(),
            "both_skl_rgb_wrong": both_wrong.float().mean().item(),
        }

        if len(gate_mean_list) > 0:
            result["fusion_gate_mean"] = torch.stack(gate_mean_list).mean().item()

        print("\n" + "=" * 90)
        print("Oracle Analysis")
        print("=" * 90)
        print(f"Num samples              : {result['num_samples']}")
        print(f"Skeleton only acc        : {result['skl_acc']:.6f}")
        print(f"RGB only acc             : {result['rgb_acc']:.6f}")
        print(f"Fusion acc               : {result['fusion_acc']:.6f}")
        print(f"Oracle(Skeleton OR RGB)  : {result['oracle_skl_rgb_acc']:.6f}")
        print(f"Oracle(Skl OR RGB OR Fus): {result['oracle_all_acc']:.6f}")
        print(f"Oracle gap vs Fusion     : {result['oracle_gap_vs_fusion']:.6f}")
        print("-" * 90)
        print(f"Both Skeleton & RGB correct : {result['both_skl_rgb_correct']:.6f}")
        print(f"Only Skeleton correct       : {result['only_skl_correct']:.6f}")
        print(f"Only RGB correct            : {result['only_rgb_correct']:.6f}")
        print(f"Both Skeleton & RGB wrong   : {result['both_skl_rgb_wrong']:.6f}")
        if "fusion_gate_mean" in result:
            print(f"Fusion gate mean            : {result['fusion_gate_mean']:.6f}")
        print("=" * 90 + "\n")

        # 保存整体预测和 correct mask，方便后续画图/查错样本。
        npz_path = os.path.join(oracle_save_dir, "oracle_predictions.npz")
        np.savez(
            npz_path,
            label=labels.numpy(),
            skl_pred=skl_pred.numpy(),
            rgb_pred=rgb_pred.numpy(),
            fusion_pred=fusion_pred.numpy(),
            skl_correct=skl_correct.numpy(),
            rgb_correct=rgb_correct.numpy(),
            fusion_correct=fusion_correct.numpy(),
            oracle_skl_rgb=oracle_skl_rgb.numpy(),
            unseen_label=np.array(list(unseen_label), dtype=np.int64),
        )
        print(f"Saved prediction masks to: {npz_path}")

        # 保存 per-class 结果。
        per_class_rows = self._per_class_stats(
            labels,
            skl_correct,
            rgb_correct,
            fusion_correct,
            unseen_label,
        )
        csv_path = os.path.join(oracle_save_dir, "oracle_per_class.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "class", "num", "skl_acc", "rgb_acc", "fusion_acc",
                    "oracle_skl_rgb_acc", "only_skl", "only_rgb", "both_wrong",
                ],
            )
            writer.writeheader()
            writer.writerows(per_class_rows)
        print(f"Saved per-class oracle results to: {csv_path}")

        return result

    @ex.capture
    def optimize(self, epoch_num=None, DA=None):
        self.log.info("oracle track")
        with torch.no_grad():
            result = self.test_oracle_epoch()
        self.log.info("Oracle result: {}".format(result))
        return result

    def initialize(self):
        self.load_data()
        self.load_model()
        self.log = Log()

    def start(self):
        self.initialize()
        self.optimize()


@ex.automain
def main(track):
    p = Processor()
    p.start()
