#tsne.py
from config import *
from dataset import DataSet
from logger import Log

import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import pi, cos
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from module.gcn.st_gcn import Model
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
from module.cross_attention_fusion import CrossAttentionFusion
from cross_mamba import MambaFusion
from temporal_mamba import TemporalMamba
from Causal import CausalIntervention as Causal


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)


class Processor:

    @ex.capture
    def load_data(self, train_list, train_label, test_list, test_label, train_rgb, test_rgb, batch_size, language_path):
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

        self.dataset['train'] = DataSet(train_list, train_label, train_rgb)
        self.dataset['test'] = DataSet(test_list, test_label, test_rgb)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            drop_last=True,
        )

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=64,
            num_workers=16,
            shuffle=False,
        )

    def load_weights(self, model=None, weight_path=None):
        checkpoint = torch.load(weight_path, map_location='cpu')

        if model is self.encoder and 'encoder' in checkpoint:
            model.load_state_dict(checkpoint['encoder'])
            print('[Loaded] encoder')
        elif model is self.adapter and 'adapter' in checkpoint:
            model.load_state_dict(checkpoint['adapter'])
            print('[Loaded] adapter')
        elif model is self.proj and 'proj' in checkpoint:
            model.load_state_dict(checkpoint['proj'])
            print('[Loaded] proj')
        elif model is self.fusion and 'fusion' in checkpoint:
            model.load_state_dict(checkpoint['fusion'])
            print('[Loaded] fusion')
        elif model is self.causal and 'causal' in checkpoint:
            model.load_state_dict(checkpoint['causal'])
            print('[Loaded] causal')
        elif model is self.temporal_mamba and 'temporal_mamba' in checkpoint:
            model.load_state_dict(checkpoint['temporal_mamba'])
            print('[Loaded] temporal_mamba')
        elif model is self.mamba_fusion and 'mamba_fusion' in checkpoint:
            model.load_state_dict(checkpoint['mamba_fusion'])
            print('[Loaded] mamba_fusion')
        elif model is self.concat_proj and 'concat_proj' in checkpoint:
            model.load_state_dict(checkpoint['concat_proj'])
            print('[Loaded] concat_proj')
        else:
            print('[Warning] cannot find matched weight for this module, skip it.')

    def adjust_learning_rate(self, optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1,
                             warmup_epoch=15, loss_mode='cos', step=[20, 30]):
        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        elif loss_mode == 'cos':
            lr = lr_min + (lr_max - lr_min) * (
                1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))
            ) / 2
        elif loss_mode == 'step':
            lr = lr_max * (0.1 ** np.sum(current_epoch >= np.array(step)))
        else:
            raise Exception('Please check loss_mode!')

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def layernorm(self, feature):
        num = feature.shape[0]
        mean = torch.mean(feature, dim=1).reshape(num, -1)
        var = torch.var(feature, dim=1).reshape(num, -1)
        out = (feature - mean) / torch.sqrt(var)
        return out

    @ex.capture
    def load_model(self, in_channels, hidden_channels, hidden_dim,
                   dropout, graph_args, edge_importance_weighting, visual_size,
                   language_size, weight_path, loss_type, fix_encoder, finetune):
        self.encoder = Model(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            graph_args=graph_args,
            edge_importance_weighting=edge_importance_weighting,
        ).cuda()

        self.adapter = Linear().cuda()

        self.proj = nn.Sequential(
            nn.LayerNorm(language_size),
            nn.Linear(language_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        ).cuda()

        self.concat_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
        ).cuda()

        self.fusion = CrossAttentionFusion(
            feature_dim=512,
            num_heads=8,
            dropout=0.3,
        ).cuda()

        self.causal = Causal(dim=512).cuda()
        self.mamba_fusion = MambaFusion(num_layers=4).cuda()
        self.temporal_mamba = nn.Sequential(TemporalMamba(512), TemporalMamba(512)).cuda()

        if loss_type == "kl" or loss_type == "klv2" or loss_type == "kl+cosface" or loss_type == "kl+sphereface" or "kl+margin":
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
            raise Exception('loss_type Error!')

        self.logit_scale = self.adapter.get_logit_scale()
        self.logit_scale_v2 = self.adapter.get_logit_scale_v2()

        # t-SNE 需要加载训练后的权重。
        # 这里沿用你原 main.py 的逻辑：fix_encoder 或 finetune 为 True 时加载 weight_path。
        if fix_encoder or finetune:
            print('[Info] Loading weights from:', weight_path)
            self.load_weights(self.encoder, weight_path)
            self.load_weights(self.adapter, weight_path)
            self.load_weights(self.proj, weight_path)
            self.load_weights(self.causal, weight_path)
            self.load_weights(self.mamba_fusion, weight_path)
        else:
            print('[Warning] fix_encoder=False and finetune=False.')
            print('[Warning] 当前模型可能是随机初始化，t-SNE 和指标没有意义。')
            print('[Warning] 请在 config.py 中设置 fix_encoder=True 或 finetune=True，并检查 weight_path。')

    @staticmethod
    def _global_pool_feature(feature):
        """
        把特征统一变成 (B, 512)。
        skeleton_feat / rgb_feat / fusion_feat 常见形状是 (B, 8, 512)，这里对时间维求平均。
        """
        if feature.dim() == 3:
            return feature.mean(dim=1)
        if feature.dim() == 2:
            return feature
        return feature.reshape(feature.shape[0], -1)

    @staticmethod
    def _sample_by_class(labels, sample_num=20, seed=0):
        """
        每类最多采样 sample_num 个样本。
        sample_num <= 0 表示不采样，使用全部样本。
        """
        if sample_num is None or int(sample_num) <= 0:
            return np.arange(len(labels), dtype=np.int64)

        rng = np.random.default_rng(seed)
        labels_np = labels.cpu().numpy()
        selected = []

        for c in np.unique(labels_np):
            idx = np.where(labels_np == c)[0]
            if len(idx) > int(sample_num):
                idx = rng.choice(idx, int(sample_num), replace=False)
            selected.extend(idx.tolist())

        return np.array(selected, dtype=np.int64)

    @staticmethod
    def _build_text_index(labels, unseen_label):
        """
        labels 是数据集真实类别编号，例如 [4, 19, 31, 47, 51]。
        unseen_label 是 zero-shot split，例如 split_1 = [4, 19, 31, 47, 51]。
        unseen_language_512 的行号是 0~4，所以必须做真实类别 -> split 内部位置的映射。
        """
        if isinstance(unseen_label, torch.Tensor):
            unseen_label_cpu = unseen_label.detach().cpu().long()
        else:
            unseen_label_cpu = torch.tensor(unseen_label, dtype=torch.long)

        label_to_pos = {int(cls): i for i, cls in enumerate(unseen_label_cpu.tolist())}
        labels_list = labels.detach().cpu().long().tolist()

        missing = sorted(set(int(x) for x in labels_list if int(x) not in label_to_pos))
        if len(missing) > 0:
            raise ValueError(
                '\n[Error] 下面这些 test label 不在 unseen_label 中：{}\n'
                'unseen_label = {}\n'
                '这通常说明 split 编号和数据集 label 编号不一致，比如 0-based / 1-based 搞反。\n'.format(
                    missing, unseen_label_cpu.tolist()
                )
            )

        text_index = torch.tensor([label_to_pos[int(x)] for x in labels_list], dtype=torch.long)
        return text_index, unseen_label_cpu

    @staticmethod
    def _run_tsne(feature, random_state=0):
        n = int(feature.shape[0])
        if n <= 2:
            raise ValueError('t-SNE 至少需要 3 个点，当前只有 {} 个点。'.format(n))

        # perplexity 必须小于样本数。
        perplexity = min(30, max(2, (n - 1) // 3))
        if perplexity >= n:
            perplexity = max(1, n - 1)

        embedding = TSNE(
            n_components=2,
            metric='cosine',
            perplexity=perplexity,
            random_state=random_state,
            init='random',
            learning_rate='auto',
        ).fit_transform(feature)
        return embedding

    @staticmethod
    def _get_class_prototype(feature, label):
        """
        feature: (N, 512)
        label:   (N,)
        return:
            proto_feature: (C, 512)
            proto_label:   (C,)
        """
        classes = torch.unique(label, sorted=True)
        proto_list = []
        proto_label = []

        for c in classes:
            idx = label == c
            proto = feature[idx].mean(dim=0)
            proto = F.normalize(proto, dim=0)
            proto_list.append(proto)
            proto_label.append(c)

        proto_feature = torch.stack(proto_list, dim=0)
        proto_label = torch.stack(proto_label, dim=0)
        return proto_feature, proto_label

    @staticmethod
    def _plot_modal_text_tsne(modal_feature, modal_label, text_feature, text_label,
                              title, modal_name, modal_marker, save_path):
        """
        样本级 t-SNE：
        modal_feature: (N, 512)，Skeleton/RGB/Fusion 样本特征
        modal_label:   (N,)，样本真实类别编号
        text_feature:  (C, 512)，每类一个文本原型
        text_label:    (C,)，文本原型真实类别编号
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        feature = torch.cat([modal_feature, text_feature], dim=0).numpy()
        embedding = Processor._run_tsne(feature)

        n = len(modal_feature)
        modal_2d = embedding[:n]
        text_2d = embedding[n:]

        modal_label_np = modal_label.cpu().numpy()
        text_label_np = text_label.cpu().numpy()
        classes = np.unique(np.concatenate([modal_label_np, text_label_np]))

        cmap = plt.cm.get_cmap('tab20', len(classes))
        color_map = {int(c): cmap(i) for i, c in enumerate(classes)}

        plt.figure(figsize=(10, 8))

        for c in classes:
            c = int(c)
            idx = modal_label_np == c
            if np.any(idx):
                plt.scatter(
                    modal_2d[idx, 0],
                    modal_2d[idx, 1],
                    c=[color_map[c]],
                    marker=modal_marker,
                    s=14,
                    alpha=0.65,
                    linewidths=0,
                )

            tidx = text_label_np == c
            if np.any(tidx):
                plt.scatter(
                    text_2d[tidx, 0],
                    text_2d[tidx, 1],
                    c=[color_map[c]],
                    marker='x',
                    s=100,
                    linewidths=2,
                )

        plt.title(title)
        plt.xlabel('t-SNE Dim 1')
        plt.ylabel('t-SNE Dim 2')
        plt.legend(
            handles=[
                plt.Line2D([0], [0], marker=modal_marker, color='w', label=modal_name,
                           markerfacecolor='gray', markersize=8),
                plt.Line2D([0], [0], marker='x', color='gray', label='Text',
                           linestyle='None', markersize=8),
            ],
            loc='best',
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def _plot_prototype_tsne(modal_proto, proto_label, text_proto,
                             title, modal_name, modal_marker, save_path):
        """
        类中心 t-SNE：每个类别只有一个 modal prototype 和一个 text prototype，
        并用线连接同一类别的两个点。这个更适合展示对齐距离。
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        feature = torch.cat([modal_proto, text_proto], dim=0).numpy()
        embedding = Processor._run_tsne(feature)

        c_num = len(modal_proto)
        modal_2d = embedding[:c_num]
        text_2d = embedding[c_num:]
        labels = proto_label.cpu().numpy()

        cmap = plt.cm.get_cmap('tab20', c_num)
        colors = [cmap(i) for i in range(c_num)]

        plt.figure(figsize=(8, 6))

        for i, c in enumerate(labels):
            plt.scatter(
                modal_2d[i, 0], modal_2d[i, 1],
                c=[colors[i]], marker=modal_marker, s=130,
                alpha=0.85,
            )
            plt.scatter(
                text_2d[i, 0], text_2d[i, 1],
                c=[colors[i]], marker='x', s=170,
                linewidths=2.2,
            )
            plt.plot(
                [modal_2d[i, 0], text_2d[i, 0]],
                [modal_2d[i, 1], text_2d[i, 1]],
                c=colors[i], linewidth=1.2, alpha=0.7,
            )
            plt.text(
                modal_2d[i, 0], modal_2d[i, 1],
                str(int(c)), fontsize=8, alpha=0.75,
            )
            plt.text(
                text_2d[i, 0], text_2d[i, 1],
                str(int(c)), fontsize=8, alpha=0.75,
            )

        plt.title(title)
        plt.xlabel('t-SNE Dim 1')
        plt.ylabel('t-SNE Dim 2')
        plt.legend(
            handles=[
                plt.Line2D([0], [0], marker=modal_marker, color='w', label=modal_name,
                           markerfacecolor='gray', markersize=9),
                plt.Line2D([0], [0], marker='x', color='gray', label='Text',
                           linestyle='None', markersize=9),
            ],
            loc='best',
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def _write_metrics(output_dir, lines, class_rows):
        os.makedirs(output_dir, exist_ok=True)

        txt_path = os.path.join(output_dir, 'alignment_metrics.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

        csv_path = os.path.join(output_dir, 'per_class_alignment.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'class', 'num',
                    'skeleton_text_cosine', 'rgb_text_cosine', 'fusion_text_cosine',
                    'skeleton_text_l2', 'rgb_text_l2', 'fusion_text_l2',
                ]
            )
            writer.writeheader()
            for row in class_rows:
                writer.writerow(row)

        return txt_path, csv_path

    @ex.capture
    def generate_tsne(self, unseen_label, sample_num=0, output_dir='tsne_output'):
        """
        sample_num:
            0 或负数：样本级 t-SNE 画全部测试样本
            正数：每个类别最多画 sample_num 个样本，比如 20
        output_dir:
            输出图片和指标的文件夹
        """
        self.encoder.eval()
        self.adapter.eval()
        self.proj.eval()
        self.fusion.eval()
        self.causal.eval()
        self.mamba_fusion.eval()
        self.temporal_mamba.eval()

        loader = self.data_loader['test']

        all_skeleton = []
        all_rgb = []
        all_fusion = []
        all_label = []
        acc_list = []

        # get_acc 使用 split 内部文本特征，预测后再通过 unseen_label 映射回真实类别。
        unseen_language = self.full_language[unseen_label]
        unseen_language_512 = self.proj(unseen_language)
        unseen_language_512 = F.normalize(unseen_language_512, dim=1).detach().cpu()

        print('[Info] Start extracting Skeleton/RGB/Fusion features for t-SNE...')

        for data, label, rgb in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            rgb_feat = rgb.type(torch.FloatTensor).cuda()

            skeleton_feat = self.encoder(data)
            skeleton_feat = self.adapter(skeleton_feat)

            mamba_fused = self.mamba_fusion(rgb_feat, skeleton_feat)
            causal_out = self.causal(mamba_fused)
            mamba_fused = causal_out['feature']

            acc_batch, pred = get_acc(mamba_fused, unseen_language_512.cuda(), unseen_label, label)
            acc_list.append(acc_batch.detach().cpu())

            sk_single = self._global_pool_feature(skeleton_feat)
            rgb_single = self._global_pool_feature(rgb_feat)
            fusion_single = self._global_pool_feature(mamba_fused)

            sk_single = F.normalize(sk_single, dim=1)
            rgb_single = F.normalize(rgb_single, dim=1)
            fusion_single = F.normalize(fusion_single, dim=1)

            all_skeleton.append(sk_single.detach().cpu())
            all_rgb.append(rgb_single.detach().cpu())
            all_fusion.append(fusion_single.detach().cpu())
            all_label.append(label.detach().cpu())

        all_skeleton = torch.cat(all_skeleton, dim=0)
        all_rgb = torch.cat(all_rgb, dim=0)
        all_fusion = torch.cat(all_fusion, dim=0)
        all_label = torch.cat(all_label, dim=0).long()

        full_text_feature = self.proj(self.full_language)
        full_text_feature = F.normalize(full_text_feature, dim=1).detach().cpu()

        sample_text_from_full = full_text_feature[all_label]
        sample_text_from_split = unseen_language_512[text_index]
        text_mapping_diff = (sample_text_from_full - sample_text_from_split).abs().max().item()

        sample_text = sample_text_from_full

        sk_cos_each = F.cosine_similarity(all_skeleton, sample_text, dim=1)
        rgb_cos_each = F.cosine_similarity(all_rgb, sample_text, dim=1)
        fusion_cos_each = F.cosine_similarity(all_fusion, sample_text, dim=1)

        sk_l2_each = torch.norm(all_skeleton - sample_text, dim=1)
        rgb_l2_each = torch.norm(all_rgb - sample_text, dim=1)
        fusion_l2_each = torch.norm(all_fusion - sample_text, dim=1)

        cos_sk = sk_cos_each.mean().item()
        cos_rgb = rgb_cos_each.mean().item()
        cos_fusion = fusion_cos_each.mean().item()

        euc_sk = sk_l2_each.mean().item()
        euc_rgb = rgb_l2_each.mean().item()
        euc_fusion = fusion_l2_each.mean().item()

        test_acc = torch.stack(acc_list).mean().item() if len(acc_list) > 0 else 0.0
        self.test_acc = test_acc

        class_rows = []
        for c in torch.unique(all_label, sorted=True):
            idx = all_label == c
            row = {
                'class': int(c.item()),
                'num': int(idx.sum().item()),
                'skeleton_text_cosine': float(sk_cos_each[idx].mean().item()),
                'rgb_text_cosine': float(rgb_cos_each[idx].mean().item()),
                'fusion_text_cosine': float(fusion_cos_each[idx].mean().item()),
                'skeleton_text_l2': float(sk_l2_each[idx].mean().item()),
                'rgb_text_l2': float(rgb_l2_each[idx].mean().item()),
                'fusion_text_l2': float(fusion_l2_each[idx].mean().item()),
            }
            class_rows.append(row)

        os.makedirs(output_dir, exist_ok=True)
        metric_path, class_csv_path = self._write_metrics(output_dir, metric_lines, class_rows)

        selected = self._sample_by_class(all_label, sample_num=sample_num, seed=0)

        sk_plot = all_skeleton[selected]
        rgb_plot = all_rgb[selected]
        fusion_plot = all_fusion[selected]
        label_plot = all_label[selected]
        text_label_plot = torch.unique(label_plot, sorted=True).long()
        text_plot = full_text_feature[text_label_plot]

        skeleton_save_path = os.path.join(output_dir, 'Skeleton_Text_TSNE.png')
        rgb_save_path = os.path.join(output_dir, 'RGB_Text_TSNE.png')
        fusion_save_path = os.path.join(output_dir, 'Fusion_Text_TSNE.png')

        self._plot_modal_text_tsne(
            modal_feature=sk_plot,
            modal_label=label_plot,
            text_feature=text_plot,
            text_label=text_label_plot,
            title='Skeleton-Text Alignment',
            modal_name='Skeleton',
            modal_marker='o',
            save_path=skeleton_save_path,
        )

        self._plot_modal_text_tsne(
            modal_feature=rgb_plot,
            modal_label=label_plot,
            text_feature=text_plot,
            text_label=text_label_plot,
            title='RGB-Text Alignment',
            modal_name='RGB',
            modal_marker='s',
            save_path=rgb_save_path,
        )

        self._plot_modal_text_tsne(
            modal_feature=fusion_plot,
            modal_label=label_plot,
            text_feature=text_plot,
            text_label=text_label_plot,
            title='Fusion-Text Alignment',
            modal_name='Fusion',
            modal_marker='D',
            save_path=fusion_save_path,
        )

        # =========================================================
        # F. Prototype-level t-SNE：更适合看“同类模态点与文本点距离”
        # =========================================================
        sk_proto, proto_label = self._get_class_prototype(all_skeleton, all_label)
        rgb_proto, _ = self._get_class_prototype(all_rgb, all_label)
        fusion_proto, _ = self._get_class_prototype(all_fusion, all_label)
        text_proto = full_text_feature[proto_label]
        text_proto = F.normalize(text_proto, dim=1)

        sk_proto_save_path = os.path.join(output_dir, 'Skeleton_Text_Prototype_TSNE.png')
        rgb_proto_save_path = os.path.join(output_dir, 'RGB_Text_Prototype_TSNE.png')
        fusion_proto_save_path = os.path.join(output_dir, 'Fusion_Text_Prototype_TSNE.png')

        self._plot_prototype_tsne(
            modal_proto=sk_proto,
            proto_label=proto_label,
            text_proto=text_proto,
            title='Prototype-level Skeleton-Text Alignment',
            modal_name='Skeleton',
            modal_marker='o',
            save_path=sk_proto_save_path,
        )

        self._plot_prototype_tsne(
            modal_proto=rgb_proto,
            proto_label=proto_label,
            text_proto=text_proto,
            title='Prototype-level RGB-Text Alignment',
            modal_name='RGB',
            modal_marker='s',
            save_path=rgb_proto_save_path,
        )

        self._plot_prototype_tsne(
            modal_proto=fusion_proto,
            proto_label=proto_label,
            text_proto=text_proto,
            title='Prototype-level Fusion-Text Alignment',
            modal_name='Fusion',
            modal_marker='D',
            save_path=fusion_proto_save_path,
        )

        print('[Saved]', skeleton_save_path)
        print('[Saved]', rgb_save_path)
        print('[Saved]', fusion_save_path)
        print('[Saved]', sk_proto_save_path)
        print('[Saved]', rgb_proto_save_path)
        print('[Saved]', fusion_proto_save_path)
        print('[Saved]', metric_path)
        print('[Saved]', class_csv_path)

    def initialize(self):
        self.load_data()
        self.load_model()
        self.log = Log()

    def start(self):
        self.initialize()
        with torch.no_grad():
            self.generate_tsne()


@ex.automain
def main(track):
    p = Processor()
    p.start()
