# evaluate_fusion.py
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, ConcatDataset
import os
import json

# 导入您的模块
from dataset import DataSet
from module.adapter import Adapter
from module.cross_attention_fusion import CrossAttentionFusion
from module.gcn.st_gcn import Model

# ========== 配置参数 ==========
DATASET = 'ntu60'
SPLIT = '1'

# 数据路径
DATA_BASE = f'./data/zeroshot/{DATASET}/split_{SPLIT}/'
LANGUAGE_PATH = f'./data/language/{DATASET}_des_embeddings.npy'

# 具体文件路径
SEEN_TRAIN_DATA = DATA_BASE + 'seen_train_data.npy'
SEEN_TRAIN_LABEL = DATA_BASE + 'seen_train_label.npy'
SEEN_TRAIN_RGB = DATA_BASE + 'seen_train_data_rgb.npy'

SEEN_TEST_DATA = DATA_BASE + 'seen_test_data.npy'
SEEN_TEST_LABEL = DATA_BASE + 'seen_test_label.npy'
SEEN_TEST_RGB = DATA_BASE + 'seen_test_data_rgb.npy'

UNSEEN_TEST_DATA = DATA_BASE + 'unseen_data.npy'
UNSEEN_TEST_LABEL = DATA_BASE + 'unseen_label.npy'
UNSEEN_TEST_RGB = DATA_BASE + 'unseen_data_rgb.npy'

# 模型路径
ENCODER_PATH = None
ADAPTER_PATH = None
FUSION_PATH = None

# 模型参数
IN_CHANNELS = 3
HIDDEN_CHANNELS = 64
HIDDEN_DIM = 256
GRAPH_ARGS = {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
EDGE_IMPORTANCE_WEIGHTING = True

# 评估参数
BATCH_SIZE = 64
FEATURE_DIM = 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")
print(f"Dataset: {DATASET}, Split: {SPLIT}")

# ========== 获取类别信息 ==========
seen_train_label = np.load(SEEN_TRAIN_LABEL)
seen_classes = np.unique(seen_train_label)

if os.path.exists(UNSEEN_TEST_LABEL):
    unseen_label = np.load(UNSEEN_TEST_LABEL)
    unseen_classes = np.unique(unseen_label)
else:
    unseen_classes = []

print(f"Seen classes: {len(seen_classes)}")
print(f"Unseen classes: {len(unseen_classes)}")

# ========== 加载测试数据 ==========
print("\nLoading test data...")

test_datasets = []

# 加载seen测试数据
print("  Loading seen test data...")
seen_test = DataSet(SEEN_TEST_DATA, SEEN_TEST_LABEL, SEEN_TEST_RGB)
test_datasets.append(seen_test)
print(f"    Loaded {len(seen_test)} seen test samples")

# 加载unseen测试数据
print("  Loading unseen test data...")
unseen_test = DataSet(UNSEEN_TEST_DATA, UNSEEN_TEST_LABEL, UNSEEN_TEST_RGB)
test_datasets.append(unseen_test)
print(f"    Loaded {len(unseen_test)} unseen test samples")

# 合并数据集
test_dataset = ConcatDataset(test_datasets)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"\nTotal test samples: {len(test_dataset)}")

# ========== 加载文本特征并映射到512维 ==========
print("\nLoading and mapping text features...")
all_text_features = np.load(LANGUAGE_PATH)  # [60, 768]
print(f"Original text features shape: {all_text_features.shape}")

# 创建文本映射层: 768 -> 512
class TextMapping(nn.Module):
    def __init__(self, input_dim=768, output_dim=512):
        super(TextMapping, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.mapping(x)

text_mapping = TextMapping(input_dim=768, output_dim=FEATURE_DIM).to(DEVICE)
text_mapping.eval()

# 获取seen和unseen的文本特征并映射
with torch.no_grad():
    seen_text_features = torch.from_numpy(all_text_features[seen_classes]).float().to(DEVICE)
    seen_text_features = text_mapping(seen_text_features)  # [55, 512]
    seen_text_features = seen_text_features.cpu().numpy()  # 使用 .detach() 自动处理
    seen_text_features = seen_text_features / (np.linalg.norm(seen_text_features, axis=1, keepdims=True) + 1e-8)

    if len(unseen_classes) > 0:
        unseen_text_features = torch.from_numpy(all_text_features[unseen_classes]).float().to(DEVICE)
        unseen_text_features = text_mapping(unseen_text_features)  # [5, 512]
        unseen_text_features = unseen_text_features.cpu().numpy()
        unseen_text_features = unseen_text_features / (np.linalg.norm(unseen_text_features, axis=1, keepdims=True) + 1e-8)
    else:
        unseen_text_features = np.array([])

print(f"Seen text features (512-dim) shape: {seen_text_features.shape}")
if len(unseen_classes) > 0:
    print(f"Unseen text features (512-dim) shape: {unseen_text_features.shape}")

# ========== 加载模型 ==========
print("\nLoading models...")

# 1. 骨架编码器
encoder = Model(
    in_channels=IN_CHANNELS,
    hidden_channels=HIDDEN_CHANNELS,
    hidden_dim=HIDDEN_DIM,
    graph_args=GRAPH_ARGS,
    edge_importance_weighting=EDGE_IMPORTANCE_WEIGHTING
).to(DEVICE)

if ENCODER_PATH and os.path.exists(ENCODER_PATH):
    encoder.load_state_dict(torch.load(ENCODER_PATH))
    print(f"  Loaded encoder from {ENCODER_PATH}")
else:
    print("  Warning: No encoder loaded, using random weights!")

encoder.eval()

# 2. Adapter: 256 -> 512
adapter = Adapter(hidden_size=HIDDEN_DIM, output_size=FEATURE_DIM).to(DEVICE)
if ADAPTER_PATH and os.path.exists(ADAPTER_PATH):
    adapter.load_state_dict(torch.load(ADAPTER_PATH))
    print(f"  Loaded adapter from {ADAPTER_PATH}")
else:
    print("  Warning: No adapter loaded, using random weights!")
adapter.eval()

# 3. 融合模块
fusion = CrossAttentionFusion(feature_dim=FEATURE_DIM, num_heads=8, dropout=0.1).to(DEVICE)
if FUSION_PATH and os.path.exists(FUSION_PATH):
    fusion.load_state_dict(torch.load(FUSION_PATH))
    print(f"  Loaded fusion from {FUSION_PATH}")
else:
    print("  Warning: No fusion loaded, using random weights!")
fusion.eval()

print("\nAll models loaded successfully!")

# ========== 评估函数 ==========
def evaluate(loader, use_fusion=True):
    """评估特征"""
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, label, rgb in tqdm(loader, desc=f"Evaluating {'fusion' if use_fusion else 'skeleton'}"):
            # 数据移到GPU
            data = data.type(torch.FloatTensor).to(DEVICE)
            rgb_feat = rgb.type(torch.FloatTensor).to(DEVICE)
            
            # 1. 提取骨架特征: [B, 3, 50, 25, 2] -> [B, 256]
            skeleton_feat = encoder(data)
            
            # 2. 通过adapter映射到512维: [B, 256] -> [B, 512]
            skeleton_512 = adapter(skeleton_feat)
            
            # 3. 融合（如果需要）
            if use_fusion and fusion is not None:
                features = fusion(skeleton_512, rgb_feat)  # [B, 512]
            else:
                features = skeleton_512  # [B, 512]
            
            all_features.append(features.cpu())
            all_labels.append(label)
    
    features = torch.cat(all_features, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # 归一化
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    results = {}
    
    # 1. 常规ZSL（只评估unseen类）
    if len(unseen_text_features) > 0:
        similarity = np.dot(features_norm, unseen_text_features.T)
        preds = np.argmax(similarity, axis=1)
        
        unseen_mask = np.isin(labels, unseen_classes)
        if unseen_mask.sum() > 0:
            test_labels = labels[unseen_mask]
            preds_unseen = preds[unseen_mask]
            
            label_mapping = {idx: i for i, idx in enumerate(unseen_classes)}
            test_labels_mapped = np.array([label_mapping[l] for l in test_labels])
            
            acc = accuracy_score(test_labels_mapped, preds_unseen)
            results['zsl_acc'] = acc
            print(f"  ZSL Acc (unseen only): {acc:.4f} ({unseen_mask.sum()} samples)")
    
    # 2. 广义ZSL（seen + unseen）
    if len(unseen_text_features) > 0:
        all_text = np.vstack([seen_text_features, unseen_text_features])
        all_classes = list(seen_classes) + list(unseen_classes)
    else:
        all_text = seen_text_features
        all_classes = list(seen_classes)
    
    similarity_all = np.dot(features_norm, all_text.T)
    preds_all = np.argmax(similarity_all, axis=1)
    
    label_to_idx = {label: i for i, label in enumerate(all_classes)}
    labels_mapped = np.array([label_to_idx[l] for l in labels])
    
    results['gzsl_acc'] = accuracy_score(labels_mapped, preds_all)
    print(f"  GZSL Acc (all classes): {results['gzsl_acc']:.4f}")
    
    # 3. 分别计算seen和unseen的准确率
    if len(unseen_classes) > 0:
        seen_mask = np.isin(labels, seen_classes)
        unseen_mask = np.isin(labels, unseen_classes)
        
        if seen_mask.sum() > 0:
            seen_acc = accuracy_score(labels_mapped[seen_mask], preds_all[seen_mask])
            results['seen_acc'] = seen_acc
            print(f"  Seen Acc: {seen_acc:.4f} ({seen_mask.sum()} samples)")
        
        if unseen_mask.sum() > 0:
            unseen_acc = accuracy_score(labels_mapped[unseen_mask], preds_all[unseen_mask])
            results['unseen_acc'] = unseen_acc
            print(f"  Unseen Acc: {unseen_acc:.4f} ({unseen_mask.sum()} samples)")
        
        if 'seen_acc' in results and 'unseen_acc' in results:
            h_score = 2 * results['seen_acc'] * results['unseen_acc'] / (results['seen_acc'] + results['unseen_acc'] + 1e-8)
            results['h_score'] = h_score
            print(f"  H-Score: {h_score:.4f}")
    
    return results

# ========== 运行评估 ==========
print("\n" + "="*60)
print("EVALUATING SKELETON FEATURES (256 -> 512 via Adapter)")
print("="*60)
skeleton_results = evaluate(test_loader, use_fusion=False)

print("\n" + "="*60)
print("EVALUATING FUSED FEATURES (Skeleton + RGB via Cross Attention)")
print("="*60)
fused_results = evaluate(test_loader, use_fusion=True)

# ========== 输出结果 ==========
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print(f"\n{'Metric':<15} {'Skeleton':<12} {'Fused':<12} {'Improvement':<12}")
print("-" * 51)

metrics = ['zsl_acc', 'gzsl_acc', 'seen_acc', 'unseen_acc', 'h_score']
improvements = {}

for metric in metrics:
    sk_val = skeleton_results.get(metric, None)
    fu_val = fused_results.get(metric, None)
    
    if sk_val is not None and fu_val is not None:
        improvement = fu_val - sk_val
        improvements[metric] = improvement
        print(f"{metric.upper():<15} {sk_val:<12.4f} {fu_val:<12.4f} {improvement:+.4f}")

# 判断
print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if 'zsl_acc' in improvements:
    if improvements['zsl_acc'] > 0.01:
        print("✓ Fusion significantly improves ZSL performance!")
    elif improvements['zsl_acc'] > 0:
        print("✓ Fusion slightly improves ZSL performance")
    elif improvements['zsl_acc'] < -0.01:
        print("✗ Fusion degrades ZSL performance")
    else:
        print("→ Fusion has minimal effect on ZSL performance")

if 'h_score' in improvements:
    if improvements['h_score'] > 0.01:
        print("✓ Fusion improves H-score (better balance between seen/unseen)")
    elif improvements['h_score'] < -0.01:
        print("✗ Fusion degrades H-score")
    else:
        print("→ Fusion has minimal effect on H-score")

# 保存结果
results = {
    'skeleton': {k: float(v) for k, v in skeleton_results.items()},
    'fused': {k: float(v) for k, v in fused_results.items()},
    'improvements': {k: float(v) for k, v in improvements.items()},
    'config': {
        'dataset': DATASET,
        'split': SPLIT,
        'batch_size': BATCH_SIZE,
        'feature_dim': FEATURE_DIM
    }
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to results.json")