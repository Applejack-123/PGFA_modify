#mytest.py
from config import *
from dataset import DataSet 
from logger import Log

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import pi, cos
from tqdm import tqdm

from module.gcn.st_gcn import Model
from module.shift_gcn import Model as ShiftGCN
from module.adapter import Adapter, Linear
from KLLoss import KLLoss, KDLoss
from tool import gen_label, create_logits, get_acc, create_sim_matrix, gen_label_from_text_sim, get_m_theta, get_acc_v2
from module.cross_attention_fusion import CrossAttentionFusion

from Causal import CausalIntervention as Causal

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0)

test_list = "../PGFA/data/zeroshot/ntu60/split_1/unseen_data.npy"
test_label = "../PGFA/data/zeroshot/ntu60/split_1/unseen_label.npy"
test_rgb = "../../datasets/clip_temporal/unseen_temporal_rgb.npy"
datasett = DataSet(test_list, test_label,test_rgb)
loader = torch.utils.data.DataLoader(
            dataset=datasett,
            batch_size=128,
            num_workers=16,
            shuffle=True,
            drop_last=True)

encoder = Model(in_channels=3, hidden_channels=16,
               hidden_dim=256,dropout=0.5, 
               graph_args={
               "layout" : 'ntu-rgb+d',
               "strategy" : 'spatial'
               },
               edge_importance_weighting=True,
               )
encoder = encoder.cuda()

for data, label,rgb in tqdm(loader):
    data = data.type(torch.FloatTensor).cuda()
    print('data:',data.shape)
    feat = encoder(data)
    print('feat:',feat.shape)
    
    
    '''
data: torch.Size([128, 3, 50, 25, 2])
      torch.Size([256, 256, 13, 25])
      torch.Size([256, 256, 1, 1])
feat: torch.Size([128, 256])
    '''