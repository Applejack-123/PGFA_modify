#main.py
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

from module.gcn.st_gcnV2 import Model
from module.shift_gcn import Model as ShiftGCN
from module.adapter import Adapter, Linear
from KLLoss import KLLoss, KDLoss
from tool import gen_label, create_logits, get_acc, create_sim_matrix, gen_label_from_text_sim, get_m_theta, get_acc_v2

from cross_mamba import MambaFusion
from module.skeleton_mamba_encoder import SkeletonMambaEncoder
from align_mamba2_fusion import AlignMamba2Fusion
#from Causal import CausalIntervention as Causal
from rgb_only_mlp import RGBOnlyModule as RGBModule
from module.skeleton_guided_rgb_causal_debias import SkeletonGuidedRGBCausalDebias
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0)

# %%
class Processor:

    @ex.capture
    def load_data(self, train_list, train_label, test_list, test_label, train_rgb, test_rgb,  batch_size, language_path):
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
        self.full_language = torch.Tensor(self.full_language)
        self.full_language = self.full_language.cuda()
        self.dataset['train'] = DataSet(train_list, train_label , train_rgb)
        self.dataset['test'] = DataSet(test_list, test_label , test_rgb)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            drop_last=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=64,
            num_workers=16,
            shuffle=False)


    def load_weights(self, model=None, weight_path=None):
        checkpoint = torch.load(weight_path)
        
        if model is self.encoder and 'encoder' in checkpoint:
            model.load_state_dict(checkpoint['encoder'])
        elif model is self.proj and 'proj' in checkpoint:
            model.load_state_dict(checkpoint['proj'])
        elif model is self.rgb_mlp and 'rgb_mlp' in checkpoint:
            model.load_state_dict(checkpoint['rgb_mlp'])
        elif model is self.fusion and 'fusion' in checkpoint:
            missing, unexpected = model.load_state_dict(checkpoint['fusion'],strict=False)
        else:
            raise Exception('cannot found the weight Error!')
                
        
    def adjust_learning_rate(self,optimizer,current_epoch, max_epoch,lr_min=0,lr_max=0.1,warmup_epoch=15, loss_mode='cos', step=[20,30]):
        if current_epoch < warmup_epoch:
            lr = lr_max * (current_epoch+1) / warmup_epoch
        elif loss_mode == 'cos':
            lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        elif loss_mode == 'step':
            lr = lr_max * (0.1 ** np.sum(current_epoch >= np.array(step)))
        else:
            raise Exception('Please check loss_mode!')
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr* param_group.get('lr_scale', 1.0)
            # if i == 0:
            #     param_group['lr'] = lr * 0.1
            # else:
            #     param_group['lr'] = lr
    
    def layernorm(self, feature):

        num = feature.shape[0]
        mean = torch.mean(feature, dim=1).reshape(num, -1)
        var = torch.var(feature, dim=1).reshape(num, -1)
        out = (feature-mean) / torch.sqrt(var)

        return out

    @ex.capture
    def load_model(self, in_channels, hidden_channels, hidden_dim,
                    dropout, graph_args, edge_importance_weighting,
                    visual_size, language_size, weight_path, loss_type,
                    fix_encoder):

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
            nn.Dropout(0.3)
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
        '''
        self.rgb_causal_debias = SkeletonGuidedRGBCausalDebias(
            dim=512,
            hidden_dim=1024,
            dropout=0.1,
            use_temporal_conv=True,
            fusion_mode="gate",
            detach_skeleton_anchor=False
        )
        '''
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

        if loss_type == "kl" or loss_type == "klv2" or loss_type == "kl+cosface" or loss_type == "kl+sphereface":
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
            
        #ckpt=torch.load('./output/model/split_1_skl.pt',map_location='cuda')
        #self.fusion.logit_scale.data.copy_(ckpt["logit_scale"].to("cuda"))
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
        
        self.optimizer = torch.optim.AdamW(
            [
                #{"params": self.encoder.parameters(), "lr": lr, 'lr_scale': 1.0},
                #{"params": self.proj.parameters(), "lr": lr, 'lr_scale': 1.0},
                #{'params': self.rgb_mlp.parameters(),'lr': lr, 'lr_scale': 1.0},
                {"params": self.fusion.parameters(), "lr": lr, 'lr_scale': 1.0},
                #{"params": self.fusion.logit_scale, "lr": lr, 'lr_scale': 1.0},
            ],
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        '''
        self.optimizer = torch.optim.SGD([
            {'params': self.encoder.parameters(),'lr': lr, 'lr_scale': 1.0},
            {'params': self.proj.parameters(),'lr': lr, 'lr_scale': 1.0},
            #{'params': self.rgb_mlp.parameters(),'lr': lr, 'lr_scale': 1.0},
            #{'params': self.fusion.parameters(),'lr': lr, 'lr_scale': 1.0},
            {"params": self.fusion.logit_scale, "lr": lr, 'lr_scale': 1.0},
            ],
             weight_decay=weight_decay,
             momentum=0.9,
             nesterov=False
             )
        '''
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=40, eta_min=1e-6)
        
    @ex.capture
    def optimize(self, epoch_num, DA,split,lr):
        self.log.info("main track")
        self.log.info("split_{}".format(split))
        self.log.info("lr={}".format(lr))
        
        with torch.no_grad():
            self.test_epoch(epoch=-1)
        self.log.info("before train test acc: {}".format(self.test_acc))
        
        for epoch in range(epoch_num):
            self.train_epoch(epoch)

            with torch.no_grad():
                self.test_epoch(epoch=epoch)

            self.log.info("epoch [{}] train loss: {}".format(epoch, self.dim_loss))
            self.log.info("epoch [{}] test acc: {}".format(epoch, self.test_acc))
            self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch, self.best_acc))
            
        #self.scheduler.step()

    @ex.capture
    def train_epoch(self, epoch, lr, loss_mode, step, loss_type, alpha, beta, m, fix_encoder):
        self.encoder.train()
        self.proj.train()
        self.fusion.train()
        self.rgb_mlp.train()
        if fix_encoder:
            self.encoder.eval()
            self.proj.eval()
            self.rgb_mlp.eval()
            #self.fusion.eval()
            
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=50, lr_max=lr, warmup_epoch=5, loss_mode=loss_mode, step=step)
        running_loss = []
        loader = self.data_loader['train']
        for data, label,rgb in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()#128,3,50,25,2
            label_g = gen_label(label)
            label = label.type(torch.LongTensor).cuda()# 128
            seen_language = self.full_language[label] # 128, 768
            seen_language_512 = self.proj(seen_language)# 512
            
            skeleton_feat = self.encoder(data)# 128, 13, 256
            
            rgb_feat = rgb.type(torch.FloatTensor).cuda()# 128, 8, 512
            rgb_feat = self.rgb_mlp(rgb_feat)
            
            out = self.fusion(skeleton_feat,rgb_feat,seen_language_512,labels=label,);fusion_feature = out["fusion_feature"]
            if loss_type == "kl":
                feature = fusion_feature
                logits_per_skl, logits_per_text = create_logits(feature, seen_language_512, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=feature.dtype).cuda()
                # ground_truth = gen_label_from_text_sim(seen_language)
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                cls_loss = (loss_skls + loss_texts) / 2
                #loss = cls_loss
                align_loss = out["loss_align"]
                loss = cls_loss + align_loss
                                
            running_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        running_loss = torch.tensor(running_loss)
        self.dim_loss = running_loss.mean().item()
        
        
    @ex.capture
    def test_epoch(self, unseen_label, epoch, DA, support_factor):
        self.encoder.eval()
        self.proj.eval()
        self.fusion.eval()
        self.rgb_mlp.eval()

        loader = self.data_loader['test']
        y_true = []
        y_pred = []
        acc_list = []
        ent_list = []
        feat_list = []
        old_pred_list = []
        all_labels = []
        for data, label,rgb in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            
            unseen_language = self.full_language[unseen_label]
            unseen_language_512 = self.proj(unseen_language)
            
            all_labels += label.cpu().numpy().tolist()
            skeleton_feat = self.encoder(data)
            
            rgb_feat = rgb.type(torch.FloatTensor).cuda()
            rgb_feat = self.rgb_mlp(rgb_feat)
            
            out = self.fusion(skeleton_feat,rgb_feat,text_feat=None,labels=None,return_align_loss=False,);fusion_feature = out["fusion_feature"]
            
            feat = fusion_feature
            acc_batch, pred = get_acc(feat, unseen_language_512, unseen_label, label)
            acc_list.append(acc_batch)

        print("test unique labels:", sorted(set(all_labels)))
        print("unseen_label:", unseen_label)
        print("test labels not in unseen_label:", sorted(set(all_labels) - set(unseen_label)))
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
        torch.save({'encoder':self.encoder.state_dict(),
                    #'adapter':self.adapter.state_dict(),
                    'proj': self.proj.state_dict(),
                    'rgb_mlp': self.rgb_mlp.state_dict(),
                    'fusion': self.fusion.state_dict(),
                   }, save_path)

    def start(self):
        self.initialize()
        self.optimize()
        #self.save_model()


# %%
@ex.automain
def main(track):
    p = Processor()
    p.start()
