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

# %%
class Processor:

    @ex.capture
    def load_data(self, train_list, train_label, test_list, test_label,train_rgb, test_rgb,  batch_size, language_path):
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
        self.dataset['train'] = DataSet(train_list, train_label,train_rgb)
        self.dataset['test'] = DataSet(test_list, test_label,test_rgb)

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

    # def load_weights(self, model=None, weight_path=None):
    #     pretrained_dict = torch.load(weight_path)
    #     model.load_state_dict(pretrained_dict)
        '''
    def load_weights(self, model=None, weight_path=None):
        #pretrained_dict = torch.load(weight_path)
        checkpoint = torch.load(weight_path) #+++
        key = ('encoder' if model is self.encoder 
               else 'adapter' if model is self.adapter 
               else 'proj' 
               )
        pretrained_dict = checkpoint[key]#+++
        model.load_state_dict(pretrained_dict)
        '''
    def load_weights(self, model=None, weight_path=None):
        checkpoint = torch.load(weight_path)
    
        # 如果传入 model，只加载对应模块
        if model is not None:
            if model is self.encoder and 'encoder' in checkpoint:
                model.load_state_dict(checkpoint['encoder'])
            elif model is self.adapter and 'adapter' in checkpoint:
                model.load_state_dict(checkpoint['adapter'])
            elif model is self.proj and 'proj' in checkpoint:
                model.load_state_dict(checkpoint['proj'])
            elif model is self.fusion and 'fusion' in checkpoint:
                model.load_state_dict(checkpoint['fusion'])
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            # 不传入 model，加载所有模块
            if 'encoder' in checkpoint:
                self.encoder.load_state_dict(checkpoint['encoder'])
            if 'adapter' in checkpoint:
                self.adapter.load_state_dict(checkpoint['adapter'])
            if 'proj' in checkpoint:
                self.proj.load_state_dict(checkpoint['proj'])
            if 'fusion' in checkpoint:
                self.fusion.load_state_dict(checkpoint['fusion'])
        
    def adjust_learning_rate(self,optimizer,current_epoch, max_epoch,lr_min=0,lr_max=0.1,warmup_epoch=15, loss_mode='step', step=[50, 80]):

        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        elif loss_mode == 'cos':
            lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        elif loss_mode == 'step':
            lr = lr_max * (0.1 ** np.sum(current_epoch >= np.array(step)))
        else:
            raise Exception('Please check loss_mode!')
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
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
    def load_model(self,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting, visual_size, language_size, weight_path, loss_type, fix_encoder, finetune):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                            hidden_dim=hidden_dim,dropout=dropout, 
                            graph_args=graph_args,
                            edge_importance_weighting=edge_importance_weighting,
                            )
        self.encoder = self.encoder.cuda()
        self.adapter = Linear().cuda()
        #self.adapter = Adapter(hidden_size=256, output_size=512).cuda()

        self.proj = nn.Sequential(
        nn.LayerNorm(language_size),
        nn.Linear(language_size, 512),
        nn.ReLU(),
        nn.Dropout(0.3)
        ).cuda()

        self.fusion = CrossAttentionFusion(
        feature_dim=512,
        num_heads=8,
        dropout=0.3
        ).cuda()

        self.causal = Causal(
            feature_dim=512, 
            state_size=512
        ).cuda()
        
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
        
        if fix_encoder or finetune:
            self.load_weights(self.encoder, weight_path)
            self.load_weights(self.adapter, weight_path)
            self.load_weights(self.proj, weight_path)
            self.load_weights(self.fusion, weight_path)
        #++++++++++
        if fix_encoder:# 冻结！
            for param in self.encoder.parameters():
                param.requires_grad = False  
            for param in self.adapter.parameters():
                param.requires_grad = False
            for param in self.proj.parameters():
                param.requires_grad = False
            for param in self.fusion.parameters():
                param.requires_grad = False
        #++++++++++


    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):
        self.optimizer = torch.optim.SGD([
            {'params': self.encoder.parameters()},
            {'params': self.adapter.parameters()},
            {'params': self.proj.parameters()},  # 新增
            {'params': self.fusion.parameters()},
            #{'params': self.causal.parameters()}# 新增
            ],
             lr=lr,
             weight_decay=weight_decay,
             momentum=0.9,
             nesterov=False
             )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 100)

    @ex.capture
    def optimize(self, epoch_num, DA): # print -> log.info
        self.log.info("main track") 
        for epoch in range(epoch_num):
            self.train_epoch(epoch)
            with torch.no_grad():
                self.test_epoch(epoch=epoch)
            self.log.info("epoch [{}] train loss: {}".format(epoch,self.dim_loss))
            self.log.info("epoch [{}] test acc: {}".format(epoch,self.test_acc))
            self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))
            if DA:
                self.log.info("epoch [{}] DA test acc: {}".format(epoch,self.test_aug_acc))
                self.log.info("epoch [{}] gets the best DA acc: {}".format(self.best_aug_epoch,self.best_aug_acc))
            # if epoch > 5:
            #     self.log.info("epoch [{}] test acc: {}".format(epoch,self.test_acc))
            #     self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))
            # else:
            #     self.log.info("epoch [{}] : warm up epoch.".format(epoch))

    @ex.capture
    def train_epoch(self, epoch, lr, loss_mode, step, loss_type, alpha, beta, m, fix_encoder):
        self.encoder.train() # eval -> train
        self.adapter.train()
        self.proj.train()
        self.fusion.train()
        #self.causal.train()
        if fix_encoder:
            self.encoder.eval()
            self.adapter.train()
            self.proj.train()
            self.fusion.train()
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=100, lr_max=lr, warmup_epoch=5, loss_mode=loss_mode, step=step)
        running_loss = []
        loader = self.data_loader['train']
        for data, label,rgb in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            #print(data.shape) #128,3,50,25,2
            # label = label.type(torch.LongTensor).cuda()
            label_g = gen_label(label)
            label = label.type(torch.LongTensor).cuda()
            # print(label.shape) # 128
            # print(label) # int
            seen_language = self.full_language[label] # 128, 768
            seen_language_512=self.proj(seen_language)# 512
           
            # print(seen_language.shape)
            
            feat = self.encoder(data)
            if fix_encoder:
                feat = feat.detach()
            skleton_feat = self.adapter(feat)
            rgb_feat = rgb.type(torch.FloatTensor).cuda()

            fused_512 = self.fusion(skleton_feat,rgb_feat)
            #fused_512 = self.causal(fused_512)

            if loss_type == "kl":
                logits_per_skl, logits_per_text = create_logits(fused_512, seen_language_512, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                # ground_truth = gen_label_from_text_sim(seen_language)
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                loss = (loss_skls + loss_texts) / 2
            elif loss_type == "kl+margin":
                logits_per_skl, logits_per_text = create_logits(fused_512, seen_language_512, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                ones = torch.ones_like(ground_truth).cuda()
                ones -= m
                logits_per_skl = torch.where(ones<logits_per_skl, ones, logits_per_skl) * ground_truth + (1-ground_truth) * logits_per_skl
                logits_per_text = torch.where(ones<logits_per_text, ones, logits_per_text) * ground_truth + (1-ground_truth) * logits_per_text
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                loss = (loss_skls + loss_texts) / 2
                
            elif loss_type == "kl+cosface":
                logits_per_skl, logits_per_text = create_logits(fused_512, seen_language_512, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                logits_per_skl -= ground_truth * m
                logits_per_text -= ground_truth * m
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                loss = (loss_skls + loss_texts) / 2
            elif loss_type == "kl+sphereface":
                logits_per_skl, logits_per_text = create_logits(fused_512, seen_language_512, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                logits_per_skl = get_m_theta(logits_per_skl, m) * ground_truth + (1-ground_truth) * logits_per_skl
                logits_per_text = get_m_theta(logits_per_text, m) * ground_truth + (1-ground_truth) * logits_per_text
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                loss = (loss_skls + loss_texts) / 2
            elif loss_type == "klv2":
                logits_per_skl, logits_per_text = create_logits(fused_512, seen_language_512, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                # ground_truth = gen_label_from_text_sim(seen_language)
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                logit_skl_skl, logit_skl_skl_2 = create_logits(fused_512, skleton_feat, self.logit_scale, exp=True)
                loss_skl_skl = self.loss(logit_skl_skl, ground_truth)
                loss = alpha * (loss_skls + loss_texts) / 2 + beta * loss_skl_skl
                # loss = (loss_skls + loss_texts + loss_skl_skl)/3

            elif loss_type == "mse":
                skl_skl_sim, skl_text_sim, text_text_sim = create_sim_matrix(fused_512, seen_language_512)
                loss = self.loss(skl_text_sim, text_text_sim)*skl_text_sim.shape[0]
            elif loss_type == "kl+mse":
                logits_per_skl, logits_per_text = create_logits(fused_512, seen_language_512, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                loss_skls = self.loss_kl(logits_per_skl, ground_truth)
                loss_texts = self.loss_kl(logits_per_text, ground_truth)
                loss_kl = (loss_skls + loss_texts) / 2
                skl_skl_sim, skl_text_sim, text_text_sim = create_sim_matrix(fused_512, seen_language_512)
                loss_mse = self.loss_mse(skl_text_sim, text_text_sim) #* skl_text_sim.shape[0]
                # loss_mse += self.loss_mse(skl_skl_sim, text_text_sim) #* skl_text_sim.shape[0]
                loss = alpha * loss_kl + beta * loss_mse
            elif loss_type == "kl+kd":
                margin = 0.5
                logits_per_skl, logits_per_text = create_logits(fused_512, seen_language_512, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda() # one-hot
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                loss = (loss_skls + loss_texts) / 2
                
                logits_per_skl_v2, logits_per_text_v2 = create_logits(fused_512, seen_language_512, logit_scale=1, exp=False)
                # logits_per_skl_v2, logits_per_text_v2 = logits_per_skl, logits_per_text
                ground_truth_v2 = gen_label_from_text_sim(seen_language_512) + (ground_truth - 1) * margin # teacher logit
                loss_skls_v2 = self.kd_loss(logits_per_skl_v2, ground_truth_v2)
                loss_texts_v2 = self.kd_loss(logits_per_text_v2, ground_truth_v2)
                kd_loss = (loss_skls_v2 + loss_texts_v2) / 2
                loss = alpha * loss + beta * kd_loss
            else:
                raise Exception('loss_type Error!')
                
            running_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        running_loss = torch.tensor(running_loss)
        self.dim_loss = running_loss.mean().item()

    @ex.capture
    def test_epoch(self, unseen_label, epoch, DA, support_factor):
        self.encoder.eval()
        self.adapter.eval()
        self.proj.eval()
        self.fusion.eval()
        #self.causal.eval()

        loader = self.data_loader['test']
        y_true = []
        y_pred = []
        acc_list = []
        ent_list = []
        feat_list = []
        old_pred_list = []
        for data, label,rgb in tqdm(loader):

            # y_t = label.numpy().tolist()
            # y_true += y_t

            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            rgb_feat = rgb.type(torch.FloatTensor).cuda()

            unseen_language = self.full_language[unseen_label]
            unseen_language_512=self.proj(unseen_language)
            # inference
            feature = self.encoder(data)
            #print("after encoder:",feature.shape)
            feature = self.adapter(feature)
            #print("after adapter:",feature.shape)

            fused_512=self.fusion(feature,rgb_feat)
            #fused_512 = self.causal(fused_512)
            
            if DA:
            # acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)
                acc_batch, pred, old_pred, ent, feat = get_acc_v2(fused_512, unseen_language_512, unseen_label, label)
                ent_list.append(ent)
                feat_list.append(feat)
                old_pred_list.append(old_pred)
            else:
                acc_batch, pred = get_acc(fused_512, unseen_language_512, unseen_label, label)
        
            # y_p = pred.cpu().numpy().tolist()
            # y_pred += y_p


            acc_list.append(acc_batch)

        acc_list = torch.tensor(acc_list)
        acc = acc_list.mean()
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch
            self.save_model()
            # y_true = np.array(y_true)
            # y_pred = np.array(y_pred)
            # np.save("y_true_3.npy",y_true)
            # np.save("y_pred_3.npy",y_pred)
            # print("save ok!")
        self.test_acc = acc
        
        if DA:
            ent_all = torch.cat(ent_list)
            feat_all = torch.cat(feat_list)
            old_pred_all = torch.cat(old_pred_list)
            z_list = []
            for i in range(len(unseen_label)):
                mask = old_pred_all == i
                class_support_set = feat_all[mask]
                class_ent = ent_all[mask]
                class_len = class_ent.shape[0]
                if int(class_len*support_factor) < 1:
                    z = self.full_language[unseen_label[i:i+1]]
                else:
                    _, indices = torch.topk(-class_ent, int(class_len*support_factor))
                    z = torch.mean(class_support_set[indices], dim=0, keepdim=True)
                z_list.append(z)
                
            z_tensor = torch.cat(z_list)
            aug_acc_list = []
            for data, label,rgb in tqdm(loader):
                # y_t = label.numpy().tolist()
                # y_true += y_t

                data = data.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                rgb_feat = rgb.type(torch.FloatTensor).cuda()
                unseen_language = z_tensor
                unseen_language_512=self.proj(unseen_language)
                # inference
                feature = self.encoder(data)
                feature = self.adapter(feature)

                fused_512=self.fusion(feature,rgb_feat)
                
                # acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)
                acc_batch, pred = get_acc(fused_512, unseen_language_512, unseen_label, label)
            
                # y_p = pred.cpu().numpy().tolist()
                # y_pred += y_p
                aug_acc_list.append(acc_batch)
            aug_acc = torch.tensor(aug_acc_list).mean()
            if aug_acc > self.best_aug_acc:
                self.best_aug_acc = aug_acc
                self.best_aug_epoch = epoch
            self.test_aug_acc = aug_acc
            


    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    @ex.capture
    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'encoder':self.encoder.state_dict(),
                    'adapter':self.adapter.state_dict(),
                    'proj': self.proj.state_dict(),
                    'fusion': self.fusion.state_dict(), }, save_path)

    def start(self):
        self.initialize()
        self.optimize()
        #self.save_model()


# %%
@ex.automain
def main(track):
    p = Processor()
    p.start()
