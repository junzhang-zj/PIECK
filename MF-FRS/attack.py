import torch
import numpy as np
import torch.nn as nn
from parse import args
from data import load_dataset

class PIECKUEA(nn.Module):
    def __init__(self, target_items, m_item, dim):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
        nn.init.normal_(self._user_emb.weight, std=0.01)
        self.rank,self.old_items_emb = None,None
        
    def forward(self, user_emb, items_emb):
        scores = torch.sum(user_emb * items_emb, dim=-1)
        return scores
    
    def train_on_user_emb(self, user_emb, items_emb):
        predictions = self.forward(user_emb.requires_grad_(False),items_emb).sigmoid()
        loss = nn.BCELoss()(predictions, torch.ones(len(predictions)).to(args.device))
        return loss
    
    def train_(self,items_emb,epoch): 
        new_items_emb = items_emb.clone().detach()
        items_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
      
        if epoch==1:
            self.old_items_emb = new_items_emb.clone().detach()
            delta_norm = torch.zeros(self.m_item,1).abs().sum(dim=1, keepdim=True)
            delta_norm[self._target_] = - (1 << 10)
            _, self.rank = torch.topk(delta_norm, args.size, dim=0)
        if epoch==2:
            delta_norm = (new_items_emb-self.old_items_emb).norm(2, dim=-1, keepdim=True)
            delta_norm[self._target_] = - (1 << 10)
            _, self.rank = torch.topk(delta_norm, args.size, dim=0)
            # rank_name = 'log_final/PCA/PopSize-PCA/Rank_'+args.dataset+'_PopSize'+str(args.size)+'_epoch'+str(epoch)+'.npy'
            # self.rank = torch.tensor(np.load(rank_name)).to(items_emb.device)
            # np.save(rank_name,np.array(self.rank.cpu()))
            
        items_emb_grad = 0
        user_batch_size = int(args.size/args.uea_bn)
        for i in range (args.uea_bn):
            for t in range(args.T):
                for _ in range(args.uea_r):
                    total_loss =0
                    self._user_emb.weight.data = new_items_emb[self.rank[user_batch_size*(i):user_batch_size*(i+1)]]
                    total_loss = (1/args.uea_bn) * (1/args.T) * self.train_on_user_emb(self._user_emb.weight.data.squeeze(1),items_emb[t])
                    total_loss.backward()
                    items_emb_grad += (1/args.uea_r)*items_emb.grad
        return self._target_,items_emb_grad, None

    def eval_(self, _items_emb):
        return None, None

class PIECKIPE(nn.Module):
    def __init__(self, target_items, m_item, dim):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
        nn.init.normal_(self._user_emb.weight, std=0.01)
        self.rank, self.old_items_emb = None, None
        self.unpoprank = None
        self.weighted = 0
    def forward(self, user_emb, items_emb):
        scores = torch.sum(user_emb * items_emb, dim=-1)
        return scores
    
    def train_on_user_emb(self, user_emb, items_emb):
        predictions = self.forward(user_emb.requires_grad_(True), items_emb).sigmoid()
        loss = nn.BCELoss()(predictions, torch.ones(len(self._target_)).to(items_emb.device))
        return loss

    def train_(self, items_emb, epoch):  
        new_items_emb = items_emb.clone().detach()
        items_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
      
        if epoch == 1:
            self.old_items_emb = new_items_emb.clone().detach()
            delta_norm = torch.zeros(self.m_item, 1).abs().sum(dim=1, keepdim=True)
            delta_norm[self._target_] = -(1 << 10)
            values, self.rank = torch.topk(delta_norm, args.size, dim=0)
            denominator = np.sum(np.arange(args.size, 0, -1))
            self.weighted =  torch.tensor(np.arange(args.size, 0, -1)/denominator).to(items_emb.device)
        if epoch == 2:
            delta_norm = (new_items_emb - self.old_items_emb).norm(2, dim=-1, keepdim=True)
            delta_norm[self._target_] = -(1 << 10)
            values, self.rank = torch.topk(delta_norm, args.size, dim=0)
            s = args.size
            denominator = np.sum(np.arange(s, 0, -1))
            self.weighted =  torch.tensor(np.arange(s, 0, -1)/denominator).to(items_emb.device)
        items_emb_grad = 0
        total_loss = 0
        for t in range(args.T):
            kl_weighted  = self.weighted*torch.cosine_similarity(new_items_emb[self.rank.squeeze(1)],items_emb[t].repeat(args.size,1),dim=1)
            pos_kl_weighted = kl_weighted[kl_weighted>0]
            neg_kl_weighted = kl_weighted[kl_weighted<=0]
            pos_num = len(pos_kl_weighted)
            neg_num = len(neg_kl_weighted)
            if args.clients_limit <0.10:
                tau=0.9
            else:
                tau=0.5
            if (pos_num==10):
                loss_kl = -(torch.sum(pos_kl_weighted)/(pos_num/tau))
            elif (neg_num==10):
                loss_kl = -(torch.sum(neg_kl_weighted)/(neg_num/tau))
            else:
                loss_kl = -(torch.sum(pos_kl_weighted)/(pos_num/tau)+torch.sum(neg_kl_weighted)/(neg_num/tau))
            total_loss += (1/args.T)*(loss_kl)
        total_loss.backward()
        items_emb_grad = items_emb.grad
        return self._target_, items_emb_grad, None
    
    def eval_(self, _items_emb):
        return None, None

class PipAttackEB(nn.Module):
    def __init__(self, target_items, m_item, dim):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
    
        self.rank,self.old_items_emb = None,None
        
    def forward(self, user_emb, items_emb):
        scores = torch.sum(user_emb * items_emb, dim=-1)
        return scores
    
    def train_on_user_emb(self, user_emb, items_emb):
        predictions = self.forward(user_emb,items_emb).sigmoid()
        loss = nn.BCELoss()(predictions, torch.ones(len(self._target_)).to(args.device))
        return loss
    
    def train_(self,items_emb,epoch):
        new_items_emb = items_emb.clone().detach()
        items_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
        self._user_emb.zero_grad()
        loss_eb = self.train_on_user_emb(self._user_emb.weight, items_emb) 
        loss_eb.backward()
        items_emb_grad = items_emb.grad
        user_emb_grad = self._user_emb.weight.grad
        self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)
        return self._target_, items_emb_grad,None

    def eval_(self, _items_emb):
        return None, None
