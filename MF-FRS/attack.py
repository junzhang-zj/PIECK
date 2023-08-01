import torch
import numpy as np
import torch.nn as nn
from parse import args

class PopClient(nn.Module):
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
        predictions = self.forward(user_emb.requires_grad_(False),items_emb).sigmoid()
        loss = nn.BCELoss()(predictions, torch.ones(len(self._target_)).to(args.device))
        return loss
    
    def train_(self,items_emb,epoch): #改4
        new_items_emb = items_emb.clone().detach()
        items_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
      
        if epoch==1:
            self.old_items_emb = new_items_emb.clone().detach()
            delta_norm = torch.zeros(self.m_item,1).abs().sum(dim=1, keepdim=True)
            delta_norm[self._target_] = - (1 << 10)
            _, self.rank = torch.topk(delta_norm, args.size, dim=0)
        if epoch==2:
            # new_norm = new_items_emb.norm(2, dim=-1, keepdim=True)
            # old_norm =  self.old_items_emb.norm(2, dim=-1, keepdim=True)
            # delta_norm = (new_norm-old_norm).abs()
            delta_norm = (new_items_emb-self.old_items_emb).norm(2, dim=-1, keepdim=True)
            delta_norm[self._target_] = - (1 << 10)
            _, self.rank = torch.topk(delta_norm, args.size, dim=0)
        # else:
        #     new_norm = new_items_emb.norm(2, dim=-1, keepdim=True)
        #     old_norm =  self.old_items_emb.norm(2, dim=-1, keepdim=True)
        #     delta_norm = (new_norm-old_norm).abs()
        #     delta_norm[self._target_] = - (1 << 10)
        #     _, self.rank = torch.topk(delta_norm, 10, dim=0)
        #     self.old_items_emb = new_items_emb.clone().detach()
            
        s = args.size
        total_loss = 0
        for i in range(s):
            self._user_emb.weight.data = new_items_emb[self.rank[i]]
            total_loss += (1 / s) * self.train_on_user_emb(self._user_emb.weight, items_emb)
        total_loss.backward()
        items_emb_grad = items_emb.grad
        
        return self._target_,items_emb_grad, None

    def eval_(self, _items_emb):
        return None, None