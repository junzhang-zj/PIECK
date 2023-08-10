import torch
import torch.nn as nn
from parse import args
import numpy as np
from client import FedRecClient
import torch.nn.functional as F

class BaselineAttackClient(FedRecClient):
    def __init__(self, train_ind, m_item, dim):
        super().__init__(train_ind, [], [], m_item, dim)

    def train_(self, items_emb, linear_layers):
        a, b, c, _ = super().train_(items_emb, linear_layers)
        return a, b, c, None

    def eval_(self, _items_emb, _linear_layers):
        return None, None


class AttackClient(nn.Module):
    def __init__(self, target_items, m_item, dim):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)

    def forward(self, user_emb, items_emb, linear_layers):
        user_emb = user_emb.repeat(len(items_emb), 1)
        v = torch.cat((user_emb, items_emb), dim=-1)

        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)

    def train_on_user_emb(self, user_emb, items_emb, linear_layers):
        predictions = self.forward(user_emb.requires_grad_(False), items_emb, linear_layers)
        loss = nn.BCELoss()(predictions, torch.ones(len(self._target_)).to(args.device))
        return loss

    # def train_(self, items_emb, linear_layers)
    def train_(self, items_emb, linear_layers,epoch):
        target_items_emb = items_emb[self._target_].clone().detach()
        target_linear_layers = [[w.clone().detach(), b.clone().detach()] for w, b in linear_layers]
        items_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
        linear_layers = [[w.clone().detach().requires_grad_(True),
                          b.clone().detach().requires_grad_(True)]
                         for (w, b) in linear_layers]
        s = 10
        total_loss = 0
        for _ in range(s):
            nn.init.normal_(self._user_emb.weight, std=0.01)
            if args.attack == 'A-hum':
                for __ in range(30):
                    predictions = self.forward(self._user_emb.weight.requires_grad_(True),
                                               target_items_emb, target_linear_layers)
                    loss = nn.BCELoss()(predictions, torch.zeros(len(self._target_)).to(args.device))

                    self._user_emb.zero_grad()
                    loss.backward()
                    self._user_emb.weight.data.add_(self._user_emb.weight.grad, alpha=-args.lr)
            total_loss += (1 / s) * self.train_on_user_emb(self._user_emb.weight, items_emb, linear_layers)
        total_loss.backward()

        items_emb_grad = items_emb.grad
        linear_layers_grad = [[w.grad, b.grad] for (w, b) in linear_layers]
        return self._target_, items_emb_grad, linear_layers_grad, None

    def eval_(self, _items_emb, _linear_layers):
        return None, None


class PopClient(nn.Module):
    def __init__(self, target_items, m_item, dim):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
        
        self.rank,self.old_items_emb = None,None

    def forward(self, user_emb, items_emb, linear_layers):
        # user_emb = user_emb.repeat(len(items_emb), 1)
        v = torch.cat((user_emb, items_emb), dim=-1)

        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)

    def train_on_user_emb(self, user_emb, items_emb, linear_layers):
        predictions = self.forward(user_emb.requires_grad_(False), items_emb, linear_layers)
        loss = nn.BCELoss()(predictions, torch.ones(len(predictions)).to(args.device))
        return loss

    def train_(self, items_emb, linear_layers,epoch):
        new_items_emb = items_emb.clone().detach()
        items_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
        linear_layers = [[w.clone().detach().requires_grad_(True),
                        b.clone().detach().requires_grad_(True)]
                        for (w, b) in linear_layers] 
        if epoch==1:
            self.old_items_emb = new_items_emb.clone().detach()
            delta_norm = torch.zeros(self.m_item,1).abs().sum(dim=1, keepdim=True)
            delta_norm[self._target_] = - (1 << 10)
            _, self.rank = torch.topk(delta_norm, args.size, dim=0)
        if epoch==2:
            delta_norm = (new_items_emb-self.old_items_emb).norm(2, dim=-1, keepdim=True)
            delta_norm[self._target_] = - (1 << 10)
            _, self.rank = torch.topk(-delta_norm, args.size, dim=0)
        
        user_batch_number=int(10)
        user_batch_size = int(len(self.rank)/user_batch_number)
        total_loss = 0
        for i in range (user_batch_number):
            self._user_emb.weight.data = new_items_emb[self.rank[user_batch_size*(i):user_batch_size*(i+1)]]
            total_loss += (1 / user_batch_number) * self.train_on_user_emb(self._user_emb.weight.data.squeeze(1),items_emb.repeat(user_batch_size,1), linear_layers)
            
        total_loss.backward()
        items_emb_grad = items_emb.grad
        linear_layers_grad = [[w.grad, b.grad] for (w, b) in linear_layers]
        return self._target_, items_emb_grad, linear_layers_grad, None
        
    def eval_(self, _items_emb, _linear_layers):
        return None, None    
        
class ApproxClient(nn.Module):
    def __init__(self, target_items, m_item, dim):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
        self.rank,self.old_items_emb = None,None
        self.weighted = None

    def forward(self, user_emb, items_emb, linear_layers):
        user_emb = user_emb.repeat(len(items_emb), 1)
        v = torch.cat((user_emb, items_emb), dim=-1)

        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)

    def train_on_user_emb(self, user_emb, items_emb, linear_layers):
        predictions = self.forward(user_emb.requires_grad_(False), items_emb, linear_layers)
        loss = nn.BCELoss()(predictions, torch.ones(len(self._target_)).to(args.device))
        return loss


    def train_(self, items_emb, linear_layers,epoch): 
        new_items_emb = items_emb.clone().detach()
        items_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
        linear_layers = [[w.clone().detach().requires_grad_(True),
                        b.clone().detach().requires_grad_(True)]
                        for (w, b) in linear_layers] 
        if epoch==1:
            self.old_items_emb = new_items_emb.clone().detach()
            delta_norm = torch.zeros(self.m_item,1).abs().sum(dim=1, keepdim=True)
            delta_norm[self._target_] = - (1 << 10)
            _, self.rank = torch.topk(delta_norm, args.size, dim=0)
            denominator = np.sum(np.arange(args.size, 0, -1))
            self.weighted =  torch.tensor(np.arange(args.size, 0, -1)/denominator).to(items_emb.device)
        if epoch==2:
            delta_norm = (new_items_emb-self.old_items_emb).norm(2, dim=-1, keepdim=True)
            delta_norm[self._target_] = - (1 << 10)
            _, self.rank = torch.topk(delta_norm, args.size, dim=0)
            denominator = np.sum(np.arange(args.size, 0, -1))
            self.weighted =  torch.tensor(np.arange(args.size, 0, -1)/denominator).to(items_emb.device)
            
        s = args.size
        kl = torch.cosine_similarity(new_items_emb[self.rank.squeeze(1)],items_emb.repeat(s,1),dim=1)
        if kl.mean()>0.95:
            loss_kl = -items_emb.norm(p=2,dim=1) # prevent overfit to generate very small gradients
        else:
            kl_weighted = self.weighted*kl
            pos_kl_weighted = kl_weighted[kl_weighted>0]
            neg_kl_weighted = kl_weighted[kl_weighted<=0]
            pos_num = len(pos_kl_weighted)
            neg_num = len(neg_kl_weighted)
            if args.clients_limit <0.10:
                tau=0.9
            else:
                tau=0.2
            pop_norm = torch.mean(new_items_emb[self.rank.squeeze(1)].norm(p=2,dim=1))
            if (pos_num==s):
                loss_kl = -(torch.sum(pos_kl_weighted)/(pos_num)/tau)
            elif (neg_num==s):
                loss_kl = -(torch.sum(neg_kl_weighted)/(neg_num)/tau)
            else:
                loss_kl = -(torch.sum(pos_kl_weighted)/(pos_num/tau)+torch.sum(neg_kl_weighted)/(neg_num/tau))
                    
        loss_kl.backward()
        items_emb_grad = items_emb.grad
        linear_layers_grad = None       
        return self._target_, items_emb_grad, linear_layers_grad, None

    def eval_(self, _items_emb, _linear_layers):
        return None, None


class PipAttackEB(nn.Module):
    def __init__(self, target_items, m_item, dim):
        super().__init__()
        self._target_ = target_items
        self.m_item = m_item
        self.dim = dim
        self._user_emb = nn.Embedding(1, self.dim)
    
        self.rank,self.old_items_emb = None,None
        
    def forward(self, user_emb, items_emb, linear_layers):
        user_emb = user_emb.repeat(len(items_emb), 1)
        v = torch.cat((user_emb, items_emb), dim=-1)

        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)
    
    def train_on_user_emb(self, user_emb, items_emb, linear_layers):
        predictions = self.forward(user_emb.requires_grad_(True), items_emb, linear_layers)
        loss = nn.BCELoss()(predictions, torch.ones(len(self._target_)).to(args.device))
        return loss

    
    def train_(self,items_emb,linear_layers,epoch):
        new_items_emb = items_emb.clone().detach()
        items_emb = items_emb[self._target_].clone().detach().requires_grad_(True)
        linear_layers = [[w.clone().detach().requires_grad_(True),
                        b.clone().detach().requires_grad_(True)]
                        for (w, b) in linear_layers] 
        self._user_emb.zero_grad()
        loss_eb = self.train_on_user_emb(self._user_emb.weight, items_emb, linear_layers)
        loss_eb.backward()
        items_emb_grad = items_emb.grad
        user_emb_grad = self._user_emb.weight.grad
        self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)
        linear_layers_grad = [[w.grad, b.grad] for (w, b) in linear_layers]
        return self._target_, items_emb_grad, linear_layers_grad, None

    def eval_(self, _items_emb, _linear_layers):
        return None, None

