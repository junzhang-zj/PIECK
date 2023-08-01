import torch
import torch.nn as nn
import numpy as np
from parse import args
from evaluate import evaluate_precision, evaluate_recall, evaluate_ndcg

class FedRecClient(nn.Module):
    def __init__(self, train_ind, test_ind, target_ind, m_item, dim):
        super().__init__()
        self._train_ = train_ind
        self._test_ = test_ind
        self._target_ = []
        self.m_item = m_item
        self.dim = dim

        for i in target_ind:
            if i not in train_ind and i not in test_ind:
                self._target_.append(i)

        items, labels = [], []
        for pos_item in train_ind:
            items.append(pos_item)
            labels.append(1.)

            for _ in range(args.num_neg):
                neg_item = np.random.randint(m_item)
                while neg_item in train_ind:
                    neg_item = np.random.randint(m_item)
                items.append(neg_item)
                labels.append(0.)

        self._train_items = torch.Tensor(items).long()
        self._train_labels = torch.Tensor(labels).to(args.device)
        self._user_emb = nn.Embedding(1, dim)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=5e-1)
        nn.init.normal_(self._user_emb.weight, std=0.01)

    def forward(self, items_emb, linear_layers):
        user_emb = self._user_emb.weight.repeat(len(items_emb), 1)
        v = torch.cat((user_emb,items_emb), dim=-1)

        for i, (w, b) in enumerate(linear_layers):
            
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)

    def train_(self, items_emb, linear_layers, epoch):
        items_emb = items_emb[self._train_items].clone().detach().requires_grad_(True)
        linear_layers = [(w.clone().detach().requires_grad_(True),
                          b.clone().detach().requires_grad_(True))
                         for (w, b) in linear_layers]
        self._user_emb.zero_grad()
        predictions = self.forward(items_emb, linear_layers)
        loss = nn.BCELoss()(predictions, self._train_labels)
        loss.backward()
        user_emb_grad = self._user_emb.weight.grad
        self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)
        items_emb_grad = items_emb.grad
        linear_layers_grad = [[w.grad, b.grad] for (w, b) in linear_layers]
        return self._train_items, items_emb_grad, linear_layers_grad, loss.cpu().item()

    def eval_(self, items_emb, linear_layers):
        rating = self.forward(items_emb, linear_layers)
        if self._test_:
            ground_truth = np.random.randint(100)
            items = []
            for _ in range(99):
                neg_item = np.random.randint(self.m_item)
                while neg_item in self._train_ or neg_item in items:
                    neg_item = np.random.randint(self.m_item)
                items.append(neg_item)
            items = items[:ground_truth] + [self._test_[0]] + items[ground_truth:]
            items = torch.Tensor(items).long().to(args.device)
            sampled_hr_at_5 = evaluate_recall(rating[items], [ground_truth], 5)
            sampled_hr_at_10 = evaluate_recall(rating[items], [ground_truth], 10)
            sampled_hr_at_20 = evaluate_recall(rating[items], [ground_truth], 20)
            test_result = np.array([sampled_hr_at_5,sampled_hr_at_10,sampled_hr_at_20])
        else:
            test_result = None

        if self._target_:
            rating[self._train_] = - (1 << 10)
            er_at_5 = evaluate_recall(rating, self._target_, 5)
            er_at_10 = evaluate_recall(rating, self._target_, 10)
            er_at_20 = evaluate_recall(rating, self._target_, 20)
            target_result = np.array([er_at_5,er_at_10,er_at_20])
        else:
            target_result = None

        return test_result, target_result


class FedRecClientDefense(nn.Module):
    def __init__(self, train_ind, test_ind, target_ind, m_item, dim):
        super().__init__()
        self._train_ = train_ind
        self._test_ = test_ind
        self._target_ = []
        self.m_item = m_item
        self.dim = dim

        for i in target_ind:
            if i not in train_ind and i not in test_ind:
                self._target_.append(i)

        items, labels = [], []
        for pos_item in train_ind:
            items.append(pos_item)
            labels.append(1.)

            for _ in range(args.num_neg):
                neg_item = np.random.randint(m_item)
                while neg_item in train_ind:
                    neg_item = np.random.randint(m_item)
                items.append(neg_item)
                labels.append(0.)

        self._train_items = torch.Tensor(items).long()
        self._train_labels = torch.Tensor(labels).to(args.device)
        self._user_emb = nn.Embedding(1, dim)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=5e-1)
        nn.init.normal_(self._user_emb.weight, std=0.01)
        
        self.pop_indices,self.unpop_indices=[],[]
        self.unrank =None
        self.delta_value = None
        
    def forward(self, items_emb, linear_layers):
        user_emb = self._user_emb.weight.repeat(len(items_emb), 1)
        v = torch.cat((user_emb,items_emb), dim=-1)

        for i, (w, b) in enumerate(linear_layers):
            v = v @ w.t() + b
            if i < len(linear_layers) - 1:
                v = v.relu()
            else:
                v = v.sigmoid()
        return v.view(-1)

    def train_(self, items_emb, linear_layers, epoch):
        import torch.nn.functional as F
        new_items_emb = items_emb.clone().detach()
        items_emb = items_emb[self._train_items].clone().detach().requires_grad_(True)
        linear_layers = [(w.clone().detach().requires_grad_(True),
                          b.clone().detach().requires_grad_(True))
                         for (w, b) in linear_layers]
        s = args.regula_size
        if epoch==1:
            self.old_items_emb = new_items_emb.clone().detach()
            delta_norm = torch.zeros(self.m_item,1).abs().sum(dim=1, keepdim=True)
            self.delta_value, self.rank = torch.topk(delta_norm, s, dim=0)
            self.rank = self.rank.squeeze(1)
        if epoch==2:
            delta_norm = (new_items_emb-self.old_items_emb).norm(2, dim=-1, keepdim=True)
            self.delta_value, self.rank = torch.topk(delta_norm, s, dim=0)
            self.rank = self.rank.squeeze(1)
            _, self.unrank = torch.topk(-delta_norm, s, dim=0)
            self.unrank = self.unrank.squeeze(1)
            for i, item in enumerate(self._train_items):
                if item in self.rank.cpu():
                    self.pop_indices.append(i)
                else:
                    self.unpop_indices.append(i)
            self.pop_indices = torch.tensor(self.pop_indices)
            self.unpop_indices = torch.tensor(self.unpop_indices)
            
        denominator = np.sum(np.exp(np.arange(s, 0, -1)))
        weighted = torch.tensor(np.exp(np.arange(s, 0, -1))/denominator).to(items_emb.device)

        for e in range(1):
            self._user_emb.zero_grad()
            predictions = self.forward(items_emb, linear_layers)
            if torch.isnan(predictions).any():
                print(predictions)

            loss_bce = nn.BCELoss()(predictions, self._train_labels)
            ipe_mu =args.ipe_mu
            uea_mu =args.uea_mu
            loss_uea,loss_ipe =0,0
            if epoch > 1:
                loss_ipe=-(weighted*torch.cosine_similarity(new_items_emb[self.rank][:, None], items_emb[self.unpop_indices], dim=2).mean(dim=1)).sum()
            loss_uea = -(weighted*F.kl_div(F.log_softmax(new_items_emb[self.rank],dim=1), F.softmax(self._user_emb.weight,dim=1), reduction='none').mean(1)).sum()
            loss = loss_bce+ipe_mu*loss_ipe+uea_mu*loss_uea
            if torch.isnan(loss).any():
                print(loss)
            loss.backward()

            user_emb_grad = self._user_emb.weight.grad
            self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)
            items_emb_grad = items_emb.grad
            linear_layers_grad = [[w.grad, b.grad] for (w, b) in linear_layers]
        return self._train_items, items_emb_grad, linear_layers_grad, loss.cpu().item()

    def eval_(self, items_emb, linear_layers):
        rating = self.forward(items_emb, linear_layers)
        if self._test_:
            ground_truth = np.random.randint(100)
            items = []
            for _ in range(99):
                neg_item = np.random.randint(self.m_item)
                while neg_item in self._train_ or neg_item in items:
                    neg_item = np.random.randint(self.m_item)
                items.append(neg_item)
            items = items[:ground_truth] + [self._test_[0]] + items[ground_truth:]
            items = torch.Tensor(items).long().to(args.device)
            sampled_hr_at_5 = evaluate_recall(rating[items], [ground_truth], 5)
            sampled_hr_at_10 = evaluate_recall(rating[items], [ground_truth], 10)
            sampled_hr_at_20 = evaluate_recall(rating[items], [ground_truth], 20)
            test_result = np.array([sampled_hr_at_5,sampled_hr_at_10,sampled_hr_at_20])
        else:
            test_result = None

        if self._target_:
            rating[self._train_] = - (1 << 10)
            er_at_5 = evaluate_recall(rating, self._target_, 5)
            er_at_10 = evaluate_recall(rating, self._target_, 10)
            er_at_20 = evaluate_recall(rating, self._target_, 20)
            target_result = np.array([er_at_5,er_at_10,er_at_20])
        else:
            target_result = None

        return test_result, target_result
