import torch
import numpy as np
import torch.nn as nn
from parse import args
from eval import evaluate_recall, evaluate_ndcg


class FedRecClient(nn.Module):
    def __init__(self, train_ind, test_ind, target_ind, m_item, dim):
        super().__init__()
        self._train_ = train_ind
        self._test_ = test_ind
        self._target_ = []
        for i in target_ind:
            if i in train_ind:
                continue
            self._target_.append(i)
        self.m_item = m_item
        self._train = torch.Tensor(train_ind).long()

        data = []
        for pos_item in train_ind:
            data.append([pos_item, 1.])
            for _ in range(args.q):
                neg_item = np.random.randint(m_item)
                while neg_item in train_ind:
                    neg_item = np.random.randint(m_item)
                data.append([neg_item, 0.])
        data.sort(key=lambda x: x[0])

        self.items = torch.Tensor([x[0] for x in data]).long()
        self._labels = torch.Tensor([x[1] for x in data]).to(args.device)

        self.dim = dim
        self.items_emb_grad = None
        self._user_emb = nn.Embedding(1, dim)
        nn.init.normal_(self._user_emb.weight, std=0.01)

    def forward(self, items_emb):
        scores = torch.sum(self._user_emb.weight * items_emb, dim=-1)
        if torch.isnan(items_emb).any() or torch.isnan(self._user_emb.weight).any():
            print(items_emb)

        return scores

    def train_(self, items_emb, epoch):
        items_emb = items_emb[self.items].clone().detach().requires_grad_(True)
        self._user_emb.zero_grad()
        predictions = self.forward(items_emb).sigmoid()
        if torch.isnan(predictions).any():
            print(predictions)
        loss = nn.BCELoss()(predictions, self._labels)
        if torch.isnan(loss).any():
            print(loss)
        loss.backward()

        user_emb_grad = self._user_emb.weight.grad
        self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)
        self.items_emb_grad = items_emb.grad
        return self.items, self.items_emb_grad, loss.cpu().item()

    def eval_(self, items_emb):
        rating = self.forward(items_emb)

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
            # sampled_ndcg_at_10 = evaluate_ndcg(rating[items], [ground_truth], 10)
            test_result = np.array([sampled_hr_at_5, sampled_hr_at_10, sampled_hr_at_20])
        else:
            test_result = None

        if self._target_:
            rating[self._train] = - (1 << 10)
            er_at_5 = evaluate_recall(rating, self._target_, 5)
            er_at_10 = evaluate_recall(rating, self._target_, 10)
            er_at_20 = evaluate_recall(rating, self._target_, 20)
            target_result = np.array([er_at_5, er_at_10, er_at_20])
        else:
            target_result = None

        return test_result, target_result

class FedRecClientDefense(nn.Module):
    def __init__(self, train_ind, test_ind, target_ind, m_item, dim):
        super().__init__()
        self._train_ = train_ind
        self._test_ = test_ind
        self._target_ = []
        for i in target_ind:
            if i in train_ind:
                continue
            self._target_.append(i)
        self.m_item = m_item
        self._train = torch.Tensor(train_ind).long()
        self.rank,self.old_items_emb = None,None

        data = []
        for pos_item in train_ind:
            data.append([pos_item, 1.])
            for _ in range(1):
                neg_item = np.random.randint(m_item)
                while neg_item in train_ind:
                    neg_item = np.random.randint(m_item)
                data.append([neg_item, 0.])
        data.sort(key=lambda x: x[0])

        self.items = torch.Tensor([x[0] for x in data]).long()
        self._labels = torch.Tensor([x[1] for x in data]).to(args.device)

        self.dim = dim
        self.items_emb_grad = None
        self._user_emb = nn.Embedding(1, dim)
        nn.init.normal_(self._user_emb.weight, std=0.01)
        self.pop_indices,self.unpop_indices=[],[]
        self.unrank =None
        self.delta_value = None
    def forward(self, items_emb):
        scores = torch.sum(self._user_emb.weight * items_emb, dim=-1)
        if torch.isnan(items_emb).any() or torch.isnan(self._user_emb.weight).any():
            print(items_emb)

        return scores

    def train_(self, items_emb,epoch):
        import torch.nn.functional as F
        new_items_emb = items_emb.clone().detach()
        items_emb = items_emb[self.items].clone().detach().requires_grad_(True)
        
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
            for i, item in enumerate(self.items):
                if item in self.rank.cpu():
                    self.pop_indices.append(i)
                else:
                    self.unpop_indices.append(i)
            self.pop_indices = torch.tensor(self.pop_indices)
            self.unpop_indices = torch.tensor(self.unpop_indices)
        
        denominator = np.sum(np.power(np.arange(1, s+1),2))
        kl_weighted = torch.tensor(np.power(np.arange(s, 0, -1),2)/denominator).to(items_emb.device)
        for e in range(1):
            self._user_emb.zero_grad()
            predictions = self.forward(items_emb).sigmoid()
            if torch.isnan(predictions).any():
                print(predictions)
            loss_bce = nn.BCELoss()(predictions, self._labels) 
            ipe_mu =args.ipe_mu
            uea_mu =args.uea_mu
            loss_uea,loss_ipe =0,0
            if epoch > 1:
                loss_ipe=-(kl_weighted*torch.cosine_similarity(new_items_emb[self.rank][:, None], items_emb[self.unpop_indices], dim=2).mean(dim=1)).sum()
            loss_uea = -(kl_weighted*F.kl_div(F.log_softmax(new_items_emb[self.rank],dim=1), F.softmax(self._user_emb.weight,dim=1), reduction='none').mean(1)).sum()
            
            loss = loss_bce+ipe_mu*loss_ipe+uea_mu*loss_uea
            if torch.isnan(loss).any():
                print(loss)
            loss.backward()

            user_emb_grad = self._user_emb.weight.grad
            self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)
            self.items_emb_grad = items_emb.grad
            items_emb.data.add_(self.items_emb_grad, alpha=-args.lr)
        return self.items, self.items_emb_grad, loss.cpu().item()

    def eval_(self, items_emb):
        rating = self.forward(items_emb)

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
            # sampled_ndcg_at_10 = evaluate_ndcg(rating[items], [ground_truth], 10)
            test_result = np.array([sampled_hr_at_5, sampled_hr_at_10, sampled_hr_at_20])
        else:
            test_result = None

        if self._target_:
            rating[self._train] = - (1 << 10)
            er_at_5 = evaluate_recall(rating, self._target_, 5)
            er_at_10 = evaluate_recall(rating, self._target_, 10)
            er_at_20 = evaluate_recall(rating, self._target_, 20)
            target_result = np.array([er_at_5, er_at_10, er_at_20])
        else:
            target_result = None

        return test_result, target_result
