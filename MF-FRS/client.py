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

    def forward(self, items_emb):
        scores = torch.sum(self._user_emb.weight * items_emb, dim=-1)
        if torch.isnan(items_emb).any() or torch.isnan(self._user_emb.weight).any():
            print(items_emb)

        return scores

    # def train_(self, items_emb, reg=0.):
    def train_(self, items_emb,epoch,reg=0.): #改
        items_emb = items_emb[self.items].clone().detach().requires_grad_(True)
        self._user_emb.zero_grad()

        predictions = self.forward(items_emb).sigmoid()
        if torch.isnan(predictions).any():
            print(predictions)
        loss = nn.BCELoss()(predictions, self._labels) + \
            0.5 * (self._user_emb.weight.norm(2).pow(2) + items_emb.norm(2).pow(2)) * reg
        if torch.isnan(loss).any():
            print(loss)
        loss.backward()

        user_emb_grad = self._user_emb.weight.grad
        self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)
        self.items_emb_grad = items_emb.grad
        # * len(self._labels)
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
            sampled_ndcg_at_10 = evaluate_ndcg(rating[items], [ground_truth], 10)
            test_result = np.array([sampled_hr_at_5, sampled_hr_at_10, sampled_ndcg_at_10])
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
