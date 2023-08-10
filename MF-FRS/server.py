import torch
import torch.nn as nn
from parse import args


class FedRecServer(nn.Module):
    def __init__(self, m_item, dim):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        self.items_emb = nn.Embedding(m_item, dim)
        nn.init.normal_(self.items_emb.weight, std=0.01)
            
    def normbound(self, items_emb_grad):
        norm = items_emb_grad.norm(2, dim=-1, keepdim=True) 
        if len(norm.shape) == 1: # bias
            too_large  = norm[0] > args.grad_limit 
        else: # weights
            too_large = norm[:,0] > args.grad_limit
        items_emb_grad[too_large] /= (norm[too_large] / args.grad_limit) 
        return items_emb_grad

    def train_(self, clients, batch_clients_idx, epoch): 
        batch_loss = []
        batch_items_emb_grad = torch.zeros_like(self.items_emb.weight)
        batch_items_cnt = torch.zeros(self.m_item, 1).long().to(args.device)
        
        if args.defense != 'NoDefense' and args.defense[:9] != 'NormBound' and args.defense[:6] != 'Regula':
            batch_items = [[] for i in range(len(batch_clients_idx))]
            batch_items_grads = torch.zeros((len(batch_clients_idx), len(self.items_emb.weight),self.dim)).to(args.device)

        for idx,user in enumerate(batch_clients_idx):
            client = clients[user]
            # items, items_emb_grad, loss = client.train_(self.items_emb.weight)
            items, items_emb_grad, loss = client.train_(self.items_emb.weight,epoch) 
            batch_items_cnt[items] += 1

            with torch.no_grad():
                if args.defense == 'NormBound': 
                    items_emb_grad = self.normbound(items_emb_grad)
                if args.defense != 'NoDefense' and args.defense[:9] != 'NormBound' and args.defense[:6] != 'Regula':
                    batch_items_grads[idx,items,:] = items_emb_grad
                    if isinstance(items,list):
                        batch_items[idx] = items
                    else:
                        batch_items[idx] = items.cpu().numpy().tolist()
                
                batch_items_emb_grad[items] += items_emb_grad

            if loss is not None:
                batch_loss.append(loss)

        with torch.no_grad():
            if args.defense == 'NoDefense' or args.defense[:9] == 'NormBound' or args.defense[:6] == 'Regula':
                batch_items_cnt[batch_items_cnt == 0] = 1 
                self.items_emb.weight.data.add_(batch_items_emb_grad , alpha=-args.lr)
            else:
                import numpy as np
                import defense
                batch_current_grads = torch.zeros_like(self.items_emb.weight)
                for i in range(self.m_item):
                    user_idx = [i in x for x in batch_items]
                    if sum(user_idx) == 0:
                        batch_current_grads[i] = torch.zeros_like(self.items_emb.weight[0]).to(args.device)
                    else:
                        before_defense_grads = batch_items_grads[user_idx,i,:].cpu()
                        corrupted_count=int(sum(user_idx)*args.clients_limit)
                        current_grads = defense.defend[args.defense](np.array(before_defense_grads), sum(user_idx), corrupted_count) 
                        batch_current_grads[i] = torch.from_numpy(current_grads).to(args.device)
                self.items_emb.weight.data.add_(batch_current_grads, alpha=-args.lr)
                
        return batch_loss

    def eval_(self, clients):
        test_cnt = 0
        test_results = 0.
        target_cnt = 0
        target_results = 0.

        for client in clients:
            test_result, target_result = client.eval_(self.items_emb.weight)
            if test_result is not None:
                test_cnt += 1
                test_results += test_result
            if target_result is not None:
                target_cnt += 1
                target_results += target_result
        return test_results / test_cnt, target_results / target_cnt
