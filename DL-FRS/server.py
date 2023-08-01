import torch
import torch.nn as nn
from parse import args 


class FedRecServer(nn.Module):
    def __init__(self, m_item, dim, layers):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        self.layers = layers

        self.items_emb = nn.Embedding(m_item, dim)
        nn.init.normal_(self.items_emb.weight, std=0.01)

        layers_dim = [2 * dim] + layers + [1]
        self.linear_layers = nn.ModuleList([nn.Linear(layers_dim[i-1], layers_dim[i])
                                            for i in range(1, len(layers_dim))])
        for layer in self.linear_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            # nn.init.kaiming_uniform_(layer.weight, 1)
            nn.init.zeros_(layer.bias)
        
    def train_(self, clients, batch_clients_idx):
        items_emb = self.items_emb.weight
        linear_layers = [[layer.weight, layer.bias] for layer in self.linear_layers]
        batch_loss = []
        batch_items_emb_grad = torch.zeros_like(items_emb)
        batch_linear_layers_grad = [[torch.zeros_like(w), torch.zeros_like(b)] for (w, b) in linear_layers]

        for idx in batch_clients_idx:
            client = clients[idx]
            items, items_emb_grad, linear_layers_grad, loss = client.train_(items_emb, linear_layers)

            with torch.no_grad():
                batch_items_emb_grad[items] += items_emb_grad
                for i in range(len(linear_layers)):
                    batch_linear_layers_grad[i][0] += linear_layers_grad[i][0]
                    batch_linear_layers_grad[i][1] += linear_layers_grad[i][1]

            if loss is not None:
                batch_loss.append(loss)

        with torch.no_grad():
            self.items_emb.weight.data.add_(batch_items_emb_grad, alpha=-args.lr)
            for i in range(len(linear_layers)):
                self.linear_layers[i].weight.data.add_(batch_linear_layers_grad[i][0], alpha=-args.lr)
                self.linear_layers[i].bias.data.add_(batch_linear_layers_grad[i][1], alpha=-args.lr)
        return batch_loss

    def eval_(self, clients):
        items_emb = self.items_emb.weight
        linear_layers = [(layer.weight, layer.bias) for layer in self.linear_layers]
        test_cnt, test_results = 0, 0.
        target_cnt, target_results = 0, 0.

        with torch.no_grad():
            for client in clients:
                test_result, target_result = client.eval_(items_emb, linear_layers)
                if test_result is not None:
                    test_cnt += 1
                    test_results += test_result
                if target_result is not None:
                    target_cnt += 1
                    target_results += target_result
        return test_results / test_cnt, target_results / target_cnt

import defense
class PopServer(nn.Module):
    def __init__(self, m_item, dim, layers):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        self.layers = layers

        self.items_emb = nn.Embedding(m_item, dim)
        nn.init.normal_(self.items_emb.weight, std=0.01)

        layers_dim = [2 * dim] + layers + [1]
        self.linear_layers = nn.ModuleList([nn.Linear(layers_dim[i-1], layers_dim[i])
                                            for i in range(1, len(layers_dim))])
        for layer in self.linear_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
            
    def normbound(self, items_emb_grad):
        norm = items_emb_grad.norm(2, dim=-1, keepdim=True) 
        if len(norm.shape) == 1: # bias
            too_large  = norm[0] > args.grad_limit 
        else: # weights
            too_large = norm[:,0] > args.grad_limit
        items_emb_grad[too_large] /= (norm[too_large] / args.grad_limit) 
        return items_emb_grad
        
    def train_(self, clients, batch_clients_idx,epoch,mal_start_ind):
        items_emb = self.items_emb.weight
        linear_layers = [[layer.weight, layer.bias] for layer in self.linear_layers]
        batch_loss = []
        batch_items_emb_grad = torch.zeros_like(items_emb)
        batch_linear_layers_grad = [[torch.zeros_like(w), torch.zeros_like(b)] for (w, b) in linear_layers]
        
        batch_items_inter = torch.zeros([len(items_emb),1])
        if args.defense != 'NoDefense' and args.defense != 'NormBound' and args.defense[:6] != 'Regula':
            batch_items = [[] for i in range(len(batch_clients_idx))]
            batch_items_grads = torch.zeros((len(batch_clients_idx), len(self.items_emb.weight),self.dim)).to(args.device)
            batch_linear_grads = [[[torch.zeros_like(w), torch.zeros_like(b)] for (w, b) in linear_layers] for i in range(len(batch_clients_idx))]
        
        for idx, user in enumerate(batch_clients_idx):
            client = clients[user]
            items, items_emb_grad, linear_layers_grad, loss = client.train_(items_emb,linear_layers,epoch)

            with torch.no_grad():
                if args.defense == 'NormBound': 
                    items_emb_grad = self.normbound(items_emb_grad)
                if args.defense != 'NoDefense' and args.defense != 'NormBound' and args.defense[:6] != 'Regula':
                    batch_items_grads[idx,items,:] = items_emb_grad
                    if isinstance(items,list):
                        batch_items[idx] = items
                    else:
                        batch_items[idx] = items.cpu().numpy().tolist()

                batch_items_emb_grad[items] += items_emb_grad
                # batch_items_inter[items] += 1
                
                for i in range(len(linear_layers)):
                    if args.defense == 'NormBound':
                        if linear_layers_grad is None:
                            continue
                        linear_layers_grad[i][0] = self.normbound(linear_layers_grad[i][0])
                        linear_layers_grad[i][1] = self.normbound(linear_layers_grad[i][1])
                    
                    if args.defense != 'NoDefense' and args.defense != 'NormBound' and args.defense[:6] != 'Regula': # 增
                        if linear_layers_grad is None and i==0:
                            batch_linear_grads[idx] = None
                            continue
                        batch_linear_grads[idx][i][0] = linear_layers_grad[i][0].cpu()
                        batch_linear_grads[idx][i][1] = linear_layers_grad[i][1].cpu()
                        
                    if linear_layers_grad != None:
                        batch_linear_layers_grad[i][0] += linear_layers_grad[i][0]
                        batch_linear_layers_grad[i][1] += linear_layers_grad[i][1]

            if loss is not None:
                batch_loss.append(loss)

        with torch.no_grad():
            if args.defense == 'NoDefense' or args.defense == 'NormBound' or args.defense[:6] == 'Regula':
                batch_items_inter[batch_items_inter==0] =1
                self.items_emb.weight.data.add_(batch_items_emb_grad, alpha=-args.lr)
                for i in range(len(linear_layers)):
                    self.linear_layers[i].weight.data.add_(batch_linear_layers_grad[i][0], alpha=-args.lr)
                    self.linear_layers[i].bias.data.add_(batch_linear_layers_grad[i][1], alpha=-args.lr)
                
            else:
                import numpy as np
                batch_current_grads = torch.zeros_like(items_emb)
                for i in range(batch_items_grads.shape[1]):
                    user_idx = [i in x for x in batch_items]
                    if sum(user_idx) == 0:
                        batch_current_grads[i] = torch.zeros_like(items_emb[0]).to(args.device)
                    else:
                        before_defense_grads = batch_items_grads[user_idx,i,:].cpu()
                        corrupted_count=int(sum(user_idx)*args.clients_limit)
                        current_grads = defense.defend[args.defense](np.array(before_defense_grads), sum(user_idx), corrupted_count) 
                        batch_current_grads[i] = torch.from_numpy(current_grads).to(args.device)
                    
                self.items_emb.weight.data.add_(batch_current_grads, alpha=-args.lr)

                for i in range(len(linear_layers)):
                    pending_weight, pending_bias = [], []
                    for x in batch_linear_grads:
                        if x is None:
                            continue
                        pending_weight.append(x[i][0])
                        pending_bias.append(x[i][1])
                    for j in range(len(pending_weight)):
                        pending_weight[j] = np.array(pending_weight[j]).reshape(-1)
                        pending_bias[j] = np.array(pending_bias[j]).reshape(-1)

                    # corrupted_count = sum(batch_clients_idx>=mal_start_ind) # linear layer
                    corrupted_count = int(len(batch_clients_idx)*args.clients_limit)
                    current_weight_grad = defense.defend[args.defense](np.array(pending_weight), len(batch_clients_idx), corrupted_count)
                    current_weight_grad = torch.from_numpy(current_weight_grad.reshape(len(self.linear_layers[i].weight),-1)).to(args.device)

                    current_bias_grad = defense.defend[args.defense](np.array(pending_bias), len(batch_clients_idx), corrupted_count)
                    current_bias_grad = torch.from_numpy(current_bias_grad).to(args.device)
                    self.linear_layers[i].weight.data.add_(current_weight_grad, alpha=-args.lr)
                    self.linear_layers[i].bias.data.add_(current_bias_grad, alpha=-args.lr)  
        return batch_loss

    def eval_(self, clients):
        items_emb = self.items_emb.weight
        linear_layers = [(layer.weight, layer.bias) for layer in self.linear_layers]
        test_cnt, test_results = 0, 0.
        target_cnt, target_results = 0, 0.

        with torch.no_grad():
            for client in clients:
                test_result, target_result = client.eval_(items_emb, linear_layers)
                if test_result is not None:
                    test_cnt += 1
                    test_results += test_result
                if target_result is not None:
                    target_cnt += 1
                    target_results += target_result
        return test_results / test_cnt, target_results / target_cnt
