import torch
import random
import numpy as np
from time import time
from parse import args
from data import load_dataset

from server import FedRecServer
from client import FedRecClient


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 增
def rank(array): #降序
    temp = (-array).argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks
# 增
def Pop_Appro_Sta(before_update_emb, after_update_emb,items_popularity,epoch):
    import pandas as pd
    
    # new_norm = after_update_emb.norm(2, dim=-1, keepdim=True)
    # old_norm = before_update_emb.norm(2, dim=-1, keepdim=True)
    delta_sum = (after_update_emb - before_update_emb).abs().sum(dim=1, keepdim=True)
    delta_norm = (after_update_emb - before_update_emb).norm(2, dim=-1, keepdim=True)

    number = np.arange(len(after_update_emb))
    delta_sum_value = np.round(np.array(delta_sum.data.cpu().squeeze(-1)),4)
    delta_norm_value = np.round(np.array(delta_norm.data.cpu().squeeze(-1)),4)
    items_popularity_value = np.round(np.array(items_popularity),4)
    value_sta = pd.DataFrame({'Number':number,'popularity':items_popularity_value,'delta_norm':delta_norm_value,'delta_sum':delta_sum_value})
    value_name = 'log/Appro_Pop_l2/value_sta_data_'+args.dataset+'_epoch'+str(epoch)+'.csv'
    value_sta.to_csv(value_name,index=False,sep=',')

    delta_sum_rank = rank(delta_sum_value)
    delta_norm_rank = rank(delta_norm_value)
    items_popularity_rank = rank(items_popularity_value)
    rank_sta = pd.DataFrame({'Number':number,'popularity':items_popularity_rank,'delta_norm':delta_norm_rank,'delta_sum':delta_sum_rank})
    rank_name = 'log/Appro_Pop_l2/rank_sta__data_'+args.dataset+'_epoch'+str(epoch)+'.csv'
    rank_sta.to_csv(rank_name,index=False,sep=',')


def main():
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s " % args_str)

    t0 = time()
    m_item, all_train_ind, all_test_ind, part_train_ind, items_popularity = load_dataset(args.path + args.dataset)
    target_items = np.random.choice(m_item, 1, replace=False).tolist()

    server = FedRecServer(m_item, args.dim).to(args.device)
    clients = []
    for train_ind, test_ind in zip(all_train_ind, all_test_ind):
        clients.append(
            FedRecClient(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
        )

    malicious_clients_limit = int(len(clients) * args.clients_limit)
    if args.attack == 'FedRecAttack':
        from Attack.FedRecAttack.center import FedRecAttackCenter
        from Attack.FedRecAttack.client import FedRecAttackClient

        attack_center = FedRecAttackCenter(part_train_ind, target_items, m_item, args.dim).to(args.device)
        for _ in range(malicious_clients_limit):
            clients.append(FedRecAttackClient(attack_center, args.items_limit))
    elif args.attack == 'A-pop':
        from attack import PopClient
        for _ in range(malicious_clients_limit):
            clients.append(PopClient(target_items, m_item, args.dim).to(args.device))
    elif args.attack == 'NoAttack':
            print(args.attack)
    else:
        from Attack.baseline import BaselineAttackClient

        if args.attack == 'Random':
            for _ in range(malicious_clients_limit):
                train_ind = [i for i in target_items]
                for __ in range(args.items_limit // 2 - len(target_items)):
                    item = np.random.randint(m_item)
                    while item in train_ind:
                        item = np.random.randint(m_item)
                    train_ind.append(item)
                clients.append(BaselineAttackClient(train_ind, [], [], m_item, args.dim).to(args.device))
        elif args.attack == 'Popular':
            for i in target_items:
                items_popularity[i] = 1e10
            _, train_ind = torch.Tensor(items_popularity).topk(args.items_limit // 2)
            train_ind = train_ind.numpy().tolist()
            for _ in range(malicious_clients_limit):
                clients.append(BaselineAttackClient(train_ind, [], [], m_item, args.dim).to(args.device))
        elif args.attack == 'Bandwagon':
            for i in target_items:
                items_popularity[i] = - 1e10
            items_limit = args.items_limit // 2
            _, popular_items = torch.Tensor(items_popularity).topk(m_item // 10)
            popular_items = popular_items.numpy().tolist()

            for _ in range(malicious_clients_limit):
                train_ind = [i for i in target_items]
                train_ind += np.random.choice(popular_items, int(items_limit * 0.1), replace=False).tolist()
                rest_items = []
                for i in range(m_item):
                    if i not in train_ind:
                        rest_items.append(i)
                train_ind += np.random.choice(rest_items, items_limit - len(train_ind), replace=False).tolist()
                clients.append(BaselineAttackClient(train_ind, [], [], m_item, args.dim).to(args.device))
        else:
            print('Unknown args --attack.')
            return

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t0, len(clients), m_item,
           sum([len(i) for i in all_train_ind]),
           sum([len(i) for i in all_test_ind])))
    print("Target items: %s." % str(target_items))
    print("output format: ({HR@5,HR@10,NDCG@10}), ({ER@5},{ER@10},{ER@20})")

    # Init performance
    t1 = time()
    with torch.no_grad():
        test_result, target_result = server.eval_(clients)
    print("Iteration 0(init), (%.7f, %.7f, %.7f) on test" % tuple(test_result) +
          ", (%.7f, %.7f, %.7f) on target." % tuple(target_result) +
          " [%.1fs]" % (time() - t1))

    try:
        for epoch in range(1, args.epochs + 1):
            t1 = time()
            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)
            
            before_update_emb = server.items_emb.weight.clone().detach() # 增
            total_loss = []
            for i in range(0, len(rand_clients), args.batch_size):
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                # loss = server.train_(clients, batch_clients_idx)
                loss = server.train_(clients, batch_clients_idx, epoch) #改
                total_loss.extend(loss)
            total_loss = np.mean(total_loss).item()

            t2 = time()
            test_result, target_result = server.eval_(clients)
            print("Iteration %d, loss = %.5f [%.1fs]" % (epoch, total_loss, t2 - t1) +
                  ", (%.7f, %.7f, %.7f) on test" % tuple(test_result) +
                  ", (%.7f, %.7f, %.7f) on target." % tuple(target_result) +
                  " [%.1fs]" % (time() - t2))
            
            # popularity = np.round(np.array(items_popularity),4)
            # items_popularity_name = 'log/PCA/items_popularity_'+args.dataset+'_epoch'+str(epoch)+'.npy'
            # np.save(items_popularity_name,popularity)
            # if args.attack == 'NoAttack': # 统计item_emb和user_emb的方向
            #     after_update_item_emb = np.array(server.items_emb.weight.clone().detach().cpu()) #增
            #     after_update_user_emb = []
            #     for i in range(len(clients)):
            #         after_update_user_emb.append(np.array(clients[i]._user_emb.weight.clone().detach().cpu()[0]))
            #     after_update_user_emb = np.array(after_update_user_emb)
            #     item_emb_name = 'log/PCA/update_item_emb_'+args.dataset+'_epoch'+str(epoch)+'.npy'
            #     np.save(item_emb_name,after_update_item_emb)
            #     user_emb_name = 'log/PCA/update_user_emb_'+args.dataset+'_epoch'+str(epoch)+'.npy'
            #     np.save(user_emb_name,after_update_user_emb)
            
            # if args.attack == 'NoAttack':
            #     after_update_emb = server.items_emb.weight.clone().detach() #增
            #     Pop_Appro_Sta(before_update_emb, after_update_emb,items_popularity,epoch) #增
            
    except KeyboardInterrupt:
        pass


setup_seed(20211110)

if __name__ == "__main__":
    main()
