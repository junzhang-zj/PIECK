import torch
import random
import numpy as np
from time import time
from parse import args
from data import load_dataset
from client import FedRecClient
from server import FedRecServer, PopServer
from attack import AttackClient, BaselineAttackClient, PopClient


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
    m_item, all_train_ind, all_test_ind, items_popularity = load_dataset(args.path + args.dataset)
    _, target_items = torch.Tensor(-items_popularity).topk(1)
    target_items = target_items.tolist()  # Select the least popular item as the target item

    # server = FedRecServer(m_item, args.dim, eval(args.layers)).to(args.device) 
    server = PopServer(m_item, args.dim, eval(args.layers)).to(args.device) # 改
    
    clients = []
    for train_ind, test_ind in zip(all_train_ind, all_test_ind):
        clients.append(
            FedRecClient(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
        )

    malicious_clients_limit = int(len(clients) * args.clients_limit)
    if args.attack == 'A-ra' or args.attack == 'A-hum':
        for _ in range(malicious_clients_limit):
            clients.append(AttackClient(target_items, m_item, args.dim).to(args.device))
    elif args.attack == 'A-pop': # 增
        for _ in range(malicious_clients_limit):
            clients.append(PopClient(target_items, m_item, args.dim).to(args.device))
    elif args.attack == 'EB':
        for _ in range(malicious_clients_limit):
            clients.append(BaselineAttackClient(target_items, m_item, args.dim).to(args.device))
    elif args.attack == 'RA':
        for _ in range(malicious_clients_limit):
            train_ind = [i for i in target_items]
            for __ in range(args.items_limit - len(target_items)):
                item = np.random.randint(m_item)
                while item in train_ind:
                    item = np.random.randint(m_item)
                train_ind.append(item)
            clients.append(BaselineAttackClient(train_ind, m_item, args.dim).to(args.device))
    elif args.attack == 'PA':
        for i in target_items:
            items_popularity[i] = - 1e10
        _, popular_items = torch.Tensor(items_popularity).topk(m_item)
        popular_items = popular_items.numpy().tolist()
        popular_items = np.random.choice(popular_items, args.items_limit - len(target_items), replace=False).tolist()
        train_ind = target_items + popular_items

        for _ in range(malicious_clients_limit):
            clients.append(BaselineAttackClient(train_ind, m_item, args.dim).to(args.device))

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t0, len(clients), m_item,
           sum([len(i) for i in all_train_ind]),
           sum([len(i) for i in all_test_ind])))
    print("Target items: %s." % str(target_items))
    # print("output format: ({HR@10}), ({ER@10})")
    print("output format: ({HR@5,HR@10,HR@20}), ({ER@5},{ER@10},{ER@20})")

    # Init performance
    t1 = time()
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
            mal_start_ind = len(clients)-malicious_clients_limit # 增
            
            total_loss = []
            for i in range(0, len(rand_clients), args.batch_size):
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                # loss = server.train_(clients,batch_clients_idx)
                loss = server.train_(clients,batch_clients_idx,epoch,mal_start_ind)   # 改1
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


if __name__ == "__main__":
    setup_seed(20220110)
    main()
