import torch
import random
import numpy as np
from time import time
from parse import args
from data import load_dataset

from server import FedRecServer
from client import FedRecClient,FedRecClientDefense

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args_str = ",".join([("%s=%s" % (k, v)) for k, v in args.__dict__.items()])
    print("Arguments: %s " % args_str)

    t0 = time()
    m_item, all_train_ind, all_test_ind, part_train_ind, items_popularity = load_dataset(args.path + args.dataset)
    target_items = np.random.choice(m_item, 1, replace=False).tolist()

    server = FedRecServer(m_item, args.dim).to(args.device)
    clients = []
    for train_ind, test_ind in zip(all_train_ind, all_test_ind):
        if args.defense=='Regula':
                    clients.append(
            FedRecClientDefense(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
        )
        elif args.defense=='Regula-POS':
            clients.append(
                FedRecClient_Pen1(train_ind, test_ind, target_items, m_item, args.dim).to(args.device)
            )
        else:
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
    elif args.attack == 'PipAttackEB': 
        from attack import PipAttackEB
        for _ in range(malicious_clients_limit):
            clients.append(PipAttackEB(target_items, m_item, args.dim).to(args.device))  
    elif args.attack == 'PIECKUEA':
        from attack import PIECKUEA
        for _ in range(malicious_clients_limit):
            clients.append(PIECKUEA(target_items, m_item, args.dim).to(args.device))
    elif args.attack == 'PIECKIPE':
        from attack import PIECKIPE
        for _ in range(malicious_clients_limit):
            clients.append(PIECKIPE(target_items, m_item, args.dim).to(args.device))
    elif args.attack == 'A-ra' or args.attack == 'A-hum':
        from Attack.Ahum.client import AhumClient
        for _ in range(malicious_clients_limit):
            clients.append(AhumClient(target_items, m_item, args.dim).to(args.device))
        print(args.attack)
    elif args.attack == 'NoAttack':
        print(args.attack)
    else:
        print('Unknown args --attack.')
        return

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time() - t0, len(clients), m_item,
           sum([len(i) for i in all_train_ind]),
           sum([len(i) for i in all_test_ind])))
    print("Target items: %s." % str(target_items))
    print("output format: ({HR@5,HR@10,HR@20}), ({ER@5},{ER@10},{ER@20})")

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
            
            before_update_emb = server.items_emb.weight.clone().detach()
            total_loss = []
            for i in range(0, len(rand_clients), args.batch_size):
                com_round = i/args.batch_size+(epoch-1)*len(rand_clients)//args.batch_size
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                # loss = server.train_(clients, batch_clients_idx)
                loss = server.train_(clients, batch_clients_idx, epoch)
                total_loss.extend(loss)
            total_loss = np.mean(total_loss).item()
            t2 = time()
            test_result, target_result = server.eval_(clients)
            print("Iteration %d, Round %d, loss = %.5f [%.1fs]" % (epoch,com_round, total_loss, t2 - t1) +
                  ", (%.7f, %.7f, %.7f) on test" % tuple(test_result) +
                  ", (%.7f, %.7f, %.7f) on target." % tuple(target_result) +
                  " [%.1fs]" % (time() - t2))
            
    except KeyboardInterrupt:
        pass


setup_seed(20211110)

if __name__ == "__main__":
    main()
