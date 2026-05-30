import time
import logging
from src.util import *
from copy import deepcopy
import matplotlib.pyplot as plt


class DCFedClient:
    def __init__(self, config, client_id, client_data, client_count, client_model):
        """
        客户端：使用本地数据集训练本地模型，并将本地模型上传给服务器
        FedClient 类属性：
            config：参数
            model：本地模型
            result：用来保存使用本地数据集训练本地模型的测试结果
        """
        self.config = config
        self.id = client_id
        self.count = client_count
        self.local_data = client_data
        self.model = client_model

        self.rank_list = []
        self.result = {'NDCG@1': 0.0, 'NDCG@5': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0, 'HT@1': 0.0, 'HT@5': 0.0,
                       'HT@10': 0.0, 'HT@20': 0.0}

        self.data = []
        f = open(f"./data/{config['dataset']}.txt", 'r', encoding='ISO-8859-1')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                self.data.append([int(parts[1]), float(parts[4]), float(parts[5])])

    def client_update(self, r):
        self.rank_list = []
        config = self.config

        [user_train, user_valid, user_test, usernum, itemnum] = self.local_data
        # 计算平均轨迹长度
        cc = 0.0
        for u in user_train:
            cc += len(user_train[u])
        logging.info(f'client{self.id} average sequence length: %.2f' % (cc / len(user_train)))

        # 客户端id列表
        client_user_list = []
        client_user_list.extend(user_train.keys())
        logging.info(f"Client{self.id}_user_list: {client_user_list}")

        client_data_proceed = [user_train, user_valid, user_test, usernum, itemnum, client_user_list]
        local_model = deepcopy(self.model)
        num_batch = len(user_train) // config['client_batch_size']

        sampler = WarpSampler(user_train, usernum, itemnum, config['client_batch_size'], config['maxlen'], 1, self.data)

        local_model.train()

        # 只进行模型预测
        if config['inference_only'] == 1:
            local_model.eval()
            t_test = evaluate(local_model, client_data_proceed, config, self.data)
            print('test (NDCG@10: %.4f, HT@10: %.4f)' % (t_test[0], t_test[1]))
            self.rank_list = t_test[2]

        criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
        optimizer = torch.optim.Adam(local_model.parameters(), lr=config['lr'], betas=(0.9, 0.98))

        T = 0.0
        t0 = time.time()
        num = 0
        Last_client_loss = []
        total_results = [0 for i in range(8)]
        avg_max_results = [0 for i in range(8)]
        result_dict = {}

        for epoch in range(config['client_epochs']):
            if config['inference_only'] == 1: break  # just to decrease identition
            for step in range(num_batch):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                u, seq, pos, neg = sampler.next_batch()  # tuples to ndarray
                u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
                pos_logits, neg_logits = local_model(u, seq, pos, neg)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=config['device']), torch.zeros(neg_logits.shape,device=config['device'])
                # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
                optimizer.zero_grad()
                pos = [[sublist[0] for sublist in row] for row in pos]
                indices = np.where(pos != 0)

                loss = criterion(pos_logits[indices], pos_labels[indices])
                loss += criterion(neg_logits[indices], neg_labels[indices])
                for param in local_model.item_emb.parameters(): loss += config['l2_emb'] * torch.norm(param)
                if step == 0:
                    with open(f"Results/{config['algorithm']}/loss/clients/first_step/client{self.id}_loss.txt", "a") as f:
                        f.write(f"Round: {r}, Epoch: {epoch}, client[{self.id}]_loss: {loss.item()}\n")
                if step == num_batch - 1:
                    with open(f"Results/{config['algorithm']}/loss/clients/last_step/client{self.id}_loss.txt", "a") as f:
                        f.write(f"Round: {r}, Epoch: {epoch}, client[{self.id}]_loss: {loss.item()}\n")
                with open(f"Results/{config['algorithm']}/loss/clients/all_step/client{self.id}_loss.txt", "a") as f:
                    f.write(f"Round: {r}, Epoch: {epoch}, client[{self.id}]_loss: {loss.item()}\n")

                loss.backward()
                optimizer.step()

                # step是循环计数器，用来跟踪当前epoch的训练进度
                if self.id == config['num_clients'] - 1:
                    Last_client_loss.append(loss.item())
                if epoch % 20 == 0:
                    logging.info(f"Round:{r}, client {self.id} loss in epoch {epoch} iteration {step}: {loss.item()}")  # expected 0.4~0.6 after init few epochs

            if epoch % 20 == 0:
                result = []
                num += 1
                local_model.eval()
                t1 = time.time() - t0
                T += t1
                t_test = evaluate(local_model, client_data_proceed, config, self.data)
                # logging.info(
                #     'Round: {}, client:{} epoch:{}, time: {}, Test_Result: NDCG: {},HT {}'.format(r, self.id, epoch,
                #                                                                                   T, t_test[0],
                #                                                                                   t_test[1]))
                result.extend(t_test[0])
                result.extend(t_test[1])
                self.rank_list = t_test[2]
                for j in range(len(result)):
                    total_results[j] += result[j]
                result_dict = {'NDCG@1': t_test[0][0], 'NDCG@5': t_test[0][1], 'NDCG@10': t_test[0][2],
                               'NDCG@20': t_test[0][3], 'HT@1': t_test[1][0], 'HT@5': t_test[1][1],
                               'HT@10': t_test[1][2], 'HT@20': t_test[1][3]}
                # logging.info(f'round{r} epoch{epoch} result:{result_dict}')
        for i in range(len(total_results)):
            if num == 0:
                avg_max_results[i] = total_results[i]
            else:
                avg_max_results[i] = total_results[i] / num

            result_dict = {'NDCG@1': avg_max_results[0], 'NDCG@5': avg_max_results[1],'NDCG@10': avg_max_results[2],
                            'NDCG@20': avg_max_results[3], 'HT@1': avg_max_results[4], 'HT@5': avg_max_results[5],
                            'HT@10': avg_max_results[6], 'HT@20': avg_max_results[7]}
        # 结果画图
        if r == config['global_epochs'] - 1:
            plt.plot(Last_client_loss, label='Round: {}, Last client loss'.format(r))
            plt.xlabel('Num_iter')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f"./Results/{config['algorithm']}/Last_client_loss.png")

        self.model = deepcopy(local_model)
        self.result = result_dict
        if r == (config['global_epochs']-1):
            logging.info(f'round{r} client{self.id} result:{result_dict}')
        # return local_model, result_dict

    def client_predict(self, test_data):
        config = self.config
        local_model = deepcopy(self.model)
        local_model.eval()
        t_test = evaluate(local_model, test_data, config, self.data)
        self.rank_list = t_test[2]
        NDCG_results = t_test[0]
        HIT_results = t_test[1]
        results = {'NDCG@1': NDCG_results[0], 'NDCG@5': NDCG_results[1], 'NDCG@10': NDCG_results[2],
                   'NDCG@20': NDCG_results[3], 'HT@1': HIT_results[0], 'HT@5': HIT_results[1],
                   'HT@10': HIT_results[2], 'HT@20': HIT_results[3]}
        self.result = results
        logging.info(f'client result:{results}')
        return results