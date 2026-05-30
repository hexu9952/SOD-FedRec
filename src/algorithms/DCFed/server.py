import time
import logging
from src.util import *
from src.models import *
from copy import deepcopy
import matplotlib.pyplot as plt
from src.algorithms.DCFed.client import DCFedClient


class DCFedServer:
    """
    服务器：聚合客户端参数，更新全局模型，并使用公共数据集训练全局模型，并将新的全局模型参数传递给客户端
    FedServer 类属性：
        config：参数
        model：全局模型
        clients：客户端
        result：用来保存使用公共数据集训练全局模型的测试结果
    """
    def __init__(self, config, global_model):
        self.config = config
        self.model = global_model

        self.clients = None
        self.result = {'NDCG@1': 0.0, 'NDCG@5': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0, 'HT@1': 0.0, 'HT@5': 0.0,
                       'HT@10': 0.0, 'HT@20': 0.0}

        self.count = {}

        self.data = []
        f = open(f"./data/{config['dataset']}.txt", 'r', encoding='ISO-8859-1')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 6:
                self.data.append([int(parts[1]), float(parts[4]), float(parts[5])])


        # 添加通信量记录
        self.communication_log = []  # 记录每轮通信量
        self.downlink_total = 0  # 总下行通信量
        self.uplink_total = 0  # 总上行通信量


    # 创建客户端
    def create_client(self):
        clients = []
        for c_id in range(self.config['num_clients']):
            client_data, client_count = get_local_data(self.config['dataset'], c_id)
            client = DCFedClient(self.config, c_id, client_data, client_count, client_model=deepcopy(self.model))
            clients.append(client)
        self.clients = clients
        return clients

    def send_model(self):
        # 计算模型参数量（字节）
        model_size = 0
        for param in self.model.parameters():
            if param.requires_grad:
                # 每个参数4字节（float32）
                model_size += param.numel() * 4

        # 记录下行通信量（服务器->所有客户端）
        downlink_bytes = model_size * self.config['num_clients']
        self.downlink_total += downlink_bytes

        # 发送模型
        for client in self.clients:
            for new_param, param in zip(self.model.parameters(), client.model.parameters()):
                param.data = new_param.data.clone()

        return model_size  # 返回每个模型的大小
    # 接收客户端权重
    def receive_model(self):
        # 计算模型参数量（字节）
        model_size = 0
        for param in self.model.parameters():
            if param.requires_grad:
                model_size += param.numel() * 4

        # 记录上行通信量（所有客户端->服务器）
        uplink_bytes = model_size * self.config['num_clients']
        self.uplink_total += uplink_bytes

        for client in self.clients:
            self.count[client.id] = client.count

        return model_size  # 返回每个模型的大小



    # 聚合客户端模型参数
    """ FedAvg: https://arxiv.org/pdf/1602.05629.pdf"""
    def aggregate(self, count):

        w_global = self.model.state_dict()

        sum_count = 0
        for key in count:
            sum_count += count[key]
        w_mul = []
        tmp = copy.deepcopy(w_global)

        for j in range(self.config['num_clients']):
            w_avg = copy.deepcopy(self.clients[j].model.state_dict())

            for i in w_avg.keys():
                if "user_embedding" in i or "embedding_user" in i or "user_bias" in i:
                    tmp[i][j] = w_avg[i][j]
                else:
                    w_avg[i] = torch.mul(w_avg[i], count[j])

            w_mul.append(w_avg)

        w_updated = copy.deepcopy(w_mul[0])
        for k in w_updated.keys():
            if "user_embedding" in k or "embedding_user" in k or "user_bias" in k:
                w_updated[k] = tmp[k]
            for i in range(1, len(w_mul)):
                w_updated[k] += w_mul[i][k]
            w_updated[k] = w_updated[k] / sum_count
        return w_updated

    def aggregate1(self, clusters, count, pos):
        num_cluster = []
        nums = 0
        for i in range(len(clusters)):
            num_cluster.append(pos[i])
            nums += pos[i]
        # for cluster in clusters:
        #     num_cluster.append(len(cluster))
        #     nums += len(cluster)
        print("num_cluster:", num_cluster)
        w_global = self.model.state_dict()
        w_updated = []

        for cluster in clusters:
            sum_count = 0
            for i in cluster:
                sum_count += count[i]
            print("sum_count:", sum_count)
            w_mul = []

            for j in cluster:
                w_avg = copy.deepcopy(self.clients[j].model.state_dict())

                for i in w_avg.keys():
                    w_avg[i] = torch.mul(w_avg[i], count[j])

                w_mul.append(w_avg)

            w_update = copy.deepcopy(w_mul[0])

            for k in w_update.keys():
                for i in range(1, len(w_mul)):
                    w_update[k] += w_mul[i][k]
                w_update[k] = w_update[k] / sum_count
            w_updated.append(w_update)

        w_mul = []
        for i in range(len(w_updated)):
            w_avg = copy.deepcopy(w_updated[i])
            for j in w_avg.keys():
                w_avg[j] = torch.mul(w_avg[j], num_cluster[i])
            w_mul.append(w_avg)
        update = copy.deepcopy(w_mul[0])
        for k in update.keys():
            for i in range(1, len(w_updated)):
                update[k] += w_mul[i][k]
            update[k] = update[k] / nums
        return update

    def aggregate2(self, clusters, count):

        w_global = self.model.state_dict()
        w_updated = []

        for cluster in clusters:
            sum_count = 0
            for i in cluster:
                sum_count += count[i]
            w_mul = []
            tmp = copy.deepcopy(w_global)

            for j in cluster:
                w_avg = copy.deepcopy(self.clients[j].model.state_dict())

                for i in w_avg.keys():
                    w_avg[i] = torch.mul(w_avg[i], count[j])

                w_mul.append(w_avg)

            w_update = copy.deepcopy(w_mul[0])

            for k in w_update.keys():
                for i in range(1, len(w_mul)):
                    w_update[k] += w_mul[i][k]
                w_update[k] = w_update[k] / sum_count
            w_updated.append(w_update)


        update = w_updated[0]
        for k in update.keys():
            for i in range(1, len(w_updated)):
                update[k] += w_updated[i][k]
            update[k] = update[k] / len(clusters)
        return update

    def aggregate3(self, clusters, count):

        w_global = self.model.state_dict()
        w_updated = []

        for cluster in clusters:
            sum_count = len(cluster)
            w_mul = []

            for j in cluster:
                w_avg = copy.deepcopy(self.clients[j].model.state_dict())

                # for i in w_avg.keys():
                #     w_avg[i] = torch.mul(w_avg[i], count[j])

                w_mul.append(w_avg)

            w_update = copy.deepcopy(w_mul[0])

            for k in w_update.keys():
                for i in range(1, len(w_mul)):
                    w_update[k] += w_mul[i][k]
                w_update[k] = w_update[k] / sum_count
            w_updated.append(w_update)

        update = w_updated[0]
        for k in update.keys():
            for i in range(1, len(w_updated)):
                update[k] += w_updated[i][k]
            update[k] = update[k] / len(clusters)
        return update

    def aggregate4(self, clusters, count):
        num_cluster = []
        nums = 0
        for cluster in clusters:
            num_cluster.append(len(cluster))
            nums += len(cluster)

        w_global = self.model.state_dict()
        w_updated = []

        for cluster in clusters:
            sum_count = len(cluster)
            w_mul = []

            for j in cluster:
                w_avg = copy.deepcopy(self.clients[j].model.state_dict())
                w_mul.append(w_avg)

            w_update = copy.deepcopy(w_mul[0])
            for k in w_update.keys():
                for i in range(1, len(w_mul)):
                    w_update[k] += w_mul[i][k]
                w_update[k] = w_update[k] / sum_count
            w_updated.append(w_update)

        w_mul = []
        for i in range(len(w_updated)):
            w_avg = copy.deepcopy(w_updated[i])
            for j in w_avg.keys():
                w_avg[j] = torch.mul(w_avg[j], num_cluster[i])
            w_mul.append(w_avg)
        update = copy.deepcopy(w_mul[0])
        for k in update.keys():
            for i in range(1, len(w_updated)):
                update[k] += w_mul[i][k]
            update[k] = update[k] / nums
        return update

    def aggregate5(self, clusters, count):
        num_count = []
        total_count = 0
        for cluster in clusters:
            cluster_count = 0
            for user in cluster:
                cluster_count += count[user]
            num_count.append(cluster_count)
            total_count += cluster_count

        w_global = self.model.state_dict()
        w_updated = []

        for cluster in clusters:
            sum_count = len(cluster)
            w_mul = []

            for j in cluster:
                w_avg = copy.deepcopy(self.clients[j].model.state_dict())
                w_mul.append(w_avg)

            w_update = copy.deepcopy(w_mul[0])
            for k in w_update.keys():
                for i in range(1, len(w_mul)):
                    w_update[k] += w_mul[i][k]
                w_update[k] = w_update[k] / sum_count
            w_updated.append(w_update)

        w_mul = []
        for i in range(len(w_updated)):
            w_avg = copy.deepcopy(w_updated[i])
            for j in w_avg.keys():
                w_avg[j] = torch.mul(w_avg[j], num_count[i])
            w_mul.append(w_avg)
        update = copy.deepcopy(w_mul[0])
        for k in update.keys():
            for i in range(1, len(w_updated)):
                update[k] += w_mul[i][k]
            update[k] = update[k] / total_count
        return update


    def server_update1(self, w_update, test_data, r):

        config = self.config

        # 参数聚合,更新全局模型
        self.model.load_state_dict(w_update)

        local_model = deepcopy(self.model)
        local_model.eval()
        t_test = evaluate(local_model, test_data, config, self.data)
        NDCG_results = t_test[0]
        HIT_results = t_test[1]
        results = {'NDCG@1': NDCG_results[0], 'NDCG@5': NDCG_results[1], 'NDCG@10': NDCG_results[2],
                   'NDCG@20': NDCG_results[3], 'HT@1': HIT_results[0], 'HT@5': HIT_results[1],
                   'HT@10': HIT_results[2], 'HT@20': HIT_results[3]}
        self.result = results
        logging.info(f'round{r} server result:{results}')

        # 记录本轮通信量
        model_size = 0
        for param in self.model.parameters():
            if param.requires_grad:
                model_size += param.numel() * 4

        round_downlink = model_size * config['num_clients']
        round_uplink = model_size * config['num_clients']

        self.communication_log.append({
            'round': r,
            'downlink_bytes': round_downlink,
            'uplink_bytes': round_uplink,
            'model_size_bytes': model_size,
            'num_clients': config['num_clients']
        })

        # 记录到日志
        logging.info(f'Round {r} Communication: Downlink={round_downlink / (1024 ** 2):.2f} MB, '
                     f'Uplink={round_uplink / (1024 ** 2):.2f} MB, '
                     f'Model Size={model_size / (1024 ** 2):.2f} MB')

        return results







