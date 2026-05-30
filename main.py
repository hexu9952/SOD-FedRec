import os
import random
import time
import json
from itertools import count

import torch
import logging
import argparse
from src.util import *
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from copy import deepcopy
from src.models import SASRec
from src.models import SASRec1
from src.models import SASRec_JointSpatial_NYC
from src.models import SASRec_JointSpatial_TKY
from src.models import SASRec_tky
from src.models import SASRec_wtPOI_NYC
from src.models import SASRec_wtPOI_tky
from src.models import SASRec_concat
from src.algorithms.DCFed.server import DCFedServer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# 参数
# config:FL训练参数
file_obj = open('config.json', 'r')
config = json.load(file_obj)

# args:SASRec模型参数
parser = argparse.ArgumentParser()
parser.add_argument('--maxlen', default=400, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--device', default=config["device"], type=str)
args = parser.parse_args()


if __name__ == '__main__':
    """
    1.设置日志，记录终端上的输出信息
    
    2.获取全局数据，用于初始化全局模型
    3.初始化全局模型
    4.初始化服务器，传入参数和全局模型
    5.服务器初始化客户端，调用server.create_client()方法
    6.初始化结果列表，保存训练结果
    7.训练，训练过程中记录客户端/服务器/总的训练时长
    8.保存训练结果
    9.测试：在测试数据集上分别测试服务器模型和客户端模型性能
    """

    # 1.设置日志。日志文件保存在 ./Logs/{config['algorithm']}/log.txt 路径中，记录所有终端输出信息
    if not os.path.exists(f"./Logs/{config['algorithm']}"):
        os.mkdir(f"./Logs/{config['algorithm']}")
    filename = f"./Logs/{config['algorithm']}/"
    set_logger(f"{filename}log.txt")

    # 2.获取全局数据，用于初始化全局模型
    global_data = get_global_data(config['dataset'])
    [user_train, user_valid, user_test, global_usernum, global_itemnum] = global_data
    logging.info(f"original length of dataset: {len(user_train.keys())}, usernum of original dataset:{global_usernum}, "
                 f"itemnum of original dataset:{global_itemnum}\n")

    # 3.初始化全局模型，SASRec/SSEPT
    # global_model = SASRec1(global_usernum, global_itemnum, args).to(config['device'])
    # global_model = SASRec_tky(global_usernum, global_itemnum, args).to(config['device'])
    # global_model = SASRec_wtPOI_NYC(global_usernum, global_itemnum, args).to(config['device'])
    global_model = SASRec_wtPOI_tky(global_usernum, global_itemnum, args).to(config['device'])
    # global_model = SASRec_concat(global_usernum, global_itemnum, args).to(config['device'])
    # global_model = SSEPT(global_usernum, global_itemnum, args).to(config['device'])
    # global_model = SASRec_JointSpatial_NYC(global_usernum, global_itemnum, args).to(config['device'])
    # global_model = SASRec_JointSpatial_TKY(global_usernum, global_itemnum, args).to(config['device'])

    # 4.初始化服务器
    # 获取放在服务器端的公共数据集，初始化一个服务器，初始化参数包括：FL参数信息(config)、全局模型(global_model)、公共数据集(global_data)
    server_name = config['algorithm'] + 'Server'
    server = eval(server_name)(config, global_model=deepcopy(global_model))    # deepcopy
    logging.info("Server is successfully initialized")

    # 5.初始化客户端
    # 服调用服务器create_client()函数，创建客户端
    clients = server.create_client()
    logging.info("Clients are successfully initialized")

    # 6.初始化客户端/服务器结果列表（客户端结果列表中包含每个客户端的结果字典）
    client_results_list = []
    for i in range(config['num_clients']):
        client_results_list.append(
            {'NDCG@1': 0.0, 'NDCG@5': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0, 'HT@1': 0.0, 'HT@5': 0.0, 'HT@10': 0.0,
             'HT@20': 0.0})
    server_avg_result = {'NDCG@1': 0.0, 'NDCG@5': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0, 'HT@1': 0.0, 'HT@5': 0.0,
                         'HT@10': 0.0, 'HT@20': 0.0}
    server_results_dict = {'NDCG@1': 0.0, 'NDCG@5': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0, 'HT@1': 0.0, 'HT@5': 0.0,
                           'HT@10': 0.0, 'HT@20': 0.0}

    # 用于记录训练时长（记录训练时长的方法可以进一步优化，只记录需要的部分，考虑清楚需要哪些时长）
    total_time = {}
    server_time = {}
    client_time = {}
    for idx in range(config['num_clients']):
        client_time[idx] = {}
    all_client_time = {}

    if config['algorithm'] == 'FedHyper':
        local_lr = []
        for i in range(config['num_clients']):
            local_lr.append(config['lr'])

        origin_vector = parameters_to_vector(global_model.parameters())
        gradient_vector = origin_vector - origin_vector

    # 获取test_data
    test_data = get_test_data(config['test_dataset'])
    data = []
    f = open(f"./data/{config['dataset']}.txt", 'r', encoding='ISO-8859-1')
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 6:
            data.append([int(parts[1]), float(parts[4]), float(parts[5])])
    # 7.开始训练
    for r in range(config["global_epochs"]):
        logging.info(f"\nCommunication Round:{r}")
        t1 = time.perf_counter()
        # 发送模型到客户端，每轮训练开始都把最新的全局模型以及需要的参数更新给客户端
        server.send_model()

        # 客户端本地训练
        # t4-t3:一轮训练中所有客户端训练总时间
        t3 = time.perf_counter()

        clients_rank_lists = []
        for idx in range(config['num_clients']):
            # t6-t5:一轮训练中每个客户端训练时间
            t5 = time.perf_counter()
            # 客户端本地更新，训练本地模型
            if config['algorithm'] == 'FedHyper':
                clients[idx].client_update(r, lrs=local_lr, gtl=gradient_vector)
            else:
                clients[idx].client_update(r)
            result = clients[idx].client_predict(test_data)

            clients_rank_lists.append(clients[idx].rank_list)
            result_dict = clients[idx].result
            # 保存客户端每轮更新结果，最后平均，保存平均结果作为该客户端训练结果

            for key in result_dict:
                client_results_list[idx][key] += result_dict[key]
            # 保存最后一轮客户端模型，用于测试客户端模型性能
            if r == config["global_epochs"] - 1:
                # pass
                torch.save(clients[idx].model,
                           f"Results/{config['algorithm']}/model_weight/clients/all_clients/" + f"last_round_client{idx}_model" + ".pt")
            # 记录一个global_epoch,单个client训练时长
            t6 = time.perf_counter()
            c_time = t6 - t5
            client_time[idx][r] = c_time

        # 记录一个global_epoch,所有client的总训练时长
        t4 = time.perf_counter()
        all_c_time = t4 - t3
        all_client_time[r] = all_c_time
        logging.info("client_update has completed")

        # 服务器更新
        logging.info("server_update begin")
        # 服务器端更新
        server.receive_model()

        # 将二维列表转换成NumPy数组
        data_array = np.array(clients_rank_lists)
        # 定义要分成的簇的数量
        num_clusters = 7
        # 使用K均值算法进行聚类
        clustering = KMeans(n_clusters=num_clusters, random_state=0).fit(data_array)
        clusters = clustering.labels_
        # 打印每个簇中的客户端
        cluster_lists = [[] for _ in range(num_clusters)]
        for i, cluster in enumerate(clusters):
            cluster_lists[cluster].append(i)
        logging.info("clusters:%s", cluster_lists)

        rank_list = []
        for cluster in cluster_lists:
            rank = 0
            for user in cluster:
                for j in clients_rank_lists[user]:
                    rank += int(j)
            rank = rank / len(cluster)
            rank_list.append(rank)

        sorted_rank_list = sorted(rank_list, reverse=True)
        rank_positions = [sorted_rank_list.index(rank) + 1 for rank in rank_list]
        print(rank_positions)
        # 服务器接收客户端模型参数，以及全局参数聚合需要的参数信息

        w_updated = server.aggregate1(cluster_lists, server.count, rank_positions)
        # w_updated = server.aggregate(server.count)

        # 服务器聚合客户端模型参数，更新全局模型，使用公共数据集训练全局模型
        result = server.server_update1(w_updated, test_data, r)

        # 保存每轮服务器全局模型训练结果，最后平均，保存平均结果作为服务器（全局模型）训练结果
        result_dict = server.result
        for key in result_dict:
            server_results_dict[key] += result_dict[key]

        server_result = result_dict
        for key in result_dict:
            server_result[key] = round(result_dict[key], 6)
        with open(f"Results/{config['algorithm']}/each_round_server_results.txt", "a") as f:
            f.write(f"round{r} server results:{server_result}\n")

        logging.info("server_update has completed")
        # 保存最后一轮服务器模型参数，用于测试训练的全局模型性能
        if r == config["global_epochs"] - 1:
            # pass
            torch.save(server.model, f"Results/{config['algorithm']}/model_weight/server/" + "last_round_server_model.pt")

        # 记录一个global_epoch训练总时长和server端训练时长
        t2 = time.perf_counter()
        all_time = t2 - t1
        s_time = all_time - all_c_time
        total_time[r] = all_time
        server_time[r] = s_time

    # 保存训练时长
    with open(f"Results/{config['algorithm']}/time.txt", "a") as f:
        f.write(f"computation:\n")
        server_all_time = round(sum(server_time.values()), 6)
        server_avg_time = round(sum(server_time.values()) / config['global_epochs'], 6)

        all_client_all_time = round(sum(all_client_time.values()), 6)
        all_client_avg_time = round(sum(all_client_time.values()) / config['global_epochs'], 6)

        acat = 0
        for i in client_time.keys():
            for j in client_time[i].keys():
                acat += client_time[i][j]
        one_client_all_time = round(acat / config['num_clients'], 6)
        one_client_avg_time = round(acat / (config['global_epochs'] * config['num_clients']), 6)

        f.write(f"server all time:{server_all_time}, server avg time:{server_avg_time}")
        f.write(f"all client all time:{all_client_all_time}, all client avg time:{all_client_avg_time}")
        f.write(f"one client all time:{one_client_all_time}, one client avg time:{one_client_avg_time}")

    # 保存通信量记录
    with open(f"Results/{config['algorithm']}/communication_log.txt", "w") as f:
        f.write("Round Communication Log\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Round':<6} {'Downlink(MB)':<15} {'Uplink(MB)':<15} {'Total(MB)':<15} {'Model Size(MB)':<15}\n")
        f.write("-" * 60 + "\n")

        total_downlink_mb = 0
        total_uplink_mb = 0

        for log in server.communication_log:
            round_num = log['round']
            downlink_mb = log['downlink_bytes'] / (1024 ** 2)
            uplink_mb = log['uplink_bytes'] / (1024 ** 2)
            total_mb = downlink_mb + uplink_mb
            model_mb = log['model_size_bytes'] / (1024 ** 2)

            total_downlink_mb += downlink_mb
            total_uplink_mb += uplink_mb

            f.write(f"{round_num:<6} {downlink_mb:<15.2f} {uplink_mb:<15.2f} {total_mb:<15.2f} {model_mb:<15.2f}\n")

        f.write("-" * 60 + "\n")
        f.write(
            f"{'Total':<6} {total_downlink_mb:<15.2f} {total_uplink_mb:<15.2f} {(total_downlink_mb + total_uplink_mb):<15.2f} {'-':<15}\n")

        # 计算平均每轮通信量
        avg_downlink_mb = total_downlink_mb / config['global_epochs']
        avg_uplink_mb = total_uplink_mb / config['global_epochs']
        f.write(
            f"{'Avg':<6} {avg_downlink_mb:<15.2f} {avg_uplink_mb:<15.2f} {(avg_downlink_mb + avg_uplink_mb):<15.2f} {'-':<15}\n")

    # 在训练结果文件中也添加通信量摘要
    with open(
            f"./Results/{config['algorithm']}/" + f"{config['dataset']}" + f"_global_epoch_{config['global_epochs']}" + f"_local_epoch_{config['client_epochs']}" + '_train_result.txt',
            'a') as f:
        f.write("\n\nCommunication Summary:\n")
        f.write(f"Total Downlink Communication: {server.downlink_total / (1024 ** 3):.3f} GB\n")
        f.write(f"Total Uplink Communication: {server.uplink_total / (1024 ** 3):.3f} GB\n")
        f.write(f"Total Communication: {(server.downlink_total + server.uplink_total) / (1024 ** 3):.3f} GB\n")
        f.write(
            f"Average per round: {(server.downlink_total + server.uplink_total) / (1024 ** 3) / config['global_epochs']:.3f} GB\n")
    # 8.对之前保存的每轮客户端/服务器结果进行平均，保存训练结果
    for key in server_results_dict:
        server_avg_result[key] = round(server_results_dict[key] / config['global_epochs'], 6)
    for c in range(config['num_clients']):
        for key in client_results_list[c]:
            client_results_list[c][key] /= (config['global_epochs'])
            client_results_list[c][key] = round(client_results_list[c][key], 6)
    print_client_result(client_results_list, config, server_avg_result)
    logging.info("Results has saved!")

    # 9.测试
    # 在测试数据集上评估服务器模型性能
    logging.info("The server model is being evaluated on the test dataset\n")
    s_test_total_result = {'NDCG@1': 0, 'NDCG@5': 0, 'NDCG@10': 0, 'NDCG@20': 0,
                           'HIT@1': 0, 'HIT@5': 0, 'HIT@10': 0, 'HIT@20': 0}
    s_test_avg_results = {'NDCG@1': 0, 'NDCG@5': 0, 'NDCG@10': 0, 'NDCG@20': 0,
                          'HIT@1': 0, 'HIT@5': 0, 'HIT@10': 0, 'HIT@20': 0}
    for i in range(config['num_test']):
        model = torch.load(f"Results/{config['algorithm']}/model_weight/server/" + 'last_round_server_model' + '.pt')
        server_dataset = get_test_data(config['test_dataset'])
        result = []
        model.eval()
        t_test = evaluate(model, server_dataset, config, data)
        result.extend(t_test[0])
        result.extend(t_test[1])
        result_dict = {'NDCG@1': result[0], 'NDCG@5': result[1], 'NDCG@10': result[2], 'NDCG@20': result[3],
                       'HIT@1': result[4], 'HIT@5': result[5], 'HIT@10': result[6], 'HIT@20': result[7]}
        logging.info(f"Server_result NO:{i}: {result_dict}")
        for key in s_test_total_result:
            s_test_total_result[key] += result_dict[key]
    for key in s_test_avg_results:
        s_test_avg_results[key] = s_test_total_result[key] / (config['num_test'])
        s_test_avg_results[key] = round(s_test_avg_results[key], 6)

    logging.info(f'Server avg_testing: {s_test_avg_results}')

    # 在测试数据集上评估客户端模型性能
    logging.info("The client model is being evaluated on the test dataset\n")
    c_test_total_result = {'NDCG@1': 0, 'NDCG@5': 0, 'NDCG@10': 0, 'NDCG@20': 0,
                           'HIT@1': 0, 'HIT@5': 0, 'HIT@10': 0, 'HIT@20': 0}
    c_test_avg_results = {'NDCG@1': 0, 'NDCG@5': 0, 'NDCG@10': 0, 'NDCG@20': 0,
                          'HIT@1': 0, 'HIT@5': 0, 'HIT@10': 0, 'HIT@20': 0}
    for c in range(config['num_clients']):

        client_model = torch.load(f"Results/{config['algorithm']}/model_weight/clients/all_clients/" + f'last_round_client{c}_model' + '.pt')
        server_model = torch.load(f"Results/{config['algorithm']}/model_weight/server/" + 'last_round_server_model' + '.pt')
        model_equal(client_model, server_model)
        client_data = get_test_data(config['test_dataset'])
        result = []
        client_model.eval()
        t_test = evaluate(client_model, client_data, config, data)
        result.extend(t_test[0])
        result.extend(t_test[1])
        result_dict = {'NDCG@1': result[0], 'NDCG@5': result[1], 'NDCG@10': result[2],
                       'NDCG@20': result[3],
                       'HIT@1': result[4], 'HIT@5': result[5], 'HIT@10': result[6], 'HIT@20': result[7]}
        logging.info(f"Client_NO:{c} result: {result_dict}")

        with open(f"Results/{config['algorithm']}/all_client_test_results.txt", "a") as f:
            f.write(f"client{c} test results:{result_dict}\n")

        for key in c_test_total_result:
            c_test_total_result[key] += result_dict[key]
    for key in c_test_total_result:
        c_test_avg_results[key] = c_test_total_result[key] / config['num_clients']
        c_test_avg_results[key] = round(c_test_avg_results[key], 6)
    logging.info(f"Avg_testing result for {config['num_clients']} Clients results: {c_test_avg_results}")

    with open(f"Results/{config['algorithm']}/" + 'total_test_result.txt', "w") as f:
        f.write(
            "Args:   \n ,num_client: {}\n, global_epochs: {}\n ,client_epochs:{}\n ,server_epochs:{}\n ,eva_epochs:{}\n ,maxlen:{}\n dataset:{}\n,datalen_all_client:{}\n,datalen_all_server:{}\n,dropout_rate:{}\n,learning_rate:{}\n,device:{}\n,client_batch_size:{}\n, server_batch_size:{}\n, num_neg:{}, aggregation_type:{}\n, num_test\n".format(
                config['num_clients'], config['global_epochs'], config['client_epochs'], config['server_epochs'],
                config['eval_epochs'], config['maxlen'], config['dataset'], config['datalen_client'],
                config['datalen_server'], config['dropout_rate'], config['lr'], config['device'],
                config['client_batch_size'], config['server_batch_size'], config['num_neg'], config['algorithm'],
                config['num_test']))

        f.write(f"\navg result of 9 time testing_Server:\n {s_test_avg_results}\n")
        f.write(f"\navg result of all clients:\n {c_test_avg_results}\n")

    logging.info("The server/client test data is saved in the test_data file\n")
