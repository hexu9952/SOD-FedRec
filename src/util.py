import copy
import torch
import random
import logging
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


# 设置日志
def set_logger(log_path):
    """
    设置日志，保存日志信息到 log_path 中。可以将在终端的输出信息保存下来。
    """
    logger = logging.getLogger()
    logger.handlers.clear()

    logger.setLevel(logging.INFO)
    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


# 取样
def random_neq(user, data):
    t = random.choice(data)
    while t[0] in [row[0] for row in user]:
        t = random.choice(data)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED, data):
    def sample():
        user_list = []
        for key in user_train:
            user_list.append(key)

        user = random.choice(user_list)
        # 从user集中随机生成一个user_id
        while len(user_train[user]) <= 1: user = random.choice(user_list)
        # 当此user的训练item小于等于1时，则重新取一个user

        # 定义整数a和浮点数b、c
        a = 0
        b = 0.0
        c = 0.0
        # 创建二维NumPy数组
        seq = np.array([[a, b, c] for _ in range(maxlen)])
        pos = np.array([[a, b, c] for _ in range(maxlen)])
        neg = np.array([[a, b, c] for _ in range(maxlen)])

        # 初始化seq, pos, neg
        nxt = user_train[user][-1]
        idx = maxlen - 1

        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:

                neg[idx] = random_neq(user_train[user], data)

            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size, maxlen, n_workers, data):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      data
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# 获取全局数据
def get_global_data(dataset):
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    user_list = []
    num_user = 1000
    usernum = 0
    itemnum = 0

    f = open('./data/' + dataset + '.txt', 'r', encoding='ISO-8859-1')
    for line in f:
        if len(user_list) > num_user:
            break
        u, i, v_cat_id, v_cat, lat, lon, time, time_UTC = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        if u in user_list:
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
        else:
            user_list.append(u)
            # 添加，每个用户的第一个签到记录
            User[u].append(i)

    f.close()
    user_list.pop(-1)

    # for user in User:
    # 修改，User中有1001个用户
    for user in user_list:
        nfeedback = len(User[user])

        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    global_data = [user_train, user_valid, user_test, usernum, itemnum]
    return global_data


# 获取客户端本地数据，返回值count表示该客户端示例数
def get_local_data(dataset, c_id):
    User = defaultdict(list)
    user_traj = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    user_list = []
    item_list = []
    usernum = 0
    itemnum = 0
    count = 0

    f = open('./data/' + dataset + '/client_{}_data.txt'.format(c_id), 'r', encoding='ISO-8859-1')
    for line in f:
        u_id, v_id, v_cat_id, v_cat, lat, lon, time, time_UTC = line.rstrip().split('\t')
        u_id = int(u_id)
        v_id = int(v_id)
        lat = float(lat)
        lon = float(lon)
        if u_id in user_list:
            usernum = max(u_id, usernum)
            itemnum = max(v_id, itemnum)
            # User[u_id].append(v_id)
            User[u_id].append([v_id, lat, lon])
        else:
            user_list.append(u_id)
            # User[u_id].append(v_id)
            User[u_id].append([v_id, lat, lon])

    for user in User:
        nfeedback = len(User[user])
        count += len(User[user])

        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    client_data = [user_train, user_valid, user_test, usernum, itemnum]
    return client_data, count


# 获取公共数据集
def get_server_data(dataset):
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    user_list = []
    item_list = []

    usernum = 0
    itemnum = 0

    f = open('./data/' + dataset + '/server_data.txt', 'r', encoding='ISO-8859-1')
    for line in f:
        if len(user_list) > 999:
            break
        u, i, v_cat_id, v_cat, lat, lon, time, time_UTC = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        if u in user_list:
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
        else:
            user_list.append(u)
            # 添加，每个用户的第一个签到记录
            User[u].append(i)

    for user in User:
        nfeedback = len(User[user])

        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    server_data = [user_train, user_valid, user_test, usernum, itemnum]
    return server_data


def print_client_result(client_results_list, config, ser_avg_results):
    # print("正在处理输出结果\n")
    logging.info("Processing the output")
    client_data_ndcg10 = []
    client_data_ht10 = []
    total_client = {'NDCG@1': 0.0, 'NDCG@5': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0,
                    'HT@1': 0.0, 'HT@5': 0.0, 'HT@10': 0.0, 'HT@20': 0.0}
    average_client = {'NDCG@1': 0.0, 'NDCG@5': 0.0, 'NDCG@10': 0.0, 'NDCG@20': 0.0,
                    'HT@1': 0.0, 'HT@5': 0.0, 'HT@10': 0.0, 'HT@20': 0.0}

    for i in range(len(client_results_list)):
        client_data_ndcg10.append(client_results_list[i]["NDCG@10"])
    for i in range(len(client_results_list)):
        client_data_ht10.append(client_results_list[i]["HT@10"])

    # Step 2: 获取前五个客户端的训练指标
    top_10_clients = sorted(client_results_list, key=lambda x: x['NDCG@10'], reverse=True)[:10]

    # Step 3: 获取最差的五个客户端的训练指标
    worst_10_clients = sorted(client_results_list, key=lambda x: x['NDCG@10'])[:10]

    # Step 4: 计算全部客户端的平均训练指标
    for key in total_client:
        for c in range(config['num_clients']):
            total_client[key] += client_results_list[c][key]

    for key in total_client:
        average_client[key] = round(total_client[key] / (config['num_clients']), 6)

    with open(f"./Results/{config['algorithm']}/" + f"{config['dataset']}" + f"_global_epoch_{config['global_epochs']}" + f"_local_epoch_{config['client_epochs']}" + '_train_result.txt', 'w') as f:
        f.write(
            "Args:   \n ,num_client: {}\n, global_epochs: {}\n ,client_epochs:{}\n ,server_epochs:{}\n ,eva_epochs:{}\n ,maxlen:{}\n dataset:{}\n,datalen_all_client:{}\n,datalen_all_server:{}\n,dropout_rate:{}\n,learning_rate:{}\n,device:{}\n,client_batch_size:{}\n, server_batch_size:{}\n, num_neg:{}, aggregation_type:{}\n, num_test\n".format(
                config['num_clients'], config['global_epochs'], config['client_epochs'], config['server_epochs'],
                config['eval_epochs'], config['maxlen'], config['dataset'], config['datalen_client'],
                config['datalen_server'], config['dropout_rate'], config['lr'], config['device'],
                config['client_batch_size'], config['server_batch_size'], config['num_neg'], config['algorithm'],
                config['num_test']))

        f.write("\n Top 10 clients sorted by NDCG@10:\n")
        for client in top_10_clients:
            f.write(str(client) + "\n")

        f.write("\n Worst 10 clients sorted by NDCG@10:\n")
        for client in worst_10_clients:
            f.write(str(client) + "\n")

        f.write("\n Average client result:\n")
        f.write(str(average_client) + "\n")

        f.write("\n\n all round avg_Server_results:\n")
        f.write(str(ser_avg_results) + "\n")



def get_test_data(fname): # 1000-1050
    input_file = f"./data/" + fname
    usernum = 0
    itemnum = 0
    user_list = []
    item_list = []
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open(input_file + '.txt', 'r', encoding='ISO-8859-1')
    for line in f:
        u, i, v_cat_id, v_cat, lat, lon, time, time_UTC = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        lat = float(lat)
        lon = float(lon)
        if u in user_list:
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            # User[u].append(i)
            User[u].append([i, lat, lon])
        else:
            user_list.append(u)
            # User[u].append(i)
            User[u].append([i, lat, lon])

    f.close()
    user_list.pop(-1)
    for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])

    test_dataset = [user_train, user_valid, user_test, usernum, itemnum, user_list]
    return test_dataset


# 模型评估
def evaluate(model, dataset, config, data, embding = False):
    item_emb_list = []
    lat_emb_list = []
    lon_emb_list = []
    spatial_emb_list = []

    [train, valid, test, usernum, itemnum, userlist] = copy.deepcopy(dataset)

    NDCG_list = [0, 0, 0, 0]
    HT_list = [0, 0, 0, 0]
    valid_user = 0.0
    temp = [1, 5, 10, 20]
    # 模型结果
    rank_list = []

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = userlist
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        # seq = np.zeros([config['maxlen']], dtype=np.int32)
        seq = np.array([[0, 0.0, 0.0] for _ in range(config['maxlen'])])
        idx = config['maxlen'] - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        # user_tuple = [row[0] for row in train[u]]
        # rated = set(user_tuple)
        rated = train[u]
        rated.append([0, 0.0, 0.0])

        item_idx = [test[u][0]]
        for _ in range(config['num_neg']):
            t = random.choice(data)
            while t in rated: t = random.choice(data)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        item_emb = model.cached_item_feat  # shape: (1, D)
        # item_emb_list.append(item_emb.squeeze(0).numpy())

        if hasattr(model, "cached_lat_feat") and hasattr(model, "cached_lon_feat"):
            lat_emb = model.cached_lat_feat
            lon_emb = model.cached_lon_feat
            lat_emb_list.append(lat_emb.squeeze(0).numpy())
            lon_emb_list.append(lon_emb.squeeze(0).numpy())

        elif hasattr(model, "cached_spatial_feat"):
            spatial_emb = model.cached_spatial_feat
            spatial_emb_list.append(spatial_emb.squeeze(0).numpy())

        prediction = predictions[0]
        rank = prediction.argsort().argsort()[0].item()
        rank_list.append(rank)
        valid_user += 1
        for k in temp:
            if rank < k:
                NDCG_list[temp.index(k)] += 1 / np.log2(rank + 2)
                HT_list[temp.index(k)] += 1

    for j in range(4):
        NDCG_list[j] = NDCG_list[j] / valid_user
        HT_list[j] = HT_list[j] / valid_user

    NDCG_list = [round(x, 6) for x in NDCG_list]
    HIT_list = [round(x, 6) for x in HT_list]
    if embding:
        if len(lat_emb_list) > 0:  # SOD
            return NDCG_list, HIT_list, rank_list, item_emb_list, lat_emb_list, lon_emb_list

        else:  # JointSpatial
            return NDCG_list, HIT_list, rank_list, item_emb_list, spatial_emb_list

    else:
        return NDCG_list, HIT_list, rank_list



def model_equal(model1, model2):
    weights1 = model1.state_dict()  # Get the weight parameters of model 1
    weights2 = model2.state_dict()  # Get the weight parameters of model 2

    if weights1.keys() == weights2.keys():
        for key in weights1.keys():
            if not torch.all(torch.eq(weights1[key], weights2[key])):
                logging.info("The weight parameters of the two models are different")
                break
        else:
            logging.info("The weight parameters of the two models are the same")
    else:
        logging.info("The weight parameters of the two models are different")

def cluster(rank_list):
    num = len(rank_list)
    lenth = len(rank_list[0])
    distance = [[0 for _ in range(num)] for _ in range(num)]
    for l in range(lenth):
        for i in range(num):
            for j in range(num):
                distance[i][j] += rank_list[j][l] - rank_list[i][l]



import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def case_study_user(
    model,
    user_id,
    user_txt_path,
    dataset,
    config,
    data,
    save_dir,
    L=50
):
    """
    Case study for a single user:
    - t-SNE of modality embeddings
    - Attention heatmaps
    - Spatial ranking visualization
    """

    train, valid, test, *_ = dataset

    # ========== 1. build input sequence ==========
    seq = np.array([[0, 0.0, 0.0] for _ in range(config['maxlen'])])
    idx = config['maxlen'] - 1
    seq[idx] = valid[user_id][0]
    idx -= 1
    for i in reversed(train[user_id]):
        seq[idx] = i
        idx -= 1
        if idx == -1:
            break

    # ========== 2. build candidate set ==========
    rated = train[user_id] + [[0, 0.0, 0.0]]

    item_idx = [test[user_id][0]]  # GT
    for _ in range(config['num_neg']):
        t = random.choice(data)
        while t in rated:
            t = random.choice(data)
        item_idx.append(t)

    # ========== 3. model prediction ==========
    scores = -model.predict(
        np.array([user_id]),
        np.array([seq]),
        item_idx,
        return_embeddings=False
    )[0]

    rank = scores.argsort().argsort()[0].item()
    top1_idx = scores.argmin().item()

    # # ========== 4. t-SNE ==========
    # item_emb = model.cached_item_seq_feat[0, -L:].numpy()
    # lat_emb = model.cached_lat_seq_feat[0, -L:].numpy()
    # lon_emb = model.cached_lon_seq_feat[0, -L:].numpy()
    #
    # X = np.concatenate([item_emb, lat_emb, lon_emb], axis=0)
    #
    # labels = (
    #         ["POI"] * len(item_emb)
    #         + ["Longitude"] * len(lon_emb)
    #         + ["Latitude"] * len(lat_emb)
    # )
    # perp = min(30, max(2, X.shape[0] // 5))
    # tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    #
    # X_2d = tsne.fit_transform(X)
    #
    # plt.figure(figsize=(5.5, 5.5))
    #
    # n = len(item_emb)
    # idx_item = range(0, n)
    # idx_lon = range(n, 2 * n)
    # idx_lat = range(2 * n, 3 * n)
    #
    # plt.scatter(X_2d[idx_item, 0], X_2d[idx_item, 1],
    #             s=30, c="#4C72B0", alpha=0.8,
    #             label="POI Embedding")
    #
    # plt.scatter(X_2d[idx_lon, 0], X_2d[idx_lon, 1],
    #             s=30, c="#55A868", alpha=0.8,
    #             label="Longitude Embedding")
    #
    # plt.scatter(X_2d[idx_lat, 0], X_2d[idx_lat, 1],
    #             s=30, c="#C44E52", alpha=0.8,
    #             label="Latitude Embedding")
    #
    # # ===== axis labels =====
    # plt.xlabel("t-SNE Dimension 1", fontsize=13)
    # plt.ylabel("t-SNE Dimension 2", fontsize=13)
    #
    # # ===== ticks (weak but present) =====
    # plt.tick_params(axis='both',
    #                 which='major',
    #                 labelsize=11,
    #                 length=4,
    #                 width=0.8)
    #
    # # ===== academic grid =====
    # plt.grid(True,
    #          linestyle="--",
    #          linewidth=0.7,
    #          alpha=0.5,
    #          zorder=0)
    #
    # # ===== legend =====
    # plt.legend(fontsize=11,
    #            frameon=False,
    #            loc="best")
    #
    # # plt.title(
    # #     f"User {user_id}: t-SNE Visualization of Learned Embeddings",
    # #     fontsize=12
    # # )
    #
    # plt.tight_layout()
    # plt.savefig(f"{save_dir}/tsne.png", dpi=1000, bbox_inches="tight")
    # plt.close()
    #
    # # ========== 5. attention heatmaps ==========
    # item_attn = model.last_item_attn.squeeze().numpy()[-L:, -L:]
    # lat_attn = model.last_lat_attn.squeeze().numpy()[-L:, -L:]
    # lon_attn = model.last_lon_attn.squeeze().numpy()[-L:, -L:]
    #
    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # for ax, attn, title, cmap in zip(
    #         axes,
    #         [item_attn, lon_attn, lat_attn],
    #         ["POI Attention", "Longitude Attention", "Latitude Attention"],
    #         ["Greens", "Blues", "Reds"]
    # ):
    #     im = ax.imshow(attn, cmap=cmap)
    #     ax.set_title(title)
    #     ax.set_xlabel("Key")
    #     ax.set_ylabel("Query")
    #
    # plt.tight_layout()
    # plt.savefig(f"{save_dir}/attention.png", dpi=1000)
    # plt.close()

    # ========== 6. spatial ranking visualization ==========
    # ---- build POI lookup table ----
    poi2coord = {p[0]: (p[1], p[2]) for p in data}

    # ---- model input history (train + last valid) ----
    history = train[user_id] + [valid[user_id][0]]

    hist_lats = [poi2coord[p[0]][0] for p in history]
    hist_lons = [poi2coord[p[0]][1] for p in history]

    last_lat, last_lon = hist_lats[-1], hist_lons[-1]

    # ---- GT & Top-1 coordinates ----
    gt_poi = item_idx[0][0]
    top1_poi = item_idx[top1_idx][0]

    gt_lat, gt_lon = poi2coord[gt_poi]
    top1_lat, top1_lon = poi2coord[top1_poi]

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(8, 8))

    # ---- axis range with margin ----
    lat_min, lat_max = min(hist_lats), max(hist_lats)
    lon_min, lon_max = min(hist_lons), max(hist_lons)

    lat_margin = (lat_max - lat_min) * 0.1
    lon_margin = (lon_max - lon_min) * 0.1

    # ---- history points ----
    ax.scatter(hist_lons, hist_lats,
               c='gray', alpha=0.5, s=80,
               edgecolors='black', linewidth=0.4,
               label='History Check-ins')

    # ---- last input ----
    ax.scatter(last_lon, last_lat,
               c='orange', s=260, marker='*',
               edgecolors='black',
               label='Last Input')

    # ---- GT & Top-1 ----
    ax.scatter(gt_lon, gt_lat,
               c='green', s=240, marker='P',
               edgecolors='black',
               label='Ground Truth')

    ax.scatter(top1_lon, top1_lat,
               c='red', s=240, marker='X',
               edgecolors='black',
               label='Top-1 Prediction')

    # ---- annotate GT rank ----
    ax.annotate(f"Rank: {rank + 1}",
                (gt_lon, gt_lat),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=15,
                fontweight='bold',
                color='green')

    # ---- labels & title ----
    ax.set_xlabel("Longitude", fontsize=16)
    ax.set_ylabel("Latitude", fontsize=16)
    # ax.set_title(f"User {user_id}: Spatial Ranking Result",fontsize=14, fontweight='bold')

    # ---- axis range ----
    ax.set_xlim(lon_min - lon_margin, lon_max + lon_margin)
    ax.set_ylim(lat_min - lat_margin, lat_max + lat_margin)

    # ---- grid ----
    ax.grid(True, alpha=0.3, linestyle='--')

    # ---- legend ----
    ax.legend(
        fontsize=15,
        frameon=False,
        loc='best'
    )

    plt.tight_layout()
    plt.savefig(f"{save_dir}/ranking_map.png", dpi=600, bbox_inches="tight")
    plt.close()