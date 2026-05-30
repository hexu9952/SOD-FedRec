import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.lat_emb = torch.nn.Embedding(438, args.hidden_units, padding_idx=0)
        self.lon_emb = torch.nn.Embedding(592, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    # 序列转特征
    def log2feats(self, log_seqs):
        item_seqs = [[sublist[0] for sublist in row] for row in log_seqs]
        item_seqs = np.array(item_seqs)
        # print("item_seqs:", item_seqs)
        seqs = self.item_emb(torch.LongTensor(item_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        lat_seqs = [[(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in row] for row in log_seqs]
        lat_seqs = np.array(lat_seqs)
        # print("lat_seqs:", lat_seqs)
        seqs1 = self.lat_emb(torch.LongTensor(lat_seqs).to(self.dev))
        seqs1 *= self.lat_emb.embedding_dim ** 0.05
        seqs += seqs1


        lon_seqs = [[(round(((sublist[2])+74.27476645)*1000) if sublist[2] != 0 else 0) for sublist in row] for row in log_seqs]
        lon_seqs = np.array(lon_seqs)
        # print("lon_seqs:", lon_seqs)
        seqs2 = self.lon_emb(torch.LongTensor(lon_seqs).to(self.dev))
        seqs2 *= self.lat_emb.embedding_dim ** 0.05
        seqs += seqs2


        # 加入空间特征 def log2feats(self, log_seqs, spatial_feats):
        # spatial_emb = self.spatial_emb(torch.Tensor(spatial_feats).to(self.dev))
        # seqs += spatial_emb

        positions = np.tile(np.array(range(item_seqs.shape[1])), [item_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(item_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)    # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # 在这前面加入特征
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)   # (U, T, C) -> (U, -1, C)
        return log_feats

    # 训练模型
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)    # user_ids hasn't been used yet

        pos_items = np.array([[sublist[0] for sublist in row] for row in pos_seqs])
        pos_lats = np.array([[(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in row] for row in pos_seqs])
        pos_lons = np.array([[(round(((sublist[2])+74.27476645)*1000) if sublist[2] != 0 else 0) for sublist in row] for row in pos_seqs])
        # print("pos_lond:", pos_lons)
        pos_items_embs = self.item_emb(torch.LongTensor(pos_items).to(self.dev))
        pos_lats_embs = self.lat_emb(torch.LongTensor(pos_lats).to(self.dev))
        pos_lons_embs = self.lon_emb(torch.LongTensor(pos_lons).to(self.dev))
        pos_embs = pos_items_embs + pos_lats_embs + pos_lons_embs

        neg_items = np.array([[sublist[0] for sublist in row] for row in neg_seqs])
        neg_lats = np.array(
            [[(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in row] for row in neg_seqs])
        neg_lons = np.array(
            [[(round(((sublist[2]) + 74.27476645) * 1000) if sublist[2] != 0 else 0) for sublist in row] for row in neg_seqs])
        neg_items_emb = self.item_emb(torch.LongTensor(neg_items).to(self.dev))
        neg_lats_emb = self.lat_emb(torch.LongTensor(neg_lats).to(self.dev))
        neg_lons_emb = self.lon_emb(torch.LongTensor(neg_lons).to(self.dev))
        neg_embs = neg_items_emb + neg_lats_emb + neg_lons_emb

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    # 使用模型预测
    def predict(self, user_ids, log_seqs, item_indices):    # for inference
        log_feats = self.log2feats(log_seqs)    # user_ids hasn't been used yet
        # 从特征向量log_feats中选择最后一个时间步的特征作为最终的特征表示final_feat。
        final_feat = log_feats[:, -1, :]    # only use last QKV classifier, a waste

        items = np.array([sublist[0] for sublist in item_indices])
        lats = np.array([(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in item_indices])
        lons = np.array([(round(((sublist[2]) + 74.27476645) * 1000) if sublist[2] != 0 else 0) for sublist in item_indices])
        item_embs = self.item_emb(torch.LongTensor(items).to(self.dev))  # (U, I, C)
        lat_emb = self.lat_emb(torch.LongTensor(lats).to(self.dev))
        lon_emb = self.lon_emb(torch.LongTensor(lons).to(self.dev))
        embs = item_embs + lat_emb + lon_emb

        logits = embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits   # preds # (U, I)


class SASRec1(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec1, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        # 方向权重
        self.last_item_attn = None
        self.last_lon_attn = None
        self.last_lat_attn = None

        #Embding
        self.cached_item_feat = None
        self.cached_lat_feat = None
        self.cached_lon_feat = None

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.lat_emb = torch.nn.Embedding(438, args.hidden_units, padding_idx=0)
        self.lon_emb = torch.nn.Embedding(592, args.hidden_units, padding_idx=0)
        # self.lat_emb = torch.nn.Embedding(536, args.hidden_units, padding_idx=0)
        # self.lon_emb = torch.nn.Embedding(664, args.hidden_units, padding_idx=0)

        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    # 序列转特征
    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)    # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # 在这前面加入特征
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_weights = self.attention_layers[i](
                Q, seqs, seqs,
                attn_mask=attention_mask,
                need_weights=True,
                average_attn_weights=True
            )
            if i == 0:
                self.last_item_attn = attn_weights.detach().cpu()
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)   # (U, T, C) -> (U, -1, C)
        return log_feats

    def lat2feats(self, log_seqs):
        seqs = self.lat_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.lat_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)    # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # 在这前面加入特征
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_weights = self.attention_layers[i](
                Q, seqs, seqs,
                attn_mask=attention_mask,
                need_weights=True,
                average_attn_weights=True
            )
            if i == 0:
                self.last_lat_attn = attn_weights.detach().cpu()
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)   # (U, T, C) -> (U, -1, C)
        return log_feats

    def lon2feats(self, log_seqs):
        seqs = self.lon_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.lon_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)    # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # 在这前面加入特征
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_weights = self.attention_layers[i](
                Q, seqs, seqs,
                attn_mask=attention_mask,
                need_weights=True,
                average_attn_weights=True
            )
            if i == 0:
                self.last_lat_attn = attn_weights.detach().cpu()
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)   # (U, T, C) -> (U, -1, C)
        self.last_lon_attn = attn_weights.detach().cpu()

        return log_feats

    # 训练模型
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        item_seqs = np.array([[sublist[0] for sublist in row] for row in log_seqs])
        lat_seqs = np.array([[(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in row] for row in log_seqs])
        lon_seqs = np.array([[(round(((sublist[2]) + 74.27476645) * 1000) if sublist[2] != 0 else 0) for sublist in row] for row in log_seqs])
        # lat_seqs = np.array([[(round((sublist[1] - 35.51018469) * 1500) if sublist[1] != 0 else 0) for sublist in row] for row in log_seqs])
        # lon_seqs = np.array([[(round(((sublist[2]) - 139.4708776) * 1500) if sublist[2] != 0 else 0) for sublist in row] for row in log_seqs])

        item_feats = self.log2feats(item_seqs)    # user_ids hasn't been used yet
        lat_feats = self.lat2feats(lat_seqs)    # user_ids hasn't been used yet
        lon_feats = self.lon2feats(lon_seqs)    # user_ids hasn't been used yet

        pos_items = np.array([[sublist[0] for sublist in row] for row in pos_seqs])
        pos_lats = np.array([[(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in row] for row in pos_seqs])
        pos_lons = np.array([[(round(((sublist[2]) + 74.27476645) * 1000) if sublist[2] != 0 else 0) for sublist in row] for row in pos_seqs])
        # pos_lats = np.array([[(round((sublist[1] - 35.51018469) * 1500) if sublist[1] != 0 else 0) for sublist in row] for row in pos_seqs])
        # pos_lons = np.array([[(round(((sublist[2]) - 139.4708776) * 1500) if sublist[2] != 0 else 0) for sublist in row] for row in pos_seqs])
        pos_items_embs = self.item_emb(torch.LongTensor(pos_items).to(self.dev))
        pos_lats_embs = self.lat_emb(torch.LongTensor(pos_lats).to(self.dev))
        pos_lons_embs = self.lon_emb(torch.LongTensor(pos_lons).to(self.dev))

        neg_items = np.array([[sublist[0] for sublist in row] for row in neg_seqs])
        neg_lats = np.array([[(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in row] for row in neg_seqs])
        neg_lons = np.array([[(round(((sublist[2]) + 74.27476645) * 1000) if sublist[2] != 0 else 0) for sublist in row] for row in neg_seqs])
        # neg_lats = np.array([[(round((sublist[1] - 35.51018469) * 1500) if sublist[1] != 0 else 0) for sublist in row] for row in neg_seqs])
        # neg_lons = np.array([[(round(((sublist[2]) - 139.4708776) * 1500) if sublist[2] != 0 else 0) for sublist in row] for row in neg_seqs])
        neg_items_embs = self.item_emb(torch.LongTensor(neg_items).to(self.dev))
        neg_lats_embs = self.lat_emb(torch.LongTensor(neg_lats).to(self.dev))
        neg_lons_embs = self.lon_emb(torch.LongTensor(neg_lons).to(self.dev))

        pos_items_logits = (item_feats * pos_items_embs).sum(dim=-1)
        pos_lats_logits = (lat_feats * pos_lats_embs).sum(dim=-1)
        pos_lons_logits = (lon_feats * pos_lons_embs).sum(dim=-1)
        pos_logits = pos_lons_logits + pos_lats_logits +pos_items_logits

        neg_items_logits = (item_feats * neg_items_embs).sum(dim=-1)
        neg_lats_logits = (lat_feats * neg_lats_embs).sum(dim=-1)
        neg_lons_logits = (lon_feats * neg_lons_embs).sum(dim=-1)
        neg_logits = neg_lats_logits + neg_lons_logits +neg_items_logits

        return pos_logits, neg_logits # pos_pred, neg_pred

    # 使用模型预测
    def predict(self, user_ids, log_seqs, item_indices, return_embeddings = False):    # for inference
        item_seqs = np.array([[sublist[0] for sublist in row] for row in log_seqs])
        lat_seqs = np.array([[(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in row] for row in log_seqs])
        lon_seqs = np.array([[(round(((sublist[2]) + 74.27476645) * 1000) if sublist[2] != 0 else 0) for sublist in row] for row in log_seqs])
        # lat_seqs = np.array([[(round((sublist[1] - 35.51018469) * 1500) if sublist[1] != 0 else 0) for sublist in row] for row in log_seqs])
        # lon_seqs = np.array([[(round(((sublist[2]) - 139.4708776) * 1500) if sublist[2] != 0 else 0) for sublist in row] for row in log_seqs])

        item_feats = self.log2feats(item_seqs)  # user_ids hasn't been used yet
        lat_feats = self.lat2feats(lat_seqs)  # user_ids hasn't been used yet
        lon_feats = self.lon2feats(lon_seqs)  # user_ids hasn't been used yet
        # 从特征向量log_feats中选择最后一个时间步的特征作为最终的特征表示final_feat。
        final_item_feat = item_feats[:, -1, :]    # only use last QKV classifier, a waste
        final_lat_feat = lat_feats[:, -1, :]    # only use last QKV classifier, a waste
        final_lon_feat = lon_feats[:, -1, :]    # only use last QKV classifier, a waste

        # ===== cache embeddings for visualization =====
        self.cached_item_feat = final_item_feat.detach().cpu()
        self.cached_lat_feat = final_lat_feat.detach().cpu()
        self.cached_lon_feat = final_lon_feat.detach().cpu()

        items = np.array([sublist[0] for sublist in item_indices])
        lats = np.array([(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in item_indices])
        lons = np.array([(round(((sublist[2]) + 74.27476645) * 1000) if sublist[2] != 0 else 0) for sublist in item_indices])
        # lats = np.array([(round((sublist[1] - 35.51018469) * 1500) if sublist[1] != 0 else 0) for sublist in item_indices])
        # lons = np.array([(round(((sublist[2]) - 139.4708776) * 1500) if sublist[2] != 0 else 0) for sublist in item_indices])
        item_embs = self.item_emb(torch.LongTensor(items).to(self.dev))  # (U, I, C)
        lat_embs = self.lat_emb(torch.LongTensor(lats).to(self.dev))
        lon_embs = self.lon_emb(torch.LongTensor(lons).to(self.dev))

        item_logits = item_embs.matmul(final_item_feat.unsqueeze(-1)).squeeze(-1)
        lat_logits = lat_embs.matmul(final_lat_feat.unsqueeze(-1)).squeeze(-1)
        lon_logits = lon_embs.matmul(final_lon_feat.unsqueeze(-1)).squeeze(-1)
        logits = item_logits + lat_logits + lon_logits
        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        if return_embeddings:
            return logits, final_item_feat, final_lat_feat, final_lon_feat
        else:
            return logits
        # return logits   # preds # (U, I)

class SASRec_tky(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec_tky, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        # attention
        self.last_item_attn = None
        self.last_lon_attn = None
        self.last_lat_attn = None
        # Embding
        self.cached_item_feat = None
        self.cached_lat_feat = None
        self.cached_lon_feat = None
        # ===== cache for case study =====
        self.cached_candidate_items = None
        self.cached_candidate_lats = None
        self.cached_candidate_lons = None
        # ===== cache for case study (sequence-level) =====
        self.cached_item_seq_feat = None
        self.cached_lat_seq_feat = None
        self.cached_lon_seq_feat = None

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        # self.lat_emb = torch.nn.Embedding(438, args.hidden_units, padding_idx=0)
        # self.lon_emb = torch.nn.Embedding(592, args.hidden_units, padding_idx=0)
        self.lat_emb = torch.nn.Embedding(536, args.hidden_units, padding_idx=0)
        self.lon_emb = torch.nn.Embedding(664, args.hidden_units, padding_idx=0)

        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    # 序列转特征
    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)    # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # 在这前面加入特征
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_weights = self.attention_layers[i](
                Q, seqs, seqs,
                attn_mask=attention_mask,
                need_weights=True,
                average_attn_weights=True
            )
            if i == 0:
                self.last_item_attn = attn_weights.detach().cpu()
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)   # (U, T, C) -> (U, -1, C)
        return log_feats

    def lat2feats(self, log_seqs):
        seqs = self.lat_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.lat_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)    # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # 在这前面加入特征
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_weights = self.attention_layers[i](
                Q, seqs, seqs,
                attn_mask=attention_mask,
                need_weights=True,
                average_attn_weights=True
            )
            if i == 0:
                self.last_lat_attn = attn_weights.detach().cpu()
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)   # (U, T, C) -> (U, -1, C)
        return log_feats

    def lon2feats(self, log_seqs):
        seqs = self.lon_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.lon_emb.embedding_dim ** 0.5

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)    # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        # 在这前面加入特征
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, attn_weights = self.attention_layers[i](
                Q, seqs, seqs,
                attn_mask=attention_mask,
                need_weights=True,
                average_attn_weights=True
            )
            if i == 0:
                self.last_lon_attn = attn_weights.detach().cpu()
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)   # (U, T, C) -> (U, -1, C)
        return log_feats

    # 训练模型
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        item_seqs = np.array([[sublist[0] for sublist in row] for row in log_seqs])
        # lat_seqs = np.array([[(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in row] for row in log_seqs])
        # lon_seqs = np.array([[(round(((sublist[2]) + 74.27476645) * 1000) if sublist[2] != 0 else 0) for sublist in row] for row in log_seqs])
        lat_seqs = np.array([[(round((sublist[1] - 35.51018469) * 1500) if sublist[1] != 0 else 0) for sublist in row] for row in log_seqs])
        lon_seqs = np.array([[(round(((sublist[2]) - 139.4708776) * 1500) if sublist[2] != 0 else 0) for sublist in row] for row in log_seqs])

        item_feats = self.log2feats(item_seqs)    # user_ids hasn't been used yet
        lat_feats = self.lat2feats(lat_seqs)    # user_ids hasn't been used yet
        lon_feats = self.lon2feats(lon_seqs)    # user_ids hasn't been used yet

        pos_items = np.array([[sublist[0] for sublist in row] for row in pos_seqs])
        # pos_lats = np.array([[(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in row] for row in pos_seqs])
        # pos_lons = np.array([[(round(((sublist[2]) + 74.27476645) * 1000) if sublist[2] != 0 else 0) for sublist in row] for row in pos_seqs])
        pos_lats = np.array([[(round((sublist[1] - 35.51018469) * 1500) if sublist[1] != 0 else 0) for sublist in row] for row in pos_seqs])
        pos_lons = np.array([[(round(((sublist[2]) - 139.4708776) * 1500) if sublist[2] != 0 else 0) for sublist in row] for row in pos_seqs])
        pos_items_embs = self.item_emb(torch.LongTensor(pos_items).to(self.dev))
        pos_lats_embs = self.lat_emb(torch.LongTensor(pos_lats).to(self.dev))
        pos_lons_embs = self.lon_emb(torch.LongTensor(pos_lons).to(self.dev))

        neg_items = np.array([[sublist[0] for sublist in row] for row in neg_seqs])
        # neg_lats = np.array([[(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in row] for row in neg_seqs])
        # neg_lons = np.array([[(round(((sublist[2]) + 74.27476645) * 1000) if sublist[2] != 0 else 0) for sublist in row] for row in neg_seqs])
        neg_lats = np.array([[(round((sublist[1] - 35.51018469) * 1500) if sublist[1] != 0 else 0) for sublist in row] for row in neg_seqs])
        neg_lons = np.array([[(round(((sublist[2]) - 139.4708776) * 1500) if sublist[2] != 0 else 0) for sublist in row] for row in neg_seqs])
        neg_items_embs = self.item_emb(torch.LongTensor(neg_items).to(self.dev))
        neg_lats_embs = self.lat_emb(torch.LongTensor(neg_lats).to(self.dev))
        neg_lons_embs = self.lon_emb(torch.LongTensor(neg_lons).to(self.dev))

        pos_items_logits = (item_feats * pos_items_embs).sum(dim=-1)
        pos_lats_logits = (lat_feats * pos_lats_embs).sum(dim=-1)
        pos_lons_logits = (lon_feats * pos_lons_embs).sum(dim=-1)
        pos_logits = pos_lons_logits + pos_lats_logits +pos_items_logits

        neg_items_logits = (item_feats * neg_items_embs).sum(dim=-1)
        neg_lats_logits = (lat_feats * neg_lats_embs).sum(dim=-1)
        neg_lons_logits = (lon_feats * neg_lons_embs).sum(dim=-1)
        neg_logits = neg_lats_logits + neg_lons_logits +neg_items_logits

        return pos_logits, neg_logits # pos_pred, neg_pred

    # 使用模型预测
    def predict(self, user_ids, log_seqs, item_indices, return_embeddings = False):    # for inference
        item_seqs = np.array([[sublist[0] for sublist in row] for row in log_seqs])
        # lat_seqs = np.array([[(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in row] for row in log_seqs])
        # lon_seqs = np.array([[(round(((sublist[2]) + 74.27476645) * 1000) if sublist[2] != 0 else 0) for sublist in row] for row in log_seqs])
        lat_seqs = np.array([[(round((sublist[1] - 35.51018469) * 1500) if sublist[1] != 0 else 0) for sublist in row] for row in log_seqs])
        lon_seqs = np.array([[(round(((sublist[2]) - 139.4708776) * 1500) if sublist[2] != 0 else 0) for sublist in row] for row in log_seqs])

        item_feats = self.log2feats(item_seqs)  # user_ids hasn't been used yet
        lat_feats = self.lat2feats(lat_seqs)  # user_ids hasn't been used yet
        lon_feats = self.lon2feats(lon_seqs)  # user_ids hasn't been used yet
        # ===== cache full sequence embeddings (for case study) =====
        self.cached_item_seq_feat = item_feats.detach().cpu()
        self.cached_lat_seq_feat = lat_feats.detach().cpu()
        self.cached_lon_seq_feat = lon_feats.detach().cpu()

        # 从特征向量log_feats中选择最后一个时间步的特征作为最终的特征表示final_feat。
        final_item_feat = item_feats[:, -1, :]    # only use last QKV classifier, a waste
        final_lat_feat = lat_feats[:, -1, :]    # only use last QKV classifier, a waste
        final_lon_feat = lon_feats[:, -1, :]    # only use last QKV classifier, a waste

        self.cached_item_feat = final_item_feat.detach().cpu()
        self.cached_lat_feat = final_lat_feat.detach().cpu()
        self.cached_lon_feat = final_lon_feat.detach().cpu()
        # ===== cache embeddings for visualization =====
        self.cached_item_feat = final_item_feat.detach().cpu()
        self.cached_lat_feat = final_lat_feat.detach().cpu()
        self.cached_lon_feat = final_lon_feat.detach().cpu()
        items = np.array([sublist[0] for sublist in item_indices])
        # lats = np.array([(round((sublist[1] - 40.55085247) * 1000) if sublist[1] != 0 else 0) for sublist in item_indices])
        # lons = np.array([(round(((sublist[2]) + 74.27476645) * 1000) if sublist[2] != 0 else 0) for sublist in item_indices])
        lats = np.array([(round((sublist[1] - 35.51018469) * 1500) if sublist[1] != 0 else 0) for sublist in item_indices])
        lons = np.array([(round(((sublist[2]) - 139.4708776) * 1500) if sublist[2] != 0 else 0) for sublist in item_indices])
        item_embs = self.item_emb(torch.LongTensor(items).to(self.dev))  # (U, I, C)
        lat_embs = self.lat_emb(torch.LongTensor(lats).to(self.dev))
        lon_embs = self.lon_emb(torch.LongTensor(lons).to(self.dev))

        item_logits = item_embs.matmul(final_item_feat.unsqueeze(-1)).squeeze(-1)
        lat_logits = lat_embs.matmul(final_lat_feat.unsqueeze(-1)).squeeze(-1)
        lon_logits = lon_embs.matmul(final_lon_feat.unsqueeze(-1)).squeeze(-1)
        logits = item_logits + lat_logits + lon_logits
        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        self.cached_candidate_items = items
        self.cached_candidate_lats = lats
        self.cached_candidate_lons = lons

        if return_embeddings:
            return logits, final_item_feat, final_lat_feat, final_lon_feat
        else:
            return logits
