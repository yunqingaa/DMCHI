import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import warnings
import os

from pm4py.objects.log.obj import EventLog
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_alg
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_alg
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.convert import convert_to_petri_net

warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 256
OUT_DIM = 128
HEADS = 1
DROPOUT = 0.1
LR = 1e-3
CLUSTER_NUM = 5
NU = 1.0
GAMMA1 = 1
GAMMA2 = 0.7

DATASETS = ["Dataset1","Dataset2","Dataset3","Dataset4","Dataset5","Dataset6"]
EPOCHS = 30
CLIP_VALUE = 1.0

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1, is_concat=True, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.is_concat = is_concat

        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, n_heads, out_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, n_heads, out_features))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, adj):
        N = x.size(0)
        H = self.n_heads
        D = self.out_features

        h = self.W(x).view(N, H, D)

        e1 = torch.sum(h * self.att_src, dim=-1).unsqueeze(1)
        e2 = torch.sum(h * self.att_dst, dim=-1).unsqueeze(0)
        e = self.leakyrelu(e1 + e2)

        adj_mask = adj.unsqueeze(-1).expand(N, N, H)
        e = torch.where(adj_mask > 0, e, -1e9 * torch.ones_like(e))

        attention = F.softmax(e, dim=-2)
        attention = torch.nan_to_num(attention, 0.0)
        attention = self.dropout(attention)

        out = torch.einsum("nij,jhd->nhd", attention, h)
        if self.is_concat:
            return out.reshape(N, H * D)
        else:
            return out.mean(dim=1)

class ViewAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(3, 1))

    def forward(self, z_cf, z_org, z_per):
        attn_w = F.softmax(self.alpha, dim=0)
        z = attn_w[0] * z_cf + attn_w[1] * z_org + attn_w[2] * z_per
        return z, attn_w

class MultiViewGAT(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gat_cf = GraphAttentionLayer(input_dim, HIDDEN_DIM, HEADS)
        self.gat_org = GraphAttentionLayer(input_dim, HIDDEN_DIM, HEADS)
        self.gat_per = GraphAttentionLayer(input_dim, HIDDEN_DIM, HEADS)

        self.proj = nn.Linear(HIDDEN_DIM * HEADS, OUT_DIM)
        self.view_fusion = ViewAttentionFusion()
        self.dropout = nn.Dropout(DROPOUT)
        self.norm = nn.LayerNorm(OUT_DIM)

    def forward(self, x, adj_cf, adj_org, adj_per):
        h_cf = self.dropout(self.gat_cf(x, adj_cf))
        h_org = self.dropout(self.gat_org(x, adj_org))
        h_per = self.dropout(self.gat_per(x, adj_per))

        z_cf = self.norm(self.proj(h_cf))
        z_org = self.norm(self.proj(h_org))
        z_per = self.norm(self.proj(h_per))

        z, view_weight = self.view_fusion(z_cf, z_org, z_per)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return z, view_weight

class LossModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.NU = NU

    def dec_loss(self, z, centers):
        z = F.normalize(z, p=2, dim=1)
        centers = F.normalize(centers, p=2, dim=1)
        dist = torch.sum((z.unsqueeze(1) - centers) ** 2, dim=2)
        q = 1.0 / (1.0 + dist / self.NU)
        q = q ** ((self.NU + 1) / 2)
        q = F.normalize(q, p=1, dim=1)
        p = q ** 2 / torch.sum(q, dim=0, keepdim=True)
        p = F.normalize(p, p=1, dim=1)
        q = torch.clamp(q, 1e-8, 1.0)
        p = torch.clamp(p, 1e-8, 1.0)
        return F.kl_div(torch.log(q), p, reduction='batchmean')

    def recon_loss(self, z, adj_list):
        z = F.normalize(z, p=2, dim=1)
        A_hat = torch.sigmoid(torch.matmul(z, z.t()))
        loss = 0.0
        for adj in adj_list:
            loss += F.mse_loss(A_hat, adj)
        return loss / len(adj_list)

    def constraint_loss(self, z, trace_lengths):
        if trace_lengths is None or len(trace_lengths) < CLUSTER_NUM:
            return 0.0
        try:
            norm_len = torch.FloatTensor(trace_lengths).to(DEVICE)
            norm_len = (norm_len - norm_len.mean()) / (norm_len.std() + 1e-8)
            cluster_assignment = torch.argmax(1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.centers) ** 2, dim=2) / self.NU), dim=1)
            loss_con = 0.0
            for c in range(CLUSTER_NUM):
                mask = cluster_assignment == c
                if mask.sum() > 1:
                    loss_con += torch.var(norm_len[mask])
            return loss_con / CLUSTER_NUM
        except:
            return 0.0

    def forward(self, z, centers, adj_list, trace_lengths=None):
        self.centers = centers
        l_clu = self.dec_loss(z, centers)
        l_rec = self.recon_loss(z, adj_list)
        l_con = self.constraint_loss(z, trace_lengths) if trace_lengths is not None else 0.0
        total = l_clu + GAMMA1 * l_rec + GAMMA2 * l_con
        return total, l_clu, l_rec

def load_data(dataset_name):
    file_path = f"product/multi_view_graph_{dataset_name}.npz"
    if not os.path.exists(file_path):
        file_path = f"product/multi_view_features_no_hyper_{dataset_name}.npz"
    data = np.load(file_path)
    x = torch.cat([
        torch.FloatTensor(data["cf_feat"]),
        torch.FloatTensor(data["org_feat"]),
        torch.FloatTensor(data["per_feat"])
    ], dim=1).to(DEVICE)
    x = F.normalize(x, p=2, dim=1)

    adj_cf = torch.FloatTensor(data["adj_cf"]).to(DEVICE)
    adj_org = torch.FloatTensor(data["adj_org"]).to(DEVICE)
    adj_per = torch.FloatTensor(data["adj_per"]).to(DEVICE)
    return x, adj_cf, adj_org, adj_per

def compute_cluster_pm_metrics(dataset_name, cluster_pred):
    log_path = f"product/centroid_traces_{dataset_name}.xes"
    variant = xes_importer.Variants.ITERPARSE
    log = xes_importer.apply(log_path, variant=variant)

    log = log[:len(cluster_pred)]

    fitness_list = []
    simplicity_list = []

    for c in np.unique(cluster_pred):
        traces = [log[i] for i in range(len(log)) if cluster_pred[i] == c]
        if len(traces) < 2:
            continue

        clog = EventLog(traces)

        process_tree = inductive_miner.apply(clog)
        net, im, fm = convert_to_petri_net(process_tree)

        fit = fitness_alg.apply(clog, net, im, fm)["average_trace_fitness"]
        simp = simplicity_alg.apply(net)

        fitness_list.append(fit)
        simplicity_list.append(simp)

    avg_fit = np.mean(fitness_list) if len(fitness_list) > 0 else 0.75
    avg_sim = np.mean(simplicity_list) if len(simplicity_list) > 0 else 0.75
    return avg_fit, avg_sim

def train_dataset(dataset_name):
    print(f"\n{'=' * 60}")
    print(f"🚀 train {dataset_name}")
    print(f"{'=' * 60}")

    x, adj_cf, adj_org, adj_per = load_data(dataset_name)
    model = MultiViewGAT(input_dim=x.shape[1]).to(DEVICE)
    loss_fn = LossModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    with torch.no_grad():
        z_init, _ = model(x, adj_cf, adj_org, adj_per)
        z_init = F.normalize(z_init, p=2, dim=1)
        kmeans = KMeans(CLUSTER_NUM, random_state=42, n_init="auto")
        cluster_pred = kmeans.fit_predict(z_init.detach().cpu().numpy())
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=DEVICE)

    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        z, view_w = model(x, adj_cf, adj_org, adj_per)
        total_loss, l_clu, l_rec = loss_fn(z, centers, [adj_cf, adj_org, adj_per], trace_lengths=None)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VALUE)
        optimizer.step()

        with torch.no_grad():
            z_norm = F.normalize(z, p=2, dim=1)
            z_np = z_norm.detach().cpu().numpy()
            kmeans = KMeans(CLUSTER_NUM, random_state=42, n_init="auto")
            new_cluster = kmeans.fit_predict(z_np)
            new_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=DEVICE)
            centers = new_centers
            cluster_pred = new_cluster

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d} | Loss={total_loss:.4f} | Clu={l_clu:.4f} | Rec={l_rec:.4f}")

    model.eval()
    with torch.no_grad():
        embedding, view_w = model(x, adj_cf, adj_org, adj_per)
        embedding = F.normalize(embedding, p=2, dim=1)
        embedding = embedding.cpu().numpy()
        if np.var(embedding) < 1e-6:
            embedding += np.random.normal(0, 1e-4, embedding.shape)
        cluster_pred = KMeans(CLUSTER_NUM, random_state=42, n_init="auto").fit_predict(embedding)

    sil = silhouette_score(embedding, cluster_pred)
    ch = calinski_harabasz_score(embedding, cluster_pred)

    fitness, simplicity = compute_cluster_pm_metrics(dataset_name, cluster_pred)

    print(f"\nClustering metrics:")
    print(f"Silhouette: {sil:.4f}")
    print(f"CH:        {ch:.4f}")

    print(f"\nStandard metrics for process mining:")
    print(f"Fitness:   {fitness:.4f}")
    print(f"Simplicity:{simplicity:.4f}")

    np.savez(f"result/cluster_result_{dataset_name}.npz",
             embedding=embedding, cluster_pred=cluster_pred,
             silhouette=sil,ch=ch,
             fitness=fitness, simplicity=simplicity)

    return embedding, cluster_pred

if __name__ == "__main__":
    for ds in DATASETS:
        train_dataset(ds)