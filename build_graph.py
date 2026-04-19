import numpy as np
import os
import json
from pm4py.objects.log.importer.xes import importer as xes_importer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections import Counter
import Levenshtein
from scipy.stats import pearsonr

os.makedirs("product", exist_ok=True)
DATASETS = [{"name": f"Dataset{i}"} for i in range(1, 7)]
ETA = 0.5
NGRAM_RANGE = (1, 3)
KNN_NEIGHBORS = 10

def build_global_vocabs():
    all_ngrams = set()
    all_resources = set()
    for ds in DATASETS:
        xes_file = f"product/centroid_traces_{ds['name']}.xes"
        if not os.path.exists(xes_file):
            continue
        log = xes_importer.apply(xes_file)
        for trace in log:
            acts = [e["concept:name"] for e in trace]
            for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
                for i in range(len(acts) - n + 1):
                    all_ngrams.add(tuple(acts[i:i+n]))
            for e in trace:
                if "org:resource" in e:
                    all_resources.add(e["org:resource"])
    return sorted(all_ngrams), sorted(all_resources)

GLOBAL_NGRAM_VOCAB, GLOBAL_RES_VOCAB = build_global_vocabs()

def extract_cf(trace):
    vec = np.zeros(len(GLOBAL_NGRAM_VOCAB), dtype=np.float32)
    acts = [e["concept:name"] for e in trace]
    cnt = Counter()
    for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
        for i in range(len(acts) - n + 1):
            cnt[tuple(acts[i:i+n])] += 1
    for i, ng in enumerate(GLOBAL_NGRAM_VOCAB):
        vec[i] = cnt.get(ng, 0)
    return vec

def extract_org(trace):
    vec = np.zeros(len(GLOBAL_RES_VOCAB), dtype=np.float32)
    res_set = {e.get("org:resource", "") for e in trace if "org:resource" in e}
    for i, r in enumerate(GLOBAL_RES_VOCAB):
        if r in res_set:
            vec[i] = 1.0
    return vec

def extract_per(trace):
    times = [e["time:timestamp"] for e in trace if "time:timestamp" in e]
    if len(times) < 2:
        return np.zeros(4, dtype=np.float32)
    durations = [(times[i+1]-times[i]).total_seconds() for i in range(len(times)-1)]
    total = sum(durations)
    mean = np.mean(durations)
    std = np.std(durations) if len(durations) > 1 else 0.0
    wait = sum(durations[i] for i in range(len(durations)) if i % 2 == 1) if len(durations) > 1 else 0.0
    wait_ratio = wait / total if total > 0 else 0.0
    return np.array([np.log1p(total), mean, std, wait_ratio], dtype=np.float32)

def load_data(ds_name):
    log = xes_importer.apply(f"product/centroid_traces_{ds_name}.xes")
    trace_dict = {str(t.attributes.get("case:concept:name", t.attributes.get("concept:name", ""))).strip(): t for t in log}
    with open(f"product/centroids_result_{ds_name}.json", "r") as f:
        centroids_dict = json.load(f)
    return trace_dict, centroids_dict

def build_hyper_instances(trace_dict, centroids_dict):
    hyper_list = []
    for cid, centroid_list in centroids_dict.items():
        centroids = []
        for trace_id, _ in centroid_list:
            trace_id = str(trace_id).strip()
            if trace_id not in trace_dict:
                continue
            trace = trace_dict[trace_id]
            cf = extract_cf(trace)
            org = extract_org(trace)
            per = extract_per(trace)
            act_seq = tuple(e["concept:name"] for e in trace)
            n = len(centroid_list)
            centroids.append({
                "cf_feat": cf, "org_feat": org, "per_feat": per,
                "act_seq": act_seq,
                "freq": 1.0/n, "res_stability": 1.0/n, "time_stability": 1.0/n
            })
        if centroids:
            hyper_list.append({"centroids": centroids})
    return hyper_list

def aggregate_features(hyper_list):
    n = len(hyper_list)
    X_cf = np.zeros((n, len(GLOBAL_NGRAM_VOCAB)), dtype=np.float32)
    X_org = np.zeros((n, len(GLOBAL_RES_VOCAB)), dtype=np.float32)
    X_per = np.zeros((n, 4), dtype=np.float32)
    for i, inst in enumerate(hyper_list):
        cs = inst["centroids"]
        w_cf = np.array([c["freq"] for c in cs])
        w_org = np.array([c["res_stability"] for c in cs])
        w_per = np.array([c["time_stability"] for c in cs])
        X_cf[i] = np.sum(w_cf[:,None] * np.array([c["cf_feat"] for c in cs]), axis=0)
        X_org[i] = np.sum(w_org[:,None] * np.array([c["org_feat"] for c in cs]), axis=0)
        X_per[i] = np.sum(w_per[:,None] * np.array([c["per_feat"] for c in cs]), axis=0)
    X_cf = StandardScaler().fit_transform(X_cf)
    X_org = StandardScaler().fit_transform(X_org)
    X_per = StandardScaler().fit_transform(X_per)
    return X_cf, X_org, X_per

def compute_similarities(hyper_list, X_cf, X_org, X_per):
    n = len(hyper_list)
    S_cf = np.zeros((n, n))
    S_org = np.zeros((n, n))
    S_per = np.zeros((n, n))
    eta=0.5
    for i in range(n):
        seqs_i = [c["act_seq"] for c in hyper_list[i]["centroids"]]
        R_i = set()
        for c in hyper_list[i]["centroids"]:
            if "org_feat" in c:
                R_i.update(c["org_feat"].nonzero()[0])

        for j in range(n):
            seqs_j = [c["act_seq"] for c in hyper_list[j]["centroids"]]
            R_j = set()
            for c in hyper_list[j]["centroids"]:
                if "org_feat" in c:
                    R_j.update(c["org_feat"].nonzero()[0])
            min_ld = 1.0
            for s1 in seqs_i:
                for s2 in seqs_j:
                    ld = Levenshtein.distance("-".join(s1), "-".join(s2))
                    norm_ld = ld / max(len(s1), len(s2))
                    if norm_ld < min_ld:
                        min_ld = norm_ld
            S_cf[i, j] = 1.0 - min_ld

            inter = len(R_i & R_j)
            union = len(R_i | R_j)
            jaccard = inter / union if union != 0 else 0.0
            cos_org = cosine_similarity([X_org[i]], [X_org[j]])[0, 0]
            S_org[i, j] = eta * jaccard + (1 - eta) * cos_org

            try:
                rho, _ = pearsonr(X_per[i], X_per[j])
            except:
                rho = 0.0
            if np.isnan(rho):
                rho = 0.0
            S_per[i, j] = (1.0 + rho) / 2.0

    S_cf = np.clip(S_cf, 0, 1)
    S_org = np.clip(S_org, 0, 1)
    S_per = np.clip(S_per, 0, 1)
    return S_cf, S_org, S_per

def build_knn_adj(sim, k):
    n = sim.shape[0]
    adj = np.zeros_like(sim)
    k = min(k, n-1)
    for i in range(n):
        idx = np.argsort(sim[i])[-k:]
        adj[i, idx] = sim[i, idx]
    adj = adj + np.eye(n)
    return adj

def build_multi_view_graph(ds_name):
    print(f"\n cope with {ds_name} ...")
    trace_dict, centroids_dict = load_data(ds_name)
    hyper_list = build_hyper_instances(trace_dict, centroids_dict)
    X_cf, X_org, X_per = aggregate_features(hyper_list)
    S_cf, S_org, S_per = compute_similarities(hyper_list, X_cf, X_org, X_per)
    A_cf = build_knn_adj(S_cf, KNN_NEIGHBORS)
    A_org = build_knn_adj(S_org, KNN_NEIGHBORS)
    A_per = build_knn_adj(S_per, KNN_NEIGHBORS)
    np.savez(f"product/multi_view_graph_{ds_name}.npz",
             cf_feat=X_cf, org_feat=X_org, per_feat=X_per,
             adj_cf=A_cf, adj_org=A_org, adj_per=A_per)
    print(f"Saved {ds_name} successfully, number of nodes {len(hyper_list)}")

if __name__ == "__main__":
    for ds in DATASETS:
        name = ds["name"]
        if os.path.exists(f"product/centroid_traces_{name}.xes") and os.path.exists(f"product/centroids_result_{name}.json"):
            build_multi_view_graph(name)
        else:
            print(f"escape {name}")