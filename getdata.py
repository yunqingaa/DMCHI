from pm4py.objects.log.importer.xes import importer as xes_importer
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from collections import Counter, defaultdict
from scipy.stats import entropy
import os

def extract_ngram_features(sequence, n_list=[1, 2, 3]):
    ngram_all = []
    for n in n_list:
        if len(sequence) >= n:
            grams = [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]
            ngram_all.extend(["-".join(g) for g in grams])
    return Counter(ngram_all)

def compute_organization_features(resource_seq, group_seq=None):
    unique_resources = len(set(resource_seq))
    res_count = Counter(resource_seq)
    res_prob = np.array(list(res_count.values())) / sum(res_count.values()) if sum(res_count.values()) > 0 else np.array([0])
    resource_concentration = entropy(res_prob) if len(res_count.values()) > 0 else 0
    unique_groups = len(set(group_seq)) if group_seq is not None else 0
    group_weight = max(Counter(group_seq).values()) / len(group_seq) if (group_seq and len(group_seq) > 0) else 0
    switch_count = 0
    if group_seq and len(group_seq) > 1:
        for i in range(1, len(group_seq)):
            if group_seq[i] != group_seq[i - 1]:
                switch_count += 1
    cross_department = switch_count / (len(group_seq) - 1) if (group_seq and len(group_seq) > 1) else 0
    return [unique_resources, resource_concentration, unique_groups, group_weight, cross_department]

def compute_performance_features(timestamps):
    if len(timestamps) < 2:
        return [0, 0, 0, 0, 0]
    ts = [t.timestamp() for t in timestamps]
    durations = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    total_time = ts[-1] - ts[0]
    mean_dur = np.mean(durations) if len(durations) > 0 else 0
    max_dur = np.max(durations) if len(durations) > 0 else 0
    std_dur = np.std(durations) if len(durations) > 0 else 0
    cv = std_dur / mean_dur if mean_dur != 0 else 0
    wait_time = mean_dur
    return [total_time, mean_dur, max_dur, cv, wait_time]

def get_feature(dataset,log, original_trace_ids, dataset_name):
    trace_activities = []
    trace_resources = []
    trace_groups = []
    trace_timestamps = []
    valid_trace_ids = []
    for idx, trace in enumerate(log):
        acts = [e["concept:name"] for e in trace if "concept:name" in e]
        res = [e.get("org:resource", "Unknown") for e in trace]
        grp = [e.get("org:group", "Unknown") for e in trace]
        ts = [e.get("time:timestamp", None) for e in trace if e.get("time:timestamp") is not None]

        if len(acts) > 0 and len(ts) > 1:
            trace_activities.append(acts)
            trace_resources.append(res)
            trace_groups.append(grp)
            trace_timestamps.append(ts)
            valid_trace_ids.append(original_trace_ids[idx])

    num_traces = len(trace_activities)
    print(f"\nvalid trace num:{num_traces}")

    trace_activities_str = ["-".join(act_list) for act_list in trace_activities]
    seq_frequency = Counter(trace_activities_str)
    valid_freq = [seq_frequency[seq] for seq in trace_activities_str]

    ngram_counter = defaultdict(int)
    for acts in trace_activities:
        for gram in extract_ngram_features(acts):
            ngram_counter[gram] += 1
    vocab = [g for g, c in ngram_counter.items() if c >= 2]
    cf_features = []
    for acts in trace_activities:
        gram_cnt = extract_ngram_features(acts)
        feat = [gram_cnt.get(g, 0) for g in vocab]
        norm_feat = np.array(feat) / sum(feat) if sum(feat) != 0 else np.zeros(len(feat))
        cf_features.append(norm_feat)
    cf_matrix = np.array(cf_features)

    org_features = []
    for res, grp in zip(trace_resources, trace_groups):
        feat = compute_organization_features(res, grp)
        org_features.append(feat)
    org_matrix = MinMaxScaler().fit_transform(np.array(org_features))

    perf_features = []
    for ts in trace_timestamps:
        feat = compute_performance_features(ts)
        perf_features.append(feat)
    perf_matrix = MinMaxScaler().fit_transform(np.array(perf_features))

    fused_matrix = np.hstack([cf_matrix, org_matrix, perf_matrix])
    svd = TruncatedSVD(n_components=min(50, fused_matrix.shape[1] - 1), random_state=42)
    F = svd.fit_transform(fused_matrix)
    save_path = f'./product/multi_view_features_{dataset}.npz'
    os.makedirs('./product', exist_ok=True)
    np.savez(save_path,
             reduced=F,
             real_ids=valid_trace_ids,
             activities=trace_activities_str,
             frequencies=valid_freq)
    print("\nfeature saved")
    return F, valid_trace_ids

def readfile(dataset,filename):
    dataset_name = os.path.splitext(os.path.basename(filename))[0]

    variant = xes_importer.Variants.ITERPARSE
    log = xes_importer.apply(filename, variant=variant)
    print(f"\n【{dataset_name}】trace num: {len(log)}")

    original_trace_ids = []
    for idx, trace in enumerate(log):
        trace_id = trace.attributes.get("concept:name", f"unknown_{idx}")
        trace_id = str(trace_id)
        original_trace_ids.append(trace_id)

    return get_feature(dataset,log, original_trace_ids, dataset_name)

if __name__ == '__main__':
    DATASETS = [
        {"name": "Dataset1", "path": "../data/BPI_Challenge_2012.xes"},
        {"name": "Dataset2", "path": "../data/BPI_Challenge_2013.xes"},
        {"name": "Dataset3", "path": "../data/BPI_Challenge_2017.xes"},
        {"name": "Dataset4", "path": "../data/BPI_Challenge_2019.xes"},
        {"name": "Dataset5", "path": "../data/Hospital Billing.xes"},
        {"name": "Dataset6", "path": "../data/PermitLog.xes"},
    ]
    print("=" * 60)
    for ds in DATASETS:
        ds_name = ds["name"]
        ds_path = ds["path"]
        if os.path.exists(ds_path):
            readfile(ds_name,ds_path)
        else:
            print(f"not exist：{ds_path}")

    print("\n🎉 complete-features")