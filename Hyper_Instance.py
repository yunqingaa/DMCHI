import numpy as np
import json
import Levenshtein
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.obj import EventLog
import os

os.makedirs("product", exist_ok=True)

def generate_fine_grained_clusters(dataset_name,OVER_CLUSTER_NUM, MIN_SAMPLES, MAX_AVG_DISTANCE):
    os.makedirs("../result", exist_ok=True)
    FEATURE_PATH = f"product/multi_view_features_{dataset_name}.npz"
    data = np.load(FEATURE_PATH)
    F = data['reduced']
    real_trace_ids = [str(tid) for tid in data['real_ids']]
    trace_frequencies = [int(freq) for freq in data['frequencies']]

    print(f"✅ Feature loading complete | Total number of trajectories: {F.shape[0]}")
    kmeans = KMeans(n_clusters=OVER_CLUSTER_NUM, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(F)
    centroids = kmeans.cluster_centers_

    clusters_variants = {}
    new_cluster_id = 0
    for cluster_id in range(OVER_CLUSTER_NUM):
        sample_indices = np.where(cluster_labels == cluster_id)[0]
        sample_size = len(sample_indices)
        if sample_size < MIN_SAMPLES:
            continue
        cluster_features = F[sample_indices]
        distances = pairwise_distances(cluster_features, [centroids[cluster_id]]).flatten()
        avg_dist = np.mean(distances)
        if avg_dist > MAX_AVG_DISTANCE:
            continue
        cluster_result = [{real_trace_ids[idx]: trace_frequencies[idx]} for idx in sample_indices]
        clusters_variants[str(new_cluster_id)] = cluster_result
        new_cluster_id += 1
    CLUSTER_TEMP_PATH = f"product/clusters_variants_{dataset_name}.json"
    with open(CLUSTER_TEMP_PATH, 'w', encoding='utf-8') as f:
        json.dump(clusters_variants, f, ensure_ascii=False, indent=2)
    print("=" * 60)
    print(f"🎯 【{dataset_name}】Clustering complete | Number of valid subclusters: {len(clusters_variants)}")
    print("=" * 60)
    return clusters_variants, CLUSTER_TEMP_PATH

def load_trace_mapping(dataset_name):
    FEATURE_PATH = f"product/multi_view_features_{dataset_name}.npz"
    data = np.load(FEATURE_PATH, allow_pickle=True)
    real_ids = [str(tid) for tid in data['real_ids'].tolist()]
    activities = data['activities'].tolist()
    frequencies = [int(f) for f in data['frequencies'].tolist()]
    id2seq = {tid: seq for tid, seq in zip(real_ids, activities)}
    id2freq = {tid: freq for tid, freq in zip(real_ids, frequencies)}
    return id2seq, id2freq

def calculate_edit_distance(seq1, seq2):
    return Levenshtein.distance(seq1, seq2)

def sort_centroids_by_frequency(centro_by_cluster):
    centroid_sums = {label: sum(freq for _, freq in centroids) for label, centroids in centro_by_cluster.items()}
    sorted_labels = sorted(centroid_sums.items(), key=lambda x: x[1], reverse=True)
    sorted_result = {label: sorted(centro_by_cluster[label], key=lambda x: x[1], reverse=True) for label, _ in
                     sorted_labels}
    return sorted_result

def select_multi_centroids(data, k, threshold, id2seq, id2freq):
    centroids = {}

    for cluster_id, traces_list in data.items():
        all_traces = [(list(t.keys())[0], list(t.values())[0]) for t in traces_list]

        unique_seq = {}
        for tid, freq in all_traces:
            seq = id2seq[tid]
            if seq not in unique_seq:
                unique_seq[seq] = (tid, freq)
        unique_traces = list(unique_seq.values())

        unique_traces.sort(key=lambda x: x[1], reverse=True)

        selected = []
        remaining = unique_traces.copy()

        while len(selected) < k and remaining:
            base = remaining.pop(0)
            base_seq = id2seq[base[0]]
            selected.append(base)

            temp_remaining = []
            for trace in remaining:
                trace_seq = id2seq[trace[0]]
                dist = calculate_edit_distance(base_seq, trace_seq)

                if dist <= threshold and len(selected) < k:
                    selected.append(trace)
                else:
                    temp_remaining.append(trace)

            remaining = temp_remaining

        centroids[cluster_id] = selected[:k]

    return sort_centroids_by_frequency(centroids)

def filter_and_save_log(centroid_data, dataset_name, raw_log_path):
    centroid_ids = set()
    for cluster in centroid_data.values():
        for tid, _ in cluster:
            centroid_ids.add(str(tid).strip())
    print(f"\n✅ 【{dataset_name}】Total number of centroid trajectories to be screened：{len(centroid_ids)}")
    if len(centroid_ids) == 0:
        print("No centroid ID found, exiting export")
        return None

    EXPORT_XES_PATH = f"product/centroid_traces_{dataset_name}.xes"
    log = xes_importer.apply(raw_log_path)
    filtered_traces = [trace for trace in log if str(trace.attributes.get("case:concept:name",
                                                                          trace.attributes.get("concept:name",
                                                                                               ""))).strip() in centroid_ids]
    filtered_log = EventLog(filtered_traces)
    xes_exporter.apply(filtered_log, EXPORT_XES_PATH)
    print(f"✅ 【{dataset_name}】Number of tracks successfully exported：{len(filtered_log)}")
    print(f"The centre of mass trajectory has been saved：{EXPORT_XES_PATH}")
    return EXPORT_XES_PATH

def run_centroid_pipeline(
        dataset_name,
        RAW_XES_PATH,
        OVER_CLUSTER_NUM=50,
        MIN_SAMPLES=3,
        MAX_AVG_DISTANCE=1.5,
        K=3,
        EDIT_DISTANCE_THRESHOLD=2
):

    print("\n" + "=" * 80)
    print(f"🚀 【{dataset_name}】Operating parameters：Ko={OVER_CLUSTER_NUM}, p={K}, edit_thresh={EDIT_DISTANCE_THRESHOLD}")
    print("=" * 80)

    cluster_result, _ = generate_fine_grained_clusters(dataset_name, OVER_CLUSTER_NUM, MIN_SAMPLES, MAX_AVG_DISTANCE)
    id2seq, id2freq = load_trace_mapping(dataset_name)
    final_centroids = select_multi_centroids(cluster_result, K, EDIT_DISTANCE_THRESHOLD, id2seq, id2freq)
    CENTROID_JSON_PATH = f"product/centroids_result_{dataset_name}.json"
    with open(CENTROID_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_centroids, f, ensure_ascii=False, indent=2)

    export_path = filter_and_save_log(final_centroids, dataset_name, RAW_XES_PATH)

    return export_path

if __name__ == "__main__":
    DATASETS = [
        {"name": "Dataset1", "path": "../data/BPI_Challenge_2012.xes"},
        {"name": "Dataset2", "path": "../data/BPI_Challenge_2013.xes"},
        {"name": "Dataset3", "path": "../data/BPI_Challenge_2017.xes"},
        {"name": "Dataset4", "path": "../data/BPI_Challenge_2019.xes"},
        {"name": "Dataset5", "path": "../data/Hospital Billing.xes"},
        {"name": "Dataset6", "path": "../data/PermitLog.xes"},
    ]

    for ds in DATASETS:
        ds_name = ds["name"]
        ds_path = ds["path"]
        if os.path.exists(ds_path):
            run_centroid_pipeline(
                dataset_name=ds_name,
                RAW_XES_PATH=ds_path,
                OVER_CLUSTER_NUM=50,
                K=3
            )
        else:
            print(f"【{ds_name}】The original log file does not exist:{ds_path}")

    print("\nAll dataset hyper-instances have been generated!")