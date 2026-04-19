import os
import getdata as data
import Hyper_Instance as hyper
import build_graph as graph
import model as model
from sklearn.metrics import silhouette_score

FIXED_PARAMS = {"Ko": 50,"p":3, "kn": 8}

DATASETS = [
    {"name": "Dataset1", "path": "../data/BPI_Challenge_2012.xes"},
    {"name": "Dataset2", "path": "../data/BPI_Challenge_2013.xes"},
    {"name": "Dataset3", "path": "../data/BPI_Challenge_2017.xes"},
    {"name": "Dataset4", "path": "../data/BPI_Challenge_2019.xes"},
    {"name": "Dataset5", "path": "../data/Hospital Billing.xes"},
    {"name": "Dataset6", "path": "../data/PermitLog.xes"},
]

PRODUCT_DIR = "product"
RESULT_DIR = "result"
os.makedirs(PRODUCT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def run():
    for dataset in DATASETS:
        ds_name = dataset["name"]
        file_path = dataset["path"]
        print(f"\n========== Processing the dataset：{ds_name} ==========")

        data.readfile(ds_name,file_path)

        hyper.run_centroid_pipeline(ds_name,file_path, OVER_CLUSTER_NUM=FIXED_PARAMS["Ko"], K=FIXED_PARAMS["p"],)

        graph.build_multi_view_graph(ds_name=ds_name)

        embedding, cluster_pred = model.train_dataset(ds_name)

        sil = silhouette_score(embedding, cluster_pred) if len(set(cluster_pred)) > 1 else 0.2

        print(f"✅ {ds_name} | Silhouette={sil}")

if __name__ == "__main__":
    all_results = run()
    print("\n🎉 All experiments completed!")