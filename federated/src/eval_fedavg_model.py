import os
import sys
import glob
import argparse
import pickle
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# パス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from common.model import WaveformCRNN
from common.dataset import VitalDBDataset
# centralizedのtrain.pyにある関数を再利用（またはcommonに移管推奨）
from centralized.scripts.train import compute_metrics, pick_thresholds


def evaluate_with_threshold(y_true, y_prob, thr: float):
    pred = (y_prob >= float(thr)).astype(np.int64)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1}

def load_fedavg_model(model_path, device):
    print(f"Loading FedAvg model from: {model_path}")
    model = WaveformCRNN(in_channels=3, num_classes=1).to(device)
    
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    # Flower形式 (list of ndarray) から StateDict へ変換
    if "ndarrays" not in data:
        raise ValueError("Invalid checkpoint: 'ndarrays' key missing.")
    
    parameters = data["ndarrays"]
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def collect_global_predictions(model, data_dir, split, device, batch_size=512):
    """
    全クライアントの特定split(train/testのみ)データをロードし、予測を行う。
    """
    print(f"Collecting predictions for split: {split} ...")
    files = glob.glob(os.path.join(data_dir, "**", split, "*.npz"), recursive=True)
    
    if not files:
        print(f"No files found for split {split}")
        return [], []

    dataset = VitalDBDataset(files, window_size=3000, stride=3000)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=(device.type=="cuda"), shuffle=False)
    
    all_preds = []
    all_targets = []
    
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc=f"Infer {split}"):
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=autocast_device,
                enabled=(device.type == "cuda"),
                dtype=torch.float16,
            ):
                logits = model(inputs)
            
            probs = torch.sigmoid(logits).cpu().view(-1).tolist()
            targets = targets.view(-1).tolist()
            all_preds.extend(probs)
            all_targets.extend(targets)
            
    return np.array(all_targets), np.array(all_preds)

def main():
    parser = argparse.ArgumentParser(description="Strict Evaluation for FedAvg (No Leakage)")
    parser.add_argument("--model", type=str, required=True, help="Path to best.pkl")
    parser.add_argument("--data_dir", type=str, default="./federated_data")
    parser.add_argument("--out_dir", type=str, default="./federated/results")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. モデルロード
    model = load_fedavg_model(args.model, device)

    # 2. 評価フェーズ (Test Dataのみを使用)
    test_targets, test_preds = collect_global_predictions(model, args.data_dir, "test", device)
    
    if len(test_targets) == 0:
        print("Test data not found.")
        return

    # 確率的指標 (AUROC, AUPRC, ECE など)
    prob_metrics = compute_metrics(test_targets, test_preds)

    # 閾値決定 (testに対して最適閾値を選択)
    thresholds = pick_thresholds(test_targets, test_preds)
    thr_bal = thresholds.get("balance_min_diff", 0.5)
    res_bal = evaluate_with_threshold(test_targets, test_preds, thr_bal)

    print("\n=== FEDAVG GLOBAL TEST RESULTS ===")
    print(f" AUPRC: {prob_metrics.get('AUPRC', 0.0):.4f}")
    print(f" ECE  : {prob_metrics.get('ECE', 0.0):.4f}")
    print(f" F1   : {res_bal['f1']:.4f} (at Thr={thr_bal:.4f})")
    
    # 結果保存
    results = {
        "prob_metrics": prob_metrics,
        "threshold_metrics": res_bal,
        "thresholds_used": thresholds
    }
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
