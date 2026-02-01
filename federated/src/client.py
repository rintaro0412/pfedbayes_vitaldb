import os
import sys
from collections import OrderedDict
from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import flwr as fl

# パス解決
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# 修正: losses.py から FocalLoss をインポート
try:
    from losses import FocalLoss
except ImportError:
    from common.losses import FocalLoss

from common.dataset import VitalDBDataset
from common.model import WaveformCRNN
from common.utils import calc_comprehensive_metrics

class VitalDBClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: Union[int, str],
        train_dataset: Dataset,
        device,
        eval_dataset: Dataset | None = None,
        batch_size: int = 64,
        epochs: int = 1,
        dataloader_workers: int = 0,
        pin_memory: bool | None = None,
        pos_weight: float | None = None,
        # Focal Loss採用のため auto_pos_weight は使用しない方針とするが互換性のため残す
        auto_pos_weight: bool = False, 
    ):
        self.cid = cid
        self.train_ds = train_dataset
        self.eval_ds = eval_dataset
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataloader_workers = dataloader_workers
        self.pin_memory = pin_memory if pin_memory is not None else (device.type == "cuda")
        self.pos_weight = pos_weight

    def _create_loader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.dataloader_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.dataloader_workers > 0)
        )

    def _build_model(self):
        # 修正: 集中学習に合わせて in_channels=3 (ABP, ECG, PPG), Attention付きCRNN
        model = WaveformCRNN(in_channels=3, num_classes=1, lstm_hidden=128)
        return model.to(self.device)

    def get_parameters(self, config):
        model = self._build_model()
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters, model):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        model = self._build_model()
        self.set_parameters(parameters, model)
        model.train()

        # 修正: Focal Loss の採用 (集中学習と統一)
        criterion = FocalLoss(alpha=0.25, gamma=2.0).to(self.device)
        
        # Optimizer (集中学習と同じ設定: lr=1e-3, weight_decay=1e-4)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        train_loader = self._create_loader(self.train_ds, shuffle=True)
        
        epoch_loss = 0.0
        # AMP (Automatic Mixed Precision) の適用
        scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))
        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"

        for _ in range(self.epochs):
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float().view(-1, 1)

                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast(
                    device_type=autocast_device,
                    enabled=(self.device.type == "cuda"),
                    dtype=torch.float16,
                ):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

        # 返却値: 更新されたパラメータ, データ数, ログ用メトリクス
        updated_params = [val.cpu().numpy() for _, val in model.state_dict().items()]
        
        return updated_params, len(self.train_ds), {"loss": epoch_loss / len(train_loader)}

    def evaluate(self, parameters, config):
        # Server側からの要求でローカル評価を行うメソッド
        # train/testのみの設計に合わせ、テストデータ(eval_ds)で評価する
        target_ds = self.eval_ds
        if target_ds is None or len(target_ds) == 0:
            return 0.0, 0, {}

        model = self._build_model()
        self.set_parameters(parameters, model)
        model.eval()

        criterion = FocalLoss(alpha=0.25, gamma=2.0).to(self.device)
        val_loader = self._create_loader(target_ds, shuffle=False)
        autocast_device = "cuda" if self.device.type == "cuda" else "cpu"

        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True).float().view(-1, 1)

                with torch.amp.autocast(
                    device_type=autocast_device,
                    enabled=(self.device.type == "cuda"),
                    dtype=torch.float16,
                ):
                    logits = model(inputs)
                    loss = criterion(logits, targets)

                total_loss += loss.item() * targets.size(0)
                probs = torch.sigmoid(logits).detach().cpu().view(-1).tolist()
                all_preds.extend(probs)
                all_targets.extend(targets.detach().cpu().view(-1).tolist())

        avg_loss = total_loss / len(target_ds)
        
        # 修正: Accuracyではなく AUPRC を返す (Server側の集計用)
        # ここでは簡易的な計算のみ行い、厳密な評価は eval_fedavg_model.py で行う
        from sklearn.metrics import average_precision_score
        try:
            auprc = average_precision_score(all_targets, all_preds)
        except:
            auprc = 0.0

        return float(avg_loss), len(target_ds), {"AUPRC": float(auprc), "Loss": float(avg_loss)}
