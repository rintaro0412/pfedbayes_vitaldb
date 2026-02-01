import os
import pickle
from typing import Optional, Tuple, Callable, Any
import flwr as fl
from flwr.common import parameters_to_ndarrays, Parameters
from flwr.common import Metrics

def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    keys = metrics[0][1].keys()
    aggregated: Metrics = {}
    total_examples = sum(num for num, _ in metrics)
    for key in keys:
        s = sum(num * m[key] for num, m in metrics)
        aggregated[key] = s / total_examples
    return aggregated

class BestAUPRCStrategy(fl.server.strategy.FedAvg):
    """
    検証メトリクス(AUPRCなど)が最大となるラウンドのグローバル重みを保存する。
    """
    def __init__(
        self,
        *,
        checkpoint_dir: Optional[str],
        best_metric_key: str = "AUPRC",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_dir = checkpoint_dir
        self.best_metric_key = best_metric_key
        self.best_metric_value = float("-inf")
        self.best_round: Optional[int] = None
        self.best_checkpoint_path: Optional[str] = None
        self._latest_parameters: Optional[Parameters] = None

        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> tuple[Optional[Parameters], Optional[Metrics]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is not None:
            self._latest_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: list[BaseException],
    ) -> tuple[Optional[float], dict[str, Any]]:

        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        if (
            metrics is not None
            and self.best_metric_key in metrics
            and self._latest_parameters is not None
        ):
            current_val = metrics[self.best_metric_key]
            if current_val > self.best_metric_value:
                self.best_metric_value = current_val
                self.best_round = server_round
                self._save_checkpoint(server_round, current_val)

        return loss, metrics

    def _save_checkpoint(self, server_round: int, metric_val: float) -> None:
        if not self.checkpoint_dir or self._latest_parameters is None:
            return

        ndarrays = parameters_to_ndarrays(self._latest_parameters)
        payload = {
            "round": server_round,
            "metric_key": self.best_metric_key,
            "metric_value": float(metric_val),
            "ndarrays": ndarrays,
        }
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "best.pkl")
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        self.best_checkpoint_path = path
        print(f"[BestAUPRCStrategy] Updated best {self.best_metric_key}: {metric_val:.4f} at round {server_round}. Saved to {path}")

# 互換用エイリアス
BestAUCStrategy = BestAUPRCStrategy


def get_strategy(
    on_fit_config_fn=None,
    checkpoint_dir: Optional[str] = None,
    best_metric_key: str = "AUPRC",
    round_logger=None,  # optional, ignored for now (kept for compatibility)
) -> fl.server.strategy.FedAvg:
    return BestAUPRCStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config_fn,
        checkpoint_dir=checkpoint_dir,
        best_metric_key=best_metric_key,
    )
