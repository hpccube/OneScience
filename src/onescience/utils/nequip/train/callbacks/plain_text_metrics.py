"""Plain-text metric reporting for non-interactive training jobs."""

from __future__ import annotations

import math

import torch
from lightning.pytorch.callbacks import Callback


def _format_metric(value) -> str | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, (int, float)):
        value = float(value)
        if not math.isfinite(value):
            return str(value)
        return f"{value:.8g}"
    return None


class PlainTextMetricsLogger(Callback):
    """Print one flush-safe metric line per epoch and after testing."""

    @staticmethod
    def _metric_text(trainer) -> str:
        entries = []
        for name, value in sorted(trainer.callback_metrics.items()):
            if "_epoch/" not in name:
                continue
            formatted = _format_metric(value)
            if formatted is not None:
                entries.append(f"{name}={formatted}")
        return " ".join(entries)

    def on_validation_end(self, trainer, pl_module) -> None:
        del pl_module
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        metrics = self._metric_text(trainer)
        print(
            f"[progress] epoch={trainer.current_epoch + 1}/{trainer.max_epochs} "
            f"global_step={trainer.global_step} {metrics}".rstrip(),
            flush=True,
        )

    def on_test_end(self, trainer, pl_module) -> None:
        del pl_module
        if not trainer.is_global_zero:
            return
        metrics = self._metric_text(trainer)
        print(f"[test] {metrics}".rstrip(), flush=True)
