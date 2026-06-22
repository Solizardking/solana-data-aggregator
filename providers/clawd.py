"""Clawd Dataset provider — serves Solana Clawd training dataset statistics."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from metrics.dataset import DatasetMetric, DatasetMetricType
from providers.base import BaseProvider

# Manifests are bundled in datasets/manifests/ relative to the repo root.
_MANIFEST_DIR = Path(__file__).parent.parent / "datasets" / "manifests"

_MANIFEST_FILES = {
    "core_ai": _MANIFEST_DIR / "core_ai_manifest.json",
    "tx_foundation": _MANIFEST_DIR / "tx_foundation_manifest.json",
    "realtime_research": _MANIFEST_DIR / "realtime_research_manifest.json",
    "nvidia_trading_factory": _MANIFEST_DIR / "nvidia_trading_factory_manifest.json",
}


def _load_manifest(key: str) -> dict:
    path = _MANIFEST_FILES[key]
    with path.open() as f:
        return json.load(f)


def _parse_date(iso_str: str) -> str:
    """Return YYYY-MM-DD from an ISO-8601 timestamp string."""
    return iso_str[:10]


class Clawd(BaseProvider):
    """Read Solana Clawd training dataset manifests and expose statistics as metrics.

    No API key is required — data is bundled in datasets/manifests/.
    """

    METRIC_MAP: Dict[str, Any] = {
        # core_ai
        "core_ai_total_examples": {
            "manifest": "core_ai",
            "path": ["stats", "total_examples"],
            "methodology": "Deduplicated merge of solana_clawd_merged.jsonl + core-ai knowledge chunks",
            "methodology_url": "https://huggingface.co/datasets/solanaclawd/solana-clawd-core-ai-instruct",
        },
        "core_ai_train_examples": {"manifest": "core_ai", "path": ["stats", "splits", "train"]},
        "core_ai_eval_examples": {"manifest": "core_ai", "path": ["stats", "splits", "eval"]},
        "core_ai_test_examples": {"manifest": "core_ai", "path": ["stats", "splits", "test"]},
        # tx_foundation
        "tx_foundation_total_examples": {
            "manifest": "tx_foundation",
            "path": ["num_examples"],
            "methodology": "One Solana transaction context per NeMo CPT record",
            "methodology_url": "https://huggingface.co/datasets/solanaclawd/solana-tx-foundation-cpt",
        },
        "tx_foundation_train_examples": {"manifest": "tx_foundation", "path": ["splits", "train"]},
        "tx_foundation_eval_examples": {"manifest": "tx_foundation", "path": ["splits", "eval"]},
        "tx_foundation_test_examples": {"manifest": "tx_foundation", "path": ["splits", "test"]},
        # realtime_research
        "realtime_research_total_examples": {
            "manifest": "realtime_research",
            "path": ["counts", "examples"],
            "methodology": "Chunked PDFs, notebooks, and parquet files from 28 Solana research sources",
            "methodology_url": "https://huggingface.co/datasets/solanaclawd/solana-clawd-realtime-research-instruct",
        },
        "realtime_research_train_examples": {"manifest": "realtime_research", "path": ["splits", "train"]},
        "realtime_research_eval_examples": {"manifest": "realtime_research", "path": ["splits", "eval"]},
        "realtime_research_test_examples": {"manifest": "realtime_research", "path": ["splits", "test"]},
        # nvidia_trading_factory
        "nvidia_trading_factory_total_examples": {
            "manifest": "nvidia_trading_factory",
            "path": ["counts", "examples"],
            "methodology": "cuFOLIO CVaR optimizer, Vulcan strategy configs, and market scenario generation",
            "methodology_url": "https://huggingface.co/datasets/solanaclawd/solana-clawd-nvidia-trading-factory-instruct",
        },
        "nvidia_trading_factory_train_examples": {"manifest": "nvidia_trading_factory", "path": ["splits", "train"]},
        "nvidia_trading_factory_eval_examples": {"manifest": "nvidia_trading_factory", "path": ["splits", "eval"]},
        "nvidia_trading_factory_test_examples": {"manifest": "nvidia_trading_factory", "path": ["splits", "test"]},
        # aggregate
        "clawd_dataset_grand_total": {
            "manifest": None,  # computed from all manifests
            "path": [],
            "methodology": "Sum of core_ai + tx_foundation + realtime_research + nvidia_trading_factory totals",
            "methodology_url": "https://huggingface.co/solanaclawd",
        },
    }

    def __init__(self) -> None:
        super().__init__(
            name="Clawd",
            base_url="https://huggingface.co/solanaclawd",
            api_key="",
        )
        self._manifests: Dict[str, Optional[dict]] = {}

    # -- private helpers ----------------------------------------------------

    def _manifest(self, key: str) -> dict:
        if key not in self._manifests:
            self._manifests[key] = _load_manifest(key)
        return self._manifests[key]  # type: ignore[return-value]

    @staticmethod
    def _dig(data: dict, path: List[str]) -> Any:
        """Traverse nested dict by key path."""
        node: Any = data
        for key in path:
            node = node[key]
        return node

    def _resolve_value(self, metric: str) -> tuple[float, str]:
        """Return (value, iso_date) for the given metric key."""
        cfg = self.METRIC_MAP[metric]

        if metric == "clawd_dataset_grand_total":
            totals = [
                self._dig(self._manifest("core_ai"), ["stats", "total_examples"]),
                self._dig(self._manifest("tx_foundation"), ["num_examples"]),
                self._dig(self._manifest("realtime_research"), ["counts", "examples"]),
                self._dig(self._manifest("nvidia_trading_factory"), ["counts", "examples"]),
            ]
            # Use latest generated_at among manifests
            dates = [
                self._manifest("core_ai").get("created_at", "2026-06-19"),
                self._manifest("tx_foundation").get("generated_at", "2026-06-21"),
                self._manifest("realtime_research").get("generated_at", "2026-06-19"),
                self._manifest("nvidia_trading_factory").get("generated_at", "2026-06-21"),
            ]
            latest_date = max(_parse_date(d) for d in dates)
            return float(sum(totals)), latest_date

        manifest_key = cfg["manifest"]
        data = self._manifest(manifest_key)
        value = self._dig(data, cfg["path"])

        # Prefer generated_at / created_at from manifest
        raw_date = data.get("generated_at") or data.get("created_at", "2026-06-19")
        return float(value), _parse_date(raw_date)

    # -- BaseProvider interface ----------------------------------------------

    def fetch_rows(
        self,
        metric: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Return a single snapshot row per metric; manifests are point-in-time."""
        value, iso_date = self._resolve_value(metric)
        # Only include the row if its manifest date falls within the requested range
        if start_date <= iso_date <= end_date:
            return [{"date": iso_date, "value": value}]
        return []

    def get_metric(
        self,
        metric: str,
        date: str,
        chain: str = "solana",
    ) -> Optional[DatasetMetric]:
        """Return a DatasetMetric for the given metric key, or None if out of range."""
        value, iso_date = self._resolve_value(metric)
        if iso_date != date:
            return None
        metric_type = DatasetMetricType(metric)
        return DatasetMetric.from_metric_type(
            metric_type,
            datetime.date.fromisoformat(iso_date),
            value,
        )
