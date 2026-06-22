"""Clawd Dataset provider — serves Solana Clawd training dataset statistics."""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from metrics.dataset import DatasetMetric, DatasetMetricType
from providers.base import BaseProvider

_DATASET_DIR = Path(__file__).parent.parent / "datasets"
_MANIFEST_DIR = _DATASET_DIR / "manifests"

_MANIFEST_FILES = {
    "core_ai": _MANIFEST_DIR / "core_ai_manifest.json",
    "tx_foundation": _MANIFEST_DIR / "tx_foundation_manifest.json",
    "realtime_research": _MANIFEST_DIR / "realtime_research_manifest.json",
    "nvidia_trading_factory": _MANIFEST_DIR / "nvidia_trading_factory_manifest.json",
}

_JUPITER_TXNS_FILE = _DATASET_DIR / "jupiter_txs.jsonl"
_SIGNAL_FILE = _DATASET_DIR / "signal_discovery_report.json"

# SOL / USDC token decimal places
_SOL_DECIMALS = 1e9
_USDC_DECIMALS = 1e6


def _load_manifest(key: str) -> dict:
    path = _MANIFEST_FILES[key]
    with path.open() as f:
        return json.load(f)


def _parse_date(iso_str: str) -> str:
    """Return YYYY-MM-DD from an ISO-8601 timestamp string."""
    return iso_str[:10]


def _parse_jupiter_txns() -> Dict[str, Any]:
    """Parse jupiter_txs.jsonl and return aggregated DEX statistics."""
    if not _JUPITER_TXNS_FILE.is_file():
        return {}

    quotes: List[Dict[str, Any]] = []
    with _JUPITER_TXNS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue

            ts_m = re.search(r'\[(\d{4}-\d{2}-\d{2})', text)
            route_m = re.search(r'Route:\s+(\S+)\s+→\s+(\S+)\s+via', text)
            in_m = re.search(r'Input:\s+(\d+)\s+raw units of', text)
            out_m = re.search(r'Output:\s+(\d+)\s+raw units of', text)
            pi_m = re.search(r'Price impact:\s+([\d.]+)%', text)

            if not (ts_m and route_m and in_m and out_m):
                continue

            quotes.append({
                "date": ts_m.group(1),
                "in_tok": route_m.group(1),
                "out_tok": route_m.group(2),
                "in_raw": int(in_m.group(1)),
                "out_raw": int(out_m.group(1)),
                "price_impact_pct": float(pi_m.group(1)) if pi_m else 0.0,
            })

    if not quotes:
        return {}

    # SOL price from SOL→USDC quotes
    sol_usdc = [q for q in quotes if q["in_tok"] == "SOL" and q["out_tok"] == "USDC"]
    sol_price = (
        sum(q["out_raw"] / _USDC_DECIMALS / (q["in_raw"] / _SOL_DECIMALS) for q in sol_usdc)
        / len(sol_usdc)
        if sol_usdc
        else 0.0
    )

    avg_pi_bps = sum(q["price_impact_pct"] * 100 for q in quotes) / len(quotes)
    unique_routes = len({f"{q['in_tok']}→{q['out_tok']}" for q in quotes})
    snapshot_date = max(q["date"] for q in quotes)

    return {
        "date": snapshot_date,
        "sol_price_usd": sol_price,
        "quote_count": float(len(quotes)),
        "avg_price_impact_bps": avg_pi_bps,
        "routes_count": float(unique_routes),
    }


def _parse_signal_report() -> Dict[str, Any]:
    """Parse signal_discovery_report.json and return market signal statistics."""
    if not _SIGNAL_FILE.is_file():
        return {}

    with _SIGNAL_FILE.open() as f:
        data = json.load(f)

    return {
        "date": _parse_date(data.get("timestamp", "2026-06-20")),
        "avg_confidence": float(data.get("avg_confidence", 0.0)),
        "markets_count": float(data.get("n_markets", 0)),
    }


class Clawd(BaseProvider):
    """Read Solana Clawd training dataset manifests and expose statistics as metrics.

    No API key is required — data is bundled in datasets/.
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
            "manifest": None,
            "path": [],
            "methodology": "Sum of core_ai + tx_foundation + realtime_research + nvidia_trading_factory totals",
            "methodology_url": "https://huggingface.co/solanaclawd",
        },
        # Jupiter DEX
        "jupiter_sol_price_usd": {
            "source": "jupiter",
            "key": "sol_price_usd",
            "methodology": "Mean of (USDC output / SOL input) across all SOL→USDC Jupiter quotes; SOL=9 decimals, USDC=6 decimals",
            "methodology_url": "https://station.jup.ag/docs/apis/swap-api",
        },
        "jupiter_quote_count": {"source": "jupiter", "key": "quote_count"},
        "jupiter_avg_price_impact_bps": {"source": "jupiter", "key": "avg_price_impact_bps"},
        "jupiter_routes_count": {"source": "jupiter", "key": "routes_count"},
        # Market signals
        "signal_avg_confidence": {
            "source": "signal",
            "key": "avg_confidence",
            "methodology": "Mean confidence from signal_discovery_report.json generated by the Clawd autoresearch pipeline",
        },
        "signal_markets_count": {"source": "signal", "key": "markets_count"},
    }

    def __init__(self) -> None:
        super().__init__(
            name="Clawd",
            base_url="https://huggingface.co/solanaclawd",
            api_key="",
        )
        self._manifests: Dict[str, Optional[dict]] = {}
        self._jupiter: Optional[Dict[str, Any]] = None
        self._signal: Optional[Dict[str, Any]] = None

    # -- private helpers ----------------------------------------------------

    def _manifest(self, key: str) -> dict:
        if key not in self._manifests:
            self._manifests[key] = _load_manifest(key)
        return self._manifests[key]  # type: ignore[return-value]

    def _jupiter_data(self) -> Dict[str, Any]:
        if self._jupiter is None:
            self._jupiter = _parse_jupiter_txns()
        return self._jupiter

    def _signal_data(self) -> Dict[str, Any]:
        if self._signal is None:
            self._signal = _parse_signal_report()
        return self._signal

    @staticmethod
    def _dig(data: dict, path: List[str]) -> Any:
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
            dates = [
                self._manifest("core_ai").get("created_at", "2026-06-19"),
                self._manifest("tx_foundation").get("generated_at", "2026-06-21"),
                self._manifest("realtime_research").get("generated_at", "2026-06-19"),
                self._manifest("nvidia_trading_factory").get("generated_at", "2026-06-21"),
            ]
            return float(sum(totals)), max(_parse_date(d) for d in dates)

        source = cfg.get("source")

        if source == "jupiter":
            data = self._jupiter_data()
            return float(data[cfg["key"]]), data["date"]

        if source == "signal":
            data = self._signal_data()
            return float(data[cfg["key"]]), data["date"]

        # Default: manifest path
        manifest_data = self._manifest(cfg["manifest"])
        value = self._dig(manifest_data, cfg["path"])
        raw_date = manifest_data.get("generated_at") or manifest_data.get("created_at", "2026-06-19")
        return float(value), _parse_date(raw_date)

    # -- BaseProvider interface ----------------------------------------------

    def fetch_rows(
        self,
        metric: str,
        start_date: str,
        end_date: str,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Return a single snapshot row per metric; all data is point-in-time."""
        value, iso_date = self._resolve_value(metric)
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
