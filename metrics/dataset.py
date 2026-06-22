"""Dataset metric models for Solana Clawd training dataset statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from numbers import Real

from metrics.base import BaseMetric


class DatasetMetricType(str, Enum):
    """Supported Clawd dataset metric categories."""

    CORE_AI_TOTAL = "core_ai_total_examples"
    CORE_AI_TRAIN = "core_ai_train_examples"
    CORE_AI_EVAL = "core_ai_eval_examples"
    CORE_AI_TEST = "core_ai_test_examples"

    TX_FOUNDATION_TOTAL = "tx_foundation_total_examples"
    TX_FOUNDATION_TRAIN = "tx_foundation_train_examples"
    TX_FOUNDATION_EVAL = "tx_foundation_eval_examples"
    TX_FOUNDATION_TEST = "tx_foundation_test_examples"

    REALTIME_RESEARCH_TOTAL = "realtime_research_total_examples"
    REALTIME_RESEARCH_TRAIN = "realtime_research_train_examples"
    REALTIME_RESEARCH_EVAL = "realtime_research_eval_examples"
    REALTIME_RESEARCH_TEST = "realtime_research_test_examples"

    NVIDIA_TRADING_FACTORY_TOTAL = "nvidia_trading_factory_total_examples"
    NVIDIA_TRADING_FACTORY_TRAIN = "nvidia_trading_factory_train_examples"
    NVIDIA_TRADING_FACTORY_EVAL = "nvidia_trading_factory_eval_examples"
    NVIDIA_TRADING_FACTORY_TEST = "nvidia_trading_factory_test_examples"

    CLAWD_DATASET_GRAND_TOTAL = "clawd_dataset_grand_total"


_METRIC_METADATA: dict[DatasetMetricType, dict[str, str]] = {
    DatasetMetricType.CORE_AI_TOTAL: {
        "name": "Core AI Dataset Total Examples",
        "unit": "Count",
        "description": "Total instruction-tuning examples in the Solana Clawd Core AI dataset",
        "methodology": "Deduplicated merge of solana_clawd_merged.jsonl and core-ai knowledge chunks",
        "methodology_url": "https://huggingface.co/datasets/solanaclawd/solana-clawd-core-ai-instruct",
    },
    DatasetMetricType.CORE_AI_TRAIN: {
        "name": "Core AI Dataset Train Split",
        "unit": "Count",
        "description": "Training-split examples in the Solana Clawd Core AI dataset (90%)",
    },
    DatasetMetricType.CORE_AI_EVAL: {
        "name": "Core AI Dataset Eval Split",
        "unit": "Count",
        "description": "Eval-split examples in the Solana Clawd Core AI dataset (5%)",
    },
    DatasetMetricType.CORE_AI_TEST: {
        "name": "Core AI Dataset Test Split",
        "unit": "Count",
        "description": "Test-split examples in the Solana Clawd Core AI dataset (5%)",
    },
    DatasetMetricType.TX_FOUNDATION_TOTAL: {
        "name": "TX Foundation CPT Total Examples",
        "unit": "Count",
        "description": "Total continual-pretraining records in the Solana Transaction Foundation dataset",
        "methodology": "One Solana transaction context per NeMo CPT record",
        "methodology_url": "https://huggingface.co/datasets/solanaclawd/solana-tx-foundation-cpt",
    },
    DatasetMetricType.TX_FOUNDATION_TRAIN: {
        "name": "TX Foundation CPT Train Split",
        "unit": "Count",
        "description": "Training-split records in the Solana Transaction Foundation CPT dataset (90%)",
    },
    DatasetMetricType.TX_FOUNDATION_EVAL: {
        "name": "TX Foundation CPT Eval Split",
        "unit": "Count",
        "description": "Eval-split records in the Solana Transaction Foundation CPT dataset (5%)",
    },
    DatasetMetricType.TX_FOUNDATION_TEST: {
        "name": "TX Foundation CPT Test Split",
        "unit": "Count",
        "description": "Test-split records in the Solana Transaction Foundation CPT dataset (5%)",
    },
    DatasetMetricType.REALTIME_RESEARCH_TOTAL: {
        "name": "Realtime Research Dataset Total Examples",
        "unit": "Count",
        "description": "Total instruction-tuning examples in the Solana Clawd Realtime Research dataset",
        "methodology": "Chunked PDFs, notebooks, and parquet files from 28 Solana research sources",
        "methodology_url": "https://huggingface.co/datasets/solanaclawd/solana-clawd-realtime-research-instruct",
    },
    DatasetMetricType.REALTIME_RESEARCH_TRAIN: {
        "name": "Realtime Research Dataset Train Split",
        "unit": "Count",
        "description": "Training-split examples in the Realtime Research dataset (90%)",
    },
    DatasetMetricType.REALTIME_RESEARCH_EVAL: {
        "name": "Realtime Research Dataset Eval Split",
        "unit": "Count",
        "description": "Eval-split examples in the Realtime Research dataset (5%)",
    },
    DatasetMetricType.REALTIME_RESEARCH_TEST: {
        "name": "Realtime Research Dataset Test Split",
        "unit": "Count",
        "description": "Test-split examples in the Realtime Research dataset (5%)",
    },
    DatasetMetricType.NVIDIA_TRADING_FACTORY_TOTAL: {
        "name": "NVIDIA Trading Factory Dataset Total Examples",
        "unit": "Count",
        "description": "Total instruction-tuning examples in the Solana Clawd NVIDIA Trading Factory dataset",
        "methodology": "cuFOLIO CVaR optimizer, Vulcan strategy configs, and market scenario generation",
        "methodology_url": "https://huggingface.co/datasets/solanaclawd/solana-clawd-nvidia-trading-factory-instruct",
    },
    DatasetMetricType.NVIDIA_TRADING_FACTORY_TRAIN: {
        "name": "NVIDIA Trading Factory Dataset Train Split",
        "unit": "Count",
        "description": "Training-split examples in the NVIDIA Trading Factory dataset (90%)",
    },
    DatasetMetricType.NVIDIA_TRADING_FACTORY_EVAL: {
        "name": "NVIDIA Trading Factory Dataset Eval Split",
        "unit": "Count",
        "description": "Eval-split examples in the NVIDIA Trading Factory dataset (5%)",
    },
    DatasetMetricType.NVIDIA_TRADING_FACTORY_TEST: {
        "name": "NVIDIA Trading Factory Dataset Test Split",
        "unit": "Count",
        "description": "Test-split examples in the NVIDIA Trading Factory dataset (5%)",
    },
    DatasetMetricType.CLAWD_DATASET_GRAND_TOTAL: {
        "name": "Clawd Dataset Grand Total",
        "unit": "Count",
        "description": "Total examples across all official Solana Clawd training datasets",
        "methodology": "Sum of core_ai + tx_foundation + realtime_research + nvidia_trading_factory totals",
        "methodology_url": "https://huggingface.co/solanaclawd",
    },
}


@dataclass
class DatasetMetric(BaseMetric):
    """Concrete metric model for Clawd training dataset statistics."""

    metric_type: DatasetMetricType

    @classmethod
    def from_metric_type(
        cls,
        metric_type: DatasetMetricType,
        date: date,
        value: Real,
    ) -> "DatasetMetric":
        """Build a dataset metric using canonical metadata."""
        metadata = _METRIC_METADATA[metric_type]
        return cls(
            metric_type=metric_type,
            name=metadata["name"],
            unit=metadata["unit"],
            description=metadata["description"],
            date=date,
            value=float(value),
        )
