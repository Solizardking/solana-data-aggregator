"""Unit tests for the Clawd dataset provider."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from metrics.dataset import DatasetMetric, DatasetMetricType
from providers.clawd import Clawd, _MANIFEST_DIR, _MANIFEST_FILES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORE_AI_MANIFEST = {
    "created_at": "2026-06-19T15:14:18.067073+00:00",
    "stats": {
        "total_examples": 35173,
        "splits": {"train": 31655, "eval": 1759, "test": 1759},
    },
}

TX_FOUNDATION_MANIFEST = {
    "generated_at": "2026-06-21T18:10:26+00:00",
    "num_examples": 19542,
    "splits": {"train": 17587, "eval": 977, "test": 978},
}

REALTIME_RESEARCH_MANIFEST = {
    "generated_at": "2026-06-19T18:08:46+00:00",
    "counts": {"examples": 29058},
    "splits": {"train": 26152, "eval": 1452, "test": 1454},
}

NVIDIA_TRADING_FACTORY_MANIFEST = {
    "generated_at": "2026-06-21T17:09:08+00:00",
    "counts": {"examples": 195},
    "splits": {"train": 175, "eval": 10, "test": 10},
}

_ALL_MANIFESTS = {
    "core_ai": CORE_AI_MANIFEST,
    "tx_foundation": TX_FOUNDATION_MANIFEST,
    "realtime_research": REALTIME_RESEARCH_MANIFEST,
    "nvidia_trading_factory": NVIDIA_TRADING_FACTORY_MANIFEST,
}


@pytest.fixture()
def provider(tmp_path: Path) -> Clawd:
    """Return a Clawd provider with manifests pre-loaded (bypasses filesystem)."""
    p = Clawd()
    p._manifests = dict(_ALL_MANIFESTS)
    return p


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestClawdInit:
    def test_name(self, provider: Clawd) -> None:
        assert provider.name == "Clawd"

    def test_no_api_key_required(self) -> None:
        """Clawd provider instantiates without any API key."""
        p = Clawd()
        assert p.api_key == ""

    def test_metric_map_not_empty(self, provider: Clawd) -> None:
        assert len(provider.METRIC_MAP) > 0


# ---------------------------------------------------------------------------
# fetch_rows
# ---------------------------------------------------------------------------


class TestFetchRows:
    def test_core_ai_total_in_range(self, provider: Clawd) -> None:
        rows = provider.fetch_rows("core_ai_total_examples", "2026-06-01", "2026-06-30")
        assert len(rows) == 1
        assert rows[0]["value"] == 35173.0
        assert rows[0]["date"] == "2026-06-19"

    def test_out_of_range_returns_empty(self, provider: Clawd) -> None:
        rows = provider.fetch_rows("core_ai_total_examples", "2025-01-01", "2025-12-31")
        assert rows == []

    def test_tx_foundation_total(self, provider: Clawd) -> None:
        rows = provider.fetch_rows("tx_foundation_total_examples", "2026-06-01", "2026-06-30")
        assert rows[0]["value"] == 19542.0

    def test_tx_foundation_train_split(self, provider: Clawd) -> None:
        rows = provider.fetch_rows("tx_foundation_train_examples", "2026-06-01", "2026-06-30")
        assert rows[0]["value"] == 17587.0

    def test_realtime_research_total(self, provider: Clawd) -> None:
        rows = provider.fetch_rows("realtime_research_total_examples", "2026-06-01", "2026-06-30")
        assert rows[0]["value"] == 29058.0

    def test_nvidia_trading_factory_total(self, provider: Clawd) -> None:
        rows = provider.fetch_rows("nvidia_trading_factory_total_examples", "2026-06-01", "2026-06-30")
        assert rows[0]["value"] == 195.0

    def test_grand_total(self, provider: Clawd) -> None:
        rows = provider.fetch_rows("clawd_dataset_grand_total", "2026-06-01", "2026-06-30")
        assert len(rows) == 1
        expected = 35173 + 19542 + 29058 + 195
        assert rows[0]["value"] == float(expected)

    def test_eval_split(self, provider: Clawd) -> None:
        rows = provider.fetch_rows("core_ai_eval_examples", "2026-06-01", "2026-06-30")
        assert rows[0]["value"] == 1759.0

    def test_test_split(self, provider: Clawd) -> None:
        rows = provider.fetch_rows("core_ai_test_examples", "2026-06-01", "2026-06-30")
        assert rows[0]["value"] == 1759.0


# ---------------------------------------------------------------------------
# get_metric
# ---------------------------------------------------------------------------


class TestGetMetric:
    def test_returns_dataset_metric(self, provider: Clawd) -> None:
        metric = provider.get_metric("core_ai_total_examples", "2026-06-19")
        assert isinstance(metric, DatasetMetric)
        assert metric.metric_type == DatasetMetricType.CORE_AI_TOTAL
        assert metric.value == 35173.0
        assert metric.unit == "Count"

    def test_wrong_date_returns_none(self, provider: Clawd) -> None:
        metric = provider.get_metric("core_ai_total_examples", "2025-01-01")
        assert metric is None

    def test_grand_total_metric_object(self, provider: Clawd) -> None:
        metric = provider.get_metric("clawd_dataset_grand_total", "2026-06-21")
        assert isinstance(metric, DatasetMetric)
        assert metric.metric_type == DatasetMetricType.CLAWD_DATASET_GRAND_TOTAL
        expected = float(35173 + 19542 + 29058 + 195)
        assert metric.value == expected


# ---------------------------------------------------------------------------
# Manifest files on disk
# ---------------------------------------------------------------------------


class TestManifestFiles:
    def test_manifest_dir_exists(self) -> None:
        assert _MANIFEST_DIR.is_dir(), f"{_MANIFEST_DIR} not found — run integration setup"

    @pytest.mark.parametrize("key", ["core_ai", "tx_foundation", "realtime_research", "nvidia_trading_factory"])
    def test_manifest_file_exists(self, key: str) -> None:
        path = _MANIFEST_FILES[key]
        assert path.is_file(), f"Missing manifest: {path}"

    @pytest.mark.parametrize("key", ["core_ai", "tx_foundation", "realtime_research", "nvidia_trading_factory"])
    def test_manifest_is_valid_json(self, key: str) -> None:
        path = _MANIFEST_FILES[key]
        data = json.loads(path.read_text())
        assert isinstance(data, dict)
