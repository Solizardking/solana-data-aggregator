"""FastAPI web server — run providers on demand and export datasets via browser."""

from __future__ import annotations

import csv
import datetime
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from providers.allium import Allium
from providers.artemis import Artemis
from providers.base import BaseProvider
from providers.blockworks import Blockworks
from providers.clawd import Clawd
from providers.defillama import DefiLlama
from providers.dune import Dune
from providers.rwa import Rwa
from providers.stakewiz import Stakewiz
from providers.token_terminal import TokenTerminal
from providers.validators_app import ValidatorsApp

app = FastAPI(title="Solana Data Aggregator", version="1.0.0")

PROVIDER_REGISTRY: List[tuple[str, type, Optional[str]]] = [
    ("clawd", Clawd, None),
    ("stakewiz", Stakewiz, None),
    ("allium", Allium, "ALLIUM_API_KEY"),
    ("artemis", Artemis, "ARTEMIS_API_KEY"),
    ("blockworks", Blockworks, "BLOCKWORKS_API_KEY"),
    ("defillama", DefiLlama, "DEFILLAMA_API_KEY"),
    ("dune", Dune, "DUNE_API_KEY"),
    ("rwa", Rwa, "RWA_API_KEY"),
    ("token_terminal", TokenTerminal, "TOKEN_TERMINAL_API_KEY"),
    ("validators_app", ValidatorsApp, "VALIDATORS_APP_API_TOKEN"),
]

# In-memory last-run cache
_last_results: Dict[str, Any] = {}
_last_run_meta: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    providers: List[str] = []  # empty = all available
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _available_providers() -> List[Dict[str, Any]]:
    result = []
    for name, cls, env_var in PROVIDER_REGISTRY:
        has_key = env_var is None or bool(os.environ.get(env_var))
        result.append({
            "name": name,
            "env_var": env_var,
            "available": has_key,
            "metrics": list(cls.METRIC_MAP.keys()),
        })
    return result


def _build_provider(name: str) -> Optional[BaseProvider]:
    for pname, cls, env_var in PROVIDER_REGISTRY:
        if pname != name:
            continue
        if env_var and not os.environ.get(env_var):
            return None
        return cls()
    return None


def _run_provider(provider: BaseProvider, start: str, end: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for metric in provider.METRIC_MAP:
        try:
            rows = provider.fetch_rows(metric, start, end)
            results[metric] = rows
        except Exception as exc:
            results[metric] = {"error": str(exc)}
    return results


def _flatten_for_export(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert nested results into flat rows for CSV/JSONL export."""
    rows = []
    for provider_name, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        for metric_name, data in metrics.items():
            if isinstance(data, list):
                for row in data:
                    rows.append({
                        "provider": provider_name,
                        "metric": metric_name,
                        "date": row.get("date", ""),
                        "value": row.get("value", ""),
                    })
            elif isinstance(data, dict) and "error" in data:
                rows.append({
                    "provider": provider_name,
                    "metric": metric_name,
                    "date": "",
                    "value": "",
                    "error": data["error"],
                })
    return rows


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/api/providers")
def get_providers() -> List[Dict[str, Any]]:
    """List all providers with availability status."""
    return _available_providers()


@app.post("/api/run")
def run_providers(req: RunRequest) -> Dict[str, Any]:
    """Run one or more providers and return results."""
    global _last_results, _last_run_meta

    today = datetime.date.today()
    end = datetime.date.fromisoformat(req.end_date) if req.end_date else today
    start = datetime.date.fromisoformat(req.start_date) if req.start_date else end - datetime.timedelta(days=6)

    requested = set(req.providers) if req.providers else None

    results: Dict[str, Any] = {}
    skipped: List[str] = []
    errors: List[str] = []

    for name, cls, env_var in PROVIDER_REGISTRY:
        if requested and name not in requested:
            continue
        if env_var and not os.environ.get(env_var):
            skipped.append(name)
            continue
        try:
            provider = cls()
            results[name] = _run_provider(provider, start.isoformat(), end.isoformat())
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    _last_results = results
    _last_run_meta = {
        "ran_at": datetime.datetime.utcnow().isoformat() + "Z",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "providers_run": list(results.keys()),
        "skipped": skipped,
        "errors": errors,
    }

    return {"meta": _last_run_meta, "results": results}


@app.get("/api/results")
def get_results() -> Dict[str, Any]:
    """Return the last run results."""
    if not _last_results:
        raise HTTPException(status_code=404, detail="No results yet — run providers first.")
    return {"meta": _last_run_meta, "results": _last_results}


@app.get("/api/export/json")
def export_json() -> StreamingResponse:
    """Export last results as JSON."""
    if not _last_results:
        raise HTTPException(status_code=404, detail="No results yet.")
    payload = json.dumps({"meta": _last_run_meta, "results": _last_results}, indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(payload.encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=solana_data_export.json"},
    )


@app.get("/api/export/csv")
def export_csv() -> StreamingResponse:
    """Export last results as CSV (flat rows)."""
    if not _last_results:
        raise HTTPException(status_code=404, detail="No results yet.")
    rows = _flatten_for_export(_last_results)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["provider", "metric", "date", "value", "error"])
    writer.writeheader()
    for row in rows:
        writer.writerow({**{"error": ""}, **row})
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=solana_data_export.csv"},
    )


@app.get("/api/export/jsonl")
def export_jsonl() -> StreamingResponse:
    """Export last results as JSONL (one record per line)."""
    if not _last_results:
        raise HTTPException(status_code=404, detail="No results yet.")
    rows = _flatten_for_export(_last_results)
    buf = "\n".join(json.dumps(r, default=str) for r in rows)
    return StreamingResponse(
        io.BytesIO(buf.encode()),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=solana_data_export.jsonl"},
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8787, reload=True)
