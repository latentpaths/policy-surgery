"""Experimental-record integrity and portable checkpoint paths, without inference."""

import hashlib
import json

from policy_surgery import analysis, io


def test_historical_absolute_checkpoint_paths_resolve_to_this_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "ROOT", tmp_path)
    path = io.checkpoint_path(
        "/another-machine/home/private/results/campaign/search/run/candidates/00.npz"
    )
    assert path == tmp_path / "results/campaign/search/run/candidates/00.npz"


def test_article_numbers_recompute_from_frozen_episodes():
    result = analysis.verify_records()
    assert result["llm_runs"] == 18 and result["runs"] == 36
    assert result["valid_candidates"] == 384
    assert result["accepted_llm_policies"] == 4
    assert result["reversal_successes"] == 226


def test_numeric_reproduction_preserves_raw_records(tmp_path):
    measured = analysis.GATE / "evaluation.json"
    before = hashlib.sha256(measured.read_bytes()).hexdigest()
    analysis.reproduce(tmp_path)
    assert hashlib.sha256(measured.read_bytes()).hexdigest() == before
    rows = json.loads((tmp_path / "results.json").read_text())
    crosswind = next(r for r in rows if r["run"] == "crosswind-llm-completion-seed0")
    assert (
        crosswind["gained"],
        crosswind["lost"],
        crosswind["base_successes"],
        crosswind["edited_successes"],
    ) == (32, 17, 25, 40)
