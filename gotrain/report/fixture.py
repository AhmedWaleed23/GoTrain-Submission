"""Build a clearly-labelled synthetic home directory so the dashboard can be demonstrated
without GPU time or real data. The marker file FIXTURE.json makes the report banner it."""

from __future__ import annotations

from pathlib import Path

from gotrain.artifacts.store import FilesystemArtifactStore
from gotrain.artifacts.verify import verify_version
from gotrain.config import Budget, Limits, RunConfig
from gotrain.experiments.runner import run_experiment
from gotrain.experiments.store import ExperimentStore
from gotrain.paths import Home
from gotrain.util.io import write_json
from gotrain.util.timing import utc_now_iso


def build_fixture_home(root: Path, with_agent: bool = True, with_final_eval: bool = True, log=print) -> Home:
    home = Home(Path(root)).ensure()
    write_json(home.root / "FIXTURE.json", {"created_at": utc_now_iso(), "note": "synthetic fixture for dashboard demonstration; not real results"})
    quiet = lambda s: None
    cls_runs = [
        RunConfig(task="synthetic", model="cnn_small", lr=1e-3, epochs=3, batch_size=16, max_minutes=2, num_workers=0, tag="baseline", hypothesis="fixed reference configuration"),
        RunConfig(task="synthetic", model="cnn_small", lr=3e-3, epochs=3, batch_size=16, max_minutes=2, num_workers=0, tag="manual", hypothesis="higher lr", screening=True, subset_fraction=0.5),
        RunConfig(task="synthetic", model="resnet_small", lr=1e-3, epochs=3, batch_size=16, max_minutes=2, num_workers=0, tag="manual", hypothesis="deeper model"),
    ]
    seg_runs = [
        RunConfig(task="synthetic", model="unet_small", lr=1e-3, epochs=3, batch_size=8, max_minutes=2, num_workers=0, tag="baseline", hypothesis="fixed reference configuration"),
        RunConfig(task="synthetic", model="unet_small", lr=3e-3, epochs=3, batch_size=8, max_minutes=2, num_workers=0, tag="manual", hypothesis="higher lr"),
    ]
    for run in cls_runs + seg_runs:
        run_experiment(home, run, source="synthetic", device="cpu", log=quiet)
    bad = RunConfig(task="synthetic", model="cnn_small", lr=1e-3, epochs=1, batch_size=16, max_minutes=2, num_workers=0, tag="manual", hypothesis="simulated failure", seed=99)
    rec = run_experiment(home, bad, source="synthetic", device="cpu", log=quiet)
    ExperimentStore(home).update(rec["experiment_id"], status="failed", error="simulated failure for the fixture (not a real error)")
    if with_agent:
        try:
            from gotrain.agent.fake_model import ReactivePolicy, ScriptedModel
            from gotrain.agent.runner import run_agent

            run_agent(home, ["synthetic"], ScriptedModel(ReactivePolicy("synthetic", "cnn_small", epochs=3)), Budget(agent_session_minutes=10, agent_max_experiments=3, agent_run_max_minutes=2), Limits.load(), device="cpu", source="synthetic", max_rounds=2, log=quiet, model_id="scripted-fake")
        except ImportError:
            log("strands not installed; fixture has no agent session")
    store = ExperimentStore(home)
    registry = FilesystemArtifactStore(home)
    best = store.best("synthetic")
    if best:
        v = registry.store_model(best["experiment_id"], note="fixture: selected model")
        verify_version(registry, v["version_id"])
        registry.promote_model(v["version_id"], "fixture: best validation score", by="fixture")
    if with_final_eval and best:
        from gotrain.final_eval.evaluate_test import final_evaluate
        from gotrain.final_eval.freeze import freeze_selection

        freeze_selection(home, ["synthetic"], by="fixture", log=quiet)
        final_evaluate(home, device="cpu", acknowledge=True, num_workers=0, log=quiet)
    return home
