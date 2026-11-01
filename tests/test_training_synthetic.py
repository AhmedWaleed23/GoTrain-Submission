import json
from pathlib import Path

import pytest
import torch

from gotrain.config import Limits, RunConfig, load_task_config
from gotrain.data.normalization import Normalizer, fit_band_stats, load_stats, save_stats
from gotrain.data.synthetic import SyntheticClassification, synthetic_splits
from gotrain.data.transforms import TransformedDataset
from gotrain.models.factory import build_model
from gotrain.train.checkpoint import load_checkpoint, save_checkpoint
from gotrain.train.trainer import Trainer, TrainingAborted
from gotrain.util.io import iter_jsonl


def _setup(task_type, tmp_path, epochs=3, **run_kwargs):
    tr, va = synthetic_splits(task_type, n=32, size=32)
    stats = fit_band_stats(tr)
    norm = Normalizer.from_stats(stats)
    model_name = "cnn_small" if task_type == "classification" else "unet_small"
    kwargs = {"batch_size": 16, "lr": 3e-3, "max_minutes": 5, "num_workers": 0}
    kwargs.update(run_kwargs)
    run = RunConfig(task="synthetic", model=model_name, epochs=epochs, **kwargs)
    model = build_model(model_name, tr.meta.in_channels, tr.meta.n_classes, task_type)
    trainer = Trainer(
        run, model, TransformedDataset(tr, norm, task_type, augment=True, input_size=run.input_size), TransformedDataset(va, norm, task_type, input_size=run.input_size),
        task_type, tr.meta.n_classes, tr.meta.class_names, tmp_path, "cpu", log=lambda s: None,
    )
    return trainer, tr, va, norm


@pytest.mark.parametrize("task_type", ["classification", "segmentation"])
def test_forward_backward_reduces_loss_and_writes_artifacts(task_type, tmp_path):
    trainer, *_ = _setup(task_type, tmp_path, epochs=3)
    state = trainer.fit()
    rows = list(iter_jsonl(tmp_path / "curves.jsonl"))
    assert state["status"] == "completed"
    assert state["stop_reason"] == "max_epochs"
    assert len(rows) == 3 and [r["epoch"] for r in rows] == [1, 2, 3]
    assert rows[-1]["train_loss"] < rows[0]["train_loss"]
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    assert state["best_metric"] is not None and state["best_sha256"]


def test_checkpoint_roundtrip_reproduces_logits_and_detects_tampering(tmp_path):
    trainer, tr, va, norm = _setup("classification", tmp_path, epochs=2)
    trainer.fit()
    ck = load_checkpoint(tmp_path / "last.pt", expected_sha256=read_state(tmp_path)["last_sha256"])
    fresh = build_model("cnn_small", tr.meta.in_channels, tr.meta.n_classes, "classification")
    fresh.load_state_dict(ck["model_state"])
    fresh.eval()
    trainer.model.eval()
    x = torch.stack([TransformedDataset(va, norm, "classification")[i][0] for i in range(4)])
    with torch.no_grad():
        torch.testing.assert_close(fresh(x), trainer.model.cpu()(x), atol=1e-6, rtol=1e-5)
    (tmp_path / "best.pt").write_bytes((tmp_path / "best.pt").read_bytes()[:-10] + b"0123456789")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(tmp_path / "best.pt", expected_sha256=read_state(tmp_path)["best_sha256"])


def read_state(d: Path) -> dict:
    return json.loads((d / "trainer_state.json").read_text())


def test_interrupt_then_resume_does_not_repeat_epochs(tmp_path):
    trainer, *_ = _setup("classification", tmp_path, epochs=4)
    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return read_state(tmp_path).get("epoch_completed", 0) >= 2 if (tmp_path / "trainer_state.json").exists() else False

    trainer.stop_requested = stop_after_two
    state = trainer.fit()
    assert state["status"] == "interrupted" and state["epoch_completed"] == 2

    trainer2, *_ = _setup("classification", tmp_path, epochs=4)
    state2 = trainer2.fit(resume=True)
    assert state2["status"] == "completed"
    assert state2["epoch_completed"] == 4
    assert state2["resume_count"] == 1
    epochs = [r["epoch"] for r in iter_jsonl(tmp_path / "curves.jsonl")]
    assert epochs == [1, 2, 3, 4]

    trainer3, *_ = _setup("classification", tmp_path, epochs=4)
    state3 = trainer3.fit(resume=True)
    assert state3["epoch_completed"] == 4
    assert [r["epoch"] for r in iter_jsonl(tmp_path / "curves.jsonl")] == [1, 2, 3, 4]


def test_time_budget_stops_training(tmp_path):
    trainer, *_ = _setup("segmentation", tmp_path, epochs=50, max_minutes=0.02)
    state = trainer.fit()
    assert state["status"] == "completed"
    assert state["stop_reason"] == "time_budget"
    assert state["epoch_completed"] >= 1


def test_early_stopping(tmp_path):
    trainer, *_ = _setup("classification", tmp_path, epochs=30, early_stopping_patience=2)
    state = trainer.fit()
    assert state["stop_reason"] in ("early_stopping", "max_epochs")
    if state["stop_reason"] == "early_stopping":
        assert state["epoch_completed"] < 30


class _OOMOnce(torch.nn.Module):
    def __init__(self, inner, threshold):
        super().__init__()
        self.inner = inner
        self.threshold = threshold
        self.raised = 0

    def forward(self, x):
        if self.training and x.shape[0] > self.threshold:
            self.raised += 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB (simulated)")
        return self.inner(x)


def test_oom_recovery_halves_batch_size_and_records_it(tmp_path):
    trainer, *_ = _setup("classification", tmp_path, epochs=2)
    trainer.model = _OOMOnce(trainer.model, threshold=8)
    state = trainer.fit()
    assert state["status"] == "completed"
    assert state["effective_batch_size"] == 8
    assert state["adjustments"][0]["kind"] == "batch_size" and state["adjustments"][0]["reason"] == "cuda_oom"
    assert trainer.model.raised == 1


def test_oom_unrecoverable_aborts(tmp_path):
    trainer, *_ = _setup("classification", tmp_path, epochs=2)
    trainer.model = _OOMOnce(trainer.model, threshold=0)
    with pytest.raises(TrainingAborted):
        trainer.fit()
    assert read_state(tmp_path)["status"] == "failed"


class _NaNModel(torch.nn.Module):
    def __init__(self, inner, bad_calls):
        super().__init__()
        self.inner = inner
        self.bad_calls = bad_calls

    def forward(self, x):
        out = self.inner(x)
        if self.training and self.bad_calls > 0:
            self.bad_calls -= 1
            return out * float("nan")
        return out


def test_nonfinite_loss_recovery_lowers_lr(tmp_path):
    trainer, *_ = _setup("classification", tmp_path, epochs=2)
    trainer.model = _NaNModel(trainer.model, bad_calls=2)
    state = trainer.fit()
    assert state["status"] == "completed"
    assert any(a["kind"] == "lr" and a["reason"] == "nonfinite_loss" for a in state["adjustments"])
    assert state["effective_lr"] == pytest.approx(3e-4)


def test_normalization_fit_persist_and_tamper(tmp_path):
    tr, va = synthetic_splits("classification", n=16, size=16)
    stats = fit_band_stats(tr)
    assert stats["split"] == "train" and len(stats["mean"]) == tr.meta.in_channels
    path = tmp_path / "norm.json"
    h = save_stats(stats, path)
    loaded = load_stats(path)
    assert loaded["stats_hash"] == h and loaded["mean"] == stats["mean"]
    x = Normalizer.from_stats(loaded)(tr[0][0])
    assert torch.isfinite(x).all()
    text = path.read_text().replace('"n_samples_used": 16', '"n_samples_used": 15')
    path.write_text(text)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_stats(path)
    with pytest.raises(ValueError, match="train split only"):
        fit_band_stats(va)


def test_validation_dataset_is_deterministic():
    tr, va = synthetic_splits("segmentation", n=4, size=16)
    norm = Normalizer.from_stats(fit_band_stats(tr))
    ds = TransformedDataset(va, norm, "segmentation", augment=False)
    a = ds[1]
    b = ds[1]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_run_config_validation_catches_bad_values():
    limits = Limits.load()
    task = load_task_config("m-eurosat")
    assert RunConfig(task="m-eurosat", model="cnn_small").validate(limits, task) == []
    problems = RunConfig(task="m-eurosat", model="unet_small", lr=10.0, batch_size=1000, subset_fraction=0.1).validate(limits, task)
    joined = " ".join(problems)
    assert "lr" in joined and "batch_size" in joined and "screening" in joined and "segmentation" in joined
    with pytest.raises(ValueError, match="unknown keys"):
        RunConfig.from_dict({"task": "m-eurosat", "model": "cnn_small", "bogus": 1})


def test_synthetic_test_split_requires_token():
    from gotrain.data.base import FinalEvalToken, TestSplitAccessError

    with pytest.raises(TestSplitAccessError):
        SyntheticClassification("test")
    assert len(SyntheticClassification("test", token=FinalEvalToken("unit test"))) > 0


def test_validation_metrics_use_native_mask_resolution_with_input_size(tmp_path):
    trainer, tr, va, norm = _setup("segmentation", tmp_path, epochs=1, input_size=16)
    trainer.fit()
    ck = load_checkpoint(tmp_path / "best.pt")
    assert ck["val_metrics"]["n_pixels"] == len(va) * 32 * 32
    x, y = trainer.valid_ds[0]
    assert tuple(x.shape[-2:]) == (16, 16) and tuple(y.shape) == (32, 32)
    xt, yt = trainer.train_ds[0]
    assert tuple(xt.shape[-2:]) == (16, 16) and tuple(yt.shape) == (16, 16)


def test_resume_drops_orphan_curve_rows_and_stale_best(tmp_path):
    from gotrain.util.io import append_jsonl

    trainer, *_ = _setup("classification", tmp_path, epochs=4)
    trainer.stop_requested = lambda: (tmp_path / "trainer_state.json").exists() and read_state(tmp_path).get("epoch_completed", 0) >= 2
    trainer.fit()
    append_jsonl(tmp_path / "curves.jsonl", {"epoch": 3, "train_loss": 9.9, "orphan": True})
    (tmp_path / "best.pt").write_bytes((tmp_path / "last.pt").read_bytes())
    trainer2, *_ = _setup("classification", tmp_path, epochs=4)
    state = trainer2.fit(resume=True)
    epochs = [r["epoch"] for r in iter_jsonl(tmp_path / "curves.jsonl")]
    assert epochs == [1, 2, 3, 4]
    assert not any(r.get("orphan") for r in iter_jsonl(tmp_path / "curves.jsonl"))
    assert any(a["kind"] == "best_checkpoint_reset" for a in state["adjustments"])
    assert state["best_metric"] is not None and state["best_sha256"]
    load_checkpoint(tmp_path / "best.pt", expected_sha256=state["best_sha256"])
