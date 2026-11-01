import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from gotrain.config import RunConfig
from gotrain.data.base import FinalEvalToken, TestSplitAccessError
from gotrain.data.synthetic import SyntheticSegmentation, synthetic_splits
from gotrain.data.transforms import logits_to_mask_resolution, resize_pair
from gotrain.experiments.runner import run_experiment
from gotrain.experiments.store import ExperimentStore
from gotrain.paths import Home
from gotrain.review import metrics as rm
from gotrain.review.build import assert_review_split, build_review, default_out_dir, load_review
from gotrain.review.model_source import from_bundle, from_home
from gotrain.review.notes import NOTES_FORMAT, new_export, run_identity, to_csv, validate_import
from gotrain.review.page import render_review_page
from gotrain.review.signals import image_priority, normalized_entropy, rank_queue
from gotrain.util.io import read_json, write_json


def test_entropy_uniform_and_near_certain():
    uniform = np.full((2, 4, 4), 0.5)
    assert np.allclose(normalized_entropy(uniform, axis=0), 1.0)
    certain = np.zeros((2, 4, 4))
    certain[0] = 0.999
    certain[1] = 0.001
    ent = normalized_entropy(certain, axis=0)
    assert ent.max() < 0.02
    three = np.full((3, 2, 2), 1 / 3)
    assert np.allclose(normalized_entropy(three, axis=0), 1.0)
    with pytest.raises(ValueError):
        normalized_entropy(np.full((2, 2, 2), 0.7), axis=0)


def test_image_priority_top_fraction_and_ranking_ties():
    ent = np.zeros((10, 10))
    ent[0, :10] = 1.0
    assert image_priority(ent, 0.10) == pytest.approx(1.0)
    assert image_priority(ent, 0.50) == pytest.approx(0.2)
    order = rank_queue({"b": 0.5, "a": 0.5, "c": 0.9, "d": 0.1})
    assert order == ["c", "a", "b", "d"]
    assert rank_queue({"b": 0.5, "a": 0.5}) == rank_queue({"a": 0.5, "b": 0.5})


def test_logits_resized_to_native_mask_and_masks_not_bilinear():
    x = torch.rand(3, 32, 32)
    y = torch.randint(0, 2, (32, 32))
    xs, ys = resize_pair(x, y, 16, "segmentation", resize_mask=False)
    assert tuple(xs.shape[-2:]) == (16, 16) and tuple(ys.shape) == (32, 32)
    logits = torch.randn(1, 2, 16, 16)
    up = logits_to_mask_resolution(logits, ys.unsqueeze(0))
    assert tuple(up.shape[-2:]) == (32, 32)
    assert set(torch.unique(ys).tolist()) <= {0, 1}
    with pytest.raises(ValueError):
        rm.misclassified_pixels(np.zeros((4, 4)), np.zeros((5, 5)))


def test_review_split_rejects_test_before_io():
    with pytest.raises(PermissionError):
        assert_review_split("test")
    with pytest.raises(ValueError):
        assert_review_split("val")
    assert_review_split("valid")
    with pytest.raises(TestSplitAccessError):
        SyntheticSegmentation("test")


def test_foreground_iou_and_error_capture_accounting():
    assert rm.foreground_iou(np.zeros((2, 2)), np.zeros((2, 2))) == (1.0, True)
    pred = np.array([[1, 1], [0, 0]])
    target = np.array([[1, 0], [0, 0]])
    iou, empty = rm.foreground_iou(pred, target)
    assert iou == pytest.approx(0.5) and not empty
    assert rm.misclassified_pixels(pred, target) == 1
    ranked = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    errors = {s: 0 for s in ranked}
    errors["a"] = 50
    errors["b"] = 30
    errors["j"] = 20
    cap = rm.error_capture(ranked, errors)
    assert cap["applicable"] and cap["total_misclassified_pixels"] == 100
    assert cap["budgets"]["10%"]["images"] == 1 and cap["budgets"]["10%"]["proportion"] == pytest.approx(0.5)
    assert cap["budgets"]["20%"]["proportion"] == pytest.approx(0.8)
    assert cap["budgets"]["30%"]["proportion"] == pytest.approx(0.8)
    assert rm.budget_sizes(403) == {"10%": 41, "20%": 81, "30%": 121}
    curve = rm.cumulative_capture_curve(ranked, errors)
    assert curve[0] == pytest.approx(0.5) and curve[-1] == pytest.approx(1.0)


def test_random_baseline_reproducible_and_zero_errors_not_applicable():
    ids = [f"s{i}" for i in range(50)]
    rng = np.random.default_rng(0)
    errors = {s: int(v) for s, v in zip(ids, rng.integers(0, 100, 50))}
    a = rm.random_baseline(ids, errors)
    b = rm.random_baseline(ids, errors)
    assert a == b and a["n_queues"] == 200 and a["seed"] == 42
    for row in a["budgets"].values():
        assert 0 <= row["p5"] <= row["mean"] <= row["p95"] <= 1
    zero = {s: 0 for s in ids}
    cap = rm.error_capture(ids, zero)
    base = rm.random_baseline(ids, zero)
    assert not cap["applicable"] and cap["budgets"]["10%"]["proportion"] is None
    assert not base["applicable"] and base["budgets"]["10%"]["mean"] is None
    comp = rm.compare_to_random(cap, base)
    assert not comp["applicable"] and comp["budgets"]["10%"]["beats_random_mean"] is None
    assert rm.cumulative_capture_curve(ids, zero) is None


def _trained_home(tmp_path) -> tuple[Home, dict]:
    home = Home(tmp_path / "home").ensure()
    rec = run_experiment(home, RunConfig(task="synthetic", model="unet_small", epochs=2, batch_size=8, max_minutes=2, num_workers=0, input_size=16), source="synthetic", device="cpu", log=lambda s: None)
    ExperimentStore(home).set_best("synthetic", rec["experiment_id"], by="test", rationale="only run")
    return home, rec


def test_build_on_synthetic_fixture_is_labelled_and_consistent(tmp_path):
    home, rec = _trained_home(tmp_path)
    source = from_home(home, "synthetic", device="cpu")
    assert source.input_size == 16 and source.mode == "home"
    _, valid = synthetic_splits("segmentation", seed=0)
    out = default_out_dir(home, source, None)
    res = build_review(home, source, out, device="cpu", batch_size=8, num_workers=0, valid_dataset=valid, log=lambda s: None)
    assert res["n_images"] == 32 and not res["partial"]
    review = load_review(out)
    prov = review["provenance"]
    assert prov["synthetic"] is True and prov["data"]["split"] == "valid"
    assert prov["inference"]["precision"] == "fp32" and prov["inference"]["logits_upsampled_to_native"]
    items = review["items"]
    assert all(it["mask_shape"] == [32, 32] for it in items)
    assert [it["sample_id"] for it in sorted(items, key=lambda i: i["rank"])] == review["queue"]
    assert review["queue"] == rank_queue({it["sample_id"]: it["priority"] for it in items})
    assert (out / "cache" / "0000_rgb.jpeg").exists() or (out / "cache" / "0000_rgb.png").exists()
    assert prov["metric_comparison"]["within_tolerance"] or prov["metric_comparison"]["abs_diff"] < 0.05
    page = render_review_page(review)
    txt = page.read_text()
    assert "SYNTHETIC FIXTURE" in txt and "Review solar-panel predictions" in txt
    assert "<script src" not in txt and "https://" not in txt.replace("http://www.w3.org/2000/svg", "")
    assert '"demo":true' in txt
    partial_out = default_out_dir(home, source, 4)
    res2 = build_review(home, source, partial_out, device="cpu", num_workers=0, valid_dataset=valid, limit=4, log=lambda s: None)
    assert res2["partial"] and res2["n_images"] == 4
    assert "PARTIAL" in render_review_page(load_review(partial_out)).read_text()
    with pytest.raises(FileExistsError, match="different model, split or limit"):
        build_review(home, source, partial_out, device="cpu", num_workers=0, valid_dataset=valid, limit=5, log=lambda s: None)


def test_review_does_not_select_screening_or_partial_runs(tmp_path):
    home = Home(tmp_path / "home").ensure()
    rec = run_experiment(home, RunConfig(task="synthetic", model="unet_small", epochs=1, batch_size=8, max_minutes=2, num_workers=0, screening=True, subset_fraction=0.5), source="synthetic", device="cpu", log=lambda s: None)
    with pytest.raises(ValueError, match="screening"):
        from_home(home, "synthetic", device="cpu", experiment_id=rec["experiment_id"])
    with pytest.raises(FileNotFoundError, match="no selected experiment"):
        from_home(home, "synthetic", device="cpu")


def _make_bundle(tmp_path, home, rec) -> Path:
    from gotrain.artifacts.store import FilesystemArtifactStore

    store = FilesystemArtifactStore(home)
    m = store.store_model(rec["experiment_id"], note="bundle test")
    vdir = store.version_dir(m["version_id"])
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("manifest.json", "checkpoint.pt", "reference.npz"):
        (bundle / name).write_bytes((vdir / name).read_bytes())
    from gotrain.data.normalization import fit_band_stats, save_stats

    train, _ = synthetic_splits("segmentation", seed=0)
    save_stats(fit_band_stats(train), bundle / "normalization.json")
    return bundle


def test_bundle_loading_checks_hashes_and_reference(tmp_path):
    home, rec = _trained_home(tmp_path)
    bundle = _make_bundle(tmp_path, home, rec)
    src = from_bundle(bundle, device="cpu")
    assert src.mode == "bundle" and src.reference_check["within_tolerance"] and src.input_size == 16
    ck = bundle / "checkpoint.pt"
    ck.write_bytes(ck.read_bytes()[:-4] + b"zzzz")
    with pytest.raises(ValueError, match="checkpoint hash"):
        from_bundle(bundle, device="cpu")
    ck.unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        from_bundle(bundle, device="cpu")
    with pytest.raises(FileNotFoundError):
        from_bundle(tmp_path / "nope", device="cpu")


def test_bundle_normalization_mismatch_and_reference_mismatch(tmp_path):
    home, rec = _trained_home(tmp_path)
    bundle = _make_bundle(tmp_path, home, rec)
    norm = read_json(bundle / "normalization.json")
    norm["mean"][0] += 1.0
    from gotrain.data.normalization import save_stats

    save_stats({k: v for k, v in norm.items() if k != "stats_hash"}, bundle / "normalization.json")
    with pytest.raises(ValueError, match="normalization hash"):
        from_bundle(bundle, device="cpu")
    ref = np.load(bundle / "reference.npz")
    np.savez_compressed(bundle / "reference.npz", inputs=ref["inputs"], logits=ref["logits"] + 5.0, indices=ref["indices"])
    manifest = read_json(bundle / "manifest.json")
    from gotrain.data.normalization import fit_band_stats

    train, _ = synthetic_splits("segmentation", seed=0)
    save_stats(fit_band_stats(train), bundle / "normalization.json")
    with pytest.raises(ValueError, match="reference logits"):
        from_bundle(bundle, device="cpu")


def test_notes_export_import_roundtrip_and_identity_check(tmp_path):
    home, rec = _trained_home(tmp_path)
    source = from_home(home, "synthetic", device="cpu")
    _, valid = synthetic_splits("segmentation", seed=0)
    out = default_out_dir(home, source, None)
    build_review(home, source, out, device="cpu", num_workers=0, valid_dataset=valid, log=lambda s: None)
    review = load_review(out)
    ident = run_identity(review["provenance"])
    tricky = 'He said "fine",\nthen <b>bold</b> & done\r\n'
    payload = new_export(ident, {"s1": {"decision": "needs_correction", "note": tricky, "updated_at": "2026-01-01T00:00:00Z"}, "s2": {"decision": None, "note": "plain"}}, reviewer="tester")
    text = json.dumps(payload)
    entries = validate_import(json.loads(text), ident)
    assert entries[0]["note"] == tricky and entries[0]["decision"] == "needs_correction"
    csv_text = to_csv(payload)
    import csv
    import io

    rows = list(csv.reader(io.StringIO(csv_text)))
    assert rows[0][0] == "sample_id" and rows[1][3] == tricky and rows[1][2] == "Needs correction"
    other = dict(payload)
    other["identity"] = {**ident, "identity_hash": "0" * 64}
    with pytest.raises(ValueError, match="different run"):
        validate_import(other, ident)
    bad = dict(payload)
    bad["reviews"] = [{"sample_id": "x", "decision": "approve"}]
    with pytest.raises(ValueError, match="unknown decision"):
        validate_import(bad, ident)
    with pytest.raises(ValueError, match="format"):
        validate_import({"format": "other"}, ident)
    with pytest.raises(ValueError):
        new_export(ident, {"s": {"decision": "bogus"}})


def test_no_training_or_fitting_calls_in_review_package():
    src = ""
    for name in ("build", "model_source", "page", "signals", "metrics", "notes"):
        src += Path(f"gotrain/review/{name}.py").read_text()
    for forbidden in ("fit_band_stats", "Trainer(", ".backward(", "optim.", "final_eval", "FinalEvalToken", "split=\"test\"", "'test'"):
        assert forbidden not in src, forbidden
