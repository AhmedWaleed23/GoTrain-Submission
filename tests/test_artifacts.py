import pytest

from gotrain.artifacts.store import FilesystemArtifactStore, dataset_identity
from gotrain.artifacts.verify import verify_version
from gotrain.config import RunConfig
from gotrain.experiments.runner import run_experiment
from gotrain.paths import Home
from gotrain.util.io import iter_jsonl, read_json, write_json


def _home(tmp_path) -> Home:
    return Home(tmp_path / "home").ensure()


def _train(home, model="cnn_small", lr=1e-3):
    return run_experiment(home, RunConfig(task="synthetic", model=model, lr=lr, epochs=2, batch_size=16, max_minutes=2, num_workers=0), source="synthetic", device="cpu", log=lambda s: None)


def test_store_list_retrieve_verify_promote_roundtrip(tmp_path):
    home = _home(tmp_path)
    rec = _train(home)
    store = FilesystemArtifactStore(home)
    m = store.store_model(rec["experiment_id"], note="v1")
    assert m["version_id"].startswith("mv-synthetic-0001-")
    assert m["checkpoint_sha256"] == rec["checkpoint"]["best_sha256"]
    assert m["metric"]["value"] == rec["result"]["best_val_metric"]
    assert m["architecture"]["name"] == "cnn_small" and m["preprocessing"]["normalization_stats_hash"]
    assert (store.version_dir(m["version_id"]) / "reference.npz").exists()
    rows = store.list_versions()
    assert [r["version_id"] for r in rows] == [m["version_id"]] and rows[0]["promoted"] is False
    got = store.retrieve_model(m["version_id"], dest=tmp_path / "out")
    assert got["sha256_verified"] and got["copied"] and (tmp_path / "out" / "checkpoint.pt").exists()
    res = verify_version(store, m["version_id"])
    assert res["verified"] and res["max_abs_diff"] == 0.0 and res["argmax_agreement"] == 1.0
    assert res["preprocessing_identity"]["ok"]
    assert store.list_versions()[0]["verification"] is True
    p = store.promote_model(m["version_id"], "only version so far", by="test")
    assert store.promoted("synthetic")["version_id"] == p["version_id"]
    kinds = [e["kind"] for e in iter_jsonl(home.registry / "events.jsonl")]
    for k in ("store_model", "retrieve_model", "verify_model", "promote_model"):
        assert k in kinds


def test_versions_are_immutable_and_promotion_keeps_history(tmp_path):
    home = _home(tmp_path)
    a = _train(home, lr=1e-3)
    b = _train(home, lr=2e-3)
    store = FilesystemArtifactStore(home)
    va = store.store_model(a["experiment_id"])
    with pytest.raises(ValueError, match="immutable"):
        store.store_model(a["experiment_id"])
    vb = store.store_model(b["experiment_id"], parent_version=va["version_id"])
    assert vb["parent_version"] == va["version_id"]
    store.promote_model(va["version_id"], "first", by="test")
    store.promote_model(vb["version_id"], "second", by="test")
    assert store.promoted("synthetic")["previous"] == va["version_id"]
    assert read_json(store.version_dir(va["version_id"]) / "manifest.json")["version_id"] == va["version_id"]
    with pytest.raises(ValueError):
        store.store_model(a["experiment_id"], parent_version="mv-does-not-exist")


def test_corrupted_checkpoint_fails_clearly(tmp_path):
    home = _home(tmp_path)
    rec = _train(home)
    store = FilesystemArtifactStore(home)
    m = store.store_model(rec["experiment_id"])
    ck = store.version_dir(m["version_id"]) / "checkpoint.pt"
    ck.write_bytes(ck.read_bytes()[:-3] + b"abc")
    with pytest.raises(ValueError, match="does not match manifest"):
        store.retrieve_model(m["version_id"])
    res = verify_version(store, m["version_id"])
    assert res["verified"] is False and "retrieve failed" in res["error"]


def test_mismatched_manifest_hash_fails(tmp_path):
    home = _home(tmp_path)
    rec = _train(home)
    store = FilesystemArtifactStore(home)
    m = store.store_model(rec["experiment_id"])
    idx = read_json(store.index_path)
    idx["versions"][m["version_id"]]["checkpoint_sha256"] = "0" * 64
    write_json(store.index_path, idx)
    with pytest.raises(ValueError):
        store.retrieve_model(m["version_id"])


def test_index_recovers_from_manifest_when_index_lost(tmp_path):
    home = _home(tmp_path)
    rec = _train(home)
    store = FilesystemArtifactStore(home)
    m = store.store_model(rec["experiment_id"])
    store.index_path.unlink()
    rows = FilesystemArtifactStore(home).list_versions()
    assert rows[0]["version_id"] == m["version_id"]


def test_identifiers_and_paths_are_validated(tmp_path):
    store = FilesystemArtifactStore(_home(tmp_path))
    for bad in ("../x", "mv-../../etc", "not-a-version", "mv-synthetic-0001-abc/../.."):
        with pytest.raises(ValueError):
            store.get_version(bad)
    with pytest.raises(FileNotFoundError):
        store.get_version("mv-synthetic-9999-deadbeef")
    with pytest.raises(ValueError):
        store.store_dataset("bad name", tmp_path)


def test_dataset_registration_reference_and_copy(tmp_path):
    home = _home(tmp_path)
    src = tmp_path / "data" / "segmentation_v1.0" / "m-fake"
    src.mkdir(parents=True)
    (src / "task_specs.pkl").write_bytes(b"pickle")
    write_json(src / "default_partition.json", {"train": ["a"], "valid": ["b"], "test": ["c"]})
    for n in ("a", "b", "c"):
        (src / f"{n}.hdf5").write_bytes(b"x" * 10)
    store = FilesystemArtifactStore(home)
    rec = store.store_dataset("m-fake", src, mode="reference")
    assert rec["mode"] == "reference" and rec["location"] == str(src.resolve())
    got = store.retrieve_dataset("m-fake")
    assert got["identity_verified"] and not got["copied"]
    copy = store.store_dataset("m-fake", src, mode="copy")
    assert copy["location"].startswith(str(home.registry)) and dataset_identity(copy["location"])["identity_hash"] == rec["identity"]["identity_hash"]
    again = store.store_dataset("m-fake", src, mode="copy")
    assert "reused" in again["mode"]
    (src / "a.hdf5").write_bytes(b"y" * 10)
    store.store_dataset("m-fake", src, mode="reference")
    (src / "a.hdf5").write_bytes(b"z" * 11)
    with pytest.raises(ValueError, match="changed since registration"):
        store.retrieve_dataset("m-fake")
