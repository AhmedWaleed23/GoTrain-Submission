"""Completion orchestration failures must not be reported as success."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gotrain.completion import complete_tasks


class FakeHome:
    def __init__(self, root):
        self.root = Path(root)
        self.selection = self.root / "selection"
        self.final_eval = self.root / "final_eval"
        self.experiments = self.root / "experiments"

    def ensure(self):
        for path in (self.selection, self.final_eval, self.experiments):
            path.mkdir(parents=True, exist_ok=True)
        return self


class FakeOperations:
    def __init__(self):
        self.reviews, self.verifications = [], []
        self.bad_verify, self.missing = set(), set()
        self.bad_review = set()
        self.dashboard_error = False

    def selection(self, task):
        if task in self.missing:
            raise ValueError("No selected model")
        return {"experiment_id": "exp-" + task, "checkpoint_sha256": "123", "selected_by": "agent"}

    def store(self, task, selected):
        return {"version_id": "mv-" + task, "reused": True}

    def verify(self, version_id):
        self.verifications.append(version_id)
        return {"verified": version_id not in self.bad_verify}

    def review(self, task, version_id):
        self.reviews.append(task)
        if task in self.bad_review:
            raise RuntimeError("review failed")
        return {"page": task + "/review.html", "partial": False, "n_images": 66}

    def report(self):
        if self.dashboard_error:
            raise RuntimeError("dashboard failed")
        return "reports/dashboard.html"


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = FakeHome(self.temp.name).ensure()
        self.ops = FakeOperations()

    def run_completion(self, tasks):
        return complete_tasks(self.home, tasks, operations=self.ops, log=lambda _: None)

    def test_binary_reviews_and_classification_skip_are_explicit(self):
        result = self.run_completion(["m-pv4ger-seg", "m-nz-cattle", "m-eurosat"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.ops.reviews, ["m-pv4ger-seg", "m-nz-cattle"])
        self.assertEqual(result["tasks"]["m-eurosat"]["stages"]["review"]["status"], "skipped")
        self.assertTrue(result["selection_and_final_evidence_unchanged"])
        self.assertEqual(json.loads(Path(result["summary_path"]).read_text()), result)

    def test_verification_failure_blocks_review_but_not_other_tasks(self):
        self.ops.bad_verify.add("mv-m-nz-cattle")
        result = self.run_completion(["m-nz-cattle", "m-pv4ger-seg"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(self.ops.reviews, ["m-pv4ger-seg"])
        self.assertEqual(result["tasks"]["m-nz-cattle"]["stages"]["verification"]["status"], "failed")

    def test_missing_selection_fails_without_verification(self):
        self.ops.missing.add("m-eurosat")
        result = self.run_completion(["m-eurosat"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.ops.verifications, [])
        self.assertEqual(result["dashboard"]["status"], "completed")

    def test_review_error_is_visible_and_summary_persisted(self):
        self.ops.bad_review.add("m-nz-cattle")
        result = self.run_completion(["m-nz-cattle"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("review failed", result["tasks"]["m-nz-cattle"]["stages"]["review"]["error"])
        self.assertTrue((self.home.root / "completion/latest.json").is_file())

    def test_report_failure_does_not_return_completed(self):
        self.ops.dashboard_error = True
        result = self.run_completion(["m-eurosat"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["dashboard"]["status"], "failed")

    def test_frozen_and_test_evidence_preserved_across_reruns(self):
        files = [self.home.selection / "manifest.json", self.home.final_eval / "results.json", self.home.final_eval / "exposure.jsonl"]
        for path in files:
            path.write_bytes(b"original evidence\n")
        first = self.run_completion(["m-eurosat"])
        second = self.run_completion(["m-eurosat"])
        self.assertNotEqual(first["summary_path"], second["summary_path"])
        self.assertTrue(second["tasks"]["m-eurosat"]["stages"]["storage"]["reused"])
        for path in files:
            self.assertEqual(path.read_bytes(), b"original evidence\n")

    def test_changed_selection_is_not_reported_as_completed(self):
        original = self.ops.selection
        counts = {}
        def changed(task):
            counts[task] = counts.get(task, 0) + 1
            result = original(task)
            if counts[task] > 1:
                result["experiment_id"] = "changed"
            return result
        self.ops.selection = changed
        result = self.run_completion(["m-eurosat"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["tasks"]["m-eurosat"]["stages"]["selection_identity"]["status"], "failed")

    def test_invalid_limits_rejected_before_work(self):
        with self.assertRaises(ValueError):
            complete_tasks(self.home, ["m-eurosat"], batch_size=0, operations=self.ops)
        self.assertEqual(self.ops.verifications, [])


def test_real_synthetic_completion_reuses_version_and_never_opens_test(tmp_path):
    """Run real storage/verification/reporting, not fake operations (SageMaker)."""
    from gotrain.config import RunConfig
    from gotrain.paths import Home
    from gotrain.experiments.runner import run_experiment
    from gotrain.experiments.store import ExperimentStore
    from gotrain.artifacts.store import FilesystemArtifactStore
    home = Home(tmp_path / "runs").ensure()
    record = run_experiment(home, RunConfig(task="synthetic", model="cnn_small", epochs=1,
                            batch_size=8, max_minutes=1, num_workers=0), source="synthetic",
                            device="cpu", log=lambda _: None)
    ExperimentStore(home).set_best("synthetic", record["experiment_id"], by="test", rationale="synthetic completion integration")
    first = complete_tasks(home, ["synthetic"], log=lambda _: None)
    assert first["status"] == "completed", first
    second = complete_tasks(home, ["synthetic"], log=lambda _: None)
    assert second["status"] == "completed", second
    assert len(FilesystemArtifactStore(home).list_versions()) == 1
    assert second["tasks"]["synthetic"]["stages"]["storage"]["reused"]
    assert not list(home.final_eval.iterdir())
    assert Path(second["dashboard"]["path"]).is_file()


def test_agent_completion_flag_connects_to_workflow(monkeypatch, tmp_path):
    """Exercise the actual cmd_agent handoff without Bedrock or GPU training."""
    import gotrain.agent.runner
    import gotrain.agent.bedrock
    import gotrain.completion
    from gotrain.cli import build_parser
    calls = []
    monkeypatch.setattr(gotrain.agent.bedrock, "make_bedrock_model", lambda **_: object())
    monkeypatch.setattr(gotrain.agent.runner, "run_agent", lambda *a, **kw: {
        "session_id": "fake-session", "status": "completed", "tool_calls": 0,
        "experiments_launched": 0, "rounds": 1, "finished_tasks": {}, "errors": []})
    def completed(home, tasks, **kwargs):
        calls.append((tasks, kwargs))
        return {"status": "completed"}
    monkeypatch.setattr(gotrain.completion, "complete_tasks", completed)
    args = build_parser().parse_args(["--home", str(tmp_path), "agent", "--task", "m-eurosat", "--complete", "--completion-batch-size", "4"])
    assert args.fn(args) == 0
    assert calls[0][0] == ["m-eurosat"]
    assert calls[0][1]["batch_size"] == 4
