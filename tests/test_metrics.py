import numpy as np
import pytest
import torch

from gotrain.train.metrics import (
    ConfusionAccumulator,
    classification_metrics,
    compute_metrics,
    segmentation_metrics,
)


def test_classification_small_manual_case():
    acc = ConfusionAccumulator(3)
    acc.update(torch.tensor([0, 1, 1, 2]), torch.tensor([0, 1, 2, 2]))
    cm = acc.matrix()
    assert cm.tolist() == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    m = classification_metrics(cm, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(0.75)
    # class a: f1 1.0 ; class b: p=0.5 r=1 f1=2/3 ; class c: p=1 r=0.5 f1=2/3
    assert m["macro_f1"] == pytest.approx((1.0 + 2 / 3 + 2 / 3) / 3)
    assert m["per_class"][1]["precision"] == pytest.approx(0.5)
    assert m["n_samples"] == 4


def test_accumulates_over_batches_not_averaged():
    acc = ConfusionAccumulator(2)
    acc.update(torch.tensor([1, 1, 1, 1]), torch.tensor([1, 1, 1, 1]))
    acc.update(torch.tensor([0]), torch.tensor([1]))
    m = classification_metrics(acc.matrix())
    assert m["accuracy"] == pytest.approx(4 / 5)


def test_segmentation_manual_case_with_absent_class():
    pred = torch.tensor([[0, 0], [1, 1]])
    target = torch.tensor([[0, 1], [1, 1]])
    acc = ConfusionAccumulator(3)
    acc.update(pred, target)
    m = segmentation_metrics(acc.matrix(), ["bg", "fg", "never"])
    # class0: tp1 fp1 fn0 -> 0.5 ; class1: tp2 fp0 fn1 -> 2/3 ; class2 absent -> excluded
    assert m["per_class"][0]["iou"] == pytest.approx(0.5)
    assert m["per_class"][1]["iou"] == pytest.approx(2 / 3)
    assert m["per_class"][2]["iou"] is None
    assert m["miou"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert m["n_classes_present"] == 2
    assert m["pixel_accuracy"] == pytest.approx(0.75)


def test_segmentation_absent_in_target_but_predicted_counts_as_zero():
    pred = torch.tensor([2, 0])
    target = torch.tensor([0, 0])
    acc = ConfusionAccumulator(3)
    acc.update(pred, target)
    m = segmentation_metrics(acc.matrix())
    assert m["per_class"][2]["iou"] == pytest.approx(0.0)
    assert m["miou"] == pytest.approx((0.5 + 0.0) / 2)


def test_ignore_index_is_dropped():
    acc = ConfusionAccumulator(2, ignore_index=255)
    acc.update(torch.tensor([1, 0, 1]), torch.tensor([1, 255, 0]))
    assert acc.total == 2
    assert acc.n_ignored == 1
    m = segmentation_metrics(acc.matrix())
    assert m["pixel_accuracy"] == pytest.approx(0.5)


def test_out_of_range_labels_fail_loudly():
    acc = ConfusionAccumulator(2)
    with pytest.raises(ValueError):
        acc.update(torch.tensor([0]), torch.tensor([2]))
    with pytest.raises(ValueError):
        acc.update(torch.tensor([0]), torch.tensor([-1]))


def test_binary_foreground_iou_and_exclusion():
    pred = torch.tensor([1, 1, 0, 0])
    target = torch.tensor([1, 0, 0, 0])
    acc = ConfusionAccumulator(2)
    acc.update(pred, target)
    m = segmentation_metrics(acc.matrix(), ["bg", "fg"], exclude_from_diagnostic_miou=[0])
    assert m["foreground_iou"] == pytest.approx(0.5)
    assert m["miou_excluding"]["value"] == pytest.approx(0.5)


def test_compute_metrics_dispatch():
    cm = np.array([[2, 0], [0, 2]])
    assert compute_metrics("classification", cm)["accuracy"] == 1.0
    assert compute_metrics("segmentation", cm)["miou"] == 1.0
    with pytest.raises(ValueError):
        compute_metrics("regression", cm)


torchmetrics = pytest.importorskip("torchmetrics")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_torchmetrics_reference(seed):
    g = torch.Generator().manual_seed(seed)
    k = 5
    # Leave class 4 out of the target so an absent class exists in some batches.
    target = torch.randint(0, k - 1, (6, 9, 9), generator=g)
    pred = torch.randint(0, k, (6, 9, 9), generator=g)
    acc = ConfusionAccumulator(k)
    for i in range(6):
        acc.update(pred[i], target[i])
    ours = segmentation_metrics(acc.matrix())
    ref = torchmetrics.JaccardIndex(task="multiclass", num_classes=k, average="macro")
    for i in range(6):
        ref.update(pred[i], target[i])
    assert ours["miou"] == pytest.approx(ref.compute().item(), abs=1e-6)

    cls_target = target.reshape(-1)
    cls_pred = pred.reshape(-1)
    acc2 = ConfusionAccumulator(k)
    acc2.update(cls_pred, cls_target)
    ref_acc = torchmetrics.Accuracy(task="multiclass", num_classes=k)
    assert classification_metrics(acc2.matrix())["accuracy"] == pytest.approx(
        ref_acc(cls_pred, cls_target).item(), abs=1e-6
    )
    ref_f1 = torchmetrics.F1Score(task="multiclass", num_classes=k, average="macro")
    assert classification_metrics(acc2.matrix())["macro_f1"] == pytest.approx(
        ref_f1(cls_pred, cls_target).item(), abs=1e-6
    )
