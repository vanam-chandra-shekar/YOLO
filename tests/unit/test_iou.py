import torch
from utils.iou import intersection_over_union
import pytest


def test_perfect_overlap_corners():
    box_pred = torch.tensor([[0, 0, 2, 2]], dtype=torch.float32)
    box_label = torch.tensor([[0, 0, 2, 2]], dtype=torch.float32)
    iou = intersection_over_union(box_pred, box_label, box_format="corners")
    assert torch.allclose(iou, torch.tensor([1.0]))


def test_no_overlap_corners():
    box_pred = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
    box_label = torch.tensor([[2, 2, 3, 3]], dtype=torch.float32)
    iou = intersection_over_union(box_pred, box_label, box_format="corners")
    assert torch.allclose(iou, torch.tensor([0.0]))


def test_partial_overlap_corners():
    box_pred = torch.tensor([[0, 0, 2, 2]], dtype=torch.float32)
    box_label = torch.tensor([[1, 1, 3, 3]], dtype=torch.float32)
    expected_iou = torch.tensor([1.0 / (4 + 4 - 1)])  # area = 1 / 7
    iou = intersection_over_union(box_pred, box_label, box_format="corners")
    assert torch.allclose(iou, expected_iou, atol=1e-6)


def test_midpoint_format():
    box_pred = torch.tensor([[1, 1, 2, 2]], dtype=torch.float32)  # → (0,0,2,2)
    box_label = torch.tensor([[2, 2, 2, 2]], dtype=torch.float32)  # → (1,1,3,3)
    expected_iou = torch.tensor([1.0 / (4 + 4 - 1)])  # intersection = 1
    iou = intersection_over_union(box_pred, box_label, box_format="midpoint")
    assert torch.allclose(iou, expected_iou, atol=1e-6)


def test_batch_multiple_boxes():
    box_preds = torch.tensor([
        [0, 0, 2, 2],   # perfect
        [0, 0, 1, 1],   # no overlap
        [1, 1, 3, 3],   # partial overlap
    ], dtype=torch.float32)

    box_labels = torch.tensor([
        [0, 0, 2, 2],   # perfect
        [2, 2, 3, 3],   # no overlap
        [2, 2, 4, 4],   # partial overlap
    ], dtype=torch.float32)

    ious = intersection_over_union(box_preds, box_labels, box_format="corners")
    assert torch.allclose(ious[0], torch.tensor(1.0))
    assert torch.allclose(ious[1], torch.tensor(0.0))
    assert (ious[2] > 0) and (ious[2] < 1)


def test_invalid_format_raises():
    box_pred = torch.tensor([[0, 0, 2, 2]], dtype=torch.float32)
    box_label = torch.tensor([[0, 0, 2, 2]], dtype=torch.float32)
    with pytest.raises(ValueError):
        intersection_over_union(box_pred, box_label, box_format="invalid")


def test_iou_zero_area_box():
    box_pred = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
    box_label = torch.tensor([[1, 1, 2, 2]], dtype=torch.float32)
    iou = intersection_over_union(box_pred, box_label, box_format="corners")
    assert torch.allclose(iou, torch.tensor([0.0]))


def test_iou_same_center_different_size_midpoint():
    box_pred = torch.tensor([[1, 1, 2, 2]], dtype=torch.float32)
    box_label = torch.tensor([[1, 1, 4, 4]], dtype=torch.float32)
    # pred = (0,0)-(2,2), label = (-1, -1)-(3,3), intersection = (0,0)-(2,2) = 4
    expected_iou = 4 / (16 + 4 - 4)  # = 4 / 16 = 0.25
    iou = intersection_over_union(box_pred, box_label, box_format="midpoint")
    assert torch.allclose(iou, torch.tensor([expected_iou]), atol=1e-6)
