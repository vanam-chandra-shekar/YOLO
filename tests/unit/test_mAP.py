import pytest
import torch

from utils.mAP import mean_average_precision

# Test data shared across tests
t1_preds = [
    [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
    [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
    [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
]
t1_targets = [
    [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
    [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
    [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
]
t1_expected = 1.0

t2_preds = [
    [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
    [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
    [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
]
t2_targets = [
    [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
    [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
    [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
]
t2_expected = 1.0

t3_preds = [
    [0, 1, 0.9, 0.55, 0.2, 0.3, 0.2],
    [0, 1, 0.8, 0.35, 0.6, 0.3, 0.2],
    [0, 1, 0.7, 0.8, 0.7, 0.2, 0.2],
]
t3_targets = [
    [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
    [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
    [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
]
t3_expected = 0.0

t4_preds = [
    [0, 0, 0.9, 0.15, 0.25, 0.1, 0.1],
    [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
    [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
]
t4_targets = [
    [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
    [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
    [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
]
t4_expected = 5 / 18

epsilon = 1e-4

@pytest.mark.parametrize("preds, targets, expected, num_classes", [
    (t1_preds, t1_targets, t1_expected, 1),
    (t2_preds, t2_targets, t2_expected, 1),
    (t3_preds, t3_targets, t3_expected, 2),
    (t4_preds, t4_targets, t4_expected, 1),
])
def test_mean_average_precision(preds, targets, expected, num_classes):
    mAP = mean_average_precision(
        preds,
        targets,
        iou_threshold=0.5,
        box_format="midpoint",
        num_classes=num_classes
    )
    assert abs(mAP - expected) < epsilon
