import torch
import pytest
from utils.nms import non_max_suppression


def test_nms_suppresses_overlapping_boxes():
    bboxes = [
        [0, 0.9, 0, 0, 2, 2],   # high conf
        [0, 0.8, 0.5, 0.5, 2, 2],  # higher overlap with above
        [0, 0.7, 4, 4, 6, 6],   # no overlap
    ]
    result = non_max_suppression(bboxes, iou_threshold=0.5, probability_threshold=0.0, box_format="corners")
    assert len(result) == 2
    assert result[0] == bboxes[0]
    assert result[1] == bboxes[2]


def test_nms_different_classes_not_suppressed():
    bboxes = [
        [0, 0.9, 0, 0, 2, 2],   # class 0
        [1, 0.8, 0, 0, 2, 2],   # class 1, same box, should be kept
    ]
    result = non_max_suppression(bboxes, iou_threshold=0.5, probability_threshold=0.0, box_format="corners")
    assert len(result) == 2

def test_nms_filters_low_confidence():
    bboxes = [
        [0, 0.4, 0, 0, 2, 2],  # below threshold
        [0, 0.9, 3, 3, 5, 5],  # above threshold
    ]
    result = non_max_suppression(bboxes, iou_threshold=0.5, probability_threshold=0.5, box_format="corners")
    assert len(result) == 1
    assert result[0] == bboxes[1]

def test_nms_empty_input():
    result = non_max_suppression([], iou_threshold=0.5, probability_threshold=0.5)
    assert result == []

def test_nms_midpoint_format():
    bboxes = [
        [0, 0.9, 2, 2, 2, 2],   # center (2,2), width=2, height=2
        [0, 0.8, 2, 2, 2, 2],   # overlaps exactly
    ]
    result = non_max_suppression(bboxes, iou_threshold=0.5, probability_threshold=0.0, box_format="midpoint")
    assert len(result) == 1
