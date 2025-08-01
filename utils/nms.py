import torch
from .iou import intersection_over_union


def non_max_suppression(
        bboxes,
        iou_threshold,
        probability_threshold,
        box_format="corners",
):
    """
    Applies Non-Maximum Suppression (NMS) to eliminate overlapping bounding boxes.

    Args:
        bboxes (list): List of boxes [class, conf, x1, y1, x2, y2] or midpoint format.
        iou_threshold (float): IoU threshold for removing overlapping boxes.
        probability_threshold (float): Minimum confidence to keep a box.
        box_format (str): "corners" or "midpoint" format.

    Returns:
        list: Filtered list of boxes after NMS.
    """

    #bboxes = [[1,0.9,x1,y1,x2,y2],[],[]] 
    
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > probability_threshold]
    bboxes = sorted(bboxes , key=lambda x : x[1] , reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]).unsqueeze(0),
                torch.tensor(box[2:]).unsqueeze(0),
                box_format=box_format
            ) < iou_threshold
        ]
    
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms
