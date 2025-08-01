import torch
from collections import Counter
from .iou import intersection_over_union

def mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_threshold=0.5,
        box_format="corners",
        num_classes = 20
):
    # pred_boxes : List : [[train_idx , class_pred , prob_score , x1,y1,x2,y2]]
    average_precision = []
    epsilon = 1e-6

    for c in range(num_classes):

        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        
        for gt in true_boxes:
            if gt[1] == c:
                ground_truths.append(gt)
        
        amount_bbox = Counter((gt[0] for gt in ground_truths))

        amount_bbox = {k: torch.zeros(v , dtype=torch.bool) for k,v in amount_bbox.items()}

        detections.sort(key=lambda x:x[2] , reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        total_true_bbox = len(ground_truths)

        for didx , detection in enumerate(detections):

            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gt = len(ground_truth_img)

            best_iou = 0
            best_gt_idx= -1

            for idx , gt in enumerate(ground_truth_img):

                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            
            if best_iou > iou_threshold:

                if amount_bbox[detection[0]][best_gt_idx] == 0:
                    TP[didx] = 1
                    amount_bbox[detection[0]][best_gt_idx] =1
                else:
                    FP[didx] = 1
            else:
                FP[didx] = 1
        
        TP_cumsum = torch.cumsum(TP , dim=0)
        FP_cumsum = torch.cumsum(FP , dim=0)

        recall = TP_cumsum / total_true_bbox + epsilon
        precision = torch.divide(TP_cumsum , (TP_cumsum + FP_cumsum + epsilon))
        precision = torch.cat((torch.tensor([1]) , precision))
        recall = torch.cat((torch.tensor([0]) , recall))

        average_precision.append(torch.trapz(precision,recall))

        return sum(average_precision) / len(average_precision)