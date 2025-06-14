def compute_metrics(predictions, ground_truth, label_list, time_tolerance=2.0):
    """
    predictions: [(timestamp, [label, ...]), ...]
    ground_truth: [(timestamp, [label, ...]), ...]
    """
    num_labels = len(label_list)
    label_to_idx = {label: i for i, label in enumerate(label_list)}

    def binarize(events):
        bin_vector = [0] * num_labels
        for label in events:
            if label in label_to_idx:
                bin_vector[label_to_idx[label]] = 1
        return bin_vector

    # Match predictions to ground truth clips using time tolerance
    y_true = []
    y_pred = []

    gt_used = set()
    for pt, pred_labels in predictions:
        best_match = None
        for i, (gt_time, gt_labels) in enumerate(ground_truth):
            if i in gt_used:
                continue
            if abs(gt_time - pt) <= time_tolerance:
                best_match = i
                break

        if best_match is not None:
            y_true.append(binarize(ground_truth[best_match][1]))
            gt_used.add(best_match)
        else:
            y_true.append([0] * num_labels)

        y_pred.append(binarize(pred_labels))

    # Add unused ground truth as false negatives
    for i, (gt_time, gt_labels) in enumerate(ground_truth):
        if i not in gt_used:
            y_true.append(binarize(gt_labels))
            y_pred.append([0] * num_labels)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": multilabel_confusion_matrix(y_true, y_pred)
    }

    return metrics
