import numpy as np

# Define the confusion matrix
conf_matrix = np.array([
    [50, 30, 0],  # Actual a
    [10, 20, 2],  # Actual b
    [8, 10, 30]   # Actual c
])

# Number of classes
num_classes = conf_matrix.shape[0]

# Initialize lists for storing results
precision, recall, f1_score = [], [], []

# Compute metrics for each class
for i in range(num_classes):
    TP = conf_matrix[i, i]  # True Positives
    FN = sum(conf_matrix[:, i]) - TP  # False Positives
    FP = sum(conf_matrix[i, :]) - TP  # False Negatives

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)

# Print the results in table format
print("| Class | Precision | Recall | F1-score |")
print("|-------|-----------|--------|-----------|")
for i, cls in enumerate(['a', 'b', 'c']):
    print(f"| {cls} | {precision[i]:.3f} | {recall[i]:.3f} | {f1_score[i]:.3f} |")