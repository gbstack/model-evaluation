import pandas as pd
import argparse
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import matplotlib.pyplot as plt


def round_by_threshold(y_pred, threshold=0.5):
    y_pred_copy = y_pred.copy()
    y_pred_copy[y_pred >= threshold] = 1
    y_pred_copy[y_pred < threshold] = 0
    return y_pred_copy


parser = argparse.ArgumentParser()
parser.add_argument("pred_result", help="prediction result file to evaluate (multiple files use comma to separate)")
parser.add_argument("--model-names", "-m", help="Model names (same order as predictions parameter)")
parser.add_argument("--test-name", "-t", help="name of test data set", nargs="*")
parser.add_argument("--threshold", "-T", help="threshold for prediction", type=float, default=0.5)
parser.add_argument("--no-threshold", "-N", help="performance evaluation without threshold (AUC,AUPR)",
                    action="store_true")
parser.add_argument("--evaluation-output", "-o", help="output for result evaluation")
args = parser.parse_args()
test_names = args.test_name
output_file = args.evaluation_output
th = args.threshold

if args.no_threshold:
    evaluation_df = pd.DataFrame(index=["Sen", "Spe", "Pre", "Acc", "F1", "AUC", "AUPR"])
else:
    evaluation_df = pd.DataFrame(index=["Sen", "Spe", "Pre", "Acc", "F1"])

roc_curves = []
pr_curves = []

model_names = []
if args.model_names is not None:
    model_names = args.model_names.split(',')

paths = args.pred_result.split(',')

for path in paths:

    extension = path.split(".")[-1]

    if extension == 'csv':
        result_df = pd.read_csv(path, header=[0, 1])
    elif extension == 'tsv':
        result_df = pd.read_table(path, header=[0, 1])

    for dataset in test_names:
        print(result_df.get(dataset) is None, result_df.get('gt') is None)
        if result_df.get(dataset) is not None:
            gt_column = result_df[dataset, "label"]
            pred_column = result_df[dataset, "predicted"]
        else:
            gt_column = result_df["gt"]
            pred_column = result_df["pred"]
        tn, fp, fn, tp = confusion_matrix(gt_column.dropna(), round_by_threshold(pred_column.dropna(), th)).ravel()
        print("Evaluation of the %s set " % dataset)
        sen = float(tp) / (fn + tp)
        pre = float(tp) / (tp + fp)
        spe = float(tn) / (tn + fp)
        acc = float(tn + tp) / (tn + fp + fn + tp)
        f1 = (2 * sen * pre) / (sen + pre)
        print("\tSen : ", sen)
        print("\tSpe : ", spe)
        print("\tAcc : ", acc)
        print("\tPrecision : ", pre)
        print("\tF1 : ", f1)
        result_dic = {"Acc": acc, "Sen": sen, "Pre": pre, "F1": f1, "Spe": spe}
        if args.no_threshold:
            fpr, tpr, thresholds_AUC = roc_curve(gt_column, pred_column)
            AUC = auc(fpr, tpr)
            precision, recall, thresholds = precision_recall_curve(gt_column, pred_column)
            AUPR = auc(recall, precision)

            roc_curves.append([fpr, tpr])
            pr_curves.append([precision, recall])
            print("\tArea Under ROC Curve(AUC): %0.3f" % AUC)
            print("\tArea Under PR Curve(AUPR): %0.3f" % AUPR)
            print("=================================================")
            result_dic.update({"AUC": AUC, "AUPR": AUPR})
        evaluation_df[dataset] = pd.Series(result_dic)

fig = plt.figure(figsize=(10, 5))
fig.add_subplot(1, 2, 1)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
for i, roc_curve in enumerate(roc_curves):
    if len(model_names) > 0:
        plt.plot(*roc_curves[i], label=model_names[i])
    else:
        plt.plot(*roc_curves[i])
plt.legend()
ax = plt.gca()
ax.set_xlim(0.0, 1)
ax.set_ylim(0.0, 1.0)

fig.add_subplot(1, 2, 2)
plt.xlabel('Recall')
plt.ylabel('Precision')
for i, roc_curve in enumerate(roc_curves):
    if len(model_names) > 0:
        plt.plot(*pr_curves[i], label=model_names[i])
    else:
        plt.plot(*pr_curves[i])
plt.legend()
plt.tight_layout(pad=0)
# plt.axis('off')
ax = plt.gca()
ax.set_xlim(0.0, 1)
ax.set_ylim(0.0, 1.0)
plt.show()

evaluation_output = args.evaluation_output
if evaluation_output:
    print("save to %s" % output_file)
    dir_name, file_name = os.path.split(evaluation_output)
    if not os.path.isdir(dir_name):
        os.system("mkdir -p " + dir_name)
        print("No directory named %s : create directory" % dir_name)
    evaluation_df.to_csv(evaluation_output)
