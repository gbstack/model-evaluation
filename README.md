`python evaluate.py pred.csv -N`

Evaluate prediction file _pred.csv_, it should contain at least two columns, one ground truth lael column with header "label", another predicted result column with header "predicted". `-N` option means no threshold is given to calculate metrics, in this case AUC and AUPR will be generated, ROC curve and PR curve will be plotted.

    Evaluation of the test set
        Sen :  0.03664921465968586
        Spe :  1.0
        Acc :  0.5195822454308094
        Precision :  1.0
        F1 :  0.07070707070707069
        Area Under ROC Curve(AUC): 0.924
        Area Under PR Curve(AUPR): 0.939
    =================================================
 