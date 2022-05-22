import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from sklearn import svm 
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

def heatmap(c, cols, sx =5, sy =5 ):    
    plt.figure(figsize =(sx, sy))
    plt.imshow(c)
    plt.xticks(np.arange(len(cols)), labels=cols, rotation=45, ha="right",rotation_mode="anchor")
    plt.yticks(np.arange(len(cols)), labels=cols)

    # Loop over data dimensions and create text annotations.
    for i in range(len(cols)):
        for j in range(len(cols)):
            text = plt.text(j, i, round(c[i, j], 2),
                           ha="center", va="center", color="w")


    return plt.show()

def plot_learning_curves(model, X_train, y_train, X_val, y_val):
    from sklearn.metrics import mean_squared_error
    """
    input:
        model:pipeline object
        X_train, y_train: training data
        X_val, y_val: test data
    """
    train_errors, val_errors = [], []
    for m in range(2, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="training data")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="validation data")
    plt.legend(loc="upper right", fontsize=14)   
    plt.xlabel("Training set size", fontsize=14) 
    plt.ylabel("RMSE", fontsize=14)     
    return 0

def ROC_curve(X, y, classifier, variaties):


    # Binarize the output
    y_bin = label_binarize(y, classes=[1, 2, 3, 4,5 ,6 ,7])
    n_classes = y_bin.shape[1]
    
    # shuffle and split training and test sets
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.3)
    
    # Learn to predict each class against the other
    clf = OneVsRestClassifier( classifier)
    
    y_score_bin = clf.fit(X_train_bin, y_train_bin).decision_function(X_test_bin)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    lw =2
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i  in range(n_classes):

        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize = (10, 5))


    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", 'blue', 'green', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(variaties[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve One vs Rest")
    plt.legend(loc="lower right")
    plt.show()


def normalize(X):
    # Normalize the data
    centered = X - X.mean(axis = 0)
    zscores = centered / centered.std(axis=0)
    return zscores