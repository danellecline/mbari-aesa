#!/usr/bin/env python
__author__    = 'Danelle Cline'
__copyright__ = '2016'
__license__   = 'GPL v3'
__contact__   = 'dcline at mbari.org'
__doc__ = '''
Utility class to run and plot various classifier performance plots
appended with -new
@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: production
@license: GPL
'''

import conf
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from collections import namedtuple
from sklearn.learning_curve import learning_curve
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics

from sklearn import svm
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


Results = namedtuple("Results", ["y_score", "y_pred", "y_test", "f_score", "precision", "accuracy"])
t = np.arange(0.0, 1.0, 0.1)
s = np.sin(2*np.pi*t)
linestyles = ['-', ':', '*', '--']
markers = []
for m in Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass

styles = markers + [
    r'$\lambda$',
    r'$\bowtie$',
    r'$\circlearrowleft$',
    r'$\clubsuit$',
    r'$\checkmark$']

colors = ('b', 'c', 'm', 'y', 'k', 'b', 'r',)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def subplot_learning_curve(title, train_scores, test_scores, train_sizes, ylim=None):
    """
    Generate a simple plot of the test and training learning curve. Assumes this is a
    subplot
    """
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score");
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score");

    plt.legend(loc="lower left")

def learning_cv(title, X, y, cv, clf_class, train_sizes, **kwargs):
    """
    Helper function to run generic estimator and generate subplot of the learning curve.
    """
    lim = (0.70, 1.01)
    # choose the estimator
    estimator = clf_class(**kwargs)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator, X=X, y=y, cv=cv, train_sizes=train_sizes, n_jobs=8)
    subplot_learning_curve(title, train_scores, test_scores, train_sizes=train_sizes, ylim=lim)

def run_cv(n_classes, cv, X, y, clf_class, **kwargs):
    y_pred = y.copy()
    f_score = []
    precision = []
    accuracy = []

    for ii, jj in cv:
        y_train, y_test = y[ii], y[jj]
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[jj] = clf.predict(X_test)

        f_score.append(f1_score(y_test, y_pred[jj], average="binary"))
        precision.append(precision_score(y_test, y_pred[jj], average="binary"))
        accuracy.append(recall_score(y_test, y_pred[jj], average="binary"))

    clf = OneVsRestClassifier(clf_class(**kwargs))
    y_b = label_binarize(y, classes=range(n_classes))
    X_train, X_test, y_train, y_test = train_test_split(X, y_b, test_size=.2,
                                                    random_state=True)
    try:
        y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    except Exception as ex:
        y_score = np.ones(X_test.shape)
        print ex

    return Results(y_score=y_score, y_pred=y_pred, y_test=y_test, f_score=f_score, precision=precision, accuracy=accuracy)

def run_cm_roc(n_classes, cv, X, y, clf_class, **kwargs):
    r = run_cv(n_classes, cv, X, y, clf_class, **kwargs)
    cm = metrics.confusion_matrix(y, r.y_pred)
    return cm, r

def plot_roc(ax, results, n_classes, class_name):
    # compute Precision-Recall per each class and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(results.y_test, results.y_score[:, i], pos_label=i)
        y_true = (results.y_test == i)
        average_precision[i] = average_precision_score(y_true, results.y_score[:, i])
        key = class_name[i]
        if conf.class_dict.has_key(key):
            name = conf.class_dict[key]
        else:
            name = i

        color = colors[i % len(colors)]
        if i < len(linestyles):
            ax.plot(recall[i], precision[i], linestyles[i], color=color, markersize=10,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(name, average_precision[i]));
        else:
            style = styles[(i - len(linestyles)) % len(styles)]
            ax.plot(recall[i], precision[i], linestyle='None', marker=style, color=color, markersize=10,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(name, average_precision[i]));



    avg_f_score = np.average(results.f_score)
    avg_precision = np.average(results.precision)
    avg_accuracy = np.average(results.accuracy)
    annotation = '(averages) f_score:{0:0.2f} precision:{1:0.2f} accuracy:{2:0.2f}'\
                    .format(avg_f_score, avg_precision, avg_accuracy)

    ax.annotate(annotation, xy=(1, 0), xycoords='axes fraction', horizontalalignment='right',
                verticalalignment='bottom', fontsize=10)

    ax.legend(loc="lower left")

def plot_training_matrix(x, y, cv, append_title, train_sizes):
    fix, ax = plt.subplots(figsize=(16, 12))
    plt.suptitle('Learning Curve of Various Classifiers ' + append_title);
    plt.subplot(2, 2, 1)
    gamma = 0.0
    C = 1
    kernel = 'linear'
    svm_title = 'SVM, %s, C=%.6f $\gamma=%.6f$)' % (kernel, C, gamma)
    print 'SVM'
    learning_cv(svm_title, x, y, cv, svm.SVC, train_sizes=train_sizes, probability=True, C=C, gamma=gamma)
    '''C=C, cache_size=200, class_weight=None,
                coef0=0.0, degree=3, gamma=gamma, kernel=kernel, max_iter=-1, probability=True, random_state=None,
                shrinking=True, tol=0.001, verbose=False)'''
    plt.subplot(2, 2, 2);
    print 'Decision Tree'
    learning_cv('Decision Tree', x, y, cv, tree.DecisionTreeClassifier, train_sizes=train_sizes)
    plt.subplot(2, 2, 3);
    print 'Random Forest'
    learning_cv('Random Forest', x, y, cv, ensemble.RandomForestClassifier, train_sizes=train_sizes)
    plt.subplot(2, 2, 4);
    print 'Gradient Boosting'
    learning_cv('Gradient Boosting', x, y, cv, ensemble.GradientBoostingClassifier, train_sizes=train_sizes)
    return plt


def plot_confusion_matrix(X, y, cv, append_title, labels):
    n_classes = len(labels)
    gamma = 0.0
    C = 1
    kernel = 'linear'
    svm_title = 'SVM, %s, C=%.6f $\gamma=%.6f$)' % (kernel, C, gamma)
    print 'SVM'
    svm_cm, svm_results = run_cm_roc(n_classes, cv, X, y, svm.SVC, probability=True, C=C, gamma=gamma)
    ''' cache_size=200, class_weight=None,
                coef0=0.0, degree=3, gamma=gamma, kernel=kernel, max_iter=-1, probability=True, random_state=None,
                shrinking=True, tol=0.001, verbose=False)'''
    print 'Decision Tree'
    decision_cm, decision_results = run_cm_roc(n_classes, cv, X, y, tree.DecisionTreeClassifier)
    print 'Random Forest'
    random_forest_cm, random_forest_results = run_cm_roc(n_classes, cv, X, y, ensemble.RandomForestClassifier)
    print 'Gradient Boosting'
    grad_reg_cm, grad_reg_results = run_cm_roc(n_classes, cv, X, y, ensemble.GradientBoostingClassifier)

    cm = {
        1: {
            'cm': svm_cm,
            'results': svm_results,
            'title': svm_title,
           },
        2: {
            'cm': decision_cm,
            'results': decision_results,
            'title': 'Decision Tree',
           },
        3: {
            'cm': random_forest_cm,
            'results': random_forest_results,
            'title': 'Random Forest',
           },
        4: {
            'cm': grad_reg_cm,
            'results': grad_reg_results,
            'title': 'Gradient Boosting',
           }
    }

    fig = plt.figure(figsize=(16, 12));
    fig.suptitle('Confusion Matrix and Precision/Recall Curves of Various Classifiers, ' + append_title)
    sns.set()

    with sns.color_palette("husl", 100):
        gs = gridspec.GridSpec(4,2)
        sns.set()

        for ii, values in cm.items():
            results = values['results']
            matrix = values['cm']
            title = values['title']
            ax = fig.add_subplot(gs[(ii - 1)*2])
            ax.set_title(title)
            print matrix
            sns.heatmap(matrix, ax=ax, annot=True, fmt='d', linewidths=.5, yticklabels=labels, xticklabels=labels)
            ax = fig.add_subplot(gs[ii*2 - 1])
            ax.set_title(title)
            plot_roc(ax, results, n_classes=n_classes, class_name=labels)

    return plt