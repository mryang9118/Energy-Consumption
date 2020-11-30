"""
@Project: Energy-Consumption   
@Description: visualize the evaluation result
@Time:2020/10/29 16:37                      
 
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_curve(model_name, sizes, training_scores, testing_scores):
    # Mean and Standard Deviation of training scores
    train_scores_mean = np.mean(training_scores, axis=1)
    train_scores_std = np.std(training_scores, axis=1)

    # Mean and Standard Deviation of testing scores
    test_scores_mean = np.mean(testing_scores, axis=1)
    test_scores_std = np.std(testing_scores, axis=1)

    # box-like grid
    plt.grid()
    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    # dotted red line is for training scores and green line is for cross-validation score
    plt.plot(sizes, train_scores_mean, linestyle='--', linewidth=2, marker='o', markersize=5, color="r",
             label="Training score")
    plt.plot(sizes, test_scores_mean, linewidth=2, marker='s', markersize=5, color="g", label="Cross-validation score")

    # Drawing plot
    plt.title("Learning Curve For %s" % model_name)
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_importance, feature_names):
    for i, v in enumerate(feature_importance):
        print('Feature: %s, Score: %.5f' % (feature_names[i], v))
    plt.title('Feature Importance')
    plt.xlabel('Feature Name')
    plt.ylabel('Proportion')
    plt.grid()
    plt.bar([x for x in range(len(feature_importance))], feature_importance)
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.tight_layout()
    plt.show()