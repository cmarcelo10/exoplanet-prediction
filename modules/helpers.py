import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score

def format_cv_results(cv_scores_df, model_name, column_mapper):
    # Generate a dataframe to show the results in one line -
    # Take the mean of each metric, swap columns with rows and drop the "fit time" and "score time" columns
    cv_scores_df = pd.DataFrame(cv_scores_df).drop(index=[0, 1]).mean(axis=0)
    out = pd.DataFrame(cv_scores_df, columns=[model_name]).transpose().rename(axis=1, mapper=column_mapper)
    return out

def print_model_params(params_dict):
    for key, value in params_dict.items():
        print(f"{key}: {value}")

def plot_feature_importances(importances, title='Feature Importances'):
    importances = importances.mean(axis=1).sort_values()
    plt.figure(figsize=(15, 8))
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    ax = sns.barplot(x=importances.values, y=importances.index, orient='h', width=0.6)
    for i, v in enumerate(importances.values):
        ax.text(v, i, f"{v:.3g}", va='center', ha='left', fontsize=9, color='black')
    return ax

def get_scores(y_true, y_pred):
    a = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return {'Accuracy':a, 'Precision': p, 'Recall': r}

def assess_model(estimator, X_train, y_train, X_test, y_test):
    y_pred_train = estimator.predict(X_train)
    y_pred_test = estimator.predict(X_test)
    train_scores = get_scores(y_train, y_pred_train)
    test_scores = get_scores(y_test, y_pred_test)
    score_types = ['Accuracy', 'Precision', 'Recall']
    model_scores = {}
    for metric in score_types:
        model_scores[f'Train {metric}'] = train_scores[metric]
        model_scores[f'Test {metric}'] = test_scores[metric]
    
    return model_scores, y_pred_test

