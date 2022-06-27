from sklearn import metrics
import pandas as pd

def nice_conf_mat(y_true, y_pred, row_dict=None, column_dict=None):
    if row_dict==None:
        row_dict = {0:'True Negative', 1:'True Positive'}
    if column_dict==None:
        column_dict = {0:'Predicted Negative', 1:'Predicted Positive'}
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm = pd.DataFrame(cm).rename(row_dict).rename(column_dict, axis=1)
    display(cm)
    return cm