from sklearn.metrics import accuracy_score, f1_score

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy using sklearn's accuracy_score for consistency.
    """
    return accuracy_score(y_true, y_pred)

def calculate_f1_score(y_true, y_pred):
    """
    Calculate the weighted F1 score using sklearn's f1_score.
    """
    return f1_score(y_true, y_pred, average='weighted')