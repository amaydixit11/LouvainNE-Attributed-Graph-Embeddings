from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def evaluate(y_true, y_pred, mask=None):
    """
    Computes performance metrics.
    """
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()
        
    if mask is not None:
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro
    }
