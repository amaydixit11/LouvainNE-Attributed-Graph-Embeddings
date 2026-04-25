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
        # Empty array check
        if not np.any(mask):
            return {"accuracy": 0.0, "f1_macro": 0.0, "f1_micro": 0.0}
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {"accuracy": 0.0, "f1_macro": 0.0, "f1_micro": 0.0}
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro
    }

def calibration_analysis(preds, labels, confidences, mask=None):
    """
    Checks if confidence correlates with error.
    Returns accuracy per confidence bucket.
    """
    if mask is not None:
        preds = preds[mask]
        labels = labels[mask]
        confidences = confidences[mask]
        
    correct = (preds == labels).astype(float)
    
    # 1. Pearson Correlation
    from scipy.stats import pearsonr
    corr = pearsonr(confidences, correct)[0] if len(correct) > 1 else 0
    
    # 2. Bucketed Accuracy (Calibration)
    # 0.0-0.2, 0.2-0.4, ...
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_accs = []
    for i in range(len(bins)-1):
        idx = (confidences >= bins[i]) & (confidences < bins[i+1])
        if np.any(idx):
            bin_acc = np.mean(correct[idx])
            bin_accs.append(bin_acc)
        else:
            bin_accs.append(0.0)
            
    return {
        "correlation": float(corr),
        "bin_accuracies": bin_accs,
        "bins": bins
    }
