import torch
from cleanlab.filter import find_label_issues
import numpy as np
def get_masks(ema_output, weak_masks, p_maps):
    return weak_masks

def confinence_learning(labels, pred_probs):
    ordered_label_issues = find_label_issues(
    labels=labels,
    pred_probs=pred_probs, # out-of-sample predicted probabilities from any model
    return_indices_ranked_by='self_confidence',
    )
    return ordered_label_issues
    
def update_errormaps(error_maps, probaility_maps):
    pass
def label_refinement(error_maps, pred_probs):
    pass