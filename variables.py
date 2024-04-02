labels = ["Pleasantness", \
            "Anticipated Effort", \
            "Attentional Activity", \
            "Certainty", \
            "Objective Experience", \
            "Self-Other Agency", \
            "Situational Control", \
            "Advice", \
            "Trope"]
metrics = ["total_accuracy", \
            "total_recall", \
            "total_precision", \
            "per_label_recall", \
            "per_label_precision", \
            "macro_f1", \
            "macro_recall", \
            "macro_precision", \
            "f1_score", \
            "auroc_score"]
loss_funcs = [
    'simple_cross_entropy_loss', \
    'penalize_label_loss', \
    'penalize_length_loss', \
    'bce_with_logits_loss', \
    'contrastive_loss', \
    'mse_loss', \
    'binary_cross_entropy_loss'
]