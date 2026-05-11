import numpy as np

def make_text_labels(labels, classes):
    # LABEL = ['Go-there', 'Distract-target', 'Follow-me', 'Explore-here', 'Terminate']
    LABEL = classes
    text_labels = np.array([LABEL[i] for i in labels])
    return text_labels


def pad_trial(x: np.ndarray, target_len: int):
    """Zero-pad / truncate to target_len (last dimension)."""
    if x.shape[-1] >= target_len:
        return x[..., :target_len]
    pad_w = target_len - x.shape[-1]
    return np.pad(x, ((0, 0), (0, 0), (0, pad_w)), mode="constant")