from patlib import Path

def is_truly_positive(label_path):
    """
    Сhecks if the labels file contains real data and not just spaces.
    """
    if not label_path.exists():
        return False
    with open(label_path, 'r') as f:
        for line in f:
            if line.strip():
                return True
    return False
