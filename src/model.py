from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_eval_logreg(X, y, feature_mask=None, C=1.0, max_iter=200, random_state=42, test_size=0.2):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    if feature_mask is not None:
        Xtr = Xtr[:, feature_mask]
        Xte = Xte[:, feature_mask]
    clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=None)
    clf.fit(Xtr, ytr)
    return clf, (Xtr, Xte, ytr, yte)
