import numpy as np
from sklearn.metrics import accuracy_score, f1_score,roc_auc_score

def _fitness(clf, Xtr, ytr, Xte, yte, metric: str, pos_label=None):
    if metric == "accuracy":
        clf.fit(Xtr, ytr)
        return clf.score(Xte, yte)
    elif metric == "f1":
        if pos_label is None:
            return 0.0
        clf.fit(Xtr, ytr)
        y_pred = clf.predict(Xte)
        y_true_bin = (yte == pos_label).astype(int)
        y_pred_bin = (y_pred == pos_label).astype(int)
        return f1_score(y_true_bin, y_pred_bin, zero_division=0)
    elif metric == "roc_auc":
        if pos_label is None:
            return 0.0
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        classes = list(clf.classes_)
        if pos_label not in classes or len(classes) != 2:
            return 0.0
        pos_idx = classes.index(pos_label)
        y_prob = proba[:, pos_idx]
        y_true_bin = (yte == pos_label).astype(int)
        try:
            return roc_auc_score(y_true_bin, y_prob)
        except Exception:
            return 0.0
    else:
        return 0.0
def evaluate_mask(mask, Xtr, ytr, Xte, yte, build_clf, metric="accuracy", pos_label=None):
    if mask.sum() == 0:
        return 0.0
    clf = build_clf()
    return _fitness(clf, Xtr[:, mask], ytr, Xte[:, mask], yte, metric, pos_label)

def ga_feature_select(Xtr, ytr, Xte, yte, build_clf, cfg):
    rng = np.random.default_rng(cfg["random_state"])
    n_feat = Xtr.shape[1]
    max_features = cfg.get("max_features")
    metric = cfg.get("metric", "accuracy")
    pos_label = cfg.get("pos_label", None)

    pop_size = cfg["population_size"]; gens = cfg["generations"]
    pc = cfg["crossover_prob"]; pm = cfg["mutation_prob"]; k = cfg["tournament_k"]; elit = cfg["elitism"]

    def init_ind():
        mask = rng.random(n_feat) < 0.5
        if max_features:
            idx = np.where(mask)[0]
            if idx.size > max_features:
                keep = rng.choice(idx, size=max_features, replace=False)
                mask = np.zeros(n_feat, dtype=bool); mask[keep] = True
        if mask.sum() == 0:
            mask[rng.integers(0, n_feat)] = True
        return mask

    pop = [init_ind() for _ in range(pop_size)]
    scores = np.array([evaluate_mask(m, Xtr, ytr, Xte, yte, build_clf, metric, pos_label) for m in pop])

    for _ in range(gens):
        new_pop = [pop[int(scores.argmax())].copy() for _ in range(elit)]

        def select():
            idx = rng.integers(0, pop_size, size=k)
            return pop[idx[scores[idx].argmax()]].copy()

        while len(new_pop) < pop_size:
            p1, p2 = select(), select()
            do_cx = (n_feat>1) and (rng.random() < pc)
            if do_cx:
                cx = rng.integers(1, n_feat)
                c1 = np.concatenate([p1[:cx], p2[cx:]])
                c2 = np.concatenate([p2[:cx], p1[cx:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in (c1, c2):
                mut_mask = rng.random(n_feat) < pm
                c[mut_mask] = ~c[mut_mask]
                if max_features and c.sum() > max_features:
                    ones = np.where(c)[0]
                    drop = rng.choice(ones, size=c.sum()-max_features, replace=False)
                    c[drop] = False
                if c.sum() == 0:
                    c[rng.integers(0, n_feat)] = True
            new_pop += [c1, c2]
        pop = new_pop[:pop_size]
        scores = np.array([evaluate_mask(m, Xtr, ytr, Xte, yte, build_clf, metric, pos_label) for m in pop])

    best_idx = int(scores.argmax())
    return pop[best_idx], float(scores[best_idx]), scores
