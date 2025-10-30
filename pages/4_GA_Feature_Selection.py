import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.ga import ga_feature_select
from src.preprocess import split_cols
from src.metrics import summary
from src.io_utils import load_cfg
from src.ui import inject_watermark
from pathlib import Path

inject_watermark(
    Path("logo") / "logo.svg",
    size="100vmin", opacity=0.08,
    position="center 55%"
)
import plotly.express as px

st.title("Genetic Algorithm Feature Selection")
cfg = load_cfg()
df = st.session_state.get("df"); target = st.session_state.get("target")
pp = st.session_state.get("preprocess")
if df is None or target is None or pp is None: st.stop()

X, y, *_ = split_cols(df, target)
Xt = pp["ct"].transform(X)
feat_names = pp["feat_names"]

# GA params
col1, col2, col3 = st.columns(3)
with col1:
    pop = st.number_input("Population", 10, 200, int(cfg["ga"]["population_size"]))
    gens = st.number_input("Generations", 5, 200, int(cfg["ga"]["generations"]))
with col2:
    pc = st.slider("Crossover prob", 0.0, 1.0, float(cfg["ga"]["crossover_prob"]))
    pm = st.slider("Mutation prob", 0.0, 1.0, float(cfg["ga"]["mutation_prob"]))
with col3:
    k = st.number_input("Tournament k", 2, 10, int(cfg["ga"]["tournament_k"]))
    elit = st.number_input("Elitism", 0, 5, int(cfg["ga"]["elitism"]))
cap = st.number_input("Max features (0 = no cap)", 0, Xt.shape[1], 0)

# New: fitness metric selector
metric = st.selectbox("Fitness metric", options=["accuracy", "f1", "roc_auc"], index=0)

# If metric needs a positive class, let user choose
pos_label = None
if metric in ("f1", "roc_auc"):
    classes = np.unique(y)
    # choose the second class by default if binary
    default_idx = 1 if classes.size > 1 else 0
    pos_label = st.selectbox("Positive class for fitness", options=list(classes), index=default_idx)

Xtr, Xte, ytr, yte = train_test_split(Xt, y,
                                      test_size=cfg["data"]["test_size"],
                                      stratify=y,
                                      random_state=cfg["data"]["random_state"]
                                      )

def build_clf():
    return LogisticRegression(C=cfg["model"]["C"], max_iter=cfg["model"]["max_iter"])

ga_cfg = dict(
    population_size=pop, generations=gens, tournament_k=k,
    crossover_prob=pc, mutation_prob=pm, elitism=elit,
    max_features=(None if cap==0 else int(cap)),
    random_state=cfg["data"]["random_state"],
    metric=metric, pos_label=pos_label
)

if st.button("Run GA"):
    mask, best_score, _ = ga_feature_select(Xtr, ytr, Xte, yte, build_clf, ga_cfg)
    st.session_state["ga_mask"] = mask
    st.success(f"Best {metric}: {best_score:.4f}")
    chosen = np.array(feat_names)[mask]
    st.write(f"Selected {mask.sum()} / {Xt.shape[1]} features")
    st.dataframe({"feature": chosen})

if "ga_mask" in st.session_state:
    mask = st.session_state["ga_mask"]
    clf = build_clf().fit(Xtr[:, mask], ytr)
    proba = clf.predict_proba(Xte[:, mask])
    classes = clf.classes_
    pos_label = st.selectbox("Positive class (GA)", options=list(classes), index=1 if len(classes) > 1 else 0, key="ga_pos_label")
    pos_idx = list(classes).index(pos_label)

    y_prob = proba[:, pos_idx]
    y_pred_lbl = clf.predict(Xte[:, mask])
    y_true_lbl = yte
    y_true_bin = (y_true_lbl == pos_label).astype(int)
    y_pred_bin = (y_pred_lbl == pos_label).astype(int)

    st.subheader("GA-selected model performance")
    st.table({k: [v] for k, v in summary(y_true_bin, y_prob, y_pred_bin).items()})

