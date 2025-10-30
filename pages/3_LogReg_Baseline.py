import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.metrics import summary, curves
from src.preprocess import split_cols
from src.io_utils import load_cfg
from src.ui import inject_watermark
from pathlib import Path

inject_watermark(
    Path("logo") / "logo.svg",
                  size="100vmin", opacity=0.08,
                  position="center 55%"
                  )


st.title("Baseline Logistic Regression")
cfg = load_cfg()
df = st.session_state.get("df"); target = st.session_state.get("target")
pp = st.session_state.get("preprocess")
if df is None or target is None or pp is None: st.stop()

X, y, *_ = split_cols(df, target)
ct = pp["ct"]; feat_names = pp["feat_names"]
Xt = ct.transform(X)
Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=cfg["data"]["random_state"])
C = st.slider("C (inverse regularization)", 0.01, 10.0, float(cfg["model"]["C"]))
clf = LogisticRegression(C=C, max_iter=cfg["model"]["max_iter"])
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)
classes = clf.classes_
pos_label = st.selectbox("Positive class", options=list(classes), index=1 if len(classes) > 1 else 0)
pos_idx = list(classes).index(pos_label)

y_prob = proba[:, pos_idx]                  # prob of pos_label
y_pred_lbl = clf.predict(Xte)               # label predictions aligned with y types
y_true_lbl = yte

# binarize for binary metrics
y_true_bin = (y_true_lbl == pos_label).astype(int)
y_pred_bin = (y_pred_lbl == pos_label).astype(int)


st.subheader("Performance")
st.table({k: [v] for k, v in summary(y_true_bin, y_prob, y_pred_bin).items()})
import plotly.express as px

f = curves(y_true_bin, y_prob)
fpr, tpr = f["roc"]
cm = f["cm"]

roc_fig = px.line(x=fpr, y=tpr, labels={"x": "FPR", "y": "TPR"}, title="ROC Curve")
st.plotly_chart(roc_fig, use_container_width=True)

cm_fig = px.imshow(cm, text_auto=True, labels=dict(x="Pred", y="True", color="Count"),
                   title="Confusion Matrix")
st.plotly_chart(cm_fig, use_container_width=True)

