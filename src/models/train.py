"""Train regression, classification, and clustering models. Saves artifacts under artifacts/models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    CV_FOLDS,
    FIGURE_DIR,
    MODEL_DIR,
    N_ITER_SEARCH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.data.load import load_raw_dataset  # noqa: E402
from src.data.preprocess import build_preprocessor, prepare_modelling_frame, split_xy  # noqa: E402
from src.data.validate import validate_raw_frame  # noqa: E402
from src.models.evaluate import classification_metrics, regression_metrics, save_json  # noqa: E402
from src.models.explain import permutation_table, shap_global_and_local  # noqa: E402
from src.models.classifiers import EncodedLabelClassifier  # noqa: E402


CLASS_LABELS = ["high_support", "medium_support", "low_support"]


def _maybe_xgb_regressor():
    try:
        from xgboost import XGBRegressor

        return XGBRegressor(
            random_state=RANDOM_STATE,
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=1,
        )
    except Exception:
        return None


def _maybe_xgb_classifier():
    try:
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=1,
        )
        return EncodedLabelClassifier(estimator)
    except Exception:
        return None


def _optional_boosting_models():
    """Return optional LightGBM/CatBoost estimators when their packages are installed."""
    models = {}
    try:
        from lightgbm import LGBMRegressor
        models["lightgbm"] = LGBMRegressor(n_estimators=120, verbosity=-1, random_state=RANDOM_STATE, n_jobs=1)
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor
        models["catboost"] = CatBoostRegressor(iterations=120, verbose=False, random_seed=RANDOM_STATE, thread_count=1)
    except Exception:
        pass
    return models


def _pipe(model) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def _search(pipeline, param_dist, X, y, scoring):
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        cv=CV_FOLDS,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=0,
    )
    search.fit(X, y)
    return search


def train() -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_dataset()
    quality = validate_raw_frame(raw)
    prepared = prepare_modelling_frame(raw)
    processed_path = ROOT / "data" / "processed" / "student_prepared.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(processed_path, index=False)

    X, y_reg, y_cls = split_xy(prepared)
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
        X,
        y_reg,
        y_cls,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_cls,
    )

    # --- Regression ---
    dummy_reg = DummyRegressor(strategy="mean")
    dummy_reg.fit(X_train, y_reg_train)
    dummy_reg_pred = dummy_reg.predict(X_test)

    xgb_reg = _maybe_xgb_regressor()
    regressors = {
        "linear_regression": _pipe(LinearRegression()),
        "decision_tree": _pipe(DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=6)),
        "random_forest": _pipe(RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=160, max_depth=8)),
        "gradient_boosting": _pipe(GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=120, max_depth=3)),
        "svr": _pipe(SVR(C=2.0, epsilon=0.4, kernel="rbf")),
    }
    if xgb_reg is not None:
        regressors["xgboost"] = _pipe(xgb_reg)
    for name, estimator in _optional_boosting_models().items():
        regressors[name] = _pipe(estimator)

    rf_search = _search(
        _pipe(RandomForestRegressor(random_state=RANDOM_STATE)),
        {
            "model__n_estimators": [80, 120, 180],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 2, 4],
        },
        X_train,
        y_reg_train,
        scoring="neg_root_mean_squared_error",
    )
    regressors["random_forest_tuned"] = rf_search.best_estimator_

    reg_scores = {"baseline_mean": regression_metrics(y_reg_test, dummy_reg_pred)}
    best_reg_name, best_reg, best_reg_rmse = None, None, 1e9
    reg_cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for name, model in regressors.items():
        if name != "random_forest_tuned":
            model.fit(X_train, y_reg_train)
        pred = model.predict(X_test)
        metrics = regression_metrics(y_reg_test, pred)
        cv_rmse = -cross_val_score(model, X_train, y_reg_train, cv=reg_cv,
                                   scoring="neg_root_mean_squared_error", n_jobs=1).mean()
        metrics["cv_rmse"] = float(cv_rmse)
        residuals = (y_reg_test - pred).astype(float)
        metrics["residual_mean"] = float(residuals.mean())
        metrics["residual_std"] = float(residuals.std())
        reg_scores[name] = metrics
        if cv_rmse < best_reg_rmse:
            best_reg_name, best_reg, best_reg_rmse = name, model, cv_rmse

    # --- Classification ---
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_cls_train)
    dummy_pred = dummy_clf.predict(X_test)

    xgb_clf = _maybe_xgb_classifier()
    classifiers = {
        "logistic_regression": _pipe(
            LogisticRegression(max_iter=400, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "decision_tree": _pipe(DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6, class_weight="balanced")),
        "random_forest": _pipe(
            RandomForestClassifier(
                random_state=RANDOM_STATE, n_estimators=160, max_depth=8, class_weight="balanced"
            )
        ),
        "gradient_boosting": _pipe(GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=120, max_depth=3)),
        "knn": _pipe(KNeighborsClassifier(n_neighbors=7)),
        "svc": _pipe(CalibratedClassifierCV(
            estimator=SVC(class_weight="balanced", random_state=RANDOM_STATE, C=1.5),
            method="sigmoid",
            cv=3,
            ensemble=False,
        )),
    }
    if xgb_clf is not None:
        classifiers["xgboost"] = _pipe(xgb_clf)
    try:
        from lightgbm import LGBMClassifier
        classifiers["lightgbm"] = _pipe(LGBMClassifier(n_estimators=120, verbosity=-1, random_state=RANDOM_STATE, n_jobs=1, class_weight="balanced"))
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        classifiers["catboost"] = _pipe(CatBoostClassifier(iterations=120, verbose=False, random_seed=RANDOM_STATE, thread_count=1, auto_class_weights="Balanced"))
    except Exception:
        pass

    gb_search = _search(
        _pipe(GradientBoostingClassifier(random_state=RANDOM_STATE)),
        {
            "model__n_estimators": [80, 120, 160],
            "model__max_depth": [2, 3, 4],
            "model__learning_rate": [0.05, 0.08, 0.12],
        },
        X_train,
        y_cls_train,
        scoring="f1_macro",
    )
    classifiers["gradient_boosting_tuned"] = gb_search.best_estimator_

    cls_scores = {
        "baseline_majority": classification_metrics(y_cls_test, dummy_pred, labels=CLASS_LABELS)
    }
    class_counts = y_cls_train.value_counts().to_dict()
    best_cls_name, best_cls, best_f1 = None, None, -1
    cls_cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    for name, model in classifiers.items():
        if name != "gradient_boosting_tuned":
            model.fit(X_train, y_cls_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        metrics = classification_metrics(y_cls_test, pred, y_proba=proba, labels=CLASS_LABELS)
        cv_f1 = cross_val_score(model, X_train, y_cls_train, cv=cls_cv,
                                scoring="f1_macro", n_jobs=1).mean()
        metrics["cv_f1_macro"] = float(cv_f1)
        cls_scores[name] = metrics
        if cv_f1 > best_f1:
            best_cls_name, best_cls, best_f1 = name, model, cv_f1

    # --- Clustering on train features only, never using G3 ---
    cluster_pre = build_preprocessor()
    X_train_trans = cluster_pre.fit_transform(X_train)
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_trans)
    cluster_profile = (
        pd.DataFrame(X_train.assign(cluster=train_clusters))
        .groupby("cluster")[["studytime", "absences", "failures", "engagement_score", "attendance_proxy"]]
        .mean()
        .round(3)
        .to_dict()
    )

    # Residual analysis and confusion matrix (assignment evaluation evidence)
    best_reg_pred = best_reg.predict(X_test)
    residuals = y_reg_test.to_numpy(dtype=float) - best_reg_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(best_reg_pred, residuals, alpha=0.55, c="#2c6e49")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Predicted G3")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title(f"Residuals — {best_reg_name}")
    axes[1].hist(residuals, bins=20, color="#bc4749", edgecolor="white")
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Residual")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "11_regression_residuals.png", dpi=140)
    plt.close(fig)

    best_cls_pred = best_cls.predict(X_test)
    cm = confusion_matrix(y_cls_test, best_cls_pred, labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {best_cls_name}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "12_confusion_matrix.png", dpi=140)
    plt.close(fig)

    # --- Robustness: missing studytime on a copy of the test set ---
    X_perturbed = X_test.copy()
    X_perturbed["studytime"] = np.nan
    robust_pred = best_reg.predict(X_perturbed)
    robustness = regression_metrics(y_reg_test, robust_pred)

    perm_reg = permutation_table(best_reg, X_test, y_reg_test, list(X_test.columns), scoring="r2")
    perm_cls = permutation_table(best_cls, X_test, y_cls_test, list(X_test.columns), scoring="f1_macro")
    shap_pack = shap_global_and_local(best_reg, X_test.head(40), n_local=3)

    X_train.sample(n=min(40, len(X_train)), random_state=RANDOM_STATE).to_csv(
        MODEL_DIR / "explanation_background.csv", index=False
    )

    joblib.dump(best_reg, MODEL_DIR / "regression_pipeline.joblib")
    joblib.dump(best_cls, MODEL_DIR / "classification_pipeline.joblib")
    joblib.dump(kmeans, MODEL_DIR / "kmeans.joblib")
    joblib.dump(cluster_pre, MODEL_DIR / "cluster_preprocessor.joblib")
    joblib.dump(dummy_reg, MODEL_DIR / "baseline_regressor.joblib")

    feature_meta = {"features": list(X.columns), "class_labels": CLASS_LABELS}
    (MODEL_DIR / "feature_meta.json").write_text(json.dumps(feature_meta, indent=2), encoding="utf-8")

    summary = {
        "data_quality": quality,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_balance_train": {k: int(v) for k, v in class_counts.items()},
        "class_balance_test": {k: int(v) for k, v in y_cls_test.value_counts().to_dict().items()},
        "leakage_excluded": ["G1", "G2", "sex"],
        "best_regressor": best_reg_name,
        "best_classifier": best_cls_name,
        "regression": reg_scores,
        "classification": cls_scores,
        "tuning": {
            "random_forest_best_params": rf_search.best_params_,
            "gradient_boosting_best_params": gb_search.best_params_,
        },
        "clustering": cluster_profile,
        "robustness_missing_studytime": robustness,
        "permutation_importance_regression": perm_reg.head(15).to_dict(orient="records"),
        "permutation_importance_classification": perm_cls.head(15).to_dict(orient="records"),
        "shap": shap_pack,
        "selection_rationale": (
            f"The deployed regressor is {best_reg_name} because it had the lowest training-fold cross-validation RMSE "
            f"({best_reg_rmse:.3f}) among models that never saw G1/G2. The deployed classifier is "
            f"{best_cls_name} because it had the highest training-fold cross-validation macro-F1 ({best_f1:.3f}), which treats "
            "the smaller high-support group as equally important as the majority medium group."
        ),
    }
    save_json(MODEL_DIR / "metrics.json", summary)
    perm_reg.to_csv(MODEL_DIR / "perm_importance_regression.csv", index=False)
    perm_cls.to_csv(MODEL_DIR / "perm_importance_classification.csv", index=False)
    print(json.dumps({"best_regressor": best_reg_name, "best_classifier": best_cls_name}, indent=2))
    print("Saved artifacts to", MODEL_DIR)
    return summary


if __name__ == "__main__":
    train()
