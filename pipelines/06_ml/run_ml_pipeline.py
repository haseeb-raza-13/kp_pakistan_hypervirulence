"""
ML pipeline for hvKP/CRhvKP prediction from genomic features.
RF + XGBoost (Optuna HPO) + LightGBM + SHAP interpretability.
MLflow tracks every run.
"""
import subprocess, mlflow, mlflow.sklearn, mlflow.xgboost
import logging, shap, optuna, warnings
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb, lightgbm as lgb

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
mlflow.set_experiment("kp_pakistan_ml")

HV_GENES = ["iucA","iucB","iucC","iucD","iutA","iroB","iroC","iroD","iroN",
            "fyuA","ybtS","ybtE","irp1","irp2","clbA","clbB","clbQ",
            "rmpA","rmpA2","mrkD","peg344","allS","kfu","ureA"]
BLACARB = ["blaKPC","blaNDM","blaOXA_48","blaIMP","blaVIM"]


def build_features(kleborate_path: str) -> tuple:
    """Build binary feature matrix from Kleborate output."""
    df = pd.read_csv(kleborate_path, sep="\t")
    feats = {}
    for gene in HV_GENES + BLACARB:
        cols = [c for c in df.columns if gene.lower().replace("_","") in c.lower().replace("_","")]
        if cols:
            feats[gene] = df[cols[0]].apply(lambda v: 0 if str(v) in ["","nan","-","none","0"] else 1)
        else:
            feats[gene] = pd.Series(0, index=df.index)
    X = pd.DataFrame(feats, index=df.index)

    # ST and K-locus encoding
    for col, prefix in [("ST","ST"), ("K_locus","Klocus")]:
        if col in df.columns:
            counts = df[col].astype(str).value_counts()
            top = counts[counts>=3].index.tolist()
            dummies = pd.get_dummies(df[col].astype(str).where(df[col].astype(str).isin(top),"Other"),
                                     prefix=prefix)
            X = pd.concat([X, dummies], axis=1)

    y = df.get("hvkp_label", pd.Series(0, index=df.index)).astype(int)
    X = X.fillna(0).astype(float)
    logger.info(f"Features: {X.shape} | hvKP+: {y.sum()}/{len(y)}")
    return X, y, X.columns.tolist()


def optuna_xgboost(X, y, n_trials=100) -> dict:
    def obj(trial):
        clf = xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators",100,1000),
            max_depth=trial.suggest_int("max_depth",3,10),
            learning_rate=trial.suggest_float("lr",0.01,0.3,log=True),
            subsample=trial.suggest_float("ss",0.6,1.0),
            colsample_bytree=trial.suggest_float("cbt",0.6,1.0),
            eval_metric="auc", use_label_encoder=False, random_state=42
        )
        return cross_validate(clf, X, y,
            cv=StratifiedKFold(5,shuffle=True,random_state=42),
            scoring="roc_auc", n_jobs=-1)["test_score"].mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"XGBoost best AUROC: {study.best_value:.4f}")
    return study.best_params


def shap_plots(model, X, feature_names, output_dir, name):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    try:
        exp = shap.TreeExplainer(model)
        vals = exp.shap_values(X)
        for plot_type, suffix in [("bar","_bar"), ("dot","_beeswarm")]:
            plt.figure(figsize=(12,8))
            shap.summary_plot(vals, X, feature_names=feature_names,
                               show=False, max_display=20,
                               plot_type="bar" if plot_type=="bar" else None)
            plt.title(f"SHAP {plot_type.title()} — {name}\nK. pneumoniae Pakistan hvKP")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/shap{suffix}_{name}.png", dpi=300)
            plt.close()
    except Exception as e:
        logger.warning(f"SHAP {name}: {e}")


def run_ml_pipeline():
    out_dir = Path("data/processed/ml"); out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path("reports/ml"); rep_dir.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = build_features("data/processed/virulence/kleborate_classified.tsv")
    cw = compute_class_weight("balanced", classes=np.unique(y), y=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_xgb = optuna_xgboost(X, y, n_trials=100)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost_Optuna": xgb.XGBClassifier(
            **best_xgb, scale_pos_weight=cw[0]/cw[1],
            use_label_encoder=False, eval_metric="auc", random_state=42),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1),
        "LogisticReg_L1": Pipeline([
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l1", solver="liblinear", C=0.1,
                class_weight="balanced", max_iter=1000, random_state=42))
        ]),
    }

    comparison = {}
    with mlflow.start_run(run_name="ml_model_comparison"):
        mlflow.log_params({"cv_folds":5,"n_samples":len(X),
                            "n_features":X.shape[1],"n_hvkp":int(y.sum())})

        for name, model in models.items():
            with mlflow.start_run(run_name=f"model_{name}", nested=True):
                cv_res = cross_validate(model, X, y, cv=cv,
                    scoring=["roc_auc","average_precision","f1"], n_jobs=-1)
                m = {"cv_auroc_mean": cv_res["test_roc_auc"].mean(),
                     "cv_auroc_std": cv_res["test_roc_auc"].std(),
                     "cv_auprc_mean": cv_res["test_average_precision"].mean(),
                     "cv_f1_mean": cv_res["test_f1"].mean()}
                comparison[name] = m
                for k,v in m.items(): mlflow.log_metric(k, v)
                model.fit(X, y)
                fm = model.named_steps.get("clf",model) if hasattr(model,"named_steps") else model
                shap_plots(fm, X, feature_names, str(rep_dir), name)
                if "XGBoost" in name: mlflow.xgboost.log_model(model, f"model_{name}")
                else: mlflow.sklearn.log_model(model, f"model_{name}")
                logger.info(f"{name}: AUROC={m['cv_auroc_mean']:.3f}±{m['cv_auroc_std']:.3f}")

        comp_df = pd.DataFrame(comparison).T
        comp_df.to_csv(str(rep_dir/"model_comparison.tsv"), sep="\t")
        mlflow.log_artifact(str(rep_dir/"model_comparison.tsv"))

    subprocess.run(["git","add","-A"], check=True)
    subprocess.run(["git","commit","-m","analysis: ML pipeline complete"], check=True)


if __name__ == "__main__":
    run_ml_pipeline()
