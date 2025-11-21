"""
dmv.py  (fixed)
Titanic data wrangling + model comparison script (sklearn compatibility fixes)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

# Use a relative output directory (works on macOS, Linux, Windows)
OUTDIR = Path("./titanic_project_outputs")
OUTDIR.mkdir(parents=True, exist_ok=True)

# helper to create OneHotEncoder with the right kwarg for sklearn version
def make_onehot_encoder(**kwargs):
    # sklearn 1.2+ uses sparse_output, older versions use sparse
    ver = sklearn.__version__
    try:
        major, minor = (int(x) for x in ver.split(".")[:2])
    except Exception:
        major, minor = 1, 0
    if (major, minor) >= (1, 2):
        # new signature
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kwargs)
    else:
        # older signature
        return OneHotEncoder(handle_unknown="ignore", sparse=False, **kwargs)

def load_titanic():
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
        print("Loaded titanic dataset from seaborn.")
    except Exception:
        csv_path = Path("titanic.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print("Loaded titanic dataset from titanic.csv in working directory.")
        else:
            raise FileNotFoundError("Could not find titanic dataset. Install seaborn or place titanic.csv in working directory.")
    return df

def basic_eda(df):
    print("Shape:", df.shape)
    print("Missing values per column:")
    print(df.isnull().sum())

def feature_engineer(df):
    df = df.copy()
    # Title from name if available
    if "name" in df.columns:
        df["Title"] = df["name"].str.extract(r',\s*([^\.]+)\.').iloc[:,0].str.strip().fillna("Unknown")
        common = df["Title"].value_counts().loc[lambda x: x>10].index
        df["Title"] = df["Title"].where(df["Title"].isin(common), other="Other")
    else:
        df["Title"] = "Unknown"
    # Family size
    if set(["sibsp","parch"]).issubset(df.columns):
        df["FamilySize"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1
    else:
        df["FamilySize"] = 1
    df["IsAlone"] = (df["FamilySize"]==1).astype(int)
    # Fare: treat non-positive as missing
    if "fare" in df.columns:
        df.loc[df["fare"]<=0, "fare"] = np.nan
    return df

def preprocess_and_save(df_orig):
    results = {}
    df = df_orig.copy()
    target_candidates = ["survived","Survived","Survived?"]
    target = None
    for c in target_candidates:
        if c in df.columns:
            target = c
            break
    if target is None:
        raise ValueError("Could not find target column 'survived' in dataset.")

    numeric_feats = ["age","fare","FamilySize"]
    categorical_feats = ["sex","pclass","embarked","Title"]
    numeric_feats = [c for c in numeric_feats if c in df.columns]
    categorical_feats = [c for c in categorical_feats if c in df.columns]

    colsA = list(set([target] + numeric_feats + categorical_feats + ["IsAlone"]))
    dfA = df[colsA].dropna().reset_index(drop=True)
    pathA = OUTDIR / "cleaned_dropna.csv"
    dfA.to_csv(pathA, index=False)

    # median/mode impute
    dfB = df[colsA].copy()
    if numeric_feats:
        num_imp = SimpleImputer(strategy="median")
        dfB[numeric_feats] = num_imp.fit_transform(dfB[numeric_feats])
    if categorical_feats:
        cat_imp = SimpleImputer(strategy="most_frequent")
        dfB[categorical_feats] = cat_imp.fit_transform(dfB[categorical_feats])
    pathB = OUTDIR / "cleaned_median_mode_impute.csv"
    dfB.to_csv(pathB, index=False)

    # KNN impute (numeric) + ordinal encode cats for distance
    dfC = df[colsA].copy()
    ord_enc = None
    if categorical_feats:
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        dfC_cats = dfC[categorical_feats].astype(object).fillna("MISSING")
        dfC[categorical_feats] = ord_enc.fit_transform(dfC_cats)
    knn_imp = KNNImputer(n_neighbors=5)
    cols_knn = [c for c in numeric_feats + categorical_feats if c in dfC.columns]
    dfC[cols_knn] = knn_imp.fit_transform(dfC[cols_knn])
    pathC = OUTDIR / "cleaned_knn_impute.csv"
    dfC.to_csv(pathC, index=False)

    results["A"] = {"df": dfA, "path": pathA}
    results["B"] = {"df": dfB, "path": pathB}
    results["C"] = {"df": dfC, "path": pathC, "ordinal_mapping": (ord_enc.categories_ if ord_enc is not None else None)}
    return results, numeric_feats, categorical_feats, target

def build_and_evaluate(results, numeric_feats, categorical_feats, target):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "HistGB": HistGradientBoostingClassifier(random_state=42)
    }
    summary = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for strat, info in results.items():
        df = info["df"].copy()
        X = df.drop(columns=[target])
        y = df[target].astype(int)

        # create OneHotEncoder with compatibility
        ohe = make_onehot_encoder()
        preproc = ColumnTransformer([
            ("num", StandardScaler(), [c for c in numeric_feats if c in X.columns]),
            ("cat", ohe, [c for c in categorical_feats if c in X.columns])
        ], remainder="passthrough")

        for mname, m in models.items():
            pipe = Pipeline([("pre", preproc), ("model", m)])
            cvres = cross_validate(pipe, X, y, cv=skf, scoring=["accuracy","precision","recall","f1","roc_auc"], return_train_score=False)
            metrics = {k: float(np.mean(v)) for k,v in cvres.items() if not k.startswith("estimator")}
            summary_key = f"{strat}_{mname}"
            summary[summary_key] = {"metrics": metrics, "cv_results": cvres}

            # Fit final model and save
            pipe.fit(X, y)
            model_path = OUTDIR / f"model_{summary_key}.joblib"
            joblib.dump(pipe, model_path)
            summary[summary_key]["model_path"] = str(model_path)

            # ROC + confusion matrix using a single train/test split for visuals
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            pipe.fit(Xtr, ytr)
            if hasattr(pipe.named_steps["model"], "predict_proba"):
                yscore = pipe.predict_proba(Xte)[:, 1]
            else:
                try:
                    yscore = pipe.decision_function(Xte)
                except Exception:
                    yscore = pipe.predict(Xte)
            fpr, tpr, _ = roc_curve(yte, yscore)
            plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC curve {summary_key}")
            plt.grid(True)
            rocpath = OUTDIR / f"roc_{summary_key}.png"
            plt.savefig(rocpath)
            plt.close()
            summary[summary_key]["roc_path"] = str(rocpath)

            ypred = pipe.predict(Xte)
            cm = confusion_matrix(yte, ypred)
            plt.figure()
            plt.imshow(cm)
            plt.title(f"Confusion matrix {summary_key}")
            plt.xlabel("Predicted"); plt.ylabel("Actual")
            plt.colorbar()
            cmpath = OUTDIR / f"cm_{summary_key}.png"
            plt.savefig(cmpath)
            plt.close()
            summary[summary_key]["cm_path"] = str(cmpath)

    # Save a small summary csv
    rows = []
    for k, v in summary.items():
        row = {"run": k}
        for met, val in v["metrics"].items():
            row[met] = val
        row["model_path"] = v.get("model_path", "")
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    summary_path = OUTDIR / "model_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return summary, summary_path

def main():
    df = load_titanic()
    basic_eda(df)
    df_fe = feature_engineer(df)
    results, numeric_feats, categorical_feats, target = preprocess_and_save(df_fe)
    summary, summary_path = build_and_evaluate(results, numeric_feats, categorical_feats, target)
    print("Saved outputs to:", OUTDIR)
    print("Summary CSV:", summary_path)
    for p in sorted(OUTDIR.iterdir()):
        print(p.name)

if __name__ == "__main__":
    main()
