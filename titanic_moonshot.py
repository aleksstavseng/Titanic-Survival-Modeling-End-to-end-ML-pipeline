from pathlib import Path
import re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# --- Stier relativt til denne fila (unngår "No such file") ---
BASE = Path(__file__).resolve().parent
TRAIN_PATH = BASE / "data" / "train.csv"
TEST_PATH  = BASE / "data" / "test.csv"
assert TRAIN_PATH.exists(), "Mangler data/train.csv"
assert TEST_PATH.exists(),  "Mangler data/test.csv"

# --- Les data ---
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

y = train["Survived"].astype(int)
X_train_raw = train.drop(columns=["Survived"])
X_test_raw  = test.copy()

# --- Feature engineering (kun felter fra datasettet) ---
def extract_title(name: str) -> str:
    m = re.search(r",\s*([^\.]+)\.", str(name))
    return m.group(1).strip() if m else "Unknown"

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Tittel fra navn
    out["Title"] = out["Name"].apply(extract_title).replace({
        "Mlle":"Miss","Ms":"Miss","Mme":"Mrs","Lady":"Rare","Countess":"Rare","Sir":"Rare",
        "Jonkheer":"Rare","Don":"Rare","Dona":"Rare","Capt":"Officer","Col":"Officer",
        "Major":"Officer","Dr":"Officer","Rev":"Officer"
    })
    common = {"Mr","Mrs","Miss","Master","Officer","Rare"}
    out["Title"] = out["Title"].where(out["Title"].isin(common), "Rare")

    # Familie
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # Billett
    out["Ticket_str"] = out["Ticket"].astype(str)
    out["TicketPrefix"] = out["Ticket_str"].str.replace(r"[0-9\.\/ ]", "", regex=True).str.upper().replace("", "NONE")
    ticket_counts = out["Ticket_str"].value_counts()
    out["TicketGroupSize"] = out["Ticket_str"].map(ticket_counts).clip(lower=1, upper=8)

    # Lugar
    out["CabinDeck"] = out["Cabin"].astype(str).str[0].str.upper()
    out.loc[out["Cabin"].isna(), "CabinDeck"] = "Unknown"
    out["HasCabin"] = out["Cabin"].notna().astype(int)

    # Pris
    out["Fare"] = out["Fare"].astype(float)
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"].replace(0, 1)
    out["FareLog"] = np.log1p(out["Fare"])
    out["FarePerPersonLog"] = np.log1p(out["FarePerPerson"])

    # Kombinasjon
    out["Pclass_Sex"] = out["Pclass"].astype(str) + "_" + out["Sex"].astype(str)
    return out

full = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
full = make_features(full)

cat_cols = ["Sex","Embarked","Title","CabinDeck","TicketPrefix","Pclass_Sex"]
num_cols = ["Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone",
            "TicketGroupSize","FarePerPerson","FareLog","FarePerPersonLog"]

full = full[cat_cols + num_cols].copy()
X_train = full.iloc[:len(X_train_raw)].reset_index(drop=True)
X_test  = full.iloc[len(X_train_raw):].reset_index(drop=True)

# --- Preprosess + modeller (kun scikit-learn for stabilitet) ---
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", num_pipe, num_cols),
                         ("cat", cat_pipe, cat_cols)])

hgb = HistGradientBoostingClassifier(random_state=42, max_depth=3, max_iter=400, learning_rate=0.06)
rf  = RandomForestClassifier(n_estimators=400, random_state=42)
lr  = LogisticRegression(max_iter=1000)

ensemble = VotingClassifier(estimators=[("hgb", hgb), ("rf", rf), ("lr", lr)],
                            voting="soft", weights=[2,2,1])
pipe = Pipeline([("pre", pre), ("clf", ensemble)])

# --- 5-fold CV + hold-out ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for i, (tr, va) in enumerate(cv.split(X_train, y), 1):
    X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
    y_tr, y_va = y.iloc[tr], y.iloc[va]
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_va)
    acc = accuracy_score(y_va, pred)
    scores.append(acc)
    print(f"Fold {i} accuracy: {acc:.3f}")
mean_acc, std_acc = float(np.mean(scores)), float(np.std(scores))
print(f"Mean CV accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

X_tr, X_te, y_tr, y_te = train_test_split(X_train, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(X_tr, y_tr)
hold_acc = accuracy_score(y_te, pipe.predict(X_te))
print(f"Hold-out accuracy: {hold_acc:.3f}")

# --- Tren på alt og lag prediksjoner på test ---
pipe.fit(X_train, y)
test_pred = pipe.predict(X_test)

# --- Skriv ut filer ---
(BASE / "results").mkdir(parents=True, exist_ok=True)
(BASE / "models").mkdir(parents=True, exist_ok=True)

pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred}).to_csv(BASE / "submission.csv", index=False)
with open(BASE / "results" / "model_report.md", "w") as f:
    f.write("# Model Report\n")
    f.write(f"- Mean CV accuracy: {mean_acc:.3f} ± {std_acc:.3f}\n")
    f.write(f"- Hold-out accuracy: {hold_acc:.3f}\n")
joblib.dump(pipe, BASE / "models" / "ensemble_pipeline.pkl")

print("Skrev submission.csv, results/model_report.md og models/ensemble_pipeline.pkl")
