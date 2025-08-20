# titanic_moonshot.py  (pandas-get_dummies, alltid tett/dense)
from pathlib import Path
import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib


# ---------- Stier ----------
BASE = Path(__file__).resolve().parent
TRAIN_PATH = BASE / "data" / "train.csv"
TEST_PATH  = BASE / "data" / "test.csv"
assert TRAIN_PATH.exists(), "Mangler data/train.csv"
assert TEST_PATH.exists(),  "Mangler data/test.csv"


# ---------- Data ----------
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

y = train_df["Survived"].astype(int)
X_train_raw = train_df.drop(columns=["Survived"])
X_test_raw  = test_df.copy()


# ---------- Feature engineering (kun pandas) ----------
def extract_title(name: str) -> str:
    m = re.search(r",\s*([^\.]+)\.", str(name))
    return m.group(1).strip() if m else "Unknown"

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Tittel
    out["Title"] = out["Name"].apply(extract_title).replace({
        "Mlle":"Miss","Ms":"Miss","Mme":"Mrs",
        "Lady":"Rare","Countess":"Rare","Sir":"Rare","Jonkheer":"Rare","Don":"Rare","Dona":"Rare",
        "Capt":"Officer","Col":"Officer","Major":"Officer","Dr":"Officer","Rev":"Officer"
    })
    common = {"Mr","Mrs","Miss","Master","Officer","Rare"}
    out["Title"] = out["Title"].where(out["Title"].isin(common), "Rare")

    # Familie
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # Billett
    out["Ticket_str"] = out["Ticket"].astype(str)
    out["TicketPrefix"] = (
        out["Ticket_str"].str.replace(r"[0-9\.\/ ]", "", regex=True).str.upper().replace("", "NONE")
    )
    ticket_counts = out["Ticket_str"].value_counts()
    out["TicketGroupSize"] = out["Ticket_str"].map(ticket_counts).clip(lower=1, upper=8)

    # Kabin
    out["CabinDeck"] = out["Cabin"].astype(str).str[0].str.upper()
    out.loc[out["Cabin"].isna(), "CabinDeck"] = "Unknown"
    out["HasCabin"] = out["Cabin"].notna().astype(int)

    # Pris
    out["Fare"] = out["Fare"].astype(float)
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"].replace(0, 1)
    out["FareLog"] = np.log1p(out["Fare"])
    out["FarePerPersonLog"] = np.log1p(out["FarePerPerson"])

    # Samspill
    out["Pclass_Sex"] = out["Pclass"].astype(str) + "_" + out["Sex"].astype(str)
    return out

full = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
full = make_features(full)

cat_cols = ["Sex","Embarked","Title","CabinDeck","TicketPrefix","Pclass_Sex"]
num_cols = ["Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone",
            "TicketGroupSize","FarePerPerson","FareLog","FarePerPersonLog"]

# Imputer numerisk med median, kategorisk med "Unknown"
num_imp = SimpleImputer(strategy="median")
full[num_cols] = num_imp.fit_transform(full[num_cols])

for c in cat_cols:
    full[c] = full[c].astype(str).fillna("Unknown")

# One-hot via pandas (alltid DENSE)
full_dummies = pd.get_dummies(full[cat_cols], drop_first=False)
X_num = full[num_cols].reset_index(drop=True)
X_all = pd.concat([X_num, full_dummies.reset_index(drop=True)], axis=1)

# Splitt tilbake til train/test
n_train = len(X_train_raw)
X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test  = X_all.iloc[n_train:].reset_index(drop=True)


# ---------- Modell ----------
clf = HistGradientBoostingClassifier(
    random_state=42, max_depth=3, max_iter=400, learning_rate=0.06
)

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for i, (tr_idx, va_idx) in enumerate(cv.split(X_train, y), 1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_va)
    acc = accuracy_score(y_va, pred)
    print(f"Fold {i} accuracy: {acc:.3f}")
    scores.append(acc)

mean_acc = float(np.mean(scores))
std_acc  = float(np.std(scores))
print(f"Mean CV accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

# Hold-out
X_tr, X_te, y_tr, y_te = train_test_split(
    X_train, y, test_size=0.2, stratify=y, random_state=42
)
clf.fit(X_tr, y_tr)
hold_acc = accuracy_score(y_te, clf.predict(X_te))
print(f"Hold-out accuracy: {hold_acc:.3f}")

# Tren på alt og predikér test
clf.fit(X_train, y)
test_pred = clf.predict(X_test)

# ---------- Lagre ----------
(BASE / "results").mkdir(parents=True, exist_ok=True)
(BASE / "models").mkdir(parents=True, exist_ok=True)

pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": test_pred}).to_csv(
    BASE / "submission.csv", index=False
)

with open(BASE / "results" / "model_report.md", "w") as f:
    f.write("# Model Report\n")
    f.write(f"- Mean CV accuracy: {mean_acc:.3f} ± {std_acc:.3f}\n")
    f.write(f"- Hold-out accuracy: {hold_acc:.3f}\n")

joblib.dump(clf, BASE / "models" / "ensemble_hgb.pkl")
print("Skrev submission.csv, results/model_report.md og models/ensemble_hgb.pkl")
