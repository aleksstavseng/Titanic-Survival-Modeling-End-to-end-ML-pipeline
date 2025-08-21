# titanic_moonshot.py  — tett/dense pipeline uten OneHotEncoder/ColumnTransformer
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


# ------------------ Stier og laster data ------------------
BASE = Path(__file__).resolve().parent
TRAIN_PATH = BASE / "data" / "train.csv"
TEST_PATH  = BASE / "data" / "test.csv"

assert TRAIN_PATH.exists(), "Mangler data/train.csv"
assert TEST_PATH.exists(),  "Mangler data/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

y = train_df["Survived"].astype(int)
X_train_raw = train_df.drop(columns=["Survived"])
X_test_raw  = test_df.copy()


# ------------------ Feature engineering ------------------
def extract_title(name: str) -> str:
    """Hent tittel fra Name-feltet, normaliser til få kategorier."""
    m = re.search(r",\s*([^\.]+)\.", str(name))
    title = m.group(1).strip() if m else "Unknown"
    title_map = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Lady": "Rare", "Countess": "Rare", "Sir": "Rare", "Jonkheer": "Rare",
        "Don": "Rare", "Dona": "Rare",
        "Capt": "Officer", "Col": "Officer", "Major": "Officer", "Dr": "Officer", "Rev": "Officer"
    }
    title = title_map.get(title, title)
    if title not in {"Mr", "Mrs", "Miss", "Master", "Officer", "Rare"}:
        title = "Rare"
    return title


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Kjønn binært
    d["Sex"] = d["Sex"].map({"male": 0, "female": 1}).fillna(0).astype(int)

    # Embarked til små heltall
    emb_order = ["C", "Q", "S"]
    emb_map = {k: i for i, k in enumerate(emb_order)}
    d["Embarked"] = d["Embarked"].map(emb_map).fillna(0).astype(int)

    # Tittel fra navn
    d["Title"] = d["Name"].apply(extract_title)

    # Familie
    d["SibSp"] = d["SibSp"].fillna(0)
    d["Parch"] = d["Parch"].fillna(0)
    d["FamilySize"] = (d["SibSp"] + d["Parch"] + 1).astype(int)
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)

    # Ticket-grupper
    d["Ticket_str"] = d["Ticket"].astype(str)
    d["TicketPrefix"] = d["Ticket_str"].str.replace(r"[0-9\.\/ ]", "", regex=True).str.upper().replace("", "NONE")
    grp_counts = d["Ticket_str"].value_counts()
    d["TicketGroupSize"] = d["Ticket_str"].map(grp_counts).clip(lower=1, upper=8)

    # Cabin
    d["CabinDeck"] = d["Cabin"].astype(str).str[0].str.upper()
    d.loc[d["Cabin"].isna(), "CabinDeck"] = "Unknown"
    d["HasCabin"] = d["Cabin"].notna().astype(int)

    # Pris
    d["Fare"] = d["Fare"].astype(float)
    d["Age"]  = d["Age"].astype(float)
    d["FarePerPerson"]    = d["Fare"] / d["FamilySize"].replace(0, 1)
    d["FareLog"]          = np.log1p(d["Fare"])
    d["FarePerPersonLog"] = np.log1p(d["FarePerPerson"])

    # Samspill
    d["Pclass_Sex"] = d["Pclass"].astype(str) + "_" + d["Sex"].astype(str)

    return d


# Bygg features på concatenert sett for å få like dummy-kolonner
full = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
full = build_features(full)

# Kolonnelister
cat_cols = ["Title", "CabinDeck", "TicketPrefix", "Pclass_Sex"]
num_cols = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "TicketGroupSize",
    "FarePerPerson", "FareLog", "FarePerPersonLog", "Embarked"
]

# Imputer numerisk (fit på train-del)
n_train = len(X_train_raw)
imp = SimpleImputer(strategy="median")
full.loc[: n_train - 1, num_cols] = imp.fit_transform(full.loc[: n_train - 1, num_cols])
full.loc[n_train:,     num_cols] = imp.transform(full.loc[n_train:,     num_cols])

# Kategorier til str og dummies
for c in cat_cols:
    full[c] = full[c].astype(str).fillna("Unknown")

dummies = pd.get_dummies(full[cat_cols], drop_first=False)
X_num = full[num_cols].reset_index(drop=True)
X_all = pd.concat([X_num, dummies.reset_index(drop=True)], axis=1)

# Split tilbake
X_train = X_all.iloc[:n_train].reset_index(drop=True)
X_test  = X_all.iloc[n_train:].reset_index(drop=True)


# ------------------ Modell og evaluering ------------------
clf = HistGradientBoostingClassifier(
    random_state=42,
    max_depth=3,
    learning_rate=0.06,
    max_iter=400
)

# 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for i, (tr, va) in enumerate(cv.split(X_train, y), 1):
    clf.fit(X_train.iloc[tr], y.iloc[tr])
    pred = clf.predict(X_train.iloc[va])
    acc = accuracy_score(y.iloc[va], pred)
    print(f"Fold {i} acc: {acc:.3f}")
    scores.append(acc)
mean_cv = float(np.mean(scores))
std_cv  = float(np.std(scores))
print(f"Mean CV acc: {mean_cv:.3f} ± {std_cv:.3f}")

# Hold-out
Xt, Xh, yt, yh = train_test_split(X_train, y, test_size=0.2, stratify=y, random_state=42)
clf.fit(Xt, yt)
hold_acc = accuracy_score(yh, clf.predict(Xh))
print(f"Hold-out acc: {hold_acc:.3f}")

# Tren på alt og predikér test
clf.fit(X_train, y)
test_pred = clf.predict(X_test)


# ------------------ Lagre artefakter ------------------
(BASE / "results").mkdir(parents=True, exist_ok=True)
(BASE / "models").mkdir(parents=True, exist_ok=True)

# submission.csv
pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": test_pred}).to_csv(
    BASE / "submission.csv", index=False
)

# enkel rapport
with open(BASE / "results" / "model_report.md", "w") as f:
    f.write("# Model report\n")
    f.write(f"- Mean CV accuracy: {mean_cv:.3f} ± {std_cv:.3f}\n")
    f.write(f"- Hold-out accuracy: {hold_acc:.3f}\n")

# modell
joblib.dump(clf, BASE / "models" / "hgb_min.pkl")

print("Skrev submission.csv, results/model_report.md og models/hgb_min.pkl")
