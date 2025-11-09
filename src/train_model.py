from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np

def run_cross_validation(x, y, preprocessor, base_preprocessor, n_splits=10):
    accs_pre, accs_base = [], []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(x, y):
        x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 파생 변수 전처리
        x_tr_pre = preprocessor.fit_transform(x_tr)
        x_val_pre = preprocessor.transform(x_val)
        x_tr_pre, y_tr_smote = SMOTE(random_state=42).fit_resample(x_tr_pre, y_tr)

        model = RandomForestClassifier(random_state=42)
        model.fit(x_tr_pre, y_tr_smote)
        accs_pre.append(accuracy_score(y_val, model.predict(x_val_pre)))

        # 베이스라인
        x_tr_base = base_preprocessor.fit_transform(x_tr)
        x_val_base = base_preprocessor.transform(x_val)
        x_tr_base, y_tr_base = SMOTE(random_state=42).fit_resample(x_tr_base, y_tr)
        base_model = RandomForestClassifier(random_state=42)
        base_model.fit(x_tr_base, y_tr_base)
        accs_base.append(accuracy_score(y_val, base_model.predict(x_val_base)))

    return np.mean(accs_base), np.mean(accs_pre)
