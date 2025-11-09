import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class AddNewFeature(BaseEstimator, TransformerMixin):
    """임상 수치 기반 파생변수 생성"""
    def t4_category(self, x):
        if x < 6: return 'T4_Low'
        elif x > 11.98: return 'T4_High'
        else: return 'T4_Normal'
    def t3_category(self, x):
        if x < 1.4: return 'T3_Low'
        elif x > 3: return 'T3_High'
        else: return 'T3_Normal'
    def age_category(self, x):
        if x < 30: return 'Young'
        elif x < 50: return 'Middle'
        elif x < 65: return 'Senior'
        else: return 'Elderly'

    def transform(self, X, y=None):
        X = X.copy()
        X['T3_Cat'] = X['T3_Result'].apply(self.t3_category)
        X['T4_Cat'] = X['T4_Result'].apply(self.t4_category)
        X['Race&T3'] = X['Race'].astype(str) + '_' + X['T3_Cat'].astype(str)
        X['Family&Iodine'] = X['Family_Background'].astype(str) + '_' + X['Iodine_Deficiency'].astype(str)
        X['T3&T4'] = X['T3_Cat'].astype(str) + '_' + X['T4_Cat'].astype(str)
        return X

    def fit(self, X, y=None):
        return self

class Dropper(BaseEstimator, TransformerMixin):
    """특정 피처 제거"""
    def __init__(self, columns):
        self.columns = columns

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)

    def fit(self, X, y=None):
        return self

class CustomEncoder(BaseEstimator, TransformerMixin):
    """범주형 원핫 + 수치형 스케일링"""
    def __init__(self):
        self.encoder = None

    def fit(self, X, y=None):
        num_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(exclude=np.number).columns
        self.encoder = ColumnTransformer([
            ('OneHot', OneHotEncoder(sparse_output=False), cat_cols),
            ('Scaler', StandardScaler(), num_cols)
        ])
        self.encoder.fit(X)
        return self

    def transform(self, X, y=None):
        encoded = self.encoder.transform(X)
        cat_features = self.encoder.named_transformers_['OneHot'].get_feature_names_out()
        num_features = X.select_dtypes(include=np.number).columns
        return pd.DataFrame(encoded, columns=list(cat_features) + list(num_features))

def create_preprocessor():
    """전체 전처리 파이프라인 반환"""
    useless = ['T3_Result', 'T4_Result', 'TSH_Result', 'Age', 'Nodule_Size', 
               'T3_Cat', 'T4_Cat', 'T3_Result_Cat']
    return Pipeline([
        ('add_features', AddNewFeature()),
        ('dropper', Dropper(useless)),
        ('encoder', CustomEncoder())
    ])
