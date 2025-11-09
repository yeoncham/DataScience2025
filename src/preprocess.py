class add_new_feature(BaseEstimator, TransformerMixin):
    def t4_category(self, x):
        if x < 6:
            return 'T4_Low'
        elif x > 11.98:
            return 'T4_High'
        else:
            return 'T4_Normal'
    def t3_category(self, x):
        if x < 1.4:
            return 'T3_Low'
        elif x > 3:
            return 'T3_High'
        else:
            return 'T3_Normal'
    def tsh_category(self, x):
        if x < 0.27:
            return 'TSH_Low'
        elif x > 4.2:
            return 'TSH_High'
        else:
            return 'TSH_Normal'
    
    def age_category(self, x):
        if x < 30:
            return 'Young'
        elif x < 50:
            return 'Middle'
        elif x < 65:
            return 'Senior'
        else:
            return 'Elderly'
    
    def nodule_size_category(self, x):
        if x < 1.0:
            return 'Small'
        elif x < 2.0:
            return 'Medium'
        elif x < 4.0:
            return 'Large'
        else:
            return 'VeryLarge'
    
    def fit(self, x, y=None):
        return self
    def transform(self, x, y=None):
        x = x.copy()
        
        x['T3_Result_Cat'] = pd.cut(x['T3_Result'], 
                                   bins=[-np.inf, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, np.inf],
                                   labels=[0, 1, 2, 3, 4, 5, 6, 7])
        
        x['T3_Cat'] = x['T3_Result'].apply(lambda x : self.t3_category(x))
        x['T4_Cat'] = x['T4_Result'].apply(lambda x : self.t4_category(x))
        
        x['Race&T3'] = x['Race'].astype(str) + x['T3_Result_Cat'].astype(str)
        x['Family&Iodine'] = x['Family_Background'].astype(str) + x['Iodine_Deficiency'].astype(str)
        x['T3&T4'] = x['T3_Cat'].astype(str) + x['T4_Cat'].astype(str)
        return x

class dropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        
    def fit(self, x, y=None):
        return self
        
    def transform(self, x, y=None):
        x = x.copy()
        return x.drop(self.columns, axis=1)

class custom_encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = None
        self.num_feature = None
        self.cat_feature = None
        self.OneHot_feature = None
    def fit(self, x, y=None):
        self.num_feature = x.select_dtypes('number').columns.to_list()
        self.cat_feature = x.select_dtypes(['object', 'category']).columns.to_list()
        self.encoder = ColumnTransformer([
            ('OneHot', OneHotEncoder(sparse_output=False), self.cat_feature),
            ('Scaler', StandardScaler(), self.num_feature)
        ])
        self.encoder.fit(x)
        self.OneHot_feature = self.encoder.named_transformers_['OneHot'].get_feature_names_out(self.cat_feature).tolist()
        
        return self
    def transform(self, x, y=None):
        x = x.copy()
        encoded = self.encoder.transform(x)
        
        return pd.DataFrame(
            encoded,
            columns=self.OneHot_feature + self.num_feature
        )

# 파이프라인 구성 
useless_feature = ['T3_Result', 'T4_Result', 'TSH_Result', 
                   'Age', 'Nodule_Size', 
                   'T3_Cat', 'T4_Cat', 'T3_Result_Cat']
preprocessor = Pipeline([
    ('add_new_feature', add_new_feature()),
    ('dropper', dropper(useless_feature)),
    ('encoder', custom_encoder())
])

dumy_pre = Pipeline([
    ('add_feature', add_new_feature()),
    ('dropper', dropper(useless_feature+['Race&T3', 'Family&Iodine', 'T3&T4'])), 
    ('encoder', custom_encoder())
])
