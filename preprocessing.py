from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
import pandas as pd


def pre(x):
    data_types_dict = {
        'number.of.people.being.liable.to.provide.maintenance.for': str,
        'number.of.existing.credits.at.this.bank': str
    }
    x = x.astype(data_types_dict)
    SimpleImputer.get_feature_names_out = (lambda self, names=None:
                                           self.feature_names_in_)
    KBinsDiscretizer.get_feature_names_out = (lambda self, names=None:
                                              self.feature_names_in_)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = x.select_dtypes(include=['object']).columns.to_numpy()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, ['duration.in.month', 'credit.amount', 'age.in.years']),
            ('num2', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'),
             ['installment.rate.in.percentage.of.disposable.income', 'present.residence.since']),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    preprocessor.feature_names_in_ = x.columns
    x_dummy = preprocessor.fit_transform(x)
    x_dummy = pd.DataFrame(data=x_dummy, columns=preprocessor.get_feature_names_out())
    return x_dummy
