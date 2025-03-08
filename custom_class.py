import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
#Custom Transformer Class
class NewFeatureTransformer(BaseEstimator, TransformerMixin):
     def fit(self, x, y=None):
          return self
     def transform(self, x):
          x['ratio'] = x['thalach']/x['trestbps']
          x=pd.DataFrame(x.loc[:, 'ratio'])
          return x.values
