from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import sklearn

class DtypeTransform(BaseEstimator, TransformerMixin):

    def __init__(self, coltypes):
        """
        Sets column dtypes.

        >>> coltypes = {
        >>>     'cont_num_cols': ['age', 'income'],
        >>>     'disc_num_cols': ['count'],
        >>>     'categoric_cols': ['day_of_week'],
        >>> }
        >>> trans = mlflow.sklearnwrapper.DtypeTransform(coltypes)
        >>> df = trans.fit_transform(df)

        """
        self.coltypes = coltypes

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        df = X.copy()
        for col in self.coltypes['cont_num_cols']:
            if col in df.columns:
                df[col] = df[col].astype('float64')
                df[col] = df[col].fillna(df[col].mean())
        for col in self.coltypes['disc_num_cols']:
            if col in df.columns:
                df[col] = df[col].astype('int64')
                df[col] = df[col].fillna(0)
        for col in self.coltypes['categoric_cols']:
            if col in df.columns:
                df[col] = df[col].astype('object')
                df[col] = df[col].fillna('UNKNOWN')

        return df

class BucketTransform(BaseEstimator, TransformerMixin):

    def __init__(self, id_col, target_col, min_thresh, pctiles=np.linspace(0,90,10, dtype=int)):
        self.id_col = id_col
        self.target_col = target_col
        self.min_thresh = min_thresh
        self.pctiles = pctiles
        self.bucket_lookup = defaultdict(lambda: '{}_NA'.format(self.id_col))

    def fit(self, X, y, **fit_params):
        X = X.copy()
        y = y.copy()

        if self.id_col not in X.columns:
            print('WARNING: {} not in X'.format(self.id_col))
            return self

        df = X.join(y)
        # Ensure percentiles are sorted in descending order
        self.pctiles[::-1].sort() # sorts inplace
        buckets = df.copy().groupby(self.id_col, as_index=False)[self.target_col].agg(['count','sum'])
        buckets[self.target_col] = buckets['sum'] / buckets['count']
        buckets[self.id_col] = buckets.index.values
        lbl_prfx = '{}_'.format(self.id_col)
        pctile_lbls = []
        pctile_targets = []
        for pctile in self.pctiles:
            lbl = '{}{}'.format(lbl_prfx, pctile)
            target = np.percentile(buckets[self.target_col], pctile)
            pctile_lbls.append(lbl)
            pctile_targets.append(target)
        def get_bucket_lbl(row):
            for lbl, target in zip(pctile_lbls, pctile_targets):
                if (row['count'] > self.min_thresh) and (row[self.target_col] >= target):
                    return lbl
            return '{}NA'.format(lbl_prfx)
        buckets['bucket'] = buckets.apply(get_bucket_lbl, axis=1)

        # Build bucket lookup dict
        for _id in buckets[self.id_col].unique():
            bucket = buckets.loc[buckets[self.id_col] == _id]['bucket'].values[0]
            self.bucket_lookup[str(_id)] = bucket

        self.categories_ = np.unique(list(self.bucket_lookup.values()))
        self.buckets_ = buckets

        return self

    def transform(self, X, y=None, **transform_params):
        X = X.copy()

        if self.id_col not in X.columns:
            print('WARNING: {} not in X'.format(self.id_col))
            return X

        def get_bucket(row):
            _id = row[self.id_col]
            return self.bucket_lookup[str(_id)]

        bucket_col = '{}_bucket'.format(self.id_col)
        X[bucket_col] = X.apply(get_bucket, axis=1)
        dummies = pd.get_dummies(X[bucket_col])

        # Add missing columns (only needed for test/predict sets)
        missing_cols = set(self.categories_) - set(dummies.columns)
        for c in missing_cols:
            dummies[c] = 0

        # Join new columns back to original data
        X = X.join(dummies)

        # Drop original and intermediate cols (e.g. affiliate_id and affiliate_id_bucket)
        X = X.drop(columns=[self.id_col, bucket_col])

        # Need to sort columns since adding missing cols does not guarantee order
        X = X.reindex(sorted(X.columns), axis=1)

        return X


class EmptyTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        return X


class PipelineModel:

    def __init__(self, pipeline, pred_proba=False):
        """
        A wrapper around an scikit-learn pipeline with ability to transform raw input in the same
        manner as in training.

        :param pipeline: an sklearn.pipeline
        :param pred_proba: boolean, whether or not to use pred_proba or just pred as final predictor

        >>> # Build pipeline
        >>> dt_trans = mlflow.sklearnwrapper.DtypeTransform(coltypes)
        >>> numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        >>> categoric_transormer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        >>> numeric_cols = coltypes['cont_num_cols'] + coltypes['disc_num_cols']
        >>> preprocessor = ColumnTransformer(transformers=[
        >>>     ('categoric_transformer', categoric_transormer, coltypes['categoric_cols']),
        >>>     ('numeric_transformer', numeric_transformer, numeric_cols),
        >>> ])
        >>> clf = LogisticRegression(penalty=penalty, C=C, random_state=123)
        >>> pipe = Pipeline(steps=[
        >>>     ('dtype_trans', dt_trans),
        >>>     ('preprocessor', preprocessor),
        >>>     ('clf', clf),
        >>> ])
        >>> pipe.fit(X_train, y_train)
        >>>
        >>> # Create pipeline wrapper
        >>> pipeline_wrapper = mlflow.sklearnwrapper.SKLearnPipelineWrapper(pipe)
        >>> pipeline_wrapper.fit(X_train, y_train)
        >>>
        >>> # Evaluate Metrics
        >>> preds = pipeline_wrapper._predict(X_test)
        >>> pred_probas = pipeline_wrapper._predict_proba(X_test)
        >>> (acc, auc) = eval_metrics(y_test, preds, pred_probas)
        """
        self.pipeline = pipeline
        self.pred_proba = pred_proba

    def fit(self, X, y):
        return self.pipeline.fit(X, y)

    def predict(self, df):
        """Used for serving pipeline in production."""
        if self.pred_proba:
            return self._predict_proba(df)
        else:
            return self._predict(df)

    def _predict(self, df):
        """Used during training for model evaluation."""
        return self.pipeline.predict(df)

    def _predict_proba(self, df):
        """Used during training for model evaluation."""
        return self.pipeline.predict_proba(df)[:,1]

class PipelineEnsemble:

    def __init__(self, pipelines, aggregator):
        """
        A wrapper around multiple scikit-learn pipelines with ability to transform raw input in the same
        manner as in training, as well as combine the outputs of mulitple predictions into one
        aggregate prediction.

        :param pipelines: an sklearn.pipeline

        >>> # Build pipeline
        >>> dt_trans = mlflow.sklearnwrapper.DtypeTransform(coltypes)
        >>> numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        >>> categoric_transormer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        >>> numeric_cols = coltypes['cont_num_cols'] + coltypes['disc_num_cols']
        >>> preprocessor = ColumnTransformer(transformers=[
        >>>     ('categoric_transformer', categoric_transormer, coltypes['categoric_cols']),
        >>>     ('numeric_transformer', numeric_transformer, numeric_cols),
        >>> ])
        >>> conv_clf = LogisticRegression(penalty=penalty, C=C, random_state=123)
        >>> ltv_regr = RandomForestRegressor(
        >>>     n_estimators=n_estimators,
        >>>     n_jobs=-1,
        >>>     max_features=max_features,
        >>>     min_samples_leaf=min_samples_leaf,
        >>>     min_samples_split=min_samples_split,
        >>>     max_depth=max_depth,
        >>>     random_state=123)
        >>> preprocess_steps=[
        >>>     ('dtype_trans', dt_trans),
        >>>     ('preprocessor', preprocessor)]
        >>> conv_steps = preprocess_steps + [('conv_clf', conv_clf)]
        >>> ltv_steps = preprocess_steps + [('ltv_regr', ltv_regr)]
        >>> conv_pipe = Pipeline(steps=conv_steps)
        >>> ltv_pipe = Pipeline(steps=ltv_steps)
        >>>
        >>> # Create pipeline wrapper
        >>> aggregator = lambda x, y: x * y
        >>> ensemble = mlflow.sklearnwrapper.PipelineEnsemble([conv_pipe, ltv_pipe], aggregator)
        >>> ensemble.fit(X_train, y_train)
        >>>
        >>> # Evaluate Metrics
        >>> conv_preds = pipeline_wrapper._predict(X_test, pipe_index=0)
        >>> conv_pred_probas = pipeline_wrapper._predict_proba(X_test, pipe_index=0)
        >>> ltv_preds = pipeline_wrapper._predict(X_test, pipe_index=1)
        >>> (conv_acc, conv_auc) = eval_clf_metrics(conv_y_test, conv_preds, conv_pred_probas)
        >>> ltv_r2 = eval_ltv_metrics(ltv_y_test, ltv_preds)
        """
        self.pipelines = pipelines
        self.aggregator = aggregator

    def fit(self, X, y=None):
        raise NotImplementedError

    def predict(self, df):
        """Used for serving pipeline in production."""
        preds = [pipe.predict(df) for pipe in self.pipelines]
        return self.aggregator(*preds)
