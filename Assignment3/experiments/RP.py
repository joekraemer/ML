from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.linalg import pinv
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.random_projection import SparseRandomProjection

from .base import ExperimentDetails
from .basedimred import BaseDimRedExperiment


def pairwise_dist_corr(x1, x2):
    assert x1.shape[0] == x2.shape[0]

    d1 = pairwise_distances(x1)
    d2 = pairwise_distances(x2)
    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]


def reconstruction_error(projections, x):
    w = projections.components_
    if sps.issparse(w):
        w = w.todense()
    p = pinv(w)
    reconstructed = ((p @ w) @ (x.T)).T  # Unproject projected data
    errors = np.square(x - reconstructed)
    return np.nanmean(errors)


# http://datascience.stackexchange.com/questions/6683/feature-selection-using-feature-importances-in-random-forests-with-scikit-learn
class ImportanceSelect(BaseEstimator, TransformerMixin):
    def __init__(self, model, n=1):
        self.model = model
        self.n = n

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return self

    def transform(self, X):
        return X[:, self.model.feature_importances_.argsort()[::-1][:self.n]]


class RPExperiment(BaseDimRedExperiment):

    def __init__(self, details: ExperimentDetails, cfg):
        super().__init__(details, cfg, experiment_name="RP",
                         transformer=SparseRandomProjection)
        self._dims = cfg.dr_analysis.dims

    async def perform(self):
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/RP.py
        self.log("Performing {}".format(self.experiment_name))

        tmp = defaultdict(dict)
        for i, dim in product(range(10), self._dims):
            rp = SparseRandomProjection(random_state=i, n_components=dim)
            tmp[dim][i] = pairwise_dist_corr(rp.fit_transform(self._details.ds.training_x), self._details.ds.training_x)
        tmp = pd.DataFrame(tmp).T
        tmp.to_csv(self._out.format('{}_scree1.csv'.format(self._details.ds_name)))

        tmp = defaultdict(dict)
        for i, dim in product(range(10), self._dims):
            rp = SparseRandomProjection(random_state=i, n_components=dim)
            rp.fit(self._details.ds.training_x)
            tmp[dim][i] = reconstruction_error(rp, self._details.ds.training_x)
        tmp = pd.DataFrame(tmp).T
        tmp.to_csv(self._out.format('{}_scree2.csv'.format(self._details.ds_name)))

        hdf_path = self.save_best_transformation()
        updated_ds = self._details.ds.reload_from_hdf(hdf_path=hdf_path, hdf_ds_name=self._details.ds_name,
                                                      preprocess=False)
        self._evaluate_best_transformer(updated_ds)
        return
