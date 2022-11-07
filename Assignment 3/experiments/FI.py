import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from .base import ExperimentDetails
from .basedimred import BaseDimRedExperiment


class FI(TransformerMixin):
    def __init__(self, features_to_keep):
        self.features_to_keep = features_to_keep

    def fit(self, x, y=None, **fit_params):
        cols = list(self.features_to_keep)
        return

    def transform(self, x):
        cols = list(self.features_to_keep)
        return x[:, cols]

    def fit_transform(self, x, y=None, **fit_params):
        return self.transform(x)


class FIExperiment(BaseDimRedExperiment):
    def __init__(self, details: ExperimentDetails, cfg):
        super().__init__(details, cfg, experiment_name="FI", transformer=FI)

        self._part2_files_string = '{}_feature_importance.csv'  # Need to override this because the plotting is so different
        self.features_to_keep = self._bv.features_to_keep

    async def _generate_part2_artifacts(self) -> NDFrame:
        # Train the RF model and dump importances
        rf = RandomForestClassifier(random_state=self.seed)
        rf.fit(self._details.ds.training_x, self._details.ds.training_y)
        importances = self._calc_permutation_importance(rf)

        return importances

    def _create_transformer(self) -> TransformerMixin:
        """Create a new transformer object with the best n_components/n_clusters"""
        return FI(self.features_to_keep)

    def save_best_transformation(self) -> str:
        """ Perform transformation of the training data with the best n_components/n_clusters selected.
        Then save the dataframe to file for later plotting. """
        hdf_path = self._out.format('{}_dataset.hdf'.format(self._details.ds_name))

        temp_df = pd.DataFrame(self._details.ds.training_x)
        ds_features = temp_df[self.features_to_keep].to_numpy()  # Feature Reduced dataset
        ds_2 = pd.DataFrame(
            np.hstack((ds_features, np.atleast_2d(self._details.ds.training_y).T)))  # Add back the class labels
        cols = list(self.features_to_keep)
        cols.append('Class')
        ds_2.columns = list(cols)
        ds_2.to_hdf(hdf_path, self._details.ds_name, complib='blosc', complevel=9)
        return hdf_path

    def _calc_permutation_importance(self, classifier) -> NDFrame:
        result = permutation_importance(classifier, self._details.ds.training_x, self._details.ds.training_y,
                                        n_repeats=10, random_state=self.seed, n_jobs=2)

        sorted_importances_idx = result.importances_mean.argsort()
        importances = pd.DataFrame(result.importances[sorted_importances_idx].T,
                                   columns=sorted_importances_idx)  # TODO: add column names, would have to be when the dataset is loaded
        return importances
