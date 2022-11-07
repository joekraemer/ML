from typing import List

import pandas as pd
from pandas.core.generic import NDFrame
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SequentialFeatureSelector, SelectorMixin
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

from .base import ExperimentDetails
from .basedimred import BaseDimRedExperiment


class SFSExperiment(BaseDimRedExperiment):
    def __init__(self, details: ExperimentDetails, cfg):
        super().__init__(details, cfg, experiment_name="SFS", transformer=SequentialFeatureSelector)

        self._dims = cfg.dr_analysis.dims
        self._part2_files_string = '{}_part2_analysis.csv'  # Need to override this because the plotting is so different

    def _create_transformer(self) -> TransformerMixin:
        return SequentialFeatureSelector(estimator=DecisionTreeClassifier(), n_features_to_select=self._bv.dims,
                                         cv=self._cv)

    async def _generate_part2_artifacts(self) -> NDFrame:
        transformer = SequentialFeatureSelector(estimator=DecisionTreeClassifier(), n_features_to_select=self._bv.dims,
                                                cv=self._cv)
        columns = ['fit_time_mean', 'fit_time_std', 'test_score_mean', 'test_score_std', 'train_score_mean',
                   'train_score_std']
        results = []
        valid_dims = [dim for dim in self._dims if dim < self._details.ds.training_x.shape[
            1]]  # doesn't work with less than the original datset size
        for dim in valid_dims:
            transformer.set_params(n_features_to_select=dim)
            transformer.fit(self._details.ds.training_x, self._details.ds.training_y)

            new_result_row = self._cv_with_dummy_classifier(transformer)
            results.append(new_result_row)

        return pd.DataFrame(results, columns=columns, index=valid_dims)

    def _cv_with_dummy_classifier(self, transformer: SelectorMixin) -> List[float]:
        transformed_training_x = transformer.transform(self._details.ds.training_x)
        res = cross_validate(
            DecisionTreeClassifier(),
            transformed_training_x,
            self._details.ds.training_y,
            scoring=self._scorer,
            cv=self._cv,
            n_jobs=self._threads,
            return_train_score=True)

        return [res['fit_time'].mean(), res['fit_time'].std(), res['test_score'].mean(), res['test_score'].std(), res[
            'train_score'].mean(), res['train_score'].std()]

    def save_additional_data(self) -> None:
        """Save the mixing matrix"""
        hdf_path = self._out.format('{}_mixing_matrix.csv'.format(self._details.ds_name))

        transformer = self._create_transformer()
        if transformer is None:
            return
        _ = transformer.fit_transform(self._details.ds.training_x)  # Feature Reduced dataset
        ds = pd.DataFrame(transformer.mixing_)
        ds.to_csv(hdf_path)
        return
