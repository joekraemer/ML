import os
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame
from sklearn.base import TransformerMixin

from .base import BaseExperiment, ExperimentDetails
from .clustering import GeneralClusteringExperiment


class BaseDimRedExperiment(BaseExperiment):
    """
    Base class for Dimensionality experiments
    """

    def __init__(self, details: ExperimentDetails, cfg, experiment_name: str,
                 transformer=None):
        """
        :param details: Experiment details
        """
        super().__init__(details=details, cfg=cfg, experiment_name=experiment_name,
                         transformer=transformer)
        self._part2_files_string = '{}_scree.csv'

    def _create_transformer(self) -> TransformerMixin:
        return self.TransformerMixin(random_state=self.seed, n_components=self._bv.dims)

    async def perform(self) -> None:
        """ Perform Part 2. Perform the transformation, save artifacts and run the dummy classifier analysis."""
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/ICA.py
        self.log("Performing {}".format(self.experiment_name))

        df = await self._generate_part2_artifacts()
        df.to_csv(self._out.format(self._part2_files_string.format(self._details.ds_name)))

        hdf_path = self.save_best_transformation()
        self.save_additional_data()

        updated_ds = self._details.ds.reload_from_hdf(hdf_path=hdf_path, hdf_ds_name=self._details.ds_name,
                                                      preprocess=False)

        self._evaluate_best_transformer(updated_ds)
        return

    async def _generate_part2_artifacts(self) -> NDFrame:
        """Generates df for part 2 plotting."""
        pass

    def save_additional_data(self) -> None:
        """Save additional data from the best transformer. For example, mixing maxtrix from ICA"""
        pass

    def save_best_transformation(self) -> str:
        """ Perform transformation of the training data with the best n_components/n_clusters selected.
        Then save the dataframe to file for later plotting. """
        hdf_path = self._out.format('{}_dataset.hdf'.format(self._details.ds_name))

        self.save_additional_data()

        transformer = self._create_transformer()
        if transformer is None:
            return

        ds_features = transformer.fit_transform(self._details.ds.training_x)  # Feature Reduced dataset
        ds_2 = pd.DataFrame(
            np.hstack((ds_features, np.atleast_2d(self._details.ds.training_y).T)))  # Add back the class labels
        cols = list(range(ds_2.shape[1]))
        cols[-1] = 'Class'
        ds_2.columns = cols
        ds_2.to_hdf(hdf_path, self._details.ds_name, complib='blosc', complevel=9)
        return hdf_path

    async def perform_cluster(self) -> None:
        self.log('Running clustering for {}'.format(self.experiment_name))

        hdf_path = self._out.format('{}_mixing_matrix.hdf'.format(self._details.ds_name))
        if not os.path.exists(hdf_path):
            hdf_path = self.save_best_transformation()  # Just in case we didn't run part 1/2

        # Run clustering as a subexperiment
        updated_ds = self._details.ds.reload_from_hdf(hdf_path=hdf_path, hdf_ds_name=self._details.ds_name,
                                                      preprocess=False)
        clustering_output_dir = self._out.format('clustering/')

        if not os.path.exists(clustering_output_dir):
            os.makedirs(clustering_output_dir)

        clustering_output_dir = clustering_output_dir + "{}"

        clustering_details = deepcopy(self._details)
        clustering_details.ds = updated_ds

        ce = GeneralClusteringExperiment(clustering_details, self.cfg)
        await ce.perform_for_subexperiment(clustering_output_dir, self)
        return
