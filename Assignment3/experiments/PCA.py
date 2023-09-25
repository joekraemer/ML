import pandas as pd
from pandas import Series
from sklearn.decomposition import PCA

from .base import ExperimentDetails
from .basedimred import BaseDimRedExperiment


class PCAExperiment(BaseDimRedExperiment):

    def __init__(self, details: ExperimentDetails, cfg):
        super().__init__(details, cfg, experiment_name="PCA", transformer=PCA)

        self._dims = cfg.dr_analysis.dims

    async def _generate_part2_artifacts(self) -> Series:
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/PCA.py
        pca = PCA(random_state=self.seed)
        pca.fit(self._details.ds.training_x)
        return pd.Series(data=pca.explained_variance_, index=range(1, min(pca.explained_variance_.shape[0], 500) + 1))

    def save_additional_data(self) -> None:
        """Save the components and explained_variance_ of each of the componenents"""
        transformer = self._create_transformer()
        _ = transformer.fit_transform(self._details.ds.training_x)  # Feature Reduced dataset
        ds = pd.DataFrame(transformer.components_)
        ds.to_csv(self._out.format('{}_components.csv'.format(self._details.ds_name)))
        ds = pd.DataFrame(transformer.explained_variance_)
        ds.to_csv(self._out.format('{}_explained_variance.csv'.format(self._details.ds_name)))
        return
