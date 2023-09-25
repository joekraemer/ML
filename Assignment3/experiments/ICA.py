import pandas as pd
from pandas import Series
from sklearn.decomposition import FastICA

from .base import ExperimentDetails
from .basedimred import BaseDimRedExperiment


class ICAExperiment(BaseDimRedExperiment):
    def __init__(self, details: ExperimentDetails, cfg):
        super().__init__(details, cfg, experiment_name="ICA", transformer=FastICA)

        self._dims = cfg.dr_analysis.dims

    async def _generate_part2_artifacts(self) -> Series:
        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/ICA.py
        ica = FastICA(random_state=self.seed)
        kurt = {}
        for dim in self._dims:
            ica.set_params(n_components=dim)
            tmp = ica.fit_transform(self._details.ds.training_x)
            tmp = pd.DataFrame(tmp)
            tmp = tmp.kurt(axis=0)
            kurt[dim] = tmp.abs().mean()

        return pd.Series(kurt)

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
