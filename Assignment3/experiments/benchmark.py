from .base import ExperimentDetails, BaseExperiment


class BenchmarkExperiment(BaseExperiment):
    def __init__(self, details: ExperimentDetails, cfg):
        super().__init__(details, cfg, experiment_name='benchmark')

    async def perform(self) -> None:
        """
        Nothing for benchmark to do without NN
        :return:
        """

        self._evaluate_best_transformer(self._details.ds)
        return

    def _create_transformer(self):
        return None

    async def perform_cluster(self):
        self.log('Clustering for a specific dim is not run for {}'.format(self.experiment_name))
