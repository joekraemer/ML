

class BaseExperiment:
    def __init__(self, cfg, experiment, solver):
        self.cfg = cfg
        self.experiment = experiment
        self.solver = solver
        pass

    async def run(self):
        raise NotImplementedError
