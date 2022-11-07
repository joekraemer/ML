import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List

import experiments
import meadowrun
import numpy as np
import pandas as pd
from meadowrun import PipRequirementsFile
from sklearn.base import TransformerMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_validate, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO: Move this to a common lib?
OUTPUT_DIRECTORY = './output'
PATH_TO_VENV = '/Users/wchen/PycharmProjects/ML/venv/bin/python'
PATH_TO_REQUIREMENTS = '/Users/wchen/PycharmProjects/ML/Assignment 3/requirements_aws.txt'


def run_gs_fit_locally(gs, x, y):
    """
    Run the GS on AWS

    Could be incorporated into the class, but this makes the size of the pickled function sent to AWS 54 instead of 48000

    Args:
        gs: GS object
    """
    gs.fit(x, y)

    best_estimator = gs.best_estimator_.fit(x, y)
    final_estimator = best_estimator._final_estimator
    best_params = pd.DataFrame([best_estimator.get_params()])
    final_estimator_params = pd.DataFrame([final_estimator.get_params()])
    return best_params, final_estimator_params, best_estimator, gs


@dataclass
class ExperimentDetails:
    """
    Experiment details
    """
    ds: object
    ds_name: str

    def __str__(self):
        return 'ExperimentDetails(ds={}, name={})'.format(
            self.ds,
            self.ds_name,
        )


class BaseExperiment(ABC):
    def __init__(self, details: ExperimentDetails, cfg, experiment_name: str,
                 transformer=None):
        self._details = details
        self._verbose = cfg.verbose
        self.experiment_name = experiment_name
        self.cfg = cfg
        self.seed = cfg.seed

        out = '{}/{}'.format(OUTPUT_DIRECTORY, self.experiment_name)
        if not os.path.exists(out):
            os.makedirs(out)
        self._out = '{}/{}'.format(out, '{}')

        self._scorer = "f1_weighted"
        self._cv = StratifiedShuffleSplit(n_splits=cfg.cv_splits, test_size=0.2, random_state=0)
        self._aws = cfg.aws
        self._threads = cfg.threads
        self.nn_early_stopping = cfg.nn_early_stopping

        # Grid Search Parameters
        self._nn_arch = self.generate_nn_arch(cfg.gs_params.nodes, cfg.gs_params.layers)
        self._nn_lr = cfg.gs_params.lr
        self._nn_alpha = cfg.gs_params.alpha

        self.loss_curves_n_curves = cfg.loss_curves_n_curves
        self.learning_curves_n_training_sizes = cfg.learning_curves_n_training_sizes

        self._bv = self._extract_best_values_from_cfg(details, cfg, experiment_name)

        self.TransformerMixin = transformer

        self._dummyclassifier = DecisionTreeClassifier()

    @staticmethod
    def _extract_best_values_from_cfg(details: ExperimentDetails, cfg, algo_name: str):
        return cfg[details.ds_name].get(algo_name.lower())

    @staticmethod
    def generate_nn_arch(nodes: List[int], layers: List[int]) -> List[tuple]:
        arch_list = []
        for layer in layers:
            for node in nodes:
                arch_list.append(tuple(node for _ in range(layer)))
        return arch_list

    @abstractmethod
    def _create_transformer(self) -> TransformerMixin:
        """Create a new transformer object with the best n_components/n_clusters"""
        pass

    async def perform(self):
        pass

    def save_best_transformation(self):
        """ Not right now """
        # TODO: add support for saving clustering transformation datasets
        pass

    async def perform_nn_grid_search(self):
        """ Perform a Neural network grid search on the modified dataset. Used for exploring for part 4 and part 5"""

        grid = {'NN__alpha': self._nn_alpha,
                'NN__learning_rate_init': self._nn_lr,
                'NN__hidden_layer_sizes': self._nn_arch}

        transformer = self._create_transformer()
        mlp = MLPClassifier(activation='relu', max_iter=4000, early_stopping=self.nn_early_stopping,
                            learning_rate='constant',
                            random_state=self.seed)

        stages = [('transformer', transformer), ('NN', mlp)] if transformer is not None else [('NN', mlp)]
        pipe = Pipeline(stages, memory=experiments.pipeline_memory)
        await self.run_gs(pipe, grid)
        return

    async def perform_nn_performance_analysis(self):
        """Performance Analysis of modified dataset with best NN parameters. Satisfies part 4 and 5"""

        transformer = self._create_transformer()
        mlp = MLPClassifier(hidden_layer_sizes=self._bv.arch, alpha=self._bv.alpha, learning_rate_init=self._bv.lr,
                            activation='relu', max_iter=4000, learning_rate='constant',
                            early_stopping=self.nn_early_stopping, random_state=self.seed)
        stages = [('transformer', transformer), ('NN', mlp)] if transformer is not None else [('NN', mlp)]
        pipe = Pipeline(stages, memory=experiments.pipeline_memory)

        self.log("Generating Loss Curves")
        self._generate_loss_curve_data(pipe)
        self.log("Generating Learning Curves")
        self._generate_learning_curve_data(pipe)
        self.log("Predict Test Data")
        self._evaluate_test_data_with_nn(pipe)
        self.log("Done")
        return

    def _generate_loss_curve_data(self, pipe):
        loss_curves = []
        train_times = []

        for _ in range(0, self.loss_curves_n_curves):
            res = cross_validate(
                pipe,
                self._details.ds.training_x,
                self._details.ds.training_y,
                scoring=self._scorer,
                cv=self._cv,
                n_jobs=self._threads,
                return_train_score=True,
                return_estimator=True
            )

            for e in res['estimator']:
                # e is actually the pipeline, have to dig to the correct step to find the mlp classifier
                for step_tuple in e.steps:
                    classifier = step_tuple[1]
                    if isinstance(classifier, MLPClassifier):
                        loss_curves.append(classifier.loss_curve_)

            train_times.extend(res['fit_time'].tolist())

        loss_curves_df = pd.DataFrame(loss_curves).T
        loss_curves_df.to_csv(
            self._out.format('{}_loss_curves.csv'.format(self._details.ds_name)),
            index=False)

        train_times_df = pd.DataFrame(train_times)
        train_times_df.to_csv(
            self._out.format('{}_train_times.csv'.format(self._details.ds_name)),
            index=False)

        self.log("Loss Curves complete")
        return

    def _generate_learning_curve_data(self, pipe):

        train_sizes = np.linspace(0.1, 1.0, self.learning_curves_n_training_sizes)

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            pipe,
            self._details.ds.training_x,
            self._details.ds.training_y,
            scoring=self._scorer,
            cv=self._cv,
            n_jobs=self._threads,
            train_sizes=train_sizes,
            return_times=True,
        )

        test_scores_df = pd.DataFrame(test_scores).set_index(train_sizes)
        train_scores_df = pd.DataFrame(train_scores).set_index(train_sizes)

        test_scores_df.to_csv(
            self._out.format(
                '{}_learning_curve_test_scores.csv'.format(self._details.ds_name)))

        train_scores_df.to_csv(
            self._out.format(
                '{}_learning_curve_train_scores.csv'.format(self._details.ds_name)))

        self.log("Learning Curves complete")
        return

    def _generate_test_scores(self, pipe):
        """Run the best NN through the test data set and save the scores"""
        # TODO might want to add train time here, would have to split the pipe up I think
        pred_y = pipe.predict(self._details.ds.test_x)
        score = f1_score(self._details.ds.test_y, pred_y, average='weighted')
        self._save_lines_to_file(['Test score: {}'.format(score)], filename='test_score')
        self.log("Test Scores complete")
        return

    def _evaluate_best_transformer(self, ds) -> None:
        """Generates train test scores and wall clock time for part 2 evaluations."""
        curr_time = datetime.now()
        _ = self._dummyclassifier.fit(ds.training_x, ds.training_y)
        train_time = datetime.now() - curr_time

        pred_y = self._dummyclassifier.predict(ds.testing_x)
        score = f1_score(ds.testing_y, pred_y, average='weighted')
        self._save_lines_to_file(['Train time: {}'.format(train_time.total_seconds()), 'Test score: {}'.format(score)],
                                 filename='part2_dummy_classifier_score')
        return

    def _evaluate_test_data_with_nn(self, pipe):
        """Generates train test scores and wall clock time for part 2 evaluations."""
        _ = pipe.fit(self._details.ds.training_x, self._details.ds.training_y)

        pred_y = pipe.predict(self._details.ds.testing_x)
        score = f1_score(self._details.ds.testing_y, pred_y, average='weighted')
        self._save_lines_to_file(['Test score: {}'.format(score)],
                                 filename='part45_nn_classifier_score')
        return

    @abstractmethod
    async def perform_cluster(self) -> None:
        pass

    async def run_gs(self, pipe, grid, type: str = None):
        gs = GridSearchCV(pipe, grid, verbose=10, cv=self._cv, scoring="f1_weighted", n_jobs=self._threads,
                          error_score="raise", return_train_score=True)

        # TODO refactor this so function is declared at the initialization
        if self._aws:
            best_params, final_estimator_params, best_estimator, gs = await self._run_gs_fit_aws(gs,
                                                                                                 self._details.ds.training_x,
                                                                                                 self._details.ds.training_y)
        else:
            best_params, final_estimator_params, best_estimator, gs = run_gs_fit_locally(gs,
                                                                                         self._details.ds.training_x,
                                                                                         self._details.ds.training_y)
        self.log("Grid search complete. Logging...")

        ds_name_and_type = type + '_' + self._details.ds_name if type else self._details.ds_name
        ds_name_and_type = ds_name_and_type + '_' + self.experiment_name

        best_params.to_csv(self._out.format('{}_best_params.csv'.format(ds_name_and_type)), index=False)
        final_estimator_params.to_csv(self._out.format('{}_final_estimator_params.csv'.format(ds_name_and_type)),
                                      index=False)

        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(self._out.format('{}_gs_cv_results.csv'.format(ds_name_and_type)))
        self.log("Done")
        return gs, best_estimator

    async def _run_gs_fit_aws(self, gs, x, y):
        best_params, final_estimator_params, best_estimator, gs = await meadowrun.run_function(
            run_gs_fit_locally,
            meadowrun.AllocEC2Instance(),
            meadowrun.Resources(logical_cpu=self._threads, memory_gb=16, max_eviction_rate=80),
            meadowrun.Deployment.mirror_local(
                interpreter=PipRequirementsFile(path_to_requirements_file=PATH_TO_REQUIREMENTS, python_version='3.8')),
            args=(gs, x, y),
            wait_for_result=True
        )

        return best_params, final_estimator_params, best_estimator, gs

    def log(self, msg, *args):
        """
        If the learner has verbose set to true, log the message with the given parameters using string.format
        :param msg: The log message
        :param args: The arguments
        :return: None
        """
        if self._verbose:
            logger.info(msg.format(*args))

    def _save_lines_to_file(self, lines: List[str], filename: str):
        path = self._out.format('{}_{}.txt'.format(self._details.ds_name, filename))
        with open(path, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        return
