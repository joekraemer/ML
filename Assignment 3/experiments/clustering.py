from collections import Counter
from collections import Counter
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from experiments.base import BaseExperiment, ExperimentDetails
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans as KMeansTransformer
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score as acc, homogeneity_score, rand_score, completeness_score
from sklearn.metrics import adjusted_mutual_info_score as ami, silhouette_score as sil_score, \
    silhouette_samples as sil_samples
from sklearn.mixture import GaussianMixture as GMM


def cluster_acc(y, cluster_labels):
    """ Attempts to compute the accuracy of teh clustering labels vs the actual data labels.

        There are some issues here because the target value is computed every time for each cluster label.
    """
    assert (y.shape == cluster_labels.shape)
    pred = np.empty_like(y)
    for label in set(cluster_labels):
        mask = cluster_labels == label
        sub = y[mask]
        target = Counter(sub).most_common(1)[0][0]
        pred[mask] = target
    return acc(y, pred)


class CustomGMM(GMM):
    def transform(self, x):
        return self.predict_proba(x)


class LabelKMeans(KMeansTransformer):
    def transform(self, x):
        trans = self.transform(x)
        trans['cluster'] = trans.labels_
        return trans


class GeneralClusteringExperiment(BaseExperiment):
    def __init__(self, details, cfg):
        super().__init__(details, cfg, experiment_name='clustering')

        self._clusters = cfg.cluster_analysis.clusters

    async def perform(self):
        self.log("Performing {}".format(self.experiment_name))
        await self.__perform_clustering_analysis(self._out)
        return

    async def perform_for_subexperiment(self, custom_out, main_experiment):
        self.log("Performing {} as part of {}".format(self.experiment_name, main_experiment.experiment_name))
        await self.__perform_clustering_analysis(output_dir=custom_out)
        return

    async def __perform_clustering_analysis(self, output_dir: str):
        """
        This function logs various stats like sse, ll, bic, sil, of clustering of the ExperimentDetails given at construction.

        :param custom_out:  used for experiment that need to call this as part of a larger experiment
        """

        # Adapted from https://github.com/JonathanTay/CS-7641-assignment-3/blob/master/clustering.py
        sse = defaultdict(list)
        ll = defaultdict(list)
        bic = defaultdict(list)
        sil = defaultdict(lambda: defaultdict(list))
        sil_s = np.empty(shape=(2 * len(self._clusters) * self._details.ds.training_x.shape[0], 4), dtype='<U21')
        acc = defaultdict(lambda: defaultdict(float))
        adj_mi = defaultdict(lambda: defaultdict(float))
        rand = defaultdict(lambda: defaultdict(float))
        homogeneity = defaultdict(lambda: defaultdict(float))
        completeness = defaultdict(lambda: defaultdict(float))

        km = KMeansTransformer(random_state=self.seed)
        gmm = GMM(random_state=self.seed)

        curr_time = datetime.now()
        j = 0
        for k in self._clusters:
            km.set_params(n_clusters=k)
            gmm.set_params(n_components=k)
            km.fit(self._details.ds.training_x)
            gmm.fit(self._details.ds.training_x)

            km_labels = km.predict(self._details.ds.training_x)
            gmm_labels = gmm.predict(self._details.ds.training_x)

            sil[k]['Kmeans'] = sil_score(self._details.ds.training_x, km_labels)
            sil[k]['GMM'] = sil_score(self._details.ds.training_x, gmm_labels)

            km_sil_samples = sil_samples(self._details.ds.training_x, km_labels)
            gmm_sil_samples = sil_samples(self._details.ds.training_x, gmm_labels)
            # There has got to be a better way to do this, but I can't brain right now
            for i, x in enumerate(km_sil_samples):
                sil_s[j] = [k, 'Kmeans', round(x, 6), km_labels[i]]
                j += 1
            for i, x in enumerate(gmm_sil_samples):
                sil_s[j] = [k, 'GMM', round(x, 6), gmm_labels[i]]
                j += 1

            sse[k] = [km.score(self._details.ds.training_x)]
            ll[k] = [gmm.score(self._details.ds.training_x)]
            bic[k] = [gmm.bic(self._details.ds.training_x)]

            acc[k]['Kmeans'] = cluster_acc(self._details.ds.training_y, km_labels)
            acc[k]['GMM'] = cluster_acc(self._details.ds.training_y, gmm_labels)

            adj_mi[k]['Kmeans'] = ami(self._details.ds.training_y, km_labels)
            adj_mi[k]['GMM'] = ami(self._details.ds.training_y, gmm_labels)

            rand[k]['Kmeans'] = rand_score(self._details.ds.training_y, km_labels)
            rand[k]['GMM'] = rand_score(self._details.ds.training_y, gmm_labels)

            homogeneity[k]['Kmeans'] = homogeneity_score(self._details.ds.training_y, km_labels)
            homogeneity[k]['GMM'] = homogeneity_score(self._details.ds.training_y, gmm_labels)

            completeness[k]['Kmeans'] = completeness_score(self._details.ds.training_y, km_labels)
            completeness[k]['GMM'] = completeness_score(self._details.ds.training_y, gmm_labels)

            self.log("Cluster: {}, time: {}".format(k, datetime.now() - curr_time))

        sse = (-pd.DataFrame(sse)).T
        sse.index.name = 'k'
        sse.columns = ['{} sse (left)'.format(self._details.ds_name)]

        ll = pd.DataFrame(ll).T
        ll.index.name = 'k'
        ll.columns = ['{} log-likelihood'.format(self._details.ds_name)]

        bic = pd.DataFrame(bic).T
        bic.index.name = 'k'
        bic.columns = ['{} BIC'.format(self._details.ds_name)]

        sil = pd.DataFrame(sil).T
        sil_s = pd.DataFrame(sil_s, columns=['k', 'type', 'score', 'label']).set_index('k')  # .T
        # sil_s = sil_s.T
        acc = pd.DataFrame(acc).T
        adj_mi = pd.DataFrame(adj_mi).T
        rand = pd.DataFrame(rand).T
        completeness = pd.DataFrame(completeness).T
        homogeneity = pd.DataFrame(homogeneity).T

        sil.index.name = 'k'
        sil_s.index.name = 'k'
        acc.index.name = 'k'
        adj_mi.index.name = 'k'
        rand.index.name = 'k'
        completeness.index.name = 'k'
        homogeneity.index.name = 'k'

        sse.to_csv(output_dir.format('{}_sse.csv'.format(self._details.ds_name)))
        ll.to_csv(output_dir.format('{}_logliklihood.csv'.format(self._details.ds_name)))
        bic.to_csv(output_dir.format('{}_bic.csv'.format(self._details.ds_name)))
        sil.to_csv(output_dir.format('{}_sil_score.csv'.format(self._details.ds_name)))
        sil_s.to_csv(output_dir.format('{}_sil_samples.csv'.format(self._details.ds_name)))
        acc.to_csv(output_dir.format('{}_acc.csv'.format(self._details.ds_name)))
        adj_mi.to_csv(output_dir.format('{}_adj_mi.csv'.format(self._details.ds_name)))
        rand.to_csv(output_dir.format('{}_rand_index.csv'.format(self._details.ds_name)))
        completeness.to_csv(output_dir.format('{}_completeness.csv'.format(self._details.ds_name)))
        homogeneity.to_csv(output_dir.format('{}_homogeneity.csv'.format(self._details.ds_name)))

        self.save_best_transformation(output_dir)
        return

    def _create_transformer(self) -> TransformerMixin:
        """ Not needed """
        pass

    def save_best_transformation(self, output_dir: str):
        km = self._extract_best_values_from_cfg_for_clustering('kmeans')
        gmm = self._extract_best_values_from_cfg_for_clustering('gmm')
        km = KMeansTransformer(random_state=self.seed, n_clusters=km.clusters)
        gmm = GMM(random_state=self.seed, n_components=gmm.clusters)

        self._save_best_transformation_single_cluster_algo(gmm, 'GMM', output_dir)
        self._save_best_transformation_single_cluster_algo(km, 'Kmeans', output_dir)
        return

    def _save_best_transformation_single_cluster_algo(self, cluster_algo, name: str, output_dir: str):
        cluster_algo.fit(self._details.ds.training_x)
        labels = cluster_algo.predict(self._details.ds.training_x)
        self._perform_TSNE(labels, name, output_dir)
        return

    async def perform_nn_grid_search(self):
        """This happens in the other classes"""
        pass

    async def perform_nn_performance_analysis(self):
        """This happens in the other classes"""
        pass

    def _perform_TSNE(self, y_labels, cluster_algo_name: str, output_dir: str):
        # y_labels used to be self._details.ds.training_y
        self.log("Creating TSNE mapping")
        # creates plots for part 2 and part 4
        self._details.ds.training_x2D = TSNE(verbose=10, random_state=self.seed, n_iter=5000).fit_transform(
            self._details.ds.training_x
        )

        ds_2d = pd.DataFrame(np.hstack((self._details.ds.training_x2D, np.atleast_2d(y_labels).T)),
                             columns=['x', 'y', 'target'])
        ds_2d.to_csv(output_dir.format('{}_{}_2D.csv'.format(self._details.ds_name, cluster_algo_name)))
        self.log("Done")
        return

    def _extract_best_values_from_cfg_for_clustering(self, algo_name: str):
        return self.cfg[self._details.ds_name].get(algo_name.lower())

    async def perform_cluster(self):
        self.log('Clustering for a specific dim is not run for {}'.format(self.experiment_name))


class SpecificClusteringExperiment(BaseExperiment):
    def __init__(self, details: ExperimentDetails, cfg, experiment_name: str,
                 transformer=None):
        """
        :param details: Experiment details
        """
        super().__init__(details=details, cfg=cfg, experiment_name=experiment_name,
                         transformer=transformer)

    async def perform(self):
        """This happens in the other general clustering experiment class"""
        pass

    async def perform_cluster(self):
        self.log('Clustering for a specific dim is not run for {}'.format(self.experiment_name))


class KMeansExperiment(SpecificClusteringExperiment):
    def __init__(self, details: ExperimentDetails, cfg):
        super().__init__(details=details, cfg=cfg, experiment_name='kmeans',
                         transformer=KMeansTransformer)

    def _create_transformer(self) -> TransformerMixin:
        return self.TransformerMixin(random_state=self.seed, n_clusters=self._bv.clusters)


class GMMExperiment(SpecificClusteringExperiment):
    def __init__(self, details: ExperimentDetails, cfg):
        super().__init__(details=details, cfg=cfg, experiment_name='gmm',
                         transformer=CustomGMM)

    def _create_transformer(self) -> TransformerMixin:
        return self.TransformerMixin(random_state=self.seed, n_components=self._bv.clusters)
