import pandas as pd
from sklearn.cluster import KMeans
from lifelines.statistics import multivariate_logrank_test
from sklearn import metrics


class ClusterProcessor:

    def __init__(self, data, sur_data):
        self.data = data
        self.sur_data = sur_data

    def KmeansCluster(self, nclusters):
        """
        Clusters the data using K-means algorithm.

        Parameters:
        - nclusters: the number of clusters to form.

        Returns:
        - An array of cluster labels.
        """
        K_mod = KMeans(n_clusters=nclusters)
        K_mod.fit(self.data)
        clusters = K_mod.predict(self.data)
        return clusters

    def LogRankp(self, nclusters):
        """
        Performs the Log-rank test for clustering quality evaluation.

        Parameters:
        - nclusters: the number of clusters to evaluate.

        Returns:
        - The p-value of the Log-rank test and an array of cluster labels.
        """
        clusters = self.KmeansCluster(nclusters)
        self.sur_data['Type'] = clusters
        pvalue = multivariate_logrank_test(self.sur_data['OS.time'], self.sur_data['Type'], self.sur_data['OS'])
        return pvalue, clusters

    def compute_indexes(self, maxclusters):
        """
        Computes and prints the clustering evaluation indexes for different cluster numbers.

        Parameters:
        - maxclusters: the maximum number of clusters to evaluate.
        """
        for i in range(2, maxclusters+1):
            pvalue, clusters = self.LogRankp(i)
            estimator = KMeans(n_clusters=i)
            estimator.fit(self.data)

            print("Number of clusters: ", i)
            print("Silhouette score: ", metrics.silhouette_score(self.data, estimator.labels_, metric='euclidean'))
            print("P-value: ", pvalue.p_value)


def do_km_plot(survive_data, pvalue, cindex, cancer_type, model_name):
    # import necessary packages
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    # extract relevant data
    values = np.asarray(survive_data['Type'])
    events = np.asarray(survive_data['OS'])
    times = np.asarray(survive_data['OS.time'])

    # set plotting style
    sns.set(style='ticks', context='notebook', font_scale=1.5)

    # create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # customize plot style
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)  # set thickness of x-axis line
    ax.spines['left'].set_linewidth(1.5)  # set thickness of y-axis line

    # fit and plot Kaplan-Meier survival curves for each cluster
    kaplan = KaplanMeierFitter()
    for label in set(values):
        kaplan.fit(times[values == label],
                   event_observed=events[values == label],
                   label='cluster {0}'.format(label))
        kaplan.plot_survival_function(ax=ax, ci_alpha=0)
        ax.legend(loc=1, frameon=False)

    # customize plot labels and title based on whether C-index was calculated or not
    if cindex == None:
        ax.set_xlabel('days', fontsize=20)
        ax.set_ylabel('Survival Probability', fontsize=20)
        ax.set_title('{1} \n Cancer: {0}    p-value.:{2: .1e} '.format(
            cancer_type, model_name, pvalue),
            fontsize=18,
            fontweight='bold')
    else:
        ax.set_xlabel('days', fontsize=20)
        ax.set_title('{1} \n Cancer: {0}  p-value.:{2: .1e}   Cindex: {3: .2f}'.format(
            cancer_type, model_name, pvalue, cindex),
            fontsize=18,
            fontweight='bold')

    # save plot as a .tiff file
    fig.savefig('./' + str(cancer_type) + model_name + '.tiff', dpi=300)
