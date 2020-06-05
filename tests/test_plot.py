# -*- coding: utf-8 -*-
"""Test of plot module.

This module test the various functions present in the plot module.

Assertion and resulting images are tested.
"""
from collections import Counter
import unittest
import unittest.mock
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import bff.plot as bplt


class TestPlot(unittest.TestCase):
    """
    Unittest of Fancy module.
    """
    # Set the set to have the same result each time.
    np.random.seed(42)

    history = {
        'loss': [3.3517270615352457, 3.0337808328178063, 2.742552079627262,
                 2.510529179309481, 2.4773420348239305, 2.4090258640020935,
                 2.3345777257603015, 2.336566770496081, 2.34276949460782,
                 2.321525989465378, 2.3300879552735756, 2.3224288386915197,
                 2.324129374183003, 2.3158747431021838, 2.3194296072475873,
                 2.2962934024369894, 2.296843618603807, 2.298411148876401,
                 2.302087271033819, 2.2869889256942213],
        'acc': [0.385427135, 0.49045226, 0.510552765, 0.650552765, 0.70030151,
                0.76070351, 0.810552765, 0.80552764, 0.81045226, 0.80050251,
                0.81557789, 0.82060302, 0.810552765, 0.80552764, 0.8056784,
                0.85075377, 0.81557789, 0.81060302, 0.80552764, 0.81572865],
        'val_loss': [3.3074077556791077, 3.006745302662272, 2.8061152659403104,
                     2.6056061252970226, 2.45513324273213, 2.4046621198808954,
                     2.304321059871107, 2.304280655512054, 2.3042611346560324,
                     2.3042683235268466, 2.3044002410326705, 2.304716517416279,
                     2.3049982415602894, 2.305085456921962, 2.3051163034046187,
                     2.3052417696192022, 2.3052861982219377, 2.305426104982545,
                     2.305481707112173, 2.3055578968795793],
        'val_acc': [0.41610487, 0.51111111, 0.52485643, 0.66360799, 0.71360799,
                    0.75985019, 0.81111111, 0.80861423, 0.80861423, 0.80486891,
                    0.80362048, 0.806129835, 0.80238452, 0.80113608, 0.80113608,
                    0.80739076, 0.80988764, 0.80113608, 0.806129835, 0.80363296]
    }

    history_without_val = {k: v for k, v in history.items() if k in ['loss', 'acc']}
    history_mult = {k: v[:-4] for k, v in history.items()}

    y_true = [1.87032178, 1.22725664, 9.38496685, 7.91451104, 7.60794146,
              9.65912261, 2.54053964, 7.31815866, 5.91692937, 2.78676838,
              7.92586481, 2.31337877, 1.78432016, 9.55596989, 6.64471696,
              3.33907423, 7.49321025, 7.14822795, 4.11686499, 2.40202043]
    y_pred = [1.85161709, 1.33317135, 9.45246137, 7.91986758, 7.54877922,
              9.71532022, 3.56777447, 7.88673475, 5.56090322, 2.78851836,
              6.70636033, 2.67531555, 1.13061356, 8.29287223, 6.27275223,
              2.49572863, 7.14305019, 8.53578604, 3.99890533, 2.35510298]

    y_true_matrix = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    y_pred_matrix = [1, 2, 3, 2, 3, 1, 1, 2, 3]

    y_true_matrix_cat = ['dog', 'cat', 'bird', 'dog', 'cat', 'bird', 'dog', 'cat', 'bird']
    y_pred_matrix_cat = ['dog', 'cat', 'bird', 'cat', 'bird', 'dog', 'dog', 'cat', 'bird']

    # Timeseries for testing.
    AXIS = {'x': 'darkorange', 'y': 'green', 'z': 'steelblue'}

    data = (pd.DataFrame(np.random.randint(0, 100, size=(60 * 60, 3)), columns=AXIS.keys())
            .set_index(pd.date_range('2018-01-01', periods=60 * 60, freq='S'))
            .rename_axis('datetime'))

    # Data with missing values.
    data_miss = (data
                 .drop(pd.date_range('2018-01-01 00:05', '2018-01-01 00:07', freq='S'))
                 .drop(pd.date_range('2018-01-01 00:40', '2018-01-01 00:41', freq='S'))
                 .drop(pd.date_range('2018-01-01 00:57', '2018-01-01 00:59', freq='S'))
                 )
    # Counter for bar chart.
    counter = Counter({'xelqo': 300, 'nisqo': 397, 'bff': 7454, 'eszo': 300, 'hedo': 26,
                       'sevcyk': 13, 'ajet': 31, 'zero': 10, 'exudes': 4, 'frazzio': 2})
    # Counter for pie chart.
    counter_pie = Counter({'xelqo': 300, 'nisqo': 397, 'bff': 454, 'eszo': 300, 'hedo': 100})
    # Dictionary for bar chart.
    dict_to_plot = {'Red': 15, 'Green': 50, 'Blue': 24}
    dict_to_plot_numerical = {1: 1_798, 2: 12_000, 3: 2_933}

    # Data for tsne.
    X, y = datasets.make_circles(n_samples=30, factor=.5, noise=.05, random_state=42)
    df = pd.DataFrame(X).assign(label=y)
    tsne = TSNE(n_iter=250)
    tsne_results = tsne.fit_transform(df.drop('label', axis='columns'))
    df_tsne = df[['label']].assign(tsne_1=tsne_results[:, 0], tsne_2=tsne_results[:, 1])

    # Data for kmeans.
    pca = PCA(n_components=2).fit_transform(tsne_results)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(pca)
    df_kmeans = pd.DataFrame({'kmeans_1': pca[:, 0], 'kmeans_2': pca[:, 1],
                              'label': kmeans.predict(pca)})

    def test_get_n_colors(self):
        """
        Test of the `get_n_colors` function.
        """
        # Check assertion if cmap does not exists.
        self.assertRaises(AssertionError, bplt.get_n_colors, 10, 'fakecmap')

        colors = bplt.get_n_colors(3)
        res = [np.array([0.5, 0., 1., 1.]),
               np.array([0.50392157, 0.99998103, 0.70492555, 1.]),
               np.array([1.0000000e+00, 1.2246468e-16, 6.1232340e-17, 1.0000000e+00])]
        np.testing.assert_allclose(colors, res)

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix(self):
        """
        Test of the `plot_confusion_matrix` function.
        """
        ax = bplt.plot_confusion_matrix(self.y_true_matrix, self.y_pred_matrix)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_annotation_fmt(self):
        """
        Test of the `plot_confusion_matrix` function.

        Check with the `annotation_fmt` option.
        """
        ax = bplt.plot_confusion_matrix(self.y_true_matrix, self.y_pred_matrix,
                                        normalize='all', annotation_fmt='.4f')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_cbar_fmt(self):
        """
        Test of the `plot_confusion_matrix` function.

        Check with the `cbar_fmt` option.
        """
        cbar_fmt = FuncFormatter(lambda x, p: format(float(x), ',g'))
        ax = bplt.plot_confusion_matrix(self.y_true_matrix, self.y_pred_matrix,
                                        normalize='all', cbar_fmt=cbar_fmt)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_fmt_thousand(self):
        """
        Test of the `plot_confusion_matrix` function.

        Check the format when there are more than on thousand examples.
        """
        ax = bplt.plot_confusion_matrix(self.y_true_matrix * 2000, self.y_pred_matrix * 2000)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_labels_filter(self):
        """
        Test of the `plot_confusion_matrix` function.

        Check with the `labels_filter` option.
        """
        ax = bplt.plot_confusion_matrix(self.y_true_matrix, self.y_pred_matrix,
                                        labels_filter=[3, 1])
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_normalize(self):
        """
        Test of the `plot_confusion_matrix` function.

        Check with the `normalize` option.
        """
        ax = bplt.plot_confusion_matrix(self.y_true_matrix, self.y_pred_matrix,
                                        normalize='all')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_sample_weight(self):
        """
        Test of the `plot_confusion_matrix` function.

        Check with the `sample_weigth` option.
        """
        weights = range(1, len(self.y_true_matrix) + 1)
        ax = bplt.plot_confusion_matrix(self.y_true_matrix, self.y_pred_matrix,
                                        sample_weight=weights)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_stats_acc(self):
        """
        Test of the `plot_confusion_matrix` function.

        Check with the `stats` option.
        """
        ax = bplt.plot_confusion_matrix(self.y_true_matrix, self.y_pred_matrix,
                                        stats='accuracy')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_stats_error(self):
        """
        Test of the `plot_confusion_matrix` function.

        Check with the `stats` option when the key of the stat is wrong.
        """
        # Check the error message using a mock.
        with unittest.mock.patch('logging.Logger.error') as mock_logging:
            ax = bplt.plot_confusion_matrix(self.y_true_matrix_cat, self.y_pred_matrix_cat,
                                            stats='acc')
            mock_logging.assert_called_with("Wrong key acc, possible values: "
                                            "['precision', 'recall', 'f1-score', 'support'].")
            return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_stats_prec(self):
        """
        Test of the `plot_confusion_matrix` function.

        Check with the `stats` option with precision for all classes.
        """
        ax = bplt.plot_confusion_matrix(self.y_true_matrix, self.y_pred_matrix,
                                        stats='precision')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_stats_fscore(self):
        """
        Test of the `plot_confusion_matrix` function.

        Check with the `stats` option with categorical values.
        """
        ax = bplt.plot_confusion_matrix(self.y_true_matrix_cat, self.y_pred_matrix_cat,
                                        stats='f1-score')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_ticklabels_cat(self):
        """
        Test of the `plot_confusion_matrix` function.

        Use categorical predictions.
        """
        ax = bplt.plot_confusion_matrix(self.y_true_matrix_cat, self.y_pred_matrix_cat)

        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_ticklabels_false(self):
        """
        Test of the `plot_confusion_matrix` function.

        Check with the `ticklabels` option set to False.
        """
        ax = bplt.plot_confusion_matrix(self.y_true_matrix, self.y_pred_matrix,
                                        ticklabels=False)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_confusion_matrix_ticklabels_n_labels(self):
        """
        Test of the `plot_confusion_matrix` function.

        Check with the `ticklabels` option and print every 2 labels.
        """
        ax = bplt.plot_confusion_matrix(self.y_true_matrix, self.y_pred_matrix,
                                        ticklabels=2)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_correlation(self):
        """
        Test of the `plot_correlation` function.
        """
        ax = bplt.plot_correlation(self.data)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_correlation_with_ax(self):
        """
        Test of the `plot_correlation` function.
        """
        # Create fake data for one of the plot.
        df_tmp = pd.DataFrame({'x': [123, 27, 38, 45, 67], 'y': [456, 45.4, 32, 34, 90]})
        df_corr = df_tmp.corr()

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10), dpi=80)
        bplt.plot_correlation(df_corr, already_computed=True, ax=axes[0],
                              rotation_xticks=0, title='Correlation between x and y')
        bplt.plot_correlation(self.data, ax=axes[1], method='spearman')
        return fig

    @pytest.mark.mpl_image_compare
    def test_plot_counter(self):
        """
        Test of the `plot_counter` function.
        """
        ax = bplt.plot_counter(self.counter)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_counter_horizontal(self):
        """
        Test of the `plot_counter` function.

        Check the behaviour with `vertical=False`.
        """
        ax = bplt.plot_counter(self.counter, vertical=False, threshold=300,
                               title='Bar chart of fake companies turnover [Bn.]',
                               label_x='Company', label_y='Turnover',
                               grid='x', rotation_xticks=45, figsize=(10, 7))
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_counter_dict(self):
        """
        Test of the `plot_counter` function.

        Check the behaviour when using a dictionary.
        """
        ax = bplt.plot_counter(self.dict_to_plot, grid=None)
        return ax.figure

    def test_plot_history(self):
        """
        Test of the `plot_history` function.

        Only checks the assertions.
        """
        self.assertRaises(AssertionError, bplt.plot_history, self.history, 'f1_score')

    @pytest.mark.mpl_image_compare
    def test_plot_history_default(self):
        """
        Test of the `plot_history` function.
        """
        ax = bplt.plot_history(self.history, title='Model history with random data',
                               grid='both', figsize=(10, 5))
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_history_with_ax(self):
        """
        Test of the `plot_history` function.

        An ax is provided for the plot.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=80)
        # When creating the ax separately, style seems to be 'classic'.
        bplt.plot_history(self.history, axes=ax)
        return fig

    @pytest.mark.mpl_image_compare
    def test_plot_history_with_metric(self):
        """
        Test of the `plot_history` function.

        Metrics is plotted as well as the loss.
        """
        axes = bplt.plot_history(self.history, metric='acc', figsize=(10, 6))
        return axes[0].figure

    @pytest.mark.mpl_image_compare
    def test_plot_history_with_multiple_ticks(self):
        """
        Test of the `plot_history` function.

        Check that the decimal are removes if the number of ticks
        is a multiple of the number of epochs.
        """
        axes = bplt.plot_history(self.history_mult, metric='acc')
        return axes[0].figure

    @pytest.mark.mpl_image_compare
    def test_plot_history_without_val(self):
        """
        Test of the `plot_history` function.

        Test without any validation (`val_loss`).
        """
        axes = bplt.plot_history(self.history_without_val, metric='acc')
        return axes[0].figure

    @pytest.mark.mpl_image_compare
    def test_plot_history_without_val_and_metric(self):
        """
        Test of the `plot_history` function.

        Test without any validation (`val_loss`) and metric.
        """
        ax = bplt.plot_history(self.history_without_val, color='r')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_kmeans(self):
        """
        Test of the `plot_kmeans` function.
        """
        ax = bplt.plot_kmeans(self.df_kmeans, cmap='plasma')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_kmeans_with_centroids(self):
        """
        Test of the `plot_kmeans` function.

        Check the behaviour with centroids.
        """
        ax = bplt.plot_kmeans(self.df_kmeans, centroids=self.kmeans.cluster_centers_)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_pca_explained_variance_ratio(self):
        """
        Test of the `plot_pca_explained_variance_ratio` function.
        """
        pca = PCA(n_components=30)
        pca.fit(np.random.randint(0, 100, size=(1000, 60)))
        ax = bplt.plot_pca_explained_variance_ratio(pca, grid='y')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_pca_explained_variance_ratio_with_hline(self):
        """
        Test of the `plot_pca_explained_variance_ratio` function.

        The `hline` option is given.
        """
        pca = PCA(n_components=30)
        pca.fit(np.random.randint(0, 100, size=(1000, 60)))
        ax = bplt.plot_pca_explained_variance_ratio(pca, title='PCA with hline option', hline=0.55)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_pca_explained_variance_ratio_with_limits(self):
        """
        Test of the `plot_pca_explained_variance_ratio` function.
        """
        pca = PCA(n_components=30)
        pca.fit(np.random.randint(0, 100, size=(1000, 60)))
        ax = bplt.plot_pca_explained_variance_ratio(pca, lim_x=(0, 20), lim_y=(0, 0.5))
        return ax.figure

    def test_plot_pie(self):
        """
        Test of the `plot_pie` function.

        Only checks the assertions.
        """
        self.assertRaises(AssertionError, bplt.plot_pie, self.dict_to_plot, colors=['r', 'b'])

    @pytest.mark.mpl_image_compare
    def test_plot_pie_counter(self):
        """
        Test of the `plot_pie` function.
        """
        data = dict(self.counter_pie.most_common(4))
        ax = bplt.plot_pie(data, explode=0.01, title='', startangle=10)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_pie_full(self):
        """
        Test of the `plot_pie` function.

        Check the behaviour with `circle=False` and legend.
        """
        ax = bplt.plot_pie(self.counter_pie, circle=False,
                           title='Pie chart of fake companies turnover [Bn.]',
                           loc='best')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_pie_dict(self):
        """
        Test of the `plot_pie` function.

        Check the behaviour when using a dictionary and custom colors.
        """
        ax = bplt.plot_pie(self.dict_to_plot, colors=['r', 'g', 'b'])
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_predictions(self):
        """
        Test of the `plot_predictions` function.
        """
        ax = bplt.plot_predictions(self.y_true, self.y_pred)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_predictions_with_x(self):
        """
        Test of the `plot_predictions` function with x parameters for plot.

        Size of both given x are not the same.
        """
        x_true = pd.date_range('2018-01-01', periods=len(self.y_true), freq='H')
        x_pred = pd.date_range('2018-01-01', periods=len(self.y_pred), freq='H')
        ax = bplt.plot_predictions(self.y_true[:-5], self.y_pred, x_true=x_true[:-5], x_pred=x_pred,
                                   title='Model predictions with datetime')
        return ax.figure

    def test_plot_predictions_wrong_length(self):
        """
        Test of the `plot_predictions` function with x length different from y.
        """
        x_true = pd.date_range('2018-01-01', periods=len(self.y_true), freq='H')
        x_pred = pd.date_range('2018-01-01', periods=len(self.y_pred), freq='H')
        self.assertRaises(AssertionError, bplt.plot_predictions, self.y_true[:-5],
                          self.y_pred, x_true=x_true, x_pred=x_pred)

    def test_plot_series(self):
        """
        Test of the `plot_series` function.

        Only checks the assertions.
        """
        # Check assertion when no datetime index.
        columns = ['name', 'age', 'country']
        df_no_date = pd.DataFrame([['John', 24, 'China'],
                                   ['Mary', 20, 'China'],
                                   ['Jane', 25, 'Switzerland'],
                                   ['James', 28, 'China']],
                                  columns=columns)
        self.assertRaises(AssertionError, bplt.plot_series, df_no_date, 'age')

        # Check assertion when column doesn't exist.
        df_date = pd.DataFrame({'datetime': pd.date_range('2019-02-01', periods=50, freq='S'),
                                'values': np.random.randn(50)}).set_index('datetime')
        self.assertRaises(AssertionError, bplt.plot_series, df_date, 'col')

    @pytest.mark.mpl_image_compare
    def test_plot_series_all_same_axis(self):
        """
        Test of the `plot_series` function.

        Check the behaviour with all plots on the same ax.
        """
        ax = bplt.plot_series(self.data, 'x', groupby='3T', title='Plot of all axis',
                              color=self.AXIS['x'])
        for k in list(self.AXIS.keys())[1:]:
            bplt.plot_series(self.data, k, groupby='3T', ax=ax, color=self.AXIS[k])
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_series_mult_axis(self):
        """
        Test of the `plot_series` function.

        Check the behaviour with multiple axis on the figure.
        """
        fig, axes = plt.subplots(nrows=len(self.AXIS), ncols=1,
                                 figsize=(14, len(self.AXIS) * 3), dpi=80)
        for i, k in enumerate(self.AXIS.keys()):
            bplt.plot_series(self.data, k, ax=axes[i], title=f'Plot of axis - {k}',
                             color=self.AXIS[k])
        return fig

    @pytest.mark.mpl_image_compare
    def test_plot_series_with_peaks(self):
        """
        Test of the `plot_series` function.

        Check the behaviour with peaks and resampling.
        """
        ax = bplt.plot_series(self.data, 'x', groupby='2T', with_peaks=True,
                              title='Plot of x with peaks')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_series_with_missing_datetimes(self):
        """
        Test of the `plot_series` function.

        Check the behaviour with missing datetimes.
        """
        ax = bplt.plot_series(self.data_miss, 'x', groupby='S', with_missing_datetimes=True,
                              title='Plot of x with missing datetimes')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_series_with_missing_datetimes_groupby(self):
        """
        Test of the `plot_series` function.

        Check the behaviour with missing datetimes and groupby.
        """
        ax = bplt.plot_series(self.data_miss, 'x', groupby='T', with_missing_datetimes=True,
                              title='Plot of x with missing datetimes')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_series_with_sem(self):
        """
        Test of the `plot_series` function.

        Check the behaviour with sem and resampling.
        """
        ax = bplt.plot_series(self.data, 'x', groupby='3T', with_sem=True,
                              title='Plot of x with standard error of the mean (sem)')
        return ax.figure

    def test_plot_true_vs_pred(self):
        """
        Test of the `plot_true_vs_pred` function.

        Only checks the assertions.
        """
        __, ax = plt.subplots(1, 1, figsize=(14, 7), dpi=80)
        self.assertRaises(AssertionError, bplt.plot_true_vs_pred, self.y_true, self.y_pred,
                          ax=ax, with_histograms=True)

    @pytest.mark.mpl_image_compare
    def test_plot_true_vs_pred_default(self):
        """
        Test of the `plot_true_vs_pred` function.
        """
        ax = bplt.plot_true_vs_pred(self.y_true, self.y_pred)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_true_vs_pred_with_ax(self):
        """
        Test of the `plot_true_vs_pred` function.

        An ax is provided for the plot.
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 7), dpi=80)
        bplt.plot_true_vs_pred(self.y_true, self.y_pred, ax=ax)
        return fig

    @pytest.mark.mpl_image_compare
    def test_plot_true_vs_pred_with_correlation(self):
        """
        Test of the `plot_true_vs_pred` function.

        The option `with_correlation=True` is used.
        """
        ax = bplt.plot_true_vs_pred(self.y_true, self.y_pred,
                                    with_correlation=True, marker='.', c='r')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_true_vs_pred_with_histograms(self):
        """
        Test of the `plot_true_vs_pred` function.

        The option `with_histograms=True` is used.
        """
        axes = bplt.plot_true_vs_pred(self.y_true, self.y_pred, with_determination=False,
                                      with_histograms=True, marker='.', c='r')
        return axes[0].figure

    @pytest.mark.mpl_image_compare
    def test_plot_true_vs_pred_with_identity(self):
        """
        Test of the `plot_true_vs_pred` function.

        The option `with_identity=True` is used.
        """
        ax = bplt.plot_true_vs_pred(self.y_true, self.y_pred,
                                    with_identity=True, marker='.', c='r')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_tsne(self):
        """
        Test of the `plot_tsne` function.
        """
        ax = bplt.plot_tsne(self.df_tsne, label_col='label', labels=['Ko', 'Ok'])
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_tsne_with_colors(self):
        """
        Test of the `plot_tsne` function.

        Test with custom colors.
        """
        ax = bplt.plot_tsne(self.df_tsne, label_col='label', colors=['r', 'b'])
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_tsne_without_label(self):
        """
        Test of the `plot_tsne` function.

        Test without label.
        """
        ax = bplt.plot_tsne(self.df_tsne)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_tsne_without_label_with_color(self):
        """
        Test of the `plot_tsne` function.

        Test without label but with custom colors.
        """
        cmap = cm.get_cmap('rainbow')
        colors = list(cmap(np.linspace(0, 1, self.df_tsne.shape[0])))
        ax = bplt.plot_tsne(self.df_tsne, colors=colors, label_x='Dim x', label_y='Dim_y')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_tsne_warning_colors(self):
        """
        Test of the `plot_tsne` function.

        Test the warning regarding the colors.
        """
        # Check the error message using a mock.
        with unittest.mock.patch('logging.Logger.warning') as mock_logging:
            ax = bplt.plot_tsne(self.df_tsne, label_col='label', colors=['r'])
            mock_logging.assert_called_with('Number of colors does not match the number '
                                            'of labels (1/2), using last color for missing ones.')
            return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_tsne_warning_labels(self):
        """
        Test of the `plot_tsne` function.

        Test the warning regarding the labels.
        """
        # Check the error message using a mock.
        with unittest.mock.patch('logging.Logger.warning') as mock_logging:
            ax = bplt.plot_tsne(self.df_tsne, label_col='label', labels='True')
            mock_logging.assert_called_with('Not enough labels (1/2).')
            return ax.figure

    @pytest.mark.mpl_image_compare
    def test_set_thousands_separator_both(self):
        """
        Test of the `set_thousands_separator` function.

        The option `which='both'` is used with a custom value for
        the number of decimal.
        """
        ax = bplt.plot_counter(self.dict_to_plot_numerical,
                               title='Bar chart with custom thousands separator')
        ax = bplt.set_thousands_separator(ax, nb_decimals=3)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_set_thousands_separator_x(self):
        """
        Test of the `set_thousands_separator` function.

        Test with multiple subplots.

        The option `which='x'` is used.

        Note: formatting of first ax is weird with pytest.
        """
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(20, 10), dpi=80)
        bplt.plot_true_vs_pred(self.y_true, self.y_pred, ax=axes[0], title='Left plot')
        bplt.plot_true_vs_pred(self.y_true, self.y_pred, ax=axes[1], title='Right plot')
        bplt.set_thousands_separator(axes[1], 'x', 3)
        bplt.set_thousands_separator(axes, 'y', 2)
        return fig
