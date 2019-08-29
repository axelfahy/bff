# -*- coding: utf-8 -*-
"""Test of plot module.

This module test the various functions present in the plot module.

Assertion and resulting images are tested.
"""
import unittest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import bff.plot as bplt


class TestPlot(unittest.TestCase):
    """
    Unittest of Fancy module.
    """
    # Set the set to have the same result each time.
    np.random.seed(42)

    history = {
        'loss': [2.3517270615352457, 2.3737808328178063, 2.342552079627262,
                 2.310529179309481, 2.3773420348239305, 2.3290258640020935,
                 2.3345777257603015, 2.336566770496081, 2.34276949460782,
                 2.321525989465378, 2.3300879552735756, 2.3224288386915197,
                 2.324129374183003, 2.3158747431021838, 2.3194296072475873,
                 2.2962934024369894, 2.296843618603807, 2.298411148876401,
                 2.302087271033819, 2.2869889256942213],
        'acc': [0.085427135, 0.09045226, 0.110552765, 0.110552765, 0.06030151,
                0.14070351, 0.110552765, 0.10552764, 0.09045226, 0.10050251,
                0.11557789, 0.12060302, 0.110552765, 0.10552764, 0.1356784,
                0.15075377, 0.11557789, 0.12060302, 0.10552764, 0.14572865],
        'val_loss': [2.3074077556791077, 2.306745302662272, 2.3061152659403104,
                     2.3056061252970226, 2.30513324273213, 2.3046621198808954,
                     2.304321059871107, 2.304280655512054, 2.3042611346560324,
                     2.3042683235268466, 2.3044002410326705, 2.304716517416279,
                     2.3049982415602894, 2.305085456921962, 2.3051163034046187,
                     2.3052417696192022, 2.3052861982219377, 2.305426104982545,
                     2.305481707112173, 2.3055578968795793],
        'val_acc': [0.11610487, 0.11111111, 0.11485643, 0.11360799, 0.11360799,
                    0.11985019, 0.11111111, 0.10861423, 0.10861423, 0.10486891,
                    0.10362048, 0.096129835, 0.09238452, 0.09113608, 0.09113608,
                    0.08739076, 0.08988764, 0.09113608, 0.096129835, 0.09363296]
    }
    y_true = [1.87032178, 1.22725664, 9.38496685, 7.91451104, 7.60794146,
              9.65912261, 2.54053964, 7.31815866, 5.91692937, 2.78676838,
              7.92586481, 2.31337877, 1.78432016, 9.55596989, 6.64471696,
              3.33907423, 7.49321025, 7.14822795, 4.11686499, 2.40202043]

    y_pred = [1.85161709, 1.33317135, 9.45246137, 7.91986758, 7.54877922,
              9.71532022, 3.56777447, 7.88673475, 5.56090322, 2.78851836,
              6.70636033, 2.67531555, 1.13061356, 8.29287223, 6.27275223,
              2.49572863, 7.14305019, 8.53578604, 3.99890533, 2.35510298]

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
        axes = bplt.plot_history(self.history, metric='acc')
        return axes[0].figure

    @pytest.mark.mpl_image_compare
    def test_plot_predictions(self):
        """
        Test of the `plot_predictions` function.
        """
        ax = bplt.plot_predictions(self.y_true, self.y_pred)
        return ax.figure

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
        ax = bplt.plot_series(self.data, 'x', groupby='3T', title=f'Plot of all axis',
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
                              title=f'Plot of x with peaks')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_series_with_missing_datetimes(self):
        """
        Test of the `plot_series` function.

        Check the behaviour with missing datetimes.
        """
        ax = bplt.plot_series(self.data_miss, 'x', groupby='S', with_missing_datetimes=True,
                              title=f'Plot of x with missing datetimes')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_series_with_missing_datetimes_groupby(self):
        """
        Test of the `plot_series` function.

        Check the behaviour with missing datetimes and groupby.
        """
        ax = bplt.plot_series(self.data_miss, 'x', groupby='T', with_missing_datetimes=True,
                              title=f'Plot of x with missing datetimes')
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_series_with_sem(self):
        """
        Test of the `plot_series` function.

        Check the behaviour with sem and resampling.
        """
        ax = bplt.plot_series(self.data, 'x', groupby='3T', with_sem=True,
                              title=f'Plot of x with standard error of the mean (sem)')
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
        Test of the `plot_predictions` function.
        """
        ax = bplt.plot_true_vs_pred(self.y_true, self.y_pred)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_true_vs_pred_with_ax(self):
        """
        Test of the `plot_predictions` function.

        An ax is provided for the plot.
        """
        __, ax = plt.subplots(1, 1, figsize=(14, 7), dpi=80)
        ax = bplt.plot_true_vs_pred(self.y_true, self.y_pred, ax=ax)
        return ax.figure

    @pytest.mark.mpl_image_compare
    def test_plot_true_vs_pred_with_histograms(self):
        """
        Test of the `plot_predictions` function.

        The option `with_histograms=True` is used.
        """
        axes = bplt.plot_true_vs_pred(self.y_true, self.y_pred,
                                      with_histograms=True, marker='.', c='r')
        return axes[0].figure
