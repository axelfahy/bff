# -*- coding: utf-8 -*-
"""Test of Fancy module

This module test the various functions present in the Fancy module.
"""
import datetime
import unittest
import unittest.mock
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.api.types import CategoricalDtype
import pandas.util.testing as tm

from bff.fancy import (cast_to_category_pd, concat_with_categories, get_peaks, idict,
                       mem_usage_pd, parse_date, value_2_list, sliding_window)


class TestFancy(unittest.TestCase):
    """
    Unittest of Fancy module.
    """

    def test_cast_to_category_pd(self):
        """
        Test of the `cast_to_category_pd` function.
        """
        columns = ['name', 'age', 'country']
        df = pd.DataFrame([['John', 24, 'China'],
                           ['Mary', 20, 'China'],
                           ['Jane', 25, 'Switzerland'],
                           ['Greg', 23, 'China'],
                           ['James', 28, 'China']],
                          columns=columns)
        original_types = {'name': np.dtype('O'), 'age': np.dtype('int64'),
                          'country': np.dtype('O')}
        self.assertDictEqual(df.dtypes.to_dict(), original_types)

        df_optimized = cast_to_category_pd(df)

        tm.assert_frame_equal(df, df_optimized, check_dtype=False, check_categorical=False)

        country_type = CategoricalDtype(categories=['China', 'Switzerland'], ordered=False)
        optimized_types = {'name': np.dtype('O'), 'age': np.dtype('int64'),
                           'country': country_type}
        self.assertDictEqual(df_optimized.dtypes.to_dict(), optimized_types)

    def test_concat_with_categories(self):
        """
        Test of the `concat_with_categories` function.
        """
        column_types = {'name': 'object',
                        'color': 'category',
                        'country': 'category'}
        columns = list(column_types.keys())
        df_left = pd.DataFrame([['John', 'red', 'China'],
                                ['Jane', 'blue', 'Switzerland']],
                               columns=columns).astype(column_types)
        df_right = pd.DataFrame([['Mary', 'yellow', 'France'],
                                 ['Fred', 'blue', 'Italy']],
                                columns=columns).astype(column_types)
        df_res = pd.DataFrame([['John', 'red', 'China'],
                               ['Jane', 'blue', 'Switzerland'],
                               ['Mary', 'yellow', 'France'],
                               ['Fred', 'blue', 'Italy']],
                              columns=columns).astype(column_types)
        df_concat = concat_with_categories(df_left, df_right,
                                           ignore_index=True)
        # Check the content of the DataFrame.
        tm.assert_frame_equal(df_res, df_concat)
        # Check the types of the DataFrame's columns.
        self.assertTrue(pd.api.types.is_object_dtype(df_concat['name']))
        self.assertTrue(pd.api.types.is_categorical_dtype(df_concat['color']))
        self.assertTrue(pd.api.types.is_categorical_dtype(
                        df_concat['country']))

        # Check assertion if columns don't match.
        df_left_wrong = pd.DataFrame([['John', 'XXL', 'China']],
                                     columns=['name', 'size', 'country'])
        self.assertRaises(AssertionError, concat_with_categories, df_left_wrong, df_right)

    def test_get_peaks(self):
        """
        Test of the `get_peaks` function.
        """
        # Creation of a serie with peaks 9 and 12.
        values = [4, 5, 9, 3, 2, 1, 2, 1, 3, 4, 12, 9, 6, 3, 2, 4, 5]
        dates = pd.date_range('2019-06-20', periods=len(values), freq='T')
        s = pd.Series(values, index=dates)

        # Compute the peaks.
        peak_dates, peak_values = get_peaks(s)

        peak_dates_res = [np.datetime64('2019-06-20T00:02'),
                          np.datetime64('2019-06-20T00:10')]
        peak_values_res = [9., 12.]

        assert_array_equal(peak_dates, peak_dates_res)
        assert_array_equal(peak_values, peak_values_res)

        # Check assertion if index is not of type `datetime`.
        self.assertRaises(AssertionError, get_peaks, pd.Series(values, index=range(len(values))))

    def test_idict(self):
        """
        Test of the `idict` function.
        """
        valid_dict = {1: 4, 2: 5, 3: 6}
        another_valid_dict = {'1': 4, 2: '5', 3: '6'}
        dataloss_dict = {1: 4, 2: 4, 3: 6}
        invalid_dict = {1: [1], 2: [2], 3: [3]}

        self.assertEqual(idict(valid_dict), {4: 1, 5: 2, 6: 3})
        self.assertEqual(idict(another_valid_dict), {4: '1', '5': 2, '6': 3})
        self.assertEqual(idict(dataloss_dict), {4: 2, 6: 3})
        self.assertRaises(TypeError, idict, invalid_dict)

    def test_mem_usage_pd(self):
        """
        Test of the `mem_usage_pd` function.
        """
        df = pd.DataFrame({'A': [f'value{i}' for i in range(100000)],
                           'B': [i for i in range(100000)],
                           'C': [float(i) for i in range(100000)]}).set_index('A')

        test_1 = mem_usage_pd(df)
        res_1 = {'total': '7.90 MB'}
        self.assertDictEqual(test_1, res_1)

        test_2 = mem_usage_pd(df, details=True)
        res_2 = {'Index': {'6.38 MB', 'Index type'},
                 'B': {'0.76 MB', np.dtype('int64')},
                 'C': {'0.76 MB', np.dtype('float64')},
                 'total': '7.90 MB'}
        self.assertDictEqual(test_2, res_2)

        serie = df.reset_index()['B']

        test_3 = mem_usage_pd(serie)
        res_3 = {'total': '0.76 MB'}
        self.assertDictEqual(test_3, res_3)

        # Check the warning message using a mock.
        with unittest.mock.patch('logging.Logger.warning') as mock_logging:
            mem_usage_pd(serie, details=True)
            mock_logging.assert_called_with('Details is only available for DataFrames.')

        # Check for exception if not a pandas object.
        self.assertRaises(AttributeError, mem_usage_pd, {'a': 1, 'b': 2})

    def test_parse_date(self):
        """
        Test of the `parse_date` decorator.
        """
        # Creation of a dummy function to apply the decorator on.
        @parse_date
        def dummy_function(**kwargs):
            return kwargs

        list_parses = ['20190325',
                       'Mon, 21 March, 2015',
                       '2019-03-09 08:03:01',
                       'March 27 2019']

        list_results = [datetime.datetime(2019, 3, 25, 0, 0),
                        datetime.datetime(2015, 3, 21, 0, 0),
                        datetime.datetime(2019, 3, 9, 8, 3, 1),
                        datetime.datetime(2019, 3, 27, 0, 0)]

        for parse, result in zip(list_parses, list_results):
            self.assertEqual(dummy_function(date=parse)['date'], result)

        # Should work with custom fields for date.
        # Creation of a dummy function with custom fields for the date.
        @parse_date(date_fields=['date_start', 'date_end'])
        def dummy_function_custom(**kwargs):
            return kwargs

        parse_1 = dummy_function_custom(date_start='20181008',
                                        date_end='2019-03-09')
        self.assertEqual(parse_1['date_start'], datetime.datetime(2018, 10, 8))
        self.assertEqual(parse_1['date_end'], datetime.datetime(2019, 3, 9))

        # Should not parse if wrong format
        self.assertEqual(dummy_function(date='wrong format')['date'],
                         'wrong format')

    def test_sliding_window(self):
        """
        Test of the `sliding_window` function.
        """
        # Should work with step of 1.
        self.assertEqual(list(sliding_window('abcdef', 2, 1)),
                         ['ab', 'bc', 'cd', 'de', 'ef'])
        # Should work with numpy arrays.
        res_np_1 = list(sliding_window(np.array([1, 2, 3, 4, 5, 6]), 5, 5))
        np.testing.assert_array_equal(res_np_1[0], np.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(res_np_1[1], np.array([6]))
        # Should work when step and windows size are the same.
        # In this case, each element will only be present once.
        self.assertEqual(list(sliding_window('abcdef', 2, 2)),
                         ['ab', 'cd', 'ef'])
        # Should work with and odd number of elements.
        self.assertEqual(list(sliding_window('abcdefg', 1, 1)),
                         ['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        self.assertEqual(list(sliding_window('abcdefg', 2, 2)),
                         ['ab', 'cd', 'ef', 'g'])
        # Should work if lenght of sequence is the same as window size.
        self.assertEqual(list(sliding_window('abcdefg', 7, 3)),
                         ['abcdefg'])
        # Should work if last chunk is not full.
        self.assertEqual(list(sliding_window('abcdefgh', 6, 4)),
                         ['abcdef', 'efgh'])
        self.assertEqual(list(sliding_window('abcdefgh', 6, 5)),
                         ['abcdef', 'fgh'])
        self.assertEqual(list(sliding_window('abcdefghi', 6, 5)),
                         ['abcdef', 'fghi'])
        # Should work with longer sequence.
        seq_1 = 'abcdefghijklmnopqrstuvwxyz'
        res_1 = ['abcdef', 'ghijkl', 'mnopqr', 'stuvwx', 'yz']
        self.assertEqual(list(sliding_window(seq_1, 6, 6)), res_1)
        res_2 = ['abcdef', 'defghi', 'ghijkl', 'jklmno', 'mnopqr', 'pqrstu',
                 'stuvwx', 'vwxyz']
        self.assertEqual(list(sliding_window(seq_1, 6, 3)), res_2)

        # Check for exceptions.
        # Should raise an exception if the sequence is not iterable.
        with self.assertRaises(TypeError):
            list(sliding_window(3, 2, 1))
        # Should raise an exception if step is not an integer.
        with self.assertRaises(TypeError):
            list(sliding_window(seq_1, 2, 1.0))
        with self.assertRaises(TypeError):
            list(sliding_window(seq_1, 2, '1'))
        # Should raise an exception if window size is not an integer.
        with self.assertRaises(TypeError):
            list(sliding_window(seq_1, 2.0, 1))
        with self.assertRaises(TypeError):
            list(sliding_window(seq_1, '2', 1))
        # Should raise an exception if window size is smaller
        # than step or <= 0.
        with self.assertRaises(ValueError):
            list(sliding_window(seq_1, 2, 3))
        with self.assertRaises(ValueError):
            list(sliding_window(seq_1, -1, -1))
        # Should raise an exception if the step is smaller or equal than 0.
        with self.assertRaises(ValueError):
            list(sliding_window(seq_1, 2, 0))
        with self.assertRaises(ValueError):
            list(sliding_window(seq_1, 2, -1))
        # Should raise an exception if length of sequence
        # is smaller than the window size.
        with self.assertRaises(ValueError):
            list(sliding_window('abc', 4, 1))

    def test_value_2_list(self):
        """
        Test of the `value_2_list` function.
        """
        # A list should remain a list.
        self.assertEqual(value_2_list(seq=[1, 2, 3]), {'seq': [1, 2, 3]})
        # A single integer should result in a list with one integer.
        self.assertEqual(value_2_list(age=42), {'age': [42]})
        # A single string should result in a list with one string.
        self.assertEqual(value_2_list(name='John Doe'), {'name': ['John Doe']})
        # A tuple should remain a tuple.
        self.assertEqual(value_2_list(children=('Jane Doe', 14)),
                         {'children': ('Jane Doe', 14)})
        # A dictionary should result in a list with a dictionary.
        self.assertEqual(value_2_list(info={'name': 'John Doe', 'age': 42}),
                         {'info': [{'name': 'John Doe', 'age': 42}]})
        # Passing a non-keyword argument should raise an exception.
        self.assertRaises(TypeError, value_2_list, [1, 2, 3])
        # Passing multiple keyword arguments should work.
        self.assertEqual(value_2_list(name='John Doe', age=42,
                                      children=('Jane Doe', 14)),
                         {'name': ['John Doe'], 'age': [42],
                          'children': ('Jane Doe', 14)})


if __name__ == '__main__':
    unittest.main()
