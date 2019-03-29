# -*- coding: utf-8 -*-
"""Test of FancyPythonThings

This module test the various functions present in the FancyPythonThings module.
"""
import datetime
import unittest

from fancy import parse_date, value_2_list


class TestFancyPythonThings(unittest.TestCase):
    """
    Unittest of FancyPythonThings.
    """

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

    def test_plot_history(self):
        """
        Test of the plot history function.

        This is not a real unittest since this is a plot.
        The test only checks if it is running smoothly.
        """
        pass

    def test_sliding_window(self):
        """
        Test of the `sliding_window` function.
        """
        pass

    def test_value_2_list(self):
        pass

if __name__ == '__main__':
    unittest.main()
