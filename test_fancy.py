# -*- coding: utf-8 -*-
"""Test of FancyPythonThings

This module test the various functions present in the FancyPythonThings module.
"""
import datetime
import unittest

from fancy import parse_date, value_2_list, sliding_window


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
        self.assertEqual(list(sliding_window('abcdef', 2, 1)), ['ab', 'bc', 'cd', 'de', 'ef'])

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
