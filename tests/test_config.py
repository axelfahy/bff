# -*- coding: utf-8 -*-
"""Test of config module

This module test the configuration loader
"""
import filecmp
from pathlib import Path
import unittest

from bff.config import FancyConfig


class TestiFancyConfig(unittest.TestCase):
    """
    Unittest of config module.
    """

    def test_create_config(self):
        """
        Test of loading the config when it does not exists.
        """
        # Default configuration
        config_default = FancyConfig()

        default_path_ok = Path(__file__).resolve().parent.parent.joinpath('bff/config.yml')
        default_path_ko = Path.home().resolve().joinpath('config.yml')

        dest = Path(__file__).resolve().parent.joinpath('config.yml')
        config_a = FancyConfig(dest, default_path_ok)

        # Check if the file is correctly copied.
        self.assertTrue(filecmp.cmp(dest, default_path_ok))

        # Check if the file is not overridden.
        path_conf_b = Path(__file__).resolve().parent.joinpath('conf_b.yml')
        with path_conf_b.open(mode='w', encoding='utf-8') as f:
            f.write('testfile: True')
        config_b = FancyConfig(dest, path_conf_b)

        self.assertFalse(filecmp.cmp(dest, path_conf_b))
        # Check that the configurations are the same.
        self.assertEqual(config_a, config_b)

        # Removes the configs.
        dest.unlink()
        path_conf_b.unlink()

        # Create a config that is not called config.
        config_c = FancyConfig(path_conf_b, default_path_ok)
        self.assertEqual(config_a, config_c)

        # Remove the files.
        path_conf_b.unlink()

        # If default config path is wrong, should fail.
        with self.assertRaises(FileNotFoundError):
            FancyConfig(dest, default_path_ko)

    def test_access_config(self):
        """
        Test the access of the configuration once loaded.

        Should be accessible as a property.
        """
        default_path_ok = Path(__file__).resolve().parent.parent.joinpath('bff/config.yml')
        dest = Path(__file__).resolve().parent.joinpath('config.yml')
        config = FancyConfig(dest, default_path_ok)

        self.assertEqual(config['env'], 'prod')
        self.assertEqual(config['database']['user'], 'Chew')
        self.assertEqual(config['imports']['star_wars'], ['ewok', 'bantha'])

        # Check the error message using a mock.
        with unittest.mock.patch('logging.Logger.error') as mock_logging:
            config['error']
            mock_logging.assert_called_with('Configuration for error does not exist.')

        # Remove the file.
        dest.unlink()
