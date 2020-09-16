"""
FancyConfig, configuration loader.

Tool to load the configuration from a configuration file (`config.yml`).
"""
from collections.abc import Mapping
import logging
from pathlib import Path
import pprint
import yaml
from typing import Union

LOGGER = logging.getLogger(__name__)


class FancyConfig(Mapping):
    """
    Class to load the configuration file.

    This class behaves like a dictionary that loads a
    configuration file in yaml format.

    If the configuration file does not exist, creates it from template.

    Examples
    --------
    >>> config = FancyConfig()
    >>> print(config)
    { 'database': { 'host': '127.0.0.1',
                    'name': 'porgs',
                    'port': 3306,
                    'pwd': 'bacca',
                    'user': 'Chew'},
      'env': 'prod',
      'imports': {'star_wars': ['ewok', 'bantha']}}
    """

    def __init__(self,
                 path_config_to_load: Union[str, Path] = (Path.home()
                                                          .joinpath('.config/fancyconfig.yml')),
                 default_config_path: Union[str, Path] = (Path(__file__).resolve()
                                                          .parent.joinpath('config.yml'))):
        """
        Initialization of configuration.

        If the folder to store the configuration does not exist, create it.
        If configuration file does not exist, copy it from default one.

        Parameters
        ----------
        path_config_to_load : Path, default '~/.config/'
            Directory to store the configuration file and load the configuration from.
        default_config_path: Path, default 'config.yml' current directory.
            Name of the configuration file.
        """
        # Create config file if does not exist.

        if isinstance(path_config_to_load, str):
            path_config_to_load = Path(path_config_to_load).resolve()

        if isinstance(default_config_path, str):
            default_config_path = Path(default_config_path).resolve()

        if not path_config_to_load.exists():
            LOGGER.info((f'Configuration file does not exist, '
                         f'creating it from {default_config_path}'))
            # Creating folder of configuration (parent of file).
            path_config_to_load.parent.mkdir(parents=True, exist_ok=True)
            # Copy the configuration file.
            path_config_to_load.write_bytes(default_config_path.read_bytes())

        with path_config_to_load.open(mode='r', encoding='utf-8') as yaml_config_file:
            self._config = yaml.safe_load(yaml_config_file)

    def __getitem__(self, item):
        """Getter of the class."""
        try:
            return self._config[item]
        except KeyError:
            LOGGER.error(f'Configuration for {item} does not exist.')

    def __iter__(self):
        """Iterator of the class."""
        return iter(self._config)

    def __len__(self):
        """Lenght of the config."""
        return len(self._config)

    def __repr__(self):
        """Representation of the config."""
        return f'{super().__repr__}\n{str(self._config)}'

    def __str__(self):
        """
        __str__ method.

        Pretty representation of the config.
        """
        pretty = pprint.PrettyPrinter(indent=2)
        return pretty.pformat(self._config)
