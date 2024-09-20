""" JSON Module

"""
from abc import ABC, abstractmethod

import simplejson


class Serializable(ABC):
    """Python Interface to export Python objects to JSON format.

    The child class must implement 'dumps' property.
    """

    @abstractmethod
    def dumps(self) -> dict:
        """Property to generate the JSON formatted data of the current instance."""
        pass


def json_read(filepath) -> dict:
    """Read the JSON file with the given filepath.

    Parameters
    ----------
    filepath :

    Returns
    -------
    dict
        The JSON data of the given file as a dict or list.

    """
    with open(filepath, "rb") as json_file:
        json_data = simplejson.load(json_file)
        return json_data


def json_write(filepath, json_data, indent=4):
    """Write the given json_data to the JSON file with the given filepath.

    Parameters
    ----------
    filepath :
    json_data :
    indent :
    """
    with open(filepath, "w") as json_file:
        simplejson.dump(json_data, json_file, indent=indent)
