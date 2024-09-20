"""Time Module

"""

from datetime import datetime
from uuid import uuid4


def timestamp(ms=True):
    """

    Parameters
    ----------
    ms :

    Returns
    -------

    """
    dt = datetime.now()
    if ms:
        return dt.microsecond
    return dt


def uuid():
    """

    Returns
    -------

    """
    return str(uuid4())


def eventid():
    """

    Returns
    -------

    """
    return f"{timestamp()}-{uuid()}".replace(" ", "-")
