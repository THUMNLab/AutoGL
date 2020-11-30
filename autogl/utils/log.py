"""
Log utils. Used to manage the output format of AutoGL
"""

import logging


def get_logger(name):
    """
    Get the logger of given name

    Parameters
    ----------
    name: str
        The name of logger

    Returns
    -------
    logger: Logger
        The logger generated
    """
    return logging.getLogger(name)
