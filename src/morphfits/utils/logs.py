"""Logging setup and utility functions.
"""

# Imports


from time import gmtime
import logging
import logging.handlers
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from pathlib import Path

from .. import LOGS_ROOT


# Constants


## Logging


LOGGER_FILENAME = LOGS_ROOT / "morphfits.log"
"""Default log file path.
"""


LOGGER_MAX_BYTES = 1000000
"""Default maximum bytes to be stored in each log file.
"""


LOGGER_BACKUP_COUNT = 4
"""Default number of log files before rotation.
"""


## Format


LOGGING_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
"""Default logging date format, in ISO-8601 compliant format, e.g.
`2024-05-01T12:43:21`."""


LOGGING_COLOUR_ESCAPES = {
    "BLACK": "\033[30m",
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
    "WHITE": "\033[37m",
    "RESET": "\033[39m",
}
"""Colour codes for terminal.
"""


# Classes


def resolve_record_name(name: str) -> str:
    """Resolve a log record's module name to a more legible name, if it wasn't
    already.

    Parameters
    ----------
    name : str
        Record module name, e.g. `tqdm.tqdm`.

    Returns
    -------
    str
        Legible record name.
    """
    # if "gunicorn" in name:
    #     return "GUNICORN"
    # elif "uvicorn" in name:
    #     return "UVICORN"
    # else:
    #     return name
    return name


class UTCFormatter(logging.Formatter):
    """Logging formatter which uses GMT (UTC).

    Parameters
    ----------
    logging.Formatter : class
        Base formatter class to convert log records to text.

    Attributes
    ----------
    converter : function
        Time converter.
    """

    converter = gmtime

    def __init__(self, datefmt: str = LOGGING_DATE_FORMAT):
        super().__init__(fmt="", datefmt=datefmt)


class FileFormatter(UTCFormatter):
    """Logfile formatter which uses UTC.

    Parameters
    ----------
    UTCFormatter : class
        Base formatter class which uses UTC time.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a str.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to format (line in log).

        Returns
        -------
        str
            Formatted log record.
        """
        self._style._fmt = (
            "%(asctime)-20s"
            + f"{record.levelname.ljust(8)}"
            + f"{resolve_record_name(record.name).ljust(12)}"
            + f"%(message)s"
        )
        return super().format(record)


class StreamFormatter(UTCFormatter):
    """Stream log formatter which uses UTC.

    Parameters
    ----------
    UTCFormatter : class
        Base formatter class which uses UTC time.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a coloured str for display.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to format (line in log).

        Returns
        -------
        str
            Formatted log record.
        """
        # Setup format
        format_string = f"{LOGGING_COLOUR_ESCAPES['MAGENTA']}%(asctime)-20s"

        # Set colour of logging level
        if (record.levelno == DEBUG) or (record.levelno == INFO):
            format_string += LOGGING_COLOUR_ESCAPES["BLUE"]
        elif (record.levelno == WARNING) or (record.levelno == ERROR):
            format_string += LOGGING_COLOUR_ESCAPES["YELLOW"]
        elif record.levelno == CRITICAL:
            format_string += LOGGING_COLOUR_ESCAPES["RED"]
        else:
            format_string += LOGGING_COLOUR_ESCAPES["WHITE"]

        # Set format to created string and format log line
        self._style._fmt = (
            format_string
            + f"{record.levelname.ljust(8)}"
            + f"{LOGGING_COLOUR_ESCAPES['CYAN']}{resolve_record_name(record.name).ljust(12)}"
            + f"{LOGGING_COLOUR_ESCAPES['GREEN']}%(message)s"
        )
        return super().format(record)


# Functions


def create_logger(
    logger: logging.Logger = logging.getLogger(),
    filename: str | Path = LOGGER_FILENAME,
    level: str = logging._levelToName[DEBUG],
    max_bytes: int = LOGGER_MAX_BYTES,
    backup_count: int = LOGGER_BACKUP_COUNT,
) -> logging.Logger:
    """Create or update a logger object with a rotating file and stream handler.

    Parameters
    ----------
    logger : Logger, optional
        Logger object to create or update, by default a new logger.
    filename : str | Path, optional
        Name or path to file to log to, by default `logs/ramotswe_broker.logs`.
    level : str, optional
        Logging level to output, by default `WARNING`.
    max_bytes : int, optional
        Maximum bytes to be stored in each log file, by default 1,000,000.
    backup_count : int, optional
        Count of log files before rotation (first one is overwritten with new
        data), by default 4.

    Returns
    -------
    Logger
        Logger object for this program.
    """
    # Validate file directory
    if filename == LOGGER_FILENAME:
        LOGS_ROOT.mkdir(exist_ok=True)

    # Validate level
    level = level.upper()
    if level not in [
        logging._levelToName[level] for level in [DEBUG, INFO, WARNING, ERROR, CRITICAL]
    ]:
        raise ValueError(
            f"Logging level {level} invalid; must be standard Python level."
        )

    # Setup rotating log file handler
    handler = logging.handlers.RotatingFileHandler(
        filename=filename, maxBytes=max_bytes, backupCount=backup_count
    )
    handler.setFormatter(FileFormatter())
    handler.setLevel(level)

    # Setup stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(StreamFormatter())
    stream_handler.setLevel(level)

    # Add handlers to logger and return
    logger.addHandler(handler)
    logger.addHandler(stream_handler)
    logger.setLevel(level)
    return logger
