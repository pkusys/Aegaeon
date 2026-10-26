"""Logging configuration for Aegaeon."""

import logging
import sys
import os

_FORMAT = "%(levelname)s %(asctime)s %(message)s"
_DATE_FORMAT = "%H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = None
_default_handler = None
_file_handler = None


def _setup_logger():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    global _root_logger, _default_handler, _file_handler
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    if _root_logger is None:
        _root_logger = logging.getLogger("aegaeon")

        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _default_handler.setFormatter(fmt)

        log_file = os.environ.get("AEGAEON_LOG_FILE", "output.log")
        print(f"Logging to {log_file}")
        _file_handler = logging.FileHandler(log_file)
        _file_handler.setLevel(logging.INFO)
        _file_handler.setFormatter(fmt)

        _root_logger.addHandler(_default_handler)
        _root_logger.addHandler(_file_handler)

    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


def init_logger(name: str):
    return logging.getLogger(name)
