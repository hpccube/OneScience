import logging

from onescience.datapipes.meta import DatapipeMetaData


class Datapipe:
    """The base class for all datapipes in onescience.

    Parameters
    ----------
    meta : DatapipeMetaData, optional
        Meta data class for storing info regarding model, by default None
    """

    def __init__(self, meta: DatapipeMetaData = None):
        super().__init__()

        if not meta or not isinstance(meta, DatapipeMetaData):
            self.meta = DatapipeMetaData()
        else:
            self.meta = meta

        self.logger = logging.getLogger("core.datapipe")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)

    def debug(self):
        """Turn on debug logging"""
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[%(asctime)s - %(levelname)s - {self.meta.name}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        # TODO: set up debug log
