import logging


def setup_logger(level=logging.DEBUG):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    _logger = logging.getLogger(__name__)
    _logger.setLevel(level)
    _logger.addHandler(handler)
    return _logger
