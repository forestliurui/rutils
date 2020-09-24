import logging
import logging.config
import os

logger = logging.getLogger(__name__)

"""
To use logger, add the following in a file
    import logging
    logger = logging.getLogger(__name__)

    def func():
        something
        logger.info("msg")
        something

To set up the logger, add the following in a file
   import logging

   from logger import setup_logger
   logger = logging.getLogger(__name__)

   def func():
       setup_logger("log_id")
       logger.debug("debug_msg")
"""

def setup_default():
    logging.config.fileConfig('./logging.conf', disable_existing_loggers=False)

def setup_logger(log_id="temp", folder="~/log"):
    # change the path of the log file
    folder = os.path.expanduser(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = "{}.log".format(log_id)
    filepath = os.path.join(folder, filename)

    import configparser
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(__file__), "logging.conf"))
    config["handler_fileHandler"]["args"] = "('{}',)".format(filepath)
    logging.config.fileConfig(config, disable_existing_loggers=False)
    logger.info("==============Init Logger to {}============".format(filepath))