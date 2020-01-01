import logging
import sys
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    # filename='../log/test.log',
                    # filemode='w'
                    stream=sys.stdout
                    )

class BasicLogging():
    def __init__(self):
        BasicLogging.info('==============Init Logger to StdOut============')
    
    @staticmethod
    def set_level(level):
        logging.setLevel(level)

    @staticmethod
    def debug(msg):
        logging.debug(msg)

    @staticmethod
    def info(msg):
        logging.info(msg)

    @staticmethod
    def warning(msg):
        logging.warning(msg)

    @staticmethod
    def error(msg):
        logging.error(msg)

    @staticmethod
    def critical(msg):
        logging.critical(msg)


class FileLogging():
    def __init__(self, log_id=None, folder='~/log'):

        # expand ~ to its full path
        folder = os.path.expanduser(folder)

        if log_id is None:
          logger1 = logging.getLogger('to_screen')
          logger1.setLevel(logging.INFO)
          ch = logging.StreamHandler(sys.stdout)
          formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
          ch.setFormatter(formatter)
          logger1.addHandler(ch)
          self.logger = logger1

          self.logger.info('==============Init Logger to screen ============')
        else:
          logger2 = logging.getLogger('to_file')
          logger2.setLevel(logging.INFO)

          log_file = folder+'/{}.log'.format(log_id)
          fh = logging.FileHandler(log_file)

          formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
          fh.setFormatter(formatter)

          logger2.addHandler(fh)

          self.logger = logger2
          
          self.logger.info('==============Init Logger to '+ log_file  +'============')


    def set_level(self, level):
        """
        set the level for logger

        inputs:
          - level: a string from ['DEBUG', 'INFO', 'WARNING', 'ERROR']

        returns:
          - None. Set the logger to the specified level
        """
        if level == 'DEBUG':
            self.logger.setLevel(logging.DEBUG)
        elif level == 'INFO':
            self.logger.setLevel(logging.INFO)
        elif level == 'WARNING':
            self.logger.setLevel(logging.WARNING)
        elif level == 'ERROR':
            self.logger.setLevel(logging.ERROR)
        else:
            raise ValueError('level %s not supported!'%level)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)