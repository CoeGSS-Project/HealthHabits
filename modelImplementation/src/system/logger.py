import sys
import os


class Logger(object):
    def __init__(self, target="stdout", loglevel="all"):
        self._target = target
        self._loglevel = loglevel

        if self._target == "stdout":
            self._logCall = self._logSTD
        else:
            self._logCall = self._logSTD

    def log(self, msg):
        self._logCall(msg)

    def _logSTD(self, msg):
        sys.stdout.write(msg)
        sys.stdout.flush()
