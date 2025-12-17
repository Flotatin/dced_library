import logging
import traceback
from typing import Any, Callable

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal


logger = logging.getLogger(__name__)


class TaskSignals(QObject):
    """Signals utilisés par les tâches longues exécutées en fond."""

    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()


class TaskRunnable(QRunnable):
    """Exécute un callable dans un thread du pool Qt et relaye les signaux."""

    def __init__(self, fn: Callable, *args: Any, **kwargs: Any):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()

    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            logger.exception("Background task failed")
            self.signals.error.emit(traceback.format_exc())
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
