import logging
from typing import Any, List

from PyQt5.QtCore import QThreadPool, QTimer, Qt
from PyQt5.QtWidgets import QApplication

from background_tasks import TaskRunnable

logger = logging.getLogger(__name__)


class BackgroundTaskMixin:
    def _init_task_state(self) -> None:
        self.thread_pool = QThreadPool.globalInstance()
        self._running_tasks = []
        self._disabled_widgets = []
        self._task_watchdogs = {}
        self._busy_task_count = 0
        self._busy_task_labels = []

    def _task_widgets_default(self) -> List[Any]:
        return [
            getattr(self, "previous_button", None),
            getattr(self, "play_stop_button", None),
            getattr(self, "next_button", None),
            getattr(self, "slider", None),
        ]

    def _set_busy_state(self, busy: bool, message: str = "", widgets=None):
        if widgets is None:
            widgets = self._task_widgets_default()

        if busy:
            label = message or "Traitement en cours…"
            self._busy_task_labels.append(label)
            self._busy_task_count += 1
            if self._busy_task_count == 1:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self._disabled_widgets = []
                for w in widgets:
                    if w is not None and w.isEnabled():
                        w.setEnabled(False)
                        self._disabled_widgets.append(w)
            if self._busy_task_count == 1:
                status = f"⏳ {label}"
            else:
                status = f"⏳ {self._busy_task_count} tâches en cours (dernier: {label})"
            self.statusBar().showMessage(status, 0)
            if hasattr(self, "text_box_msg"):
                self.text_box_msg.setText(status)
        else:
            if message and message in self._busy_task_labels:
                self._busy_task_labels.remove(message)
            elif self._busy_task_labels:
                self._busy_task_labels.pop(0)
            self._busy_task_count = max(0, self._busy_task_count - 1)
            if self._busy_task_count > 0:
                current = self._busy_task_labels[-1] if self._busy_task_labels else "Traitements en cours…"
                status = f"⏳ {self._busy_task_count} tâches en cours (actuelle: {current})"
                self.statusBar().showMessage(status, 0)
                if hasattr(self, "text_box_msg"):
                    self.text_box_msg.setText(status)
                return

            QApplication.restoreOverrideCursor()
            for w in self._disabled_widgets:
                try:
                    w.setEnabled(True)
                except Exception:
                    pass
            self._disabled_widgets = []
            done_label = message or "Traitement terminé"
            self.statusBar().showMessage(f"✅ {done_label}", 3000)
            if hasattr(self, "text_box_msg"):
                self.text_box_msg.setText(f"✅ {done_label}")

    def _submit_background_task(
        self,
        func,
        result_slot=None,
        description: str = "",
        widgets=None,
        timeout_ms: int = 20000,
    ):
        worker = TaskRunnable(func)
        if result_slot is not None:
            worker.signals.result.connect(result_slot)
        worker.signals.error.connect(
            lambda traceback_msg, desc=description: self._on_task_error(traceback_msg, desc)
        )

        def _cleanup():
            self._set_busy_state(False, message=description, widgets=widgets)
            timer = self._task_watchdogs.pop(worker, None)
            if timer:
                timer.stop()
                timer.deleteLater()
            try:
                self._running_tasks.remove(worker)
            except ValueError:
                pass

        worker.signals.finished.connect(_cleanup)

        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(
            lambda: logger.warning("Long running task (> %sms): %s", timeout_ms, description)
        )
        timer.start(timeout_ms)
        self._task_watchdogs[worker] = timer

        self._set_busy_state(True, message=description, widgets=widgets)
        self.thread_pool.start(worker)
        self._running_tasks.append(worker)

    def _on_task_error(self, traceback_msg: str, description: str = ""):
        logger.error("Background task error (%s):\n%s", description or "task", traceback_msg)
        label = description or "tâche en arrière-plan"
        self.text_box_msg.setText(f"❌ Erreur pendant « {label} ». Voir logs.")
        if "Sauvegarde CEDd" in label and hasattr(self, "_save_cedd_in_progress"):
            self._save_cedd_in_progress = False
