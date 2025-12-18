import logging
from typing import Optional, Any, Dict, List

import pyqtgraph as pg
from PyQt5.QtCore import QThreadPool, QTimer, Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QApplication

from background_tasks import TaskRunnable
from theme_config import STYLE_TEMPLATE, THEMES, make_c_m

logger = logging.getLogger(__name__)


class ThemeMixin:
    def _init_theme_state(self, default_theme: str = "dark") -> None:
        self.current_theme = default_theme
        self._theme_cache = {}

    # -----------------------------
    # Gestion du thème / palette Qt
    # -----------------------------
    def _get_theme_resources(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Récupère (ou construit) les ressources mises en cache pour un thème."""

        if name is None:
            name = getattr(self, "current_theme", "dark")

        base_theme = THEMES.get(name, THEMES["dark"])
        cache_entry = self._theme_cache.setdefault(name, {})

        if cache_entry.get("theme") is not base_theme:
            cache_entry.update(
                {
                    "theme": base_theme,
                    "palette": None,
                    "pen_cache": {},
                    "brush_cache": {},
                    "stylesheet": None,
                }
            )

        if cache_entry.get("palette") is None:
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor(base_theme["window"]))
            palette.setColor(QPalette.WindowText, QColor(base_theme["text"]))
            palette.setColor(QPalette.Base, QColor(base_theme["background"]))
            palette.setColor(QPalette.Text, QColor(base_theme["text"]))
            cache_entry["palette"] = palette

        return cache_entry

    def _get_theme(self, name: Optional[str] = None):
        return self._get_theme_resources(name)["theme"]

    def _cache_key(self, spec: Any):
        if isinstance(spec, dict):
            try:
                return ("dict", tuple(sorted(spec.items())))
            except TypeError:
                return ("dict_repr", repr(spec))
        return ("val", spec)

    def _mk_pen(self, spec, theme_name: Optional[str] = None):
        cache_entry = self._get_theme_resources(theme_name)
        key = self._cache_key(spec)
        pens = cache_entry["pen_cache"]
        if key not in pens:
            pens[key] = pg.mkPen(**spec) if isinstance(spec, dict) else pg.mkPen(spec)
        return pens[key]

    def _mk_brush(self, spec, theme_name: Optional[str] = None):
        cache_entry = self._get_theme_resources(theme_name)
        key = self._cache_key(spec)
        brushes = cache_entry["brush_cache"]
        if key not in brushes:
            brushes[key] = pg.mkBrush(spec)
        return brushes[key]

    def _report_warning(self, message: str):
        """Loggue un avertissement et l'affiche dans la zone de statut si possible."""

        logger.warning(message)
        if hasattr(self, "text_box_msg"):
            self.text_box_msg.setText(message)

    def _require_attributes(self, attrs, context: str) -> bool:
        """Vérifie la présence des attributs indispensables et loggue si absent."""

        missing = [attr for attr in attrs if not hasattr(self, attr)]
        if missing:
            self._report_warning(
                f"{context} interrompu : attribut(s) manquant(s) {', '.join(missing)}."
            )
            return False
        return True

    def _build_stylesheet(self, theme):
        try:
            return STYLE_TEMPLATE.safe_substitute(
                window=theme["window"],
                background=theme["background"],
                text=theme["text"],
                accent=theme["accent"],
                accent_hover=theme["accent_hover"],
                accent_pressed=theme["accent_pressed"],
                input_background=theme["input_background"],
                selection=theme["selection"],
                selection_text=theme["selection_text"],
                menu_background=theme["menu_background"],
                button_text=theme["button_text"],
            )
        except Exception as exc:
            self._report_warning(f"Failed to build stylesheet: {exc}")
            return ""

    def _apply_plot_item_theme(self, plot_item, theme):
        """Applique le thème à un objet PyQtGraph :
        - GraphicsLayoutWidget
        - PlotItem
        - ViewBox
        """

        if plot_item is None:
            return

        if isinstance(plot_item, pg.GraphicsLayoutWidget):
            vb = plot_item.ci.getViewBox()
            if vb is not None:
                vb.setBackgroundColor(theme["plot_background"])
            return

        if isinstance(plot_item, pg.ViewBox):
            plot_item.setBackgroundColor(theme["plot_background"])
            return

        vb = plot_item.getViewBox()
        if vb is not None and hasattr(vb, "setBackgroundColor"):
            vb.setBackgroundColor(theme["plot_background"])

        axis_pen = self._mk_pen(theme["axis_pen"])
        text_pen = self._mk_pen(theme["text"])

        for name in ("bottom", "left", "right", "top"):
            axis = plot_item.getAxis(name)
            if axis is not None:
                axis.setPen(axis_pen)
                axis.setTextPen(text_pen)

        plot_item.showGrid(x=True, y=True, alpha=theme["grid_alpha"])

    def _apply_theme_toggle_state(self):
        """Met à jour l'interrupteur light/dark sans déclencher de signaux."""

        if not hasattr(self, "theme_toggle_button"):
            return

        self.theme_toggle_button.blockSignals(True)
        self.theme_toggle_button.setChecked(self.current_theme == "light")
        self.theme_toggle_button.setText(
            "Light mode" if self.current_theme == "light" else "Dark mode"
        )
        self.theme_toggle_button.blockSignals(False)

    def _apply_theme_stylesheet(self, theme):
        """Applique la feuille de style Qt en fonction du thème calculé."""

        cache_entry = self._get_theme_resources(self.current_theme)

        if not cache_entry.get("stylesheet"):
            cache_entry["stylesheet"] = self._build_stylesheet(theme)

        stylesheet = cache_entry["stylesheet"]
        if not stylesheet:
            self._report_warning("Feuille de style vide : thème invalide ou incomplet.")
            return

        self.setPalette(cache_entry["palette"])
        self.setStyleSheet(stylesheet)

    def _refresh_visible_plots(self, plots, theme):
        for plot in plots:
            if plot is None:
                continue
            if hasattr(plot, "isVisible") and not plot.isVisible():
                continue
            self._apply_plot_item_theme(plot, theme)

    def _apply_theme(self, theme_name: str):
        self.current_theme = theme_name if theme_name in THEMES else "dark"
        theme = self._get_theme(self.current_theme)

        self._apply_theme_toggle_state()
        self._apply_theme_stylesheet(theme)
        self._apply_theme_to_spectrum(theme)
        self._apply_theme_to_ddac(theme)
        self._apply_theme_to_text_view(theme)
        self._apply_axis_fonts()

    def _toggle_theme(self, checked: bool):
        self._apply_theme("light" if checked else "dark")

    def _apply_theme_to_text_view(self, theme):
        if hasattr(self, "pg_text"):
            self.pg_text.setBackgroundColor(theme["plot_background"])

    def _apply_axis_fonts(self):
        font = pg.QtGui.QFont("Segoe UI", 11)

        def _set_axes_font(plot_item):
            if plot_item is None:
                return
            for name in ("bottom", "left"):
                axis = plot_item.getAxis(name)
                if axis is not None:
                    axis.setTickFont(font)

        for plot in (
            getattr(self, "pg_zoom", None),
            getattr(self, "pg_dy", None),
            getattr(self, "pg_spectrum", None),
            getattr(self, "pg_P", None),
            getattr(self, "pg_dPdt", None),
            getattr(self, "pg_sigma", None),
            getattr(self, "pg_dlambda", None),
        ):
            _set_axes_font(plot)

    def _get_run_id(self, ced):
        if hasattr(ced, "CEDd_path") and ced.CEDd_path:
            return ced.CEDd_path
        return f"memory_{id(ced)}"

    def _apply_theme_to_spectrum(self, theme):
        raise NotImplementedError

    def _apply_theme_to_ddac(self, theme):
        raise NotImplementedError


class BackgroundTaskMixin:
    def _init_task_state(self) -> None:
        self.thread_pool = QThreadPool.globalInstance()
        self._running_tasks = []
        self._disabled_widgets = []
        self._task_watchdogs = {}

    def _task_widgets_default(self) -> List[Any]:
        """Widgets désactivés pendant l'exécution d'une tâche en arrière-plan."""

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
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage(message, 5000)
            self._disabled_widgets = []
            for w in widgets:
                if w is not None and w.isEnabled():
                    w.setEnabled(False)
                    self._disabled_widgets.append(w)
        else:
            QApplication.restoreOverrideCursor()
            for w in self._disabled_widgets:
                try:
                    w.setEnabled(True)
                except Exception:
                    pass
            self._disabled_widgets = []

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
        worker.signals.error.connect(self._on_task_error)

        def _cleanup():
            self._set_busy_state(False, widgets=widgets)
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

    def _on_task_error(self, traceback_msg: str):
        logger.error("Background task error:\n%s", traceback_msg)
        self.text_box_msg.setText("Erreur dans une tâche en arrière-plan. Voir logs.")


class RunStateMixin:
    def _init_run_state(self) -> None:
        """Initialise l'état partagé entre les vues Spectrum et dDAC."""

        self.runs: Dict[str, Any] = {}
        self.current_run_id: Optional[str] = None
        self.file_index_map: Dict[str, int] = {}
