import logging
from typing import Optional, Any, Dict

import pyqtgraph as pg
from PyQt5.QtGui import QColor, QPalette

from theme_config import STYLE_TEMPLATE, THEMES, make_c_m

logger = logging.getLogger(__name__)


class ThemeMixin:
    def _init_theme_state(self, default_theme: str = "dark") -> None:
        self.current_theme = default_theme
        self._theme_cache = {}

    def _get_theme_resources(self, name: Optional[str] = None) -> Dict[str, Any]:
        if name is None:
            name = getattr(self, "current_theme", "dark")

        base_theme = THEMES.get(name, THEMES["dark"])
        cache_entry = self._theme_cache.setdefault(name, {})

        if cache_entry.get("theme") is not base_theme:
            cache_entry.update(
                {"theme": base_theme, "palette": None, "pen_cache": {}, "brush_cache": {}, "stylesheet": None}
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
        logger.warning(message)
        if hasattr(self, "text_box_msg"):
            self.text_box_msg.setText(message)

    def _require_attributes(self, attrs, context: str) -> bool:
        missing = [attr for attr in attrs if not hasattr(self, attr)]
        if missing:
            self._report_warning(f"{context} interrompu : attribut(s) manquant(s) {', '.join(missing)}.")
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
        if not hasattr(self, "theme_toggle_button"):
            return
        self.theme_toggle_button.blockSignals(True)
        self.theme_toggle_button.setChecked(self.current_theme == "light")
        self.theme_toggle_button.setText("Light mode" if self.current_theme == "light" else "Dark mode")
        self.theme_toggle_button.blockSignals(False)

    def _apply_theme_stylesheet(self, theme):
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

    def _apply_pen_mapping(self, pen_mapping, pens) -> None:
        for attr, pen_key in pen_mapping.items():
            item = getattr(self, attr, None)
            if item is None:
                continue
            item.setPen(self._mk_pen(pens[pen_key]))

    def _apply_brush_mapping(self, brush_mapping, pens) -> None:
        for attr, brush_key in brush_mapping.items():
            item = getattr(self, attr, None)
            if item is None:
                continue
            item.setBrush(self._mk_brush(pens[brush_key]))

    def _apply_theme_to_spectrum(self, theme):
        if not self._require_attributes(
            (
                "pg_spec",
                "pg_zoom",
                "pg_dy",
                "pg_spectrum",
                "curve_spec_data",
                "curve_spec_fit",
                "curve_spec_brut",
                "curve_spec_blfit",
                "curve_spec_pic_select",
                "curve_dy",
                "line_dy_zero",
                "curve_zoom_data",
                "curve_zoom_data_brut",
                "curve_zoom_pic",
                "vline_spec",
                "hline_spec",
                "vline_dy",
                "vline_zoom",
                "cross_zoom",
                "pg_text_label",
            ),
            "Application du thème Spectrum",
        ):
            return
        if self.pg_spec.isVisible():
            self.pg_spec.setBackground(theme["plot_background"])
        self._refresh_visible_plots((self.pg_zoom, self.pg_dy, self.pg_spectrum), theme)
        self._apply_pen_mapping(self._spectrum_pen_mapping, theme["pens"])
        self._apply_brush_mapping(self._spectrum_brush_mapping, theme["pens"])
        self.pg_text_label.setColor(theme["pens"]["text_item"])

    def _apply_theme_to_ddac(self, theme):
        if not self._require_attributes(
            (
                "pg_ddac",
                "pg_P",
                "pg_dPdt",
                "pg_sigma",
                "pg_movie",
                "pg_dlambda",
                "line_t_P",
                "line_t_dPdt",
                "line_t_sigma",
                "line_t_dlambda",
                "line_t_spec_dlambda",
                "line_nspec",
                "line_p0",
                "scatter_P",
                "scatter_dPdt",
                "scatter_sigma",
                "scatter_dlambda",
            ),
            "Application du thème dDAC",
        ):
            return
        if self.pg_ddac.isVisible():
            self.pg_ddac.setBackground(theme["plot_background"])
        self._refresh_visible_plots((self.pg_P, self.pg_dPdt, self.pg_sigma, self.pg_movie, self.pg_dlambda), theme)
        self._apply_pen_mapping(self._ddac_pen_mapping, theme["pens"])
        if hasattr(self, "c_m_base"):
            self.c_m_base = make_c_m(self.current_theme)
        self._recolor_all_runs()

    def _stylize_plot(self, plot_item, x_label=None, y_label=None, show_grid=True):
        if plot_item is None:
            return
        if x_label:
            plot_item.setLabel("bottom", x_label)
        if y_label:
            plot_item.setLabel("left", y_label)
        if show_grid:
            plot_item.showGrid(x=True, y=True, alpha=self._get_theme()["grid_alpha"])
        plot_item.setMouseEnabled(x=True, y=True)

    def _create_marker_line(
        self, *, angle: float, pos=None, movable: bool = False, pen_key: str = "selection_line", z_value: Optional[float] = None
    ):
        pen_spec = self._get_theme()["pens"].get(pen_key, {"color": "#ffff00"})
        line = pg.InfiniteLine(pos, angle=angle, movable=movable, pen=self._mk_pen(pen_spec))
        if z_value is not None:
            line.setZValue(z_value)
        return line

    def _create_scatter_marker(
        self,
        *,
        symbol: str = "+",
        size: int = 10,
        pen_key: str = "scatter",
        brush=None,
        z_value: Optional[float] = None,
    ):
        pen_spec = self._get_theme()["pens"].get(pen_key, {"color": "#ffffff", "width": 2})
        scatter = pg.ScatterPlotItem(
            x=[], y=[], pen=self._mk_pen(pen_spec), brush=self._mk_brush(brush) if brush is not None else None, size=size, symbol=symbol
        )
        if z_value is not None:
            scatter.setZValue(z_value)
        return scatter
