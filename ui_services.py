import logging
import copy
import os
import traceback
from typing import Optional, Any, Dict, List

import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from domain_math import (
    fit_score,
    gauge_lambda_from_pressure,
    gauge_pressure_from_lambda,
    ruby_lambda_from_temperature,
)
from ui_tasks import BackgroundTaskMixin
from ui_theme import ThemeMixin

logger = logging.getLogger(__name__)


class CedCreationServiceMixin:
    """Services métier liés à la création d'objets CEDd (hors logique UI)."""

    def _resolve_movie_folder_for_creation(self):
        return self.loaded_filename_movie if self.chk_use_movie.isChecked() else None

    def _build_new_cedd(self, *, fit: bool):
        from Bibli_python import CL_FD_Update as CL

        folder_movie = self._resolve_movie_folder_for_creation()
        return CL.CEDd(
            self.loaded_filename_spectro,
            Gauges_init=copy.deepcopy(self.Spectrum.Gauges),
            data_Oscillo=self.loaded_filename_oscilo,
            folder_Movie=folder_movie,
            fit=fit,
            time_index=[2, 4],
            type_filtre=self.Spectrum.type_filtre,
            param_f=self.Spectrum.param_f,
        )

    def _finalize_new_cedd(self, new_cedd, folder_cedd: str):
        name_ced = os.path.basename(self.loaded_filename_oscilo)
        new_cedd.CEDd_path = os.path.join(folder_cedd, (name_ced + ".CEDUp"))
        self.PRINT_CEDd(objet_run=new_cedd, item=None)

    def _corr_summary_for_specs(self, spec_indices):
        """Met à jour uniquement les lignes Summary nécessaires."""
        if self.RUN is None:
            return

        if isinstance(spec_indices, int):
            indices = [spec_indices]
        else:
            indices = sorted({int(i) for i in spec_indices if i is not None})

        if not indices:
            return

        for idx in indices:
            if 0 <= idx < len(self.RUN.Spectra):
                self.RUN.Corr_Summary(num_spec=idx, All=False)

        if (
            hasattr(self, "_summary_needs_full_rebuild")
            and self._summary_needs_full_rebuild(self.RUN)
        ):
            self.RUN.Corr_Summary(All=True)

    def _mark_summary_dirty(self, spec_indices):
        """Marque des index de spectres comme modifiés (dirty)."""
        if isinstance(spec_indices, int):
            self._summary_dirty_indices.add(int(spec_indices))
            return
        for idx in spec_indices:
            if idx is not None:
                self._summary_dirty_indices.add(int(idx))

    def _flush_summary_dirty(self):
        """Applique Corr_Summary uniquement sur les index dirty, puis vide la file."""
        if not self._summary_dirty_indices:
            return
        indices = sorted(self._summary_dirty_indices)
        self._summary_dirty_indices.clear()
        self._corr_summary_for_specs(indices)


class FileFolderServiceMixin:
    """Services de gestion des fichiers/dossiers CEDd."""

    def f_load_latest_file(self, folder, extend, dir_name, dir_label):
        from Bibli_python import CL_FD_Update as CL

        if not folder or not os.path.isdir(folder):
            dir_label.setText(":folder missing")
            return None

        dir_name, f = CL.Load_last(Folder=folder, extend=extend)
        if dir_name is None:
            dir_label.setText(":no file")
            return None
        dir_label.setText(f":{f}")
        return dir_name

    def load_latest_file(self):
        from datetime import datetime

        folder_start_spectro = os.path.join(self.folder_start, "Aquisition_ANDOR_Banc_CEDd")
        folders = [d for d in os.listdir(folder_start_spectro) if os.path.isdir(os.path.join(folder_start_spectro, d))]
        folders_dates = []
        for fold in folders:
            try:
                datetime.strptime(fold, "%y%m%d")
                folders_dates.append(fold)
            except ValueError:
                pass

        if folders_dates:
            latest_folder = max(folders_dates, key=lambda d: datetime.strptime(d, "%y%m%d"))
            latest_folder_spectro = os.path.join(folder_start_spectro, latest_folder)
            self.loaded_filename_spectro = self.f_load_latest_file(
                latest_folder_spectro, ".asc", self.loaded_filename_spectro, self.dir_label_spectro
            )
            self.f_data_spectro()

        self.loaded_filename_oscilo = self.f_load_latest_file(
            os.path.join(self.folder_start, "Aquisition_LECROY_Banc_CEDd"),
            None,
            self.loaded_filename_oscilo,
            self.dir_label_oscilo,
        )
        self.loaded_filename_movie = self.f_load_latest_file(
            os.path.join(self.folder_start, "Aquisition_PHANTOME_Banc_CEDd"),
            ".cine",
            self.loaded_filename_movie,
            self.dir_label_movie,
        )

    def f_filter_files(self):
        filter_text = self.search_bar.text().lower()
        filtered_files = [os.path.basename(f) for f in self.liste_chemins_fichiers if filter_text in os.path.basename(f)]
        self.liste_fichiers.clear()
        self.liste_fichiers.addItems(filtered_files)

    def parcourir_dossier(self):
        options = QFileDialog.Options()
        self.dossier_selectionne = QFileDialog.getExistingDirectory(self, "Sélectionner un dossier", options=options)
        if self.dossier_selectionne:
            with os.scandir(self.dossier_selectionne) as entries:
                files = sorted(
                    [entry for entry in entries if entry.is_file()],
                    key=lambda entry: entry.stat().st_ctime_ns,
                    reverse=True,
                )
            self.liste_fichiers.clear()
            self.liste_fichiers.addItems([entry.name for entry in files])
            self.liste_chemins_fichiers = [entry.path for entry in files]

    def f_select_directory(self, file_name, file_label, name, type_file=".asc"):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter(f"Text Files (*{type_file});;All Files (*)")
        dialog.setViewMode(QFileDialog.Detail)

        if file_name is None:
            dialog.setDirectory(self.folder_start)
        else:
            dialog.setDirectory(os.path.dirname(file_name))

        if dialog.exec_():
            file_name = dialog.selectedFiles()[0]
            file_label.setText(f"Selected directory: {os.path.basename(file_name)}")
            return file_name

    def f_data_spectro(self):
        import numpy as np
        import pandas as pd

        if self.loaded_filename_spectro:
            try:
                file_stat = os.stat(self.loaded_filename_spectro)
                cache_hit = (
                    self._spectro_cache["path"] == self.loaded_filename_spectro
                    and self._spectro_cache["mtime_ns"] == file_stat.st_mtime_ns
                    and self._spectro_cache["size"] == file_stat.st_size
                    and self._spectro_cache["dataframe"] is not None
                )
                if cache_hit:
                    self.data_Spectro = self._spectro_cache["dataframe"].copy(deep=False)
                else:
                    self.data_Spectro = pd.read_csv(
                        self.loaded_filename_spectro,
                        sep=r"\s+",
                        header=None,
                        skiprows=43,
                        engine="python",
                    )
                    self._spectro_cache.update(
                        {
                            "path": self.loaded_filename_spectro,
                            "mtime_ns": file_stat.st_mtime_ns,
                            "size": file_stat.st_size,
                            "dataframe": self.data_Spectro,
                        }
                    )
            except Exception:
                self.text_box_msg.setText("ERROR FILE")
                return

        if len(self.data_Spectro.columns) == 2:
            wave = self.data_Spectro.iloc[:, 0]
            iua = self.data_Spectro.iloc[:, 1]
            wave_unique = np.unique(wave)
            num_spec = len(wave) // len(wave_unique)
            if num_spec >= 1:
                iua = iua.values.reshape(num_spec, len(wave_unique)).T
                self.data_Spectro = pd.DataFrame(
                    np.column_stack([wave_unique, iua]),
                    columns=[0] + [i + 1 for i in range(num_spec)],
                )

        n_spec = max(0, self.data_Spectro.shape[1] - 1)
        if hasattr(self, "spinbox_spec_index"):
            self.spinbox_spec_index.blockSignals(True)
            self.spinbox_spec_index.setRange(0, max(0, n_spec - 1))
            self.spinbox_spec_index.setValue(0)
            self.spinbox_spec_index.blockSignals(False)
        self.text_box_msg.setText(f"{n_spec} spectres chargés depuis le fichier spectro.")

    def select_spectro_file(self):
        self.loaded_filename_spectro = self.f_select_directory(
            self.loaded_filename_spectro, self.dir_label_spectro, "Spectrum", type_file=".asc"
        )
        self.f_data_spectro()

    def select_oscilo_file(self):
        self.loaded_filename_oscilo = self.f_select_directory(
            self.loaded_filename_oscilo, self.dir_label_oscilo, "Oscilo", type_file=""
        )

    def select_movie_file(self):
        self.loaded_filename_movie = self.f_select_directory(
            self.loaded_filename_movie, self.dir_label_movie, "Movie", type_file=".cine"
        )


class SpectrumWorkflowMixin:
    """Services liés à la création/sauvegarde de spectres."""

    def _plot_gauge_reference_lines(self, gauge, x_center):
        """Trace les lignes verticales des pics d'une jauge autour de ``x_center``."""
        m = (
            float(max(self.Spectrum.y_corr))
            if self.Spectrum is not None and self.Spectrum.y_corr is not None
            else 1.0
        )
        self.f_dell_lines()
        for i, _ in enumerate(gauge.pics):
            ctr = x_center + gauge.deltaP0i[i][0]
            y_top = m * gauge.deltaP0i[i][1]
            item = self.pg_spectrum.plot(
                [ctr, ctr],
                [0.0, y_top],
                pen=pg.mkPen(gauge.color_print[0], width=2),
            )
            self.lines.append(item)

    def CREAT_new_Spectrum(self):
        from Bibli_python import CL_FD_Update as CL
        import numpy as np

        save_gauges = []
        if isinstance(self.Spectrum, CL.Spectre):
            save_gauges = copy.deepcopy(self.Spectrum.Gauges)

        self.CLEAR_ALL(empty=False)
        self.bit_bypass = True

        if hasattr(self, "spinbox_spec_index"):
            idx = int(self.spinbox_spec_index.value())
            n_spec = idx + 1
        else:
            n_spec = 1

        max_col = self.data_Spectro.shape[1] - 1
        n_spec = max(1, min(n_spec, max_col))

        self.text_box_msg.setText(f"New spec n°{n_spec}")
        x = np.array(self.data_Spectro[0])
        y = np.array(self.data_Spectro[n_spec])

        new_spectrum = CL.Spectre(x, y, Gauges=save_gauges)

        try:
            self.LOAD_Spectrum(Spectrum=new_spectrum)
        except TypeError:
            self.Spectrum = new_spectrum
            self.LOAD_Spectrum()

        self.bit_bypass = False

    def SAVE_CEDd(self, run_async: bool = True):
        from Bibli_python import CL_FD_Update as CL

        if getattr(self, "_save_cedd_in_progress", False):
            self.text_box_msg.setText("⏳ Sauvegarde CEDd déjà en cours…")
            return

        run_snapshot = copy.deepcopy(self.RUN)
        run_snapshot.Spectra[self.index_spec] = copy.deepcopy(self.Spectrum)

        if run_async and hasattr(self, "_submit_background_task"):
            self._save_cedd_in_progress = True
            self._submit_background_task(
                lambda: CL.SAVE_CEDd(run_snapshot),
                result_slot=self._on_save_cedd_done,
                description="Sauvegarde CEDd…",
            )
            return

        self._save_cedd_in_progress = True
        try:
            CL.SAVE_CEDd(run_snapshot)
            self._on_save_cedd_done(None)
        finally:
            self._save_cedd_in_progress = False

    @pyqtSlot(object)
    def _on_save_cedd_done(self, _payload):
        self._save_cedd_in_progress = False
        self.text_box_msg.setText("Save CEDd SUCCES")

    def f_dell_lines(self):
        for item in self.lines:
            try:
                self.pg_spectrum.removeItem(item)
            except Exception:
                pass
        self.lines = []

    def f_p_move(self, J_select, value):
        J_select.P = round(value, 3)
        try:
            x, self.deltalambdaP = gauge_lambda_from_pressure(J_select, value)
        except Exception:
            x = 0.0
            self.deltalambdaP = 0.0

        self._plot_gauge_reference_lines(J_select, x)

        self.bit_bypass = True
        try:
            self.spinbox_x.setValue(x + self.deltalambdaT)
        finally:
            self.bit_bypass = False
        return J_select

    def spinbox_p_move(self, value):
        if self.bit_modif_PTlambda:
            return
        try:
            if self.is_loading_gauge:
                self.Gauge_select.lamb_fit = self.Gauge_select.inv_f_P(value) + self.deltalambdaT
                self.Gauge_select = self.f_p_move(self.Gauge_select, value)
            if self.bit_modif_jauge and self.Spectrum is not None and self.index_jauge >= 0:
                g = self.Spectrum.Gauges[self.index_jauge]
                g.lamb_fit = g.inv_f_P(value) + self.deltalambdaT
                self.Spectrum.Gauges[self.index_jauge] = self.f_p_move(g, value)
            self.save_value = value
        except Exception as e:
            print("spinbox_p_move error:", e)

    def f_x_move(self, J_select, value):
        J_select.lamb_fit = value
        try:
            J_select.P = gauge_pressure_from_lambda(J_select, value, self.deltalambdaT)
        except Exception:
            J_select.P = 0.0

        self._plot_gauge_reference_lines(J_select, value)

        self.bit_modif_PTlambda = True
        try:
            self.spinbox_P.setValue(J_select.P)
        finally:
            self.bit_modif_PTlambda = False
        return J_select

    def spinbox_x_move(self, value):
        if self.bit_modif_PTlambda:
            return
        try:
            if self.is_loading_gauge:
                self.Gauge_select.lamb_fit = value
                self.Gauge_select = self.f_x_move(self.Gauge_select, value)
            if self.bit_modif_jauge and self.Spectrum is not None and self.index_jauge >= 0:
                g = self.Spectrum.Gauges[self.index_jauge]
                g.lamb_fit = value
                self.Spectrum.Gauges[self.index_jauge] = self.f_x_move(g, value)
            self.save_value = value
        except Exception as e:
            print("spinbox_x_move error:", e)

    def f_t_move(self, J_select, value):
        J_select.T = round(value, 3)
        try:
            x, self.deltalambdaT = ruby_lambda_from_temperature(J_select, value)
        except Exception:
            x = 0.0
            self.deltalambdaT = 0.0

        self._plot_gauge_reference_lines(J_select, x)

        self.bit_modif_PTlambda = True
        try:
            self.spinbox_x.setValue(x)
        finally:
            self.bit_modif_PTlambda = False
        return J_select

    def spinbox_t_move(self, value):
        if self.bit_modif_PTlambda:
            return
        try:
            if self.is_loading_gauge:
                self.Gauge_select.lamb_fit = round(
                    ruby_lambda_from_temperature(self.Gauge_select, value)[0]
                )
                self.Gauge_select = self.f_t_move(self.Gauge_select, value)

            if self.bit_modif_jauge and self.Spectrum is not None and self.index_jauge >= 0:
                g = self.Spectrum.Gauges[self.index_jauge]
                g.lamb_fit = round(ruby_lambda_from_temperature(g, value)[0], 3)
                self.Spectrum.Gauges[self.index_jauge] = self.f_t_move(g, value)
            self.save_value = value
        except Exception as e:
            print("spinbox_t_move error:", e)


class GaugeWorkflowMixin:
    """Gestion métier des jauges (sélection, ajout, suppression, méta-données)."""

    def f_gauge_select(self):
        from Bibli_python import CL_FD_Update as CL

        col1 = self.Gauge_type_selector.model().item(
            self.Gauge_type_selector.currentIndex()
        ).background().color().getRgb()
        self.Gauge_type_selector.setStyleSheet(
            "background-color: rgba{};\tselection-background-color: gray;".format(col1)
        )
        go = False
        new_g = self.Gauge_type_selector.currentText()
        if self.Spectrum is None:
            go = True
        elif new_g not in [g.name for g in self.Spectrum.Gauges]:
            go = True

        if go:
            self.is_loading_gauge = True
            self.bit_modif_jauge = False
            self.Gauge_select = CL.Gauge(name=new_g)
            self.Gauge_select.P = self.spinbox_P.value()
            self.lamb0_entry.setText(str(self.Gauge_select.lamb0))
            self.name_spe_entry.setText(str(self.Gauge_select.name_spe))
            self.f_dell_lines()
            self.f_p_move(self.Gauge_select, value=self.Gauge_select.P)

            self.name_gauge.setText("Add ?")
            self.name_gauge.setStyleSheet("background-color: red;")
            if self.Gauge_select.name == "Ruby":
                self.spinbox_T.setEnabled(True)
            else:
                self.spinbox_T.setEnabled(False)
                self.spinbox_T.setValue(293)
                self.deltalambdaT = 0
        else:
            self.index_jauge = [ga.name for ga in self.Spectrum.Gauges].index(
                self.Gauge_type_selector.currentText()
            )
            self.LOAD_Gauge()

    def f_lambda0(self):
        lambda0 = str(self.Spectrum.Gauges[self.index_jauge].lamb0)
        try:
            self.Spectrum.Gauges[self.index_jauge].lamb0 = float(self.lamb0_entry.text())
        except Exception as e:
            self.lamb0_entry.setText(lambda0)
            print("ERROR:", e, "in lambda0")

        if self.Gauge_init_box.isChecked() and self.RUN is not None:
            self.RUN.Gauges_init[self.index_jauge].lamb0 = float(self.lamb0_entry.text())
            self.RUN.Gauges_init[self.index_jauge].T = self.spinbox_T.value()
            self.RUN.Corr_Summary(All=True)
            self.REFRESH()

    def f_name_spe(self):
        name_spe = str(self.Spectrum.Gauges[self.index_jauge].name_spe)
        try:
            self.Spectrum.Gauges[self.index_jauge].name_spe = str(self.name_spe_entry.text())
            self.Spectrum.Gauges[self.index_jauge].spe = 0
            if self.Gauge_init_box.isChecked() and self.RUN is not None:
                self.RUN.Gauges_init[self.index_jauge].name_spe = str(self.name_spe_entry.text())
                self.Spectrum.Gauges[self.index_jauge].spe
                self.REFRESH()
        except Exception as e:
            self.name_spe_entry.setText(name_spe)
            print("ERROR:", e, "in lambda0")

    def f_index_gauge(self, spec):
        l_name = [ga.name for ga in spec.Gauges]
        try:
            index = l_name.index(self.Gauge_type_selector.currentText())
            print(index)
        except Exception:
            index = None
            self.Gauge_type_selector.setCurrentIndex(
                l_name.index(self.Gauge_type_selector.currentText())
            )
        return index

    def ADD_gauge(self):
        from Bibli_python import CL_FD_Update as CL

        new_g = self.Gauge_type_selector.currentText()
        if new_g not in self.list_name_gauges:
            self.bit_modif_jauge = False
        if self.bit_modif_jauge is True:
            return print("if you want this gauges DELL AND RELOAD")
        self.is_loading_gauge = False
        new_Jauge = CL.Gauge(name=new_g)
        new_Jauge.P = self.spinbox_P.value()
        new_Jauge.lamb_fit = new_Jauge.inv_f_P(new_Jauge.P)
        self.Spectrum.Gauges.append(new_Jauge)

        self.index_jauge = len(self.Spectrum.Gauges) - 1
        self.Update_var(new_Jauge.name)
        self.LOAD_Gauge()
        self.name_gauge.setText("In")
        self.name_gauge.setStyleSheet("background-color: green;")
        self.Auto_pic()

    def Dell_Jauge(self):
        if self.index_jauge == -1:
            return print("jauge not select")
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Warning dell Jauge")
        text = (
            "You going to dell "
            + self.Spectrum.Gauges[self.index_jauge].name
            + '\n Press "v" for Validate "c" for Cancel'
        )
        msg_box.setText(text)

        v_button = msg_box.addButton("Validate", QMessageBox.AcceptRole)
        a_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)

        msg_box.setDefaultButton(v_button)

        def on_key_press(event):
            if event.key() == Qt.Key_V:
                v_button.click()
            elif event.key() == Qt.Key_C:
                a_button.click()

        msg_box.keyPressEvent = on_key_press
        msg_box.exec_()

        if msg_box.clickedButton() == v_button:
            if self.index_jauge == 0 and len(self.Param0) > 2:
                self.z1[self.index_jauge + 1] = self.z1[self.index_jauge]
                self.z2[self.index_jauge + 1] = self.z2[self.index_jauge]
                (
                    self.X_s[self.index_jauge + 1],
                    self.X_e[self.index_jauge + 1],
                    self.Zone_fit[self.index_jauge + 1],
                ) = (
                    self.X_s[self.index_jauge],
                    self.X_e[self.index_jauge],
                    self.Zone_fit[self.index_jauge],
                )
            del (self.Nom_pic[self.index_jauge])
            del (self.Param0[self.index_jauge])
            del (self.list_text_pic[self.index_jauge])
            del (self.J[self.index_jauge])
            del (self.plot_pic_fit[self.index_jauge])
            del (self.list_y_fit_start[self.index_jauge])
            del (self.list_name_gauges[self.index_jauge])
            try:
                del (self.Param_FIT[self.index_jauge])
            except Exception as e:
                print("del(Param_FIT[J])", e)
            del (self.X_s[self.index_jauge])
            del (self.X_e[self.index_jauge])
            del (self.z1[self.index_jauge])
            del (self.z2[self.index_jauge])
            del (self.Zone_fit[self.index_jauge])
            del (self.Spectrum.Gauges[self.index_jauge])
            self.text_box_msg.setText("JAUGE DELL")
            self.name_gauge.setText("Add ?")
            self.name_gauge.setStyleSheet("background-color: red;")
            self.Print_fit_start()
            self.index_jauge -= 1
            if len(self.Param0) != 0:
                self.Gauge_type_selector.setCurrentIndex(
                    self.liste_type_Gauge.index(self.list_name_gauges[self.index_jauge])
                )
                self.LOAD_Gauge()
            else:
                self.f_gauge_select()
        else:
            print("Function stopped.")


class FitWorkflowMixin:
    """Services de préparation/validation des fits spectres.

    This mixin centralises fit-specific helpers so ``MainWindow`` keeps
    orchestration logic while reusable fit plumbing stays isolated.
    """

    def _build_pic_fit_problem(self, gauge_indices):
        from Bibli_python import CL_FD_Update as CL
        import numpy as np

        list_F = []
        initial_guess = []
        bounds_min = []
        bounds_max = []

        first_j = next(iter(gauge_indices))
        x_min = float(self.Param0[first_j][0][0])
        x_max = float(self.Param0[first_j][0][0])
        inter = float(self.inter_entry.value())

        for j in gauge_indices:
            g = self.Spectrum.Gauges[j]
            for i in range(self.J[j]):
                ctr, ampH, sigma, coef_spe, model_fit = self.Param0[j][i]
                pic = g.pics[i]
                pic.Update(
                    ctr=float(ctr),
                    ampH=float(ampH),
                    coef_spe=coef_spe,
                    sigma=float(sigma),
                    model_fit=model_fit,
                    inter=inter,
                )
                x_min = min(x_min, float(ctr) - float(sigma) * int(self.sigma_pic_fit_entry.value()))
                x_max = max(x_max, float(ctr) + float(sigma) * int(self.sigma_pic_fit_entry.value()))
                list_F.append(pic.f_model)
                initial_guess.extend([ctr, ampH, sigma])
                for c in coef_spe:
                    initial_guess.append(c)
                bounds_min.extend([pic.ctr[1][0], pic.ampH[1][0], pic.sigma[1][0]])
                bounds_max.extend([pic.ctr[1][1], pic.ampH[1][1], pic.sigma[1][1]])
                for c in pic.coef_spe:
                    bounds_min.append(c[1][0])
                    bounds_max.append(c[1][1])
            g.Update_model()

        return (
            list_F,
            np.array(initial_guess, dtype=float),
            np.array(bounds_min, dtype=float),
            np.array(bounds_max, dtype=float),
            float(x_min),
            float(x_max),
        )

    def _propose_and_confirm_fit(self, x_fit, y_fit, fit, color, text_base, use_abs=False):
        if self.Spectrum.dY is not None:
            resid_new = (y_fit - fit)
            new_score = fit_score(resid_new, use_abs=use_abs)
            old_score = fit_score(self.Spectrum.dY, use_abs=use_abs)
            is_better = new_score < old_score
        else:
            is_better = True

        text_fit = f"{text_base} BEST you can Validate" if is_better else f"{text_base} LESS GOOD you can Cancel"
        temp_curve_fit = None
        temp_curve_dy = None
        if not self.bit_bypass:
            temp_curve_fit = self.pg_spectrum.plot(x_fit, fit, pen=pg.mkPen(color, width=2, style=Qt.DashLine))
            temp_curve_dy = self.pg_dy.plot(x_fit, y_fit - fit, pen=pg.mkPen(color, width=2, style=Qt.DashLine))
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("CURVE FIT DONE")
            msg_box.setText(text_fit + '\n Save fit Press "v" Cancel Press "c"')
            v_button = msg_box.addButton("Validate", QMessageBox.AcceptRole)
            a_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)
            msg_box.setDefaultButton(v_button)

            def on_key_press(event):
                if event.key() == Qt.Key_V:
                    v_button.click()
                elif event.key() == Qt.Key_C:
                    a_button.click()

            msg_box.keyPressEvent = on_key_press
            msg_box.exec_()
            accepted = (msg_box.clickedButton() == v_button)
        else:
            accepted = is_better

        try:
            self.pg_spectrum.removeItem(temp_curve_fit)
        except Exception:
            pass
        try:
            self.pg_dy.removeItem(temp_curve_dy)
        except Exception:
            pass

        return accepted, is_better

    def _fit_bypass_state(self) -> bool:
        return bool(getattr(self, "_multi_fit_running", False) or self.bit_bypass)

    def _assign_fit_params_to_param0(self, params, *, nc0_slice_stop: int):
        import numpy as np

        ij_3 = ij_4 = ij_5 = 0
        params_list = list(params)
        for i, gauge in enumerate(self.Spectrum.Gauges):
            for j, _pic in enumerate(gauge.pics):
                n_c = len(self.Param0[i][j][3])
                start_idx = 3 * ij_3 + 4 * ij_4 + 5 * ij_5
                end_idx = start_idx + 3

                if n_c == 0:
                    self.Param0[i][j][:nc0_slice_stop] = params_list[start_idx:end_idx]
                    ij_3 += 1
                elif n_c == 1:
                    self.Param0[i][j][:4] = params_list[start_idx:end_idx] + [
                        np.array([params_list[end_idx]])
                    ]
                    ij_4 += 1
                elif n_c == 2:
                    self.Param0[i][j][:4] = params_list[start_idx:end_idx] + [
                        np.array(params_list[end_idx : end_idx + 2])
                    ]
                    ij_5 += 1

    def _refresh_fit_plot_cache_from_param0(self):
        for i, gauge in enumerate(self.Spectrum.Gauges):
            for j, pic in enumerate(gauge.pics):
                pic.Update(
                    ctr=float(self.Param0[i][j][0]),
                    ampH=float(self.Param0[i][j][1]),
                    coef_spe=self.Param0[i][j][3],
                    sigma=float(self.Param0[i][j][2]),
                    inter=float(self.inter_entry.value()),
                )
                params_f = pic.model.make_params()
                y_plot = pic.model.eval(params_f, x=self.Spectrum.wnb)
                self.list_y_fit_start[i][j] = y_plot
                new_name = (
                    f"{self.Nom_pic[i][j]}   X0:{self.Param0[i][j][0]}"
                    f"   Y0:{self.Param0[i][j][1]}"
                    f"   sigma:{self.Param0[i][j][2]}"
                    f"   Coef:{self.Param0[i][j][3]}"
                    f" ; Modele:{self.Param0[i][j][4]}"
                )
                self.list_text_pic[i][j] = str(new_name)

    def FIT_lmfitVScurvfit_ONE_GAUGE(self):
        """Fit uniquement sur la jauge actuellement sélectionnée (sans traçage Matplotlib)."""
        from Bibli_python import CL_FD_Update as CL
        from scipy.optimize import curve_fit
        import numpy as np

        save_jauge = self.index_jauge
        save_pic = self.index_pic_select

        self.Param_FIT = []
        self.nb_jauges = len(self.Spectrum.Gauges)
        j = self.index_jauge

        if j < 0 or j >= self.nb_jauges:
            self.text_box_msg.setText("FIT ONE GAUGE : aucune jauge sélectionnée.")
            return

        (
            list_F,
            initial_guess,
            bounds_min,
            bounds_max,
            _x_min,
            _x_max,
        ) = self._build_pic_fit_problem([j])
        bounds = (bounds_min, bounds_max)

        if (
            self.X_e[j] is not None
            and self.X_s[j] is not None
            and self.Spectrum.Gauges[j].indexX is not None
        ):
            self.Zone_fit[j] = np.where(
                (self.Spectrum.wnb >= self.X_s[j]) & (self.Spectrum.wnb <= self.X_e[j])
            )[0]
            x_fit = np.array(self.Spectrum.wnb[self.Zone_fit[j]])
            self.Spectrum.Gauges[j].indexX = self.Zone_fit[j]
            y_fit = np.array(self.Spectrum.spec[self.Spectrum.indexX])
        else:
            x_fit = np.array(self.Spectrum.x_corr)
            y_fit = np.array(self.Spectrum.y_corr)

        sum_function = CL.Gen_sum_F(list_F)
        self._submit_background_task(
            lambda: {
                "params": curve_fit(
                    sum_function, x_fit, y_fit, p0=initial_guess, bounds=bounds
                )[0],
                "x_fit": x_fit,
                "y_fit": y_fit,
                "fit": None,
                "gauge_index": j,
                "save_jauge": save_jauge,
                "save_pic": save_pic,
                "sum_function": sum_function,
            },
            result_slot=self._apply_single_fit_result,
            description="Fit jauge en cours…",
        )

    @pyqtSlot(object)
    def _apply_single_fit_result(self, payload):
        if not payload or "params" not in payload:
            self.text_box_msg.setText("FIT ERROR")
            return

        params = payload["params"]
        x_fit = payload["x_fit"]
        y_fit = payload["y_fit"]
        fit = payload.get("fit")
        j = payload.get("gauge_index", -1)
        save_jauge = payload.get("save_jauge", self.index_jauge)
        save_pic = payload.get("save_pic", self.index_pic_select)
        sum_function = payload.get("sum_function")

        if j < 0 or j >= len(self.Spectrum.Gauges):
            self.text_box_msg.setText("FIT ERROR (index)")
            return

        if fit is None and sum_function is not None:
            fit = sum_function(x_fit, *params)

        accepted, _is_better = self._propose_and_confirm_fit(
            x_fit=x_fit, y_fit=y_fit, fit=fit, color="r", text_base="Curve_fit", use_abs=True
        )

        if not accepted:
            self.Spectrum.bit_fit = True
            self.bit_fit_T = True
            self.text_box_msg.setText("BAD FIT $R^2$ INCREAS")
            return

        self.Spectrum.Gauges[j].Y = fit + self.Spectrum.blfit
        self.Spectrum.Gauges[j].X = x_fit
        self.Spectrum.Gauges[j].dY = y_fit - fit
        self.Spectrum.lamb_fit = params[0]

        self._assign_fit_params_to_param0(params, nc0_slice_stop=3)
        self._refresh_fit_plot_cache_from_param0()

        self.Spectrum.bit_fit = True
        self.text_box_msg.setText("FIT TOTAL \n DONE")
        self.bit_fit_T = True
        self.index_jauge = save_jauge
        self.index_pic_select = save_pic
        self.LOAD_Gauge()
        self.Print_fit_start()

    def FIT_lmfitVScurvfit(self, run_asynchron=True):
        """Fit global sur toutes les jauges (sans tracés Matplotlib)."""
        from Bibli_python import CL_FD_Update as CL
        from scipy.optimize import curve_fit
        import numpy as np

        save_jauge = self.index_jauge
        save_pic = self.index_pic_select
        self.Param_FIT = []
        self.nb_jauges = len(self.Spectrum.Gauges)
        if self.nb_jauges == 0:
            self.text_box_msg.setText("FIT : aucune jauge dans le spectre.")
            return

        gauge_indices = range(self.nb_jauges)
        (
            list_F,
            initial_guess,
            bounds_min,
            bounds_max,
            x_min,
            x_max,
        ) = self._build_pic_fit_problem(gauge_indices)
        bounds = (bounds_min, bounds_max)

        self.Spectrum.model = None
        for j in gauge_indices:
            g = self.Spectrum.Gauges[j]
            if self.Spectrum.model is None:
                self.Spectrum.model = g.model
            else:
                self.Spectrum.model += g.model
        if not self._ensure_spectrum_treatment_current():
            deg_baseline, filtre_type, param = self._current_baseline_params()
            self.Spectrum.Data_treatement(
                deg_baseline=deg_baseline,
                type_filtre=filtre_type,
                param_f=param,
            )

        if self.zone_spectrum_box.isChecked():
            self.Spectrum.indexX = np.where(
                (self.Spectrum.wnb >= x_min) & (self.Spectrum.wnb <= x_max)
            )[0]
            x_sub = self.Spectrum.wnb[self.Spectrum.indexX]
            y_sub = self.Spectrum.y_corr[self.Spectrum.indexX]
            blfit = self.Spectrum.blfit[self.Spectrum.indexX]
        else:
            if self.X_e[0] is not None and self.X_s[0] is not None and self.Spectrum.indexX is not None:
                self.Zone_fit[0] = np.where(
                    (self.Spectrum.wnb >= self.X_s[0]) & (self.Spectrum.wnb <= self.X_e[0])
                )[0]
                x_sub = self.Spectrum.wnb[self.Zone_fit[0]]
                self.Spectrum.indexX = self.Zone_fit[0]
                for j in self.Spectrum.Gauges:
                    j.indexX = self.Zone_fit[0]
                y_sub = self.Spectrum.y_corr[self.Spectrum.indexX]
                blfit = self.Spectrum.blfit[self.Spectrum.indexX]
            else:
                y_sub = self.Spectrum.y_corr
                blfit = self.Spectrum.blfit
                x_sub = self.Spectrum.wnb
                self.Spectrum.indexX = None
        y_sub, x_sub, blfit = np.array(y_sub), np.array(x_sub), np.array(blfit)

        if self.vslmfit.isChecked():
            self.Spectrum.FIT()
            for i, j in enumerate(self.Spectrum.Gauges):
                for k, p in enumerate(j.pics):
                    params_f = p.model.make_params()
                    y_plot = p.model.eval(params_f, x=self.Spectrum.wnb)
                    self.list_y_fit_start[i][k] = y_plot

        sum_function = CL.Gen_sum_F(list_F)
        payload = {
            "x_sub": x_sub,
            "y_sub": y_sub,
            "blfit": blfit,
            "gauge_indices": list(gauge_indices),
            "save_jauge": save_jauge,
            "save_pic": save_pic,
            "sum_function": sum_function,
            "bit_bypass": self._fit_bypass_state(),
        }
        if run_asynchron:
            self._submit_background_task(
                lambda: {
                    **payload,
                    "params": curve_fit(
                        sum_function, x_sub, y_sub, p0=initial_guess, bounds=bounds
                    )[0],
                },
                result_slot=self._apply_global_fit_result,
                description="Fit global en cours…",
            )
        else:
            payload["params"] = curve_fit(
                sum_function, x_sub, y_sub, p0=initial_guess, bounds=bounds
            )[0]
            self._apply_global_fit_result(payload)

    @pyqtSlot(object)
    def _apply_global_fit_result(self, payload):
        if not payload or "params" not in payload:
            self.text_box_msg.setText("FIT ERROR")
            return

        params = payload["params"]
        x_sub = payload.get("x_sub")
        y_sub = payload.get("y_sub")
        blfit = payload.get("blfit")
        sum_function = payload.get("sum_function")
        save_jauge = payload.get("save_jauge", self.index_jauge)
        save_pic = payload.get("save_pic", self.index_pic_select)
        self.bit_bypass = payload.get("bit_bypass", self.bit_bypass)
        if sum_function is None:
            self.text_box_msg.setText("FIT ERROR")
            return

        fit = sum_function(x_sub, *params)
        accepted, _is_better = self._propose_and_confirm_fit(
            x_fit=x_sub, y_fit=y_sub, fit=fit, color="m", text_base="Curve_fit", use_abs=False
        )
        if not accepted:
            # Fit non convergent/qualité faible :
            # on conserve malgré tout une estimation approchée pour alimenter
            # les calculs de pression en aval.
            self.Spectrum.Y = fit + blfit
            self.Spectrum.X = x_sub
            self.Spectrum.dY = y_sub - fit
            self.Spectrum.lamb_fit = params[0]
            self._assign_fit_params_to_param0(params, nc0_slice_stop=4)
            self._refresh_fit_plot_cache_from_param0()
            for gauge in self.Spectrum.Gauges:
                gauge.bit_fit = True
            for i, gauge in enumerate(self.Spectrum.Gauges):
                gauge.lamb_fit = self.Param0[i][0][0]
            self.Spectrum.bit_fit = True
            self.bit_fit_T = True
            self.Spectrum.Calcul_study(mini=False)
            self.text_box_msg.setText("BAD FIT r^2 INCREAS (approximation conservée)")
            self.Print_fit_start()
            return

        self.Spectrum.Y = fit + blfit
        self.Spectrum.X = x_sub
        self.Spectrum.dY = y_sub - fit
        self.Spectrum.lamb_fit = params[0]

        self._assign_fit_params_to_param0(params, nc0_slice_stop=4)
        self._refresh_fit_plot_cache_from_param0()
        for i, gauge in enumerate(self.Spectrum.Gauges):
            gauge.lamb_fit = self.Param0[i][0][0]
            gauge.bit_fit = True

        self.Spectrum.bit_fit = True
        self.Spectrum.Calcul_study(mini=False)
        self.text_box_msg.setText("FIT TOTAL \n DONE")
        self.bit_fit_T = True
        self.index_jauge = save_jauge
        self.index_pic_select = save_pic
        if not getattr(self, "_multi_fit_fast_mode", False):
            self.LOAD_Gauge()
            self.Print_fit_start()


class KeyboardShortcutMixin:
    """Gestion centralisée des key press events et raccourcis clavier."""

    def _should_ignore_keypress(self, event) -> bool:
        if self.viewer is not None and self.focusWidget() == self.viewer:
            print("focus in Lecroy")
            return True
        return False

    def _handle_python_shortcuts(self, key, modifiers) -> bool:
        from Bibli_python import Oscilloscope_LeCroy_vLABO as Oscilo

        if key == Qt.Key_Return and modifiers == Qt.ControlModifier:
            self.execute_code()
            return True

        if key == Qt.Key_L and modifiers & Qt.ShiftModifier:
            self.viewer = Oscilo.OscilloscopeViewer(
                folder=os.path.join(self.folder_start, "Aquisition_LECROY_Banc_CEDd")
            )
            self.viewer.show()
            return True
        return False

    def _handle_setup_shortcuts(self, key, modifiers):
        if not getattr(self, "setup_mode", False):
            return None
        if key == Qt.Key_C:
            return self.Click_Confirme
        if key == Qt.Key_Z and modifiers & Qt.ShiftModifier:
            return self.Click_Zone
        if key == Qt.Key_U and modifiers & Qt.ShiftModifier:
            return self.Undo_pic
        if key == Qt.Key_Return and modifiers & Qt.ShiftModifier:
            return self.Click_Clear
        if key == Qt.Key_W:
            return self.Undo_pic_select
        return None

    def _handle_toggle_shortcuts(self, key) -> bool:
        if key == Qt.Key_Q:
            self.select_clic_box.setChecked(not self.select_clic_box.isChecked())
            return True
        if key == Qt.Key_F:
            self.fit_start_box.setChecked(not self.fit_start_box.isChecked())
            return True
        if key == Qt.Key_M:
            self.movie_select_box.setChecked(not self.movie_select_box.isChecked())
            return True
        if key == Qt.Key_H:
            self.spectrum_select_box.setChecked(not self.spectrum_select_box.isChecked())
            return True
        return False

    def _run_calcul_study(self, mini: bool):
        try:
            self.Spectrum.Calcul_study(mini=mini)
        except Exception:
            self.Print_error(traceback.format_exc())

    def _resolve_global_action(self, key, modifiers):
        if key == Qt.Key_B and modifiers & Qt.ShiftModifier:
            return self.Baseline_spectrum, "Baseline_spectrum", False
        if key == Qt.Key_E and modifiers & Qt.ShiftModifier:
            return self.CREAT_new_CEDd, "CREAT new file CEDd", False
        if key == Qt.Key_K and modifiers & Qt.ShiftModifier:
            return self.CLEAR_CEDd, "CLEAR CEDd", True
        if key == Qt.Key_T and modifiers & Qt.ShiftModifier:
            return self.CREAT_new_Spectrum, "CREAT new Spectrum", False
        if key == Qt.Key_A and modifiers & Qt.ShiftModifier:
            return self.FIT_lmfitVScurvfit, "FIT lmfit VS curvfit", False
        if key == Qt.Key_Y and modifiers & Qt.ShiftModifier:
            return self.Auto_pic, None, False
        if key == Qt.Key_P:
            return self.toggle_colonne, None, False
        if key == Qt.Key_Z:
            return lambda: self.toggle_cam_region(), None, False
        if key == Qt.Key_I:
            return lambda: self.toggle_fit_region(), None, False
        if key == Qt.Key_O and modifiers & Qt.ShiftModifier:
            return self.Dell_Jauge, None, False
        if key == Qt.Key_R:
            if modifiers & Qt.ShiftModifier:
                return self.Replace_pic_fit, None, False
            return self.Replace_pic, None, False
        if key == Qt.Key_N and modifiers & Qt.ShiftModifier:
            return self.REFRESH, "Refresh data CEDd", False
        if key == Qt.Key_X and modifiers & Qt.ShiftModifier:
            return self.SAVE_CEDd, "Save CEDd", False
        if key == Qt.Key_J and modifiers & Qt.ShiftModifier:
            return self.ADD_gauge, None, False
        if key == Qt.Key_0:
            return self.f_lambda0, None, False
        return None

    def _dispatch_key_action(self, action):
        if action is None:
            return
        func, name_f, set_bypass = action
        if set_bypass:
            self.bit_bypass = True
        try:
            func()
            if name_f is not None and name_f != "Save CEDd":
                self.text_box_msg.setText(name_f + "SUCCES")
        except Exception:
            self.Print_error(traceback.format_exc())
        if set_bypass:
            self.bit_bypass = False

    def keyPressEvent(self, event):
        key = event.key()
        modifiers = event.modifiers()

        if self._should_ignore_keypress(event):
            return
        if self._handle_python_shortcuts(key, modifiers):
            return
        setup_action = self._handle_setup_shortcuts(key, modifiers)
        if setup_action is not None:
            setup_action()
            return
        if self._handle_toggle_shortcuts(key):
            return
        if key == Qt.Key_S:
            self._run_calcul_study(mini=bool(modifiers & Qt.ShiftModifier))
            return
        action = self._resolve_global_action(key, modifiers)
        self._dispatch_key_action(action)


class RunStateMixin:
    def _init_run_state(self) -> None:
        """Initialise l'état partagé entre les vues Spectrum et dDAC."""

        self.runs: Dict[str, Any] = {}
        self.current_run_id: Optional[str] = None
        self.file_index_map: Dict[str, int] = {}
