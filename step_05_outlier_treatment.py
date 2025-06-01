# step_05_outlier_treatment.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

try:
    from pyod.models.hbos import HBOS as PyOD_HBOS
    from pyod.models.ecod import ECOD as PyOD_ECOD
except ImportError:
    print("Warning: PyOD library not found. HBOS and ECOD outlier detection will not be available.")
    PyOD_HBOS = None
    PyOD_ECOD = None

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import io
from PIL import Image
import traceback
import seaborn as sns


# --- DPG Tags for Step 5 ---
TAG_OT_STEP_GROUP = "step5_outlier_treatment_group"

# Univariate Tab Tags
TAG_OT_UNIVARIATE_TAB = "step5_univariate_outlier_tab"
TAG_OT_DETECT_METHOD_RADIO_UNI = "step5_ot_detect_method_radio_uni"
TAG_OT_IQR_MULTIPLIER_INPUT = "step5_ot_iqr_multiplier_input"
TAG_OT_HBOS_N_BINS_INPUT = "step5_ot_hbos_n_bins_input"
TAG_OT_ECOD_CONTAM_INPUT_UNI = "step5_ot_ecod_contam_input_uni"
TAG_OT_DETECT_BUTTON_UNI = "step5_ot_detect_button_uni"
TAG_OT_RECOMMEND_PARAMS_BUTTON_UNI = "step5_ot_recommend_params_button_uni"
TAG_OT_DETECTION_RESULTS_TABLE_UNI = "step5_ot_detection_results_table_uni"
TAG_OT_VISUALIZATION_GROUP_UNI = "step5_ot_visualization_group_uni"
TAG_OT_VISUALIZATION_PLOT_IMAGE_UNI = "step5_ot_visualization_plot_image_uni"
TAG_OT_DEFAULT_PLOT_TEXTURE_UNI = "step5_ot_default_plot_texture_uni"

# Multivariate Tab Tags
TAG_OT_MULTIVARIATE_TAB = "step5_multivariate_outlier_tab"
TAG_OT_MVA_VAR_METHOD_RADIO = "step5_ot_mva_var_method_radio"
TAG_OT_MVA_CUSTOM_COLS_TABLE = "step5_ot_mva_custom_cols_table" # 테이블로 변경됨
TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT = "step5_ot_mva_iso_forest_contam_input"
TAG_OT_MVA_DETECT_BUTTON = "step5_ot_mva_detect_button"
TAG_OT_MVA_RESULTS_TEXT = "step5_ot_mva_results_text"
TAG_OT_MVA_VISUALIZATION_IMAGE = "step5_ot_mva_visualization_image"
TAG_OT_DEFAULT_PLOT_TEXTURE_MVA = "step5_ot_default_plot_texture_mva" # 이름 수정됨

# Common Log Tags
TAG_OT_LOG_TEXT_AREA = "step5_ot_log_text_area"
TAG_OT_LOG_TEXT = "step5_ot_log_text" # 태그 이름 일관성 (step_ -> step5_) - 다른 곳도 확인 필요하면 알려주세요.

# Univariate Treatment Button Tags
TAG_OT_RECOMMEND_TREATMENTS_BUTTON_UNI = "step5_ot_recommend_treatments_button_uni"
TAG_OT_RESET_TREATMENTS_BUTTON_UNI = "step5_ot_reset_treatments_button_uni"
TAG_OT_APPLY_TREATMENT_BUTTON_UNI = "step5_ot_apply_treatment_button_uni"
TAG_OT_TREATMENT_TABLE_UNI = "step5_ot_treatment_table_uni"


# --- Constants for Filtering (Univariate) ---
MIN_UNIQUE_VALUES_FOR_OUTLIER_DETECTION_UNI = 10
MAX_MISSING_RATIO_FOR_OUTLIER_DETECTION_UNI = 0.5
MIN_VARIANCE_FOR_OUTLIER_DETECTION_UNI = 1e-5

# --- Module State Variables ---
_main_app_callbacks: Optional[Dict[str, Any]] = None
_util_funcs: Optional[Dict[str, Any]] = None
_current_df_for_this_step: Optional[pd.DataFrame] = None

# Univariate State
_df_with_uni_detected_outliers: Optional[pd.DataFrame] = None
_df_after_uni_treatment: Optional[pd.DataFrame] = None # 단변량 처리 후 최종 DF (main_app으로 전달될 수 있음)
_uni_detected_outlier_indices: Dict[str, np.ndarray] = {}
_uni_outlier_summary_data: List[Dict[str, Any]] = []
_uni_columns_eligible_for_detection: List[str] = []
_uni_selected_detection_method: str = "IQR"
_uni_iqr_multiplier: float = 1.5
_uni_hbos_n_bins: int = 20
_uni_ecod_contamination: float = 0.1
_uni_treatment_selections: Dict[str, Dict[str, Any]] = {}
_uni_active_plot_texture_id: Optional[str] = None
_uni_currently_visualized_column: Optional[str] = None
_uni_all_selectable_tags_in_table: List[str] = []

# Multivariate State
_mva_variable_selection_method: str = "All Numeric"
_mva_custom_selected_columns: List[str] = []
_mva_custom_col_checkbox_tags: Dict[str, int] = {} # For table with checkboxes
_mva_iso_forest_contamination: Union[str, float] = 'auto'
_df_with_mva_outliers: Optional[pd.DataFrame] = None
_mva_outlier_row_indices: Optional[np.ndarray] = None
_mva_active_plot_texture_id: Optional[str] = None


# Default Parameter Constants
DEFAULT_UNI_IQR_MULTIPLIER = 1.5
DEFAULT_UNI_HBOS_N_BINS = 20
DEFAULT_UNI_ECOD_CONTAMINATION = 0.1
DEFAULT_MVA_ISO_FOREST_CONTAMINATION = 'auto'
RECOMMENDED_UNI_TREATMENT_METHOD = "Treat as Missing"


def _s5_plot_to_dpg_texture(fig: plt.Figure, desired_dpi: int = 90) -> Tuple[Optional[str], int, int]:
    img_data_buf = io.BytesIO()
    try:
        fig.savefig(img_data_buf, format="png", bbox_inches='tight', dpi=desired_dpi)
        img_data_buf.seek(0)
        pil_image = Image.open(img_data_buf)
        if pil_image.mode != 'RGBA': pil_image = pil_image.convert('RGBA')
        img_width, img_height = pil_image.size
        if img_width == 0 or img_height == 0:
            _log_message(f"Error: Plot image has zero dimension ({img_width}x{img_height}). Cannot create texture.")
            plt.close(fig)
            return None, 0, 0
        texture_data_np = np.array(pil_image).astype(np.float32) / 255.0
        texture_data_flat_list = texture_data_np.ravel().tolist()
        if not dpg.does_item_exist("texture_registry"):
            dpg.add_texture_registry(tag="texture_registry", show=False)
        texture_tag = dpg.generate_uuid()
        dpg.add_static_texture(width=img_width, height=img_height, default_value=texture_data_flat_list, tag=texture_tag, parent="texture_registry")
        return texture_tag, img_width, img_height
    except Exception as e:
        _log_message(f"Error converting plot to DPG texture: {e}\n{traceback.format_exc()}")
        return None, 0, 0
    finally:
        plt.close(fig)

def _log_message(message: str):
    if not dpg.is_dearpygui_running(): return
    log_text_tag_to_use = TAG_OT_LOG_TEXT # 공용 로그 태그 사용
    log_area_tag_to_use = TAG_OT_LOG_TEXT_AREA

    if dpg.does_item_exist(log_text_tag_to_use):
        current_log = dpg.get_value(log_text_tag_to_use)
        max_log_entries = 200 
        log_lines = current_log.splitlines()
        if len(log_lines) >= max_log_entries:
            log_lines = log_lines[-(max_log_entries-1):]
        
        new_log_entry = f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}"
        new_log = "\n".join(log_lines) + "\n" + new_log_entry if current_log else new_log_entry
        
        dpg.set_value(log_text_tag_to_use, new_log.strip())
        
        if dpg.does_item_exist(log_area_tag_to_use):
            item_config = dpg.get_item_configuration(log_area_tag_to_use)
            is_shown = item_config.get('show', True)
            item_info = dpg.get_item_info(log_area_tag_to_use)
            is_child_window = item_info['type'] == "mvAppItemType::mvChildWindow" if item_info else False

            if is_shown and is_child_window:
                dpg.set_y_scroll(log_area_tag_to_use, -1.0)

# --- Univariate Callbacks and Logic ---
def _reset_uni_treatment_selections_to_default():
    global _uni_treatment_selections, _uni_outlier_summary_data
    _log_message("Resetting all univariate treatment selections to 'Do Not Treat'.")
    cols_with_detected_outliers = [item['Column'] for item in _uni_outlier_summary_data if item.get('Detected Outliers', 0) > 0]
    if not cols_with_detected_outliers:
        _log_message("No univariate columns with detected outliers to reset treatment for.")
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Info", "No univariate outlier treatments to reset. Run detection first.")
        return
    for col_name in cols_with_detected_outliers:
        _uni_treatment_selections[col_name] = {"method": "Do Not Treat"}
    _populate_uni_treatment_table()
    _log_message("All univariate treatment selections have been reset to 'Do Not Treat'.")
    if _util_funcs and '_show_simple_modal_message' in _util_funcs:
        _util_funcs['_show_simple_modal_message']("Info", "All univariate outlier treatment selections reset.")

def _update_uni_parameter_fields_visibility():
    if not dpg.is_dearpygui_running(): return
    show_iqr = _uni_selected_detection_method == "IQR"
    show_hbos = _uni_selected_detection_method == "HBOS"
    show_ecod = _uni_selected_detection_method == "ECOD"
    
    iqr_elements = [TAG_OT_IQR_MULTIPLIER_INPUT, TAG_OT_IQR_MULTIPLIER_INPUT + "_label"]
    hbos_elements = [TAG_OT_HBOS_N_BINS_INPUT, TAG_OT_HBOS_N_BINS_INPUT + "_label"]
    ecod_elements = [TAG_OT_ECOD_CONTAM_INPUT_UNI, TAG_OT_ECOD_CONTAM_INPUT_UNI + "_label"]

    for item_tag in iqr_elements:
        if dpg.does_item_exist(item_tag): dpg.configure_item(item_tag, show=show_iqr)
    for item_tag in hbos_elements:
        if dpg.does_item_exist(item_tag): dpg.configure_item(item_tag, show=show_hbos)
    for item_tag in ecod_elements:
        if dpg.does_item_exist(item_tag): dpg.configure_item(item_tag, show=show_ecod)

def _on_uni_detection_method_change(sender, app_data: str, user_data):
    global _uni_selected_detection_method
    _uni_selected_detection_method = app_data
    _log_message(f"Univariate outlier detection method changed to: {_uni_selected_detection_method}")
    _update_uni_parameter_fields_visibility()

def _on_uni_iqr_multiplier_change(sender, app_data: float, user_data):
    global _uni_iqr_multiplier
    _uni_iqr_multiplier = app_data
    _log_message(f"Univariate IQR multiplier set to: {_uni_iqr_multiplier}")

def _on_uni_hbos_n_bins_change(sender, app_data: int, user_data):
    global _uni_hbos_n_bins
    if app_data >= 2 : 
        _uni_hbos_n_bins = app_data
        _log_message(f"Univariate HBOS n_bins set to: {_uni_hbos_n_bins}")
    else:
        dpg.set_value(sender, _uni_hbos_n_bins) 
        _log_message(f"Univariate HBOS n_bins must be >= 2. Reverted to {_uni_hbos_n_bins}.")
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Input Error", "HBOS n_bins must be greater than or equal to 2.")

def _on_uni_ecod_contamination_change(sender, app_data: float, user_data):
    global _uni_ecod_contamination
    if 0.0 < app_data <= 0.5: 
        _uni_ecod_contamination = app_data
        _log_message(f"Univariate ECOD contamination set to: {_uni_ecod_contamination:.4f}")
    else:
        dpg.set_value(sender, _uni_ecod_contamination) 
        _log_message(f"Univariate ECOD contamination must be (0.0, 0.5]. Reverted to {_uni_ecod_contamination:.4f}.")
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Input Error", "ECOD contamination must be between 0 (exclusive) and 0.5 (inclusive).")

def _set_uni_recommended_detection_parameters():
    global _uni_iqr_multiplier, _uni_hbos_n_bins, _uni_ecod_contamination
    _uni_iqr_multiplier = DEFAULT_UNI_IQR_MULTIPLIER
    _uni_hbos_n_bins = DEFAULT_UNI_HBOS_N_BINS
    _uni_ecod_contamination = DEFAULT_UNI_ECOD_CONTAMINATION
    
    if dpg.does_item_exist(TAG_OT_IQR_MULTIPLIER_INPUT):
        dpg.set_value(TAG_OT_IQR_MULTIPLIER_INPUT, _uni_iqr_multiplier)
    if dpg.does_item_exist(TAG_OT_HBOS_N_BINS_INPUT):
        dpg.set_value(TAG_OT_HBOS_N_BINS_INPUT, _uni_hbos_n_bins)
    if dpg.does_item_exist(TAG_OT_ECOD_CONTAM_INPUT_UNI):
        dpg.set_value(TAG_OT_ECOD_CONTAM_INPUT_UNI, _uni_ecod_contamination)
        
    _log_message(f"Recommended univariate detection parameters set: IQR Mult={_uni_iqr_multiplier}, HBOS n_bins={_uni_hbos_n_bins}, ECOD Contam={_uni_ecod_contamination:.2f}")
    _update_uni_parameter_fields_visibility() 
    if _util_funcs and '_show_simple_modal_message' in _util_funcs:
        _util_funcs['_show_simple_modal_message']("Info", "Recommended univariate detection parameters have been applied.")

def _detect_outliers_uni_iqr(series: pd.Series, multiplier: float) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(series.dtype) or series.empty: return np.array([])
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr_value = q3 - q1
    if iqr_value == 0 or pd.isna(iqr_value): return np.array([])
    lower_bound = q1 - multiplier * iqr_value
    upper_bound = q3 + multiplier * iqr_value
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    return series.index[outlier_mask].to_numpy()

def _detect_outliers_uni_hbos(series: pd.Series, n_bins_param: int, contamination_param: float) -> np.ndarray:
    if PyOD_HBOS is None: _log_message("Error: PyOD_HBOS model not loaded. Is PyOD installed?"); return np.array([])
    if not pd.api.types.is_numeric_dtype(series.dtype): return np.array([])
    series_cleaned = series.dropna()
    if len(series_cleaned) < 2 or len(series_cleaned.unique()) < 2 : return np.array([])
    actual_n_bins = min(n_bins_param, len(series_cleaned.unique()) -1) if len(series_cleaned.unique()) > 1 else 1
    if actual_n_bins < 2 : actual_n_bins = 2 
    try:
        model = PyOD_HBOS(n_bins=actual_n_bins, contamination=contamination_param)
        model.fit(series_cleaned.values.reshape(-1, 1))
        return series_cleaned.index[model.labels_ == 1].to_numpy()
    except Exception as e: _log_message(f"  Error during HBOS for {series.name}: {e}"); return np.array([])

def _detect_outliers_uni_ecod(series: pd.Series, contamination_param: float) -> np.ndarray:
    if PyOD_ECOD is None: _log_message("Error: PyOD_ECOD model not loaded. Is PyOD installed?"); return np.array([])
    if not pd.api.types.is_numeric_dtype(series.dtype): return np.array([])
    series_cleaned = series.dropna()
    if len(series_cleaned) < 2 : return np.array([])
    try:
        model = PyOD_ECOD(contamination=contamination_param)
        model.fit(series_cleaned.values.reshape(-1, 1))
        return series_cleaned.index[model.labels_ == 1].to_numpy()
    except Exception as e: _log_message(f"  Error during ECOD for {series.name}: {e}"); return np.array([])

def _filter_columns_for_uni_detection(df: pd.DataFrame) -> List[str]:
    eligible_cols = []
    if df is None: return eligible_cols
    s1_col_types = _main_app_callbacks.get('get_column_analysis_types', lambda: {})() if _main_app_callbacks else {}
    _log_message("Column filtering for univariate outlier detection:")
    for col_name in df.columns:
        series = df[col_name]
        is_numeric_s1 = "Numeric" in s1_col_types.get(col_name, "") and "Binary" not in s1_col_types.get(col_name, "")
        is_numeric_pandas = pd.api.types.is_numeric_dtype(series.dtype)
        if not (is_numeric_s1 or is_numeric_pandas): continue
        if series.nunique() < MIN_UNIQUE_VALUES_FOR_OUTLIER_DETECTION_UNI: continue
        s1_type = s1_col_types.get(col_name, "")
        if any(k in s1_type for k in ["Categorical", "Binary", "Text"]): continue
        if (series.isnull().sum() / len(series) if len(series) > 0 else 1.0) > MAX_MISSING_RATIO_FOR_OUTLIER_DETECTION_UNI: continue
        series_dropped_na = series.dropna()
        if series_dropped_na.empty: continue
        variance = series_dropped_na.var()
        if pd.isna(variance) or variance < MIN_VARIANCE_FOR_OUTLIER_DETECTION_UNI: continue
        eligible_cols.append(col_name)
        _log_message(f"  Eligible for univariate detection: '{col_name}'")
    if not eligible_cols: _log_message("  No columns found eligible for univariate detection.")
    return eligible_cols

def _run_uni_outlier_detection_logic():
    global _current_df_for_this_step, _uni_detected_outlier_indices, _uni_outlier_summary_data, \
           _df_with_uni_detected_outliers, _uni_columns_eligible_for_detection, _uni_currently_visualized_column
    _log_message("Run Univariate Outlier Detection button clicked.")
    try:
        if _current_df_for_this_step is None:
            _log_message("Error: No data for univariate outlier detection.")
            if _util_funcs and '_show_simple_modal_message' in _util_funcs:
                _util_funcs['_show_simple_modal_message']("Error", "No data for univariate outlier detection.")
            return

        _log_message(f"--- Starting Univariate Outlier Detection (Method: {_uni_selected_detection_method}) ---")
        if _uni_selected_detection_method == "IQR":
            _log_message(f"  Parameters: IQR Multiplier={_uni_iqr_multiplier}")
        elif _uni_selected_detection_method == "HBOS":
            _log_message(f"  Parameters: HBOS n_bins={_uni_hbos_n_bins}, Contamination={_uni_ecod_contamination:.4f}")
        elif _uni_selected_detection_method == "ECOD":
            _log_message(f"  Parameters: ECOD Contamination={_uni_ecod_contamination:.4f}")

        _uni_columns_eligible_for_detection = _filter_columns_for_uni_detection(_current_df_for_this_step)
        if not _uni_columns_eligible_for_detection:
            _log_message("Warning: No columns suitable for univariate outlier detection.")
            if _util_funcs and '_show_simple_modal_message' in _util_funcs:
                _util_funcs['_show_simple_modal_message']("Warning", "No columns suitable for univariate outlier detection.")
            _uni_outlier_summary_data.clear(); _uni_detected_outlier_indices.clear()
            _populate_uni_detection_results_table(); _populate_uni_treatment_table()
            _uni_currently_visualized_column = None; _clear_uni_visualization_plot()
            return

        _uni_detected_outlier_indices.clear(); _uni_outlier_summary_data.clear()
        _df_with_uni_detected_outliers = _current_df_for_this_step.copy()
        any_detection_performed_successfully = False
        first_col_with_outliers_for_auto_vis = None

        for col_name in _uni_columns_eligible_for_detection:
            series = _current_df_for_this_step[col_name]
            col_outlier_indices = np.array([])
            if _uni_selected_detection_method == "IQR":
                col_outlier_indices = _detect_outliers_uni_iqr(series, _uni_iqr_multiplier)
            elif _uni_selected_detection_method == "HBOS":
                col_outlier_indices = _detect_outliers_uni_hbos(series, _uni_hbos_n_bins, _uni_ecod_contamination)
            elif _uni_selected_detection_method == "ECOD":
                col_outlier_indices = _detect_outliers_uni_ecod(series, _uni_ecod_contamination)
            
            if len(col_outlier_indices) > 0: any_detection_performed_successfully = True
            _uni_detected_outlier_indices[col_name] = col_outlier_indices
            outlier_flag_col_name = f"{col_name}_is_outlier"
            _df_with_uni_detected_outliers[outlier_flag_col_name] = False
            if len(col_outlier_indices) > 0:
                _df_with_uni_detected_outliers.loc[col_outlier_indices, outlier_flag_col_name] = True
                if first_col_with_outliers_for_auto_vis is None: first_col_with_outliers_for_auto_vis = col_name
            num_outliers = len(col_outlier_indices)
            _uni_outlier_summary_data.append({"Column": col_name, "Detected Outliers": num_outliers, "Percentage (%)": f"{(num_outliers / len(series) * 100 if len(series) > 0 else 0):.2f}"})

        _populate_uni_detection_results_table(); _populate_uni_treatment_table()
        
        if first_col_with_outliers_for_auto_vis:
            _uni_currently_visualized_column = first_col_with_outliers_for_auto_vis
            _generate_uni_combined_plot_texture(first_col_with_outliers_for_auto_vis)
        elif _uni_columns_eligible_for_detection:
            _uni_currently_visualized_column = _uni_columns_eligible_for_detection[0]
            _generate_uni_combined_plot_texture(_uni_columns_eligible_for_detection[0])
        else: _uni_currently_visualized_column = None; _clear_uni_visualization_plot()
        _log_message("--- Univariate Outlier Detection Finished ---")
        if _util_funcs and '_show_simple_modal_message' in _util_funcs: _util_funcs['_show_simple_modal_message']("Detection Complete", "Univariate outlier detection finished.")
    except Exception as e:
        _log_message(f"CRITICAL ERROR in _run_uni_outlier_detection_logic: {e}"); traceback.print_exc()
        if _util_funcs and '_show_simple_modal_message' in _util_funcs: _util_funcs['_show_simple_modal_message']("Critical Error", f"Error during univariate detection: {e}")

def _populate_uni_detection_results_table():
    global _uni_all_selectable_tags_in_table
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_DETECTION_RESULTS_TABLE_UNI): return
    dpg.delete_item(TAG_OT_DETECTION_RESULTS_TABLE_UNI, children_only=True)
    _uni_all_selectable_tags_in_table.clear()
    if not _uni_outlier_summary_data:
        dpg.add_table_column(label="Info", parent=TAG_OT_DETECTION_RESULTS_TABLE_UNI, width_stretch=True)
        with dpg.table_row(parent=TAG_OT_DETECTION_RESULTS_TABLE_UNI): dpg.add_text("No univariate detection results. Run detection.")
        return
    headers = ["Column (Click to Visualize)", "Detected Outliers", "Percentage (%)"]
    for i, header in enumerate(headers): dpg.add_table_column(label=header, parent=TAG_OT_DETECTION_RESULTS_TABLE_UNI, init_width_or_weight=([0.5,0.25,0.25][i]), width_stretch=(i==0))
    for i, row_data in enumerate(_uni_outlier_summary_data):
        with dpg.table_row(parent=TAG_OT_DETECTION_RESULTS_TABLE_UNI):
            col_name_str = str(row_data.get("Column", ""))
            tag = f"uni_selectable_s5_{i}_{''.join(filter(str.isalnum, col_name_str))}"
            _uni_all_selectable_tags_in_table.append(tag)
            dpg.add_selectable(label=col_name_str, tag=tag, user_data=col_name_str, callback=_on_uni_row_selectable_clicked, span_columns=False)
            dpg.add_text(str(row_data.get("Detected Outliers", "")))
            dpg.add_text(str(row_data.get("Percentage (%)", "")))

def _on_uni_row_selectable_clicked(sender, app_data_is_selected: bool, user_data_col_name: str):
    global _uni_currently_visualized_column, _uni_all_selectable_tags_in_table
    if app_data_is_selected:
        for tag in _uni_all_selectable_tags_in_table:
            if tag != sender and dpg.does_item_exist(tag) and dpg.get_value(tag): dpg.set_value(tag, False)
        _uni_currently_visualized_column = user_data_col_name
        _generate_uni_combined_plot_texture(user_data_col_name)
        _log_message(f"Visualizing univariate outlier plots for: {user_data_col_name}")

def _generate_uni_combined_plot_texture(column_name: str):
    global _uni_active_plot_texture_id, _df_with_uni_detected_outliers, _current_df_for_this_step, _uni_iqr_multiplier
    if _df_with_uni_detected_outliers is None or column_name not in _df_with_uni_detected_outliers.columns or \
       _current_df_for_this_step is None or column_name not in _current_df_for_this_step.columns:
        _clear_uni_visualization_plot(); _log_message(f"Plot error: Data for '{column_name}' not ready."); return

    original_series = _current_df_for_this_step[column_name].dropna()
    outlier_flag_col = f"{column_name}_is_outlier"
    if original_series.empty or not pd.api.types.is_numeric_dtype(original_series.dtype) or \
       outlier_flag_col not in _df_with_uni_detected_outliers.columns:
        _clear_uni_visualization_plot(); _log_message(f"Plot error: Invalid data or flags for '{column_name}'."); return

    fig_width, fig_height = 15.6, 5.85 # 플롯 크기 확장 
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    plt.subplots_adjust(wspace=0.3)

    axes[0].boxplot(original_series, vert=True, patch_artist=True, medianprops={'color':'#FF0000', 'linewidth': 1.5}, flierprops={'marker':'o', 'markersize':4, 'markerfacecolor':'#FF7F50', 'alpha':0.6})
    axes[0].set_xticks([]); axes[0].set_ylabel(column_name, fontsize=9)
    axes[0].set_title(f"Box Plot: {column_name}", fontsize=10); axes[0].tick_params(axis='y', labelsize=8); axes[0].grid(True, linestyle='--', alpha=0.6)
    q1, q3 = original_series.quantile(0.25), original_series.quantile(0.75)
    iqr_val = q3 - q1
    lower_b, upper_b = q1 - _uni_iqr_multiplier * iqr_val, q3 + _uni_iqr_multiplier * iqr_val
    axes[0].axhline(upper_b, color='orangered', linestyle='--', linewidth=1.2, label=f'Upper ({_uni_iqr_multiplier}*IQR): {upper_b:.2f}')
    axes[0].axhline(lower_b, color='orangered', linestyle='--', linewidth=1.2, label=f'Lower ({_uni_iqr_multiplier}*IQR): {lower_b:.2f}')
    axes[0].legend(fontsize=7, loc='upper right', framealpha=0.5)

    scatter_df = pd.DataFrame({'index': _df_with_uni_detected_outliers.index, 'value': _df_with_uni_detected_outliers[column_name], 'is_outlier': _df_with_uni_detected_outliers[outlier_flag_col]})
    sns.scatterplot(data=scatter_df, x='index', y='value', hue='is_outlier', style='is_outlier', palette={True: "red", False: "cornflowerblue"}, markers={True: "X", False: "o"}, alpha=0.7, size='is_outlier', sizes={True: 50, False: 20}, ax=axes[1], legend='brief')
    median_val = original_series.median()
    axes[1].axhline(median_val, color='forestgreen', linestyle=':', linewidth=1.2, label=f'Median: {median_val:.2f}')
    axes[1].set_title(f"Scatter Plot (vs. Index): {column_name}", fontsize=10); axes[1].set_xlabel("Data Index", fontsize=9); axes[1].set_ylabel(column_name, fontsize=9)
    axes[1].legend(fontsize=8, loc='upper right', framealpha=0.5); axes[1].tick_params(axis='both', which='major', labelsize=8); axes[1].grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(f"Univariate Outlier Visualization for {column_name}", fontsize=12, y=0.99 if fig_height > 5 else 1.02) 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    texture_tag, tex_w, tex_h = _s5_plot_to_dpg_texture(fig)
    if _uni_active_plot_texture_id and _uni_active_plot_texture_id != TAG_OT_DEFAULT_PLOT_TEXTURE_UNI and dpg.does_item_exist(_uni_active_plot_texture_id):
        try: dpg.delete_item(_uni_active_plot_texture_id)
        except Exception as e: _log_message(f"Error deleting uni tex: {e}")
    if texture_tag and tex_w > 0 and tex_h > 0:
        _uni_active_plot_texture_id = texture_tag
        if dpg.does_item_exist(TAG_OT_VISUALIZATION_PLOT_IMAGE_UNI):
            img_parent_w = dpg.get_item_width(TAG_OT_VISUALIZATION_GROUP_UNI); disp_w = min(tex_w, (img_parent_w - 20 if img_parent_w and img_parent_w > 20 else 880) )
            disp_h = int(tex_h * (disp_w / tex_w)) if tex_w > 0 else tex_h
            dpg.configure_item(TAG_OT_VISUALIZATION_PLOT_IMAGE_UNI, texture_tag=_uni_active_plot_texture_id, width=int(disp_w), height=int(disp_h), show=True)
    else: _clear_uni_visualization_plot(); _log_message(f"Failed to generate uni plot for '{column_name}'.")

def _clear_uni_visualization_plot():
    global _uni_active_plot_texture_id
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_OT_VISUALIZATION_PLOT_IMAGE_UNI) and dpg.does_item_exist(TAG_OT_DEFAULT_PLOT_TEXTURE_UNI):
        cfg = dpg.get_item_configuration(TAG_OT_DEFAULT_PLOT_TEXTURE_UNI); w,h = (cfg.get('width',100), cfg.get('height',30)) if cfg else (100,30)
        dpg.configure_item(TAG_OT_VISUALIZATION_PLOT_IMAGE_UNI, texture_tag=TAG_OT_DEFAULT_PLOT_TEXTURE_UNI, width=w, height=h, show=True)
    if _uni_active_plot_texture_id and _uni_active_plot_texture_id != TAG_OT_DEFAULT_PLOT_TEXTURE_UNI and dpg.does_item_exist(_uni_active_plot_texture_id):
        try: dpg.delete_item(_uni_active_plot_texture_id)
        except Exception as e: _log_message(f"Error deleting uni tex: {e}")
    _uni_active_plot_texture_id = TAG_OT_DEFAULT_PLOT_TEXTURE_UNI

def _populate_uni_treatment_table():
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_TREATMENT_TABLE_UNI): return
    dpg.delete_item(TAG_OT_TREATMENT_TABLE_UNI, children_only=True)
    cols_to_treat = [item['Column'] for item in _uni_outlier_summary_data if item.get('Detected Outliers', 0) > 0]
    if not cols_to_treat:
        dpg.add_table_column(label="Info", parent=TAG_OT_TREATMENT_TABLE_UNI, width_stretch=True)
        with dpg.table_row(parent=TAG_OT_TREATMENT_TABLE_UNI): dpg.add_text("No univariate columns with detected outliers to treat.")
        return
    headers = ["Column Name", "Treatment Method", "Parameters (Lower %tile / Abs Lower)", "Parameters (Upper %tile / Abs Upper)"]
    col_widths_treatment = [0.3, 0.3, 0.2, 0.2] 
    for i, header in enumerate(headers): dpg.add_table_column(label=header, parent=TAG_OT_TREATMENT_TABLE_UNI, init_width_or_weight=col_widths_treatment[i], width_stretch=True)
    treatment_options = ["Do Not Treat", "Treat as Missing", "Ratio-based Capping", "Absolute Value Capping"]
    for col_name in cols_to_treat:
        with dpg.table_row(parent=TAG_OT_TREATMENT_TABLE_UNI):
            dpg.add_text(col_name)
            current_col_treatment = _uni_treatment_selections.get(col_name, {"method": "Do Not Treat"})
            current_method = current_col_treatment.get("method", "Do Not Treat")
            s_col = "".join(filter(str.isalnum, col_name))
            dpg.add_combo(treatment_options, default_value=current_method, width=-1, tag=f"s5_uni_treat_combo_{s_col}", callback=_on_uni_treatment_method_change, user_data={"col_name": col_name})
            with dpg.group(horizontal=True): 
                dpg.add_input_int(width=70, default_value=current_col_treatment.get("lower_percentile", 5), show=(current_method == "Ratio-based Capping"), min_value=1, max_value=20, tag=f"s5_uni_lp_{s_col}", callback=_on_uni_treatment_param_change, user_data={"col_name": col_name, "param_name": "lower_percentile"})
                dpg.add_input_float(width=100, default_value=current_col_treatment.get("abs_lower_bound", 0.0), show=(current_method == "Absolute Value Capping"), tag=f"s5_uni_alb_{s_col}", callback=_on_uni_treatment_param_change, user_data={"col_name": col_name, "param_name": "abs_lower_bound"})
                if current_method not in ["Ratio-based Capping", "Absolute Value Capping"]: dpg.add_text("N/A", show=True) 
            with dpg.group(horizontal=True): 
                dpg.add_input_int(width=70, default_value=current_col_treatment.get("upper_percentile", 95), show=(current_method == "Ratio-based Capping"), min_value=80, max_value=99, tag=f"s5_uni_up_{s_col}", callback=_on_uni_treatment_param_change, user_data={"col_name": col_name, "param_name": "upper_percentile"})
                dpg.add_input_float(width=100, default_value=current_col_treatment.get("abs_upper_bound", 0.0), show=(current_method == "Absolute Value Capping"), tag=f"s5_uni_aub_{s_col}", callback=_on_uni_treatment_param_change, user_data={"col_name": col_name, "param_name": "abs_upper_bound"})
                if current_method not in ["Ratio-based Capping", "Absolute Value Capping"]: dpg.add_text("N/A", show=True)

def _on_uni_treatment_method_change(sender, app_data_method: str, user_data: Dict):
    col_name = user_data["col_name"]
    current_treatment = _uni_treatment_selections.get(col_name, {}).copy()
    current_treatment["method"] = app_data_method
    _uni_treatment_selections[col_name] = current_treatment
    _log_message(f"Univariate Treatment for '{col_name}': {app_data_method}")
    _populate_uni_treatment_table()

def _on_uni_treatment_param_change(sender, app_data, user_data: Dict):
    col_name = user_data["col_name"]
    param_name = user_data["param_name"]
    current_treatment = _uni_treatment_selections.get(col_name, {}).copy()
    if param_name == "lower_percentile":
        try: val = int(app_data); assert 1 <= val <= 20; current_treatment[param_name] = val
        except: _log_message(f"Warn: Invalid lower %tile '{col_name}'."); dpg.set_value(sender, current_treatment.get(param_name, 5)); return
    elif param_name == "upper_percentile":
        try: val = int(app_data); assert 80 <= val <= 99; current_treatment[param_name] = val
        except: _log_message(f"Warn: Invalid upper %tile '{col_name}'."); dpg.set_value(sender, current_treatment.get(param_name, 95)); return
    elif param_name in ["abs_lower_bound", "abs_upper_bound"]:
        try: current_treatment[param_name] = float(app_data)
        except: _log_message(f"Warn: Invalid abs bound '{col_name}'."); dpg.set_value(sender, current_treatment.get(param_name, 0.0)); return
    _uni_treatment_selections[col_name] = current_treatment
    _log_message(f"Uni Treatment param for '{col_name}', '{param_name}' set to {app_data}")

def _set_uni_recommended_treatments_logic():
    global _uni_treatment_selections, _uni_outlier_summary_data
    cols_to_treat = [item['Column'] for item in _uni_outlier_summary_data if item.get('Detected Outliers', 0) > 0]
    if not cols_to_treat: _log_message("No uni columns with outliers for recommended treatments."); return
    for col_name in cols_to_treat: _uni_treatment_selections[col_name] = {"method": RECOMMENDED_UNI_TREATMENT_METHOD}
    _log_message(f"Recommended uni treatment ('{RECOMMENDED_UNI_TREATMENT_METHOD}') set."); _populate_uni_treatment_table()
    if _util_funcs and '_show_simple_modal_message' in _util_funcs: _util_funcs['_show_simple_modal_message']("Info", "Recommended univariate treatments set.")

def _apply_uni_outlier_treatment_logic():
    global _current_df_for_this_step, _df_after_uni_treatment, _uni_detected_outlier_indices, _uni_treatment_selections, _df_after_treatment
    _log_message("Apply Univariate Outlier Treatments button clicked.")
    try:
        if _current_df_for_this_step is None: 
            if _main_app_callbacks: _main_app_callbacks['step5_outlier_treatment_complete'](None); return
        if not _uni_detected_outlier_indices and not any(item.get('Detected Outliers',0) > 0 for item in _uni_outlier_summary_data):
            if _main_app_callbacks: _main_app_callbacks['step5_outlier_treatment_complete'](_current_df_for_this_step.copy()); return

        _df_after_treatment = _current_df_for_this_step.copy() # 최종 결과를 담을 DF, 초기화
        _log_message("--- Starting Univariate Outlier Treatment Application ---")
        treatment_applied_any = False
        for col_name, treatment_params in _uni_treatment_selections.items():
            if col_name not in _df_after_treatment.columns or col_name not in _uni_detected_outlier_indices: continue
            outlier_idx = _uni_detected_outlier_indices[col_name]
            if len(outlier_idx) == 0: continue
            method = treatment_params.get("method", "Do Not Treat")
            series_before_treat = _df_after_treatment[col_name].copy() # clip은 원본 변경 방지 위해 복사본에 적용
            if method == "Treat as Missing": _df_after_treatment.loc[outlier_idx, col_name] = np.nan; treatment_applied_any = True
            elif method == "Ratio-based Capping":
                lp, up = treatment_params.get("lower_percentile"), treatment_params.get("upper_percentile")
                if lp is None or up is None or not (1<=lp<=20 and 80<=up<=99 and lp < up): continue
                ref_series = _current_df_for_this_step[col_name].dropna(); 
                if ref_series.empty: continue
                lv, uv = ref_series.quantile(lp/100.0), ref_series.quantile(up/100.0)
                if pd.isna(lv) or pd.isna(uv): continue
                _df_after_treatment.loc[outlier_idx, col_name] = series_before_treat.loc[outlier_idx].clip(lower=lv, upper=uv); treatment_applied_any = True
            elif method == "Absolute Value Capping":
                al, au = treatment_params.get("abs_lower_bound"), treatment_params.get("abs_upper_bound")
                if al is None or au is None or al >= au : continue
                _df_after_treatment.loc[outlier_idx, col_name] = series_before_treat.loc[outlier_idx].clip(lower=al, upper=au); treatment_applied_any = True
            if method != "Do Not Treat": _log_message(f"  Univariate treatment '{method}' applied to '{col_name}'.")
        
        if treatment_applied_any: _log_message("--- Univariate Outlier Treatment Application Finished ---")
        else: _log_message("--- No univariate outlier treatments were applied. ---")
        
        if _main_app_callbacks: _main_app_callbacks['step5_outlier_treatment_complete'](_df_after_treatment) # 단일 처리 결과 전달
    
    except Exception as e:
        _log_message(f"CRITICAL ERROR in _apply_uni_outlier_treatment_logic: {e}"); traceback.print_exc()
        if _main_app_callbacks: _main_app_callbacks['step5_outlier_treatment_complete'](None)

# --- Multivariate Callbacks and Logic (Scaffolding) ---
def _on_mva_var_method_change(sender, app_data: str, user_data):
    global _mva_variable_selection_method
    _mva_variable_selection_method = app_data
    _log_message(f"Multivariate variable selection method: {_mva_variable_selection_method}")
    show_custom_table = (_mva_variable_selection_method == "Custom")
    if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE):
        dpg.configure_item(TAG_OT_MVA_CUSTOM_COLS_TABLE, show=show_custom_table)
    if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE + "_label"):
        dpg.configure_item(TAG_OT_MVA_CUSTOM_COLS_TABLE + "_label", show=show_custom_table)
    if show_custom_table and _current_df_for_this_step is not None:
        _populate_mva_custom_cols_table(_current_df_for_this_step)

def _on_mva_custom_col_checkbox_change(sender, app_data_is_checked: bool, user_data_col_name: str):
    global _mva_custom_selected_columns
    if app_data_is_checked:
        if user_data_col_name not in _mva_custom_selected_columns: _mva_custom_selected_columns.append(user_data_col_name)
    else:
        if user_data_col_name in _mva_custom_selected_columns: _mva_custom_selected_columns.remove(user_data_col_name)
    _log_message(f"MVA custom columns updated: {_mva_custom_selected_columns}")

def _populate_mva_custom_cols_table(df: Optional[pd.DataFrame]): # Optional[pd.DataFrame]으로 변경
    global _mva_custom_col_checkbox_tags, _mva_custom_selected_columns
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE): return
    dpg.delete_item(TAG_OT_MVA_CUSTOM_COLS_TABLE, children_only=True)
    _mva_custom_col_checkbox_tags.clear()
    if df is None:
        dpg.add_table_column(label="Info", parent=TAG_OT_MVA_CUSTOM_COLS_TABLE, width_stretch=True)
        with dpg.table_row(parent=TAG_OT_MVA_CUSTOM_COLS_TABLE): dpg.add_text("No data to select columns from.")
        return
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_cols:
        dpg.add_table_column(label="Info", parent=TAG_OT_MVA_CUSTOM_COLS_TABLE, width_stretch=True)
        with dpg.table_row(parent=TAG_OT_MVA_CUSTOM_COLS_TABLE): dpg.add_text("No numeric columns available.")
        return
    dpg.add_table_column(label="Select", parent=TAG_OT_MVA_CUSTOM_COLS_TABLE, width_fixed=True, init_width_or_weight=70)
    dpg.add_table_column(label="Column Name", parent=TAG_OT_MVA_CUSTOM_COLS_TABLE, width_stretch=True)
    for col_name in numeric_cols:
        with dpg.table_row(parent=TAG_OT_MVA_CUSTOM_COLS_TABLE):
            checkbox_tag = f"mva_cb_{''.join(filter(str.isalnum, col_name))}_{dpg.generate_uuid()}" # 더 고유한 태그
            _mva_custom_col_checkbox_tags[col_name] = checkbox_tag
            is_selected = col_name in _mva_custom_selected_columns
            dpg.add_checkbox(tag=checkbox_tag, default_value=is_selected, user_data=col_name, callback=_on_mva_custom_col_checkbox_change)
            dpg.add_text(col_name)

def _on_mva_iso_forest_contam_change(sender, app_data_str: str, user_data):
    global _mva_iso_forest_contamination
    cleaned_app_data_str = app_data_str.strip()
    if not cleaned_app_data_str or cleaned_app_data_str.lower() == 'auto':
        _mva_iso_forest_contamination = 'auto'
        if dpg.get_value(sender) != "": dpg.set_value(sender, "")
        _log_message("MVA Isolation Forest contamination set to: 'auto'")
    else:
        try:
            val = float(cleaned_app_data_str)
            if 0.0001 <= val <= 0.5: _mva_iso_forest_contamination = val; _log_message(f"MVA Isolation Forest contamination set to: {val:.4f}")
            else: _mva_iso_forest_contamination = 'auto'; dpg.set_value(sender, ""); _log_message(f"MVA Contam value {val} out of range. Using 'auto'.")
        except ValueError: _mva_iso_forest_contamination = 'auto'; dpg.set_value(sender, ""); _log_message(f"Invalid float for MVA contam: '{cleaned_app_data_str}'. Using 'auto'.")

def _get_mva_columns_to_analyze() -> List[str]:
    if _current_df_for_this_step is None: return []
    all_numeric_cols = [col for col in _current_df_for_this_step.columns if pd.api.types.is_numeric_dtype(_current_df_for_this_step[col])]
    if _mva_variable_selection_method == "All Numeric":
        _log_message(f"MVA using all {len(all_numeric_cols)} numeric columns.")
        return all_numeric_cols
    elif _mva_variable_selection_method == "Recommended":
        _log_message("MVA recommended column selection is a TODO. Using all numeric columns for now.")
        return all_numeric_cols 
    elif _mva_variable_selection_method == "Custom":
        valid_custom_cols = [col for col in _mva_custom_selected_columns if col in all_numeric_cols]
        _log_message(f"MVA using {len(valid_custom_cols)} custom selected numeric columns: {valid_custom_cols}")
        return valid_custom_cols
    return []

def _detect_outliers_mva_iso_forest(df_subset: pd.DataFrame, contamination: Union[str, float]) -> np.ndarray:
    if df_subset.empty or df_subset.shape[1] == 0: _log_message("MVA IsoForest: Input empty."); return np.array([])
    df_numeric_imputed = df_subset.copy()
    for col in df_numeric_imputed.columns:
        if pd.api.types.is_numeric_dtype(df_numeric_imputed[col]):
            if df_numeric_imputed[col].isnull().any(): df_numeric_imputed[col].fillna(df_numeric_imputed[col].median(), inplace=True)
        else: _log_message(f"  MVA IsoForest: Non-numeric col '{col}'. Skipping."); return np.array([])
    if df_numeric_imputed.shape[0] < 2 : _log_message("MVA IsoForest: Not enough samples."); return np.array([])
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    try:
        model.fit(df_numeric_imputed); predictions = model.predict(df_numeric_imputed)
    except ValueError as e:
        _log_message(f"  MVA IsoForest error (contam: {contamination}): {e}. Trying 'auto'.")
        if contamination != 'auto':
            try: model_auto = IsolationForest(contamination='auto', random_state=42, n_estimators=100); model_auto.fit(df_numeric_imputed); predictions = model_auto.predict(df_numeric_imputed)
            except Exception as e_auto: _log_message(f"  MVA IsoForest with 'auto' also failed: {e_auto}"); return np.array([])
        else: return np.array([])
    return df_numeric_imputed.index[predictions == -1].to_numpy()

def _run_mva_outlier_detection_logic():
    global _current_df_for_this_step, _df_with_mva_outliers, _mva_outlier_row_indices
    _log_message("Run Multivariate Outlier Detection button clicked.")
    try:
        if _current_df_for_this_step is None: 
            _log_message("Error: No data loaded for MVA detection.")
            if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, "Error: No data.")
            return
        cols_to_analyze = _get_mva_columns_to_analyze()
        if not cols_to_analyze or len(cols_to_analyze) < 2:
             _log_message("MVA Error: Select at least 2 numeric columns.")
             if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, "Error: Select at least 2 numeric columns.")
             return
        df_subset = _current_df_for_this_step[cols_to_analyze]
        _log_message(f"--- Starting MVA Detection (Isolation Forest) on {len(cols_to_analyze)} columns ---")
        _log_message(f"  Parameters: Contamination='{_mva_iso_forest_contamination}'")
        _mva_outlier_row_indices = _detect_outliers_mva_iso_forest(df_subset, _mva_iso_forest_contamination)
        num_outliers = len(_mva_outlier_row_indices)
        total_rows = len(_current_df_for_this_step)
        percentage = (num_outliers / total_rows * 100) if total_rows > 0 else 0
        summary = f"MVA Detection Complete:\n- Method: Isolation Forest\n- Columns: {len(cols_to_analyze)}\n- Outlier Rows: {num_outliers} ({percentage:.2f}%)"
        _log_message(summary)
        if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, summary)
        _df_with_mva_outliers = _current_df_for_this_step.copy()
        _df_with_mva_outliers['is_mva_outlier'] = False # MVA 결과 플래그 컬럼
        if num_outliers > 0: _df_with_mva_outliers.loc[_mva_outlier_row_indices, 'is_mva_outlier'] = True
        _generate_mva_plot_texture()
        if _util_funcs and '_show_simple_modal_message' in _util_funcs: _util_funcs['_show_simple_modal_message']("MVA Complete", f"MVA detection finished. Found {num_outliers} outlier rows.")
    except Exception as e: 
        _log_message(f"CRITICAL ERROR in _run_mva_outlier_detection_logic: {e}"); traceback.print_exc()
        if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, f"Error: {e}")

def _generate_mva_plot_texture():
    global _mva_active_plot_texture_id, _df_with_mva_outliers, _current_df_for_this_step
    _log_message("Generating MVA plot (UMAP)...")
    _clear_mva_visualization_plot() 
    if _df_with_mva_outliers is None or 'is_mva_outlier' not in _df_with_mva_outliers.columns:
        _log_message("MVA Plot: No MVA outlier detection data."); return
    cols_for_plot = _get_mva_columns_to_analyze()
    if not cols_for_plot or len(cols_for_plot) < 2: _log_message("MVA Plot: Not enough columns for UMAP."); return
    try:
        from umap import UMAP
        df_for_umap = _current_df_for_this_step[cols_for_plot].copy()
        for col in df_for_umap.columns: df_for_umap[col].fillna(df_for_umap[col].median(), inplace=True)
        if df_for_umap.shape[0] < 2: _log_message("MVA Plot: Not enough samples for UMAP."); return
        
        n_neighbors_umap = min(15, df_for_umap.shape[0]-1 if df_for_umap.shape[0]>1 else 1)
        if n_neighbors_umap <= 0: n_neighbors_umap = 1 # n_neighbors는 0보다 커야 함

        reducer = UMAP(n_neighbors=n_neighbors_umap, n_components=2, min_dist=0.1, random_state=42)
        embedding = reducer.fit_transform(df_for_umap)
        
        fig_width, fig_height = 15.6, 5.85 # 확장된 플롯 크기
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        hue_data = _df_with_mva_outliers['is_mva_outlier']
        
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=hue_data, palette={True: "red", False: "cornflowerblue"}, style=hue_data, markers={True:"X", False:"o"}, alpha=0.7, s=30, ax=ax, legend="brief")
        ax.set_title(f"Multivariate Outliers (UMAP of {len(cols_for_plot)} Vars)", fontsize=10); ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9); ax.tick_params(labelsize=8); ax.grid(True, alpha=0.3)
        if hue_data.nunique() > 1 : ax.legend(title="MVA Outlier", fontsize=8, loc='best')
        else: ax.legend().remove() if ax.get_legend() else None # 범례가 있으면 제거
        fig.suptitle("Multivariate Outlier Visualization", fontsize=12, y=0.99); plt.tight_layout(rect=[0,0.03,1,0.95])
        
        texture_tag, tex_w, tex_h = _s5_plot_to_dpg_texture(fig)
        if _mva_active_plot_texture_id and _mva_active_plot_texture_id != TAG_OT_DEFAULT_PLOT_TEXTURE_MVA and dpg.does_item_exist(_mva_active_plot_texture_id): # MVA 기본 텍스처 사용
            try: dpg.delete_item(_mva_active_plot_texture_id)
            except Exception as e: _log_message(f"Error deleting MVA tex: {e}")
        if texture_tag and tex_w > 0 and tex_h > 0:
            _mva_active_plot_texture_id = texture_tag
            if dpg.does_item_exist(TAG_OT_MVA_VISUALIZATION_IMAGE):
                parent_w = dpg.get_item_width(dpg.get_item_parent(TAG_OT_MVA_VISUALIZATION_IMAGE))
                disp_w = min(tex_w, (parent_w-20 if parent_w and parent_w>20 else 880))
                disp_h = int(tex_h * (disp_w/tex_w)) if tex_w > 0 else tex_h
                dpg.configure_item(TAG_OT_MVA_VISUALIZATION_IMAGE, texture_tag=_mva_active_plot_texture_id, width=int(disp_w), height=int(disp_h), show=True)
            _log_message("MVA UMAP plot generated.")
        else: _clear_mva_visualization_plot(); _log_message("Failed to generate MVA plot.")
    except ImportError: _log_message("MVA Plot Error: UMAP library not found. Please install 'umap-learn'.")
    except Exception as e: _log_message(f"Error generating MVA plot: {e}\n{traceback.format_exc()}"); _clear_mva_visualization_plot()

def _clear_mva_visualization_plot():
    global _mva_active_plot_texture_id
    if not dpg.is_dearpygui_running(): return
    # TAG_OT_MVA_DEFAULT_PLOT_TEXTURE 사용
    if dpg.does_item_exist(TAG_OT_MVA_VISUALIZATION_IMAGE) and dpg.does_item_exist(TAG_OT_DEFAULT_PLOT_TEXTURE_MVA):
        cfg = dpg.get_item_configuration(TAG_OT_DEFAULT_PLOT_TEXTURE_MVA); 
        w,h = (cfg.get('width',100), cfg.get('height',30)) if cfg else (100,30)
        dpg.configure_item(TAG_OT_MVA_VISUALIZATION_IMAGE, texture_tag=TAG_OT_DEFAULT_PLOT_TEXTURE_MVA, width=w, height=h, show=True)
    if _mva_active_plot_texture_id and _mva_active_plot_texture_id != TAG_OT_DEFAULT_PLOT_TEXTURE_MVA and dpg.does_item_exist(_mva_active_plot_texture_id):
        try: dpg.delete_item(_mva_active_plot_texture_id)
        except Exception as e: _log_message(f"Error deleting MVA tex: {e}")
    _mva_active_plot_texture_id = TAG_OT_DEFAULT_PLOT_TEXTURE_MVA


# --- Main UI Creation and Update Functions ---
def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _main_app_callbacks, _util_funcs, _uni_active_plot_texture_id, _mva_active_plot_texture_id
    _main_app_callbacks = main_callbacks
    if 'get_util_funcs' in main_callbacks: _util_funcs = main_callbacks['get_util_funcs']()
    if not dpg.does_item_exist("texture_registry"): dpg.add_texture_registry(tag="texture_registry", show=False)
    
    # 기본 텍스처 생성 (UNI & MVA)
    default_texture_tags = {
        "_UNI": TAG_OT_DEFAULT_PLOT_TEXTURE_UNI,
        "_MVA": TAG_OT_DEFAULT_PLOT_TEXTURE_MVA
    }
    active_id_vars = {
        "_UNI": "_uni_active_plot_texture_id",
        "_MVA": "_mva_active_plot_texture_id"
    }
    for suffix, tag_constant_name in default_texture_tags.items():
        if not dpg.does_item_exist(tag_constant_name):
            try: dpg.add_static_texture(width=100, height=30, default_value=[0.0]*100*30*4, tag=tag_constant_name, parent="texture_registry")
            except Exception as e: print(f"Error creating default texture {tag_constant_name}: {e}")
        globals()[active_id_vars[suffix]] = tag_constant_name # 전역 변수 초기화


    main_callbacks['register_step_group_tag'](step_name, TAG_OT_STEP_GROUP)
    with dpg.group(tag=TAG_OT_STEP_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---"); dpg.add_separator()

        with dpg.tab_bar(tag="step5_outlier_tab_bar"):
            # --- Univariate Tab ---
            with dpg.tab(label="Univariate Outlier Detection", tag=TAG_OT_UNIVARIATE_TAB):
                dpg.add_text("1. Configure & Run Univariate Outlier Detection", color=[255, 255, 0])
                with dpg.group(horizontal=True):
                    dpg.add_text("Detection Method:")
                    dpg.add_radio_button(["IQR", "HBOS", "ECOD"], tag=TAG_OT_DETECT_METHOD_RADIO_UNI, default_value=_uni_selected_detection_method, horizontal=True, callback=_on_uni_detection_method_change)
                dpg.add_text("Detection Parameters (applied to eligible columns):")
                with dpg.group(horizontal=True, tag=TAG_OT_IQR_MULTIPLIER_INPUT + "_group_parent_uni"):
                    dpg.add_text("IQR Multiplier:", tag=TAG_OT_IQR_MULTIPLIER_INPUT + "_label")
                    dpg.add_input_float(tag=TAG_OT_IQR_MULTIPLIER_INPUT, width=120, default_value=_uni_iqr_multiplier, step=0.1, callback=_on_uni_iqr_multiplier_change)
                with dpg.group(horizontal=True, tag=TAG_OT_HBOS_N_BINS_INPUT + "_group_parent_uni"):
                    dpg.add_text("HBOS n_bins:", tag=TAG_OT_HBOS_N_BINS_INPUT + "_label")
                    dpg.add_input_int(tag=TAG_OT_HBOS_N_BINS_INPUT, width=120, default_value=_uni_hbos_n_bins, step=1, min_value=2, min_clamped=True, callback=_on_uni_hbos_n_bins_change)
                with dpg.group(horizontal=True, tag=TAG_OT_ECOD_CONTAM_INPUT_UNI + "_group_parent_uni"):
                    dpg.add_text("ECOD Contamination (0.0-0.5):", tag=TAG_OT_ECOD_CONTAM_INPUT_UNI + "_label")
                    dpg.add_input_float(tag=TAG_OT_ECOD_CONTAM_INPUT_UNI, width=120, default_value=_uni_ecod_contamination, min_value=0.0001, max_value=0.5, min_clamped=True, max_clamped=True, step=0.01, format="%.4f", callback=_on_uni_ecod_contamination_change)
                _update_uni_parameter_fields_visibility()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Run Univariate Detection", tag=TAG_OT_DETECT_BUTTON_UNI, width=-1, height=30, callback=_run_uni_outlier_detection_logic)
                    dpg.add_button(label="Set Recommended Univariate Params", tag=TAG_OT_RECOMMEND_PARAMS_BUTTON_UNI, width=-1, height=30, callback=_set_uni_recommended_detection_parameters)
                dpg.add_spacer(height=5)
                dpg.add_text("2. Univariate Detection Summary & Visualization (Click row in table to visualize)", color=[255, 255, 0])
                with dpg.table(tag=TAG_OT_DETECTION_RESULTS_TABLE_UNI, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=120, borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                    dpg.add_table_column(label="Info", parent=TAG_OT_DETECTION_RESULTS_TABLE_UNI, width_stretch=True); dpg.add_table_row(parent=TAG_OT_DETECTION_RESULTS_TABLE_UNI); dpg.add_text("Run univariate detection.", parent=dpg.last_item())
                with dpg.group(tag=TAG_OT_VISUALIZATION_GROUP_UNI, horizontal=False):
                    img_w_uni, img_h_uni = 910, 390 # 확장된 플롯 크기 반영
                    dpg.add_image(texture_tag=_uni_active_plot_texture_id or TAG_OT_DEFAULT_PLOT_TEXTURE_UNI, tag=TAG_OT_VISUALIZATION_PLOT_IMAGE_UNI, show=True, width=img_w_uni, height=img_h_uni) 
                dpg.add_spacer(height=5)
                dpg.add_text("3. Configure Univariate Outlier Treatment", color=[255, 255, 0])
                with dpg.table(tag=TAG_OT_TREATMENT_TABLE_UNI, header_row=True, resizable=True,
                               policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=150, scrollX=True,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                    # 테이블 초기화 (이전과 동일)
                    dpg.add_table_column(label="Info", parent=TAG_OT_TREATMENT_TABLE_UNI, width_stretch=True)
                    with dpg.table_row(parent=TAG_OT_TREATMENT_TABLE_UNI):
                        dpg.add_text("Run univariate detection to configure treatments.", parent=dpg.last_item())
                
                # 문제의 버튼 그룹을 다시 명확하게 정의합니다.
                with dpg.group(horizontal=True, tag="uni_treatment_buttons_group"): # 그룹에 고유 태그 부여 (디버깅용)
                    dpg.add_button(label="Set Recommended Treatments",
                                   tag=TAG_OT_RECOMMEND_TREATMENTS_BUTTON_UNI, 
                                   width=-1, height=30, 
                                   callback=_set_uni_recommended_treatments_logic)
                                   
                    dpg.add_button(label="Reset Treatment Selections", 
                                   tag=TAG_OT_RESET_TREATMENTS_BUTTON_UNI, # 상수 태그 사용 확인
                                   width=-1, height=30, 
                                   callback=_reset_uni_treatment_selections_to_default)
                                   
                    dpg.add_button(label="Apply Selected Treatments", 
                                   tag=TAG_OT_APPLY_TREATMENT_BUTTON_UNI, # 상수 태그 사용 확인
                                   width=-1, height=30, 
                                   callback=_apply_uni_outlier_treatment_logic)

            # --- Multivariate Tab ---
            with dpg.tab(label="Multivariate Outlier Detection", tag=TAG_OT_MULTIVARIATE_TAB):
                dpg.add_text("1. Configure & Run Multivariate Outlier Detection", color=[255, 255, 0])
                dpg.add_text("Detection Method: Isolation Forest (on selected numeric columns)")
                with dpg.group(horizontal=True):
                    dpg.add_text("Contamination ('auto' or 0.0001-0.5):", tag=TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT + "_label")
                    dpg.add_input_text(tag=TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT, width=120, default_value="auto", hint="e.g., 0.1 or auto", callback=_on_mva_iso_forest_contam_change)
                
                dpg.add_text("Select Variables for Multivariate Analysis:")
                dpg.add_radio_button(items=["All Numeric Columns", "Recommended Columns (TODO)", "Select Custom Columns"], 
                                     tag=TAG_OT_MVA_VAR_METHOD_RADIO, default_value=_mva_variable_selection_method, 
                                     horizontal=True, callback=_on_mva_var_method_change)
                
                dpg.add_text("Custom Numeric Columns (if selected):", tag=TAG_OT_MVA_CUSTOM_COLS_TABLE + "_label", show=False) # 테이블용 레이블
                # 테이블을 자식 창으로 감싸서 스크롤 및 크기 제어 용이하게 함
                with dpg.child_window(tag=TAG_OT_MVA_CUSTOM_COLS_TABLE + "_child", show=False, height=150, border=True):
                    # 실제 테이블은 여기에 생성됨 (TAG_OT_MVA_CUSTOM_COLS_TABLE 사용)
                    # _populate_mva_custom_cols_table 함수가 이 자식 창 내에 테이블을 그림
                    pass 
                                
                dpg.add_button(label="Run Multivariate Detection", tag=TAG_OT_MVA_DETECT_BUTTON, width=-1, height=30, callback=_run_mva_outlier_detection_logic)
                dpg.add_spacer(height=5)

                dpg.add_text("2. Multivariate Detection Summary & Visualization", color=[255, 255, 0])
                dpg.add_text("Summary will appear here.", tag=TAG_OT_MVA_RESULTS_TEXT, wrap=-1)
                dpg.add_spacer(height=5)
                with dpg.group(tag="mva_visualization_parent_group", horizontal=False): # MVA 시각화 이미지 부모 그룹
                    img_w_mva, img_h_mva = 910, 390  # 확장된 플롯 크기 반영
                    dpg.add_image(texture_tag=_mva_active_plot_texture_id or TAG_OT_DEFAULT_PLOT_TEXTURE_MVA, 
                                  tag=TAG_OT_MVA_VISUALIZATION_IMAGE, show=True, 
                                  width=img_w_mva, height=img_h_mva)
                
                dpg.add_spacer(height=10)
                dpg.add_text("Multivariate outlier treatment: Typically involves row removal or flagging.", color=(150,150,150))


        dpg.add_separator()
        dpg.add_text("Processing Log (Common for Step 5)", color=[255, 255, 0])
        with dpg.child_window(tag=TAG_OT_LOG_TEXT_AREA, height=80, border=True): 
            dpg.add_text("Logs will appear here...", tag=TAG_OT_LOG_TEXT, wrap=-1)
            
    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui(df_input_for_step: Optional[pd.DataFrame], main_callbacks: dict):
    global _main_app_callbacks, _util_funcs, _current_df_for_this_step
    if not dpg.is_dearpygui_running(): return

    if not _main_app_callbacks: _main_app_callbacks = main_callbacks
    if not _util_funcs and 'get_util_funcs' in _main_app_callbacks: _util_funcs = main_callbacks['get_util_funcs']()
    
    is_new_data = _current_df_for_this_step is not df_input_for_step
    _current_df_for_this_step = df_input_for_step

    if not dpg.does_item_exist(TAG_OT_STEP_GROUP): return

    if _current_df_for_this_step is None or is_new_data:
        reset_outlier_treatment_state() 
        if dpg.does_item_exist(TAG_OT_LOG_TEXT):
            msg = "Data loaded. Configure detection." if _current_df_for_this_step is not None else "Load data for Step 5."
            dpg.set_value(TAG_OT_LOG_TEXT, msg)
    
    # MVA 사용자 지정 컬럼 테이블 채우기 (update_ui 호출 시마다 df 상태에 따라)
    if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE + "_child"): # 자식창이 존재하면
         _populate_mva_custom_cols_table(_current_df_for_this_step) # 테이블 내용 채우기
    
    _update_uni_parameter_fields_visibility()
    # MVA 변수 선택 UI 가시성 업데이트
    show_custom_table_mva = (_mva_variable_selection_method == "Custom")
    if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE + "_child"): # 자식창의 show/hide
        dpg.configure_item(TAG_OT_MVA_CUSTOM_COLS_TABLE + "_child", show=show_custom_table_mva)
    if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE + "_label"):
        dpg.configure_item(TAG_OT_MVA_CUSTOM_COLS_TABLE + "_label", show=show_custom_table_mva)

def reset_outlier_treatment_state(): 
    global _current_df_for_this_step, _df_with_uni_detected_outliers, _df_after_uni_treatment, _uni_active_plot_texture_id
    global _uni_detected_outlier_indices, _uni_outlier_summary_data, _uni_treatment_selections, _uni_columns_eligible_for_detection
    global _uni_selected_detection_method, _uni_iqr_multiplier, _uni_hbos_n_bins, _uni_ecod_contamination, _uni_currently_visualized_column, _uni_all_selectable_tags_in_table
    global _mva_variable_selection_method, _mva_custom_selected_columns, _mva_iso_forest_contamination, _df_with_mva_outliers, _mva_outlier_row_indices, _mva_active_plot_texture_id, _mva_custom_col_checkbox_tags

    _current_df_for_this_step = None; 
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_LOG_TEXT): dpg.set_value(TAG_OT_LOG_TEXT, "State reset. Logs cleared.")
    
    _df_with_uni_detected_outliers = None; _df_after_uni_treatment = None
    _uni_detected_outlier_indices.clear(); _uni_outlier_summary_data.clear(); _uni_treatment_selections.clear(); 
    _uni_columns_eligible_for_detection.clear(); _uni_all_selectable_tags_in_table.clear()
    _uni_selected_detection_method = "IQR"; _uni_iqr_multiplier = DEFAULT_UNI_IQR_MULTIPLIER
    _uni_hbos_n_bins = DEFAULT_UNI_HBOS_N_BINS; _uni_ecod_contamination = DEFAULT_UNI_ECOD_CONTAMINATION
    _uni_currently_visualized_column = None
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_DETECT_METHOD_RADIO_UNI): dpg.set_value(TAG_OT_DETECT_METHOD_RADIO_UNI, _uni_selected_detection_method)
        if dpg.does_item_exist(TAG_OT_IQR_MULTIPLIER_INPUT): dpg.set_value(TAG_OT_IQR_MULTIPLIER_INPUT, _uni_iqr_multiplier)
        if dpg.does_item_exist(TAG_OT_HBOS_N_BINS_INPUT): dpg.set_value(TAG_OT_HBOS_N_BINS_INPUT, _uni_hbos_n_bins)
        if dpg.does_item_exist(TAG_OT_ECOD_CONTAM_INPUT_UNI): dpg.set_value(TAG_OT_ECOD_CONTAM_INPUT_UNI, _uni_ecod_contamination)
        _populate_uni_detection_results_table(); _populate_uni_treatment_table(); _clear_uni_visualization_plot()
    _update_uni_parameter_fields_visibility()

    _mva_variable_selection_method = "All Numeric"; _mva_custom_selected_columns.clear(); _mva_custom_col_checkbox_tags.clear()
    _mva_iso_forest_contamination = DEFAULT_MVA_ISO_FOREST_CONTAMINATION
    _df_with_mva_outliers = None; _mva_outlier_row_indices = None
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_MVA_VAR_METHOD_RADIO): dpg.set_value(TAG_OT_MVA_VAR_METHOD_RADIO, _mva_variable_selection_method)
        if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE + "_child"): # 자식창 제어
            _populate_mva_custom_cols_table(None) 
            dpg.configure_item(TAG_OT_MVA_CUSTOM_COLS_TABLE + "_child", show=False)
        if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE + "_label"):
            dpg.configure_item(TAG_OT_MVA_CUSTOM_COLS_TABLE + "_label", show=False)
        if dpg.does_item_exist(TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT): dpg.set_value(TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT, "auto" if _mva_iso_forest_contamination == 'auto' else str(_mva_iso_forest_contamination))
        if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, "Run multivariate detection.")
        _clear_mva_visualization_plot()
    print("Step 5: Outlier Treatment state (Univariate & Multivariate) has been reset.")

def get_outlier_treatment_settings_for_saving() -> dict: 
    return {
        "uni_selected_detection_method": _uni_selected_detection_method,
        "uni_iqr_multiplier": _uni_iqr_multiplier,
        "uni_hbos_n_bins": _uni_hbos_n_bins,
        "uni_ecod_contamination": _uni_ecod_contamination,
        "uni_treatment_selections": _uni_treatment_selections.copy(),
        "mva_variable_selection_method": _mva_variable_selection_method,
        "mva_custom_selected_columns": _mva_custom_selected_columns[:],
        "mva_iso_forest_contamination": _mva_iso_forest_contamination,
    }

def apply_outlier_treatment_settings_and_process(df_input: pd.DataFrame, settings: dict, main_callbacks: dict): 
    global _uni_selected_detection_method, _uni_iqr_multiplier, _uni_hbos_n_bins, _uni_ecod_contamination, _uni_treatment_selections
    global _mva_variable_selection_method, _mva_custom_selected_columns, _mva_iso_forest_contamination
    global _main_app_callbacks, _current_df_for_this_step, _df_after_uni_treatment, _df_with_uni_detected_outliers, _uni_outlier_summary_data, _uni_detected_outlier_indices, _uni_currently_visualized_column
    global _df_with_mva_outliers, _mva_outlier_row_indices
    if not _main_app_callbacks: _main_app_callbacks = main_callbacks
    _current_df_for_this_step = df_input
    _df_after_uni_treatment = None; _df_with_uni_detected_outliers = None; _uni_outlier_summary_data.clear(); _uni_detected_outlier_indices.clear(); _uni_currently_visualized_column = None
    _df_with_mva_outliers = None; _mva_outlier_row_indices = None; 
    
    _uni_selected_detection_method = settings.get("uni_selected_detection_method", "IQR")
    _uni_iqr_multiplier = settings.get("uni_iqr_multiplier", DEFAULT_UNI_IQR_MULTIPLIER)
    _uni_hbos_n_bins = settings.get("uni_hbos_n_bins", DEFAULT_UNI_HBOS_N_BINS) 
    _uni_ecod_contamination = settings.get("uni_ecod_contamination", DEFAULT_UNI_ECOD_CONTAMINATION) 
    _uni_treatment_selections = settings.get("uni_treatment_selections", {}).copy()
    _mva_variable_selection_method = settings.get("mva_variable_selection_method", "All Numeric")
    _mva_custom_selected_columns = settings.get("mva_custom_selected_columns", [])[:] 
    _mva_iso_forest_contamination = settings.get("mva_iso_forest_contamination", DEFAULT_MVA_ISO_FOREST_CONTAMINATION)

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_DETECT_METHOD_RADIO_UNI): dpg.set_value(TAG_OT_DETECT_METHOD_RADIO_UNI, _uni_selected_detection_method)
        if dpg.does_item_exist(TAG_OT_IQR_MULTIPLIER_INPUT): dpg.set_value(TAG_OT_IQR_MULTIPLIER_INPUT, _uni_iqr_multiplier)
        if dpg.does_item_exist(TAG_OT_HBOS_N_BINS_INPUT): dpg.set_value(TAG_OT_HBOS_N_BINS_INPUT, _uni_hbos_n_bins)
        if dpg.does_item_exist(TAG_OT_ECOD_CONTAM_INPUT_UNI): dpg.set_value(TAG_OT_ECOD_CONTAM_INPUT_UNI, _uni_ecod_contamination)
        if dpg.does_item_exist(TAG_OT_MVA_VAR_METHOD_RADIO): dpg.set_value(TAG_OT_MVA_VAR_METHOD_RADIO, _mva_variable_selection_method)
        if dpg.does_item_exist(TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT): 
            contam_val = _mva_iso_forest_contamination; dpg.set_value(TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT, "auto" if contam_val == 'auto' else str(contam_val))
            
    update_ui(df_input, main_callbacks) 
    _log_message("Step 5 Outlier Treatment settings applied. UI reflects saved parameters. Please run detection and treatment manually if needed.")