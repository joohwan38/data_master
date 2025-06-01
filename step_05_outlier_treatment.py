# step_05_outlier_treatment.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

# PyOD 모델 임포트 - 이 부분이 파일 상단에 추가되어야 합니다.
try:
    from pyod.models.hbos import HBOS as PyOD_HBOS
    from pyod.models.ecod import ECOD as PyOD_ECOD
except ImportError:
    # PyOD가 설치되지 않았을 경우를 대비한 로그 또는 예외 처리 (여기서는 print로 간단히)
    print("Warning: PyOD library not found. HBOS and ECOD outlier detection will not be available.")
    PyOD_HBOS = None
    PyOD_ECOD = None

import matplotlib.pyplot as plt
import io
from PIL import Image
import traceback

# --- DPG Tags for Step 5 ---
TAG_OT_STEP_GROUP = "step5_outlier_treatment_group"
TAG_OT_DETECT_METHOD_RADIO = "step5_ot_detect_method_radio"
TAG_OT_IQR_MULTIPLIER_INPUT = "step5_ot_iqr_multiplier_input"
# Isolation Forest 태그 제거
TAG_OT_HBOS_N_BINS_INPUT = "step5_ot_hbos_n_bins_input" # HBOS용 새 태그
TAG_OT_ECOD_CONTAM_INPUT = "step5_ot_ecod_contam_input" # ECOD용 새 태그

TAG_OT_DETECT_BUTTON = "step5_ot_detect_button"
TAG_OT_RECOMMEND_PARAMS_BUTTON = "step5_ot_recommend_params_button"
TAG_OT_DETECTION_RESULTS_TABLE = "step5_ot_detection_results_table"

TAG_OT_VISUALIZATION_GROUP = "step5_ot_visualization_group"
TAG_OT_VISUALIZATION_PLOT_IMAGE = "step5_ot_visualization_plot_image"
TAG_OT_DEFAULT_PLOT_TEXTURE = "step5_ot_default_plot_texture"

TAG_OT_TREATMENT_TABLE = "step5_ot_treatment_table"
TAG_OT_RECOMMEND_TREATMENTS_BUTTON = "step5_ot_recommend_treatments_button"
TAG_OT_RESET_TREATMENTS_BUTTON = "step5_ot_reset_treatments_button" # 이전 추가 사항
TAG_OT_APPLY_TREATMENT_BUTTON = "step5_ot_apply_treatment_button"
TAG_OT_LOG_TEXT_AREA = "step5_ot_log_text_area"
TAG_OT_LOG_TEXT = "step_ot_log_text"

# --- Constants for Filtering ---
MIN_UNIQUE_VALUES_FOR_OUTLIER_DETECTION = 30 # 필요시 조정
MAX_MISSING_RATIO_FOR_OUTLIER_DETECTION = 0.5
MIN_VARIANCE_FOR_OUTLIER_DETECTION = 1e-5

# --- Module State Variables ---
_main_app_callbacks: Optional[Dict[str, Any]] = None
_util_funcs: Optional[Dict[str, Any]] = None
_current_df_for_this_step: Optional[pd.DataFrame] = None
_df_with_detected_outliers: Optional[pd.DataFrame] = None
_df_after_treatment: Optional[pd.DataFrame] = None
_detected_outlier_indices: Dict[str, np.ndarray] = {}
_outlier_summary_data: List[Dict[str, Any]] = []
_columns_eligible_for_detection: List[str] = []

_selected_detection_method: str = "IQR" # 기본값 IQR 유지
_iqr_multiplier: float = 1.5
_hbos_n_bins: int = 20 # HBOS 기본 n_bins
_ecod_contamination: float = 0.1 # ECOD 기본 contamination (0.0 ~ 0.5)

_treatment_selections: Dict[str, Dict[str, Any]] = {}
_active_plot_texture_id: Optional[str] = None
_currently_visualized_column: Optional[str] = None
_all_selectable_tags_in_table: List[str] = []

DEFAULT_IQR_MULTIPLIER = 1.5
DEFAULT_HBOS_N_BINS = 20
DEFAULT_ECOD_CONTAMINATION = 0.1
RECOMMENDED_TREATMENT_METHOD = "Treat as Missing"

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
    if dpg.does_item_exist(TAG_OT_LOG_TEXT):
        current_log = dpg.get_value(TAG_OT_LOG_TEXT)
        max_log_entries = 200 
        log_lines = current_log.splitlines()
        if len(log_lines) >= max_log_entries:
            log_lines = log_lines[-(max_log_entries-1):]
        
        new_log_entry = f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}"
        new_log = "\n".join(log_lines) + "\n" + new_log_entry if current_log else new_log_entry
        
        dpg.set_value(TAG_OT_LOG_TEXT, new_log.strip())
        
        if dpg.does_item_exist(TAG_OT_LOG_TEXT_AREA):
            item_config = dpg.get_item_configuration(TAG_OT_LOG_TEXT_AREA)
            is_shown = item_config.get('show', True)
            item_info = dpg.get_item_info(TAG_OT_LOG_TEXT_AREA)
            is_child_window = item_info['type'] == "mvAppItemType::mvChildWindow" if item_info else False

            if is_shown and is_child_window:
                dpg.set_y_scroll(TAG_OT_LOG_TEXT_AREA, -1.0)

def _reset_treatment_selections_to_default():
    global _treatment_selections, _outlier_summary_data
    _log_message("Resetting all treatment selections to 'Do Not Treat'.")
    cols_with_detected_outliers = [item['Column'] for item in _outlier_summary_data if item.get('Detected Outliers', 0) > 0]
    if not cols_with_detected_outliers:
        _log_message("No columns with detected outliers to reset treatment for.")
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Info", "No outlier treatments to reset. Run detection first.")
        return
    for col_name in cols_with_detected_outliers:
        _treatment_selections[col_name] = {"method": "Do Not Treat"}
    _populate_treatment_table()
    _log_message("All treatment selections have been reset to 'Do Not Treat'.")
    if _util_funcs and '_show_simple_modal_message' in _util_funcs:
        _util_funcs['_show_simple_modal_message']("Info", "All outlier treatment selections have been reset to 'Do Not Treat'.")


def _update_parameter_fields_visibility():
    if not dpg.is_dearpygui_running(): return
    show_iqr = _selected_detection_method == "IQR"
    show_hbos = _selected_detection_method == "HBOS"
    show_ecod = _selected_detection_method == "ECOD"
    
    iqr_elements = [TAG_OT_IQR_MULTIPLIER_INPUT, TAG_OT_IQR_MULTIPLIER_INPUT + "_label"]
    hbos_elements = [TAG_OT_HBOS_N_BINS_INPUT, TAG_OT_HBOS_N_BINS_INPUT + "_label"]
    ecod_elements = [TAG_OT_ECOD_CONTAM_INPUT, TAG_OT_ECOD_CONTAM_INPUT + "_label"]

    for item_tag in iqr_elements:
        if dpg.does_item_exist(item_tag): dpg.configure_item(item_tag, show=show_iqr)
    for item_tag in hbos_elements:
        if dpg.does_item_exist(item_tag): dpg.configure_item(item_tag, show=show_hbos)
    for item_tag in ecod_elements:
        if dpg.does_item_exist(item_tag): dpg.configure_item(item_tag, show=show_ecod)

def _on_detection_method_change(sender, app_data: str, user_data):
    global _selected_detection_method
    _selected_detection_method = app_data
    _log_message(f"Outlier detection method changed to: {_selected_detection_method}")
    _update_parameter_fields_visibility()

def _on_iqr_multiplier_change(sender, app_data: float, user_data):
    global _iqr_multiplier
    _iqr_multiplier = app_data
    _log_message(f"IQR multiplier set to: {_iqr_multiplier}")

def _on_hbos_n_bins_change(sender, app_data: int, user_data):
    global _hbos_n_bins
    if app_data > 0 : # n_bins는 0보다 커야 함
        _hbos_n_bins = app_data
        _log_message(f"HBOS n_bins set to: {_hbos_n_bins}")
    else:
        dpg.set_value(sender, _hbos_n_bins) # 유효하지 않은 값이면 이전 값으로 복원
        _log_message(f"HBOS n_bins must be > 0. Reverted to {_hbos_n_bins}.")
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Input Error", "HBOS n_bins must be greater than 0.")


def _on_ecod_contamination_change(sender, app_data: float, user_data):
    global _ecod_contamination
    if 0.0 < app_data <= 0.5: # ECOD contamination 범위 (0 초과, 0.5 이하)
        _ecod_contamination = app_data
        _log_message(f"ECOD contamination set to: {_ecod_contamination:.4f}")
    else:
        dpg.set_value(sender, _ecod_contamination) # 유효하지 않은 값이면 이전 값으로 복원
        _log_message(f"ECOD contamination must be (0.0, 0.5]. Reverted to {_ecod_contamination:.4f}.")
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Input Error", "ECOD contamination must be between 0 (exclusive) and 0.5 (inclusive).")


def _set_recommended_detection_parameters():
    global _iqr_multiplier, _hbos_n_bins, _ecod_contamination
    _iqr_multiplier = DEFAULT_IQR_MULTIPLIER
    _hbos_n_bins = DEFAULT_HBOS_N_BINS
    _ecod_contamination = DEFAULT_ECOD_CONTAMINATION
    
    if dpg.does_item_exist(TAG_OT_IQR_MULTIPLIER_INPUT):
        dpg.set_value(TAG_OT_IQR_MULTIPLIER_INPUT, _iqr_multiplier)
    if dpg.does_item_exist(TAG_OT_HBOS_N_BINS_INPUT):
        dpg.set_value(TAG_OT_HBOS_N_BINS_INPUT, _hbos_n_bins)
    if dpg.does_item_exist(TAG_OT_ECOD_CONTAM_INPUT):
        dpg.set_value(TAG_OT_ECOD_CONTAM_INPUT, _ecod_contamination)
        
    _log_message(f"Recommended detection parameters set: IQR Mult={_iqr_multiplier}, HBOS n_bins={_hbos_n_bins}, ECOD Contam={_ecod_contamination:.2f}")
    _update_parameter_fields_visibility()
    if _util_funcs and '_show_simple_modal_message' in _util_funcs:
        _util_funcs['_show_simple_modal_message']("Info", "Recommended detection parameters have been applied.")

def _detect_outliers_iqr(series: pd.Series, multiplier: float) -> np.ndarray:
    # (이전과 동일)
    if not pd.api.types.is_numeric_dtype(series.dtype) or series.empty: return np.array([])
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr_value = q3 - q1
    if iqr_value == 0 or pd.isna(iqr_value): return np.array([])
    lower_bound = q1 - multiplier * iqr_value
    upper_bound = q3 + multiplier * iqr_value
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    return series.index[outlier_mask].to_numpy()

def _detect_outliers_hbos(series: pd.Series, n_bins_param: int, contamination_param: float) -> np.ndarray:
    if PyOD_HBOS is None:
        _log_message("Error: PyOD_HBOS model not loaded. Is PyOD installed?")
        return np.array([])
    if not pd.api.types.is_numeric_dtype(series.dtype): return np.array([])
    series_cleaned = series.dropna()
    if len(series_cleaned) < 2 or len(series_cleaned.unique()) < 2 : # HBOS는 최소 2개의 고유값 필요
        _log_message(f"  HBOS: Skipped for {series.name} due to insufficient unique data points after NaN removal ({len(series_cleaned.unique())} unique).")
        return np.array([])
    
    # n_bins는 샘플 수보다 작아야 함 (PyOD HBOS 내부 제약 조건)
    actual_n_bins = min(n_bins_param, len(series_cleaned.unique()) -1) if len(series_cleaned.unique()) > 1 else 1
    if actual_n_bins < 2 : actual_n_bins = 2 # 최소 n_bins
    if actual_n_bins != n_bins_param:
         _log_message(f"  HBOS: Adjusted n_bins for {series.name} from {n_bins_param} to {actual_n_bins} due to data size.")

    try:
        model = PyOD_HBOS(n_bins=actual_n_bins, contamination=contamination_param)
        model.fit(series_cleaned.values.reshape(-1, 1))
        predictions = model.labels_ # 0 for inliers, 1 for outliers
    except Exception as e:
        _log_message(f"  Error during HBOS for {series.name} (n_bins={actual_n_bins}, contam={contamination_param}): {e}")
        return np.array([])
    outlier_mask_on_cleaned = predictions == 1
    return series_cleaned.index[outlier_mask_on_cleaned].to_numpy()

def _detect_outliers_ecod(series: pd.Series, contamination_param: float) -> np.ndarray:
    if PyOD_ECOD is None:
        _log_message("Error: PyOD_ECOD model not loaded. Is PyOD installed?")
        return np.array([])
    if not pd.api.types.is_numeric_dtype(series.dtype): return np.array([])
    series_cleaned = series.dropna()
    if len(series_cleaned) < 2 : # ECOD도 최소 샘플 수 필요
        _log_message(f"  ECOD: Skipped for {series.name} due to insufficient data points after NaN removal.")
        return np.array([])
    try:
        model = PyOD_ECOD(contamination=contamination_param)
        model.fit(series_cleaned.values.reshape(-1, 1))
        predictions = model.labels_
    except Exception as e:
        _log_message(f"  Error during ECOD for {series.name} (contam={contamination_param}): {e}")
        return np.array([])
    outlier_mask_on_cleaned = predictions == 1
    return series_cleaned.index[outlier_mask_on_cleaned].to_numpy()

def _detect_outliers_iso_forest(series: pd.Series, contamination_param: Union[str, float]) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(series.dtype): return np.array([])
    series_cleaned = series.dropna()
    if len(series_cleaned) < 2: return np.array([])
    
    contam_value_for_model = contamination_param
    if isinstance(contamination_param, str) and contamination_param.lower() != 'auto':
        try: contam_value_for_model = float(contamination_param)
        except ValueError: contam_value_for_model = 'auto' # 변환 실패시 'auto'
    
    if isinstance(contam_value_for_model, float) and not (0 < contam_value_for_model <= 0.5):
        _log_message(f"  IsoForest: Contam value {contam_value_for_model} out of (0, 0.5]. Using 'auto'.")
        contam_value_for_model = 'auto'

    model = IsolationForest(contamination=contam_value_for_model, random_state=42, n_estimators=100)
    try:
        predictions = model.fit_predict(series_cleaned.values.reshape(-1, 1))
    except ValueError as e:
        _log_message(f"  IsoForest error for {series.name} (contam: {contam_value_for_model}): {e}. Trying 'auto'.")
        if contam_value_for_model != 'auto':
            try:
                model_auto = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
                predictions = model_auto.fit_predict(series_cleaned.values.reshape(-1, 1))
                _log_message(f"  Retried IsoForest for {series.name} with 'auto'.")
            except Exception as e_auto:
                _log_message(f"  IsoForest with 'auto' also failed for {series.name}: {e_auto}")
                return np.array([])
        else: return np.array([])
            
    outlier_mask_on_cleaned = predictions == -1
    return series_cleaned.index[outlier_mask_on_cleaned].to_numpy()

def _filter_columns_for_detection(df: pd.DataFrame) -> List[str]:
    eligible_cols = []
    if df is None: return eligible_cols
    s1_col_types = {}
    if _main_app_callbacks and 'get_column_analysis_types' in _main_app_callbacks:
        s1_col_types = _main_app_callbacks['get_column_analysis_types']()
    
    _log_message("Column filtering for outlier detection:")
    for col_name in df.columns:
        series = df[col_name]
        is_numeric_s1 = "Numeric" in s1_col_types.get(col_name, "") and "Binary" not in s1_col_types.get(col_name, "")
        is_numeric_pandas = pd.api.types.is_numeric_dtype(series.dtype)
        if not (is_numeric_s1 or is_numeric_pandas):
            continue
        nunique = series.nunique()
        if nunique < MIN_UNIQUE_VALUES_FOR_OUTLIER_DETECTION:
            continue
        s1_type = s1_col_types.get(col_name, "")
        if "Categorical" in s1_type or "Binary" in s1_type or "Text" in s1_type:
            continue
        missing_ratio = series.isnull().sum() / len(series) if len(series) > 0 else 1.0
        if missing_ratio > MAX_MISSING_RATIO_FOR_OUTLIER_DETECTION:
            continue
        series_dropped_na = series.dropna()
        if series_dropped_na.empty:
            continue
        variance = series_dropped_na.var()
        if pd.isna(variance) or variance < MIN_VARIANCE_FOR_OUTLIER_DETECTION:
            continue
        eligible_cols.append(col_name)
        _log_message(f"  Eligible for detection: '{col_name}'")
    if not eligible_cols: _log_message("  No columns found eligible after filtering.")
    return eligible_cols

def _run_outlier_detection_logic():
    global _current_df_for_this_step, _detected_outlier_indices, _outlier_summary_data, _df_with_detected_outliers, _columns_eligible_for_detection, _currently_visualized_column
    
    _log_message("Run Outlier Detection button clicked.")
    # ... (try-except 블록 시작 및 _current_df_for_this_step None 체크는 이전과 동일) ...
    try:
        if _current_df_for_this_step is None:
            _log_message("Error: No data loaded for outlier detection.")
            if _util_funcs and '_show_simple_modal_message' in _util_funcs:
                _util_funcs['_show_simple_modal_message']("Error", "No data available to detect outliers.")
            return

        _log_message(f"--- Starting Univariate Outlier Detection (Method: {_selected_detection_method}) ---")
        if _selected_detection_method == "IQR":
            _log_message(f"  Parameters: IQR Multiplier={_iqr_multiplier}")
        elif _selected_detection_method == "HBOS":
            _log_message(f"  Parameters: HBOS n_bins={_hbos_n_bins}, Contamination={_ecod_contamination}") # HBOS도 contamination 사용 가능 (PyOD HBOS 기본 파라미터) - ECOD의 contamination을 공유
        elif _selected_detection_method == "ECOD":
            _log_message(f"  Parameters: ECOD Contamination={_ecod_contamination}")

        _columns_eligible_for_detection = _filter_columns_for_detection(_current_df_for_this_step)

        if not _columns_eligible_for_detection:
            # ... (이전과 동일: 적합 컬럼 없음 처리) ...
            _log_message("Warning: No columns found suitable for outlier detection after filtering.")
            if _util_funcs and '_show_simple_modal_message' in _util_funcs:
                _util_funcs['_show_simple_modal_message']("Warning", "No columns were found suitable for automatic outlier detection based on current criteria.")
            _outlier_summary_data.clear(); _detected_outlier_indices.clear()
            _populate_detection_results_table(); _populate_treatment_table()
            _currently_visualized_column = None
            _clear_visualization_plot()
            return

        _detected_outlier_indices.clear(); _outlier_summary_data.clear()
        _df_with_detected_outliers = _current_df_for_this_step.copy()
        any_detection_performed_successfully = False # 성공적으로 탐지된 경우만 True
        first_col_with_outliers_for_auto_vis = None

        for col_name in _columns_eligible_for_detection:
            series = _current_df_for_this_step[col_name]
            col_outlier_indices = np.array([]) # 초기화
            
            if _selected_detection_method == "IQR":
                try:
                    col_outlier_indices = _detect_outliers_iqr(series, _iqr_multiplier)
                    _log_message(f"  IQR: Found {len(col_outlier_indices)} outliers for '{col_name}'.")
                    if len(col_outlier_indices) > 0: any_detection_performed_successfully = True
                except Exception as e_iqr: _log_message(f"  Error during IQR for '{col_name}': {e_iqr}")
            
            elif _selected_detection_method == "HBOS":
                if PyOD_HBOS is None: _log_message("HBOS skipped: PyOD library not available."); continue
                try: # HBOS는 contamination으로 ECOD의 것을 임시 사용 (파라미터 통일 차원)
                    col_outlier_indices = _detect_outliers_hbos(series, _hbos_n_bins, _ecod_contamination)
                    _log_message(f"  HBOS: Found {len(col_outlier_indices)} outliers for '{col_name}'.")
                    if len(col_outlier_indices) > 0: any_detection_performed_successfully = True
                except Exception as e_hbos: _log_message(f"  Error during HBOS for '{col_name}': {e_hbos}")
            
            elif _selected_detection_method == "ECOD":
                if PyOD_ECOD is None: _log_message("ECOD skipped: PyOD library not available."); continue
                try:
                    col_outlier_indices = _detect_outliers_ecod(series, _ecod_contamination)
                    _log_message(f"  ECOD: Found {len(col_outlier_indices)} outliers for '{col_name}'.")
                    if len(col_outlier_indices) > 0: any_detection_performed_successfully = True
                except Exception as e_ecod: _log_message(f"  Error during ECOD for '{col_name}': {e_ecod}")

            _detected_outlier_indices[col_name] = col_outlier_indices # set()으로 합치지 않고, 선택된 단일 방법의 결과만 저장
            
            outlier_flag_col_name = f"{col_name}_is_outlier"
            _df_with_detected_outliers[outlier_flag_col_name] = False
            if len(col_outlier_indices) > 0:
                _df_with_detected_outliers.loc[col_outlier_indices, outlier_flag_col_name] = True
                if first_col_with_outliers_for_auto_vis is None:
                    first_col_with_outliers_for_auto_vis = col_name

            num_outliers = len(col_outlier_indices)
            total_rows = len(series)
            percentage_outliers = (num_outliers / total_rows) * 100 if total_rows > 0 else 0
            _outlier_summary_data.append({"Column": col_name, "Detected Outliers": num_outliers, "Percentage (%)": f"{percentage_outliers:.2f}"})

        # ... (이하 _populate_detection_results_table, 시각화 호출, 완료 메시지 등은 이전과 유사하게, any_detection_performed_successfully 사용) ...
        _populate_detection_results_table()
        _populate_treatment_table()
        
        if first_col_with_outliers_for_auto_vis:
            _currently_visualized_column = first_col_with_outliers_for_auto_vis
            _generate_combined_plot_texture(first_col_with_outliers_for_auto_vis)
        elif _columns_eligible_for_detection:
            _currently_visualized_column = _columns_eligible_for_detection[0]
            _generate_combined_plot_texture(_columns_eligible_for_detection[0])
        else:
            _currently_visualized_column = None
            _clear_visualization_plot()

        if any_detection_performed_successfully or _columns_eligible_for_detection :
            _log_message("--- Univariate Outlier Detection Finished ---")
            if _util_funcs and '_show_simple_modal_message' in _util_funcs:
                msg = "Univariate Outlier detection finished."
                if not any_detection_performed_successfully and _columns_eligible_for_detection:
                    msg += " No outliers found in eligible columns."
                elif not _columns_eligible_for_detection:
                     msg = "Univariate Outlier detection finished. No columns were eligible."
                _util_funcs['_show_simple_modal_message']("Detection Complete", msg)
        else:
            _log_message("--- No outlier detection was performed (no eligible columns or errors). ---")
            if _util_funcs and '_show_simple_modal_message' in _util_funcs:
                _util_funcs['_show_simple_modal_message']("Detection Info", "No columns were eligible for outlier detection or detection failed.")
                
    except Exception as e:
        _log_message(f"CRITICAL ERROR in _run_outlier_detection_logic: {e}")
        traceback.print_exc()
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Critical Error", f"An unexpected error occurred during outlier detection: {e}")

def _populate_detection_results_table():
    global _all_selectable_tags_in_table
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_DETECTION_RESULTS_TABLE): return
    
    dpg.delete_item(TAG_OT_DETECTION_RESULTS_TABLE, children_only=True)
    _all_selectable_tags_in_table.clear()

    if not _outlier_summary_data:
        dpg.add_table_column(label="Info", parent=TAG_OT_DETECTION_RESULTS_TABLE, width_stretch=True)
        with dpg.table_row(parent=TAG_OT_DETECTION_RESULTS_TABLE):
            dpg.add_text("No detection results. Run detection.")
        return

    headers = ["Column (Click to Visualize)", "Detected Outliers", "Percentage (%)"]
    col_widths = [0.5, 0.25, 0.25] 

    for i, header in enumerate(headers):
        dpg.add_table_column(label=header, parent=TAG_OT_DETECTION_RESULTS_TABLE, 
                             init_width_or_weight=col_widths[i] if i < len(col_widths) else 0.0, 
                             width_stretch=True if i==0 else False) 

    for i, row_data in enumerate(_outlier_summary_data):
        with dpg.table_row(parent=TAG_OT_DETECTION_RESULTS_TABLE):
            column_name_str = str(row_data.get("Column", ""))
            
            selectable_tag = f"selectable_col_s5_{i}_{''.join(filter(str.isalnum, column_name_str))}"
            _all_selectable_tags_in_table.append(selectable_tag)
            
            dpg.add_selectable(label=column_name_str, tag=selectable_tag,
                               user_data=column_name_str, callback=_on_row_selectable_clicked,
                               span_columns=False) 
            
            dpg.add_text(str(row_data.get("Detected Outliers", "")))
            dpg.add_text(str(row_data.get("Percentage (%)", "")))

def _on_row_selectable_clicked(sender, app_data_is_selected: bool, user_data_col_name: str):
    global _currently_visualized_column, _all_selectable_tags_in_table
    if app_data_is_selected: 
        for tag in _all_selectable_tags_in_table:
            if tag != sender and dpg.does_item_exist(tag): 
                if dpg.get_value(tag): 
                    dpg.set_value(tag, False)
        _currently_visualized_column = user_data_col_name
        _generate_combined_plot_texture(user_data_col_name)
        _log_message(f"Visualizing outlier plots for: {user_data_col_name}")

def _generate_combined_plot_texture(column_name: str):
    global _active_plot_texture_id, _df_with_detected_outliers, _current_df_for_this_step
    if _df_with_detected_outliers is None or column_name not in _df_with_detected_outliers.columns or \
       _current_df_for_this_step is None or column_name not in _current_df_for_this_step.columns : # _current_df_for_this_step도 확인
        _clear_visualization_plot()
        _log_message(f"Cannot generate plot for '{column_name}': Data or column not available.")
        return

    original_series_for_boxplot = _current_df_for_this_step[column_name].dropna()
    outlier_flag_col = f"{column_name}_is_outlier"

    if not pd.api.types.is_numeric_dtype(original_series_for_boxplot.dtype) or original_series_for_boxplot.empty:
        _clear_visualization_plot()
        _log_message(f"Cannot generate plot for '{column_name}': Not numeric or empty after dropna.")
        return
    
    if outlier_flag_col not in _df_with_detected_outliers.columns:
        _clear_visualization_plot()
        _log_message(f"Outlier flag column '{outlier_flag_col}' not found for '{column_name}'. Run detection first.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(6, 7)) 
    plt.subplots_adjust(hspace=0.4) 

    axes[0].boxplot(original_series_for_boxplot, vert=False, patch_artist=True, medianprops={'color':'red', 'linewidth': 1.5})
    axes[0].set_yticks([]) 
    axes[0].set_xlabel(column_name)
    axes[0].set_title(f"Box Plot: {column_name}", fontsize=10)

    scatter_df = pd.DataFrame({
        'index': _df_with_detected_outliers.index,
        'value': _df_with_detected_outliers[column_name], 
        'is_outlier': _df_with_detected_outliers[outlier_flag_col]
    })
    normal_points = scatter_df[~scatter_df['is_outlier']]
    outlier_points = scatter_df[scatter_df['is_outlier']]

    axes[1].scatter(normal_points['index'], normal_points['value'], color='blue', label='Normal', alpha=0.6, s=15)
    axes[1].scatter(outlier_points['index'], outlier_points['value'], color='red', label='Outlier', marker='x', alpha=0.8, s=30)
    
    median_val = original_series_for_boxplot.median()
    axes[1].axhline(median_val, color='green', linestyle='--', linewidth=1, label=f'Median: {median_val:.2f}')
    axes[1].set_title(f"Scatter Plot (vs. Index): {column_name}", fontsize=10)
    axes[1].set_xlabel("Data Index", fontsize=9)
    axes[1].set_ylabel(column_name, fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis='both', which='major', labelsize=8)

    fig.suptitle(f"Outlier Visualization for {column_name}", fontsize=12, y=0.98)
    texture_tag, tex_w, tex_h = _s5_plot_to_dpg_texture(fig)

    if _active_plot_texture_id and _active_plot_texture_id != TAG_OT_DEFAULT_PLOT_TEXTURE and dpg.does_item_exist(_active_plot_texture_id):
        if dpg.does_item_exist(_active_plot_texture_id): 
            try: dpg.delete_item(_active_plot_texture_id)
            except Exception as e_del: _log_message(f"Error deleting old texture {_active_plot_texture_id}: {e_del}")
    
    if texture_tag and tex_w > 0 and tex_h > 0:
        _active_plot_texture_id = texture_tag
        if dpg.does_item_exist(TAG_OT_VISUALIZATION_PLOT_IMAGE):
            img_widget_width = dpg.get_item_width(TAG_OT_VISUALIZATION_GROUP) - 20 
            if img_widget_width is None or img_widget_width <=0 : img_widget_width = 580 
            display_w = min(tex_w, img_widget_width) 
            display_h = int(tex_h * (display_w / tex_w)) if tex_w > 0 else tex_h 
            dpg.configure_item(TAG_OT_VISUALIZATION_PLOT_IMAGE, texture_tag=_active_plot_texture_id, width=display_w, height=display_h, show=True)
        _log_message(f"Combined outlier plots generated for '{column_name}'.")
    else:
        _clear_visualization_plot()
        _log_message(f"Failed to generate plot texture for '{column_name}'.")

def _clear_visualization_plot():
    global _active_plot_texture_id
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_OT_VISUALIZATION_PLOT_IMAGE) and dpg.does_item_exist(TAG_OT_DEFAULT_PLOT_TEXTURE):
        def_tex_info = dpg.get_item_configuration(TAG_OT_DEFAULT_PLOT_TEXTURE)
        def_w, def_h = def_tex_info.get('width',100), def_tex_info.get('height',30)
        dpg.configure_item(TAG_OT_VISUALIZATION_PLOT_IMAGE, texture_tag=TAG_OT_DEFAULT_PLOT_TEXTURE, width=def_w, height=def_h, show=True)
    if _active_plot_texture_id and _active_plot_texture_id != TAG_OT_DEFAULT_PLOT_TEXTURE and dpg.does_item_exist(_active_plot_texture_id):
        try: dpg.delete_item(_active_plot_texture_id)
        except Exception as e_del_clear: _log_message(f"Error deleting texture in clear: {_active_plot_texture_id}: {e_del_clear}")
    _active_plot_texture_id = TAG_OT_DEFAULT_PLOT_TEXTURE

# --- Treatment 관련 함수들 (_on_treatment_method_change, _on_treatment_param_change, _populate_treatment_table, _set_recommended_treatments_logic, _apply_outlier_treatment_logic)은 이전과 동일하게 유지 ---
# (위에 이미 제공된 함수들이므로 생략합니다. 이전 답변의 코드를 참조해주세요.)
def _on_treatment_method_change(sender, app_data_method: str, user_data: Dict):
    col_name = user_data["col_name"]
    current_treatment = _treatment_selections.get(col_name, {}).copy()
    current_treatment["method"] = app_data_method
    _treatment_selections[col_name] = current_treatment
    _log_message(f"Treatment for '{col_name}': {app_data_method}")
    _populate_treatment_table()

def _on_treatment_param_change(sender, app_data, user_data: Dict):
    col_name = user_data["col_name"]
    param_name = user_data["param_name"]
    current_treatment = _treatment_selections.get(col_name, {}).copy()
    if param_name == "lower_percentile":
        try:
            val = int(app_data)
            if not (1 <= val <= 20):
                _log_message(f"Warn: Lower percentile for '{col_name}' must be 1-20.")
                dpg.set_value(sender, current_treatment.get(param_name, 5)); return
            current_treatment[param_name] = val
        except ValueError: _log_message(f"Warn: Invalid lower percentile '{col_name}'."); dpg.set_value(sender, current_treatment.get(param_name, 5)); return
    elif param_name == "upper_percentile":
        try:
            val = int(app_data)
            if not (80 <= val <= 99):
                _log_message(f"Warn: Upper percentile for '{col_name}' must be 80-99.")
                dpg.set_value(sender, current_treatment.get(param_name, 95)); return
            current_treatment[param_name] = val
        except ValueError: _log_message(f"Warn: Invalid upper percentile '{col_name}'."); dpg.set_value(sender, current_treatment.get(param_name, 95)); return
    elif param_name in ["abs_lower_bound", "abs_upper_bound"]:
        try: current_treatment[param_name] = float(app_data)
        except ValueError: _log_message(f"Warn: Invalid abs bound '{param_name}' for '{col_name}'."); dpg.set_value(sender, current_treatment.get(param_name, 0.0)); return
    _treatment_selections[col_name] = current_treatment
    _log_message(f"Treatment param for '{col_name}', '{param_name}' set to {app_data}")

def _populate_treatment_table():
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_TREATMENT_TABLE): return
    dpg.delete_item(TAG_OT_TREATMENT_TABLE, children_only=True)
    cols_to_treat = [item['Column'] for item in _outlier_summary_data if item.get('Detected Outliers', 0) > 0]
    if not cols_to_treat:
        dpg.add_table_column(label="Info", parent=TAG_OT_TREATMENT_TABLE, width_stretch=True)
        with dpg.table_row(parent=TAG_OT_TREATMENT_TABLE): dpg.add_text("No columns with detected outliers to treat.")
        return
    headers = ["Column Name", "Treatment Method", "Parameters (Lower %tile / Abs Lower)", "Parameters (Upper %tile / Abs Upper)"]
    col_widths_treatment = [0.3, 0.3, 0.2, 0.2] 

    for i, header in enumerate(headers): 
        dpg.add_table_column(label=header, parent=TAG_OT_TREATMENT_TABLE, 
                             init_width_or_weight=col_widths_treatment[i], width_stretch=True)

    treatment_options = ["Do Not Treat", "Treat as Missing", "Ratio-based Capping", "Absolute Value Capping"]
    for col_name in cols_to_treat:
        with dpg.table_row(parent=TAG_OT_TREATMENT_TABLE):
            dpg.add_text(col_name)
            current_col_treatment = _treatment_selections.get(col_name, {"method": "Do Not Treat"})
            current_method = current_col_treatment.get("method", "Do Not Treat")
            sanitized_col_name = "".join(filter(str.isalnum, col_name))
            combo_tag = f"s5_treat_combo_{sanitized_col_name}"
            lower_perc_tag = f"s5_lower_perc_{sanitized_col_name}"
            upper_perc_tag = f"s5_upper_perc_{sanitized_col_name}"
            abs_lower_tag = f"s5_abs_lower_{sanitized_col_name}"
            abs_upper_tag = f"s5_abs_upper_{sanitized_col_name}"
            
            dpg.add_combo(treatment_options, default_value=current_method, width=-1, tag=combo_tag, callback=_on_treatment_method_change, user_data={"col_name": col_name})
            
            with dpg.group(horizontal=True): 
                dpg.add_input_int(tag=lower_perc_tag, width=70, default_value=current_col_treatment.get("lower_percentile", 5), show=(current_method == "Ratio-based Capping"), min_value=1, max_value=20, min_clamped=True, max_clamped=True, callback=_on_treatment_param_change, user_data={"col_name": col_name, "param_name": "lower_percentile"})
                dpg.add_input_float(tag=abs_lower_tag, width=100, default_value=current_col_treatment.get("abs_lower_bound", 0.0), show=(current_method == "Absolute Value Capping"), callback=_on_treatment_param_change, user_data={"col_name": col_name, "param_name": "abs_lower_bound"})
                if current_method not in ["Ratio-based Capping", "Absolute Value Capping"]: dpg.add_text("N/A", show=True) 
            
            with dpg.group(horizontal=True): 
                dpg.add_input_int(tag=upper_perc_tag, width=70, default_value=current_col_treatment.get("upper_percentile", 95), show=(current_method == "Ratio-based Capping"), min_value=80, max_value=99, min_clamped=True, max_clamped=True, callback=_on_treatment_param_change, user_data={"col_name": col_name, "param_name": "upper_percentile"})
                dpg.add_input_float(tag=abs_upper_tag, width=100, default_value=current_col_treatment.get("abs_upper_bound", 0.0), show=(current_method == "Absolute Value Capping"), callback=_on_treatment_param_change, user_data={"col_name": col_name, "param_name": "abs_upper_bound"})
                if current_method not in ["Ratio-based Capping", "Absolute Value Capping"]: dpg.add_text("N/A", show=True)

def _set_recommended_treatments_logic():
    global _treatment_selections
    cols_to_treat = [item['Column'] for item in _outlier_summary_data if item.get('Detected Outliers', 0) > 0]
    if not cols_to_treat:
        _log_message("No columns with detected outliers for recommended treatments.")
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Info", "No detected outliers to recommend treatment for.")
        return
    for col_name in cols_to_treat: _treatment_selections[col_name] = {"method": RECOMMENDED_TREATMENT_METHOD}
    _log_message(f"Recommended treatment ('{RECOMMENDED_TREATMENT_METHOD}') set for all applicable columns.")
    _populate_treatment_table()
    if _util_funcs and '_show_simple_modal_message' in _util_funcs:
        _util_funcs['_show_simple_modal_message']("Info", f"Recommended treatment ('{RECOMMENDED_TREATMENT_METHOD}') set.")

def _apply_outlier_treatment_logic():
    global _current_df_for_this_step, _df_after_treatment, _detected_outlier_indices, _treatment_selections
    _log_message("Apply Outlier Treatments button clicked.")
    if _util_funcs and '_show_simple_modal_message' in _util_funcs:
        pass 
    try:
        if _current_df_for_this_step is None:
            _log_message("Error: No data loaded to apply treatment.");
            if _util_funcs and '_show_simple_modal_message' in _util_funcs: _util_funcs['_show_simple_modal_message']("Error", "No data to apply treatments.")
            if _main_app_callbacks: _main_app_callbacks['step5_outlier_treatment_complete'](None)
            return
        if not _detected_outlier_indices and not any(item.get('Detected Outliers',0) > 0 for item in _outlier_summary_data):
            _log_message("Info: No outliers were previously detected. Run detection first or no outliers to treat.")
            if _util_funcs and '_show_simple_modal_message' in _util_funcs: _util_funcs['_show_simple_modal_message']("Info", "No outlier detection results found or no outliers to treat.")
            if _main_app_callbacks: _main_app_callbacks['step5_outlier_treatment_complete'](_current_df_for_this_step.copy()) 
            return

        _df_after_treatment = _current_df_for_this_step.copy()
        _log_message("--- Starting Outlier Treatment Application ---")
        treatment_applied_any = False
        for col_name, treatment_params in _treatment_selections.items():
            if col_name not in _df_after_treatment.columns or col_name not in _detected_outlier_indices:
                _log_message(f"Skipping treatment for '{col_name}': Not in data or no outliers detected for it."); continue
            outlier_idx_for_col = _detected_outlier_indices[col_name] 
            if len(outlier_idx_for_col) == 0: _log_message(f"No detected outliers in '{col_name}' to treat."); continue
            
            method = treatment_params.get("method", "Do Not Treat")
            _log_message(f"Applying to '{col_name}' (Method: {method}, {len(outlier_idx_for_col)} outliers):")
            try:
                current_col_series_before_treatment = _df_after_treatment[col_name].copy() 

                if method == "Treat as Missing":
                    _df_after_treatment.loc[outlier_idx_for_col, col_name] = np.nan
                    _log_message(f"  Treated {len(outlier_idx_for_col)} outliers in '{col_name}' as NaN."); treatment_applied_any = True
                elif method == "Ratio-based Capping":
                    lower_p = treatment_params.get("lower_percentile"); upper_p = treatment_params.get("upper_percentile")
                    if lower_p is None or upper_p is None: _log_message(f"  Skip Ratio Cap for '{col_name}': Percentiles not set."); continue
                    if not (1 <= lower_p <= 20 and 80 <= upper_p <= 99 and lower_p < upper_p) : _log_message(f"  Skip Ratio Cap '{col_name}': Invalid percentiles."); continue
                    reference_series_for_quantile = _current_df_for_this_step[col_name].dropna() 
                    if reference_series_for_quantile.empty : _log_message(f"  Skip Ratio Cap '{col_name}': Reference series for quantile is empty."); continue
                    lower_val = reference_series_for_quantile.quantile(lower_p / 100.0)
                    upper_val = reference_series_for_quantile.quantile(upper_p / 100.0)
                    if pd.isna(lower_val) or pd.isna(upper_val) : _log_message(f"  Skip Ratio Cap '{col_name}': Quantile calculation resulted in NaN."); continue
                    _df_after_treatment.loc[outlier_idx_for_col, col_name] = current_col_series_before_treatment.loc[outlier_idx_for_col].clip(lower=lower_val, upper=upper_val)
                    _log_message(f"  Applied Ratio Capping to outliers in '{col_name}' (Range: {lower_val:.2f}-{upper_val:.2f})."); treatment_applied_any = True
                elif method == "Absolute Value Capping":
                    abs_lower = treatment_params.get("abs_lower_bound"); abs_upper = treatment_params.get("abs_upper_bound")
                    if abs_lower is None or abs_upper is None: _log_message(f"  Skip Abs Cap '{col_name}': Bounds not set."); continue
                    if abs_lower >= abs_upper: _log_message(f"  Skip Abs Cap '{col_name}': Lower bound must be >= upper."); continue
                    _df_after_treatment.loc[outlier_idx_for_col, col_name] = current_col_series_before_treatment.loc[outlier_idx_for_col].clip(lower=abs_lower, upper=abs_upper)
                    _log_message(f"  Applied Absolute Capping to outliers in '{col_name}' (Range: {abs_lower:.2f}-{abs_upper:.2f})."); treatment_applied_any = True
                elif method == "Do Not Treat": _log_message(f"  '{col_name}' set to 'Do Not Treat'.")
            except Exception as e_treat_col: _log_message(f"  Error treating '{col_name}': {e_treat_col}")
        
        if treatment_applied_any:
            _log_message("--- Outlier Treatment Application Finished ---")
            if _util_funcs and '_show_simple_modal_message' in _util_funcs: _util_funcs['_show_simple_modal_message']("Treatment Complete", "Selected outlier treatments applied.")
        else:
            _log_message("--- No outlier treatments were applied. ---")
            if _util_funcs and '_show_simple_modal_message' in _util_funcs: _util_funcs['_show_simple_modal_message']("Treatment Info", "No treatments selected or applied.")
        
        if _main_app_callbacks: _main_app_callbacks['step5_outlier_treatment_complete'](_df_after_treatment)
    
    except Exception as e:
        _log_message(f"CRITICAL ERROR in _apply_outlier_treatment_logic: {e}")
        traceback.print_exc()
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Critical Error", f"An unexpected error occurred during outlier treatment: {e}")
        if _main_app_callbacks: _main_app_callbacks['step5_outlier_treatment_complete'](None) 


def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _main_app_callbacks, _util_funcs, _active_plot_texture_id
    _main_app_callbacks = main_callbacks
    if 'get_util_funcs' in main_callbacks: _util_funcs = main_callbacks['get_util_funcs']()

    if not dpg.does_item_exist("texture_registry"):
        dpg.add_texture_registry(tag="texture_registry", show=False)
    
    if not dpg.does_item_exist(TAG_OT_DEFAULT_PLOT_TEXTURE):
        default_texture_data = [0.0] * (100 * 30 * 4) 
        try:
            dpg.add_static_texture(width=100, height=30, default_value=default_texture_data, tag=TAG_OT_DEFAULT_PLOT_TEXTURE, parent="texture_registry")
        except Exception as e: print(f"Error creating default texture: {e}")
    _active_plot_texture_id = TAG_OT_DEFAULT_PLOT_TEXTURE


    main_callbacks['register_step_group_tag'](step_name, TAG_OT_STEP_GROUP)
    with dpg.group(tag=TAG_OT_STEP_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} (Univariate) ---"); dpg.add_separator() # 탭 이름 명시 (임시)
        dpg.add_text("1. Configure & Run Univariate Outlier Detection", color=[255, 255, 0])
        with dpg.group(horizontal=True):
            dpg.add_text("Detection Method:")
            # "Both" 옵션 제거, PyOD 모델 추가
            dpg.add_radio_button(["IQR", "HBOS", "ECOD"], tag=TAG_OT_DETECT_METHOD_RADIO, 
                                 default_value=_selected_detection_method, horizontal=True, 
                                 callback=_on_detection_method_change)
        dpg.add_text("Detection Parameters (applied to eligible columns):")
        
        # IQR Parameters
        with dpg.group(horizontal=True, tag=TAG_OT_IQR_MULTIPLIER_INPUT + "_group"): # 그룹에 태그 부여 가능성 (그러나 show/hide는 개별 위젯에)
            dpg.add_text("IQR Multiplier:", tag=TAG_OT_IQR_MULTIPLIER_INPUT + "_label")
            dpg.add_input_float(tag=TAG_OT_IQR_MULTIPLIER_INPUT, width=120, default_value=_iqr_multiplier, 
                                step=0.1, callback=_on_iqr_multiplier_change)
        
        # HBOS Parameters
        with dpg.group(horizontal=True, tag=TAG_OT_HBOS_N_BINS_INPUT + "_group"):
            dpg.add_text("HBOS n_bins:", tag=TAG_OT_HBOS_N_BINS_INPUT + "_label")
            dpg.add_input_int(tag=TAG_OT_HBOS_N_BINS_INPUT, width=120, default_value=_hbos_n_bins, 
                              step=1, min_value=2, min_clamped=True, # HBOS n_bins는 최소 2 이상
                              callback=_on_hbos_n_bins_change)
        
        # ECOD Parameters
        with dpg.group(horizontal=True, tag=TAG_OT_ECOD_CONTAM_INPUT + "_group"):
            dpg.add_text("ECOD Contamination (0.0-0.5):", tag=TAG_OT_ECOD_CONTAM_INPUT + "_label")
            dpg.add_input_float(tag=TAG_OT_ECOD_CONTAM_INPUT, width=120, default_value=_ecod_contamination, 
                               min_value=0.0001, max_value=0.5, min_clamped=True, max_clamped=True, # 범위 제한
                               step=0.01, format="%.4f", callback=_on_ecod_contamination_change)

        _update_parameter_fields_visibility() # 초기 UI 생성 시 호출
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Run Univariate Outlier Detection", tag=TAG_OT_DETECT_BUTTON, width=-1, height=30, callback=_run_outlier_detection_logic)
            dpg.add_button(label="Set Recommended Detection Params", tag=TAG_OT_RECOMMEND_PARAMS_BUTTON, width=-1, height=30, callback=_set_recommended_detection_parameters)
        dpg.add_spacer(height=10)

        dpg.add_text("2. Detection Summary & Visualization (Click row in table to visualize)", color=[255, 255, 0])
        with dpg.table(tag=TAG_OT_DETECTION_RESULTS_TABLE, header_row=True, resizable=True, 
                       policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=150, 
                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            dpg.add_table_column(label="Info", parent=TAG_OT_DETECTION_RESULTS_TABLE, width_stretch=True)
            with dpg.table_row(parent=TAG_OT_DETECTION_RESULTS_TABLE):
                dpg.add_text("Run detection to see summary.")
        
        with dpg.group(tag=TAG_OT_VISUALIZATION_GROUP, horizontal=False):
            def_tex_cfg = dpg.get_item_configuration(TAG_OT_DEFAULT_PLOT_TEXTURE) if dpg.does_item_exist(TAG_OT_DEFAULT_PLOT_TEXTURE) else {'width':100, 'height':30}
            img_w, img_h = 600, 350 
            dpg.add_image(texture_tag=_active_plot_texture_id or TAG_OT_DEFAULT_PLOT_TEXTURE, 
                          tag=TAG_OT_VISUALIZATION_PLOT_IMAGE, show=True, 
                          width=img_w, height=img_h) 
        dpg.add_spacer(height=10)

        dpg.add_text("3. Configure Outlier Treatment", color=[255, 255, 0])
        with dpg.table(tag=TAG_OT_TREATMENT_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=200, scrollX=True, borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            dpg.add_table_column(label="Info", parent=TAG_OT_TREATMENT_TABLE, width_stretch=True)
            with dpg.table_row(parent=TAG_OT_TREATMENT_TABLE):
                dpg.add_text("Run detection to configure treatments.")
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Set Recommended Treatments for All", tag=TAG_OT_RECOMMEND_TREATMENTS_BUTTON, 
                           width=-1, height=30, callback=_set_recommended_treatments_logic) # 원래 width=-1, 테스트용 고정값 제거
            dpg.add_button(label="Reset Treatment Selections", tag=TAG_OT_RESET_TREATMENTS_BUTTON,
                           width=-1, height=30, callback=_reset_treatment_selections_to_default)
            dpg.add_button(label="Apply Selected Outlier Treatments", tag=TAG_OT_APPLY_TREATMENT_BUTTON, 
                           width=-1, height=30, callback=_apply_outlier_treatment_logic)
        dpg.add_spacer(height=15)

        dpg.add_text("4. Processing Log", color=[255, 255, 0])
        with dpg.child_window(tag=TAG_OT_LOG_TEXT_AREA, height=120, border=True):
            dpg.add_text("Logs will appear here...", tag=TAG_OT_LOG_TEXT, wrap=-1)
            
    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui(df_input_for_step: Optional[pd.DataFrame], main_callbacks: dict):
    global _main_app_callbacks, _util_funcs, _current_df_for_this_step, _currently_visualized_column
    if not dpg.is_dearpygui_running(): return

    if not _main_app_callbacks: _main_app_callbacks = main_callbacks
    if not _util_funcs and 'get_util_funcs' in _main_app_callbacks: _util_funcs = _main_app_callbacks['get_util_funcs']()
    
    is_new_data = _current_df_for_this_step is not df_input_for_step
    _current_df_for_this_step = df_input_for_step

    if not dpg.does_item_exist(TAG_OT_STEP_GROUP): return

    if _current_df_for_this_step is None or is_new_data:
        _columns_eligible_for_detection.clear()
        _outlier_summary_data.clear()
        _detected_outlier_indices.clear()
        _df_with_detected_outliers = None
        _populate_detection_results_table()
        _populate_treatment_table()
        _currently_visualized_column = None
        _clear_visualization_plot()
        if dpg.does_item_exist(TAG_OT_LOG_TEXT):
            msg = "Data loaded. Configure and run outlier detection." if _current_df_for_this_step is not None else "Load data for Step 5."
            dpg.set_value(TAG_OT_LOG_TEXT, msg)
    
    _update_parameter_fields_visibility()

def reset_outlier_treatment_state():
    global _current_df_for_this_step, _df_with_detected_outliers, _df_after_treatment, _active_plot_texture_id
    global _detected_outlier_indices, _outlier_summary_data, _treatment_selections, _columns_eligible_for_detection
    global _selected_detection_method, _iqr_multiplier, _hbos_n_bins, _ecod_contamination, _currently_visualized_column, _all_selectable_tags_in_table
    
    _current_df_for_this_step = None; _df_with_detected_outliers = None; _df_after_treatment = None
    _detected_outlier_indices.clear(); _outlier_summary_data.clear(); _treatment_selections.clear(); 
    _columns_eligible_for_detection.clear(); _all_selectable_tags_in_table.clear()
    # 기본값 재설정
    _selected_detection_method = "IQR"
    _iqr_multiplier = DEFAULT_IQR_MULTIPLIER
    _hbos_n_bins = DEFAULT_HBOS_N_BINS
    _ecod_contamination = DEFAULT_ECOD_CONTAMINATION
    _currently_visualized_column = None

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_DETECT_METHOD_RADIO): dpg.set_value(TAG_OT_DETECT_METHOD_RADIO, _selected_detection_method)
        if dpg.does_item_exist(TAG_OT_IQR_MULTIPLIER_INPUT): dpg.set_value(TAG_OT_IQR_MULTIPLIER_INPUT, _iqr_multiplier)
        if dpg.does_item_exist(TAG_OT_HBOS_N_BINS_INPUT): dpg.set_value(TAG_OT_HBOS_N_BINS_INPUT, _hbos_n_bins)
        if dpg.does_item_exist(TAG_OT_ECOD_CONTAM_INPUT): dpg.set_value(TAG_OT_ECOD_CONTAM_INPUT, _ecod_contamination)
        
        _populate_detection_results_table(); _populate_treatment_table()
        _clear_visualization_plot()

        if dpg.does_item_exist(TAG_OT_LOG_TEXT): dpg.set_value(TAG_OT_LOG_TEXT, "State reset. Logs cleared.")
    
    _update_parameter_fields_visibility()
    print("Step 5: Outlier Treatment state has been reset.")

def get_outlier_treatment_settings_for_saving() -> dict:
    return {
        "selected_detection_method": _selected_detection_method,
        "iqr_multiplier": _iqr_multiplier,
        "hbos_n_bins": _hbos_n_bins, # HBOS 파라미터 저장
        "ecod_contamination": _ecod_contamination, # ECOD 파라미터 저장
        "treatment_selections": _treatment_selections.copy()
    }

def apply_outlier_treatment_settings_and_process(df_input: pd.DataFrame, settings: dict, main_callbacks: dict):
    global _selected_detection_method, _iqr_multiplier, _hbos_n_bins, _ecod_contamination, _treatment_selections
    global _main_app_callbacks, _current_df_for_this_step, _df_after_treatment, _df_with_detected_outliers, _outlier_summary_data, _detected_outlier_indices, _currently_visualized_column
    
    if not _main_app_callbacks: _main_app_callbacks = main_callbacks
    
    _current_df_for_this_step = df_input; _df_after_treatment = None; _df_with_detected_outliers = None
    _detected_outlier_indices.clear(); _outlier_summary_data.clear(); _currently_visualized_column = None

    _selected_detection_method = settings.get("selected_detection_method", "IQR")
    _iqr_multiplier = settings.get("iqr_multiplier", DEFAULT_IQR_MULTIPLIER)
    _hbos_n_bins = settings.get("hbos_n_bins", DEFAULT_HBOS_N_BINS) # HBOS 파라미터 로드
    _ecod_contamination = settings.get("ecod_contamination", DEFAULT_ECOD_CONTAMINATION) # ECOD 파라미터 로드
    _treatment_selections = settings.get("treatment_selections", {}).copy()
    
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_DETECT_METHOD_RADIO): dpg.set_value(TAG_OT_DETECT_METHOD_RADIO, _selected_detection_method)
        if dpg.does_item_exist(TAG_OT_IQR_MULTIPLIER_INPUT): dpg.set_value(TAG_OT_IQR_MULTIPLIER_INPUT, _iqr_multiplier)
        if dpg.does_item_exist(TAG_OT_HBOS_N_BINS_INPUT): dpg.set_value(TAG_OT_HBOS_N_BINS_INPUT, _hbos_n_bins)
        if dpg.does_item_exist(TAG_OT_ECOD_CONTAM_INPUT): dpg.set_value(TAG_OT_ECOD_CONTAM_INPUT, _ecod_contamination)
            
    update_ui(df_input, main_callbacks)
    _log_message("Step 5 Outlier Treatment settings applied. UI reflects saved parameters. Please run detection and treatment manually if needed.")