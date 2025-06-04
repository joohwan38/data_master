# step_05a_univariate_outliers.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

try:
    from pyod.models.hbos import HBOS as PyOD_HBOS
    from pyod.models.ecod import ECOD as PyOD_ECOD
except ImportError:
    # _log_message_func를 여기서 직접 호출할 수 없으므로, 초기화 시점에 전달받아야 함
    # print("Warning: PyOD library not found. HBOS and ECOD outlier detection will not be available.")
    PyOD_HBOS = None
    PyOD_ECOD = None

import matplotlib.pyplot as plt
import seaborn as sns
# traceback, io, PIL 등은 _s5_plot_to_dpg_texture_parent 함수가 처리하므로 직접 임포트 불필요할 수 있음
# 단, matplotlib.pyplot, seaborn 등은 여기서 직접 사용하므로 임포트

# --- DPG Tags for Univariate Tab ---
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
# TAG_OT_DEFAULT_PLOT_TEXTURE_UNI는 부모로부터 태그 문자열을 받아서 사용

TAG_OT_RECOMMEND_TREATMENTS_BUTTON_UNI = "step5_ot_recommend_treatments_button_uni"
TAG_OT_RESET_TREATMENTS_BUTTON_UNI = "step5_ot_reset_treatments_button_uni"
TAG_OT_APPLY_TREATMENT_BUTTON_UNI = "step5_ot_apply_treatment_button_uni"
TAG_OT_TREATMENT_TABLE_UNI = "step5_ot_treatment_table_uni"

# --- Constants for Filtering (Univariate) ---
MIN_UNIQUE_VALUES_FOR_OUTLIER_DETECTION_UNI = 10
MAX_MISSING_RATIO_FOR_OUTLIER_DETECTION_UNI = 0.5
MIN_VARIANCE_FOR_OUTLIER_DETECTION_UNI = 1e-5

# --- Default Parameter Constants ---
DEFAULT_UNI_IQR_MULTIPLIER = 1.5
DEFAULT_UNI_HBOS_N_BINS = 20
DEFAULT_UNI_ECOD_CONTAMINATION = 0.1
RECOMMENDED_UNI_TREATMENT_METHOD = "Treat as Missing"


# --- Module State Variables (Univariate) ---
# 공유 유틸리티 및 콜백 (부모 모듈에서 설정)
_shared_utils_uni: Optional[Dict[str, Any]] = None

# 단변량 내부 상태
_df_with_uni_detected_outliers: Optional[pd.DataFrame] = None
_df_after_uni_treatment: Optional[pd.DataFrame] = None
_uni_detected_outlier_indices: Dict[str, np.ndarray] = {}
_uni_outlier_summary_data: List[Dict[str, Any]] = []
_uni_columns_eligible_for_detection: List[str] = []
_uni_selected_detection_method: str = "IQR"
_uni_iqr_multiplier: float = DEFAULT_UNI_IQR_MULTIPLIER
_uni_hbos_n_bins: int = DEFAULT_UNI_HBOS_N_BINS
_uni_ecod_contamination: float = DEFAULT_UNI_ECOD_CONTAMINATION
_uni_treatment_selections: Dict[str, Dict[str, Any]] = {}
_uni_active_plot_texture_id: Optional[str] = None # 부모의 TAG_OT_DEFAULT_PLOT_TEXTURE_UNI로 초기화될 것
_uni_currently_visualized_column: Optional[str] = None
_uni_all_selectable_tags_in_table: List[str] = []


# --- Helper Functions ---
# _log_message, _s5_plot_to_dpg_texture 등은 _shared_utils_uni를 통해 호출

def _log_uni(message: str):
    if _shared_utils_uni and 'log_message_func' in _shared_utils_uni:
        _shared_utils_uni['log_message_func'](f"[UniOutlier] {message}")

def _show_simple_modal_uni(title: str, message: str, width: int = 450, height: int = 200):
    if _shared_utils_uni and 'util_funcs_common' in _shared_utils_uni and \
       '_show_simple_modal_message' in _shared_utils_uni['util_funcs_common']:
        _shared_utils_uni['util_funcs_common']['_show_simple_modal_message'](title, message, width, height)
    else:
        _log_uni(f"Modal display function not available. Title: {title}, Msg: {message}")


# --- Univariate Callbacks and Logic ---
def _reset_uni_treatment_selections_to_default():
    global _uni_treatment_selections, _uni_outlier_summary_data
    _log_uni("Resetting all univariate treatment selections to 'Do Not Treat'.")
    # ... (기존 로직과 동일, _log_message -> _log_uni, _show_simple_modal_message -> _show_simple_modal_uni)
    cols_with_detected_outliers = [item['Column'] for item in _uni_outlier_summary_data if item.get('Detected Outliers', 0) > 0]
    if not cols_with_detected_outliers:
        _log_uni("No univariate columns with detected outliers to reset treatment for.")
        _show_simple_modal_uni("Info", "No univariate outlier treatments to reset. Run detection first.")
        return
    for col_name in cols_with_detected_outliers:
        _uni_treatment_selections[col_name] = {"method": "Do Not Treat"}
    _populate_uni_treatment_table()
    _log_uni("All univariate treatment selections have been reset to 'Do Not Treat'.")
    _show_simple_modal_uni("Info", "All univariate outlier treatment selections reset.")


def _update_uni_parameter_fields_visibility():
    if not dpg.is_dearpygui_running(): return
    show_iqr = _uni_selected_detection_method == "IQR"
    show_hbos = _uni_selected_detection_method == "HBOS" if PyOD_HBOS is not None else False
    show_ecod = _uni_selected_detection_method == "ECOD" if PyOD_ECOD is not None else False
    
    # ... (기존 로직과 동일)
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
    _log_uni(f"Univariate outlier detection method changed to: {_uni_selected_detection_method}")
    _update_uni_parameter_fields_visibility()

def _on_uni_iqr_multiplier_change(sender, app_data: float, user_data):
    global _uni_iqr_multiplier
    _uni_iqr_multiplier = app_data
    _log_uni(f"Univariate IQR multiplier set to: {_uni_iqr_multiplier}")

def _on_uni_hbos_n_bins_change(sender, app_data: int, user_data):
    global _uni_hbos_n_bins
    if app_data >= 2 :
        _uni_hbos_n_bins = app_data
        _log_uni(f"Univariate HBOS n_bins set to: {_uni_hbos_n_bins}")
    else:
        dpg.set_value(sender, _uni_hbos_n_bins)
        _log_uni(f"Univariate HBOS n_bins must be >= 2. Reverted to {_uni_hbos_n_bins}.")
        _show_simple_modal_uni("Input Error", "HBOS n_bins must be greater than or equal to 2.")


def _on_uni_ecod_contamination_change(sender, app_data: float, user_data):
    global _uni_ecod_contamination
    if 0.0 < app_data <= 0.5:
        _uni_ecod_contamination = app_data
        _log_uni(f"Univariate ECOD contamination set to: {_uni_ecod_contamination:.4f}")
    else:
        dpg.set_value(sender, _uni_ecod_contamination)
        _log_uni(f"Univariate ECOD contamination must be (0.0, 0.5]. Reverted to {_uni_ecod_contamination:.4f}.")
        _show_simple_modal_uni("Input Error", "ECOD contamination must be between 0 (exclusive) and 0.5 (inclusive).")

def _set_uni_recommended_detection_parameters(sender, app_data, user_data):
    global _uni_iqr_multiplier, _uni_hbos_n_bins, _uni_ecod_contamination
    _uni_iqr_multiplier = DEFAULT_UNI_IQR_MULTIPLIER
    _uni_hbos_n_bins = DEFAULT_UNI_HBOS_N_BINS
    _uni_ecod_contamination = DEFAULT_UNI_ECOD_CONTAMINATION
    
    if dpg.does_item_exist(TAG_OT_IQR_MULTIPLIER_INPUT):
        dpg.set_value(TAG_OT_IQR_MULTIPLIER_INPUT, _uni_iqr_multiplier)
    if PyOD_HBOS and dpg.does_item_exist(TAG_OT_HBOS_N_BINS_INPUT):
        dpg.set_value(TAG_OT_HBOS_N_BINS_INPUT, _uni_hbos_n_bins)
    if PyOD_ECOD and dpg.does_item_exist(TAG_OT_ECOD_CONTAM_INPUT_UNI):
        dpg.set_value(TAG_OT_ECOD_CONTAM_INPUT_UNI, _uni_ecod_contamination)
        
    _log_uni(f"Recommended univariate detection parameters set: IQR Mult={_uni_iqr_multiplier}, HBOS n_bins={_uni_hbos_n_bins}, ECOD Contam={_uni_ecod_contamination:.2f}")
    _update_uni_parameter_fields_visibility()
    _show_simple_modal_uni("Info", "Recommended univariate detection parameters have been applied.")


def _detect_outliers_uni_iqr(series: pd.Series, multiplier: float) -> np.ndarray:
    # ... (기존 로직과 동일)
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
    if PyOD_HBOS is None: _log_uni("Error: PyOD_HBOS model not loaded. Is PyOD installed?"); return np.array([])
    # ... (기존 로직과 동일, _log_message -> _log_uni)
    if not pd.api.types.is_numeric_dtype(series.dtype): return np.array([])
    series_cleaned = series.dropna()
    if len(series_cleaned) < 2 or len(series_cleaned.unique()) < 2 : return np.array([])
    actual_n_bins = min(n_bins_param, len(series_cleaned.unique()) -1) if len(series_cleaned.unique()) > 1 else 1
    if actual_n_bins < 2 : actual_n_bins = 2
    try:
        model = PyOD_HBOS(n_bins=actual_n_bins, contamination=contamination_param)
        model.fit(series_cleaned.values.reshape(-1, 1))
        return series_cleaned.index[model.labels_ == 1].to_numpy()
    except Exception as e: _log_uni(f"  Error during HBOS for {series.name}: {e}"); return np.array([])


def _detect_outliers_uni_ecod(series: pd.Series, contamination_param: float) -> np.ndarray:
    if PyOD_ECOD is None: _log_uni("Error: PyOD_ECOD model not loaded. Is PyOD installed?"); return np.array([])
    # ... (기존 로직과 동일, _log_message -> _log_uni)
    if not pd.api.types.is_numeric_dtype(series.dtype): return np.array([])
    series_cleaned = series.dropna()
    if len(series_cleaned) < 2 : return np.array([])
    try:
        model = PyOD_ECOD(contamination=contamination_param)
        model.fit(series_cleaned.values.reshape(-1, 1))
        return series_cleaned.index[model.labels_ == 1].to_numpy()
    except Exception as e: _log_uni(f"  Error during ECOD for {series.name}: {e}"); return np.array([])


def _filter_columns_for_uni_detection(df: pd.DataFrame) -> List[str]:
    eligible_cols = []
    if df is None: return eligible_cols
    
    s1_col_types = {}
    if _shared_utils_uni and 'main_app_callbacks' in _shared_utils_uni and \
       'get_column_analysis_types' in _shared_utils_uni['main_app_callbacks']:
        s1_col_types = _shared_utils_uni['main_app_callbacks']['get_column_analysis_types']()
        
    _log_uni("Column filtering for univariate outlier detection:")
    # ... (기존 로직과 동일, _log_message -> _log_uni)
    for col_name in df.columns:
        series = df[col_name]
        is_numeric_s1 = "Numeric" in s1_col_types.get(col_name, "") and "Binary" not in s1_col_types.get(col_name, "")
        is_numeric_pandas = pd.api.types.is_numeric_dtype(series.dtype)
        if not (is_numeric_s1 or is_numeric_pandas): continue
        if series.nunique() < MIN_UNIQUE_VALUES_FOR_OUTLIER_DETECTION_UNI: continue
        s1_type = s1_col_types.get(col_name, "")
        if any(k in s1_type for k in ["Categorical", "Binary", "Text"]): continue # Text도 제외 고려
        if (series.isnull().sum() / len(series) if len(series) > 0 else 1.0) > MAX_MISSING_RATIO_FOR_OUTLIER_DETECTION_UNI: continue
        series_dropped_na = series.dropna()
        if series_dropped_na.empty: continue
        variance = series_dropped_na.var()
        if pd.isna(variance) or variance < MIN_VARIANCE_FOR_OUTLIER_DETECTION_UNI: continue
        eligible_cols.append(col_name)
        _log_uni(f"  Eligible for univariate detection: '{col_name}'")
    if not eligible_cols: _log_uni("  No columns found eligible for univariate detection.")
    return eligible_cols


def _run_uni_outlier_detection_logic(sender, app_data, user_data):
    global _df_with_uni_detected_outliers, _uni_detected_outlier_indices, _uni_outlier_summary_data
    global _uni_columns_eligible_for_detection, _uni_currently_visualized_column
    
    _log_uni("Run Univariate Outlier Detection button clicked.")
    # ... (기존 로직과 동일, _log_message -> _log_uni, _show_simple_modal_message -> _show_simple_modal_uni)
    # _current_df_for_this_step 대신 _shared_utils_uni['get_current_df_func']() 사용
    current_df = _shared_utils_uni['get_current_df_func']()
    if current_df is None:
        _log_uni("Error: No data for univariate outlier detection.")
        _show_simple_modal_uni("Error", "No data for univariate outlier detection.")
        return

    _log_uni(f"--- Starting Univariate Outlier Detection (Method: {_uni_selected_detection_method}) ---")
    if _uni_selected_detection_method == "IQR":
        _log_uni(f"  Parameters: IQR Multiplier={_uni_iqr_multiplier}")
    elif _uni_selected_detection_method == "HBOS":
        _log_uni(f"  Parameters: HBOS n_bins={_uni_hbos_n_bins}, Contamination={_uni_ecod_contamination:.4f}") # ECOD Contam 사용 주의
    elif _uni_selected_detection_method == "ECOD":
        _log_uni(f"  Parameters: ECOD Contamination={_uni_ecod_contamination:.4f}")


    _uni_columns_eligible_for_detection = _filter_columns_for_uni_detection(current_df)
    if not _uni_columns_eligible_for_detection:
        _log_uni("Warning: No columns suitable for univariate outlier detection.")
        _show_simple_modal_uni("Warning", "No columns suitable for univariate outlier detection.")
        _uni_outlier_summary_data.clear(); _uni_detected_outlier_indices.clear()
        _populate_uni_detection_results_table(); _populate_uni_treatment_table()
        _uni_currently_visualized_column = None; _clear_uni_visualization_plot()
        return

    _uni_detected_outlier_indices.clear(); _uni_outlier_summary_data.clear()
    _df_with_uni_detected_outliers = current_df.copy()
    # ... (이하 로직 동일, 내부에서 _current_df_for_this_step 대신 current_df 사용)
    first_col_with_outliers_for_auto_vis = None
    for col_name in _uni_columns_eligible_for_detection:
        series = current_df[col_name]
        col_outlier_indices = np.array([])
        if _uni_selected_detection_method == "IQR":
            col_outlier_indices = _detect_outliers_uni_iqr(series, _uni_iqr_multiplier)
        elif _uni_selected_detection_method == "HBOS" and PyOD_HBOS:
            # HBOS는 ECOD의 contamination을 쓰지 않도록 주의. HBOS 자체의 contamination 파라미터가 있지만,
            # 현재 UI에서는 ECOD 것을 공유하고 있음. 이 부분은 명확한 분리 또는 UI 수정 필요.
            # 여기서는 ECOD contam을 임시로 사용. (원래 코드 참조)
            col_outlier_indices = _detect_outliers_uni_hbos(series, _uni_hbos_n_bins, _uni_ecod_contamination)
        elif _uni_selected_detection_method == "ECOD" and PyOD_ECOD:
            col_outlier_indices = _detect_outliers_uni_ecod(series, _uni_ecod_contamination)
        
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
    elif _uni_columns_eligible_for_detection: # 이상치가 없더라도 첫번째 eligible 컬럼 시각화 시도
        _uni_currently_visualized_column = _uni_columns_eligible_for_detection[0]
        _generate_uni_combined_plot_texture(_uni_columns_eligible_for_detection[0])
    else: 
        _uni_currently_visualized_column = None; _clear_uni_visualization_plot()
    _log_uni("--- Univariate Outlier Detection Finished ---")
    _show_simple_modal_uni("Detection Complete", "Univariate outlier detection finished.")


def _populate_uni_detection_results_table():
    global _uni_all_selectable_tags_in_table
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_DETECTION_RESULTS_TABLE_UNI): return
    dpg.delete_item(TAG_OT_DETECTION_RESULTS_TABLE_UNI, children_only=True)
    _uni_all_selectable_tags_in_table.clear() # 선택 가능한 태그 목록 초기화
    # ... (기존 로직과 동일)
    if not _uni_outlier_summary_data:
        dpg.add_table_column(label="Info", parent=TAG_OT_DETECTION_RESULTS_TABLE_UNI, width_stretch=True)
        with dpg.table_row(parent=TAG_OT_DETECTION_RESULTS_TABLE_UNI): dpg.add_text("No univariate detection results. Run detection.")
        return
    headers = ["Column (Click to Visualize)", "Detected Outliers", "Percentage (%)"]
    # 컬럼 너비 조정 가능
    col_widths_summary = [0.5, 0.25, 0.25] 
    for i, header in enumerate(headers): 
        dpg.add_table_column(label=header, parent=TAG_OT_DETECTION_RESULTS_TABLE_UNI, 
                             init_width_or_weight=col_widths_summary[i], 
                             width_stretch=(i==0)) # 첫 번째 컬럼만 stretch

    for i, row_data in enumerate(_uni_outlier_summary_data):
        with dpg.table_row(parent=TAG_OT_DETECTION_RESULTS_TABLE_UNI):
            col_name_str = str(row_data.get("Column", ""))
            # 태그 고유성 강화
            tag = f"uni_selectable_s5_{i}_{''.join(filter(str.isalnum, col_name_str))}_{dpg.generate_uuid()}"
            _uni_all_selectable_tags_in_table.append(tag)
            # selectable은 전체 row에 걸쳐서 동작하도록 span_columns=True (기본값 False)
            dpg.add_selectable(label=col_name_str, tag=tag, user_data=col_name_str, 
                               callback=_on_uni_row_selectable_clicked, span_columns=False) # 원래 코드 False 유지
            dpg.add_text(str(row_data.get("Detected Outliers", "")))
            dpg.add_text(str(row_data.get("Percentage (%)", "")))

def _on_uni_row_selectable_clicked(sender, app_data_is_selected: bool, user_data_col_name: str):
    global _uni_currently_visualized_column
    if app_data_is_selected: # selectable이 True일 때만 (선택될 때)
        # 다른 selectable들의 선택 상태 해제
        for tag_iter in _uni_all_selectable_tags_in_table:
            if tag_iter != sender and dpg.does_item_exist(tag_iter) and dpg.get_value(tag_iter):
                dpg.set_value(tag_iter, False)
        
        _uni_currently_visualized_column = user_data_col_name
        _generate_uni_combined_plot_texture(user_data_col_name)
        _log_uni(f"Visualizing univariate outlier plots for: {user_data_col_name}")


def _generate_uni_combined_plot_texture(column_name: str):
    global _uni_active_plot_texture_id
    
    # --- 1. 필요한 콜백 및 유틸리티 함수 가져오기 ---
    if not _shared_utils_uni or 'main_app_callbacks' not in _shared_utils_uni or \
       'util_funcs_common' not in _shared_utils_uni:
        _log_uni(f"Plot error for '{column_name}': Shared utilities or main_app_callbacks not fully initialized.")
        _clear_uni_visualization_plot()
        return

    main_callbacks = _shared_utils_uni['main_app_callbacks']
    util_funcs = _shared_utils_uni['util_funcs_common']
    
    create_widget_func = util_funcs.get('create_analyzable_image_widget')
    if not callable(create_widget_func):
        _log_uni(f"Plot error for '{column_name}': create_analyzable_image_widget function not found in util_funcs.")
        _clear_uni_visualization_plot()
        return

    # --- 2. 데이터 준비 (기존 로직과 거의 동일) ---
    current_df = _shared_utils_uni['get_current_df_func']()
    default_texture_tag_from_shared = _shared_utils_uni.get('default_uni_plot_texture_tag') # 기본 텍스처 태그

    if _df_with_uni_detected_outliers is None or column_name not in _df_with_uni_detected_outliers.columns or \
       current_df is None or column_name not in current_df.columns:
        _clear_uni_visualization_plot()
        _log_uni(f"Plot error: Data for '{column_name}' not ready for univariate plot.")
        return

    original_series = current_df[column_name].dropna()
    outlier_flag_col = f"{column_name}_is_outlier"
    if original_series.empty or not pd.api.types.is_numeric_dtype(original_series.dtype) or \
       outlier_flag_col not in _df_with_uni_detected_outliers.columns:
        _clear_uni_visualization_plot()
        _log_uni(f"Plot error: Invalid data or flags for univariate plot of '{column_name}'.")
        return
    
    fig_width, fig_height = 15.6, 5.85 
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    plt.subplots_adjust(wspace=0.3) # 서브플롯 간 간격

    # Box Plot
    axes[0].boxplot(original_series, vert=True, patch_artist=True, 
                    medianprops={'color':'#FF0000', 'linewidth': 1.5},
                    flierprops={'marker':'o', 'markersize':4, 'markerfacecolor':'#FF7F50', 'alpha':0.6})
    axes[0].set_xticks([]) 
    axes[0].set_ylabel(column_name, fontsize=9)
    axes[0].set_title(f"Box Plot: {column_name}", fontsize=10)
    axes[0].tick_params(axis='y', labelsize=8)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    # IQR Bounds for Box Plot (using current _uni_iqr_multiplier)
    q1, q3 = original_series.quantile(0.25), original_series.quantile(0.75)
    iqr_val = q3 - q1
    lower_b, upper_b = q1 - _uni_iqr_multiplier * iqr_val, q3 + _uni_iqr_multiplier * iqr_val
    axes[0].axhline(upper_b, color='orangered', linestyle='--', linewidth=1.2, label=f'Upper ({_uni_iqr_multiplier}*IQR): {upper_b:.2f}')
    axes[0].axhline(lower_b, color='orangered', linestyle='--', linewidth=1.2, label=f'Lower ({_uni_iqr_multiplier}*IQR): {lower_b:.2f}')
    axes[0].legend(fontsize=7, loc='upper right', framealpha=0.5)


    # Scatter Plot (vs Index, with outliers highlighted)
    # _df_with_uni_detected_outliers에서 해당 컬럼과 플래그 컬럼을 사용
    scatter_df = pd.DataFrame({
        'index': _df_with_uni_detected_outliers.index, # 원본 DF의 인덱스 사용
        'value': _df_with_uni_detected_outliers[column_name], # 이상치 탐지 로직이 적용된 DF의 값
        'is_outlier': _df_with_uni_detected_outliers[outlier_flag_col]
    })
    sns.scatterplot(data=scatter_df, x='index', y='value', hue='is_outlier', 
                    style='is_outlier', palette={True: "red", False: "cornflowerblue"}, 
                    markers={True: "X", False: "o"}, alpha=0.7, 
                    size='is_outlier', sizes={True: 50, False: 20}, 
                    ax=axes[1], legend='brief') # legend='auto' or 'brief'

    median_val = original_series.median() # 원본 시리즈의 중앙값
    axes[1].axhline(median_val, color='forestgreen', linestyle=':', linewidth=1.2, label=f'Median: {median_val:.2f}')
    axes[1].set_title(f"Scatter Plot (vs. Index): {column_name}", fontsize=10)
    axes[1].set_xlabel("Data Index", fontsize=9)
    axes[1].set_ylabel(column_name, fontsize=9)
    if scatter_df['is_outlier'].nunique() > 1 : # 두 종류 이상의 hue가 있을 때만 범례 표시
        axes[1].legend(fontsize=8, loc='upper right', framealpha=0.5)
    else: # 단일 종류면 범례 제거 (에러 방지)
        if axes[1].get_legend() is not None: axes[1].legend().remove()

    axes[1].tick_params(axis='both', which='major', labelsize=8)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(f"Univariate Outlier Visualization for {column_name}", fontsize=12, y=0.99 if fig_height > 5 else 1.02) # y 조정으로 제목 위치 확보
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 전체 레이아웃 조정으로 suptitle 공간 확보

    uni_plot_shared_callbacks = {
        'cache_image_data_func': main_callbacks.get('cache_image_data_func'),
        'ask_for_ollama_confirmation': main_callbacks.get('ask_for_ollama_confirmation')
        # 'initiate_ollama_analysis'는 더 이상 create_analyzable_image_widget에서 직접 필요하지 않습니다.
        # 'ask_for_ollama_confirmation'을 통해 main_app에서 처리합니다.
    }

    # DPG 이미지 위젯이 실제로 위치할 부모 DPG 아이템의 태그
    # (create_univariate_ui 함수에서 TAG_OT_VISUALIZATION_GROUP_UNI로 생성됨)
    image_parent_dpg_tag = TAG_OT_VISUALIZATION_GROUP_UNI
    
    # DPG 이미지 위젯 자체의 태그
    # (create_univariate_ui 함수에서 TAG_OT_VISUALIZATION_PLOT_IMAGE_UNI로 생성됨)
    image_widget_dpg_tag = TAG_OT_VISUALIZATION_PLOT_IMAGE_UNI
    
    image_title_for_ollama = f"단변량 이상치 분석: {column_name}"

    # 이전에 사용되던 텍스처 ID (있다면 전달하여 이전 텍스처 삭제 유도)
    existing_texture_to_replace = _uni_active_plot_texture_id

    new_texture_id, DPG_image_widget_tag_returned = create_widget_func(
        parent_dpg_tag=image_parent_dpg_tag,
        fig=fig, # 위에서 생성한 matplotlib figure 객체
        image_widget_tag=image_widget_dpg_tag,
        image_title=image_title_for_ollama,
        shared_callbacks=uni_plot_shared_callbacks, # 구성한 콜백 전달
        default_texture_tag=default_texture_tag_from_shared, # 기본 텍스처
        existing_dpg_texture_id=existing_texture_to_replace # 이전 텍스처 ID
    )

    if new_texture_id:
        _uni_active_plot_texture_id = new_texture_id # 새로 생성된 텍스처 ID로 업데이트
        _log_uni(f"Univariate plot for '{column_name}' updated/created with analyzable widget. New texture: {new_texture_id}")
    else:
        _log_uni(f"Failed to create/update analyzable univariate plot for '{column_name}'. Clearing plot.")
        _clear_uni_visualization_plot() # 실패 시 기본 이미지로 설정


def _clear_uni_visualization_plot():
    global _uni_active_plot_texture_id
    if not dpg.is_dearpygui_running(): return
    
    default_texture_tag = _shared_utils_uni['default_uni_plot_texture_tag'] if _shared_utils_uni else None
    if not default_texture_tag:
        _log_uni("Error: Default uni plot texture tag not available for clearing plot.")
        return

    if dpg.does_item_exist(TAG_OT_VISUALIZATION_PLOT_IMAGE_UNI) and dpg.does_item_exist(default_texture_tag):
        # 기본 텍스처의 크기를 가져와서 설정
        cfg = dpg.get_item_configuration(default_texture_tag)
        w,h = (cfg.get('width',100), cfg.get('height',30)) if cfg else (100,30)
        dpg.configure_item(TAG_OT_VISUALIZATION_PLOT_IMAGE_UNI, texture_tag=default_texture_tag, width=w, height=h, show=True)
    
    if _uni_active_plot_texture_id and _uni_active_plot_texture_id != default_texture_tag and dpg.does_item_exist(_uni_active_plot_texture_id):
        try:
            dpg.delete_item(_uni_active_plot_texture_id)
        except Exception as e:
            _log_uni(f"Error deleting active uni plot texture: {e}") # 디버깅 메시지
    _uni_active_plot_texture_id = default_texture_tag


def _populate_uni_treatment_table():
    # ... (기존 로직과 동일)
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_TREATMENT_TABLE_UNI): return
    dpg.delete_item(TAG_OT_TREATMENT_TABLE_UNI, children_only=True)
    cols_to_treat = [item['Column'] for item in _uni_outlier_summary_data if item.get('Detected Outliers', 0) > 0]
    if not cols_to_treat:
        dpg.add_table_column(label="Info", parent=TAG_OT_TREATMENT_TABLE_UNI, width_stretch=True)
        with dpg.table_row(parent=TAG_OT_TREATMENT_TABLE_UNI): dpg.add_text("No univariate columns with detected outliers to treat.")
        return
    headers = ["Column Name", "Treatment Method", "Parameters (Lower %tile / Abs Lower)", "Parameters (Upper %tile / Abs Upper)"]
    col_widths_treatment = [0.3, 0.3, 0.2, 0.2] 
    for i, header in enumerate(headers): dpg.add_table_column(label=header, parent=TAG_OT_TREATMENT_TABLE_UNI, init_width_or_weight=col_widths_treatment[i], width_stretch=True) # 모든 컬럼 stretch 가능하도록
    treatment_options = ["Do Not Treat", "Treat as Missing", "Ratio-based Capping", "Absolute Value Capping"]
    for col_name in cols_to_treat:
        with dpg.table_row(parent=TAG_OT_TREATMENT_TABLE_UNI):
            dpg.add_text(col_name)
            current_col_treatment = _uni_treatment_selections.get(col_name, {"method": "Do Not Treat"})
            current_method = current_col_treatment.get("method", "Do Not Treat")
            s_col = "".join(filter(str.isalnum, col_name)) # 태그용 문자열 정리
            dpg.add_combo(treatment_options, default_value=current_method, width=-1, 
                          tag=f"s5_uni_treat_combo_{s_col}_{dpg.generate_uuid()}", # 태그 고유성 강화
                          callback=_on_uni_treatment_method_change, user_data={"col_name": col_name})
            
            # Lower param group
            with dpg.group(horizontal=True): # 파라미터 입력 필드를 그룹으로 묶어 가시성 동시 제어
                tag_lp = f"s5_uni_lp_{s_col}_{dpg.generate_uuid()}"
                tag_alb = f"s5_uni_alb_{s_col}_{dpg.generate_uuid()}"
                dpg.add_input_int(width=70, default_value=current_col_treatment.get("lower_percentile", 5), 
                                  show=(current_method == "Ratio-based Capping"), 
                                  min_value=1, max_value=20, tag=tag_lp,
                                  callback=_on_uni_treatment_param_change, user_data={"col_name": col_name, "param_name": "lower_percentile"})
                dpg.add_input_float(width=100, default_value=current_col_treatment.get("abs_lower_bound", 0.0), 
                                    show=(current_method == "Absolute Value Capping"), 
                                    tag=tag_alb,
                                    callback=_on_uni_treatment_param_change, user_data={"col_name": col_name, "param_name": "abs_lower_bound"})
                if current_method not in ["Ratio-based Capping", "Absolute Value Capping"]: 
                    dpg.add_text("N/A", show=True) # 항상 N/A 텍스트 표시 (가시성은 그룹으로 제어 안 함)

            # Upper param group
            with dpg.group(horizontal=True):
                tag_up = f"s5_uni_up_{s_col}_{dpg.generate_uuid()}"
                tag_aub = f"s5_uni_aub_{s_col}_{dpg.generate_uuid()}"
                dpg.add_input_int(width=70, default_value=current_col_treatment.get("upper_percentile", 95), 
                                  show=(current_method == "Ratio-based Capping"), 
                                  min_value=80, max_value=99, tag=tag_up,
                                  callback=_on_uni_treatment_param_change, user_data={"col_name": col_name, "param_name": "upper_percentile"})
                dpg.add_input_float(width=100, default_value=current_col_treatment.get("abs_upper_bound", 0.0), 
                                    show=(current_method == "Absolute Value Capping"), 
                                    tag=tag_aub,
                                    callback=_on_uni_treatment_param_change, user_data={"col_name": col_name, "param_name": "abs_upper_bound"})
                if current_method not in ["Ratio-based Capping", "Absolute Value Capping"]: 
                    dpg.add_text("N/A", show=True)


def _on_uni_treatment_method_change(sender, app_data_method: str, user_data: Dict):
    # ... (기존 로직과 동일, _log_message -> _log_uni)
    col_name = user_data["col_name"]
    current_treatment = _uni_treatment_selections.get(col_name, {}).copy() # 복사본 사용
    current_treatment["method"] = app_data_method
    # 선택된 메소드에 따라 기본 파라미터 설정 (옵션)
    if app_data_method == "Ratio-based Capping":
        current_treatment.setdefault("lower_percentile", 5)
        current_treatment.setdefault("upper_percentile", 95)
    elif app_data_method == "Absolute Value Capping":
        # 절대값은 데이터에 따라 달라지므로 기본값 설정이 어려울 수 있음. 0.0 또는 이전 값 유지.
        current_treatment.setdefault("abs_lower_bound", 0.0) 
        current_treatment.setdefault("abs_upper_bound", 0.0)
    
    _uni_treatment_selections[col_name] = current_treatment
    _log_uni(f"Univariate Treatment for '{col_name}': {app_data_method}")
    _populate_uni_treatment_table() # 테이블을 다시 그려서 파라미터 필드 가시성 업데이트


def _on_uni_treatment_param_change(sender, app_data, user_data: Dict):
    # ... (기존 로직과 동일, _log_message -> _log_uni)
    col_name = user_data["col_name"]
    param_name = user_data["param_name"]
    current_treatment = _uni_treatment_selections.get(col_name, {}).copy()
    
    try:
        if param_name == "lower_percentile":
            val = int(app_data); 
            if not (1 <= val <= 20): raise ValueError("Lower percentile out of range 1-20")
            # 상위 퍼센타일보다 작아야 함
            upper_perc = current_treatment.get("upper_percentile", 95)
            if val >= upper_perc : raise ValueError(f"Lower percentile {val} must be less than upper percentile {upper_perc}")
            current_treatment[param_name] = val
        elif param_name == "upper_percentile":
            val = int(app_data); 
            if not (80 <= val <= 99): raise ValueError("Upper percentile out of range 80-99")
            # 하위 퍼센타일보다 커야 함
            lower_perc = current_treatment.get("lower_percentile", 5)
            if val <= lower_perc: raise ValueError(f"Upper percentile {val} must be greater than lower percentile {lower_perc}")
            current_treatment[param_name] = val
        elif param_name in ["abs_lower_bound", "abs_upper_bound"]:
            val = float(app_data)
            # 절대값 상하한 관계 체크 (옵션)
            if param_name == "abs_lower_bound" and "abs_upper_bound" in current_treatment:
                if val >= current_treatment["abs_upper_bound"]: raise ValueError("Abs lower bound must be less than abs upper bound")
            if param_name == "abs_upper_bound" and "abs_lower_bound" in current_treatment:
                if val <= current_treatment["abs_lower_bound"]: raise ValueError("Abs upper bound must be greater than abs lower bound")
            current_treatment[param_name] = val
        else:
            _log_uni(f"Warning: Unknown univariate treatment parameter '{param_name}' for column '{col_name}'.")
            return # 알 수 없는 파라미터면 업데이트하지 않음
            
        _uni_treatment_selections[col_name] = current_treatment
        _log_uni(f"Univariate Treatment parameter for '{col_name}', '{param_name}' set to {app_data}")

    except ValueError as e:
        _log_uni(f"Warning: Invalid input for '{param_name}' on column '{col_name}'. Error: {e}. Reverting.")
        # 잘못된 값 입력 시 이전 값으로 되돌리기 (sender 값 변경)
        # 또는 UI에서 입력 제한 (min_value, max_value 등 활용)
        if dpg.does_item_exist(sender):
             dpg.set_value(sender, current_treatment.get(param_name, 0)) # 기본값으로 복원 시도
        _populate_uni_treatment_table() # 테이블을 다시 그려서 올바른 값 표시


def _set_uni_recommended_treatments_logic(sender, app_data, user_data):
    global _uni_treatment_selections
    # ... (기존 로직과 동일, _log_message -> _log_uni, _show_simple_modal_message -> _show_simple_modal_uni)
    cols_to_treat = [item['Column'] for item in _uni_outlier_summary_data if item.get('Detected Outliers', 0) > 0]
    if not cols_to_treat: 
        _log_uni("No univariate columns with outliers for recommended treatments.")
        _show_simple_modal_uni("Info", "No univariate columns with outliers to set recommended treatments for.")
        return
    for col_name in cols_to_treat: 
        _uni_treatment_selections[col_name] = {"method": RECOMMENDED_UNI_TREATMENT_METHOD}
    _log_uni(f"Recommended univariate treatment ('{RECOMMENDED_UNI_TREATMENT_METHOD}') set for all eligible columns.")
    _populate_uni_treatment_table()
    _show_simple_modal_uni("Info", "Recommended univariate treatments have been set.")


def _apply_uni_outlier_treatment_logic(sender, app_data, user_data):
    global _df_after_uni_treatment # 이 변수는 단변량 처리 결과 DF를 저장
    
    _log_uni("Apply Univariate Outlier Treatments button clicked.")
    # ... (기존 로직과 동일, _log_message -> _log_uni)
    # _current_df_for_this_step 대신 _shared_utils_uni['get_current_df_func']() 사용
    # _main_app_callbacks['step5_outlier_treatment_complete'] 사용
    current_df_for_step = _shared_utils_uni['get_current_df_func']()
    main_app_callbacks = _shared_utils_uni['main_app_callbacks']

    if current_df_for_step is None:
        _log_uni("Error: No data to apply univariate treatments.")
        if main_app_callbacks and 'step5_outlier_treatment_complete' in main_app_callbacks:
            main_app_callbacks['step5_outlier_treatment_complete'](None)
        _show_simple_modal_uni("Error", "No data available to apply treatments.")
        return

    # _uni_detected_outlier_indices가 비어있고, _uni_outlier_summary_data에도 감지된 이상치가 없다면
    # 원본 DF를 그대로 완료 콜백으로 전달
    if not _uni_detected_outlier_indices and not any(item.get('Detected Outliers',0) > 0 for item in _uni_outlier_summary_data):
        _log_uni("No outliers were previously detected or no detection was run. No univariate treatments applied.")
        if main_app_callbacks and 'step5_outlier_treatment_complete' in main_app_callbacks:
            main_app_callbacks['step5_outlier_treatment_complete'](current_df_for_step.copy()) # 원본 복사본 전달
        _show_simple_modal_uni("Info", "No univariate outliers detected to treat. Original data passed through.")
        return

    _df_after_uni_treatment = current_df_for_step.copy() # 처리 결과를 담을 DF
    _log_uni("--- Starting Univariate Outlier Treatment Application ---")
    treatment_applied_to_any_column = False

    for col_name, treatment_params in _uni_treatment_selections.items():
        if col_name not in _df_after_uni_treatment.columns or col_name not in _uni_detected_outlier_indices:
            _log_uni(f"Skipping treatment for '{col_name}': column not in data or no outlier indices.")
            continue
        
        outlier_idx = _uni_detected_outlier_indices[col_name]
        if len(outlier_idx) == 0:
            _log_uni(f"No detected outliers for '{col_name}' in current indices. Skipping treatment.")
            continue

        method = treatment_params.get("method", "Do Not Treat")
        # 원본 시리즈 (처리 전 값 참조용, clip은 복사본에 적용)
        # _df_after_uni_treatment에서 직접 수정하므로, clip 전에 값을 가져올 필요 없음
        # series_before_treat = _df_after_uni_treatment[col_name].copy() 

        if method == "Treat as Missing":
            _df_after_uni_treatment.loc[outlier_idx, col_name] = np.nan
            treatment_applied_to_any_column = True
            _log_uni(f"  Univariate treatment '{method}' applied to '{col_name}'.")
        elif method == "Ratio-based Capping":
            lp = treatment_params.get("lower_percentile")
            up = treatment_params.get("upper_percentile")
            if lp is None or up is None or not (1<=lp<=20 and 80<=up<=99 and lp < up):
                _log_uni(f"  Invalid percentile parameters for Ratio-based Capping on '{col_name}'. Skipping.")
                continue
            
            # Capping을 위한 참조 시리즈는 현재 처리 중인 DF가 아닌, *원본 데이터* 또는 *탐지 시점의 데이터*를 사용하는 것이 일반적
            # 여기서는 current_df_for_step (스텝 시작 시점의 DF)를 사용
            ref_series_for_capping = current_df_for_step[col_name].dropna()
            if ref_series_for_capping.empty:
                _log_uni(f"  Reference series for capping '{col_name}' is empty. Skipping.")
                continue
            
            lower_cap_value = ref_series_for_capping.quantile(lp/100.0)
            upper_cap_value = ref_series_for_capping.quantile(up/100.0)

            if pd.isna(lower_cap_value) or pd.isna(upper_cap_value):
                _log_uni(f"  Could not determine cap values for '{col_name}' (NaN). Skipping.")
                continue

            # clip은 Series 객체에 적용해야 하므로, _df_after_uni_treatment.loc[outlier_idx, col_name]에 직접 적용 시 주의
            # 해당 위치의 값들을 가져와 clip 후 다시 할당
            values_to_cap = _df_after_uni_treatment.loc[outlier_idx, col_name]
            _df_after_uni_treatment.loc[outlier_idx, col_name] = values_to_cap.clip(lower=lower_cap_value, upper=upper_cap_value)
            treatment_applied_to_any_column = True
            _log_uni(f"  Univariate treatment '{method}' (lower:{lower_cap_value:.2f}, upper:{upper_cap_value:.2f}) applied to '{col_name}'.")

        elif method == "Absolute Value Capping":
            abs_lower_bound = treatment_params.get("abs_lower_bound")
            abs_upper_bound = treatment_params.get("abs_upper_bound")
            if abs_lower_bound is None or abs_upper_bound is None or abs_lower_bound >= abs_upper_bound:
                _log_uni(f"  Invalid absolute bound parameters for Absolute Value Capping on '{col_name}'. Skipping.")
                continue

            values_to_cap_abs = _df_after_uni_treatment.loc[outlier_idx, col_name]
            _df_after_uni_treatment.loc[outlier_idx, col_name] = values_to_cap_abs.clip(lower=abs_lower_bound, upper=abs_upper_bound)
            treatment_applied_to_any_column = True
            _log_uni(f"  Univariate treatment '{method}' (lower:{abs_lower_bound:.2f}, upper:{abs_upper_bound:.2f}) applied to '{col_name}'.")
        
        # "Do Not Treat"는 아무 작업도 하지 않음

    if treatment_applied_to_any_column:
        _log_uni("--- Univariate Outlier Treatment Application Finished ---")
        _show_simple_modal_uni("Treatment Applied", "Selected univariate outlier treatments have been applied.")
    else:
        _log_uni("--- No univariate outlier treatments were actively applied (either 'Do Not Treat' or no valid parameters/outliers). ---")
        _show_simple_modal_uni("Treatment Info", "No univariate outlier treatments were actively applied. Original data (or data from previous step) passed through.")

    if main_app_callbacks and 'step5_outlier_treatment_complete' in main_app_callbacks:
        main_app_callbacks['step5_outlier_treatment_complete'](_df_after_uni_treatment) # 처리된 DF 전달


# --- Main UI Creation and Update Functions for Univariate ---
def create_univariate_ui(parent_tab_bar_tag: str, shared_utilities: dict):
    global _shared_utils_uni, _uni_active_plot_texture_id
    _shared_utils_uni = shared_utilities
    
    # PyOD 라이브러리 누락 시 사용자에게 알림
    if PyOD_HBOS is None or PyOD_ECOD is None:
        _log_uni("Warning: PyOD library (for HBOS/ECOD) not found. Some detection methods will be unavailable.")
        # _show_simple_modal_uni("Library Missing", "PyOD library not found. HBOS and ECOD detection methods will be disabled.")


    # 부모로부터 default texture tag를 가져와 _uni_active_plot_texture_id 초기화
    _uni_active_plot_texture_id = _shared_utils_uni.get('default_uni_plot_texture_tag', None)


    with dpg.tab(label="Univariate Outlier Detection", tag=TAG_OT_UNIVARIATE_TAB, parent=parent_tab_bar_tag):
        dpg.add_text("1. Configure & Run Univariate Outlier Detection", color=[255, 255, 0])
        with dpg.group(horizontal=True):
            dpg.add_text("Detection Method:")
            # HBOS, ECOD 사용 가능 여부에 따라 항목 조절
            uni_methods = ["IQR"]
            if PyOD_HBOS: uni_methods.append("HBOS")
            if PyOD_ECOD: uni_methods.append("ECOD")
            dpg.add_radio_button(uni_methods, tag=TAG_OT_DETECT_METHOD_RADIO_UNI, default_value=_uni_selected_detection_method, horizontal=True, callback=_on_uni_detection_method_change)
        
        dpg.add_text("Detection Parameters (applied to eligible columns):")
        with dpg.group(horizontal=True, tag=TAG_OT_IQR_MULTIPLIER_INPUT + "_group_parent_uni"): # Label과 Input을 그룹화
            dpg.add_text("IQR Multiplier:", tag=TAG_OT_IQR_MULTIPLIER_INPUT + "_label")
            dpg.add_input_float(tag=TAG_OT_IQR_MULTIPLIER_INPUT, width=120, default_value=_uni_iqr_multiplier, step=0.1, callback=_on_uni_iqr_multiplier_change)
        
        if PyOD_HBOS:
            with dpg.group(horizontal=True, tag=TAG_OT_HBOS_N_BINS_INPUT + "_group_parent_uni"):
                dpg.add_text("HBOS n_bins:", tag=TAG_OT_HBOS_N_BINS_INPUT + "_label")
                dpg.add_input_int(tag=TAG_OT_HBOS_N_BINS_INPUT, width=120, default_value=_uni_hbos_n_bins, step=1, min_value=2, min_clamped=True, callback=_on_uni_hbos_n_bins_change)
        
        if PyOD_ECOD:
            with dpg.group(horizontal=True, tag=TAG_OT_ECOD_CONTAM_INPUT_UNI + "_group_parent_uni"):
                dpg.add_text("ECOD Contamination (0.0-0.5):", tag=TAG_OT_ECOD_CONTAM_INPUT_UNI + "_label") # 범위 명시
                dpg.add_input_float(tag=TAG_OT_ECOD_CONTAM_INPUT_UNI, width=120, default_value=_uni_ecod_contamination, min_value=0.0001, max_value=0.5, min_clamped=True, max_clamped=True, step=0.01, format="%.4f", callback=_on_uni_ecod_contamination_change)
        
        _update_uni_parameter_fields_visibility() # 초기 가시성 설정

        with dpg.group(horizontal=True):
            dpg.add_button(label="Run Univariate Detection", tag=TAG_OT_DETECT_BUTTON_UNI, width=-1, height=30, callback=_run_uni_outlier_detection_logic)
            dpg.add_button(label="Set Recommended Univariate Params", tag=TAG_OT_RECOMMEND_PARAMS_BUTTON_UNI, width=-1, height=30, callback=_set_uni_recommended_detection_parameters)
        dpg.add_spacer(height=5)
        
        dpg.add_text("2. Univariate Detection Summary & Visualization (Click row in table to visualize)", color=[255, 255, 0])
        with dpg.table(tag=TAG_OT_DETECTION_RESULTS_TABLE_UNI, header_row=True, resizable=True, 
                       policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=120, # 고정 높이
                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            dpg.add_table_column(label="Info", parent=TAG_OT_DETECTION_RESULTS_TABLE_UNI, width_stretch=True)
            with dpg.table_row(parent=TAG_OT_DETECTION_RESULTS_TABLE_UNI):
                dpg.add_text("Run univariate detection.", parent=dpg.last_item())
        
        with dpg.group(tag=TAG_OT_VISUALIZATION_GROUP_UNI, horizontal=False): # Plot 이미지 부모 그룹
             # 초기 이미지 크기는 기본 텍스처 크기에 맞추거나, 적절한 기본값으로 설정
            default_tex_tag = _shared_utils_uni.get('default_uni_plot_texture_tag')
            init_w, init_h = 100, 30 # 기본 텍스처 크기 (부모에서 생성된 크기)
            if default_tex_tag and dpg.does_item_exist(default_tex_tag):
                cfg = dpg.get_item_configuration(default_tex_tag)
                init_w = cfg.get('width', init_w)
                init_h = cfg.get('height', init_h)

            dpg.add_image(texture_tag=default_tex_tag or "", # 초기에는 기본 텍스처
                          tag=TAG_OT_VISUALIZATION_PLOT_IMAGE_UNI, show=True, 
                          width=init_w, height=init_h) # 크기 유동적으로 변경될 것
        dpg.add_spacer(height=5)

        dpg.add_text("3. Configure Univariate Outlier Treatment", color=[255, 255, 0])
        with dpg.table(tag=TAG_OT_TREATMENT_TABLE_UNI, header_row=True, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=150, scrollX=True, # 높이 및 가로 스크롤
                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            dpg.add_table_column(label="Info", parent=TAG_OT_TREATMENT_TABLE_UNI, width_stretch=True)
            with dpg.table_row(parent=TAG_OT_TREATMENT_TABLE_UNI):
                dpg.add_text("Run univariate detection to configure treatments.", parent=dpg.last_item())
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Set Recommended Treatments", tag=TAG_OT_RECOMMEND_TREATMENTS_BUTTON_UNI, width=-1, height=30, callback=_set_uni_recommended_treatments_logic)
            dpg.add_button(label="Reset Treatment Selections", tag=TAG_OT_RESET_TREATMENTS_BUTTON_UNI, width=-1, height=30, callback=_reset_uni_treatment_selections_to_default)
            dpg.add_button(label="Apply Selected Treatments", tag=TAG_OT_APPLY_TREATMENT_BUTTON_UNI, width=-1, height=30, callback=_apply_uni_outlier_treatment_logic)


def update_univariate_ui(df_input: Optional[pd.DataFrame], shared_utilities: dict, is_new_data: bool):
    global _shared_utils_uni
    if not dpg.is_dearpygui_running(): return
    _shared_utils_uni = shared_utilities # 최신 유틸리티로 업데이트

    if not dpg.does_item_exist(TAG_OT_UNIVARIATE_TAB): return # 탭이 없으면 업데이트 중지

    current_df_for_uni = df_input # _shared_utils_uni['get_current_df_func']() 대신 직접 받은 df 사용

    if current_df_for_uni is None or is_new_data:
        # 새 데이터가 로드되거나 데이터가 없어지면 단변량 관련 상태 초기화
        reset_univariate_state_internal(called_from_parent_reset=False) 
        if current_df_for_uni is not None:
             _log_uni("New data loaded for Univariate Outlier Detection. Please re-run detection.")
        # else:
        #      _log_uni("No data for Univariate Outlier Detection.")
             
    # UI 요소들의 값이나 가시성 업데이트 (예: 파라미터 필드 가시성)
    _update_uni_parameter_fields_visibility()
    
    # 테이블 데이터 업데이트 (필요시, 현재는 run_detection 시점에 업데이트)
    # _populate_uni_detection_results_table()
    # _populate_uni_treatment_table()

    # 시각화 클리어 (옵션: 데이터 변경 시 이전 시각화 유지 여부 결정)
    # _clear_uni_visualization_plot()


def reset_univariate_state_internal(called_from_parent_reset=True):
    """
    단변량 이상치 모듈의 내부 상태를 초기화합니다.
    부모의 전체 리셋 호출과 내부 UI 업데이트 시의 리셋을 구분.
    """
    global _df_with_uni_detected_outliers, _df_after_uni_treatment, _uni_active_plot_texture_id
    global _uni_detected_outlier_indices, _uni_outlier_summary_data, _uni_treatment_selections, _uni_columns_eligible_for_detection
    global _uni_selected_detection_method, _uni_iqr_multiplier, _uni_hbos_n_bins, _uni_ecod_contamination, _uni_currently_visualized_column, _uni_all_selectable_tags_in_table

    _df_with_uni_detected_outliers = None; _df_after_uni_treatment = None
    _uni_detected_outlier_indices.clear(); _uni_outlier_summary_data.clear(); _uni_treatment_selections.clear();
    _uni_columns_eligible_for_detection.clear(); _uni_all_selectable_tags_in_table.clear()
    
    _uni_selected_detection_method = "IQR"; _uni_iqr_multiplier = DEFAULT_UNI_IQR_MULTIPLIER
    _uni_hbos_n_bins = DEFAULT_UNI_HBOS_N_BINS; _uni_ecod_contamination = DEFAULT_UNI_ECOD_CONTAMINATION
    _uni_currently_visualized_column = None
    
    # _uni_active_plot_texture_id는 _shared_utils_uni를 통해 기본값으로 설정
    if _shared_utils_uni and 'default_uni_plot_texture_tag' in _shared_utils_uni:
        _uni_active_plot_texture_id = _shared_utils_uni['default_uni_plot_texture_tag']
    else: # 안전장치
        _uni_active_plot_texture_id = None 

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_DETECT_METHOD_RADIO_UNI): dpg.set_value(TAG_OT_DETECT_METHOD_RADIO_UNI, _uni_selected_detection_method)
        if dpg.does_item_exist(TAG_OT_IQR_MULTIPLIER_INPUT): dpg.set_value(TAG_OT_IQR_MULTIPLIER_INPUT, _uni_iqr_multiplier)
        if PyOD_HBOS and dpg.does_item_exist(TAG_OT_HBOS_N_BINS_INPUT): dpg.set_value(TAG_OT_HBOS_N_BINS_INPUT, _uni_hbos_n_bins)
        if PyOD_ECOD and dpg.does_item_exist(TAG_OT_ECOD_CONTAM_INPUT_UNI): dpg.set_value(TAG_OT_ECOD_CONTAM_INPUT_UNI, _uni_ecod_contamination)
        
        _populate_uni_detection_results_table() # 빈 테이블로 업데이트
        _populate_uni_treatment_table()         # 빈 테이블로 업데이트
        if _shared_utils_uni: # _shared_utils_uni가 설정된 이후에만 호출 가능
             _clear_uni_visualization_plot()
    
    _update_uni_parameter_fields_visibility() # 파라미터 필드 가시성도 초기화

    if not called_from_parent_reset: # 부모의 전체 리셋이 아닐 때만 자체 로그 남김
        _log_uni("Univariate outlier state reset due to data change.")


def reset_univariate_state(): # 부모 모듈에서 호출될 공개 함수
    reset_univariate_state_internal(called_from_parent_reset=True)
    _log_uni("Univariate outlier state has been reset by parent.")


def get_univariate_settings() -> dict:
    return {
        "uni_selected_detection_method": _uni_selected_detection_method,
        "uni_iqr_multiplier": _uni_iqr_multiplier,
        "uni_hbos_n_bins": _uni_hbos_n_bins,
        "uni_ecod_contamination": _uni_ecod_contamination,
        "uni_treatment_selections": _uni_treatment_selections.copy(),
        # 시각화 상태 (_uni_currently_visualized_column) 등은 저장하지 않음 (런타임 상태)
    }

def apply_univariate_settings(df_input: Optional[pd.DataFrame], settings: dict, shared_utilities: dict):
    global _uni_selected_detection_method, _uni_iqr_multiplier, _uni_hbos_n_bins, _uni_ecod_contamination, _uni_treatment_selections
    global _shared_utils_uni # Ensure shared_utils is set for this context
    
    _shared_utils_uni = shared_utilities # Update shared utilities

    _uni_selected_detection_method = settings.get("uni_selected_detection_method", "IQR")
    _uni_iqr_multiplier = settings.get("uni_iqr_multiplier", DEFAULT_UNI_IQR_MULTIPLIER)
    _uni_hbos_n_bins = settings.get("uni_hbos_n_bins", DEFAULT_UNI_HBOS_N_BINS)
    _uni_ecod_contamination = settings.get("uni_ecod_contamination", DEFAULT_UNI_ECOD_CONTAMINATION)
    _uni_treatment_selections = settings.get("uni_treatment_selections", {}).copy()

    # UI 업데이트
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_DETECT_METHOD_RADIO_UNI): dpg.set_value(TAG_OT_DETECT_METHOD_RADIO_UNI, _uni_selected_detection_method)
        if dpg.does_item_exist(TAG_OT_IQR_MULTIPLIER_INPUT): dpg.set_value(TAG_OT_IQR_MULTIPLIER_INPUT, _uni_iqr_multiplier)
        if PyOD_HBOS and dpg.does_item_exist(TAG_OT_HBOS_N_BINS_INPUT): dpg.set_value(TAG_OT_HBOS_N_BINS_INPUT, _uni_hbos_n_bins)
        if PyOD_ECOD and dpg.does_item_exist(TAG_OT_ECOD_CONTAM_INPUT_UNI): dpg.set_value(TAG_OT_ECOD_CONTAM_INPUT_UNI, _uni_ecod_contamination)
        
        _update_uni_parameter_fields_visibility() # 파라미터 필드 가시성
        # 테이블은 detection 결과가 없으므로 빈 상태로 표시되거나,
        # settings에 detection 결과가 있다면 복원 (현재는 detection 결과 저장 안 함)
        _populate_uni_detection_results_table() 
        _populate_uni_treatment_table() # 저장된 treatment selections 기반으로 테이블 다시 그림
        _clear_uni_visualization_plot() # 시각화 초기화

    _log_uni("Univariate outlier settings applied from saved state.")
    # update_univariate_ui(df_input, shared_utilities, True) # is_new_data=True로 간주하여 UI 전체 갱신 및 상태 일부 초기화