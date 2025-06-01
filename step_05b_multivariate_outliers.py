# step_05b_multivariate_outliers.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import traceback

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
except ImportError:
    shap = None

try:
    from umap import UMAP
except ImportError:
    UMAP = None

# --- DPG Tags ---
TAG_OT_MULTIVARIATE_TAB = "step5_multivariate_outlier_tab"
TAG_OT_MVA_VAR_METHOD_RADIO = "step5_ot_mva_var_method_radio"
TAG_OT_MVA_CUSTOM_COLS_TABLE = "step5_ot_mva_custom_cols_table"
TAG_OT_MVA_CUSTOM_COLS_TABLE_CHILD = TAG_OT_MVA_CUSTOM_COLS_TABLE + "_child"
TAG_OT_MVA_CUSTOM_COLS_TABLE_LABEL = TAG_OT_MVA_CUSTOM_COLS_TABLE + "_label"
TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT = "step5_ot_mva_iso_forest_contam_input"
TAG_OT_MVA_DETECT_BUTTON = "step5_ot_mva_detect_button"
TAG_OT_MVA_RESULTS_TEXT = "step5_ot_mva_results_text"
TAG_OT_MVA_UMAP_PLOT_WINDOW = "step5_mva_umap_plot_window"
TAG_OT_MVA_UMAP_PLOT = "step5_mva_umap_plot"
TAG_OT_MVA_UMAP_X_AXIS = "step5_mva_umap_x_axis"
TAG_OT_MVA_UMAP_Y_AXIS = "step5_mva_umap_y_axis"
TAG_OT_MVA_UMAP_LEGEND = "step5_mva_umap_legend"
TAG_OT_MVA_SCATTER_SERIES_NORMAL = "step5_mva_scatter_normal"
TAG_OT_MVA_SCATTER_SERIES_OUTLIER = "step5_mva_scatter_outlier"
TAG_OT_MVA_SCATTER_SERIES_SELECTED = "step5_mva_scatter_selected"
TAG_OT_MVA_SELECTED_OUTLIER_INFO_GROUP = "step5_mva_selected_outlier_info_group"
TAG_OT_MVA_SELECTED_OUTLIER_INDEX_TEXT = "step5_mva_selected_outlier_index_text"
TAG_OT_MVA_SELECTED_OUTLIER_TABLE_CHILD = "step5_mva_selected_outlier_table_child"
TAG_OT_MVA_SELECTED_OUTLIER_TABLE = "step5_mva_selected_outlier_table"
TAG_OT_MVA_SHAP_PLOT_IMAGE = "step5_mva_shap_plot_image"
TAG_OT_MVA_SHAP_PLOT_PLACEHOLDER_TEXT = TAG_OT_MVA_SHAP_PLOT_IMAGE + "_placeholder_text"

GLOBAL_MOUSE_HANDLER_REGISTRY_TAG = "global_mouse_event_handler_for_mva_plot"

# New Tags for Group Analysis
TAG_OT_MVA_GROUP_ANALYSIS_COLLAPSING_HEADER = "step5_mva_group_analysis_collapsing_header"
TAG_OT_MVA_RUN_GROUP_ANALYSIS_BUTTON = "step5_mva_run_group_analysis_button"
TAG_OT_MVA_GROUP_STATS_TABLE = "step5_mva_group_stats_table"
TAG_OT_MVA_GROUP_NUM_FEATURE_COMBO = "step5_mva_group_num_feature_combo"
TAG_OT_MVA_GROUP_DIST_PLOT_IMAGE = "step5_mva_group_dist_plot_image"
TAG_OT_MVA_GROUP_DIST_PLOT_PLACEHOLDER = TAG_OT_MVA_GROUP_DIST_PLOT_IMAGE + "_placeholder"
TAG_OT_MVA_GROUP_CAT_FEATURE_COMBO = "step5_mva_group_cat_feature_combo"
TAG_OT_MVA_GROUP_FREQ_PLOT_IMAGE = "step5_mva_group_freq_plot_image"
TAG_OT_MVA_GROUP_FREQ_PLOT_PLACEHOLDER = TAG_OT_MVA_GROUP_FREQ_PLOT_IMAGE + "_placeholder"

# --- Constants ---
DEFAULT_MVA_ISO_FOREST_CONTAMINATION = 'auto'
UMAP_PLOT_WINDOW_HEIGHT = 380 # UMAP 플롯 창 높이 조정
SELECTED_POINT_INFO_GROUP_WIDTH = 480 
SHAP_PLOT_IMAGE_WIDTH = SELECTED_POINT_INFO_GROUP_WIDTH - 40
SELECTED_POINT_TABLE_CHILD_HEIGHT = 220 # 선택된 포인트 테이블 높이 조정
GROUP_ANALYSIS_PLOT_WIDTH = SELECTED_POINT_INFO_GROUP_WIDTH - 40 # 그룹 분석 플롯 너비
GROUP_ANALYSIS_PLOT_HEIGHT = 300 # 그룹 분석 플롯 높이

# --- Module State Variables ---
_shared_utils_mva: Optional[Dict[str, Any]] = None
_s1_column_types_cache: Optional[Dict[str, str]] = None
_mva_variable_selection_method: str = "All Numeric Columns"
_mva_custom_selected_columns: List[str] = [] # <--- 이 변수가 여기서 초기화됩니다.
_mva_custom_col_checkbox_tags: Dict[str, str] = {}
_mva_iso_forest_contamination: Union[str, float] = DEFAULT_MVA_ISO_FOREST_CONTAMINATION

# ... (기존 상태 변수들)
_df_with_mva_outliers: Optional[pd.DataFrame] = None
_mva_iso_forest_model: Optional[IsolationForest] = None
_mva_cols_analyzed_for_if: List[str] = []
_mva_umap_embedding: Optional[np.ndarray] = None
_mva_umap_original_indices: Optional[np.ndarray] = None
_mva_outlier_scores: Optional[np.ndarray] = None
_mva_selected_point_original_index: Optional[Any] = None
_mva_shap_plot_active_texture_id: Optional[str] = None

# New State Variables for Group Analysis
_mva_group_stats_df: Optional[pd.DataFrame] = None # 비교 통계량 저장
_mva_group_numerical_cols_for_plot: List[str] = []
_mva_group_categorical_cols_for_plot: List[str] = []
_mva_group_dist_plot_texture_id: Optional[str] = None
_mva_group_freq_plot_texture_id: Optional[str] = None
_normal_group_df_cache: Optional[pd.DataFrame] = None # 그룹 분석용 정상 그룹 DF 캐시
_outlier_group_df_cache: Optional[pd.DataFrame] = None # 그룹 분석용 이상치 그룹 DF 캐시


# --- Helper Functions ---
def _log_mva(message: str): # ... (이전과 동일)
    if _shared_utils_mva and 'log_message_func' in _shared_utils_mva:
        _shared_utils_mva['log_message_func'](f"[MVAOutlier] {message}")

def _show_simple_modal_mva(title: str, message: str, width: int = 450, height: int = 200): # ... (이전과 동일)
    if _shared_utils_mva and 'util_funcs_common' in _shared_utils_mva and \
       '_show_simple_modal_message' in _shared_utils_mva['util_funcs_common']:
        _shared_utils_mva['util_funcs_common']['_show_simple_modal_message'](title, message, width, height)

def _get_s1_column_types() -> Dict[str, str]: # ... (이전과 동일)
    global _s1_column_types_cache
    if _s1_column_types_cache is None: 
        if _shared_utils_mva and 'main_app_callbacks' in _shared_utils_mva and \
           'get_column_analysis_types' in _shared_utils_mva['main_app_callbacks']:
            _s1_column_types_cache = _shared_utils_mva['main_app_callbacks']['get_column_analysis_types']()
        else: _s1_column_types_cache = {}
    return _s1_column_types_cache

def _get_feature_type(feature_name: str, df: pd.DataFrame) -> str: # ... (이전과 동일)
    s1_types = _get_s1_column_types()
    if feature_name in s1_types:
        s1_type = s1_types[feature_name]
        if "Numeric" in s1_type and "Binary" not in s1_type: return "Numeric"
        if "Categorical" in s1_type or "Text" in s1_type or "Binary" in s1_type: return "Categorical"
    if feature_name in df.columns:
        dtype = df[feature_name].dtype
        if pd.api.types.is_numeric_dtype(dtype): return "Numeric"
        if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_bool_dtype(dtype): return "Categorical"
    return "Other"

# --- Multivariate Callbacks and Logic ---
# _update_mva_custom_cols_ui_visibility, _on_mva_var_method_change, ... _detect_outliers_mva_iso_forest,
# _run_mva_outlier_detection_logic, _perform_umap_and_update_dpg_plot, _clear_dpg_umap_plot_series,
# _update_dpg_umap_plot_series, _on_mva_umap_plot_click 등은 이전 답변 내용과 거의 동일하게 유지.
# (이전 답변의 최종본을 사용한다고 가정)
# ... (이전 함수들 붙여넣기 - 공간 절약을 위해 생략) ...
def _update_mva_custom_cols_ui_visibility(): # 이전과 동일
    if not dpg.is_dearpygui_running(): return
    show_custom_table = (_mva_variable_selection_method == "Select Custom Columns")
    if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE_CHILD):
        dpg.configure_item(TAG_OT_MVA_CUSTOM_COLS_TABLE_CHILD, show=show_custom_table)
    if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE_LABEL):
        dpg.configure_item(TAG_OT_MVA_CUSTOM_COLS_TABLE_LABEL, show=show_custom_table)

def _on_mva_var_method_change(sender, app_data: str, user_data): # 이전과 동일
    global _mva_variable_selection_method
    _mva_variable_selection_method = app_data
    _log_mva(f"Multivariate variable selection method: {_mva_variable_selection_method}")
    _update_mva_custom_cols_ui_visibility()
    if _mva_variable_selection_method == "Select Custom Columns":
        current_df = _shared_utils_mva['get_current_df_func']() if _shared_utils_mva else None
        if current_df is not None: _populate_mva_custom_cols_table(current_df)

def _on_mva_custom_col_checkbox_change(sender, app_data_is_checked: bool, user_data_col_name: str): # 이전과 동일
    global _mva_custom_selected_columns
    if app_data_is_checked:
        if user_data_col_name not in _mva_custom_selected_columns: _mva_custom_selected_columns.append(user_data_col_name)
    else:
        if user_data_col_name in _mva_custom_selected_columns: _mva_custom_selected_columns.remove(user_data_col_name)
    _log_mva(f"MVA custom columns updated: {_mva_custom_selected_columns}")

def _populate_mva_custom_cols_table(df: Optional[pd.DataFrame]): # 이전과 동일
    global _mva_custom_col_checkbox_tags
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE): return
    dpg.delete_item(TAG_OT_MVA_CUSTOM_COLS_TABLE, children_only=True)
    _mva_custom_col_checkbox_tags.clear()
    if df is None:
        with dpg.table_row(parent=TAG_OT_MVA_CUSTOM_COLS_TABLE): dpg.add_text("No data.")
        return
    s1_col_types = _get_s1_column_types()
    numeric_cols_for_mva = [col for col in df.columns if _get_feature_type(col, df) == "Numeric"]
    if not numeric_cols_for_mva:
        with dpg.table_row(parent=TAG_OT_MVA_CUSTOM_COLS_TABLE): dpg.add_text("No suitable numeric columns.")
        return
    for col_name in numeric_cols_for_mva:
        with dpg.table_row(parent=TAG_OT_MVA_CUSTOM_COLS_TABLE):
            checkbox_tag = f"mva_cb_{''.join(filter(str.isalnum, col_name))}"
            _mva_custom_col_checkbox_tags[col_name] = checkbox_tag
            dpg.add_checkbox(tag=checkbox_tag, default_value=(col_name in _mva_custom_selected_columns), user_data=col_name, callback=_on_mva_custom_col_checkbox_change)
            dpg.add_text(col_name)

def _on_mva_iso_forest_contam_change(sender, app_data_str: str, user_data):
    global _mva_iso_forest_contamination
    
    # app_data_str은 input_text 위젯에서 현재 입력된 문자열 값입니다.
    cleaned_user_input = app_data_str.strip()
    
    new_contamination_state = _mva_iso_forest_contamination # 현재 상태를 기본값으로 가정
    ui_should_be_updated_to = cleaned_user_input # UI는 일단 사용자 입력대로 두되, 필요시 수정

    if not cleaned_user_input: # 사용자가 모든 글자를 지워 입력창이 비었을 경우
        new_contamination_state = 'auto'
        # UI를 'auto'로 되돌릴지, 아니면 빈칸으로 둘지는 선택사항입니다.
        # 여기서는 'auto'로 상태를 변경하고, UI는 사용자가 지운 상태(빈칸)를 잠시 유지하도록 합니다.
        # 사용자가 다른 곳을 클릭하거나 Enter를 치면 최종적으로 'auto'로 설정될 수 있도록 유도.
        # 다만, 즉시 'auto'로 복원하는 것이 일관성 있을 수 있습니다.
        ui_should_be_updated_to = 'auto' 
    elif cleaned_user_input.lower() == 'auto':
        new_contamination_state = 'auto'
        ui_should_be_updated_to = 'auto' # 대소문자 구분 없이 'auto'로 통일
    else:
        try:
            val = float(cleaned_user_input)
            if 0.0001 <= val <= 0.5:
                new_contamination_state = val
                # 유효한 숫자 범위이므로 UI는 사용자 입력값 그대로 둠 (또는 포맷팅 된 값으로 변경)
                # ui_should_be_updated_to = f"{val:.4f}" # 예: 소수점 4자리 포맷팅
            else: # 숫자이지만 범위 밖
                new_contamination_state = 'auto' # 또는 이전 유효값으로 복원
                ui_should_be_updated_to = 'auto'
                _show_simple_modal_mva("Input Error", f"Contamination value {val} is out of range (0.0001-0.5). Reverted to 'auto'.")
        except ValueError: # 숫자로 변환 불가 (그리고 'auto'도 아님)
            new_contamination_state = 'auto' # 또는 이전 유효값으로 복원
            ui_should_be_updated_to = 'auto'
            _show_simple_modal_mva("Input Error", f"Invalid input '{cleaned_user_input}'. Please use 'auto' or a number between 0.0001 and 0.5.")

    # 실제 상태 변수 업데이트 (변경되었을 경우에만)
    if _mva_iso_forest_contamination != new_contamination_state:
        _mva_iso_forest_contamination = new_contamination_state
        _log_mva(f"MVA Isolation Forest contamination state updated to: '{_mva_iso_forest_contamination}'")

    # UI 값 업데이트 (콜백이 반환된 후 DPG가 자동으로 UI를 업데이트하지만, 강제 동기화가 필요할 경우)
    # 사용자의 현재 타이핑을 방해하지 않도록, 최종적으로 UI에 반영되어야 하는 값이 현재 UI 값과 다를 때만 설정
    current_dpg_value = dpg.get_value(sender)
    if current_dpg_value != ui_should_be_updated_to:
        dpg.set_value(sender, ui_should_be_updated_to)

def _get_mva_columns_to_analyze(current_df: pd.DataFrame) -> List[str]: # 이전과 동일
    global _mva_cols_analyzed_for_if
    _mva_cols_analyzed_for_if = []
    if current_df is None: return []
    all_suitable_numeric_cols = [col for col in current_df.columns if _get_feature_type(col, current_df) == "Numeric"]
    if not all_suitable_numeric_cols: _log_mva("No suitable numeric columns for MVA."); return []
    if _mva_variable_selection_method == "All Numeric Columns": _mva_cols_analyzed_for_if = all_suitable_numeric_cols[:]
    elif _mva_variable_selection_method == "Recommended Columns (TODO)": _mva_cols_analyzed_for_if = all_suitable_numeric_cols[:]
    elif _mva_variable_selection_method == "Select Custom Columns":
        valid_custom_cols = [col for col in _mva_custom_selected_columns if col in all_suitable_numeric_cols]
        _mva_cols_analyzed_for_if = valid_custom_cols[:] if valid_custom_cols else all_suitable_numeric_cols[:]
    _log_mva(f"MVA using {len(_mva_cols_analyzed_for_if)} columns: {_mva_cols_analyzed_for_if}")
    return _mva_cols_analyzed_for_if

def _detect_outliers_mva_iso_forest(df_subset: pd.DataFrame, contamination_val: Union[str, float]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]: # 이전과 동일
    global _mva_outlier_scores, _mva_iso_forest_model
    _mva_outlier_scores = None; _mva_iso_forest_model = None
    if df_subset.empty or df_subset.shape[1] == 0: _log_mva("MVA IF: Input empty."); return None, None
    df_numeric_imputed = df_subset.copy()
    for col in df_numeric_imputed.columns:
        if pd.api.types.is_numeric_dtype(df_numeric_imputed[col]) and df_numeric_imputed[col].isnull().any():
            df_numeric_imputed[col].fillna(df_numeric_imputed[col].median(), inplace=True)
    if df_numeric_imputed.shape[0] < 2: _log_mva("MVA IF: Not enough samples."); return None, None
    valid_contamination = contamination_val
    _mva_iso_forest_model = IsolationForest(contamination=valid_contamination, random_state=42, n_estimators=100)
    try:
        _mva_iso_forest_model.fit(df_numeric_imputed)
        predictions = _mva_iso_forest_model.predict(df_numeric_imputed)
        scores = _mva_iso_forest_model.decision_function(df_numeric_imputed)
        _mva_outlier_scores = scores
    except Exception as e:
        _log_mva(f"MVA IF error with contam '{valid_contamination}': {e}. Trying 'auto'.")
        try:
            _mva_iso_forest_model = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
            _mva_iso_forest_model.fit(df_numeric_imputed)
            predictions = _mva_iso_forest_model.predict(df_numeric_imputed)
            scores = _mva_iso_forest_model.decision_function(df_numeric_imputed)
            _mva_outlier_scores = scores
        except Exception as e_auto:
            _log_mva(f"MVA IF with 'auto' also failed: {e_auto}"); _mva_iso_forest_model=None; return None, None
    return df_numeric_imputed.index[predictions == -1].to_numpy(), scores

def _run_mva_outlier_detection_logic(sender, app_data, user_data): # 이전과 동일
    global _df_with_mva_outliers, _mva_outlier_row_indices, _mva_outlier_scores
    _log_mva("Run MVA Detection button clicked.")
    current_df = _shared_utils_mva['get_current_df_func']()
    if current_df is None: _log_mva("Error: No data for MVA."); _show_simple_modal_mva("Error", "No data for MVA."); return
    cols_to_analyze = _get_mva_columns_to_analyze(current_df)
    if not cols_to_analyze or len(cols_to_analyze) < 2:
         _log_mva("MVA Error: Min 2 numeric cols required."); _show_simple_modal_mva("Error", "Min 2 numeric cols for MVA."); return
    df_subset_for_mva = current_df[cols_to_analyze].copy()
    _log_mva(f"--- Starting MVA Detection (IF) on {len(_mva_cols_analyzed_for_if)} columns ---")
    _log_mva(f"  Contamination: '{_mva_iso_forest_contamination}'")
    detected_indices, _ = _detect_outliers_mva_iso_forest(df_subset_for_mva, _mva_iso_forest_contamination)
    _mva_outlier_row_indices = detected_indices
    if _mva_iso_forest_model is None or _mva_outlier_row_indices is None:
        _log_mva("MVA Detection or IF model training failed."); _show_simple_modal_mva("Failed", "MVA Detection failed. Check logs."); return
    num_outliers = len(_mva_outlier_row_indices)
    total_rows = len(current_df)
    percentage = (num_outliers / total_rows * 100) if total_rows > 0 else 0
    summary = f"MVA Detection (IF):\n- Cols: {len(_mva_cols_analyzed_for_if)}\n- Outliers: {num_outliers} ({percentage:.2f}%)\n- Contam: '{_mva_iso_forest_contamination}'"
    _log_mva(summary)
    if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, summary)
    _df_with_mva_outliers = current_df.copy()
    _df_with_mva_outliers['is_mva_outlier'] = False
    if num_outliers > 0: _df_with_mva_outliers.loc[_mva_outlier_row_indices, 'is_mva_outlier'] = True
    _perform_umap_and_update_dpg_plot(df_subset_for_mva)
    _show_simple_modal_mva("MVA Detection Complete", f"Found {num_outliers} outlier rows.")
    _clear_selected_point_info()
    _clear_group_analysis_ui_elements() # 그룹 분석 UI도 초기화

def _perform_umap_and_update_dpg_plot(df_for_umap_input: pd.DataFrame): # 이전과 동일
    global _mva_umap_embedding, _mva_umap_original_indices
    _log_mva("Performing UMAP for DPG plot...")
    if UMAP is None: _log_mva("UMAP lib not found."); _show_simple_modal_mva("Lib Missing", "Install 'umap-learn'."); _clear_dpg_umap_plot_series(); return
    if df_for_umap_input.empty: _log_mva("UMAP input empty."); _clear_dpg_umap_plot_series(); return
    df_for_umap = df_for_umap_input.copy()
    for col in df_for_umap.columns:
        if df_for_umap[col].isnull().any(): df_for_umap[col].fillna(df_for_umap[col].median(), inplace=True)
    if df_for_umap.shape[0] < 2: _log_mva("Not enough samples for UMAP."); _clear_dpg_umap_plot_series(); return
    n_neighbors_umap = min(15, df_for_umap.shape[0] - 1 if df_for_umap.shape[0] > 1 else 1)
    if n_neighbors_umap <= 0: n_neighbors_umap = 1
    try:
        reducer = UMAP(n_neighbors=n_neighbors_umap, n_components=2, min_dist=0.1, random_state=42, n_jobs=1)
        _mva_umap_embedding = reducer.fit_transform(df_for_umap)
        _mva_umap_original_indices = df_for_umap.index.to_numpy()
        _update_dpg_umap_plot_series()
        _log_mva("UMAP embedding calculated & DPG plot updated.")
    except Exception as e: _log_mva(f"UMAP calc/plot error: {e}"); _clear_dpg_umap_plot_series()

def _clear_dpg_umap_plot_series(): # 이전과 동일
    tags_to_delete = [TAG_OT_MVA_SCATTER_SERIES_NORMAL, TAG_OT_MVA_SCATTER_SERIES_OUTLIER, TAG_OT_MVA_SCATTER_SERIES_SELECTED]
    for tag in tags_to_delete:
        if dpg.does_item_exist(tag): dpg.delete_item(tag)

def _update_dpg_umap_plot_series(selected_embedding_idx: Optional[int] = None): # 이전과 동일
    _clear_dpg_umap_plot_series()
    if _mva_umap_embedding is None or _df_with_mva_outliers is None or _mva_umap_original_indices is None:
        _log_mva("Cannot update UMAP: embedding or outlier/original index data missing."); return
    outlier_flags_for_embedding = _df_with_mva_outliers.loc[_mva_umap_original_indices, 'is_mva_outlier'].to_numpy()
    normal_indices_in_embedding = np.where(outlier_flags_for_embedding == False)[0]
    outlier_indices_in_embedding = np.where(outlier_flags_for_embedding == True)[0]
    if len(normal_indices_in_embedding) > 0:
        dpg.add_scatter_series(x=_mva_umap_embedding[normal_indices_in_embedding, 0].tolist(), y=_mva_umap_embedding[normal_indices_in_embedding, 1].tolist(), label="Normal", parent=TAG_OT_MVA_UMAP_Y_AXIS, tag=TAG_OT_MVA_SCATTER_SERIES_NORMAL)
    if len(outlier_indices_in_embedding) > 0:
        dpg.add_scatter_series(x=_mva_umap_embedding[outlier_indices_in_embedding, 0].tolist(), y=_mva_umap_embedding[outlier_indices_in_embedding, 1].tolist(), label="Outlier", parent=TAG_OT_MVA_UMAP_Y_AXIS, tag=TAG_OT_MVA_SCATTER_SERIES_OUTLIER)
    if selected_embedding_idx is not None and 0 <= selected_embedding_idx < len(_mva_umap_embedding):
        dpg.add_scatter_series(x=[_mva_umap_embedding[selected_embedding_idx, 0]], y=[_mva_umap_embedding[selected_embedding_idx, 1]], label="Selected", parent=TAG_OT_MVA_UMAP_Y_AXIS, tag=TAG_OT_MVA_SCATTER_SERIES_SELECTED)
    dpg.fit_axis_data(TAG_OT_MVA_UMAP_X_AXIS); dpg.fit_axis_data(TAG_OT_MVA_UMAP_Y_AXIS)
    if dpg.does_item_exist(TAG_OT_MVA_UMAP_LEGEND):
        dpg.configure_item(TAG_OT_MVA_UMAP_LEGEND, show=(len(normal_indices_in_embedding) > 0 or len(outlier_indices_in_embedding) > 0 or selected_embedding_idx is not None))

def _on_mva_umap_plot_click(sender, app_data, user_data): # 이전과 동일
    global _mva_selected_point_original_index
    if not dpg.is_item_hovered(TAG_OT_MVA_UMAP_PLOT): return
    _log_mva(f"UMAP Plot area clicked (via global handler). Sender: {sender}, Clicked Button: {app_data}")
    if _mva_umap_embedding is None or _df_with_mva_outliers is None or _mva_umap_original_indices is None: _log_mva("UMAP data not ready."); return
    plot_mouse_pos = dpg.get_plot_mouse_pos()
    if plot_mouse_pos is None or len(plot_mouse_pos) < 2: _log_mva("Could not get plot mouse pos."); return # 선택 해제 로직 추가 가능
    clicked_x, clicked_y = plot_mouse_pos[0], plot_mouse_pos[1]
    _log_mva(f"Plot click data coordinates: x={clicked_x:.2f}, y={clicked_y:.2f}")
    distances = np.sqrt(np.sum((_mva_umap_embedding - np.array([clicked_x, clicked_y]))**2, axis=1))
    if len(distances) == 0: _log_mva("No points in UMAP embedding."); return
    nearest_point_idx_in_embedding = np.argmin(distances)
    x_axis_tag_to_query = TAG_OT_MVA_UMAP_X_AXIS
    if not dpg.does_item_exist(x_axis_tag_to_query): _log_mva(f"X-axis '{x_axis_tag_to_query}' does not exist."); return
    x_limits = dpg.get_axis_limits(x_axis_tag_to_query)
    data_dist_threshold = (x_limits[1] - x_limits[0]) * 0.05 if x_limits[0] != x_limits[1] else 0.1
    if x_limits[0] == 0.0 and x_limits[1] == 0.0 and len(_mva_umap_embedding) > 0: # 축 초기화 상태 보정
        min_x_emb, max_x_emb = np.min(_mva_umap_embedding[:,0]), np.max(_mva_umap_embedding[:,0])
        data_dist_threshold = (max_x_emb - min_x_emb) * 0.05 if min_x_emb != max_x_emb else 0.1
    if distances[nearest_point_idx_in_embedding] > data_dist_threshold :
        _log_mva(f"No point selected, click too far (dist: {distances[nearest_point_idx_in_embedding]:.2f} > thres: {data_dist_threshold:.2f}).")
        if _mva_selected_point_original_index is not None: _clear_selected_point_info(); _update_dpg_umap_plot_series()
        return
    newly_selected_original_index = _mva_umap_original_indices[nearest_point_idx_in_embedding]
    if newly_selected_original_index == _mva_selected_point_original_index:
        _log_mva(f"Point {newly_selected_original_index} re-clicked. Deselecting.")
        _clear_selected_point_info(); _update_dpg_umap_plot_series()
        return
    _mva_selected_point_original_index = newly_selected_original_index
    _log_mva(f"UMAP point selected. OriginalDF Index: {_mva_selected_point_original_index}, Embedding Index: {nearest_point_idx_in_embedding}")
    _update_selected_point_info_ui()
    _update_dpg_umap_plot_series(selected_embedding_idx=nearest_point_idx_in_embedding)

def _clear_selected_point_info(): # "Original Values" 테이블 헤더 유지하도록 수정
    global _mva_selected_point_original_index, _mva_shap_plot_active_texture_id
    _mva_selected_point_original_index = None
    if dpg.does_item_exist(TAG_OT_MVA_SELECTED_OUTLIER_INDEX_TEXT):
        dpg.set_value(TAG_OT_MVA_SELECTED_OUTLIER_INDEX_TEXT, "Selected Point: (None - Click a point on UMAP)")
    if dpg.does_item_exist(TAG_OT_MVA_SELECTED_OUTLIER_TABLE):
        dpg.delete_item(TAG_OT_MVA_SELECTED_OUTLIER_TABLE, children_only=True)
        with dpg.table_row(parent=TAG_OT_MVA_SELECTED_OUTLIER_TABLE): # Placeholder
            for i in range(5): dpg.add_text("...") # 컬럼 수에 맞게
            
    default_shap_texture = _shared_utils_mva.get("default_shap_plot_texture_tag", "") if _shared_utils_mva else ""
    if _mva_shap_plot_active_texture_id and _mva_shap_plot_active_texture_id != default_shap_texture and dpg.does_item_exist(_mva_shap_plot_active_texture_id):
        try: dpg.delete_item(_mva_shap_plot_active_texture_id)
        except Exception as e: _log_mva(f"Error deleting active SHAP texture: {e}")
    _mva_shap_plot_active_texture_id = default_shap_texture
    if dpg.does_item_exist(TAG_OT_MVA_SHAP_PLOT_IMAGE):
        dpg.configure_item(TAG_OT_MVA_SHAP_PLOT_IMAGE, texture_tag=default_shap_texture or "", show=False)
    if dpg.does_item_exist(TAG_OT_MVA_SHAP_PLOT_PLACEHOLDER_TEXT):
        dpg.configure_item(TAG_OT_MVA_SHAP_PLOT_PLACEHOLDER_TEXT, show=True)
        dpg.set_value(TAG_OT_MVA_SHAP_PLOT_PLACEHOLDER_TEXT, "SHAP plot will appear here.")


def _update_selected_point_info_ui(): # 이전 답변의 강화된 버전 사용
    if _mva_selected_point_original_index is None or _df_with_mva_outliers is None: _clear_selected_point_info(); return
    current_df_for_context = _df_with_mva_outliers # 이상치 플래그가 포함된 DF 사용
    normal_df = current_df_for_context[current_df_for_context['is_mva_outlier'] == False]
    if dpg.does_item_exist(TAG_OT_MVA_SELECTED_OUTLIER_INDEX_TEXT):
        status = "Outlier" if current_df_for_context.loc[_mva_selected_point_original_index, 'is_mva_outlier'] else "Normal"
        score_text = ""
        if _mva_outlier_scores is not None and _mva_umap_original_indices is not None:
            emb_idx_list = np.where(_mva_umap_original_indices == _mva_selected_point_original_index)[0]
            if len(emb_idx_list) > 0 and 0 <= emb_idx_list[0] < len(_mva_outlier_scores):
                score_text = f"(Score: {_mva_outlier_scores[emb_idx_list[0]]:.3f})"
        dpg.set_value(TAG_OT_MVA_SELECTED_OUTLIER_INDEX_TEXT, f"Selected Point Index: {_mva_selected_point_original_index} (Status: {status}) {score_text}")
    if dpg.does_item_exist(TAG_OT_MVA_SELECTED_OUTLIER_TABLE):
        dpg.delete_item(TAG_OT_MVA_SELECTED_OUTLIER_TABLE, children_only=True)
        selected_data_series = current_df_for_context.loc[_mva_selected_point_original_index]
        for feature, value in selected_data_series.items():
            if feature == 'is_mva_outlier': continue
            feature_type = _get_feature_type(feature, current_df_for_context)
            norm_range_txt, z_score_txt, norm_freq_txt, val_color = "N/A", "N/A", "N/A", (255,255,255,255)
            if feature_type == "Numeric" and feature in normal_df.columns and not normal_df[feature].empty:
                norm_series = normal_df[feature].dropna()
                if len(norm_series) > 1:
                    p05, p95 = norm_series.quantile(0.05), norm_series.quantile(0.95)
                    mean, std = norm_series.mean(), norm_series.std()
                    norm_range_txt = f"{p05:.3f}~{p95:.3f}"
                    if pd.notna(value) and std > 1e-6 :
                        z = (value - mean) / std; z_score_txt = f"{z:.2f}"
                        if abs(z) > 3: val_color = (255,0,0,255) # Red
                        elif abs(z) > 2: val_color = (255,165,0,255) # Orange
                    elif pd.notna(value) and (value < p05 or value > p95) and val_color == (255,255,255,255):
                        val_color = (255,255,0,255) # Yellow
            elif feature_type == "Categorical" and feature in normal_df.columns and not normal_df[feature].empty:
                norm_series_cat = normal_df[feature].dropna()
                if not norm_series_cat.empty:
                    freq = norm_series_cat.value_counts(normalize=True).get(value,0)*100
                    norm_freq_txt = f"{freq:.1f}%"
                    if freq < 1: val_color = (255,100,100,255)
                    elif freq < 5: val_color = (255,200,0,255)
            with dpg.table_row(parent=TAG_OT_MVA_SELECTED_OUTLIER_TABLE):
                dpg.add_text(str(feature))
                dpg.add_text(f"{value:.4f}" if isinstance(value, (float,np.floating)) else str(value), color=val_color)
                dpg.add_text(norm_range_txt); dpg.add_text(z_score_txt); dpg.add_text(norm_freq_txt)
    _calculate_and_display_shap_values()

def _calculate_and_display_shap_values(): # 이전 답변의 최종본 사용 (거의 동일)
    global _mva_shap_plot_active_texture_id
    # ... (이전 코드와 동일, 로깅 메시지 등 약간의 톤 다운 가능)
    placeholder_tag = TAG_OT_MVA_SHAP_PLOT_PLACEHOLDER_TEXT
    image_tag = TAG_OT_MVA_SHAP_PLOT_IMAGE
    if _mva_selected_point_original_index is None or _mva_iso_forest_model is None or not _mva_cols_analyzed_for_if:
        if dpg.does_item_exist(placeholder_tag): dpg.configure_item(placeholder_tag, show=True); dpg.set_value(placeholder_tag, "Select point for SHAP.")
        if dpg.does_item_exist(image_tag): dpg.configure_item(image_tag, show=False)
        return
    if shap is None:
        if dpg.does_item_exist(placeholder_tag): dpg.configure_item(placeholder_tag, show=True); dpg.set_value(placeholder_tag, "SHAP library not installed.")
        if dpg.does_item_exist(image_tag): dpg.configure_item(image_tag, show=False); return
    current_df_for_shap = _df_with_mva_outliers
    if current_df_for_shap is None: _log_mva("SHAP: current_df (with flags) is None."); return
    data_for_shap_series = current_df_for_shap.loc[_mva_selected_point_original_index, _mva_cols_analyzed_for_if]
    data_for_shap_df_imputed = pd.DataFrame([data_for_shap_series])
    for col in data_for_shap_df_imputed.columns:
        if data_for_shap_df_imputed[col].isnull().any():
            col_median_for_shap = current_df_for_shap[col].median() 
            data_for_shap_df_imputed[col].fillna(col_median_for_shap, inplace=True)
    if data_for_shap_df_imputed.isnull().any().any(): _log_mva("SHAP Error: Data has NaNs after imputation."); return
    try:
        explainer = shap.TreeExplainer(_mva_iso_forest_model)
        shap_values_for_point = explainer.shap_values(data_for_shap_df_imputed.iloc[0])
        shap_values_to_plot = shap_values_for_point[1] if isinstance(shap_values_for_point, list) and len(shap_values_for_point)==2 else shap_values_for_point
        fig_height = max(4, len(_mva_cols_analyzed_for_if) * 0.35 + 1.5)
        fig_width_inches = (SHAP_PLOT_IMAGE_WIDTH - 20) / 80 
        fig, ax = plt.subplots(figsize=(fig_width_inches , fig_height ))
        shap_series = pd.Series(shap_values_to_plot, index=_mva_cols_analyzed_for_if).sort_values(ascending=True)
        colors = ['#d62728' if x > 0 else '#1f77b4' for x in shap_series.values]
        shap_series.plot(kind='barh', ax=ax, color=colors)
        ax.set_xlabel("SHAP Value (Contribution to Anomaly Score)"); ax.set_title(f"SHAP Values for Point: {str(_mva_selected_point_original_index)[:30]}", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8); plt.tight_layout()
        plot_texture_func = _shared_utils_mva['plot_to_dpg_texture_func']
        texture_tag_new, tex_w, tex_h = plot_texture_func(fig); plt.close(fig)
        default_shap_texture = _shared_utils_mva.get("default_shap_plot_texture_tag", "")
        if _mva_shap_plot_active_texture_id and _mva_shap_plot_active_texture_id != default_shap_texture and dpg.does_item_exist(_mva_shap_plot_active_texture_id):
            dpg.delete_item(_mva_shap_plot_active_texture_id)
        if texture_tag_new and tex_w > 0 and tex_h > 0:
            _mva_shap_plot_active_texture_id = texture_tag_new
            disp_h = int(tex_h * (SHAP_PLOT_IMAGE_WIDTH / tex_w)) if tex_w > 0 else tex_h
            dpg.configure_item(image_tag, texture_tag=_mva_shap_plot_active_texture_id, width=SHAP_PLOT_IMAGE_WIDTH, height=disp_h, show=True)
            if dpg.does_item_exist(placeholder_tag): dpg.configure_item(placeholder_tag, show=False)
        else:
            if dpg.does_item_exist(placeholder_tag): dpg.set_value(placeholder_tag, "Error generating SHAP plot."); dpg.configure_item(placeholder_tag, show=True)
            if dpg.does_item_exist(image_tag): dpg.configure_item(image_tag, show=False)
    except Exception as e:
        _log_mva(f"Error in SHAP calculation/plot: {e}\n{traceback.format_exc()}")
        if dpg.does_item_exist(placeholder_tag): dpg.set_value(placeholder_tag, f"SHAP Error: {e}"); dpg.configure_item(placeholder_tag, show=True)
        if dpg.does_item_exist(image_tag): dpg.configure_item(image_tag, show=False)

# --- 이상치 그룹 분석 관련 함수 ---
def _run_group_analysis(sender, app_data, user_data):
    global _mva_group_stats_df, _mva_group_numerical_cols_for_plot, _mva_group_categorical_cols_for_plot
    global _normal_group_df_cache, _outlier_group_df_cache

    _log_mva("Run Group Analysis button clicked.")
    if _df_with_mva_outliers is None or 'is_mva_outlier' not in _df_with_mva_outliers.columns:
        _show_simple_modal_mva("Error", "Please run MVA detection first to define outlier groups.")
        return

    _normal_group_df_cache = _df_with_mva_outliers[_df_with_mva_outliers['is_mva_outlier'] == False].copy()
    _outlier_group_df_cache = _df_with_mva_outliers[_df_with_mva_outliers['is_mva_outlier'] == True].copy()

    if _outlier_group_df_cache.empty:
        _show_simple_modal_mva("Info", "No outliers detected in the dataset to perform group analysis.")
        _clear_group_analysis_ui_elements() # UI 초기화
        return
    if _normal_group_df_cache.empty:
        _show_simple_modal_mva("Info", "No normal data points found to perform group analysis (all points are outliers?).")
        _clear_group_analysis_ui_elements()
        return

    current_df_context = _df_with_mva_outliers # 전체 데이터셋 컨텍스트

    # 분석에 사용된 컬럼 또는 전체 사용 가능한 컬럼으로 확장 가능
    # 여기서는 _mva_cols_analyzed_for_if (MVA 탐지에 사용된 컬럼) 기준으로 분석
    cols_for_group_analysis = _mva_cols_analyzed_for_if if _mva_cols_analyzed_for_if else \
                              [col for col in current_df_context.columns if col != 'is_mva_outlier']


    _mva_group_numerical_cols_for_plot = [col for col in cols_for_group_analysis if _get_feature_type(col, current_df_context) == "Numeric"]
    _mva_group_categorical_cols_for_plot = [col for col in cols_for_group_analysis if _get_feature_type(col, current_df_context) == "Categorical"]

    _populate_group_stats_table(_normal_group_df_cache, _outlier_group_df_cache, _mva_group_numerical_cols_for_plot)

    if dpg.does_item_exist(TAG_OT_MVA_GROUP_NUM_FEATURE_COMBO):
        dpg.configure_item(TAG_OT_MVA_GROUP_NUM_FEATURE_COMBO, items=_mva_group_numerical_cols_for_plot, default_value="")
    if dpg.does_item_exist(TAG_OT_MVA_GROUP_CAT_FEATURE_COMBO):
        dpg.configure_item(TAG_OT_MVA_GROUP_CAT_FEATURE_COMBO, items=_mva_group_categorical_cols_for_plot, default_value="")
    
    _clear_group_dist_plot() # 이전 플롯 초기화
    _clear_group_freq_plot() # 이전 플롯 초기화
    _log_mva("Group analysis complete. Stats table and feature selectors populated.")


def _populate_group_stats_table(normal_df: pd.DataFrame, outlier_df: pd.DataFrame, num_cols: List[str]):
    if not dpg.does_item_exist(TAG_OT_MVA_GROUP_STATS_TABLE): return
    dpg.delete_item(TAG_OT_MVA_GROUP_STATS_TABLE, children_only=True)

    if not num_cols:
        with dpg.table_row(parent=TAG_OT_MVA_GROUP_STATS_TABLE):
            dpg.add_text("No numerical features to compare.") 
            dpg.add_text("") 
            dpg.add_text("") 
            dpg.add_text("") 
        return

    stats_to_calc = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    
    # 테이블의 실제 컬럼 수를 가져옵니다.
    num_defined_columns = 0
    if dpg.does_item_exist(TAG_OT_MVA_GROUP_STATS_TABLE):
        table_children_slot_1 = dpg.get_item_children(TAG_OT_MVA_GROUP_STATS_TABLE, 1)
        if table_children_slot_1: # slot 1에 자식이 있는지 확인
             num_defined_columns = len(table_children_slot_1)
    
    if num_defined_columns == 0 : # 컬럼이 정의되지 않은 경우 (예외 처리)
        _log_mva("Error: Group stats table has no columns defined.")
        return


    for col_name in num_cols:
        normal_stats = normal_df[col_name].describe().reindex(stats_to_calc)
        outlier_stats = outlier_df[col_name].describe().reindex(stats_to_calc)

        for i, stat_name in enumerate(stats_to_calc):
            with dpg.table_row(parent=TAG_OT_MVA_GROUP_STATS_TABLE):
                if i == 0: 
                    dpg.add_text(col_name, color=(200,200,0))
                else:
                    dpg.add_text("") 
                
                dpg.add_text(stat_name) 
                dpg.add_text(f"{normal_stats.get(stat_name, np.nan):.3f}")
                dpg.add_text(f"{outlier_stats.get(stat_name, np.nan):.3f}")
        
        if num_cols.index(col_name) < len(num_cols) - 1: 
            with dpg.table_row(parent=TAG_OT_MVA_GROUP_STATS_TABLE):
                # 수정된 부분: 테이블의 실제 컬럼 수만큼 반복하여 구분선 추가
                for _ in range(num_defined_columns): 
                    dpg.add_separator()

def _on_group_num_feature_select(sender, selected_feature: str, user_data):
    global _mva_group_dist_plot_texture_id
    if not selected_feature or _normal_group_df_cache is None or _outlier_group_df_cache is None:
        _clear_group_dist_plot(); return
    _log_mva(f"Numerical feature selected for group dist plot: {selected_feature}")

    fig, ax = plt.subplots(figsize=( (GROUP_ANALYSIS_PLOT_WIDTH-20)/80 , GROUP_ANALYSIS_PLOT_HEIGHT/100 )) # DPI 고려
    sns.boxplot(data=[_normal_group_df_cache[selected_feature].dropna(), 
                       _outlier_group_df_cache[selected_feature].dropna()], 
                ax=ax, notch=True, showmeans=True, meanline=True,
                palette=['#1f77b4', '#d62728'], flierprops={'markerfacecolor':'gray', 'markeredgecolor':'gray', 'markersize':3, 'alpha':0.5}) # 이상치 마커 스타일
    ax.set_xticklabels(['Normal', 'Outlier'])
    ax.set_title(f"Distribution: {selected_feature}", fontsize=10)
    ax.set_ylabel("Value", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()

    plot_texture_func = _shared_utils_mva['plot_to_dpg_texture_func']
    texture_tag_new, tex_w, tex_h = plot_texture_func(fig); plt.close(fig)

    default_texture = _shared_utils_mva.get("default_group_dist_plot_texture_tag", "")
    if _mva_group_dist_plot_texture_id and _mva_group_dist_plot_texture_id != default_texture and dpg.does_item_exist(_mva_group_dist_plot_texture_id):
        dpg.delete_item(_mva_group_dist_plot_texture_id)
    
    if texture_tag_new and tex_w > 0 and tex_h > 0:
        _mva_group_dist_plot_texture_id = texture_tag_new
        disp_h = int(tex_h * (GROUP_ANALYSIS_PLOT_WIDTH / tex_w)) if tex_w > 0 else tex_h
        dpg.configure_item(TAG_OT_MVA_GROUP_DIST_PLOT_IMAGE, texture_tag=_mva_group_dist_plot_texture_id, width=GROUP_ANALYSIS_PLOT_WIDTH, height=disp_h, show=True)
        if dpg.does_item_exist(TAG_OT_MVA_GROUP_DIST_PLOT_PLACEHOLDER): dpg.configure_item(TAG_OT_MVA_GROUP_DIST_PLOT_PLACEHOLDER, show=False)
    else: _clear_group_dist_plot()


def _on_group_cat_feature_select(sender, selected_feature: str, user_data):
    global _mva_group_freq_plot_texture_id
    if not selected_feature or _normal_group_df_cache is None or _outlier_group_df_cache is None:
        _clear_group_freq_plot(); return
    _log_mva(f"Categorical feature selected for group freq plot: {selected_feature}")

    normal_counts = _normal_group_df_cache[selected_feature].value_counts(normalize=True, dropna=False) * 100
    outlier_counts = _outlier_group_df_cache[selected_feature].value_counts(normalize=True, dropna=False) * 100
    
    df_plot = pd.DataFrame({'Normal (%)': normal_counts, 'Outlier (%)': outlier_counts}).fillna(0).sort_index()

    fig, ax = plt.subplots(figsize=( (GROUP_ANALYSIS_PLOT_WIDTH-20)/80 , GROUP_ANALYSIS_PLOT_HEIGHT/100 ))
    df_plot.plot(kind='bar', ax=ax, color=['#1f77b4', '#d62728'], width=0.8)
    ax.set_title(f"Frequency: {selected_feature}", fontsize=10)
    ax.set_ylabel("Percentage (%)", fontsize=9)
    ax.set_xlabel("Category", fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8, ha='right')
    ax.tick_params(axis='y', labelsize=8)
    ax.legend(fontsize=8)
    plt.tight_layout()

    plot_texture_func = _shared_utils_mva['plot_to_dpg_texture_func']
    texture_tag_new, tex_w, tex_h = plot_texture_func(fig); plt.close(fig)

    default_texture = _shared_utils_mva.get("default_group_freq_plot_texture_tag", "")
    if _mva_group_freq_plot_texture_id and _mva_group_freq_plot_texture_id != default_texture and dpg.does_item_exist(_mva_group_freq_plot_texture_id):
        dpg.delete_item(_mva_group_freq_plot_texture_id)

    if texture_tag_new and tex_w > 0 and tex_h > 0:
        _mva_group_freq_plot_texture_id = texture_tag_new
        disp_h = int(tex_h * (GROUP_ANALYSIS_PLOT_WIDTH / tex_w)) if tex_w > 0 else tex_h
        dpg.configure_item(TAG_OT_MVA_GROUP_FREQ_PLOT_IMAGE, texture_tag=_mva_group_freq_plot_texture_id, width=GROUP_ANALYSIS_PLOT_WIDTH, height=disp_h, show=True)
        if dpg.does_item_exist(TAG_OT_MVA_GROUP_FREQ_PLOT_PLACEHOLDER): dpg.configure_item(TAG_OT_MVA_GROUP_FREQ_PLOT_PLACEHOLDER, show=False)
    else: _clear_group_freq_plot()

def _clear_group_dist_plot():
    global _mva_group_dist_plot_texture_id
    default_tag = _shared_utils_mva.get("default_group_dist_plot_texture_tag", "") if _shared_utils_mva else ""
    if _mva_group_dist_plot_texture_id and _mva_group_dist_plot_texture_id != default_tag and dpg.does_item_exist(_mva_group_dist_plot_texture_id):
        dpg.delete_item(_mva_group_dist_plot_texture_id)
    _mva_group_dist_plot_texture_id = default_tag
    if dpg.does_item_exist(TAG_OT_MVA_GROUP_DIST_PLOT_IMAGE):
        dpg.configure_item(TAG_OT_MVA_GROUP_DIST_PLOT_IMAGE, texture_tag=default_tag or "", show=False)
    if dpg.does_item_exist(TAG_OT_MVA_GROUP_DIST_PLOT_PLACEHOLDER):
        dpg.configure_item(TAG_OT_MVA_GROUP_DIST_PLOT_PLACEHOLDER, show=True)
        dpg.set_value(TAG_OT_MVA_GROUP_DIST_PLOT_PLACEHOLDER, "Select a numerical feature.")

def _clear_group_freq_plot():
    global _mva_group_freq_plot_texture_id
    default_tag = _shared_utils_mva.get("default_group_freq_plot_texture_tag", "") if _shared_utils_mva else ""
    if _mva_group_freq_plot_texture_id and _mva_group_freq_plot_texture_id != default_tag and dpg.does_item_exist(_mva_group_freq_plot_texture_id):
        dpg.delete_item(_mva_group_freq_plot_texture_id)
    _mva_group_freq_plot_texture_id = default_tag
    if dpg.does_item_exist(TAG_OT_MVA_GROUP_FREQ_PLOT_IMAGE):
        dpg.configure_item(TAG_OT_MVA_GROUP_FREQ_PLOT_IMAGE, texture_tag=default_tag or "", show=False)
    if dpg.does_item_exist(TAG_OT_MVA_GROUP_FREQ_PLOT_PLACEHOLDER):
        dpg.configure_item(TAG_OT_MVA_GROUP_FREQ_PLOT_PLACEHOLDER, show=True)
        dpg.set_value(TAG_OT_MVA_GROUP_FREQ_PLOT_PLACEHOLDER, "Select a categorical feature.")

def _clear_group_analysis_ui_elements():
    if dpg.does_item_exist(TAG_OT_MVA_GROUP_STATS_TABLE):
        dpg.delete_item(TAG_OT_MVA_GROUP_STATS_TABLE, children_only=True)
        with dpg.table_row(parent=TAG_OT_MVA_GROUP_STATS_TABLE): # Placeholder
             for i in range(4): dpg.add_text("...")
    if dpg.does_item_exist(TAG_OT_MVA_GROUP_NUM_FEATURE_COMBO): dpg.configure_item(TAG_OT_MVA_GROUP_NUM_FEATURE_COMBO, items=[], default_value="")
    if dpg.does_item_exist(TAG_OT_MVA_GROUP_CAT_FEATURE_COMBO): dpg.configure_item(TAG_OT_MVA_GROUP_CAT_FEATURE_COMBO, items=[], default_value="")
    _clear_group_dist_plot()
    _clear_group_freq_plot()


# --- Main UI Creation and Update Functions for Multivariate ---
def create_multivariate_ui(parent_tab_bar_tag: str, shared_utilities: dict):
    global _shared_utils_mva, _mva_shap_plot_active_texture_id
    global _mva_group_dist_plot_texture_id, _mva_group_freq_plot_texture_id # 전역변수 선언
    _shared_utils_mva = shared_utilities
    _mva_shap_plot_active_texture_id = _shared_utils_mva.get("default_shap_plot_texture_tag", "")
    _mva_group_dist_plot_texture_id = _shared_utils_mva.get("default_group_dist_plot_texture_tag", "")
    _mva_group_freq_plot_texture_id = _shared_utils_mva.get("default_group_freq_plot_texture_tag", "")

    # ... (shap, umap import checks) ...

    with dpg.tab(label="Multivariate Outlier Detection", tag=TAG_OT_MULTIVARIATE_TAB, parent=parent_tab_bar_tag):
        with dpg.group(horizontal=True): 
            left_panel_width = - (SELECTED_POINT_INFO_GROUP_WIDTH + 20) 
            with dpg.child_window(width=left_panel_width, border=False, tag="mva_left_panel_outer_child"):
                # ... (왼쪽 패널 상단 UI: 설정, UMAP 플롯 등) ...
                dpg.add_text("1. Configure & Run Multivariate Outlier Detection", color=[255, 255, 0])
                dpg.add_text("Detection Method: Isolation Forest (on selected numeric columns)")
                with dpg.group(horizontal=True):
                    dpg.add_text("Contamination ('auto' or 0.0001-0.5):", tag=TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT + "_label")
                    dpg.add_input_text(tag=TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT, width=120, default_value=str(_mva_iso_forest_contamination), hint="e.g., 0.1 or auto", callback=_on_mva_iso_forest_contam_change,
                                       on_enter=True)
                dpg.add_text("Select Variables for Multivariate Analysis:")
                dpg.add_radio_button(items=["All Numeric Columns", "Recommended Columns (TODO)", "Select Custom Columns"], tag=TAG_OT_MVA_VAR_METHOD_RADIO, default_value=_mva_variable_selection_method, horizontal=True, callback=_on_mva_var_method_change)
                dpg.add_text("Custom Numeric Columns (if selected):", tag=TAG_OT_MVA_CUSTOM_COLS_TABLE_LABEL, show=False)
                with dpg.child_window(tag=TAG_OT_MVA_CUSTOM_COLS_TABLE_CHILD, show=False, height=100, border=True): 
                    with dpg.table(tag=TAG_OT_MVA_CUSTOM_COLS_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                        dpg.add_table_column(label="Select", parent=TAG_OT_MVA_CUSTOM_COLS_TABLE, width_fixed=True, init_width_or_weight=70)
                        dpg.add_table_column(label="Column Name", parent=TAG_OT_MVA_CUSTOM_COLS_TABLE, width_stretch=True)
                dpg.add_button(label="Run Multivariate Detection", tag=TAG_OT_MVA_DETECT_BUTTON, width=-1, height=30, callback=_run_mva_outlier_detection_logic)
                dpg.add_spacer(height=5)
                dpg.add_text("2. Multivariate Detection Summary & Visualization", color=[255, 255, 0])
                dpg.add_text("Summary will appear here.", tag=TAG_OT_MVA_RESULTS_TEXT, wrap=-1)
                dpg.add_spacer(height=5)
                dpg.add_text("UMAP Visualization (Click a point to see details):")
                with dpg.child_window(tag=TAG_OT_MVA_UMAP_PLOT_WINDOW, height=UMAP_PLOT_WINDOW_HEIGHT, border=True):
                    with dpg.plot(tag=TAG_OT_MVA_UMAP_PLOT, label="UMAP Projection", height=-1, width=-1, no_title=True): 
                        dpg.add_plot_legend(tag=TAG_OT_MVA_UMAP_LEGEND, show=False)
                        dpg.add_plot_axis(dpg.mvXAxis, label="UMAP 1", tag=TAG_OT_MVA_UMAP_X_AXIS)
                        dpg.add_plot_axis(dpg.mvYAxis, label="UMAP 2", tag=TAG_OT_MVA_UMAP_Y_AXIS)
                _clear_dpg_umap_plot_series()

                # --- Outlier Group Analysis Section ---
                dpg.add_spacer(height=10)
                with dpg.collapsing_header(label="4. Outlier Group vs. Normal Group Analysis", tag=TAG_OT_MVA_GROUP_ANALYSIS_COLLAPSING_HEADER, default_open=False):
                    dpg.add_button(label="Run Group Analysis", tag=TAG_OT_MVA_RUN_GROUP_ANALYSIS_BUTTON, callback=_run_group_analysis, width=-1)
                    dpg.add_text("Comparative Descriptive Statistics (Numerical Features):")
                    with dpg.child_window(height=180, border=True): # 테이블 높이
                        with dpg.table(tag=TAG_OT_MVA_GROUP_STATS_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, borders_outerH=True, borders_innerV=True):
                            dpg.add_table_column(label="Feature", parent=TAG_OT_MVA_GROUP_STATS_TABLE, init_width_or_weight=0.3)
                            dpg.add_table_column(label="Statistic", parent=TAG_OT_MVA_GROUP_STATS_TABLE, init_width_or_weight=0.2)
                            dpg.add_table_column(label="Normal Group", parent=TAG_OT_MVA_GROUP_STATS_TABLE, init_width_or_weight=0.25)
                            dpg.add_table_column(label="Outlier Group", parent=TAG_OT_MVA_GROUP_STATS_TABLE, init_width_or_weight=0.25)
                    
                    dpg.add_spacer(height=5)
                    dpg.add_text("Numerical Feature Distribution Comparison (Box Plot):")
                    dpg.add_combo(items=[], tag=TAG_OT_MVA_GROUP_NUM_FEATURE_COMBO, width=-1, callback=_on_group_num_feature_select, default_value="", no_preview=True)
                    dpg.add_text("Select a numerical feature.", tag=TAG_OT_MVA_GROUP_DIST_PLOT_PLACEHOLDER)
                    dpg.add_image(texture_tag=_shared_utils_mva.get("default_group_dist_plot_texture_tag",""), tag=TAG_OT_MVA_GROUP_DIST_PLOT_IMAGE, show=False, width=GROUP_ANALYSIS_PLOT_WIDTH)

                    dpg.add_spacer(height=5)
                    dpg.add_text("Categorical Feature Frequency Comparison (Bar Chart):")
                    dpg.add_combo(items=[], tag=TAG_OT_MVA_GROUP_CAT_FEATURE_COMBO, width=-1, callback=_on_group_cat_feature_select, default_value="", no_preview=True)
                    dpg.add_text("Select a categorical feature.", tag=TAG_OT_MVA_GROUP_FREQ_PLOT_PLACEHOLDER)
                    dpg.add_image(texture_tag=_shared_utils_mva.get("default_group_freq_plot_texture_tag",""), tag=TAG_OT_MVA_GROUP_FREQ_PLOT_IMAGE, show=False, width=GROUP_ANALYSIS_PLOT_WIDTH)


            with dpg.child_window(width=SELECTED_POINT_INFO_GROUP_WIDTH, border=False, tag=TAG_OT_MVA_SELECTED_OUTLIER_INFO_GROUP):
                dpg.add_text("3. Selected Point Details & Feature Contributions", color=[255, 255, 0])
                dpg.add_text("Selected Point: (None - Click a point on UMAP)", tag=TAG_OT_MVA_SELECTED_OUTLIER_INDEX_TEXT, wrap=-1)
                dpg.add_text("Original Values & Context:")
                with dpg.child_window(tag=TAG_OT_MVA_SELECTED_OUTLIER_TABLE_CHILD, height=SELECTED_POINT_TABLE_CHILD_HEIGHT, border=True): 
                    with dpg.table(tag=TAG_OT_MVA_SELECTED_OUTLIER_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                        dpg.add_table_column(label="Feature", parent=TAG_OT_MVA_SELECTED_OUTLIER_TABLE, init_width_or_weight=0.30)
                        dpg.add_table_column(label="Value", parent=TAG_OT_MVA_SELECTED_OUTLIER_TABLE, init_width_or_weight=0.15)
                        dpg.add_table_column(label="Normal Range (5-95%)", parent=TAG_OT_MVA_SELECTED_OUTLIER_TABLE, init_width_or_weight=0.25)
                        dpg.add_table_column(label="Z-Score", parent=TAG_OT_MVA_SELECTED_OUTLIER_TABLE, init_width_or_weight=0.15)
                        dpg.add_table_column(label="Normal Freq. (%)", parent=TAG_OT_MVA_SELECTED_OUTLIER_TABLE, init_width_or_weight=0.15)
                
                dpg.add_spacer(height=10)
                dpg.add_text("SHAP Value Contributions (to anomaly score):")
                # ... (SHAP 이미지 및 플레이스홀더 UI는 이전과 동일) ...
                default_shap_texture_tag_val = _shared_utils_mva.get("default_shap_plot_texture_tag", "")
                init_w_shap, init_h_shap = 10,10 
                if default_shap_texture_tag_val and dpg.does_item_exist(default_shap_texture_tag_val):
                    cfg = dpg.get_item_configuration(default_shap_texture_tag_val)
                    init_w_shap = cfg.get('width', init_w_shap); init_h_shap = cfg.get('height', init_h_shap)
                dpg.add_text("SHAP plot will appear here.", tag=TAG_OT_MVA_SHAP_PLOT_PLACEHOLDER_TEXT)
                dpg.add_image(texture_tag=default_shap_texture_tag_val, tag=TAG_OT_MVA_SHAP_PLOT_IMAGE, show=False, width=SHAP_PLOT_IMAGE_WIDTH)

        _clear_selected_point_info()
        _clear_group_analysis_ui_elements() # 그룹 분석 UI도 초기화

    if not dpg.does_item_exist(GLOBAL_MOUSE_HANDLER_REGISTRY_TAG):
        with dpg.handler_registry(tag=GLOBAL_MOUSE_HANDLER_REGISTRY_TAG):
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=_on_mva_umap_plot_click)
            _log_mva(f"Global mouse click handler registry '{GLOBAL_MOUSE_HANDLER_REGISTRY_TAG}' created.")


def update_multivariate_ui(df_input: Optional[pd.DataFrame], shared_utilities: dict, is_new_data: bool):
    global _shared_utils_mva, _s1_column_types_cache # 캐시 사용 명시
    if not dpg.is_dearpygui_running(): return
    _shared_utils_mva = shared_utilities
    if not dpg.does_item_exist(TAG_OT_MULTIVARIATE_TAB): return
    current_df_for_mva = df_input
    if current_df_for_mva is None or is_new_data:
        _s1_column_types_cache = None # 새 데이터 시 S1 타입 캐시 초기화
        reset_multivariate_state_internal(called_from_parent_reset=False)
        if current_df_for_mva is not None:
            _log_mva("New data for MVA. Re-configure & run detection.")
            _populate_mva_custom_cols_table(current_df_for_mva)
    _update_mva_custom_cols_ui_visibility()


def reset_multivariate_state_internal(called_from_parent_reset=True):
    global _mva_variable_selection_method, _mva_custom_selected_columns, _mva_custom_col_checkbox_tags
    global _mva_iso_forest_contamination, _df_with_mva_outliers, _mva_outlier_row_indices
    global _mva_iso_forest_model, _mva_cols_analyzed_for_if, _mva_umap_embedding, _mva_outlier_scores
    global _mva_selected_point_original_index, _mva_shap_plot_active_texture_id, _mva_umap_original_indices
    global _s1_column_types_cache # S1 타입 캐시 초기화
    global _mva_group_stats_df, _mva_group_numerical_cols_for_plot, _mva_group_categorical_cols_for_plot
    global _mva_group_dist_plot_texture_id, _mva_group_freq_plot_texture_id
    global _normal_group_df_cache, _outlier_group_df_cache


    _s1_column_types_cache = None # S1 타입 캐시 초기화
    _mva_variable_selection_method = "All Numeric Columns"
    # ... (기존 상태 변수 초기화)
    _mva_custom_selected_columns.clear(); _mva_custom_col_checkbox_tags.clear()
    _mva_iso_forest_contamination = DEFAULT_MVA_ISO_FOREST_CONTAMINATION
    _df_with_mva_outliers = None; _mva_outlier_row_indices = None
    _mva_iso_forest_model = None; _mva_cols_analyzed_for_if = []
    _mva_umap_embedding = None; _mva_outlier_scores = None; _mva_umap_original_indices = None
    
    if _shared_utils_mva: # shared_utils가 있을 때만 default tag 접근
        _mva_shap_plot_active_texture_id = _shared_utils_mva.get("default_shap_plot_texture_tag", "")
        _mva_group_dist_plot_texture_id = _shared_utils_mva.get("default_group_dist_plot_texture_tag", "")
        _mva_group_freq_plot_texture_id = _shared_utils_mva.get("default_group_freq_plot_texture_tag", "")
    else:
        _mva_shap_plot_active_texture_id = ""
        _mva_group_dist_plot_texture_id = ""
        _mva_group_freq_plot_texture_id = ""


    # 그룹 분석 관련 상태 초기화
    _mva_group_stats_df = None
    _mva_group_numerical_cols_for_plot = []
    _mva_group_categorical_cols_for_plot = []
    _normal_group_df_cache = None
    _outlier_group_df_cache = None


    if dpg.is_dearpygui_running():
        # ... (기존 UI 초기화)
        if dpg.does_item_exist(TAG_OT_MVA_VAR_METHOD_RADIO): dpg.set_value(TAG_OT_MVA_VAR_METHOD_RADIO, _mva_variable_selection_method)
        if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE): _populate_mva_custom_cols_table(None)
        _update_mva_custom_cols_ui_visibility()
        if dpg.does_item_exist(TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT): dpg.set_value(TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT, str(_mva_iso_forest_contamination))
        if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, "Run multivariate detection.")
        _clear_dpg_umap_plot_series()
        _clear_selected_point_info() 
        _clear_group_analysis_ui_elements() # 그룹 분석 UI 요소들 초기화
        
    if not called_from_parent_reset: _log_mva("MVA state reset due to data change.")


# reset_multivariate_state, get_multivariate_settings, apply_multivariate_settings 함수는 이전과 동일
def reset_multivariate_state(): reset_multivariate_state_internal(True); _log_mva("MVA state reset by parent.")
def get_multivariate_settings() -> dict:
    return {"mva_variable_selection_method": _mva_variable_selection_method, "mva_custom_selected_columns": _mva_custom_selected_columns[:], "mva_iso_forest_contamination": _mva_iso_forest_contamination}
def apply_multivariate_settings(df_input: Optional[pd.DataFrame], settings: dict, shared_utilities: dict):
    global _mva_variable_selection_method, _mva_custom_selected_columns, _mva_iso_forest_contamination, _shared_utils_mva, _s1_column_types_cache
    _shared_utils_mva = shared_utilities
    _s1_column_types_cache = None # 설정 적용 시 S1 타입 캐시도 초기화 (새 데이터 로드 가정)
    _mva_variable_selection_method = settings.get("mva_variable_selection_method", "All Numeric Columns")
    _mva_custom_selected_columns = settings.get("mva_custom_selected_columns", [])[:]
    _mva_iso_forest_contamination = settings.get("mva_iso_forest_contamination", DEFAULT_MVA_ISO_FOREST_CONTAMINATION)
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_MVA_VAR_METHOD_RADIO): dpg.set_value(TAG_OT_MVA_VAR_METHOD_RADIO, _mva_variable_selection_method)
        if dpg.does_item_exist(TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT): dpg.set_value(TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT, str(_mva_iso_forest_contamination))
        if df_input is not None: _populate_mva_custom_cols_table(df_input)
        else: _populate_mva_custom_cols_table(None)
        _update_mva_custom_cols_ui_visibility()
        if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, "Settings applied. Run detection.")
        _clear_dpg_umap_plot_series(); _clear_selected_point_info(); _clear_group_analysis_ui_elements()
    _log_mva("MVA settings applied from saved state.")