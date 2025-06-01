# step_05b_multivariate_outliers.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
# UMAP 임포트는 나중에 DPG 네이티브 플롯 구현 시 필요
# from umap import UMAP 

# traceback, io, PIL 등은 _s5_plot_to_dpg_texture_parent 함수가 처리

# --- DPG Tags for Multivariate Tab ---
TAG_OT_MULTIVARIATE_TAB = "step5_multivariate_outlier_tab"
TAG_OT_MVA_VAR_METHOD_RADIO = "step5_ot_mva_var_method_radio"
TAG_OT_MVA_CUSTOM_COLS_TABLE = "step5_ot_mva_custom_cols_table"
TAG_OT_MVA_CUSTOM_COLS_TABLE_CHILD = TAG_OT_MVA_CUSTOM_COLS_TABLE + "_child" # 자식창 태그
TAG_OT_MVA_CUSTOM_COLS_TABLE_LABEL = TAG_OT_MVA_CUSTOM_COLS_TABLE + "_label" # 레이블 태그

TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT = "step5_ot_mva_iso_forest_contam_input"
TAG_OT_MVA_DETECT_BUTTON = "step5_ot_mva_detect_button"
TAG_OT_MVA_RESULTS_TEXT = "step5_ot_mva_results_text"
TAG_OT_MVA_VISUALIZATION_IMAGE = "step5_ot_mva_visualization_image"
TAG_OT_MVA_VISUALIZATION_PARENT_GROUP = "mva_visualization_parent_group" # 시각화 부모 그룹
# TAG_OT_DEFAULT_PLOT_TEXTURE_MVA는 부모로부터 태그 문자열을 받아서 사용


# --- Constants ---
DEFAULT_MVA_ISO_FOREST_CONTAMINATION = 'auto'


# --- Module State Variables (Multivariate) ---
_shared_utils_mva: Optional[Dict[str, Any]] = None

_mva_variable_selection_method: str = "All Numeric" # 초기값: "All Numeric Columns"
_mva_custom_selected_columns: List[str] = []
_mva_custom_col_checkbox_tags: Dict[str, str] = {} # Checkbox tag 저장 (str로 변경)
_mva_iso_forest_contamination: Union[str, float] = DEFAULT_MVA_ISO_FOREST_CONTAMINATION
_df_with_mva_outliers: Optional[pd.DataFrame] = None # MVA 탐지 결과 (플래그 포함)
_mva_outlier_row_indices: Optional[np.ndarray] = None
_mva_active_plot_texture_id: Optional[str] = None # 부모의 TAG_OT_DEFAULT_PLOT_TEXTURE_MVA로 초기화될 것
_mva_umap_embedding: Optional[np.ndarray] = None # UMAP 결과 저장 (DPG 네이티브 플롯용)
_mva_outlier_scores: Optional[np.ndarray] = None # Isolation Forest 스코어 저장


# --- Helper Functions ---
def _log_mva(message: str):
    if _shared_utils_mva and 'log_message_func' in _shared_utils_mva:
        _shared_utils_mva['log_message_func'](f"[MVAOutlier] {message}")

def _show_simple_modal_mva(title: str, message: str, width: int = 450, height: int = 200):
    if _shared_utils_mva and 'util_funcs_common' in _shared_utils_mva and \
       '_show_simple_modal_message' in _shared_utils_mva['util_funcs_common']:
        _shared_utils_mva['util_funcs_common']['_show_simple_modal_message'](title, message, width, height)
    else:
        _log_mva(f"Modal display function not available. Title: {title}, Msg: {message}")

# --- Multivariate Callbacks and Logic ---

def _update_mva_custom_cols_ui_visibility():
    """MVA 사용자 지정 컬럼 테이블 및 레이블의 가시성을 업데이트합니다."""
    if not dpg.is_dearpygui_running(): return
    show_custom_table = (_mva_variable_selection_method == "Select Custom Columns") # 선택지 이름과 일치
    if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE_CHILD):
        dpg.configure_item(TAG_OT_MVA_CUSTOM_COLS_TABLE_CHILD, show=show_custom_table)
    if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE_LABEL):
        dpg.configure_item(TAG_OT_MVA_CUSTOM_COLS_TABLE_LABEL, show=show_custom_table)


def _on_mva_var_method_change(sender, app_data: str, user_data):
    global _mva_variable_selection_method
    _mva_variable_selection_method = app_data
    _log_mva(f"Multivariate variable selection method: {_mva_variable_selection_method}")
    _update_mva_custom_cols_ui_visibility() # 가시성 업데이트 호출
    
    # "Custom" 선택 시 테이블 채우기 (현재 DF가 있다면)
    if _mva_variable_selection_method == "Select Custom Columns":
        current_df = _shared_utils_mva['get_current_df_func']() if _shared_utils_mva else None
        if current_df is not None:
            _populate_mva_custom_cols_table(current_df)


def _on_mva_custom_col_checkbox_change(sender, app_data_is_checked: bool, user_data_col_name: str):
    global _mva_custom_selected_columns
    if app_data_is_checked:
        if user_data_col_name not in _mva_custom_selected_columns:
            _mva_custom_selected_columns.append(user_data_col_name)
    else:
        if user_data_col_name in _mva_custom_selected_columns:
            _mva_custom_selected_columns.remove(user_data_col_name)
    _log_mva(f"MVA custom columns updated: {_mva_custom_selected_columns}")


def _populate_mva_custom_cols_table(df: Optional[pd.DataFrame]):
    global _mva_custom_col_checkbox_tags, _mva_custom_selected_columns
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE): return
    
    dpg.delete_item(TAG_OT_MVA_CUSTOM_COLS_TABLE, children_only=True) # 테이블 내부만 삭제
    _mva_custom_col_checkbox_tags.clear()

    if df is None:
        with dpg.table_row(parent=TAG_OT_MVA_CUSTOM_COLS_TABLE): # 부모는 테이블 태그
            dpg.add_text("No data to select columns from.")
        return

    # 수치형 컬럼 필터링 시 S1 타입 고려 (utils._get_numeric_cols와 유사하게)
    s1_col_types = {}
    if _shared_utils_mva and 'main_app_callbacks' in _shared_utils_mva and \
       'get_column_analysis_types' in _shared_utils_mva['main_app_callbacks']:
        s1_col_types = _shared_utils_mva['main_app_callbacks']['get_column_analysis_types']()

    numeric_cols_for_mva = []
    for col in df.columns:
        is_numeric_s1 = "Numeric" in s1_col_types.get(col, "") # S1에서 Numeric으로 지정
        is_numeric_pandas = pd.api.types.is_numeric_dtype(df[col])
        # Binary는 보통 MVA에서 제외하나, 포함하고 싶다면 조건 수정
        is_binary_s1 = "Binary" in s1_col_types.get(col,"") 
        
        if (is_numeric_s1 and not is_binary_s1) or (is_numeric_pandas and df[col].nunique() > 2) : # 고유값 2 초과 수치형
             numeric_cols_for_mva.append(col)
             
    if not numeric_cols_for_mva:
        with dpg.table_row(parent=TAG_OT_MVA_CUSTOM_COLS_TABLE):
            dpg.add_text("No suitable numeric columns available for MVA.")
        return
    
    # 테이블 헤더는 한 번만 추가 (create_ui에서)
    # 여기서는 row만 추가/삭제

    for col_name in numeric_cols_for_mva:
        with dpg.table_row(parent=TAG_OT_MVA_CUSTOM_COLS_TABLE):
            # 태그 고유성을 위해 UUID 대신 컬럼 이름 사용 (충돌 가능성 낮음)
            checkbox_tag = f"mva_cb_{''.join(filter(str.isalnum, col_name))}"
            _mva_custom_col_checkbox_tags[col_name] = checkbox_tag # 태그 저장
            is_selected = col_name in _mva_custom_selected_columns
            dpg.add_checkbox(tag=checkbox_tag, default_value=is_selected, user_data=col_name, callback=_on_mva_custom_col_checkbox_change)
            dpg.add_text(col_name)


def _on_mva_iso_forest_contam_change(sender, app_data_str: str, user_data):
    global _mva_iso_forest_contamination
    # ... (기존 로직과 동일, _log_message -> _log_mva)
    cleaned_app_data_str = app_data_str.strip()
    if not cleaned_app_data_str or cleaned_app_data_str.lower() == 'auto':
        _mva_iso_forest_contamination = 'auto'
        if dpg.get_value(sender) != "auto": dpg.set_value(sender, "auto") # UI 'auto'로 통일
        _log_mva("MVA Isolation Forest contamination set to: 'auto'")
    else:
        try:
            val = float(cleaned_app_data_str)
            if 0.0001 <= val <= 0.5: 
                _mva_iso_forest_contamination = val
                _log_mva(f"MVA Isolation Forest contamination set to: {val:.4f}")
            else: 
                _mva_iso_forest_contamination = 'auto'
                dpg.set_value(sender, "auto") # 범위 벗어나면 'auto'로
                _log_mva(f"MVA Contam value {val} out of range (0.0001-0.5). Using 'auto'.")
                _show_simple_modal_mva("Input Error", "Contamination must be 'auto' or a value between 0.0001 and 0.5.")
        except ValueError: 
            _mva_iso_forest_contamination = 'auto'
            dpg.set_value(sender, "auto") # 잘못된 입력이면 'auto'로
            _log_mva(f"Invalid float for MVA contam: '{cleaned_app_data_str}'. Using 'auto'.")
            _show_simple_modal_mva("Input Error", "Invalid input for contamination. Please enter 'auto' or a numeric value.")


def _get_mva_columns_to_analyze(current_df: pd.DataFrame) -> List[str]: # current_df를 인자로 받음
    # ... (기존 로직과 동일, _log_message -> _log_mva)
    # _current_df_for_this_step 대신 current_df 사용
    if current_df is None: return []
    
    s1_col_types = {}
    if _shared_utils_mva and 'main_app_callbacks' in _shared_utils_mva and \
       'get_column_analysis_types' in _shared_utils_mva['main_app_callbacks']:
        s1_col_types = _shared_utils_mva['main_app_callbacks']['get_column_analysis_types']()

    all_suitable_numeric_cols = []
    for col in current_df.columns:
        is_numeric_s1 = "Numeric" in s1_col_types.get(col, "")
        is_numeric_pandas = pd.api.types.is_numeric_dtype(current_df[col])
        is_binary_s1 = "Binary" in s1_col_types.get(col,"")
        
        if (is_numeric_s1 and not is_binary_s1) or \
           (is_numeric_pandas and current_df[col].nunique(dropna=False) > 2): # 고유값 2 초과 (Binary 제외)
             all_suitable_numeric_cols.append(col)

    if not all_suitable_numeric_cols:
        _log_mva("No suitable numeric columns found in the current dataset for MVA.")
        return []

    if _mva_variable_selection_method == "All Numeric Columns": # UI와 일치
        _log_mva(f"MVA using all {len(all_suitable_numeric_cols)} suitable numeric columns.")
        return all_suitable_numeric_cols
    elif _mva_variable_selection_method == "Recommended Columns (TODO)": # UI와 일치
        _log_mva("MVA recommended column selection is a TODO. Using all suitable numeric columns for now.")
        return all_suitable_numeric_cols # 현재는 전체 반환
    elif _mva_variable_selection_method == "Select Custom Columns": # UI와 일치
        valid_custom_cols = [col for col in _mva_custom_selected_columns if col in all_suitable_numeric_cols]
        if not valid_custom_cols:
             _log_mva("MVA custom selection: No valid numeric columns selected or available. Using all suitable numeric columns as fallback.")
             return all_suitable_numeric_cols # 선택된 유효 컬럼 없으면 전체 사용 (또는 에러 처리)
        _log_mva(f"MVA using {len(valid_custom_cols)} custom selected suitable numeric columns: {valid_custom_cols}")
        return valid_custom_cols
    return []


def _detect_outliers_mva_iso_forest(df_subset: pd.DataFrame, contamination_val: Union[str, float]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    global _mva_outlier_scores
    _mva_outlier_scores = None # 초기화
    # ... (기존 로직과 동일, _log_message -> _log_mva)
    # decision_function 결과도 반환하도록 수정
    if df_subset.empty or df_subset.shape[1] == 0: 
        _log_mva("MVA IsoForest: Input DataFrame subset is empty or has no columns."); return None, None
    
    df_numeric_imputed = df_subset.copy()
    # 결측치 처리: 각 컬럼의 중앙값으로 대체
    for col in df_numeric_imputed.columns:
        if pd.api.types.is_numeric_dtype(df_numeric_imputed[col]):
            if df_numeric_imputed[col].isnull().any():
                median_val = df_numeric_imputed[col].median()
                df_numeric_imputed[col].fillna(median_val, inplace=True)
                _log_mva(f"  MVA IsoForest: NaN in column '{col}' imputed with median ({median_val}).")
        else: # 이론상 여기까지 오면 안됨 (_get_mva_columns_to_analyze에서 필터링)
            _log_mva(f"  MVA IsoForest: Non-numeric column '{col}' encountered unexpectedly. Skipping imputation for it."); 
            # return None, None # 또는 해당 컬럼 제외하고 진행
    
    if df_numeric_imputed.shape[0] < 2 : 
        _log_mva("MVA IsoForest: Not enough samples after potential imputation (less than 2)."); return None, None

    # Contamination 값 유효성 검사 (숫자일 경우 범위 확인)
    valid_contamination = contamination_val
    if isinstance(contamination_val, str) and contamination_val.lower() != 'auto':
        try:
            cont_float = float(contamination_val)
            if not (0.0001 <= cont_float <= 0.5): valid_contamination = 'auto'
            else: valid_contamination = cont_float
        except ValueError: valid_contamination = 'auto'
    elif isinstance(contamination_val, (int, float)):
        if not (0.0001 <= contamination_val <= 0.5): valid_contamination = 'auto'


    model = IsolationForest(contamination=valid_contamination, random_state=42, n_estimators=100) # n_jobs=-1 고려 가능
    
    try:
        model.fit(df_numeric_imputed)
        predictions = model.predict(df_numeric_imputed) # -1 for outliers, 1 for inliers
        # decision_function: 양수일수록 정상, 음수일수록 이상치 (값이 작을수록 이상치)
        # score_samples: 부호 반대 (값이 작을수록 정상, 클수록 이상치) - scikit-learn 버전 확인
        # 여기서는 decision_function 사용 (일반적)
        scores = model.decision_function(df_numeric_imputed) 
        _mva_outlier_scores = scores # 스코어 저장
    except ValueError as e:
        _log_mva(f"  MVA IsoForest error with contamination '{valid_contamination}': {e}. Trying 'auto' if not already.")
        if valid_contamination != 'auto':
            try: 
                model_auto = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
                model_auto.fit(df_numeric_imputed)
                predictions = model_auto.predict(df_numeric_imputed)
                scores = model_auto.decision_function(df_numeric_imputed)
                _mva_outlier_scores = scores
                _log_mva("  MVA IsoForest successfully run with 'auto' contamination.")
            except Exception as e_auto: 
                _log_mva(f"  MVA IsoForest with 'auto' contamination also failed: {e_auto}"); return None, None
        else: # 이미 'auto'였으면 실패
            return None, None
            
    outlier_indices = df_numeric_imputed.index[predictions == -1].to_numpy()
    return outlier_indices, scores # scores도 반환 (현재는 _mva_outlier_scores 전역변수로 저장)


def _run_mva_outlier_detection_logic(sender, app_data, user_data):
    global _df_with_mva_outliers, _mva_outlier_row_indices, _mva_outlier_scores, _mva_umap_embedding
    
    _log_mva("Run Multivariate Outlier Detection button clicked.")
    # ... (기존 로직과 동일, _log_message -> _log_mva, _show_simple_modal_message -> _show_simple_modal_mva)
    # _current_df_for_this_step 대신 _shared_utils_mva['get_current_df_func']() 사용
    current_df = _shared_utils_mva['get_current_df_func']()
    if current_df is None:
        _log_mva("Error: No data loaded for MVA detection.")
        if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, "Error: No data available.")
        _show_simple_modal_mva("Error", "No data available for Multivariate Outlier Detection.")
        return

    cols_to_analyze = _get_mva_columns_to_analyze(current_df)
    if not cols_to_analyze or len(cols_to_analyze) < 2: # MVA는 보통 2개 이상 변수 필요
         _log_mva("MVA Error: Not enough suitable numeric columns selected or available for analysis (minimum 2 required).")
         if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, "Error: Select at least 2 suitable numeric columns.")
         _show_simple_modal_mva("Error", "Not enough suitable numeric columns for MVA (minimum 2 required).")
         return
    
    df_subset_for_mva = current_df[cols_to_analyze]
    _log_mva(f"--- Starting MVA Detection (Isolation Forest) on {len(cols_to_analyze)} columns ---")
    _log_mva(f"  Parameters: Contamination='{_mva_iso_forest_contamination}'")
    
    detected_indices, detected_scores = _detect_outliers_mva_iso_forest(df_subset_for_mva, _mva_iso_forest_contamination)
    _mva_outlier_row_indices = detected_indices # 전역변수에도 할당 (기존 코드 호환)
    # _mva_outlier_scores는 _detect_outliers_mva_iso_forest 내부에서 이미 전역변수에 할당됨

    if _mva_outlier_row_indices is None: # 탐지 실패
        _log_mva("MVA Outlier detection failed.")
        if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, "MVA detection failed. Check logs.")
        _show_simple_modal_mva("Detection Failed", "Multivariate outlier detection failed. Please check the logs for more details.")
        return

    num_outliers = len(_mva_outlier_row_indices)
    total_rows = len(current_df)
    percentage = (num_outliers / total_rows * 100) if total_rows > 0 else 0
    
    summary = f"MVA Detection Complete (Isolation Forest):\n" \
              f"- Columns Analyzed: {len(cols_to_analyze)}\n" \
              f"- Detected Outlier Rows: {num_outliers} ({percentage:.2f}% of total)\n" \
              f"- Contamination Setting Used: '{_mva_iso_forest_contamination}'" # 실제 사용된 contamination 값 표시 필요
    _log_mva(summary)
    if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, summary)
    
    _df_with_mva_outliers = current_df.copy()
    _df_with_mva_outliers['is_mva_outlier'] = False # MVA 결과 플래그 컬럼
    if num_outliers > 0:
        _df_with_mva_outliers.loc[_mva_outlier_row_indices, 'is_mva_outlier'] = True
    
    _generate_mva_plot_texture() # 시각화 생성
    _show_simple_modal_mva("MVA Detection Complete", f"Multivariate outlier detection finished. Found {num_outliers} potential outlier rows.")
    
    # 다변량 처리 적용 버튼은 아직 없으므로, main_app으로의 콜백은 여기서 발생하지 않음.
    # 단변량 처리와 별개로 MVA는 탐지만 수행.


def _generate_mva_plot_texture():
    global _mva_active_plot_texture_id, _mva_umap_embedding
    
    _log_mva("Generating MVA plot (UMAP)...")
    _clear_mva_visualization_plot() # 이전 플롯 정리
    
    current_df = _shared_utils_mva['get_current_df_func']()
    plot_texture_func = _shared_utils_mva['plot_to_dpg_texture_func']

    if _df_with_mva_outliers is None or 'is_mva_outlier' not in _df_with_mva_outliers.columns:
        _log_mva("MVA Plot Error: No MVA outlier detection data available (run detection first)."); return
    
    cols_for_plot = _get_mva_columns_to_analyze(current_df) # current_df 전달
    if not cols_for_plot or len(cols_for_plot) < 2: # UMAP은 최소 2개 변수 필요
        _log_mva("MVA Plot Error: Not enough columns selected or available for UMAP (minimum 2)."); return

    try:
        from umap import UMAP # 여기서 UMAP 임포트
    except ImportError:
        _log_mva("MVA Plot Error: UMAP library not found. Please install 'umap-learn' to visualize MVA results.")
        _show_simple_modal_mva("Library Missing", "UMAP library not found. Please install 'umap-learn' for MVA visualization.")
        return

    df_for_umap = current_df[cols_for_plot].copy()
    # UMAP을 위해 결측치 처리 (이미 _detect_outliers_mva_iso_forest에서 했을 수 있지만, 여기서 한 번 더 보장)
    for col in df_for_umap.columns:
        if df_for_umap[col].isnull().any():
            median_val = df_for_umap[col].median()
            df_for_umap[col].fillna(median_val, inplace=True)
            _log_mva(f"  UMAP Preprocessing: NaN in column '{col}' imputed with median ({median_val}).")
            
    if df_for_umap.shape[0] < 2: # UMAP은 최소 2개 샘플 필요
        _log_mva("MVA Plot Error: Not enough samples for UMAP (minimum 2)."); return
    
    # UMAP 파라미터 (n_neighbors는 샘플 수보다 작아야 함)
    n_neighbors_umap = min(15, df_for_umap.shape[0] - 1 if df_for_umap.shape[0] > 1 else 1)
    if n_neighbors_umap <= 0: n_neighbors_umap = 1 # n_neighbors는 0보다 커야 함

    try:
        reducer = UMAP(n_neighbors=n_neighbors_umap, n_components=2, min_dist=0.1, random_state=42, n_jobs=1) # n_jobs=1로 안정성 확보
        _mva_umap_embedding = reducer.fit_transform(df_for_umap) # 임베딩 결과 저장
        
        fig_width, fig_height = 15.6, 5.85 
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        hue_data = _df_with_mva_outliers['is_mva_outlier'] # is_mva_outlier 플래그 사용
        
        # 이상치 스코어에 따른 점 크기/색상 변화 (옵션)
        # score_norm = None
        # if _mva_outlier_scores is not None and len(_mva_outlier_scores) == len(_mva_umap_embedding):
        #     # 스코어를 0-1로 정규화 (값이 작을수록 이상치이므로, 1 - score 또는 -score 사용)
        #     # 예: score_norm = (-_mva_outlier_scores - np.min(-_mva_outlier_scores)) / (np.max(-_mva_outlier_scores) - np.min(-_mva_outlier_scores) + 1e-6)
        #     pass # 아직 미적용

        sns.scatterplot(
            x=_mva_umap_embedding[:, 0], 
            y=_mva_umap_embedding[:, 1], 
            hue=hue_data, 
            palette={True: "red", False: "cornflowerblue"}, 
            style=hue_data, 
            markers={True:"X", False:"o"}, 
            alpha=0.7, s=30, # size=score_norm * 100 if score_norm is not None else 30,
            ax=ax, legend="brief"
        )
        ax.set_title(f"Multivariate Outliers (UMAP of {len(cols_for_plot)} Vars)", fontsize=10)
        ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(labelsize=8); ax.grid(True, alpha=0.3)
        
        if hue_data.nunique() > 1 : ax.legend(title="MVA Outlier", fontsize=8, loc='best')
        else: 
            if ax.get_legend() is not None: ax.legend().remove()

        fig.suptitle("Multivariate Outlier Visualization", fontsize=12, y=0.99)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        
        texture_tag, tex_w, tex_h = plot_texture_func(fig)
        plt.close(fig) # matplotlib figure 객체 닫기

        default_mva_texture_tag = _shared_utils_mva['default_mva_plot_texture_tag']
        if _mva_active_plot_texture_id and _mva_active_plot_texture_id != default_mva_texture_tag and dpg.does_item_exist(_mva_active_plot_texture_id):
            try: dpg.delete_item(_mva_active_plot_texture_id)
            except Exception as e: _log_mva(f"Error deleting old MVA plot texture: {e}")
        
        if texture_tag and tex_w > 0 and tex_h > 0:
            _mva_active_plot_texture_id = texture_tag
            if dpg.does_item_exist(TAG_OT_MVA_VISUALIZATION_IMAGE):
                # 이미지 부모의 너비를 가져와서 이미지 크기 동적 조절
                img_parent_width_mva = dpg.get_item_width(TAG_OT_MVA_VISUALIZATION_PARENT_GROUP)
                display_width_mva = tex_w
                if img_parent_width_mva and img_parent_width_mva > 20 :
                     display_width_mva = min(tex_w, img_parent_width_mva - 20)
                
                display_height_mva = int(tex_h * (display_width_mva / tex_w)) if tex_w > 0 else tex_h
                
                dpg.configure_item(TAG_OT_MVA_VISUALIZATION_IMAGE, texture_tag=_mva_active_plot_texture_id, 
                                   width=int(display_width_mva), height=int(display_height_mva), show=True)
            _log_mva("MVA UMAP plot generated and displayed.")
        else: 
            _clear_mva_visualization_plot() # 실패 시 기본 이미지로
            _log_mva("Failed to generate or display MVA UMAP plot.")

    except Exception as e:
        _log_mva(f"Error during MVA plot generation (UMAP): {e}\n{traceback.format_exc()}")
        _clear_mva_visualization_plot()
        _show_simple_modal_mva("Plot Error", f"Failed to generate MVA visualization: {e}")


def _clear_mva_visualization_plot():
    global _mva_active_plot_texture_id
    if not dpg.is_dearpygui_running(): return

    default_texture_tag = _shared_utils_mva['default_mva_plot_texture_tag'] if _shared_utils_mva else None
    if not default_texture_tag:
        _log_mva("Error: Default MVA plot texture tag not available for clearing plot.")
        return

    if dpg.does_item_exist(TAG_OT_MVA_VISUALIZATION_IMAGE) and dpg.does_item_exist(default_texture_tag):
        cfg = dpg.get_item_configuration(default_texture_tag)
        w,h = (cfg.get('width',100), cfg.get('height',30)) if cfg else (100,30)
        dpg.configure_item(TAG_OT_MVA_VISUALIZATION_IMAGE, texture_tag=default_texture_tag, width=w, height=h, show=True)

    if _mva_active_plot_texture_id and _mva_active_plot_texture_id != default_texture_tag and dpg.does_item_exist(_mva_active_plot_texture_id):
        try:
            dpg.delete_item(_mva_active_plot_texture_id)
        except Exception as e:
            _log_mva(f"Error deleting active MVA plot texture: {e}")
    _mva_active_plot_texture_id = default_texture_tag


# --- Main UI Creation and Update Functions for Multivariate ---
def create_multivariate_ui(parent_tab_bar_tag: str, shared_utilities: dict):
    global _shared_utils_mva, _mva_active_plot_texture_id
    _shared_utils_mva = shared_utilities
    _mva_active_plot_texture_id = _shared_utils_mva.get('default_mva_plot_texture_tag', None)

    with dpg.tab(label="Multivariate Outlier Detection", tag=TAG_OT_MULTIVARIATE_TAB, parent=parent_tab_bar_tag):
        dpg.add_text("1. Configure & Run Multivariate Outlier Detection", color=[255, 255, 0])
        dpg.add_text("Detection Method: Isolation Forest (on selected numeric columns)")
        with dpg.group(horizontal=True):
            dpg.add_text("Contamination ('auto' or 0.0001-0.5):", 
                         tag=TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT + "_label") # 명확한 레이블 태그
            dpg.add_input_text(tag=TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT, width=120, 
                               default_value=str(_mva_iso_forest_contamination), # 문자열로 초기값 설정
                               hint="e.g., 0.1 or auto", 
                               callback=_on_mva_iso_forest_contam_change)
        
        dpg.add_text("Select Variables for Multivariate Analysis:")
        # 라디오 버튼 선택지 이름 수정 (일관성 및 명확성)
        dpg.add_radio_button(items=["All Numeric Columns", "Recommended Columns (TODO)", "Select Custom Columns"], 
                             tag=TAG_OT_MVA_VAR_METHOD_RADIO, default_value=_mva_variable_selection_method, 
                             horizontal=True, callback=_on_mva_var_method_change)
        
        dpg.add_text("Custom Numeric Columns (if selected):", tag=TAG_OT_MVA_CUSTOM_COLS_TABLE_LABEL, show=False)
        with dpg.child_window(tag=TAG_OT_MVA_CUSTOM_COLS_TABLE_CHILD, show=False, height=150, border=True):
            # 테이블 헤더 (한 번만 생성)
            with dpg.table(tag=TAG_OT_MVA_CUSTOM_COLS_TABLE, header_row=True, resizable=True, 
                           policy=dpg.mvTable_SizingStretchProp, scrollY=True, 
                           borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                dpg.add_table_column(label="Select", parent=TAG_OT_MVA_CUSTOM_COLS_TABLE, width_fixed=True, init_width_or_weight=70)
                dpg.add_table_column(label="Column Name", parent=TAG_OT_MVA_CUSTOM_COLS_TABLE, width_stretch=True)
                # 초기에는 비어있음, _populate_mva_custom_cols_table 함수가 내용 채움
                                
        dpg.add_button(label="Run Multivariate Detection", tag=TAG_OT_MVA_DETECT_BUTTON, width=-1, height=30, callback=_run_mva_outlier_detection_logic)
        dpg.add_spacer(height=5)

        dpg.add_text("2. Multivariate Detection Summary & Visualization", color=[255, 255, 0])
        dpg.add_text("Summary will appear here after running detection.", tag=TAG_OT_MVA_RESULTS_TEXT, wrap=-1)
        dpg.add_spacer(height=5)
        with dpg.group(tag=TAG_OT_MVA_VISUALIZATION_PARENT_GROUP, horizontal=False):
            default_mva_tex_tag = _shared_utils_mva.get('default_mva_plot_texture_tag')
            init_w_mva, init_h_mva = 100, 30
            if default_mva_tex_tag and dpg.does_item_exist(default_mva_tex_tag):
                cfg = dpg.get_item_configuration(default_mva_tex_tag)
                init_w_mva = cfg.get('width', init_w_mva)
                init_h_mva = cfg.get('height', init_h_mva)
            
            dpg.add_image(texture_tag=default_mva_tex_tag or "", 
                          tag=TAG_OT_MVA_VISUALIZATION_IMAGE, show=True, 
                          width=init_w_mva, height=init_h_mva)
        
        dpg.add_spacer(height=10)
        dpg.add_text("Multivariate outlier treatment: Typically involves row removal or flagging (manual review recommended).", color=(150,150,150))
        # 향후 MVA 처리 버튼 추가 가능


def update_multivariate_ui(df_input: Optional[pd.DataFrame], shared_utilities: dict, is_new_data: bool):
    global _shared_utils_mva
    if not dpg.is_dearpygui_running(): return
    _shared_utils_mva = shared_utilities

    if not dpg.does_item_exist(TAG_OT_MULTIVARIATE_TAB): return

    current_df_for_mva = df_input # 직접 받은 df 사용

    if current_df_for_mva is None or is_new_data:
        reset_multivariate_state_internal(called_from_parent_reset=False)
        if current_df_for_mva is not None:
            _log_mva("New data loaded for Multivariate Outlier Detection. Please re-configure and run detection.")
            _populate_mva_custom_cols_table(current_df_for_mva) # 새 데이터로 테이블 다시 채우기
        # else:
        #     _log_mva("No data for Multivariate Outlier Detection.")
            
    _update_mva_custom_cols_ui_visibility() # 사용자 지정 컬럼 테이블 가시성 업데이트

    # MVA는 현재 적용(treatment) 버튼이 없으므로, update_ui에서 처리할 추가적인 상태는 많지 않음.
    # 주로 파라미터 UI 업데이트 및 테이블 내용 업데이트 정도.


def reset_multivariate_state_internal(called_from_parent_reset=True):
    global _mva_variable_selection_method, _mva_custom_selected_columns, _mva_custom_col_checkbox_tags
    global _mva_iso_forest_contamination, _df_with_mva_outliers, _mva_outlier_row_indices
    global _mva_active_plot_texture_id, _mva_umap_embedding, _mva_outlier_scores

    _mva_variable_selection_method = "All Numeric Columns" # 초기값 일치
    _mva_custom_selected_columns.clear()
    _mva_custom_col_checkbox_tags.clear()
    _mva_iso_forest_contamination = DEFAULT_MVA_ISO_FOREST_CONTAMINATION
    _df_with_mva_outliers = None
    _mva_outlier_row_indices = None
    _mva_umap_embedding = None
    _mva_outlier_scores = None
    
    if _shared_utils_mva and 'default_mva_plot_texture_tag' in _shared_utils_mva:
        _mva_active_plot_texture_id = _shared_utils_mva['default_mva_plot_texture_tag']
    else:
        _mva_active_plot_texture_id = None

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_MVA_VAR_METHOD_RADIO): 
            dpg.set_value(TAG_OT_MVA_VAR_METHOD_RADIO, _mva_variable_selection_method)
        
        # 사용자 지정 컬럼 테이블 초기화
        if dpg.does_item_exist(TAG_OT_MVA_CUSTOM_COLS_TABLE):
             _populate_mva_custom_cols_table(None) # 빈 데이터로 테이블 내용 지우기
        _update_mva_custom_cols_ui_visibility() # 가시성도 초기 상태로

        if dpg.does_item_exist(TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT): 
            dpg.set_value(TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT, str(_mva_iso_forest_contamination))
        if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): 
            dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, "Run multivariate detection.")
        
        if _shared_utils_mva: # _shared_utils_mva 설정 확인
            _clear_mva_visualization_plot()

    if not called_from_parent_reset:
        _log_mva("Multivariate outlier state reset due to data change.")


def reset_multivariate_state(): # 부모 모듈에서 호출
    reset_multivariate_state_internal(called_from_parent_reset=True)
    _log_mva("Multivariate outlier state has been reset by parent.")

def get_multivariate_settings() -> dict:
    return {
        "mva_variable_selection_method": _mva_variable_selection_method,
        "mva_custom_selected_columns": _mva_custom_selected_columns[:],
        "mva_iso_forest_contamination": _mva_iso_forest_contamination,
        # 탐지 결과(_df_with_mva_outliers, _mva_outlier_row_indices 등)는 저장하지 않음
    }

def apply_multivariate_settings(df_input: Optional[pd.DataFrame], settings: dict, shared_utilities: dict):
    global _mva_variable_selection_method, _mva_custom_selected_columns, _mva_iso_forest_contamination
    global _shared_utils_mva
    
    _shared_utils_mva = shared_utilities

    _mva_variable_selection_method = settings.get("mva_variable_selection_method", "All Numeric Columns")
    _mva_custom_selected_columns = settings.get("mva_custom_selected_columns", [])[:]
    _mva_iso_forest_contamination = settings.get("mva_iso_forest_contamination", DEFAULT_MVA_ISO_FOREST_CONTAMINATION)

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_MVA_VAR_METHOD_RADIO): 
            dpg.set_value(TAG_OT_MVA_VAR_METHOD_RADIO, _mva_variable_selection_method)
        if dpg.does_item_exist(TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT): 
            dpg.set_value(TAG_OT_MVA_ISO_FOREST_CONTAM_INPUT, str(_mva_iso_forest_contamination))
        
        # 사용자 지정 컬럼 테이블 업데이트 및 가시성 설정
        if df_input is not None: # 입력 데이터가 있을 때만 테이블 채우기 시도
             _populate_mva_custom_cols_table(df_input) # 저장된 _mva_custom_selected_columns 반영
        else:
             _populate_mva_custom_cols_table(None)
        _update_mva_custom_cols_ui_visibility()

        if dpg.does_item_exist(TAG_OT_MVA_RESULTS_TEXT): 
            dpg.set_value(TAG_OT_MVA_RESULTS_TEXT, "Settings applied. Run multivariate detection.")
        _clear_mva_visualization_plot()

    _log_mva("Multivariate outlier settings applied from saved state.")
    # update_multivariate_ui(df_input, shared_utilities, True)