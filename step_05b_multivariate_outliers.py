# step_05b_multivariate_outliers.py (Manual Column Selection 제거됨)
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import utils
import functools
import io
from PIL import Image
import traceback

try:
    from pyod.models.iforest import IForest as PyOD_IForest
    import umap
    import shap
except ImportError as e:
    print(f"Warning: Missing one or more libraries (PyOD, UMAP, SHAP) for MVA Outliers: {e}")
    PyOD_IForest, umap, shap = None, None, None

TAG_OT_MULTIVARIATE_TAB = "step5_multivariate_outlier_tab"
TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO = "step5_ot_mva_col_selection_mode_radio"
TAG_OT_MVA_CONTAMINATION_INPUT = "step5_ot_mva_contamination_input"
TAG_OT_MVA_DETECT_BUTTON = "step5_ot_mva_detect_button"
# TAG_OT_MVA_RECOMMEND_PARAMS_BUTTON - 제거됨
TAG_OT_MVA_VISUALIZATION_GROUP = "step5_ot_mva_visualization_group"
TAG_OT_MVA_UMAP_PLOT_IMAGE = "step5_ot_mva_umap_plot_image"
TAG_OT_MVA_PCA_PLOT_IMAGE = "step5_ot_mva_pca_plot_image"
TAG_OT_MVA_UMAP_SECTION_GROUP = "umap_image_section_group"
TAG_OT_MVA_PCA_SECTION_GROUP = "pca_image_section_group"
TAG_OT_MVA_OUTLIER_INSTANCES_TABLE = "step5_ot_mva_outlier_instances_table"
TAG_OT_MVA_INSTANCE_STATS_TABLE = "step5_ot_mva_instance_stats_table"
TAG_OT_MVA_SHAP_PLOT_IMAGE = "step5_ot_mva_shap_plot_image"
TAG_OT_MVA_SHAP_PARENT_GROUP = "shap_image_parent_group_in_tab"
TAG_OT_MVA_BOXPLOT_GROUP = "step5_ot_mva_boxplot_group"
TAG_OT_MVA_APPLY_TREATMENT_BUTTON = "step5_ot_mva_apply_treatment_button" # 새로 추가
TAG_MVA_BULK_TREATMENT_COMBO = "step5_mva_bulk_treatment_combo"
TAG_MVA_EXECUTE_BULK_CHANGE_BUTTON = "step5_mva_execute_bulk_change_button"

DEFAULT_MVA_CONTAMINATION = 0.1
MAX_OUTLIER_INSTANCES_TO_SHOW = 30
TOP_N_VARIABLES_FOR_BOXPLOT = 10
MIN_SAMPLES_FOR_IFOREST = 5
MIN_FEATURES_FOR_IFOREST = 2

_shared_utils_mva: Optional[Dict[str, Any]] = None
_current_df_for_mva: Optional[pd.DataFrame] = None
_df_with_mva_outliers: Optional[pd.DataFrame] = None
_mva_model: Optional[Any] = None
_mva_shap_explainer: Optional[Any] = None
_mva_shap_values_for_selected: Optional[np.ndarray] = None
_mva_eligible_numeric_cols: List[str] = []
_mva_selected_columns_for_detection: List[str] = [] # "Recommended" 모드에서 채워질 수 있음
_mva_column_selection_mode: str = "All Numeric" # 기본값 변경 또는 유지
_mva_contamination: float = DEFAULT_MVA_CONTAMINATION
_mva_outlier_instances_summary: List[Dict[str, Any]] = []
_mva_active_umap_texture_id: Optional[str] = None
_mva_active_pca_texture_id: Optional[str] = None
_mva_active_shap_texture_id: Optional[str] = None
_mva_selected_outlier_instance_idx: Optional[Any] = None
_mva_all_selectable_tags_in_instances_table: List[str] = []
_mva_top_gap_vars_for_boxplot: List[str] = []
_mva_boxplot_image_tags: List[str] = []
_mva_last_recommendation_details_str: Optional[str] = None
_mva_treatment_selections: Dict[Any, str] = {} # 새로 추가: 이상치 처리 방법을 저장할 딕셔너리 (인덱스 -> 방법)


def _log_mva(message: str):
    if _shared_utils_mva and 'log_message_func' in _shared_utils_mva:
        _shared_utils_mva['log_message_func'](f"[MVAOutlier] {message}")

def _show_simple_modal_mva(title: str, message: str, width: int = 450, height: int = 200):
    if _shared_utils_mva and 'util_funcs_common' in _shared_utils_mva and \
       '_show_simple_modal_message' in _shared_utils_mva['util_funcs_common']:
        _shared_utils_mva['util_funcs_common']['_show_simple_modal_message'](title, message, width, height)

def _s5_plot_to_dpg_texture_mva(fig: matplotlib.figure.Figure, desired_dpi: int = 90) -> Tuple[Optional[str], int, int, Optional[bytes]]:
    if _shared_utils_mva and 'plot_to_dpg_texture_func' in _shared_utils_mva:
        return _shared_utils_mva['plot_to_dpg_texture_func'](fig, desired_dpi)
    return None, 0, 0, None

def _remove_old_ai_buttons(parent_tag: str, button_alias_prefix: str):
    if dpg.is_dearpygui_running() and dpg.does_item_exist(parent_tag):
        children_slots = dpg.get_item_children(parent_tag, 1)
        for child_tag_slot in children_slots:
            item_info = dpg.get_item_info(child_tag_slot)
            if item_info['type'] == "mvAppItemType::mvButton":
                alias = dpg.get_item_alias(child_tag_slot)
                if alias and alias.startswith(button_alias_prefix):
                    try: dpg.delete_item(child_tag_slot)
                    except Exception as e_del: _log_mva(f"Error deleting old AI button {alias}: {e_del}")

def _on_mva_col_selection_mode_change(sender, app_data: str, user_data):
    global _mva_column_selection_mode
    _mva_column_selection_mode = app_data
    _update_selected_columns_for_mva_detection()

def _on_mva_contamination_change(sender, app_data: float, user_data):
    global _mva_contamination
    if 0.0 < app_data <= 0.5: _mva_contamination = app_data
    else:
        dpg.set_value(sender, _mva_contamination)
        _show_simple_modal_mva("Input Error", "Contamination must be (0.0, 0.5].")

def _set_mva_recommended_parameters(sender, app_data, user_data):
    global _mva_contamination, _mva_column_selection_mode
    _mva_contamination = DEFAULT_MVA_CONTAMINATION
    _mva_column_selection_mode = "Recommended" # "Manual"이 없으므로 이쪽이 더 적절할 수 있음
    if dpg.does_item_exist(TAG_OT_MVA_CONTAMINATION_INPUT): dpg.set_value(TAG_OT_MVA_CONTAMINATION_INPUT, _mva_contamination)
    if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO): dpg.set_value(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO, _mva_column_selection_mode)
    _update_selected_columns_for_mva_detection()
    _show_simple_modal_mva("Info", "Recommended MVA parameters applied.")

def _get_eligible_numeric_cols_for_mva(df: pd.DataFrame) -> List[str]:
    if df is None: return []
    numeric_cols = []
    if not (_shared_utils_mva and 'main_app_callbacks' in _shared_utils_mva): return df.select_dtypes(include=np.number).columns.tolist()
    s1_col_types = _shared_utils_mva['main_app_callbacks'].get('get_column_analysis_types', lambda: {})()
    for col in df.columns:
        s1_type = s1_col_types.get(col, "")
        is_s1_num_not_binary = "Numeric" in s1_type and "Binary" not in s1_type
        is_pandas_num_many_unique = pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].nunique() > 5
        if is_s1_num_not_binary or is_pandas_num_many_unique:
            if df[col].isnull().sum() / len(df) < 0.8 and df[col].dropna().var() > 1e-6:
                numeric_cols.append(col)
    return numeric_cols

def _update_selected_columns_for_mva_detection():
    global _mva_selected_columns_for_detection, _mva_eligible_numeric_cols, _mva_last_recommendation_details_str
    _mva_last_recommendation_details_str = None
    current_df = _shared_utils_mva['get_current_df_func']() if _shared_utils_mva and 'get_current_df_func' in _shared_utils_mva else None
    if current_df is None:
        _mva_eligible_numeric_cols, _mva_selected_columns_for_detection = [], []
        return

    _mva_eligible_numeric_cols = _get_eligible_numeric_cols_for_mva(current_df)
    reason = "all eligible numeric features (fallback)."
    
    if _mva_column_selection_mode == "All Numeric":
        _mva_selected_columns_for_detection = _mva_eligible_numeric_cols[:]
    elif _mva_column_selection_mode == "Recommended":
        target_var, target_type, relevance_scores = None, None, []
        if _shared_utils_mva and 'main_app_callbacks' in _shared_utils_mva:
            cb = _shared_utils_mva['main_app_callbacks']
            target_var, target_type = cb.get('get_selected_target_variable')(), cb.get('get_selected_target_variable_type')()
        
        eligible_for_relevance = [col for col in _mva_eligible_numeric_cols if col != target_var]
        if target_var and target_type and eligible_for_relevance and _shared_utils_mva and \
           'util_funcs_common' in _shared_utils_mva and 'calculate_feature_target_relevance' in _shared_utils_mva['util_funcs_common']:
            relevance_scores = _shared_utils_mva['util_funcs_common']['calculate_feature_target_relevance'](
                current_df, target_var, target_type, eligible_for_relevance, _shared_utils_mva.get('main_app_callbacks'))
        
        if relevance_scores:
            num_to_select = max(MIN_FEATURES_FOR_IFOREST, min(20, int(len(eligible_for_relevance) * 0.20))) # 최소 MIN_FEATURES_FOR_IFOREST 개 이상 선택
            _mva_selected_columns_for_detection = [feat for feat, _ in relevance_scores[:num_to_select]]
            if len(_mva_selected_columns_for_detection) < MIN_FEATURES_FOR_IFOREST and len(_mva_eligible_numeric_cols) >= MIN_FEATURES_FOR_IFOREST: # 추천 컬럼 수가 너무 적으면 전체에서 채움
                _mva_selected_columns_for_detection = _mva_eligible_numeric_cols[:MIN_FEATURES_FOR_IFOREST]
            reason = f"top {len(_mva_selected_columns_for_detection)} features by relevance to '{target_var}'."
        else: 
            _mva_selected_columns_for_detection = _mva_eligible_numeric_cols[:max(MIN_FEATURES_FOR_IFOREST, 20)] if len(_mva_eligible_numeric_cols) >= MIN_FEATURES_FOR_IFOREST else _mva_eligible_numeric_cols[:]

    _mva_last_recommendation_details_str = f"Features: {reason}"

def _run_mva_outlier_detection_logic(sender, app_data, user_data):
    global _df_with_mva_outliers, _mva_model, _mva_shap_explainer, _mva_outlier_instances_summary
    _log_mva("Run MVA Outlier Detection clicked.")
    _mva_treatment_selections.clear() # 새로 탐지 시 이전 처리 방법 선택 초기화
    if dpg.does_item_exist(TAG_MVA_BULK_TREATMENT_COMBO):
        dpg.set_value(TAG_MVA_BULK_TREATMENT_COMBO, "--- Select Bulk Action ---")
    current_df_orig = _shared_utils_mva['get_current_df_func']() if _shared_utils_mva and 'get_current_df_func' in _shared_utils_mva else None
    if current_df_orig is None:
        _show_simple_modal_mva("Error", "No data for MVA detection.")
        return

    _update_selected_columns_for_mva_detection()
    if not _mva_selected_columns_for_detection or len(_mva_selected_columns_for_detection) < MIN_FEATURES_FOR_IFOREST:
        _show_simple_modal_mva("Error", f"Select at least {MIN_FEATURES_FOR_IFOREST} numeric features for MVA detection.")
        return

    df_subset_raw = current_df_orig[_mva_selected_columns_for_detection].copy()
    df_for_detection_processed = pd.DataFrame(index=df_subset_raw.index)
    for col in df_subset_raw.columns:
        try:
            converted_col = pd.to_numeric(df_subset_raw[col], errors='coerce')
            if converted_col.isnull().all():
                _log_mva(f"Column '{col}' is all NaN after to_numeric conversion. Skipping for MVA detection.")
                continue
            if converted_col.isnull().any():
                # 학습 데이터 생성 시에는 평균값 등으로 채우는 것이 일반적
                # 여기서는 current_df_orig (초기 입력 DF)에서 평균을 가져옴
                mean_val = current_df_orig[col].dropna().mean()
                _log_mva(f"Filling NaNs in '{col}' with mean value: {mean_val if pd.notna(mean_val) else '0 (mean was NaN)'}")
                converted_col.fillna(mean_val if pd.notna(mean_val) else 0, inplace=True)
            df_for_detection_processed[col] = converted_col
        except Exception as e_conv:
            _log_mva(f"Error converting column '{col}' to numeric for MVA: {e_conv}. Skipping this column.")
            continue
    
    if df_for_detection_processed.empty or df_for_detection_processed.shape[1] < MIN_FEATURES_FOR_IFOREST:
        _show_simple_modal_mva("Data Error", f"Not enough valid numeric features after conversion/cleaning for MVA. Min required: {MIN_FEATURES_FOR_IFOREST}.")
        return
        
    df_for_detection = df_for_detection_processed.dropna(axis=0) # NaN이 있는 행 전체 제거 (IForest는 NaN 처리 못함)
    if len(df_for_detection) < MIN_SAMPLES_FOR_IFOREST:
        _show_simple_modal_mva("Error", f"Not enough samples after NaN row removal for MVA. Min required: {MIN_SAMPLES_FOR_IFOREST}. Got: {len(df_for_detection)}.")
        return
        
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_for_detection.values)
        
        _mva_model = PyOD_IForest(contamination=_mva_contamination, random_state=42, n_jobs=-1)
        _mva_model.fit(df_for_detection.values) # 원본 스케일 (그러나 NaN 처리된) 데이터로 학습
        outlier_scores, outlier_labels = _mva_model.decision_scores_, _mva_model.labels_

        _df_with_mva_outliers = current_df_orig.copy()
        _df_with_mva_outliers.loc[df_for_detection.index, 'mva_outlier_score'] = outlier_scores
        _df_with_mva_outliers.loc[df_for_detection.index, 'mva_is_outlier'] = outlier_labels.astype(bool)
        _df_with_mva_outliers['mva_is_outlier'] = _df_with_mva_outliers['mva_is_outlier'].fillna(False)
        min_s = np.min(outlier_scores) if len(outlier_scores) > 0 else 0
        _df_with_mva_outliers['mva_outlier_score'] = _df_with_mva_outliers['mva_outlier_score'].fillna(min_s -1) # NaN이었던 원본 행에 대한 처리
        
        # --- SHAP Explainer 초기화 부분 수정 ---
        if shap:
            _mva_shap_explainer = None # 시도 전에 None으로 초기화
            try:
                _log_mva("Attempting to initialize SHAP TreeExplainer...")
                _log_mva(f"MVA model detector type for SHAP: {type(_mva_model.detector_)}")

                df_for_detection_shap = df_for_detection.copy() # 원본 df_for_detection은 유지
                df_for_detection_shap.columns = df_for_detection_shap.columns.astype(str)
                
                # --- 추가된 부분: SHAP에 전달하기 전 데이터 타입을 float으로 명시적 변환 ---
                _log_mva(f"Data types of df_for_detection_shap BEFORE explicit astype(float):\n{df_for_detection_shap.dtypes.to_string()}")
                for col in df_for_detection_shap.columns:
                    try:
                        # 이전 단계에서 pd.to_numeric 및 NaN 처리가 완료되었다고 가정하고 float으로 변환 시도
                        df_for_detection_shap[col] = df_for_detection_shap[col].astype(float)
                    except ValueError as e_astype:
                        # 만약 astype(float)에서 오류 발생 시, 해당 컬럼을 다시 한번 pd.to_numeric으로 변환 시도
                        # 이는 이전 단계의 숫자 변환이 불완전했음을 의미할 수 있음
                        _log_mva(f"Warning: Could not convert column '{col}' to float directly for SHAP. Error: {e_astype}. "
                                 f"Original dtype in df_for_detection: {df_for_detection[col].dtype}. Trying pd.to_numeric with coerce and fillna(0).")
                        # df_for_detection 단계에서 이미 NaN 행이 제거되었으므로, 여기서 coerce 후 NaN이 발생하면 안됨.
                        # fillna(0)은 예비 조치이며, 데이터 특성에 따라 다른 값(예: 평균)이 더 적절할 수 있음.
                        df_for_detection_shap[col] = pd.to_numeric(df_for_detection_shap[col], errors='coerce').fillna(0)
                _log_mva(f"Data types of df_for_detection_shap AFTER explicit astype(float):\n{df_for_detection_shap.dtypes.to_string()}")
                feature_names_for_shap = df_for_detection_shap.columns.tolist()

                _log_mva(f"Data for SHAP explainer (df_for_detection_shap - first 3 rows, shape {df_for_detection_shap.shape}):\n{df_for_detection_shap.head(3).to_string()}")
                _log_mva(f"Feature names for SHAP: {feature_names_for_shap}")

                # TreeExplainer 초기화
                explainer_instance = shap.TreeExplainer(
                    _mva_model.detector_,  # PyOD IForest의 내부 scikit-learn IsolationForest 모델
                    data=df_for_detection_shap,  # 학습에 사용된 데이터와 유사한 분포의 배경 데이터 (DataFrame 형태)
                    feature_names=feature_names_for_shap # 특징 이름 명시
                )
                _mva_shap_explainer = explainer_instance # 성공 시 할당
                _log_mva(f"SHAP TreeExplainer initialized successfully. Type: {type(_mva_shap_explainer)}")

            except BaseException as e_sh_init_base: # 모든 예외 (SystemExit, KeyboardInterrupt 포함) 포착
                # 표준 print를 사용하여 콘솔에 직접 오류 출력 (로깅 함수 문제 회피)
                print("\n--- CRITICAL SHAP TreeExplainer INIT ERROR (BaseException block) ---")
                print(f"Error Type: {type(e_sh_init_base)}")
                print(f"Error Message: {str(e_sh_init_base)}")
                import traceback
                print("Traceback:")
                traceback.print_exc() # 표준 traceback 출력
                print("--- END CRITICAL SHAP TreeExplainer INIT ERROR ---\n")
                
                # _log_mva도 시도 (다른 로그와 함께 기록될 수 있도록)
                _log_mva(f"CRITICAL SHAP TreeExplainer init error (caught BaseException): {type(e_sh_init_base).__name__} - {str(e_sh_init_base)}")
                _mva_shap_explainer = None # 실패 시 명시적으로 None 할당
        else:
            _log_mva("SHAP library not imported or not considered active.")
            _mva_shap_explainer = None
        # --- SHAP Explainer 초기화 부분 수정 끝 ---

        _mva_outlier_instances_summary = []
        # ... (이하 기존 코드와 동일하게 outlier instances 요약 및 UI 업데이트)
        detected_df = _df_with_mva_outliers[_df_with_mva_outliers['mva_is_outlier'] == True]
        if not detected_df.empty and 'mva_outlier_score' in detected_df.columns:
            detected_df = detected_df.sort_values(by='mva_outlier_score', ascending=False)
        for idx, row in detected_df.head(MAX_OUTLIER_INSTANCES_TO_SHOW).iterrows():
            _mva_outlier_instances_summary.append({"Original Index": idx, "MVA Outlier Score": f"{row['mva_outlier_score']:.4f}"})
        
        _populate_mva_outlier_instances_table()
        _generate_mva_umap_pca_plots(scaled_data, df_for_detection.index, outlier_labels) # scaled_data와 원본 인덱스 전달
        _generate_mva_boxplots_for_comparison()

        num_total, num_outliers = len(current_df_orig), _df_with_mva_outliers['mva_is_outlier'].sum()
        ratio = (num_outliers / num_total) * 100 if num_total > 0 else 0
        if dpg.does_item_exist("mva_summary_text_status"): dpg.set_value("mva_summary_text_status", "Detection Complete. Summary:")
        if dpg.does_item_exist("mva_summary_text_used_features"): dpg.set_value("mva_summary_text_used_features", f"  - Features Used: {df_for_detection.shape[1]} (out of {_mva_selected_columns_for_detection if _mva_selected_columns_for_detection else 'N/A selected'}) from {df_for_detection.shape[0]} samples")
        if dpg.does_item_exist("mva_summary_text_detected_outliers"): dpg.set_value("mva_summary_text_detected_outliers", f"  - Detected Outliers: {num_outliers} samples")
        if dpg.does_item_exist("mva_summary_text_outlier_ratio"): dpg.set_value("mva_summary_text_outlier_ratio", f"  - Outlier Ratio: {ratio:.2f}% of {num_total}")
        
        msg = "MVA detection finished." + (f"\n\n[Info on Features Used]\n{_mva_last_recommendation_details_str}" if _mva_last_recommendation_details_str else "")
        _show_simple_modal_mva("Detection Complete", msg, height=min(max(200, (msg.count('\n') + 2) * 22), 450), width=500) # 너비와 높이 조정
    except Exception as e_det:
        _log_mva(f"MVA detection logic error: {e_det}\n{traceback.format_exc()}")
        _show_simple_modal_mva("Detection Error", f"An error occurred during MVA detection: {str(e_det)[:200]}")
        _df_with_mva_outliers, _mva_model, _mva_shap_explainer = None, None, None
        _clear_all_mva_visualizations(); _populate_mva_outlier_instances_table()

def _populate_mva_outlier_instances_table():
    global _mva_all_selectable_tags_in_instances_table, _mva_treatment_selections
    if not (dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_MVA_OUTLIER_INSTANCES_TABLE)): return
    dpg.delete_item(TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, children_only=True)
    _mva_all_selectable_tags_in_instances_table.clear()
    
    if not _mva_outlier_instances_summary:
        dpg.add_table_column(label="Info", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE)
        with dpg.table_row(parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE): dpg.add_text("No MVA outliers detected or run detection.")
        _clear_mva_instance_details(); return

    dpg.add_table_column(label="Original Index", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, init_width_or_weight=0.25)
    dpg.add_table_column(label="MVA Outlier Score", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, init_width_or_weight=0.35)
    dpg.add_table_column(label="Treatment Method", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, init_width_or_weight=0.40)

    treatment_options = ["Do Not Treat", "Remove Row", "Treat as Missing"]

    for i, item in enumerate(_mva_outlier_instances_summary):
        idx, score = item["Original Index"], item["MVA Outlier Score"]
        tag = f"mva_instance_sel_{i}_{idx}_{dpg.generate_uuid()}"
        _mva_all_selectable_tags_in_instances_table.append(tag)
        with dpg.table_row(parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE):
            dpg.add_selectable(label=str(idx), tag=tag, user_data=idx, callback=_on_mva_outlier_instance_selected)
            dpg.add_text(score)
            
            current_treatment = _mva_treatment_selections.get(idx, "Do Not Treat")
            dpg.add_combo(treatment_options, default_value=current_treatment, width=-1,
                          user_data=idx, callback=_on_mva_treatment_method_change)

def _on_mva_treatment_method_change(sender, app_data: str, user_data_idx: Any):
    """MVA 이상치 처리 방법 콤보박스 변경 시 콜백"""
    global _mva_treatment_selections
    _mva_treatment_selections[user_data_idx] = app_data
    _log_mva(f"MVA Treatment for index '{user_data_idx}' set to '{app_data}'.")

def _apply_mva_treatment_logic():
    """선택된 다변량 이상치 처리 방법을 적용하는 함수"""
    global _df_with_mva_outliers
    _log_mva("Apply MVA Treatments button clicked.")

    if _df_with_mva_outliers is None:
        _show_simple_modal_mva("Error", "No detection data available to apply treatments.")
        return
        
    treatments_to_apply = {idx: method for idx, method in _mva_treatment_selections.items() if method != "Do Not Treat"}

    if not treatments_to_apply:
        _show_simple_modal_mva("Info", "No treatments were selected to be applied.")
        return

    df_processed = _df_with_mva_outliers.copy()
    indices_to_drop = [idx for idx, method in treatments_to_apply.items() if method == "Remove Row"]
    indices_to_nan = [idx for idx, method in treatments_to_apply.items() if method == "Treat as Missing"]

    if indices_to_drop:
        # drop하려는 인덱스가 실제 DataFrame에 있는지 확인
        valid_indices_to_drop = [idx for idx in indices_to_drop if idx in df_processed.index]
        if valid_indices_to_drop:
            df_processed.drop(index=valid_indices_to_drop, inplace=True)
            _log_mva(f"Removed {len(valid_indices_to_drop)} rows.")

    if indices_to_nan:
        # NaN 처리하려는 인덱스가 실제 DataFrame에 있는지 확인
        valid_indices_to_nan = [idx for idx in indices_to_nan if idx in df_processed.index]
        if valid_indices_to_nan:
            # 모든 컬럼을 NaN으로 처리
            df_processed.loc[valid_indices_to_nan, :] = np.nan
            _log_mva(f"Treated {len(valid_indices_to_nan)} rows as missing (all columns).")

    # 처리 후, 부모의 콜백 함수 호출하여 다음 단계로 데이터 전달
    if _shared_utils_mva and 'main_app_callbacks' in _shared_utils_mva:
        _shared_utils_mva['main_app_callbacks']['step5_outlier_treatment_complete'](df_processed)
        _show_simple_modal_mva("Success", f"MVA outlier treatments applied.\n- {len(indices_to_drop)} rows removed.\n- {len(indices_to_nan)} rows set to NaN.")
    
    # 처리 후에는 선택 상태 초기화
    _mva_treatment_selections.clear()
    _populate_mva_outlier_instances_table() # 테이블 다시 그려서 콤보박스 초기화

def _on_mva_outlier_instance_selected(sender, app_data_is_selected: bool, user_data_original_idx: Any):
    global _mva_selected_outlier_instance_idx
    if app_data_is_selected:
        for tag_iter in _mva_all_selectable_tags_in_instances_table:
            if tag_iter != sender and dpg.does_item_exist(tag_iter) and dpg.get_value(tag_iter): dpg.set_value(tag_iter, False)
        _mva_selected_outlier_instance_idx = user_data_original_idx
        _display_mva_instance_statistics(user_data_original_idx)
        _generate_mva_shap_plot_for_instance(user_data_original_idx)
    elif _mva_selected_outlier_instance_idx == user_data_original_idx:
         _mva_selected_outlier_instance_idx = None; _clear_mva_instance_details()

def _display_mva_instance_statistics(original_idx: Any):
    if not (dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_MVA_INSTANCE_STATS_TABLE)): return
    dpg.delete_item(TAG_OT_MVA_INSTANCE_STATS_TABLE, children_only=True)
    if _df_with_mva_outliers is None or original_idx not in _df_with_mva_outliers.index:
        dpg.add_table_column(label="Info", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE)
        with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE): dpg.add_text("Instance data not available.")
        return

    instance_data = _df_with_mva_outliers.loc[original_idx]
    stats_list = []
    cols_to_show = _mva_selected_columns_for_detection[:] if _mva_selected_columns_for_detection else []
    src_df_stats = _current_df_for_mva

    for feat_name in cols_to_show:
        if feat_name not in instance_data: continue
        val = instance_data[feat_name]
        mean_s, median_s, z_s, z_abs_key = "N/A", "N/A", "N/A", 0.0
        if src_df_stats is not None and feat_name in src_df_stats.columns:
            overall_series = src_df_stats[feat_name].dropna()
            if not overall_series.empty and pd.api.types.is_numeric_dtype(overall_series.dtype):
                mean_v, median_v, std_v = overall_series.mean(), overall_series.median(), overall_series.std()
                mean_s, median_s = f"{mean_v:.2f}", f"{median_v:.2f}"
                if pd.notna(std_v) and std_v > 1e-9 and pd.notna(val) and isinstance(val, (int, float, np.number)) and pd.notna(mean_v):
                    z_score_val = (val - mean_v) / std_v
                    z_abs_key, z_s = abs(z_score_val), f"{z_score_val:.2f}"
        stats_list.append({"feat": feat_name, "val_s": f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val), 
                           "mean_s": mean_s, "median_s": median_s, "z_s": z_s, "z_key": z_abs_key if pd.notna(z_abs_key) else -1})
    
    sorted_stats = sorted(stats_list, key=lambda x: x["z_key"], reverse=True)
    headers = ["Feature", "Value", "Mean", "Median", "Z-score Dist."]
    for i_h, h_name in enumerate(headers): dpg.add_table_column(label=h_name, parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, init_width_or_weight=0.20)
    for item in sorted_stats:
        with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE):
            dpg.add_text(item["feat"]); dpg.add_text(item["val_s"]); dpg.add_text(item["mean_s"]); dpg.add_text(item["median_s"]); dpg.add_text(item["z_s"])
    if 'mva_outlier_score' in instance_data:
        with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE):
            dpg.add_text("MVA Outlier Score", color=[255,255,0]); dpg.add_text(f"{instance_data['mva_outlier_score']:.4f}"); dpg.add_text("-"); dpg.add_text("-"); dpg.add_text("-")

def _clear_mva_instance_details():
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_OT_MVA_INSTANCE_STATS_TABLE):
        dpg.delete_item(TAG_OT_MVA_INSTANCE_STATS_TABLE, children_only=True)
        dpg.add_table_column(label="Info", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE)
        with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE): dpg.add_text("Select an outlier instance.")
    _clear_mva_shap_plot()

def _plot_cleanup_and_set(image_widget_tag: str, active_texture_id_var_name: str, new_texture_tag: Optional[str], default_texture_key: str, w: int, h: int):
    # w와 h는 matplotlib에서 생성된 원본 이미지의 너비와 높이지만,
    # SHAP 이미지의 경우 이 값을 최종 크기 결정에 사용하지 않습니다.
    global _mva_active_umap_texture_id, _mva_active_pca_texture_id, _mva_active_shap_texture_id
    current_active_texture_id = globals().get(active_texture_id_var_name)
    default_tex = _shared_utils_mva.get(default_texture_key) if _shared_utils_mva else None
    
    _log_mva(f"_plot_cleanup_and_set for '{image_widget_tag}'. Matplotlib original w={w}, h={h}. New texture: {new_texture_tag is not None}")

    # 이전 텍스처 삭제 로직 (동일)
    if current_active_texture_id and current_active_texture_id != default_tex and dpg.does_item_exist(current_active_texture_id):
        try:
            dpg.delete_item(current_active_texture_id)
        except Exception as e:
            _log_mva(f"Error deleting texture '{current_active_texture_id}': {e}")

    if new_texture_tag: # 새 텍스처가 유효할 때
        globals()[active_texture_id_var_name] = new_texture_tag
        if dpg.is_dearpygui_running() and dpg.does_item_exist(image_widget_tag):
            
            display_w: int
            display_h: int

            if image_widget_tag == TAG_OT_MVA_SHAP_PLOT_IMAGE:
                # --- SHAP 이미지: 무조건 찌그러뜨려서 영역에 맞춤 ---
                shap_parent_container_tag = "shap_content_child_window" # 이미지가 실제로 그려질 child_window
                
                # 1. 목표 표시 너비 (display_w) 결정:
                #    shap_parent_container_tag의 현재 너비를 가져오려고 시도.
                #    실패 시 (0 또는 None), main_app.py 레이아웃 기반의 '예상되는' 너비 사용.
                container_measured_width = 0
                if dpg.does_item_exist(shap_parent_container_tag):
                    container_measured_width = dpg.get_item_width(shap_parent_container_tag)
                
                _log_mva(f"SHAP plot: '{shap_parent_container_tag}' measured width: {container_measured_width}")

                if container_measured_width and container_measured_width > 20: # 측정된 너비가 유효하면 사용
                    display_w = container_measured_width - 10 # 양쪽 여백 5px씩 고려
                else:
                    # 측정 실패 시 Fallback: content_area 너비가 1000px이고, SHAP 컬럼이 테이블의 55%를 차지하며,
                    # 테이블 셀 내부 패딩 등을 고려한 예상 너비. 이 값은 실제 UI와 일치해야 함.
                    # (1000 * 0.55) - (테이블 셀 패딩 + child_window 내부 패딩 등)
                    # 이 값을 실제 레이아웃을 보고 정확하게 계산하거나, 여러 번의 테스트를 통해 최적화해야 합니다.
                    # 예시로, content_area가 1000, 테이블 컬럼 비율 0.55, 내부 여백 총 20px 가정 -> 1000 * 0.55 - 20 = 530
                    estimated_container_width = 530 # <<-- 이 값을 실제 UI에 맞게 조정하세요!
                    display_w = estimated_container_width
                    _log_mva(f"SHAP plot: '{shap_parent_container_tag}' width is {container_measured_width}. Using estimated display_w: {display_w}")
                
                # 2. 목표 표시 높이 (display_h) 결정:
                #    shap_parent_container_tag의 고정 높이(500px)에서, 그 안에 있는 다른 UI 요소들의 높이를 제외.
                container_fixed_height = 500 # 'shap_content_child_window'의 고정 높이
                
                # SHAP 이미지 위젯 위에 있는 'AI 분석 버튼'과 그 아래 'Spacer'의 높이, 그리고 '상태 메시지 텍스트'의 높이를 합산.
                # 추가적인 상하 내부 패딩/여백도 고려.
                # 이 값들은 create_multivariate_ui 함수에서 해당 위젯들의 실제 구성에 따라 달라집니다.
                # get_item_height는 아이템이 그려진 후에 정확하므로, 여기서는 추정치나 고정값을 사용할 수 있습니다.
                button_tag_above = "mva_shap_plot_ai_analyze_button"
                spacer_height_above = 5 # 버튼 아래 spacer
                status_text_tag_above = "mva_shap_status_text"
                
                height_of_other_elements = 0
                if dpg.does_item_exist(button_tag_above) and dpg.is_item_shown(button_tag_above):
                     height_of_other_elements += (dpg.get_item_height(button_tag_above) or 30) + spacer_height_above
                
                if dpg.does_item_exist(status_text_tag_above) and dpg.is_item_shown(status_text_tag_above):
                    # 상태 텍스트는 내용에 따라 높이가 변할 수 있으므로, 대략적인 최대 예상 높이를 사용하거나,
                    # 실제 UI에서 차지하는 공간을 보고 값을 정합니다.
                    height_of_other_elements += (dpg.get_item_height(status_text_tag_above) or 20) 
                
                height_of_other_elements += 15 # 이미지 위젯 자체의 상하 여백/패딩 (임의값)

                display_h = container_fixed_height - height_of_other_elements
                display_h = max(50, display_h) # 최소 높이 50px 보장 (너무 작으면 보이지 않음)

                _log_mva(f"SHAP plot: Calculated height_of_other_elements: {height_of_other_elements}")
                _log_mva(f"SHAP plot: Forcing display size (찌그러뜨림): display_w={display_w}, display_h={display_h}")

            else:
                # --- 다른 이미지들(UMAP, PCA 등)의 경우: 기존 로직 (예: 종횡비 유지) ---
                parent_item_for_other_images = dpg.get_item_parent(image_widget_tag)
                parent_container_w_others = w # 기본값은 원본 이미지 너비
                if parent_item_for_other_images and dpg.does_item_exist(parent_item_for_other_images):
                    parent_container_w_others = dpg.get_item_width(parent_item_for_other_images) or w
                
                display_w = min(w, parent_container_w_others - 10 if parent_container_w_others > 10 else parent_container_w_others)
                if w > 0 : # 원본 이미지 너비가 0보다 클 때만 비율 계산
                    display_h = int(h * (display_w / w)) # 종횡비 유지
                else:
                    parent_container_h_others = 300 # 기본 높이
                    if parent_item_for_other_images and dpg.does_item_exist(parent_item_for_other_images):
                         parent_container_h_others = dpg.get_item_height(parent_item_for_other_images) or 300
                    display_h = parent_container_h_others
                _log_mva(f"Non-SHAP plot '{image_widget_tag}': Ratio-based size. display_w={display_w}, display_h={display_h}")

            # 최종적으로 display_w, display_h가 너무 작지 않도록 보정
            display_w = max(10, display_w)
            display_h = max(10, display_h)

            _log_mva(f"Final DPG configure for '{image_widget_tag}': width={display_w}, height={display_h}, texture_tag='{new_texture_tag}'")
            dpg.configure_item(image_widget_tag, texture_tag=new_texture_tag, width=display_w, height=display_h, show=True)

    elif default_tex and dpg.does_item_exist(default_tex) and dpg.does_item_exist(image_widget_tag):
        cfg = dpg.get_item_configuration(default_tex)
        def_w, def_h = (cfg.get('width', 100), cfg.get('height', 30))
        dpg.configure_item(image_widget_tag, texture_tag=default_tex, width=def_w, height=def_h, show=True)
        globals()[active_texture_id_var_name] = default_tex
        _log_mva(f"Configuring '{image_widget_tag}' with default texture: width={def_w}, height={def_h}")
    elif dpg.is_dearpygui_running() and dpg.does_item_exist(image_widget_tag):
         dpg.configure_item(image_widget_tag, show=False)
         _log_mva(f"Hiding '{image_widget_tag}' as no valid new or default texture and it exists.")

def _clear_mva_umap_plot():
    _remove_old_ai_buttons(TAG_OT_MVA_UMAP_SECTION_GROUP, "MVA_UMAP_AI_Button_")
    _plot_cleanup_and_set(TAG_OT_MVA_UMAP_PLOT_IMAGE, '_mva_active_umap_texture_id', None, 'default_umap_texture_tag', 0, 0)
def _clear_mva_pca_plot():
    _remove_old_ai_buttons(TAG_OT_MVA_PCA_SECTION_GROUP, "MVA_PCA_AI_Button_")
    _plot_cleanup_and_set(TAG_OT_MVA_PCA_PLOT_IMAGE, '_mva_active_pca_texture_id', None, 'default_pca_texture_tag', 0, 0)
def _clear_mva_shap_plot():
    _remove_old_ai_buttons(TAG_OT_MVA_SHAP_PARENT_GROUP, "MVA_SHAP_AI_Button_")
    _plot_cleanup_and_set(TAG_OT_MVA_SHAP_PLOT_IMAGE, '_mva_active_shap_texture_id', None, 'default_shap_plot_texture_tag', 0, 0)
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_MVA_SHAP_PARENT_GROUP):
        if dpg.does_item_exist("mva_shap_status_text"): dpg.delete_item("mva_shap_status_text")
        if not (_mva_selected_outlier_instance_idx and _mva_shap_explainer):
            dpg.add_text("Select an instance for SHAP or SHAP N/A.", parent=TAG_OT_MVA_SHAP_PARENT_GROUP, tag="mva_shap_status_text", before=TAG_OT_MVA_SHAP_PLOT_IMAGE)

def _clear_mva_boxplots():
    global _mva_boxplot_image_tags
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_OT_MVA_BOXPLOT_GROUP):
        dpg.delete_item(TAG_OT_MVA_BOXPLOT_GROUP, children_only=True)
    for tex_tag in _mva_boxplot_image_tags:
        if dpg.does_item_exist(tex_tag):
            try: dpg.delete_item(tex_tag)
            except Exception as e: _log_mva(f"Error deleting boxplot tex {tex_tag}: {e}")
    _mva_boxplot_image_tags.clear()

def _clear_all_mva_visualizations():
    _clear_mva_umap_plot(); _clear_mva_pca_plot(); _clear_mva_shap_plot(); _clear_mva_boxplots()
    _mva_selected_outlier_instance_idx = None

def _generate_mva_umap_pca_plots(data_for_reduction: np.ndarray, original_indices: pd.Index, outlier_labels: np.ndarray):
    global _mva_active_umap_texture_id, _mva_active_pca_texture_id
    _log_mva("Attempting UMAP/PCA plots...")
    # 이전 AI 버튼 및 텍스처 정리
    _clear_mva_umap_plot() 
    _clear_mva_pca_plot()
    if not _shared_utils_mva: _log_mva("Shared utils missing for UMAP/PCA."); return
    main_cb, plot_func = _shared_utils_mva.get('main_app_callbacks'), _shared_utils_mva.get('plot_to_dpg_texture_func')
    if not (plot_func and main_cb): _log_mva("Plot func or main_cb missing for UMAP/PCA."); return
    if data_for_reduction is None or len(data_for_reduction) < 2 or data_for_reduction.shape[1] < 2 or \
       len(data_for_reduction) != len(outlier_labels) or len(data_for_reduction) != len(original_indices):
        _log_mva("Insufficient/mismatched data for UMAP/PCA."); return

    # UMAP
    if umap and dpg.does_item_exist(TAG_OT_MVA_UMAP_SECTION_GROUP):
        _remove_old_ai_buttons(TAG_OT_MVA_UMAP_SECTION_GROUP, "MVA_UMAP_AI_Button_") # 해당 섹션 그룹 내 버튼 정리
        try:
            fig_u, ax_u = plt.subplots(figsize=(6, 4.5)) # figsize 조정
            reducer_u = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
            embed_u = reducer_u.fit_transform(data_for_reduction)
            ax_u.scatter(embed_u[:,0], embed_u[:,1], c=outlier_labels, cmap='coolwarm', s=15, alpha=0.7)
            ax_u.set_title("UMAP Projection", fontsize=10)
            # ... (legend 등)
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor=plt.cm.coolwarm(0.0), markersize=5), plt.Line2D([0], [0], marker='o', color='w', label='Outlier', markerfacecolor=plt.cm.coolwarm(1.0), markersize=5)]
            ax_u.legend(handles=legend_elements, loc='best', fontsize=7)
            plt.tight_layout(); res_u = plot_func(fig_u); plt.close(fig_u)
            
            t_u,w_u,h_u,b_u = (res_u if res_u and len(res_u)==4 else (None,0,0,None))
            _log_mva(f"UMAP plot_texture_func result: tex={t_u is not None}, w={w_u}, h={h_u}, bytes={b_u is not None}")
            _plot_cleanup_and_set(TAG_OT_MVA_UMAP_PLOT_IMAGE, '_mva_active_umap_texture_id', t_u, 'default_umap_texture_tag', w_u, h_u)
            if b_u and t_u:
                btn_tag_u = f"MVA_UMAP_AI_Button_{dpg.generate_uuid()}"
                act_u = functools.partial(utils.confirm_and_run_ai_analysis,b_u,"MVA_UMAP",btn_tag_u,main_cb)
                dpg.add_button(label="💡 Analyze UMAP", tag=btn_tag_u, parent=TAG_OT_MVA_UMAP_SECTION_GROUP, width=-1, height=30, callback=lambda s,a,ud:act_u())
        except Exception as e: _log_mva(f"UMAP gen error: {e}\n{traceback.format_exc()}")

    # PCA
    if dpg.does_item_exist(TAG_OT_MVA_PCA_SECTION_GROUP):
        _remove_old_ai_buttons(TAG_OT_MVA_PCA_SECTION_GROUP, "MVA_PCA_AI_Button_")
        try:
            fig_p, ax_p = plt.subplots(figsize=(6, 4.5)) # figsize 조정
            pca_m = PCA(n_components=2, random_state=42)
            embed_p = pca_m.fit_transform(data_for_reduction)
            ax_p.scatter(embed_p[:,0], embed_p[:,1], c=outlier_labels, cmap='viridis', s=15, alpha=0.7)
            ax_p.set_title("PCA Projection", fontsize=10); ax_p.set_xlabel(f"PC1 ({pca_m.explained_variance_ratio_[0]*100:.1f}%)"); ax_p.set_ylabel(f"PC2 ({pca_m.explained_variance_ratio_[1]*100:.1f}%)")
            # ... (legend 등)
            legend_elements_pca = [plt.Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor=plt.cm.viridis(0.2), markersize=5), plt.Line2D([0], [0], marker='o', color='w', label='Outlier', markerfacecolor=plt.cm.viridis(0.8), markersize=5)]
            ax_p.legend(handles=legend_elements_pca, loc='best', fontsize=7)
            plt.tight_layout(); res_p = plot_func(fig_p); plt.close(fig_p)
            
            t_p,w_p,h_p,b_p = (res_p if res_p and len(res_p)==4 else (None,0,0,None))
            _log_mva(f"PCA plot_texture_func result: tex={t_p is not None}, w={w_p}, h={h_p}, bytes={b_p is not None}")
            _plot_cleanup_and_set(TAG_OT_MVA_PCA_PLOT_IMAGE, '_mva_active_pca_texture_id', t_p, 'default_pca_texture_tag', w_p, h_p)
            if b_p and t_p:
                btn_tag_p = f"MVA_PCA_AI_Button_{dpg.generate_uuid()}"
                act_p = functools.partial(utils.confirm_and_run_ai_analysis,b_p,"MVA_PCA",btn_tag_p,main_cb)
                dpg.add_button(label="💡 Analyze PCA", tag=btn_tag_p, parent=TAG_OT_MVA_PCA_SECTION_GROUP, width=-1, height=30, callback=lambda s,a,ud:act_p())
        except Exception as e: _log_mva(f"PCA gen error: {e}\n{traceback.format_exc()}")

def _generate_mva_shap_plot_for_instance(original_idx: Any):
    global _mva_active_shap_texture_id # DPG 텍스처 ID 관리를 위함
    _log_mva(f"Attempting SHAP plot generation for instance {original_idx}")

    # SHAP 분석 버튼 및 상태 메시지 태그 정의
    shap_ai_button_fixed_alias = "mva_shap_plot_ai_analyze_button"
    status_text_tag = "mva_shap_status_text" # UI 생성 시 이 태그로 add_text가 되어 있어야 함

    # 이전 AI 분석 버튼이 있다면 삭제
    if dpg.is_dearpygui_running() and dpg.does_item_exist(shap_ai_button_fixed_alias):
        try:
            dpg.delete_item(shap_ai_button_fixed_alias)
        except Exception as e_del_btn:
            _log_mva(f"Could not delete old SHAP AI button '{shap_ai_button_fixed_alias}': {e_del_btn}")

    # 이전 SHAP 플롯 및 상태 메시지 초기화
    _clear_mva_shap_plot() # 이 함수는 내부적으로 _plot_cleanup_and_set을 호출하여 이미지와 상태 텍스트를 정리

    # 필수 유틸리티 및 데이터 존재 여부 확인
    if not _shared_utils_mva:
        _log_mva("Shared utils missing for SHAP plot generation.")
        if dpg.is_dearpygui_running() and dpg.does_item_exist(status_text_tag):
            dpg.set_value(status_text_tag, "SHAP Error: Shared utilities not available.")
            dpg.configure_item(status_text_tag, show=True, color=[255, 0, 0])
        return

    main_cb = _shared_utils_mva.get('main_app_callbacks')
    plot_func = _shared_utils_mva.get('plot_to_dpg_texture_func')
    current_df_for_shap_values = _df_with_mva_outliers # SHAP 값 계산에 사용될 DataFrame (mva_outlier_score 등 포함 가능)
                                                 # SHAP explainer 학습 시 사용된 특성들이 있어야 함.

    # SHAP 실행을 위한 전제 조건 확인
    prereqs_met = (
        shap and plot_func and main_cb and # shap 라이브러리, plot 함수, 메인 콜백
        current_df_for_shap_values is not None and
        original_idx in current_df_for_shap_values.index and # 선택된 인스턴스가 DataFrame에 있는지
        _mva_shap_explainer and # SHAP explainer가 초기화되었는지
        _mva_selected_columns_for_detection and # MVA 탐지에 사용된 컬럼(SHAP 특성) 목록이 있는지
        len(_mva_selected_columns_for_detection) > 0
    )

    if not prereqs_met:
        _log_mva("SHAP plot prerequisites not met. Cannot generate plot.")
        if dpg.is_dearpygui_running() and dpg.does_item_exist(status_text_tag):
            msg = "SHAP prerequisites not met. Possible_reasons:\n"
            if not shap: msg += "- SHAP library not loaded.\n"
            if not _mva_shap_explainer: msg += "- SHAP explainer not initialized (run MVA detection).\n"
            if not _mva_selected_columns_for_detection: msg += "- No features selected for MVA/SHAP.\n"
            if current_df_for_shap_values is None or original_idx not in current_df_for_shap_values.index:
                msg += "- Selected instance data not found.\n"
            dpg.set_value(status_text_tag, msg)
            dpg.configure_item(status_text_tag, show=True, color=[200, 200, 100])
        # _clear_mva_shap_plot()가 이미 호출되었으므로, 여기서 이미지 위젯을 숨기거나 기본값으로 설정할 필요 없음
        return

    fig_s = None # Matplotlib Figure 객체 초기화
    try:
        # 선택된 인스턴스의 데이터 추출 (MVA 탐지에 사용된 특성들만)
        instance_series = current_df_for_shap_values.loc[original_idx, _mva_selected_columns_for_detection]
        # SHAP 값 계산을 위해 DataFrame 형태로 변환
        instance_df_for_shap_calc = pd.DataFrame([instance_series.values], columns=_mva_selected_columns_for_detection)

        # NaN 값 처리 (이론적으로는 MVA 탐지 과정에서 처리되었어야 하나, 안전을 위해 확인)
        if instance_df_for_shap_calc.isnull().values.any():
            _log_mva(f"NaNs found in instance data for SHAP (idx: {original_idx}). Filling with 0 for SHAP calculation. This should ideally not happen.")
            # MVA 탐지 시 사용했던 'df_for_detection'의 평균 등으로 채우는 것이 더 정확할 수 있으나,
            # 여기서는 explainer가 이미 학습된 상태이므로, SHAP 값 계산 시 NaN을 허용하지 않는다면 0으로 채움.
            # IsolationForest는 NaN을 처리하지 못하므로, _mva_shap_explainer 학습 데이터에는 NaN이 없었어야 함.
            instance_df_for_shap_calc.fillna(0, inplace=True)

        # SHAP 값 계산
        # TreeExplainer의 shap_values는 때때로 여러 출력에 대한 값 리스트를 반환할 수 있음 (특히 multi-output 모델).
        # IsolationForest는 단일 출력이므로, 결과 배열의 형태를 확인하고 적절히 인덱싱 필요.
        shap_values_raw = _mva_shap_explainer.shap_values(instance_df_for_shap_calc)
        
        # shap_values_raw의 형태에 따라 적절한 SHAP 값 추출
        # 통상적인 경우 (단일 인스턴스, 단일 출력), shap_values_raw는 (1, num_features) 형태의 2D 배열이거나,
        # explainer 구현에 따라 (num_outputs, 1, num_features) 또는 리스트 형태일 수 있음.
        # PyOD IForest의 경우 내부 scikit-learn IsolationForest를 사용하며, shap_values는 보통 (n_samples, n_features)
        shap_vals_for_exp = shap_values_raw[0, :] if isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 2 and shap_values_raw.shape[0] == 1 else shap_values_raw

        # expected_value (base value) 가져오기
        # 이 또한 explainer 구현 및 모델에 따라 스칼라 또는 배열일 수 있음
        expected_value = _mva_shap_explainer.expected_value
        if hasattr(expected_value, "__len__") and not isinstance(expected_value, (str, bytes)): # 배열 형태인지 확인
             # IsolationForest의 경우 expected_value가 단일 값 배열(예: [0.5])일 수 있음
            base_value_for_exp = expected_value[0] if len(expected_value) > 0 else 0.0
        else:
            base_value_for_exp = expected_value # 스칼라 값

        # SHAP Explanation 객체 생성
        shap_explanation_obj = shap.Explanation(
            values=shap_vals_for_exp,
            base_values=base_value_for_exp,
            data=instance_df_for_shap_calc.iloc[0].values, # 원본 특성 값
            feature_names=_mva_selected_columns_for_detection
        )
        
        num_features_to_display = min(len(_mva_selected_columns_for_detection), 15) # Waterfall 플롯에 표시할 최대 특성 수

        # --- Matplotlib Figure 크기 동적 설정 ---
        parent_width_pixels = 400 # DearPyGui 아이템 너비 측정 실패 시 기본값
        dpi = 90 # plot_to_dpg_texture_func의 기본 DPI와 맞춤

        # Matplotlib 그림 너비 계산 (이전 대화의 개선된 로직 통합)
        shap_group_tag_for_width = TAG_OT_MVA_SHAP_PARENT_GROUP
        table_item_tag_for_width = None
        # SHAP 이미지가 속한 테이블 셀은 전체 테이블 너비의 약 55%를 차지하도록 설정됨
        if dpg.is_dearpygui_running() and dpg.does_item_exist(shap_group_tag_for_width):
            cell_item_tag = dpg.get_item_parent(shap_group_tag_for_width)
            if cell_item_tag and dpg.does_item_exist(cell_item_tag):
                row_item_tag = dpg.get_item_parent(cell_item_tag)
                if row_item_tag and dpg.does_item_exist(row_item_tag):
                    table_item_tag_for_width = dpg.get_item_parent(row_item_tag)
        
        calculated_from_table = False
        if table_item_tag_for_width and dpg.does_item_exist(table_item_tag_for_width) and dpg.get_item_info(table_item_tag_for_width)['type'] == 'mvAppItemType::mvTable':
            table_width = dpg.get_item_width(table_item_tag_for_width)
            if table_width and table_width > 0:
                parent_width_pixels = int(table_width * 0.52) # 55%에서 스크롤바, 내부패딩 등 여유 더 제외
                calculated_from_table = True
                _log_mva(f"Matplotlib parent_width_pixels for SHAP based on table '{table_item_tag_for_width}' (table_width {table_width}): {parent_width_pixels}")

        if not calculated_from_table:
            content_area_tag = "content_area" # main_app.py에 정의된 태그
            content_area_width = 0
            if dpg.is_dearpygui_running() and dpg.does_item_exist(content_area_tag):
                content_area_width = dpg.get_item_width(content_area_tag)
            
            if content_area_width and content_area_width > 0:
                # content_area 너비의 약 45-50% (SHAP 영역이 차지하는 대략적 비율)
                parent_width_pixels = int(content_area_width * 0.45)
                _log_mva(f"Matplotlib parent_width_pixels for SHAP based on '{content_area_tag}' (width {content_area_width}): {parent_width_pixels}")
            else:
                parent_width_pixels = 700 # Fallback 값 증가 (이전 400)
                _log_mva(f"Matplotlib parent_width_pixels for SHAP fallback to: {parent_width_pixels}")
        _log_mva(f"Final parent_width_pixels for Matplotlib SHAP plot: {parent_width_pixels}")

        fig_width_inches = max(5.0, min(12.0, (parent_width_pixels / dpi) if parent_width_pixels > 0 else 5.0)) # 최소 5인치, 최대 12인치

        # Matplotlib 그림 높이 계산 (이전 대화의 개선된 로직 통합)
        shap_child_window_actual_height_px = 500 # 'shap_content_child_window'의 고정 높이
        # 버튼(30), 스페이서(5), 상태텍스트(20 가정), 상하패딩(20*2=40 가정) -> 30+5+20+40 = 95
        # 실제 이미지 가용 높이는 shap_child_window_actual_height_px - (다른 요소들 높이 합)
        image_drawable_height_px = shap_child_window_actual_height_px - 100 # 여유 공간 및 기타 UI 요소 높이 제외 (대략적)
        max_fig_height_inches = image_drawable_height_px / dpi if image_drawable_height_px > 0 else 3.0

        # 특성 수에 따라 높이 계산, 상한선 적용, 계수 조정 (예: 0.3~0.45)
        calculated_fig_height_inches = max(3.0, num_features_to_display * 0.40) # 최소 3인치
        fig_height_inches = min(calculated_fig_height_inches, max_fig_height_inches)
        fig_height_inches = max(fig_height_inches, 3.0) # 최종적으로 최소 3인치 보장

        _log_mva(f"SHAP plot Matplotlib dimensions: {fig_width_inches:.1f}W x {fig_height_inches:.1f}H inches "
                 f"(n_feat_disp={num_features_to_display}, max_fig_H_inches_allowed={max_fig_height_inches:.1f})")
        
        plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=dpi) # DPI 명시
        shap.waterfall_plot(shap_explanation_obj, max_display=num_features_to_display, show=False)
        
        fig_s = plt.gcf() # 현재 Figure 객체 가져오기
        
        # 플롯 제목 설정
        title_str = f"SHAP Analysis - Instance {original_idx}"
        if fig_s.axes: # 축이 있다면 첫 번째 축에 제목 설정
            fig_s.axes[0].set_title(title_str, fontsize=10)
        else: # 축이 없다면 Figure 전체에 제목 설정
            fig_s.suptitle(title_str, fontsize=10)
            
        # 내용이 잘리지 않도록 여백 조정 (값은 실험적으로 최적화)
        # left: 특성 이름, bottom: x축 레이블, right/top: 여유 공간
        plt.subplots_adjust(left=0.35, right=0.95, top=0.90, bottom=0.15 if num_features_to_display > 3 else 0.10)
        # 또는 plt.tight_layout(pad=1.2, h_pad=1.0, w_pad=1.0) # tight_layout은 때로 figsize를 변경할 수 있음

        # Matplotlib Figure를 DPG 텍스처로 변환
        res_s = plot_func(fig_s) # _s5_plot_to_dpg_texture_mva 호출
        t_s, w_s, h_s, b_s = (res_s if res_s and len(res_s) == 4 else (None, 0, 0, None))
        _log_mva(f"plot_func for SHAP returned: tex_exists={t_s is not None}, w={w_s}, h={h_s}, bytes_exist={b_s is not None}")

        # AI 분석 버튼 추가 (성공적으로 이미지 바이트가 생성된 경우)
        if b_s and t_s and dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_MVA_SHAP_PARENT_GROUP):
            # functools.partial을 사용하여 콜백 함수와 인자들을 미리 준비
            action_for_ai_button = functools.partial(
                utils.confirm_and_run_ai_analysis, # utils.py의 함수
                b_s, # image_bytes
                f"Outlier_Analysis_SHAP_Instance_{original_idx}", # chart_name
                shap_ai_button_fixed_alias, # ai_button_tag
                main_cb # main_callbacks
            )
            # 버튼 추가 (SHAP 이미지 위젯 위에 배치)
            dpg.add_button(label="💡 Analyze SHAP Plot", tag=shap_ai_button_fixed_alias,
                           parent=TAG_OT_MVA_SHAP_PARENT_GROUP, width=-1, height=30,
                           callback=lambda sender, app_data, user_data: action_for_ai_button(), 
                           before=TAG_OT_MVA_SHAP_PLOT_IMAGE) # 이미지 앞에 버튼 추가
            # 버튼과 이미지 사이에 약간의 간격 추가
            dpg.add_spacer(height=5, parent=TAG_OT_MVA_SHAP_PARENT_GROUP, before=TAG_OT_MVA_SHAP_PLOT_IMAGE)

        # DPG 이미지 위젯 업데이트
        # _plot_cleanup_and_set 함수는 SHAP 이미지의 경우 종횡비를 무시하고 영역에 맞추도록 수정되었다고 가정
        _plot_cleanup_and_set(
            image_widget_tag=TAG_OT_MVA_SHAP_PLOT_IMAGE,
            active_texture_id_var_name='_mva_active_shap_texture_id', # 전역변수 이름
            new_texture_tag=t_s,
            default_texture_key='default_shap_plot_texture_tag', # shared_utils_mva에 정의된 키
            w=w_s, # Matplotlib에서 생성된 이미지의 원본 픽셀 너비
            h=h_s  # Matplotlib에서 생성된 이미지의 원본 픽셀 높이
        )

        if t_s: # 텍스처가 성공적으로 생성되었다면
            _log_mva(f"SHAP waterfall plot for instance {original_idx} generated and displayed.")
            if dpg.is_dearpygui_running() and dpg.does_item_exist(status_text_tag):
                dpg.configure_item(status_text_tag, show=False) # 성공 메시지 대신 플롯이 보이므로 상태 텍스트 숨김
        else: # 텍스처 생성 실패
            _log_mva(f"SHAP waterfall plot texture generation failed for instance {original_idx}.")
            if dpg.is_dearpygui_running() and dpg.does_item_exist(status_text_tag):
                dpg.set_value(status_text_tag, "SHAP plot generation failed (texture error).")
                dpg.configure_item(status_text_tag, show=True, color=[255, 0, 0])
    
    except Exception as e_shap_plot:
        _log_mva(f"Error during SHAP plot generation for instance {original_idx}: {e_shap_plot}\n{traceback.format_exc()}")
        if dpg.is_dearpygui_running() and dpg.does_item_exist(status_text_tag):
            error_message_display = f"SHAP Plot Error: {str(e_shap_plot)[:150]}" # 오류 메시지 간결화
            dpg.set_value(status_text_tag, error_message_display)
            dpg.configure_item(status_text_tag, show=True, color=[255, 0, 0])
        # 오류 발생 시 이미지 위젯을 숨기거나 기본 이미지로 되돌릴 수 있음 (이미 _clear_mva_shap_plot에서 처리)
    finally:
        if fig_s: # Matplotlib Figure 객체가 생성되었다면 닫아서 리소스 해제
            plt.close(fig_s)
            _log_mva(f"Matplotlib figure for SHAP instance {original_idx} closed.")


def _find_top_gap_variables_for_boxplot() -> List[str]:
    if _df_with_mva_outliers is None or 'mva_is_outlier' not in _df_with_mva_outliers.columns: return []
    eligible = [c for c in _get_eligible_numeric_cols_for_mva(_df_with_mva_outliers) if c not in ['mva_outlier_score', 'mva_is_outlier']]
    if not eligible: return []
    gaps = []
    for col in eligible:
        norm_data = _df_with_mva_outliers.loc[_df_with_mva_outliers['mva_is_outlier'] == False, col].dropna()
        out_data = _df_with_mva_outliers.loc[_df_with_mva_outliers['mva_is_outlier'] == True, col].dropna()
        if len(norm_data) < 2 or len(out_data) < 2: continue
        gap = abs(norm_data.median() - out_data.median())
        if pd.notna(gap) and gap > 1e-6: gaps.append((col, gap))
    gaps.sort(key=lambda x: x[1], reverse=True)
    return [var for var, _ in gaps[:TOP_N_VARIABLES_FOR_BOXPLOT]]

def _generate_mva_boxplots_for_comparison():
    global _mva_boxplot_image_tags
    _log_mva("Attempting MVA boxplots...")
    _clear_mva_boxplots()
    top_vars = _find_top_gap_variables_for_boxplot()
    if not (_shared_utils_mva and top_vars and _df_with_mva_outliers is not None):
        if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_MVA_BOXPLOT_GROUP):
            if dpg.does_item_exist(TAG_OT_MVA_BOXPLOT_GROUP):
                 dpg.add_text("No data for MVA boxplots.", parent=TAG_OT_MVA_BOXPLOT_GROUP)
        return
    main_cb, plot_func = _shared_utils_mva.get('main_app_callbacks'), _shared_utils_mva.get('plot_to_dpg_texture_func')
    if not (plot_func and main_cb): 
        _log_mva("Plot function or main_callbacks missing for MVA boxplots.")
        return

    cols_per_row = 2; parent_cw = TAG_OT_MVA_BOXPLOT_GROUP
    if not dpg.does_item_exist(parent_cw): 
        _log_mva(f"Parent container {parent_cw} for boxplots does not exist.")
        return
        
    parent_w = dpg.get_item_width(parent_cw)
    if parent_w is None or parent_w <= 0: 
        _log_mva(f"Warning: Could not get valid width for {parent_cw}. Using fallback width 700.")
        parent_w = 700 
    
    plot_grp_w = (parent_w - (10 * (cols_per_row +1))) // cols_per_row 
    plot_grp_w = max(250, plot_grp_w) 

    _log_mva(f"Generating MVA boxplots for vars: {top_vars}. Parent width: {parent_w}, Plot group width: {plot_grp_w}")

    for i in range(0, len(top_vars), cols_per_row):
        with dpg.group(horizontal=True, parent=parent_cw) as row_grp:
            for j in range(cols_per_row):
                if i + j < len(top_vars):
                    var = top_vars[i+j]
                    with dpg.group(horizontal=False, parent=row_grp, width=plot_grp_w) as plot_cell_grp:
                        fig, ax = None, None 
                        t_b, w_b, h_b, b_b = None, 0, 0, None 
                        try:
                            fig_width_inches = max(5, plot_grp_w / 80) 
                            fig, ax = plt.subplots(figsize=(fig_width_inches, 4.0)) 
                            
                            norm_s = _df_with_mva_outliers.loc[~_df_with_mva_outliers['mva_is_outlier'], var].dropna()
                            out_s = _df_with_mva_outliers.loc[_df_with_mva_outliers['mva_is_outlier'], var].dropna()
                            data, lbls = [], []
                            if not norm_s.empty: data.append(norm_s); lbls.append("Normal")
                            if not out_s.empty: data.append(out_s); lbls.append("Outlier")
                            
                            if data:
                                try:
                                    bp_data = ax.boxplot(data, labels=lbls, patch_artist=True, medianprops={'color':'red','lw':1.5})
                                    for k_patch, p_color in enumerate(['lightblue', 'lightcoral'][:len(data)]): bp_data['boxes'][k_patch].set_facecolor(p_color)
                                    ax.set_title(f"{var}", fontsize=9); ax.grid(True,ls='--',alpha=0.6); ax.tick_params(axis='both', which='major', labelsize=7)
                                except Exception as e_bp:
                                    _log_mva(f"Error during ax.boxplot for variable '{var}': {e_bp}")
                                    ax.text(0.5,0.5,f"Boxplot error for {var}",ha='center',va='center', color='red')
                            else: 
                                ax.text(0.5,0.5,f"No data for {var}",ha='center',va='center')
                            
                            plt.tight_layout(); res_b = plot_func(fig); 
                            t_b,w_b,h_b,b_b = (res_b if res_b and len(res_b)==4 else (None,0,0,None))
                            _log_mva(f"Boxplot for '{var}': tex={t_b is not None}, w={w_b}, h={h_b}, bytes={b_b is not None}")

                            if t_b and w_b > 0 and dpg.does_item_exist(plot_cell_grp): 
                                cell_w = dpg.get_item_width(plot_cell_grp) 
                                disp_w = min(w_b, cell_w - 10 if cell_w > 10 else cell_w)
                                disp_h = int(h_b * (disp_w/w_b)) if w_b > 0 else h_b
                                dpg.add_image(t_b, width=disp_w, height=disp_h, parent=plot_cell_grp)
                                _mva_boxplot_image_tags.append(t_b) 
                                dpg.add_spacer(height=5, parent=plot_cell_grp) # Add spacer after image
                                # AI 분석 버튼 관련 코드 제거됨
                            else: 
                                if dpg.does_item_exist(plot_cell_grp):
                                     dpg.add_text(f"Error generating plot for '{var}'.",parent=plot_cell_grp, color=[255,0,0])
                        except Exception as e_plot_gen: 
                            _log_mva(f"General error generating boxplot for {var}: {e_plot_gen}\n{traceback.format_exc()}")
                            if dpg.does_item_exist(plot_cell_grp):
                                 dpg.add_text(f"Plot error for {var}.",parent=plot_cell_grp, color=[255,0,0])
                        finally:
                            if fig: plt.close(fig) 

                elif cols_per_row > 1: 
                    with dpg.group(horizontal=False, parent=row_grp, width=plot_grp_w): 
                        dpg.add_spacer(height=1)
        dpg.add_spacer(height=10, parent=parent_cw)

def create_multivariate_ui(parent_tab_bar_tag: str, shared_utilities: dict):
    global _shared_utils_mva
    _shared_utils_mva = shared_utilities
    if not all([PyOD_IForest, umap, shap]):
        _log_mva("One or more MVA libraries (PyOD, UMAP, SHAP) are missing. MVA tab might be limited.")

    with dpg.tab(label="Multivariate Outlier Detection", tag=TAG_OT_MULTIVARIATE_TAB, parent=parent_tab_bar_tag):
        dpg.add_text("1. Configure & Run Multivariate Detection (Isolation Forest)", color=[255, 255, 0])
        with dpg.group(horizontal=True):
            dpg.add_text("Column Selection Mode:")
            dpg.add_radio_button(["All Numeric", "Recommended"], tag=TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO, default_value=_mva_column_selection_mode, horizontal=True, callback=_on_mva_col_selection_mode_change)
        with dpg.group(horizontal=True):
            dpg.add_text("Contamination (0.0-0.5):")
            dpg.add_input_float(tag=TAG_OT_MVA_CONTAMINATION_INPUT, width=120, default_value=_mva_contamination, min_value=0.0001, max_value=0.5, step=0.01, format="%.4f", callback=_on_mva_contamination_change)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Run Multivariate Detection", tag=TAG_OT_MVA_DETECT_BUTTON, width=-1, height=30, callback=_run_mva_outlier_detection_logic, enabled=bool(PyOD_IForest))
        dpg.add_separator()

        with dpg.tab_bar(tag="mva_results_tab_bar"):
            with dpg.tab(label="Overview: UMAP & PCA", tag="mva_tab_overview"):
                with dpg.group(tag="mva_summary_info_group"):
                    dpg.add_text("Detection Summary:", color=[255, 255, 0])
                    dpg.add_text("Run detection for summary.", tag="mva_summary_text_status")
                    dpg.add_text("", tag="mva_summary_text_used_features"); dpg.add_text("", tag="mva_summary_text_detected_outliers"); dpg.add_text("", tag="mva_summary_text_outlier_ratio")
                dpg.add_separator()
                dpg.add_text("2. UMAP & PCA Projection (Outliers Highlighted)", color=[255, 255, 0])
                
                # UMAP & PCA 섹션을 위한 테이블 레이아웃
                with dpg.table(header_row=False, borders_innerH=False, borders_outerH=False, borders_innerV=False, borders_outerV=False,
                               resizable=True, policy=dpg.mvTable_SizingStretchProp,
                               parent="mva_tab_overview"): # 부모는 Overview 탭
                    dpg.add_table_column(label="UMAP Section", width_stretch=True) # 너비 자동 분배
                    dpg.add_table_column(label="PCA Section", width_stretch=True)  # 너비 자동 분배

                    with dpg.table_row():
                        with dpg.table_cell(): # UMAP Cell
                            with dpg.group(tag=TAG_OT_MVA_UMAP_SECTION_GROUP, horizontal=False): 
                                dpg.add_text("UMAP Projection", color=[200,200,200] if umap else [255,100,100])
                                default_umap_tex = _shared_utils_mva.get('default_umap_texture_tag') if _shared_utils_mva else None
                                cfg_u_w, cfg_u_h = (10,10) # 기본값
                                if default_umap_tex and dpg.does_item_exist(default_umap_tex):
                                    cfg_u = dpg.get_item_configuration(default_umap_tex)
                                    cfg_u_w, cfg_u_h = cfg_u.get('width',200), cfg_u.get('height',150)
                                
                                dpg.add_image(texture_tag=default_umap_tex or "", tag=TAG_OT_MVA_UMAP_PLOT_IMAGE, 
                                              width=cfg_u_w, height=cfg_u_h, show=bool(umap))
                                if not umap: dpg.add_text("UMAP N/A (library missing)", color=[255,100,100])
                                # AI 버튼은 _generate_mva_umap_pca_plots 에서 이 그룹에 동적으로 추가됨

                        with dpg.table_cell(): # PCA Cell
                            with dpg.group(tag=TAG_OT_MVA_PCA_SECTION_GROUP, horizontal=False):
                                dpg.add_text("PCA Projection", color=[200,200,200])
                                default_pca_tex = _shared_utils_mva.get('default_pca_texture_tag') if _shared_utils_mva else None
                                cfg_p_w, cfg_p_h = (10,10) # 기본값
                                if default_pca_tex and dpg.does_item_exist(default_pca_tex):
                                    cfg_p = dpg.get_item_configuration(default_pca_tex)
                                    cfg_p_w, cfg_p_h = cfg_p.get('width',200), cfg_p.get('height',150)

                                dpg.add_image(texture_tag=default_pca_tex or "", tag=TAG_OT_MVA_PCA_PLOT_IMAGE, 
                                              width=cfg_p_w, height=cfg_p_h, show=True)
                                # AI 버튼은 _generate_mva_umap_pca_plots 에서 이 그룹에 동적으로 추가됨
            # ... (Detected Instances & Details 탭, Variable Box Plots 탭은 기존과 거의 동일하게 유지)
            with dpg.tab(label="Detected Instances & Details", tag="mva_tab_details"):
                with dpg.group():
                    dpg.add_text("Bulk Treatment for All Detected Outliers", color=[200, 200, 0])
                    with dpg.group(horizontal=True):
                        dpg.add_combo(
                            items=["--- Select Bulk Action ---", "Remove All Detected", "Treat All as Missing", "Reset All Selections"],
                            tag=TAG_MVA_BULK_TREATMENT_COMBO,
                            default_value="--- Select Bulk Action ---",
                            width=-150
                        )
                        dpg.add_button(
                            label="Execute Bulk Change",
                            tag=TAG_MVA_EXECUTE_BULK_CHANGE_BUTTON,
                            callback=_execute_bulk_treatment_change,
                            width=-1
                        )
                dpg.add_spacer(height=5)
                dpg.add_text("4. Detected Multivariate Outlier Instances (Max 30, by Score)", color=[255, 255, 0])
                with dpg.table(tag=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=220, borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                    dpg.add_table_column(label="Info", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, width_stretch=True)
                    with dpg.table_row(parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE):
                        dpg.add_text("Run MVA detection.")

                # 'Apply MVA Treatments' 버튼을 테이블 위젯 *밖으로* 이동시킵니다.
                dpg.add_button(label="Apply MVA Treatments", tag=TAG_OT_MVA_APPLY_TREATMENT_BUTTON, width=-1, height=30,
                               callback=_apply_mva_treatment_logic)
                
                dpg.add_separator(parent="mva_tab_details")

                # SHAP 섹션과 통계 테이블 섹션을 위한 테이블 레이아웃
                with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp,
                               borders_innerH=True, borders_outerH=True, # 명확성을 위해 테두리 추가 (선택 사항)
                               parent="mva_tab_details"): # 부모 명시
                    
                    # 테이블 열 정의 (예: SHAP 55%, 통계 45% 비율로 너비 설정)
                    dpg.add_table_column(label="SHAP Area", width_stretch=True, init_width_or_weight=0.55)
                    dpg.add_table_column(label="Statistics Area", width_stretch=True, init_width_or_weight=0.45)

                    with dpg.table_row():
                        # --- SHAP 섹션 (왼쪽 셀) ---
                        with dpg.table_cell():
                            with dpg.group(tag=TAG_OT_MVA_SHAP_PARENT_GROUP):
                                dpg.add_text("5. SHAP Values for Selected Instance", color=[255, 255, 0])
                                
                                # child_window 생성 - autosize_x=True로 자동 크기 조정
                                with dpg.child_window(tag="shap_content_child_window", 
                                                    border=False,
                                                    autosize_x=True,  # X축 자동 크기
                                                    autosize_y=False,  # Y축은 고정
                                                    height=500):
                                    
                                    # SHAP 상태 메시지
                                    default_shap_status_text = "Select an instance to see SHAP details."
                                    if not shap: default_shap_status_text = "SHAP library not available."
                                    dpg.add_text(default_shap_status_text, tag="mva_shap_status_text",
                                                 show=True, color=[200,200,200])

                                    # SHAP 이미지 위젯 - width=-1로 부모에 맞춤
                                    default_shap_tex = _shared_utils_mva.get('default_shap_plot_texture_tag') if _shared_utils_mva else None
                                    
                                    dpg.add_image(texture_tag=default_shap_tex or "", 
                                                tag=TAG_OT_MVA_SHAP_PLOT_IMAGE,
                                                width=-1,  # 부모 child_window 너비에 자동 맞춤
                                                height=-1,  # 높이도 자동
                                                show=bool(shap))# wrap 활성화

                                # SHAP 이미지 위젯 (크기는 _plot_cleanup_and_set에서 조정됨)
                                # default_shap_tex = _shared_utils_mva.get('default_shap_plot_texture_tag') if _shared_utils_mva else None
                                # cfg_s_w, cfg_s_h = (250, 200) # 초기 플레이스홀더 크기
                                # if default_shap_tex and dpg.does_item_exist(default_shap_tex):
                                #     cfg_s = dpg.get_item_configuration(default_shap_tex)
                                #     cfg_s_w, cfg_s_h = cfg_s.get('width',250), cfg_s.get('height',200)
                                
                                # dpg.add_image(texture_tag=default_shap_tex or "", tag=TAG_OT_MVA_SHAP_PLOT_IMAGE,
                                #               width=cfg_s_w, height=cfg_s_h, show=bool(shap))
                                # AI 분석 버튼은 _generate_mva_shap_plot_for_instance에서 TAG_OT_MVA_SHAP_PARENT_GROUP에 추가됨
                                # 버튼의 width=-1은 TAG_OT_MVA_SHAP_PARENT_GROUP (즉, 이 테이블 셀)의 너비를 채우게 됨.

                        # --- 통계 테이블 섹션 (오른쪽 셀) ---
                        with dpg.table_cell():
                            with dpg.group(): # 통계 테이블을 감싸는 그룹 (선택 사항)
                                dpg.add_text("6. Statistics for Selected Instance", color=[255, 255, 0])
                                with dpg.table(tag=TAG_OT_MVA_INSTANCE_STATS_TABLE, header_row=True, resizable=True,
                                               policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=450, # 스크롤 위해 높이 고정
                                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                                   dpg.add_table_column(label="Info", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, width_stretch=True)
                                   with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE):
                                       dpg.add_text("Select an outlier instance for details.")
            
            # ... (Variable Box Plots 탭은 기존과 거의 동일하게 유지)
            with dpg.tab(label="Variable Box Plots", tag="mva_tab_boxplots"):
                dpg.add_text("3. Variable Comparison: Outlier vs. Normal (Top 10 by Median Gap)", color=[255, 255, 0])
                with dpg.child_window(tag=TAG_OT_MVA_BOXPLOT_GROUP, border=True): # 스크롤 가능하도록 child_window 사용
                    dpg.add_text("Run detection to see boxplots.")

def update_multivariate_ui(df_input: Optional[pd.DataFrame], shared_utilities: dict, is_new_data: bool):
    global _shared_utils_mva, _current_df_for_mva
    if not dpg.is_dearpygui_running(): return
    _shared_utils_mva = shared_utilities
    if not dpg.does_item_exist(TAG_OT_MULTIVARIATE_TAB): return
    _current_df_for_mva = df_input
    if _current_df_for_mva is None or is_new_data:
        reset_multivariate_state_internal(called_from_parent_reset=False)
    if _current_df_for_mva is not None or is_new_data :
        _update_selected_columns_for_mva_detection()

def reset_multivariate_state_internal(called_from_parent_reset=True):
    global _df_with_mva_outliers, _mva_model, _mva_shap_explainer, _mva_eligible_numeric_cols, \
           _mva_selected_columns_for_detection, _mva_column_selection_mode, _mva_contamination, \
           _mva_outlier_instances_summary, _mva_selected_outlier_instance_idx, \
           _mva_all_selectable_tags_in_instances_table, _mva_top_gap_vars_for_boxplot

    _df_with_mva_outliers, _mva_model, _mva_shap_explainer = None, None, None
    _mva_eligible_numeric_cols, _mva_selected_columns_for_detection = [], []
    _mva_column_selection_mode, _mva_contamination = "All Numeric", DEFAULT_MVA_CONTAMINATION # 기본값
    _mva_outlier_instances_summary, _mva_all_selectable_tags_in_instances_table, _mva_top_gap_vars_for_boxplot = [], [], []
    _mva_selected_outlier_instance_idx = None
    _mva_treatment_selections.clear()

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO): dpg.set_value(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO, _mva_column_selection_mode)
        if dpg.does_item_exist(TAG_OT_MVA_CONTAMINATION_INPUT): dpg.set_value(TAG_OT_MVA_CONTAMINATION_INPUT, _mva_contamination)
        for tag_suffix in ["status", "used_features", "detected_outliers", "outlier_ratio"]:
            if dpg.does_item_exist(f"mva_summary_text_{tag_suffix}"): dpg.set_value(f"mva_summary_text_{tag_suffix}", "" if tag_suffix !="status" else "Run detection.")
        _populate_mva_outlier_instances_table(); _clear_all_mva_visualizations()
    if not called_from_parent_reset: _log_mva("MVA outlier state reset (internal).")

def reset_multivariate_state(): reset_multivariate_state_internal(True); _log_mva("MVA outlier state reset by parent.")

def get_multivariate_settings() -> dict:
    return {
        "mva_col_sel_mode": _mva_column_selection_mode, 
        "mva_contam": _mva_contamination,
        "mva_treatments": _mva_treatment_selections.copy() # 이 줄을 추가합니다.
    }

def apply_multivariate_settings(df_input: Optional[pd.DataFrame], settings: dict, shared_utilities: dict):
    global _shared_utils_mva, _current_df_for_mva, _mva_column_selection_mode, _mva_contamination
    _shared_utils_mva = shared_utilities; _current_df_for_mva = df_input
    _mva_column_selection_mode = settings.get("mva_col_sel_mode", "All Numeric")
    _mva_contamination = settings.get("mva_contam", DEFAULT_MVA_CONTAMINATION)
    _mva_treatment_selections = settings.get("mva_treatments", {}).copy() # 이 줄을 추가합니다.

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO): dpg.set_value(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO, _mva_column_selection_mode)
        if dpg.does_item_exist(TAG_OT_MVA_CONTAMINATION_INPUT): dpg.set_value(TAG_OT_MVA_CONTAMINATION_INPUT, _mva_contamination)
        if _current_df_for_mva is not None:
            _update_selected_columns_for_mva_detection()
            _populate_mva_outlier_instances_table()
            _clear_all_mva_visualizations()
            
    _log_mva("MVA outlier settings applied. Re-run detection if needed.")

# --- 아래 함수 전체를 새로 추가합니다 ---
def _execute_bulk_treatment_change():
    """Bulk Treatment 콤보박스에서 선택된 액션을 모든 이상치에 대해 준비(stage)합니다."""
    global _mva_treatment_selections
    
    if not dpg.does_item_exist(TAG_MVA_BULK_TREATMENT_COMBO): return
        
    selected_action = dpg.get_value(TAG_MVA_BULK_TREATMENT_COMBO)
    
    if selected_action == "--- Select Bulk Action ---":
        _show_simple_modal_mva("Info", "Please select a bulk action from the dropdown menu.")
        return

    if not _mva_outlier_instances_summary:
        _show_simple_modal_mva("Info", "There are no detected outliers to apply a bulk action to.")
        dpg.set_value(TAG_MVA_BULK_TREATMENT_COMBO, "--- Select Bulk Action ---")
        return

    method_map = {
        "Remove All Detected": "Remove Row",
        "Treat All as Missing": "Treat as Missing",
        "Reset All Selections": "Do Not Treat"
    }
    target_method = method_map.get(selected_action)
    if target_method is None: return

    # 모든 탐지된 이상치에 대해 처리 방법을 상태 변수에 저장
    for instance_summary in _mva_outlier_instances_summary:
        idx = instance_summary["Original Index"]
        _mva_treatment_selections[idx] = target_method
        
    _log_mva(f"Bulk action '{selected_action}' staged for all {len(_mva_outlier_instances_summary)} detected outliers.")
    
    # 테이블 UI를 새로고침하여 변경된 드롭다운 값들을 보여줌
    _populate_mva_outlier_instances_table()
    
    # 액션 실행 후 콤보박스 초기화
    dpg.set_value(TAG_MVA_BULK_TREATMENT_COMBO, "--- Select Bulk Action ---")