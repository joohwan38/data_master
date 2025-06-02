# step_05b_multivariate_outliers.py
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

# 라이브러리 직접 임포트 (try-except 제거)
from pyod.models.iforest import IForest as PyOD_IForest
import umap
import shap


# --- DPG Tags for Multivariate Tab ---
TAG_OT_MULTIVARIATE_TAB = "step5_multivariate_outlier_tab"

# Left Panel - Top (Detection & UMAP/PCA)
TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO = "step5_ot_mva_col_selection_mode_radio"
TAG_OT_MVA_MANUAL_COLUMN_SELECTOR_GROUP = "step5_ot_mva_manual_col_selector_group"
TAG_OT_MVA_COLUMN_SELECTOR_MULTI = "step5_ot_mva_col_selector_multi"
TAG_OT_MVA_CONTAMINATION_INPUT = "step5_ot_mva_contamination_input"
TAG_OT_MVA_DETECT_BUTTON = "step5_ot_mva_detect_button"
TAG_OT_MVA_RECOMMEND_PARAMS_BUTTON = "step5_ot_mva_recommend_params_button"

TAG_OT_MVA_VISUALIZATION_GROUP = "step5_ot_mva_visualization_group" # UMAP & PCA parent
TAG_OT_MVA_UMAP_PLOT_IMAGE = "step5_ot_mva_umap_plot_image"
TAG_OT_MVA_PCA_PLOT_IMAGE = "step5_ot_mva_pca_plot_image"
# 기본 텍스처는 부모 모듈의 TAG_OT_DEFAULT_PLOT_TEXTURE_MVA 또는 TAG_OT_MVA_SHAP_DEFAULT_TEXTURE 등을 활용하거나,
# 이 모듈에서 필요시 추가 정의하고 부모 모듈의 create_ui에서 생성 요청
TAG_OT_MVA_UMAP_DEFAULT_TEXTURE = "step5_ot_mva_umap_default_texture" # UMAP용 기본 텍스처 태그 (부모에 추가 필요 가정)
TAG_OT_MVA_PCA_DEFAULT_TEXTURE = "step5_ot_mva_pca_default_texture"   # PCA용 기본 텍스처 태그 (부모에 추가 필요 가정)


# Right Panel (Outlier Instances & SHAP)
TAG_OT_MVA_OUTLIER_INSTANCES_TABLE = "step5_ot_mva_outlier_instances_table"
TAG_OT_MVA_INSTANCE_STATS_TABLE = "step5_ot_mva_instance_stats_table"
TAG_OT_MVA_SHAP_PLOT_IMAGE = "step5_ot_mva_shap_plot_image"
# SHAP 기본 텍스처는 부모 모듈의 TAG_OT_MVA_SHAP_DEFAULT_TEXTURE 사용 가정

# Left Panel - Bottom (Boxplots)
TAG_OT_MVA_BOXPLOT_GROUP = "step5_ot_mva_boxplot_group"
# Boxplot 이미지는 동적으로 생성하여 추가/삭제 관리 (개별 태그 필요시 동적 생성)


# --- Constants ---
DEFAULT_MVA_CONTAMINATION = 0.1
MAX_OUTLIER_INSTANCES_TO_SHOW = 30
TOP_N_VARIABLES_FOR_BOXPLOT = 10
MIN_SAMPLES_FOR_IFOREST = 5 # Isolation Forest 실행을 위한 최소 샘플 수
MIN_FEATURES_FOR_IFOREST = 2 # Isolation Forest 실행을 위한 최소 특성 수

# --- Module State Variables (Multivariate) ---
_shared_utils_mva: Optional[Dict[str, Any]] = None
_current_df_for_mva: Optional[pd.DataFrame] = None # 이 스텝에 입력된 DF
_df_with_mva_outliers: Optional[pd.DataFrame] = None # 이상치 점수/플래그 추가된 DF
_mva_model: Optional[Any] = None # 학습된 Isolation Forest 모델
_mva_shap_explainer: Optional[Any] = None
_mva_shap_values_for_selected: Optional[np.ndarray] = None

_mva_eligible_numeric_cols: List[str] = []
_mva_selected_columns_for_detection: List[str] = []
_mva_column_selection_mode: str = "All Numeric" # "All Numeric", "Recommended", "Manual"
_mva_contamination: float = DEFAULT_MVA_CONTAMINATION
_mva_outlier_instances_summary: List[Dict[str, Any]] = [] # 우측 테이블 데이터
_mva_active_umap_texture_id: Optional[str] = None
_mva_active_pca_texture_id: Optional[str] = None
_mva_active_shap_texture_id: Optional[str] = None
_mva_selected_outlier_instance_idx: Optional[Any] = None # 원본 DF의 인덱스
_mva_all_selectable_tags_in_instances_table: List[str] = []
_mva_top_gap_vars_for_boxplot: List[str] = []
_mva_boxplot_image_tags: List[str] = [] # 생성된 boxplot 이미지 태그들
_mva_last_recommendation_details_str: Optional[str] = None

def _log_mva(message: str):
    if _shared_utils_mva and 'log_message_func' in _shared_utils_mva:
        _shared_utils_mva['log_message_func'](f"[MVAOutlier] {message}")

def _show_simple_modal_mva(title: str, message: str, width: int = 450, height: int = 200):
    if _shared_utils_mva and 'util_funcs_common' in _shared_utils_mva and \
       '_show_simple_modal_message' in _shared_utils_mva['util_funcs_common']:
        _shared_utils_mva['util_funcs_common']['_show_simple_modal_message'](title, message, width, height)
    else:
        _log_mva(f"Modal display function not available. Title: {title}, Msg: {message}")

def _s5_plot_to_dpg_texture_mva(fig: 'plt.Figure', desired_dpi: int = 90) -> Tuple[Optional[str], int, int]:
    if _shared_utils_mva and 'plot_to_dpg_texture_func' in _shared_utils_mva:
        return _shared_utils_mva['plot_to_dpg_texture_func'](fig, desired_dpi)
    _log_mva("Error: Plot to DPG texture function not available from shared utils.")
    return None, 0, 0


def _update_mva_param_visibility():
    if not dpg.is_dearpygui_running(): return
    is_manual_mode = _mva_column_selection_mode == "Manual"
    if dpg.does_item_exist(TAG_OT_MVA_MANUAL_COLUMN_SELECTOR_GROUP):
        dpg.configure_item(TAG_OT_MVA_MANUAL_COLUMN_SELECTOR_GROUP, show=is_manual_mode)

def _on_mva_col_selection_mode_change(sender, app_data: str, user_data):
    global _mva_column_selection_mode
    _mva_column_selection_mode = app_data
    _log_mva(f"Multivariate column selection mode changed to: {_mva_column_selection_mode}")
    _update_mva_param_visibility()
    # 모드 변경 시 선택된 컬럼 목록 업데이트 필요
    _update_selected_columns_for_mva_detection()

def _on_mva_manual_cols_selected(sender, app_data: List[str], user_data):
    global _mva_selected_columns_for_detection
    _mva_selected_columns_for_detection = app_data
    _log_mva(f"Manually selected columns for MVA: {app_data}")

def _on_mva_contamination_change(sender, app_data: float, user_data):
    global _mva_contamination
    if 0.0 < app_data <= 0.5:
        _mva_contamination = app_data
        _log_mva(f"Multivariate contamination set to: {_mva_contamination:.4f}")
    else:
        dpg.set_value(sender, _mva_contamination) # 이전 값으로 복원
        err_msg = "Contamination must be between 0 (exclusive) and 0.5 (inclusive)."
        _log_mva(f"Invalid input for contamination: {app_data}. {err_msg}")
        _show_simple_modal_mva("Input Error", err_msg)

def _set_mva_recommended_parameters(sender, app_data, user_data):
    global _mva_contamination, _mva_column_selection_mode
    _mva_contamination = DEFAULT_MVA_CONTAMINATION
    _mva_column_selection_mode = "Recommended" # 추천 모드로 변경

    if dpg.does_item_exist(TAG_OT_MVA_CONTAMINATION_INPUT):
        dpg.set_value(TAG_OT_MVA_CONTAMINATION_INPUT, _mva_contamination)
    if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO):
        dpg.set_value(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO, _mva_column_selection_mode)

    _update_mva_param_visibility()
    _update_selected_columns_for_mva_detection() # 추천 컬럼으로 업데이트
    _log_mva(f"Recommended MVA parameters set: Contamination={_mva_contamination}, Mode='Recommended'")
    _show_simple_modal_mva("Info", "Recommended multivariate detection parameters have been applied.")

def _get_eligible_numeric_cols_for_mva(df: pd.DataFrame) -> List[str]:
    if df is None: return []
    numeric_cols = []
    if _shared_utils_mva and 'util_funcs_common' in _shared_utils_mva and \
       '_get_numeric_cols' in _shared_utils_mva['util_funcs_common']:
        # utils._get_numeric_cols는 모든 수치형을 반환. 여기서는 추가 필터링 가능.
        # 예를 들어, 분산이 매우 작거나 고유값이 거의 없는 수치형 컬럼 제외.
        # 현재는 utils에서 반환하는 모든 numeric cols를 사용.
        # 좀 더 정교하게 하려면 step_01_data_loading의 분석 타입을 참조하여 'Numeric'이면서 'Binary'가 아닌 것만.
        
        s1_col_types = {}
        if 'main_app_callbacks' in _shared_utils_mva and \
        'get_column_analysis_types' in _shared_utils_mva['main_app_callbacks']:
            s1_col_types = _shared_utils_mva['main_app_callbacks']['get_column_analysis_types']()

        all_cols = df.columns
        for col in all_cols:
            s1_type = s1_col_types.get(col, "")
            is_numeric_s1 = "Numeric" in s1_type and "Binary" not in s1_type
            is_numeric_pandas = pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].nunique() > 5 # 고유값 많은 수치형
            
            if is_numeric_s1 or is_numeric_pandas:
                # 추가 조건: 결측치가 너무 많거나, 분산이 0에 가까운 컬럼은 제외 가능
                if df[col].isnull().sum() / len(df) < 0.8 and df[col].dropna().var() > 1e-6:
                    numeric_cols.append(col)
        return numeric_cols
    return df.select_dtypes(include=np.number).columns.tolist()


def _update_selected_columns_for_mva_detection():
    global _mva_selected_columns_for_detection, _mva_eligible_numeric_cols, _mva_last_recommendation_details_str
    
    _mva_last_recommendation_details_str = None # 매번 초기화
    current_df = _shared_utils_mva['get_current_df_func']() if _shared_utils_mva and 'get_current_df_func' in _shared_utils_mva else None
    # ... (기존 current_df None 체크 및 _mva_eligible_numeric_cols 업데이트 로직) ...
    if current_df is None: # current_df가 None이면 eligible_cols도 비게 되므로, 이 부분은 유지
        _mva_eligible_numeric_cols = []
        _mva_selected_columns_for_detection = []
        if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTOR_MULTI):
            dpg.configure_item(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, items=[])
            dpg.set_value(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, "")
        return

    _mva_eligible_numeric_cols = _get_eligible_numeric_cols_for_mva(current_df) # 최신 eligible cols 가져오기

    if _mva_column_selection_mode == "All Numeric":
        _mva_selected_columns_for_detection = _mva_eligible_numeric_cols[:]
        _mva_last_recommendation_details_str = "Used all eligible numeric features."
    elif _mva_column_selection_mode == "Recommended":
        target_var = None
        target_var_type = None
        recommendation_reason = "all eligible numeric features (fallback criteria)." # 기본 Fallback 사유

        if _shared_utils_mva and 'main_app_callbacks' in _shared_utils_mva:
            get_target_var_func = _shared_utils_mva['main_app_callbacks'].get('get_selected_target_variable')
            get_target_type_func = _shared_utils_mva['main_app_callbacks'].get('get_selected_target_variable_type')
            if get_target_var_func: target_var = get_target_var_func()
            if get_target_type_func: target_var_type = get_target_type_func()

        # 연관성 계산 대상 컬럼 (타겟 변수 제외)
        eligible_cols_for_relevance = [col for col in _mva_eligible_numeric_cols if col != target_var]

        if target_var and target_var_type and eligible_cols_for_relevance and current_df is not None:
            print(f"[DEBUG] Recommended mode: Calculating relevance with target '{target_var}' ({target_var_type}) for {len(eligible_cols_for_relevance)} features.")
            
            relevance_scores = []
            # utils.calculate_feature_target_relevance 함수 호출 부분 (이전 답변의 예시 활용)
            # 이 함수는 main_app_callbacks를 통해 Step 1의 컬럼 타입 정보를 활용할 수 있어야 함
            if 'util_funcs_common' in _shared_utils_mva and 'calculate_feature_target_relevance' in _shared_utils_mva['util_funcs_common']:
                 relevance_scores = _shared_utils_mva['util_funcs_common']['calculate_feature_target_relevance'](
                     current_df, target_var, target_var_type, eligible_cols_for_relevance,
                     _shared_utils_mva.get('main_app_callbacks') 
                 )
            
            if relevance_scores:
                n1 = 20
                n2 = int(len(eligible_cols_for_relevance) * 0.20) # 적격 후보군의 20%
                num_to_select = max(n1, n2)
                
                _mva_selected_columns_for_detection = [feat for feat, score in relevance_scores[:num_to_select]]
                recommendation_reason = (f"top {len(_mva_selected_columns_for_detection)} features based on relevance to target '{target_var}' "
                                         f"(selected from {len(eligible_cols_for_relevance)} candidates using N=max(20, 20%)).")
                print(f"[DEBUG] Recommended columns ({len(_mva_selected_columns_for_detection)}): {_mva_selected_columns_for_detection[:5]}...") # 일부만 출력
            else:
                print("[DEBUG] Recommended mode: Relevance scores empty. Falling back.")
                _mva_selected_columns_for_detection = _mva_eligible_numeric_cols[:]
        else:
            print("[DEBUG] Recommended mode: No target or not enough data for relevance. Falling back.")
            _mva_selected_columns_for_detection = _mva_eligible_numeric_cols[:]
        
        _mva_last_recommendation_details_str = f"Recommended features selected: {recommendation_reason}"

    elif _mva_column_selection_mode == "Manual":
        _mva_selected_columns_for_detection = [col for col in _mva_selected_columns_for_detection if col in _mva_eligible_numeric_cols]
        _mva_last_recommendation_details_str = f"Used {len(_mva_selected_columns_for_detection)} manually selected features."
    
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTOR_MULTI):
        dpg.configure_item(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, items=_mva_eligible_numeric_cols)
        # Manual 모드일 때만 현재 선택된 것을 Listbox에 반영하려고 시도 (단일 선택 한계 인지)
        if _mva_column_selection_mode == "Manual" and _mva_selected_columns_for_detection:
            dpg.set_value(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, _mva_selected_columns_for_detection[0]) # 첫번째 선택된 것만 표시 시도
        elif _mva_column_selection_mode != "Manual":
             dpg.set_value(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, "") 

def _update_mva_boxplots_for_comparison():
    print("[DEBUG] Called: _update_mva_boxplots_for_comparison")
    if not dpg.is_dearpygui_running():
        print("[DEBUG] _update_mva_boxplots_for_comparison: DPG not running. Skipping.")
        return

    if _df_with_mva_outliers is None:
        print("[DEBUG] _update_mva_boxplots_for_comparison: No MVA outlier data. Clearing boxplots.")
        _clear_mva_boxplots()
        if dpg.does_item_exist(TAG_OT_MVA_BOXPLOT_GROUP):
             # 기존 자식들을 먼저 지우고 텍스트 추가
            dpg.delete_item(TAG_OT_MVA_BOXPLOT_GROUP, children_only=True)
            dpg.add_text("Run detection to see boxplots.", parent=TAG_OT_MVA_BOXPLOT_GROUP)
        return

    print("[DEBUG] _update_mva_boxplots_for_comparison: Generating MVA boxplots...")
    _generate_mva_boxplots_for_comparison()


def _run_mva_outlier_detection_logic(sender, app_data, user_data):
    global _df_with_mva_outliers, _mva_model, _mva_shap_explainer, _mva_outlier_instances_summary
    
    print("[DEBUG] Called: _run_mva_outlier_detection_logic")
    current_df = _shared_utils_mva['get_current_df_func']()

    if current_df is None:
        print("[DEBUG] _run_mva_outlier_detection_logic: current_df is None. Aborting.")
        _show_simple_modal_mva("Error", "No data for multivariate outlier detection.")
        return

    _update_selected_columns_for_mva_detection() 
    print(f"[DEBUG] _run_mva_outlier_detection_logic: Selected columns for detection: {_mva_selected_columns_for_detection}")
    
    if not _mva_selected_columns_for_detection or len(_mva_selected_columns_for_detection) < MIN_FEATURES_FOR_IFOREST:
        print(f"[DEBUG] _run_mva_outlier_detection_logic: Not enough features. Selected: {len(_mva_selected_columns_for_detection)}")
        _show_simple_modal_mva("Error", f"Please select at least {MIN_FEATURES_FOR_IFOREST} numeric features.")
        return

    df_for_detection_raw = current_df[_mva_selected_columns_for_detection].copy()
    df_for_detection = df_for_detection_raw.dropna()
    print(f"[DEBUG] _run_mva_outlier_detection_logic: Shape of df_for_detection (after dropna): {df_for_detection.shape}")
    
    if len(df_for_detection) < MIN_SAMPLES_FOR_IFOREST:
        print(f"[DEBUG] _run_mva_outlier_detection_logic: Not enough samples after dropna. Samples: {len(df_for_detection)}")
        _show_simple_modal_mva("Error", f"Not enough data rows (after NaN removal) to run detection. Min required: {MIN_SAMPLES_FOR_IFOREST}.")
        return
        
    print(f"[DEBUG] _run_mva_outlier_detection_logic: Starting IForest with Contam={_mva_contamination}")

    try:
        scaler = StandardScaler()
        scaled_data_for_reduction = scaler.fit_transform(df_for_detection.values)

        _mva_model = PyOD_IForest(contamination=_mva_contamination, random_state=42, n_jobs=-1)
        _mva_model.fit(df_for_detection.values) 

        outlier_scores = _mva_model.decision_scores_
        outlier_labels = _mva_model.labels_

        _df_with_mva_outliers = current_df.copy()
        _df_with_mva_outliers.loc[df_for_detection.index, 'mva_outlier_score'] = outlier_scores
        _df_with_mva_outliers.loc[df_for_detection.index, 'mva_is_outlier'] = outlier_labels.astype(bool)
        
        _df_with_mva_outliers['mva_is_outlier'] = _df_with_mva_outliers['mva_is_outlier'].fillna(False)
        min_score_val = np.min(outlier_scores) if len(outlier_scores) > 0 else 0
        _df_with_mva_outliers['mva_outlier_score'] = _df_with_mva_outliers['mva_outlier_score'].fillna(min_score_val - 1)

        print(f"[DEBUG] _run_mva_outlier_detection_logic: _df_with_mva_outliers shape: {_df_with_mva_outliers.shape}")
        if 'mva_is_outlier' in _df_with_mva_outliers.columns:
            print(f"[DEBUG] _run_mva_outlier_detection_logic: mva_is_outlier value counts:\n{_df_with_mva_outliers['mva_is_outlier'].value_counts(dropna=False)}")
        if 'mva_outlier_score' in _df_with_mva_outliers.columns:
            print(f"[DEBUG] _run_mva_outlier_detection_logic: mva_outlier_score (non-NaN) describe:\n{_df_with_mva_outliers['mva_outlier_score'].dropna().describe()}")
        
        if 'shap' in globals() and shap is not None:
            try:
                _mva_shap_explainer = shap.TreeExplainer(_mva_model.detector_, data=df_for_detection) # Pass data to explainer for background
                print("[DEBUG] _run_mva_outlier_detection_logic: SHAP TreeExplainer initialized.")
            except Exception as e_shap_init:
                print(f"[DEBUG] _run_mva_outlier_detection_logic: Error initializing SHAP explainer: {e_shap_init}")
                _mva_shap_explainer = None
        else:
            _mva_shap_explainer = None
            print("[DEBUG] _run_mva_outlier_detection_logic: SHAP library not available.")

        _mva_outlier_instances_summary = []
        # NaN이 아닌 'mva_is_outlier' 값만 필터링 (이미 fillna(False) 처리됨)
        detected_outliers_df = _df_with_mva_outliers[_df_with_mva_outliers['mva_is_outlier'] == True]
        
        if not detected_outliers_df.empty and 'mva_outlier_score' in detected_outliers_df.columns:
            detected_outliers_df = detected_outliers_df.sort_values(by='mva_outlier_score', ascending=False)
        
        print(f"[DEBUG] _run_mva_outlier_detection_logic: Number of actual detected outliers: {len(detected_outliers_df)}")

        for original_idx, row in detected_outliers_df.head(MAX_OUTLIER_INSTANCES_TO_SHOW).iterrows():
            _mva_outlier_instances_summary.append({
                "Original Index": original_idx,
                "MVA Outlier Score": f"{row['mva_outlier_score']:.4f}"
            })
        
        if _mva_outlier_instances_summary:
            print(f"[DEBUG] _run_mva_outlier_detection_logic: First item in summary: {_mva_outlier_instances_summary[0]}")

        _populate_mva_outlier_instances_table()
        _generate_mva_umap_pca_plots(scaled_data_for_reduction, df_for_detection.index, outlier_labels) # outlier_labels은 df_for_detection에 해당
        _update_mva_boxplots_for_comparison()

        if _df_with_mva_outliers is not None and _current_df_for_mva is not None:
            num_total_samples = len(_current_df_for_mva)
            num_detected_outliers = _df_with_mva_outliers['mva_is_outlier'].sum() # True 값의 합계
            outlier_ratio = (num_detected_outliers / num_total_samples) * 100 if num_total_samples > 0 else 0
            
            # 전체 변수 수는 원본 입력 DF(_current_df_for_mva)의 컬럼 수
            # (단, 여기서 ID나 타겟 등 분석에 사용되지 않을 컬럼 제외 필요시 추가 로직)
            num_total_features_in_input = len(_current_df_for_mva.columns)
            num_used_features = len(_mva_selected_columns_for_detection)

            if dpg.does_item_exist("mva_summary_text_status"):
                dpg.set_value("mva_summary_text_status", "Detection Complete. Summary:")
            if dpg.does_item_exist("mva_summary_text_total_features"):
                dpg.set_value("mva_summary_text_total_features", f"  - Total Features in Input Data: {num_total_features_in_input}")
            if dpg.does_item_exist("mva_summary_text_used_features"):
                dpg.set_value("mva_summary_text_used_features", f"  - Features Used for Detection: {num_used_features}")
            if dpg.does_item_exist("mva_summary_text_detected_outliers"):
                dpg.set_value("mva_summary_text_detected_outliers", f"  - Detected Outlier Instances: {num_detected_outliers} samples")
            if dpg.does_item_exist("mva_summary_text_outlier_ratio"):
                dpg.set_value("mva_summary_text_outlier_ratio", f"  - Outlier Ratio: {outlier_ratio:.2f}% of total {num_total_samples} samples")
        else:
            if dpg.does_item_exist("mva_summary_text_status"):
                dpg.set_value("mva_summary_text_status", "Could not generate summary (data missing).")

        completion_message = "Multivariate outlier detection finished."
        if _mva_column_selection_mode == "Recommended" and _mva_last_recommendation_details_str:
            completion_message += f"\n\n[Recommendation Info]\n{_mva_last_recommendation_details_str}"
        elif _mva_last_recommendation_details_str: # Manual 또는 All Numeric일 때도 간단한 정보 추가
             completion_message += f"\n\n[Selection Info]\n{_mva_last_recommendation_details_str}"

        # 모달창의 높이를 메시지 길이에 따라 조절하거나 충분히 크게 설정
        # 예시: 메시지 줄 수에 따라 높이 동적 조절 (간단한 방식)
        num_lines = completion_message.count('\n') + 1
        modal_height = min(max(200, num_lines * 25), 400) # 최소 200, 최대 400, 줄당 25px

        _show_simple_modal_mva("Detection Complete", completion_message, height=modal_height)

        
    except Exception as e:
        print(f"[DEBUG] _run_mva_outlier_detection_logic: Error during MVA detection: {e}")
        import traceback
        print(f"[DEBUG] _run_mva_outlier_detection_logic: Traceback: {traceback.format_exc()}")
        _show_simple_modal_mva("Detection Error", f"An error occurred during multivariate detection: {e}")
        _df_with_mva_outliers = None
        _mva_model = None
        _mva_shap_explainer = None
        _clear_all_mva_visualizations()
        _populate_mva_outlier_instances_table()

def _populate_mva_outlier_instances_table():
    global _mva_all_selectable_tags_in_instances_table
    print("[DEBUG] Called: _populate_mva_outlier_instances_table")
    dpg.add_table_column(label="Test Col 1", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE)
    dpg.add_table_column(label="Test Col 2", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE)
    with dpg.table_row(parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE):
        dpg.add_text("Test Data 1")
        dpg.add_text("Test Data 2")
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_MVA_OUTLIER_INSTANCES_TABLE):
        print("[DEBUG] _populate_mva_outlier_instances_table: DPG not running or table tag not found.")
        return
    dpg.delete_item(TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, children_only=True)
    _mva_all_selectable_tags_in_instances_table.clear()

    if not _mva_outlier_instances_summary:
        print("[DEBUG] _populate_mva_outlier_instances_table: No outlier instances to show.")
        dpg.add_table_column(label="Info", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE)
        with dpg.table_row(parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE):
            dpg.add_text("No multivariate outliers detected or detection not run.")
        _clear_mva_instance_details()
        return

    headers = ["Original Index", "MVA Outlier Score (Higher is more outlier)"]
    dpg.add_table_column(label=headers[0], parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, init_width_or_weight=0.4)
    dpg.add_table_column(label=headers[1], parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, init_width_or_weight=0.6)

    print(f"[DEBUG] _populate_mva_outlier_instances_table: Populating table with {len(_mva_outlier_instances_summary)} items.")
    for i, item_data in enumerate(_mva_outlier_instances_summary):
        original_idx = item_data["Original Index"]
        score_str = item_data["MVA Outlier Score"]
        
        tag = f"mva_instance_selectable_{i}_{original_idx}_{dpg.generate_uuid()}"
        _mva_all_selectable_tags_in_instances_table.append(tag)
        
        with dpg.table_row(parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE):
            dpg.add_selectable(label=str(original_idx), tag=tag, user_data=original_idx,
                               callback=_on_mva_outlier_instance_selected, span_columns=False)
            dpg.add_text(score_str)
            # print(f"[DEBUG] _populate_mva_outlier_instances_table: Added row for index {original_idx}")


def _on_mva_outlier_instance_selected(sender, app_data_is_selected: bool, user_data_original_idx: Any):
    global _mva_selected_outlier_instance_idx
    if app_data_is_selected:
        for tag_iter in _mva_all_selectable_tags_in_instances_table:
            if tag_iter != sender and dpg.does_item_exist(tag_iter) and dpg.get_value(tag_iter):
                dpg.set_value(tag_iter, False) # 다른 선택 해제
        
        _mva_selected_outlier_instance_idx = user_data_original_idx
        _log_mva(f"MVA Outlier instance selected: Index {user_data_original_idx}")
        _display_mva_instance_statistics(user_data_original_idx)
        _generate_mva_shap_plot_for_instance(user_data_original_idx)
    else: # 선택 해제 시 (다시 클릭)
        if _mva_selected_outlier_instance_idx == user_data_original_idx: # 현재 선택된 것을 다시 클릭하여 해제
             _mva_selected_outlier_instance_idx = None
             _clear_mva_instance_details()


def _display_mva_instance_statistics(original_idx: Any):
    print(f"[DEBUG] Called: _display_mva_instance_statistics for index {original_idx}")
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_MVA_INSTANCE_STATS_TABLE):
        print("[DEBUG] _display_mva_instance_statistics: DPG not running or stats table tag not found.")
        return
    dpg.delete_item(TAG_OT_MVA_INSTANCE_STATS_TABLE, children_only=True)

    if _df_with_mva_outliers is None or original_idx not in _df_with_mva_outliers.index:
        print("[DEBUG] _display_mva_instance_statistics: Instance data not available in _df_with_mva_outliers.")
        dpg.add_table_column(label="Info", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE)
        with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE):
            dpg.add_text("Selected instance data not available.")
        return

    instance_data_series = _df_with_mva_outliers.loc[original_idx]
    
    # Z-score 계산 및 정렬을 위한 데이터 준비
    feature_stats_list = []
    # MVA 탐지에 사용된 컬럼들만 대상으로 함 (또는 모든 수치형 컬럼으로 확장 가능)
    cols_to_process_for_stats = _mva_selected_columns_for_detection[:] if _mva_selected_columns_for_detection else []

    source_df_for_overall_stats = _current_df_for_mva # 스텝 시작 시 DF 기준

    if source_df_for_overall_stats is None:
        print("[DEBUG] Stats Table: source_df_for_overall_stats is None. Cannot calculate overall stats for Z-score sort.")
        # 이 경우, 정렬 없이 기본 순서대로 표시하거나, 빈 테이블 처리.
        # 여기서는 Z-score 없이 기본 순서대로 표시 (아래 루프에서 N/A 처리됨)

    for feature_name in cols_to_process_for_stats:
        if feature_name in instance_data_series: # 현재 인스턴스에 해당 피처가 있는지 확인
            value = instance_data_series[feature_name]
            
            overall_mean_val, overall_median_val, z_score_abs_val = np.nan, np.nan, 0.0 # 정렬키를 위한 실제 float 값
            overall_mean_str, overall_median_str, z_score_str = "N/A", "N/A", "N/A" # 표시용 문자열

            if source_df_for_overall_stats is not None and feature_name in source_df_for_overall_stats.columns:
                feature_series_overall = source_df_for_overall_stats[feature_name].dropna()
                if not feature_series_overall.empty and pd.api.types.is_numeric_dtype(feature_series_overall.dtype):
                    overall_mean_val = feature_series_overall.mean()
                    overall_median_val = feature_series_overall.median()
                    std_val = feature_series_overall.std()
                    
                    overall_mean_str = f"{overall_mean_val:.4f}"
                    overall_median_str = f"{overall_median_val:.4f}"
                    
                    # Z-score 계산
                    if pd.notna(std_val) and std_val > 1e-9 and pd.notna(value) and pd.api.types.is_numeric_dtype(type(value)) and pd.notna(overall_mean_val):
                        z_score = (value - overall_mean_val) / std_val
                        z_score_abs_val = abs(z_score)
                        z_score_str = f"{z_score:.2f}"
                    elif pd.notna(value) and pd.api.types.is_numeric_dtype(type(value)):
                         z_score_str = "N/A (std~0 or no mean)"
            
            feature_stats_list.append({
                "feature": feature_name,
                "value_str": f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value),
                "mean_str": overall_mean_str,
                "median_str": overall_median_str,
                "z_score_str": z_score_str,
                "z_score_abs_sort_key": z_score_abs_val if pd.notna(z_score_abs_val) else -1 # NaN이면 정렬 시 뒤로
            })

    # Z-score 절댓값 기준으로 내림차순 정렬 (전체 통계량 계산 가능했을 때만 의미 있음)
    sorted_feature_stats_list = sorted(feature_stats_list, key=lambda x: x["z_score_abs_sort_key"], reverse=True)
    
    print(f"[DEBUG] Stats Table: Features sorted by Z-score (abs) for instance {original_idx}: {[item['feature'] for item in sorted_feature_stats_list[:5]]}")

    # 컬럼 헤더 추가
    dpg.add_table_column(label="Feature", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, init_width_or_weight=0.25)
    dpg.add_table_column(label="Value", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, init_width_or_weight=0.15)
    dpg.add_table_column(label="Overall Mean", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, init_width_or_weight=0.20)
    dpg.add_table_column(label="Overall Median", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, init_width_or_weight=0.20)
    dpg.add_table_column(label="Z-score Dist.", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, init_width_or_weight=0.20)

    # 정렬된 순서대로 테이블 행 생성
    for stats_item in sorted_feature_stats_list:
        with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE):
            dpg.add_text(stats_item["feature"])
            dpg.add_text(stats_item["value_str"])
            dpg.add_text(stats_item["mean_str"])
            dpg.add_text(stats_item["median_str"])
            dpg.add_text(stats_item["z_score_str"])
    
    # mva_outlier_score 행 추가 (이전과 동일, 이 행은 정렬과 무관하게 마지막에 표시)
    if 'mva_outlier_score' in instance_data_series:
        with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE):
            dpg.add_text("MVA Outlier Score", color=[255,255,0])
            dpg.add_text(f"{instance_data_series['mva_outlier_score']:.4f}")
            dpg.add_text("-") 
            dpg.add_text("-")
            dpg.add_text("-")

def _clear_mva_instance_details():
    if not dpg.is_dearpygui_running(): return
    # 통계량 테이블 클리어
    if dpg.does_item_exist(TAG_OT_MVA_INSTANCE_STATS_TABLE):
        dpg.delete_item(TAG_OT_MVA_INSTANCE_STATS_TABLE, children_only=True)
        dpg.add_table_column(label="Info", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE)
        with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE):
            dpg.add_text("Select an outlier instance from the table above.")
    # SHAP 플롯 클리어
    _clear_mva_shap_plot()


# --- Visualization Functions (UMAP, PCA, SHAP, Boxplots) ---

def _generate_mva_umap_pca_plots(data_for_reduction: np.ndarray, original_indices: pd.Index, outlier_labels: np.ndarray):
    global _mva_active_umap_texture_id, _mva_active_pca_texture_id
    
    _clear_mva_umap_plot()
    _clear_mva_pca_plot()

    if data_for_reduction is None or len(data_for_reduction) < 2 or data_for_reduction.shape[1] < 2:
        _log_mva("Not enough data for UMAP/PCA. Skipping visualization.")
        return

    if len(data_for_reduction) != len(outlier_labels) or len(data_for_reduction) != len(original_indices):
        _log_mva("Data length mismatch for UMAP/PCA. Skipping visualization.")
        return

    # UMAP
    if umap:
        try:
            _log_mva("Generating UMAP plot...")
            reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            embedding_umap = reducer_umap.fit_transform(data_for_reduction) # 스케일링/NaN제거된 데이터 사용
            
            fig_umap, ax_umap = plt.subplots(figsize=(7, 5)) # 가로 배치 고려하여 적절한 크기
            scatter_umap = ax_umap.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=outlier_labels, cmap='coolwarm', s=15, alpha=0.7)
            ax_umap.set_title("UMAP Projection of Outliers", fontsize=10)
            ax_umap.set_xlabel("UMAP 1", fontsize=8); ax_umap.set_ylabel("UMAP 2", fontsize=8)
            ax_umap.tick_params(axis='both', which='major', labelsize=7)
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='blue', markersize=5),
                               plt.Line2D([0], [0], marker='o', color='w', label='Outlier', markerfacecolor='red', markersize=5)]
            ax_umap.legend(handles=legend_elements, loc='best', fontsize=7)
            plt.tight_layout()
            
            tex_tag_umap, w_umap, h_umap = _s5_plot_to_dpg_texture_mva(fig_umap)
            plt.close(fig_umap)

            if tex_tag_umap and w_umap > 0 and h_umap > 0:
                _mva_active_umap_texture_id = tex_tag_umap
                if dpg.does_item_exist(TAG_OT_MVA_UMAP_PLOT_IMAGE):
                    dpg.configure_item(TAG_OT_MVA_UMAP_PLOT_IMAGE, texture_tag=_mva_active_umap_texture_id, width=w_umap, height=h_umap)
            _log_mva("UMAP plot generated." if tex_tag_umap else "UMAP plot generation failed.")
        except Exception as e_umap:
            _log_mva(f"Error generating UMAP plot: {e_umap}")
    else:
        _log_mva("UMAP library not available. Skipping UMAP plot.")

    # PCA
    try:
        _log_mva("Generating PCA plot...")
        pca = PCA(n_components=2, random_state=42)
        embedding_pca = pca.fit_transform(data_for_reduction) # 스케일링/NaN제거된 데이터 사용
        
        fig_pca, ax_pca = plt.subplots(figsize=(7, 5)) # 가로 배치 고려
        scatter_pca = ax_pca.scatter(embedding_pca[:, 0], embedding_pca[:, 1], c=outlier_labels, cmap='viridis', s=15, alpha=0.7)
        ax_pca.set_title("PCA Projection of Outliers", fontsize=10)
        ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=8)
        ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=8)
        ax_pca.tick_params(axis='both', which='major', labelsize=7)
        # PCA는 UMAP과 다른 색상맵 및 범례 사용 가능
        legend_elements_pca = [plt.Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor=plt.cm.viridis(0.2), markersize=5),
                               plt.Line2D([0], [0], marker='o', color='w', label='Outlier', markerfacecolor=plt.cm.viridis(0.8), markersize=5)]
        ax_pca.legend(handles=legend_elements_pca, loc='best', fontsize=7)

        plt.tight_layout()

        tex_tag_pca, w_pca, h_pca = _s5_plot_to_dpg_texture_mva(fig_pca)
        plt.close(fig_pca)

        if tex_tag_pca and w_pca > 0 and h_pca > 0:
            _mva_active_pca_texture_id = tex_tag_pca
            if dpg.does_item_exist(TAG_OT_MVA_PCA_PLOT_IMAGE):
                dpg.configure_item(TAG_OT_MVA_PCA_PLOT_IMAGE, texture_tag=_mva_active_pca_texture_id, width=w_pca, height=h_pca)
        _log_mva("PCA plot generated." if tex_tag_pca else "PCA plot generation failed.")
    except Exception as e_pca:
        _log_mva(f"Error generating PCA plot: {e_pca}")


def _generate_mva_shap_plot_for_instance(original_idx: Any):
    global _mva_active_shap_texture_id, _mva_shap_values_for_selected
    _clear_mva_shap_plot() # 이전 SHAP 플롯 클리어

    # SHAP 라이브러리 존재 여부 확인
    if not ('shap' in globals() and shap is not None) :
        if dpg.does_item_exist(TAG_OT_MVA_SHAP_PLOT_IMAGE):
            dpg.configure_item(TAG_OT_MVA_SHAP_PLOT_IMAGE, show=False)
            shap_plot_parent = dpg.get_item_parent(TAG_OT_MVA_SHAP_PLOT_IMAGE) if dpg.does_item_exist(TAG_OT_MVA_SHAP_PLOT_IMAGE) else None
            if shap_plot_parent and dpg.does_item_exist(shap_plot_parent):
                if dpg.does_item_exist("mva_shap_status_text"): dpg.delete_item("mva_shap_status_text") # 이전 메시지 삭제
                dpg.add_text("SHAP N/A (library missing)", parent=shap_plot_parent, color=[255,100,100], tag="mva_shap_status_text")
        return

    # 필요한 데이터 존재 여부 확인
    if _df_with_mva_outliers is None or original_idx not in _df_with_mva_outliers.index:
        print("[DEBUG] _generate_mva_shap_plot_for_instance: Instance data not found in _df_with_mva_outliers.")
        return
    if _mva_shap_explainer is None or not _mva_selected_columns_for_detection:
        print("[DEBUG] _generate_mva_shap_plot_for_instance: SHAP explainer not ready or no columns for detection.")
        return

    try:
        instance_series = _df_with_mva_outliers.loc[original_idx, _mva_selected_columns_for_detection]
        instance_df_for_shap = pd.DataFrame([instance_series.values], columns=_mva_selected_columns_for_detection)

        # NaN 값 처리 (SHAP 계산 전)
        if instance_df_for_shap.isnull().values.any():
            print(f"[DEBUG] _generate_mva_shap_plot_for_instance: Instance {original_idx} has NaNs. Imputing with mean from source DF.")
            if _current_df_for_mva is not None: # _current_df_for_mva는 스텝 시작 시의 DF
                for col in instance_df_for_shap.columns:
                    if instance_df_for_shap[col].isnull().any():
                        mean_val = _current_df_for_mva[col].dropna().mean()
                        instance_df_for_shap[col].fillna(mean_val, inplace=True)
            else:
                print("[DEBUG] _generate_mva_shap_plot_for_instance: _current_df_for_mva is None for NaN imputation. Filling with 0.")
                instance_df_for_shap.fillna(0, inplace=True) # 대체값으로 0 사용

        # SHAP 값 계산
        shap_values_instance_raw = _mva_shap_explainer.shap_values(instance_df_for_shap)
        
        # SHAP 값 형태에 따른 처리 (IForest는 보통 단일 출력)
        if isinstance(shap_values_instance_raw, list):
            # explainer가 여러 모델을 감싸는 경우 등, 현재 IForest에는 해당 안될 가능성 높음
            _mva_shap_values_for_selected = shap_values_instance_raw[0][0, :] if len(shap_values_instance_raw[0].shape) == 2 else shap_values_instance_raw[0]
        else: # NumPy 배열일 경우
            _mva_shap_values_for_selected = shap_values_instance_raw[0, :] if len(shap_values_instance_raw.shape) == 2 else shap_values_instance_raw

        print(f"[DEBUG] _generate_mva_shap_plot_for_instance: SHAP values calculated. Shape: {_mva_shap_values_for_selected.shape}")

        # SHAP Explanation 객체 생성
        explainer_expected_value = _mva_shap_explainer.expected_value
        if hasattr(explainer_expected_value, "__len__") and not isinstance(explainer_expected_value, (str, bytes)): # 배열인 경우 첫 번째 값 사용 (IForest는 보통 스칼라)
            explainer_expected_value = explainer_expected_value[0]

        shap_explanation = shap.Explanation(
            values=_mva_shap_values_for_selected,             # (n_features,)
            base_values=explainer_expected_value,            # 스칼라
            data=instance_df_for_shap.iloc[0].values,      # (n_features,)
            feature_names=_mva_selected_columns_for_detection
        )

        # Waterfall plot 생성
        max_display_shap = 20 # 사용자가 요청한 최대 표시 피처 수
        num_features_to_display = min(len(_mva_selected_columns_for_detection), max_display_shap)
        
        # Figure 크기 조절 (표시될 피처 수에 따라)
        fig_height_shap = max(4.5, num_features_to_display * 0.4) # 피처당 약 0.4 인치 높이, 최소 4.5인치
        fig_shap_waterfall = plt.figure(figsize=(8, fig_height_shap)) # 새로운 Figure 생성

        shap.waterfall_plot(
            shap_explanation,
            max_display=num_features_to_display,
            show=False # 자동 plt.show() 방지
        )
        
        # Waterfall plot은 기본적으로 제목을 포함할 수 있으나, 필요시 추가 설정 가능
        # plt.title(f"SHAP Waterfall Plot - Instance {original_idx}", fontsize=10) # 필요한 경우
        plt.tight_layout() # 레이아웃 조절

        # 현재 Figure (fig_shap_waterfall)를 DPG 텍스처로 변환
        tex_tag_shap, w_shap, h_shap = _s5_plot_to_dpg_texture_mva(fig_shap_waterfall)
        plt.close(fig_shap_waterfall) # 사용한 Figure 닫기 (메모리 관리)

        if tex_tag_shap and w_shap > 0 and h_shap > 0:
            _mva_active_shap_texture_id = tex_tag_shap
            if dpg.does_item_exist(TAG_OT_MVA_SHAP_PLOT_IMAGE):
                dpg.configure_item(TAG_OT_MVA_SHAP_PLOT_IMAGE, texture_tag=_mva_active_shap_texture_id, width=w_shap, height=h_shap, show=True)
                # 메시지 텍스트가 있다면 삭제
                if dpg.does_item_exist("mva_shap_status_text"): dpg.delete_item("mva_shap_status_text")
        else: # 텍스처 생성 실패 시
             if dpg.does_item_exist(TAG_OT_MVA_SHAP_PLOT_IMAGE):
                dpg.configure_item(TAG_OT_MVA_SHAP_PLOT_IMAGE, show=True) # 기본 이미지로 보이도록 시도
                shap_plot_parent = dpg.get_item_parent(TAG_OT_MVA_SHAP_PLOT_IMAGE)
                if shap_plot_parent and dpg.does_item_exist(shap_plot_parent):
                    if dpg.does_item_exist("mva_shap_status_text"): dpg.delete_item("mva_shap_status_text")
                    dpg.add_text("Failed to generate SHAP waterfall plot.", parent=shap_plot_parent, color=[255,0,0], tag="mva_shap_status_text")
        print(f"[DEBUG] _generate_mva_shap_plot_for_instance: SHAP waterfall plot {'generated' if tex_tag_shap else 'generation failed'}.")

    except Exception as e_shap_plot:
        print(f"[DEBUG] _generate_mva_shap_plot_for_instance: Error generating SHAP waterfall plot for instance {original_idx}: {e_shap_plot}")
        import traceback
        print(f"[DEBUG] _generate_mva_shap_plot_for_instance: Traceback: {traceback.format_exc()}")
        if dpg.does_item_exist(TAG_OT_MVA_SHAP_PLOT_IMAGE): # 이미지 위젯이 있다면
            shap_plot_parent = dpg.get_item_parent(TAG_OT_MVA_SHAP_PLOT_IMAGE)
            if shap_plot_parent and dpg.does_item_exist(shap_plot_parent):
                if dpg.does_item_exist("mva_shap_status_text"): dpg.delete_item("mva_shap_status_text")
                dpg.add_text(f"SHAP Plot Error: {e_shap_plot}", parent=shap_plot_parent, color=[255,0,0], tag="mva_shap_status_text")

def _find_top_gap_variables_for_boxplot() -> List[str]:
    global _mva_top_gap_vars_for_boxplot
    print("[DEBUG] Called: _find_top_gap_variables_for_boxplot")
    _mva_top_gap_vars_for_boxplot = []
    if _df_with_mva_outliers is None or 'mva_is_outlier' not in _df_with_mva_outliers.columns:
        print("[DEBUG] _find_top_gap_variables_for_boxplot: _df_with_mva_outliers is None or 'mva_is_outlier' column missing.")
        return []

    eligible_cols = _get_eligible_numeric_cols_for_mva(_df_with_mva_outliers)
    print(f"[DEBUG] _find_top_gap_variables_for_boxplot: Eligible columns for boxplot comparison: {eligible_cols}")
    if not eligible_cols:
        print("[DEBUG] _find_top_gap_variables_for_boxplot: No eligible columns found.")
        return []

    gaps = []
    for col in eligible_cols:
        if col in ['mva_outlier_score', 'mva_is_outlier']: continue

        # 정확한 인덱싱을 위해 .loc 사용 및 boolean Series 직접 전달
        normal_data = _df_with_mva_outliers.loc[_df_with_mva_outliers['mva_is_outlier'] == False, col].dropna()
        outlier_data = _df_with_mva_outliers.loc[_df_with_mva_outliers['mva_is_outlier'] == True, col].dropna()
        
        # print(f"[DEBUG] _find_top_gap_variables_for_boxplot: Processing column '{col}'. Normal data points: {len(normal_data)}, Outlier data points: {len(outlier_data)}")

        if len(normal_data) < 2 or len(outlier_data) < 2:
            # print(f"[DEBUG] _find_top_gap_variables_for_boxplot: Not enough data for column '{col}'. Skipping.")
            continue
        
        median_normal = normal_data.median()
        median_outlier = outlier_data.median()
        gap = abs(median_normal - median_outlier)
        # print(f"[DEBUG] _find_top_gap_variables_for_boxplot: Column '{col}' -> Normal Median: {median_normal:.2f}, Outlier Median: {median_outlier:.2f}, Gap: {gap:.2f}")
        
        if pd.notna(gap) and gap > 1e-6 : # 유의미한 차이
            gaps.append((col, gap))

    gaps.sort(key=lambda x: x[1], reverse=True)
    _mva_top_gap_vars_for_boxplot = [var for var, score in gaps[:TOP_N_VARIABLES_FOR_BOXPLOT]]
    print(f"[DEBUG] _find_top_gap_variables_for_boxplot: Top variables selected: {_mva_top_gap_vars_for_boxplot}")
    return _mva_top_gap_vars_for_boxplot


def _generate_mva_boxplots_for_comparison():
    global _mva_boxplot_image_tags
    
    _clear_mva_boxplots() # 이전 이미지 및 텍스처 태그 리스트 클리어
    top_vars = _find_top_gap_variables_for_boxplot() # 이 함수가 _mva_top_gap_vars_for_boxplot를 업데이트함

    if not top_vars or _df_with_mva_outliers is None:
        print("[DEBUG] _generate_mva_boxplots_for_comparison: No top variables or no outlier data. Nothing to plot.")
        if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_MVA_BOXPLOT_GROUP):
            # dpg.delete_item(TAG_OT_MVA_BOXPLOT_GROUP, children_only=True) # 이미 _clear_mva_boxplots에서 처리
            dpg.add_text("Run detection to see boxplots or no significant gaps found.", parent=TAG_OT_MVA_BOXPLOT_GROUP)
        return

    num_plots_actual = len(top_vars)
    cols_per_row = 2 
    
    parent_group_for_boxplots = TAG_OT_MVA_BOXPLOT_GROUP
    if not dpg.does_item_exist(parent_group_for_boxplots):
        print(f"[DEBUG] _generate_mva_boxplots_for_comparison: Error - Boxplot parent group '{parent_group_for_boxplots}' does not exist.")
        return
    
    print(f"[DEBUG] _generate_mva_boxplots_for_comparison: Generating {num_plots_actual} boxplots in group '{parent_group_for_boxplots}'.")

    for i in range(0, num_plots_actual, cols_per_row):
        with dpg.group(horizontal=True, parent=parent_group_for_boxplots) as row_group: # 각 행의 가로 그룹
            for j in range(cols_per_row):
                if i + j < num_plots_actual:
                    var_name = top_vars[i+j]
                    print(f"[DEBUG] _generate_mva_boxplots_for_comparison: Plotting for variable '{var_name}'.")
                    try:
                        fig, ax = plt.subplots(figsize=(5, 4)) # 개별 boxplot 크기
                        
                        normal_series = _df_with_mva_outliers.loc[_df_with_mva_outliers['mva_is_outlier'] == False, var_name].dropna()
                        outlier_series = _df_with_mva_outliers.loc[_df_with_mva_outliers['mva_is_outlier'] == True, var_name].dropna()
                        
                        data_to_plot, labels = [], []
                        if not normal_series.empty: data_to_plot.append(normal_series); labels.append("Normal")
                        if not outlier_series.empty: data_to_plot.append(outlier_series); labels.append("Outlier")

                        if data_to_plot:
                            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, vert=True,
                                            medianprops={'color': '#FF0000', 'linewidth':1.5})
                            colors = ['lightblue', 'lightcoral'][:len(data_to_plot)]
                            for patch, color_val in zip(bp['boxes'], colors): patch.set_facecolor(color_val)

                            ax.set_title(f"{var_name}", fontsize=9)
                            ax.tick_params(axis='both', which='major', labelsize=7)
                            ax.grid(True, linestyle='--', alpha=0.6)
                            plt.tight_layout()

                            tex_tag, w, h = _s5_plot_to_dpg_texture_mva(fig)
                            plt.close(fig)

                            if tex_tag and w > 0 and h > 0:
                                dpg.add_image(tex_tag, width=w, height=h, parent=row_group) # 현재 행 그룹에 이미지 추가
                                _mva_boxplot_image_tags.append(tex_tag) 
                                print(f"[DEBUG] _generate_mva_boxplots_for_comparison: Added boxplot image for '{var_name}' with tex_tag '{tex_tag}'.")
                            else:
                                dpg.add_text(f"Error generating boxplot for {var_name}", parent=row_group)
                                print(f"[DEBUG] _generate_mva_boxplots_for_comparison: Failed to generate texture for '{var_name}'.")
                        else:
                             dpg.add_text(f"No data for boxplot: {var_name}", parent=row_group)
                             print(f"[DEBUG] _generate_mva_boxplots_for_comparison: No data to plot for '{var_name}'.")
                    except Exception as e_box:
                        print(f"[DEBUG] _generate_mva_boxplots_for_comparison: Error generating boxplot for {var_name}: {e_box}")
                        dpg.add_text(f"Plot error: {var_name}", parent=row_group)



def _clear_mva_umap_plot():
    global _mva_active_umap_texture_id
    if not dpg.is_dearpygui_running(): return
    default_tex = _shared_utils_mva.get('default_mva_plot_texture_tag') or _shared_utils_mva.get('default_umap_texture_tag', TAG_OT_MVA_UMAP_DEFAULT_TEXTURE) # 부모의 기본 태그 또는 이 파일 내 정의
    
    if dpg.does_item_exist(TAG_OT_MVA_UMAP_PLOT_IMAGE) and dpg.does_item_exist(default_tex):
        cfg = dpg.get_item_configuration(default_tex); w,h = (cfg.get('width',100), cfg.get('height',30))
        dpg.configure_item(TAG_OT_MVA_UMAP_PLOT_IMAGE, texture_tag=default_tex, width=w, height=h)
    
    if _mva_active_umap_texture_id and _mva_active_umap_texture_id != default_tex and dpg.does_item_exist(_mva_active_umap_texture_id):
        try: dpg.delete_item(_mva_active_umap_texture_id)
        except Exception as e: _log_mva(f"Error deleting active UMAP texture: {e}")
    _mva_active_umap_texture_id = default_tex


def _clear_mva_pca_plot():
    global _mva_active_pca_texture_id
    if not dpg.is_dearpygui_running(): return
    default_tex = _shared_utils_mva.get('default_mva_plot_texture_tag') or _shared_utils_mva.get('default_pca_texture_tag', TAG_OT_MVA_PCA_DEFAULT_TEXTURE)

    if dpg.does_item_exist(TAG_OT_MVA_PCA_PLOT_IMAGE) and dpg.does_item_exist(default_tex):
        cfg = dpg.get_item_configuration(default_tex); w,h = (cfg.get('width',100), cfg.get('height',30))
        dpg.configure_item(TAG_OT_MVA_PCA_PLOT_IMAGE, texture_tag=default_tex, width=w, height=h)

    if _mva_active_pca_texture_id and _mva_active_pca_texture_id != default_tex and dpg.does_item_exist(_mva_active_pca_texture_id):
        try: dpg.delete_item(_mva_active_pca_texture_id)
        except Exception as e: _log_mva(f"Error deleting active PCA texture: {e}")
    _mva_active_pca_texture_id = default_tex

def _clear_mva_shap_plot():
    global _mva_active_shap_texture_id
    if not dpg.is_dearpygui_running(): return
    default_tex = _shared_utils_mva.get('default_shap_plot_texture_tag') # 부모에서 전달된 SHAP 기본 텍스처

    if dpg.does_item_exist(TAG_OT_MVA_SHAP_PLOT_IMAGE) and default_tex and dpg.does_item_exist(default_tex):
        cfg = dpg.get_item_configuration(default_tex); w,h = (cfg.get('width',100), cfg.get('height',30))
        dpg.configure_item(TAG_OT_MVA_SHAP_PLOT_IMAGE, texture_tag=default_tex, width=w, height=h)
    
    if _mva_active_shap_texture_id and _mva_active_shap_texture_id != default_tex and dpg.does_item_exist(_mva_active_shap_texture_id):
        try: dpg.delete_item(_mva_active_shap_texture_id)
        except Exception as e: _log_mva(f"Error deleting active SHAP texture: {e}")
    _mva_active_shap_texture_id = default_tex


def _clear_mva_boxplots():
    global _mva_boxplot_image_tags
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_OT_MVA_BOXPLOT_GROUP):
        dpg.delete_item(TAG_OT_MVA_BOXPLOT_GROUP, children_only=True) # 그룹 내 모든 자식 (이미지들) 삭제
    
    # 생성된 텍스처들도 삭제 (메모리 관리)
    for tex_tag in _mva_boxplot_image_tags:
        if dpg.does_item_exist(tex_tag):
            try: dpg.delete_item(tex_tag)
            except Exception as e: _log_mva(f"Error deleting boxplot texture {tex_tag}: {e}")
    _mva_boxplot_image_tags.clear()


def _clear_all_mva_visualizations():
    _clear_mva_umap_plot()
    _clear_mva_pca_plot()
    _clear_mva_shap_plot()
    _clear_mva_boxplots()
    _mva_selected_outlier_instance_idx = None # 선택된 인스턴스도 초기화

# --- Main UI Creation and Update ---
def create_multivariate_ui(parent_tab_bar_tag: str, shared_utilities: dict):
    global _shared_utils_mva
    _shared_utils_mva = shared_utilities

    # 부모로부터 기본 텍스처 태그 가져오기
    default_umap_tex = _shared_utils_mva.get('default_umap_texture_tag')
    default_pca_tex = _shared_utils_mva.get('default_pca_texture_tag')
    default_shap_tex = _shared_utils_mva.get('default_shap_plot_texture_tag')

    # 이 Multivariate 탭 전체를 담는 부모 (parent_tab_bar_tag에 추가될 탭)
    with dpg.tab(label="Multivariate Outlier Detection", tag=TAG_OT_MULTIVARIATE_TAB, parent=parent_tab_bar_tag):
        
        # --- 상단: 설정 및 실행 버튼 ---
        dpg.add_text("1. Configure & Run Multivariate Detection (Isolation Forest)", color=[255, 255, 0])
        with dpg.group(horizontal=True):
            dpg.add_text("Column Selection Mode:")
            dpg.add_radio_button(["All Numeric", "Recommended", "Manual"], tag=TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO, default_value=_mva_column_selection_mode, horizontal=True, callback=_on_mva_col_selection_mode_change)
        
        with dpg.group(tag=TAG_OT_MVA_MANUAL_COLUMN_SELECTOR_GROUP, show=False):
            dpg.add_text("Select Columns (click to toggle):")
            dpg.add_listbox([], tag=TAG_OT_MVA_COLUMN_SELECTOR_MULTI, num_items=6, callback=_on_mva_manual_cols_selected, width=-1)
        
        dpg.add_spacer(height=5)
        with dpg.group(horizontal=True):
            dpg.add_text("Contamination (0.0-0.5):")
            dpg.add_input_float(tag=TAG_OT_MVA_CONTAMINATION_INPUT, width=120, default_value=_mva_contamination, min_value=0.0001, max_value=0.5, step=0.01, format="%.4f", callback=_on_mva_contamination_change)
        
        with dpg.group(horizontal=True):
            enable_detection_button = 'PyOD_IForest' in globals() and PyOD_IForest is not None
            dpg.add_button(label="Run Multivariate Detection", tag=TAG_OT_MVA_DETECT_BUTTON, width=-1, height=30, callback=_run_mva_outlier_detection_logic, enabled=enable_detection_button)
            if not enable_detection_button:
                dpg.add_text(" (PyOD N/A)", color=[255,100,100], parent=dpg.last_item())
            dpg.add_button(label="Set Recommended MVA Params", tag=TAG_OT_MVA_RECOMMEND_PARAMS_BUTTON, width=-1, height=30, callback=_set_mva_recommended_parameters)
        dpg.add_separator()

        # --- 결과 표시를 위한 탭 영역 ---
        with dpg.tab_bar(tag="mva_results_tab_bar"):

            # 탭: Overview (UMAP & PCA)
            with dpg.tab(label="Overview: UMAP & PCA", tag="mva_tab_overview"):
                with dpg.group(tag="mva_summary_info_group"): # 요약 정보를 담을 그룹
                    dpg.add_text("Detection Summary:", color=[255, 255, 0])
                    dpg.add_text("Run detection to see the summary.", tag="mva_summary_text_status") # 초기 메시지
                    # 실제 요약 정보가 표시될 텍스트 위젯들 (초기에는 비어있거나 기본 메시지)
                    dpg.add_text("", tag="mva_summary_text_total_features")
                    dpg.add_text("", tag="mva_summary_text_used_features")
                    dpg.add_text("", tag="mva_summary_text_detected_outliers")
                    dpg.add_text("", tag="mva_summary_text_outlier_ratio")
                dpg.add_separator()
                dpg.add_text("2. UMAP & PCA Projection (Outliers Highlighted)", color=[255, 255, 0])
                with dpg.group(tag=TAG_OT_MVA_VISUALIZATION_GROUP, horizontal=True): # 가로 배치
                    init_w, init_h = 400, 350 # UMAP/PCA 이미지 크기 조절 (탭 내부 공간 고려)
                    
                    texture_to_use_for_umap = default_umap_tex if default_umap_tex and dpg.does_item_exist(default_umap_tex) else ""
                    show_umap_image = 'umap' in globals() and umap is not None
                    umap_image_group = dpg.add_group() # UMAP 이미지 또는 메시지용 그룹
                    if texture_to_use_for_umap and show_umap_image :
                        cfg_umap = dpg.get_item_configuration(texture_to_use_for_umap)
                        init_w_u, init_h_u = cfg_umap.get('width', init_w), cfg_umap.get('height', init_h)
                        dpg.add_image(texture_tag=texture_to_use_for_umap, tag=TAG_OT_MVA_UMAP_PLOT_IMAGE, width=init_w_u, height=init_h_u, parent=umap_image_group)
                    elif show_umap_image:
                        dpg.add_text("UMAP texture missing.", color=[255,0,0], parent=umap_image_group)
                    else:
                        dpg.add_text("UMAP N/A (library missing)", color=[255,100,100], parent=umap_image_group)
                        if dpg.does_item_exist(TAG_OT_MVA_UMAP_PLOT_IMAGE): dpg.configure_item(TAG_OT_MVA_UMAP_PLOT_IMAGE, show=False)

                    texture_to_use_for_pca = default_pca_tex if default_pca_tex and dpg.does_item_exist(default_pca_tex) else ""
                    pca_image_group = dpg.add_group() # PCA 이미지 또는 메시지용 그룹
                    if texture_to_use_for_pca :
                        cfg_pca = dpg.get_item_configuration(texture_to_use_for_pca)
                        init_w_p, init_h_p = cfg_pca.get('width', init_w), cfg_pca.get('height', init_h)
                        dpg.add_image(texture_tag=texture_to_use_for_pca, tag=TAG_OT_MVA_PCA_PLOT_IMAGE, width=init_w_p, height=init_h_p, parent=pca_image_group)
                    else:
                        dpg.add_text("PCA texture missing.", color=[255,0,0], parent=pca_image_group)

            # 탭: Detected Instances & Details
            with dpg.tab(label="Detected Instances & Details", tag="mva_tab_details"):
                dpg.add_text("4. Detected Multivariate Outlier Instances (Max 30, by Score)", color=[255, 255, 0])
                with dpg.table(tag=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=220, 
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                    dpg.add_table_column(label="Original Index", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, init_width_or_weight=0.4)
                    dpg.add_table_column(label="MVA Outlier Score", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, init_width_or_weight=0.6)
                    with dpg.table_row(parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE):
                        dpg.add_text("Run MVA detection.")
                        dpg.add_text("")
                
                # 통계량 테이블과 SHAP 플롯을 가로로 배치할 그룹
                with dpg.group(horizontal=True): # 가로 배치 시작
                    # 그룹 1: SHAP Values for Selected Instance (너비 40%)
                    with dpg.group(width=-0.4): # 전체 가로 그룹 너비의 40%
                        dpg.add_text("5. SHAP Values for Selected Instance", color=[255, 255, 0])
                        shap_image_parent_group_in_tab = dpg.add_group() 
                        
                        default_shap_tex = _shared_utils_mva.get('default_shap_plot_texture_tag')
                        init_w_shap, init_h_shap = -1, 430 # 너비는 그룹에 맞추고, 높이는 테이블과 비슷하게
                        texture_to_use_for_shap_init = ""

                        if default_shap_tex and dpg.does_item_exist(default_shap_tex):
                            texture_to_use_for_shap_init = default_shap_tex
                        
                        show_shap_image = 'shap' in globals() and shap is not None
                        
                        if texture_to_use_for_shap_init : 
                            dpg.add_image(texture_tag=texture_to_use_for_shap_init, tag=TAG_OT_MVA_SHAP_PLOT_IMAGE,
                                        width=init_w_shap, height=init_h_shap, parent=shap_image_parent_group_in_tab, show=False) 
                        
                        if not show_shap_image:
                             dpg.add_text("SHAP N/A (library missing)", color=[255,100,100], parent=shap_image_parent_group_in_tab, tag="mva_shap_status_text")
                        elif not _mva_selected_outlier_instance_idx: 
                             dpg.add_text("Select an instance to see SHAP values.", parent=shap_image_parent_group_in_tab, tag="mva_shap_status_text")

                    # 그룹 2: Statistics for Selected Instance (너비 60%)
                    with dpg.group(width=0): # 남은 공간 모두 사용 (60%)
                        dpg.add_text("6. Statistics for Selected Instance", color=[255, 255, 0])
                        with dpg.table(tag=TAG_OT_MVA_INSTANCE_STATS_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=450, # 높이 증가
                                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                            dpg.add_table_column(label="Feature", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, init_width_or_weight=0.25)
                            dpg.add_table_column(label="Value", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, init_width_or_weight=0.20)
                            dpg.add_table_column(label="Mean", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, init_width_or_weight=0.20)
                            dpg.add_table_column(label="Median", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, init_width_or_weight=0.20)
                            dpg.add_table_column(label="Z-Dist.", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, init_width_or_weight=0.20)
                            with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE):
                                dpg.add_text("Select an instance.")
                                dpg.add_text("")
                                dpg.add_text("")
                                dpg.add_text("")
                                dpg.add_text("")

            # 탭: Variable Box Plots
            with dpg.tab(label="Variable Box Plots", tag="mva_tab_boxplots"):
                dpg.add_text("3. Variable Comparison: Outlier vs. Normal (Top 10 by Median Gap)", color=[255, 255, 0])
                # Box Plot 영역은 내용이 많아질 수 있으므로 Child Window 사용 권장
                with dpg.child_window(tag=TAG_OT_MVA_BOXPLOT_GROUP, border=True): # 높이는 자동으로 채워지도록 하거나, 고정값 후 스크롤
                    dpg.add_text("Run detection to see boxplots.")

    _update_mva_param_visibility()

def update_multivariate_ui(df_input: Optional[pd.DataFrame], shared_utilities: dict, is_new_data: bool):
    global _shared_utils_mva, _current_df_for_mva
    if not dpg.is_dearpygui_running(): return
    _shared_utils_mva = shared_utilities
    
    # 부모로부터 전달된 default texture tag 업데이트 (필요시)
    # _mva_active_umap_texture_id = _shared_utils_mva.get('default_umap_texture_tag', TAG_OT_MVA_UMAP_DEFAULT_TEXTURE)
    # ... 등등

    if not dpg.does_item_exist(TAG_OT_MULTIVARIATE_TAB): return

    _current_df_for_mva = df_input

    if _current_df_for_mva is None or is_new_data:
        _log_mva("New data or no data for MVA. Resetting MVA state.")
        reset_multivariate_state_internal(called_from_parent_reset=False)
        if _current_df_for_mva is not None:
            _update_selected_columns_for_mva_detection() # 새 데이터에 대한 eligible 컬럼 업데이트
    
    # UI 요소 값 업데이트 (예: listbox 아이템)
    if _current_df_for_mva is not None and dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTOR_MULTI):
        if is_new_data: # 새 데이터일 때만 명시적으로 listbox 아이템 업데이트
            _update_selected_columns_for_mva_detection()


def reset_multivariate_state_internal(called_from_parent_reset=True):
    global _current_df_for_mva, _df_with_mva_outliers, _mva_model, _mva_shap_explainer
    global _mva_eligible_numeric_cols, _mva_selected_columns_for_detection, _mva_column_selection_mode, _mva_contamination
    global _mva_outlier_instances_summary, _mva_selected_outlier_instance_idx, _mva_all_selectable_tags_in_instances_table
    global _mva_top_gap_vars_for_boxplot

    # _current_df_for_mva는 update_multivariate_ui에서 관리. 여기서는 detection 결과물 위주로 리셋.
    _df_with_mva_outliers = None
    _mva_model = None
    _mva_shap_explainer = None
    _mva_eligible_numeric_cols = []
    _mva_selected_columns_for_detection = []
    _mva_column_selection_mode = "All Numeric"
    _mva_contamination = DEFAULT_MVA_CONTAMINATION
    _mva_outlier_instances_summary = []
    _mva_selected_outlier_instance_idx = None
    _mva_all_selectable_tags_in_instances_table.clear()
    _mva_top_gap_vars_for_boxplot = []

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO):
            dpg.set_value(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO, _mva_column_selection_mode)
        if dpg.does_item_exist(TAG_OT_MVA_CONTAMINATION_INPUT):
            dpg.set_value(TAG_OT_MVA_CONTAMINATION_INPUT, _mva_contamination)
        if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTOR_MULTI):
            dpg.configure_item(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, items=[]) # eligible 목록 비우기
            dpg.set_value(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, []) # 선택값도 비우기
        if dpg.does_item_exist("mva_summary_text_status"):
            dpg.set_value("mva_summary_text_status", "Run detection to see the summary.")
        for tag_suffix in ["total_features", "used_features", "detected_outliers", "outlier_ratio"]:
            if dpg.does_item_exist(f"mva_summary_text_{tag_suffix}"):
                dpg.set_value(f"mva_summary_text_{tag_suffix}", "")
        
        _update_mva_param_visibility()
        _populate_mva_outlier_instances_table() # 빈 테이블
        _clear_all_mva_visualizations()

    if not called_from_parent_reset:
        _log_mva("Multivariate outlier state reset (internal).")


def reset_multivariate_state(): # 부모 모듈에서 호출
    reset_multivariate_state_internal(called_from_parent_reset=True)
    _log_mva("Multivariate outlier state has been reset by parent.")


def get_multivariate_settings() -> dict:
    return {
        "mva_column_selection_mode": _mva_column_selection_mode,
        "mva_selected_columns_for_detection": _mva_selected_columns_for_detection[:], # 복사본
        "mva_contamination": _mva_contamination,
        # 탐지 결과나 모델은 저장하지 않음 (런타임에 재계산)
    }

def apply_multivariate_settings(df_input: Optional[pd.DataFrame], settings: dict, shared_utilities: dict):
    global _shared_utils_mva, _current_df_for_mva
    global _mva_column_selection_mode, _mva_selected_columns_for_detection, _mva_contamination
    
    _shared_utils_mva = shared_utilities
    _current_df_for_mva = df_input # 현재 DF 설정

    _mva_column_selection_mode = settings.get("mva_column_selection_mode", "All Numeric")
    _mva_selected_columns_for_detection = settings.get("mva_selected_columns_for_detection", [])[:]
    _mva_contamination = settings.get("mva_contamination", DEFAULT_MVA_CONTAMINATION)

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO):
            dpg.set_value(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO, _mva_column_selection_mode)
        if dpg.does_item_exist(TAG_OT_MVA_CONTAMINATION_INPUT):
            dpg.set_value(TAG_OT_MVA_CONTAMINATION_INPUT, _mva_contamination)
        
        _update_mva_param_visibility()
        if _current_df_for_mva is not None: # DF가 있어야 eligible cols 업데이트 가능
            _update_selected_columns_for_mva_detection() # listbox 아이템 및 선택값 업데이트
        else: # DF가 없으면 listbox 비우기
             if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTOR_MULTI):
                dpg.configure_item(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, items=[])
                dpg.set_value(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, [])


        # 이전 탐지 결과는 복원하지 않으므로, 테이블과 플롯은 초기 상태로.
        _populate_mva_outlier_instances_table()
        _clear_all_mva_visualizations()

    _log_mva("Multivariate outlier settings applied from saved state. Please re-run detection if needed.")
    # update_multivariate_ui(df_input, shared_utilities, True) # is_new_data=True로 간주하여 UI 전체 갱신