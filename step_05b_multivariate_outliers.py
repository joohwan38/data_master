# step_05b_multivariate_outliers.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pyod.models.iforest import IForest as PyOD_IForest
import umap
import shap

# --- DPG Tags for Multivariate Tab ---
TAG_OT_MULTIVARIATE_TAB = "step5_multivariate_outlier_tab"
TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO = "step5_ot_mva_col_selection_mode_radio"
TAG_OT_MVA_MANUAL_COLUMN_SELECTOR_GROUP = "step5_ot_mva_manual_col_selector_group"
TAG_OT_MVA_COLUMN_SELECTOR_MULTI = "step5_ot_mva_col_selector_multi"
TAG_OT_MVA_CONTAMINATION_INPUT = "step5_ot_mva_contamination_input"
TAG_OT_MVA_DETECT_BUTTON = "step5_ot_mva_detect_button"
TAG_OT_MVA_RECOMMEND_PARAMS_BUTTON = "step5_ot_mva_recommend_params_button"
TAG_OT_MVA_VISUALIZATION_GROUP = "step5_ot_mva_visualization_group"
TAG_OT_MVA_UMAP_PLOT_IMAGE = "step5_ot_mva_umap_plot_image"
TAG_OT_MVA_PCA_PLOT_IMAGE = "step5_ot_mva_pca_plot_image"
TAG_OT_MVA_UMAP_DEFAULT_TEXTURE = "step5_ot_mva_umap_default_texture"
TAG_OT_MVA_PCA_DEFAULT_TEXTURE = "step5_ot_mva_pca_default_texture"
TAG_OT_MVA_SHAP_PLOT_IMAGE = "step5_ot_mva_shap_plot_image"
TAG_OT_MVA_UMAP_PARENT_GROUP = "step5_ot_mva_umap_parent_group"
TAG_OT_MVA_PCA_PARENT_GROUP = "step5_ot_mva_pca_parent_group"
TAG_OT_MVA_SHAP_PARENT_GROUP = "step5_ot_mva_shap_parent_group"
TAG_OT_MVA_OUTLIER_INSTANCES_TABLE = "step5_ot_mva_outlier_instances_table"
TAG_OT_MVA_INSTANCE_STATS_TABLE = "step5_ot_mva_instance_stats_table"
TAG_OT_MVA_BOXPLOT_GROUP = "step5_ot_mva_boxplot_group"

# --- Constants ---
DEFAULT_MVA_CONTAMINATION = 0.1
MAX_OUTLIER_INSTANCES_TO_SHOW = 30
TOP_N_VARIABLES_FOR_BOXPLOT = 10
MIN_SAMPLES_FOR_IFOREST = 5
MIN_FEATURES_FOR_IFOREST = 2

# --- Module State Variables ---
_shared_utils_mva: Optional[Dict[str, Any]] = None
_current_df_for_mva: Optional[pd.DataFrame] = None
_df_with_mva_outliers: Optional[pd.DataFrame] = None
_mva_model: Optional[Any] = None
_mva_shap_explainer: Optional[Any] = None
_mva_scaler: Optional[StandardScaler] = None
_mva_eligible_numeric_cols: List[str] = []
_mva_selected_columns_for_detection: List[str] = []
_mva_column_selection_mode: str = "All Numeric"
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

def _log_mva(message: str, level: str = "DEBUG"): # level 인자 추가 및 기본값 설정
    if _shared_utils_mva and 'log_message_func' in _shared_utils_mva:
        # main_app의 log_message_func가 level 인자를 받을 수 있도록 수정되었거나,
        # 여기서 level을 메시지에 포함시켜 전달해야 함.
        # 여기서는 메시지에 레벨을 포함시키는 방식으로 가정.
        _shared_utils_mva['log_message_func'](f"[{level.upper()}] [MVAOutlier] {message}")
    else:
        # main_app의 로거가 없을 경우를 대비한 기본 print
        print(f"[{level.upper()}] [MVAOutlier] {message}")

def _show_simple_modal_mva(title: str, message: str, width: int = 450, height: int = 200):
    if _shared_utils_mva and 'util_funcs_common' in _shared_utils_mva and \
       '_show_simple_modal_message' in _shared_utils_mva['util_funcs_common']:
        _shared_utils_mva['util_funcs_common']['_show_simple_modal_message'](title, message, width, height)
    else:
        _log_mva(f"Modal display function not available. Title: {title}, Msg: {message}")

def _get_shared_callbacks_for_image_util() -> Dict[str, Callable]:
    if not _shared_utils_mva or 'main_app_callbacks' not in _shared_utils_mva:
        _log_mva("오류: main_app_callbacks를 찾을 수 없어 이미지 유틸 콜백을 구성할 수 없습니다.")
        return {}
    main_callbacks = _shared_utils_mva['main_app_callbacks']
    return {
        'cache_image_data_func': main_callbacks.get('cache_image_data_func'),
        'initiate_ollama_analysis': main_callbacks.get('initiate_ollama_analysis')
    }

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
    _update_selected_columns_for_mva_detection()

def _on_mva_manual_cols_selected(sender, app_data: str, user_data):
    global _mva_selected_columns_for_detection
    selected_item = app_data
    if selected_item: _mva_selected_columns_for_detection = [selected_item]
    else: _mva_selected_columns_for_detection = []
    _log_mva(f"Manually selected column for MVA: {_mva_selected_columns_for_detection}")

def _on_mva_contamination_change(sender, app_data: float, user_data):
    global _mva_contamination
    if 0.0 < app_data <= 0.5: _mva_contamination = app_data; _log_mva(f"Multivariate contamination set to: {_mva_contamination:.4f}")
    else:
        dpg.set_value(sender, _mva_contamination)
        err_msg = "Contamination must be between 0 (exclusive) and 0.5 (inclusive)."
        _log_mva(f"Invalid input for contamination: {app_data}. {err_msg}"); _show_simple_modal_mva("Input Error", err_msg)

def _set_mva_recommended_parameters(sender, app_data, user_data):
    global _mva_contamination, _mva_column_selection_mode
    _mva_contamination = DEFAULT_MVA_CONTAMINATION; _mva_column_selection_mode = "Recommended"
    if dpg.does_item_exist(TAG_OT_MVA_CONTAMINATION_INPUT): dpg.set_value(TAG_OT_MVA_CONTAMINATION_INPUT, _mva_contamination)
    if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO): dpg.set_value(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO, _mva_column_selection_mode)
    _update_mva_param_visibility(); _update_selected_columns_for_mva_detection()
    _log_mva(f"Recommended MVA parameters set: Contamination={_mva_contamination}, Mode='Recommended'")
    _show_simple_modal_mva("Info", "Recommended multivariate detection parameters have been applied.")

def _get_eligible_numeric_cols_for_mva(df: pd.DataFrame) -> List[str]:
    if df is None: return []
    numeric_cols = []; s1_col_types = {}
    if _shared_utils_mva and 'main_app_callbacks' in _shared_utils_mva and 'get_column_analysis_types' in _shared_utils_mva['main_app_callbacks']:
        s1_col_types = _shared_utils_mva['main_app_callbacks']['get_column_analysis_types']()
    for col in df.columns:
        s1_type = s1_col_types.get(col, ""); is_numeric_s1 = "Numeric" in s1_type and "Binary" not in s1_type
        is_numeric_pandas = pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].nunique() > 5
        if (is_numeric_s1 or is_numeric_pandas) and df[col].isnull().sum() / len(df) < 0.8 and df[col].dropna().var(ddof=0) > 1e-6:
            numeric_cols.append(col)
    return numeric_cols

def _update_selected_columns_for_mva_detection():
    global _mva_selected_columns_for_detection, _mva_eligible_numeric_cols, _mva_last_recommendation_details_str
    _mva_last_recommendation_details_str = None
    current_df = _shared_utils_mva.get('get_current_df_func', lambda: None)()
    if current_df is None:
        _mva_eligible_numeric_cols, _mva_selected_columns_for_detection = [], []
        if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTOR_MULTI):
            dpg.configure_item(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, items=[]); dpg.set_value(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, "")
        return
    _mva_eligible_numeric_cols = _get_eligible_numeric_cols_for_mva(current_df)
    if _mva_column_selection_mode == "All Numeric":
        _mva_selected_columns_for_detection = _mva_eligible_numeric_cols[:]; _mva_last_recommendation_details_str = "Used all eligible numeric features."
    elif _mva_column_selection_mode == "Recommended":
        target_var, target_var_type, recommendation_reason = None, None, "all eligible numeric features (fallback criteria)."
        if _shared_utils_mva and 'main_app_callbacks' in _shared_utils_mva:
            get_target_var_func = _shared_utils_mva['main_app_callbacks'].get('get_selected_target_variable')
            get_target_type_func = _shared_utils_mva['main_app_callbacks'].get('get_selected_target_variable_type')
            if get_target_var_func: target_var = get_target_var_func()
            if get_target_type_func: target_var_type = get_target_type_func()
        eligible_cols_for_relevance = [col for col in _mva_eligible_numeric_cols if col != target_var]
        if target_var and target_var_type and eligible_cols_for_relevance and current_df is not None:
            relevance_scores = []
            if _shared_utils_mva and 'util_funcs_common' in _shared_utils_mva and 'calculate_feature_target_relevance' in _shared_utils_mva['util_funcs_common']:
                 relevance_scores = _shared_utils_mva['util_funcs_common']['calculate_feature_target_relevance'](current_df, target_var, target_var_type, eligible_cols_for_relevance, _shared_utils_mva.get('main_app_callbacks'))
            if relevance_scores:
                n1, n2 = 20, int(len(eligible_cols_for_relevance) * 0.20); num_to_select = max(n1, n2)
                _mva_selected_columns_for_detection = [feat for feat, score in relevance_scores[:num_to_select]]
                recommendation_reason = f"top {len(_mva_selected_columns_for_detection)} features by relevance to target '{target_var}' (N=max(20, 20%))."
            else: _mva_selected_columns_for_detection = _mva_eligible_numeric_cols[:]
        else: _mva_selected_columns_for_detection = _mva_eligible_numeric_cols[:]
        _mva_last_recommendation_details_str = f"Recommended features selected: {recommendation_reason}"
    elif _mva_column_selection_mode == "Manual":
        _mva_selected_columns_for_detection = [col for col in _mva_selected_columns_for_detection if col in _mva_eligible_numeric_cols]
        _mva_last_recommendation_details_str = f"Used {len(_mva_selected_columns_for_detection)} manually selected features."
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTOR_MULTI):
        dpg.configure_item(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, items=_mva_eligible_numeric_cols)
        dpg.set_value(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, _mva_selected_columns_for_detection[0] if _mva_column_selection_mode == "Manual" and _mva_selected_columns_for_detection else "")

def _update_mva_boxplots_for_comparison():
    _log_mva("Called: _update_mva_boxplots_for_comparison")
    if not dpg.is_dearpygui_running(): _log_mva("_update_mva_boxplots_for_comparison: DPG not running."); return
    if _df_with_mva_outliers is None:
        _log_mva("_update_mva_boxplots_for_comparison: No MVA outlier data. Clearing boxplots.")
        _clear_mva_boxplots()
        if dpg.does_item_exist(TAG_OT_MVA_BOXPLOT_GROUP):
            dpg.delete_item(TAG_OT_MVA_BOXPLOT_GROUP, children_only=True)
            dpg.add_text("Run detection to see boxplots.", parent=TAG_OT_MVA_BOXPLOT_GROUP)
        return
    _log_mva("_update_mva_boxplots_for_comparison: Generating MVA boxplots...")
    _generate_mva_boxplots_for_comparison()

def _run_mva_outlier_detection_logic(sender, app_data, user_data):
    global _df_with_mva_outliers, _mva_model, _mva_shap_explainer, _mva_outlier_instances_summary, _mva_scaler
    current_df = _shared_utils_mva.get('get_current_df_func', lambda: None)()
    if current_df is None: _show_simple_modal_mva("Error", "No data for multivariate outlier detection."); return
    _update_selected_columns_for_mva_detection()
    if not _mva_selected_columns_for_detection or len(_mva_selected_columns_for_detection) < MIN_FEATURES_FOR_IFOREST:
        _show_simple_modal_mva("Error", f"Please select at least {MIN_FEATURES_FOR_IFOREST} numeric features."); return
    df_for_detection_raw = current_df[_mva_selected_columns_for_detection].copy()
    df_for_detection = df_for_detection_raw.dropna()
    if len(df_for_detection) < MIN_SAMPLES_FOR_IFOREST:
        _show_simple_modal_mva("Error", f"Not enough data rows (after NaN removal). Min required: {MIN_SAMPLES_FOR_IFOREST}."); return
    try:
        _mva_scaler = StandardScaler()
        scaled_data_for_iforest_and_reduction = _mva_scaler.fit_transform(df_for_detection.values)
        _mva_model = PyOD_IForest(contamination=_mva_contamination, random_state=42, n_jobs=-1)
        _mva_model.fit(scaled_data_for_iforest_and_reduction)
        outlier_scores, outlier_labels = _mva_model.decision_scores_, _mva_model.labels_
        _df_with_mva_outliers = current_df.copy()
        _df_with_mva_outliers.loc[df_for_detection.index, 'mva_outlier_score'] = outlier_scores
        _df_with_mva_outliers.loc[df_for_detection.index, 'mva_is_outlier'] = outlier_labels.astype(bool)
        _df_with_mva_outliers['mva_is_outlier'] = _df_with_mva_outliers['mva_is_outlier'].fillna(False)
        min_score_val = np.min(outlier_scores) if len(outlier_scores) > 0 else 0
        _df_with_mva_outliers['mva_outlier_score'] = _df_with_mva_outliers['mva_outlier_score'].fillna(min_score_val - 1)
        if 'shap' in globals() and shap is not None:
            try: _mva_shap_explainer = shap.TreeExplainer(_mva_model.detector_, data=pd.DataFrame(scaled_data_for_iforest_and_reduction, columns=df_for_detection.columns))
            except Exception as e_shap_init: _log_mva(f"Error initializing SHAP explainer: {e_shap_init}"); _mva_shap_explainer = None
        else: _mva_shap_explainer = None
        _mva_outlier_instances_summary = []
        detected_outliers_df = _df_with_mva_outliers[_df_with_mva_outliers['mva_is_outlier'] == True]
        if not detected_outliers_df.empty and 'mva_outlier_score' in detected_outliers_df.columns:
            detected_outliers_df = detected_outliers_df.sort_values(by='mva_outlier_score', ascending=False)
        for original_idx, row in detected_outliers_df.head(MAX_OUTLIER_INSTANCES_TO_SHOW).iterrows():
            _mva_outlier_instances_summary.append({"Original Index": original_idx, "MVA Outlier Score": f"{row['mva_outlier_score']:.4f}"})
        _populate_mva_outlier_instances_table()
        _generate_mva_umap_pca_plots(scaled_data_for_iforest_and_reduction, df_for_detection.index, outlier_labels)
        _update_mva_boxplots_for_comparison()
        if _df_with_mva_outliers is not None and _current_df_for_mva is not None:
            num_total_samples, num_detected_outliers = len(_current_df_for_mva), _df_with_mva_outliers['mva_is_outlier'].sum()
            outlier_ratio = (num_detected_outliers / num_total_samples) * 100 if num_total_samples > 0 else 0
            num_total_features_in_input, num_used_features = len(_current_df_for_mva.columns), len(_mva_selected_columns_for_detection)
            for tag, val in [("status", "Detection Complete. Summary:"),
                             ("total_features", f"  - Total Features in Input Data: {num_total_features_in_input}"),
                             ("used_features", f"  - Features Used for Detection: {num_used_features}"),
                             ("detected_outliers", f"  - Detected Outlier Instances: {num_detected_outliers} samples"),
                             ("outlier_ratio", f"  - Outlier Ratio: {outlier_ratio:.2f}% of total {num_total_samples} samples")]:
                if dpg.does_item_exist(f"mva_summary_text_{tag}"): dpg.set_value(f"mva_summary_text_{tag}", val)
        completion_message = "Multivariate outlier detection finished."
        if _mva_column_selection_mode == "Recommended" and _mva_last_recommendation_details_str: completion_message += f"\n\n[Recommendation Info]\n{_mva_last_recommendation_details_str}"
        elif _mva_last_recommendation_details_str: completion_message += f"\n\n[Selection Info]\n{_mva_last_recommendation_details_str}"
        _show_simple_modal_mva("Detection Complete", completion_message, height=min(max(200, (completion_message.count('\n') + 1) * 25), 400))
    except Exception as e:
        _log_mva(f"Error during MVA detection: {e}\n{traceback.format_exc()}")
        _show_simple_modal_mva("Detection Error", f"An error occurred: {e}")
        _df_with_mva_outliers, _mva_model, _mva_shap_explainer, _mva_scaler = None, None, None, None
        _clear_all_mva_visualizations(); _populate_mva_outlier_instances_table()

def _populate_mva_outlier_instances_table():
    global _mva_all_selectable_tags_in_instances_table
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_MVA_OUTLIER_INSTANCES_TABLE): return
    dpg.delete_item(TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, children_only=True); _mva_all_selectable_tags_in_instances_table.clear()
    if not _mva_outlier_instances_summary:
        dpg.add_table_column(label="Info", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE)
        with dpg.table_row(parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE): dpg.add_text("No multivariate outliers detected or not run.")
        _clear_mva_instance_details(); return
    dpg.add_table_column(label="Original Index", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, init_width_or_weight=0.4)
    dpg.add_table_column(label="MVA Outlier Score (Higher is more outlier)", parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, init_width_or_weight=0.6)
    for i, item_data in enumerate(_mva_outlier_instances_summary):
        original_idx, score_str = item_data["Original Index"], item_data["MVA Outlier Score"]
        tag = f"mva_instance_selectable_{i}_{original_idx}_{dpg.generate_uuid()}"
        _mva_all_selectable_tags_in_instances_table.append(tag)
        with dpg.table_row(parent=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE):
            dpg.add_selectable(label=str(original_idx), tag=tag, user_data=original_idx, callback=_on_mva_outlier_instance_selected, span_columns=False)
            dpg.add_text(score_str)

def _on_mva_outlier_instance_selected(sender, app_data_is_selected: bool, user_data_original_idx: Any):
    global _mva_selected_outlier_instance_idx
    if app_data_is_selected:
        for tag_iter in _mva_all_selectable_tags_in_instances_table:
            if tag_iter != sender and dpg.does_item_exist(tag_iter) and dpg.get_value(tag_iter): dpg.set_value(tag_iter, False)
        _mva_selected_outlier_instance_idx = user_data_original_idx
        _log_mva(f"MVA Outlier instance selected: Index {user_data_original_idx}")
        _display_mva_instance_statistics(user_data_original_idx); _generate_mva_shap_plot_for_instance(user_data_original_idx)
    elif _mva_selected_outlier_instance_idx == user_data_original_idx:
         _mva_selected_outlier_instance_idx = None; _clear_mva_instance_details()

def _display_mva_instance_statistics(original_idx: Any):
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_MVA_INSTANCE_STATS_TABLE): return
    dpg.delete_item(TAG_OT_MVA_INSTANCE_STATS_TABLE, children_only=True)
    if _df_with_mva_outliers is None or original_idx not in _df_with_mva_outliers.index:
        dpg.add_table_column(label="Info", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE)
        with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE): dpg.add_text("Selected instance data not available."); return
    instance_data_series = _df_with_mva_outliers.loc[original_idx]; feature_stats_list = []
    cols_to_process_for_stats = _mva_selected_columns_for_detection[:] if _mva_selected_columns_for_detection else []
    source_df_for_overall_stats = _current_df_for_mva
    for feature_name in cols_to_process_for_stats:
        if feature_name in instance_data_series:
            value = instance_data_series[feature_name]; overall_mean_val, overall_median_val, z_score_abs_val = np.nan, np.nan, 0.0
            overall_mean_str, overall_median_str, z_score_str = "N/A", "N/A", "N/A"
            if source_df_for_overall_stats is not None and feature_name in source_df_for_overall_stats.columns:
                feature_series_overall = source_df_for_overall_stats[feature_name].dropna()
                if not feature_series_overall.empty and pd.api.types.is_numeric_dtype(feature_series_overall.dtype):
                    overall_mean_val, overall_median_val, std_val = feature_series_overall.mean(), feature_series_overall.median(), feature_series_overall.std()
                    overall_mean_str, overall_median_str = f"{overall_mean_val:.4f}", f"{overall_median_val:.4f}"
                    if pd.notna(std_val) and std_val > 1e-9 and pd.notna(value) and pd.api.types.is_numeric_dtype(type(value)) and pd.notna(overall_mean_val):
                        z_score = (value - overall_mean_val) / std_val; z_score_abs_val, z_score_str = abs(z_score), f"{z_score:.2f}"
                    elif pd.notna(value) and pd.api.types.is_numeric_dtype(type(value)): z_score_str = "N/A (std~0)"
            feature_stats_list.append({"feature": feature_name, "value_str": f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value),
                                       "mean_str": overall_mean_str, "median_str": overall_median_str, "z_score_str": z_score_str,
                                       "z_score_abs_sort_key": z_score_abs_val if pd.notna(z_score_abs_val) else -1})
    sorted_feature_stats_list = sorted(feature_stats_list, key=lambda x: x["z_score_abs_sort_key"], reverse=True)
    headers = ["Feature", "Value", "Overall Mean", "Overall Median", "Z-score Dist."]
    widths = [0.25, 0.15, 0.20, 0.20, 0.20]
    for lbl, w in zip(headers, widths): dpg.add_table_column(label=lbl, parent=TAG_OT_MVA_INSTANCE_STATS_TABLE, init_width_or_weight=w)
    for stats_item in sorted_feature_stats_list:
        with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE):
            for key in ["feature", "value_str", "mean_str", "median_str", "z_score_str"]: dpg.add_text(stats_item[key])
    if 'mva_outlier_score' in instance_data_series:
        with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE):
            dpg.add_text("MVA Outlier Score", color=[255,255,0]); dpg.add_text(f"{instance_data_series['mva_outlier_score']:.4f}"); dpg.add_text("-"); dpg.add_text("-"); dpg.add_text("-")

def _clear_mva_instance_details():
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_OT_MVA_INSTANCE_STATS_TABLE):
        dpg.delete_item(TAG_OT_MVA_INSTANCE_STATS_TABLE, children_only=True)
        dpg.add_table_column(label="Info", parent=TAG_OT_MVA_INSTANCE_STATS_TABLE)
        with dpg.table_row(parent=TAG_OT_MVA_INSTANCE_STATS_TABLE): dpg.add_text("Select an outlier instance from the table above.")
    _clear_mva_shap_plot()

def _generate_mva_umap_pca_plots(data_for_reduction: np.ndarray, original_indices: pd.Index, outlier_labels: np.ndarray):
    global _mva_active_umap_texture_id, _mva_active_pca_texture_id
    
    # utils.py의 create_analyzable_image_widget 함수 가져오기
    create_widget_func = None
    if _shared_utils_mva and 'util_funcs_common' in _shared_utils_mva and \
       'create_analyzable_image_widget' in _shared_utils_mva['util_funcs_common']:
        create_widget_func = _shared_utils_mva['util_funcs_common']['create_analyzable_image_widget']
    
    if not callable(create_widget_func):
        _log_mva("Error: create_analyzable_image_widget function not available in shared utils.", "ERROR")
        return

    # Ollama 연동 및 이미지 캐싱에 필요한 콜백 가져오기
    mva_shared_image_callbacks = {
    'cache_image_data_func': _shared_utils_mva['main_app_callbacks'].get('cache_image_data_func'),
    'ask_for_ollama_confirmation': _shared_utils_mva['main_app_callbacks'].get('ask_for_ollama_confirmation')
    # 주석 처리된 부분에 대한 설명:
    # 'initiate_ollama_analysis'는 이제 'ask_for_ollama_confirmation' 콜백 내부에서
    # 사용자가 "예"를 선택했을 때 main_app.py를 통해 간접적으로 호출됩니다.
    # (정확히는 main_app.py의 _ollama_analysis_confirmed_callback -> initiate_ollama_analysis_with_window)
    # 따라서 utils.py의 create_analyzable_image_widget 함수는 더 이상
    # 'initiate_ollama_analysis' 콜백을 직접 받아서 사용할 필요가 없습니다.
    # 클릭 시 'ask_for_ollama_confirmation'만 호출하면 되기 때문입니다.
}
    if not mva_shared_image_callbacks.get('cache_image_data_func') or \
       not mva_shared_image_callbacks.get('initiate_ollama_analysis'):
        _log_mva("UMAP/PCA 생성 실패: 필수 이미지 유틸리티 콜백 누락.")
        return

    prev_umap_tex_id = _mva_active_umap_texture_id
    prev_pca_tex_id = _mva_active_pca_texture_id
    
    # 기존 시각화 클리어 (텍스처 자체는 utils 함수가 관리)
    _clear_mva_umap_plot() # 이 함수는 위젯을 기본 상태로 돌리고 핸들러 해제
    _clear_mva_pca_plot()  # 이 함수는 위젯을 기본 상태로 돌리고 핸들러 해제

    if data_for_reduction is None or len(data_for_reduction) < 2 or data_for_reduction.shape[1] < 2:
        _log_mva("Not enough data for UMAP/PCA. Skipping visualization.")
        return
    
    if len(data_for_reduction) != len(outlier_labels) or len(data_for_reduction) != len(original_indices): _log_mva("UMAP/PCA 데이터 길이 불일치"); return
    create_widget_func = _shared_utils_mva.get('util_funcs_common', {}).get('create_analyzable_image_widget')
    if not callable(create_widget_func): _log_mva("UMAP/PCA 생성 실패: 위젯 생성 함수 없음."); return
    if umap:
        try:
            _log_mva("Generating UMAP plot...")
            reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
            embedding_umap = reducer_umap.fit_transform(data_for_reduction)
            
            fig_umap, ax_umap = plt.subplots(figsize=(7, 5))
            # ... (UMAP 플롯 그리기) ...
            ax_umap.scatter(embedding_umap[:, 0], embedding_umap[:, 1], c=outlier_labels, cmap='coolwarm', s=15, alpha=0.7)
            ax_umap.set_title("UMAP Projection", fontsize=10); ax_umap.set_xlabel("UMAP 1", fontsize=8); ax_umap.set_ylabel("UMAP 2", fontsize=8)
            ax_umap.tick_params(axis='both',which='major',labelsize=7); legend_elements = [plt.Line2D([0],[0],marker='o',color='w',label='Normal',markerfacecolor='blue',markersize=5), plt.Line2D([0],[0],marker='o',color='w',label='Outlier',markerfacecolor='red',markersize=5)]
            ax_umap.legend(handles=legend_elements, loc='best', fontsize=7); plt.tight_layout()

            # create_analyzable_image_widget 직접 호출
            new_tex_id, _ = create_widget_func(
                parent_dpg_tag=TAG_OT_MVA_UMAP_PARENT_GROUP,
                fig=fig_umap,
                image_widget_tag=TAG_OT_MVA_UMAP_PLOT_IMAGE,
                image_title="UMAP Projection",
                shared_callbacks=shared_image_callbacks,
                default_texture_tag=_shared_utils_mva.get('default_umap_texture_tag', TAG_OT_MVA_UMAP_DEFAULT_TEXTURE),
                existing_dpg_texture_id=prev_umap_tex_id
            )
            _mva_active_umap_texture_id = str(new_tex_id) if new_tex_id else None

            _log_mva("UMAP plot " + ("generated" if new_tex_id else "failed"))
        except Exception as e_umap:
            _log_mva(f"UMAP plot error: {e_umap}")
            if 'fig_umap' in locals() and hasattr(fig_umap, 'canvas') and fig_umap.canvas.manager:
                plt.close(fig_umap)
    else:
        _log_mva("UMAP library N/A.")

    # PCA (위와 유사하게 create_widget_func 직접 호출)
    try:
        _log_mva("Generating PCA plot...")
        pca = PCA(n_components=2, random_state=42)
        embedding_pca = pca.fit_transform(data_for_reduction)
        fig_pca, ax_pca = plt.subplots(figsize=(7, 5))
        # ... (PCA 플롯 그리기) ...
        ax_pca.scatter(embedding_pca[:, 0], embedding_pca[:, 1], c=outlier_labels, cmap='viridis', s=15, alpha=0.7)
        ax_pca.set_title("PCA Projection", fontsize=10); ax_pca.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=8); ax_pca.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=8)
        ax_pca.tick_params(axis='both',which='major',labelsize=7); legend_elements_pca = [plt.Line2D([0],[0],marker='o',color='w',label='Normal',markerfacecolor=plt.cm.viridis(0.2),markersize=5), plt.Line2D([0],[0],marker='o',color='w',label='Outlier',markerfacecolor=plt.cm.viridis(0.8),markersize=5)]
        ax_pca.legend(handles=legend_elements_pca, loc='best', fontsize=7); plt.tight_layout()

        new_tex_id, _ = create_widget_func(
            parent_dpg_tag=TAG_OT_MVA_PCA_PARENT_GROUP, # create_ui에서 정의한 PCA용 부모 그룹 태그
            fig=fig_pca,
            image_widget_tag=TAG_OT_MVA_PCA_PLOT_IMAGE,
            image_title="PCA Projection",
            shared_callbacks=shared_image_callbacks,
            default_texture_tag=_shared_utils_mva.get('default_pca_texture_tag', TAG_OT_MVA_PCA_DEFAULT_TEXTURE),
            existing_dpg_texture_id=prev_pca_tex_id
        )
        _mva_active_pca_texture_id = str(new_tex_id) if new_tex_id else None
        _log_mva("PCA plot " + ("generated" if new_tex_id else "failed"))
    except Exception as e_pca:
        _log_mva(f"PCA plot error: {e_pca}")
        if 'fig_pca' in locals() and hasattr(fig_pca, 'canvas') and fig_pca.canvas.manager:
            plt.close(fig_pca)

def _generate_mva_shap_plot_for_instance(original_idx: Any):
    global _mva_active_shap_texture_id, _mva_scaler
    shared_image_callbacks = _get_shared_callbacks_for_image_util()
    if not shared_image_callbacks.get('cache_image_data_func') or not shared_image_callbacks.get('initiate_ollama_analysis'): _log_mva("SHAP: 콜백 누락"); return
    prev_shap_tex_id = _mva_active_shap_texture_id; _clear_mva_shap_plot()
    create_widget_func = _shared_utils_mva.get('util_funcs_common', {}).get('create_analyzable_image_widget')
    if not callable(create_widget_func): _log_mva("SHAP: 위젯 생성 함수 없음"); return
    shap_plot_parent_group = TAG_OT_MVA_SHAP_PARENT_GROUP
    if not dpg.does_item_exist(shap_plot_parent_group): _log_mva(f"SHAP 부모 그룹 '{shap_plot_parent_group}' 없음"); return
    if not ('shap' in globals() and shap is not None):
        if dpg.does_item_exist(TAG_OT_MVA_SHAP_PLOT_IMAGE): dpg.configure_item(TAG_OT_MVA_SHAP_PLOT_IMAGE, show=False)
        if dpg.does_item_exist("mva_shap_status_text"): dpg.delete_item("mva_shap_status_text")
        dpg.add_text("SHAP N/A (library missing)", parent=shap_plot_parent_group, color=[255,100,100], tag="mva_shap_status_text"); return
    if _df_with_mva_outliers is None or original_idx not in _df_with_mva_outliers.index: _log_mva(f"SHAP {original_idx}: Instance data 없음"); return
    if _mva_shap_explainer is None or not _mva_selected_columns_for_detection: _log_mva(f"SHAP {original_idx}: Explainer/columns 없음"); return
    if _mva_scaler is None: _log_mva(f"SHAP {original_idx}: Scaler 없음"); return
    try:
        instance_series_raw = _df_with_mva_outliers.loc[original_idx, _mva_selected_columns_for_detection]
        instance_df_for_shap_raw = pd.DataFrame([instance_series_raw.values], columns=_mva_selected_columns_for_detection)
        if instance_df_for_shap_raw.isnull().values.any():
            if _current_df_for_mva is not None:
                for col in instance_df_for_shap_raw.columns:
                    if instance_df_for_shap_raw[col].isnull().any(): instance_df_for_shap_raw[col].fillna(_current_df_for_mva[col].dropna().mean(), inplace=True)
            else: instance_df_for_shap_raw.fillna(0, inplace=True)
        instance_df_for_shap_scaled_values = _mva_scaler.transform(instance_df_for_shap_raw)
        instance_df_for_shap_scaled = pd.DataFrame(instance_df_for_shap_scaled_values, columns=_mva_selected_columns_for_detection)
        shap_values_instance_raw = _mva_shap_explainer.shap_values(instance_df_for_shap_scaled)
        shap_values_for_plot = shap_values_instance_raw[0,:] if len(shap_values_instance_raw.shape) == 2 else shap_values_instance_raw
        explainer_expected_value = _mva_shap_explainer.expected_value
        if hasattr(explainer_expected_value, "__len__") and not isinstance(explainer_expected_value, (str, bytes)): explainer_expected_value = explainer_expected_value[0]
        shap_explanation = shap.Explanation(values=shap_values_for_plot, base_values=explainer_expected_value, data=instance_df_for_shap_scaled.iloc[0].values, feature_names=_mva_selected_columns_for_detection)
        max_display_shap = 15; num_features_to_display = min(len(_mva_selected_columns_for_detection), max_display_shap)
        fig_height_shap = max(4.0, num_features_to_display * 0.35 + 1.0); fig_shap_waterfall = plt.figure(figsize=(7, fig_height_shap))
        shap.waterfall_plot(shap_explanation, max_display=num_features_to_display, show=False); plt.tight_layout()
        new_tex_id, _ = create_widget_func(parent_dpg_tag=shap_plot_parent_group, fig=fig_shap_waterfall, image_widget_tag=TAG_OT_MVA_SHAP_PLOT_IMAGE, image_title=f"SHAP Waterfall - Instance {original_idx}", shared_callbacks=shared_image_callbacks, default_texture_tag=_shared_utils_mva.get('default_shap_plot_texture_tag'), existing_dpg_texture_id=prev_shap_tex_id)
        _mva_active_shap_texture_id = new_tex_id
        if new_tex_id:
            if dpg.does_item_exist("mva_shap_status_text"): dpg.delete_item("mva_shap_status_text")
        else:
            if dpg.does_item_exist("mva_shap_status_text"): dpg.delete_item("mva_shap_status_text")
            dpg.add_text("Failed to generate SHAP plot.", parent=shap_plot_parent_group, color=[255,0,0], tag="mva_shap_status_text")
        _log_mva(f"SHAP waterfall plot {'generated' if new_tex_id else 'failed'}.")
    except Exception as e_shap_plot:
        _log_mva(f"Error generating SHAP waterfall plot for instance {original_idx}: {e_shap_plot}\n{traceback.format_exc()}")
        if dpg.does_item_exist("mva_shap_status_text"): dpg.delete_item("mva_shap_status_text")
        dpg.add_text(f"SHAP Plot Error: {str(e_shap_plot)[:100]}...", parent=shap_plot_parent_group, color=[255,0,0], tag="mva_shap_status_text")

def _find_top_gap_variables_for_boxplot() -> List[str]:
    global _mva_top_gap_vars_for_boxplot; _mva_top_gap_vars_for_boxplot = []
    if _df_with_mva_outliers is None or 'mva_is_outlier' not in _df_with_mva_outliers.columns: return []
    eligible_cols = _get_eligible_numeric_cols_for_mva(_df_with_mva_outliers);
    if not eligible_cols: return []
    gaps = []
    for col in eligible_cols:
        if col in ['mva_outlier_score', 'mva_is_outlier']: continue
        normal_data = _df_with_mva_outliers.loc[_df_with_mva_outliers['mva_is_outlier'] == False, col].dropna()
        outlier_data = _df_with_mva_outliers.loc[_df_with_mva_outliers['mva_is_outlier'] == True, col].dropna()
        if len(normal_data) < 2 or len(outlier_data) < 2: continue
        median_normal, median_outlier = normal_data.median(), outlier_data.median(); gap = abs(median_normal - median_outlier)
        if pd.notna(gap) and gap > 1e-6 : gaps.append((col, gap))
    gaps.sort(key=lambda x: x[1], reverse=True)
    _mva_top_gap_vars_for_boxplot = [var for var, score in gaps[:TOP_N_VARIABLES_FOR_BOXPLOT]]
    return _mva_top_gap_vars_for_boxplot

def _generate_mva_boxplots_for_comparison():
    global _mva_boxplot_image_tags
    shared_image_callbacks = _get_shared_callbacks_for_image_util()
    if not shared_image_callbacks.get('cache_image_data_func') or not shared_image_callbacks.get('initiate_ollama_analysis'):
        _log_mva("Boxplot: 콜백 누락")
        return
    
    _clear_mva_boxplots()
    top_vars = _find_top_gap_variables_for_boxplot()
    if not top_vars or _df_with_mva_outliers is None:
        if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_MVA_BOXPLOT_GROUP):
            dpg.add_text("Run detection or no significant gaps for boxplots.", parent=TAG_OT_MVA_BOXPLOT_GROUP)
        return

    num_plots_actual, cols_per_row = len(top_vars), 2
    parent_group_for_boxplots = TAG_OT_MVA_BOXPLOT_GROUP
    
    # 부모 그룹 존재 확인
    if not dpg.does_item_exist(parent_group_for_boxplots):
        _log_mva(f"Error: 부모 그룹 '{parent_group_for_boxplots}'가 존재하지 않음.", "ERROR")
        return

    create_widget_func = _shared_utils_mva.get('util_funcs_common', {}).get('create_analyzable_image_widget')
    if not callable(create_widget_func):
        _log_mva("Boxplot: 위젯 생성 함수 없음")
        return

    for i in range(0, num_plots_actual, cols_per_row):
        # 각 행별로 그룹을 생성하고 DPG가 완전히 처리되도록 보장
        row_group_tag = f"mva_boxplot_row_group_{i}_{dpg.generate_uuid()}"
        
        # 그룹 생성 후 존재 확인
        with dpg.group(horizontal=True, parent=parent_group_for_boxplots, tag=row_group_tag):
            pass  # 빈 그룹 먼저 생성
        
        # DPG가 그룹을 완전히 처리하도록 잠시 대기
        if not dpg.does_item_exist(row_group_tag):
            _log_mva(f"Error: 행 그룹 '{row_group_tag}' 생성 실패", "ERROR")
            continue
            
        for j in range(cols_per_row):
            if i + j < num_plots_actual:
                var_name = top_vars[i+j]
                try:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    normal_series = _df_with_mva_outliers.loc[_df_with_mva_outliers['mva_is_outlier'] == False, var_name].dropna()
                    outlier_series = _df_with_mva_outliers.loc[_df_with_mva_outliers['mva_is_outlier'] == True, var_name].dropna()
                    data_to_plot, labels = [], []
                    
                    if not normal_series.empty:
                        data_to_plot.append(normal_series)
                        labels.append("Normal")
                    if not outlier_series.empty:
                        data_to_plot.append(outlier_series)
                        labels.append("Outlier")
                    
                    if data_to_plot:
                        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, vert=True, 
                                       medianprops={'color':'#FF0000','linewidth':1.5})
                        colors = ['lightblue','lightcoral'][:len(data_to_plot)]
                        [patch.set_facecolor(color_val) for patch, color_val in zip(bp['boxes'], colors)]
                        ax.set_title(f"{var_name}", fontsize=9)
                        ax.tick_params(axis='both', which='major', labelsize=7)
                        ax.grid(True, linestyle='--', alpha=0.6)
                        plt.tight_layout()
                        
                        boxplot_widget_tag = f"mva_boxplot_image_{var_name.replace(' ', '_').replace('.', '_')}_{dpg.generate_uuid()}"
                        
                        # 다시 한 번 부모 그룹 존재 확인
                        if dpg.does_item_exist(row_group_tag):
                            new_tex_id, _ = create_widget_func(
                                parent_dpg_tag=row_group_tag,  # 확실히 존재하는 그룹 사용
                                fig=fig,
                                image_widget_tag=boxplot_widget_tag,
                                image_title=f"Boxplot - {var_name}",
                                shared_callbacks=shared_image_callbacks,
                                default_texture_tag=None,
                                existing_dpg_texture_id=None
                            )
                            if new_tex_id:
                                _mva_boxplot_image_tags.append(new_tex_id)
                                _log_mva(f"Boxplot 생성 성공: {var_name} -> {new_tex_id}")
                            else:
                                _log_mva(f"Error: Boxplot 텍스처 생성 실패: {var_name}", "ERROR")
                                dpg.add_text(f"Error boxplot: {var_name}", parent=row_group_tag)
                        else:
                            _log_mva(f"Error: 행 그룹 '{row_group_tag}'이 더 이상 존재하지 않음", "ERROR")
                            plt.close(fig)  # Figure 리소스 정리
                            dpg.add_text(f"Group error: {var_name}", parent=parent_group_for_boxplots)
                    else:
                        plt.close(fig)  # 데이터가 없을 때도 Figure 정리
                        if dpg.does_item_exist(row_group_tag):
                            dpg.add_text(f"No data boxplot: {var_name}", parent=row_group_tag)
                        else:
                            dpg.add_text(f"No data boxplot: {var_name}", parent=parent_group_for_boxplots)
                            
                except Exception as e:
                    _log_mva(f"Error boxplot {var_name}: {e}", "ERROR")
                    if 'fig' in locals():
                        plt.close(fig)  # 예외 발생 시에도 Figure 정리
                    if dpg.does_item_exist(row_group_tag):
                        dpg.add_text(f"Plot error: {var_name}", parent=row_group_tag)
                    else:
                        dpg.add_text(f"Plot error: {var_name}", parent=parent_group_for_boxplots)

def _safe_unbind_handler_registry(item_tag: str):
    """DPG 버전 호환성을 위한 안전한 unbind 함수"""
    try:
        if hasattr(dpg, 'unbind_item_handler_registry'):
            dpg.unbind_item_handler_registry(str(item_tag))
        elif hasattr(dpg, 'bind_item_handler_registry'):
            # 구버전에서는 None을 바인딩하여 기존 핸들러 제거
            dpg.bind_item_handler_registry(str(item_tag), None)
        _log_mva(f"Handler registry unbound from {item_tag}.")
    except Exception as e:
        _log_mva(f"Note: Error/No action unbinding handler for {item_tag}: {e}", "WARN")

def _clear_mva_umap_plot():
    if not dpg.is_dearpygui_running(): return
    umap_plot_image_tag = TAG_OT_MVA_UMAP_PLOT_IMAGE
    default_tex_tag = (_shared_utils_mva.get('default_umap_texture_tag', TAG_OT_MVA_UMAP_DEFAULT_TEXTURE)
                       if _shared_utils_mva else TAG_OT_MVA_UMAP_DEFAULT_TEXTURE)

    if dpg.does_item_exist(str(umap_plot_image_tag)):
        _log_mva(f"Clearing UMAP plot: {umap_plot_image_tag}. Info: {dpg.get_item_info(str(umap_plot_image_tag))}")
        texture_to_set, width_to_set, height_to_set, show_item = "", 100, 30, False
        if default_tex_tag and dpg.does_item_exist(str(default_tex_tag)):
            cfg = dpg.get_item_configuration(str(default_tex_tag))
            width_to_set, height_to_set = cfg.get('width',100), cfg.get('height',30)
            texture_to_set = str(default_tex_tag); show_item = True
        try:
            dpg.configure_item(str(umap_plot_image_tag), texture_tag=texture_to_set,
                               width=width_to_set, height=height_to_set, show=show_item)
        except Exception as e: _log_mva(f"Error configuring {umap_plot_image_tag}: {e}", "ERROR")
        
        # 수정된 부분: 안전한 unbind 사용
        _safe_unbind_handler_registry(str(umap_plot_image_tag))
        
        try:
            if dpg.get_item_user_data(str(umap_plot_image_tag)) is not None:
                dpg.set_item_user_data(str(umap_plot_image_tag), None)
        except Exception as e: _log_mva(f"Error clearing user_data for {umap_plot_image_tag}: {e}", "ERROR")
    else: _log_mva(f"UMAP plot item '{umap_plot_image_tag}' does not exist.", "WARN")

def _clear_mva_pca_plot():
    if not dpg.is_dearpygui_running(): return
    pca_plot_image_tag = TAG_OT_MVA_PCA_PLOT_IMAGE
    default_tex_tag = (_shared_utils_mva.get('default_pca_texture_tag', TAG_OT_MVA_PCA_DEFAULT_TEXTURE)
                       if _shared_utils_mva else TAG_OT_MVA_PCA_DEFAULT_TEXTURE)
    if dpg.does_item_exist(str(pca_plot_image_tag)):
        _log_mva(f"Clearing PCA plot: {pca_plot_image_tag}. Info: {dpg.get_item_info(str(pca_plot_image_tag))}")
        texture_to_set, width_to_set, height_to_set, show_item = "", 100, 30, False
        if default_tex_tag and dpg.does_item_exist(str(default_tex_tag)):
            cfg = dpg.get_item_configuration(str(default_tex_tag))
            width_to_set, height_to_set = cfg.get('width',100), cfg.get('height',30)
            texture_to_set = str(default_tex_tag); show_item = True
        try:
            dpg.configure_item(str(pca_plot_image_tag), texture_tag=texture_to_set,
                               width=width_to_set, height=height_to_set, show=show_item)
        except Exception as e: _log_mva(f"Error configuring {pca_plot_image_tag}: {e}", "ERROR")
        
        # 수정된 부분: 안전한 unbind 사용
        _safe_unbind_handler_registry(str(pca_plot_image_tag))
        
        try:
            if dpg.get_item_user_data(str(pca_plot_image_tag)) is not None:
                dpg.set_item_user_data(str(pca_plot_image_tag), None)
        except Exception as e: _log_mva(f"Error clearing user_data for {pca_plot_image_tag}: {e}", "ERROR")
    else: _log_mva(f"PCA plot item '{pca_plot_image_tag}' does not exist.", "WARN")

def _clear_mva_shap_plot():
    if not dpg.is_dearpygui_running(): return
    shap_plot_image_tag = TAG_OT_MVA_SHAP_PLOT_IMAGE
    default_tex_tag = _shared_utils_mva.get('default_shap_plot_texture_tag') if _shared_utils_mva else None
    if dpg.does_item_exist(str(shap_plot_image_tag)):
        _log_mva(f"Clearing SHAP plot: {shap_plot_image_tag}. Info: {dpg.get_item_info(str(shap_plot_image_tag))}")
        shap_status_text_parent = dpg.get_item_parent(str(shap_plot_image_tag))
        texture_to_set, width_to_set, height_to_set, show_item = "", 100, 30, False
        if default_tex_tag and dpg.does_item_exist(str(default_tex_tag)):
            cfg = dpg.get_item_configuration(str(default_tex_tag))
            width_to_set,height_to_set = cfg.get('width',100),cfg.get('height',30)
            texture_to_set=str(default_tex_tag); show_item=True
        try:
            dpg.configure_item(str(shap_plot_image_tag),texture_tag=texture_to_set,width=width_to_set,height=height_to_set,show=show_item)
        except Exception as e: _log_mva(f"Error configuring {shap_plot_image_tag}: {e}", "ERROR")
        
        # 수정된 부분: 안전한 unbind 사용
        _safe_unbind_handler_registry(str(shap_plot_image_tag))
        
        try:
            if dpg.get_item_user_data(str(shap_plot_image_tag)) is not None:
                dpg.set_item_user_data(str(shap_plot_image_tag), None)
        except Exception as e: _log_mva(f"Error clearing user_data for {shap_plot_image_tag}: {e}", "ERROR")
        
        if dpg.does_item_exist("mva_shap_status_text"): dpg.delete_item("mva_shap_status_text")
        if shap_status_text_parent and dpg.does_item_exist(str(shap_status_text_parent)):
            children_info = dpg.get_item_children(str(shap_status_text_parent), 1) if dpg.does_item_exist(str(shap_status_text_parent)) else {}
            before_target = str(shap_plot_image_tag) if isinstance(children_info, dict) and str(shap_plot_image_tag) in children_info else \
                           (str(shap_plot_image_tag) if isinstance(children_info, list) and str(shap_plot_image_tag) in children_info else "")
            dpg.add_text("Select an instance to see SHAP values.",parent=str(shap_status_text_parent),tag="mva_shap_status_text",before=before_target)
    else: _log_mva(f"SHAP plot item '{shap_plot_image_tag}' does not exist.", "WARN")

def _clear_mva_boxplots():
    global _mva_boxplot_image_tags
    if not dpg.is_dearpygui_running():
        _log_mva("DPG not running, cannot clear boxplots.")
        return
    
    # 기존 boxplot 텍스처들 삭제
    for tex_tag in _mva_boxplot_image_tags:
        if tex_tag and dpg.does_item_exist(str(tex_tag)):
            try:
                dpg.delete_item(str(tex_tag))
                _log_mva(f"Boxplot texture {tex_tag} deleted.", "INFO")
            except Exception as e:
                _log_mva(f"Error deleting boxplot texture {tex_tag}: {e}", "WARN")
    _mva_boxplot_image_tags.clear()
    
    # 부모 그룹 존재 확인 후 자식들만 삭제
    if dpg.does_item_exist(TAG_OT_MVA_BOXPLOT_GROUP):
        try:
            # 자식 위젯들만 삭제 (부모 그룹은 유지)
            dpg.delete_item(TAG_OT_MVA_BOXPLOT_GROUP, children_only=True)
            _log_mva("Boxplot group children cleared successfully.")
        except Exception as e:
            _log_mva(f"Error clearing boxplot group children: {e}", "ERROR")
    else:
        _log_mva(f"Warning: Boxplot parent group '{TAG_OT_MVA_BOXPLOT_GROUP}' does not exist.", "WARN")

def _clear_all_mva_visualizations():
    _clear_mva_umap_plot(); _clear_mva_pca_plot(); _clear_mva_shap_plot(); _clear_mva_boxplots()
    _mva_selected_outlier_instance_idx = None

def create_multivariate_ui(parent_tab_bar_tag: str, shared_utilities: dict):
    global _shared_utils_mva
    _shared_utils_mva = shared_utilities
    default_umap_tex = _shared_utils_mva.get('default_umap_texture_tag', TAG_OT_MVA_UMAP_DEFAULT_TEXTURE)
    default_pca_tex = _shared_utils_mva.get('default_pca_texture_tag', TAG_OT_MVA_PCA_DEFAULT_TEXTURE)
    default_shap_tex = _shared_utils_mva.get('default_shap_plot_texture_tag') # main_app에서 이 키로 전달 가정

    with dpg.tab(label="Multivariate Outlier Detection", tag=TAG_OT_MULTIVARIATE_TAB, parent=parent_tab_bar_tag):
        dpg.add_text("1. Configure & Run Multivariate Detection (Isolation Forest)", color=[255, 255, 0])
        with dpg.group(horizontal=True):
            dpg.add_text("Column Selection Mode:"); dpg.add_radio_button(["All Numeric", "Recommended", "Manual"], tag=TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO, default_value=_mva_column_selection_mode, horizontal=True, callback=_on_mva_col_selection_mode_change)
        with dpg.group(tag=TAG_OT_MVA_MANUAL_COLUMN_SELECTOR_GROUP, show=False):
            dpg.add_text("Select Column (Single Click):"); dpg.add_listbox([], tag=TAG_OT_MVA_COLUMN_SELECTOR_MULTI, num_items=6, callback=_on_mva_manual_cols_selected, width=-1)
        dpg.add_spacer(height=5)
        with dpg.group(horizontal=True):
            dpg.add_text("Contamination (0.0-0.5):"); dpg.add_input_float(tag=TAG_OT_MVA_CONTAMINATION_INPUT, width=120, default_value=_mva_contamination, min_value=0.0001, max_value=0.5, step=0.01, format="%.4f", callback=_on_mva_contamination_change)
        with dpg.group(horizontal=True):
            enable_detection_button = 'PyOD_IForest' in globals() and PyOD_IForest is not None
            dpg.add_button(label="Run Multivariate Detection", tag=TAG_OT_MVA_DETECT_BUTTON, width=-1, height=30, callback=_run_mva_outlier_detection_logic, enabled=enable_detection_button)
            if not enable_detection_button: dpg.add_text(" (PyOD N/A)", color=[255,100,100], parent=dpg.last_item())
            dpg.add_button(label="Set Recommended MVA Params", tag=TAG_OT_MVA_RECOMMEND_PARAMS_BUTTON, width=-1, height=30, callback=_set_mva_recommended_parameters)
        dpg.add_separator()
        with dpg.tab_bar(tag="mva_results_tab_bar"):
            with dpg.tab(label="Overview: UMAP & PCA", tag="mva_tab_overview"):
                with dpg.group(tag="mva_summary_info_group"):
                    dpg.add_text("Detection Summary:", color=[255,255,0]); dpg.add_text("Run detection to see summary.", tag="mva_summary_text_status")
                    for suffix in ["total_features", "used_features", "detected_outliers", "outlier_ratio"]: dpg.add_text("", tag=f"mva_summary_text_{suffix}")
                dpg.add_separator(); dpg.add_text("2. UMAP & PCA Projection", color=[255,255,0])
                with dpg.group(tag=TAG_OT_MVA_VISUALIZATION_GROUP, horizontal=True):
                    init_w, init_h = 400, 350
                    with dpg.group(tag=TAG_OT_MVA_UMAP_PARENT_GROUP, parent=TAG_OT_MVA_VISUALIZATION_GROUP):
                        dpg.add_image(texture_tag=default_umap_tex if default_umap_tex and dpg.does_item_exist(default_umap_tex) else "", tag=TAG_OT_MVA_UMAP_PLOT_IMAGE, width=init_w, height=init_h, show=bool(default_umap_tex and dpg.does_item_exist(default_umap_tex)))
                        if not ('umap' in globals() and umap): dpg.add_text("UMAP N/A",color=[255,100,100])
                        elif not (default_umap_tex and dpg.does_item_exist(default_umap_tex)): dpg.add_text("UMAP texture missing",color=[255,0,0])
                    with dpg.group(tag=TAG_OT_MVA_PCA_PARENT_GROUP, parent=TAG_OT_MVA_VISUALIZATION_GROUP):
                        dpg.add_image(texture_tag=default_pca_tex if default_pca_tex and dpg.does_item_exist(default_pca_tex) else "", tag=TAG_OT_MVA_PCA_PLOT_IMAGE, width=init_w, height=init_h, show=bool(default_pca_tex and dpg.does_item_exist(default_pca_tex)))
                        if not (default_pca_tex and dpg.does_item_exist(default_pca_tex)): dpg.add_text("PCA texture missing",color=[255,0,0])
            with dpg.tab(label="Detected Instances & Details", tag="mva_tab_details"):
                dpg.add_text("4. Detected Multivariate Outlier Instances", color=[255,255,0])
                with dpg.table(tag=TAG_OT_MVA_OUTLIER_INSTANCES_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=220, borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                    dpg.add_table_column(label="Original Index"); dpg.add_table_column(label="MVA Outlier Score")
                    with dpg.table_row(): dpg.add_text("Run MVA detection."); dpg.add_text("")
                with dpg.group(horizontal=True):
                    with dpg.group(width=-0.4, tag=TAG_OT_MVA_SHAP_PARENT_GROUP) as shap_image_parent_actual:
                        dpg.add_text("5. SHAP Values for Selected Instance", color=[255,255,0])
                        init_w_shap, init_h_shap = -1, 430; show_shap_lib = 'shap' in globals() and shap
                        dpg.add_image(texture_tag=default_shap_tex if default_shap_tex and dpg.does_item_exist(default_shap_tex) else "", tag=TAG_OT_MVA_SHAP_PLOT_IMAGE, width=init_w_shap, height=init_h_shap, parent=shap_image_parent_actual, show=bool(show_shap_lib and default_shap_tex and dpg.does_item_exist(default_shap_tex)))
                        if not show_shap_lib: dpg.add_text("SHAP N/A",color=[255,100,100],tag="mva_shap_status_text")
                        elif not (default_shap_tex and dpg.does_item_exist(default_shap_tex)): dpg.add_text("SHAP texture/instance missing",color=[255,0,0],tag="mva_shap_status_text")
                        elif dpg.does_item_exist(TAG_OT_MVA_SHAP_PLOT_IMAGE) and not (dpg.get_value(TAG_OT_MVA_SHAP_PLOT_IMAGE) if dpg.does_item_exist(TAG_OT_MVA_SHAP_PLOT_IMAGE) else True): # 숨겨져 있다면 메시지
                            if not dpg.does_item_exist("mva_shap_status_text"): dpg.add_text("Select an instance for SHAP.",tag="mva_shap_status_text")
                    with dpg.group(width=0):
                        dpg.add_text("6. Statistics for Selected Instance", color=[255,255,0])
                        with dpg.table(tag=TAG_OT_MVA_INSTANCE_STATS_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=450, borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                            for lbl in ["Feature","Value","Mean","Median","Z-Dist."]: dpg.add_table_column(label=lbl)
                            with dpg.table_row(): dpg.add_text("Select an instance."); [dpg.add_text("") for _ in range(4)]
            with dpg.tab(label="Variable Box Plots", tag="mva_tab_boxplots"):
                dpg.add_text("3. Variable Comparison: Outlier vs. Normal", color=[255,255,0])
                with dpg.child_window(tag=TAG_OT_MVA_BOXPLOT_GROUP, border=True): dpg.add_text("Run detection to see boxplots.")
    _update_mva_param_visibility()

def update_multivariate_ui(df_input: Optional[pd.DataFrame], shared_utilities: dict, is_new_data: bool):
    global _shared_utils_mva, _current_df_for_mva; _shared_utils_mva = shared_utilities
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OT_MULTIVARIATE_TAB): return
    _current_df_for_mva = df_input
    if _current_df_for_mva is None or is_new_data:
        _log_mva("New/no data for MVA. Resetting state.")
        reset_multivariate_state_internal(called_from_parent_reset=False)
        if _current_df_for_mva is not None: _update_selected_columns_for_mva_detection()
    elif _current_df_for_mva is not None:
        global _mva_eligible_numeric_cols
        _mva_eligible_numeric_cols = _get_eligible_numeric_cols_for_mva(_current_df_for_mva)
        if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTOR_MULTI):
            dpg.configure_item(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, items=_mva_eligible_numeric_cols)

def reset_multivariate_state_internal(called_from_parent_reset=True):
    global _df_with_mva_outliers, _mva_model, _mva_shap_explainer, _mva_scaler, _mva_eligible_numeric_cols, _mva_selected_columns_for_detection, _mva_column_selection_mode, _mva_contamination, _mva_outlier_instances_summary, _mva_selected_outlier_instance_idx, _mva_all_selectable_tags_in_instances_table, _mva_top_gap_vars_for_boxplot
    _df_with_mva_outliers, _mva_model, _mva_shap_explainer, _mva_scaler = None,None,None,None
    _mva_eligible_numeric_cols, _mva_selected_columns_for_detection = [],[]
    _mva_column_selection_mode, _mva_contamination = "All Numeric", DEFAULT_MVA_CONTAMINATION
    _mva_outlier_instances_summary, _mva_selected_outlier_instance_idx = [],None
    _mva_all_selectable_tags_in_instances_table.clear(); _mva_top_gap_vars_for_boxplot = []
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO): dpg.set_value(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO, _mva_column_selection_mode)
        if dpg.does_item_exist(TAG_OT_MVA_CONTAMINATION_INPUT): dpg.set_value(TAG_OT_MVA_CONTAMINATION_INPUT, _mva_contamination)
        if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTOR_MULTI): dpg.configure_item(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, items=[]); dpg.set_value(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, "")
        if dpg.does_item_exist("mva_summary_text_status"): dpg.set_value("mva_summary_text_status", "Run detection to see summary.")
        for suffix in ["total_features","used_features","detected_outliers","outlier_ratio"]:
            if dpg.does_item_exist(f"mva_summary_text_{suffix}"): dpg.set_value(f"mva_summary_text_{suffix}", "")
        _update_mva_param_visibility(); _populate_mva_outlier_instances_table(); _clear_all_mva_visualizations()
    if not called_from_parent_reset: _log_mva("Multivariate outlier state reset (internal).")

def reset_multivariate_state():
    reset_multivariate_state_internal(called_from_parent_reset=True); _log_mva("Multivariate outlier state reset by parent.")

def get_multivariate_settings() -> dict:
    return {"mva_column_selection_mode":_mva_column_selection_mode, "mva_selected_columns_for_detection":_mva_selected_columns_for_detection[:], "mva_contamination":_mva_contamination}

def apply_multivariate_settings(df_input:Optional[pd.DataFrame], settings:dict, shared_utilities:dict):
    global _shared_utils_mva, _current_df_for_mva, _mva_column_selection_mode, _mva_selected_columns_for_detection, _mva_contamination
    _shared_utils_mva, _current_df_for_mva = shared_utilities, df_input
    _mva_column_selection_mode = settings.get("mva_column_selection_mode", "All Numeric")
    _mva_selected_columns_for_detection = settings.get("mva_selected_columns_for_detection", [])[:]
    _mva_contamination = settings.get("mva_contamination", DEFAULT_MVA_CONTAMINATION)
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO): dpg.set_value(TAG_OT_MVA_COLUMN_SELECTION_MODE_RADIO, _mva_column_selection_mode)
        if dpg.does_item_exist(TAG_OT_MVA_CONTAMINATION_INPUT): dpg.set_value(TAG_OT_MVA_CONTAMINATION_INPUT, _mva_contamination)
        _update_mva_param_visibility()
        if _current_df_for_mva is not None: _update_selected_columns_for_mva_detection()
        else:
            if dpg.does_item_exist(TAG_OT_MVA_COLUMN_SELECTOR_MULTI): dpg.configure_item(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, items=[]); dpg.set_value(TAG_OT_MVA_COLUMN_SELECTOR_MULTI, "")
        _populate_mva_outlier_instances_table(); _clear_all_mva_visualizations()
    _log_mva("Multivariate outlier settings applied. Re-run detection if needed.")