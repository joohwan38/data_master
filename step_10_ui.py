# step_10_ui.py - UI 담당 (수정된 버전)

import dearpygui.dearpygui as dpg
import pandas as pd
from typing import Dict, List, Optional, Any
import step_10_analysis as analysis
import step_10_visualization as viz

# --- DPG Tags ---
TAG_S10_GROUP = "step10_advanced_analysis_group"
TAG_S10_UPPER_VIZ_WINDOW = "step10_upper_viz_window"
TAG_S10_LOWER_CONTROL_PANEL = "step10_lower_control_panel"
TAG_S10_VIZ_TAB_BAR = "step10_viz_tab_bar"
TAG_S10_DF_SELECTOR = "step10_df_selector"
TAG_S10_METHOD_SELECTOR = "step10_method_selector"
TAG_S10_DYNAMIC_OPTIONS_AREA = "step10_dynamic_options_area"
TAG_S10_VAR_AVAILABLE_LIST = "step10_var_available_list"
TAG_S10_VAR_SELECTED_LIST = "step10_var_selected_list"
TAG_S10_VAR_SEARCH_INPUT = "step10_var_search_input"
TAG_S10_RUN_BUTTON = "step10_run_analysis_button"
TAG_S10_ADD_TO_DF_CHECK = "step10_add_results_to_df_check"
TAG_S10_NEW_DF_NAME_INPUT = "step10_new_df_name_input"

# --- Module State ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_available_dfs: Dict[str, pd.DataFrame] = {}
_selected_df_name: str = ""
_selected_vars: List[str] = []

def initialize(main_callbacks: dict):
    """UI 모듈 초기화"""
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    
    # 분석 및 시각화 모듈 초기화
    analysis.initialize(main_callbacks)
    viz.initialize(main_callbacks)

def _update_df_list():
    """데이터프레임 목록 업데이트"""
    global _available_dfs, _selected_df_name
    if not _module_main_callbacks:
        return
    
    all_dfs_from_main = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    _available_dfs = {k: v for k, v in all_dfs_from_main.items() if k != '0. Original Data'}
    
    if dpg.does_item_exist(TAG_S10_DF_SELECTOR):
        df_names = list(_available_dfs.keys())
        current_selection = dpg.get_value(TAG_S10_DF_SELECTOR)
        
        dpg.configure_item(TAG_S10_DF_SELECTOR, items=df_names)
        
        if current_selection not in df_names:
            new_selection = df_names[0] if df_names else ""
            dpg.set_value(TAG_S10_DF_SELECTOR, new_selection)
            _on_df_selected(None, new_selection, None)

def _on_df_selected(sender, app_data, user_data):
    """데이터프레임 선택 시 호출"""
    global _selected_df_name, _selected_vars
    _selected_df_name = app_data
    _selected_vars.clear()
    _update_variable_lists()

def _on_method_selected(sender, app_data, user_data):
    """분석 방법론 선택 시 호출"""
    global _selected_vars
    _selected_vars.clear()
    _create_dynamic_options(app_data)
    _update_variable_lists()

def _update_variable_lists():
    """변수 목록 업데이트"""
    if not _selected_df_name or _selected_df_name not in _available_dfs:
        for list_tag in [TAG_S10_VAR_AVAILABLE_LIST, TAG_S10_VAR_SELECTED_LIST]:
            if dpg.does_item_exist(list_tag):
                dpg.delete_item(list_tag, children_only=True)
                dpg.add_table_column(parent=list_tag)
        return
    
    df = _available_dfs[_selected_df_name]
    s1_analysis_types = _module_main_callbacks.get('get_column_analysis_types', lambda: {})()

    # 전역 제외: Step 1에서 '제외'로 표시된 변수 필터링
    eligible_vars = [
        var for var in df.columns 
        if s1_analysis_types.get(var) != "분석에서 제외 (Exclude)"
    ]

    # 방법론별 필터링
    method = dpg.get_value(TAG_S10_METHOD_SELECTOR) if dpg.does_item_exist(TAG_S10_METHOD_SELECTOR) else ""
    
    if method in ["K-Means", "Hierarchical", "DBSCAN", "PCA", "Factor Analysis", 
                  "Pearson", "Spearman", "Kendall", "Linear"]:
        # 수치형 변수만
        numeric_vars = []
        for var in eligible_vars:
            s1_type = s1_analysis_types.get(var)
            if s1_type and "Numeric" in s1_type:
                numeric_vars.append(var)
            elif pd.api.types.is_numeric_dtype(df[var].dtype):
                numeric_vars.append(var)
        eligible_vars = numeric_vars
    
    elif method == "Logistic":
        # 수치형 + 이진 범주형
        valid_vars = []
        for var in eligible_vars:
            if pd.api.types.is_numeric_dtype(df[var].dtype):
                valid_vars.append(var)
            elif df[var].nunique() <= 10:  # 카테고리가 적은 범주형도 포함
                valid_vars.append(var)
        eligible_vars = valid_vars
    
    elif method == "ANOVA":
        # 모든 변수 포함 (범주형과 연속형 구분은 분석 시 처리)
        pass
    
    elif method == "Time Series":
        # 수치형 변수만
        numeric_vars = []
        for var in eligible_vars:
            if pd.api.types.is_numeric_dtype(df[var].dtype):
                numeric_vars.append(var)
        eligible_vars = numeric_vars

    # 검색어 필터링
    if dpg.does_item_exist(TAG_S10_VAR_SEARCH_INPUT):
        search_term = dpg.get_value(TAG_S10_VAR_SEARCH_INPUT).lower()
        if search_term:
            eligible_vars = [v for v in eligible_vars if search_term in v.lower()]
    
    # 이미 선택된 변수 제외
    available_vars = sorted([v for v in eligible_vars if v not in _selected_vars])
    
    # UI 목록 업데이트
    if dpg.does_item_exist(TAG_S10_VAR_AVAILABLE_LIST):
        dpg.delete_item(TAG_S10_VAR_AVAILABLE_LIST, children_only=True)
        dpg.add_table_column(parent=TAG_S10_VAR_AVAILABLE_LIST)
        for var in available_vars:
            with dpg.table_row(parent=TAG_S10_VAR_AVAILABLE_LIST):
                dpg.add_selectable(label=var, user_data=var, span_columns=True)

    if dpg.does_item_exist(TAG_S10_VAR_SELECTED_LIST):
        dpg.delete_item(TAG_S10_VAR_SELECTED_LIST, children_only=True)
        dpg.add_table_column(parent=TAG_S10_VAR_SELECTED_LIST)
        for var in sorted(_selected_vars):
            with dpg.table_row(parent=TAG_S10_VAR_SELECTED_LIST):
                dpg.add_selectable(label=var, user_data=var, span_columns=True)

def _move_variables(direction: str):
    """변수를 좌우로 이동"""
    global _selected_vars
    
    if direction == ">>":
        # 모든 가능한 변수 추가
        all_available = []
        if dpg.does_item_exist(TAG_S10_VAR_AVAILABLE_LIST):
            rows = dpg.get_item_children(TAG_S10_VAR_AVAILABLE_LIST, 1)
            for row in rows:
                selectable = dpg.get_item_children(row, 1)[0]
                all_available.append(dpg.get_item_user_data(selectable))
        _selected_vars = sorted(list(set(_selected_vars + all_available)))
        
    elif direction == "<<":
        # 모든 선택된 변수 제거
        _selected_vars.clear()
        
    elif direction == ">":
        # 선택된 변수들만 추가
        source_list_tag = TAG_S10_VAR_AVAILABLE_LIST
        vars_to_move = []
        if dpg.does_item_exist(source_list_tag):
            table_rows = dpg.get_item_children(source_list_tag, 1)
            for row in table_rows:
                selectable_item = dpg.get_item_children(row, 1)[0]
                if dpg.get_value(selectable_item):
                    var = dpg.get_item_user_data(selectable_item)
                    if var: vars_to_move.append(var)
        _selected_vars = sorted(list(set(_selected_vars + vars_to_move)))
        
    elif direction == "<":
        # 선택된 변수들만 제거
        source_list_tag = TAG_S10_VAR_SELECTED_LIST
        vars_to_remove = []
        if dpg.does_item_exist(source_list_tag):
            table_rows = dpg.get_item_children(source_list_tag, 1)
            for row in table_rows:
                selectable_item = dpg.get_item_children(row, 1)[0]
                if dpg.get_value(selectable_item):
                    var = dpg.get_item_user_data(selectable_item)
                    if var: vars_to_remove.append(var)
        _selected_vars = [v for v in _selected_vars if v not in vars_to_remove]

    _update_variable_lists()

def _create_dynamic_options(method: str):
    """선택된 방법론에 따라 동적 옵션 UI 생성"""
    if not dpg.does_item_exist(TAG_S10_DYNAMIC_OPTIONS_AREA):
        return
    
    dpg.delete_item(TAG_S10_DYNAMIC_OPTIONS_AREA, children_only=True)
    
    if method == "K-Means":
        dpg.add_text("K-Means Options", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Number of Clusters:")
            dpg.add_input_int(default_value=3, min_value=2, max_value=20, 
                            tag="kmeans_n_clusters", width=100)
        
        dpg.add_checkbox(label="Standardize variables", default_value=True, 
                        tag="kmeans_standardize", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_checkbox(label="Generate Elbow Plot", default_value=True,
                        tag="kmeans_elbow", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
    elif method == "Hierarchical":
        dpg.add_text("Hierarchical Clustering Options", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Number of Clusters:")
            dpg.add_input_int(default_value=3, min_value=2, max_value=20,
                            tag="hier_n_clusters", width=100)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Linkage Method:")
            dpg.add_combo(["ward", "complete", "average", "single"], 
                         default_value="ward", tag="hier_linkage", width=150)
        
        dpg.add_checkbox(label="Standardize variables", default_value=True,
                        tag="hier_standardize", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_checkbox(label="Generate Dendrogram", default_value=True,
                        tag="hier_dendrogram", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
    elif method == "DBSCAN":
        dpg.add_text("DBSCAN Options", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Epsilon:")
            dpg.add_input_float(default_value=0.5, min_value=0.01, max_value=10.0,
                              tag="dbscan_eps", width=100, format="%.2f")
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Min Samples:")
            dpg.add_input_int(default_value=5, min_value=1, max_value=50,
                            tag="dbscan_min_samples", width=100)
        
        dpg.add_checkbox(label="Standardize variables", default_value=True,
                        tag="dbscan_standardize", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
    elif method == "PCA":
        dpg.add_text("PCA Options", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Number of Components:")
            dpg.add_input_int(default_value=2, min_value=1, max_value=10,
                            tag="pca_n_components", width=100)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Variance Threshold (%):")
            dpg.add_input_float(default_value=85.0, min_value=50.0, max_value=99.0,
                              tag="pca_variance_threshold", width=100, format="%.1f")
        
        dpg.add_checkbox(label="Standardize variables", default_value=True,
                        tag="pca_standardize", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_checkbox(label="Generate Scree Plot", default_value=True,
                        tag="pca_scree", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_checkbox(label="Generate Biplot", default_value=True,
                        tag="pca_biplot", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
    elif method == "Factor Analysis":
        dpg.add_text("Factor Analysis Options", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Number of Factors:")
            dpg.add_input_int(default_value=2, min_value=1, max_value=10,
                            tag="fa_n_factors", width=100)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Max Iterations:")
            dpg.add_input_int(default_value=1000, min_value=100, max_value=5000,
                            tag="fa_max_iter", width=100)
        
        dpg.add_checkbox(label="Standardize variables", default_value=True,
                        tag="fa_standardize", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_checkbox(label="Generate Scree Plot", default_value=True,
                        tag="fa_scree", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
    elif method in ["Pearson", "Spearman", "Kendall"]:
        dpg.add_text(f"{method} Correlation Options", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Significance Level:")
            dpg.add_combo(["0.001", "0.01", "0.05", "0.10"], default_value="0.05",
                         tag="corr_alpha", width=100)
        
    elif method in ["Linear", "Logistic"]:
        dpg.add_text(f"{method} Regression Options", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        # 타겟 변수 선택
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Target Variable:")
            target_var = _module_main_callbacks.get('get_selected_target_variable', lambda: None)() if _module_main_callbacks else None
            if target_var:
                dpg.add_text(f"{target_var} (from main panel)", color=(200, 200, 200))
            else:
                dpg.add_text("Not selected", color=(255, 100, 100))
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Test Size (%):")
            dpg.add_input_float(default_value=20.0, min_value=10.0, max_value=50.0,
                              tag="reg_test_size", width=100, format="%.1f")
        
        dpg.add_checkbox(label="Standardize features", default_value=False,
                        tag="reg_standardize", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        if method == "Linear":
            dpg.add_checkbox(label="Include diagnostic plots", default_value=True,
                            tag="reg_diagnostics", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
    elif method == "ANOVA":
        dpg.add_text("ANOVA Options", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        # 타겟 변수 선택
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Target Variable (Continuous):")
            target_var = _module_main_callbacks.get('get_selected_target_variable', lambda: None)() if _module_main_callbacks else None
            if target_var:
                dpg.add_text(f"{target_var} (from main panel)", color=(200, 200, 200))
            else:
                dpg.add_text("Not selected", color=(255, 100, 100))
        
        dpg.add_checkbox(label="Include interaction effects", default_value=False,
                        tag="anova_interactions", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_checkbox(label="Perform post-hoc tests", default_value=True,
                        tag="anova_posthoc", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
    elif method == "Time Series":
        dpg.add_text("Time Series Analysis Options", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        dpg.add_text("Select exactly one variable for time series analysis", 
                    color=(200, 200, 200), parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        dpg.add_checkbox(label="Seasonal decomposition", default_value=True,
                        tag="ts_seasonal", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Period (for seasonal):")
            dpg.add_input_int(default_value=12, min_value=2, max_value=365,
                            tag="ts_period", width=100)
    
    # 공통 출력 옵션
    dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    dpg.add_text("Output Options:", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    
    if method in ["K-Means", "Hierarchical", "DBSCAN"]:
        dpg.add_checkbox(label="Add cluster labels to DataFrame", default_value=True,
                        tag=TAG_S10_ADD_TO_DF_CHECK, parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    elif method in ["PCA", "Factor Analysis"]:
        dpg.add_checkbox(label="Add component/factor scores to DataFrame", default_value=True,
                        tag=TAG_S10_ADD_TO_DF_CHECK, parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    elif method in ["Linear", "Logistic"]:
        dpg.add_checkbox(label="Add predictions to DataFrame", default_value=True,
                        tag=TAG_S10_ADD_TO_DF_CHECK, parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    else:
        dpg.add_checkbox(label="Save results to new DataFrame", default_value=False,
                        tag=TAG_S10_ADD_TO_DF_CHECK, parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    
    with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
        dpg.add_text("New DataFrame name:")
        dpg.add_input_text(default_value="", tag=TAG_S10_NEW_DF_NAME_INPUT,
                         width=200, hint="Leave empty to update current")

def _get_method_options(method: str) -> Dict[str, Any]:
    """현재 UI에서 설정된 옵션들을 수집"""
    options = {}
    
    if method == "K-Means":
        options = {
            'n_clusters': dpg.get_value("kmeans_n_clusters") if dpg.does_item_exist("kmeans_n_clusters") else 3,
            'standardize': dpg.get_value("kmeans_standardize") if dpg.does_item_exist("kmeans_standardize") else True,
            'elbow': dpg.get_value("kmeans_elbow") if dpg.does_item_exist("kmeans_elbow") else True
        }
    elif method == "Hierarchical":
        options = {
            'n_clusters': dpg.get_value("hier_n_clusters") if dpg.does_item_exist("hier_n_clusters") else 3,
            'linkage': dpg.get_value("hier_linkage") if dpg.does_item_exist("hier_linkage") else "ward",
            'standardize': dpg.get_value("hier_standardize") if dpg.does_item_exist("hier_standardize") else True,
            'dendrogram': dpg.get_value("hier_dendrogram") if dpg.does_item_exist("hier_dendrogram") else True
        }
    elif method == "DBSCAN":
        options = {
            'eps': dpg.get_value("dbscan_eps") if dpg.does_item_exist("dbscan_eps") else 0.5,
            'min_samples': dpg.get_value("dbscan_min_samples") if dpg.does_item_exist("dbscan_min_samples") else 5,
            'standardize': dpg.get_value("dbscan_standardize") if dpg.does_item_exist("dbscan_standardize") else True
        }
    elif method == "PCA":
        options = {
            'n_components': dpg.get_value("pca_n_components") if dpg.does_item_exist("pca_n_components") else 2,
            'variance_threshold': dpg.get_value("pca_variance_threshold") if dpg.does_item_exist("pca_variance_threshold") else 85.0,
            'standardize': dpg.get_value("pca_standardize") if dpg.does_item_exist("pca_standardize") else True,
            'scree': dpg.get_value("pca_scree") if dpg.does_item_exist("pca_scree") else True,
            'biplot': dpg.get_value("pca_biplot") if dpg.does_item_exist("pca_biplot") else True
        }
    elif method == "Factor Analysis":
        options = {
            'n_factors': dpg.get_value("fa_n_factors") if dpg.does_item_exist("fa_n_factors") else 2,
            'max_iter': dpg.get_value("fa_max_iter") if dpg.does_item_exist("fa_max_iter") else 1000,
            'standardize': dpg.get_value("fa_standardize") if dpg.does_item_exist("fa_standardize") else True,
            'scree': dpg.get_value("fa_scree") if dpg.does_item_exist("fa_scree") else True
        }
    elif method in ["Pearson", "Spearman", "Kendall"]:
        alpha_str = dpg.get_value("corr_alpha") if dpg.does_item_exist("corr_alpha") else "0.05"
        options = {
            'alpha': float(alpha_str)
        }
    elif method in ["Linear", "Logistic"]:
        target_var = _module_main_callbacks.get('get_selected_target_variable', lambda: None)() if _module_main_callbacks else None
        options = {
            'target_variable': target_var,
            'test_size': dpg.get_value("reg_test_size") if dpg.does_item_exist("reg_test_size") else 20.0,
            'standardize': dpg.get_value("reg_standardize") if dpg.does_item_exist("reg_standardize") else False
        }
        if method == "Linear":
            options['diagnostics'] = dpg.get_value("reg_diagnostics") if dpg.does_item_exist("reg_diagnostics") else True
    elif method == "ANOVA":
        target_var = _module_main_callbacks.get('get_selected_target_variable', lambda: None)() if _module_main_callbacks else None
        options = {
            'target_variable': target_var,
            'include_interactions': dpg.get_value("anova_interactions") if dpg.does_item_exist("anova_interactions") else False,
            'posthoc': dpg.get_value("anova_posthoc") if dpg.does_item_exist("anova_posthoc") else True
        }
    elif method == "Time Series":
        options = {
            'seasonal_decompose': dpg.get_value("ts_seasonal") if dpg.does_item_exist("ts_seasonal") else True,
            'period': dpg.get_value("ts_period") if dpg.does_item_exist("ts_period") else 12
        }
    
    return options

def _run_analysis():
    """분석 실행"""
    method = dpg.get_value(TAG_S10_METHOD_SELECTOR)
    
    # 방법별 변수 개수 확인
    min_vars = 1
    max_vars = None
    
    if method in ["K-Means", "Hierarchical", "DBSCAN", "PCA", "Pearson", "Spearman", "Kendall"]:
        min_vars = 2
    elif method == "Factor Analysis":
        min_vars = 3
    elif method == "Time Series":
        min_vars = 1
        max_vars = 1
    elif method in ["Linear", "Logistic", "ANOVA"]:
        min_vars = 1
    
    if not _selected_df_name or not _selected_vars:
        _util_funcs['_show_simple_modal_message']("Error", 
            f"Please select a DataFrame and at least {min_vars} variable(s) for analysis.")
        return
    
    if max_vars and len(_selected_vars) > max_vars:
        _util_funcs['_show_simple_modal_message']("Error", 
            f"Please select exactly {max_vars} variable(s) for {method} analysis.")
        return
    
    # 회귀분석/ANOVA의 경우 타겟 변수 확인
    if method in ["Linear", "Logistic", "ANOVA"]:
        target_var = _module_main_callbacks.get('get_selected_target_variable', lambda: None)() if _module_main_callbacks else None
        if not target_var:
            _util_funcs['_show_simple_modal_message']("Error", 
                f"Please select a target variable in the main panel for {method} analysis.")
            return
    
    df = _available_dfs[_selected_df_name]
    
    # 분석 파라미터 수집
    analysis_params = {
        'df_name': _selected_df_name,
        'df': df,
        'method': method,
        'variables': _selected_vars.copy(),
        'options': _get_method_options(method),
        'add_to_df': dpg.get_value(TAG_S10_ADD_TO_DF_CHECK) if dpg.does_item_exist(TAG_S10_ADD_TO_DF_CHECK) else False,
        'new_df_name': dpg.get_value(TAG_S10_NEW_DF_NAME_INPUT) if dpg.does_item_exist(TAG_S10_NEW_DF_NAME_INPUT) else ""
    }
    
    # 분석 실행
    success, results = analysis.run_analysis(analysis_params)
    
    if success and results:
        viz.create_results_tab(TAG_S10_VIZ_TAB_BAR, results)
        
        if analysis_params['add_to_df']:
            _handle_dataframe_addition(results, analysis_params)

def _handle_dataframe_addition(results: Dict[str, Any], params: Dict[str, Any]):
    """분석 결과를 DataFrame에 추가"""
    df_name = params['df_name']
    if df_name not in _available_dfs:
        return

    df = _available_dfs[df_name]
    
    # 결과 타입에 따른 처리
    if 'labels' in results:  # 군집분석
        analysis.add_clustering_results_to_dataframe(df, results, params)
    elif 'factor_scores' in results:  # 요인분석
        analysis.add_factor_results_to_dataframe(df, results, params)
    elif 'component_scores' in results:  # PCA
        analysis.add_pca_results_to_dataframe(df, results, params)
    elif 'correlation_matrix' in results:  # 상관분석
        analysis.add_correlation_results_to_dataframe(df, results, params)
    elif 'predictions_test' in results:  # 회귀분석
        analysis.add_regression_results_to_dataframe(df, results, params)
    
    _update_df_list()

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """Step 10 UI 생성"""
    initialize(main_callbacks)
    
    main_callbacks['register_step_group_tag'](step_name, TAG_S10_GROUP)

    with dpg.group(tag=TAG_S10_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        
        total_height = 1000 
        upper_height = int(total_height * 0.65)
        lower_height = int(total_height * 0.35)
        
        # 상부 시각화 영역
        with dpg.child_window(height=upper_height, border=True, tag=TAG_S10_UPPER_VIZ_WINDOW):
            dpg.add_text("Analysis Results", color=(255, 255, 0))
            dpg.add_separator()
            
            with dpg.tab_bar(tag=TAG_S10_VIZ_TAB_BAR):
                with dpg.tab(label="Instructions"):
                    dpg.add_text("1. Select a DataFrame")
                    dpg.add_text("2. Choose an analysis method")
                    dpg.add_text("3. Select variables")
                    dpg.add_text("4. Configure options and run")
                    dpg.add_separator()
                    dpg.add_text("Available Methods:", color=(255, 255, 0))
                    dpg.add_text("• Clustering: K-Means, Hierarchical, DBSCAN")
                    dpg.add_text("• Dimension Reduction: PCA, Factor Analysis")
                    dpg.add_text("• Correlation: Pearson, Spearman, Kendall")
                    dpg.add_text("• Regression: Linear, Logistic (with statsmodels)")
                    dpg.add_text("• ANOVA: One-way, Multi-way, ANCOVA")
                    dpg.add_text("• Time Series: Decomposition, Stationarity tests")
        
        # 하부 컨트롤 패널
        with dpg.child_window(height=lower_height, border=True, tag=TAG_S10_LOWER_CONTROL_PANEL):
            with dpg.group(horizontal=True):
                # 좌측 선택 영역
                left_width = 250
                with dpg.child_window(width=left_width, border=True):
                    dpg.add_text("Data Source", color=(255, 255, 0))
                    dpg.add_combo(label="DataFrame", tag=TAG_S10_DF_SELECTOR,
                                callback=_on_df_selected, width=-1)
                    
                    dpg.add_separator()
                    dpg.add_text("Analysis Method", color=(255, 255, 0))
                    
                    # 탭바로 카테고리별 분류
                    with dpg.tab_bar(tag="method_selection_tabs"):
                        
                        # 군집분석 탭
                        with dpg.tab(label="Clustering"):
                            dpg.add_radio_button(
                                items=["K-Means", "Hierarchical", "DBSCAN"],
                                default_value="K-Means",
                                callback=lambda s, a, u: [
                                    dpg.set_value(TAG_S10_METHOD_SELECTOR, a),
                                    _on_method_selected(s, a, u)
                                ]
                            )
                        
                        # 차원축소 탭
                        with dpg.tab(label="Dimension"):
                            dpg.add_radio_button(
                                items=["PCA", "Factor Analysis"],
                                default_value="PCA",
                                callback=lambda s, a, u: [
                                    dpg.set_value(TAG_S10_METHOD_SELECTOR, a),
                                    _on_method_selected(s, a, u)
                                ]
                            )
                        
                        # 상관분석 탭
                        with dpg.tab(label="Correlation"):
                            dpg.add_radio_button(
                                items=["Pearson", "Spearman", "Kendall"],
                                default_value="Pearson",
                                callback=lambda s, a, u: [
                                    dpg.set_value(TAG_S10_METHOD_SELECTOR, a),
                                    _on_method_selected(s, a, u)
                                ]
                            )
                        
                        # 회귀분석 탭
                        with dpg.tab(label="Regression"):
                            dpg.add_radio_button(
                                items=["Linear", "Logistic"],
                                default_value="Linear",
                                callback=lambda s, a, u: [
                                    dpg.set_value(TAG_S10_METHOD_SELECTOR, a),
                                    _on_method_selected(s, a, u)
                                ]
                            )
                        
                        # 통계 분석 탭 (새로 추가)
                        with dpg.tab(label="Statistical"):
                            dpg.add_radio_button(
                                items=["ANOVA", "Time Series"],
                                default_value="ANOVA",
                                callback=lambda s, a, u: [
                                    dpg.set_value(TAG_S10_METHOD_SELECTOR, a),
                                    _on_method_selected(s, a, u)
                                ]
                            )
                    
                    # Hidden combo to store selected method
                    dpg.add_combo(tag=TAG_S10_METHOD_SELECTOR, show=False, 
                                default_value="K-Means", callback=_on_method_selected)

                # 우측 동적 영역
                with dpg.child_window(border=True):
                    with dpg.group(horizontal=True):
                        # 변수 선택 영역
                        with dpg.group(width=200):
                            dpg.add_text("Variable Selection", color=(255, 255, 0))
                            dpg.add_input_text(label="Search", 
                                            tag=TAG_S10_VAR_SEARCH_INPUT,
                                            callback=lambda: _update_variable_lists(),
                                            width=-1)
                            
                            dpg.add_text("Available Variables:")
                            with dpg.table(header_row=False, tag=TAG_S10_VAR_AVAILABLE_LIST, 
                                     height=150, borders_outerH=True, borders_outerV=True, scrollY=True, policy=dpg.mvTable_SizingStretchProp):
                                dpg.add_table_column()
                        
                        # 이동 버튼
                        with dpg.group():
                            dpg.add_spacer(height=50)
                            dpg.add_button(label=">", width=30, 
                                        callback=lambda: _move_variables(">"))
                            dpg.add_button(label=">>", width=30, 
                                        callback=lambda: _move_variables(">>"))
                            dpg.add_button(label="<", width=30, 
                                        callback=lambda: _move_variables("<"))
                            dpg.add_button(label="<<", width=30, 
                                        callback=lambda: _move_variables("<<"))
                        
                        # 선택된 변수
                        with dpg.group(width=200):
                            dpg.add_text("Selected Variables:")
                            with dpg.table(header_row=False, tag=TAG_S10_VAR_SELECTED_LIST,
                                         height=190, borders_outerH=True, borders_outerV=True, scrollY=True, policy=dpg.mvTable_SizingStretchProp):
                                dpg.add_table_column()
                        
                        dpg.add_spacer(width=20)
                        
                        # 동적 옵션 영역
                        with dpg.group(tag=TAG_S10_DYNAMIC_OPTIONS_AREA):
                            pass
                    
                    # 실행 버튼
                    dpg.add_separator()
                    dpg.add_button(label="Run Analysis", tag=TAG_S10_RUN_BUTTON,
                                callback=_run_analysis,
                                width=-1, height=30)

    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui():
    """UI 업데이트"""
    _update_df_list()
    
    if dpg.does_item_exist(TAG_S10_METHOD_SELECTOR):
        method = dpg.get_value(TAG_S10_METHOD_SELECTOR)
        if method:
            _create_dynamic_options(method)

def reset_state():
    """상태 초기화"""
    global _selected_df_name, _selected_vars

    _selected_df_name = ""
    _selected_vars.clear()

    analysis.reset_state()
    viz.reset_state()

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_S10_VIZ_TAB_BAR):
            tabs = dpg.get_item_children(TAG_S10_VIZ_TAB_BAR, 1)
            for tab in tabs[1:]:  # Instructions 탭은 유지
                if dpg.does_item_exist(tab):
                    dpg.delete_item(tab)
        
        update_ui()