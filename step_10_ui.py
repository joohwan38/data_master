# step_10_ui.py - UI 담당

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

# --- 분석 방법론 목록 ---
ANALYSIS_METHODS = {
    "군집분석 (Clustering)": ["K-Means", "Hierarchical", "DBSCAN"],
    "차원축소 (Dimension Reduction)": ["PCA", "Factor Analysis", "t-SNE"],
    "회귀분석 (Regression)": ["Linear", "Logistic", "Polynomial"],
    "상관분석 (Correlation)": ["Pearson", "Spearman", "Kendall"],
    "시계열분석 (Time Series)": ["Trend", "Seasonal Decomposition"],
    "판별분석 (Discriminant)": ["LDA", "QDA"]
}

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
    """데이터프레임 목록 업데이트 (수정: original_df 제외)"""
    global _available_dfs, _selected_df_name
    if not _module_main_callbacks:
        return
    
    all_dfs_from_main = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    
    # --- [사용자 요청 수정] '0. Original Data' 제외 ---
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
    """분석 방법론 선택 시 호출 (수정: 변수 목록 초기화 및 업데이트)"""
    global _selected_vars
    
    # --- [사용자 요청 수정] 메서드 변경 시 선택된 변수 초기화 및 목록 새로고침 ---
    _selected_vars.clear()
    _create_dynamic_options(app_data)
    _update_variable_lists()

def _update_variable_lists():
    """변수 목록 업데이트 (수정: 전역 제외 및 방법론별 필터링)"""
    if not _selected_df_name or _selected_df_name not in _available_dfs:
        for list_tag in [TAG_S10_VAR_AVAILABLE_LIST, TAG_S10_VAR_SELECTED_LIST]:
            if dpg.does_item_exist(list_tag):
                dpg.delete_item(list_tag, children_only=True)
                dpg.add_table_column(parent=list_tag)
        return
    
    df = _available_dfs[_selected_df_name]
    
    # 1. 필터링을 위해 Step 1 분석 타입 가져오기
    s1_analysis_types = _module_main_callbacks.get('get_column_analysis_types', lambda: {})()

    # 2. 전역 제외: Step 1에서 '제외'로 표시된 변수 필터링
    eligible_vars = [
        var for var in df.columns 
        if s1_analysis_types.get(var) != "분석에서 제외 (Exclude)"
    ]

    # 3. 방법론별 필터링
    method = dpg.get_value(TAG_S10_METHOD_SELECTOR) if dpg.does_item_exist(TAG_S10_METHOD_SELECTOR) else ""
    
    if method in ["K-Means", "Hierarchical", "DBSCAN", "PCA", "Factor Analysis"]:
        # 군집분석 및 차원축소의 경우, 수치형 변수만 표시
        numeric_vars = []
        for var in eligible_vars:
            s1_type = s1_analysis_types.get(var)
            if s1_type:
                if "Numeric" in s1_type:
                    numeric_vars.append(var)
            elif pd.api.types.is_numeric_dtype(df[var].dtype):
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
    """변수를 좌우로 이동 (테이블 구조에 맞게 수정)"""
    global _selected_vars
    
    source_list_tag = None
    
    if direction in [">", ">>"]:
        source_list_tag = TAG_S10_VAR_AVAILABLE_LIST
        if direction == ">>":
            all_available_now = []
            if dpg.does_item_exist(source_list_tag):
                rows = dpg.get_item_children(source_list_tag, 1)
                for row in rows:
                    selectable = dpg.get_item_children(row, 1)[0]
                    all_available_now.append(dpg.get_item_user_data(selectable))

            _selected_vars = sorted(list(set(_selected_vars + all_available_now)))
            _update_variable_lists()
            return
            
    elif direction in ["<", "<<"]:
        source_list_tag = TAG_S10_VAR_SELECTED_LIST
        if direction == "<<":
            _selected_vars.clear()
            _update_variable_lists()
            return

    if not source_list_tag or not dpg.does_item_exist(source_list_tag):
        return

    vars_to_move = []
    table_rows = dpg.get_item_children(source_list_tag, 1)

    for row in table_rows:
        selectable_item = dpg.get_item_children(row, 1)[0]
        if dpg.get_value(selectable_item):
            var = dpg.get_item_user_data(selectable_item)
            if var: vars_to_move.append(var)

    if direction == ">":
        _selected_vars = sorted(list(set(_selected_vars + vars_to_move)))
    elif direction == "<":
        _selected_vars = [v for v in _selected_vars if v not in vars_to_move]

    _update_variable_lists()

def _create_dynamic_options(method: str):
    """선택된 방법론에 따라 동적 옵션 UI 생성"""
    if not dpg.does_item_exist(TAG_S10_DYNAMIC_OPTIONS_AREA):
        return
    
    dpg.delete_item(TAG_S10_DYNAMIC_OPTIONS_AREA, children_only=True)
    
    if method == "K-Means":
        dpg.add_text("K-Means Clustering Options", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Number of Clusters (k):")
            dpg.add_input_int(default_value=3, min_value=2, max_value=20, 
                            tag="kmeans_n_clusters", width=100)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Initialization Method:")
            dpg.add_combo(["k-means++", "random"], default_value="k-means++",
                         tag="kmeans_init", width=150)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Max Iterations:")
            dpg.add_input_int(default_value=300, min_value=10, max_value=1000,
                            tag="kmeans_max_iter", width=100)
        
        dpg.add_checkbox(label="Standardize variables before clustering",
                        default_value=True, tag="kmeans_standardize",
                        parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        dpg.add_checkbox(label="Generate Elbow Plot (for optimal k)",
                        default_value=True, tag="kmeans_elbow",
                        parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    
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
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Distance Metric:")
            dpg.add_combo(["euclidean", "manhattan", "cosine"],
                         default_value="euclidean", tag="hier_metric", width=150)
        
        dpg.add_checkbox(label="Standardize variables before clustering",
                        default_value=True, tag="hier_standardize",
                        parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        dpg.add_checkbox(label="Generate Dendrogram",
                        default_value=True, tag="hier_dendrogram",
                        parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    
    elif method == "DBSCAN":
        dpg.add_text("DBSCAN Clustering Options", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Epsilon (eps):")
            dpg.add_input_float(default_value=0.5, min_value=0.01, max_value=10.0,
                              tag="dbscan_eps", width=100, format="%.2f")
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Min Samples:")
            dpg.add_input_int(default_value=5, min_value=1, max_value=50,
                            tag="dbscan_min_samples", width=100)
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Distance Metric:")
            dpg.add_combo(["euclidean", "manhattan", "cosine"],
                         default_value="euclidean", tag="dbscan_metric", width=150)
        
        dpg.add_checkbox(label="Standardize variables before clustering",
                        default_value=True, tag="dbscan_standardize",
                        parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    
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
        
        with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
            dpg.add_text("Tolerance:")
            dpg.add_input_float(default_value=0.01, min_value=0.001, max_value=0.1,
                              tag="fa_tol", width=100, format="%.3f")
        
        dpg.add_checkbox(label="Standardize variables before analysis",
                        default_value=True, tag="fa_standardize",
                        parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
        
        dpg.add_checkbox(label="Generate Scree Plot",
                        default_value=True, tag="fa_scree",
                        parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    
    # 공통 옵션들
    dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    dpg.add_text("Output Options:", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    
    if method in ["K-Means", "Hierarchical", "DBSCAN"]:
        dpg.add_checkbox(label="Add cluster labels to original DataFrame",
                        default_value=True, tag=TAG_S10_ADD_TO_DF_CHECK,
                        parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    elif method == "Factor Analysis":
        dpg.add_checkbox(label="Add factor scores to original DataFrame",
                        default_value=True, tag=TAG_S10_ADD_TO_DF_CHECK,
                        parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    
    with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
        dpg.add_text("New DataFrame name (if creating):")
        dpg.add_input_text(default_value="", tag=TAG_S10_NEW_DF_NAME_INPUT,
                         width=200, hint="Leave empty to auto-generate")

def _run_analysis():
    """분석 실행 - 분석 모듈로 위임"""
    method = dpg.get_value(TAG_S10_METHOD_SELECTOR)
    
    if not _selected_df_name or not _selected_vars:
        _util_funcs['_show_simple_modal_message']("Error", 
            "Please select a DataFrame and at least 2 variables for analysis.")
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
    
    # 분석 실행 및 결과 처리
    success, results = analysis.run_analysis(analysis_params)
    
    if success and results:
        # 결과를 시각화 모듈로 전달
        viz.create_results_tab(TAG_S10_VIZ_TAB_BAR, results)
        
        # DataFrame에 결과 추가 처리
        if analysis_params['add_to_df']:
            _handle_dataframe_addition(results, analysis_params)

def _get_method_options(method: str) -> Dict[str, Any]:
    """현재 UI에서 설정된 옵션들을 딕셔너리로 수집"""
    options = {}
    
    if method == "K-Means":
        options = {
            'n_clusters': dpg.get_value("kmeans_n_clusters") if dpg.does_item_exist("kmeans_n_clusters") else 3,
            'init': dpg.get_value("kmeans_init") if dpg.does_item_exist("kmeans_init") else "k-means++",
            'max_iter': dpg.get_value("kmeans_max_iter") if dpg.does_item_exist("kmeans_max_iter") else 300,
            'standardize': dpg.get_value("kmeans_standardize") if dpg.does_item_exist("kmeans_standardize") else True,
            'elbow': dpg.get_value("kmeans_elbow") if dpg.does_item_exist("kmeans_elbow") else True
        }
    elif method == "Hierarchical":
        options = {
            'n_clusters': dpg.get_value("hier_n_clusters") if dpg.does_item_exist("hier_n_clusters") else 3,
            'linkage': dpg.get_value("hier_linkage") if dpg.does_item_exist("hier_linkage") else "ward",
            'metric': dpg.get_value("hier_metric") if dpg.does_item_exist("hier_metric") else "euclidean",
            'standardize': dpg.get_value("hier_standardize") if dpg.does_item_exist("hier_standardize") else True,
            'dendrogram': dpg.get_value("hier_dendrogram") if dpg.does_item_exist("hier_dendrogram") else True
        }
    elif method == "DBSCAN":
        options = {
            'eps': dpg.get_value("dbscan_eps") if dpg.does_item_exist("dbscan_eps") else 0.5,
            'min_samples': dpg.get_value("dbscan_min_samples") if dpg.does_item_exist("dbscan_min_samples") else 5,
            'metric': dpg.get_value("dbscan_metric") if dpg.does_item_exist("dbscan_metric") else "euclidean",
            'standardize': dpg.get_value("dbscan_standardize") if dpg.does_item_exist("dbscan_standardize") else True
        }
    elif method == "Factor Analysis":
        options = {
            'n_factors': dpg.get_value("fa_n_factors") if dpg.does_item_exist("fa_n_factors") else 2,
            'max_iter': dpg.get_value("fa_max_iter") if dpg.does_item_exist("fa_max_iter") else 1000,
            'tol': dpg.get_value("fa_tol") if dpg.does_item_exist("fa_tol") else 0.01,
            'standardize': dpg.get_value("fa_standardize") if dpg.does_item_exist("fa_standardize") else True,
            'scree': dpg.get_value("fa_scree") if dpg.does_item_exist("fa_scree") else True
        }
    
    return options

def _handle_dataframe_addition(results: Dict[str, Any], params: Dict[str, Any]):
    """분석 결과를 DataFrame에 추가하는 처리"""
    df_name = params['df_name']
    if df_name not in _available_dfs:
        return

    df = _available_dfs[df_name]
    
    # 결과 타입에 따른 처리
    if 'labels' in results:  # 군집분석
        analysis.add_clustering_results_to_dataframe(df, results, params)
    elif 'factor_scores' in results:  # 요인분석
        analysis.add_factor_results_to_dataframe(df, results, params)
    
    # UI 업데이트
    _update_df_list()

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """Step 10 UI 생성"""
    initialize(main_callbacks)
    
    main_callbacks['register_step_group_tag'](step_name, TAG_S10_GROUP)

    with dpg.group(tag=TAG_S10_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        
        # 전체 높이 계산
        total_height = 1000 
        upper_height = int(total_height * 0.65)
        lower_height = int(total_height * 0.35)
        
        # 상부 시각화 영역 (65%)
        with dpg.child_window(height=upper_height, border=True, tag=TAG_S10_UPPER_VIZ_WINDOW):
            dpg.add_text("Analysis Results", color=(255, 255, 0))
            dpg.add_separator()
            
            # 탭바 (분석 결과들이 탭으로 추가됨)
            with dpg.tab_bar(tag=TAG_S10_VIZ_TAB_BAR):
                with dpg.tab(label="Instructions"):
                    dpg.add_text("1. Select a DataFrame from the dropdown below")
                    dpg.add_text("2. Choose an analysis method")
                    dpg.add_text("3. Select variables for analysis")
                    dpg.add_text("4. Configure options and click 'Run Analysis'")
                    dpg.add_text("5. Results will appear in new tabs")
        
        # 하부 컨트롤 패널 (35%)
        with dpg.child_window(height=lower_height, border=True, tag=TAG_S10_LOWER_CONTROL_PANEL):
            with dpg.group(horizontal=True):
                # 좌측 선택 영역 (20%)
                left_width = 250
                with dpg.child_window(width=left_width, border=True):
                    dpg.add_text("Data Source", color=(255, 255, 0))
                    dpg.add_combo(label="DataFrame", tag=TAG_S10_DF_SELECTOR,
                                callback=_on_df_selected, width=-1)
                    
                    dpg.add_separator()
                    dpg.add_text("Analysis Method", color=(255, 255, 0))
                    
                    # 분석 방법론 트리
                    for category, methods in ANALYSIS_METHODS.items():
                        with dpg.tree_node(label=category, default_open=True):
                            if category == "군집분석 (Clustering)":
                                dpg.add_radio_button(
                                    items=methods,
                                    default_value="K-Means",
                                    callback=lambda s, a, u: [
                                        dpg.set_value(TAG_S10_METHOD_SELECTOR, a),
                                        _on_method_selected(s, a, u)
                                    ]
                                )
                            elif category == "차원축소 (Dimension Reduction)":
                                implemented_methods = ["Factor Analysis"]
                                
                                for method in methods:
                                    if method in implemented_methods:
                                        dpg.add_selectable(
                                            label=method,
                                            user_data=method,
                                            callback=lambda s, a, u: [
                                                dpg.set_value(TAG_S10_METHOD_SELECTOR, u),
                                                _on_method_selected(s, u, None)
                                            ]
                                        )
                                    else:
                                        dpg.add_text(f"{method} (Not implemented)", 
                                                color=(150, 150, 150))
                            else:
                                for method in methods:
                                    dpg.add_text(f"{method} (Not implemented)", 
                                            color=(150, 150, 150))
                    
                    # Hidden combo to store selected method
                    dpg.add_combo(tag=TAG_S10_METHOD_SELECTOR, show=False, 
                                default_value="K-Means", callback=_on_method_selected)

                # 우측 동적 영역 (80%)
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
                                     height=150,borders_outerH=True, borders_outerV=True,scrollY=True, policy=dpg.mvTable_SizingStretchProp):
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
                                         height=190,borders_outerH=True, borders_outerV=True, scrollY=True, policy=dpg.mvTable_SizingStretchProp):
                                dpg.add_table_column()
                        
                        dpg.add_spacer(width=20)
                        
                        # 동적 옵션 영역
                        with dpg.group(tag=TAG_S10_DYNAMIC_OPTIONS_AREA):
                            # 이 부분은 _on_method_selected에 의해 동적으로 채워짐
                            pass
                    
                    # 실행 버튼
                    dpg.add_separator()
                    dpg.add_button(label="Run Analysis", tag=TAG_S10_RUN_BUTTON,
                                callback=_run_analysis,
                                width=-1, height=30)

    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui():
    """UI 업데이트"""
    # DataFrame 목록 업데이트
    _update_df_list()

    # 초기 동적 옵션 생성
    if dpg.does_item_exist(TAG_S10_METHOD_SELECTOR):
        method = dpg.get_value(TAG_S10_METHOD_SELECTOR)
        if method:
            _create_dynamic_options(method)

def reset_state():
    """상태 초기화"""
    global _selected_df_name, _selected_vars

    _selected_df_name = ""
    _selected_vars.clear()

    # 분석 및 시각화 모듈 상태 초기화
    analysis.reset_state()
    viz.reset_state()

    # UI 초기화
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_S10_VIZ_TAB_BAR):
            # Instructions 탭을 제외한 모든 탭 제거
            tabs = dpg.get_item_children(TAG_S10_VIZ_TAB_BAR, 1)
            for tab in tabs[1:]:  # 첫 번째 탭(Instructions)은 유지
                if dpg.does_item_exist(tab):
                    dpg.delete_item(tab)
        
        update_ui()