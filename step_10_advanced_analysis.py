# step_10_advanced_analysis.py

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import traceback
import datetime
import io
import base64
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import utils
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage

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
TAG_S10_PROGRESS_MODAL = "step10_progress_modal"
TAG_S10_PROGRESS_TEXT = "step10_progress_text"
TAG_S10_ADD_TO_DF_CHECK = "step10_add_results_to_df_check"
TAG_S10_NEW_DF_NAME_INPUT = "step10_new_df_name_input"

# --- 분석 방법론 목록 ---
ANALYSIS_METHODS = {
    "군집분석 (Clustering)": ["K-Means", "Hierarchical", "DBSCAN"],
    "차원축소 (Dimension Reduction)": ["PCA", "t-SNE"],
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
_analysis_results: Dict[str, Any] = {}
_viz_tab_counter: int = 0
_texture_tags: List[str] = []

# --- Helper Functions ---

def _update_df_list():
    """데이터프레임 목록 업데이트"""
    global _available_dfs, _selected_df_name
    if not _module_main_callbacks:
        return
    
    _available_dfs = _module_main_callbacks['get_all_available_dfs']()
    
    if dpg.does_item_exist(TAG_S10_DF_SELECTOR):
        df_names = list(_available_dfs.keys())
        dpg.configure_item(TAG_S10_DF_SELECTOR, items=df_names)
        
        if _selected_df_name not in df_names and df_names:
            _selected_df_name = df_names[0]
            dpg.set_value(TAG_S10_DF_SELECTOR, _selected_df_name)
        elif not df_names:
            _selected_df_name = ""

def _on_df_selected(sender, app_data, user_data):
    """데이터프레임 선택 시 호출"""
    global _selected_df_name, _selected_vars
    _selected_df_name = app_data
    _selected_vars.clear()
    _update_variable_lists()

def _on_method_selected(sender, app_data, user_data):
    """분석 방법론 선택 시 호출"""
    _create_dynamic_options(app_data)

def _update_variable_lists():
    """변수 목록 업데이트"""
    if not _selected_df_name or _selected_df_name not in _available_dfs:
        return
    
    df = _available_dfs[_selected_df_name]
    available_vars = list(df.columns)
    
    # 검색어 필터링
    if dpg.does_item_exist(TAG_S10_VAR_SEARCH_INPUT):
        search_term = dpg.get_value(TAG_S10_VAR_SEARCH_INPUT).lower()
        if search_term:
            available_vars = [v for v in available_vars if search_term in v.lower()]
    
    # 이미 선택된 변수는 제외
    available_vars = [v for v in available_vars if v not in _selected_vars]
    
    # 사용 가능한 변수 목록 업데이트
    if dpg.does_item_exist(TAG_S10_VAR_AVAILABLE_LIST):
        dpg.delete_item(TAG_S10_VAR_AVAILABLE_LIST, children_only=True)
        for var in available_vars:
            dpg.add_selectable(label=var, parent=TAG_S10_VAR_AVAILABLE_LIST, 
                             width=-1, user_data=var)
    
    # 선택된 변수 목록 업데이트
    if dpg.does_item_exist(TAG_S10_VAR_SELECTED_LIST):
        dpg.delete_item(TAG_S10_VAR_SELECTED_LIST, children_only=True)
        for var in _selected_vars:
            dpg.add_selectable(label=var, parent=TAG_S10_VAR_SELECTED_LIST,
                             width=-1, user_data=var)

def _move_variables(direction: str):
    """변수를 좌우로 이동"""
    global _selected_vars
    
    if direction in [">", ">>"]:
        # 오른쪽으로 이동 (선택)
        source_list = TAG_S10_VAR_AVAILABLE_LIST
        if direction == ">>":
            # 모든 변수 이동
            if _selected_df_name in _available_dfs:
                df = _available_dfs[_selected_df_name]
                _selected_vars = list(df.columns)
        else:
            # 선택된 변수만 이동
            if dpg.does_item_exist(source_list):
                for child in dpg.get_item_children(source_list, 1):
                    if dpg.get_value(child):
                        var = dpg.get_item_user_data(child)
                        if var not in _selected_vars:
                            _selected_vars.append(var)
    
    elif direction in ["<", "<<"]:
        # 왼쪽으로 이동 (제거)
        if direction == "<<":
            # 모든 변수 제거
            _selected_vars.clear()
        else:
            # 선택된 변수만 제거
            source_list = TAG_S10_VAR_SELECTED_LIST
            if dpg.does_item_exist(source_list):
                vars_to_remove = []
                for child in dpg.get_item_children(source_list, 1):
                    if dpg.get_value(child):
                        vars_to_remove.append(dpg.get_item_user_data(child))
                for var in vars_to_remove:
                    if var in _selected_vars:
                        _selected_vars.remove(var)
    
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
    
    # 공통 옵션들
    dpg.add_separator(parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    dpg.add_text("Output Options:", parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    
    dpg.add_checkbox(label="Add cluster labels to original DataFrame",
                    default_value=True, tag=TAG_S10_ADD_TO_DF_CHECK,
                    parent=TAG_S10_DYNAMIC_OPTIONS_AREA)
    
    with dpg.group(horizontal=True, parent=TAG_S10_DYNAMIC_OPTIONS_AREA):
        dpg.add_text("New DataFrame name (if creating):")
        dpg.add_input_text(default_value="", tag=TAG_S10_NEW_DF_NAME_INPUT,
                         width=200, hint="Leave empty to auto-generate")

def _run_clustering_analysis():
    """군집분석 실행"""
    global _viz_tab_counter, _analysis_results
    
    if not _selected_df_name or not _selected_vars:
        _util_funcs['_show_simple_modal_message']("Error", 
            "Please select a DataFrame and at least 2 variables for clustering.")
        return
    
    df = _available_dfs[_selected_df_name]
    method = dpg.get_value(TAG_S10_METHOD_SELECTOR)
    
    # Progress modal
    _util_funcs['show_dpg_progress_modal']("Running Analysis", 
        f"Performing {method} clustering...", 
        modal_tag=TAG_S10_PROGRESS_MODAL, 
        text_tag=TAG_S10_PROGRESS_TEXT)
    
    try:
        # 데이터 준비
        X = df[_selected_vars].dropna()
        
        if len(X) < 10:
            raise ValueError("Not enough data points for clustering (need at least 10)")
        
        # 표준화 여부 확인
        standardize_tag = f"{method.lower().replace('-', '')}_standardize"
        if dpg.does_item_exist(standardize_tag) and dpg.get_value(standardize_tag):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # 분석 실행
        if method == "K-Means":
            results = _run_kmeans(X, X_scaled)
        elif method == "Hierarchical":
            results = _run_hierarchical(X, X_scaled)
        elif method == "DBSCAN":
            results = _run_dbscan(X, X_scaled)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 결과를 새 탭에 표시
        _viz_tab_counter += 1
        tab_name = f"{method}_{_viz_tab_counter}"
        results['method'] = method
        results['df_name'] = _selected_df_name
        results['variables'] = _selected_vars.copy()
        
        _create_results_tab(tab_name, results)
        
        # DataFrame에 결과 추가
        if dpg.get_value(TAG_S10_ADD_TO_DF_CHECK) and 'labels' in results:
            _add_results_to_dataframe(results)
        
    except Exception as e:
        _util_funcs['_show_simple_modal_message']("Analysis Error", 
            f"Error during {method} analysis:\n{str(e)}")
        traceback.print_exc()
    
    finally:
        _util_funcs['hide_dpg_progress_modal'](TAG_S10_PROGRESS_MODAL)

def _run_kmeans(X: pd.DataFrame, X_scaled: np.ndarray) -> Dict[str, Any]:
    """K-Means 군집분석 실행"""
    n_clusters = dpg.get_value("kmeans_n_clusters")
    init = dpg.get_value("kmeans_init")
    max_iter = dpg.get_value("kmeans_max_iter")
    
    # K-Means 모델 학습
    kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, 
                    n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    results = {
        'labels': labels,
        'centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'n_clusters': n_clusters,
        'X': X,
        'X_scaled': X_scaled
    }
    
    # 실루엣 점수 계산
    if n_clusters > 1:
        results['silhouette_avg'] = silhouette_score(X_scaled, labels)
        results['silhouette_samples'] = silhouette_samples(X_scaled, labels)
    
    # Elbow plot을 위한 데이터
    if dpg.get_value("kmeans_elbow"):
        inertias = []
        K_range = range(2, min(11, len(X)))
        for k in K_range:
            km = KMeans(n_clusters=k, init=init, max_iter=max_iter, 
                       n_init=10, random_state=42)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
        results['elbow_data'] = {'K': list(K_range), 'inertia': inertias}
    
    return results

def _run_hierarchical(X: pd.DataFrame, X_scaled: np.ndarray) -> Dict[str, Any]:
    """계층적 군집분석 실행"""
    n_clusters = dpg.get_value("hier_n_clusters")
    linkage_method = dpg.get_value("hier_linkage")
    metric = dpg.get_value("hier_metric")
    
    # 계층적 군집분석
    hc = AgglomerativeClustering(n_clusters=n_clusters, 
                                linkage=linkage_method,
                                metric=metric)
    labels = hc.fit_predict(X_scaled)
    
    results = {
        'labels': labels,
        'n_clusters': n_clusters,
        'X': X,
        'X_scaled': X_scaled
    }
    
    # 실루엣 점수
    if n_clusters > 1:
        results['silhouette_avg'] = silhouette_score(X_scaled, labels)
    
    # Dendrogram 데이터
    if dpg.get_value("hier_dendrogram"):
        linkage_matrix = linkage(X_scaled, method=linkage_method, metric=metric)
        results['linkage_matrix'] = linkage_matrix
    
    return results

def _run_dbscan(X: pd.DataFrame, X_scaled: np.ndarray) -> Dict[str, Any]:
    """DBSCAN 군집분석 실행"""
    eps = dpg.get_value("dbscan_eps")
    min_samples = dpg.get_value("dbscan_min_samples")
    metric = dpg.get_value("dbscan_metric")
    
    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    results = {
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'X': X,
        'X_scaled': X_scaled
    }
    
    # 실루엣 점수 (노이즈 제외)
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 0:
            results['silhouette_avg'] = silhouette_score(X_scaled[mask], labels[mask])
    
    return results

def _create_results_tab(tab_name: str, results: Dict[str, Any]):
    """결과를 새 탭에 표시"""
    if not dpg.does_item_exist(TAG_S10_VIZ_TAB_BAR):
        return
    
    # 새 탭 생성
    with dpg.tab(label=tab_name, parent=TAG_S10_VIZ_TAB_BAR, 
                 closable=True, tag=f"tab_{tab_name}"):
        
        # 스크롤 가능한 child window
        with dpg.child_window(border=False, tag=f"scroll_{tab_name}"):
            
            # 요약 정보
            dpg.add_text(f"Method: {results['method']}", color=(255, 255, 0))
            dpg.add_text(f"DataFrame: {results['df_name']}")
            dpg.add_text(f"Variables: {', '.join(results['variables'])}")
            dpg.add_text(f"Number of clusters: {results['n_clusters']}")
            
            if 'silhouette_avg' in results:
                dpg.add_text(f"Silhouette Score: {results['silhouette_avg']:.3f}")
            
            if 'n_noise' in results:
                dpg.add_text(f"Noise points: {results['n_noise']}")
            
            dpg.add_separator()
            
            # 시각화
            if results['method'] == "K-Means":
                _create_kmeans_visualizations(f"viz_{tab_name}", results)
            elif results['method'] == "Hierarchical":
                _create_hierarchical_visualizations(f"viz_{tab_name}", results)
            elif results['method'] == "DBSCAN":
                _create_dbscan_visualizations(f"viz_{tab_name}", results)
            
            # Export 버튼
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Export to Excel", 
                             callback=lambda: _export_results(results, 'excel'))
                dpg.add_button(label="Export to HTML", 
                             callback=lambda: _export_results(results, 'html'))

def _create_kmeans_visualizations(parent_tag: str, results: Dict[str, Any]):
    """K-Means 결과 시각화"""
    plot_func = _util_funcs.get('plot_to_dpg_texture', utils.plot_to_dpg_texture)
    
    # 1. Elbow Plot
    if 'elbow_data' in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(results['elbow_data']['K'], results['elbow_data']['inertia'], 
                'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method For Optimal k')
        ax.grid(True, alpha=0.3)
        
        tex_tag, w, h, img_bytes = plot_func(fig)
        if tex_tag:
            _texture_tags.append(tex_tag)
            dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
        plt.close(fig)
        dpg.add_separator(parent=parent_tag)
    
    # 2. Cluster Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels, counts = np.unique(results['labels'], return_counts=True)
    bars = ax.bar(unique_labels, counts, color=plt.cm.viridis(np.linspace(0, 1, len(unique_labels))))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Points')
    ax.set_title('Cluster Distribution')
    ax.set_xticks(unique_labels)
    
    # 막대 위에 개수 표시
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    tex_tag, w, h, img_bytes = plot_func(fig)
    if tex_tag:
        _texture_tags.append(tex_tag)
        dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
    plt.close(fig)
    dpg.add_separator(parent=parent_tag)
    
    # 3. Silhouette Plot
    if 'silhouette_samples' in results:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        y_lower = 10
        for i in range(results['n_clusters']):
            cluster_silhouette_values = results['silhouette_samples'][results['labels'] == i]
            cluster_silhouette_values.sort()
            
            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.viridis(float(i) / results['n_clusters'])
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        ax.set_xlabel("Silhouette Coefficient Values")
        ax.set_ylabel("Cluster Label")
        ax.set_title("Silhouette Plot for Each Cluster")
        ax.axvline(x=results['silhouette_avg'], color="red", linestyle="--", 
                  label=f"Average: {results['silhouette_avg']:.3f}")
        ax.legend()
        
        tex_tag, w, h, img_bytes = plot_func(fig)
        if tex_tag:
            _texture_tags.append(tex_tag)
            dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
        plt.close(fig)
        dpg.add_separator(parent=parent_tag)
    
    # 4. Scatter Plot (if 2D)
    if len(results['variables']) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(results['X'].iloc[:, 0], results['X'].iloc[:, 1], 
                           c=results['labels'], cmap='viridis', alpha=0.6, s=50)
        
        # 클러스터 중심점 표시 (표준화된 데이터를 원래 스케일로 변환)
        if dpg.get_value("kmeans_standardize"):
            # 역변환이 필요하지만, 여기서는 간단히 표시만
            ax.scatter(results['centers'][:, 0], results['centers'][:, 1], 
                      c='red', s=200, alpha=0.8, marker='x', linewidths=3)
        
        ax.set_xlabel(results['variables'][0])
        ax.set_ylabel(results['variables'][1])
        ax.set_title('Cluster Scatter Plot (First 2 Variables)')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        tex_tag, w, h, img_bytes = plot_func(fig)
        if tex_tag:
            _texture_tags.append(tex_tag)
            dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
        plt.close(fig)

def _create_hierarchical_visualizations(parent_tag: str, results: Dict[str, Any]):
    """계층적 군집분석 결과 시각화"""
    plot_func = _util_funcs.get('plot_to_dpg_texture', utils.plot_to_dpg_texture)
    
    # 1. Dendrogram
    if 'linkage_matrix' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(results['linkage_matrix'], ax=ax, truncate_mode='level', p=6)
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Sample Index or (Cluster Size)')
        ax.set_ylabel('Distance')
        
        tex_tag, w, h, img_bytes = plot_func(fig)
        if tex_tag:
            _texture_tags.append(tex_tag)
            dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
        plt.close(fig)
        dpg.add_separator(parent=parent_tag)
    
    # 2. Cluster Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels, counts = np.unique(results['labels'], return_counts=True)
    bars = ax.bar(unique_labels, counts, color=plt.cm.viridis(np.linspace(0, 1, len(unique_labels))))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Points')
    ax.set_title('Cluster Distribution')
    ax.set_xticks(unique_labels)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')

    tex_tag, w, h, img_bytes = plot_func(fig)
    if tex_tag:
        _texture_tags.append(tex_tag)
        dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
    plt.close(fig)
    dpg.add_separator(parent=parent_tag)

    # 3. Scatter Plot (if 2D)
    if len(results['variables']) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(results['X'].iloc[:, 0], results['X'].iloc[:, 1], 
                            c=results['labels'], cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel(results['variables'][0])
        ax.set_ylabel(results['variables'][1])
        ax.set_title('Cluster Scatter Plot (First 2 Variables)')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        
        tex_tag, w, h, img_bytes = plot_func(fig)
        if tex_tag:
            _texture_tags.append(tex_tag)
            dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
        plt.close(fig)

def _create_dbscan_visualizations(parent_tag: str, results: Dict[str, Any]):
    plot_func = _util_funcs.get('plot_to_dpg_texture', utils.plot_to_dpg_texture)

    # 1. Cluster Distribution (including noise)
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels, counts = np.unique(results['labels'], return_counts=True)

    # 노이즈(-1)와 클러스터를 구분하여 색상 지정
    colors = []
    labels_text = []
    for label in unique_labels:
        if label == -1:
            colors.append('red')
            labels_text.append('Noise')
        else:
            colors.append(plt.cm.viridis(label / max(1, results['n_clusters'])))
            labels_text.append(f'Cluster {label}')

    bars = ax.bar(range(len(unique_labels)), counts, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Points')
    ax.set_title('DBSCAN Cluster Distribution')
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(labels_text, rotation=45, ha='right')

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')

    tex_tag, w, h, img_bytes = plot_func(fig)
    if tex_tag:
        _texture_tags.append(tex_tag)
        dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
    plt.close(fig)
    dpg.add_separator(parent=parent_tag)

    # 2. Scatter Plot (if 2D)
    if len(results['variables']) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 노이즈 포인트와 클러스터 포인트를 분리하여 표시
        mask_noise = results['labels'] == -1
        mask_clusters = results['labels'] != -1
        
        # 클러스터 포인트
        if mask_clusters.any():
            scatter = ax.scatter(results['X'].iloc[mask_clusters, 0], 
                                results['X'].iloc[mask_clusters, 1],
                                c=results['labels'][mask_clusters], 
                                cmap='viridis', alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        # 노이즈 포인트
        if mask_noise.any():
            ax.scatter(results['X'].iloc[mask_noise, 0], 
                        results['X'].iloc[mask_noise, 1],
                        c='red', marker='x', s=50, label='Noise')
            ax.legend()
        
        ax.set_xlabel(results['variables'][0])
        ax.set_ylabel(results['variables'][1])
        ax.set_title('DBSCAN Cluster Scatter Plot (First 2 Variables)')
        
        tex_tag, w, h, img_bytes = plot_func(fig)
        if tex_tag:
            _texture_tags.append(tex_tag)
            dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
        plt.close(fig)

def _add_results_to_dataframe(results: Dict[str, Any]):
    """분석 결과를 DataFrame에 추가"""
    df_name = results['df_name']
    if df_name not in _available_dfs:
        return

    df = _available_dfs[df_name]
    X = results['X']
    labels = results['labels']

    # 클러스터 레이블을 원본 DataFrame에 추가
    cluster_col_name = f"cluster_{results['method'].lower().replace('-', '_')}"

    # X의 인덱스를 사용하여 원본 DataFrame에 레이블 추가
    df.loc[X.index, cluster_col_name] = labels

    # 새 DataFrame 생성 여부 확인
    new_df_name = dpg.get_value(TAG_S10_NEW_DF_NAME_INPUT).strip()
    if new_df_name:
        # 새 DataFrame 생성
        new_df = df.copy()
        if _module_main_callbacks and 'step8_derivation_complete' in _module_main_callbacks:
            _module_main_callbacks['step8_derivation_complete'](new_df_name, new_df)
            _util_funcs['_show_simple_modal_message']("Success", 
                f"New DataFrame '{new_df_name}' created with cluster results.")
    else:
        # 기존 DataFrame 업데이트
        _util_funcs['_show_simple_modal_message']("Success", 
            f"Cluster labels added to '{df_name}' as column '{cluster_col_name}'.")

    # UI 업데이트
    _update_df_list()

def _export_results(results: Dict[str, Any], format: str):
    """결과 내보내기"""
    if format == 'excel':
        _export_to_excel(results)
    elif format == 'html':
        _export_to_html(results)

def _export_to_excel(results: Dict[str, Any]):
    """Excel로 내보내기"""
    try:        
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        filename = f"{current_date}_{results['method']}_Analysis.xlsx"
        
        with dpg.file_dialog(
            directory_selector=False, show=True, 
            callback=lambda s, a: _save_excel_file(a['file_path_name'], results),
            default_filename=filename, width=700, height=400, modal=True
        ):
            dpg.add_file_extension(".xlsx", color=(0, 255, 0, 255))
            
    except ImportError:
        _util_funcs['_show_simple_modal_message']("Error", 
            "openpyxl not installed. Cannot export to Excel.")

def _save_excel_file(filepath: str, results: Dict[str, Any]):
    """Excel 파일 저장"""
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 요약 정보
            summary_data = {
                'Metric': ['Method', 'DataFrame', 'Variables', 'Number of Clusters'],
                'Value': [results['method'], results['df_name'], 
                        ', '.join(results['variables']), results['n_clusters']]
            }
            
            if 'silhouette_avg' in results:
                summary_data['Metric'].append('Silhouette Score')
                summary_data['Value'].append(f"{results['silhouette_avg']:.3f}")
            
            if 'n_noise' in results:
                summary_data['Metric'].append('Noise Points')
                summary_data['Value'].append(results['n_noise'])
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 클러스터 할당
            cluster_df = results['X'].copy()
            cluster_df['Cluster'] = results['labels']
            cluster_df.to_excel(writer, sheet_name='Cluster_Assignments')
            
            # 클러스터별 통계
            cluster_stats = cluster_df.groupby('Cluster').agg(['mean', 'std', 'count'])
            cluster_stats.to_excel(writer, sheet_name='Cluster_Statistics')
        
        _util_funcs['_show_simple_modal_message']("Success", 
            f"Results exported to:\n{filepath}")
            
    except Exception as e:
        _util_funcs['_show_simple_modal_message']("Export Error", 
            f"Failed to export: {str(e)}")

def _export_to_html(results: Dict[str, Any]):
    """HTML로 내보내기"""
    # HTML 내보내기 구현 (step_02a_sva.py의 로직 참조)
    pass

    # --- Main UI Creation ---

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """Step 10 UI 생성"""
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()

    main_callbacks['register_step_group_tag'](step_name, TAG_S10_GROUP)

    with dpg.group(tag=TAG_S10_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        
        # 전체 높이 계산
        total_height = 1000 
        # if dpg.is_dearpygui_running() else 600
        upper_height = int(total_height * 0.65)
        lower_height = int(total_height * 0.35)
        # print(total_height, upper_height, lower_height)
        
        # 상부 시각화 영역 (70%)
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
        
        # 하부 컨트롤 패널 (30%)
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
                        with dpg.tree_node(label=category):
                            for method in methods:
                                if category == "군집분석 (Clustering)":
                                    dpg.add_selectable(label=method, 
                                                    callback=lambda s, a, u: [
                                                        dpg.set_value(TAG_S10_METHOD_SELECTOR, u),
                                                        _on_method_selected(s, u, None)
                                                    ],
                                                    user_data=method)
                                else:
                                    dpg.add_text(f"{method} (Not implemented)", 
                                                color=(150, 150, 150))
                    
                    # Hidden combo to store selected method
                    dpg.add_combo(tag=TAG_S10_METHOD_SELECTOR, show=False, 
                                default_value="K-Means")
                
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
                            with dpg.child_window(height=150, border=True, 
                                                tag=TAG_S10_VAR_AVAILABLE_LIST):
                                pass
                        
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
                            with dpg.child_window(height=190, border=True, 
                                                tag=TAG_S10_VAR_SELECTED_LIST):
                                pass
                        
                        dpg.add_spacer(width=20)
                        
                        # 동적 옵션 영역
                        with dpg.group(tag=TAG_S10_DYNAMIC_OPTIONS_AREA):
                            dpg.add_text("Select an analysis method to see options")
                    
                    # 실행 버튼
                    dpg.add_separator()
                    dpg.add_button(label="Run Analysis", tag=TAG_S10_RUN_BUTTON,
                                callback=_run_clustering_analysis,
                                width=-1, height=30)

    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui():
    global _texture_tags

    # 이전 텍스처 정리
    for tag in _texture_tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    _texture_tags.clear()

    # DataFrame 목록 업데이트
    _update_df_list()

    # 초기 동적 옵션 생성
    if dpg.does_item_exist(TAG_S10_METHOD_SELECTOR):
        method = dpg.get_value(TAG_S10_METHOD_SELECTOR)
        if method:
            _create_dynamic_options(method)

def reset_state():
    global _selected_df_name, _selected_vars, _analysis_results, _viz_tab_counter, _texture_tags

    _selected_df_name = ""
    _selected_vars.clear()
    _analysis_results.clear()
    _viz_tab_counter = 0

    # 텍스처 정리
    for tag in _texture_tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    _texture_tags.clear()

    # UI 초기화
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_S10_VIZ_TAB_BAR):
            # Instructions 탭을 제외한 모든 탭 제거
            tabs = dpg.get_item_children(TAG_S10_VIZ_TAB_BAR, 1)
            for tab in tabs[1:]:  # 첫 번째 탭(Instructions)은 유지
                if dpg.does_item_exist(tab):
                    dpg.delete_item(tab)
        
        update_ui()