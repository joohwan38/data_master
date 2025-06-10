# step_10_visualization.py - 시각화 담당 (간소화 버전)

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings('ignore')

# --- Module State ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_viz_tab_counter: int = 0
_texture_tags: List[str] = []

def initialize(main_callbacks: dict):
    """시각화 모듈 초기화"""
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()

def create_results_tab(tab_bar_tag: str, results: Dict[str, Any]):
    """결과를 새 탭에 표시 - SAS Output 스타일"""
    global _viz_tab_counter
    
    if not dpg.does_item_exist(tab_bar_tag):
        return
    
    _viz_tab_counter += 1
    tab_name = f"{results['method']}_{_viz_tab_counter}"
    
    with dpg.tab(label=tab_name, parent=tab_bar_tag, closable=True, tag=f"tab_{tab_name}"):
        with dpg.child_window(border=False, tag=f"scroll_{tab_name}"):
            # 요약 정보
            _create_summary_info(results)
            dpg.add_separator()
            
            # 통계 테이블
            _create_statistical_tables(results)
            dpg.add_separator()
            
            # 시각화
            viz_parent_tag = f"viz_{tab_name}"
            _create_visualizations(viz_parent_tag, results)
            
            # Export 버튼
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Export to Excel", callback=lambda: _export_results(results, 'excel'))
                dpg.add_button(label="Export to HTML", callback=lambda: _export_results(results, 'html'))

def _create_summary_info(results: Dict[str, Any]):
    """요약 정보 생성"""
    dpg.add_text(f"Method: {results['method']}", color=(255, 255, 0))
    dpg.add_text(f"DataFrame: {results['df_name']}")
    dpg.add_text(f"Variables: {', '.join(results['variables'])}")
    
    if 'n_clusters' in results:
        dpg.add_text(f"Number of clusters: {results['n_clusters']}")
    if 'n_factors' in results:
        dpg.add_text(f"Number of factors: {results['n_factors']}")
    if 'n_components' in results:
        dpg.add_text(f"Number of components: {results['n_components']}")
    if 'silhouette_avg' in results:
        dpg.add_text(f"Silhouette Score: {results['silhouette_avg']:.3f}")
    if 'explained_variance_ratio' in results:
        total_var = np.sum(results['explained_variance_ratio']) * 100
        dpg.add_text(f"Total Explained Variance: {total_var:.1f}%")
    if 'n_noise' in results:
        dpg.add_text(f"Noise points: {results['n_noise']}")

def _create_statistical_tables(results: Dict[str, Any]):
    """통계 테이블 생성"""
    method = results['method']
    
    if method in ["K-Means", "Hierarchical", "DBSCAN"]:
        _create_clustering_tables(results)
    elif method == "Factor Analysis":
        _create_factor_tables(results)
    elif method == "PCA":
        _create_pca_tables(results)
    elif method in ["Pearson", "Spearman", "Kendall"]:
        _create_correlation_tables(results)
    elif method in ["Linear", "Logistic"]:
        _create_regression_tables(results)

def _create_clustering_tables(results: Dict[str, Any]):
    """군집분석 테이블"""
    # 클러스터 요약
    dpg.add_text("Cluster Summary", color=(255, 255, 0))
    unique_labels, counts = np.unique(results['labels'], return_counts=True)
    percentages = (counts / len(results['labels'])) * 100
    
    summary_data = []
    for label, count, pct in zip(unique_labels, counts, percentages):
        cluster_name = f"Cluster {label}" if label != -1 else "Noise"
        summary_data.append({"Cluster": cluster_name, "Frequency": count, "Percentage": f"{pct:.1f}%"})
    
    _create_table("Cluster Summary", pd.DataFrame(summary_data))
    
    # 품질 지표
    if 'silhouette_avg' in results or 'inertia' in results:
        dpg.add_text("Quality Metrics", color=(255, 255, 0))
        quality_data = []
        if 'silhouette_avg' in results:
            quality_data.append({
                "Metric": "Silhouette Score", 
                "Value": f"{results['silhouette_avg']:.4f}",
                "Interpretation": _interpret_silhouette(results['silhouette_avg'])
            })
        if 'inertia' in results:
            quality_data.append({
                "Metric": "WCSS", 
                "Value": f"{results['inertia']:.2f}",
                "Interpretation": "Lower is better"
            })
        _create_table("Quality Metrics", pd.DataFrame(quality_data))

def _create_factor_tables(results: Dict[str, Any]):
    """요인분석 테이블"""
    # 로딩 행렬
    dpg.add_text("Factor Loadings", color=(255, 255, 0))
    loadings_df = pd.DataFrame(
        results['loadings'], 
        index=results['variables'],
        columns=[f'Factor {i+1}' for i in range(results['n_factors'])]
    ).reset_index().rename(columns={'index': 'Variable'})
    _create_table("Factor Loadings", loadings_df)
    
    # 설명 분산
    dpg.add_text("Variance Explained", color=(255, 255, 0))
    variance_data = []
    cumulative = 0
    for i in range(results['n_factors']):
        var_pct = results['explained_variance_ratio'][i] * 100
        cumulative += var_pct
        variance_data.append({
            "Factor": f"Factor {i+1}",
            "Variance %": f"{var_pct:.2f}%",
            "Cumulative %": f"{cumulative:.2f}%"
        })
    _create_table("Variance Explained", pd.DataFrame(variance_data))

def _create_pca_tables(results: Dict[str, Any]):
    """PCA 테이블"""
    # 설명 분산
    dpg.add_text("Principal Components", color=(255, 255, 0))
    variance_data = []
    cumulative = 0
    for i in range(results['n_components']):
        var_pct = results['explained_variance_ratio'][i] * 100
        cumulative += var_pct
        variance_data.append({
            "Component": f"PC{i+1}",
            "Eigenvalue": f"{results['explained_variance'][i]:.4f}",
            "Variance %": f"{var_pct:.2f}%",
            "Cumulative %": f"{cumulative:.2f}%"
        })
    _create_table("Principal Components", pd.DataFrame(variance_data))

def _create_correlation_tables(results: Dict[str, Any]):
    """상관분석 테이블"""
    dpg.add_text("Correlation Matrix", color=(255, 255, 0))
    
    # 상관계수 행렬을 테이블로 표시
    corr_matrix = results['correlation_matrix']
    corr_display = corr_matrix.reset_index()
    _create_table("Correlation Matrix", corr_display)

def _create_table(title: str, df: pd.DataFrame):
    """공통 테이블 생성 함수"""
    table_tag = dpg.generate_uuid()
    with dpg.table(header_row=True, resizable=True, tag=table_tag,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
        _util_funcs.get('create_table_with_data', lambda *args: None)(table_tag, df)
    dpg.add_spacer(height=10)

def _create_regression_tables(results: Dict[str, Any]):
    """회귀분석 테이블"""
    method = results['method']
    
    # 모델 계수
    dpg.add_text("Model Coefficients", color=(255, 255, 0))
    coef_df = results['coefficients'].copy()
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
    _create_table("Model Coefficients", coef_df[['Feature', 'Coefficient']])
    
    # 성능 지표
    dpg.add_text("Model Performance", color=(255, 255, 0))
    metrics = results['metrics']
    
    if method == "Linear":
        perf_data = [
            {"Metric": "R² (Training)", "Value": f"{metrics['R2_train']:.4f}"},
            {"Metric": "R² (Testing)", "Value": f"{metrics['R2_test']:.4f}"},
            {"Metric": "RMSE (Training)", "Value": f"{metrics['RMSE_train']:.4f}"},
            {"Metric": "RMSE (Testing)", "Value": f"{metrics['RMSE_test']:.4f}"}
        ]
    else:  # Logistic
        perf_data = [
            {"Metric": "Accuracy (Training)", "Value": f"{metrics['Accuracy_train']:.4f}"},
            {"Metric": "Accuracy (Testing)", "Value": f"{metrics['Accuracy_test']:.4f}"}
        ]
    
    _create_table("Performance Metrics", pd.DataFrame(perf_data))

def _create_visualizations(parent_tag: str, results: Dict[str, Any]):
    """시각화 생성"""
    method = results['method']
    plot_func = _util_funcs.get('plot_to_dpg_texture')
    
    if method == "K-Means":
        _create_kmeans_plots(parent_tag, results, plot_func)
    elif method == "Hierarchical":
        _create_hierarchical_plots(parent_tag, results, plot_func)
    elif method == "DBSCAN":
        _create_dbscan_plots(parent_tag, results, plot_func)
    elif method == "Factor Analysis":
        _create_factor_plots(parent_tag, results, plot_func)
    elif method == "PCA":
        _create_pca_plots(parent_tag, results, plot_func)
    elif method in ["Pearson", "Spearman", "Kendall"]:
        _create_correlation_plots(parent_tag, results, plot_func)
    elif method in ["Linear", "Logistic"]:
        _create_regression_plots(parent_tag, results, plot_func)

def _create_kmeans_plots(parent_tag: str, results: Dict[str, Any], plot_func):
    """K-Means 시각화"""
    # Elbow Plot
    if 'elbow_data' in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(results['elbow_data']['K'], results['elbow_data']['inertia'], 'bo-')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method')
        ax.grid(True, alpha=0.3)
        _add_plot(fig, plot_func, parent_tag)
    
    # Cluster Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels, counts = np.unique(results['labels'], return_counts=True)
    ax.bar(unique_labels, counts, color=plt.cm.viridis(np.linspace(0, 1, len(unique_labels))))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')
    ax.set_title('Cluster Distribution')
    _add_plot(fig, plot_func, parent_tag)

def _create_hierarchical_plots(parent_tag: str, results: Dict[str, Any], plot_func):
    """계층적 군집분석 시각화"""
    # Dendrogram
    if 'linkage_matrix' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(results['linkage_matrix'], ax=ax)
        ax.set_title('Dendrogram')
        _add_plot(fig, plot_func, parent_tag)

def _create_dbscan_plots(parent_tag: str, results: Dict[str, Any], plot_func):
    """DBSCAN 시각화"""
    # Cluster Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels, counts = np.unique(results['labels'], return_counts=True)
    colors = ['red' if label == -1 else plt.cm.viridis(label/max(1, results['n_clusters'])) 
              for label in unique_labels]
    ax.bar(range(len(unique_labels)), counts, color=colors)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')
    ax.set_title('DBSCAN Results')
    _add_plot(fig, plot_func, parent_tag)

def _create_factor_plots(parent_tag: str, results: Dict[str, Any], plot_func):
    """요인분석 시각화"""
    # Scree Plot
    if 'eigenvalues' in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(results['eigenvalues']) + 1), results['eigenvalues'], 'bo-')
        ax.axhline(y=1, color='red', linestyle='--', label='Kaiser Criterion')
        ax.set_xlabel('Factor Number')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Scree Plot')
        ax.legend()
        _add_plot(fig, plot_func, parent_tag)
    
    # Loadings Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    loadings_df = pd.DataFrame(
        results['loadings'], 
        index=results['variables'],
        columns=[f'Factor {i+1}' for i in range(results['n_factors'])]
    )
    sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Factor Loadings')
    _add_plot(fig, plot_func, parent_tag)

def _create_pca_plots(parent_tag: str, results: Dict[str, Any], plot_func):
    """PCA 시각화"""
    # Scree Plot
    if 'eigenvalues' in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(results['eigenvalues']) + 1), results['eigenvalues'], 'bo-')
        ax.axhline(y=1, color='red', linestyle='--', label='Kaiser Criterion')
        ax.set_xlabel('Component Number')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Scree Plot')
        ax.legend()
        _add_plot(fig, plot_func, parent_tag)
    
    # Scores Plot
    if results['n_components'] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scores = results['component_scores']
        ax.scatter(scores[:, 0], scores[:, 1], alpha=0.6)
        pc1_var = results['explained_variance_ratio'][0] * 100
        pc2_var = results['explained_variance_ratio'][1] * 100
        ax.set_xlabel(f'PC1 ({pc1_var:.1f}%)')
        ax.set_ylabel(f'PC2 ({pc2_var:.1f}%)')
        ax.set_title('PCA Scores Plot')
        _add_plot(fig, plot_func, parent_tag)

def _create_correlation_plots(parent_tag: str, results: Dict[str, Any], plot_func):
    """상관분석 시각화"""
    # Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = results['correlation_matrix']
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax, 
                square=True, fmt='.3f')
    ax.set_title(f'{results["method"]} Correlation Matrix')
    _add_plot(fig, plot_func, parent_tag)

def _create_regression_plots(parent_tag: str, results: Dict[str, Any], plot_func):
    """회귀분석 시각화"""
    method = results['method']
    
    if method == "Linear":
        # 실제값 vs 예측값
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(results['y_test'], results['predictions_test'], alpha=0.6)
        ax.plot([results['y_test'].min(), results['y_test'].max()], 
                [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted (Test Set)')
        _add_plot(fig, plot_func, parent_tag)
        
        # 잔차 플롯
        fig, ax = plt.subplots(figsize=(8, 6))
        residuals = results['y_test'] - results['predictions_test']
        ax.scatter(results['predictions_test'], residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        _add_plot(fig, plot_func, parent_tag)
    
    elif method == "Logistic":
        # 혼동 행렬
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = results['metrics']['Confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        _add_plot(fig, plot_func, parent_tag)
    
    # 특성 중요도
    fig, ax = plt.subplots(figsize=(8, 6))
    coef_df = results['coefficients'].copy()
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient')
    
    colors = ['red' if x < 0 else 'blue' for x in coef_df['Coefficient']]
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Feature Coefficients')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    _add_plot(fig, plot_func, parent_tag)

def _add_plot(fig, plot_func, parent_tag):
    """플롯을 DPG에 추가하는 헬퍼 함수"""
    tex_tag, w, h, img_bytes = plot_func(fig)
    if tex_tag:
        _texture_tags.append(tex_tag)
        dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
    plt.close(fig)
    dpg.add_separator(parent=parent_tag)

def _interpret_silhouette(score: float) -> str:
    """실루엣 점수 해석"""
    if score > 0.7:
        return "Strong"
    elif score > 0.5:
        return "Reasonable"
    elif score > 0.25:
        return "Weak"
    else:
        return "Poor"

def _export_results(results: Dict[str, Any], format: str):
    """결과 내보내기"""
    import step_10_analysis as analysis
    analysis.export_results(results, format)

def reset_state():
    """상태 초기화"""
    global _viz_tab_counter, _texture_tags
    _viz_tab_counter = 0
    
    for tag in _texture_tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    _texture_tags.clear()