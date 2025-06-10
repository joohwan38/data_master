# step_10_visualization.py - 시각화 담당

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
    """결과를 새 탭에 표시 - SAS Output 스타일로 통계 테이블과 시각화를 함께 표시"""
    global _viz_tab_counter
    
    if not dpg.does_item_exist(tab_bar_tag):
        return
    
    # 새 탭 생성
    _viz_tab_counter += 1
    tab_name = f"{results['method']}_{_viz_tab_counter}"
    
    with dpg.tab(label=tab_name, parent=tab_bar_tag, 
                 closable=True, tag=f"tab_{tab_name}"):
        
        # 스크롤 가능한 child window
        with dpg.child_window(border=False, tag=f"scroll_{tab_name}"):
            
            # 요약 정보 표시
            _create_summary_info(results)
            
            dpg.add_separator()
            
            # 통계 테이블 생성 (SAS 스타일)
            _create_statistical_tables(results)
            
            dpg.add_separator()
            
            # 시각화 생성
            viz_parent_tag = f"viz_{tab_name}"
            if results['method'] == "K-Means":
                _create_kmeans_visualizations(viz_parent_tag, results)
            elif results['method'] == "Hierarchical":
                _create_hierarchical_visualizations(viz_parent_tag, results)
            elif results['method'] == "DBSCAN":
                _create_dbscan_visualizations(viz_parent_tag, results)
            elif results['method'] == "Factor Analysis":
                _create_factor_visualizations(viz_parent_tag, results)
            
            # Export 버튼
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Export to Excel", 
                             callback=lambda: _export_results(results, 'excel'))
                dpg.add_button(label="Export to HTML", 
                             callback=lambda: _export_results(results, 'html'))

def _create_statistical_tables(results: Dict[str, Any]):
    """SAS 스타일의 통계 테이블 생성"""
    method = results['method']
    
    if method in ["K-Means", "Hierarchical", "DBSCAN"]:
        _create_clustering_tables(results)
    elif method == "Factor Analysis":
        _create_factor_analysis_tables(results)

def _create_clustering_tables(results: Dict[str, Any]):
    """군집분석 통계 테이블 생성"""
    
    # 1. 클러스터 요약 통계
    dpg.add_text("Cluster Summary Statistics", color=(255, 255, 0))
    
    # 클러스터별 빈도 테이블
    unique_labels, counts = np.unique(results['labels'], return_counts=True)
    percentages = (counts / len(results['labels'])) * 100
    
    cluster_summary_data = []
    for label, count, pct in zip(unique_labels, counts, percentages):
        cluster_name = f"Cluster {label}" if label != -1 else "Noise"
        cluster_summary_data.append({
            "Cluster": cluster_name,
            "Frequency": count,
            "Percentage": f"{pct:.1f}%"
        })
    
    cluster_summary_df = pd.DataFrame(cluster_summary_data)
    
    # 테이블 생성
    cluster_table_tag = dpg.generate_uuid()
    with dpg.table(header_row=True, resizable=True, tag=cluster_table_tag,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
        _util_funcs.get('create_table_with_data', lambda *args: None)(cluster_table_tag, cluster_summary_df)
    
    dpg.add_spacer(height=10)
    
    # 2. 변수별 클러스터 평균 (K-Means, Hierarchical만)
    if results['method'] in ["K-Means", "Hierarchical"]:
        dpg.add_text("Cluster Means by Variables", color=(255, 255, 0))
        
        X = results['X']
        labels = results['labels']
        
        # 클러스터별 평균 계산
        cluster_means_data = []
        for var in results['variables']:
            row_data = {"Variable": var}
            for cluster_id in sorted(unique_labels):
                if cluster_id != -1:  # 노이즈 제외
                    cluster_mask = labels == cluster_id
                    if cluster_mask.sum() > 0:
                        mean_val = X.loc[cluster_mask, var].mean()
                        row_data[f"Cluster {cluster_id}"] = f"{mean_val:.3f}"
            cluster_means_data.append(row_data)
        
        cluster_means_df = pd.DataFrame(cluster_means_data)
        
        means_table_tag = dpg.generate_uuid()
        with dpg.table(header_row=True, resizable=True, tag=means_table_tag,
                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            _util_funcs.get('create_table_with_data', lambda *args: None)(means_table_tag, cluster_means_df)
        
        dpg.add_spacer(height=10)
    
    # 3. 클러스터 품질 지표
    if 'silhouette_avg' in results or 'inertia' in results:
        dpg.add_text("Cluster Quality Metrics", color=(255, 255, 0))
        
        quality_data = []
        if 'silhouette_avg' in results:
            quality_data.append({
                "Metric": "Silhouette Score",
                "Value": f"{results['silhouette_avg']:.4f}",
                "Interpretation": _interpret_silhouette_score(results['silhouette_avg'])
            })
        
        if 'inertia' in results:
            quality_data.append({
                "Metric": "Within-cluster Sum of Squares (WCSS)",
                "Value": f"{results['inertia']:.2f}",
                "Interpretation": "Lower values indicate tighter clusters"
            })
        
        if results['method'] == "DBSCAN" and 'n_noise' in results:
            noise_pct = (results['n_noise'] / len(results['labels'])) * 100
            quality_data.append({
                "Metric": "Noise Points",
                "Value": f"{results['n_noise']} ({noise_pct:.1f}%)",
                "Interpretation": "Points not assigned to any cluster"
            })
        
        quality_df = pd.DataFrame(quality_data)
        
        quality_table_tag = dpg.generate_uuid()
        with dpg.table(header_row=True, resizable=True, tag=quality_table_tag,
                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            _util_funcs.get('create_table_with_data', lambda *args: None)(quality_table_tag, quality_df)

def _create_factor_analysis_tables(results: Dict[str, Any]):
    """요인분석 통계 테이블 생성"""
    
    # 1. 요인 적재값 테이블
    dpg.add_text("Factor Loadings Matrix", color=(255, 255, 0))
    
    loadings_df = pd.DataFrame(
        results['loadings'], 
        index=results['variables'],
        columns=[f'Factor {i+1}' for i in range(results['n_factors'])]
    )
    # 인덱스를 열로 변환
    loadings_display_df = loadings_df.reset_index()
    loadings_display_df = loadings_display_df.rename(columns={'index': 'Variable'})
    
    loadings_table_tag = dpg.generate_uuid()
    with dpg.table(header_row=True, resizable=True, tag=loadings_table_tag,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
        _util_funcs.get('create_table_with_data', lambda *args: None)(loadings_table_tag, loadings_display_df)
    
    dpg.add_spacer(height=10)
    
    # 2. 공통성 테이블
    dpg.add_text("Communalities", color=(255, 255, 0))
    
    communalities_data = []
    for i, var in enumerate(results['variables']):
        communalities_data.append({
            "Variable": var,
            "Communality": f"{results['communalities'][i]:.4f}",
            "Interpretation": _interpret_communality(results['communalities'][i])
        })
    
    communalities_df = pd.DataFrame(communalities_data)
    
    comm_table_tag = dpg.generate_uuid()
    with dpg.table(header_row=True, resizable=True, tag=comm_table_tag,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
        _util_funcs.get('create_table_with_data', lambda *args: None)(comm_table_tag, communalities_df)
    
    dpg.add_spacer(height=10)
    
    # 3. 설명된 분산 테이블
    dpg.add_text("Variance Explained by Factors", color=(255, 255, 0))
    
    variance_data = []
    cumulative_var = 0
    for i in range(results['n_factors']):
        factor_var = results['explained_variance_ratio'][i] * 100
        cumulative_var += factor_var
        variance_data.append({
            "Factor": f"Factor {i+1}",
            "Eigenvalue": f"{results['explained_variance'][i]:.4f}" if 'explained_variance' in results else "N/A",
            "Variance %": f"{factor_var:.2f}%",
            "Cumulative %": f"{cumulative_var:.2f}%"
        })
    
    variance_df = pd.DataFrame(variance_data)
    
    var_table_tag = dpg.generate_uuid()
    with dpg.table(header_row=True, resizable=True, tag=var_table_tag,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
        _util_funcs.get('create_table_with_data', lambda *args: None)(var_table_tag, variance_df)
    
    dpg.add_spacer(height=10)
    
    # 4. Kaiser-Meyer-Olkin (KMO) 측정 (간단한 버전)
    if 'eigenvalues' in results:
        dpg.add_text("Factor Analysis Adequacy", color=(255, 255, 0))
        
        # Kaiser Criterion 확인
        kaiser_factors = sum(1 for ev in results['eigenvalues'] if ev > 1.0)
        total_var_explained = sum(results['explained_variance_ratio']) * 100
        
        adequacy_data = [
            {
                "Measure": "Total Variance Explained",
                "Value": f"{total_var_explained:.1f}%",
                "Criterion": ">60% is adequate"
            },
            {
                "Measure": "Factors with Eigenvalue > 1",
                "Value": f"{kaiser_factors}",
                "Criterion": "Kaiser Criterion"
            },
            {
                "Measure": "Selected Factors",
                "Value": f"{results['n_factors']}",
                "Criterion": "User specified"
            }
        ]
        
        adequacy_df = pd.DataFrame(adequacy_data)
        
        adequacy_table_tag = dpg.generate_uuid()
        with dpg.table(header_row=True, resizable=True, tag=adequacy_table_tag,
                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            _util_funcs.get('create_table_with_data', lambda *args: None)(adequacy_table_tag, adequacy_df)

def _interpret_silhouette_score(score: float) -> str:
    """실루엣 점수 해석"""
    if score > 0.7:
        return "Strong cluster structure"
    elif score > 0.5:
        return "Reasonable cluster structure"
    elif score > 0.25:
        return "Weak cluster structure"
    else:
        return "No substantial cluster structure"

def _interpret_communality(comm: float) -> str:
    """공통성 해석"""
    if comm > 0.8:
        return "Excellent"
    elif comm > 0.6:
        return "Good"
    elif comm > 0.4:
        return "Acceptable"
    else:
        return "Poor (consider removal)"

def _create_kmeans_visualizations(parent_tag: str, results: Dict[str, Any]):
    """K-Means 결과 시각화"""
    plot_func = _util_funcs.get('plot_to_dpg_texture')
    
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
        if 'centers' in results:
            ax.scatter(results['centers'][:, 0], results['centers'][:, 1], 
                      c='red', s=200, alpha=0.8, marker='x', linewidths=3,
                      label='Centroids')
        
        ax.set_xlabel(results['variables'][0])
        ax.set_ylabel(results['variables'][1])
        ax.set_title('Cluster Scatter Plot (First 2 Variables)')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        if 'centers' in results:
            ax.legend()
        
        tex_tag, w, h, img_bytes = plot_func(fig)
        if tex_tag:
            _texture_tags.append(tex_tag)
            dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
        plt.close(fig)

def _create_summary_info(results: Dict[str, Any]):
    """요약 정보 생성"""
    dpg.add_text(f"Method: {results['method']}", color=(255, 255, 0))
    dpg.add_text(f"DataFrame: {results['df_name']}")
    dpg.add_text(f"Variables: {', '.join(results['variables'])}")
    
    if 'n_clusters' in results:
        dpg.add_text(f"Number of clusters: {results['n_clusters']}")
    if 'n_factors' in results:
        dpg.add_text(f"Number of factors: {results['n_factors']}")
    
    if 'silhouette_avg' in results:
        dpg.add_text(f"Silhouette Score: {results['silhouette_avg']:.3f}")
    if 'explained_variance_ratio' in results:
        total_var = np.sum(results['explained_variance_ratio']) * 100
        dpg.add_text(f"Total Explained Variance: {total_var:.1f}%")
    
    if 'n_noise' in results:
        dpg.add_text(f"Noise points: {results['n_noise']}")

def _create_statistical_tables(results: Dict[str, Any]):
    """SAS 스타일의 통계 테이블 생성"""
    method = results['method']
    
    if method in ["K-Means", "Hierarchical", "DBSCAN"]:
        _create_clustering_tables(results)
    elif method == "Factor Analysis":
        _create_factor_analysis_tables(results)

def _create_clustering_tables(results: Dict[str, Any]):
    """군집분석 통계 테이블 생성"""
    
    # 1. 클러스터 요약 통계
    dpg.add_text("Cluster Summary Statistics", color=(255, 255, 0))
    
    # 클러스터별 빈도 테이블
    unique_labels, counts = np.unique(results['labels'], return_counts=True)
    percentages = (counts / len(results['labels'])) * 100
    
    cluster_summary_data = []
    for label, count, pct in zip(unique_labels, counts, percentages):
        cluster_name = f"Cluster {label}" if label != -1 else "Noise"
        cluster_summary_data.append({
            "Cluster": cluster_name,
            "Frequency": count,
            "Percentage": f"{pct:.1f}%"
        })
    
    cluster_summary_df = pd.DataFrame(cluster_summary_data)
    
    # 테이블 생성
    cluster_table_tag = dpg.generate_uuid()
    with dpg.table(header_row=True, resizable=True, tag=cluster_table_tag,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
        _util_funcs.get('create_table_with_data', lambda *args: None)(cluster_table_tag, cluster_summary_df)
    
    dpg.add_spacer(height=10)
    
    # 2. 변수별 클러스터 평균 (K-Means, Hierarchical만)
    if results['method'] in ["K-Means", "Hierarchical"]:
        dpg.add_text("Cluster Means by Variables", color=(255, 255, 0))
        
        X = results['X']
        labels = results['labels']
        
        # 클러스터별 평균 계산
        cluster_means_data = []
        for var in results['variables']:
            row_data = {"Variable": var}
            for cluster_id in sorted(unique_labels):
                if cluster_id != -1:  # 노이즈 제외
                    cluster_mask = labels == cluster_id
                    if cluster_mask.sum() > 0:
                        mean_val = X.loc[cluster_mask, var].mean()
                        row_data[f"Cluster {cluster_id}"] = f"{mean_val:.3f}"
            cluster_means_data.append(row_data)
        
        cluster_means_df = pd.DataFrame(cluster_means_data)
        
        means_table_tag = dpg.generate_uuid()
        with dpg.table(header_row=True, resizable=True, tag=means_table_tag,
                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            _util_funcs.get('create_table_with_data', lambda *args: None)(means_table_tag, cluster_means_df)
        
        dpg.add_spacer(height=10)
    
    # 3. 클러스터 품질 지표
    if 'silhouette_avg' in results or 'inertia' in results:
        dpg.add_text("Cluster Quality Metrics", color=(255, 255, 0))
        
        quality_data = []
        if 'silhouette_avg' in results:
            quality_data.append({
                "Metric": "Silhouette Score",
                "Value": f"{results['silhouette_avg']:.4f}",
                "Interpretation": _interpret_silhouette_score(results['silhouette_avg'])
            })
        
        if 'inertia' in results:
            quality_data.append({
                "Metric": "Within-cluster Sum of Squares (WCSS)",
                "Value": f"{results['inertia']:.2f}",
                "Interpretation": "Lower values indicate tighter clusters"
            })
        
        if results['method'] == "DBSCAN" and 'n_noise' in results:
            noise_pct = (results['n_noise'] / len(results['labels'])) * 100
            quality_data.append({
                "Metric": "Noise Points",
                "Value": f"{results['n_noise']} ({noise_pct:.1f}%)",
                "Interpretation": "Points not assigned to any cluster"
            })
        
        quality_df = pd.DataFrame(quality_data)
        
        quality_table_tag = dpg.generate_uuid()
        with dpg.table(header_row=True, resizable=True, tag=quality_table_tag,
                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            _util_funcs.get('create_table_with_data', lambda *args: None)(quality_table_tag, quality_df)

def _create_factor_analysis_tables(results: Dict[str, Any]):
    """요인분석 통계 테이블 생성"""
    
    # 1. 요인 적재값 테이블
    dpg.add_text("Factor Loadings Matrix", color=(255, 255, 0))
    
    loadings_df = pd.DataFrame(
        results['loadings'], 
        index=results['variables'],
        columns=[f'Factor {i+1}' for i in range(results['n_factors'])]
    )
    # 인덱스를 열로 변환
    loadings_display_df = loadings_df.reset_index()
    loadings_display_df = loadings_display_df.rename(columns={'index': 'Variable'})
    
    loadings_table_tag = dpg.generate_uuid()
    with dpg.table(header_row=True, resizable=True, tag=loadings_table_tag,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
        _util_funcs.get('create_table_with_data', lambda *args: None)(loadings_table_tag, loadings_display_df)
    
    dpg.add_spacer(height=10)
    
    # 2. 공통성 테이블
    dpg.add_text("Communalities", color=(255, 255, 0))
    
    communalities_data = []
    for i, var in enumerate(results['variables']):
        communalities_data.append({
            "Variable": var,
            "Communality": f"{results['communalities'][i]:.4f}",
            "Interpretation": _interpret_communality(results['communalities'][i])
        })
    
    communalities_df = pd.DataFrame(communalities_data)
    
    comm_table_tag = dpg.generate_uuid()
    with dpg.table(header_row=True, resizable=True, tag=comm_table_tag,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
        _util_funcs.get('create_table_with_data', lambda *args: None)(comm_table_tag, communalities_df)
    
    dpg.add_spacer(height=10)
    
    # 3. 설명된 분산 테이블
    dpg.add_text("Variance Explained by Factors", color=(255, 255, 0))
    
    variance_data = []
    cumulative_var = 0
    for i in range(results['n_factors']):
        factor_var = results['explained_variance_ratio'][i] * 100
        cumulative_var += factor_var
        variance_data.append({
            "Factor": f"Factor {i+1}",
            "Eigenvalue": f"{results['explained_variance'][i]:.4f}" if 'explained_variance' in results else "N/A",
            "Variance %": f"{factor_var:.2f}%",
            "Cumulative %": f"{cumulative_var:.2f}%"
        })
    
    variance_df = pd.DataFrame(variance_data)
    
    var_table_tag = dpg.generate_uuid()
    with dpg.table(header_row=True, resizable=True, tag=var_table_tag,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
        _util_funcs.get('create_table_with_data', lambda *args: None)(var_table_tag, variance_df)
    
    dpg.add_spacer(height=10)
    
    # 4. Kaiser-Meyer-Olkin (KMO) 측정 (간단한 버전)
    if 'eigenvalues' in results:
        dpg.add_text("Factor Analysis Adequacy", color=(255, 255, 0))
        
        # Kaiser Criterion 확인
        kaiser_factors = sum(1 for ev in results['eigenvalues'] if ev > 1.0)
        total_var_explained = sum(results['explained_variance_ratio']) * 100
        
        adequacy_data = [
            {
                "Measure": "Total Variance Explained",
                "Value": f"{total_var_explained:.1f}%",
                "Criterion": ">60% is adequate"
            },
            {
                "Measure": "Factors with Eigenvalue > 1",
                "Value": f"{kaiser_factors}",
                "Criterion": "Kaiser Criterion"
            },
            {
                "Measure": "Selected Factors",
                "Value": f"{results['n_factors']}",
                "Criterion": "User specified"
            }
        ]
        
        adequacy_df = pd.DataFrame(adequacy_data)
        
        adequacy_table_tag = dpg.generate_uuid()
        with dpg.table(header_row=True, resizable=True, tag=adequacy_table_tag,
                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            _util_funcs.get('create_table_with_data', lambda *args: None)(adequacy_table_tag, adequacy_df)

def _interpret_silhouette_score(score: float) -> str:
    """실루엣 점수 해석"""
    if score > 0.7:
        return "Strong cluster structure"
    elif score > 0.5:
        return "Reasonable cluster structure"
    elif score > 0.25:
        return "Weak cluster structure"
    else:
        return "No substantial cluster structure"

def _interpret_communality(comm: float) -> str:
    """공통성 해석"""
    if comm > 0.8:
        return "Excellent"
    elif comm > 0.6:
        return "Good"
    elif comm > 0.4:
        return "Acceptable"
    else:
        return "Poor (consider removal)"
    """K-Means 결과 시각화"""
    plot_func = _util_funcs.get('plot_to_dpg_texture')
    
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
        if 'centers' in results:
            ax.scatter(results['centers'][:, 0], results['centers'][:, 1], 
                      c='red', s=200, alpha=0.8, marker='x', linewidths=3,
                      label='Centroids')
        
        ax.set_xlabel(results['variables'][0])
        ax.set_ylabel(results['variables'][1])
        ax.set_title('Cluster Scatter Plot (First 2 Variables)')
        plt.colorbar(scatter, ax=ax, label='Cluster')
        if 'centers' in results:
            ax.legend()
        
        tex_tag, w, h, img_bytes = plot_func(fig)
        if tex_tag:
            _texture_tags.append(tex_tag)
            dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
        plt.close(fig)

def _create_hierarchical_visualizations(parent_tag: str, results: Dict[str, Any]):
    """계층적 군집분석 결과 시각화"""
    plot_func = _util_funcs.get('plot_to_dpg_texture')
    
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
    """DBSCAN 결과 시각화"""
    plot_func = _util_funcs.get('plot_to_dpg_texture')

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

def _create_factor_visualizations(parent_tag: str, results: Dict[str, Any]):
    """요인분석 결과 시각화"""
    plot_func = _util_funcs.get('plot_to_dpg_texture')
    
    # 1. Scree Plot
    if 'eigenvalues' in results:
        fig, ax = plt.subplots(figsize=(8, 5))
        factors_range = range(1, len(results['eigenvalues']) + 1)
        ax.plot(factors_range, results['eigenvalues'], 'bo-', linewidth=2, markersize=8)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Kaiser Criterion (>1)')
        ax.set_xlabel('Factor Number')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Scree Plot')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        tex_tag, w, h, img_bytes = plot_func(fig)
        if tex_tag:
            _texture_tags.append(tex_tag)
            dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
        plt.close(fig)
        dpg.add_separator(parent=parent_tag)
    
    # 2. Factor Loadings Heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    loadings_df = pd.DataFrame(
        results['loadings'], 
        index=results['variables'],
        columns=[f'Factor {i+1}' for i in range(results['n_factors'])]
    )
    
    sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, 
                fmt='.3f', ax=ax, cbar_kws={'label': 'Loading'})
    ax.set_title('Factor Loadings Matrix')
    ax.set_xlabel('Factors')
    ax.set_ylabel('Variables')
    
    tex_tag, w, h, img_bytes = plot_func(fig)
    if tex_tag:
        _texture_tags.append(tex_tag)
        dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
    plt.close(fig)
    dpg.add_separator(parent=parent_tag)
    
    # 3. Explained Variance Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    factor_names = [f'Factor {i+1}' for i in range(results['n_factors'])]
    bars = ax.bar(factor_names, results['explained_variance_ratio'] * 100)
    ax.set_ylabel('Explained Variance (%)')
    ax.set_title('Variance Explained by Each Factor')
    ax.set_ylim(0, max(results['explained_variance_ratio'] * 100) * 1.1)
    
    # 막대 위에 퍼센트 표시
    for bar, ratio in zip(bars, results['explained_variance_ratio']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio*100:.1f}%', ha='center', va='bottom')
    
    tex_tag, w, h, img_bytes = plot_func(fig)
    if tex_tag:
        _texture_tags.append(tex_tag)
        dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
    plt.close(fig)
    dpg.add_separator(parent=parent_tag)
    
    # 4. Communalities Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(results['variables']))
    bars = ax.barh(y_pos, results['communalities'])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results['variables'])
    ax.set_xlabel('Communality')
    ax.set_title('Communalities (Proportion of Variance Explained)')
    ax.set_xlim(0, 1)
    
    # 막대 끝에 값 표시
    for i, (bar, comm) in enumerate(zip(bars, results['communalities'])):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{comm:.3f}', ha='left', va='center')
    
    tex_tag, w, h, img_bytes = plot_func(fig)
    if tex_tag:
        _texture_tags.append(tex_tag)
        dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
    plt.close(fig)
    dpg.add_separator(parent=parent_tag)
    
    # 5. Factor Scores Scatter Plot (if 2 factors)
    if results['n_factors'] >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(results['factor_scores'][:, 0], results['factor_scores'][:, 1], 
                           alpha=0.6, s=50)
        ax.set_xlabel('Factor 1')
        ax.set_ylabel('Factor 2')
        ax.set_title('Factor Scores Plot (Factor 1 vs Factor 2)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        tex_tag, w, h, img_bytes = plot_func(fig)
        if tex_tag:
            _texture_tags.append(tex_tag)
            dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
        plt.close(fig)

def _export_results(results: Dict[str, Any], format: str):
    """결과 내보내기 - 분석 모듈로 위임"""
    import step_10_analysis as analysis
    analysis.export_results(results, format)

def reset_state():
    """시각화 모듈 상태 초기화"""
    global _viz_tab_counter, _texture_tags

    _viz_tab_counter = 0

    # 텍스처 정리
    for tag in _texture_tags:
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
    _texture_tags.clear()