# step_10_visualization.py - 시각화 담당 (표 형태 AI 분석 추가)

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import warnings
warnings.filterwarnings('ignore')
import functools
import utils

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
    elif method == "ANOVA":
        _create_anova_tables(results)
    elif method == "Time Series":
        _create_time_series_tables(results)

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
    
    _create_table_with_ai("Cluster Summary", pd.DataFrame(summary_data), 
                         f"{results['method']} Cluster Distribution")
    
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
        _create_table_with_ai("Quality Metrics", pd.DataFrame(quality_data),
                             f"{results['method']} Quality Assessment")

def _create_regression_tables(results: Dict[str, Any]):
    """회귀분석 테이블 (statsmodels 결과 포함)"""
    method = results['method']
    
    # statsmodels summary가 있는 경우 전체 요약 표시
    if 'model_summary_text' in results:
        dpg.add_text("Model Summary (statsmodels)", color=(255, 255, 0))
        
        # 요약 텍스트를 고정폭 폰트로 표시
        summary_lines = results['model_summary_text'].split('\n')
        for line in summary_lines[:50]:  # 처음 50줄만 표시
            dpg.add_text(line, bullet=False)
        
        # AI 분석 버튼 추가
        ai_button_tag = dpg.generate_uuid()
        summary_text_for_ai = results['model_summary_text']
        
        def analyze_summary():
            import ollama_analyzer
            try:
                # 텍스트 기반 분석을 위한 프롬프트
                prompt = f"""다음은 {method} Regression 분석의 statsmodels 요약입니다:

{summary_text_for_ai}

이 회귀분석 결과를 해석하고 다음 사항을 한국어로 설명해주세요:
1. 모델의 전반적인 적합도 (R-squared, F-statistic)
2. 유의미한 변수들과 그 영향력
3. 모델 진단 결과 (잔차, 다중공선성 등)
4. 실무적 시사점과 주의사항
"""
                
                # AI 로그에 스트리밍
                if 'add_ai_log' in _module_main_callbacks:
                    first_chunk = True
                    for chunk in ollama_analyzer.analyze_text_with_ollama(prompt, f"{method} Regression Summary"):
                        if first_chunk:
                            _module_main_callbacks['add_ai_log'](chunk, f"{method} Regression Summary", mode="stream_start_entry")
                            first_chunk = False
                        else:
                            _module_main_callbacks['add_ai_log'](chunk, "", mode="stream_chunk_append")
                    
                    _module_main_callbacks['add_ai_log']("\n(분석 완료)", "", mode="stream_chunk_append")
                
            except Exception as e:
                if 'add_ai_log' in _module_main_callbacks:
                    _module_main_callbacks['add_ai_log'](f"Error: {str(e)}", f"{method} Regression Summary", mode="new_log_entry")
        
        dpg.add_button(label="💡 Analyze Summary with AI", tag=ai_button_tag,
                      callback=analyze_summary)
        dpg.add_separator()
    
    # 계수 테이블
    dpg.add_text("Model Coefficients", color=(255, 255, 0))
    coef_df = results['coefficients'].copy()
    _create_table_with_ai("Model Coefficients", coef_df,
                         f"{method} Regression Coefficients")
    
    # 성능 지표
    if 'diagnostics' in results:
        dpg.add_text("Diagnostic Tests", color=(255, 255, 0))
        diag = results['diagnostics']
        
        # VIF (다중공선성)
        if 'vif' in diag:
            _create_table_with_ai("Variance Inflation Factors", diag['vif'],
                                "Multicollinearity Assessment")
        
        # Breusch-Pagan (이분산성)
        if 'breusch_pagan' in diag:
            bp_data = pd.DataFrame([{
                'Test': 'Breusch-Pagan',
                'LM Statistic': f"{diag['breusch_pagan']['lm_statistic']:.4f}",
                'LM p-value': f"{diag['breusch_pagan']['lm_pvalue']:.4f}",
                'F Statistic': f"{diag['breusch_pagan']['f_statistic']:.4f}",
                'F p-value': f"{diag['breusch_pagan']['f_pvalue']:.4f}",
                'Heteroscedasticity': 'Present' if diag['breusch_pagan']['lm_pvalue'] < 0.05 else 'Not detected'
            }])
            _create_table_with_ai("Heteroscedasticity Test", bp_data,
                                "Breusch-Pagan Test Results")
        
        # Durbin-Watson (자기상관)
        if 'durbin_watson' in diag:
            dw_interpretation = "Positive autocorrelation" if diag['durbin_watson'] < 1.5 else \
                               "No autocorrelation" if 1.5 <= diag['durbin_watson'] <= 2.5 else \
                               "Negative autocorrelation"
            dw_data = pd.DataFrame([{
                'Test': 'Durbin-Watson',
                'Statistic': f"{diag['durbin_watson']:.4f}",
                'Interpretation': dw_interpretation
            }])
            _create_table_with_ai("Autocorrelation Test", dw_data,
                                "Durbin-Watson Test Results")

def _create_anova_tables(results: Dict[str, Any]):
    """ANOVA 테이블"""
    dpg.add_text(f"{results['anova_type']} Results", color=(255, 255, 0))
    
    # ANOVA 테이블
    anova_table = results['anova_table'].copy()
    anova_table = anova_table.round(4)
    _create_table_with_ai("ANOVA Table", anova_table, results['anova_type'])
    
    # Post-hoc 검정 결과
    if 'post_hoc' in results and results['post_hoc']:
        if 'tukey_hsd' in results['post_hoc']:
            dpg.add_text("Tukey HSD Post-hoc Test", color=(255, 255, 0))
            tukey = results['post_hoc']['tukey_hsd']
            # Tukey 결과를 DataFrame으로 변환
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], 
                                  columns=tukey._results_table.data[0])
            _create_table_with_ai("Tukey HSD Results", tukey_df, 
                                "Multiple Comparisons")

def _create_time_series_tables(results: Dict[str, Any]):
    """시계열 분석 테이블"""
    dpg.add_text("Time Series Analysis Results", color=(255, 255, 0))
    
    # 기본 통계
    stats_df = pd.DataFrame([results['statistics']])
    _create_table_with_ai("Time Series Statistics", stats_df,
                         "Basic Time Series Properties")
    
    # ADF 검정 결과
    adf = results['adf_test']
    adf_df = pd.DataFrame([{
        'ADF Statistic': f"{adf['adf_statistic']:.4f}",
        'p-value': f"{adf['p_value']:.4f}",
        'Used Lag': adf['used_lag'],
        'Critical Value (1%)': f"{adf['critical_values']['1%']:.4f}",
        'Critical Value (5%)': f"{adf['critical_values']['5%']:.4f}",
        'Critical Value (10%)': f"{adf['critical_values']['10%']:.4f}",
        'Stationary': 'Yes' if adf['is_stationary'] else 'No'
    }])
    _create_table_with_ai("Augmented Dickey-Fuller Test", adf_df,
                         "Stationarity Test Results")

def _create_table_with_ai(title: str, df: pd.DataFrame, context: str = ""):
    """테이블 생성 with AI 분석 버튼"""
    table_tag = dpg.generate_uuid()
    with dpg.table(header_row=True, resizable=True, tag=table_tag,
                   borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
        if _util_funcs and 'create_table_with_data' in _util_funcs:
            _util_funcs['create_table_with_data'](table_tag, df)
    
    # AI 분석 버튼 추가
    if df is not None and not df.empty:
        ai_button_tag = dpg.generate_uuid()
        
        def analyze_table():
            # DataFrame을 텍스트로 변환
            table_text = df.to_string()
            
            import ollama_analyzer
            try:
                # 텍스트 기반 분석을 위한 프롬프트
                prompt = f"""다음은 '{title}' 테이블입니다:

{table_text}

Context: {context}

이 테이블의 주요 내용을 분석하고 한국어로 설명해주세요:
1. 핵심 발견사항
2. 통계적 의미
3. 실무적 시사점
"""
                
                # AI 로그에 스트리밍
                if 'add_ai_log' in _module_main_callbacks:
                    first_chunk = True
                    for chunk in ollama_analyzer.analyze_text_with_ollama(prompt, f"Table: {title}"):
                        if first_chunk:
                            _module_main_callbacks['add_ai_log'](chunk, f"Table: {title}", mode="stream_start_entry")
                            first_chunk = False
                        else:
                            _module_main_callbacks['add_ai_log'](chunk, "", mode="stream_chunk_append")
                    
                    _module_main_callbacks['add_ai_log']("\n(분석 완료)", "", mode="stream_chunk_append")
                
            except Exception as e:
                if 'add_ai_log' in _module_main_callbacks:
                    _module_main_callbacks['add_ai_log'](f"Error: {str(e)}", f"Table: {title}", mode="new_log_entry")
        
        dpg.add_button(label=f"💡 Analyze {title}", tag=ai_button_tag,
                      callback=analyze_table, small=True)
    
    dpg.add_spacer(height=10)

def _create_factor_tables(results: Dict[str, Any]):
    """요인분석 테이블"""
    # 로딩 행렬
    dpg.add_text("Factor Loadings", color=(255, 255, 0))
    loadings_df = pd.DataFrame(
        results['loadings'], 
        index=results['variables'],
        columns=[f'Factor {i+1}' for i in range(results['n_factors'])]
    ).reset_index().rename(columns={'index': 'Variable'})
    _create_table_with_ai("Factor Loadings", loadings_df, "Factor Analysis Loadings Matrix")
    
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
    _create_table_with_ai("Variance Explained", pd.DataFrame(variance_data),
                         "Factor Analysis Variance Decomposition")

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
    _create_table_with_ai("Principal Components", pd.DataFrame(variance_data),
                         "PCA Variance Decomposition")

def _create_correlation_tables(results: Dict[str, Any]):
    """상관분석 테이블"""
    dpg.add_text("Correlation Matrix", color=(255, 255, 0))
    
    # 상관계수 행렬
    corr_matrix = results['correlation_matrix']
    corr_display = corr_matrix.reset_index()
    _create_table_with_ai("Correlation Matrix", corr_display,
                         f"{results['method']} Correlation Analysis")
    
    # p-value 행렬
    if 'p_value_matrix' in results:
        dpg.add_text("P-values", color=(255, 255, 0))
        pval_display = results['p_value_matrix'].reset_index()
        _create_table_with_ai("P-value Matrix", pval_display,
                            "Statistical Significance of Correlations")

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
    elif method == "ANOVA":
        _create_anova_plots(parent_tag, results, plot_func)
    elif method == "Time Series":
        _create_time_series_plots(parent_tag, results, plot_func)

def _create_anova_plots(parent_tag: str, results: Dict[str, Any], plot_func):
    """ANOVA 시각화"""
    # 잔차 플롯
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    residuals = results['residuals']
    fitted = results['fitted_values']
    
    # Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Histogram of Residuals')
    
    # Scale-Location plot
    standardized_residuals = residuals / np.std(residuals)
    axes[1, 1].scatter(fitted, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('√|Standardized Residuals|')
    axes[1, 1].set_title('Scale-Location Plot')
    
    plt.tight_layout()
    _add_plot(fig, plot_func, parent_tag, "ANOVA Diagnostic Plots")

def _create_time_series_plots(parent_tag: str, results: Dict[str, Any], plot_func):
    """시계열 분석 시각화"""
    ts_data = results['time_series']
    
    # 시계열 플롯
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts_data.index, ts_data.values)
    ax.set_xlabel('Time')
    ax.set_ylabel(results['variable'])
    ax.set_title('Time Series Plot')
    ax.grid(True, alpha=0.3)
    _add_plot(fig, plot_func, parent_tag, "Time Series Plot")
    
    # ACF/PACF 플롯
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(ts_data, lags=min(40, len(ts_data)//4), ax=axes[0])
    plot_pacf(ts_data, lags=min(40, len(ts_data)//4), ax=axes[1])
    plt.tight_layout()
    _add_plot(fig, plot_func, parent_tag, "ACF/PACF Plots")
    
    # 계절성 분해
    if results.get('decomposition'):
        decomp = results['decomposition']
        fig, axes = plt.subplots(4, 1, figsize=(10, 10))
        
        ts_data.plot(ax=axes[0])
        axes[0].set_title('Original Time Series')
        
        decomp.trend.plot(ax=axes[1])
        axes[1].set_title('Trend Component')
        
        decomp.seasonal.plot(ax=axes[2])
        axes[2].set_title('Seasonal Component')
        
        decomp.resid.plot(ax=axes[3])
        axes[3].set_title('Residual Component')
        
        plt.tight_layout()
        _add_plot(fig, plot_func, parent_tag, "Seasonal Decomposition")

# 기존 플롯 함수들은 그대로 유지...
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
        _add_plot(fig, plot_func, parent_tag, "Elbow Plot")
    
    # Cluster Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels, counts = np.unique(results['labels'], return_counts=True)
    ax.bar(unique_labels, counts, color=plt.cm.viridis(np.linspace(0, 1, len(unique_labels))))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')
    ax.set_title('Cluster Distribution')
    _add_plot(fig, plot_func, parent_tag, "Cluster Distribution")

def _create_hierarchical_plots(parent_tag: str, results: Dict[str, Any], plot_func):
    """계층적 군집분석 시각화"""
    # Dendrogram
    if 'linkage_matrix' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(results['linkage_matrix'], ax=ax)
        ax.set_title('Dendrogram')
        _add_plot(fig, plot_func, parent_tag, "Dendrogram")

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
    _add_plot(fig, plot_func, parent_tag, "DBSCAN Cluster Distribution")

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
        _add_plot(fig, plot_func, parent_tag, "Scree Plot")
    
    # Loadings Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    loadings_df = pd.DataFrame(
        results['loadings'], 
        index=results['variables'],
        columns=[f'Factor {i+1}' for i in range(results['n_factors'])]
    )
    sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('Factor Loadings')
    _add_plot(fig, plot_func, parent_tag, "Factor Loadings Heatmap")

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
        _add_plot(fig, plot_func, parent_tag, "Scree Plot")
    
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
        _add_plot(fig, plot_func, parent_tag, "PCA Scores Plot")

def _create_correlation_plots(parent_tag: str, results: Dict[str, Any], plot_func):
    """상관분석 시각화"""
    # Correlation Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = results['correlation_matrix']
    
    # Significance mask
    if 'p_value_matrix' in results:
        alpha = results.get('significance_level', 0.05)
        mask = results['p_value_matrix'] > alpha
        
        # Annotate with stars for significance
        annot_data = corr_matrix.copy()
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix.columns)):
                if i != j:
                    p_val = results['p_value_matrix'].iloc[i, j]
                    if p_val < 0.001:
                        annot_data.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}***"
                    elif p_val < 0.01:
                        annot_data.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}**"
                    elif p_val < 0.05:
                        annot_data.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}*"
                    else:
                        annot_data.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}"
                else:
                    annot_data.iloc[i, j] = "1.00"
        
        sns.heatmap(corr_matrix, annot=annot_data, fmt='', cmap='RdBu_r', center=0, 
                    ax=ax, square=True)
    else:
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=ax, 
                    square=True, fmt='.3f')
    
    ax.set_title(f'{results["method"]} Correlation Matrix')
    _add_plot(fig, plot_func, parent_tag, f"{results['method']} Correlation Heatmap")

def _create_regression_plots(parent_tag: str, results: Dict[str, Any], plot_func):
    """회귀분석 시각화"""
    method = results['method']
    
    if method == "Linear":
        # 진단 플롯 (4개 패널)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Residuals vs Fitted
        axes[0, 0].scatter(results['predictions_test'], results['residuals_test'], alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        
        # 2. Q-Q plot
        from scipy import stats
        stats.probplot(results['residuals_test'], dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        
        # 3. Scale-Location
        standardized_residuals = results['residuals_test'] / np.std(results['residuals_test'])
        axes[1, 0].scatter(results['predictions_test'], np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
        axes[1, 0].set_xlabel('Fitted Values')
        axes[1, 0].set_ylabel('√|Standardized Residuals|')
        axes[1, 0].set_title('Scale-Location')
        
        # 4. Residuals vs Leverage
        if 'diagnostics' in results and 'leverage' in results['diagnostics']:
            leverage = results['diagnostics']['leverage'][:len(results['residuals_test'])]
            axes[1, 1].scatter(leverage, standardized_residuals[:len(leverage)], alpha=0.6)
            axes[1, 1].set_xlabel('Leverage')
            axes[1, 1].set_ylabel('Standardized Residuals')
            axes[1, 1].set_title('Residuals vs Leverage')
        else:
            # Actual vs Predicted as alternative
            axes[1, 1].scatter(results['y_test'], results['predictions_test'], alpha=0.6)
            axes[1, 1].plot([results['y_test'].min(), results['y_test'].max()], 
                           [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
            axes[1, 1].set_xlabel('Actual Values')
            axes[1, 1].set_ylabel('Predicted Values')
            axes[1, 1].set_title('Actual vs Predicted')
        
        plt.tight_layout()
        _add_plot(fig, plot_func, parent_tag, "Regression Diagnostic Plots")
        
    elif method == "Logistic":
        # ROC Curve
        from sklearn.metrics import roc_curve, auc
        
        if 'prediction_probabilities_test' in results:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            fpr, tpr, _ = roc_curve(results['y_test'], results['prediction_probabilities_test'])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic')
            ax.legend(loc="lower right")
            _add_plot(fig, plot_func, parent_tag, "ROC Curve")
        
        # Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        _add_plot(fig, plot_func, parent_tag, "Confusion Matrix")
    
    # Feature Importance (계수 플롯)
    fig, ax = plt.subplots(figsize=(8, 6))
    coef_df = results['coefficients'].copy()
    
    # 상수항 제외
    if 'const' in coef_df['Feature'].values:
        coef_df = coef_df[coef_df['Feature'] != 'const']
    
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient')
    
    colors = ['red' if x < 0 else 'blue' for x in coef_df['Coefficient']]
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Feature Coefficients')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    _add_plot(fig, plot_func, parent_tag, "Feature Coefficients")

def _add_plot(fig, plot_func, parent_tag, chart_name="Chart"):
    """플롯을 DPG에 추가하는 헬퍼 함수 (AI 분석 버튼 포함)"""
    tex_tag, w, h, img_bytes = plot_func(fig)
    if tex_tag:
        _texture_tags.append(tex_tag)
        dpg.add_image(tex_tag, parent=parent_tag, width=w, height=h)
        
        # AI 분석 버튼 추가
        if img_bytes:
            ai_button_tag = dpg.generate_uuid()
            action_callback = functools.partial(
                utils.confirm_and_run_ai_analysis,
                img_bytes, chart_name, ai_button_tag, _module_main_callbacks
            )
            dpg.add_button(label=f"💡 Analyze {chart_name}", tag=ai_button_tag,
                          callback=lambda s, a, u: action_callback())
    
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