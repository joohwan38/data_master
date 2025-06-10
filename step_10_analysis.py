# step_10_analysis.py - 분석 로직 담당

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import traceback
import datetime
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage
import warnings
warnings.filterwarnings('ignore')

# --- Module State ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None

def initialize(main_callbacks: dict):
    """분석 모듈 초기화"""
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()

def run_analysis(params: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    통합 분석 실행 함수
    
    Args:
        params: 분석 파라미터 딕셔너리
            - df_name: DataFrame 이름
            - df: DataFrame 객체
            - method: 분석 방법
            - variables: 선택된 변수 목록
            - options: 방법별 옵션
            - add_to_df: DataFrame에 결과 추가 여부
            - new_df_name: 새 DataFrame 이름
    
    Returns:
        (성공여부, 결과딕셔너리) 튜플
    """
    try:
        # Progress modal 표시
        _util_funcs['show_dpg_progress_modal']("Running Analysis", 
            f"Performing {params['method']} analysis...", 
            modal_tag="step10_progress_modal", 
            text_tag="step10_progress_text")
        
        # 분석 방법에 따른 분기
        if params['method'] in ["K-Means", "Hierarchical", "DBSCAN"]:
            results = _run_clustering_analysis(params)
        elif params['method'] == "Factor Analysis":
            results = _run_factor_analysis(params)
        elif params['method'] == "PCA":
            results = _run_pca_analysis(params)
        elif params['method'] in ["Pearson", "Spearman", "Kendall"]:
            results = _run_correlation_analysis(params)
        elif params['method'] in ["Linear", "Logistic"]:
            results = _run_regression_analysis(params)
        else:
            raise ValueError(f"Method '{params['method']}' is not implemented yet.")
            raise ValueError(f"Method '{params['method']}' is not implemented yet.")
        
        return True, results
        
    except Exception as e:
        error_msg = f"Error during {params['method']} analysis:\n{str(e)}"
        _util_funcs['_show_simple_modal_message']("Analysis Error", error_msg)
        traceback.print_exc()
        return False, None
    
    finally:
        _util_funcs['hide_dpg_progress_modal']("step10_progress_modal")

def _run_clustering_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    """군집분석 실행"""
    df = params['df']
    method = params['method']
    variables = params['variables']
    options = params['options']
    
    # 데이터 준비
    X = df[variables].dropna()
    
    if len(X) < 10:
        raise ValueError("Not enough data points for clustering (need at least 10)")
    
    # 표준화 처리
    if options.get('standardize', True):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # 분석 실행
    if method == "K-Means":
        results = _run_kmeans(X, X_scaled, options)
    elif method == "Hierarchical":
        results = _run_hierarchical(X, X_scaled, options)
    elif method == "DBSCAN":
        results = _run_dbscan(X, X_scaled, options)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # 공통 메타데이터 추가
    results.update({
        'method': method,
        'df_name': params['df_name'],
        'variables': variables,
        'X': X,
        'X_scaled': X_scaled
    })
    
    return results

def _run_regression(X: pd.DataFrame, y: pd.Series, feature_vars: List[str], 
                   target_var: str, method: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """회귀분석 실행"""
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report, confusion_matrix
    
    # 데이터 분할
    test_size = options.get('test_size', 20.0) / 100.0
    random_state = options.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 표준화
    if options.get('standardize', True):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
        scaler = None
    
    # 모델 학습
    if method == "Linear":
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # 예측
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # 성능 지표
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        metrics = {
            'R2_train': r2_train,
            'R2_test': r2_test,
            'RMSE_train': rmse_train,
            'RMSE_test': rmse_test
        }
        
    elif method == "Logistic":
        max_iter = options.get('max_iter', 1000)
        model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        model.fit(X_train_scaled, y_train)
        
        # 예측
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        y_pred_proba_test = model.predict_proba(X_test_scaled)
        
        # 성능 지표
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        
        metrics = {
            'Accuracy_train': acc_train,
            'Accuracy_test': acc_test,
            'Classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            'Confusion_matrix': confusion_matrix(y_test, y_pred_test)
        }
    
    # 계수/특성 중요도
    coefficients = pd.DataFrame({
        'Feature': feature_vars,
        'Coefficient': model.coef_[0] if hasattr(model, 'coef_') and len(model.coef_.shape) > 1 else model.coef_
    })
    
    results = {
        'model': model,
        'scaler': scaler,
        'coefficients': coefficients,
        'metrics': metrics,
        'predictions_train': y_pred_train,
        'predictions_test': y_pred_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train': X_train,
        'X_test': X_test
    }
    
    if method == "Logistic":
        results['prediction_probabilities'] = y_pred_proba_test
    
    return results

def _run_correlation_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    """상관분석 실행"""
    df = params['df']
    variables = params['variables']
    options = params['options']
    method = params['method'].lower()  # 'Pearson' -> 'pearson'
    
    if len(variables) < 2:
        raise ValueError("Correlation analysis requires at least 2 variables")
    
    # 데이터 준비
    X = df[variables].dropna()
    
    if len(X) < 10:
        raise ValueError("Not enough data points for correlation analysis (need at least 10)")
    
    # 상관분석 실행
    results = _run_correlation(X, method, options)
    
    # 메타데이터 추가
    results.update({
        'method': params['method'],
        'df_name': params['df_name'],
        'variables': variables,
        'X': X
    })
    
    return results

def _run_regression_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    """회귀분석 실행"""
    df = params['df']
    variables = params['variables']
    options = params['options']
    method = params['method']
    
    target_var = options.get('target_variable')
    if not target_var or target_var not in df.columns:
        raise ValueError("Target variable must be selected for regression analysis")
    
    if target_var in variables:
        variables = [v for v in variables if v != target_var]  # 타겟 변수 제외
    
    if len(variables) < 1:
        raise ValueError("At least one predictor variable is required")
    
    # 데이터 준비
    X = df[variables].dropna()
    y = df.loc[X.index, target_var]
    
    if len(X) < 20:
        raise ValueError("Not enough data points for regression (need at least 20)")
    
    # 회귀분석 실행
    results = _run_regression(X, y, variables, target_var, method, options)
    
    # 메타데이터 추가
    results.update({
        'method': method,
        'df_name': params['df_name'],
        'variables': variables,
        'target_variable': target_var,
        'X': X,
        'y': y
    })
    
    return results

def _run_pca(X: pd.DataFrame, X_scaled: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
    """PCA 실행"""
    n_components = options.get('n_components', 2)
    variance_threshold = options.get('variance_threshold', 85.0) / 100.0
    
    # 모든 성분으로 PCA 실행 (Scree plot용)
    pca_full = PCA()
    pca_full.fit(X_scaled)
    
    # 지정된 성분 수로 PCA 실행
    pca = PCA(n_components=n_components, random_state=42)
    component_scores = pca.fit_transform(X_scaled)
    
    # 분산 기준으로 최적 성분 수 찾기
    optimal_components = 1
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    for i, cum_var in enumerate(cumsum_var):
        if cum_var >= variance_threshold:
            optimal_components = i + 1
            break
    
    results = {
        'component_scores': component_scores,
        'components': pca.components_,  # 주성분 벡터
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'singular_values': pca.singular_values_,
        'n_components': n_components,
        'optimal_components': optimal_components,
        'variance_threshold': variance_threshold * 100
    }
    
    # Scree plot을 위한 전체 eigenvalue
    if options.get('scree', True):
        results['eigenvalues'] = pca_full.explained_variance_
        results['full_explained_variance_ratio'] = pca_full.explained_variance_ratio_
    
    # Biplot을 위한 추가 정보
    if options.get('biplot', True):
        # 변수 로딩 (주성분과 원변수의 상관계수)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        results['loadings'] = loadings
    
    return results

def _run_pca_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    """PCA 분석 실행"""
    df = params['df']
    variables = params['variables']
    options = params['options']
    
    if len(variables) < 2:
        raise ValueError("PCA requires at least 2 variables")
    
    # 데이터 준비
    X = df[variables].dropna()
    
    if len(X) < 10:
        raise ValueError("Not enough data points for PCA (need at least 10)")
    
    # 표준화
    if options.get('standardize', True):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # PCA 실행
    results = _run_pca(X, X_scaled, options)
    
    # 메타데이터 추가
    results.update({
        'method': "PCA",
        'df_name': params['df_name'],
        'variables': variables,
        'X': X,
        'X_scaled': X_scaled
    })
    
    return results

def _run_factor_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    """요인분석 실행"""
    df = params['df']
    variables = params['variables']
    options = params['options']
    
    if len(variables) < 3:
        raise ValueError("Factor analysis requires at least 3 variables")
    
    # 데이터 준비
    X = df[variables].dropna()
    
    if len(X) < 20:
        raise ValueError("Not enough data points for factor analysis (need at least 20)")
    
    # 표준화
    if options.get('standardize', True):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # Factor Analysis 실행
    results = _run_fa(X, X_scaled, options)
    
    # 메타데이터 추가
    results.update({
        'method': "Factor Analysis",
        'df_name': params['df_name'],
        'variables': variables,
        'X': X,
        'X_scaled': X_scaled
    })
    
    return results

def _run_kmeans(X: pd.DataFrame, X_scaled: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
    """K-Means 군집분석 실행"""
    n_clusters = options.get('n_clusters', 3)
    init = options.get('init', 'k-means++')
    max_iter = options.get('max_iter', 300)
    
    # K-Means 모델 학습
    kmeans = KMeans(
        n_clusters=n_clusters, 
        init=init, 
        max_iter=max_iter,
        n_init=10, 
        random_state=42
    )
    labels = kmeans.fit_predict(X_scaled)
    
    results = {
        'labels': labels,
        'centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'n_clusters': n_clusters
    }
    
    # 실루엣 점수 계산
    if n_clusters > 1:
        results['silhouette_avg'] = silhouette_score(X_scaled, labels)
        results['silhouette_samples'] = silhouette_samples(X_scaled, labels)
    
    # Elbow plot을 위한 데이터
    if options.get('elbow', True):
        inertias = []
        K_range = range(2, min(11, len(X)))
        for k in K_range:
            km = KMeans(
                n_clusters=k, 
                init=init, 
                max_iter=max_iter,
                n_init=10, 
                random_state=42
            )
            km.fit(X_scaled)
            inertias.append(km.inertia_)
        results['elbow_data'] = {'K': list(K_range), 'inertia': inertias}
    
    return results

def _run_hierarchical(X: pd.DataFrame, X_scaled: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
    """계층적 군집분석 실행"""
    n_clusters = options.get('n_clusters', 3)
    linkage_method = options.get('linkage', 'ward')
    metric = options.get('metric', 'euclidean')
    
    # 계층적 군집분석
    hc = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
        metric=metric
    )
    labels = hc.fit_predict(X_scaled)
    
    results = {
        'labels': labels,
        'n_clusters': n_clusters
    }
    
    # 실루엣 점수
    if n_clusters > 1:
        results['silhouette_avg'] = silhouette_score(X_scaled, labels)
    
    # Dendrogram 데이터
    if options.get('dendrogram', True):
        linkage_matrix = linkage(X_scaled, method=linkage_method, metric=metric)
        results['linkage_matrix'] = linkage_matrix
    
    return results

def _run_dbscan(X: pd.DataFrame, X_scaled: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
    """DBSCAN 군집분석 실행"""
    eps = options.get('eps', 0.5)
    min_samples = options.get('min_samples', 5)
    metric = options.get('metric', 'euclidean')
    
    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    results = {
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise
    }
    
    # 실루엣 점수 (노이즈 제외)
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 0:
            results['silhouette_avg'] = silhouette_score(X_scaled[mask], labels[mask])
    
    return results

def _run_fa(X: pd.DataFrame, X_scaled: np.ndarray, options: Dict[str, Any]) -> Dict[str, Any]:
    """Factor Analysis 실행"""
    n_factors = options.get('n_factors', 2)
    max_iter = options.get('max_iter', 1000)
    tol = options.get('tol', 0.01)
    
    # Factor Analysis 모델
    fa = FactorAnalysis(
        n_components=n_factors, 
        max_iter=max_iter, 
        tol=tol, 
        random_state=42
    )
    factor_scores = fa.fit_transform(X_scaled)
    
    # 요인 적재값 (Factor Loadings)
    loadings = fa.components_.T
    
    # 공통성 (Communalities) 계산
    communalities = np.sum(loadings**2, axis=1)
    
    # 설명된 분산 계산
    explained_variance = np.var(factor_scores, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    
    results = {
        'factor_scores': factor_scores,
        'loadings': loadings,
        'communalities': communalities,
        'explained_variance': explained_variance,
        'explained_variance_ratio': explained_variance_ratio,
        'n_factors': n_factors
    }
    
    # Scree plot을 위한 eigenvalue 계산
    if options.get('scree', True):
        # PCA를 이용해 eigenvalue 계산
        pca_temp = PCA()
        pca_temp.fit(X_scaled)
        eigenvalues = pca_temp.explained_variance_
        results['eigenvalues'] = eigenvalues
    
    return results

def add_clustering_results_to_dataframe(df: pd.DataFrame, results: Dict[str, Any], params: Dict[str, Any]):
    """군집분석 결과를 DataFrame에 추가"""
    X = results['X']
    labels = results['labels']
    
    # 클러스터 레이블을 원본 DataFrame에 추가
    cluster_col_name = f"cluster_{results['method'].lower().replace('-', '_')}"
    
    # X의 인덱스를 사용하여 원본 DataFrame에 레이블 추가
    df.loc[X.index, cluster_col_name] = labels
    
    # 새 DataFrame 생성 여부 확인
    new_df_name = params.get('new_df_name', '').strip()
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
            f"Cluster labels added to '{params['df_name']}' as column '{cluster_col_name}'.")

def add_factor_results_to_dataframe(df: pd.DataFrame, results: Dict[str, Any], params: Dict[str, Any]):
    """요인분석 결과를 DataFrame에 추가"""
    X = results['X']
    factor_scores = results['factor_scores']
    
    # 요인 점수를 원본 DataFrame에 추가
    for i in range(results['n_factors']):
        factor_col_name = f"factor_{i+1}"
        df.loc[X.index, factor_col_name] = factor_scores[:, i]
    
    # 새 DataFrame 생성 여부 확인
    new_df_name = params.get('new_df_name', '').strip()
    if new_df_name:
        new_df = df.copy()
        if _module_main_callbacks and 'step8_derivation_complete' in _module_main_callbacks:
            _module_main_callbacks['step8_derivation_complete'](new_df_name, new_df)
            _util_funcs['_show_simple_modal_message']("Success", 
                f"New DataFrame '{new_df_name}' created with factor scores.")
    else:
        _util_funcs['_show_simple_modal_message']("Success", 
            f"Factor scores added to '{params['df_name']}'.")

def add_pca_results_to_dataframe(df: pd.DataFrame, results: Dict[str, Any], params: Dict[str, Any]):
    """PCA 결과를 DataFrame에 추가"""
    X = results['X']
    component_scores = results['component_scores']
    
    # 주성분 점수를 원본 DataFrame에 추가
    for i in range(results['n_components']):
        component_col_name = f"PC{i+1}"
        df.loc[X.index, component_col_name] = component_scores[:, i]
    
    # 새 DataFrame 생성 여부 확인
    new_df_name = params.get('new_df_name', '').strip()
    if new_df_name:
        new_df = df.copy()
        if _module_main_callbacks and 'step8_derivation_complete' in _module_main_callbacks:
            _module_main_callbacks['step8_derivation_complete'](new_df_name, new_df)
            _util_funcs['_show_simple_modal_message']("Success", 
                f"New DataFrame '{new_df_name}' created with PCA component scores.")
    else:
        _util_funcs['_show_simple_modal_message']("Success", 
            f"PCA component scores added to '{params['df_name']}'.")

def export_results(results: Dict[str, Any], format: str):
    """결과 내보내기"""
    if format == 'excel':
        _export_to_excel(results)
    elif format == 'html':
        _export_to_html(results)

def _export_to_excel(results: Dict[str, Any]):
    """Excel로 내보내기"""
    try:
        import dearpygui.dearpygui as dpg
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
            if results['method'] == "Factor Analysis":
                summary_data = {
                    'Metric': ['Method', 'DataFrame', 'Variables', 'Number of Factors', 'Total Explained Variance'],
                    'Value': [results['method'], results['df_name'], 
                            ', '.join(results['variables']), results['n_factors'],
                            f"{np.sum(results['explained_variance_ratio'])*100:.1f}%"]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # 요인 적재값
                loadings_df = pd.DataFrame(
                    results['loadings'], 
                    index=results['variables'],
                    columns=[f'Factor_{i+1}' for i in range(results['n_factors'])]
                )
                loadings_df.to_excel(writer, sheet_name='Factor_Loadings')
                
                # 요인 점수
                factor_scores_df = results['X'].copy()
                for i in range(results['n_factors']):
                    factor_scores_df[f'Factor_{i+1}'] = results['factor_scores'][:, i]
                factor_scores_df.to_excel(writer, sheet_name='Factor_Scores')
                
            else:  # Clustering methods
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
    """HTML로 내보내기 (향후 구현)"""
    _util_funcs['_show_simple_modal_message']("Info", 
        "HTML export not implemented yet.")

def reset_state():
    """분석 모듈 상태 초기화"""
    pass  # 현재 분석 모듈은 상태가 없으므로 빈 함수