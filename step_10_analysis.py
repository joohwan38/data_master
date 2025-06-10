# step_10_analysis.py - 분석 로직 담당 (statsmodels 통합 버전)

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

# statsmodels 추가
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
        elif params['method'] == "ANOVA":
            results = _run_anova_analysis(params)
        elif params['method'] == "Time Series":
            results = _run_time_series_analysis(params)
        else:
            raise ValueError(f"Method '{params['method']}' is not implemented yet.")
        
        return True, results
        
    except Exception as e:
        error_msg = f"Error during {params['method']} analysis:\n{str(e)}"
        _util_funcs['_show_simple_modal_message']("Analysis Error", error_msg)
        traceback.print_exc()
        return False, None
    
    finally:
        _util_funcs['hide_dpg_progress_modal']("step10_progress_modal")

def _run_regression_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    """회귀분석 실행 (statsmodels 사용)"""
    df = params['df']
    variables = params['variables']
    options = params['options']
    method = params['method']
    
    target_var = options.get('target_variable')
    if not target_var or target_var not in df.columns:
        raise ValueError("Target variable must be selected for regression analysis")
    
    if target_var in variables:
        variables = [v for v in variables if v != target_var]
    
    if len(variables) < 1:
        raise ValueError("At least one predictor variable is required")
    
    # 데이터 준비
    data_for_model = df[variables + [target_var]].dropna()
    
    if len(data_for_model) < 20:
        raise ValueError("Not enough data points for regression (need at least 20)")
    
    X = data_for_model[variables]
    y = data_for_model[target_var]
    
    # statsmodels를 사용한 회귀분석
    if method == "Linear":
        results = _run_linear_regression_sm(X, y, variables, target_var, options)
    elif method == "Logistic":
        results = _run_logistic_regression_sm(X, y, variables, target_var, options)
    
    # 메타데이터 추가
    results.update({
        'method': method,
        'df_name': params['df_name'],
        'variables': variables,
        'target_variable': target_var,
        'X': X,
        'y': y,
        'data_for_model': data_for_model
    })
    
    return results

def _run_linear_regression_sm(X: pd.DataFrame, y: pd.Series, feature_vars: List[str], 
                            target_var: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """statsmodels를 사용한 선형 회귀분석"""
    from sklearn.model_selection import train_test_split
    
    # 데이터 분할
    test_size = options.get('test_size', 20.0) / 100.0
    random_state = options.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 상수항 추가
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    
    # 모델 적합
    model = sm.OLS(y_train, X_train_sm).fit()
    
    # 예측
    y_pred_train = model.predict(X_train_sm)
    y_pred_test = model.predict(X_test_sm)
    
    # 잔차
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    # 진단 통계량
    diagnostics = {}
    
    # VIF 계산 (다중공선성)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_train.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) 
                       for i in range(len(X_train.columns))]
    diagnostics['vif'] = vif_data
    
    # Breusch-Pagan 검정 (이분산성)
    bp_test = het_breuschpagan(residuals_train, X_train_sm)
    diagnostics['breusch_pagan'] = {
        'lm_statistic': bp_test[0],
        'lm_pvalue': bp_test[1],
        'f_statistic': bp_test[2],
        'f_pvalue': bp_test[3]
    }
    
    # Durbin-Watson 검정 (자기상관)
    dw_stat = durbin_watson(residuals_train)
    diagnostics['durbin_watson'] = dw_stat
    
    # 영향력 진단
    influence = model.get_influence()
    diagnostics['cooks_distance'] = influence.cooks_distance[0]
    diagnostics['leverage'] = influence.hat_matrix_diag
    
    results = {
        'model': model,
        'model_summary': model.summary(),
        'model_summary_text': str(model.summary()),
        'predictions_train': y_pred_train,
        'predictions_test': y_pred_test,
        'residuals_train': residuals_train,
        'residuals_test': residuals_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train': X_train,
        'X_test': X_test,
        'diagnostics': diagnostics,
        'coefficients': pd.DataFrame({
            'Feature': ['const'] + feature_vars,
            'Coefficient': model.params.values,
            'Std_Error': model.bse.values,
            't_value': model.tvalues.values,
            'p_value': model.pvalues.values,
            'CI_Lower': model.conf_int()[0].values,
            'CI_Upper': model.conf_int()[1].values
        })
    }
    
    return results

def _run_logistic_regression_sm(X: pd.DataFrame, y: pd.Series, feature_vars: List[str], 
                               target_var: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """statsmodels를 사용한 로지스틱 회귀분석"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    
    # 데이터 분할
    test_size = options.get('test_size', 20.0) / 100.0
    random_state = options.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 상수항 추가
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    
    # 모델 적합
    try:
        model = sm.Logit(y_train, X_train_sm).fit(method='bfgs', maxiter=1000)
    except:
        # 수렴 실패 시 다른 방법 시도
        model = sm.Logit(y_train, X_train_sm).fit(method='newton', maxiter=1000)
    
    # 예측 확률
    y_pred_proba_train = model.predict(X_train_sm)
    y_pred_proba_test = model.predict(X_test_sm)
    
    # 예측 클래스
    threshold = options.get('threshold', 0.5)
    y_pred_train = (y_pred_proba_train >= threshold).astype(int)
    y_pred_test = (y_pred_proba_test >= threshold).astype(int)
    
    # 성능 지표
    cm_test = confusion_matrix(y_test, y_pred_test)
    cr_test = classification_report(y_test, y_pred_test, output_dict=True)
    
    # 모델 진단
    diagnostics = {}
    
    # Pseudo R-squared
    diagnostics['pseudo_r2'] = {
        'McFadden': model.prsquared,
        'LLR_pvalue': model.llr_pvalue
    }
    
    # Odds Ratios
    odds_ratios = np.exp(model.params)
    
    results = {
        'model': model,
        'model_summary': model.summary(),
        'model_summary_text': str(model.summary()),
        'predictions_train': y_pred_train,
        'predictions_test': y_pred_test,
        'prediction_probabilities_train': y_pred_proba_train,
        'prediction_probabilities_test': y_pred_proba_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train': X_train,
        'X_test': X_test,
        'confusion_matrix': cm_test,
        'classification_report': cr_test,
        'diagnostics': diagnostics,
        'coefficients': pd.DataFrame({
            'Feature': ['const'] + feature_vars,
            'Coefficient': model.params.values,
            'Std_Error': model.bse.values,
            'z_value': model.tvalues.values,
            'p_value': model.pvalues.values,
            'Odds_Ratio': odds_ratios.values,
            'CI_Lower': model.conf_int()[0].values,
            'CI_Upper': model.conf_int()[1].values
        })
    }
    
    return results

def _run_anova_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    """ANOVA 분석 실행"""
    df = params['df']
    variables = params['variables']
    options = params['options']
    
    target_var = options.get('target_variable')
    if not target_var or target_var not in df.columns:
        raise ValueError("Target variable must be selected for ANOVA")
    
    if target_var in variables:
        variables = [v for v in variables if v != target_var]
    
    # 카테고리 변수와 연속 변수 구분
    cat_vars = []
    cont_vars = []
    
    for var in variables:
        if df[var].dtype in ['object', 'category'] or df[var].nunique() < 10:
            cat_vars.append(var)
        else:
            cont_vars.append(var)
    
    if len(cat_vars) == 0:
        raise ValueError("At least one categorical variable is required for ANOVA")
    
    # 데이터 준비
    data_for_anova = df[variables + [target_var]].dropna()
    
    # ANOVA 모델 생성
    if len(cat_vars) == 1 and len(cont_vars) == 0:
        # One-way ANOVA
        formula = f"{target_var} ~ C({cat_vars[0]})"
        anova_type = "One-way ANOVA"
    else:
        # Multi-way ANOVA or ANCOVA
        formula_parts = []
        for cat_var in cat_vars:
            formula_parts.append(f"C({cat_var})")
        for cont_var in cont_vars:
            formula_parts.append(cont_var)
        
        # 상호작용 효과 포함 옵션
        if options.get('include_interactions', False) and len(cat_vars) > 1:
            formula_parts.append("*".join([f"C({cv})" for cv in cat_vars[:2]]))
        
        formula = f"{target_var} ~ " + " + ".join(formula_parts)
        anova_type = "ANCOVA" if cont_vars else "Multi-way ANOVA"
    
    # 모델 적합
    model = smf.ols(formula, data=data_for_anova).fit()
    
    # ANOVA 테이블
    anova_table = anova_lm(model, typ=2)
    
    # 사후 검정 준비 (Tukey HSD 등)
    post_hoc_results = {}
    if len(cat_vars) == 1:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        groups = data_for_anova[cat_vars[0]]
        tukey = pairwise_tukeyhsd(data_for_anova[target_var], groups)
        post_hoc_results['tukey_hsd'] = tukey
    
    results = {
        'anova_type': anova_type,
        'model': model,
        'model_summary': model.summary(),
        'model_summary_text': str(model.summary()),
        'anova_table': anova_table,
        'formula': formula,
        'categorical_vars': cat_vars,
        'continuous_vars': cont_vars,
        'post_hoc': post_hoc_results,
        'residuals': model.resid,
        'fitted_values': model.fittedvalues
    }
    
    return results

def _run_time_series_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    """시계열 분석 실행"""
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    df = params['df']
    variables = params['variables']
    options = params['options']
    
    if len(variables) != 1:
        raise ValueError("Time series analysis requires exactly one variable")
    
    ts_var = variables[0]
    
    # 시계열 데이터 준비
    ts_data = df[ts_var].dropna()
    
    if len(ts_data) < 50:
        raise ValueError("Not enough data points for time series analysis (need at least 50)")
    
    # 기본 통계
    ts_stats = {
        'mean': ts_data.mean(),
        'std': ts_data.std(),
        'min': ts_data.min(),
        'max': ts_data.max(),
        'skewness': ts_data.skew(),
        'kurtosis': ts_data.kurt()
    }
    
    # ADF 검정 (정상성)
    adf_result = adfuller(ts_data)
    adf_stats = {
        'adf_statistic': adf_result[0],
        'p_value': adf_result[1],
        'used_lag': adf_result[2],
        'nobs': adf_result[3],
        'critical_values': adf_result[4],
        'is_stationary': adf_result[1] < 0.05
    }
    
    # ACF/PACF
    nlags = min(40, len(ts_data) // 4)
    acf_values = acf(ts_data, nlags=nlags)
    pacf_values = pacf(ts_data, nlags=nlags)
    
    # 계절성 분해
    decomposition = None
    if options.get('seasonal_decompose', True) and len(ts_data) >= 24:
        period = options.get('period', 12)
        try:
            decomposition = seasonal_decompose(ts_data, model='additive', period=period)
        except:
            decomposition = None
    
    results = {
        'time_series': ts_data,
        'statistics': ts_stats,
        'adf_test': adf_stats,
        'acf': acf_values,
        'pacf': pacf_values,
        'decomposition': decomposition,
        'variable': ts_var
    }
    
    return results

# 기존 함수들은 그대로 유지...
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

def _run_correlation_analysis(params: Dict[str, Any]) -> Dict[str, Any]:
    """상관분석 실행"""
    df = params['df']
    variables = params['variables']
    options = params['options']
    method = params['method'].lower()
    
    if len(variables) < 2:
        raise ValueError("Correlation analysis requires at least 2 variables")
    
    # 데이터 준비
    X = df[variables].dropna()
    
    if len(X) < 10:
        raise ValueError("Not enough data points for correlation analysis (need at least 10)")
    
    # 상관분석 실행
    corr_matrix = X.corr(method=method)
    
    # p-values 계산
    from scipy.stats import pearsonr, spearmanr, kendalltau
    
    n_vars = len(variables)
    p_values = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                p_values[i, j] = 0.0
            else:
                x = X.iloc[:, i]
                y = X.iloc[:, j]
                
                if method == 'pearson':
                    _, p = pearsonr(x, y)
                elif method == 'spearman':
                    _, p = spearmanr(x, y)
                elif method == 'kendall':
                    _, p = kendalltau(x, y)
                
                p_values[i, j] = p
    
    p_value_matrix = pd.DataFrame(p_values, index=variables, columns=variables)
    
    results = {
        'method': params['method'],
        'df_name': params['df_name'],
        'variables': variables,
        'X': X,
        'correlation_matrix': corr_matrix,
        'p_value_matrix': p_value_matrix,
        'significance_level': options.get('alpha', 0.05)
    }
    
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

# 기존의 _run_kmeans, _run_hierarchical, _run_dbscan, _run_pca, _run_fa 함수들은 그대로 유지...
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

# DataFrame에 결과 추가하는 함수들도 그대로 유지...
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

def add_correlation_results_to_dataframe(df: pd.DataFrame, results: Dict[str, Any], params: Dict[str, Any]):
    """상관분석 결과를 DataFrame에 추가 (필요시)"""
    # 상관분석은 일반적으로 DataFrame에 추가할 내용이 없음
    pass

def add_regression_results_to_dataframe(df: pd.DataFrame, results: Dict[str, Any], params: Dict[str, Any]):
    """회귀분석 결과를 DataFrame에 추가"""
    data_for_model = results['data_for_model']
    predictions = results.get('predictions_test')
    residuals = results.get('residuals_test')
    
    if predictions is not None:
        # 예측값과 잔차를 DataFrame에 추가
        pred_col_name = f"pred_{results['method'].lower()}_{results['target_variable']}"
        resid_col_name = f"resid_{results['method'].lower()}_{results['target_variable']}"
        
        # test set의 인덱스에 예측값 추가
        test_indices = results['y_test'].index
        df.loc[test_indices, pred_col_name] = predictions
        
        if residuals is not None:
            df.loc[test_indices, resid_col_name] = residuals
    
    # 새 DataFrame 생성 여부 확인
    new_df_name = params.get('new_df_name', '').strip()
    if new_df_name:
        new_df = df.copy()
        if _module_main_callbacks and 'step8_derivation_complete' in _module_main_callbacks:
            _module_main_callbacks['step8_derivation_complete'](new_df_name, new_df)
            _util_funcs['_show_simple_modal_message']("Success", 
                f"New DataFrame '{new_df_name}' created with regression results.")
    else:
        _util_funcs['_show_simple_modal_message']("Success", 
            f"Regression results added to '{params['df_name']}'.")

# Export 함수들은 그대로 유지...
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
    """Excel 파일 저장 (statsmodels 결과 포함)"""
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 요약 정보
            method = results['method']
            
            if method in ["Linear", "Logistic"]:
                # 회귀분석 결과
                summary_data = {
                    'Metric': ['Method', 'DataFrame', 'Target Variable', 'Predictors'],
                    'Value': [method, results['df_name'], results['target_variable'], 
                            ', '.join(results['variables'])]
                }
                
                if method == "Linear":
                    summary_data['Metric'].extend(['R-squared', 'Adj. R-squared', 'F-statistic', 'AIC', 'BIC'])
                    model = results['model']
                    summary_data['Value'].extend([
                        f"{model.rsquared:.4f}",
                        f"{model.rsquared_adj:.4f}",
                        f"{model.fvalue:.4f} (p={model.f_pvalue:.4e})",
                        f"{model.aic:.2f}",
                        f"{model.bic:.2f}"
                    ])
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # 계수 테이블
                results['coefficients'].to_excel(writer, sheet_name='Coefficients', index=False)
                
                # 모델 요약 (텍스트)
                summary_text_df = pd.DataFrame({'Model Summary': [results['model_summary_text']]})
                summary_text_df.to_excel(writer, sheet_name='Model_Summary', index=False)
                
                # 진단 통계량
                if 'diagnostics' in results:
                    diag = results['diagnostics']
                    if 'vif' in diag:
                        diag['vif'].to_excel(writer, sheet_name='VIF', index=False)
                    
                    if 'breusch_pagan' in diag:
                        bp_df = pd.DataFrame([diag['breusch_pagan']])
                        bp_df.to_excel(writer, sheet_name='Breusch_Pagan_Test', index=False)
                
            elif method == "ANOVA":
                # ANOVA 결과
                summary_data = {
                    'Metric': ['ANOVA Type', 'DataFrame', 'Target Variable', 'Formula'],
                    'Value': [results['anova_type'], results['df_name'], 
                            results.get('target_variable', 'N/A'), results['formula']]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # ANOVA 테이블
                results['anova_table'].to_excel(writer, sheet_name='ANOVA_Table')
                
                # 모델 요약
                summary_text_df = pd.DataFrame({'Model Summary': [results['model_summary_text']]})
                summary_text_df.to_excel(writer, sheet_name='Model_Summary', index=False)
                
            else:
                # 기존의 다른 분석 방법들
                if method == "Factor Analysis":
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
                    
                else:  # Clustering methods
                    summary_data = {
                        'Metric': ['Method', 'DataFrame', 'Variables', 'Number of Clusters'],
                        'Value': [results['method'], results['df_name'], 
                                ', '.join(results['variables']), results['n_clusters']]
                    }
                    
                    if 'silhouette_avg' in results:
                        summary_data['Metric'].append('Silhouette Score')
                        summary_data['Value'].append(f"{results['silhouette_avg']:.3f}")
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        _util_funcs['_show_simple_modal_message']("Success", 
            f"Results exported to:\n{filepath}")
            
    except Exception as e:
        _util_funcs['_show_simple_modal_message']("Export Error", 
            f"Failed to export: {str(e)}")

def _export_to_html(results: Dict[str, Any]):
    """HTML로 내보내기"""
    _util_funcs['_show_simple_modal_message']("Info", 
        "HTML export with interactive plots coming soon.")

def reset_state():
    """분석 모듈 상태 초기화"""
    pass