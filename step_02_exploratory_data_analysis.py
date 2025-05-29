# step_02_exploratory_data_analysis.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from scipy import stats
import traceback
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# UI 태그 정의
TAG_EDA_GROUP = "step2_eda_group"
TAG_EDA_MAIN_TAB_BAR = "step2_eda_main_tab_bar"

# SVA 태그
TAG_SVA_TAB = "step2_sva_tab"
TAG_SVA_FILTER_STRENGTH_RADIO = "step2_sva_filter_strength_radio"
TAG_SVA_GROUP_BY_TARGET_CHECKBOX = "step2_sva_group_by_target_checkbox"
TAG_SVA_RUN_BUTTON = "step2_sva_run_button"
TAG_SVA_RESULTS_CHILD_WINDOW = "step2_sva_results_child_window"
TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX = "sva_var_section_"
TAG_SVA_PROGRESS_MODAL = "sva_progress_modal"
TAG_SVA_PROGRESS_TEXT = "sva_progress_text"
TAG_SVA_GROUPED_PLOT_TYPE_RADIO = "step2_sva_grouped_plot_type_radio"

REUSABLE_SVA_ALERT_MODAL_TAG = "reusable_sva_alert_modal_unique_tag"
REUSABLE_SVA_ALERT_TEXT_TAG = "reusable_sva_alert_text_unique_tag"

# MVA 태그
TAG_MVA_TAB = "step2_mva_tab"
TAG_MVA_SUB_TAB_BAR = "step2_mva_sub_tab_bar"
TAG_MVA_CORR_TAB = "step2_mva_corr_tab"
TAG_MVA_CORR_RUN_BUTTON = "step2_mva_corr_run_button"
TAG_MVA_CORR_RESULTS_GROUP = "step2_mva_corr_results_group"
TAG_MVA_PAIRPLOT_TAB = "step2_mva_pairplot_tab"
TAG_MVA_PAIRPLOT_VAR_SELECTOR = "step2_mva_pairplot_var_selector"
TAG_MVA_PAIRPLOT_HUE_COMBO = "step2_mva_pairplot_hue_combo"
TAG_MVA_PAIRPLOT_RUN_BUTTON = "step2_mva_pairplot_run_button"
TAG_MVA_PAIRPLOT_RESULTS_GROUP = "step2_mva_pairplot_results_group"
TAG_MVA_TARGET_TAB = "step2_mva_target_tab"
TAG_MVA_TARGET_INFO_TEXT = "step2_mva_target_info_text"
TAG_MVA_TARGET_FEATURE_COMBO = "step2_mva_target_feature_combo"
TAG_MVA_TARGET_RUN_BUTTON = "step2_mva_target_run_button"
TAG_MVA_TARGET_RESULTS_GROUP = "step2_mva_target_results_group"
TAG_MVA_CAT_EDA_TAB = "step2_mva_cat_eda_tab"
TAG_MVA_CAT_EDA_VAR_SELECTOR = "step2_mva_cat_eda_var_selector"
TAG_MVA_CAT_EDA_RUN_BUTTON = "step2_mva_cat_eda_run_button"
TAG_MVA_CAT_EDA_RESULTS_GROUP = "step2_mva_cat_eda_results_group"
TAG_MVA_CORR_OUTPUT_GROUP = "step2_mva_corr_output_group"

# Outlier 태그
TAG_OUTLIER_TAB = "step2_outlier_tab"
TAG_OUTLIER_METHOD_RADIO = "step2_outlier_method_radio"
TAG_OUTLIER_CAPPING_CONTROLS_GROUP = "step2_outlier_capping_controls_group"
TAG_OUTLIER_CAPPING_LOWER_PERCENTILE = "step2_outlier_capping_lower_percentile"
TAG_OUTLIER_CAPPING_UPPER_PERCENTILE = "step2_outlier_capping_upper_percentile"
TAG_OUTLIER_CAPPING_VAR_SELECTOR = "step2_outlier_capping_var_selector"
TAG_OUTLIER_IF_CONTROLS_GROUP = "step2_outlier_if_controls_group"
TAG_OUTLIER_IF_VAR_SELECTOR = "step2_outlier_if_var_selector"
TAG_OUTLIER_APPLY_BUTTON = "step2_outlier_apply_button"
TAG_OUTLIER_RESET_TO_AFTER_STEP1_BUTTON = "step2_outlier_reset_to_after_step1_button"
TAG_OUTLIER_RESULTS_TEXT = "step2_outlier_results_text"
TAG_OUTLIER_STATUS_TEXT = "step2_outlier_status_text"
TAG_MVA_OUTLIER_RESULTS_TEXT = "step2_mva_outlier_results_text"

# 전역 변수
_main_app_callbacks_eda: Dict[str, Any] = {}
_util_funcs_eda: Dict[str, Any] = {}

def _show_alert_modal(title: str, message: str):
    """경고 모달 표시"""
    if not dpg.is_dearpygui_running():
        print(f"Alert - {title}: {message}")
        return

    viewport_width = dpg.get_viewport_width()
    viewport_height = dpg.get_viewport_height()
    modal_width = 450
    modal_pos_x = (viewport_width - modal_width) // 2
    modal_pos_y = viewport_height // 3

    if not dpg.does_item_exist(REUSABLE_SVA_ALERT_MODAL_TAG):
        with dpg.window(label="Alert", modal=True, show=False, 
                       tag=REUSABLE_SVA_ALERT_MODAL_TAG,
                       no_close=True, pos=[modal_pos_x, modal_pos_y], 
                       width=modal_width, autosize=True,
                       no_saved_settings=True):
            dpg.add_text("", tag=REUSABLE_SVA_ALERT_TEXT_TAG, wrap=modal_width - 30)
            dpg.add_spacer(height=15)
            with dpg.group(horizontal=True):
                button_width = 100
                spacer_width = max(0, (modal_width - button_width - dpg.get_style_item_spacing()[0] * 2) / 2)
                dpg.add_spacer(width=spacer_width)
                dpg.add_button(label="OK", width=button_width, 
                             user_data=REUSABLE_SVA_ALERT_MODAL_TAG,
                             callback=lambda s, a, u: dpg.configure_item(u, show=False))
    
    dpg.configure_item(REUSABLE_SVA_ALERT_MODAL_TAG, label=title, show=True, 
                      pos=[modal_pos_x, modal_pos_y])
    dpg.set_value(REUSABLE_SVA_ALERT_TEXT_TAG, message)

def _calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V 계산"""
    if x is None or y is None or x.empty or y.empty:
        return 0.0
    
    try:
        temp_df = pd.DataFrame({'x': x, 'y': y}).dropna()
        if temp_df.empty or temp_df['x'].nunique() < 1 or temp_df['y'].nunique() < 1:
            return 0.0
        
        confusion_matrix = pd.crosstab(temp_df['x'], temp_df['y'])
        if confusion_matrix.empty or confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
            return 0.0
        
        chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
        n = confusion_matrix.sum().sum()
        if n == 0:
            return 0.0
        
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1 if n > 1 else 1))
        rcorr = r - (((r - 1)**2) / (n - 1 if n > 1 else 1) if r > 1 else 0)
        kcorr = k - (((k - 1)**2) / (n - 1 if n > 1 else 1) if k > 1 else 0)
        
        denominator = min((kcorr - 1 if kcorr > 1 else 0), (rcorr - 1 if rcorr > 1 else 0))
        if denominator == 0:
            return 0.0
        
        return np.sqrt(phi2corr / denominator)
    except Exception:
        return 0.0

def _get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """숫자형 컬럼 목록 반환"""
    if df is None:
        return []
    return df.select_dtypes(include=np.number).columns.tolist()

def _get_categorical_cols(df: pd.DataFrame, max_unique_for_cat: int = 20, 
                         main_callbacks: Optional[Dict] = None) -> List[str]:
    """범주형 컬럼 목록 반환"""
    if df is None:
        return []
    
    cat_cols = []
    s1_types = {}
    
    if main_callbacks and 'get_column_analysis_types' in main_callbacks:
        s1_types = main_callbacks['get_column_analysis_types']()
        if not isinstance(s1_types, dict):
            s1_types = {}
    
    for col in df.columns:
        s1_type = s1_types.get(col, "")
        is_s1_cat = "Categorical" in s1_type or "Text" in s1_type or "Binary" in s1_type
        
        if is_s1_cat:
            if df[col].nunique(dropna=False) <= max_unique_for_cat * 1.5:
                cat_cols.append(col)
        elif df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col].dtype):
            if df[col].nunique(dropna=False) <= max_unique_for_cat:
                cat_cols.append(col)
    
    # 숫자형이지만 고유값이 적은 컬럼도 범주형으로 처리
    for col in df.select_dtypes(include=np.number).columns:
        s1_type = s1_types.get(col, "")
        is_s1_strong_numeric = "Numeric" in s1_type and "Binary" not in s1_type
        
        if not is_s1_strong_numeric and df[col].nunique(dropna=False) <= max_unique_for_cat and col not in cat_cols:
            cat_cols.append(col)
    
    return list(set(cat_cols))

def _get_top_n_correlated_with_target(df: pd.DataFrame, target_var: str, 
                                     numeric_cols: List[str], top_n: int = 20) -> List[str]:
    """타겟 변수와 상관관계가 높은 상위 n개 변수 반환"""
    if df is None or target_var not in df.columns or not numeric_cols:
        return []
    
    correlations = {}
    target_series = df[target_var]
    
    if not pd.api.types.is_numeric_dtype(target_series.dtype):
        print(f"Warning: Target variable '{target_var}' is not numeric.")
        return numeric_cols[:top_n]
    
    for col in numeric_cols:
        if col == target_var:
            continue
        try:
            temp_df = df[[target_var, col]].dropna()
            if len(temp_df) < 2:
                correlations[col] = 0
                continue
            corr_val = temp_df[target_var].corr(temp_df[col])
            correlations[col] = abs(corr_val if pd.notna(corr_val) else 0)
        except Exception:
            correlations[col] = 0
    
    sorted_vars = sorted(correlations.keys(), key=lambda k: correlations[k], reverse=True)
    return sorted_vars[:top_n]

def _get_filtered_variables(df: pd.DataFrame, filter_strength: str,
                           main_callbacks: dict, target_var: str = None) -> Tuple[List[str], str]:
    """필터 조건에 따른 변수 목록 반환"""
    if df is None or df.empty:
        return [], filter_strength
    
    analysis_types = main_callbacks.get('get_column_analysis_types', lambda: {})()
    if not analysis_types or not isinstance(analysis_types, dict):
        analysis_types = {col: str(df[col].dtype) for col in df.columns}
    
    # 텍스트 타입 필터링
    cols_after_text_filter = []
    for col_name in df.columns:
        col_type = analysis_types.get(col_name, str(df[col_name].dtype))
        if isinstance(col_type, str) and any(keyword in col_type for keyword in ["Text (", "Potentially Sensitive"]):
            continue
        cols_after_text_filter.append(col_name)
    
    print(f"Text filter: {len(df.columns) - len(cols_after_text_filter)} columns excluded, {len(cols_after_text_filter)} remain.")
    
    if filter_strength == "None (All variables)":
        return cols_after_text_filter, filter_strength
    
    # Weak filter
    weakly_filtered_cols = []
    for col_name in cols_after_text_filter:
        series = df[col_name]
        col_type = analysis_types.get(col_name, str(series.dtype))
        
        # 단일값 제외
        if series.nunique(dropna=False) <= 1:
            continue
        
        # 이진 숫자 제외
        is_binary_numeric = "Numeric (Binary)" in col_type
        if not is_binary_numeric and "Numeric" in col_type:
            unique_vals = series.dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                is_binary_numeric = True
        
        if is_binary_numeric:
            continue
        
        weakly_filtered_cols.append(col_name)
    
    if filter_strength == "Weak (Exclude obvious non-analytical)":
        return weakly_filtered_cols, filter_strength
    
    # Medium/Strong filter
    if not target_var:
        msg = f"Cannot apply '{filter_strength}' filter: Target variable required."
        _show_alert_modal("Target Variable Needed", msg)
        return [], filter_strength
    
    # 관련성 점수 계산
    numeric_cols_for_ranking = [col for col in weakly_filtered_cols 
                                if "Numeric" in analysis_types.get(col, "") 
                                and "Binary" not in analysis_types.get(col, "")]
    
    if not numeric_cols_for_ranking:
        msg = f"Cannot apply '{filter_strength}' filter: No suitable numeric variables."
        _show_alert_modal("Filter Condition Not Met", msg)
        return [], filter_strength
    
    target_type = main_callbacks['get_selected_target_variable_type']()
    relevance_scores = _calculate_relevance_scores(
        df, target_var, target_type, numeric_cols_for_ranking
    )
    
    if not relevance_scores:
        msg = f"Cannot apply '{filter_strength}' filter: No relevant variables found."
        _show_alert_modal("Filter Condition Not Met", msg)
        return [], filter_strength
    
    if filter_strength == "Strong (Top 5-10 relevant)":
        return relevance_scores[:10], filter_strength
    elif filter_strength == "Medium (Top 11-20 relevant)":
        return relevance_scores[:20], filter_strength
    
    return [], filter_strength

def _calculate_relevance_scores(df: pd.DataFrame, target_var: str, 
                               target_type: str, cols: List[str]) -> List[str]:
    """변수들의 타겟 대비 관련성 점수 계산"""
    scores = []
    
    for col in cols:
        if col == target_var:
            continue
        
        score = 0.0
        try:
            temp_df = pd.concat([df[col], df[target_var]], axis=1).dropna()
            if len(temp_df) < 20:
                continue
            
            if target_type == "Categorical":
                # ANOVA F-value
                groups = [temp_df[col][temp_df[target_var] == cat] 
                         for cat in temp_df[target_var].unique()]
                valid_groups = [g for g in groups if len(g) >= 2]
                if len(valid_groups) >= 2:
                    f_val, _ = stats.f_oneway(*valid_groups)
                    score = f_val if pd.notna(f_val) and np.isfinite(f_val) else 0.0
            elif target_type == "Continuous":
                # Pearson correlation
                if pd.api.types.is_numeric_dtype(temp_df[target_var].dtype):
                    score = abs(temp_df[col].corr(temp_df[target_var]))
                    if not (pd.notna(score) and np.isfinite(score)):
                        score = 0.0
        except Exception:
            score = 0.0
        
        if score > 1e-3:
            scores.append((col, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in scores]

def _create_sva_basic_stats_table(parent_tag: str, series: pd.Series, 
                                 util_funcs: dict, analysis_type_override: str = None):
    """SVA 기본 통계 테이블 생성"""
    dpg.add_text("Basic Statistics", parent=parent_tag)
    stats_data = []
    
    # 통계 타입 결정
    s1_type = ""
    if 'main_app_callbacks' in util_funcs:
        s1_types = util_funcs['main_app_callbacks'].get('get_column_analysis_types', lambda: {})()
        s1_type = s1_types.get(series.name, str(series.dtype))
    
    is_binary_numeric = "Numeric (Binary)" in s1_type
    treat_as_categorical = (
        is_binary_numeric or
        analysis_type_override == "ForceCategoricalForBinaryNumeric" or
        "Categorical" in s1_type or
        "Text (" in s1_type or
        "Potentially Sensitive" in s1_type or
        series.nunique(dropna=False) < 5
    )
    
    is_numeric = pd.api.types.is_numeric_dtype(series.dtype)
    
    # 기본 통계
    stats_data.extend([
        {'Statistic': 'Count', 'Value': str(series.count())},
        {'Statistic': 'Missing', 'Value': str(series.isnull().sum())},
        {'Statistic': 'Missing %', 'Value': f"{series.isnull().mean()*100:.2f}%"},
        {'Statistic': 'Unique (Actual)', 'Value': str(series.nunique(dropna=False))},
        {'Statistic': 'Unique (Valid)', 'Value': str(series.nunique())}
    ])
    
    if is_numeric and not treat_as_categorical:
        # 숫자형 통계
        desc = series.describe()
        for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            if stat in desc.index:
                val = desc[stat]
                formatted = f"{val:.3f}" if isinstance(val, (int, float, np.number)) else str(val)
                stats_data.append({'Statistic': stat, 'Value': formatted})
        
        # 왜도, 첨도
        series_clean = series.dropna()
        if len(series_clean) >= 3:
            try:
                skew = series_clean.skew()
                kurt = series_clean.kurtosis()
                stats_data.append({'Statistic': 'Skewness', 
                                 'Value': f"{skew:.3f}" if pd.notna(skew) else "N/A"})
                stats_data.append({'Statistic': 'Kurtosis', 
                                 'Value': f"{kurt:.3f}" if pd.notna(kurt) else "N/A"})
            except:
                pass
    else:
        # 범주형 통계
        value_counts = series.value_counts(dropna=False).nlargest(5)
        if not value_counts.empty:
            stats_data.extend([
                {'Statistic': 'Mode (Top1)', 'Value': str(value_counts.index[0])},
                {'Statistic': 'Mode Freq (Top1)', 'Value': str(value_counts.iloc[0])}
            ])
            if len(value_counts) > 1:
                stats_data.extend([
                    {'Statistic': 'Mode (Top2)', 'Value': str(value_counts.index[1])},
                    {'Statistic': 'Mode Freq (Top2)', 'Value': str(value_counts.iloc[1])}
                ])
    
    # 테이블 생성
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        table_tag = dpg.generate_uuid()
        table_height = min(280, len(stats_df) * 22 + 40)
        
        with dpg.table(header_row=True, tag=table_tag, parent=parent_tag,
                      borders_innerH=True, borders_outerH=True, 
                      borders_innerV=True, borders_outerV=True,
                      resizable=True, policy=dpg.mvTable_SizingStretchProp, 
                      height=int(table_height), scrollY=True):
            util_funcs['create_table_with_data'](table_tag, stats_df, 
                                               parent_df_for_widths=stats_df)

def _create_sva_advanced_relations_table(parent_tag: str, series: pd.Series, 
                                       full_df: pd.DataFrame, util_funcs: dict, 
                                       col_width: int):
    """SVA 고급 관계 테이블 생성"""
    # 정규성 검정
    normality_data = []
    s1_type = ""
    if 'main_app_callbacks' in util_funcs:
        s1_types = util_funcs['main_app_callbacks'].get('get_column_analysis_types', lambda: {})()
        s1_type = s1_types.get(series.name, str(series.dtype))
    
    is_numeric_for_test = (
        pd.api.types.is_numeric_dtype(series.dtype) and
        "Binary" not in s1_type and
        series.nunique(dropna=False) >= 5
    )
    
    if is_numeric_for_test:
        series_clean = series.dropna()
        if 3 <= len(series_clean) < 5000:
            try:
                stat_sw, p_sw = stats.shapiro(series_clean.astype(float, errors='ignore'))
                normality_data.extend([
                    {'Test': 'Shapiro-Wilk W', 'Value': f"{stat_sw:.3f}"},
                    {'Test': 'p-value (SW)', 'Value': f"{p_sw:.3f}"},
                    {'Test': 'Normality (α=0.05)', 
                     'Value': "Likely Normal" if p_sw > 0.05 else "Likely Not Normal"}
                ])
            except:
                normality_data.append({'Test': 'Shapiro-Wilk', 'Value': 'Error'})
        else:
            normality_data.append({'Test': 'Shapiro-Wilk', 'Value': 'N/A (size)'})
    else:
        normality_data.append({'Test': 'Shapiro-Wilk', 'Value': 'N/A (not applicable)'})
    
    # 정규성 테이블 표시
    if normality_data:
        dpg.add_text("Normality Test:", parent=parent_tag)
        norm_df = pd.DataFrame(normality_data)
        norm_table_tag = dpg.generate_uuid()
        norm_height = min(120, len(norm_df) * 22 + 30)
        
        with dpg.table(header_row=True, tag=norm_table_tag, parent=parent_tag,
                      borders_innerH=True, borders_outerH=True, 
                      borders_innerV=True, borders_outerV=True,
                      resizable=True, policy=dpg.mvTable_SizingStretchProp, 
                      height=int(norm_height), scrollY=True):
            util_funcs['create_table_with_data'](norm_table_tag, norm_df, 
                                               parent_df_for_widths=norm_df)
        dpg.add_spacer(height=5, parent=parent_tag)
    
    # 상관 변수 표시
    dpg.add_text("Top Related Variables:", parent=parent_tag)
    related_vars_data = _get_top_correlated_vars(full_df, series.name, top_n=5)
    
    if related_vars_data:
        if len(related_vars_data) == 1 and 'Info' in related_vars_data[0]:
            dpg.add_text(related_vars_data[0]['Info'], parent=parent_tag, 
                        wrap=col_width-10 if col_width > 20 else 200)
        else:
            actual_data = [item for item in related_vars_data if 'Info' not in item]
            if actual_data:
                related_df = pd.DataFrame(actual_data)
                rel_table_tag = dpg.generate_uuid()
                rel_height = min(150, len(related_df) * 22 + 40)
                
                with dpg.table(header_row=True, tag=rel_table_tag, parent=parent_tag,
                              borders_innerH=True, borders_outerH=True, 
                              borders_innerV=True, borders_outerV=True,
                              resizable=True, policy=dpg.mvTable_SizingStretchProp, 
                              height=int(rel_height), scrollY=True):
                    util_funcs['create_table_with_data'](rel_table_tag, related_df, 
                                                       parent_df_for_widths=related_df)
            else:
                dpg.add_text("No specific related variables found.", parent=parent_tag, 
                           wrap=col_width-10)
    else:
        dpg.add_text("No correlation/association data available.", parent=parent_tag, 
                   wrap=col_width-10)

def _get_top_correlated_vars(df: pd.DataFrame, current_var: str, 
                            top_n: int = 5) -> List[Dict[str, str]]:
    """현재 변수와 상관관계가 높은 변수들 반환"""
    if df is None or current_var not in df.columns or len(df.columns) < 2:
        return [{'Info': 'Not enough data or variables'}]
    
    correlations = []
    results = []
    current_series = df[current_var].copy()
    
    if pd.api.types.is_numeric_dtype(current_series.dtype):
        # 숫자형 상관관계
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if col == current_var:
                continue
            try:
                temp_df = df[[current_var, col]].dropna()
                if len(temp_df) < 2:
                    continue
                corr_val = temp_df[current_var].corr(temp_df[col])
                if pd.notna(corr_val) and abs(corr_val) > 0.01:
                    correlations.append((col, corr_val, "Pearson Corr"))
            except:
                pass
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, val, metric in correlations[:top_n]:
            results.append({'Variable': name, 'Metric': metric, 'Value': f"{val:.3f}"})
    
    # 범주형 상관관계
    elif current_series.nunique(dropna=False) < 30 or \
         pd.api.types.is_categorical_dtype(current_series.dtype) or \
         current_series.dtype == 'object':
        candidate_cols = [col for col in df.columns if col != current_var and
                         (df[col].nunique(dropna=False) < 30 or 
                          pd.api.types.is_categorical_dtype(df[col].dtype) or 
                          df[col].dtype == 'object')]
        
        for col in candidate_cols:
            try:
                cramers_v = _calculate_cramers_v(current_series, df[col])
                if pd.notna(cramers_v) and cramers_v > 0.01:
                    correlations.append((col, cramers_v, "Cramér's V"))
            except:
                pass
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, val, metric in correlations[:top_n]:
            results.append({'Variable': name, 'Metric': metric, 'Value': f"{val:.3f}"})
    
    if not results:
        return [{'Info': 'No significant relations found.'}]
    
    return results

def _create_single_var_plot(parent_tag: str, series: pd.Series, 
                          group_by_target: pd.Series = None,
                          analysis_override: str = None,
                          grouped_plot_pref: str = "KDE",
                          util_funcs: dict = None):
    """단일 변수 플롯 생성"""
    plot_height = 290
    plot_label = f"Distribution: {series.name}"
    
    s1_type = "Unknown"
    if util_funcs and 'main_app_callbacks' in util_funcs:
        s1_types = util_funcs['main_app_callbacks'].get('get_column_analysis_types', lambda: {})()
        s1_type = s1_types.get(series.name, str(series.dtype))
    
    is_grouped = group_by_target is not None and series.name != group_by_target.name
    if is_grouped:
        plot_label += f" (Grouped by {group_by_target.name})"
    
    plot_tag = dpg.generate_uuid()
    with dpg.plot(label=plot_label, height=plot_height, width=-1, 
                 parent=parent_tag, tag=plot_tag):
        xaxis_tag = dpg.add_plot_axis(dpg.mvXAxis, label=series.name, 
                                     lock_min=False, lock_max=False, auto_fit=True)
        yaxis_tag = dpg.generate_uuid()
        dpg.add_plot_axis(dpg.mvYAxis, label="Density / Frequency", tag=yaxis_tag, 
                         lock_min=False, lock_max=False, auto_fit=True)
        
        legend_tag = dpg.add_plot_legend(parent=plot_tag, horizontal=False, 
                                       location=dpg.mvPlot_Location_NorthEast, outside=False)
        
        series_clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(series_clean) < 2:
            dpg.add_text("Not enough valid data points for plot.", 
                        parent=yaxis_tag, color=(255, 200, 0))
            if dpg.does_item_exist(legend_tag):
                dpg.delete_item(legend_tag)
            return
        
        # 플롯 타입 결정
        is_binary_numeric = "Numeric (Binary)" in s1_type
        treat_as_categorical = (
            is_binary_numeric or
            analysis_override == "ForceCategoricalForBinaryNumeric" or
            "Categorical" in s1_type or
            "Text (" in s1_type or
            "Potentially Sensitive" in s1_type or
            series_clean.nunique() < 5
        )
        
        if pd.api.types.is_numeric_dtype(series.dtype) and not treat_as_categorical:
            _create_numeric_plot(series_clean, group_by_target, grouped_plot_pref, 
                               xaxis_tag, yaxis_tag, is_grouped)
        else:
            _create_categorical_plot(series_clean, group_by_target, xaxis_tag, 
                                   yaxis_tag, legend_tag, is_grouped)

def _create_numeric_plot(series_clean: pd.Series, group_by_target: pd.Series,
                        plot_pref: str, xaxis_tag, yaxis_tag, is_grouped: bool):
    """숫자형 변수 플롯 생성"""
    if is_grouped and group_by_target is not None:
        # 그룹별 플롯
        unique_groups = sorted(group_by_target.dropna().unique())
        colors = [(0, 110, 255, 200), (255, 120, 0, 200), (0, 170, 0, 200),
                 (200, 0, 0, 200), (150, 50, 200, 200), (255, 192, 203, 200),
                 (128, 0, 128, 200)]
        
        for idx, group in enumerate(unique_groups):
            group_data = series_clean[group_by_target == group].dropna()
            if len(group_data) < 2:
                continue
            
            color = colors[idx % len(colors)]
            
            try:
                if plot_pref == "KDE" and group_data.nunique() >= 2:
                    kde = stats.gaussian_kde(group_data.astype(float))
                    kde_min, kde_max = group_data.min(), group_data.max()
                    padding = (kde_max - kde_min) * 0.05 if (kde_max - kde_min) > 1e-6 else 0.1
                    x_vals = np.linspace(kde_min - padding, kde_max + padding, 100)
                    y_vals = kde(x_vals)
                    dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), 
                                      label=f"KDE (T={group})", parent=yaxis_tag)
                elif plot_pref == "Histogram":
                    dpg.add_histogram_series(group_data.tolist(), 
                                           label=f"Hist (T={group})",
                                           density=True, bins=-1, parent=yaxis_tag, weight=0.9)
            except Exception as e:
                print(f"Plot error for group {group}: {e}")
    else:
        # 단일 플롯
        if series_clean.nunique() < 2:
            dpg.add_text("No variance in data.", parent=yaxis_tag, color=(255, 200, 0))
            return
        
        try:
            dpg.add_histogram_series(series_clean.tolist(), label="Histogram", 
                                   density=True, bins=-1, parent=yaxis_tag, weight=1.0)
        except:
            pass
        
        try:
            if series_clean.nunique() >= 2:
                kde = stats.gaussian_kde(series_clean.astype(float))
                kde_min, kde_max = series_clean.min(), series_clean.max()
                padding = (kde_max - kde_min) * 0.05 if (kde_max - kde_min) > 1e-6 else 0.1
                x_vals = np.linspace(kde_min - padding, kde_max + padding, 150)
                y_vals = kde(x_vals)
                dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), 
                                  label="KDE", parent=yaxis_tag)
        except:
            pass

def _create_categorical_plot(series_clean: pd.Series, group_by_target: pd.Series,
                           xaxis_tag, yaxis_tag, legend_tag, is_grouped: bool):
    """범주형 변수 플롯 생성"""
    top_n_categories = 10
    
    if is_grouped and group_by_target is not None:
        dpg.show_item(legend_tag)
        
        unique_groups = sorted(group_by_target.dropna().unique())
        value_counts_overall = series_clean.value_counts(dropna=False).nlargest(top_n_categories)
        categories = [str(c) for c in value_counts_overall.index.tolist()]
        
        num_groups = len(unique_groups)
        bar_width_total = 0.8
        bar_width_single = bar_width_total / num_groups if num_groups > 0 else bar_width_total
        x_positions = np.arange(len(categories))
        
        for i, group in enumerate(unique_groups):
            group_series = series_clean[group_by_target == group]
            group_counts = group_series.value_counts(dropna=False)
            
            y_values = [group_counts.get(cat, 0) for cat in value_counts_overall.index]
            x_pos = x_positions - (bar_width_total / 2) + (i * bar_width_single) + (bar_width_single / 2)
            
            dpg.add_bar_series(x_pos.tolist(), y_values, weight=bar_width_single, 
                             label=f"{group}", parent=yaxis_tag)
        
        if categories:
            dpg.set_axis_ticks(xaxis_tag, tuple(zip(categories, x_positions.tolist())))
    else:
        dpg.hide_item(legend_tag)
        
        value_counts = series_clean.value_counts(dropna=False).nlargest(top_n_categories)
        x_pos = list(range(len(value_counts)))
        labels = [str(val) for val in value_counts.index.tolist()]
        
        dpg.add_bar_series(x_pos, value_counts.values.tolist(), weight=0.7, 
                         label="Frequency", parent=yaxis_tag)
        
        if labels:
            dpg.set_axis_ticks(xaxis_tag, tuple(zip(labels, x_pos)))

def _apply_sva_filters_and_run(main_callbacks: dict):
    """SVA 필터 적용 및 실행"""
    print("Running SVA analysis...")
    
    current_df = main_callbacks['get_current_df']()
    util_funcs = main_callbacks['get_util_funcs']()
    extended_util_funcs = {**util_funcs, 'main_app_callbacks': main_callbacks}
    
    target_var = main_callbacks['get_selected_target_variable']()
    
    # 진행 모달 표시
    if not _show_progress_modal("Processing SVA", "Analyzing variables..."):
        return
    
    # 결과 영역 초기화
    if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
        dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
    else:
        _hide_progress_modal()
        _show_alert_modal("UI Error", "SVA result display area is missing.")
        return
    
    if current_df is None:
        dpg.add_text("Load data first to perform Single Variable Analysis.", 
                    parent=TAG_SVA_RESULTS_CHILD_WINDOW)
        _hide_progress_modal()
        return
    
    # 필터 설정 가져오기
    filter_strength = "Weak (Exclude obvious non-analytical)"
    if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO):
        filter_strength = dpg.get_value(TAG_SVA_FILTER_STRENGTH_RADIO)
    
    group_by_target = False
    if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
        group_by_target = dpg.get_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX)
    
    grouped_plot_pref = "KDE"
    if group_by_target and dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
        grouped_plot_pref = dpg.get_value(TAG_SVA_GROUPED_PLOT_TYPE_RADIO)
    
    # 그룹핑 설정 검증
    target_series_for_grouping = None
    if group_by_target:
        target_series_for_grouping = _validate_grouping_target(
            current_df, target_var, main_callbacks
        )
        if target_series_for_grouping is None:
            group_by_target = False
    
    # 변수 필터링
    dpg.set_value(TAG_SVA_PROGRESS_TEXT, "SVA: Filtering variables...")
    dpg.split_frame()
    
    filtered_cols, actual_filter = _get_filtered_variables(
        current_df, filter_strength, main_callbacks, target_var
    )
    
    if not filtered_cols:
        msg = f"No variables to analyze with filter: '{actual_filter}'"
        dpg.add_text(msg, parent=TAG_SVA_RESULTS_CHILD_WINDOW)
        _hide_progress_modal()
        return
    
    # 변수별 분석 실행
    total_vars = len(filtered_cols)
    print(f"Analyzing {total_vars} variables with '{actual_filter}' filter")
    
    for i, col_name in enumerate(filtered_cols):
        if not dpg.is_dearpygui_running():
            break
        
        dpg.set_value(TAG_SVA_PROGRESS_TEXT, f"SVA: Analyzing {col_name} ({i+1}/{total_vars})")
        dpg.split_frame()
        
        _create_single_variable_analysis(
            i, col_name, current_df, target_series_for_grouping,
            grouped_plot_pref, extended_util_funcs, actual_filter
        )
    
    _hide_progress_modal()
    print("SVA processing finished.")

def _show_progress_modal(title: str, message: str) -> bool:
    """진행 모달 표시"""
    if not dpg.is_dearpygui_running():
        return False
    
    if not dpg.does_item_exist(TAG_SVA_PROGRESS_MODAL):
        viewport_width = dpg.get_viewport_width()
        viewport_height = dpg.get_viewport_height()
        modal_width = 350
        modal_height = 70
        
        with dpg.window(label=title, modal=True, show=False, 
                       tag=TAG_SVA_PROGRESS_MODAL,
                       no_close=True, no_title_bar=True, 
                       pos=[(viewport_width - modal_width) // 2, 
                            (viewport_height - modal_height) // 2],
                       width=modal_width, height=modal_height,
                       no_saved_settings=True):
            dpg.add_text(message, tag=TAG_SVA_PROGRESS_TEXT)
    
    dpg.configure_item(TAG_SVA_PROGRESS_MODAL, show=True)
    dpg.set_value(TAG_SVA_PROGRESS_TEXT, message)
    dpg.split_frame()
    return True

def _hide_progress_modal():
    """진행 모달 숨기기"""
    if dpg.does_item_exist(TAG_SVA_PROGRESS_MODAL):
        dpg.configure_item(TAG_SVA_PROGRESS_MODAL, show=False)

def _validate_grouping_target(df: pd.DataFrame, target_var: str, 
                            main_callbacks: dict) -> Optional[pd.Series]:
    """그룹핑 타겟 검증"""
    if not target_var or target_var not in df.columns:
        _show_alert_modal("Grouping Info", 
                         "Target variable not selected or invalid. Grouping disabled.")
        return None
    
    unique_values = df[target_var].nunique(dropna=False)
    if not (2 <= unique_values <= 7):
        _show_alert_modal("Grouping Warning",
                         f"Target variable '{target_var}' has {unique_values} unique values.\n"
                         f"Grouping requires between 2 and 7 unique values.")
        if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
            dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
        if dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
            dpg.configure_item(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, show=False)
        return None
    
    return df[target_var]

def _create_single_variable_analysis(idx: int, col_name: str, df: pd.DataFrame,
                                   target_series: pd.Series, plot_pref: str,
                                   util_funcs: dict, filter_applied: str):
    """단일 변수 분석 생성"""
    var_section_tag = f"{TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX}{''.join(filter(str.isalnum, col_name))}_{idx}"
    
    analysis_types = util_funcs.get('main_app_callbacks', {}).get('get_column_analysis_types', lambda: {})()
    col_type = analysis_types.get(col_name, str(df[col_name].dtype))
    
    analysis_override = None
    if filter_applied == "None (All variables)" and "Numeric (Binary)" in col_type:
        analysis_override = "ForceCategoricalForBinaryNumeric"
    
    with dpg.group(tag=var_section_tag, parent=TAG_SVA_RESULTS_CHILD_WINDOW):
        # 헤더
        results_width = dpg.get_item_width(TAG_SVA_RESULTS_CHILD_WINDOW)
        header_wrap = results_width - 30 if results_width and results_width > 50 else 500
        
        dpg.add_text(f"Variable: {util_funcs['format_text_for_display'](col_name, 60)} ({idx+1})",
                    color=(255, 255, 0), wrap=header_wrap)
        dpg.add_text(f"Type: {col_type} (Dtype: {str(df[col_name].dtype)})", 
                    wrap=header_wrap)
        if analysis_override:
            dpg.add_text("Display: Treated as Categorical", 
                        color=(200, 200, 0), wrap=header_wrap)
        dpg.add_spacer(height=5)
        
        # 3열 레이아웃
        with dpg.group(horizontal=True):
            available_width = results_width if results_width and results_width > 100 else 900
            col_width = int(available_width * 0.28)
            col_width = max(200, col_width)
            
            # 기본 통계
            with dpg.group(width=col_width):
                _create_sva_basic_stats_table(dpg.last_item(), df[col_name], 
                                            util_funcs, analysis_override)
            
            dpg.add_spacer(width=10)
            
            # 고급 통계
            with dpg.group(width=col_width):
                _create_sva_advanced_relations_table(dpg.last_item(), df[col_name], 
                                                   df, util_funcs, col_width)
            
            dpg.add_spacer(width=10)
            
            # 플롯
            with dpg.group():
                _create_single_var_plot(dpg.last_item(), df[col_name], 
                                      target_series, analysis_override,
                                      plot_pref, util_funcs)
        
        dpg.add_separator()
        dpg.add_spacer(height=10)

def _sva_group_by_target_callback(sender, app_data, user_data):
    """SVA 그룹핑 체크박스 콜백"""
    is_checked = dpg.get_value(sender)
    if dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
        dpg.configure_item(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, show=is_checked)
        if not is_checked:
            dpg.set_value(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, "KDE")

def _outlier_method_changed_callback(sender, app_data, user_data):
    """아웃라이어 처리 방법 변경 콜백"""
    method = dpg.get_value(sender)
    dpg.configure_item(TAG_OUTLIER_CAPPING_CONTROLS_GROUP, show=(method == "Capping"))
    dpg.configure_item(TAG_OUTLIER_IF_CONTROLS_GROUP, show=(method == "Isolation Forest"))
                                                                   
def get_outlier_settings_for_saving() -> dict:
    """아웃라이어 설정 저장용 데이터 반환"""
    settings = {'method': "None", 'params': {}}
    
    if not dpg.is_dearpygui_running():
        return settings
    
    if dpg.does_item_exist(TAG_OUTLIER_METHOD_RADIO):
        settings['method'] = dpg.get_value(TAG_OUTLIER_METHOD_RADIO)
    
    if settings['method'] == "Capping":
        if dpg.does_item_exist(TAG_OUTLIER_CAPPING_LOWER_PERCENTILE):
            settings['params']['lower_percentile'] = dpg.get_value(TAG_OUTLIER_CAPPING_LOWER_PERCENTILE)
        if dpg.does_item_exist(TAG_OUTLIER_CAPPING_UPPER_PERCENTILE):
            settings['params']['upper_percentile'] = dpg.get_value(TAG_OUTLIER_CAPPING_UPPER_PERCENTILE)
        if dpg.does_item_exist(TAG_OUTLIER_CAPPING_VAR_SELECTOR):
            settings['params']['selected_vars_capping'] = dpg.get_value(TAG_OUTLIER_CAPPING_VAR_SELECTOR)
    
    elif settings['method'] == "Isolation Forest":
        settings['params']['contamination'] = 'auto'
        if dpg.does_item_exist(TAG_OUTLIER_IF_VAR_SELECTOR):
            settings['params']['selected_vars_if'] = dpg.get_value(TAG_OUTLIER_IF_VAR_SELECTOR)
    
    return settings

def apply_outlier_treatment_from_settings(df: pd.DataFrame, config: dict, 
                                         main_callbacks: dict) -> Tuple[pd.DataFrame, bool]:
    """설정에 따른 아웃라이어 처리 적용"""
    if df is None or not config or 'method' not in config:
        return df, False
    
    method = config.get('method')
    params = config.get('params', {})
    original_shape = df.shape
    modified_df = df.copy()
    changes_made = False
    status_messages = []
    
    numeric_cols = _get_numeric_cols(modified_df)
    
    if method == "Capping":
        changes_made = _apply_capping(modified_df, params, numeric_cols, status_messages)
    elif method == "Isolation Forest":
        changes_made = _apply_isolation_forest(modified_df, params, numeric_cols, status_messages)
    
    # 결과 메시지 생성
    final_message = "Outlier treatment applied. " if changes_made else "No data changes made. "
    final_message += f"Original shape: {original_shape}, New shape: {modified_df.shape}. "
    if status_messages:
        final_message += "Details: " + " | ".join(status_messages)
    
    if dpg.does_item_exist(TAG_OUTLIER_RESULTS_TEXT):
        dpg.set_value(TAG_OUTLIER_RESULTS_TEXT, final_message)
    
    # 변경사항을 main_app에 알림
    if main_callbacks and changes_made:
        main_callbacks.get('notify_eda_df_changed', lambda df: None)(modified_df.copy())
        main_callbacks.get('set_eda_outlier_applied_flag', lambda flag: None)(True)
    elif main_callbacks and not changes_made and method != "None":
        main_callbacks.get('set_eda_outlier_applied_flag', lambda flag: None)(True)
    elif main_callbacks and method == "None":
        main_callbacks.get('set_eda_outlier_applied_flag', lambda flag: None)(False)
    
    return modified_df, changes_made

def _apply_capping(df: pd.DataFrame, params: dict, numeric_cols: List[str], 
                  status_messages: List[str]) -> bool:
    """Capping 방식 아웃라이어 처리"""
    lower_p = params.get('lower_percentile', 1)
    upper_p = params.get('upper_percentile', 99)
    selected_vars = params.get('selected_vars_capping', [])
    
    vars_to_cap = selected_vars if selected_vars else numeric_cols
    vars_to_cap = [v for v in vars_to_cap if v in numeric_cols]
    
    if not vars_to_cap:
        status_messages.append("Capping: No numeric variables selected or available.")
        return False
    
    changes_made = False
    for col in vars_to_cap:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col].dtype):
            low_val = np.percentile(df[col].dropna(), lower_p)
            high_val = np.percentile(df[col].dropna(), upper_p)
            
            if df[col].min() < low_val or df[col].max() > high_val:
                changes_made = True
            
            df[col] = np.clip(df[col], low_val, high_val)
    
    status_messages.append(f"Capping applied ({lower_p}%-{upper_p}%) to: {', '.join(vars_to_cap)}.")
    if not changes_made and vars_to_cap:
        status_messages.append("No actual changes (values already within bounds).")
    
    return changes_made

def _apply_isolation_forest(df: pd.DataFrame, params: dict, numeric_cols: List[str], 
                           status_messages: List[str]) -> bool:
    """Isolation Forest 방식 아웃라이어 처리"""
    contamination = params.get('contamination', 'auto')
    selected_vars = params.get('selected_vars_if', [])
    
    vars_to_process = selected_vars if selected_vars else numeric_cols
    vars_to_process = [v for v in vars_to_process if v in numeric_cols]
    
    if not vars_to_process:
        status_messages.append("Isolation Forest: No numeric variables selected or available.")
        return False
    
    changes_made = False
    for col in vars_to_process:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col].dtype):
            continue
        
        series = df[[col]].dropna()
        if len(series) < 2 or series.nunique().iloc[0] < 2:
            status_messages.append(f"IF for '{col}': Skipped (insufficient data/variance).")
            continue
        
        try:
            model = IsolationForest(contamination=contamination, random_state=42)
            model.fit(series)
            is_outlier = model.predict(series) == -1
            
            outlier_indices = series.index[is_outlier]
            if len(outlier_indices) > 0:
                changes_made = True
                df.loc[outlier_indices, col] = np.nan
                status_messages.append(f"IF for '{col}': {len(outlier_indices)} outliers set to NaN.")
            else:
                status_messages.append(f"IF for '{col}': No outliers identified.")
        except Exception as e:
            status_messages.append(f"IF for '{col}': Error - {str(e)}.")
            print(f"Error applying Isolation Forest to {col}: {e}")
            traceback.print_exc()
    
    return changes_made

def _apply_outlier_treatment_button_callback(sender, app_data, user_data):
    """아웃라이어 처리 적용 버튼 콜백"""
    main_callbacks = user_data
    df_after_step1 = main_callbacks['get_df_after_step1']()
    
    if df_after_step1 is None:
        _show_alert_modal("Error", "No data from Step 1 available.")
        if dpg.does_item_exist(TAG_OUTLIER_RESULTS_TEXT):
            dpg.set_value(TAG_OUTLIER_RESULTS_TEXT, "Error: Data from Step 1 is not available.")
        return
    
    current_config = get_outlier_settings_for_saving()
    
    if dpg.does_item_exist(TAG_OUTLIER_RESULTS_TEXT):
        dpg.set_value(TAG_OUTLIER_RESULTS_TEXT, "Processing outlier treatment...")
    
    _, changes_made = apply_outlier_treatment_from_settings(
        df_after_step1.copy(), current_config, main_callbacks
    )
    
    _update_outlier_status_text(main_callbacks)

def _reset_outliers_to_after_step1_callback(sender, app_data, user_data):
    """Step 1 이후 상태로 리셋 콜백"""
    main_callbacks = user_data
    main_callbacks['reset_eda_df_to_after_step1']()
    
    if dpg.does_item_exist(TAG_OUTLIER_RESULTS_TEXT):
        dpg.set_value(TAG_OUTLIER_RESULTS_TEXT, 
                     "Outlier treatment reset to data after Step 1. "
                     "Saved outlier settings (if any) were re-applied.")
    
    _update_outlier_status_text(main_callbacks)

def _update_outlier_status_text(main_callbacks: dict):
    """아웃라이어 처리 상태 텍스트 업데이트"""
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OUTLIER_STATUS_TEXT):
        return
    
    is_applied = main_callbacks.get('get_eda_outlier_applied_flag', lambda: False)()
    
    if is_applied:
        dpg.set_value(TAG_OUTLIER_STATUS_TEXT, 
                     "Status: Outlier settings are currently reflected in the EDA DataFrame.")
        dpg.configure_item(TAG_OUTLIER_STATUS_TEXT, color=(0, 200, 0))
    else:
        dpg.set_value(TAG_OUTLIER_STATUS_TEXT, 
                     "Status: No outlier treatment currently active on EDA DataFrame.")
        dpg.configure_item(TAG_OUTLIER_STATUS_TEXT, color=(200, 200, 0))

def _create_outlier_treatment_tab_content(parent_tag: str, main_callbacks: dict):
    """아웃라이어 처리 탭 내용 생성"""
    with dpg.group(parent=parent_tag):
        dpg.add_text("Select an outlier treatment method. Changes will modify the DataFrame used for EDA.", 
                    wrap=-1)
        dpg.add_text("Note: These settings are saved with the session.", 
                    wrap=-1, color=(200, 200, 200))
        dpg.add_separator()
        
        dpg.add_text("Current Outlier Treatment Status:")
        dpg.add_text("Status: Initializing...", tag=TAG_OUTLIER_STATUS_TEXT, wrap=-1)
        dpg.add_spacer(height=5)
        
        dpg.add_radio_button(
            items=["None", "Capping", "Isolation Forest"],
            tag=TAG_OUTLIER_METHOD_RADIO, default_value="None", horizontal=True,
            callback=_outlier_method_changed_callback
        )
        dpg.add_spacer(height=10)
        
        # Capping 설정
        with dpg.group(tag=TAG_OUTLIER_CAPPING_CONTROLS_GROUP, show=False):
            dpg.add_text("Capping Settings:")
            dpg.add_text("Caps selected numeric variables at specified percentiles.", wrap=-1)
            dpg.add_input_int(label="Lower Percentile (1-20)", 
                            tag=TAG_OUTLIER_CAPPING_LOWER_PERCENTILE, 
                            default_value=1, min_value=1, max_value=20, width=150)
            dpg.add_input_int(label="Upper Percentile (80-99)", 
                            tag=TAG_OUTLIER_CAPPING_UPPER_PERCENTILE, 
                            default_value=99, min_value=80, max_value=99, width=150)
            dpg.add_text("Apply to Variables (numeric only, select none for all):")
            dpg.add_listbox(tag=TAG_OUTLIER_CAPPING_VAR_SELECTOR, width=-1, num_items=5)
        
        # Isolation Forest 설정
        with dpg.group(tag=TAG_OUTLIER_IF_CONTROLS_GROUP, show=False):
            dpg.add_text("Isolation Forest Settings:")
            dpg.add_text("Identifies outliers using Isolation Forest (contamination='auto'). "
                        "Outliers are set to NaN.", wrap=-1)
            dpg.add_text("Apply to Variables (numeric only, select none for all):")
            dpg.add_listbox(tag=TAG_OUTLIER_IF_VAR_SELECTOR, width=-1, num_items=5)
        
        dpg.add_spacer(height=15)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply Outlier Treatment to EDA Data", 
                         tag=TAG_OUTLIER_APPLY_BUTTON,
                         callback=_apply_outlier_treatment_button_callback, 
                         user_data=main_callbacks, height=30)
            dpg.add_button(label="Reset to Data Post-Step 1", 
                         tag=TAG_OUTLIER_RESET_TO_AFTER_STEP1_BUTTON,
                         callback=_reset_outliers_to_after_step1_callback, 
                         user_data=main_callbacks, height=30)
        
        dpg.add_separator()
        dpg.add_text("Application Results/Log:")
        dpg.add_text("Apply treatment to see effects.", tag=TAG_OUTLIER_RESULTS_TEXT, wrap=-1)

def _analyze_target_correlations(df: pd.DataFrame, target_var: str,
                                 numeric_cols: List[str], max_vars_total: int,
                                 parent_tag: str):
    """타겟 변수와의 상관관계 분석 (최대 변수 개수 적용)"""
    dpg.add_text(f"Analysis 2: Top {max_vars_total} Variables (incl. target) Correlated with Target '{target_var}'",
                parent=parent_tag, color=(200, 200, 0))

    if not pd.api.types.is_numeric_dtype(df[target_var].dtype):
        dpg.add_text(f"Target variable '{target_var}' is not numeric. "
                    f"This analysis requires a numeric target.",
                    parent=parent_tag, color=(255, 100, 100))
        return

    top_n_others = max_vars_total - 1
    if top_n_others < 1:
        dpg.add_text(f"Not enough variable slots (max_vars_total={max_vars_total}) to include other variables with target '{target_var}'.", parent=parent_tag)
        return

    top_correlated_others = _get_top_n_correlated_with_target(df, target_var, numeric_cols, top_n_others)

    vars_for_heatmap = [target_var]
    vars_for_heatmap.extend([v for v in top_correlated_others if v != target_var])
    vars_for_heatmap = list(dict.fromkeys(vars_for_heatmap)) # Ensure target_var is first and unique

    if len(vars_for_heatmap) >= 2:
        try:
            corr_matrix_target = df[vars_for_heatmap].corr(method='pearson')
            _create_heatmap_in_dpg(corr_matrix_target,
                                  f"Heatmap: Target '{target_var}' & Top {len(vars_for_heatmap)-1} Correlated (Total {len(vars_for_heatmap)} vars)",
                                  parent_tag, height=350)
        except Exception as e:
            dpg.add_text(f"Error creating target correlation heatmap: {e}", parent=parent_tag, color=(255,0,0))
            print(f"Error creating target correlation heatmap for {target_var} with {vars_for_heatmap}: {e}")
            traceback.print_exc()
    else:
        dpg.add_text(f"Not enough variables (min 2, found {len(vars_for_heatmap)}) for target correlation heatmap after selection.", parent=parent_tag)


def _run_correlation_analysis(df: pd.DataFrame, util_funcs: dict, main_callbacks: dict):
    """상관관계 분석 실행"""
    if not dpg.is_dearpygui_running():
        return

    results_group = TAG_MVA_CORR_RESULTS_GROUP
    if dpg.does_item_exist(results_group):
        dpg.delete_item(results_group, children_only=True)
    else:
        _show_alert_modal("UI Error", "Correlation results group is missing.")
        return

    if df is None:
        dpg.add_text("Load data first.", parent=results_group)
        return

    numeric_cols = _get_numeric_cols(df)
    if len(numeric_cols) < 2:
        dpg.add_text("Not enough numeric columns for correlation analysis (need at least 2).",
                    parent=results_group)
        return

    target_var = main_callbacks['get_selected_target_variable']()
    max_vars_heatmap = 20

    if len(numeric_cols) <= max_vars_heatmap:
        dpg.add_text(f"Correlation Matrix for {len(numeric_cols)} Numeric Variables:",
                    parent=results_group, color=(255, 255, 0))
        try:
            corr_matrix = df[numeric_cols].corr(method='pearson')
            _create_heatmap_in_dpg(corr_matrix, "Overall Correlation Heatmap (Pearson)",
                                  results_group)
        except Exception as e:
            dpg.add_text(f"Error creating overall correlation heatmap: {e}", parent=results_group, color=(255,0,0))
            print(f"Error creating overall correlation heatmap: {e}")
            traceback.print_exc()
    else:
        dpg.add_text(f"Number of numeric variables ({len(numeric_cols)}) > {max_vars_heatmap}. "
                    f"Showing targeted analyses (max {max_vars_heatmap} vars per heatmap):", parent=results_group, color=(255, 255, 0))

        _analyze_highly_correlated_vars(df, numeric_cols, results_group, max_vars_heatmap)
        dpg.add_separator(parent=results_group)

        if target_var and target_var in df.columns:
            _analyze_target_correlations(df, target_var, numeric_cols,
                                       max_vars_heatmap, results_group)
        else:
            dpg.add_text("No target variable selected or target is not in DataFrame. Target correlation analysis skipped.",
                        parent=results_group)

    _create_correlation_pairs_table(df, numeric_cols, results_group, util_funcs)

def _create_heatmap_in_dpg(data_matrix: pd.DataFrame, title: str,
                          parent_tag: str, height: int = 450):
    """DearPyGui 히트맵 생성 (SystemError 디버깅 및 축 옵션 조정)"""
    if not dpg.is_dearpygui_running():
        return

    if not dpg.does_item_exist(parent_tag):
        print(f"Error in _create_heatmap_in_dpg for '{title}': Parent tag '{parent_tag}' does not exist.")
        return

    if data_matrix is None or not isinstance(data_matrix, pd.DataFrame) or data_matrix.empty:
        dpg.add_text(f"{title}: No data to display (DataFrame is empty or None).", parent=parent_tag)
        return

    rows, cols = data_matrix.shape

    if rows <= 0 or cols <= 0:
        dpg.add_text(f"{title}: Invalid dimensions (rows={rows}, cols={cols}). Cannot draw heatmap.", parent=parent_tag)
        return

    heatmap_np_array = data_matrix.values.flatten()
    heatmap_np_array = np.nan_to_num(heatmap_np_array, nan=0.0, posinf=1.0, neginf=-1.0).astype(float)
    heatmap_data_float = heatmap_np_array.tolist()

    if len(heatmap_data_float) != rows * cols:
        dpg.add_text(f"{title}: Data length mismatch after processing (expected={rows*cols}, actual={len(heatmap_data_float)}).", parent=parent_tag)
        return

    col_labels = [str(c) for c in data_matrix.columns.tolist()]
    row_labels = [str(r) for r in data_matrix.index.tolist()]

    plot_uuid = dpg.generate_uuid()
    try:
        with dpg.plot(label=title, height=height, width=-1, parent=parent_tag,
                     tag=plot_uuid, equal_aspects=True if rows == cols else False):

            # add_plot_axis 호출 단순화 (no_zoom, no_gridlines 임시 제거)
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="", auto_fit=True)
            yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="", auto_fit=True)

            # 축 눈금 레이블 설정 (셀 중앙에 위치하도록)
            if col_labels and cols > 0:
                tick_values_x = [i + 0.5 for i in range(cols)]
                if len(col_labels) == len(tick_values_x):
                    dpg.set_axis_ticks(xaxis, tuple(zip(col_labels, tick_values_x)))
                else:
                    print(f"Warning for heatmap '{title}': X-axis label count ({len(col_labels)}) "
                          f"does not match tick value count ({len(tick_values_x)}). Using default ticks.")

            if row_labels and rows > 0:
                tick_values_y = [i + 0.5 for i in range(rows)]
                if len(row_labels) == len(tick_values_y):
                    dpg.set_axis_ticks(yaxis, tuple(zip(row_labels, tick_values_y)))
                else:
                    print(f"Warning for heatmap '{title}': Y-axis label count ({len(row_labels)}) "
                          f"does not match tick value count ({len(tick_values_y)}). Using default ticks.")

            dpg.add_heat_series(heatmap_data_float, rows=rows, cols=cols,
                               scale_min=-1.0, scale_max=1.0,
                               format='%.2f', parent=yaxis, show_tooltips=True,
                               bounds_min=(0, 0), bounds_max=(cols, rows))

        dpg.add_spacer(height=10, parent=parent_tag)

    except SystemError as se:
        error_msg = f"Failed to create heatmap '{title}' due to SystemError: {se}. Plot Tag: {plot_uuid}, Parent: {parent_tag}"
        print(error_msg)
        print(f"Details for SystemError in '{title}':")
        print(f"  Dimensions: rows={rows}, cols={cols}")
        print(f"  heatmap_data_float length: {len(heatmap_data_float)}")
        # print(f"  heatmap_data_float (sample): {heatmap_data_float[:10] if len(heatmap_data_float) > 10 else heatmap_data_float}") # 필요시 주석 해제
        print(f"  col_labels (count {len(col_labels)}): {col_labels[:5] if len(col_labels) > 5 else col_labels}")
        print(f"  row_labels (count {len(row_labels)}): {row_labels[:5] if len(row_labels) > 5 else row_labels}")
        dpg.add_text(error_msg, parent=parent_tag, color=(255, 0, 0), wrap=-1)
        traceback.print_exc() # traceback 추가
    except Exception as e:
        error_msg = f"An unexpected error occurred while creating heatmap '{title}': {e}"
        print(error_msg)
        traceback.print_exc()
        dpg.add_text(error_msg, parent=parent_tag, color=(255, 0, 0), wrap=-1)


def _analyze_highly_correlated_vars(df: pd.DataFrame, numeric_cols: List[str],
                                   parent_tag: str, max_vars_heatmap: int = 20):
    """높은 상관관계 변수 분석 (히트맵 조건 강화 및 변수 개수 제한)"""
    if not dpg.is_dearpygui_running():
        return

    if not dpg.does_item_exist(parent_tag):
        # print(f"Error in _analyze_highly_correlated_vars: Parent tag '{parent_tag}' does not exist.") # 디버깅용
        return

    dpg.add_text(f"Analysis 1: Variables with Pairwise |Correlation| >= 0.6 (Max {max_vars_heatmap} vars for heatmap)",
                parent=parent_tag, color=(200, 200, 0))

    if len(numeric_cols) < 2:
        dpg.add_text("Not enough numeric columns for pairwise correlation.", parent=parent_tag)
        return

    try:
        corr_matrix = df[numeric_cols].corr(method='pearson')
    except Exception as e:
        dpg.add_text(f"Error calculating correlation matrix: {e}", parent=parent_tag, color=(255,0,0))
        return

    highly_correlated_set = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if pd.notna(corr_val) and abs(corr_val) >= 0.6:
                highly_correlated_set.add(corr_matrix.columns[i])
                highly_correlated_set.add(corr_matrix.columns[j])

    vars_for_heatmap = sorted(list(highly_correlated_set))

    if len(vars_for_heatmap) > max_vars_heatmap:
        dpg.add_text(f"More than {max_vars_heatmap} variables found with pairwise |corr| >= 0.6. "
                     f"Displaying heatmap for the first {max_vars_heatmap} (alphabetically sorted).",
                     parent=parent_tag, color=(200, 200, 100), wrap=-1)
        vars_for_heatmap = vars_for_heatmap[:max_vars_heatmap]

    if len(vars_for_heatmap) >= 2:
        try:
            corr_matrix_high = df[vars_for_heatmap].corr(method='pearson')
        except Exception as e:
            dpg.add_text(f"Error calculating high-correlation sub-matrix: {e}", parent=parent_tag, color=(255,0,0))
            return

        if not corr_matrix_high.empty and corr_matrix_high.shape[0] >= 1 and corr_matrix_high.shape[1] >= 1:
            _create_heatmap_in_dpg(corr_matrix_high,
                                  f"Heatmap of Highly Correlated Variables (>=0.6, Top {len(vars_for_heatmap)})",
                                  parent_tag, height=350)
        else:
            dpg.add_text("Could not generate heatmap for highly correlated variables (matrix is empty or invalid).", parent=parent_tag)
    else:
        dpg.add_text("No variable pairs found with |correlation| >= 0.6, or not enough variables (min 2) to form a heatmap after filtering.", parent=parent_tag)


def _create_correlation_pairs_table(df: pd.DataFrame, numeric_cols: List[str], 
                                   parent_tag: str, util_funcs: dict):
    """상관관계 쌍 테이블 생성"""
    dpg.add_text("Highly Correlated Numeric Pairs (|Correlation| > 0.7):", parent=parent_tag)
    
    corr_matrix = df[numeric_cols].corr(method='pearson')
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append({
                    "Variable 1": corr_matrix.columns[i],
                    "Variable 2": corr_matrix.columns[j],
                    "Correlation": f"{corr_matrix.iloc[i, j]:.3f}"
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        table_tag = dpg.generate_uuid()
        
        with dpg.table(header_row=True, tag=table_tag, parent=parent_tag,
                      resizable=True, policy=dpg.mvTable_SizingFixedFit,
                      scrollY=True, height=200,
                      borders_innerH=True, borders_outerH=True, 
                      borders_innerV=True, borders_outerV=True):
            util_funcs['create_table_with_data'](table_tag, high_corr_df, 
                                               parent_df_for_widths=high_corr_df)
    else:
        dpg.add_text("No pairs with |correlation| > 0.7 found.", parent=parent_tag)

def _run_pair_plot_analysis(df: pd.DataFrame, selected_vars: list, hue_var: str, 
                           util_funcs: dict, main_callbacks: dict):
    """Pair plot 분석 실행"""
    results_group = TAG_MVA_PAIRPLOT_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group):
        return
    
    dpg.delete_item(results_group, children_only=True)
    
    if df is None:
        dpg.add_text("Load data first.", parent=results_group)
        return
    
    numeric_cols = _get_numeric_cols(df)
    if not numeric_cols:
        dpg.add_text("No numeric variables available for Pair Plot.", parent=results_group)
        return
    
    # 변수 선택 로직
    vars_for_plot = _select_vars_for_pairplot(df, selected_vars, numeric_cols, 
                                             main_callbacks, results_group)
    
    if not vars_for_plot or len(vars_for_plot) < 2:
        dpg.add_text("Not enough valid numeric variables for Pair Plot (need at least 2).", 
                    parent=results_group)
        return
    
    # Hue 변수 검증
    hue_series, hue_categories, actual_hue_var = _validate_hue_variable(df, hue_var, 
                                                                       main_callbacks)
    
    # Pair plot 생성
    _create_pair_plot_grid(df, vars_for_plot, hue_series, hue_categories, 
                          actual_hue_var, results_group)

def _select_vars_for_pairplot(df: pd.DataFrame, selected_vars: list, 
                             numeric_cols: List[str], main_callbacks: dict, 
                             parent_tag: str) -> List[str]:
    """Pair plot을 위한 변수 선택"""
    target_var = main_callbacks['get_selected_target_variable']()
    max_vars = 7
    vars_for_plot = []
    info_message = ""
    
    if selected_vars and len(selected_vars) > 0:
        vars_for_plot = [var for var in selected_vars if var in numeric_cols]
        if not vars_for_plot:
            info_message = "None of the selected variables are valid numeric columns."
        elif len(vars_for_plot) < 2:
            info_message = "Please select at least two valid numeric variables."
    else:
        if len(numeric_cols) <= max_vars:
            vars_for_plot = numeric_cols
            info_message = f"Using all {len(vars_for_plot)} available numeric variables."
        else:
            if target_var and target_var in df.columns and pd.api.types.is_numeric_dtype(df[target_var].dtype):
                info_message = f"Using top {max_vars} variables correlated with target '{target_var}'."
                vars_for_plot = _get_top_n_correlated_with_target(df, target_var, 
                                                                 numeric_cols, max_vars)
                if target_var in numeric_cols and target_var not in vars_for_plot:
                    if len(vars_for_plot) < max_vars:
                        vars_for_plot.append(target_var)
                    else:
                        vars_for_plot[-1] = target_var
                vars_for_plot = list(dict.fromkeys(vars_for_plot))
            else:
                info_message = f"Using first {max_vars} numeric variables."
                vars_for_plot = numeric_cols[:max_vars]
    
    if info_message:
        dpg.add_text(info_message, parent=parent_tag, 
                    wrap=dpg.get_item_width(parent_tag) or 600)
    
    if len(vars_for_plot) > 7:
        _show_alert_modal("Pair Plot Limit", 
                         f"Plotting first 7 variables out of {len(vars_for_plot)} for performance.")
        vars_for_plot = vars_for_plot[:7]
    
    return vars_for_plot

def _validate_hue_variable(df: pd.DataFrame, hue_var: str, 
                          main_callbacks: dict) -> Tuple[Optional[pd.Series], 
                                                        Optional[List[str]], 
                                                        Optional[str]]:
    """Hue 변수 검증"""
    if not hue_var or hue_var not in df.columns:
        return None, None, None
    
    temp_hue_series = df[hue_var]
    cat_cols = _get_categorical_cols(df[[hue_var]], max_unique_for_cat=10, 
                                    main_callbacks=main_callbacks)
    
    if hue_var in cat_cols:
        hue_categories = sorted(temp_hue_series.astype(str).dropna().unique())
        return temp_hue_series, hue_categories, hue_var
    else:
        _show_alert_modal("Hue Variable Warning", 
                         f"Hue variable '{hue_var}' has too many unique values (>10). Hue disabled.")
        return None, None, None

def _create_pair_plot_grid(df: pd.DataFrame, vars_for_plot: List[str], 
                          hue_series: Optional[pd.Series], 
                          hue_categories: Optional[List[str]], 
                          hue_var: Optional[str], parent_tag: str):
    """Pair plot 그리드 생성"""
    n_vars = len(vars_for_plot)
    plot_cell_width = max(180, int((dpg.get_item_width(parent_tag) or 800) / n_vars) - 20)
    plot_cell_height = plot_cell_width
    
    plot_title = f"Pair Plot for: {', '.join(vars_for_plot)}"
    if hue_var:
        plot_title += f" (Hue: {hue_var})"
    dpg.add_text(plot_title, parent=parent_tag)
    
    with dpg.child_window(parent=parent_tag, border=False, autosize_x=True, autosize_y=True):
        for i in range(n_vars):
            with dpg.group(horizontal=True):
                for j in range(n_vars):
                    _create_pair_plot_cell(df, vars_for_plot, i, j, 
                                         plot_cell_width, plot_cell_height,
                                         hue_series, hue_categories, n_vars)

def _create_pair_plot_cell(df: pd.DataFrame, vars: List[str], row: int, col: int,
                          width: int, height: int, hue_series: Optional[pd.Series],
                          hue_categories: Optional[List[str]], n_vars: int):
    """Pair plot 개별 셀 생성"""
    var_y = vars[row]
    var_x = vars[col]
    
    cell_label = f"{var_y} vs {var_x}" if row != col else f"Dist: {var_x}"
    cell_plot_tag = dpg.generate_uuid()
    
    with dpg.plot(width=width, height=height, label=cell_label, tag=cell_plot_tag):
        show_x_label = (row == n_vars - 1)
        show_y_label = (col == 0)
        
        px_axis = dpg.add_plot_axis(dpg.mvXAxis, 
                                   label=var_x if show_x_label else "", 
                                   no_tick_labels=not show_x_label)
        py_axis = dpg.add_plot_axis(dpg.mvYAxis, 
                                   label=var_y if show_y_label else "", 
                                   no_tick_labels=not show_y_label)
        
        if hue_series is not None and row != col:
            dpg.add_plot_legend(parent=cell_plot_tag, horizontal=True, 
                              location=dpg.mvPlot_Location_NorthEast, outside=False)
        
        if row == col:
            # 대각선: 히스토그램
            series_diag = df[var_x].dropna()
            if not series_diag.empty and series_diag.nunique() >= 1:
                if series_diag.nunique() == 1:
                    dpg.add_bar_series([0], [len(series_diag)], weight=0.5, 
                                     label=str(series_diag.iloc[0]), parent=py_axis)
                    dpg.set_axis_ticks(px_axis, [(str(series_diag.iloc[0]), 0)])
                else:
                    dpg.add_histogram_series(series_diag.tolist(), bins=-1, 
                                           density=True, label="Hist", 
                                           parent=py_axis, weight=1.0)
        else:
            # 비대각선: 산점도
            _create_scatter_plot(df, var_x, var_y, py_axis, hue_series, hue_categories)

def _create_scatter_plot(df: pd.DataFrame, var_x: str, var_y: str, 
                        yaxis_tag, hue_series: Optional[pd.Series], 
                        hue_categories: Optional[List[str]]):
    """산점도 생성"""
    series_x = df[var_x]
    series_y = df[var_y]
    
    if hue_series is not None and hue_categories is not None:
        for cat_val in hue_categories:
            mask = (hue_series.astype(str) == cat_val)
            temp_df = pd.concat([series_x[mask], series_y[mask]], axis=1).dropna()
            
            if not temp_df.empty:
                dpg.add_scatter_series(
                    temp_df.iloc[:, 0].tolist(),
                    temp_df.iloc[:, 1].tolist(),
                    label=str(cat_val), parent=yaxis_tag
                )
    else:
        temp_df = pd.concat([series_x, series_y], axis=1).dropna()
        if not temp_df.empty:
            dpg.add_scatter_series(temp_df.iloc[:, 0].tolist(), 
                                 temp_df.iloc[:, 1].tolist(), parent=yaxis_tag)

def _run_categorical_correlation_analysis(df: pd.DataFrame, util_funcs: dict, 
                                        main_callbacks: dict):
    """범주형 변수 상관관계 분석"""
    results_group = TAG_MVA_CAT_EDA_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group):
        return
    
    dpg.delete_item(results_group, children_only=True)
    
    if df is None:
        dpg.add_text("Load data first.", parent=results_group)
        return
    
    selected_vars = []
    if dpg.does_item_exist(TAG_MVA_CAT_EDA_VAR_SELECTOR):
        selected_vars = dpg.get_value(TAG_MVA_CAT_EDA_VAR_SELECTOR)
    
    all_cat_cols = _get_categorical_cols(df, max_unique_for_cat=30, main_callbacks=main_callbacks)
    
    if not selected_vars:
        cat_vars = all_cat_cols[:20] if len(all_cat_cols) > 20 else all_cat_cols
        if len(all_cat_cols) > 20:
            dpg.add_text(f"Using first 20 categorical variables for Cramer's V heatmap.", 
                        parent=results_group, color=(200, 200, 0))
    else:
        cat_vars = [var for var in selected_vars if var in all_cat_cols][:20]
        if len(selected_vars) > 20:
            dpg.add_text(f"Using first 20 valid categorical variables.", 
                        parent=results_group, color=(200, 200, 0))
    
    if len(cat_vars) < 2:
        dpg.add_text("Not enough categorical variables for Cramer's V analysis (need at least 2).", 
                    parent=results_group)
        return
    
    dpg.add_text(f"Cramer's V Matrix for: {', '.join(cat_vars)}", parent=results_group)
    
    # Cramer's V 매트릭스 계산
    cramers_v_matrix = pd.DataFrame(np.zeros((len(cat_vars), len(cat_vars))),
                                   columns=cat_vars, index=cat_vars)
    
    for i in range(len(cat_vars)):
        for j in range(i, len(cat_vars)):
            var1 = cat_vars[i]
            var2 = cat_vars[j]
            
            if var1 == var2:
                c_v = 1.0
            else:
                c_v = _calculate_cramers_v(df[var1], df[var2])
            
            cramers_v_matrix.iloc[i, j] = c_v
            if i != j:
                cramers_v_matrix.iloc[j, i] = c_v
    
    # 히트맵 표시
    heatmap_data = cramers_v_matrix.values.flatten().tolist()
    rows, cols = cramers_v_matrix.shape
    
    with dpg.plot(label="Cramer's V Heatmap (Categorical Associations)", 
                 height=450, width=-1, parent=results_group, equal_aspects=True):
        xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="")
        yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="")
        
        dpg.set_axis_ticks(xaxis, tuple(zip(cat_vars, list(range(cols)))))
        dpg.set_axis_ticks(yaxis, tuple(zip(cat_vars, list(range(rows)))))
        
        dpg.add_heat_series(heatmap_data, rows=rows, cols=cols, 
                           scale_min=0.0, scale_max=1.0,
                           format='%.2f', parent=yaxis, show_tooltips=True,
                           bounds_min=(0, 0), bounds_max=(cols, rows))

def _run_target_variable_analysis(df: pd.DataFrame, target_var: str, target_type: str,
                                 feature_var: str, util_funcs: dict, main_callbacks: dict):
    """타겟 변수 분석"""
    results_group = TAG_MVA_TARGET_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group):
        return
    
    dpg.delete_item(results_group, children_only=True)
    
    if df is None or not target_var or target_var not in df.columns or \
       not feature_var or feature_var not in df.columns or target_var == feature_var:
        dpg.add_text("Select valid and distinct target and feature variables.", 
                    parent=results_group)
        return
    
    dpg.add_text(f"Analysis: Feature '{feature_var}' vs Target '{target_var}' (Type: {target_type})", 
                parent=results_group)
    dpg.add_separator(parent=results_group)
    
    target_series = df[target_var]
    feature_series = df[feature_var]
    
    # Step 1 타입 정보 가져오기
    s1_types = main_callbacks.get('get_column_analysis_types', lambda: {})()
    feature_type = s1_types.get(feature_var, str(feature_series.dtype))
    
    is_feature_numeric = (
        ("Numeric" in feature_type and "Binary" not in feature_type) or
        (pd.api.types.is_numeric_dtype(feature_series.dtype) and feature_series.nunique() > 2)
    )
    
    if target_type == "Continuous":
        if is_feature_numeric:
            _analyze_continuous_target_numeric_feature(df, target_var, feature_var, 
                                                     target_series, feature_series, 
                                                     results_group, util_funcs)
        else:
            _analyze_continuous_target_categorical_feature(df, target_var, feature_var, 
                                                         target_series, feature_series, 
                                                         results_group, util_funcs)
    elif target_type == "Categorical":
        if is_feature_numeric:
            _analyze_categorical_target_numeric_feature(df, target_var, feature_var, 
                                                      target_series, feature_series, 
                                                      results_group, util_funcs)
        else:
            _analyze_categorical_target_categorical_feature(df, target_var, feature_var, 
                                                          target_series, feature_series, 
                                                          results_group, util_funcs)
    else:
        dpg.add_text(f"Analysis for target type '{target_type}' is not implemented.", 
                    parent=results_group)

def _analyze_continuous_target_numeric_feature(df, target_var, feature_var, 
                                             target_series, feature_series, 
                                             parent_tag, util_funcs):
    """연속형 타겟 vs 숫자형 특성 분석"""
    aligned_df = pd.concat([feature_series, target_series], axis=1).dropna()
    
    if not aligned_df.empty and len(aligned_df) >= 2:
        correlation = aligned_df.iloc[:, 0].corr(aligned_df.iloc[:, 1])
        dpg.add_text(f"Pearson Correlation: {correlation:.3f}" if pd.notna(correlation) else "Correlation: N/A", 
                    parent=parent_tag)
        
        with dpg.plot(label=f"Scatter: '{feature_var}' by '{target_var}'", 
                     height=350, width=-1, parent=parent_tag):
            dpg.add_plot_axis(dpg.mvXAxis, label=feature_var)
            yaxis = dpg.add_plot_axis(dpg.mvYAxis, label=target_var)
            dpg.add_scatter_series(aligned_df.iloc[:, 0].tolist(), 
                                 aligned_df.iloc[:, 1].tolist(), parent=yaxis)
    else:
        dpg.add_text("Not enough common data points for analysis.", parent=parent_tag)

def _analyze_continuous_target_categorical_feature(df, target_var, feature_var, 
                                                 target_series, feature_series, 
                                                 parent_tag, util_funcs):
    """연속형 타겟 vs 범주형 특성 분석"""
    dpg.add_text("Grouped Statistics (Feature's Categories vs Continuous Target):", parent=parent_tag)
    
    try:
        feature_cat = feature_series.astype(str) if feature_series.nunique() > 20 else feature_series
        
        if feature_cat.nunique() > 20:
            dpg.add_text(f"Feature has too many categories ({feature_cat.nunique()}). Max 20.", 
                        parent=parent_tag)
        else:
            grouped_stats = df.groupby(feature_cat)[target_var].agg(
                ['mean', 'median', 'std', 'count', 'min', 'max']
            ).reset_index()
            grouped_stats.columns = [str(col) for col in grouped_stats.columns]
            
            table_tag = dpg.generate_uuid()
            with dpg.table(header_row=True, tag=table_tag, parent=parent_tag, 
                         resizable=True, height=200, scrollY=True,
                         borders_innerH=True, borders_outerH=True, 
                         borders_innerV=True, borders_outerV=True):
                util_funcs['create_table_with_data'](table_tag, grouped_stats.round(3), 
                                                   parent_df_for_widths=grouped_stats.round(3))
            
            # KDE 플롯
            dpg.add_text("Distribution comparison (DPG shows grouped KDE):", parent=parent_tag)
            unique_cats = feature_cat.dropna().unique()[:7]
            
            if len(unique_cats) >= 1:
                with dpg.plot(label=f"Distribution of '{target_var}' by '{feature_var}'", 
                            height=350, width=-1, parent=parent_tag):
                    dpg.add_plot_axis(dpg.mvXAxis, label=target_var)
                    yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Density")
                    dpg.add_plot_legend(parent=dpg.last_item())
                    
                    for cat_val in unique_cats:
                        subset = target_series[feature_cat == cat_val].dropna()
                        if len(subset) > 1 and subset.nunique() > 1:
                            try:
                                kde = stats.gaussian_kde(subset.astype(float))
                                x_vals = np.linspace(subset.min(), subset.max(), 100)
                                y_vals = kde(x_vals)
                                dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), 
                                                  label=f"{feature_var}={str(cat_val)[:20]}", 
                                                  parent=yaxis)
                            except Exception as e:
                                print(f"KDE error for category {cat_val}: {e}")
    except Exception as e:
        dpg.add_text(f"Error during grouping: {e}", parent=parent_tag)

def _analyze_categorical_target_numeric_feature(df, target_var, feature_var, 
                                              target_series, feature_series, 
                                              parent_tag, util_funcs):
    """범주형 타겟 vs 숫자형 특성 분석"""
    dpg.add_text("Grouped Statistics (Target's Categories vs Numeric Feature):", parent=parent_tag)
    
    try:
        target_cat = target_series.astype(str) if target_series.nunique() > 20 else target_series
        
        if target_cat.nunique() > 20:
            dpg.add_text(f"Target has too many categories ({target_cat.nunique()}). Max 20.", 
                        parent=parent_tag)
        else:
            grouped_stats = df.groupby(target_cat)[feature_var].agg(
                ['mean', 'median', 'std', 'count', 'min', 'max']
            ).reset_index()
            grouped_stats.columns = [str(col) for col in grouped_stats.columns]
            
            table_tag = dpg.generate_uuid()
            with dpg.table(header_row=True, tag=table_tag, parent=parent_tag, 
                         resizable=True, height=200, scrollY=True,
                         borders_innerH=True, borders_outerH=True, 
                         borders_innerV=True, borders_outerV=True):
                util_funcs['create_table_with_data'](table_tag, grouped_stats.round(3), 
                                                   parent_df_for_widths=grouped_stats.round(3))
            
            # Density plots
            dpg.add_text("Overlaid Density Plots of Feature by Target Categories:", parent=parent_tag)
            unique_targets = target_cat.dropna().unique()[:7]
            
            if len(unique_targets) >= 1:
                with dpg.plot(label=f"Density of '{feature_var}' by '{target_var}'", 
                            height=350, width=-1, parent=parent_tag):
                    dpg.add_plot_axis(dpg.mvXAxis, label=feature_var)
                    yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Density")
                    dpg.add_plot_legend(parent=dpg.last_item())
                    
                    for cat_val in unique_targets:
                        subset = feature_series[target_cat == cat_val].dropna()
                        if len(subset) > 1 and subset.nunique() > 1:
                            try:
                                kde = stats.gaussian_kde(subset.astype(float))
                                x_vals = np.linspace(subset.min(), subset.max(), 100)
                                y_vals = kde(x_vals)
                                dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), 
                                                  label=f"{target_var}={str(cat_val)[:20]}", 
                                                  parent=yaxis)
                            except Exception as e:
                                print(f"KDE error for target category {cat_val}: {e}")
    except Exception as e:
        dpg.add_text(f"Error during grouping: {e}", parent=parent_tag)

def _analyze_categorical_target_categorical_feature(df, target_var, feature_var, 
                                                  target_series, feature_series, 
                                                  parent_tag, util_funcs):
    """범주형 타겟 vs 범주형 특성 분석"""
    dpg.add_text("Crosstabulation (Feature vs Target):", parent=parent_tag)
    
    try:
        ct_feature = feature_series.astype(str)
        ct_target = target_series.astype(str)
        
        if ct_feature.nunique() > 20 or ct_target.nunique() > 20:
            dpg.add_text("Too many categories (>20) for full crosstab. Showing top 20 combinations.", 
                        parent=parent_tag)
            counts_summary = df.groupby([ct_feature.name, ct_target.name]).size().reset_index(
                name='counts'
            ).nlargest(20, 'counts')
            
            table_tag = dpg.generate_uuid()
            with dpg.table(header_row=True, tag=table_tag, parent=parent_tag, 
                         resizable=True, height=250, scrollY=True,
                         borders_innerH=True, borders_outerH=True, 
                         borders_innerV=True, borders_outerV=True):
                util_funcs['create_table_with_data'](table_tag, counts_summary, 
                                                   parent_df_for_widths=counts_summary)
        else:
            # 전체 crosstab
            crosstab_abs = pd.crosstab(ct_feature, ct_target, dropna=False)
            crosstab_norm = pd.crosstab(ct_feature, ct_target, normalize='index', 
                                      dropna=False).mul(100).round(1)
            
            # Chi-squared test
            chi2_stat, p_val, dof, expected = stats.chi2_contingency(crosstab_abs)
            cramers_v = _calculate_cramers_v(feature_series, target_series)
            
            dpg.add_text(f"Chi-squared Test: stat={chi2_stat:.2f}, p-value={p_val:.3f}, "
                        f"Cramér's V={cramers_v:.3f}", parent=parent_tag)
            
            # Count 테이블
            dpg.add_text("Counts:", parent=parent_tag)
            table_tag_abs = dpg.generate_uuid()
            with dpg.table(header_row=True, tag=table_tag_abs, parent=parent_tag, 
                         resizable=True, height=180, scrollY=True,
                         borders_innerH=True, borders_outerH=True, 
                         borders_innerV=True, borders_outerV=True):
                util_funcs['create_table_with_data'](table_tag_abs, 
                                                   crosstab_abs.reset_index(), 
                                                   parent_df_for_widths=crosstab_abs.reset_index())
            
            # 비율 테이블
            dpg.add_text("Row Percentages (%):", parent=parent_tag)
            table_tag_norm = dpg.generate_uuid()
            with dpg.table(header_row=True, tag=table_tag_norm, parent=parent_tag, 
                         resizable=True, height=180, scrollY=True,
                         borders_innerH=True, borders_outerH=True, 
                         borders_innerV=True, borders_outerV=True):
                util_funcs['create_table_with_data'](table_tag_norm, 
                                                   crosstab_norm.reset_index(), 
                                                   parent_df_for_widths=crosstab_norm.reset_index())
            
            dpg.add_text("Consider Stacked/Grouped Bar charts (external tools).", parent=parent_tag)
    except Exception as e:
        dpg.add_text(f"Error creating crosstab: {e}", parent=parent_tag)

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """UI 생성"""
    global _main_app_callbacks_eda, _util_funcs_eda
    _main_app_callbacks_eda = main_callbacks
    _util_funcs_eda = main_callbacks.get('get_util_funcs', lambda: {})()
    
    main_callbacks['register_step_group_tag'](step_name, TAG_EDA_GROUP)
    
    with dpg.group(tag=TAG_EDA_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        
        with dpg.tab_bar(tag=TAG_EDA_MAIN_TAB_BAR):
            # SVA 탭
            with dpg.tab(label="Single Variable Analysis (SVA)", tag=TAG_SVA_TAB):
                _create_sva_tab_content(main_callbacks)
            
            # MVA 탭
            with dpg.tab(label="Multivariate Analysis (MVA)", tag=TAG_MVA_TAB):
                _create_mva_tab_content(main_callbacks)
            
            # Outlier 탭
            with dpg.tab(label="Outlier Treatment", tag=TAG_OUTLIER_TAB):
                _create_outlier_treatment_tab_content(TAG_OUTLIER_TAB, main_callbacks)
    
    main_callbacks['register_module_updater'](step_name, update_ui)

def _create_sva_tab_content(main_callbacks: dict):
    """SVA 탭 내용 생성"""
    with dpg.group(horizontal=True):
        with dpg.group(width=280):
            dpg.add_text("Variable Filter")
            dpg.add_radio_button(
                items=["Strong (Top 5-10 relevant)", "Medium (Top 11-20 relevant)", 
                      "Weak (Exclude obvious non-analytical)", "None (All variables)"],
                tag=TAG_SVA_FILTER_STRENGTH_RADIO, 
                default_value="Weak (Exclude obvious non-analytical)"
            )
            dpg.add_spacer(height=5)
            dpg.add_text("Filter Info:", wrap=270, color=(200, 200, 200))
            dpg.add_text("- Strong/Medium: Ranked by relevance to Target", 
                        wrap=270, color=(200, 200, 200))
            dpg.add_text("- Weak: Excludes single-value & binary numeric", 
                        wrap=270, color=(200, 200, 200))
            dpg.add_text("- None: Includes most vars (text excluded)", 
                        wrap=270, color=(200, 200, 200))
        
        dpg.add_spacer(width=10)
        
        with dpg.group():
            dpg.add_text("Grouping & Plot Option")
            dpg.add_checkbox(label="Group by Target (2-7 Unique Values)",
                           tag=TAG_SVA_GROUP_BY_TARGET_CHECKBOX, default_value=False,
                           user_data=main_callbacks,
                           callback=_sva_group_by_target_callback)
            dpg.add_radio_button(items=["KDE", "Histogram"],
                               tag=TAG_SVA_GROUPED_PLOT_TYPE_RADIO,
                               default_value="KDE", horizontal=True, show=False)
            dpg.add_spacer(height=10)
            dpg.add_button(label="Run Single Variable Analysis", 
                         tag=TAG_SVA_RUN_BUTTON,
                         callback=lambda: _apply_sva_filters_and_run(main_callbacks),
                         width=-1, height=30)
    
    dpg.add_separator()
    with dpg.child_window(tag=TAG_SVA_RESULTS_CHILD_WINDOW, border=True):
        dpg.add_text("Select filter options and click 'Run Single Variable Analysis'.")

def _create_mva_tab_content(main_callbacks: dict):
    """MVA 탭 내용 생성"""
    with dpg.tab_bar(tag=TAG_MVA_SUB_TAB_BAR):
        # 상관관계 탭
        with dpg.tab(label="Correlation (Numeric)", tag=TAG_MVA_CORR_TAB):
            dpg.add_button(label="Run Correlation Analysis", 
                         tag=TAG_MVA_CORR_RUN_BUTTON,
                         callback=lambda: _run_correlation_analysis(
                             main_callbacks['get_current_df'](), 
                             main_callbacks['get_util_funcs'](), 
                             main_callbacks
                         ))
            dpg.add_child_window(tag=TAG_MVA_CORR_RESULTS_GROUP, border=True, height=-1)
        
        # Pair Plot 탭
        with dpg.tab(label="Pair Plot (Numeric)", tag=TAG_MVA_PAIRPLOT_TAB):
            dpg.add_text("Select numeric variables (up to 7 recommended). "
                        "If none selected, defaults based on variable count.", wrap=-1)
            dpg.add_listbox(tag=TAG_MVA_PAIRPLOT_VAR_SELECTOR, width=-1, num_items=8)
            dpg.add_combo(label="Hue (Optional Categorical Var, <10 Categories)", 
                        tag=TAG_MVA_PAIRPLOT_HUE_COMBO, width=350)
            dpg.add_button(label="Generate Pair Plot", 
                         tag=TAG_MVA_PAIRPLOT_RUN_BUTTON,
                         callback=lambda: _run_pair_plot_analysis(
                             main_callbacks['get_current_df'](),
                             dpg.get_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR),
                             dpg.get_value(TAG_MVA_PAIRPLOT_HUE_COMBO),
                             main_callbacks['get_util_funcs'](),
                             main_callbacks
                         ))
            dpg.add_child_window(tag=TAG_MVA_PAIRPLOT_RESULTS_GROUP, border=True, height=-1)
        
        # Target vs Feature 탭
        with dpg.tab(label="Target vs Feature", tag=TAG_MVA_TARGET_TAB):
            dpg.add_text("Analyze relationship between features and the selected target.", 
                        tag=TAG_MVA_TARGET_INFO_TEXT, wrap=-1)
            with dpg.group(horizontal=True):
                dpg.add_combo(label="Feature Variable", 
                            tag=TAG_MVA_TARGET_FEATURE_COMBO, width=300)
                dpg.add_button(label="Analyze vs Target", 
                             tag=TAG_MVA_TARGET_RUN_BUTTON,
                             callback=lambda: _run_target_variable_analysis(
                                 main_callbacks['get_current_df'](),
                                 main_callbacks['get_selected_target_variable'](),
                                 main_callbacks['get_selected_target_variable_type'](),
                                 dpg.get_value(TAG_MVA_TARGET_FEATURE_COMBO),
                                 main_callbacks['get_util_funcs'](),
                                 main_callbacks
                             ))
            dpg.add_separator()
            dpg.add_child_window(tag=TAG_MVA_TARGET_RESULTS_GROUP, border=True, height=-1)
        
        # 범주형 상관관계 탭
        with dpg.tab(label="Correlation (Categorical)", tag=TAG_MVA_CAT_EDA_TAB):
            dpg.add_text("Analyze associations between categorical variables using Cramer's V.", 
                        wrap=-1)
            dpg.add_text("Select variables (up to 20). If none selected, defaults based on available.", 
                        wrap=-1)
            dpg.add_listbox(tag=TAG_MVA_CAT_EDA_VAR_SELECTOR, width=-1, num_items=8)
            dpg.add_button(label="Run Categorical Association Analysis", 
                         tag=TAG_MVA_CAT_EDA_RUN_BUTTON,
                         callback=lambda: _run_categorical_correlation_analysis(
                             main_callbacks['get_current_df'](),
                             main_callbacks['get_util_funcs'](),
                             main_callbacks
                         ))
            dpg.add_child_window(tag=TAG_MVA_CAT_EDA_RESULTS_GROUP, border=True, height=-1)

def update_ui(current_df: pd.DataFrame, main_callbacks: dict):
    """UI 업데이트"""
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_EDA_GROUP):
        return
    
    global _main_app_callbacks_eda, _util_funcs_eda
    _main_app_callbacks_eda = main_callbacks
    _util_funcs_eda = main_callbacks.get('get_util_funcs', lambda: {})()
    
    # SVA UI 업데이트
    if current_df is None:
        if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
            dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
            dpg.add_text("Load data to perform Single Variable Analysis.", 
                        parent=TAG_SVA_RESULTS_CHILD_WINDOW)
    
    # MVA 선택기 업데이트
    all_columns = current_df.columns.tolist() if current_df is not None else []
    numeric_cols = _get_numeric_cols(current_df) if current_df is not None else []
    cat_cols_for_hue = [""] + (_get_categorical_cols(current_df, max_unique_for_cat=10, 
                                                     main_callbacks=main_callbacks) 
                              if current_df is not None else [])
    all_cat_cols = _get_categorical_cols(current_df, max_unique_for_cat=30, 
                                       main_callbacks=main_callbacks) if current_df is not None else []
    
    # Pair Plot 선택기
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_VAR_SELECTOR):
        dpg.configure_item(TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=numeric_cols)
        if not numeric_cols:
            dpg.set_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR, [])
    
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_HUE_COMBO):
        current_hue = dpg.get_value(TAG_MVA_PAIRPLOT_HUE_COMBO)
        dpg.configure_item(TAG_MVA_PAIRPLOT_HUE_COMBO, items=cat_cols_for_hue)
        if current_hue and current_hue in cat_cols_for_hue:
            dpg.set_value(TAG_MVA_PAIRPLOT_HUE_COMBO, current_hue)
        else:
            dpg.set_value(TAG_MVA_PAIRPLOT_HUE_COMBO, "")
    
    # Target vs Feature 선택기
    target_var = main_callbacks['get_selected_target_variable']()
    if dpg.does_item_exist(TAG_MVA_TARGET_INFO_TEXT):
        if current_df is not None and target_var and target_var in all_columns:
            dpg.set_value(TAG_MVA_TARGET_INFO_TEXT, 
                         f"Analyzing features against Target: '{target_var}' "
                         f"(Type: {main_callbacks['get_selected_target_variable_type']()})")
            if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO):
                feature_candidates = [col for col in all_columns if col != target_var]
                dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=feature_candidates)
        else:
            dpg.set_value(TAG_MVA_TARGET_INFO_TEXT, 
                         "Load data and select a target variable to enable this analysis.")
            if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO):
                dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=[])
                dpg.set_value(TAG_MVA_TARGET_FEATURE_COMBO, None)
    
    # 범주형 EDA 선택기
    if dpg.does_item_exist(TAG_MVA_CAT_EDA_VAR_SELECTOR):
        dpg.configure_item(TAG_MVA_CAT_EDA_VAR_SELECTOR, items=all_cat_cols)
        if not all_cat_cols:
            dpg.set_value(TAG_MVA_CAT_EDA_VAR_SELECTOR, [])
    
    # Outlier Treatment 선택기
    if dpg.does_item_exist(TAG_OUTLIER_CAPPING_VAR_SELECTOR):
        dpg.configure_item(TAG_OUTLIER_CAPPING_VAR_SELECTOR, items=numeric_cols)
        if not numeric_cols:
            dpg.set_value(TAG_OUTLIER_CAPPING_VAR_SELECTOR, [])
    
    if dpg.does_item_exist(TAG_OUTLIER_IF_VAR_SELECTOR):
        dpg.configure_item(TAG_OUTLIER_IF_VAR_SELECTOR, items=numeric_cols)
        if not numeric_cols:
            dpg.set_value(TAG_OUTLIER_IF_VAR_SELECTOR, [])
    
    # Outlier 상태 업데이트
    _update_outlier_status_text(main_callbacks)
    
    # 데이터가 없을 때 결과 영역 초기화
    if current_df is None:
        result_areas = [
            (TAG_MVA_CORR_RESULTS_GROUP, "Load data."),
            (TAG_MVA_PAIRPLOT_RESULTS_GROUP, "Load data."),
            (TAG_MVA_TARGET_RESULTS_GROUP, "Load data and select target."),
            (TAG_MVA_CAT_EDA_RESULTS_GROUP, "Load data."),
            (TAG_OUTLIER_RESULTS_TEXT, "Load data to manage outliers.")
        ]
        
        for area, message in result_areas:
            if area == TAG_OUTLIER_RESULTS_TEXT:
                if dpg.does_item_exist(area):
                    dpg.set_value(area, message)
            else:
                if dpg.does_item_exist(area):
                    dpg.delete_item(area, children_only=True)
                    dpg.add_text(message, parent=area)

def reset_eda_ui_defaults():
    """EDA UI를 기본값으로 리셋"""
    if not dpg.is_dearpygui_running():
        return
    
    # SVA 기본값
    if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO):
        dpg.set_value(TAG_SVA_FILTER_STRENGTH_RADIO, "Weak (Exclude obvious non-analytical)")
    if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
        dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
    if dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
        dpg.set_value(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, "KDE")
        dpg.configure_item(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, show=False)
    if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
        dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
        dpg.add_text("Select filter options and click 'Run Single Variable Analysis'.", 
                    parent=TAG_SVA_RESULTS_CHILD_WINDOW)
    
    # MVA 기본값
    selectors = [
        (TAG_MVA_PAIRPLOT_VAR_SELECTOR, []),
        (TAG_MVA_PAIRPLOT_HUE_COMBO, ""),
        (TAG_MVA_TARGET_FEATURE_COMBO, None),
        (TAG_MVA_CAT_EDA_VAR_SELECTOR, [])
    ]
    
    for selector, default_val in selectors:
        if dpg.does_item_exist(selector):
            dpg.configure_item(selector, items=[] if isinstance(default_val, list) else [""])
            dpg.set_value(selector, default_val)
    
    # 결과 영역 초기화
    result_areas = [
        (TAG_MVA_CORR_RESULTS_GROUP, "Run analysis to see results."),
        (TAG_MVA_PAIRPLOT_RESULTS_GROUP, "Run analysis to see results."),
        (TAG_MVA_TARGET_RESULTS_GROUP, "Load data and select target."),
        (TAG_MVA_CAT_EDA_RESULTS_GROUP, "Run analysis to see results.")
    ]
    
    for area, message in result_areas:
        if dpg.does_item_exist(area):
            dpg.delete_item(area, children_only=True)
            dpg.add_text(message, parent=area)
    
    # Outlier Treatment 기본값
    if dpg.does_item_exist(TAG_OUTLIER_METHOD_RADIO):
        dpg.set_value(TAG_OUTLIER_METHOD_RADIO, "None")
    if dpg.does_item_exist(TAG_OUTLIER_CAPPING_LOWER_PERCENTILE):
        dpg.set_value(TAG_OUTLIER_CAPPING_LOWER_PERCENTILE, 1)
    if dpg.does_item_exist(TAG_OUTLIER_CAPPING_UPPER_PERCENTILE):
        dpg.set_value(TAG_OUTLIER_CAPPING_UPPER_PERCENTILE, 99)
    
    outlier_selectors = [TAG_OUTLIER_CAPPING_VAR_SELECTOR, TAG_OUTLIER_IF_VAR_SELECTOR]
    for selector in outlier_selectors:
        if dpg.does_item_exist(selector):
            dpg.configure_item(selector, items=[])
            dpg.set_value(selector, [])
    
    if dpg.does_item_exist(TAG_OUTLIER_CAPPING_CONTROLS_GROUP):
        dpg.configure_item(TAG_OUTLIER_CAPPING_CONTROLS_GROUP, show=False)
    if dpg.does_item_exist(TAG_OUTLIER_IF_CONTROLS_GROUP):
        dpg.configure_item(TAG_OUTLIER_IF_CONTROLS_GROUP, show=False)
    
    if dpg.does_item_exist(TAG_OUTLIER_RESULTS_TEXT):
        dpg.set_value(TAG_OUTLIER_RESULTS_TEXT, "Apply outlier treatment to see effects.")
    if dpg.does_item_exist(TAG_OUTLIER_STATUS_TEXT):
        dpg.set_value(TAG_OUTLIER_STATUS_TEXT, "Status: Initializing...")
        dpg.configure_item(TAG_OUTLIER_STATUS_TEXT, color=(180, 180, 180))                                                                   