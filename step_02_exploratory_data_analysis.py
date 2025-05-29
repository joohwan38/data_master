# step_02_exploratory_data_analysis.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from scipy import stats # For skewness, kurtosis, normality tests, chi2_contingency
import traceback # For detailed error logging

# --- UI Element Tags ---
TAG_EDA_GROUP = "step2_eda_group"
TAG_EDA_MAIN_TAB_BAR = "step2_eda_main_tab_bar"

# Single Variable Analysis (SVA) Tab
TAG_SVA_TAB = "step2_sva_tab"
TAG_SVA_FILTER_STRENGTH_RADIO = "step2_sva_filter_strength_radio"
TAG_SVA_GROUP_BY_TARGET_CHECKBOX = "step2_sva_group_by_target_checkbox"
TAG_SVA_RUN_BUTTON = "step2_sva_run_button" # SVA 실행 버튼 태그 추가
TAG_SVA_RESULTS_CHILD_WINDOW = "step2_sva_results_child_window"
TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX = "sva_var_section_"
TAG_SVA_PROGRESS_MODAL = "sva_progress_modal"
TAG_SVA_PROGRESS_TEXT = "sva_progress_text"
TAG_SVA_ALERT_MODAL_PREFIX = "sva_alert_modal_" # 기존 모달 태그 prefix

# Reusable Modal Tags (이전 답변에서 제안된 방식)
REUSABLE_SVA_ALERT_MODAL_TAG = "reusable_sva_alert_modal_unique_tag"
REUSABLE_SVA_ALERT_TEXT_TAG = "reusable_sva_alert_text_unique_tag"

# Multivariate Analysis (MVA) Tab (태그들은 기존과 동일하게 유지)
TAG_MVA_TAB = "step2_mva_tab"
TAG_MVA_SUB_TAB_BAR = "step2_mva_sub_tab_bar"
TAG_MVA_CORR_TAB = "step2_mva_corr_tab"
TAG_MVA_CORR_RUN_BUTTON = "step2_mva_corr_run_button"
TAG_MVA_CORR_HEATMAP_PLOT = "step2_mva_corr_heatmap_plot"
TAG_MVA_CORR_TABLE = "step2_mva_corr_table"
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
TAG_MVA_TARGET_PLOT_AREA_PREFIX = "mva_target_plot_area_"
TAG_SVA_GROUPED_PLOT_TYPE_RADIO = "step2_sva_grouped_plot_type_radio"



def _show_alert_modal(title: str, message: str):
    """
    Displays a modal alert window. Reuses a single modal definition for all SVA alerts.
    (이전 답변에서 제안된 Reusable Modal 방식 유지)
    """
    if not dpg.is_dearpygui_running():
        print(f"DPG not running. Modal '{title}': {message}")
        return

    if not dpg.does_item_exist(REUSABLE_SVA_ALERT_MODAL_TAG):
        with dpg.window(label="Alert", modal=True, show=False, tag=REUSABLE_SVA_ALERT_MODAL_TAG,
                        no_close=True, pos=[400, 300], width=400, autosize=True,
                        no_saved_settings=True):
            dpg.add_text("", tag=REUSABLE_SVA_ALERT_TEXT_TAG, wrap=380)
            dpg.add_spacer(height=15)
            with dpg.group(horizontal=True):
                # 버튼을 중앙에 가깝게 배치하기 위한 spacer (정확한 중앙 정렬은 어려울 수 있음)
                # dpg.add_spacer(width=dpg.get_item_width(dpg.last_item()) / 2 - 50) # 이 방식은 last_item에 따라 불안정할 수 있음
                dpg.add_spacer(width=140) # 고정값으로 대략 중앙 정렬 시도
                dpg.add_button(label="OK", width=100, user_data=REUSABLE_SVA_ALERT_MODAL_TAG,
                               callback=lambda s, a, u: dpg.configure_item(u, show=False))
    
    dpg.configure_item(REUSABLE_SVA_ALERT_MODAL_TAG, label=title, show=True)
    dpg.set_value(REUSABLE_SVA_ALERT_TEXT_TAG, message)

# --- Helper Functions (SVA and MVA common) ---

def _calculate_cramers_v(x: pd.Series, y: pd.Series):
    # (사용자 제공 파일의 내용과 동일하게 유지)
    if x is None or y is None or x.empty or y.empty: return 0.0
    try:
        temp_df = pd.DataFrame({'x': x, 'y': y}).dropna()
        if temp_df.empty or temp_df['x'].nunique() < 1 or temp_df['y'].nunique() < 1:
            return 0.0
        confusion_matrix = pd.crosstab(temp_df['x'], temp_df['y'])
        if confusion_matrix.empty or confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
            return 0.0
        chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
        n = confusion_matrix.sum().sum()
        if n == 0: return 0.0
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1 if n > 1 else 1))
        rcorr = r - (((r - 1)**2) / (n - 1 if n > 1 else 1) if r > 1 else 0)
        kcorr = k - (((k - 1)**2) / (n - 1 if n > 1 else 1) if k > 1 else 0)
        denominator = min((kcorr - 1 if kcorr > 1 else 0), (rcorr - 1 if rcorr > 1 else 0))
        if denominator == 0: return 0.0
        return np.sqrt(phi2corr / denominator)
    except Exception: return 0.0

def _get_top_correlated_vars(df: pd.DataFrame, current_var_name: str, top_n=5) -> list:
    """
    연관성이 높은 상위 N개 변수 목록을 반환합니다.
    반환값: [{'Variable': 이름, 'Metric': 측정항목, 'Value': 값}, ...] 형태의 리스트.
            정보 메시지만 있을 경우 [{'Info': 메시지}] 형태.
    """
    if df is None or current_var_name not in df.columns or len(df.columns) < 2:
        return [{'Info': 'Not enough data or variables'}]

    correlations_tuples = [] # (name, value, type_str)
    current_series = df[current_var_name]
    structured_results = []

    if pd.api.types.is_numeric_dtype(current_series.dtype):
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col_name in numeric_cols:
            if col_name == current_var_name: continue
            try:
                corr_val = current_series.corr(df[col_name])
                if pd.notna(corr_val) and abs(corr_val) > 0.01: # 약간의 연관성이라도 포함 (0.1은 너무 높을 수 있음)
                    correlations_tuples.append((col_name, corr_val, "Pearson Corr"))
            except Exception: pass
        correlations_tuples.sort(key=lambda item: abs(item[1]), reverse=True)
        for name, val, metric_type in correlations_tuples[:top_n]:
            structured_results.append({'Variable': name, 'Metric': metric_type, 'Value': f"{val:.3f}"})
        if not structured_results:
            return [{'Info': 'No strong numeric correlations found.'}]

    elif current_series.nunique(dropna=False) < 30 or pd.api.types.is_categorical_dtype(current_series.dtype) or current_series.dtype == 'object':
        candidate_cols = [col for col in df.columns if col != current_var_name and
                          (df[col].nunique(dropna=False) < 30 or pd.api.types.is_categorical_dtype(df[col].dtype) or df[col].dtype == 'object')]
        for col_name in candidate_cols:
            try:
                cramers_v = _calculate_cramers_v(current_series, df[col_name])
                if pd.notna(cramers_v) and cramers_v > 0.01: 
                    correlations_tuples.append((col_name, cramers_v, "Cramér's V"))
            except Exception: pass
        correlations_tuples.sort(key=lambda item: abs(item[1]), reverse=True)
        for name, val, metric_type in correlations_tuples[:top_n]:
            structured_results.append({'Variable': name, 'Metric': metric_type, 'Value': f"{val:.3f}"})
        if not structured_results:
            return [{'Info': 'No strong categorical associations found.'}]
    else:
        return [{'Info': 'N/A (Unsupported type for this summary)'}]

    return structured_results if structured_results else [{'Info': 'No significant relations found.'}]


def _get_numeric_cols(df: pd.DataFrame):
    if df is None: return []
    return df.select_dtypes(include=np.number).columns.tolist()

def _get_categorical_cols(df: pd.DataFrame, max_unique_for_cat=20):
    # (사용자 제공 파일의 내용과 동일하게 유지)
    if df is None: return []
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col].dtype):
            if df[col].nunique(dropna=False) <= max_unique_for_cat * 2:
                 cat_cols.append(col)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].nunique(dropna=False) <= max_unique_for_cat and col not in cat_cols :
            cat_cols.append(col)
    return list(set(cat_cols))


def _get_filtered_variables(df: pd.DataFrame, filter_strength_selected: str,
                            main_callbacks: dict, target_var: str = None) -> tuple[list, str]:
    if df is None or df.empty: return [], filter_strength_selected

    analysis_types_dict = main_callbacks.get('get_column_analysis_types', lambda: {})()
    if not analysis_types_dict:
        analysis_types_dict = {col: str(df[col].dtype) for col in df.columns}

    cols_after_text_filter = []
    for col_name in df.columns:
        col_analysis_type = analysis_types_dict.get(col_name, str(df[col_name].dtype))
        if isinstance(col_analysis_type, str) and \
           any(text_keyword in col_analysis_type for text_keyword in ["Text (", "Potentially Sensitive"]):
            continue
        cols_after_text_filter.append(col_name)
    
    print(f"DEBUG: Initial filter: {len(df.columns) - len(cols_after_text_filter)} 'Text' type columns excluded. {len(cols_after_text_filter)} candidates remain.")

    if filter_strength_selected == "None (All variables)":
        print(f"DEBUG: Filter Profile 'None' selected. Vars: {len(cols_after_text_filter)}")
        return cols_after_text_filter, "None (All variables)"

    weakly_filtered_cols = []
    for col_name in cols_after_text_filter:
        series = df[col_name]
        col_analysis_type = analysis_types_dict.get(col_name, str(series.dtype))
        if series.nunique(dropna=False) <= 1: continue
        is_numeric_binary = False
        if "Numeric" in col_analysis_type:
            unique_vals_str_set = set(series.dropna().astype(str).unique())
            if unique_vals_str_set <= {'0', '1', '0.0', '1.0'} and len(unique_vals_str_set) <= 2:
                is_numeric_binary = True
        if is_numeric_binary: continue
        weakly_filtered_cols.append(col_name)
    
    if filter_strength_selected == "Weak (Exclude obvious non-analytical)":
        print(f"DEBUG: Filter Profile 'Weak' applied. Vars: {len(weakly_filtered_cols)}")
        return weakly_filtered_cols, "Weak (Exclude obvious non-analytical)"

    # Medium/Strong 필터 로직
    print(f"DEBUG: Attempting '{filter_strength_selected}' filter. Target variable: '{target_var}'")

    numeric_cols_for_ranking = [
        col for col in weakly_filtered_cols 
        if "Numeric" in analysis_types_dict.get(col, df[col].dtype.name) and not (
            len(set(df[col].dropna().astype(str).unique())) == 2 and 
            set(df[col].dropna().astype(str).unique()) <= {'0', '1', '0.0', '1.0'}
        )
    ]
    if not numeric_cols_for_ranking:
        msg = f"Cannot apply '{filter_strength_selected}' filter: No suitable numeric variables found for ranking after initial filtering. Analysis will not run with this filter."
        _show_alert_modal("Filter Condition Not Met", msg)
        print(f"DEBUG: {msg}")
        return [], filter_strength_selected # 빈 리스트와 원래 요청된 필터 반환 (fallback 안 함)
        
    if not target_var:
        msg = f"Cannot apply '{filter_strength_selected}' filter: A Target Variable must be selected for relevance ranking. Analysis will not run with this filter."
        _show_alert_modal("Target Variable Needed", msg)
        print(f"DEBUG: {msg}")
        return [], filter_strength_selected # 빈 리스트와 원래 요청된 필터 반환 (fallback 안 함)

    target_series = df[target_var]
    user_defined_target_type = main_callbacks['get_selected_target_variable_type']() # "Categorical" 또는 "Continuous"

    relevance_scores = []
    for col_name in numeric_cols_for_ranking:
        if col_name == target_var: continue
        score = 0.0
        try:
            current_col_series = df[col_name] # 현재 분석 대상인 숫자형 변수

            if user_defined_target_type == "Categorical":
                # 목표 변수가 사용자에 의해 "범주형"으로 지정됨 -> ANOVA F-value 사용
                groups = [current_col_series[target_series == cat].dropna() for cat in target_series.dropna().unique()]
                groups = [g for g in groups if len(g) >= 2]
                if len(groups) >= 2: # ANOVA를 위한 최소 그룹 수 및 샘플 수 확인
                    f_val, p_val = stats.f_oneway(*groups)
                    score = f_val if pd.notna(f_val) else 0.0
                else:
                    print(f"Warning: Not enough valid groups for ANOVA between '{col_name}' and target '{target_var}'.")
                    score = 0.0
            elif user_defined_target_type == "Continuous":
                # 목표 변수가 사용자에 의해 "연속형"으로 지정됨
                # 목표 변수가 실제로 숫자형인지 확인 후 상관계수 계산
                if pd.api.types.is_numeric_dtype(target_series.dtype):
                    score = abs(current_col_series.corr(target_series))
                    if not pd.notna(score): score = 0.0 # NaN 처리
                else:
                    # 사용자가 연속형으로 지정했으나 실제 타입이 숫자형이 아닌 경우 경고
                    print(f"Warning: Target '{target_var}' designated 'Continuous' but is not numeric (actual dtype: {target_series.dtype}). Cannot calculate Pearson correlation for '{col_name}'.")
                    score = 0.0
            else:
                # 목표 변수 유형이 설정되지 않았거나 예기치 않은 값일 경우
                print(f"Warning: Target variable type for '{target_var}' is not set or invalid ('{user_defined_target_type}'). Relevance score for '{col_name}' set to 0.")
                score = 0.0

        except Exception as e:
            print(f"Error calculating relevance for '{col_name}' vs target '{target_var}' (User type: '{user_defined_target_type}'): {e}")
            score = 0.0

        if pd.notna(score):
            relevance_scores.append((col_name, score))
    
    relevance_scores.sort(key=lambda item: item[1], reverse=True)
    ranked_cols = [col for col, score in relevance_scores if pd.notna(score) and score > 0.001] 

    if not ranked_cols:
        msg = f"Cannot apply '{filter_strength_selected}' filter: No variables showed significant relevance to target '{target_var}'. Analysis will not run with this filter."
        _show_alert_modal("Filter Condition Not Met", msg)
        print(f"DEBUG: {msg}")
        return [], filter_strength_selected # 빈 리스트와 원래 요청된 필터 반환

    if filter_strength_selected == "Strong (Top 5-10 relevant)":
        result_cols = ranked_cols[:min(len(ranked_cols), 10)]
        print(f"DEBUG: '{filter_strength_selected}' applied. Returning {len(result_cols)} vars.")
        return result_cols, filter_strength_selected
    if filter_strength_selected == "Medium (Top 11-20 relevant)":
        result_cols = ranked_cols[:min(len(ranked_cols), 20)]
        print(f"DEBUG: '{filter_strength_selected}' applied. Returning {len(result_cols)} vars.")
        return result_cols, filter_strength_selected
    
    # 이 경우는 발생하지 않아야 하나, 안전장치로 weakly_filtered_cols 대신 빈 리스트 반환
    print(f"DEBUG: Filter strength '{filter_strength_selected}' did not match. No analysis performed.")
    return [], filter_strength_selected


def _create_sva_basic_stats_table(parent_tag: str, series: pd.Series, util_funcs: dict, analysis_type_override: str = None):
    dpg.add_text("Basic Statistics", parent=parent_tag) # 제목 추가
    stats_data = []
    is_effectively_categorical = (analysis_type_override == "ForceCategoricalForBinaryNumeric" or
                                  pd.api.types.is_categorical_dtype(series.dtype) or
                                  series.dtype == 'object' or
                                  series.nunique(dropna=False) < 5)
    is_numeric_original_dtype = pd.api.types.is_numeric_dtype(series.dtype)

    stats_data.append({'Statistic': 'Count', 'Value': str(series.count())}) # NaN 제외 개수
    stats_data.append({'Statistic': 'Missing', 'Value': str(series.isnull().sum())})
    stats_data.append({'Statistic': 'Missing %', 'Value': f"{series.isnull().mean()*100:.2f}%"})
    stats_data.append({'Statistic': 'Unique (Actual)', 'Value': str(series.nunique(dropna=False))})


    if is_numeric_original_dtype and not is_effectively_categorical:
        desc = series.describe()
        stats_to_show = ['mean', 'std', 'min', '25%', '50%', '75%', 'max'] # count는 위에서 처리
        for stat_name in stats_to_show:
            if stat_name in desc.index:
                stats_data.append({'Statistic': stat_name, 'Value': f"{desc[stat_name]:.3f}" if isinstance(desc[stat_name], (int, float)) else str(desc[stat_name])})
        
        skew, kurt = np.nan, np.nan
        series_not_na = series.dropna()
        if len(series_not_na) >=3 : # 왜도/첨도는 최소 3개 데이터 필요 (라이브러리 따라 다를 수 있음)
            try: skew = series_not_na.skew()
            except TypeError: pass # 일부 object-like 숫자형에서 에러날 수 있음
            try: kurt = series_not_na.kurtosis()
            except TypeError: pass
        stats_data.append({'Statistic': 'Skewness', 'Value': f"{skew:.3f}" if pd.notna(skew) else "N/A"})
        stats_data.append({'Statistic': 'Kurtosis', 'Value': f"{kurt:.3f}" if pd.notna(kurt) else "N/A"})
    else: 
        value_counts = series.value_counts(dropna=False)
        stats_data.append({'Statistic': 'Mode', 'Value': str(value_counts.index[0]) if not value_counts.empty else 'N/A'})
        stats_data.append({'Statistic': 'Mode Freq', 'Value': str(value_counts.iloc[0]) if not value_counts.empty else 'N/A'})
        if not value_counts.empty and len(value_counts)>1:
            stats_data.append({'Statistic': '2nd Mode', 'Value': str(value_counts.index[1])})
            stats_data.append({'Statistic': '2nd Mode Freq', 'Value': str(value_counts.iloc[1])})
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        table_tag = dpg.generate_uuid()
        # 표 높이 동적 조절 (기본 230에서 280으로 증가 시도, 또는 내용에 맞춰 최대값 설정)
        table_height = min(230 + 50, len(stats_df) * 22 + 40) # 행 높이 대략 22, 헤더 및 제목 공간 40
        with dpg.table(header_row=True, tag=table_tag, parent=parent_tag,
                       borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True,
                       resizable=True, policy=dpg.mvTable_SizingStretchProp, height=int(table_height), scrollY=True):
            util_funcs['create_table_with_data'](table_tag, stats_df, parent_df_for_widths=stats_df)


def _create_sva_advanced_relations_table(parent_tag: str, series: pd.Series, full_df: pd.DataFrame, util_funcs: dict, col_width: int):
    """표2: 정규성 검정 결과 및 상위 연관 변수 목록"""
    normality_data = []
    is_effectively_categorical = (pd.api.types.is_categorical_dtype(series.dtype) or
                                  series.dtype == 'object' or
                                  series.nunique(dropna=False) < 5) # ForceCategoricalForBinaryNumeric는 SVA 실행 시점에 타입이 변경되므로 여기서는 원본 Dtype 기반

    if pd.api.types.is_numeric_dtype(series.dtype) and not is_effectively_categorical:
        series_dropna = series.dropna()
        if 3 <= len(series_dropna) < 5000 :
            try:
                stat_sw, p_sw = stats.shapiro(series_dropna)
                normality_data.append({'Test': 'Shapiro-Wilk W', 'Value': f"{stat_sw:.3f}"})
                normality_data.append({'Test': 'p-value', 'Value': f"{p_sw:.3f}"})
                normality_data.append({'Test': 'Normality (α=0.05)', 'Value': "Likely Normal" if p_sw > 0.05 else "Likely Not Normal"})
            except Exception as e:
                normality_data.append({'Test': 'Shapiro-Wilk', 'Value': f"Error"}) # 간단히 Error로 표시
                print(f"Shapiro-Wilk error for {series.name}: {e}")
        else:
            normality_data.append({'Test': 'Shapiro-Wilk', 'Value': 'N/A (sample size)'})
    else:
        normality_data.append({'Test': 'Shapiro-Wilk', 'Value': 'N/A (not numeric)'})

    if normality_data:
        normality_df = pd.DataFrame(normality_data)
        dpg.add_text("Normality Test:", parent=parent_tag)
        norm_table_tag = dpg.generate_uuid()
        norm_table_height = min(100, len(normality_df) * 22 + 30)
        with dpg.table(header_row=True, tag=norm_table_tag, parent=parent_tag,
                       borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True,
                       resizable=True, policy=dpg.mvTable_SizingStretchProp, height=int(norm_table_height), scrollY=True):
            util_funcs['create_table_with_data'](norm_table_tag, normality_df, parent_df_for_widths=normality_df)
        dpg.add_spacer(height=5, parent=parent_tag)

    # Top Related Variables
    top_related_vars_data = _get_top_correlated_vars(full_df, series.name, top_n=5)
    dpg.add_text("Top Related Variables:", parent=parent_tag)
    if top_related_vars_data:
        if len(top_related_vars_data) == 1 and 'Info' in top_related_vars_data[0]:
             dpg.add_text(top_related_vars_data[0]['Info'], parent=parent_tag, wrap=col_width-10 if col_width > 20 else 200)
        else: # 실제 데이터가 있는 경우
            actual_data_for_table = [item for item in top_related_vars_data if 'Info' not in item]
            if actual_data_for_table:
                related_vars_df = pd.DataFrame(actual_data_for_table)
                rel_table_tag = dpg.generate_uuid()
                # 표 높이: 최대 5줄 + 헤더 + 약간의 여유. (5 * 22) + 30 + 10 = 150
                rel_table_height = min(150, len(related_vars_df) * 22 + 40) # 조금 더 여유
                with dpg.table(header_row=True, tag=rel_table_tag, parent=parent_tag,
                               borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True,
                               resizable=True, policy=dpg.mvTable_SizingStretchProp, height=int(rel_table_height), scrollY=True): # scrollY는 유지 (혹시 5줄 넘을까봐)
                    util_funcs['create_table_with_data'](rel_table_tag, related_vars_df, parent_df_for_widths=related_vars_df)
            else:
                dpg.add_text("No specific related variables found to display.", parent=parent_tag)
    else: # _get_top_correlated_vars가 빈 리스트([])를 반환할 수도 있음 (수정된 함수 기준)
        dpg.add_text("No correlation/association data determined.", parent=parent_tag)


def _create_single_var_plot(parent_group_tag: str, series: pd.Series, group_by_target_series: pd.Series = None, analysis_type_override:str=None,
                            grouped_plot_preference:str="KDE_AND_HIST"):
    plot_height = 230 + 60 # 높이 증가
    plot_label = f"Distribution: {series.name}"

    is_grouped_plotting = group_by_target_series is not None and \
                          group_by_target_series.nunique(dropna=False) >= 2 and \
                          series.name != group_by_target_series.name and \
                          (analysis_type_override != "ForceCategoricalForBinaryNumeric" or not pd.api.types.is_numeric_dtype(series.dtype))

    # 만약 시리즈 이름과 그룹바이 타겟 시리즈 이름이 같다면, 그룹핑 플롯팅을 비활성화
    # 또는 사용자에게 알림을 표시하고 플롯 생성을 건너뛸 수 있습니다.
    if group_by_target_series is not None and series.name == group_by_target_series.name:
        plot_label += " (Overall)" # 그룹핑 없이 전체 분포로 표시함을 명시
        # 그룹핑 플로팅을 비활성화하고, group_by_target_series를 None으로 설정하여 단일 플롯 로직을 따르도록 함
        is_grouped_plotting = False
        group_by_target_series = None # 단일 변수 플롯 로직을 타도록 강제
        # 또는 아래처럼 메시지 표시 후 반환
        # dpg.add_text(f"Plot for '{series.name}' grouped by itself is not generated.\nDisplaying overall distribution.", parent=parent_group_tag, color=(255, 200, 0))
        # is_grouped_plotting = False # 이후 로직에서 그룹핑 없이 처리하도록
        # group_by_target_series = None # 확실히 None으로 설정
        # (만약 여기서 완전히 종료하고 싶다면, legend_tag 확인 후 삭제하고 return)


    if is_grouped_plotting: # 이 조건은 위에서 series.name != group_by_target_series.name 이 이미 반영됨
        plot_label += f" (Grouped by {group_by_target_series.name})"

    plot_tag = dpg.generate_uuid()
    with dpg.plot(label=plot_label, height=plot_height, width=-1, parent=parent_group_tag, tag=plot_tag):
        xaxis_tag = dpg.add_plot_axis(dpg.mvXAxis, label=series.name, lock_min=False, lock_max=False, auto_fit=True)
        yaxis_tag = dpg.generate_uuid()
        dpg.add_plot_axis(dpg.mvYAxis, label="Density / Frequency", tag=yaxis_tag, lock_min=False, lock_max=False, auto_fit=True)
        
        legend_tag = dpg.add_plot_legend(parent=plot_tag, location=8, outside=True)

        series_cleaned_for_plot_initial = series.dropna()
        series_cleaned_for_plot = series_cleaned_for_plot_initial.replace([np.inf, -np.inf], np.nan).dropna()

        if len(series_cleaned_for_plot) < 2:
            dpg.add_text("Not enough valid data points for plot.", parent=yaxis_tag, color=(255, 200, 0))
            if dpg.does_item_exist(legend_tag): 
                dpg.delete_item(legend_tag)
            return

        is_effectively_categorical_for_plot = (analysis_type_override == "ForceCategoricalForBinaryNumeric" or
                                              pd.api.types.is_categorical_dtype(series.dtype) or
                                              series.dtype == 'object' or
                                              series.nunique(dropna=False) < 5)

        if pd.api.types.is_numeric_dtype(series.dtype) and not is_effectively_categorical_for_plot:
            if is_grouped_plotting and group_by_target_series is not None: # group_by_target_series가 None이 아닌지 한번 더 확인
                unique_target_groups_numeric = sorted(group_by_target_series.dropna().unique())
                
                base_colors = [(0, 110, 255), (255, 120, 0), (0, 170, 0), (200, 0, 0), (150, 50, 200)]
                alpha_value = 150

                for idx, group_name in enumerate(unique_target_groups_numeric):
                    # series.name == group_by_target_series.name 인 경우는 이미 위에서 처리되어
                    # is_grouped_plotting = False 가 되므로 이 루프에 진입하지 않거나,
                    # group_by_target_series가 None이 되어 이 블록 전체를 건너뜀.
                    group_data_initial = series_cleaned_for_plot[group_by_target_series == group_name]
                    group_data_cleaned = group_data_initial.replace([np.inf, -np.inf], np.nan).dropna()

                    # KDE를 위한 데이터 유효성 검사 강화
                    if len(group_data_cleaned) < 2 or group_data_cleaned.nunique() < 2: # 분산이 없거나 데이터가 너무 적으면 KDE 스킵
                        print(f"INFO: Group '{group_name}' for var '{series.name}' has insufficient data or no variance for KDE. Skipping plot for this group.")
                        if grouped_plot_preference == "KDE": # KDE 선택 시에만 메시지 또는 처리
                             # dpg.add_text(f"KDE skipped for group '{group_name}' (no variance/data)", parent=yaxis_tag, color=(255,165,0))
                            continue # 다음 그룹으로
                        # 히스토그램은 단일 값도 표시 가능하므로 계속 진행될 수 있음

                    current_color = base_colors[idx % len(base_colors)]
                    
                    try:
                        if grouped_plot_preference == "KDE":
                            # KDE를 위한 추가적인 데이터 유효성 검사
                            if group_data_cleaned.nunique() < 2: # 이중 체크
                                print(f"Skipping KDE for group '{group_name}' of var '{series.name}' due to no variance.")
                                continue
                            kde = stats.gaussian_kde(group_data_cleaned)
                            kde_min = group_data_cleaned.min(); kde_max = group_data_cleaned.max()
                            padding = (kde_max - kde_min) * 0.05 if (kde_max - kde_min) > 0 else 0.1
                            x_vals = np.linspace(kde_min - padding, kde_max + padding, 100)
                            y_vals = kde(x_vals)
                            line_series_tag = dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), label=f"KDE (T={group_name})", parent=yaxis_tag)
                            # 색상 적용 (간단한 방법은 아니지만, 테마나 다른 방법으로 시도 가능)

                        elif grouped_plot_preference == "Histogram":
                            hist_series = dpg.add_histogram_series(group_data_cleaned.tolist(), label=f"Hist (T={group_name})",
                                                                  density=True, bins=-1, parent=yaxis_tag, bar_scale=0.9)
                            series_theme_tag = dpg.generate_uuid()
                            with dpg.theme(tag=series_theme_tag):
                                with dpg.theme_component(dpg.mvHistogramSeries, parent=series_theme_tag):
                                    dpg.add_theme_color(dpg.mvPlotCol_Fill, (current_color[0], current_color[1], current_color[2], alpha_value), 
                                                        category=dpg.mvThemeCat_Plots)
                            dpg.bind_item_theme(hist_series, series_theme_tag)
                    except Exception as e_plot:
                        error_msg_group = f"Plot Error (Group '{group_name}', Type: {grouped_plot_preference})"
                        dpg.add_text(error_msg_group, parent=yaxis_tag, color=(255,100,100))
                        print(f"{error_msg_group} for var '{series.name}': {e_plot}\nData sample: {group_data_cleaned.head().tolist()}")
                        traceback.print_exc()
            
            else: # 단일 숫자형 시리즈 (그룹핑 없음)
                if series_cleaned_for_plot.nunique() < 2 :
                    dpg.add_text("No variance in data for histogram/density plot.", parent=yaxis_tag, color=(255,200,0))
                    if dpg.does_item_exist(legend_tag): dpg.delete_item(legend_tag)
                    return

                try:
                    dpg.add_histogram_series(series_cleaned_for_plot.tolist(), label="Histogram",
                                             density=True, bins=-1, parent=yaxis_tag, bar_scale=1.0)
                except Exception as e_hist:
                    dpg.add_text(f"Histogram Error", parent=yaxis_tag, color=(255,0,0))
                    print(f"Error plotting histogram for var '{series.name}': {e_hist}\nData sample: {series_cleaned_for_plot.head().tolist()}")
                    traceback.print_exc()
                
                try:
                    # KDE를 위한 데이터 유효성 검사
                    if series_cleaned_for_plot.nunique() < 2: # 이중 체크
                        print(f"Skipping KDE for var '{series.name}' due to no variance.")
                    else:
                        kde = stats.gaussian_kde(series_cleaned_for_plot)
                        kde_min = series_cleaned_for_plot.min(); kde_max = series_cleaned_for_plot.max()
                        padding = (kde_max - kde_min) * 0.05 if (kde_max - kde_min) > 0 else 0.1
                        x_vals = np.linspace(kde_min - padding, kde_max + padding, 150)
                        y_vals = kde(x_vals)
                        dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), label="KDE", parent=yaxis_tag)
                except Exception as e_kde:
                    dpg.add_text(f"KDE Error", parent=yaxis_tag, color=(255,100,100))
                    print(f"Error plotting KDE for var '{series.name}': {e_kde}\nData sample: {series_cleaned_for_plot.head().tolist()}")
                    traceback.print_exc()
        else: # 범주형 데이터 플롯 로직 (이 부분은 크게 변경되지 않음)
            # ... (기존 범주형 플롯 로직) ...
            top_n_categories = 10
            if is_grouped_plotting and group_by_target_series is not None and \
               group_by_target_series.nunique(dropna=False) <= 5 : 
                
                if dpg.does_item_exist(legend_tag): dpg.delete_item(legend_tag) # 범례 먼저 삭제 시도
                
                unique_target_groups_categorical = sorted(group_by_target_series.dropna().unique())
                value_counts_overall = series_cleaned_for_plot.value_counts(dropna=False).nlargest(top_n_categories)
                categories_to_plot = value_counts_overall.index.tolist()
                
                num_groups = len(unique_target_groups_categorical)
                bar_width_total = 0.8 
                bar_width_single = bar_width_total / num_groups if num_groups > 0 else bar_width_total
                x_positions = np.arange(len(categories_to_plot))

                for i, group_name in enumerate(unique_target_groups_categorical):
                    group_series = series_cleaned_for_plot[group_by_target_series == group_name]
                    group_value_counts = group_series.value_counts(dropna=False)
                    
                    y_values = [group_value_counts.get(cat, 0) for cat in categories_to_plot]
                    current_x_positions = x_positions - (bar_width_total / 2) + (i * bar_width_single) + (bar_width_single / 2)
                    dpg.add_bar_series(current_x_positions.tolist(), y_values, weight=bar_width_single, label=f"{group_name}", parent=yaxis_tag)

                if categories_to_plot and dpg.does_item_exist(xaxis_tag):
                     dpg.set_axis_ticks(xaxis_tag, tuple(zip([str(c) for c in categories_to_plot], x_positions))) # 레이블 문자열 변환
                if num_groups > 0 and dpg.does_item_exist(plot_tag): # plot_tag로 부모 지정
                    # 범례를 새로 추가하거나, 이미 있는 legend_tag를 활용 (여기서는 새로 추가)
                    dpg.add_plot_legend(parent=plot_tag, location=8, outside=True) 
            else: 
                if dpg.does_item_exist(legend_tag): dpg.delete_item(legend_tag)
                value_counts_data = series_cleaned_for_plot.value_counts(dropna=False).nlargest(top_n_categories)
                x_pos = list(range(len(value_counts_data)))
                bar_labels = [str(val) for val in value_counts_data.index.tolist()]
                dpg.add_bar_series(x_pos, value_counts_data.values.tolist(), weight=0.7, label="Frequency", parent=yaxis_tag)
                if bar_labels and dpg.does_item_exist(xaxis_tag): dpg.set_axis_ticks(xaxis_tag, tuple(zip(bar_labels, x_pos)))
                
def _apply_sva_filters_and_run(main_callbacks: dict):
    print("DEBUG: _apply_sva_filters_and_run CALLED.") # 함수 호출 시작 확인

    # --- 1. 필수 데이터 및 UI 요소 가져오기 ---
    current_df = main_callbacks['get_current_df']()
    util_funcs = main_callbacks['get_util_funcs']()
    target_var = main_callbacks['get_selected_target_variable']()
    # get_column_analysis_types 콜백이 main_callbacks에 확실히 존재하고 올바른 딕셔너리를 반환하는지 확인 필요
    analysis_types_dict_local = main_callbacks.get('get_column_analysis_types', lambda: {})()
    if not isinstance(analysis_types_dict_local, dict): # 반환 값 타입 체크
        print("ERROR: 'get_column_analysis_types' did not return a dictionary. Using empty dict.")
        analysis_types_dict_local = {}

    results_child_window_tag = TAG_SVA_RESULTS_CHILD_WINDOW
    progress_modal_tag = TAG_SVA_PROGRESS_MODAL
    progress_text_tag = TAG_SVA_PROGRESS_TEXT

    # --- 2. 이전 결과 지우기 및 진행률 표시창 준비 ---
    if dpg.does_item_exist(results_child_window_tag):
        dpg.delete_item(results_child_window_tag, children_only=True)
    else:
        print(f"ERROR: SVA results child window '{results_child_window_tag}' not found.")
        _show_alert_modal("UI Error", f"SVA result display area (tag: {results_child_window_tag}) is missing. Cannot proceed.")
        return

    if not dpg.does_item_exist(progress_modal_tag):
        with dpg.window(label="Processing SVA", modal=True, show=False, tag=progress_modal_tag,
                        no_close=True, no_title_bar=True, pos=[500,400], width=350, height=70):
            dpg.add_text("Analyzing variables, please wait...", tag=progress_text_tag)

    dpg.configure_item(progress_modal_tag, show=True)
    dpg.set_value(progress_text_tag, "SVA: Preparing analysis...")
    dpg.split_frame()

    # --- 3. 데이터 유효성 검사 ---
    if current_df is None:
        dpg.add_text("Load data first to perform Single Variable Analysis.", parent=results_child_window_tag)
        dpg.configure_item(progress_modal_tag, show=False)
        return

    # --- 4. UI에서 현재 필터 및 그룹핑 옵션 가져오기 ---
    filter_strength_selected = dpg.get_value(TAG_SVA_FILTER_STRENGTH_RADIO) if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO) else "Weak (Exclude obvious non-analytical)"
    group_by_target_flag = dpg.get_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX) if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX) else False
    
    # 그룹핑 시 플롯 타입 가져오기
    grouped_plot_preference = "KDE" # 기본값
    if group_by_target_flag and dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO) and dpg.is_item_shown(TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
        grouped_plot_preference = dpg.get_value(TAG_SVA_GROUPED_PLOT_TYPE_RADIO)
    
    target_series_for_grouping = None

    # --- 5. 그룹핑 옵션 유효성 검사 및 설정 ---
    # 이 부분이 그룹핑 체크 해제와 직접적으로 관련된 로직입니다.
    if group_by_target_flag: # 사용자가 UI에서 그룹핑을 선택한 경우
        print(f"DEBUG: User selected 'Group by Target'. Initial target_var: '{target_var}'")
        if target_var and target_var in current_df.columns:
            unique_target_values = current_df[target_var].nunique(dropna=False)
            # 수정된 조건: 타입에 관계없이 고유값 개수(2~10개)만으로 판단.
            condition_type_met = True # target_var가 유효한 컬럼이기만 하면 타입은 OK로 간주.
            condition_nunique_met = (unique_target_values >= 2 and unique_target_values <= 10)

            if condition_type_met and condition_nunique_met:
                target_series_for_grouping = current_df[target_var]
                print(f"DEBUG: Grouping by target '{target_var}' WILL BE ATTEMPTED. Conditions met (Unique values: {unique_target_values}).")
            else:
                reasons_for_failure = []
                # if not condition_type_met: # 이 조건은 이제 거의 항상 참이므로 제거하거나 단순화 가능
                #     reasons_for_failure.append(f"Target variable '{target_var}' type not suitable.")
                if not condition_nunique_met:
                    reasons_for_failure.append(f"Unique values count {unique_target_values} is not between 2-10")
                
                alert_message = (f"Target variable '{target_var}' is not suitable for grouping.\n"
                                 f"Reason(s): {'; '.join(reasons_for_failure)}.\n"
                                 f"Grouping has been disabled.")
                _show_alert_modal("Grouping Warning", alert_message)
                
                if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
                    dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
                group_by_target_flag = False
                target_series_for_grouping = None
                print(f"DEBUG: Grouping disabled for target '{target_var}'. Reasons: {'; '.join(reasons_for_failure)}")
        else:
            # (기존 'Target variable not selected or is invalid...' 로직 유지)
            alert_message = "Target variable not selected or is invalid for grouping. Grouping disabled."
            _show_alert_modal("Grouping Info", alert_message)
            if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
                dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
            group_by_target_flag = False
            target_series_for_grouping = None
            print(f"DEBUG: Grouping disabled. Target variable not selected or does not exist in DataFrame.")
    else:
        print(f"DEBUG: User did not select 'Group by Target' checkbox.")
        target_series_for_grouping = None
        group_by_target_flag = False

    # --- 6. 변수 필터링 실행 ---
    # (이하 _get_filtered_variables 호출 및 결과 처리 로직은 이전 답변의 최종본과 동일하게 유지)
    # ...
    dpg.set_value(progress_text_tag, "SVA: Filtering variables...")
    dpg.split_frame()
    # _get_filtered_variables는 (filtered_cols_list, actual_filter_applied_str) 튜플을 반환해야 합니다.
    # 만약 이 함수가 아직 수정되지 않았다면, 이전 답변을 참고하여 수정해야 합니다.
    # 여기서는 해당 함수가 수정되었다고 가정합니다.
    filtered_cols, actual_filter_applied = _get_filtered_variables(current_df, filter_strength_selected, main_callbacks, target_var)
    print(f"DEBUG: User selected filter: '{filter_strength_selected}'. Actual filter applied by _get_filtered_variables: '{actual_filter_applied}'. Num vars: {len(filtered_cols)}")

    # --- 7. Fallback 조건 또는 필터링 결과 없음 처리 ---
    conditions_not_met_for_strong_medium = (filter_strength_selected in ["Medium (Top 11-20 relevant)", "Strong (Top 5-10 relevant)"]) and \
                                           (not filtered_cols) # _get_filtered_variables가 빈 리스트를 반환하여 조건 미충족 시

    if conditions_not_met_for_strong_medium:
        print(f"INFO: SVA run aborted. User selected '{filter_strength_selected}', but conditions were not met (e.g., no target, no relevant vars).")
        # _get_filtered_variables 내부에서 이미 알림 모달이 표시되었을 것입니다.
        dpg.add_text(f"SVA not performed for '{filter_strength_selected}'.\nConditions for this filter were not met. Please check alerts/console for details.",
                     parent=results_child_window_tag, wrap=dpg.get_item_width(results_child_window_tag)-20 if dpg.get_item_width(results_child_window_tag) > 0 else 400,
                     color=(255,165,0))
        dpg.configure_item(progress_modal_tag, show=False)
        return

    if not filtered_cols: # Fallback이 아니더라도, 최종적으로 필터링된 변수가 없는 경우
        dpg.add_text(f"No variables to display based on the filter: '{actual_filter_applied}'.", parent=results_child_window_tag)
        dpg.configure_item(progress_modal_tag, show=False)
        return

    # --- 9. 각 변수에 대한 분석 및 UI 생성 (3단 레이아웃) ---
    # (이하 3단 레이아웃 생성 및 _create_sva_basic_stats_table, _create_sva_advanced_relations_table, _create_single_var_plot 호출 로직은
    #  이전 답변의 최종본과 동일하게 유지합니다. 여기서 group_by_target_flag와 target_series_for_grouping이 올바르게 사용됩니다.)
    # ...
    total_vars = len(filtered_cols)
    print(f"DEBUG: Starting SVA loop for {total_vars} variables using '{actual_filter_applied}' filter. Grouping flag: {group_by_target_flag}")

    for i, col_name in enumerate(filtered_cols):
        if not dpg.is_dearpygui_running(): break
        dpg.set_value(progress_text_tag, f"SVA: Analyzing {col_name} ({i+1}/{total_vars})")
        dpg.split_frame()

        var_section_tag_str = "".join(filter(str.isalnum, str(col_name)))
        var_section_tag = f"{TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX}{var_section_tag_str}_{i}"

        col_analysis_type = analysis_types_dict_local.get(col_name, str(current_df[col_name].dtype))
        analysis_override_for_sva = None
        # 실제 적용된 필터(actual_filter_applied)를 기준으로 override 로직 적용
        if actual_filter_applied == "None (All variables)":
            unique_vals_str_set = set(current_df[col_name].dropna().astype(str).unique())
            is_binary_numeric_for_none_filter = unique_vals_str_set <= {'0', '1', '0.0', '1.0'} and len(unique_vals_str_set) <= 2
            if "Numeric" in col_analysis_type and is_binary_numeric_for_none_filter:
                analysis_override_for_sva = "ForceCategoricalForBinaryNumeric"
        
        with dpg.group(tag=var_section_tag, parent=results_child_window_tag):
            header_text_width = dpg.get_item_width(results_child_window_tag) - 20 if dpg.get_item_width(results_child_window_tag) > 20 else 500
            dpg.add_text(f"Variable: {util_funcs['format_text_for_display'](col_name, 60)} ({i+1}/{total_vars})", color=(255, 255, 0), wrap=header_text_width)
            dpg.add_text(f"Identified Type: {col_analysis_type} (Actual Dtype: {str(current_df[col_name].dtype)})", wrap=header_text_width)
            dpg.add_spacer(height=5)
            
            available_width = dpg.get_item_width(results_child_window_tag)
            if available_width <= 0 : available_width = 900 
            
            spacing = 10 
            col_1_width = int(available_width * 0.30) - spacing
            col_2_width = int(available_width * 0.30) - spacing
            
            min_col_width = 200 
            col_1_width = max(min_col_width, col_1_width)
            col_2_width = max(min_col_width, col_2_width)

            with dpg.group(horizontal=True): # 전체 가로 그룹
                # Column 1: Basic Stats
                with dpg.group(width=col_1_width) as col1_group_tag:
                    _create_sva_basic_stats_table(col1_group_tag, current_df[col_name], util_funcs, analysis_override_for_sva)
                
                # Column 2: Advanced Stats & Relations
                with dpg.group(width=col_2_width) as col2_group_tag:
                    _create_sva_advanced_relations_table(col2_group_tag, current_df[col_name], current_df, util_funcs, col_2_width)

                # Column 3: Plot
                with dpg.group() as col3_group_tag:
                    print(f"  DEBUG [{col_name}]: Creating plot in group {col3_group_tag}. Grouping: {group_by_target_flag}, Plot Pref: {grouped_plot_preference if group_by_target_flag else 'KDE_AND_HIST'}")
                    _create_single_var_plot(col3_group_tag, current_df[col_name], 
                                            target_series_for_grouping, # group_by_target_flag에 따라 None 또는 시리즈 값 가짐
                                            analysis_override_for_sva,
                                            grouped_plot_preference if group_by_target_flag else "KDE_AND_HIST" # 다섯 번째 인자
                                           )
            dpg.add_separator()
            dpg.add_spacer(height=10)

    # --- 10. 완료 후 진행률 표시창 숨기기 ---
    dpg.configure_item(progress_modal_tag, show=False)
    print("DEBUG: SVA processing finished successfully.")


# --- Main UI Creation & Update ---
def _sva_group_by_target_callback(sender, app_data, user_data):
    """'Group by Target' 체크박스 콜백 함수"""
    main_callbacks = user_data
    is_checked = dpg.get_value(sender)
    plot_type_radio_tag = TAG_SVA_GROUPED_PLOT_TYPE_RADIO

    if dpg.does_item_exist(plot_type_radio_tag):
        dpg.configure_item(plot_type_radio_tag, show=is_checked)
        if not is_checked:
            dpg.set_value(plot_type_radio_tag, "KDE") # 체크 해제 시 기본값으로

    # 옵션 변경 시 SVA 자동 재실행 (사용자 요청에 따라 이 부분은 유지 또는 버튼 클릭으로만 실행)
    # _apply_sva_filters_and_run(main_callbacks) # 필요시 이 라인 주석 해제 또는 유지

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    main_callbacks['register_step_group_tag'](step_name, TAG_EDA_GROUP)
    with dpg.group(tag=TAG_EDA_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---"); dpg.add_separator()
        with dpg.tab_bar(tag=TAG_EDA_MAIN_TAB_BAR):
            with dpg.tab(label="Single Variable Analysis", tag=TAG_SVA_TAB):
                with dpg.group(horizontal=True): 
                    with dpg.group(width=280): # 필터 옵션 그룹
                        dpg.add_text("Variable Filter")
                        dpg.add_radio_button(
                            items=["Strong (Top 5-10 relevant)", "Medium (Top 11-20 relevant)", "Weak (Exclude obvious non-analytical)", "None (All variables)"],
                            tag=TAG_SVA_FILTER_STRENGTH_RADIO, default_value="Weak (Exclude obvious non-analytical)"
                            # callback은 Run 버튼으로 통합
                        )
                        # ... (기존 필터 설명 텍스트)
                        dpg.add_spacer(height=5) 
                        dpg.add_text("Filter Algorithm (Simplified):", wrap=250)
                        dpg.add_text("- Strong/Medium: Relevance to Target (if set) or other heuristics.", wrap=250)
                        dpg.add_text("- Weak: Excludes single-value & binary numeric variables.", wrap=250)
                    
                    dpg.add_spacer(width=10) 
                    
                    with dpg.group(): # 그룹핑, 플롯 타입 및 실행 버튼 그룹
                        dpg.add_text("Grouping Option")
                        dpg.add_checkbox(label="Group by Target (Categorical Target, 2-5 Categories)",
                                         tag=TAG_SVA_GROUP_BY_TARGET_CHECKBOX, default_value=False,
                                         user_data=main_callbacks,
                                         callback=_sva_group_by_target_callback # 콜백 연결
                                         )
                        # 그룹핑 시 플롯 타입 선택 라디오 버튼 (초기에는 숨김)
                        dpg.add_radio_button(items=["KDE", "Histogram"], 
                                             tag=TAG_SVA_GROUPED_PLOT_TYPE_RADIO, 
                                             default_value="KDE", horizontal=True, show=False)
                                             # user_data=main_callbacks,
                                             # callback=lambda s,a,u: _apply_sva_filters_and_run(u) # 필요시 콜백 추가

                        dpg.add_spacer(height=10)
                        dpg.add_button(label="Run Single Variable Analysis", tag=TAG_SVA_RUN_BUTTON,
                                       callback=lambda: _apply_sva_filters_and_run(main_callbacks),
                                       width=-1, height=30)
                dpg.add_separator()
                with dpg.child_window(tag=TAG_SVA_RESULTS_CHILD_WINDOW, border=True):
                    dpg.add_text("Select filter options and click 'Run Single Variable Analysis'.")

            # --- MVA 탭 UI (이전과 동일하게 유지) ---
            with dpg.tab(label="Multivariate Analysis", tag=TAG_MVA_TAB):
                # (MVA 탭 내용은 사용자 제공 파일과 동일하게 유지)
                with dpg.tab_bar(tag=TAG_MVA_SUB_TAB_BAR):
                    with dpg.tab(label="Correlation Analysis", tag=TAG_MVA_CORR_TAB):
                        dpg.add_button(label="Run Correlation Analysis (Numeric Vars)", tag=TAG_MVA_CORR_RUN_BUTTON,
                                       callback=lambda: _run_correlation_analysis(main_callbacks['get_current_df'](), main_callbacks['get_util_funcs']()))
                        dpg.add_child_window(tag=TAG_MVA_CORR_HEATMAP_PLOT, border=True, height=480)
                    with dpg.tab(label="Pair Plot", tag=TAG_MVA_PAIRPLOT_TAB):
                        dpg.add_text("Select numeric variables for Pair Plot (Max 5 recommended).")
                        dpg.add_listbox(tag=TAG_MVA_PAIRPLOT_VAR_SELECTOR, width=-1, num_items=8)
                        dpg.add_combo(label="Hue (Optional Categorical Var, <8 Categories)", tag=TAG_MVA_PAIRPLOT_HUE_COMBO, width=300)
                        dpg.add_button(label="Generate Pair Plot", tag=TAG_MVA_PAIRPLOT_RUN_BUTTON,
                                       callback=lambda: _run_pair_plot_analysis(
                                           main_callbacks['get_current_df'](),
                                           dpg.get_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR),
                                           dpg.get_value(TAG_MVA_PAIRPLOT_HUE_COMBO),
                                           main_callbacks['get_util_funcs']()
                                       ))
                        dpg.add_child_window(tag=TAG_MVA_PAIRPLOT_RESULTS_GROUP, border=True, height=-1)
                    with dpg.tab(label="Target Variable Analysis", tag=TAG_MVA_TARGET_TAB):
                        dpg.add_text("Analyze relationship between features and the selected target variable.", tag=TAG_MVA_TARGET_INFO_TEXT, wrap=-1)
                        with dpg.group(horizontal=True):
                            dpg.add_combo(label="Feature Variable", tag=TAG_MVA_TARGET_FEATURE_COMBO, width=300)
                            dpg.add_button(label="Analyze vs Target", tag=TAG_MVA_TARGET_RUN_BUTTON,
                                           callback=lambda: _run_target_variable_analysis(
                                               main_callbacks['get_current_df'](),
                                               main_callbacks['get_selected_target_variable'](),
                                               dpg.get_value(TAG_MVA_TARGET_FEATURE_COMBO),
                                               main_callbacks['get_util_funcs']()
                                           ))
                        dpg.add_separator()
                        dpg.add_child_window(tag=TAG_MVA_TARGET_RESULTS_GROUP, border=True, height=-1)
    
    # updater 등록 방식은 이전 답변에서 제안된 대로 유지 (main_app.py 호출 방식과 일치)
    main_callbacks['register_module_updater'](step_name, lambda df_arg, mc_arg: update_ui(df_arg, mc_arg))
    update_ui(main_callbacks['get_current_df'](), main_callbacks)


def update_ui(current_df: pd.DataFrame, main_callbacks: dict):
    # (사용자 제공 파일의 내용과 동일하게 유지 - MVA 탭의 선택지 업데이트 로직)
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_EDA_GROUP): return
    print("DEBUG: EDA Module: update_ui called to refresh selectors.") # DEBUG 프린트 변경
    
    if current_df is None:
        if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
            dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
            dpg.add_text("Load data to perform Single Variable Analysis.", parent=TAG_SVA_RESULTS_CHILD_WINDOW)
        if dpg.does_item_exist(TAG_MVA_PAIRPLOT_VAR_SELECTOR): dpg.configure_item(TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=[])
        if dpg.does_item_exist(TAG_MVA_PAIRPLOT_HUE_COMBO): dpg.configure_item(TAG_MVA_PAIRPLOT_HUE_COMBO, items=[""])
        if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO): dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=[])
        if dpg.does_item_exist(TAG_MVA_TARGET_INFO_TEXT): dpg.set_value(TAG_MVA_TARGET_INFO_TEXT, "Select a global target variable and load data.")
        return

    all_columns = current_df.columns.tolist()
    numeric_cols = _get_numeric_cols(current_df)
    # MVA Hue 콤보에는 모든 범주형 후보 표시 (EDA 필터링과 무관)
    categorical_cols_for_mva_hue = [""] + _get_categorical_cols(current_df, max_unique_for_cat=10) # MVA Hue는 고유값 10개까지 허용

    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_VAR_SELECTOR):
        current_selection = dpg.get_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR)
        dpg.configure_item(TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=numeric_cols)
        if current_selection and isinstance(current_selection, list) and all(item in numeric_cols for item in current_selection):
            try: dpg.set_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR, current_selection)
            except Exception: pass 
            
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_HUE_COMBO):
        current_hue = dpg.get_value(TAG_MVA_PAIRPLOT_HUE_COMBO)
        dpg.configure_item(TAG_MVA_PAIRPLOT_HUE_COMBO, items=categorical_cols_for_mva_hue)
        if current_hue and current_hue in categorical_cols_for_mva_hue:
            dpg.set_value(TAG_MVA_PAIRPLOT_HUE_COMBO, current_hue)
        elif categorical_cols_for_mva_hue: dpg.set_value(TAG_MVA_PAIRPLOT_HUE_COMBO, categorical_cols_for_mva_hue[0])

    selected_target_var = main_callbacks['get_selected_target_variable']()
    if dpg.does_item_exist(TAG_MVA_TARGET_INFO_TEXT):
        if selected_target_var and selected_target_var in all_columns:
            dpg.set_value(TAG_MVA_TARGET_INFO_TEXT, f"Analyzing features against Target: '{selected_target_var}'")
            if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO):
                feature_candidates = [col for col in all_columns if col != selected_target_var]
                current_feature_target = dpg.get_value(TAG_MVA_TARGET_FEATURE_COMBO)
                dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=feature_candidates)
                if current_feature_target and current_feature_target in feature_candidates:
                     dpg.set_value(TAG_MVA_TARGET_FEATURE_COMBO, current_feature_target)
                elif feature_candidates: dpg.set_value(TAG_MVA_TARGET_FEATURE_COMBO, feature_candidates[0])
        else:
            dpg.set_value(TAG_MVA_TARGET_INFO_TEXT, "Select a global target variable (top panel) to enable this analysis.")
            if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO): dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=[])
    print("DEBUG: EDA UI selectors updated.")


# --- MVA Helper Functions (사용자 제공 파일의 MVA 함수들 유지) ---
# _run_correlation_analysis, _run_pair_plot_analysis, _run_target_variable_analysis
# (이 함수들의 상세 구현은 사용자 제공 파일에 있는 것을 그대로 사용한다고 가정합니다.
#  만약 이 함수들에도 문제가 있다면 별도로 검토해야 합니다.)

def _run_correlation_analysis(df: pd.DataFrame, util_funcs: dict):
    # ... (이전 답변의 _run_correlation_analysis 전체 구현 내용) ...
    if not dpg.is_dearpygui_running(): return
    heatmap_plot_tag = TAG_MVA_CORR_HEATMAP_PLOT; corr_table_tag = TAG_MVA_CORR_TABLE; corr_tab_tag = TAG_MVA_CORR_TAB
    if dpg.does_item_exist(heatmap_plot_tag): dpg.delete_item(heatmap_plot_tag, children_only=True)
    if dpg.does_item_exist(corr_table_tag): dpg.delete_item(corr_table_tag)
    if df is None: dpg.add_text("Load data first.", parent=heatmap_plot_tag if dpg.does_item_exist(heatmap_plot_tag) else corr_tab_tag); return
    numeric_cols = _get_numeric_cols(df)
    if len(numeric_cols) < 2: dpg.add_text("Not enough numeric columns.", parent=heatmap_plot_tag if dpg.does_item_exist(heatmap_plot_tag) else corr_tab_tag); return
    corr_matrix = df[numeric_cols].corr(method='pearson'); heatmap_data = corr_matrix.values.flatten().tolist()
    rows, cols = corr_matrix.shape; col_labels = corr_matrix.columns.tolist()
    with dpg.plot(label="Correlation Heatmap", height=450, width=-1, parent=heatmap_plot_tag, equal_aspects=True):
        xaxis = dpg.add_plot_axis(dpg.mvXAxis, label=""); yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="")
        if col_labels: dpg.set_axis_ticks(xaxis, tuple(zip(col_labels, list(range(cols))))); dpg.set_axis_ticks(yaxis, tuple(zip(col_labels, list(range(rows)))))
        dpg.add_heat_series(heatmap_data, rows=rows, cols=cols, scale_min=-1.0, scale_max=1.0, format='%.2f', parent=yaxis, show_tooltips=True)
    parent_for_corr_table = corr_tab_tag
    dpg.add_text("Highly Correlated Pairs (|Correlation| > 0.7):", parent=parent_for_corr_table)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7: high_corr_pairs.append({"Variable 1": corr_matrix.columns[i], "Variable 2": corr_matrix.columns[j], "Correlation": f"{corr_matrix.iloc[i, j]:.3f}"})
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        with dpg.table(header_row=True, tag=corr_table_tag, parent=parent_for_corr_table, resizable=True, policy=dpg.mvTable_SizingFixedFit, scrollY=True, height=150, borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
            util_funcs['create_table_with_data'](corr_table_tag, high_corr_df)
    else: dpg.add_text("No pairs with |correlation| > 0.7 found.", parent=parent_for_corr_table)

def _run_pair_plot_analysis(df: pd.DataFrame, selected_vars: list, hue_var: str, util_funcs: dict):
    # ... (이전 답변의 _run_pair_plot_analysis 전체 구현 내용 - DPG 네이티브 시도는 복잡하므로 주로 메시지 표시) ...
    results_group = TAG_MVA_PAIRPLOT_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group) or df is None: return
    dpg.delete_item(results_group, children_only=True)
    if not selected_vars or not isinstance(selected_vars, list) or len(selected_vars) < 2:
        dpg.add_text("Please select 2 or more numeric variables for the Pair Plot.", parent=results_group); return
    max_pair_plot_vars = 5
    if len(selected_vars) > max_pair_plot_vars:
        _show_alert_modal("Pair Plot Limit", f"Too many variables ({len(selected_vars)}). Displaying first {max_pair_plot_vars}.")
        selected_vars = selected_vars[:max_pair_plot_vars]
    valid_selected_vars = [var for var in selected_vars if var in df.columns and pd.api.types.is_numeric_dtype(df[var].dtype)]
    if len(valid_selected_vars) < 2: dpg.add_text("Not enough valid numeric variables selected.", parent=results_group); return
    selected_vars = valid_selected_vars
    hue_categories = None; hue_series = None
    if hue_var and hue_var in df.columns:
        hue_series = df[hue_var]
        if hue_series.nunique(dropna=False) > 7:
            _show_alert_modal("Hue Variable Warning", f"Hue var '{hue_var}' has {hue_series.nunique()} categories. Max 7 recommended. Hue disabled.")
            hue_var = None; hue_series = None
        else: hue_categories = sorted(hue_series.dropna().unique())
    n_vars = len(selected_vars); plot_cell_size = 200
    dpg.add_text(f"Pair Plot for: {', '.join(selected_vars)}" + (f" (Hue: {hue_var})" if hue_var else ""), parent=results_group)
    # Simplified DPG Pair Plot
    with dpg.child_window(parent=results_group, border=False, autosize_x=True, autosize_y=True): # 스크롤을 위해
        for i in range(n_vars):
            with dpg.group(horizontal=True):
                for j in range(n_vars):
                    var_y = selected_vars[i]; var_x = selected_vars[j]
                    plot_label = f"{var_y} vs {var_x}" if i != j else f"Dist: {var_x}"
                    with dpg.plot(width=plot_cell_size, height=plot_cell_size, label=plot_label):
                        pxaxis = dpg.add_plot_axis(dpg.mvXAxis, label=var_x if i == n_vars - 1 else "", no_tick_labels=(i != n_vars -1))
                        pyaxis = dpg.add_plot_axis(dpg.mvYAxis, label=var_y if j == 0 else "", no_tick_labels=(j != 0))
                        if i == j: # Diagonal
                            s_diag = df[var_x].dropna()
                            if not s_diag.empty and len(s_diag) > 1:
                                hist_d = np.histogram(s_diag, bins='auto', density=True)
                                dpg.add_stair_series(hist_d[1], np.append(hist_d[0],0), label="Hist", parent=pyaxis)
                        else: # Off-diagonal
                            sx_data = df[var_x]; sy_data = df[var_y]
                            if hue_series is not None and hue_categories:
                                for cat_idx, cat_val in enumerate(hue_categories):
                                    mask = (hue_series == cat_val)
                                    x_plot_vals = sx_data[mask].dropna().tolist(); y_plot_vals = sy_data[mask].dropna().tolist()
                                    min_len = min(len(x_plot_vals), len(y_plot_vals))
                                    if min_len > 0: dpg.add_scatter_series(x_plot_vals[:min_len], y_plot_vals[:min_len], label=str(cat_val), parent=pyaxis)
                                if len(hue_categories) > 1: dpg.add_plot_legend(parent=dpg.last_item(2)) # Plot legend
                            else:
                                aligned_s = pd.concat([sx_data, sy_data], axis=1).dropna()
                                if not aligned_s.empty: dpg.add_scatter_series(aligned_s.iloc[:,0].tolist(), aligned_s.iloc[:,1].tolist(), parent=pyaxis)

def _run_target_variable_analysis(df: pd.DataFrame, target_var: str, feature_var: str, util_funcs: dict):
    # ... (이전 답변의 _run_target_variable_analysis 전체 구현 내용 - 타입 조합별 분석 로직) ...
    results_group = TAG_MVA_TARGET_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group) or df is None or \
       not target_var or target_var not in df.columns or \
       not feature_var or feature_var not in df.columns or target_var == feature_var:
        if dpg.does_item_exist(results_group): dpg.delete_item(results_group, children_only=True); dpg.add_text("Select valid target and feature variables.", parent=results_group)
        return
    dpg.delete_item(results_group, children_only=True); dpg.add_text(f"Analysis: Feature '{feature_var}' vs Target '{target_var}'", parent=results_group); dpg.add_separator(parent=results_group)
    target_series = df[target_var]; feature_series = df[feature_var]
    if pd.api.types.is_numeric_dtype(target_series.dtype):
        if pd.api.types.is_numeric_dtype(feature_series.dtype): 
            dpg.add_text(f"Correlation: {feature_series.corr(target_series):.3f}", parent=results_group)
            with dpg.plot(label=f"Scatter: {feature_var} by {target_var}", height=300, width=-1, parent=results_group):
                dpg.add_plot_axis(dpg.mvXAxis, label=feature_var); yaxis_scatter = dpg.add_plot_axis(dpg.mvYAxis, label=target_var)
                aligned_df = pd.concat([feature_series, target_series], axis=1).dropna()
                if not aligned_df.empty: dpg.add_scatter_series(aligned_df.iloc[:,0].tolist(), aligned_df.iloc[:,1].tolist(), parent=yaxis_scatter)
        elif feature_series.nunique(dropna=False) < 20: 
            dpg.add_text("Grouped Statistics (Feature's Categories vs Numeric Target):", parent=results_group)
            grouped_stats = df.groupby(feature_var)[target_var].agg(['mean', 'median', 'std', 'count']).reset_index()
            table_tag = dpg.generate_uuid()
            with dpg.table(header_row=True, tag=table_tag, parent=results_group, resizable=True, height=150, scrollY=True): util_funcs['create_table_with_data'](table_tag, grouped_stats)
            dpg.add_text("Consider Box/Violin plots for visual comparison.", parent=results_group)
    elif target_series.nunique(dropna=False) < 20: 
        if pd.api.types.is_numeric_dtype(feature_series.dtype): 
            dpg.add_text("Grouped Statistics (Target's Categories vs Numeric Feature):", parent=results_group)
            grouped_stats = df.groupby(target_var)[feature_var].agg(['mean', 'median', 'std', 'count']).reset_index()
            table_tag = dpg.generate_uuid()
            with dpg.table(header_row=True, tag=table_tag, parent=results_group, resizable=True, height=150, scrollY=True): util_funcs['create_table_with_data'](table_tag, grouped_stats)
            dpg.add_text("Consider overlaid Histograms/Density plots.", parent=results_group)
            unique_target_cats = sorted(target_series.dropna().unique())[:5]
            if len(unique_target_cats) >=2:
                with dpg.plot(label=f"Density of '{feature_var}' by '{target_var}' categories", height=300, width=-1, parent=results_group):
                    dpg.add_plot_axis(dpg.mvXAxis, label=feature_var); yaxis_kde_group = dpg.add_plot_axis(dpg.mvYAxis, label="Density"); dpg.add_plot_legend(parent=dpg.last_item())
                    for cat_val in unique_target_cats:
                        subset_data = feature_series[target_series == cat_val].dropna()
                        if len(subset_data) > 1:
                            try:
                                kde = stats.gaussian_kde(subset_data); x_vals = np.linspace(subset_data.min(), subset_data.max(), 150); y_vals = kde(x_vals)
                                dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), label=f"Target={cat_val}", parent=yaxis_kde_group)
                            except Exception: pass
        elif feature_series.nunique(dropna=False) < 20: 
            dpg.add_text("Crosstabulation (Feature vs Target):", parent=results_group)
            try:
                crosstab_df_abs = pd.crosstab(df[feature_var], df[target_var])
                crosstab_df_norm = pd.crosstab(df[feature_var], df[target_var], normalize='index').mul(100).round(1).astype(str) + '%'
                dpg.add_text("Counts:", parent=results_group); table_tag_abs = dpg.generate_uuid()
                with dpg.table(header_row=True, tag=table_tag_abs, parent=results_group, resizable=True, height=150, scrollY=True): util_funcs['create_table_with_data'](table_tag_abs, crosstab_df_abs.reset_index())
                dpg.add_text("Row Percentages:", parent=results_group); table_tag_norm = dpg.generate_uuid()
                with dpg.table(header_row=True, tag=table_tag_norm, parent=results_group, resizable=True, height=150, scrollY=True): util_funcs['create_table_with_data'](table_tag_norm, crosstab_df_norm.reset_index())
                dpg.add_text("Consider Stacked/Grouped Bar charts.", parent=results_group)
            except Exception as e: dpg.add_text(f"Error creating crosstab: {e}", parent=results_group)
    else: dpg.add_text(f"Target '{target_var}' type/cardinality not suitable for this analysis.", parent=results_group)
