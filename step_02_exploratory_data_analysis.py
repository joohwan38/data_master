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
TAG_SVA_RESULTS_CHILD_WINDOW = "step2_sva_results_child_window"
TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX = "sva_var_section_"
TAG_SVA_PROGRESS_MODAL = "sva_progress_modal"
TAG_SVA_PROGRESS_TEXT = "sva_progress_text"
TAG_SVA_ALERT_MODAL_PREFIX = "sva_alert_modal_"

# Multivariate Analysis (MVA) Tab
TAG_MVA_TAB = "step2_mva_tab"
TAG_MVA_SUB_TAB_BAR = "step2_mva_sub_tab_bar"
TAG_MVA_CORR_TAB = "step2_mva_corr_tab"
TAG_MVA_CORR_RUN_BUTTON = "step2_mva_corr_run_button"
TAG_MVA_CORR_HEATMAP_PLOT = "step2_mva_corr_heatmap_plot"
TAG_MVA_CORR_TABLE = "step2_mva_corr_table"
TAG_MVA_PAIRPLOT_TAB = "step2_mva_pairplot_tab"
TAG_MVA_PAIRPLOT_VAR_SELECTOR = "step2_mva_pairplot_var_selector"
TAG_MVA_PAIRPLOT_HUE_COMBO = "step2_mva_pairplot_hue_combo" # Corrected from previous (step_02_...)
TAG_MVA_PAIRPLOT_RUN_BUTTON = "step2_mva_pairplot_run_button"
TAG_MVA_PAIRPLOT_RESULTS_GROUP = "step2_mva_pairplot_results_group"
TAG_MVA_TARGET_TAB = "step2_mva_target_tab"
TAG_MVA_TARGET_INFO_TEXT = "step2_mva_target_info_text"
TAG_MVA_TARGET_FEATURE_COMBO = "step2_mva_target_feature_combo"
TAG_MVA_TARGET_RUN_BUTTON = "step2_mva_target_run_button"
TAG_MVA_TARGET_RESULTS_GROUP = "step2_mva_target_results_group"
TAG_MVA_TARGET_PLOT_AREA_PREFIX = "mva_target_plot_area_"

REUSABLE_SVA_ALERT_MODAL_TAG = "reusable_sva_alert_modal_unique_tag"
REUSABLE_SVA_ALERT_TEXT_TAG = "reusable_sva_alert_text_unique_tag"

def _show_alert_modal(title: str, message: str):
    """
    Displays a modal alert window. Reuses a single modal definition for all SVA alerts.
    """
    if not dpg.is_dearpygui_running():
        print(f"DPG not running. Modal '{title}': {message}")
        return

    if not dpg.does_item_exist(REUSABLE_SVA_ALERT_MODAL_TAG):
        # Create the modal once if it doesn't exist
        with dpg.window(label="Alert", modal=True, show=False, tag=REUSABLE_SVA_ALERT_MODAL_TAG,
                        no_close=True, # User must click OK to dismiss
                        pos=[400, 300], width=400, autosize=True,
                        no_saved_settings=True): # Important for consistent behavior
            dpg.add_text("", tag=REUSABLE_SVA_ALERT_TEXT_TAG, wrap=380) # Placeholder for message
            dpg.add_spacer(height=15)
            with dpg.group(horizontal=True): # To center the button
                dpg.add_spacer(width=dpg.get_item_width(dpg.last_item()) / 2 - 50) # Approximate centering
                dpg.add_button(label="OK", width=100, user_data=REUSABLE_SVA_ALERT_MODAL_TAG,
                               callback=lambda s, a, u: dpg.configure_item(u, show=False))
    
    # Configure and show the existing, reusable modal
    dpg.configure_item(REUSABLE_SVA_ALERT_MODAL_TAG, label=title, show=True)
    dpg.set_value(REUSABLE_SVA_ALERT_TEXT_TAG, message)
    # dpg.focus_item(REUSABLE_SVA_ALERT_MODAL_TAG) # Ensure it's in front, though modal=True should handle

# --- Helper Functions (SVA and MVA common) ---

def _calculate_cramers_v(x: pd.Series, y: pd.Series):
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
    Calculates top N correlated/associated variables.
    Returns a list of dictionaries, e.g., [{'Variable': name, 'Metric': 'Pearson', 'Value': 0.85}, ...].
    Returns empty list if no significant relations or data is unsuitable.
    """
    if df is None or current_var_name not in df.columns or len(df.columns) < 2:
        return [] # Or [{'Info': 'Not enough data or variables'}] if you want to display a message

    correlations = []
    current_series = df[current_var_name]
    structured_results = []

    if pd.api.types.is_numeric_dtype(current_series.dtype):
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col_name in numeric_cols:
            if col_name == current_var_name: continue
            try:
                corr_val = current_series.corr(df[col_name])
                if pd.notna(corr_val) and abs(corr_val) > 0.1: # Threshold for "significant"
                    correlations.append((col_name, corr_val))
            except Exception:
                pass # Silently ignore errors for individual correlations
        correlations.sort(key=lambda item: abs(item[1]), reverse=True)
        structured_results = [{'Variable': name, 'Pearson Corr': f"{val:.3f}"} for name, val in correlations[:top_n]]
        if not structured_results:
            return [{'Info': 'No strong numeric correlations found.'}]

    elif current_series.nunique(dropna=False) < 30 or pd.api.types.is_categorical_dtype(current_series.dtype) or current_series.dtype == 'object':
        # Consider only other categorical-like columns for Cramér's V
        candidate_cols = [
            col for col in df.columns 
            if col != current_var_name and 
            (df[col].nunique(dropna=False) < 30 or pd.api.types.is_categorical_dtype(df[col].dtype) or df[col].dtype == 'object')
        ]
        for col_name in candidate_cols:
            try:
                # Ensure both series have enough variability for crosstab
                temp_df_cramers = pd.DataFrame({'x': current_series, 'y': df[col_name]}).dropna()
                if temp_df_cramers['x'].nunique() < 2 or temp_df_cramers['y'].nunique() < 2:
                    continue
                cramers_v = _calculate_cramers_v(current_series, df[col_name])
                if pd.notna(cramers_v) and cramers_v > 0.1: # Threshold for "significant"
                    correlations.append((col_name, cramers_v))
            except Exception:
                pass
        correlations.sort(key=lambda item: abs(item[1]), reverse=True)
        structured_results = [{'Variable': name, "Cramér's V": f"{val:.3f}"} for name, val in correlations[:top_n]]
        if not structured_results:
            return [{'Info': 'No strong categorical associations found.'}]
    else:
        return [{'Info': 'N/A (Unsupported type for this summary)'}]

    return structured_results

def _get_numeric_cols(df: pd.DataFrame):
    if df is None: return []
    return df.select_dtypes(include=np.number).columns.tolist()

def _get_categorical_cols(df: pd.DataFrame, max_unique_for_cat=20):
    if df is None: return []
    cat_cols = []
    # 명시적인 object나 category 타입 먼저 추가
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col].dtype):
            if df[col].nunique(dropna=False) <= max_unique_for_cat * 2: # 고유값 제한을 약간 더 유연하게
                 cat_cols.append(col)
    # 숫자형이지만 고유값이 적은 경우도 추가 (중복 방지)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].nunique(dropna=False) <= max_unique_for_cat and col not in cat_cols :
            cat_cols.append(col)
    return list(set(cat_cols))


def _get_filtered_variables(df: pd.DataFrame, filter_strength: str,
                            main_callbacks: dict, target_var: str = None) -> tuple[list, str]: # 반환 타입 명시
    # ... (함수 초반부 로직은 동일) ...
    # cols_after_text_filter, weakly_filtered_cols 계산까지는 동일

    if df is None or df.empty: return [], filter_strength # 데이터 없는 경우

    # ... (analysis_types_dict, cols_after_text_filter 정의) ...
    # 이 부분은 이전과 동일
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
    
    if filter_strength == "None (All variables)":
        print(f"DEBUG: Filter Profile 'None' selected. Vars: {len(cols_after_text_filter)}")
        return cols_after_text_filter, "None (All variables)"

    weakly_filtered_cols = []
    # ... (weakly_filtered_cols 계산 로직은 동일) ...
    for col_name in cols_after_text_filter: #
        series = df[col_name] #
        col_analysis_type = analysis_types_dict.get(col_name, str(series.dtype)) #
        if series.nunique(dropna=False) <= 1: continue #
        is_numeric_binary = False #
        if "Numeric" in col_analysis_type: #
            unique_vals_str_set = set(series.dropna().astype(str).unique()) #
            if unique_vals_str_set <= {'0', '1', '0.0', '1.0'} and len(unique_vals_str_set) <= 2: # Ensure only 0 and 1 #
                is_numeric_binary = True #
        if is_numeric_binary: continue #
        weakly_filtered_cols.append(col_name) #

    if filter_strength == "Weak (Exclude obvious non-analytical)":
        print(f"DEBUG: Filter Profile 'Weak' applied. Vars: {len(weakly_filtered_cols)}")
        return weakly_filtered_cols, "Weak (Exclude obvious non-analytical)"

    # Medium/Strong 필터 로직 시작
    actual_applied_filter = filter_strength # 기본적으로는 요청된 필터
    print(f"DEBUG: Attempting '{filter_strength}' filter. Target variable: '{target_var}'")

    numeric_cols_for_ranking = [ # ... (기존과 동일) ...
        col for col in weakly_filtered_cols 
        if "Numeric" in analysis_types_dict.get(col, df[col].dtype.name) and not (
            len(set(df[col].dropna().astype(str).unique())) == 2 and 
            set(df[col].dropna().astype(str).unique()) <= {'0', '1', '0.0', '1.0'}
        )
    ]
    if not numeric_cols_for_ranking:
        msg = f"'{filter_strength}' filter requires numeric variables for ranking. None found after 'Weak' filter. Defaulting to 'Weak' filter results ({len(weakly_filtered_cols)} vars)."
        _show_alert_modal("Filter Fallback", msg)
        print(f"DEBUG: No suitable numeric columns for ranking for '{filter_strength}'. Fallback to 'Weak'.")
        return weakly_filtered_cols, "Weak (Exclude obvious non-analytical)" # Fallback
    if not target_var:
        msg = f"'{filter_strength}' filter requires a Target Variable for relevance ranking. Target not set. Defaulting to 'Weak' filter results ({len(weakly_filtered_cols)} vars)."
        _show_alert_modal("Target Variable Needed", msg)
        print(f"DEBUG: Target variable not provided for '{filter_strength}'. Fallback to 'Weak'.")
        return weakly_filtered_cols, "Weak (Exclude obvious non-analytical)" # Fallback

    # ... (relevance_scores 계산 로직은 동일) ...
    target_series = df[target_var] #
    target_analysis_type = analysis_types_dict.get(target_var, str(target_series.dtype)) #
    relevance_scores = [] #
    for col_name in numeric_cols_for_ranking: #
        if col_name == target_var: continue #
        score = 0.0 #
        try: #
            current_col_series = df[col_name] #
            if "Categorical" in target_analysis_type or ("Numeric (Binary)" in target_analysis_type and target_series.nunique(dropna=False) <=5 ): #
                groups = [current_col_series[target_series == cat].dropna() for cat in target_series.dropna().unique()] #
                groups = [g for g in groups if len(g) >= 2]; #
                if len(groups) >= 2: f_val, p_val = stats.f_oneway(*groups); score = f_val if pd.notna(f_val) else 0.0 #
            elif "Numeric" in target_analysis_type: score = abs(current_col_series.corr(target_series)) #
        except Exception: pass #
        if pd.notna(score): relevance_scores.append((col_name, score)) #
    relevance_scores.sort(key=lambda item: item[1], reverse=True) #
    ranked_cols = [col for col, score in relevance_scores if pd.notna(score) and score > 0.001] #

    if not ranked_cols:
        msg = f"No variables showed significant relevance to target '{target_var}' for '{filter_strength}' filter. Defaulting to 'Weak' filter results ({len(weakly_filtered_cols)} vars)."
        _show_alert_modal("Filter Fallback", msg)
        print(f"DEBUG: No significantly relevant columns found for '{filter_strength}'. Fallback to 'Weak'.")
        return weakly_filtered_cols, "Weak (Exclude obvious non-analytical)" # Fallback

    if filter_strength == "Strong (Top 5-10 relevant)":
        result_cols = ranked_cols[:min(len(ranked_cols), 10)]
        print(f"DEBUG: '{filter_strength}' applied. Returning {len(result_cols)} vars.")
        return result_cols, filter_strength
    if filter_strength == "Medium (Top 11-20 relevant)":
        result_cols = ranked_cols[:min(len(ranked_cols), 20)]
        print(f"DEBUG: '{filter_strength}' applied. Returning {len(result_cols)} vars.")
        return result_cols, filter_strength

    # 예기치 않은 경우, Weak 필터로 기본 설정
    print(f"DEBUG: Filter strength '{filter_strength}' not matched or unexpected issue. Defaulting to 'Weak'.")
    return weakly_filtered_cols, "Weak (Exclude obvious non-analytical)"

def _create_sva_basic_stats_table(parent_tag: str, series: pd.Series, util_funcs: dict, analysis_type_override: str = None):
    """Creates table for 5-num summary, mean, std, skew, kurt."""
    stats_data = []
    is_effectively_categorical = (analysis_type_override == "ForceCategoricalForBinaryNumeric" or
                                  pd.api.types.is_categorical_dtype(series.dtype) or
                                  series.dtype == 'object' or
                                  series.nunique(dropna=False) < 5)
    is_numeric_original_dtype = pd.api.types.is_numeric_dtype(series.dtype)

    if is_numeric_original_dtype and not is_effectively_categorical:
        desc = series.describe()
        stats_to_show = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        for stat_name in stats_to_show:
            if stat_name in desc.index:
                stats_data.append({'Statistic': stat_name, 'Value': f"{desc[stat_name]:.3f}" if isinstance(desc[stat_name], (int, float)) else str(desc[stat_name])})
        
        skew, kurt = np.nan, np.nan
        if not series.empty and series.notna().any():
            try: skew = series.skew()
            except TypeError: pass
            try: kurt = series.kurtosis()
            except TypeError: pass
        stats_data.append({'Statistic': 'Skewness', 'Value': f"{skew:.3f}"})
        stats_data.append({'Statistic': 'Kurtosis', 'Value': f"{kurt:.3f}"})
    else: # Categorical or treated as categorical
        value_counts = series.value_counts(dropna=False)
        stats_data.extend([
            {'Statistic': 'Count (incl. NA)', 'Value': str(len(series))},
            {'Statistic': 'Unique (incl. NA)', 'Value': str(series.nunique(dropna=False))},
            {'Statistic': 'Top (Mode)', 'Value': str(value_counts.index[0]) if not value_counts.empty else 'N/A'},
            {'Statistic': 'Top Freq', 'Value': str(value_counts.iloc[0]) if not value_counts.empty else 'N/A'},
            {'Statistic': 'Dtype', 'Value': str(series.dtype)}
        ])
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        table_tag = dpg.generate_uuid()
        with dpg.table(header_row=True, tag=table_tag, parent=parent_tag,
                       borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True,
                       resizable=True, policy=dpg.mvTable_SizingStretchProp, height=230): # Adjust height
            util_funcs['create_table_with_data'](table_tag, stats_df, parent_df_for_widths=stats_df)

def _create_sva_advanced_relations_table(parent_tag: str, series: pd.Series, full_df: pd.DataFrame, util_funcs: dict):
    """Creates table for Shapiro-Wilk and Top 5 Related Variables."""
    # Shapiro-Wilk Test (only for numeric, non-categorical data)
    normality_data = []
    is_effectively_categorical = (pd.api.types.is_categorical_dtype(series.dtype) or
                                  series.dtype == 'object' or
                                  series.nunique(dropna=False) < 5)
    if pd.api.types.is_numeric_dtype(series.dtype) and not is_effectively_categorical:
        series_dropna = series.dropna()
        if 3 <= len(series_dropna) < 5000 : # Shapiro-Wilk constraints
            try:
                stat_sw, p_sw = stats.shapiro(series_dropna)
                normality_data.append({'Test': 'Shapiro-Wilk W', 'Value': f"{stat_sw:.3f}"})
                normality_data.append({'Test': 'p-value', 'Value': f"{p_sw:.3f}"})
                normality_data.append({'Test': 'Normality', 'Value': "Likely Normal" if p_sw > 0.05 else "Likely Not Normal"})
            except Exception as e:
                normality_data.append({'Test': 'Shapiro-Wilk', 'Value': f"Error: {e}"})
        else:
            normality_data.append({'Test': 'Shapiro-Wilk', 'Value': 'N/A (sample size)'})
    else:
        normality_data.append({'Test': 'Shapiro-Wilk', 'Value': 'N/A (not numeric)'})

    if normality_data:
        normality_df = pd.DataFrame(normality_data)
        dpg.add_text("Normality Test:", parent=parent_tag)
        norm_table_tag = dpg.generate_uuid()
        with dpg.table(header_row=True, tag=norm_table_tag, parent=parent_tag,
                       borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True,
                       resizable=True, policy=dpg.mvTable_SizingStretchProp, height=100): # Adjust height
            util_funcs['create_table_with_data'](norm_table_tag, normality_df, parent_df_for_widths=normality_df)
        dpg.add_spacer(height=5, parent=parent_tag)

    # Top Related Variables
    top_related_vars_data = _get_top_correlated_vars(full_df, series.name, top_n=5)
    dpg.add_text("Top Related Variables:", parent=parent_tag)
    if top_related_vars_data:
        if 'Info' in top_related_vars_data[0]: # Handle message like "No correlations"
             dpg.add_text(top_related_vars_data[0]['Info'], parent=parent_tag, wrap=300) # Allow wrap
        else:
            related_vars_df = pd.DataFrame(top_related_vars_data)
            rel_table_tag = dpg.generate_uuid()
            # Adjust height for 5 rows + header
            table_height = min(150, len(related_vars_df) * 25 + 30) if len(related_vars_df)>0 else 60
            with dpg.table(header_row=True, tag=rel_table_tag, parent=parent_tag,
                           borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True,
                           resizable=True, policy=dpg.mvTable_SizingStretchProp, height=table_height):
                util_funcs['create_table_with_data'](rel_table_tag, related_vars_df, parent_df_for_widths=related_vars_df)
    else: # Should not happen if _get_top_correlated_vars always returns list with Info or data
        dpg.add_text("No correlation data.", parent=parent_tag)

def _create_single_var_plot(parent_group_tag: str, series: pd.Series, group_by_target_series: pd.Series = None, analysis_type_override:str=None):
    plot_height = 230; plot_width = -1
    plot_label = f"Distribution: {series.name}"
    if group_by_target_series is not None and analysis_type_override != "ForceCategoricalForBinaryNumeric":
        plot_label += f" (Grouped by {group_by_target_series.name})"

    plot_tag = dpg.generate_uuid()
    with dpg.plot(label=plot_label, height=plot_height, width=plot_width, parent=parent_group_tag, tag=plot_tag):
        xaxis_tag = dpg.add_plot_axis(dpg.mvXAxis, label=series.name)
        yaxis_tag = dpg.generate_uuid(); dpg.add_plot_axis(dpg.mvYAxis, label="Density / Frequency", tag=yaxis_tag)
        legend_tag = dpg.add_plot_legend(parent=plot_tag)

        series_cleaned_for_plot_initial = series.dropna() # 초기 NaN 제거

        # 무한대 값 처리 추가
        series_cleaned_for_plot = series_cleaned_for_plot_initial.replace([np.inf, -np.inf], np.nan).dropna()

        if len(series_cleaned_for_plot) < 2 : # 히스토그램을 그리기에 데이터가 충분한지 확인 (최소 2개)
            if dpg.does_item_exist(parent_group_tag): # parent_group_tag가 플롯의 부모가 아니라, 플롯을 담는 그룹의 태그여야 함
                                                     # yaxis_tag를 부모로 하여 플롯 내부에 텍스트 추가
                dpg.add_text("Not enough valid data points for plot.", parent=yaxis_tag, color=(255, 200, 0))
            if dpg.does_item_exist(legend_tag):
                 dpg.delete_item(legend_tag)
            return

        is_effectively_categorical_for_plot = (analysis_type_override == "ForceCategoricalForBinaryNumeric" or
                                              pd.api.types.is_categorical_dtype(series.dtype) or
                                              series.dtype == 'object' or
                                              series.nunique(dropna=False) < 5)

        if pd.api.types.is_numeric_dtype(series.dtype) and not is_effectively_categorical_for_plot:
            # --- Numeric Plotting ---
            if group_by_target_series is not None and group_by_target_series.nunique(dropna=False) > 1:
                unique_target_groups_numeric = sorted(group_by_target_series.dropna().unique())
                for group_name in unique_target_groups_numeric:
                    group_data_initial = series_cleaned_for_plot[group_by_target_series == group_name]
                    # 그룹 데이터에 대해서도 무한대 값 처리 및 길이 확인
                    group_data = group_data_initial.replace([np.inf, -np.inf], np.nan).dropna()

                    if len(group_data) < 2: # 그룹별 데이터 포인트가 2개 미만이면 히스토그램 생략
                        dpg.add_text(f"Group '{group_name}': Not enough data for histogram.", parent=yaxis_tag, color=(255,200,0))
                        continue
                    
                    try:
                        dpg.add_histogram_series(group_data.tolist(), label=f"Hist (T={group_name})", 
                                                 density=True, bins=-1, parent=yaxis_tag, weight=0.8)
                        kde = stats.gaussian_kde(group_data)
                        x_vals = np.linspace(group_data.min(), group_data.max(), 150)
                        y_vals = kde(x_vals)
                        dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), label=f"KDE (T={group_name})", parent=yaxis_tag)
                    except SystemError as e_sys:
                        dpg.add_text(f"Error plotting group '{group_name}' (SystemError).", parent=yaxis_tag, color=(255,0,0))
                        print(f"SystemError plotting histogram for group '{group_name}', var '{series.name}': {e_sys}\nData sample: {group_data.head().tolist()}")
                    except Exception as e_gen:
                        dpg.add_text(f"Error plotting group '{group_name}'.", parent=yaxis_tag, color=(255,0,0))
                        print(f"General error plotting histogram for group '{group_name}', var '{series.name}': {e_gen}\nData sample: {group_data.head().tolist()}")

            else: # Single numeric series
                # 이미 series_cleaned_for_plot는 위에서 무한대 처리 및 길이 확인 완료됨
                try:
                    dpg.add_histogram_series(series_cleaned_for_plot.tolist(), label="Histogram", 
                                             density=True, bins=-1, parent=yaxis_tag, weight=1.0)
                    kde = stats.gaussian_kde(series_cleaned_for_plot)
                    x_vals = np.linspace(series_cleaned_for_plot.min(), series_cleaned_for_plot.max(), 200)
                    y_vals = kde(x_vals)
                    dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), label="KDE", parent=yaxis_tag)
                except SystemError as e_sys:
                    dpg.add_text(f"Error plotting histogram (SystemError).", parent=yaxis_tag, color=(255,0,0))
                    print(f"SystemError plotting histogram for var '{series.name}': {e_sys}\nData sample: {series_cleaned_for_plot.head().tolist()}")
                except Exception as e_gen:
                    dpg.add_text(f"Error plotting histogram.", parent=yaxis_tag, color=(255,0,0))
                    print(f"General error plotting histogram for var '{series.name}': {e_gen}\nData sample: {series_cleaned_for_plot.head().tolist()}")
        else:
            # --- Categorical Plotting (Bar Charts) ---
            # (이 부분은 기존 로직 유지)
            top_n_categories = 10
            if group_by_target_series is not None and \
               group_by_target_series.nunique(dropna=False) >= 2 and \
               group_by_target_series.nunique(dropna=False) <= 5 and \
               not is_effectively_categorical_for_plot:
                
                if dpg.does_item_exist(legend_tag): dpg.delete_item(legend_tag)
                
                unique_target_groups_categorical = sorted(group_by_target_series.dropna().unique())

                # parent_group_tag의 너비를 가져올 때, 해당 아이템이 존재하는지, 너비가 유효한지 확인하는 것이 좋음
                parent_width = dpg.get_item_width(parent_group_tag) if dpg.does_item_exist(parent_group_tag) and dpg.get_item_width(parent_group_tag) else 300 # 기본 너비 설정
                plot_area_width = parent_width - 20 if parent_width > 20 else 280

                num_cols_for_plot = len(unique_target_groups_categorical)
                sub_plot_width = max(150, (plot_area_width - 10 * (num_cols_for_plot -1)) / num_cols_for_plot if num_cols_for_plot > 0 else plot_area_width)
                
                with dpg.group(horizontal=True, parent=parent_group_tag):
                    for group_name in unique_target_groups_categorical:
                        with dpg.group(parent=dpg.last_item()):
                            with dpg.plot(label=f"{series.name} (Target={group_name})", height=plot_height-30, width=int(sub_plot_width)):
                                sub_xaxis_tag = dpg.add_plot_axis(dpg.mvXAxis, label=str(group_name))
                                sub_yaxis_tag = dpg.add_plot_axis(dpg.mvYAxis, label="Frequency")
                                group_series = series_cleaned_for_plot[group_by_target_series == group_name] # series_cleaned_for_plot 사용
                                if group_series.empty: continue
                                value_counts_data = group_series.value_counts(dropna=False).nlargest(top_n_categories)
                                x_pos = list(range(len(value_counts_data)))
                                bar_labels = [str(val) for val in value_counts_data.index.tolist()]
                                dpg.add_bar_series(x_pos, value_counts_data.values.tolist(), weight=0.7, label=f"Freq (T={group_name})", parent=sub_yaxis_tag)
                                if bar_labels and dpg.does_item_exist(sub_xaxis_tag): dpg.set_axis_ticks(sub_xaxis_tag, tuple(zip(bar_labels, x_pos)))
            else: 
                if dpg.does_item_exist(legend_tag): dpg.delete_item(legend_tag)
                value_counts_data = series_cleaned_for_plot.value_counts(dropna=False).nlargest(top_n_categories) # series_cleaned_for_plot 사용
                x_pos = list(range(len(value_counts_data)))
                bar_labels = [str(val) for val in value_counts_data.index.tolist()]
                dpg.add_bar_series(x_pos, value_counts_data.values.tolist(), weight=0.7, label="Frequency", parent=yaxis_tag)
                if bar_labels and dpg.does_item_exist(xaxis_tag): dpg.set_axis_ticks(xaxis_tag, tuple(zip(bar_labels, x_pos)))

def _show_alert_modal(title: str, message: str):
    # ... (이전과 동일) ...
    if not dpg.is_dearpygui_running(): return
    modal_tag = TAG_SVA_ALERT_MODAL_PREFIX + str(hash(message))[:8] 
    if dpg.does_item_exist(modal_tag) and dpg.is_item_shown(modal_tag):
        dpg.configure_item(modal_tag, show=False); dpg.delete_item(modal_tag) 
    with dpg.window(label=title, modal=True, show=True, tag=modal_tag, no_close=False, pos=[400,300], width=400, autosize=True):
        dpg.add_text(message, wrap=380); dpg.add_spacer(height=20)
        with dpg.group(horizontal=True): dpg.add_button(label="OK", width=-1, user_data=modal_tag, callback=lambda s, a, u: dpg.configure_item(u, show=False))

def _apply_sva_filters_and_run(main_callbacks: dict):
    print("DEBUG: _apply_sva_filters_and_run called.") # 함수 호출 확인

    # --- 1. 필수 데이터 및 UI 요소 가져오기 ---
    current_df = main_callbacks['get_current_df']()
    util_funcs = main_callbacks['get_util_funcs']()
    target_var = main_callbacks['get_selected_target_variable']()
    analysis_types_dict_local = main_callbacks.get('get_column_analysis_types', lambda: {})()
    
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

    if not dpg.does_item_exist(progress_modal_tag): # 진행률 모달이 없다면 생성
        with dpg.window(label="Processing SVA", modal=True, show=False, tag=progress_modal_tag, 
                        no_close=True, no_title_bar=True, pos=[500,400], width=350, height=70):
            dpg.add_text("Analyzing variables, please wait...", tag=progress_text_tag)
    
    dpg.configure_item(progress_modal_tag, show=True)
    dpg.set_value(progress_text_tag, "SVA: Preparing analysis...")
    dpg.split_frame() # UI 즉시 업데이트

    # --- 3. 데이터 유효성 검사 ---
    if current_df is None:
        dpg.add_text("Load data first to perform Single Variable Analysis.", parent=results_child_window_tag)
        dpg.configure_item(progress_modal_tag, show=False)
        return

    # --- 4. UI에서 현재 필터 및 그룹핑 옵션 가져오기 ---
    filter_strength_selected = dpg.get_value(TAG_SVA_FILTER_STRENGTH_RADIO) if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO) else "Weak (Exclude obvious non-analytical)"
    group_by_target_flag = dpg.get_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX) if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX) else False
    target_series_for_grouping = None

    # --- 5. 그룹핑 옵션 유효성 검사 및 설정 ---
    if group_by_target_flag:
        if target_var and target_var in current_df.columns:
            unique_target_values = current_df[target_var].nunique(dropna=False)
            target_col_analysis_type = analysis_types_dict_local.get(target_var, str(current_df[target_var].dtype))
            if ("Categorical" in target_col_analysis_type or "Numeric (Binary)" in target_col_analysis_type) and \
               unique_target_values >= 2 and unique_target_values <= 5:
                target_series_for_grouping = current_df[target_var]
                print(f"DEBUG: Grouping by target '{target_var}' enabled.")
            else:
                _show_alert_modal("Grouping Warning", f"Target '{target_var}' (Type: {target_col_analysis_type}, Unique: {unique_target_values}) not suitable for grouping (requires 2-5 distinct categories). Grouping disabled.")
                if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX): dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
                group_by_target_flag = False # 플래그 업데이트
        else:
            _show_alert_modal("Grouping Info", "Target variable not selected or invalid for grouping. Grouping disabled.")
            if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX): dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
            group_by_target_flag = False # 플래그 업데이트
    
    # --- 6. 변수 필터링 실행 (`_get_filtered_variables`는 (cols_list, actual_filter_str) 반환 가정) ---
    dpg.set_value(progress_text_tag, "SVA: Filtering variables...")
    dpg.split_frame()
    filtered_cols, actual_filter_applied = _get_filtered_variables(current_df, filter_strength_selected, main_callbacks, target_var)
    print(f"DEBUG: User selected filter: '{filter_strength_selected}'. Actual filter applied: '{actual_filter_applied}'. Num vars: {len(filtered_cols)}")

    # --- 7. Fallback 조건 처리 (Medium/Strong 선택했으나 Weak으로 적용된 경우) ---
    is_fallback_to_weak = (filter_strength_selected in ["Medium (Top 11-20 relevant)", "Strong (Top 5-10 relevant)"]) and \
                          (actual_filter_applied == "Weak (Exclude obvious non-analytical)")

    if is_fallback_to_weak:
        # _get_filtered_variables 내부에서 이미 알림 모달이 표시되었을 것임
        print(f"INFO: SVA run aborted. User selected '{filter_strength_selected}', but conditions for it were not met, leading to fallback intention to '{actual_filter_applied}'.")
        dpg.add_text(f"SVA not performed for '{filter_strength_selected}'.\nConditions for this filter were not met (e.g., target variable missing or unsuitable data for ranking).\nPlease check console logs or previous alerts for details.", 
                     parent=results_child_window_tag, wrap=dpg.get_item_width(results_child_window_tag)-20 if dpg.get_item_width(results_child_window_tag) > 0 else 400,
                     color=(255,165,0)) # 주황색 경고
        dpg.configure_item(progress_modal_tag, show=False)
        return # 분석 중단

    # --- 8. 필터링된 변수가 없는 경우 처리 ---
    if not filtered_cols:
        dpg.add_text(f"No variables to display based on the filter: '{actual_filter_applied}'.", parent=results_child_window_tag)
        dpg.configure_item(progress_modal_tag, show=False)
        return

    # --- 9. 각 변수에 대한 분석 및 UI 생성 (3단 레이아웃) ---
    total_vars = len(filtered_cols)
    print(f"DEBUG: Starting SVA loop for {total_vars} variables using '{actual_filter_applied}' filter.")

    for i, col_name in enumerate(filtered_cols):
        if not dpg.is_dearpygui_running(): break # GUI 종료 시 중단
        dpg.set_value(progress_text_tag, f"SVA: Analyzing {col_name} ({i+1}/{total_vars})")
        dpg.split_frame()

        var_section_tag_str = "".join(filter(str.isalnum, str(col_name))) # 유효한 태그 문자열 생성
        var_section_tag = f"{TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX}{var_section_tag_str}_{i}" # 인덱스 추가로 완전 고유성 확보

        # 이전 섹션이 있다면 삭제 (필요 시, 그러나 results_child_window_tag 전체를 비웠으므로 중복 생성 우려는 적음)
        # if dpg.does_item_exist(var_section_tag): dpg.delete_item(var_section_tag) 

        col_analysis_type = analysis_types_dict_local.get(col_name, str(current_df[col_name].dtype))
        analysis_override_for_sva = None
        if actual_filter_applied == "None (All variables)": # 실제 적용된 필터 기준
            unique_vals_str_set = set(current_df[col_name].dropna().astype(str).unique())
            is_binary_numeric_for_none_filter = unique_vals_str_set <= {'0', '1', '0.0', '1.0'} and len(unique_vals_str_set) <= 2
            if "Numeric" in col_analysis_type and is_binary_numeric_for_none_filter:
                analysis_override_for_sva = "ForceCategoricalForBinaryNumeric"
        
        with dpg.group(tag=var_section_tag, parent=results_child_window_tag):
            # 변수명 및 타입 정보 표시
            header_text_width = dpg.get_item_width(results_child_window_tag) - 20 if dpg.get_item_width(results_child_window_tag) > 20 else 500
            dpg.add_text(f"Variable: {util_funcs['format_text_for_display'](col_name, 60)}", color=(255, 255, 0), wrap=header_text_width)
            dpg.add_text(f"Identified Analysis Type: {col_analysis_type} (Actual Dtype: {str(current_df[col_name].dtype)})", wrap=header_text_width)
            dpg.add_spacer(height=5)

            # 3단 레이아웃 컨테이너
            # col_width 계산: results_child_window_tag 너비가 확정된 후 사용해야 정확.
            # 여기서는 각 그룹이 내용을 기반으로 너비를 갖도록 하거나, 비율 기반 설정을 고려.
            # 우선은 너비 지정을 최소화하거나, child_window 너비 변경 시 다시 계산하는 로직이 필요할 수 있음.
            # 간단하게 하기 위해 각 그룹에 width를 지정하지 않고, dpg.group(horizontal=True)의 자동 배치에 맡겨봄.
            # 또는, 고정 비율로 계산 (예: 30%, 30%, 40%)
            
            available_width = dpg.get_item_width(results_child_window_tag)
            if available_width <=0 : available_width = 800 # 기본 너비 (창이 아직 완전히 그려지지 않은 경우 대비)
            
            # 대략적인 너비 비율 설정 (공백 포함하여 총합이 100%에 가깝게)
            # 실제로는 padding, spacing 등 고려 필요
            col_1_width = int(available_width * 0.30)
            col_2_width = int(available_width * 0.30)
            # col_3_width은 나머지 또는 비율. 0이면 자동.

            with dpg.group(horizontal=True):
                # Column 1: Basic Stats
                with dpg.group(width=col_1_width) as col1_group_tag:
                    print(f"  DEBUG [{col_name}]: Creating basic stats table in group {col1_group_tag}")
                    _create_sva_basic_stats_table(col1_group_tag, current_df[col_name], util_funcs, analysis_override_for_sva)
                
                # Column 2: Advanced Stats & Relations
                with dpg.group(width=col_2_width) as col2_group_tag:
                    print(f"  DEBUG [{col_name}]: Creating advanced stats table in group {col2_group_tag}")
                    _create_sva_advanced_relations_table(col2_group_tag, current_df[col_name], current_df, util_funcs)

                # Column 3: Plot (나머지 공간 사용하도록 width=0 또는 양수 지정)
                with dpg.group() as col3_group_tag: # width=0 이면 나머지 공간 자동 채움 시도
                    print(f"  DEBUG [{col_name}]: Creating plot in group {col3_group_tag}")
                    _create_single_var_plot(col3_group_tag, current_df[col_name], 
                                            target_series_for_grouping if group_by_target_flag else None, 
                                            analysis_override_for_sva)
            dpg.add_separator()
            dpg.add_spacer(height=10) # 변수 간 간격

    # --- 10. 완료 후 진행률 표시창 숨기기 ---
    dpg.configure_item(progress_modal_tag, show=False)
    print("DEBUG: SVA processing finished successfully.")


# --- MVA Helper Functions (Full Implementations - 이전 답변 내용과 동일) ---
# _run_correlation_analysis, _run_pair_plot_analysis, _run_target_variable_analysis
# 이 함수들은 이전에 제공된 상세 구현 내용을 여기에 포함합니다.

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


# --- Main UI Creation & Update ---
def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    main_callbacks['register_step_group_tag'](step_name, TAG_EDA_GROUP)
    with dpg.group(tag=TAG_EDA_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---"); dpg.add_separator()
        with dpg.tab_bar(tag=TAG_EDA_MAIN_TAB_BAR):
            with dpg.tab(label="Single Variable Analysis", tag=TAG_SVA_TAB):
                with dpg.group(horizontal=True):
                    with dpg.group(): # Filter Options Group
                        dpg.add_text("Variable Filter")
                        dpg.add_radio_button(
                            items=["Strong (Top 5-10 relevant)", "Medium (Top 11-20 relevant)", "Weak (Exclude obvious non-analytical)", "None (All variables)"],
                            tag=TAG_SVA_FILTER_STRENGTH_RADIO, default_value="Weak (Exclude obvious non-analytical)"
                            # callback 제거
                        )
                        dpg.add_spacer(height=5)
                        dpg.add_text("Filter Algorithm (Simplified):", wrap=250)
                        dpg.add_text("- Strong/Medium: Relevance to Target (if set) or other heuristics.", wrap=250)
                        dpg.add_text("- Weak: Excludes single-value & binary numeric variables.", wrap=250)
                    dpg.add_spacer(width=20)
                    with dpg.group(): # Grouping and Action Group
                        dpg.add_text("Grouping Option")
                        dpg.add_checkbox(label="Group by Target (Categorical Target, 2-5 Categories)",
                                         tag=TAG_SVA_GROUP_BY_TARGET_CHECKBOX, default_value=False
                                         # callback 제거
                                         )
                        dpg.add_spacer(height=10)
                        # "Run SVA" 버튼 추가
                        dpg.add_button(label="Run Single Variable Analysis",
                                       callback=lambda: _apply_sva_filters_and_run(main_callbacks),
                                       width=-1, height=30)

                dpg.add_separator()
                with dpg.child_window(tag=TAG_SVA_RESULTS_CHILD_WINDOW, border=False):
                    dpg.add_text("Select filter options and click 'Run Single Variable Analysis'.")
            

            with dpg.tab(label="Multivariate Analysis", tag=TAG_MVA_TAB):
                with dpg.tab_bar(tag=TAG_MVA_SUB_TAB_BAR):
                    with dpg.tab(label="Correlation Analysis", tag=TAG_MVA_CORR_TAB):
                        dpg.add_button(label="Run Correlation Analysis (Numeric Vars)", tag=TAG_MVA_CORR_RUN_BUTTON,
                                       callback=lambda: _run_correlation_analysis(main_callbacks['get_current_df'](), main_callbacks['get_util_funcs']()))
                        dpg.add_child_window(tag=TAG_MVA_CORR_HEATMAP_PLOT, border=True, height=480)
                        # Correlation table은 _run_correlation_analysis 함수 내에서 TAG_MVA_CORR_TAB에 직접 추가됨
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

    main_callbacks['register_module_updater'](step_name, lambda df_arg, mc_arg: update_ui(df_arg, mc_arg))
    
    update_ui(main_callbacks['get_current_df'](), main_callbacks)


def update_ui(current_df: pd.DataFrame, main_callbacks: dict):
    # ... (이전 답변과 동일하게 MVA 탭의 선택지 업데이트 로직 유지) ...
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_EDA_GROUP): return
    print("EDA Module: update_ui called to refresh selectors.")
    
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
    categorical_cols_for_hue = [""] + _get_categorical_cols(current_df, max_unique_for_cat=7) 

    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_VAR_SELECTOR):
        current_selection = dpg.get_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR)
        dpg.configure_item(TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=numeric_cols)
        if current_selection and isinstance(current_selection, list) and all(item in numeric_cols for item in current_selection):
            try: dpg.set_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR, current_selection)
            except Exception: pass 
            
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_HUE_COMBO):
        current_hue = dpg.get_value(TAG_MVA_PAIRPLOT_HUE_COMBO)
        dpg.configure_item(TAG_MVA_PAIRPLOT_HUE_COMBO, items=categorical_cols_for_hue)
        if current_hue and current_hue in categorical_cols_for_hue:
            dpg.set_value(TAG_MVA_PAIRPLOT_HUE_COMBO, current_hue)
        elif categorical_cols_for_hue: dpg.set_value(TAG_MVA_PAIRPLOT_HUE_COMBO, categorical_cols_for_hue[0])

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
    print("EDA UI selectors updated.")