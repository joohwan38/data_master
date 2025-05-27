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
TAG_SVA_ALERT_MODAL_PREFIX = "sva_alert_modal_" # 각 알림마다 고유 ID를 위해 접두사 사용

# Multivariate Analysis (MVA) Tab
TAG_MVA_TAB = "step2_mva_tab"
TAG_MVA_SUB_TAB_BAR = "step2_mva_sub_tab_bar"

# MVA - Correlation Tab
TAG_MVA_CORR_TAB = "step2_mva_corr_tab"
TAG_MVA_CORR_RUN_BUTTON = "step2_mva_corr_run_button"
TAG_MVA_CORR_HEATMAP_PLOT = "step2_mva_corr_heatmap_plot"
TAG_MVA_CORR_TABLE = "step2_mva_corr_table"

# MVA - Pair Plot Tab
TAG_MVA_PAIRPLOT_TAB = "step2_mva_pairplot_tab"
TAG_MVA_PAIRPLOT_VAR_SELECTOR = "step2_mva_pairplot_var_selector"
TAG_MVA_PAIRPLOT_HUE_COMBO = "step2_mva_pairplot_hue_combo"
TAG_MVA_PAIRPLOT_RUN_BUTTON = "step2_mva_pairplot_run_button"
TAG_MVA_PAIRPLOT_RESULTS_GROUP = "step2_mva_pairplot_results_group"

# MVA - Target Analysis Tab
TAG_MVA_TARGET_TAB = "step2_mva_target_tab"
TAG_MVA_TARGET_INFO_TEXT = "step2_mva_target_info_text"
TAG_MVA_TARGET_FEATURE_COMBO = "step2_mva_target_feature_combo"
TAG_MVA_TARGET_RUN_BUTTON = "step2_mva_target_run_button"
TAG_MVA_TARGET_RESULTS_GROUP = "step2_mva_target_results_group"
TAG_MVA_TARGET_PLOT_AREA_PREFIX = "mva_target_plot_area_"


# --- Helper Functions ---

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

def _get_top_correlated_vars(df: pd.DataFrame, current_var_name: str, top_n=5):
    if df is None or current_var_name not in df.columns or len(df.columns) < 2:
        return "N/A (Not enough data or variables)"
    correlations = []
    current_series = df[current_var_name]
    if pd.api.types.is_numeric_dtype(current_series.dtype):
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col_name in numeric_cols:
            if col_name == current_var_name: continue
            try:
                corr_val = current_series.corr(df[col_name])
                if pd.notna(corr_val): correlations.append((col_name, corr_val))
            except Exception: pass
        correlations.sort(key=lambda item: abs(item[1]), reverse=True)
        top_items = [f"{name} (Corr:{val:.2f})" for name, val in correlations[:top_n]]
        return ", ".join(top_items) if top_items else "No strong numeric correlations"
    elif current_series.nunique(dropna=False) < 30 or pd.api.types.is_categorical_dtype(current_series.dtype) or current_series.dtype == 'object':
        candidate_cols = [col for col in df.columns if col != current_var_name and (df[col].nunique(dropna=False) < 30 or pd.api.types.is_categorical_dtype(df[col].dtype) or df[col].dtype == 'object')]
        for col_name in candidate_cols:
            try:
                cramers_v = _calculate_cramers_v(current_series, df[col_name])
                if pd.notna(cramers_v) and cramers_v > 0.01: # 0.01 이상만 의미있게
                    correlations.append((col_name, cramers_v))
            except Exception: pass
        correlations.sort(key=lambda item: abs(item[1]), reverse=True)
        top_items = [f"{name} (V:{val:.2f})" for name, val in correlations[:top_n]]
        return ", ".join(top_items) if top_items else "No strong categorical associations"
    return "N/A (Unsupported type for this summary)"

def _get_filtered_variables(df: pd.DataFrame, filter_strength: str, target_var: str = None):
    if df is None or df.empty: return []
    all_cols = df.columns.tolist()
    if filter_strength == "None (All variables)": return all_cols
    weakly_filtered_cols = [col for col in all_cols if df[col].nunique(dropna=False) > 1]
    if filter_strength == "Weak (Exclude obvious non-analytical)": return weakly_filtered_cols
    ranked_cols = []
    if target_var and target_var in df.columns:
        relevance_scores = []
        target_series = df[target_var]
        for col in weakly_filtered_cols:
            if col == target_var: continue
            score = 0.0
            try:
                if pd.api.types.is_numeric_dtype(df[col].dtype) and pd.api.types.is_numeric_dtype(target_series.dtype):
                    score = abs(df[col].corr(target_series))
                elif (df[col].nunique(dropna=False) < 30 or df[col].dtype=='object' or pd.api.types.is_categorical_dtype(df[col].dtype)) and \
                     (target_series.nunique(dropna=False) < 30 or target_series.dtype=='object' or pd.api.types.is_categorical_dtype(target_series.dtype)):
                    score = _calculate_cramers_v(df[col], target_series)
                # TODO: Add logic for Numeric vs Categorical (e.g., F-statistic from ANOVA for numeric feature vs cat target)
            except Exception: pass
            if pd.notna(score): relevance_scores.append((col, score))
        relevance_scores.sort(key=lambda item: item[1], reverse=True)
        ranked_cols = [col for col, score in relevance_scores if pd.notna(score) and score > 0.01]
    if not ranked_cols:
        print("Warning: Using weakly filtered cols for Medium/Strong filter due to no target or scores calculated.")
        ranked_cols = weakly_filtered_cols 
    if filter_strength == "Strong (Top 5-10 relevant)": return ranked_cols[:min(len(ranked_cols), 10)]
    if filter_strength == "Medium (Top 11-20 relevant)": return ranked_cols[:min(len(ranked_cols), 20)]
    return weakly_filtered_cols

def _create_single_var_summary_table(parent_group_tag: str, series: pd.Series, df_for_relations: pd.DataFrame, util_funcs: dict, target_var: str = None):
    desc_stats_display = pd.DataFrame()
    is_numeric = pd.api.types.is_numeric_dtype(series.dtype)
    if is_numeric:
        desc = series.describe()
        stats_to_show = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        valid_stats_to_show = [s for s in stats_to_show if s in desc.index]
        desc_stats_display = desc[valid_stats_to_show].reset_index().rename(columns={'index': 'Statistic', series.name: 'Value'})
        skew, kurt = np.nan, np.nan
        if not series.empty and series.notna().any():
            try: skew = series.skew()
            except TypeError: pass # Some dtypes (like Int64) might raise TypeError
            try: kurt = series.kurtosis()
            except TypeError: pass
        extra_stats = pd.DataFrame({'Statistic': ['Skewness', 'Kurtosis'], 'Value': [f"{skew:.3f}", f"{kurt:.3f}"]})
        desc_stats_display = pd.concat([desc_stats_display, extra_stats], ignore_index=True)
        series_dropna = series.dropna()
        if len(series_dropna) >= 3 and len(series_dropna) < 5000: # Shapiro-Wilk needs at least 3 samples
            try:
                stat_sw, p_sw = stats.shapiro(series_dropna)
                normality = "Likely Normal" if p_sw > 0.05 else "Likely Not Normal"
                norm_stat = pd.DataFrame({'Statistic': ['Shapiro-Wilk p-value', 'Normality'], 'Value': [f"{p_sw:.3f}", normality]})
                desc_stats_display = pd.concat([desc_stats_display, norm_stat], ignore_index=True)
            except Exception as e: print(f"Shapiro-Wilk test error for {series.name}: {e}")
    else:
        value_counts = series.value_counts(dropna=False)
        desc_stats_display = pd.DataFrame({
            'Statistic': ['Count', 'Unique Values', 'Top (Mode)', 'Top Freq', 'Dtype'],
            'Value': [len(series), series.nunique(dropna=False), 
                      str(value_counts.index[0]) if not value_counts.empty else 'N/A', 
                      str(value_counts.iloc[0]) if not value_counts.empty else 'N/A', str(series.dtype)]})
    top_related_vars_str = _get_top_correlated_vars(df_for_relations, series.name, top_n=5)
    related_vars_df = pd.DataFrame({'Statistic': ['Top 5 Related Vars'], 'Value': [top_related_vars_str]})
    desc_stats_display = pd.concat([desc_stats_display, related_vars_df], ignore_index=True)
    if not desc_stats_display.empty:
        table_tag = dpg.generate_uuid()
        with dpg.table(header_row=True, tag=table_tag, parent=parent_group_tag, 
                       borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True, 
                       width=360, height=230, resizable=True, policy=dpg.mvTable_SizingStretchProp):
            util_funcs['create_table_with_data'](table_tag, desc_stats_display)

def _create_single_var_plot(parent_group_tag: str, series: pd.Series, group_by_target_series: pd.Series = None):
    plot_height = 230; plot_width = -1
    plot_label = f"Distribution: {series.name}"
    if group_by_target_series is not None:
        plot_label += f" (Grouped by {group_by_target_series.name})"

    with dpg.plot(label=plot_label, height=plot_height, width=plot_width, parent=parent_group_tag):
        xaxis_tag = dpg.add_plot_axis(dpg.mvXAxis, label=series.name)
        yaxis_tag = dpg.generate_uuid(); dpg.add_plot_axis(dpg.mvYAxis, label="Density / Frequency", tag=yaxis_tag)
        legend_tag = dpg.add_plot_legend(parent=dpg.last_item()) # Plot legend

        series_cleaned_for_plot = series.dropna()
        if series_cleaned_for_plot.empty: 
            dpg.add_text("No data to plot after NA removal.", parent=parent_group_tag); return

        if pd.api.types.is_numeric_dtype(series.dtype):
            if group_by_target_series is not None and group_by_target_series.nunique(dropna=False) > 1:
                unique_groups = sorted(group_by_target_series.dropna().unique())
                for group_name in unique_groups:
                    group_data = series_cleaned_for_plot[group_by_target_series == group_name]
                    if len(group_data) < 2: continue
                    hist_counts, hist_bins = np.histogram(group_data, bins='auto', density=True)
                    dpg.add_stair_series(hist_bins, np.append(hist_counts,0), label=f"T={group_name}", parent=yaxis_tag)
                    try:
                        kde = stats.gaussian_kde(group_data)
                        x_vals = np.linspace(group_data.min(), group_data.max(), 150)
                        y_vals = kde(x_vals)
                        dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), label=f"KDE(T={group_name})", parent=yaxis_tag)
                    except Exception: pass
            else:
                if len(series_cleaned_for_plot) > 1:
                    hist_counts, hist_bins = np.histogram(series_cleaned_for_plot, bins='auto', density=True)
                    dpg.add_stair_series(hist_bins, np.append(hist_counts,0), label="Hist", parent=yaxis_tag)
                    try:
                        kde = stats.gaussian_kde(series_cleaned_for_plot)
                        x_vals = np.linspace(series_cleaned_for_plot.min(), series_cleaned_for_plot.max(), 200)
                        y_vals = kde(x_vals)
                        dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), label="KDE", parent=yaxis_tag)
                    except Exception: pass
        else: # 범주형 변수
            top_n_categories = 10
            if group_by_target_series is not None and group_by_target_series.nunique(dropna=False) > 1:
                # 그룹별 범주형 데이터 시각화 (col_wrap 효과를 위한 DPG 구현 시도)
                # 각 타겟 그룹에 대해 별도의 작은 막대 그래프를 옆으로 나열 (최대 5개 그룹)
                dpg.delete_item(legend_tag) # 그룹별 플롯에서는 각 플롯에 레이블보다 전체 설명이 나을 수 있음
                
                unique_target_groups = sorted(group_by_target_series.dropna().unique())[:5] # 최대 5개 그룹
                
                # 가로로 플롯들을 담을 그룹 (스크롤 가능하게)
                # with dpg.child_window(parent=parent_group_tag, autosize_x=True, height=plot_height - 30):
                #     with dpg.group(horizontal=True):
                #         for i, group_name in enumerate(unique_target_groups):
                #             with dpg.plot(label=f"{series.name} (T={group_name})", height=plot_height-40, width=max(150, (plot_width-20*len(unique_target_groups))//len(unique_target_groups) ), parent=dpg.last_item()):
                #                 # ... (이 부분은 복잡하여, 우선 전체 빈도 표시로 대체)
                dpg.add_text("Grouped categorical plot (col_wrap style) is complex for native DPG.", parent=parent_group_tag)
                dpg.add_text("Showing overall frequency. Consider external libraries for advanced grouped categoricals.", parent=parent_group_tag)
                value_counts = series_cleaned_for_plot.value_counts(dropna=False).nlargest(top_n_categories)
                x_pos = list(range(len(value_counts)))
                bar_labels = [str(val) for val in value_counts.index.tolist()]
                dpg.add_bar_series(x_pos, value_counts.values.tolist(), weight=0.7, label="Overall Freq.", parent=yaxis_tag)
                if bar_labels and dpg.does_item_exist(xaxis_tag): dpg.set_axis_ticks(xaxis_tag, tuple(zip(bar_labels, x_pos)))
            else:
                value_counts = series_cleaned_for_plot.value_counts(dropna=False).nlargest(top_n_categories)
                x_pos = list(range(len(value_counts)))
                bar_labels = [str(val) for val in value_counts.index.tolist()]
                dpg.add_bar_series(x_pos, value_counts.values.tolist(), weight=0.7, label="Frequency", parent=yaxis_tag)
                if bar_labels and dpg.does_item_exist(xaxis_tag): dpg.set_axis_ticks(xaxis_tag, tuple(zip(bar_labels, x_pos)))
                dpg.delete_item(legend_tag)


def _show_alert_modal(title: str, message: str):
    if not dpg.is_dearpygui_running(): return
    # 모달 태그를 고유하게 만들기 위해 접두사와 메시지 일부(해시 등) 사용 가능
    modal_tag = TAG_SVA_ALERT_MODAL_PREFIX + str(hash(message))[:8] # 간단한 고유화
    
    # 이전 동일 메시지 모달이 있다면 닫고 새로 띄우거나, 내용만 업데이트
    if dpg.does_item_exist(modal_tag) and dpg.is_item_shown(modal_tag):
        dpg.configure_item(modal_tag, show=False) # 일단 닫고
        dpg.delete_item(modal_tag) # 삭제 후 재생성 (더 확실)

    with dpg.window(label=title, modal=True, show=True, tag=modal_tag, no_close=False, pos=[400,300], width=400, autosize=True):
        dpg.add_text(message, wrap=380)
        dpg.add_spacer(height=20)
        with dpg.group(horizontal=True):
            dpg.add_button(label="OK", width=-1, user_data=modal_tag, callback=lambda s, a, u: dpg.configure_item(u, show=False))

def _apply_sva_filters_and_run(main_callbacks: dict):
    if not dpg.is_dearpygui_running(): return
    current_df = main_callbacks['get_current_df']()
    util_funcs = main_callbacks['get_util_funcs']()
    target_var = main_callbacks['get_selected_target_variable']()
    
    progress_modal_tag = TAG_SVA_PROGRESS_MODAL
    progress_text_tag = TAG_SVA_PROGRESS_TEXT
    results_child_window_tag = TAG_SVA_RESULTS_CHILD_WINDOW
    filter_strength_radio_tag = TAG_SVA_FILTER_STRENGTH_RADIO
    group_by_target_checkbox_tag = TAG_SVA_GROUP_BY_TARGET_CHECKBOX

    if not dpg.does_item_exist(progress_modal_tag):
        with dpg.window(label="Processing SVA", modal=True, show=False, tag=progress_modal_tag, no_close=True, no_title_bar=True, pos=[500,400], width=350, height=70):
            dpg.add_text("Analyzing variables, please wait...", tag=progress_text_tag)
    dpg.configure_item(progress_modal_tag, show=True)
    dpg.set_value(progress_text_tag, "SVA: Preparing analysis...")
    dpg.split_frame()

    if not dpg.does_item_exist(results_child_window_tag):
        print(f"Error: SVA results child window '{results_child_window_tag}' not found.")
        if dpg.does_item_exist(progress_modal_tag): dpg.configure_item(progress_modal_tag, show=False)
        return
    dpg.delete_item(results_child_window_tag, children_only=True)

    if current_df is None:
        dpg.add_text("Load data first to perform Single Variable Analysis.", parent=results_child_window_tag)
        if dpg.does_item_exist(progress_modal_tag): dpg.configure_item(progress_modal_tag, show=False)
        return

    filter_strength = dpg.get_value(filter_strength_radio_tag) if dpg.does_item_exist(filter_strength_radio_tag) else "Weak (Exclude obvious non-analytical)"
    group_by_target_flag = dpg.get_value(group_by_target_checkbox_tag) if dpg.does_item_exist(group_by_target_checkbox_tag) else False
    
    target_series_for_grouping = None
    if group_by_target_flag:
        if target_var and target_var in current_df.columns:
            unique_target_values = current_df[target_var].nunique(dropna=False)
            if unique_target_values >= 2 and unique_target_values <= 5 :
                target_series_for_grouping = current_df[target_var]
            else:
                _show_alert_modal("Grouping Warning", f"Target variable '{target_var}' has {unique_target_values} unique values. Grouping disabled (requires 2-5 distinct values for clarity).")
                if dpg.does_item_exist(group_by_target_checkbox_tag): dpg.set_value(group_by_target_checkbox_tag, False)
        else:
            _show_alert_modal("Grouping Info", "Target variable not selected or invalid for grouping. Grouping disabled.")
            if dpg.does_item_exist(group_by_target_checkbox_tag): dpg.set_value(group_by_target_checkbox_tag, False)
        # Ensure flag is updated if an alert was shown and checkbox was unticked
        group_by_target_flag = dpg.get_value(group_by_target_checkbox_tag) if dpg.does_item_exist(group_by_target_checkbox_tag) else False


    filtered_cols = _get_filtered_variables(current_df, filter_strength, target_var)
    if not filtered_cols:
        dpg.add_text("No variables to display based on current filter.", parent=results_child_window_tag)
        if dpg.does_item_exist(progress_modal_tag): dpg.configure_item(progress_modal_tag, show=False)
        return

    total_vars = len(filtered_cols)
    for i, col_name in enumerate(filtered_cols):
        if dpg.does_item_exist(progress_text_tag): dpg.set_value(progress_text_tag, f"SVA: Analyzing {col_name} ({i+1}/{total_vars})")
        dpg.split_frame() 
        var_section_tag = TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX + "".join(filter(str.isalnum, str(col_name)))
        # Ensure parent exists before adding to it
        if not dpg.does_item_exist(results_child_window_tag): break # Parent disappeared
        if dpg.does_item_exist(var_section_tag): dpg.delete_item(var_section_tag) # Delete if exists, then recreate group for content

        with dpg.group(tag=var_section_tag, parent=results_child_window_tag): # Recreate group for fresh content
            with dpg.group(horizontal=True): # Horizontal layout for info and plot
                info_group_tag = dpg.generate_uuid()
                with dpg.group(tag=info_group_tag, width=390):
                    dpg.add_text(f"Variable: {util_funcs['format_text_for_display'](col_name, 40)}", color=(255, 255, 0), wrap=380)
                    dpg.add_text(f"Type: {str(current_df[col_name].dtype)}", wrap=380)
                    _create_single_var_summary_table(info_group_tag, current_df[col_name], current_df, util_funcs, target_var)
                dpg.add_spacer(width=10)
                plot_group_tag = dpg.generate_uuid()
                with dpg.group(tag=plot_group_tag):
                     _create_single_var_plot(plot_group_tag, current_df[col_name], target_series_for_grouping if group_by_target_flag else None)
            dpg.add_separator() 
    
    if dpg.does_item_exist(progress_modal_tag): dpg.configure_item(progress_modal_tag, show=False)
    print("SVA processing finished.")

# --- MVA Helper Functions (Full Implementations) ---
def _get_numeric_cols(df: pd.DataFrame):
    if df is None: return []
    return df.select_dtypes(include=np.number).columns.tolist()

def _get_categorical_cols(df: pd.DataFrame, max_unique_for_cat=20):
    if df is None: return []
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].nunique(dropna=False) <= max_unique_for_cat and col not in cat_cols :
            cat_cols.append(col)
    return list(set(cat_cols))

def _run_correlation_analysis(df: pd.DataFrame, util_funcs: dict):
    if not dpg.is_dearpygui_running(): return
    heatmap_plot_tag = TAG_MVA_CORR_HEATMAP_PLOT
    corr_table_tag = TAG_MVA_CORR_TABLE # This is the table itself
    corr_tab_tag = TAG_MVA_CORR_TAB     # This is the parent tab for the table
    
    # Clear previous results
    if dpg.does_item_exist(heatmap_plot_tag): dpg.delete_item(heatmap_plot_tag, children_only=True)
    if dpg.does_item_exist(corr_table_tag): dpg.delete_item(corr_table_tag) # Delete the table item to recreate

    if df is None:
        dpg.add_text("Load data first.", parent=heatmap_plot_tag if dpg.does_item_exist(heatmap_plot_tag) else corr_tab_tag)
        return

    numeric_cols = _get_numeric_cols(df)
    if len(numeric_cols) < 2:
        dpg.add_text("Not enough numeric columns for correlation analysis.", parent=heatmap_plot_tag if dpg.does_item_exist(heatmap_plot_tag) else corr_tab_tag)
        return

    corr_matrix = df[numeric_cols].corr(method='pearson') # Or 'spearman'
    heatmap_data = corr_matrix.values.flatten().tolist()
    rows, cols = corr_matrix.shape
    col_labels = corr_matrix.columns.tolist() # Same as row_labels for square matrix

    with dpg.plot(label="Correlation Heatmap", height=450, width=-1, parent=heatmap_plot_tag, equal_aspects=True):
        # dpg.add_plot_legend() # Not typical for heatmap itself, colorscale is the legend
        xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="")
        if col_labels: dpg.set_axis_ticks(xaxis, tuple(zip(col_labels, list(range(cols)))))
        yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="")
        if col_labels: dpg.set_axis_ticks(yaxis, tuple(zip(col_labels, list(range(rows))))) # Use col_labels for y-axis too
        
        dpg.add_heat_series(heatmap_data, rows=rows, cols=cols, 
                            scale_min=-1.0, scale_max=1.0, 
                            format='%.2f', parent=yaxis, show_tooltips=True) # Added show_tooltips

    # Display highly correlated pairs in a table below the heatmap plot (within the same tab)
    parent_for_corr_table = corr_tab_tag # Add table directly to the tab
    dpg.add_text("Highly Correlated Pairs (|Correlation| > 0.7):", parent=parent_for_corr_table)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i): # Avoid duplicates and self-correlation
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                pair_info = {
                    "Variable 1": corr_matrix.columns[i],
                    "Variable 2": corr_matrix.columns[j],
                    "Correlation": f"{corr_matrix.iloc[i, j]:.3f}"}
                high_corr_pairs.append(pair_info)
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        with dpg.table(header_row=True, tag=corr_table_tag, parent=parent_for_corr_table,
                       resizable=True, policy=dpg.mvTable_SizingFixedFit, scrollY=True, height=150,
                       borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
            util_funcs['create_table_with_data'](corr_table_tag, high_corr_df)
    else:
        dpg.add_text("No pairs with |correlation| > 0.7 found.", parent=parent_for_corr_table)


def _run_pair_plot_analysis(df: pd.DataFrame, selected_vars: list, hue_var: str, util_funcs: dict):
    results_group = TAG_MVA_PAIRPLOT_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group) or df is None: return
    dpg.delete_item(results_group, children_only=True)

    if not selected_vars or not isinstance(selected_vars, list) or len(selected_vars) < 2:
        dpg.add_text("Please select 2 or more numeric variables for the Pair Plot.", parent=results_group)
        return
    
    max_pair_plot_vars = 5 
    if len(selected_vars) > max_pair_plot_vars:
        _show_alert_modal("Pair Plot Limit", f"Too many variables ({len(selected_vars)}). Please select up to {max_pair_plot_vars}.")
        selected_vars = selected_vars[:max_pair_plot_vars] # Limit automatically for now
        # return # Or just stop

    valid_selected_vars = [var for var in selected_vars if var in df.columns and pd.api.types.is_numeric_dtype(df[var].dtype)]
    if len(valid_selected_vars) < 2:
        dpg.add_text("Not enough valid numeric variables selected.", parent=results_group); return
    
    selected_vars = valid_selected_vars # Use only valid numeric vars

    hue_categories = None
    hue_series = None
    if hue_var and hue_var in df.columns:
        hue_series = df[hue_var]
        if hue_series.nunique(dropna=False) > 7: # Hue 카테고리 최대 7개 권장
            _show_alert_modal("Hue Variable Warning", f"Hue variable '{hue_var}' has {hue_series.nunique()} unique values. Max 7 recommended for clarity. Hue disabled.")
            hue_var = None; hue_series = None
        else:
            hue_categories = sorted(hue_series.dropna().unique())
            # TODO: Define a color map for hue_categories
            print(f"Using hue variable: {hue_var} with categories: {hue_categories}")
    
    n_vars = len(selected_vars)
    plot_cell_size = 200 # 각 subplot 크기

    dpg.add_text(f"Pair Plot for: {', '.join(selected_vars)}" + (f" (Hue: {hue_var})" if hue_var else ""), parent=results_group)
    
    # DPG Pair Plot (Simplified)
    # This implementation is basic. For rich features, Matplotlib/Seaborn image is better.
    with dpg.group(parent=results_group): # Outer group for all plots
        for i in range(n_vars):
            with dpg.group(horizontal=True): # Each row of plots
                for j in range(n_vars):
                    var_y = selected_vars[i]
                    var_x = selected_vars[j]
                    plot_label = f"{var_y} vs {var_x}" if i != j else f"Dist: {var_x}"
                    
                    with dpg.plot(width=plot_cell_size, height=plot_cell_size, label=plot_label):
                        xaxis = dpg.add_plot_axis(dpg.mvXAxis, label=var_x if i == n_vars - 1 else "", no_tick_labels=(i != n_vars -1))
                        yaxis = dpg.add_plot_axis(dpg.mvYAxis, label=var_y if j == 0 else "", no_tick_labels=(j != 0), tag=dpg.generate_uuid())

                        if i == j: # Diagonal: Histogram
                            series_diag = df[var_x].dropna()
                            if not series_diag.empty and len(series_diag) > 1:
                                hist_data = np.histogram(series_diag, bins='auto', density=True)
                                dpg.add_stair_series(hist_data[1], np.append(hist_data[0],0), label="Hist", parent=dpg.last_item())
                        else: # Off-diagonal: Scatter plot
                            series_x_data = df[var_x]
                            series_y_data = df[var_y]
                            
                            if hue_series is not None and hue_categories:
                                for cat_idx, category in enumerate(hue_categories):
                                    mask = (hue_series == category)
                                    x_vals = series_x_data[mask].dropna().tolist()
                                    y_vals = series_y_data[mask].dropna().tolist()
                                    # Align x and y for scatter (simple approach by index, more robust would be merge)
                                    # This simple alignment might not be perfect if NaNs are not aligned
                                    min_len = min(len(x_vals), len(y_vals)) 
                                    if min_len > 0:
                                        # TODO: Assign different colors per category
                                        dpg.add_scatter_series(x_vals[:min_len], y_vals[:min_len], label=str(category), parent=dpg.last_item())
                                if len(hue_categories) > 1 : dpg.add_plot_legend(parent=dpg.last_item(2)) # Add legend to plot
                            else:
                                aligned_df = pd.concat([series_x_data, series_y_data], axis=1).dropna()
                                if not aligned_df.empty:
                                     dpg.add_scatter_series(aligned_df.iloc[:,0].tolist(), aligned_df.iloc[:,1].tolist(), parent=dpg.last_item())


def _run_target_variable_analysis(df: pd.DataFrame, target_var: str, feature_var: str, util_funcs: dict):
    results_group = TAG_MVA_TARGET_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group) or df is None or \
       not target_var or target_var not in df.columns or \
       not feature_var or feature_var not in df.columns or target_var == feature_var:
        if dpg.does_item_exist(results_group):
            dpg.delete_item(results_group, children_only=True)
            dpg.add_text("Select valid target and feature variables for analysis.", parent=results_group)
        return

    dpg.delete_item(results_group, children_only=True)
    dpg.add_text(f"Analysis: Feature '{feature_var}' vs Target '{target_var}'", parent=results_group)
    dpg.add_separator(parent=results_group)

    target_series = df[target_var]
    feature_series = df[feature_var]

    # Case 1: Numeric Target
    if pd.api.types.is_numeric_dtype(target_series.dtype):
        if pd.api.types.is_numeric_dtype(feature_series.dtype): # Numeric Feature vs Numeric Target
            dpg.add_text(f"Correlation: {feature_series.corr(target_series):.3f}", parent=results_group)
            with dpg.plot(label=f"Scatter: {feature_var} by {target_var}", height=300, width=-1, parent=results_group):
                dpg.add_plot_axis(dpg.mvXAxis, label=feature_var)
                yaxis_scatter = dpg.add_plot_axis(dpg.mvYAxis, label=target_var)
                aligned_df = pd.concat([feature_series, target_series], axis=1).dropna()
                if not aligned_df.empty:
                    dpg.add_scatter_series(aligned_df.iloc[:,0].tolist(), aligned_df.iloc[:,1].tolist(), parent=yaxis_scatter)
        
        elif feature_series.nunique(dropna=False) < 20: # Categorical-like Feature vs Numeric Target
            dpg.add_text("Grouped Statistics (Feature's Categories vs Numeric Target):", parent=results_group)
            grouped_stats = df.groupby(feature_var)[target_var].agg(['mean', 'median', 'std', 'count']).reset_index()
            table_tag = dpg.generate_uuid()
            with dpg.table(header_row=True, tag=table_tag, parent=results_group, resizable=True, height=150, scrollY=True):
                util_funcs['create_table_with_data'](table_tag, grouped_stats)
            dpg.add_text("Consider Box/Violin plots (e.g., via Matplotlib image) for visual comparison.", parent=results_group)

    # Case 2: Categorical Target (or low-cardinality numeric treated as categorical)
    elif target_series.nunique(dropna=False) < 20: 
        if pd.api.types.is_numeric_dtype(feature_series.dtype): # Numeric Feature vs Categorical Target
            dpg.add_text("Grouped Statistics (Target's Categories vs Numeric Feature):", parent=results_group)
            grouped_stats = df.groupby(target_var)[feature_var].agg(['mean', 'median', 'std', 'count']).reset_index()
            table_tag = dpg.generate_uuid()
            with dpg.table(header_row=True, tag=table_tag, parent=results_group, resizable=True, height=150, scrollY=True):
                util_funcs['create_table_with_data'](table_tag, grouped_stats)
            dpg.add_text("Consider overlaid Histograms/Density plots (e.g., via Matplotlib image).", parent=results_group)

            # Basic DPG Grouped Plot (KDE for each target category)
            # This is a simplified version of "col_wrap" idea for target analysis
            unique_target_cats = sorted(target_series.dropna().unique())[:5] # Max 5 categories for this plot
            if len(unique_target_cats) >=2:
                with dpg.plot(label=f"Density of '{feature_var}' by '{target_var}' categories", height=300, width=-1, parent=results_group):
                    dpg.add_plot_axis(dpg.mvXAxis, label=feature_var)
                    yaxis_kde_group = dpg.add_plot_axis(dpg.mvYAxis, label="Density")
                    dpg.add_plot_legend(parent=dpg.last_item())
                    for cat_val in unique_target_cats:
                        subset_data = feature_series[target_series == cat_val].dropna()
                        if len(subset_data) > 1:
                            try:
                                kde = stats.gaussian_kde(subset_data)
                                x_vals = np.linspace(subset_data.min(), subset_data.max(), 150)
                                y_vals = kde(x_vals)
                                dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), label=f"Target={cat_val}", parent=yaxis_kde_group)
                            except Exception: pass


        elif feature_series.nunique(dropna=False) < 20: # Categorical Feature vs Categorical Target
            dpg.add_text("Crosstabulation (Feature vs Target):", parent=results_group)
            try:
                crosstab_df = pd.crosstab(df[feature_var], df[target_var], normalize='index').mul(100).round(1).astype(str) + '%'
                crosstab_df_abs = pd.crosstab(df[feature_var], df[target_var])
                
                dpg.add_text("Counts:", parent=results_group)
                table_tag_abs = dpg.generate_uuid()
                with dpg.table(header_row=True, tag=table_tag_abs, parent=results_group, resizable=True, height=150, scrollY=True):
                    util_funcs['create_table_with_data'](table_tag_abs, crosstab_df_abs.reset_index())
                
                dpg.add_text("Row Percentages:", parent=results_group)
                table_tag_norm = dpg.generate_uuid()
                with dpg.table(header_row=True, tag=table_tag_norm, parent=results_group, resizable=True, height=150, scrollY=True):
                    util_funcs['create_table_with_data'](table_tag_norm, crosstab_df.reset_index())
                dpg.add_text("Consider Stacked/Grouped Bar charts (e.g., via Matplotlib image).", parent=results_group)
            except Exception as e:
                dpg.add_text(f"Error creating crosstab: {e}", parent=results_group)
    else:
        dpg.add_text(f"Target variable '{target_var}' type or cardinality not suitable for this simplified analysis.", parent=results_group)


# --- Main UI Creation & Update ---
def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    # ... (이전 답변에서 제공된 create_ui 함수 본문과 동일하게 SVA 및 MVA 탭 구조 생성) ...
    # SVA 탭 메뉴바 콜백은 _apply_sva_filters_and_run(main_callbacks)
    # MVA 탭 버튼 콜백은 각각의 _run... 함수 (예: _run_correlation_analysis 등)
    main_callbacks['register_step_group_tag'](step_name, TAG_EDA_GROUP)
    with dpg.group(tag=TAG_EDA_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---"); dpg.add_separator()
        with dpg.tab_bar(tag=TAG_EDA_MAIN_TAB_BAR):
            with dpg.tab(label="Single Variable Analysis", tag=TAG_SVA_TAB):
                with dpg.menu_bar():
                    with dpg.menu(label="Variable Filter"):
                        dpg.add_text("Filter Strength (Default: Weak)")
                        dpg.add_radio_button(
                            items=["Strong (Top 5-10 relevant)", "Medium (Top 11-20 relevant)", "Weak (Exclude obvious non-analytical)", "None (All variables)"],
                            tag=TAG_SVA_FILTER_STRENGTH_RADIO, default_value="Weak (Exclude obvious non-analytical)",
                            callback=lambda: _apply_sva_filters_and_run(main_callbacks))
                        dpg.add_separator()
                        dpg.add_text("Filter Algorithm (Simplified):", wrap = 250)
                        dpg.add_text("- Strong/Medium: Relevance to Target (if set) or other heuristics.", wrap = 250)
                        dpg.add_text("- Weak: Excludes single-value variables.", wrap = 250)
                    dpg.add_menu_item(label="Group by Target (Categorical Target, 2-5 Categories)", check=True,
                                     tag=TAG_SVA_GROUP_BY_TARGET_CHECKBOX, default_value=False,
                                     callback=lambda: _apply_sva_filters_and_run(main_callbacks))
                dpg.add_separator()
                with dpg.child_window(tag=TAG_SVA_RESULTS_CHILD_WINDOW, border=False):
                    dpg.add_text("Select filter options or load data to view Single Variable Analysis.")

            with dpg.tab(label="Multivariate Analysis", tag=TAG_MVA_TAB):
                with dpg.tab_bar(tag=TAG_MVA_SUB_TAB_BAR):
                    with dpg.tab(label="Correlation Analysis", tag=TAG_MVA_CORR_TAB):
                        dpg.add_button(label="Run Correlation Analysis (Numeric Vars)", tag=TAG_MVA_CORR_RUN_BUTTON,
                                       callback=lambda: _run_correlation_analysis(main_callbacks['get_current_df'](), main_callbacks['get_util_funcs']()))
                        dpg.add_child_window(tag=TAG_MVA_CORR_HEATMAP_PLOT, border=True, height=480) 
                        # Correlation table is added by _run_correlation_analysis directly to TAG_MVA_CORR_TAB

                    with dpg.tab(label="Pair Plot", tag=TAG_MVA_PAIRPLOT_TAB):
                        dpg.add_text("Select numeric variables for Pair Plot (Max 5 recommended).")
                        dpg.add_listbox(tag=TAG_MVA_PAIRPLOT_VAR_SELECTOR, width=-1, num_items=8) # 아이템 수 늘림
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
                        dpg.add_child_window(tag=TAG_MVA_TARGET_RESULTS_GROUP, border=True, height=-1) # 높이 -1로

    main_callbacks['register_module_updater'](step_name, lambda df, orig_df, utils, fp, mc=main_callbacks: update_ui(df, mc))
    update_ui(main_callbacks['get_current_df'](), main_callbacks)


def update_ui(current_df: pd.DataFrame, main_callbacks: dict):
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
            except Exception: pass # 값 설정 오류는 무시 (DPG 내부 문제일 수 있음)
            
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
    
    # SVA 탭의 자동 실행은 _apply_sva_filters_and_run 콜백에서 처리되므로,
    # update_ui에서는 특별히 SVA 내용을 다시 그리지 않습니다.
    # 단, 최초 데이터 로드 시에는 한 번 실행해주는 것이 좋습니다.
    # 이 로직은 main_app.py에서 EDA 탭이 처음 활성화될 때 _apply_sva_filters_and_run(main_callbacks)를
    # 한 번 호출하는 방식으로 관리하는 것이 더 명확할 수 있습니다.
    # 예를 들어, main_app.py의 switch_step_view에서:
    # if "2. Exploratory Data Analysis (EDA)" in user_data_step_name and not _eda_sva_initialized:
    #     # EDA 모듈의 SVA 초기 실행 함수 호출 (main_callbacks를 통해)
    #     # _eda_sva_initialized = True
    print("EDA UI selectors updated.")