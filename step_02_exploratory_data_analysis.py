# step_02_exploratory_data_analysis.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from scipy import stats # For skewness, kurtosis, normality tests

# --- UI Element Tags ---
TAG_EDA_GROUP = "step2_eda_group"
TAG_EDA_MAIN_TAB_BAR = "step2_eda_main_tab_bar"

# Single Variable Analysis Tab
TAG_SVA_TAB = "step2_sva_tab"
TAG_SVA_VARIABLE_COMBO = "step2_sva_variable_combo"
TAG_SVA_RUN_BUTTON = "step2_sva_run_button"
TAG_SVA_RESULTS_GROUP = "step2_sva_results_group"
TAG_SVA_SUMMARY_TABLE = "step2_sva_summary_table"
TAG_SVA_DISTRIBUTION_PLOT = "step2_sva_distribution_plot"

# Multivariate Analysis Tab
TAG_MVA_TAB = "step2_mva_tab"
TAG_MVA_SUB_TAB_BAR = "step2_mva_sub_tab_bar"

# MVA - Correlation Tab
TAG_MVA_CORR_TAB = "step2_mva_corr_tab"
TAG_MVA_CORR_RUN_BUTTON = "step2_mva_corr_run_button"
TAG_MVA_CORR_HEATMAP_PLOT = "step2_mva_corr_heatmap_plot"
TAG_MVA_CORR_TABLE = "step2_mva_corr_table" # To show high correlation pairs

# MVA - Pair Plot Tab
TAG_MVA_PAIRPLOT_TAB = "step2_mva_pairplot_tab"
TAG_MVA_PAIRPLOT_VAR_SELECTOR = "step2_mva_pairplot_var_selector" # For selecting multiple vars
TAG_MVA_PAIRPLOT_HUE_COMBO = "step2_mva_pairplot_hue_combo"
TAG_MVA_PAIRPLOT_RUN_BUTTON = "step2_mva_pairplot_run_button"
TAG_MVA_PAIRPLOT_RESULTS_GROUP = "step2_mva_pairplot_results_group" # Pair plots will be added here

# MVA - Target Analysis Tab (if target is selected)
TAG_MVA_TARGET_TAB = "step2_mva_target_tab"
TAG_MVA_TARGET_INFO_TEXT = "step2_mva_target_info_text"
TAG_MVA_TARGET_FEATURE_COMBO = "step2_mva_target_feature_combo" # Feature to compare with target
TAG_MVA_TARGET_RUN_BUTTON = "step2_mva_target_run_button"
TAG_MVA_TARGET_RESULTS_GROUP = "step2_mva_target_results_group"

# --- Helper Functions for EDA Logic ---

def _get_numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=np.number).columns.tolist()

def _get_categorical_cols(df: pd.DataFrame, max_unique_for_cat=20):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Include numeric cols with few unique values if desired (e.g. binary 0/1)
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].nunique() <= max_unique_for_cat and col not in cat_cols : # 예를 들어 고유값 20개 이하
             # 0/1 변수도 여기에 포함될 수 있음 (타입 정리 단계에서 Numeric(Binary)로 남겼다면)
            cat_cols.append(col)
    return list(set(cat_cols)) # 중복 제거


def _run_single_variable_analysis(col_name: str, df: pd.DataFrame, util_funcs: dict):
    """지정된 단일 변수에 대한 EDA를 수행하고 DPG에 결과를 표시합니다."""
    if not dpg.does_item_exist(TAG_SVA_RESULTS_GROUP) or df is None or col_name not in df.columns:
        if dpg.does_item_exist(TAG_SVA_RESULTS_GROUP):
            dpg.delete_item(TAG_SVA_RESULTS_GROUP, children_only=True)
            dpg.add_text("Select a variable and run analysis.", parent=TAG_SVA_RESULTS_GROUP)
        return

    dpg.delete_item(TAG_SVA_RESULTS_GROUP, children_only=True) # Clear previous results
    series = df[col_name]
    
    # 1. Basic Info
    dpg.add_text(f"Analysis for Variable: {col_name}", parent=TAG_SVA_RESULTS_GROUP)
    dpg.add_text(f"Data Type: {series.dtype}", parent=TAG_SVA_RESULTS_GROUP)
    missing_count = series.isnull().sum()
    missing_percent = missing_count / len(series) * 100
    dpg.add_text(f"Missing Values: {missing_count} ({missing_percent:.2f}%)", parent=TAG_SVA_RESULTS_GROUP)
    dpg.add_text(f"Unique Values: {series.nunique()}", parent=TAG_SVA_RESULTS_GROUP)
    dpg.add_separator(parent=TAG_SVA_RESULTS_GROUP)

    # 2. Descriptive Statistics
    if pd.api.types.is_numeric_dtype(series):
        dpg.add_text("Descriptive Statistics (Numeric):", parent=TAG_SVA_RESULTS_GROUP)
        desc_stats = series.describe().reset_index().rename(columns={'index': 'Statistic', col_name: 'Value'})
        # DPG 테이블로 표시 (util_funcs['create_table_with_data'] 활용)
        temp_table_tag = f"sva_desc_table_{col_name}" # Ensure unique tag
        with dpg.child_window(height=180, parent=TAG_SVA_RESULTS_GROUP, tag=temp_table_tag+"_child"):
            with dpg.table(header_row=True, tag=temp_table_tag, parent=temp_table_tag+"_child"):
                util_funcs['create_table_with_data'](temp_table_tag, desc_stats)

        skew = series.skew()
        kurt = series.kurtosis()
        dpg.add_text(f"Skewness: {skew:.3f}", parent=TAG_SVA_RESULTS_GROUP)
        dpg.add_text(f"Kurtosis: {kurt:.3f}", parent=TAG_SVA_RESULTS_GROUP)
        
        # Normality Test (Shapiro-Wilk for samples < 5000)
        if len(series.dropna()) > 2 and len(series.dropna()) < 5000:
            stat, p_value = stats.shapiro(series.dropna())
            alpha = 0.05
            normality = "Likely Normal" if p_value > alpha else "Likely Not Normal"
            dpg.add_text(f"Shapiro-Wilk Normality Test: p-value={p_value:.3f} ({normality})", parent=TAG_SVA_RESULTS_GROUP)
        dpg.add_separator(parent=TAG_SVA_RESULTS_GROUP)

        # Distribution Plot (Histogram + KDE)
        dpg.add_text("Distribution Plot:", parent=TAG_SVA_RESULTS_GROUP)
        with dpg.plot(label=f"Distribution of {col_name}", height=300, width=-1, parent=TAG_SVA_RESULTS_GROUP, tag=TAG_SVA_DISTRIBUTION_PLOT):
            dpg.add_plot_axis(dpg.mvXAxis, label=col_name)
            dpg.add_plot_axis(dpg.mvYAxis, label="Frequency / Density", tag=f"sva_yaxis_{col_name}")
            
            # Histogram
            hist_data = np.histogram(series.dropna(), bins='auto') # 'auto' for automatic bin selection
            dpg.add_histogram_series(hist_data[1][:-1], hist_data[0], label="Histogram", parent=f"sva_yaxis_{col_name}", bins=len(hist_data[0]))
            
            # KDE (approximated, if scipy is available)
            try:
                kde = stats.gaussian_kde(series.dropna())
                x_vals = np.linspace(series.min(), series.max(), 200)
                y_vals = kde(x_vals)
                # Scale KDE to histogram for better visualization - this needs careful scaling
                # For simplicity, plot separately or normalize y-axis
                # Here, just plotting KDE, might need a secondary y-axis or scaling
                # dpg.add_line_series(x_vals.tolist(), y_vals.tolist(), label="KDE", parent=f"sva_yaxis_{col_name}")
                print(f"KDE for {col_name} calculated. Min/Max Y: {y_vals.min()}/{y_vals.max()}")
            except Exception as e:
                print(f"Could not generate KDE plot for {col_name}: {e}")

    elif pd.api.types.is_categorical_dtype(series) or series.dtype == 'object' or series.nunique() < 20:
        dpg.add_text("Frequency Distribution (Categorical/Low Cardinality):", parent=TAG_SVA_RESULTS_GROUP)
        value_counts = series.value_counts(dropna=False).reset_index()
        value_counts.columns = ['Value', 'Count']
        value_counts['Percentage'] = (value_counts['Count'] / len(series) * 100).round(2).astype(str) + '%'
        
        temp_table_tag = f"sva_freq_table_{col_name}"
        with dpg.child_window(height=200, parent=TAG_SVA_RESULTS_GROUP, tag=temp_table_tag+"_child"):
            with dpg.table(header_row=True, tag=temp_table_tag, parent=temp_table_tag+"_child"):
                 util_funcs['create_table_with_data'](temp_table_tag, value_counts)
        dpg.add_separator(parent=TAG_SVA_RESULTS_GROUP)

        # Bar Plot for top N categories
        dpg.add_text("Bar Plot (Top N Categories):", parent=TAG_SVA_RESULTS_GROUP)
        top_n = min(10, series.nunique()) # Show top 10 categories
        top_n_counts = series.value_counts().nlargest(top_n)
        with dpg.plot(label=f"Top {top_n} Categories of {col_name}", height=300, width=-1, parent=TAG_SVA_RESULTS_GROUP, tag=f"sva_barplot_{col_name}"):
            dpg.add_plot_axis(dpg.mvXAxis, label="Category", no_gridlines=True)
            dpg.set_axis_limits_auto(dpg.mvXAxis) # Ensure X axis fits labels
            dpg.add_plot_axis(dpg.mvYAxis, label="Count", tag=f"sva_bar_yaxis_{col_name}")
            # DPG bar series expects numeric x-values for labels. We create positions.
            x_pos = list(range(len(top_n_counts)))
            bar_labels = [str(val) for val in top_n_counts.index.tolist()]
            dpg.add_bar_series(x_pos, top_n_counts.values.tolist(), weight=0.7, label="Count", parent=f"sva_bar_yaxis_{col_name}")
            # Set X-axis tick labels
            dpg.set_axis_ticks(dpg.last_item(), tuple(zip(bar_labels, x_pos))) # Requires DPG > 0.8.7
            # If older DPG, custom tick labels are harder. Might need to use item alias for axis.


def _run_correlation_analysis(df: pd.DataFrame, util_funcs: dict):
    """숫자형 변수 간 상관관계 분석 및 히트맵 표시."""
    if not dpg.does_item_exist(TAG_MVA_CORR_HEATMAP_PLOT) or df is None: return
    dpg.delete_item(TAG_MVA_CORR_HEATMAP_PLOT, children_only=True) # Clear previous plot
    if dpg.does_item_exist(TAG_MVA_CORR_TABLE):
        dpg.delete_item(TAG_MVA_CORR_TABLE, children_only=True)

    numeric_cols = _get_numeric_cols(df)
    if len(numeric_cols) < 2:
        dpg.add_text("Not enough numeric columns for correlation analysis.", parent=TAG_MVA_CORR_HEATMAP_PLOT)
        return

    corr_matrix = df[numeric_cols].corr(method='pearson') # Or 'spearman'

    # DPG 히트맵
    # 데이터 준비: DPG 히트맵은 1D 배열을 기대 (row-major)
    heatmap_data = corr_matrix.values.flatten().tolist()
    rows, cols = corr_matrix.shape
    col_labels = corr_matrix.columns.tolist()
    row_labels = corr_matrix.index.tolist()

    with dpg.plot(label="Correlation Heatmap", height=450, width=-1, parent=TAG_MVA_CORR_HEATMAP_PLOT, equal_aspects=True):
        dpg.add_plot_legend()
        xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="Variables")
        dpg.set_axis_ticks(xaxis, tuple(zip(col_labels, list(range(cols)))))
        yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Variables")
        dpg.set_axis_ticks(yaxis, tuple(zip(row_labels, list(range(rows)))))
        
        # 히트맵 시리즈 추가
        dpg.add_heat_series(heatmap_data, rows=rows, cols=cols, 
                            scale_min=-1.0, scale_max=1.0, # Correlation ranges from -1 to 1
                            format='%.2f', # Display format for values on heatmap cells
                            parent=yaxis) # 또는 xaxis, DPG 문서 확인

    # 상관관계 높은 쌍 테이블 표시 (예: |corr| > 0.7)
    dpg.add_text("Highly Correlated Pairs (|Correlation| > 0.7):", parent=TAG_MVA_CORR_TAB) # TAB에 직접 추가
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                pair_info = {
                    "Variable 1": corr_matrix.columns[i],
                    "Variable 2": corr_matrix.columns[j],
                    "Correlation": f"{corr_matrix.iloc[i, j]:.3f}"
                }
                high_corr_pairs.append(pair_info)
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        if dpg.does_item_exist(TAG_MVA_CORR_TABLE): # 테이블이 이미 존재하면 사용, 아니면 새로 생성
             util_funcs['create_table_with_data'](TAG_MVA_CORR_TABLE, high_corr_df)
        else: # 테이블 최초 생성
            with dpg.table(header_row=True, tag=TAG_MVA_CORR_TABLE, parent=TAG_MVA_CORR_TAB,
                           resizable=True, policy=dpg.mvTable_SizingFixedFit, scrollY=True, height=150):
                util_funcs['create_table_with_data'](TAG_MVA_CORR_TABLE, high_corr_df)
    else:
        dpg.add_text("No pairs with |correlation| > 0.7 found.", parent=TAG_MVA_CORR_TAB)


def _run_pair_plot_analysis(df: pd.DataFrame, selected_vars: list, hue_var: str, util_funcs: dict):
    """선택된 변수들로 Pair Plot (DPG 네이티브로 유사 구현)"""
    if not dpg.does_item_exist(TAG_MVA_PAIRPLOT_RESULTS_GROUP) or df is None: return
    dpg.delete_item(TAG_MVA_PAIRPLOT_RESULTS_GROUP, children_only=True)

    if not selected_vars or len(selected_vars) < 2:
        dpg.add_text("Please select 2 or more variables for the Pair Plot.", parent=TAG_MVA_PAIRPLOT_RESULTS_GROUP)
        return
    
    # 변수 개수 제한 (예: 최대 5개)
    max_pair_plot_vars = 5 
    if len(selected_vars) > max_pair_plot_vars:
        dpg.add_text(f"Too many variables ({len(selected_vars)}) for Pair Plot. Please select up to {max_pair_plot_vars}.", 
                     parent=TAG_MVA_PAIRPLOT_RESULTS_GROUP, color=(255,100,100))
        return

    # Hue 변수 유효성 검사 및 카테고리 개수 제한
    hue_categories = None
    if hue_var and hue_var in df.columns:
        hue_uniques = df[hue_var].nunique()
        max_hue_categories = 7 # Hue 카테고리 최대 개수
        if hue_uniques > max_hue_categories:
            # 사용자에게 경고 후 진행 여부 확인 (Modal 사용)
            # 여기서는 일단 메시지만 표시하고 진행 중단
            dpg.add_text(f"Hue variable '{hue_var}' has too many categories ({hue_uniques}). Max {max_hue_categories} allowed for clarity.",
                         parent=TAG_MVA_PAIRPLOT_RESULTS_GROUP, color=(255,100,100))
            hue_var = None # Hue 사용 안 함
        else:
            hue_categories = df[hue_var].dropna().unique().tolist()
            # TODO: hue_categories에 대한 색상 맵핑 필요
            print(f"Using hue variable: {hue_var} with categories: {hue_categories}")


    n_vars = len(selected_vars)
    plot_size = 180 # 각 subplot 크기

    # DPG로 Pair Plot 유사 구현 (Grid of Plots)
    # 이 부분은 Matplotlib/Seaborn 이미지를 사용하는 것이 더 간단할 수 있음
    dpg.add_text(f"Pair Plot for: {', '.join(selected_vars)}" + (f" (Hue: {hue_var})" if hue_var else ""), 
                 parent=TAG_MVA_PAIRPLOT_RESULTS_GROUP)

    # 실제 DPG Pair Plot 구현은 복잡하므로 여기서는 메시지만 남김
    # (Matplotlib + dpg.add_image 방식 추천)
    dpg.add_text("Native DPG Pair Plot implementation is complex.", parent=TAG_MVA_PAIRPLOT_RESULTS_GROUP)
    dpg.add_text("Consider using Matplotlib/Seaborn to generate a pair plot image", parent=TAG_MVA_PAIRPLOT_RESULTS_GROUP)
    dpg.add_text("and display it here using dpg.add_image().", parent=TAG_MVA_PAIRPLOT_RESULTS_GROUP)
    dpg.add_text("For now, this is a placeholder.", parent=TAG_MVA_PAIRPLOT_RESULTS_GROUP)
    
    # 만약 DPG 네이티브로 매우 단순하게 구현한다면:
    # for i in range(n_vars):
    #     with dpg.group(horizontal=True, parent=TAG_MVA_PAIRPLOT_RESULTS_GROUP):
    #         for j in range(n_vars):
    #             plot_tag = f"pair_plot_{selected_vars[i]}_{selected_vars[j]}"
    #             with dpg.plot(width=plot_size, height=plot_size, label=plot_tag): # no_title=True
    #                 # dpg.add_plot_axis(dpg.mvXAxis, label=selected_vars[j] if i == n_vars - 1 else "", no_tick_labels=i != n_vars -1)
    #                 # dpg.add_plot_axis(dpg.mvYAxis, label=selected_vars[i] if j == 0 else "", no_tick_labels=j != 0, tag=f"pair_yaxis_{i}_{j}")
    #                 dpg.add_plot_axis(dpg.mvXAxis, label=selected_vars[j])
    #                 dpg.add_plot_axis(dpg.mvYAxis, label=selected_vars[i], tag=f"pair_yaxis_{i}_{j}")

    #                 if i == j: # Diagonal: Histogram or KDE
    #                     series_diag = df[selected_vars[i]].dropna()
    #                     if not series_diag.empty:
    #                         hist_data = np.histogram(series_diag, bins='auto')
    #                         dpg.add_histogram_series(hist_data[1][:-1], hist_data[0], label="Hist", parent=f"pair_yaxis_{i}_{j}")
    #                 else: # Off-diagonal: Scatter plot
    #                     series_x = df[selected_vars[j]].dropna()
    #                     series_y = df[selected_vars[i]].dropna()
    #                     # Align series by index for scatter plot
    #                     aligned_df = pd.concat([series_x, series_y], axis=1).dropna()
    #                     if not aligned_df.empty:
    #                          dpg.add_scatter_series(aligned_df.iloc[:,0].tolist(), aligned_df.iloc[:,1].tolist(), parent=f"pair_yaxis_{i}_{j}")
    #                             # TODO: Hue 적용 로직 추가 (그룹별로 add_scatter_series 호출)

def _run_target_variable_analysis(df: pd.DataFrame, target_var: str, feature_var: str, util_funcs: dict):
    """타겟 변수와 선택된 피처 간의 관계 분석"""
    if not dpg.does_item_exist(TAG_MVA_TARGET_RESULTS_GROUP) or df is None or \
       target_var not in df.columns or feature_var not in df.columns or target_var == feature_var:
        if dpg.does_item_exist(TAG_MVA_TARGET_RESULTS_GROUP):
            dpg.delete_item(TAG_MVA_TARGET_RESULTS_GROUP, children_only=True)
            dpg.add_text("Select valid target and feature variables.", parent=TAG_MVA_TARGET_RESULTS_GROUP)
        return

    dpg.delete_item(TAG_MVA_TARGET_RESULTS_GROUP, children_only=True)
    dpg.add_text(f"Analysis: '{feature_var}' vs Target '{target_var}'", parent=TAG_MVA_TARGET_RESULTS_GROUP)

    target_series = df[target_var]
    feature_series = df[feature_var]

    # 1. 타겟 변수가 숫자형일 경우
    if pd.api.types.is_numeric_dtype(target_series):
        if pd.api.types.is_numeric_dtype(feature_series): # 숫자 vs 숫자
            dpg.add_text("Numeric Feature vs Numeric Target: Scatter Plot", parent=TAG_MVA_TARGET_RESULTS_GROUP)
            # 상관계수
            corr_val = feature_series.corr(target_series)
            dpg.add_text(f"Correlation ({feature_var} vs {target_var}): {corr_val:.3f}", parent=TAG_MVA_TARGET_RESULTS_GROUP)
            
            with dpg.plot(label=f"{feature_var} vs {target_var}", height=300, width=-1, parent=TAG_MVA_TARGET_RESULTS_GROUP):
                dpg.add_plot_axis(dpg.mvXAxis, label=feature_var)
                dpg.add_plot_axis(dpg.mvYAxis, label=target_var, tag="target_vs_num_yaxis")
                aligned_df = pd.concat([feature_series, target_series], axis=1).dropna()
                if not aligned_df.empty:
                    dpg.add_scatter_series(aligned_df.iloc[:,0].tolist(), aligned_df.iloc[:,1].tolist(), parent="target_vs_num_yaxis")

        elif feature_series.nunique() < 20 : # 범주형(으로 간주 가능한) 피처 vs 숫자 타겟
            dpg.add_text("Categorical-like Feature vs Numeric Target: Box Plot (Placeholder)", parent=TAG_MVA_TARGET_RESULTS_GROUP)
            # 그룹별 기술 통계
            grouped_stats = df.groupby(feature_var)[target_var].describe()
            dpg.add_text("Grouped Descriptive Stats:", parent=TAG_MVA_TARGET_RESULTS_GROUP)
            temp_table_tag = f"target_grouped_stats_table"
            with dpg.child_window(height=200, parent=TAG_MVA_TARGET_RESULTS_GROUP, tag=temp_table_tag+"_child"):
                with dpg.table(header_row=True, tag=temp_table_tag, parent=temp_table_tag+"_child"):
                    util_funcs['create_table_with_data'](temp_table_tag, grouped_stats.reset_index())
            # DPG 네이티브 Box Plot은 직접 구현 필요. 여기서는 Matplotlib/Seaborn 이미지 권장.
            dpg.add_text("Consider Matplotlib/Seaborn for Box/Violin plots.", parent=TAG_MVA_TARGET_RESULTS_GROUP)


    # 2. 타겟 변수가 범주형일 경우 (또는 이진 숫자형)
    elif target_series.nunique() < 20: # 범주형 타겟으로 간주
        if pd.api.types.is_numeric_dtype(feature_series): # 숫자 피처 vs 범주형 타겟
            dpg.add_text("Numeric Feature vs Categorical Target: Grouped Distribution (Placeholder)", parent=TAG_MVA_TARGET_RESULTS_GROUP)
            # 그룹별 기술 통계
            grouped_stats = df.groupby(target_var)[feature_var].describe() # 타겟으로 그룹핑
            dpg.add_text("Grouped Descriptive Stats (by Target):", parent=TAG_MVA_TARGET_RESULTS_GROUP)
            temp_table_tag = f"target_grouped_stats_table_cat"
            with dpg.child_window(height=200, parent=TAG_MVA_TARGET_RESULTS_GROUP, tag=temp_table_tag+"_child"):
                with dpg.table(header_row=True, tag=temp_table_tag, parent=temp_table_tag+"_child"):
                    util_funcs['create_table_with_data'](temp_table_tag, grouped_stats.reset_index())
            dpg.add_text("Consider Matplotlib/Seaborn for overlaid Histograms/Density plots by target category.", parent=TAG_MVA_TARGET_RESULTS_GROUP)

        elif feature_series.nunique() < 20 : # 범주형 피처 vs 범주형 타겟
            dpg.add_text("Categorical Feature vs Categorical Target: Crosstab / Stacked Bar (Placeholder)", parent=TAG_MVA_TARGET_RESULTS_GROUP)
            crosstab_df = pd.crosstab(df[feature_var], df[target_var], normalize='index') # 행 기준 비율
            dpg.add_text("Crosstab (Normalized by Feature):", parent=TAG_MVA_TARGET_RESULTS_GROUP)
            temp_table_tag = f"target_crosstab_table"
            with dpg.child_window(height=200, parent=TAG_MVA_TARGET_RESULTS_GROUP, tag=temp_table_tag+"_child"):
                with dpg.table(header_row=True, tag=temp_table_tag, parent=temp_table_tag+"_child"):
                    util_funcs['create_table_with_data'](temp_table_tag, crosstab_df.reset_index())
            dpg.add_text("Consider Matplotlib/Seaborn for Stacked/Grouped Bar charts.", parent=TAG_MVA_TARGET_RESULTS_GROUP)
    else:
        dpg.add_text("Target variable type not suitable for this simplified analysis (e.g. high cardinality text).", parent=TAG_MVA_TARGET_RESULTS_GROUP)


# --- Main UI Creation and Update Functions ---
def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """EDA 스텝 UI 생성: 단일 변수 분석, 다변량 분석 탭 구성"""
    main_callbacks['register_step_group_tag'](step_name, TAG_EDA_GROUP)

    with dpg.group(tag=TAG_EDA_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()

        with dpg.tab_bar(tag=TAG_EDA_MAIN_TAB_BAR):
            # --- 1. 단일 변수 분석 탭 ---
            with dpg.tab(label="Single Variable Analysis", tag=TAG_SVA_TAB):
                dpg.add_text("Select a variable to see its detailed analysis.")
                with dpg.group(horizontal=True):
                    dpg.add_combo(label="Variable", tag=TAG_SVA_VARIABLE_COMBO, width=300) # 아이템은 update_ui에서 채움
                    dpg.add_button(label="Run Analysis", tag=TAG_SVA_RUN_BUTTON,
                                   callback=lambda: _run_single_variable_analysis(
                                       dpg.get_value(TAG_SVA_VARIABLE_COMBO),
                                       main_callbacks['get_current_df'](),
                                       main_callbacks['get_util_funcs']()
                                   ))
                dpg.add_separator()
                dpg.add_child_window(tag=TAG_SVA_RESULTS_GROUP, border=True) # 결과 표시 영역

            # --- 2. 다변량 분석 탭 ---
            with dpg.tab(label="Multivariate Analysis", tag=TAG_MVA_TAB):
                with dpg.tab_bar(tag=TAG_MVA_SUB_TAB_BAR):
                    # 2-1. 상관관계 분석 탭
                    with dpg.tab(label="Correlation Analysis", tag=TAG_MVA_CORR_TAB):
                        dpg.add_button(label="Run Correlation Analysis (Numeric Vars)", tag=TAG_MVA_CORR_RUN_BUTTON,
                                       callback=lambda: _run_correlation_analysis(
                                           main_callbacks['get_current_df'](),
                                           main_callbacks['get_util_funcs']()
                                       ))
                        dpg.add_child_window(tag=TAG_MVA_CORR_HEATMAP_PLOT, border=True, height=480) # 히트맵용
                        # 상관관계 높은 쌍 테이블은 버튼 아래, 히트맵 위에 위치시키거나 별도 child_window 구성 가능
                        # 여기서는 버튼 아래에 바로 테이블 생성하도록 함 (위 _run_correlation_analysis 참조)

                    # 2-2. Pair Plot 탭
                    with dpg.tab(label="Pair Plot", tag=TAG_MVA_PAIRPLOT_TAB):
                        dpg.add_text("Select multiple numeric variables for Pair Plot.")
                        # 변수 다중 선택 (Listbox 사용 예시)
                        dpg.add_listbox(tag=TAG_MVA_PAIRPLOT_VAR_SELECTOR, width=-1, num_items=5) # 아이템은 update_ui에서
                        dpg.add_combo(label="Hue (Optional Categorical Var)", tag=TAG_MVA_PAIRPLOT_HUE_COMBO, width=300) # 아이템은 update_ui
                        dpg.add_button(label="Generate Pair Plot", tag=TAG_MVA_PAIRPLOT_RUN_BUTTON,
                                       callback=lambda: _run_pair_plot_analysis(
                                           main_callbacks['get_current_df'](),
                                           dpg.get_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR), # listbox는 여러개 선택된 값을 어떻게 가져오는지 확인 필요
                                           dpg.get_value(TAG_MVA_PAIRPLOT_HUE_COMBO),
                                           main_callbacks['get_util_funcs']()
                                       ))
                        dpg.add_child_window(tag=TAG_MVA_PAIRPLOT_RESULTS_GROUP, border=True, height=500)
                        # Pair Plot은 복잡해서 여기에 직접 그리기보다 이미지로 표시하는 것 고려

                    # 2-3. 타겟 변수 분석 탭
                    with dpg.tab(label="Target Variable Analysis", tag=TAG_MVA_TARGET_TAB):
                        dpg.add_text("Analyze relationship between features and the selected target variable.", tag=TAG_MVA_TARGET_INFO_TEXT)
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
                        dpg.add_child_window(tag=TAG_MVA_TARGET_RESULTS_GROUP, border=True)


    main_callbacks['register_module_updater'](step_name, lambda df, orig_df, utils, fp: update_ui(df, main_callbacks))
    update_ui(main_callbacks['get_current_df'](), main_callbacks) # 초기 UI 채우기


def update_ui(current_df: pd.DataFrame, main_callbacks: dict): # 파라미터 변경: main_callbacks 전체를 받도록
    """EDA 모듈의 UI 요소들(콤보박스 아이템 등)을 현재 데이터프레임 기준으로 업데이트합니다."""
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_EDA_GROUP):
        return

    # --- 단일 변수 분석 탭 업데이트 ---
    if dpg.does_item_exist(TAG_SVA_VARIABLE_COMBO):
        if current_df is not None and not current_df.empty:
            all_cols = current_df.columns.tolist()
            dpg.configure_item(TAG_SVA_VARIABLE_COMBO, items=all_cols)
            if all_cols: # 첫번째 컬럼을 기본값으로 설정 (선택사항)
                 # dpg.set_value(TAG_SVA_VARIABLE_COMBO, all_cols[0])
                 pass 
        else:
            dpg.configure_item(TAG_SVA_VARIABLE_COMBO, items=[])

    # --- 다변량 분석 탭 업데이트 ---
    # Pair Plot 변수 선택기 (Listbox) 업데이트
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_VAR_SELECTOR):
        if current_df is not None:
            numeric_cols = _get_numeric_cols(current_df)
            dpg.configure_item(TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=numeric_cols)
        else:
            dpg.configure_item(TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=[])
            
    # Pair Plot Hue 변수 콤보 업데이트
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_HUE_COMBO):
        if current_df is not None:
            cat_cols = [""] + _get_categorical_cols(current_df) # 첫번째는 "선택 안함"
            dpg.configure_item(TAG_MVA_PAIRPLOT_HUE_COMBO, items=cat_cols)
        else:
            dpg.configure_item(TAG_MVA_PAIRPLOT_HUE_COMBO, items=[""])

    # 타겟 변수 분석 탭 업데이트
    selected_target_var = main_callbacks['get_selected_target_variable']()
    if dpg.does_item_exist(TAG_MVA_TARGET_INFO_TEXT):
        if selected_target_var and current_df is not None and selected_target_var in current_df.columns:
            dpg.set_value(TAG_MVA_TARGET_INFO_TEXT, f"Analyzing features against Target: '{selected_target_var}'")
            if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO):
                # 타겟 자신을 제외한 모든 컬럼을 피처 후보로
                feature_candidates = [col for col in current_df.columns if col != selected_target_var]
                dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=feature_candidates)
        else:
            dpg.set_value(TAG_MVA_TARGET_INFO_TEXT, "Select a global target variable from the top panel to enable this analysis.")
            if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO):
                dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=[])


    # 각 분석 결과 영역 초기화 (데이터가 없거나 변경 시)
    # (선택사항: 사용자가 "Run" 버튼을 누를 때만 결과가 표시되므로, 여기서는 주로 콤보박스만 업데이트)
    # if current_df is None:
    #     if dpg.does_item_exist(TAG_SVA_RESULTS_GROUP): dpg.delete_item(TAG_SVA_RESULTS_GROUP, children_only=True)
    #     # ... 다른 결과 영역도 초기화 ...