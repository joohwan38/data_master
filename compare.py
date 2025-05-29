# step_02_exploratory_data_analysis.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from scipy import stats # For skewness, kurtosis, normality tests, chi2_contingency
import traceback # For detailed error logging
from sklearn.ensemble import IsolationForest # For Outlier Detection

# --- UI Element Tags (Existing + New) ---
TAG_EDA_GROUP = "step2_eda_group"
TAG_EDA_MAIN_TAB_BAR = "step2_eda_main_tab_bar"

# Single Variable Analysis (SVA) Tab (Tags assumed to be defined as in original)
TAG_SVA_TAB = "step2_sva_tab"
TAG_SVA_FILTER_STRENGTH_RADIO = "step2_sva_filter_strength_radio"
TAG_SVA_GROUP_BY_TARGET_CHECKBOX = "step2_sva_group_by_target_checkbox"
TAG_SVA_RUN_BUTTON = "step2_sva_run_button"
TAG_SVA_RESULTS_CHILD_WINDOW = "step2_sva_results_child_window"
TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX = "sva_var_section_"
TAG_SVA_PROGRESS_MODAL = "sva_progress_modal"
TAG_SVA_PROGRESS_TEXT = "sva_progress_text"
TAG_SVA_ALERT_MODAL_PREFIX = "sva_alert_modal_"
TAG_SVA_GROUPED_PLOT_TYPE_RADIO = "step2_sva_grouped_plot_type_radio"


REUSABLE_SVA_ALERT_MODAL_TAG = "reusable_sva_alert_modal_unique_tag" # Assuming this exists
REUSABLE_SVA_ALERT_TEXT_TAG = "reusable_sva_alert_text_unique_tag" # Assuming this exists


# Multivariate Analysis (MVA) Tab
TAG_MVA_TAB = "step2_mva_tab"
TAG_MVA_SUB_TAB_BAR = "step2_mva_sub_tab_bar" # This might be removed if new structure is flatter
TAG_MVA_CORR_TAB = "step2_mva_corr_tab"
TAG_MVA_CORR_RUN_BUTTON = "step2_mva_corr_run_button"
TAG_MVA_CORR_RESULTS_GROUP = "step2_mva_corr_results_group" # New group to hold all corr results
TAG_MVA_CORR_HEATMAP_PLOT = "step2_mva_corr_heatmap_plot" # Retained for specific heatmaps
TAG_MVA_CORR_TABLE = "step2_mva_corr_table" # Retained for specific tables

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
TAG_MVA_TARGET_PLOT_AREA_PREFIX = "mva_target_plot_area_" # Assuming this exists

# New MVA Tab for Categorical EDA
TAG_MVA_CAT_EDA_TAB = "step2_mva_cat_eda_tab"
TAG_MVA_CAT_EDA_VAR_SELECTOR = "step2_mva_cat_eda_var_selector"
TAG_MVA_CAT_EDA_RUN_BUTTON = "step2_mva_cat_eda_run_button"
TAG_MVA_CAT_EDA_RESULTS_GROUP = "step2_mva_cat_eda_results_group"

# New Main Tab for Outlier Treatment
TAG_OUTLIER_TAB = "step2_outlier_tab"
TAG_OUTLIER_METHOD_RADIO = "step2_outlier_method_radio"
TAG_OUTLIER_CAPPING_CONTROLS_GROUP = "step2_outlier_capping_controls_group"
TAG_OUTLIER_CAPPING_LOWER_PERCENTILE = "step2_outlier_capping_lower_percentile"
TAG_OUTLIER_CAPPING_UPPER_PERCENTILE = "step2_outlier_capping_upper_percentile"
TAG_OUTLIER_CAPPING_VAR_SELECTOR = "step2_outlier_capping_var_selector"
TAG_OUTLIER_IF_CONTROLS_GROUP = "step2_outlier_if_controls_group"
TAG_OUTLIER_IF_CONTAMINATION_SLIDER = "step2_outlier_if_contamination_slider" # Example, could be 'auto'
TAG_OUTLIER_IF_VAR_SELECTOR = "step2_outlier_if_var_selector"
TAG_OUTLIER_APPLY_BUTTON = "step2_outlier_apply_button"
TAG_OUTLIER_RESET_TO_AFTER_STEP1_BUTTON = "step2_outlier_reset_to_after_step1_button"
TAG_OUTLIER_RESULTS_TEXT = "step2_outlier_results_text"
TAG_OUTLIER_STATUS_TEXT = "step2_outlier_status_text"


# Store callbacks from main_app
_main_app_callbacks_eda = {}
_util_funcs_eda = {}

# --- Helper Functions (Existing or to be adjusted if needed) ---
# _show_alert_modal, _calculate_cramers_v, _get_numeric_cols, _get_categorical_cols
# _get_filtered_variables, _create_sva_basic_stats_table, _create_sva_advanced_relations_table,
# _create_single_var_plot, _apply_sva_filters_and_run, _sva_group_by_target_callback
# These are assumed to be present and largely unchanged unless specified.
# For brevity, only new helpers or significantly modified ones are shown below.

def _show_alert_modal(title: str, message: str):
    if not dpg.is_dearpygui_running():
        print(f"DPG not running. Modal '{title}': {message}")
        return

    viewport_width = dpg.get_viewport_width() if dpg.is_dearpygui_running() else 1000
    viewport_height = dpg.get_viewport_height() if dpg.is_dearpygui_running() else 700
    modal_width = 450 # Increased width for better message display
    
    modal_pos_x = (viewport_width - modal_width) // 2
    modal_pos_y = viewport_height // 3 

    if not dpg.does_item_exist(REUSABLE_SVA_ALERT_MODAL_TAG): # Ensure this generic modal exists
        with dpg.window(label="Alert", modal=True, show=False, tag=REUSABLE_SVA_ALERT_MODAL_TAG,
                        no_close=True, pos=[modal_pos_x, modal_pos_y], width=modal_width, autosize=True, 
                        no_saved_settings=True):
            dpg.add_text("", tag=REUSABLE_SVA_ALERT_TEXT_TAG, wrap=modal_width - 30) # Wrap text
            dpg.add_spacer(height=15)
            with dpg.group(horizontal=True):
                # Center the button
                button_width = 100
                spacer_width = (modal_width - button_width - dpg.get_style_item_spacing()[0] * 2) / 2
                if spacer_width < 0: spacer_width = 0
                dpg.add_spacer(width=spacer_width)
                dpg.add_button(label="OK", width=button_width, user_data=REUSABLE_SVA_ALERT_MODAL_TAG,
                               callback=lambda s, a, u: dpg.configure_item(u, show=False))
    
    dpg.configure_item(REUSABLE_SVA_ALERT_MODAL_TAG, label=title, show=True, pos=[modal_pos_x, modal_pos_y])
    dpg.set_value(REUSABLE_SVA_ALERT_TEXT_TAG, message)

# (Re-paste _get_numeric_cols and _get_categorical_cols for completeness if they are used by new functions)
def _get_numeric_cols(df: pd.DataFrame) -> list:
    if df is None: return []
    return df.select_dtypes(include=np.number).columns.tolist()

def _get_categorical_cols(df: pd.DataFrame, max_unique_for_cat=20, main_callbacks=None) -> list:
    if df is None: return []
    cat_cols = []
    
    # Prioritize Step 1 analysis types if available
    s1_types = {}
    if main_callbacks and 'get_column_analysis_types' in main_callbacks:
        s1_types = main_callbacks['get_column_analysis_types']()
        if not isinstance(s1_types, dict): s1_types = {}

    for col in df.columns:
        s1_type = s1_types.get(col, "")
        is_s1_cat = "Categorical" in s1_type or "Text" in s1_type or "Binary" in s1_type # Consider binary as cat for this purpose
        
        if is_s1_cat:
            if df[col].nunique(dropna=False) <= max_unique_for_cat * 1.5: # Slightly higher threshold for S1 identified cats
                 cat_cols.append(col)
        elif df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col].dtype):
            if df[col].nunique(dropna=False) <= max_unique_for_cat:
                 cat_cols.append(col)
    
    # Numeric columns with few unique values can also be treated as categorical (unless S1 marked them strongly numeric)
    for col in df.select_dtypes(include=np.number).columns:
        s1_type = s1_types.get(col, "")
        is_s1_strong_numeric = "Numeric" in s1_type and "Binary" not in s1_type # e.g. Numeric (Float), Numeric (Integer)
        
        if not is_s1_strong_numeric and df[col].nunique(dropna=False) <= max_unique_for_cat and col not in cat_cols:
            cat_cols.append(col)
            
    return list(set(cat_cols))


def _calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    if x is None or y is None or x.empty or y.empty: return 0.0
    try:
        temp_df = pd.DataFrame({'x': x, 'y': y}).dropna()
        if temp_df.empty or temp_df['x'].nunique() < 1 or temp_df['y'].nunique() < 1:
            return 0.0
        
        # Ensure data is suitable for crosstab (e.g., no all-NaN columns after dropna)
        if temp_df['x'].count() == 0 or temp_df['y'].count() == 0:
             return 0.0

        confusion_matrix = pd.crosstab(temp_df['x'], temp_df['y'])
        if confusion_matrix.empty or confusion_matrix.shape[0] < 1 or confusion_matrix.shape[1] < 1: # Allow 1xN or Nx1 tables
            return 0.0 # Or handle differently if chi2 can work with it. For Cramer's V, usually need >1 for one dim.
        
        # Chi-squared test might not be meaningful for 1xN or Nx1, but Cramer's V formula components still can be calculated.
        # However, typical Cramer's V interpretation relies on a meaningful Chi2. Let's stick to min 2x2 for chi2 part.
        if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
            # Simplified check if one var is constant: V would be 0 or undefined.
             if confusion_matrix.shape[0] == 1 or confusion_matrix.shape[1] == 1: return 0.0

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
    except Exception as e:
        # print(f"Cramer's V calculation error for series {x.name} and {y.name}: {e}")
        # traceback.print_exc()
        return 0.0

# --- New Helper Function ---
def _get_top_n_correlated_with_target(df: pd.DataFrame, target_var: str, numeric_cols: list, top_n: int = 20) -> list:
    if df is None or target_var not in df.columns or not numeric_cols:
        return []
    
    correlations = {}
    target_series = df[target_var]
    
    if not pd.api.types.is_numeric_dtype(target_series.dtype):
        # If target is not numeric, correlation is not directly applicable in this context for ranking numeric features.
        # Depending on desired behavior, could try to find numeric proxies or return empty.
        print(f"Warning: Target variable '{target_var}' is not numeric. Cannot rank numeric features by Pearson correlation.")
        return numeric_cols[:top_n] # Fallback: return first top_n numeric cols

    for col in numeric_cols:
        if col == target_var:
            continue
        try:
            # Align series by index and drop NaNs only for the pair being correlated
            temp_df_corr = df[[target_var, col]].dropna()
            if len(temp_df_corr) < 2: # Need at least 2 common data points
                correlations[col] = 0 
                continue
            corr_val = temp_df_corr[target_var].corr(temp_df_corr[col])
            correlations[col] = abs(corr_val if pd.notna(corr_val) else 0)
        except Exception:
            correlations[col] = 0
            
    sorted_vars = sorted(correlations.keys(), key=lambda k: correlations[k], reverse=True)
    return sorted_vars[:top_n]

# --- Modified MVA Core Analysis Functions ---

def _run_correlation_analysis(df: pd.DataFrame, util_funcs: dict, main_callbacks: dict):
    """
    MODIFIED: Implements new logic for correlation analysis based on variable count.
    """
    if not dpg.is_dearpygui_running(): return
    
    results_group_tag = TAG_MVA_CORR_RESULTS_GROUP # Use a general results group for this tab
    if dpg.does_item_exist(results_group_tag):
        dpg.delete_item(results_group_tag, children_only=True)
    else:
        print(f"Error: MVA Correlation results group {results_group_tag} not found.")
        _show_alert_modal("UI Error", f"MVA Correlation results group {results_group_tag} is missing.")
        return

    if df is None:
        dpg.add_text("Load data first.", parent=results_group_tag)
        return

    numeric_cols = _get_numeric_cols(df)
    if len(numeric_cols) < 2:
        dpg.add_text("Not enough numeric columns for correlation analysis (need at least 2).", parent=results_group_tag)
        return

    target_var = main_callbacks['get_selected_target_variable']()
    max_vars_direct_heatmap = 20

    def create_heatmap_in_dpg(data_matrix: pd.DataFrame, title: str, parent_tag: str, height: int = 450):
        if data_matrix.empty:
            dpg.add_text(f"{title}: No data to display or matrix is empty.", parent=parent_tag)
            return

        heatmap_data_flat = data_matrix.values.flatten().tolist()
        rows, cols = data_matrix.shape
        col_labels = data_matrix.columns.tolist()
        row_labels = data_matrix.index.tolist()

        plot_uuid = dpg.generate_uuid()
        with dpg.plot(label=title, height=height, width=-1, parent=parent_tag, tag=plot_uuid, equal_aspects=True if rows==cols else False):
            xaxis = dpg.add_plot_axis(dpg.mvXAxis, label="")
            yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="")
            if col_labels:
                dpg.set_axis_ticks(xaxis, tuple(zip(col_labels, list(range(cols)))))
            if row_labels: # Y-axis ticks are typically reversed for heatmaps if (0,0) is top-left
                dpg.set_axis_ticks(yaxis, tuple(zip(row_labels, list(range(rows)))))
            
            dpg.add_heat_series(heatmap_data_flat, rows=rows, cols=cols, scale_min=-1.0, scale_max=1.0,
                                format='%.2f', parent=yaxis, show_tooltips=True,
                                bounds_min=(0, 0), bounds_max=(cols, rows))
        dpg.add_spacer(height=10, parent=parent_tag)


    if len(numeric_cols) <= max_vars_direct_heatmap:
        dpg.add_text(f"Correlation Matrix for {len(numeric_cols)} Numeric Variables:", parent=results_group_tag, color=(255,255,0))
        corr_matrix_full = df[numeric_cols].corr(method='pearson')
        create_heatmap_in_dpg(corr_matrix_full, "Overall Correlation Heatmap (Pearson)", results_group_tag)
    else:
        dpg.add_text(f"Number of numeric variables ({len(numeric_cols)}) > {max_vars_direct_heatmap}. Showing targeted analyses:", parent=results_group_tag, color=(255,255,0))
        
        # 1-1: Heatmap of pairs with |correlation| >= 0.6
        dpg.add_text("Analysis 1: Variables with Pairwise |Correlation| >= 0.6", parent=results_group_tag, color=(200,200,0))
        corr_matrix_all_pairs = df[numeric_cols].corr(method='pearson')
        highly_correlated_vars = set()
        for i in range(len(corr_matrix_all_pairs.columns)):
            for j in range(i + 1, len(corr_matrix_all_pairs.columns)):
                if abs(corr_matrix_all_pairs.iloc[i, j]) >= 0.6:
                    highly_correlated_vars.add(corr_matrix_all_pairs.columns[i])
                    highly_correlated_vars.add(corr_matrix_all_pairs.columns[j])
        
        if len(highly_correlated_vars) >= 2:
            sorted_highly_corr_vars = sorted(list(highly_correlated_vars))
            corr_matrix_high_pairs = df[sorted_highly_corr_vars].corr(method='pearson')
            create_heatmap_in_dpg(corr_matrix_high_pairs, "Heatmap of Highly Correlated Variables (>=0.6)", results_group_tag, height=350)
        elif highly_correlated_vars: # Only one variable met the criteria with itself effectively, which is not a pair.
            dpg.add_text(f"Only one variable ('{list(highly_correlated_vars)[0]}') was part of a highly correlated pair, or not enough distinct variables to form a heatmap.", parent=results_group_tag)
        else:
            dpg.add_text("No variable pairs found with |correlation| >= 0.6.", parent=results_group_tag)
        
        dpg.add_separator(parent=results_group_tag)

        # 1-2: Heatmap of Top 20 variables correlated with the target
        dpg.add_text(f"Analysis 2: Top {max_vars_direct_heatmap} Variables Correlated with Target", parent=results_group_tag, color=(200,200,0))
        if target_var and target_var in df.columns:
            if not pd.api.types.is_numeric_dtype(df[target_var].dtype):
                dpg.add_text(f"Target variable '{target_var}' is not numeric. This analysis requires a numeric target.", parent=results_group_tag, color=(255,100,100))
            else:
                top_n_for_target_corr = _get_top_n_correlated_with_target(df, target_var, numeric_cols, top_n=max_vars_direct_heatmap)
                if len(top_n_for_target_corr) >= 1: # Need at least one other var to correlate with target
                    vars_for_target_heatmap = [target_var] + [v for v in top_n_for_target_corr if v != target_var]
                    vars_for_target_heatmap = list(dict.fromkeys(vars_for_target_heatmap)) # Keep order, remove duplicates

                    if len(vars_for_target_heatmap) >=2: # Need target + at least one other
                        corr_matrix_target_focused = df[vars_for_target_heatmap].corr(method='pearson')
                        create_heatmap_in_dpg(corr_matrix_target_focused, f"Heatmap: Target '{target_var}' & Top Correlated Vars", results_group_tag, height=350)
                    else:
                        dpg.add_text(f"Not enough other numeric variables found with strong correlation to target '{target_var}' to form a heatmap.", parent=results_group_tag)

                else:
                    dpg.add_text(f"No numeric variables found to correlate with target '{target_var}'.", parent=results_group_tag)
        else:
            dpg.add_text("No target variable selected or target is not numeric. This analysis cannot be performed.", parent=results_group_tag)

    # Highly Correlated Pairs Table (General, from full matrix if small, or from combined if large)
    # This can be duplicative if specific heatmaps already show this. For now, let's show general high correlations.
    # This part can be refined based on exact needs.
    corr_matrix_for_table = df[numeric_cols].corr(method='pearson') # Use full matrix for table source
    dpg.add_text("Highly Correlated Numeric Pairs (|Correlation| > 0.7, from all variables):", parent=results_group_tag)
    high_corr_pairs_list = []
    for i in range(len(corr_matrix_for_table.columns)):
        for j in range(i + 1, len(corr_matrix_for_table.columns)):
            if abs(corr_matrix_for_table.iloc[i, j]) > 0.7:
                high_corr_pairs_list.append({
                    "Variable 1": corr_matrix_for_table.columns[i],
                    "Variable 2": corr_matrix_for_table.columns[j],
                    "Correlation": f"{corr_matrix_for_table.iloc[i, j]:.3f}"
                })
    if high_corr_pairs_list:
        high_corr_df = pd.DataFrame(high_corr_pairs_list)
        table_tag_high_corr = dpg.generate_uuid()
        with dpg.table(header_row=True, tag=table_tag_high_corr, parent=results_group_tag,
                       resizable=True, policy=dpg.mvTable_SizingFixedFit,
                       scrollY=True, height=200, 
                       borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
            util_funcs['create_table_with_data'](table_tag_high_corr, high_corr_df, parent_df_for_widths=high_corr_df)
    else:
        dpg.add_text("No pairs with |correlation| > 0.7 found among all numeric variables.", parent=results_group_tag)


def _run_pair_plot_analysis(df: pd.DataFrame, selected_vars_from_ui: list, hue_var: str, util_funcs: dict, main_callbacks: dict):
    """
    MODIFIED: Implements new logic for Pair Plot based on variable count.
    """
    results_group = TAG_MVA_PAIRPLOT_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group): return

    dpg.delete_item(results_group, children_only=True)

    if df is None:
        dpg.add_text("Load data first.", parent=results_group)
        return

    all_numeric_cols = _get_numeric_cols(df)
    if not all_numeric_cols:
        dpg.add_text("No numeric variables available for Pair Plot.", parent=results_group)
        return

    target_var = main_callbacks['get_selected_target_variable']()
    max_vars_pairplot = 7 # Max vars for pairplot to keep it manageable, DPG native can be slow with many. User asked for Top20 if >20
    
    vars_for_plot = []
    info_message = ""

    if selected_vars_from_ui and len(selected_vars_from_ui) > 0: # User made a specific selection
        vars_for_plot = [var for var in selected_vars_from_ui if var in all_numeric_cols]
        if not vars_for_plot:
            info_message = "None of the selected variables are valid numeric columns. Please reselect."
        elif len(vars_for_plot) < 2:
            info_message = "Please select at least two valid numeric variables for the Pair Plot."
    else: # No specific selection by user, or selection was cleared - use logic based on count
        if len(all_numeric_cols) <= max_vars_pairplot : # If total numeric vars are few, use all
            vars_for_plot = all_numeric_cols
            info_message = f"Using all {len(vars_for_plot)} available numeric variables for Pair Plot."
        else: # More than max_vars_pairplot numeric variables, use top correlated with target
            if target_var and target_var in df.columns and pd.api.types.is_numeric_dtype(df[target_var].dtype):
                info_message = f"Number of numeric variables ({len(all_numeric_cols)}) > {max_vars_pairplot}. Using top {max_vars_pairplot} correlated with target '{target_var}'."
                vars_for_plot = _get_top_n_correlated_with_target(df, target_var, all_numeric_cols, top_n=max_vars_pairplot)
                # Ensure target is included if it's numeric and not already there by correlation rank
                if target_var in all_numeric_cols and target_var not in vars_for_plot:
                     if len(vars_for_plot) < max_vars_pairplot: vars_for_plot.append(target_var)
                     else: vars_for_plot[-1] = target_var # Replace last one
                vars_for_plot = list(dict.fromkeys(vars_for_plot)) # Remove duplicates if any, keep order
            else:
                info_message = f"Number of numeric variables ({len(all_numeric_cols)}) > {max_vars_pairplot}. No numeric target selected. Using first {max_vars_pairplot} numeric variables."
                vars_for_plot = all_numeric_cols[:max_vars_pairplot]


    if info_message:
        dpg.add_text(info_message, parent=results_group, wrap=dpg.get_item_width(results_group) or 600)

    if not vars_for_plot or len(vars_for_plot) < 2:
        dpg.add_text("Not enough valid numeric variables to generate Pair Plot (need at least 2).", parent=results_group)
        if selected_vars_from_ui and len(selected_vars_from_ui)>0 and (not vars_for_plot or len(vars_for_plot) < 2):
             dpg.add_text("Tip: Check your selections in the listbox.", parent=results_group, color=(200,200,0))
        return
    
    if len(vars_for_plot) > 7: # Hard cap for DPG performance if still too many
        _show_alert_modal("Pair Plot Limit", f"Plotting first 7 variables out of {len(vars_for_plot)} selected for performance reasons in DPG.")
        vars_for_plot = vars_for_plot[:7]


    hue_series = None
    hue_categories = None
    actual_hue_var_name = None

    if hue_var and hue_var in df.columns:
        temp_hue_series = df[hue_var]
        s1_types_hue = main_callbacks.get('get_column_analysis_types', lambda: {})()
        is_hue_cat_like = _get_categorical_cols(df[[hue_var]], max_unique_for_cat=10, main_callbacks=main_callbacks)
        
        if hue_var in is_hue_cat_like: # Check if it's considered categorical with <= 10 uniques
            hue_series = temp_hue_series
            # Convert to string for consistent category handling, esp. if it's numeric-categorical
            hue_categories = sorted(hue_series.astype(str).dropna().unique()) 
            actual_hue_var_name = hue_var
        else:
            _show_alert_modal("Hue Variable Warning", f"Hue variable '{hue_var}' has too many unique values (>10) or is not suitable. Hue disabled.")
            hue_var = None
    # ... (rest of the pair plot generation logic, similar to the original _run_pair_plot_analysis)
    # This part is complex to reproduce fully without the original context of how DPG plots are themed/colored for hue.
    # Assuming the original logic for drawing the grid of plots:

    n_vars = len(vars_for_plot)
    plot_cell_width = max(180, int((dpg.get_item_width(results_group) or 800) / n_vars) - 20) if n_vars > 0 else 200
    plot_cell_height = plot_cell_width 

    plot_title_text = f"Pair Plot for: {', '.join(vars_for_plot)}"
    if actual_hue_var_name:
        plot_title_text += f" (Hue: {actual_hue_var_name})"
    dpg.add_text(plot_title_text, parent=results_group)

    # Colors for hue (example, can be expanded)
    # DPG default coloring might be sufficient if not too many hue categories.
    # hue_colors = [(0,114,178,200), (230,159,0,200), (0,158,115,200), (240,228,66,200), 
    #               (213,94,0,200), (86,180,233,200), (204,121,167,200)] 

    with dpg.child_window(parent=results_group, border=False, autosize_x=True, autosize_y=True):
        for i in range(n_vars):  # Row variable
            with dpg.group(horizontal=True):  # Each row of plots
                for j in range(n_vars):  # Column variable
                    var_y_name = vars_for_plot[i]
                    var_x_name = vars_for_plot[j]

                    cell_plot_label = f"{var_y_name} vs {var_x_name}" if i != j else f"Dist: {var_x_name}"
                    
                    # Unique tag for each plot to manage legends if needed
                    cell_plot_tag = dpg.generate_uuid()

                    with dpg.plot(width=plot_cell_width, height=plot_cell_height, label=cell_plot_label, tag=cell_plot_tag):
                        show_x_label = (i == n_vars - 1)
                        show_y_label = (j == 0)

                        px_axis = dpg.add_plot_axis(dpg.mvXAxis, label=var_x_name if show_x_label else "", no_tick_labels=not show_x_label)
                        py_axis = dpg.add_plot_axis(dpg.mvYAxis, label=var_y_name if show_y_label else "", no_tick_labels=not show_y_label)
                        
                        # Add legend to each plot cell if hue is active, control its visibility
                        cell_legend_tag = None
                        if actual_hue_var_name and i !=j : # Only for scatter plots with hue
                             cell_legend_tag = dpg.add_plot_legend(parent=cell_plot_tag, horizontal=True, location=dpg.mvPlot_Location_NorthEast, outside=False)
                             # dpg.configure_item(cell_legend_tag, show= (i==0 and j==n_vars-1) ) # Example: Show legend only on one plot. Or always show.


                        if i == j: # Diagonal: Histogram or KDE
                            series_diag = df[var_x_name].dropna()
                            if not series_diag.empty and series_diag.nunique() >= 1:
                                if actual_hue_var_name and hue_series is not None and hue_categories is not None:
                                    # Grouped KDE/Hist on diagonal
                                    # This requires more complex logic to draw multiple distributions if DPG supports it easily.
                                    # For simplicity, let's do a single distribution on diagonal for now, or show text.
                                    # dpg.add_text("Grouped diag not implemented", parent=py_axis)
                                    # Fallback to simple histogram if hue is active on diagonal
                                     dpg.add_histogram_series(series_diag.tolist(), bins=-1, density=True, label="Hist", parent=py_axis, weight=1.0)

                                elif series_diag.nunique() == 1:
                                    dpg.add_bar_series([0], [len(series_diag)], weight=0.5, label=str(series_diag.iloc[0]), parent=py_axis)
                                    dpg.set_axis_ticks(px_axis, [(str(series_diag.iloc[0]), 0)])
                                else:
                                    dpg.add_histogram_series(series_diag.tolist(), bins=-1, density=True, label="Hist", parent=py_axis, weight=1.0)
                        else: # Off-diagonal: Scatter plot
                            series_x_scatter = df[var_x_name]
                            series_y_scatter = df[var_y_name]

                            if actual_hue_var_name and hue_series is not None and hue_categories is not None:
                                for cat_idx, cat_val in enumerate(hue_categories):
                                    mask = (hue_series.astype(str) == cat_val) # Ensure comparison with string category
                                    
                                    # Align data for this category
                                    temp_df_cat_scatter = pd.concat([series_x_scatter[mask], series_y_scatter[mask]], axis=1).dropna()

                                    if not temp_df_cat_scatter.empty:
                                        # DPG will auto-color different series. Explicit coloring needs themes.
                                        dpg.add_scatter_series(
                                            temp_df_cat_scatter.iloc[:, 0].tolist(),
                                            temp_df_cat_scatter.iloc[:, 1].tolist(),
                                            label=str(cat_val), parent=py_axis
                                        )
                            else: # No hue
                                temp_df_scatter = pd.concat([series_x_scatter, series_y_scatter], axis=1).dropna()
                                if not temp_df_scatter.empty:
                                    dpg.add_scatter_series(temp_df_scatter.iloc[:, 0].tolist(), temp_df_scatter.iloc[:, 1].tolist(), parent=py_axis)
    print("Pair plot generation attempt finished.")


# --- New MVA Function for Categorical EDA ---
def _run_categorical_correlation_analysis(df: pd.DataFrame, util_funcs: dict, main_callbacks: dict):
    results_group = TAG_MVA_CAT_EDA_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group): return

    dpg.delete_item(results_group, children_only=True)

    if df is None:
        dpg.add_text("Load data first.", parent=results_group)
        return

    # Use main_callbacks to pass into _get_categorical_cols
    selected_cat_vars_ui = dpg.get_value(TAG_MVA_CAT_EDA_VAR_SELECTOR) if dpg.does_item_exist(TAG_MVA_CAT_EDA_VAR_SELECTOR) else []
    
    all_potential_cat_cols = _get_categorical_cols(df, max_unique_for_cat=30, main_callbacks=main_callbacks) # Allow more uniques for selection

    if not selected_cat_vars_ui: # If user selected none, use all identified up to a limit
        cat_vars_for_analysis = all_potential_cat_cols
        if len(cat_vars_for_analysis) > 20 :
             dpg.add_text(f"More than 20 categorical variables identified. Using first 20 for Cramer's V heatmap.", parent=results_group, color=(200,200,0))
             cat_vars_for_analysis = cat_vars_for_analysis[:20]
    else:
        cat_vars_for_analysis = [var for var in selected_cat_vars_ui if var in all_potential_cat_cols]
        if len(cat_vars_for_analysis) > 20:
            dpg.add_text(f"More than 20 variables selected. Using first 20 valid categorical variables for Cramer's V heatmap.", parent=results_group, color=(200,200,0))
            cat_vars_for_analysis = cat_vars_for_analysis[:20]


    if len(cat_vars_for_analysis) < 2:
        dpg.add_text("Not enough categorical variables selected or available for Cramer's V analysis (need at least 2).", parent=results_group)
        return

    dpg.add_text(f"Cramer's V Matrix for: {', '.join(cat_vars_for_analysis)}", parent=results_group)
    
    cramers_v_matrix = pd.DataFrame(np.zeros((len(cat_vars_for_analysis), len(cat_vars_for_analysis))),
                                    columns=cat_vars_for_analysis, index=cat_vars_for_analysis)

    for i in range(len(cat_vars_for_analysis)):
        for j in range(i, len(cat_vars_for_analysis)): # Calculate upper triangle including diagonal
            var1_name = cat_vars_for_analysis[i]
            var2_name = cat_vars_for_analysis[j]
            
            # Ensure series are not all NaNs or empty before passing to _calculate_cramers_v
            series1 = df[var1_name].dropna()
            series2 = df[var2_name].dropna()

            if series1.empty or series2.empty:
                c_v = 0.0
            elif var1_name == var2_name:
                 c_v = 1.0 # Cramer's V of a variable with itself is 1
            else:
                 c_v = _calculate_cramers_v(df[var1_name], df[var2_name])
            
            cramers_v_matrix.iloc[i, j] = c_v
            if i != j: # Mirror to lower triangle
                cramers_v_matrix.iloc[j, i] = c_v
    
    # Display as heatmap
    heatmap_data_cramers = cramers_v_matrix.values.flatten().tolist()
    rows_c, cols_c = cramers_v_matrix.shape
    col_labels_c = cramers_v_matrix.columns.tolist()

    with dpg.plot(label="Cramer's V Heatmap (Categorical Associations)", height=450, width=-1, parent=results_group, equal_aspects=True):
        xaxis_c = dpg.add_plot_axis(dpg.mvXAxis, label="")
        yaxis_c = dpg.add_plot_axis(dpg.mvYAxis, label="")
        if col_labels_c:
            dpg.set_axis_ticks(xaxis_c, tuple(zip(col_labels_c, list(range(cols_c)))))
            dpg.set_axis_ticks(yaxis_c, tuple(zip(col_labels_c, list(range(rows_c))))) # Y-axis ticks

        dpg.add_heat_series(heatmap_data_cramers, rows=rows_c, cols=cols_c, scale_min=0.0, scale_max=1.0,
                            format='%.2f', parent=yaxis_c, show_tooltips=True,
                            bounds_min=(0, 0), bounds_max=(cols_c, rows_c))


# --- Outlier Treatment Functions ---
def _outlier_method_changed_callback(sender, app_data, user_data):
    method = dpg.get_value(sender)
    dpg.configure_item(TAG_OUTLIER_CAPPING_CONTROLS_GROUP, show=(method == "Capping"))
    dpg.configure_item(TAG_OUTLIER_IF_CONTROLS_GROUP, show=(method == "Isolation Forest"))

def get_outlier_settings_for_saving() -> dict:
    """
    NEW: Gathers current outlier settings from the UI.
    Called by main_app.py to save settings.
    """
    settings = {'method': "None", 'params': {}}
    if not dpg.is_dearpygui_running(): return settings # Should not happen if UI is active

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
        # if dpg.does_item_exist(TAG_OUTLIER_IF_CONTAMINATION_SLIDER): # If using a slider
        #     settings['params']['contamination'] = dpg.get_value(TAG_OUTLIER_IF_CONTAMINATION_SLIDER)
        # else: # Defaulting to auto
        settings['params']['contamination'] = 'auto' # As per request
        if dpg.does_item_exist(TAG_OUTLIER_IF_VAR_SELECTOR):
            settings['params']['selected_vars_if'] = dpg.get_value(TAG_OUTLIER_IF_VAR_SELECTOR)
            
    return settings

def apply_outlier_treatment_from_settings(df: pd.DataFrame, config: dict, main_callbacks: dict) -> tuple[pd.DataFrame, bool]:
    """
    NEW: Applies outlier treatment to df based on config.
    Called by main_app.py when loading settings or by the apply button in this module.
    Returns (modified_df, True if changes were made else False)
    """
    if df is None or not config or 'method' not in config:
        return df, False

    method = config.get('method')
    params = config.get('params', {})
    original_shape = df.shape
    modified_df = df.copy()
    changes_made = False
    
    status_messages = []

    numeric_cols_in_df = _get_numeric_cols(modified_df)

    if method == "Capping":
        lower_p = params.get('lower_percentile', 1)
        upper_p = params.get('upper_percentile', 99)
        selected_vars_capping = params.get('selected_vars_capping', []) # Empty list means apply to all numeric if not specified
        
        vars_to_cap = selected_vars_capping if selected_vars_capping else numeric_cols_in_df
        vars_to_cap = [v for v in vars_to_cap if v in numeric_cols_in_df] # Ensure they are numeric

        if not vars_to_cap:
            status_messages.append("Capping: No numeric variables selected or available.")
        else:
            for col in vars_to_cap:
                if col in modified_df.columns and pd.api.types.is_numeric_dtype(modified_df[col].dtype):
                    low_val = np.percentile(modified_df[col].dropna(), lower_p)
                    high_val = np.percentile(modified_df[col].dropna(), upper_p)
                    # Check if capping actually changes anything to set changes_made flag
                    if modified_df[col].min() < low_val or modified_df[col].max() > high_val:
                        changes_made = True
                    modified_df[col] = np.clip(modified_df[col], low_val, high_val)
            status_messages.append(f"Capping applied ({lower_p}%-{upper_p}%) to: {', '.join(vars_to_cap)}.")
            if not changes_made and vars_to_cap: # If capping was configured but no values changed
                 status_messages.append("No actual data changes from capping (values were already within bounds).")


    elif method == "Isolation Forest":
        contamination = params.get('contamination', 'auto')
        selected_vars_if = params.get('selected_vars_if', []) # Empty list means apply to all numeric if not specified

        vars_to_process_if = selected_vars_if if selected_vars_if else numeric_cols_in_df
        vars_to_process_if = [v for v in vars_to_process_if if v in numeric_cols_in_df]

        if not vars_to_process_if:
            status_messages.append("Isolation Forest: No numeric variables selected or available.")
        else:
            for col in vars_to_process_if:
                if col in modified_df.columns and pd.api.types.is_numeric_dtype(modified_df[col].dtype):
                    series = modified_df[[col]].dropna() # IF needs 2D array and no NaNs
                    if len(series) < 2 or series.nunique().iloc[0] < 2 : # Not enough data or no variance
                        status_messages.append(f"IF for '{col}': Skipped (not enough data/variance).")
                        continue
                    
                    try:
                        model = IsolationForest(contamination=contamination, random_state=42)
                        model.fit(series)
                        is_outlier = model.predict(series) == -1 # -1 indicates outlier
                        
                        # Align outlier predictions back to original df's index (for non-NaN values of the column)
                        original_indices_of_series = series.index
                        outlier_indices = original_indices_of_series[is_outlier]

                        if len(outlier_indices) > 0:
                            changes_made = True
                             # For Isolation Forest, "removal" usually means setting to NaN or imputing.
                             # Here, we'll set them to NaN, to be handled by Step 1's imputation if re-run, or kept as NaN.
                            modified_df.loc[outlier_indices, col] = np.nan
                            status_messages.append(f"IF for '{col}': {len(outlier_indices)} outliers identified and set to NaN.")
                        else:
                            status_messages.append(f"IF for '{col}': No outliers identified.")
                    except Exception as e:
                        status_messages.append(f"IF for '{col}': Error - {str(e)}.")
                        print(f"Error applying Isolation Forest to {col}: {e}")
                        traceback.print_exc()

    final_message = "Outlier treatment applied. " if changes_made else "Outlier method configured, but no data changes made. "
    final_message += f"Original shape: {original_shape}, New shape: {modified_df.shape}. "
    final_message += "Details: " + " | ".join(status_messages)
    
    if dpg.does_item_exist(TAG_OUTLIER_RESULTS_TEXT):
        dpg.set_value(TAG_OUTLIER_RESULTS_TEXT, final_message)
    
    if main_callbacks and changes_made: # Notify main_app if actual changes happened
        main_callbacks.get('notify_eda_df_changed', lambda df: None)(modified_df.copy()) # Pass a copy
        main_callbacks.get('set_eda_outlier_applied_flag', lambda flag: None)(True)
    elif main_callbacks and not changes_made and method != "None": # Configured but no changes
        # If method was "None", the flag should remain as it was or be false.
        # If a method was chosen but no data changed, we might still want to consider settings "applied" in terms of configuration.
        # main_app handles the _eda_outlier_settings_applied_once flag restoration from settings.
        # This function's role is to report if *this run* made changes.
         main_callbacks.get('set_eda_outlier_applied_flag', lambda flag: None)(True) # Config was applied, even if no data change.
    elif main_callbacks and method == "None":
        main_callbacks.get('set_eda_outlier_applied_flag', lambda flag: None)(False)


    return modified_df, changes_made


def _apply_outlier_treatment_button_callback(sender, app_data, user_data):
    main_callbacks = user_data
    df_after_step1 = main_callbacks['get_df_after_step1']() # Get the clean slate from Step 1
    
    if df_after_step1 is None:
        _show_alert_modal("Error", "No data from Step 1 available to apply outlier treatment.")
        if dpg.does_item_exist(TAG_OUTLIER_RESULTS_TEXT):
            dpg.set_value(TAG_OUTLIER_RESULTS_TEXT, "Error: Data from Step 1 is not available.")
        return

    current_outlier_config = get_outlier_settings_for_saving() # Get current UI settings
    
    if dpg.does_item_exist(TAG_OUTLIER_RESULTS_TEXT): # Clear previous messages
         dpg.set_value(TAG_OUTLIER_RESULTS_TEXT, "Processing outlier treatment...")

    # Apply settings to a copy of df_after_step1
    # The apply_outlier_treatment_from_settings function will call notify_eda_df_changed
    _, changes_were_made = apply_outlier_treatment_from_settings(df_after_step1.copy(), current_outlier_config, main_callbacks)
    
    # The main current_df is updated via the callback.
    # Update the status text about whether settings are currently active on current_df
    _update_outlier_status_text(main_callbacks)

    # Trigger update for EDA module (and potentially others if current_df impacts them)
    # main_callbacks['trigger_specific_module_update'](ANALYSIS_STEPS[1]) # Assuming ANALYSIS_STEPS[1] is EDA key
    # This is now handled by notify_eda_df_changed -> step1_processing_complete (indirectly if df is reset) or trigger_all_module_updates

def _reset_outliers_to_after_step1_callback(sender, app_data, user_data):
    main_callbacks = user_data
    
    # This will reset current_df to df_after_step1 and re-apply any *saved* outlier settings.
    # If we want to just clear current run's outliers without re-applying saved ones,
    # we'd need a different logic, perhaps by setting method to "None" and applying.
    
    # For now, using main_app's reset which re-applies active_settings.
    main_callbacks['reset_eda_df_to_after_step1']() # This resets current_df and re-applies from active_settings
    
    # After reset, update the status message. The actual _eda_outlier_settings_applied_once flag
    # is managed by main_app and apply_outlier_treatment_from_settings.
    if dpg.does_item_exist(TAG_OUTLIER_RESULTS_TEXT):
         dpg.set_value(TAG_OUTLIER_RESULTS_TEXT, "Outlier treatment reset to data after Step 1. Saved outlier settings (if any) were re-applied.")
    _update_outlier_status_text(main_callbacks)


def _update_outlier_status_text(main_callbacks):
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_OUTLIER_STATUS_TEXT):
        return
    
    is_applied = main_callbacks.get('get_eda_outlier_applied_flag', lambda: False)()
    active_settings = main_callbacks.get('get_active_settings', lambda: {})() # Need a way to get main_app's active_settings
                                                                            # This might need a new callback from main_app.
                                                                            # For now, let's assume a simplified status.

    # This is a simplified status. A more accurate one would compare current UI settings with applied settings.
    # The `_eda_outlier_settings_applied_once` flag from main_app signals if *any* outlier settings (from load or apply)
    # are considered active on the current `current_df`.
    if is_applied:
        # Check the method from currently *loaded/active* settings if possible.
        # This part is tricky without direct access to main_app's active_settings' outlier part.
        # Let's rely on the flag for a general status for now.
        dpg.set_value(TAG_OUTLIER_STATUS_TEXT, "Status: Outlier settings are currently reflected in the EDA DataFrame.")
        dpg.configure_item(TAG_OUTLIER_STATUS_TEXT, color=(0, 200, 0)) # Green
    else:
        dpg.set_value(TAG_OUTLIER_STATUS_TEXT, "Status: No outlier treatment currently active on EDA DataFrame (or treatment resulted in no changes).")
        dpg.configure_item(TAG_OUTLIER_STATUS_TEXT, color=(200, 200, 0)) # Yellow


# --- Main UI Creation & Update (Modified) ---
def _create_outlier_treatment_tab_content(parent_tag: str, main_callbacks: dict):
    with dpg.group(parent=parent_tag): # Main group for the tab content
        dpg.add_text("Select an outlier treatment method. Changes will modify the DataFrame used for EDA.", wrap=-1)
        dpg.add_text("Note: Applying treatment here updates the current EDA session's DataFrame. These settings are saved with the session.", wrap=-1, color=(200,200,200))
        dpg.add_separator()

        dpg.add_text("Current Outlier Treatment Status:", weight=dpg.mvFontWeight_Bold)
        dpg.add_text("Status: Initializing...", tag=TAG_OUTLIER_STATUS_TEXT, wrap=-1) # Updated by _update_outlier_status_text
        dpg.add_spacer(height=5)


        dpg.add_radio_button(
            items=["None", "Capping", "Isolation Forest"],
            tag=TAG_OUTLIER_METHOD_RADIO, default_value="None", horizontal=True,
            callback=_outlier_method_changed_callback
        )
        dpg.add_spacer(height=10)

        # --- Capping Controls ---
        with dpg.group(tag=TAG_OUTLIER_CAPPING_CONTROLS_GROUP, show=False):
            dpg.add_text("Capping Settings:", weight=dpg.mvFontWeight_Bold)
            dpg.add_text("Caps selected numeric variables at specified percentiles.", wrap=-1)
            dpg.add_input_int(label="Lower Percentile (1-20)", tag=TAG_OUTLIER_CAPPING_LOWER_PERCENTILE, default_value=1, min_value=1, max_value=20, width=150)
            dpg.add_input_int(label="Upper Percentile (80-99)", tag=TAG_OUTLIER_CAPPING_UPPER_PERCENTILE, default_value=99, min_value=80, max_value=99, width=150)
            dpg.add_text("Apply to Variables (numeric only, select none for all):")
            dpg.add_listbox(tag=TAG_OUTLIER_CAPPING_VAR_SELECTOR, width=-1, num_items=5) # Populated in update_ui


        # --- Isolation Forest Controls ---
        with dpg.group(tag=TAG_OUTLIER_IF_CONTROLS_GROUP, show=False):
            dpg.add_text("Isolation Forest Settings:", weight=dpg.mvFontWeight_Bold)
            dpg.add_text("Identifies outliers in selected numeric variables using Isolation Forest (contamination='auto'). Outliers are set to NaN.", wrap=-1)
            # Contamination is hardcoded to 'auto' as per request. If UI control needed:
            # dpg.add_slider_float(label="Contamination (approx. proportion of outliers)", tag=TAG_OUTLIER_IF_CONTAMINATION_SLIDER, default_value=0.05, min_value=0.01, max_value=0.5, format="%.2f")
            dpg.add_text("Apply to Variables (numeric only, select none for all):")
            dpg.add_listbox(tag=TAG_OUTLIER_IF_VAR_SELECTOR, width=-1, num_items=5) # Populated in update_ui

        dpg.add_spacer(height=15)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Apply Outlier Treatment to EDA Data", tag=TAG_OUTLIER_APPLY_BUTTON,
                           callback=_apply_outlier_treatment_button_callback, user_data=main_callbacks, height=30)
            dpg.add_button(label="Reset to Data Post-Step 1 (Re-applies Session Outliers)", 
                           tag=TAG_OUTLIER_RESET_TO_AFTER_STEP1_BUTTON,
                           callback=_reset_outliers_to_after_step1_callback, user_data=main_callbacks, height=30)
        dpg.add_separator()
        dpg.add_text("Application Results/Log:", weight=dpg.mvFontWeight_Bold)
        dpg.add_text("Apply treatment to see effects.", tag=TAG_OUTLIER_RESULTS_TEXT, wrap=-1)


def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """
    MODIFIED: Adds new MVA tabs and the Outlier Treatment tab.
    """
    global _main_app_callbacks_eda, _util_funcs_eda
    _main_app_callbacks_eda = main_callbacks
    _util_funcs_eda = main_callbacks.get('get_util_funcs', lambda: {})()
    
    main_callbacks['register_step_group_tag'](step_name, TAG_EDA_GROUP)
    with dpg.group(tag=TAG_EDA_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---"); dpg.add_separator()
        with dpg.tab_bar(tag=TAG_EDA_MAIN_TAB_BAR):
            # SVA Tab (Assumed to be defined as in original, for brevity)
            with dpg.tab(label="Single Variable Analysis (SVA)", tag=TAG_SVA_TAB):
                # ... (Original SVA UI setup using TAG_SVA_... tags) ...
                # This part is taken from the user's provided step_02 file structure
                with dpg.group(horizontal=True): 
                    with dpg.group(width=280): 
                        dpg.add_text("Variable Filter")
                        dpg.add_radio_button(
                            items=["Strong (Top 5-10 relevant)", "Medium (Top 11-20 relevant)", "Weak (Exclude obvious non-analytical)", "None (All variables)"],
                            tag=TAG_SVA_FILTER_STRENGTH_RADIO, default_value="Weak (Exclude obvious non-analytical)"
                        )
                        dpg.add_spacer(height=5) 
                        dpg.add_text("Filter Info:", wrap=270, color=(200,200,200))
                        dpg.add_text("- Strong/Medium: Numeric vars ranked by relevance to Target (if set).", wrap=270, color=(200,200,200))
                        dpg.add_text("- Weak: Excludes single-value & binary numeric vars.", wrap=270, color=(200,200,200))
                        dpg.add_text("- None: Includes most vars (text types excluded).", wrap=270, color=(200,200,200))
                    dpg.add_spacer(width=10)                     
                    with dpg.group(): 
                        dpg.add_text("Grouping & Plot Option")
                        dpg.add_checkbox(label="Group by Target (2-7 Unique Values)",
                                         tag=TAG_SVA_GROUP_BY_TARGET_CHECKBOX, default_value=False,
                                         user_data=main_callbacks, 
                                         callback=_sva_group_by_target_callback 
                                         )
                        dpg.add_radio_button(items=["KDE", "Histogram"], 
                                             tag=TAG_SVA_GROUPED_PLOT_TYPE_RADIO, 
                                             default_value="KDE", horizontal=True, show=False,
                                             )
                        dpg.add_spacer(height=10)
                        dpg.add_button(label="Run Single Variable Analysis", tag=TAG_SVA_RUN_BUTTON,
                                       callback=lambda: _apply_sva_filters_and_run(main_callbacks), # Assuming _apply_sva_filters_and_run exists
                                       width=-1, height=30)
                dpg.add_separator()
                with dpg.child_window(tag=TAG_SVA_RESULTS_CHILD_WINDOW, border=True):
                    dpg.add_text("Select filter options and click 'Run Single Variable Analysis'.")


            # MVA Tab (now a container for sub-tabs)
            with dpg.tab(label="Multivariate Analysis (MVA)", tag=TAG_MVA_TAB):
                with dpg.tab_bar(tag=TAG_MVA_SUB_TAB_BAR): # Sub-tab bar for MVA sections
                    with dpg.tab(label="Correlation (Numeric)", tag=TAG_MVA_CORR_TAB):
                        dpg.add_button(label="Run Correlation Analysis", tag=TAG_MVA_CORR_RUN_BUTTON,
                                       callback=lambda: _run_correlation_analysis(main_callbacks['get_current_df'](), main_callbacks['get_util_funcs'](), main_callbacks))
                        dpg.add_child_window(tag=TAG_MVA_CORR_RESULTS_GROUP, border=True, height=-1) # Group for all corr results

                    with dpg.tab(label="Pair Plot (Numeric)", tag=TAG_MVA_PAIRPLOT_TAB):
                        dpg.add_text("Select numeric variables (up to 7 recommended for DPG performance). If none selected, defaults based on variable count.", wrap = -1)
                        dpg.add_listbox(tag=TAG_MVA_PAIRPLOT_VAR_SELECTOR, width=-1, num_items=8)
                        dpg.add_combo(label="Hue (Optional Categorical Var, <10 Categories)", tag=TAG_MVA_PAIRPLOT_HUE_COMBO, width=350)
                        dpg.add_button(label="Generate Pair Plot", tag=TAG_MVA_PAIRPLOT_RUN_BUTTON,
                                       callback=lambda: _run_pair_plot_analysis(
                                           main_callbacks['get_current_df'](),
                                           dpg.get_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR),
                                           dpg.get_value(TAG_MVA_PAIRPLOT_HUE_COMBO),
                                           main_callbacks['get_util_funcs'](),
                                           main_callbacks
                                       ))
                        dpg.add_child_window(tag=TAG_MVA_PAIRPLOT_RESULTS_GROUP, border=True, height=-1)

                    with dpg.tab(label="Target vs Feature", tag=TAG_MVA_TARGET_TAB):
                        dpg.add_text("Analyze relationship between features and the selected target variable.", tag=TAG_MVA_TARGET_INFO_TEXT, wrap=-1)
                        with dpg.group(horizontal=True):
                            dpg.add_combo(label="Feature Variable", tag=TAG_MVA_TARGET_FEATURE_COMBO, width=300)
                            dpg.add_button(label="Analyze vs Target", tag=TAG_MVA_TARGET_RUN_BUTTON,
                                           callback=lambda: _run_target_variable_analysis( # Assuming this exists
                                               main_callbacks['get_current_df'](),
                                               main_callbacks['get_selected_target_variable'](),
                                               main_callbacks['get_selected_target_variable_type'](),
                                               dpg.get_value(TAG_MVA_TARGET_FEATURE_COMBO),
                                               main_callbacks['get_util_funcs'](),
                                               main_callbacks 
                                           ))
                        dpg.add_separator()
                        dpg.add_child_window(tag=TAG_MVA_TARGET_RESULTS_GROUP, border=True, height=-1)
                    
                    # New MVA Tab for Categorical EDA
                    with dpg.tab(label="Correlation (Categorical)", tag=TAG_MVA_CAT_EDA_TAB):
                        dpg.add_text("Analyze associations between categorical variables using Cramer's V.", wrap=-1)
                        dpg.add_text("Select variables (up to 20). If none selected, defaults based on available categoricals.", wrap=-1)
                        dpg.add_listbox(tag=TAG_MVA_CAT_EDA_VAR_SELECTOR, width=-1, num_items=8)
                        dpg.add_button(label="Run Categorical Association Analysis", tag=TAG_MVA_CAT_EDA_RUN_BUTTON,
                                       callback=lambda: _run_categorical_correlation_analysis(
                                           main_callbacks['get_current_df'](),
                                           main_callbacks['get_util_funcs'](),
                                           main_callbacks
                                       ))
                        dpg.add_child_window(tag=TAG_MVA_CAT_EDA_RESULTS_GROUP, border=True, height=-1)

            # New Main Tab for Outlier Treatment
            with dpg.tab(label="Outlier Treatment", tag=TAG_OUTLIER_TAB):
                _create_outlier_treatment_tab_content(TAG_OUTLIER_TAB, main_callbacks)

    main_callbacks['register_module_updater'](step_name, lambda df_arg, mc_arg: update_ui(df_arg, mc_arg))


def update_ui(current_df: pd.DataFrame, main_callbacks: dict):
    """
    MODIFIED: Updates UI elements for new MVA and Outlier sections.
    """
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_EDA_GROUP): return
    # print("DEBUG: EDA Module: update_ui called.")
    
    global _main_app_callbacks_eda, _util_funcs_eda
    _main_app_callbacks_eda = main_callbacks
    _util_funcs_eda = main_callbacks.get('get_util_funcs', lambda: {})()

    # SVA UI updates (as in original)
    sva_results_child = TAG_SVA_RESULTS_CHILD_WINDOW
    if current_df is None:
        if dpg.does_item_exist(sva_results_child):
            dpg.delete_item(sva_results_child, children_only=True)
            dpg.add_text("Load data to perform Single Variable Analysis.", parent=sva_results_child)
        if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO): dpg.set_value(TAG_SVA_FILTER_STRENGTH_RADIO, "Weak (Exclude obvious non-analytical)")
        if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX): dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
        if dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO): dpg.configure_item(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, show=False); dpg.set_value(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, "KDE")

    # MVA Tab UI Updates
    all_columns = current_df.columns.tolist() if current_df is not None else []
    numeric_cols = _get_numeric_cols(current_df) if current_df is not None else []
    # Pass main_callbacks for s1_type informed categorical detection
    categorical_cols_for_mva_hue = [""] + (_get_categorical_cols(current_df, max_unique_for_cat=10, main_callbacks=main_callbacks) if current_df is not None else [])
    all_cat_cols_for_selector = _get_categorical_cols(current_df, max_unique_for_cat=30, main_callbacks=main_callbacks) if current_df is not None else []


    # Pair Plot selectors
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_VAR_SELECTOR):
        # Keep previous selection if valid, otherwise it clears.
        # current_selection_pp = dpg.get_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR) 
        dpg.configure_item(TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=numeric_cols)
        # if isinstance(current_selection_pp, list) and all(item in numeric_cols for item in current_selection_pp):
        #     try: dpg.set_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR, current_selection_pp)
        #     except Exception: pass 
        # el
        if not numeric_cols: dpg.set_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR, [])


    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_HUE_COMBO):
        current_hue_pp = dpg.get_value(TAG_MVA_PAIRPLOT_HUE_COMBO)
        dpg.configure_item(TAG_MVA_PAIRPLOT_HUE_COMBO, items=categorical_cols_for_mva_hue)
        if current_hue_pp and current_hue_pp in categorical_cols_for_mva_hue:
            dpg.set_value(TAG_MVA_PAIRPLOT_HUE_COMBO, current_hue_pp)
        elif categorical_cols_for_mva_hue: dpg.set_value(TAG_MVA_PAIRPLOT_HUE_COMBO, categorical_cols_for_mva_hue[0])
        else: dpg.set_value(TAG_MVA_PAIRPLOT_HUE_COMBO, "")

    # Target vs Feature selectors
    selected_target_var_mva = main_callbacks['get_selected_target_variable']()
    if dpg.does_item_exist(TAG_MVA_TARGET_INFO_TEXT):
        if current_df is not None and selected_target_var_mva and selected_target_var_mva in all_columns:
            dpg.set_value(TAG_MVA_TARGET_INFO_TEXT, f"Analyzing features against Target: '{selected_target_var_mva}' (Type: {main_callbacks['get_selected_target_variable_type']()})")
            if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO):
                feature_candidates_mva = [col for col in all_columns if col != selected_target_var_mva]
                # current_feature_target_mva = dpg.get_value(TAG_MVA_TARGET_FEATURE_COMBO)
                dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=feature_candidates_mva)
                # if current_feature_target_mva and current_feature_target_mva in feature_candidates_mva:
                #      dpg.set_value(TAG_MVA_TARGET_FEATURE_COMBO, current_feature_target_mva)
                # elif feature_candidates_mva: dpg.set_value(TAG_MVA_TARGET_FEATURE_COMBO, feature_candidates_mva[0])
                # else: dpg.set_value(TAG_MVA_TARGET_FEATURE_COMBO, None) 
        else:
            dpg.set_value(TAG_MVA_TARGET_INFO_TEXT, "Load data and select a global target variable (top-left panel) to enable this analysis.")
            if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO): 
                dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=[])
                dpg.set_value(TAG_MVA_TARGET_FEATURE_COMBO, None) 

    # Categorical EDA selector
    if dpg.does_item_exist(TAG_MVA_CAT_EDA_VAR_SELECTOR):
        dpg.configure_item(TAG_MVA_CAT_EDA_VAR_SELECTOR, items=all_cat_cols_for_selector)
        if not all_cat_cols_for_selector: dpg.set_value(TAG_MVA_CAT_EDA_VAR_SELECTOR, [])


    # Outlier Treatment selectors
    if dpg.does_item_exist(TAG_OUTLIER_CAPPING_VAR_SELECTOR):
        dpg.configure_item(TAG_OUTLIER_CAPPING_VAR_SELECTOR, items=numeric_cols)
        if not numeric_cols: dpg.set_value(TAG_OUTLIER_CAPPING_VAR_SELECTOR, [])
    if dpg.does_item_exist(TAG_OUTLIER_IF_VAR_SELECTOR):
        dpg.configure_item(TAG_OUTLIER_IF_VAR_SELECTOR, items=numeric_cols)
        if not numeric_cols: dpg.set_value(TAG_OUTLIER_IF_VAR_SELECTOR, [])
    
    # Update outlier status text
    _update_outlier_status_text(main_callbacks)


    # Clear MVA result areas if no data
    if current_df is None:
        if dpg.does_item_exist(TAG_MVA_CORR_RESULTS_GROUP): dpg.delete_item(TAG_MVA_CORR_RESULTS_GROUP, children_only=True); dpg.add_text("Load data.", parent=TAG_MVA_CORR_RESULTS_GROUP)
        if dpg.does_item_exist(TAG_MVA_PAIRPLOT_RESULTS_GROUP): dpg.delete_item(TAG_MVA_PAIRPLOT_RESULTS_GROUP, children_only=True); dpg.add_text("Load data.", parent=TAG_MVA_PAIRPLOT_RESULTS_GROUP)
        if dpg.does_item_exist(TAG_MVA_TARGET_RESULTS_GROUP): dpg.delete_item(TAG_MVA_TARGET_RESULTS_GROUP, children_only=True); dpg.add_text("Load data and select target.", parent=TAG_MVA_TARGET_RESULTS_GROUP)
        if dpg.does_item_exist(TAG_MVA_CAT_EDA_RESULTS_GROUP): dpg.delete_item(TAG_MVA_CAT_EDA_RESULTS_GROUP, children_only=True); dpg.add_text("Load data.", parent=TAG_MVA_CAT_EDA_RESULTS_GROUP)
        if dpg.does_item_exist(TAG_OUTLIER_RESULTS_TEXT): dpg.set_value(TAG_OUTLIER_RESULTS_TEXT, "Load data to manage outliers.")

    # print("DEBUG: EDA UI selectors updated.")

# This function is called by main_app to reset EDA UI elements to their default state
# when a new file is loaded or state is reset.
def reset_eda_ui_defaults():
    if not dpg.is_dearpygui_running(): return

    # SVA Defaults
    if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO):
        dpg.set_value(TAG_SVA_FILTER_STRENGTH_RADIO, "Weak (Exclude obvious non-analytical)")
    if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
        dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
    if dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
        dpg.set_value(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, "KDE")
        dpg.configure_item(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, show=False)
    if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
        dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
        dpg.add_text("Select filter options and click 'Run Single Variable Analysis'.", parent=TAG_SVA_RESULTS_CHILD_WINDOW)

    # MVA Defaults
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_VAR_SELECTOR):
        dpg.configure_item(TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=[])
        dpg.set_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR, [])
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_HUE_COMBO):
        dpg.configure_item(TAG_MVA_PAIRPLOT_HUE_COMBO, items=[""])
        dpg.set_value(TAG_MVA_PAIRPLOT_HUE_COMBO, "")
    if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO):
        dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=[])
        dpg.set_value(TAG_MVA_TARGET_FEATURE_COMBO, None)
    if dpg.does_item_exist(TAG_MVA_CAT_EDA_VAR_SELECTOR):
        dpg.configure_item(TAG_MVA_CAT_EDA_VAR_SELECTOR, items=[])
        dpg.set_value(TAG_MVA_CAT_EDA_VAR_SELECTOR, [])


    # Clear MVA result areas
    result_areas_mva = [TAG_MVA_CORR_RESULTS_GROUP, TAG_MVA_PAIRPLOT_RESULTS_GROUP, 
                        TAG_MVA_TARGET_RESULTS_GROUP, TAG_MVA_CAT_EDA_RESULTS_GROUP]
    for area in result_areas_mva:
        if dpg.does_item_exist(area):
            dpg.delete_item(area, children_only=True)
            if area == TAG_MVA_TARGET_RESULTS_GROUP:
                dpg.add_text("Load data and select target.", parent=area)
            else:
                dpg.add_text("Run analysis to see results.", parent=area)
    
    # Outlier Treatment Defaults
    if dpg.does_item_exist(TAG_OUTLIER_METHOD_RADIO):
        dpg.set_value(TAG_OUTLIER_METHOD_RADIO, "None")
    if dpg.does_item_exist(TAG_OUTLIER_CAPPING_LOWER_PERCENTILE):
        dpg.set_value(TAG_OUTLIER_CAPPING_LOWER_PERCENTILE, 1)
    if dpg.does_item_exist(TAG_OUTLIER_CAPPING_UPPER_PERCENTILE):
        dpg.set_value(TAG_OUTLIER_CAPPING_UPPER_PERCENTILE, 99)
    if dpg.does_item_exist(TAG_OUTLIER_CAPPING_VAR_SELECTOR):
        dpg.configure_item(TAG_OUTLIER_CAPPING_VAR_SELECTOR, items=[])
        dpg.set_value(TAG_OUTLIER_CAPPING_VAR_SELECTOR, [])
    if dpg.does_item_exist(TAG_OUTLIER_IF_VAR_SELECTOR):
        dpg.configure_item(TAG_OUTLIER_IF_VAR_SELECTOR, items=[])
        dpg.set_value(TAG_OUTLIER_IF_VAR_SELECTOR, [])
    
    if dpg.does_item_exist(TAG_OUTLIER_CAPPING_CONTROLS_GROUP):
        dpg.configure_item(TAG_OUTLIER_CAPPING_CONTROLS_GROUP, show=False)
    if dpg.does_item_exist(TAG_OUTLIER_IF_CONTROLS_GROUP):
        dpg.configure_item(TAG_OUTLIER_IF_CONTROLS_GROUP, show=False)
        
    if dpg.does_item_exist(TAG_OUTLIER_RESULTS_TEXT):
        dpg.set_value(TAG_OUTLIER_RESULTS_TEXT, "Apply outlier treatment to see effects.")
    if dpg.does_item_exist(TAG_OUTLIER_STATUS_TEXT):
        dpg.set_value(TAG_OUTLIER_STATUS_TEXT, "Status: Initializing...")
        dpg.configure_item(TAG_OUTLIER_STATUS_TEXT, color=(180, 180, 180)) # Default color


# --- (Original SVA functions like _get_filtered_variables, _create_sva_basic_stats_table, etc. would be here) ---
# --- (Original MVA functions like _run_target_variable_analysis, etc. would be here, if not modified above) ---