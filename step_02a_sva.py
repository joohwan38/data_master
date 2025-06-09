# step_02a_sva.py (Seaborn 경고 및 trace trap 오류 수정 버전)
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
import utils 
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import tkinter as tk
from tkinter import filedialog


warnings.filterwarnings('ignore', category=RuntimeWarning)
# Seaborn의 향후 변경에 대한 경고는 무시하지 않고 코드를 수정하여 해결합니다.
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')


# --- 태그 정의 ---
TAG_SVA_STEP_GROUP = "sva_step_group"
TAG_SVA_FILTER_STRENGTH_RADIO = "sva_filter_strength_radio"
TAG_SVA_GROUP_BY_TARGET_CHECKBOX = "sva_group_by_target_checkbox"
TAG_SVA_RUN_BUTTON = "sva_run_button"
TAG_SVA_RESULTS_CHILD_WINDOW = "sva_results_child_window"
TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX = "sva_var_section_prefix_"
TAG_SVA_MODULE_PROGRESS_MODAL = "sva_module_specific_progress_modal"
TAG_SVA_MODULE_PROGRESS_TEXT = "sva_module_specific_progress_text"
TAG_SVA_MODULE_ALERT_MODAL = "sva_module_specific_alert_modal"
TAG_SVA_MODULE_ALERT_TEXT = "sva_module_specific_alert_text"
TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO = "sva_grouped_numeric_plot_type_radio"
TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO = "sva_grouped_cat_plot_type_radio"
X_AXIS_ROTATED_TICKS_THEME_TAG = "sva_x_axis_rotated_theme"
TAG_SVA_EXPORT_BUTTON = "sva_export_button"

# --- 모듈 상태 변수 ---
_sva_main_app_callbacks: Dict[str, Any] = {}
_sva_util_funcs: Dict[str, Any] = {}
_sva_results_cache: Dict[str, Dict[str, Any]] = {}

def get_sva_settings_for_saving() -> Dict[str, Any]:
    settings = {}
    if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO):
        settings['filter_strength'] = dpg.get_value(TAG_SVA_FILTER_STRENGTH_RADIO)
    if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
        settings['group_by_target'] = dpg.get_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX)
    if dpg.does_item_exist(TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO):
        settings['grouped_numeric_plot_type'] = dpg.get_value(TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO)
    if dpg.does_item_exist(TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO):
        settings['grouped_cat_plot_type'] = dpg.get_value(TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO)
    return settings

def apply_sva_settings_from_loaded(settings: Dict[str, Any], current_df: Optional[pd.DataFrame], main_callbacks: Dict[str, Any]):
    if not dpg.is_dearpygui_running(): return

    if 'filter_strength' in settings and dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO):
        dpg.set_value(TAG_SVA_FILTER_STRENGTH_RADIO, settings['filter_strength'])
    
    group_by_target_val = settings.get('group_by_target', False)
    if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
        dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, group_by_target_val)
    
    _sva_group_by_target_cb(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, group_by_target_val, main_callbacks)
    
    if dpg.does_item_exist(TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO):
        if 'grouped_numeric_plot_type' in settings:
            dpg.set_value(TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO, settings['grouped_numeric_plot_type'])

    if dpg.does_item_exist(TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO):
        if 'grouped_cat_plot_type' in settings:
            dpg.set_value(TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO, settings['grouped_cat_plot_type'])

    if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
        dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
        dpg.add_text("Settings loaded. Click 'Run Single Variable Analysis' to update results.", parent=TAG_SVA_RESULTS_CHILD_WINDOW)
    
    if dpg.does_item_exist(TAG_SVA_EXPORT_BUTTON):
        dpg.configure_item(TAG_SVA_EXPORT_BUTTON, show=False)


def _sva_setup_axis_themes():
    if not dpg.does_item_exist(X_AXIS_ROTATED_TICKS_THEME_TAG):
        with dpg.theme(tag=X_AXIS_ROTATED_TICKS_THEME_TAG):
            with dpg.theme_component(dpg.mvXAxis): pass

def _sva_get_filtered_variables(df: pd.DataFrame, strength: str, callbacks: dict, target: str = None) -> Tuple[List[str], str]:
    if df is None or df.empty: return [], strength
    types = callbacks.get('get_column_analysis_types', lambda: {})()
    types = types if isinstance(types, dict) else {c: str(df[c].dtype) for c in df.columns}
    
    cols = [c for c in df.columns if not any(k in types.get(c, str(df[c].dtype)) for k in ["Text (", "Potentially Sensitive"])]
    if strength == "None (All variables)": return cols, strength
    
    weak_cols = []
    for c_name in cols:
        s = df[c_name]
        if s.nunique(dropna=False) <= 1: continue
        is_bin_num = "Numeric (Binary)" in types.get(c_name, str(s.dtype)) or \
                     ("Numeric" in types.get(c_name, str(s.dtype)) and len(s.dropna().unique())==2 and set(s.dropna().unique()).issubset({0,1,0.0,1.0}))
        if is_bin_num: continue
        weak_cols.append(c_name)
        
    if strength == "Weak (Exclude obvious non-analytical)": return weak_cols, strength
    
    if not target: 
        utils.show_dpg_alert_modal("Target Required", f"Filter '{strength}' requires a target variable to be selected.", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
        return [], strength
    
    candidate_cols = weak_cols
    if not candidate_cols:
        utils.show_dpg_alert_modal("Filter Error", f"Filter '{strength}': No candidate variables found after 'Weak' filtering.", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
        return [], strength
        
    target_type = callbacks['get_selected_target_variable_type']()
    relevance_tuples = utils.calculate_feature_target_relevance(
        df, target, target_type, candidate_cols, callbacks 
    )
    relevant_scores_vars = [var_name for var_name, score in relevance_tuples] 

    if not relevant_scores_vars: 
        utils.show_dpg_alert_modal("Filter Error", f"Filter '{strength}': Could not calculate relevance for any variables.", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
        return [], strength
        
    return relevant_scores_vars[:10 if strength == "Strong (Top 5-10 relevant)" else 20], strength

def _sva_create_basic_stats_table(parent: str, series: pd.Series, u_funcs: dict, analysis_override: str = None) -> pd.DataFrame:
    dpg.add_text("Basic Statistics", parent=parent)
    stats_data = [{'S': 'Count', 'V': str(series.count())}, {'S': 'Missing', 'V': str(series.isnull().sum())},
                  {'S': 'Missing %', 'V': f"{series.isnull().mean()*100:.2f}%"},
                  {'S': 'Unique (Actual)', 'V': str(series.nunique(dropna=False))},
                  {'S': 'Unique (Valid)', 'V': str(series.nunique())}]
    s1_types_func = u_funcs['main_app_callbacks'].get('get_column_analysis_types', lambda: {})
    s1_types_dict = s1_types_func() 
    s1_type = s1_types_dict.get(series.name, str(series.dtype))
    is_bin_num = "Numeric (Binary)" in s1_type
    treat_cat = is_bin_num or analysis_override == "ForceCategoricalForBinaryNumeric" or \
                any(k in s1_type for k in ["Categorical", "Text (", "Potentially Sensitive"]) or series.nunique(dropna=False) < 5
    if pd.api.types.is_numeric_dtype(series.dtype) and not treat_cat:
        desc = series.describe()
        for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            if stat in desc.index: stats_data.append({'S': stat, 'V': f"{desc[stat]:.3f}" if isinstance(desc[stat], (int,float,np.number)) else str(desc[stat])})
        clean_s = series.dropna();
        if len(clean_s) >= 3:
            try:
                stats_data.append({'S': 'Skewness', 'V': f"{clean_s.skew():.3f}" if pd.notna(clean_s.skew()) else "N/A"})
                stats_data.append({'S': 'Kurtosis', 'V': f"{clean_s.kurtosis():.3f}" if pd.notna(clean_s.kurtosis()) else "N/A"})
            except: pass
    else:
        v_counts = series.value_counts(dropna=False).nlargest(5)
        if not v_counts.empty:
            stats_data.extend([{'S': 'Mode (Top1)', 'V': str(v_counts.index[0])}, {'S': 'Mode Freq (Top1)', 'V': str(v_counts.iloc[0])}])
            if len(v_counts) > 1: stats_data.extend([{'S': 'Mode (Top2)', 'V': str(v_counts.index[1])}, {'S': 'Mode Freq (Top2)', 'V': str(v_counts.iloc[1])}])
    
    df_stats = pd.DataFrame(stats_data).rename(columns={'S':'Statistic', 'V':'Value'})
    if not df_stats.empty:
        tbl_tag, tbl_h = dpg.generate_uuid(), min(280, len(df_stats) * 22 + 40)
        with dpg.table(header_row=True, tag=tbl_tag, parent=parent, policy=dpg.mvTable_SizingStretchProp, height=int(tbl_h), scrollY=True,
                      borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True, resizable=True):
            u_funcs['create_table_with_data'](tbl_tag, df_stats, parent_df_for_widths=df_stats)

    return df_stats

def _sva_create_advanced_relations_table(parent: str, series: pd.Series, full_df: pd.DataFrame, u_funcs: dict, col_w: int) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    norm_data = []
    df_norm, df_rel = None, None
    s1_types_func = u_funcs['main_app_callbacks'].get('get_column_analysis_types', lambda: {})
    s1_types_dict = s1_types_func() 
    s1_type = s1_types_dict.get(series.name, str(series.dtype))
    is_num_test = pd.api.types.is_numeric_dtype(series.dtype) and "Binary" not in s1_type and series.nunique(dropna=False) >= 5
    if is_num_test:
        clean_s = series.dropna()
        if 3 <= len(clean_s) < 5000:
            try:
                sw_stat, sw_p = stats.shapiro(clean_s.astype(float, errors='ignore'))
                norm_data.extend([{'T': 'Shapiro-Wilk W', 'V': f"{sw_stat:.3f}"}, {'T': 'p-value (SW)', 'V': f"{sw_p:.3f}"},
                                  {'T': 'Normality (α=0.05)', 'V': "Likely Normal" if sw_p > 0.05 else "Likely Not Normal"}])
            except: norm_data.append({'T': 'Shapiro-Wilk', 'V': 'Error'})
        else: norm_data.append({'T': 'Shapiro-Wilk', 'V': 'N/A (size)'})
    else: norm_data.append({'T': 'Shapiro-Wilk', 'V': 'N/A (type)'})
    if norm_data:
        dpg.add_text("Normality Test:", parent=parent)
        df_norm = pd.DataFrame(norm_data).rename(columns={'T':'Test', 'V':'Value'})
        tbl_tag, tbl_h = dpg.generate_uuid(), min(120, len(df_norm) * 22 + 30)
        with dpg.table(header_row=True, tag=tbl_tag, parent=parent, policy=dpg.mvTable_SizingStretchProp, height=int(tbl_h), scrollY=True,
                      borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True, resizable=True):
            u_funcs['create_table_with_data'](tbl_tag, df_norm, parent_df_for_widths=df_norm)
        dpg.add_spacer(height=5, parent=parent)
    
    dpg.add_text("Top Related Variables:", parent=parent)
    rel_vars_data = _sva_get_top_correlated(full_df, series.name, top_n=5)
    if rel_vars_data:
        if len(rel_vars_data) == 1 and 'Info' in rel_vars_data[0]: dpg.add_text(rel_vars_data[0]['Info'], parent=parent, wrap=col_w-10)
        else:
            actual_data = [item for item in rel_vars_data if 'Info' not in item]
            if actual_data:
                df_rel = pd.DataFrame(actual_data)
                tbl_tag, tbl_h = dpg.generate_uuid(), min(150, len(df_rel) * 22 + 40)
                with dpg.table(header_row=True, tag=tbl_tag, parent=parent, policy=dpg.mvTable_SizingStretchProp, height=int(tbl_h), scrollY=True,
                              borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True, resizable=True):
                    u_funcs['create_table_with_data'](tbl_tag, df_rel, parent_df_for_widths=df_rel)
            else: dpg.add_text("No specific related variables found.", parent=parent, wrap=col_w-10)
    else: dpg.add_text("No correlation/association data.", parent=parent, wrap=col_w-10)
    
    return df_norm, df_rel

def _sva_get_top_correlated(df: pd.DataFrame, cur_var: str, top_n: int = 5) -> List[Dict[str, str]]:
    if df is None or cur_var not in df.columns or len(df.columns) < 2: return [{'Info': 'Not enough data'}]
    corrs, results, series = [], [], df[cur_var].copy()
    if pd.api.types.is_numeric_dtype(series.dtype):
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            if col == cur_var: continue
            try:
                temp_df = df[[cur_var, col]].dropna();
                if len(temp_df) >= 2 :
                    val = temp_df[cur_var].corr(temp_df[col])
                    if pd.notna(val) and abs(val) > 0.01: corrs.append((col, val, "Pearson"))
            except: pass
    elif series.nunique(dropna=False) < 30 or pd.api.types.is_categorical_dtype(series.dtype) or series.dtype == 'object':
        cand_cols = [c for c in df.columns if c != cur_var and (df[c].nunique(dropna=False) < 30 or pd.api.types.is_categorical_dtype(df[c].dtype) or df[c].dtype == 'object')]
        for col in cand_cols:
            try:
                c_v = utils.calculate_cramers_v(series, df[col]) 
                if pd.notna(c_v) and c_v > 0.01: corrs.append((col, c_v, "Cramér's V"))
            except: pass
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, val, metric in corrs[:top_n]: results.append({'Variable': name, 'Metric': metric, 'Value': f"{val:.3f}"})
    return results if results else [{'Info': 'No significant relations.'}]

# --- [MODIFIED] Plotting function to fix deprecation warning and crash ---
def _sva_create_plot(parent: str, series: pd.Series, group_target: Optional[pd.Series] = None,
                     an_override: Optional[str] = None, numeric_plot_pref: str = "Box Plot",
                     cat_plot_pref: str = "100% Stacked (Proportions)",
                     u_funcs: Optional[Dict] = None) -> Optional[bytes]:
    plot_container_tag = parent
    clean_s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_s.empty:
        dpg.add_text("No valid data for plot.", parent=plot_container_tag, color=(255, 200, 0))
        return None

    s1_types_func = u_funcs['main_app_callbacks'].get('get_column_analysis_types', lambda: {})
    s1_types_dict = s1_types_func()
    s1_type_str = s1_types_dict.get(series.name, str(series.dtype))
    is_binary_numeric = "Numeric (Binary)" in s1_type_str
    treat_as_categorical = is_binary_numeric or an_override == "ForceCategoricalForBinaryNumeric" or \
                           any(k in s1_type_str for k in ["Categorical", "Text (", "Potentially Sensitive"]) or \
                           (clean_s.nunique() < 5)
    is_grouped = group_target is not None
    
    fig, ax = None, None 
    img_bytes = None
    try:
        plot_texture_func = u_funcs.get('plot_to_dpg_texture', utils.plot_to_dpg_texture)
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=90)
        
        if pd.api.types.is_numeric_dtype(series.dtype) and not treat_as_categorical:
            if is_grouped:
                plot_data = pd.DataFrame({'value': clean_s, 'group': group_target}).dropna()
                if numeric_plot_pref == "Box Plot":
                    # --- FIX START ---
                    # Assign x-variable 'group' to hue and disable legend to fix warning
                    sns.boxplot(x='group', y='value', data=plot_data, ax=ax, hue='group', palette="viridis", legend=False)
                    # --- FIX END ---
                    ax.set_title(f"Box Plot of {series.name} by {group_target.name}")
                    ax.set_xlabel(group_target.name)
                else:
                    stat_type = "density" if numeric_plot_pref == "Overlaid Density (KDE)" else "count"
                    kde_flag = True if numeric_plot_pref == "Overlaid Density (KDE)" else False
                    sns.histplot(data=plot_data, x='value', hue='group', fill=True, common_norm=False, stat=stat_type,
                                 kde=kde_flag, palette="viridis", alpha=0.5, bins=30)
                    ax.set_title(f"{stat_type.capitalize()} of {series.name} by {group_target.name}")
                ax.set_ylabel(series.name)
            else: 
                sns.histplot(clean_s, kde=True, ax=ax, bins=30)
                ax.set_title(f"Distribution of {series.name}")
                ax.set_xlabel(series.name)
                ax.set_ylabel("Density")
        else:
            # Cast to string to avoid potential CategoricalDtype issues with seaborn
            plot_series = series.astype(str) if isinstance(series.dtype, pd.CategoricalDtype) else series

            if is_grouped:
                if cat_plot_pref == "100% Stacked (Proportions)":
                    contingency_table = pd.crosstab(plot_series, group_target)
                    proportions_df = contingency_table.div(contingency_table.sum(axis=1), axis=0)
                    proportions_df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', alpha=0.75, width=0.8)
                    ax.set_title(f"Proportions of {group_target.name} within each {series.name}")
                    ax.set_ylabel("Proportion")
                    ax.legend(title=group_target.name, bbox_to_anchor=(1.02, 1), loc='upper left')
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
                else: 
                    sns.countplot(x=plot_series, hue=group_target, ax=ax, palette="viridis", order=plot_series.value_counts().index[:15])
                    ax.set_title(f"Counts of {series.name} by {group_target.name}")
                    ax.set_ylabel("Count")
            else: 
                # --- FIX START ---
                # Assign y-variable 'series' to hue and disable legend to fix warning
                sns.countplot(y=plot_series, ax=ax, order=plot_series.value_counts().index[:15], hue=plot_series, palette="viridis", legend=False)
                # --- FIX END ---
                ax.set_title(f"Frequency of {series.name} (Top 15)")
                ax.set_ylabel(series.name)
                ax.set_xlabel("Count")

        ax.grid(axis='y', linestyle='--', alpha=0.6)
        if ax.get_xticklabels(): plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()

        tex_tag, w, h, img_bytes = plot_texture_func(fig)
        
        if tex_tag and w > 0:
            container_w = dpg.get_item_width(plot_container_tag) or 600
            display_w = min(w, container_w - 20)
            display_h = int(h * (display_w / w)) if w > 0 else h
            dpg.add_image(tex_tag, parent=plot_container_tag, width=display_w, height=display_h)
        else:
            dpg.add_text("Failed to render plot.", parent=plot_container_tag, color=(255, 100, 0))
    
    except Exception as e:
        dpg.add_text(f"Error creating plot:\n{e}", parent=plot_container_tag, color=(255, 0, 0))
        print(f"Plotting Error for '{series.name}':\n{traceback.format_exc()}")
    finally:
        if fig: plt.close(fig)
    
    return img_bytes

def _sva_run_analysis(callbacks: dict):
    global _sva_results_cache
    
    _sva_results_cache.clear()
    if dpg.does_item_exist(TAG_SVA_EXPORT_BUTTON):
        dpg.configure_item(TAG_SVA_EXPORT_BUTTON, show=False)

    df = callbacks['get_current_df']()
    u_funcs = {**_sva_util_funcs, 'main_app_callbacks': callbacks} 
    target = callbacks['get_selected_target_variable']()

    if not utils.show_dpg_progress_modal("Processing SVA", "Analyzing variables...", modal_tag=TAG_SVA_MODULE_PROGRESS_MODAL, text_tag=TAG_SVA_MODULE_PROGRESS_TEXT): return

    if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW): dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
    else: utils.hide_dpg_progress_modal(TAG_SVA_MODULE_PROGRESS_MODAL); utils.show_dpg_alert_modal("UI Error", "SVA result area missing.", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT); return
    
    if df is None: dpg.add_text("Load data for SVA.", parent=TAG_SVA_RESULTS_CHILD_WINDOW); utils.hide_dpg_progress_modal(TAG_SVA_MODULE_PROGRESS_MODAL); return

    strength = dpg.get_value(TAG_SVA_FILTER_STRENGTH_RADIO)
    grp_flag = dpg.get_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX)
    numeric_plot_pref = dpg.get_value(TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO) if grp_flag else "Box Plot"
    cat_plot_pref = dpg.get_value(TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO) if grp_flag else "100% Stacked (Proportions)"
    target_s_grp = _sva_validate_grouping(df, target, callbacks) if grp_flag else None
    if target_s_grp is None and grp_flag: grp_flag = False
        
    dpg.set_value(TAG_SVA_MODULE_PROGRESS_TEXT, "SVA: Filtering variables..."); dpg.split_frame()
    filt_cols, act_filter = _sva_get_filtered_variables(df, strength, callbacks, target)
    
    if not filt_cols: dpg.add_text(f"No vars for filter: '{act_filter}'", parent=TAG_SVA_RESULTS_CHILD_WINDOW); utils.hide_dpg_progress_modal(TAG_SVA_MODULE_PROGRESS_MODAL); return
        
    total_vars = len(filt_cols)
    for i, col_name in enumerate(filt_cols):
        if not dpg.is_dearpygui_running(): break
        dpg.set_value(TAG_SVA_MODULE_PROGRESS_TEXT, f"SVA: Analyzing {col_name} ({i+1}/{total_vars})"); dpg.split_frame()
        _sva_create_section(i, col_name, df, target_s_grp, numeric_plot_pref, cat_plot_pref, u_funcs, act_filter, _sva_results_cache)

    utils.hide_dpg_progress_modal(TAG_SVA_MODULE_PROGRESS_MODAL)

    if _sva_results_cache:
        if dpg.does_item_exist(TAG_SVA_EXPORT_BUTTON):
            dpg.configure_item(TAG_SVA_EXPORT_BUTTON, show=True)


def _sva_validate_grouping(df: pd.DataFrame, target_var: str, callbacks: dict) -> Optional[pd.Series]:
    if not target_var or target_var not in df.columns: 
        utils.show_dpg_alert_modal("Grouping Info", "Target invalid. Grouping disabled.", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
        return None
    unique_n = df[target_var].nunique(dropna=False)
    if not (2 <= unique_n <= 7):
        utils.show_dpg_alert_modal("Grouping Warning", f"Target '{target_var}' has {unique_n} unique values (need 2-7).", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
        if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX): dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
        _sva_group_by_target_cb(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False, callbacks)
        return None
    return df[target_var]

def _sva_create_section(idx: int, col_name: str, df: pd.DataFrame, target_s: Optional[pd.Series], 
                        numeric_plot_pref: str, cat_plot_pref: str, u_funcs: dict, 
                        filt_applied: str, results_cache: dict):
    
    sec_tag = f"{TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX}{''.join(filter(str.isalnum, col_name))}_{idx}"
    s1_types_dict = u_funcs['main_app_callbacks'].get('get_column_analysis_types', lambda: {})()
    col_type = s1_types_dict.get(col_name, str(df[col_name].dtype))
    an_override = "ForceCategoricalForBinaryNumeric" if filt_applied == "None (All variables)" and "Numeric (Binary)" in col_type else None
    
    with dpg.group(tag=sec_tag, parent=TAG_SVA_RESULTS_CHILD_WINDOW):
        res_w = (dpg.get_item_width(TAG_SVA_RESULTS_CHILD_WINDOW) or 900) 
        head_wrap = res_w - 30 if res_w > 50 else 500
        dpg.add_text(f"Var: {u_funcs['format_text_for_display'](col_name, 60)} ({idx+1})", color=(255,255,0), wrap=head_wrap)
        dpg.add_text(f"Type: {col_type} (Dtype: {str(df[col_name].dtype)})", wrap=head_wrap)
        if an_override: dpg.add_text("Display: Treated as Categorical", color=(200,200,0), wrap=head_wrap)
        dpg.add_spacer(height=5)
        
        basic_stats_df, norm_df, rel_df, plot_bytes = None, None, None, None
        
        with dpg.group(horizontal=True):
            item_w = max(200, int(res_w * 0.28))
            with dpg.group(width=item_w): 
                basic_stats_df = _sva_create_basic_stats_table(dpg.last_item(), df[col_name], u_funcs, an_override)
            dpg.add_spacer(width=10)
            with dpg.group(width=item_w): 
                norm_df, rel_df = _sva_create_advanced_relations_table(dpg.last_item(), df[col_name], df, u_funcs, item_w)
            dpg.add_spacer(width=10)
            with dpg.group(): 
                plot_bytes = _sva_create_plot(dpg.last_item(), df[col_name], target_s, an_override, numeric_plot_pref, cat_plot_pref, u_funcs)

        results_cache[col_name] = {
            'basic_stats_df': basic_stats_df, 'normality_df': norm_df,
            'related_vars_df': rel_df, 'plot_bytes': plot_bytes
        }

        dpg.add_separator(); dpg.add_spacer(height=10)


def _sva_group_by_target_cb(sender, app_data, user_data):
    main_callbacks = user_data
    if app_data:
        target = main_callbacks['get_selected_target_variable']()
        if not target:
            utils.show_dpg_alert_modal("Target Not Selected", 
                                       "To use the grouping feature, please select a target variable from the top panel first.", 
                                       modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
            dpg.set_value(sender, False); return
    dpg.configure_item(TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO, show=app_data)
    dpg.configure_item(TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO + "_label", show=app_data)
    dpg.configure_item(TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO, show=app_data)
    dpg.configure_item(TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO + "_label", show=app_data)
    if not app_data:
        dpg.set_value(TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO, "Box Plot")
        dpg.set_value(TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO, "100% Stacked (Proportions)")

def _show_sva_export_dialog():
    if not _sva_results_cache:
        utils.show_dpg_alert_modal("No Results", "No SVA results available to export. Please run analysis first.",
                                   modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
        return

    try:
        root = tk.Tk()
        root.withdraw() 
        file_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML Report", "*.html"), ("Excel Data", "*.xlsx")],
            title="Save SVA Results"
        )
        root.destroy()

        if file_path:
            if file_path.endswith('.xlsx'):
                _export_sva_to_excel(file_path)
            else:
                _export_sva_to_html(file_path)
    except Exception as e:
        print(f"File dialog error: {e}")
        traceback.print_exc()
        utils.show_dpg_alert_modal("Export Error", f"An error occurred while trying to save the file:\n{e}",
                                   modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)

def _export_sva_to_excel(file_path: str):
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for var_name, data in _sva_results_cache.items():
                sanitized_name = "".join(c for c in var_name if c.isalnum() or c in (' ', '_')).rstrip()[:30]
                
                if data['basic_stats_df'] is not None and not data['basic_stats_df'].empty:
                    data['basic_stats_df'].to_excel(writer, sheet_name=sanitized_name, startrow=0, index=False)
                
                start_row_adv = len(data['basic_stats_df']) + 3 if data['basic_stats_df'] is not None else 2
                
                if data['normality_df'] is not None and not data['normality_df'].empty:
                    data['normality_df'].to_excel(writer, sheet_name=sanitized_name, startrow=start_row_adv, index=False)

                start_row_rel = start_row_adv + (len(data['normality_df']) + 3 if data['normality_df'] is not None else 2)
                
                if data['related_vars_df'] is not None and not data['related_vars_df'].empty:
                    data['related_vars_df'].to_excel(writer, sheet_name=sanitized_name, startrow=start_row_rel, index=False)

        utils.show_dpg_alert_modal("Export Successful", f"SVA results exported to:\n{file_path}",
                                   modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
    except Exception as e:
        traceback.print_exc()
        utils.show_dpg_alert_modal("Excel Export Error", f"Failed to export to Excel:\n{e}",
                                   modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)

def _export_sva_to_html(file_path: str):
    try:
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Single Variable Analysis Report</title>
            <style>
                body { font-family: sans-serif; margin: 2em; } h1, h2 { color: #333; }
                .variable-section { border: 1px solid #ccc; border-radius: 8px; padding: 1em; margin-bottom: 2em; overflow: auto; }
                .flex-container { display: flex; flex-wrap: wrap; gap: 2em; align-items: flex-start; }
                .stats-container { flex: 1; min-width: 300px; }
                .plot-container { flex: 2; min-width: 400px; text-align: center; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 1em; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; border: 1px solid #eee; }
            </style>
        </head>
        <body><h1>Single Variable Analysis Report</h1>
        """

        for var_name, data in _sva_results_cache.items():
            html += f"<div class='variable-section'><h2>Variable: {var_name}</h2>"
            html += "<div class='flex-container'>"
            
            html += "<div class='stats-container'>"
            if data['basic_stats_df'] is not None:
                html += "<h3>Basic Statistics</h3>" + data['basic_stats_df'].to_html(index=False, classes='stats-table')
            if data['normality_df'] is not None:
                html += "<h3>Normality Test</h3>" + data['normality_df'].to_html(index=False, classes='stats-table')
            if data['related_vars_df'] is not None:
                html += "<h3>Top Related Variables</h3>" + data['related_vars_df'].to_html(index=False, classes='stats-table')
            html += "</div>"
            
            html += "<div class='plot-container'>"
            if data['plot_bytes']:
                encoded_img = base64.b64encode(data['plot_bytes']).decode('utf-8')
                html += f"<h3>Distribution Plot</h3><img src='data:image/png;base64,{encoded_img}'>"
            else:
                html += "<p>No plot generated for this variable.</p>"
            html += "</div></div></div>"

        html += "</body></html>"

        with open(file_path, 'w', encoding='utf-8') as f: f.write(html)
        utils.show_dpg_alert_modal("Export Successful", f"SVA report exported to:\n{file_path}",
                                   modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
    except Exception as e:
        traceback.print_exc()
        utils.show_dpg_alert_modal("HTML Export Error", f"Failed to export to HTML:\n{e}",
                                   modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)


def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _sva_main_app_callbacks, _sva_util_funcs
    _sva_main_app_callbacks = main_callbacks
    _sva_util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    _sva_setup_axis_themes() 

    with dpg.group(tag=TAG_SVA_STEP_GROUP, parent=parent_container_tag):
        dpg.add_text(f"--- {step_name} ---"); dpg.add_separator()
        with dpg.group(horizontal=True):
            with dpg.group(width=280):
                dpg.add_text("Variable Filter")
                dpg.add_radio_button(items=["Strong (Top 5-10 relevant)", "Medium (Top 11-20 relevant)", 
                                          "Weak (Exclude obvious non-analytical)", "None (All variables)"],
                                     tag=TAG_SVA_FILTER_STRENGTH_RADIO, default_value="Weak (Exclude obvious non-analytical)")
                dpg.add_spacer(height=5)
                dpg.add_text("Filter Info:", wrap=270, color=(200,200,200))
                dpg.add_text("- Strong/Medium: Vs Target", wrap=270, color=(200,200,200))
                dpg.add_text("- Weak: Excl. single-val & bin-num", wrap=270, color=(200,200,200))
            dpg.add_spacer(width=10)
            with dpg.group():
                dpg.add_text("Grouping & Plot Option")
                dpg.add_checkbox(label="Group by Target (2-7 Unique)", tag=TAG_SVA_GROUP_BY_TARGET_CHECKBOX, default_value=False, user_data=main_callbacks, callback=_sva_group_by_target_cb)
                dpg.add_text(" - For Numeric Vars:", show=False, tag=TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO + "_label")
                dpg.add_radio_button(items=["Box Plot", "Overlaid Density (KDE)", "Overlaid Histogram"], tag=TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO, default_value="Box Plot", horizontal=True, show=False)
                dpg.add_text(" - For Categorical Vars:", show=False, tag=TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO + "_label")
                dpg.add_radio_button(items=["100% Stacked (Proportions)", "Side-by-side (Counts)"], tag=TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO, default_value="100% Stacked (Proportions)", horizontal=True, show=False)
                dpg.add_spacer(height=10)
                dpg.add_button(label="Run Single Variable Analysis", tag=TAG_SVA_RUN_BUTTON, callback=lambda: _sva_run_analysis(main_callbacks), width=-1, height=30)
        dpg.add_separator()
        dpg.add_button(label="Export Results...", tag=TAG_SVA_EXPORT_BUTTON, callback=lambda: _show_sva_export_dialog(), width=-1, show=False)
        dpg.add_spacer(height=5)
        with dpg.child_window(tag=TAG_SVA_RESULTS_CHILD_WINDOW, border=True):
            dpg.add_text("Select filter options and click 'Run Single Variable Analysis'.")
    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui(current_df: pd.DataFrame, main_callbacks: dict):
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_SVA_STEP_GROUP): return
    global _sva_main_app_callbacks, _sva_util_funcs, _sva_results_cache
    _sva_main_app_callbacks = main_callbacks
    _sva_util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    if current_df is None and dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
        dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
        dpg.add_text("Load data to perform Single Variable Analysis.", parent=TAG_SVA_RESULTS_CHILD_WINDOW)
        _sva_results_cache.clear()
        if dpg.does_item_exist(TAG_SVA_EXPORT_BUTTON):
            dpg.configure_item(TAG_SVA_EXPORT_BUTTON, show=False)


def reset_sva_ui_defaults():
    global _sva_results_cache
    _sva_results_cache.clear()
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO): dpg.set_value(TAG_SVA_FILTER_STRENGTH_RADIO, "Weak (Exclude obvious non-analytical)")
    if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX): dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
    if dpg.does_item_exist(TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO):
        dpg.set_value(TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO, "Box Plot")
        dpg.configure_item(TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO, show=False)
        dpg.configure_item(TAG_SVA_GROUPED_NUMERIC_PLOT_TYPE_RADIO + "_label", show=False)
    if dpg.does_item_exist(TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO):
        dpg.set_value(TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO, "100% Stacked (Proportions)")
        dpg.configure_item(TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO, show=False)
        dpg.configure_item(TAG_SVA_GROUPED_CAT_PLOT_TYPE_RADIO + "_label", show=False)
    if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
        dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
        dpg.add_text("Select filter options and click 'Run Single Variable Analysis'.", parent=TAG_SVA_RESULTS_CHILD_WINDOW)
    if dpg.does_item_exist(TAG_SVA_EXPORT_BUTTON):
        dpg.configure_item(TAG_SVA_EXPORT_BUTTON, show=False)