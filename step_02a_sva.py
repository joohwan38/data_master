# step_02a_sva.py (동적 UI 및 분석 로직 개선 버전)
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
import datetime
from openpyxl.drawing.image import Image

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')

# --- 태그 정의 ---
TAG_SVA_STEP_GROUP = "sva_step_group"
TAG_SVA_FILTER_STRENGTH_RADIO = "sva_filter_strength_radio"
TAG_SVA_RUN_BUTTON = "sva_run_button"
TAG_SVA_RESULTS_CHILD_WINDOW = "sva_results_child_window"
TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX = "sva_var_section_prefix_"
TAG_SVA_MODULE_PROGRESS_MODAL = "sva_module_specific_progress_modal"
TAG_SVA_MODULE_PROGRESS_TEXT = "sva_module_specific_progress_text"
TAG_SVA_MODULE_ALERT_MODAL = "sva_module_specific_alert_modal"
TAG_SVA_MODULE_ALERT_TEXT = "sva_module_specific_alert_text"
TAG_SVA_EXPORT_BUTTON = "sva_export_button"

# --- [신규] 동적 UI 관련 태그 ---
TAG_SVA_ANALYSIS_MODE_RADIO = "sva_analysis_mode_radio"
TAG_SVA_TARGET_STATUS_TEXT = "sva_target_status_text"
# 그룹 비교 (범주형 타겟) 옵션
TAG_SVA_GROUP_COMPARE_OPTIONS_GROUP = "sva_group_compare_options_group"
TAG_SVA_GC_NUMERIC_PLOT_RADIO = "sva_gc_numeric_plot_radio"
TAG_SVA_GC_CAT_PLOT_RADIO = "sva_gc_cat_plot_radio"
# 연속형 관계 분석 (연속형 타겟) 옵션
TAG_SVA_CONTINUOUS_RELATION_OPTIONS_GROUP = "sva_continuous_relation_options_group"
TAG_SVA_CR_NUMERIC_PLOT_RADIO = "sva_cr_numeric_plot_radio"
TAG_SVA_CR_CAT_PLOT_RADIO = "sva_cr_cat_plot_radio"


# --- 모듈 상태 변수 ---
_sva_main_app_callbacks: Dict[str, Any] = {}
_sva_util_funcs: Dict[str, Any] = {}
_sva_results_cache: Dict[str, Dict[str, Any]] = {}

def get_sva_settings_for_saving() -> Dict[str, Any]:
    settings = {}
    if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO):
        settings['filter_strength'] = dpg.get_value(TAG_SVA_FILTER_STRENGTH_RADIO)
    if dpg.does_item_exist(TAG_SVA_ANALYSIS_MODE_RADIO):
        settings['analysis_mode'] = dpg.get_value(TAG_SVA_ANALYSIS_MODE_RADIO)
    # 각 모드별 플롯 설정 저장
    if dpg.does_item_exist(TAG_SVA_GC_NUMERIC_PLOT_RADIO):
        settings['gc_numeric_plot'] = dpg.get_value(TAG_SVA_GC_NUMERIC_PLOT_RADIO)
    if dpg.does_item_exist(TAG_SVA_GC_CAT_PLOT_RADIO):
        settings['gc_cat_plot'] = dpg.get_value(TAG_SVA_GC_CAT_PLOT_RADIO)
    if dpg.does_item_exist(TAG_SVA_CR_CAT_PLOT_RADIO):
        settings['cr_cat_plot'] = dpg.get_value(TAG_SVA_CR_CAT_PLOT_RADIO)
    return settings


def apply_sva_settings_from_loaded(settings: Dict[str, Any], current_df: Optional[pd.DataFrame], main_callbacks: Dict[str, Any]):
    if not dpg.is_dearpygui_running(): return

    # 필터 강도 적용
    if 'filter_strength' in settings and dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO):
        dpg.set_value(TAG_SVA_FILTER_STRENGTH_RADIO, settings['filter_strength'])
    
    # UI 상태 업데이트 (가장 먼저 호출)
    _sva_update_target_analysis_ui()

    # 저장된 분석 모드 적용
    if 'analysis_mode' in settings and dpg.does_item_exist(TAG_SVA_ANALYSIS_MODE_RADIO):
        dpg.set_value(TAG_SVA_ANALYSIS_MODE_RADIO, settings['analysis_mode'])

    # 각 플롯 설정 적용
    if 'gc_numeric_plot' in settings and dpg.does_item_exist(TAG_SVA_GC_NUMERIC_PLOT_RADIO):
        dpg.set_value(TAG_SVA_GC_NUMERIC_PLOT_RADIO, settings['gc_numeric_plot'])
    if 'gc_cat_plot' in settings and dpg.does_item_exist(TAG_SVA_GC_CAT_PLOT_RADIO):
        dpg.set_value(TAG_SVA_GC_CAT_PLOT_RADIO, settings['gc_cat_plot'])
    if 'cr_cat_plot' in settings and dpg.does_item_exist(TAG_SVA_CR_CAT_PLOT_RADIO):
        dpg.set_value(TAG_SVA_CR_CAT_PLOT_RADIO, settings['cr_cat_plot'])
    
    if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
        dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
        dpg.add_text("Settings loaded. Click 'Run Single Variable Analysis' to update results.", parent=TAG_SVA_RESULTS_CHILD_WINDOW)
    
    if dpg.does_item_exist(TAG_SVA_EXPORT_BUTTON):
        dpg.configure_item(TAG_SVA_EXPORT_BUTTON, show=False)


def _sva_get_filtered_variables(df: pd.DataFrame, strength: str, callbacks: dict, target: str = None) -> Tuple[List[str], str]:
    if df is None or df.empty: return [], strength
    types = callbacks.get('get_column_analysis_types', lambda: {})()
    types = types if isinstance(types, dict) else {c: str(df[c].dtype) for c in df.columns}
    
    EXCLUSION_TYPES = ["Text (", "Potentially Sensitive", "분석에서 제외 (Exclude)"]
    cols = [c for c in df.columns if not any(k in types.get(c, str(df[c].dtype)) for k in EXCLUSION_TYPES)]

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

# --- [수정] 플롯 생성 함수: 새로운 분석 모드에 맞춰 로직 확장 ---
def _sva_create_plot(parent: str, 
                     series: pd.Series, 
                     group_target: Optional[pd.Series] = None,
                     analysis_mode: str = "사용 안 함",
                     plot_options: Dict[str, str] = None,
                     an_override: Optional[str] = None, 
                     u_funcs: Optional[Dict] = None) -> Optional[bytes]:
    plot_container_tag = parent
    plot_opts = plot_options or {}
    
    # 데이터 유효성 검사
    clean_s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean_s.empty:
        dpg.add_text("No valid data for plot.", parent=plot_container_tag, color=(255, 200, 0))
        return None

    # 변수 타입 결정
    s1_types_func = u_funcs['main_app_callbacks'].get('get_column_analysis_types', lambda: {})
    s1_types_dict = s1_types_func()
    s1_type_str = s1_types_dict.get(series.name, str(series.dtype))
    is_binary_numeric = "Numeric (Binary)" in s1_type_str
    treat_as_categorical = is_binary_numeric or an_override == "ForceCategoricalForBinaryNumeric" or \
                           any(k in s1_type_str for k in ["Categorical", "Text (", "Potentially Sensitive"]) or \
                           (clean_s.nunique() < 5)

    fig, ax = None, None 
    img_bytes = None
    try:
        plot_texture_func = u_funcs.get('plot_to_dpg_texture', utils.plot_to_dpg_texture)
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=90)
        
        # --- 분석 모드에 따른 분기 ---
        # 1. 그룹별 비교 (범주형 타겟)
        if analysis_mode == "그룹별 비교 (범주형 타겟)" and group_target is not None:
            plot_data = pd.DataFrame({'value': series, 'group': group_target}).dropna()
            
            if pd.api.types.is_numeric_dtype(series.dtype) and not treat_as_categorical:
                # X: 연속형, Y: 범주형
                numeric_plot_pref = plot_opts.get('gc_numeric_plot', 'Box Plot')
                if numeric_plot_pref == "Box Plot":
                    sns.boxplot(x='group', y='value', data=plot_data, ax=ax, hue='group', palette="viridis", legend=False)
                    ax.set_title(f"Box Plot of {series.name} by {group_target.name}")
                    ax.set_xlabel(group_target.name)
                else: # Overlaid Density/Histogram
                    stat_type = "density" if numeric_plot_pref == "Overlaid Density (KDE)" else "count"
                    kde_flag = True if numeric_plot_pref == "Overlaid Density (KDE)" else False
                    sns.histplot(data=plot_data, x='value', hue='group', fill=True, common_norm=False, stat=stat_type,
                                 kde=kde_flag, palette="viridis", alpha=0.5, bins=30)
                    ax.set_title(f"{stat_type.capitalize()} of {series.name} by {group_target.name}")
                ax.set_ylabel(series.name)

            else: # X: 범주형, Y: 범주형
                cat_plot_pref = plot_opts.get('gc_cat_plot', '100% Stacked (Proportions)')
                plot_series = series.astype(str)
                order = plot_series.value_counts().index[:15]
                
                if cat_plot_pref == "100% Stacked (Proportions)":
                    contingency_table = pd.crosstab(plot_series, group_target.astype(str))
                    proportions_df = contingency_table.div(contingency_table.sum(axis=1), axis=0)
                    proportions_df.loc[order].plot(kind='bar', stacked=True, ax=ax, colormap='viridis', alpha=0.75, width=0.8)
                    ax.set_title(f"Proportions of {group_target.name} in {series.name}")
                    ax.set_ylabel("Proportion")
                    ax.legend(title=group_target.name, bbox_to_anchor=(1.02, 1), loc='upper left')
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
                else: # Side-by-side (Counts)
                    sns.countplot(x=plot_series, hue=group_target, ax=ax, palette="viridis", order=order)
                    ax.set_title(f"Counts of {series.name} by {group_target.name}")
                    ax.set_ylabel("Count")

        # 2. 연속형 관계 분석 (연속형 타겟)
        elif analysis_mode == "연속형 관계 분석 (연속형 타겟)" and group_target is not None:
            plot_data = pd.DataFrame({'feature': series, 'target': group_target}).dropna()

            if pd.api.types.is_numeric_dtype(series.dtype) and not treat_as_categorical:
                # X: 연속형, Y: 연속형 -> 회귀선 포함 산점도
                sns.regplot(x='feature', y='target', data=plot_data, ax=ax, scatter_kws={'alpha': 0.4})
                ax.set_title(f"Relationship between {series.name} and {group_target.name}")
                ax.set_xlabel(series.name)
                ax.set_ylabel(group_target.name)
            else:
                # X: 범주형, Y: 연속형 -> 바이올린 플롯
                plot_series = series.astype(str)
                order = plot_series.value_counts().index[:15]
                sns.violinplot(x='feature', y='target', data=plot_data, ax=ax, palette="viridis", order=order)
                ax.set_title(f"Distribution of {group_target.name} by {series.name}")
                ax.set_xlabel(series.name)
                ax.set_ylabel(group_target.name)

        # 3. 사용 안 함 (기본 단일 변수 분석)
        else:
            if pd.api.types.is_numeric_dtype(series.dtype) and not treat_as_categorical:
                sns.histplot(clean_s, kde=True, ax=ax, bins=30)
                ax.set_title(f"Distribution of {series.name}")
                ax.set_xlabel(series.name)
                ax.set_ylabel("Density")
            else:
                plot_series = series.astype(str)
                order = plot_series.value_counts().index[:15]
                sns.countplot(y=plot_series, ax=ax, order=order, hue=plot_series, palette="viridis", legend=False)
                ax.set_title(f"Frequency of {series.name} (Top 15)")
                ax.set_ylabel(series.name)
                ax.set_xlabel("Count")

        # 공통 플롯 설정
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


# --- [수정] 분석 실행 함수: 단순화된 로직 ---
def _sva_run_analysis(callbacks: dict):
    global _sva_results_cache
    
    _sva_results_cache.clear()
    if dpg.does_item_exist(TAG_SVA_EXPORT_BUTTON):
        dpg.configure_item(TAG_SVA_EXPORT_BUTTON, show=False)

    df = callbacks['get_current_df']()
    u_funcs = {**_sva_util_funcs, 'main_app_callbacks': callbacks} 
    target_var = callbacks['get_selected_target_variable']()

    if not utils.show_dpg_progress_modal("Processing SVA", "Analyzing variables...", modal_tag=TAG_SVA_MODULE_PROGRESS_MODAL, text_tag=TAG_SVA_MODULE_PROGRESS_TEXT): return

    if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW): dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
    else: utils.hide_dpg_progress_modal(TAG_SVA_MODULE_PROGRESS_MODAL); utils.show_dpg_alert_modal("UI Error", "SVA result area missing.", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT); return
    
    if df is None: dpg.add_text("Load data for SVA.", parent=TAG_SVA_RESULTS_CHILD_WINDOW); utils.hide_dpg_progress_modal(TAG_SVA_MODULE_PROGRESS_MODAL); return

    strength = dpg.get_value(TAG_SVA_FILTER_STRENGTH_RADIO)
    analysis_mode = dpg.get_value(TAG_SVA_ANALYSIS_MODE_RADIO)
    
    # UI에서 현재 활성화된 플롯 옵션 값 가져오기
    plot_options = {
        'gc_numeric_plot': dpg.get_value(TAG_SVA_GC_NUMERIC_PLOT_RADIO),
        'gc_cat_plot': dpg.get_value(TAG_SVA_GC_CAT_PLOT_RADIO),
    }

    target_series = df[target_var] if target_var and analysis_mode != "사용 안 함" else None
        
    dpg.set_value(TAG_SVA_MODULE_PROGRESS_TEXT, "SVA: Filtering variables..."); dpg.split_frame()
    filt_cols, act_filter = _sva_get_filtered_variables(df, strength, callbacks, target_var)
    
    if not filt_cols: dpg.add_text(f"No vars for filter: '{act_filter}'", parent=TAG_SVA_RESULTS_CHILD_WINDOW); utils.hide_dpg_progress_modal(TAG_SVA_MODULE_PROGRESS_MODAL); return
        
    total_vars = len(filt_cols)
    for i, col_name in enumerate(filt_cols):
        if not dpg.is_dearpygui_running(): break
        if col_name == target_var: continue # 자기 자신은 분석에서 제외
        dpg.set_value(TAG_SVA_MODULE_PROGRESS_TEXT, f"SVA: Analyzing {col_name} ({i+1}/{total_vars})"); dpg.split_frame()
        _sva_create_section(i, col_name, df, target_series, analysis_mode, plot_options, u_funcs, act_filter, _sva_results_cache)

    utils.hide_dpg_progress_modal(TAG_SVA_MODULE_PROGRESS_MODAL)

    if _sva_results_cache:
        if dpg.does_item_exist(TAG_SVA_EXPORT_BUTTON):
            dpg.configure_item(TAG_SVA_EXPORT_BUTTON, show=True)


def _sva_create_section(idx: int, col_name: str, df: pd.DataFrame, 
                        target_s: Optional[pd.Series], 
                        analysis_mode: str, 
                        plot_options: dict, 
                        u_funcs: dict, 
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
                plot_bytes = _sva_create_plot(dpg.last_item(), df[col_name], target_s, analysis_mode, plot_options, an_override, u_funcs)

        results_cache[col_name] = {
            'basic_stats_df': basic_stats_df, 'normality_df': norm_df,
            'related_vars_df': rel_df, 'plot_bytes': plot_bytes
        }

        dpg.add_separator(); dpg.add_spacer(height=10)

def _sva_on_analysis_mode_change(sender, app_data, user_data):
    """사용자가 분석 모드 라디오 버튼을 직접 변경했을 때 호출되는 콜백"""
    if not dpg.is_dearpygui_running(): return

    mode = dpg.get_value(TAG_SVA_ANALYSIS_MODE_RADIO)
    gc_group = TAG_SVA_GROUP_COMPARE_OPTIONS_GROUP
    cr_group = TAG_SVA_CONTINUOUS_RELATION_OPTIONS_GROUP

    # 선택된 모드에 따라 옵션 그룹을 보여주거나 숨김
    dpg.configure_item(gc_group, show=(mode == "그룹별 비교 (범주형 타겟)"))
    dpg.configure_item(cr_group, show=(mode == "연속형 관계 분석 (연속형 타겟)"))


def _sva_update_target_analysis_ui():
    """메인 앱의 목표 변수 선택이 변경될 때마다 호출되어야 합니다."""
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_SVA_ANALYSIS_MODE_RADIO):
        return

    df = _sva_main_app_callbacks.get('get_current_df', lambda: None)()
    target_var = _sva_main_app_callbacks.get('get_selected_target_variable', lambda: None)()
    
    gc_group = TAG_SVA_GROUP_COMPARE_OPTIONS_GROUP
    cr_group = TAG_SVA_CONTINUOUS_RELATION_OPTIONS_GROUP
    
    # 초기화
    dpg.configure_item(gc_group, show=False)
    dpg.configure_item(cr_group, show=False)
    
    if df is None or not target_var:
        dpg.set_value(TAG_SVA_ANALYSIS_MODE_RADIO, "사용 안 함")
        dpg.set_value(TAG_SVA_TARGET_STATUS_TEXT, "분석할 목표 변수를 선택해주세요.")
        dpg.configure_item(TAG_SVA_ANALYSIS_MODE_RADIO, enabled=False)
        return

    dpg.configure_item(TAG_SVA_ANALYSIS_MODE_RADIO, enabled=True)
    target_series = df[target_var]
    nunique = target_series.nunique()
    
    # 1. 연속형 변수인지 확인
    # (고유값이 30개를 초과하는 수치형 변수를 연속형으로 간주)
    if pd.api.types.is_numeric_dtype(target_series.dtype) and nunique > 30:
        dpg.set_value(TAG_SVA_ANALYSIS_MODE_RADIO, "연속형 관계 분석 (연속형 타겟)")
        dpg.set_value(TAG_SVA_TARGET_STATUS_TEXT, f"타겟 '{target_var}' (연속형)에 대한 관계 분석이 활성화됩니다.")
        dpg.configure_item(cr_group, show=True)
    # 2. 그룹 분석에 적합한 범주형 변수인지 확인
    elif 2 <= nunique <= 30:
        dpg.set_value(TAG_SVA_ANALYSIS_MODE_RADIO, "그룹별 비교 (범주형 타겟)")
        dpg.set_value(TAG_SVA_TARGET_STATUS_TEXT, f"타겟 '{target_var}' (고유값 {nunique}개)에 대한 그룹 분석이 활성화됩니다.")
        dpg.configure_item(gc_group, show=True)
    # 3. 그 외 (분석에 부적합)
    else:
        dpg.set_value(TAG_SVA_ANALYSIS_MODE_RADIO, "사용 안 함")
        if nunique < 2:
            reason = f"고유값이 1개뿐입니다."
        else: # nunique > 30 이고 범주형
            reason = f"고유값이 너무 많습니다 ({nunique}개)."
        dpg.set_value(TAG_SVA_TARGET_STATUS_TEXT, f"타겟 '{target_var}'은(는) 분석에 부적합합니다: {reason}")
        _sva_on_analysis_mode_change(None, None, None)



def _show_sva_export_dialog():
    if not _sva_results_cache:
        utils.show_dpg_alert_modal("No Results", "No SVA results available to export.", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
        return
    
    current_date_str = datetime.datetime.now().strftime("%Y%m%d")
    default_filename = f"{current_date_str}_EDA_Report"
    
    # 파일 확장자 필터 추가
    with dpg.file_dialog(
        directory_selector=False, show=True, callback=_sva_export_callback, 
        tag="sva_export_file_dialog_id", default_filename=default_filename,
        width=700, height=400, modal=True
    ):
        dpg.add_file_extension(".xlsx", color=(0, 255, 0, 255))
        dpg.add_file_extension(".html", color=(0, 255, 255, 255))

def _sva_export_callback(sender, app_data):
    file_path = app_data.get('file_path_name')
    if file_path:
        if not (file_path.endswith('.xlsx') or file_path.endswith('.html')):
            file_path += '.xlsx' # 기본값
            
        if file_path.endswith('.xlsx'):
            _export_sva_to_excel(file_path)
        else:
            _export_sva_to_html(file_path)

def _export_sva_to_excel(file_path: str):
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for var_name, data in _sva_results_cache.items():
                sanitized_name = "".join(c for c in var_name if c.isalnum() or c in (' ', '_')).rstrip()[:30]
                
                start_row_next = 0
                if data['basic_stats_df'] is not None and not data['basic_stats_df'].empty:
                    data['basic_stats_df'].to_excel(writer, sheet_name=sanitized_name, startrow=start_row_next, index=False)
                    start_row_next += len(data['basic_stats_df']) + 2
                if data['normality_df'] is not None and not data['normality_df'].empty:
                    data['normality_df'].to_excel(writer, sheet_name=sanitized_name, startrow=start_row_next, index=False)
                    start_row_next += len(data['normality_df']) + 2
                if data['related_vars_df'] is not None and not data['related_vars_df'].empty:
                    data['related_vars_df'].to_excel(writer, sheet_name=sanitized_name, startrow=start_row_next, index=False)

                if data['plot_bytes']:
                    worksheet = writer.sheets[sanitized_name]
                    img = Image(io.BytesIO(data['plot_bytes']))
                    worksheet.add_image(img, 'D2')

        utils.show_dpg_alert_modal("Export Successful", f"SVA results exported to:\n{file_path}", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
    except Exception as e:
        traceback.print_exc()
        utils.show_dpg_alert_modal("Excel Export Error", f"Failed to export to Excel:\n{e}", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)

def _export_sva_to_html(file_path: str):
    # HTML 내보내기 로직은 변경 없음
    # ... (이전 코드와 동일) ...
    pass

# --- [수정] UI 생성 함수 ---
def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _sva_main_app_callbacks, _sva_util_funcs
    _sva_main_app_callbacks = main_callbacks
    _sva_util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()

    with dpg.group(tag=TAG_SVA_STEP_GROUP, parent=parent_container_tag):
        dpg.add_text(f"--- {step_name} ---"); dpg.add_separator()
        
        with dpg.group(horizontal=True):
            # 왼쪽: 변수 필터
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
            
            # 오른쪽: 목표 변수 활용 분석
            with dpg.group():
                dpg.add_text("--- 목표 변수 활용 분석 ---", color=(255, 255, 0))
                dpg.add_text("분석할 목표 변수를 선택해주세요.", tag=TAG_SVA_TARGET_STATUS_TEXT, wrap=400)
                dpg.add_radio_button(
                    items=["그룹별 비교 (범주형 타겟)", "연속형 관계 분석 (연속형 타겟)", "사용 안 함"],
                    tag=TAG_SVA_ANALYSIS_MODE_RADIO, default_value="사용 안 함", horizontal=True, enabled=False,
                    callback=_sva_on_analysis_mode_change  # --- ◀ [수정] 이 줄 추가
                )
                dpg.add_separator()

                # 그룹 비교 (범주형 타겟) 옵션
                with dpg.group(tag=TAG_SVA_GROUP_COMPARE_OPTIONS_GROUP, show=False):
                    dpg.add_text("그룹별 비교 플롯 옵션")
                    dpg.add_text(" - For Numeric Vars:")
                    dpg.add_radio_button(items=["Box Plot", "Overlaid Density (KDE)"], tag=TAG_SVA_GC_NUMERIC_PLOT_RADIO, default_value="Box Plot", horizontal=True)
                    dpg.add_text(" - For Categorical Vars:")
                    dpg.add_radio_button(items=["100% Stacked (Proportions)", "Side-by-side (Counts)"], tag=TAG_SVA_GC_CAT_PLOT_RADIO, default_value="100% Stacked (Proportions)", horizontal=True)
                
                # 연속형 관계 분석 (연속형 타겟) 옵션
                with dpg.group(tag=TAG_SVA_CONTINUOUS_RELATION_OPTIONS_GROUP, show=False):
                    dpg.add_text("연속형 관계 분석 플롯 옵션")
                    dpg.add_text(" - For Numeric Vars: 회귀선 포함 산점도 (고정)")
                    dpg.add_text(" - For Categorical Vars: Violin Plot (고정)")

        dpg.add_separator()
        dpg.add_button(label="Run Single Variable Analysis", tag=TAG_SVA_RUN_BUTTON, callback=lambda: _sva_run_analysis(main_callbacks), width=-1, height=30)
        dpg.add_separator()
        dpg.add_button(label="Export Results...", tag=TAG_SVA_EXPORT_BUTTON, callback=_show_sva_export_dialog, width=-1, show=False)
        dpg.add_spacer(height=5)
        with dpg.child_window(tag=TAG_SVA_RESULTS_CHILD_WINDOW, border=True):
            dpg.add_text("옵션을 선택하고 'Run Single Variable Analysis'를 클릭하세요.")
            
    # 모듈의 UI 업데이트 함수를 메인 콜백에 등록
    main_callbacks['register_module_updater'](step_name, update_ui)
    # [중요] SVA 모듈의 동적 UI 업데이트 함수를 별도로 등록하거나 반환
    main_callbacks['register_sva_ui_updater'] = _sva_update_target_analysis_ui


def update_ui(current_df: pd.DataFrame, main_callbacks: dict):
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_SVA_STEP_GROUP): return
    global _sva_main_app_callbacks, _sva_util_funcs, _sva_results_cache
    _sva_main_app_callbacks = main_callbacks
    _sva_util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    
    # 데이터 로드/언로드 시 UI 상태 업데이트
    _sva_update_target_analysis_ui()
    
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
    if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO): 
        dpg.set_value(TAG_SVA_FILTER_STRENGTH_RADIO, "Weak (Exclude obvious non-analytical)")
    
    # UI 리셋 시 동적 업데이트 함수 호출
    _sva_update_target_analysis_ui()
    
    # 플롯 옵션 기본값으로 리셋
    if dpg.does_item_exist(TAG_SVA_GC_NUMERIC_PLOT_RADIO): 
        dpg.set_value(TAG_SVA_GC_NUMERIC_PLOT_RADIO, "Box Plot")
    if dpg.does_item_exist(TAG_SVA_GC_CAT_PLOT_RADIO): 
        dpg.set_value(TAG_SVA_GC_CAT_PLOT_RADIO, "100% Stacked (Proportions)")

    if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
        dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
        dpg.add_text("Select filter options and click 'Run Single Variable Analysis'.", parent=TAG_SVA_RESULTS_CHILD_WINDOW)
    if dpg.does_item_exist(TAG_SVA_EXPORT_BUTTON):
        dpg.configure_item(TAG_SVA_EXPORT_BUTTON, show=False)