# step_02a_sva.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
import utils # utils 모듈 임포트

warnings.filterwarnings('ignore', category=RuntimeWarning)

TAG_SVA_STEP_GROUP = "sva_step_group"
TAG_SVA_FILTER_STRENGTH_RADIO = "sva_filter_strength_radio"
TAG_SVA_GROUP_BY_TARGET_CHECKBOX = "sva_group_by_target_checkbox"
TAG_SVA_RUN_BUTTON = "sva_run_button"
TAG_SVA_RESULTS_CHILD_WINDOW = "sva_results_child_window"
TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX = "sva_var_section_prefix_"
# SVA 모듈 전용 프로그레스/알림 태그 정의 (utils의 기본 태그와 다르게 하여 충돌 방지)
TAG_SVA_MODULE_PROGRESS_MODAL = "sva_module_specific_progress_modal"
TAG_SVA_MODULE_PROGRESS_TEXT = "sva_module_specific_progress_text"
TAG_SVA_MODULE_ALERT_MODAL = "sva_module_specific_alert_modal"
TAG_SVA_MODULE_ALERT_TEXT = "sva_module_specific_alert_text"

TAG_SVA_GROUPED_PLOT_TYPE_RADIO = "sva_grouped_plot_type_radio"
X_AXIS_ROTATED_TICKS_THEME_TAG = "sva_x_axis_rotated_theme" # SVA용 X축 테마

_sva_main_app_callbacks: Dict[str, Any] = {}
_sva_util_funcs: Dict[str, Any] = {}

def get_sva_settings_for_saving() -> Dict[str, Any]:
    """SVA 탭의 현재 설정을 저장용 딕셔너리로 반환합니다."""
    settings = {}
    if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO):
        settings['filter_strength'] = dpg.get_value(TAG_SVA_FILTER_STRENGTH_RADIO)
    if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
        settings['group_by_target'] = dpg.get_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX)
    if dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
        settings['grouped_plot_type'] = dpg.get_value(TAG_SVA_GROUPED_PLOT_TYPE_RADIO)
    # 추가적으로 저장할 SVA 관련 UI 상태가 있다면 여기에 추가
    return settings

def apply_sva_settings_from_loaded(settings: Dict[str, Any], current_df: Optional[pd.DataFrame], main_callbacks: Dict[str, Any]):
    """저장된 설정을 SVA UI에 적용합니다."""
    if not dpg.is_dearpygui_running():
        return

    if 'filter_strength' in settings and dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO):
        dpg.set_value(TAG_SVA_FILTER_STRENGTH_RADIO, settings['filter_strength'])
    
    group_by_target_val = settings.get('group_by_target', False)
    if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
        dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, group_by_target_val)
    
    if dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
        if 'grouped_plot_type' in settings:
            dpg.set_value(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, settings['grouped_plot_type'])
        dpg.configure_item(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, show=group_by_target_val)

    # 설정 로드 후 SVA 결과창을 어떻게 할지 결정 필요
    # 예: 이전 결과가 있었다면 "설정이 로드되었습니다. 분석을 다시 실행하세요." 같은 메시지 표시
    if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
        dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
        dpg.add_text("Settings loaded. Click 'Run Single Variable Analysis' to update results.", parent=TAG_SVA_RESULTS_CHILD_WINDOW)
    
    # 필요하다면 여기서 update_ui를 호출하여 UI의 다른 부분을 갱신할 수 있습니다.
    # update_ui(current_df, main_callbacks) # current_df가 필요할 수 있음


def _sva_setup_axis_themes():
    if not dpg.does_item_exist(X_AXIS_ROTATED_TICKS_THEME_TAG):
        with dpg.theme(tag=X_AXIS_ROTATED_TICKS_THEME_TAG):
            with dpg.theme_component(dpg.mvXAxis): pass # 실제 스타일은 사용자가 이 태그에 직접 정의

def _sva_get_filtered_variables(df: pd.DataFrame, strength: str, callbacks: dict, target: str = None) -> Tuple[List[str], str]:
    if df is None or df.empty: return [], strength
    types = callbacks.get('get_column_analysis_types', lambda: {})()
    types = types if isinstance(types, dict) else {c: str(df[c].dtype) for c in df.columns}
    
    cols = [c for c in df.columns if not any(k in types.get(c, str(df[c].dtype)) for k in ["Text (", "Potentially Sensitive"])]
    if strength == "None (All variables)": return cols, strength
    
    weak_cols = []
    for c_name in cols:
        s, c_type = df[c_name], types.get(c_name, str(df[c_name].dtype))
        if s.nunique(dropna=False) <= 1: continue
        is_bin_num = "Numeric (Binary)" in c_type or \
                     ("Numeric" in c_type and len(s.dropna().unique())==2 and set(s.dropna().unique()).issubset({0,1,0.0,1.0}))
        if is_bin_num: continue
        weak_cols.append(c_name)
        
    if strength == "Weak (Exclude obvious non-analytical)": return weak_cols, strength
    if not target: utils.show_dpg_alert_modal("Target Required", f"Filter '{strength}': Target var needed.", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT); return [], strength
        
    num_cols_rank = [c for c in weak_cols if "Numeric" in types.get(c,"") and "Binary" not in types.get(c,"")]
    if not num_cols_rank: utils.show_dpg_alert_modal("Filter Error", f"Filter '{strength}': No numeric vars.", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT); return [], strength
        
    target_type = callbacks['get_selected_target_variable_type']()
    relevant_scores = _sva_calculate_relevance(df, target, target_type, num_cols_rank)
    if not relevant_scores: utils.show_dpg_alert_modal("Filter Error", f"Filter '{strength}': No relevant vars.", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT); return [], strength
    return relevant_scores[:10 if strength == "Strong (Top 5-10 relevant)" else 20], strength

def _sva_calculate_relevance(df: pd.DataFrame, target_var: str, target_type: str, cols: List[str]) -> List[str]:
    scores = []
    for col in cols:
        if col == target_var: continue
        score = 0.0
        try:
            temp_df = pd.concat([df[col], df[target_var]], axis=1).dropna()
            if len(temp_df) < 20: continue
            if target_type == "Categorical": 
                groups = [temp_df[col][temp_df[target_var] == cat] for cat in temp_df[target_var].unique()]
                valid_groups = [g for g in groups if len(g) >= 2]
                if len(valid_groups) >= 2:
                    f_val, _ = stats.f_oneway(*valid_groups)
                    score = f_val if pd.notna(f_val) and np.isfinite(f_val) else 0.0
            elif target_type == "Continuous" and pd.api.types.is_numeric_dtype(temp_df[col].dtype):
                corr_val = temp_df[col].corr(temp_df[target_var])
                score = abs(corr_val) if pd.notna(corr_val) and np.isfinite(corr_val) else 0.0
        except Exception: score = 0.0
        if score > 1e-3 : scores.append((col, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in scores]

def _sva_create_basic_stats_table(parent: str, series: pd.Series, u_funcs: dict, analysis_override: str = None):
    dpg.add_text("Basic Statistics", parent=parent)
    stats_data = [{'S': 'Count', 'V': str(series.count())}, {'S': 'Missing', 'V': str(series.isnull().sum())},
                  {'S': 'Missing %', 'V': f"{series.isnull().mean()*100:.2f}%"},
                  {'S': 'Unique (Actual)', 'V': str(series.nunique(dropna=False))},
                  {'S': 'Unique (Valid)', 'V': str(series.nunique())}]
    s1_types_func = u_funcs['main_app_callbacks'].get('get_column_analysis_types', lambda: {})
    s1_types_dict = s1_types_func() # 함수를 호출하여 딕셔너리를 얻음
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
    if stats_data:
        df_stats = pd.DataFrame(stats_data).rename(columns={'S':'Statistic', 'V':'Value'})
        tbl_tag, tbl_h = dpg.generate_uuid(), min(280, len(df_stats) * 22 + 40)
        with dpg.table(header_row=True, tag=tbl_tag, parent=parent, policy=dpg.mvTable_SizingStretchProp, height=int(tbl_h), scrollY=True,
                      borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True, resizable=True):
            u_funcs['create_table_with_data'](tbl_tag, df_stats, parent_df_for_widths=df_stats)

def _sva_create_advanced_relations_table(parent: str, series: pd.Series, full_df: pd.DataFrame, u_funcs: dict, col_w: int):
    norm_data = []
    s1_types_func = u_funcs['main_app_callbacks'].get('get_column_analysis_types', lambda: {})
    s1_types_dict = s1_types_func() # 함수를 호출하여 컬럼 분석 타입 딕셔너리를 얻음
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
                c_v = utils.calculate_cramers_v(series, df[col]) # utils의 함수 사용
                if pd.notna(c_v) and c_v > 0.01: corrs.append((col, c_v, "Cramér's V"))
            except: pass
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for name, val, metric in corrs[:top_n]: results.append({'Variable': name, 'Metric': metric, 'Value': f"{val:.3f}"})
    return results if results else [{'Info': 'No significant relations.'}]

def _sva_create_plot(parent: str, series: pd.Series, group_target: Optional[pd.Series] = None,
                     an_override: Optional[str] = None, grp_plot_pref: str = "KDE",
                     u_funcs: Optional[Dict] = None):

    plot_lbl = f"Dist: {series.name}"
    if group_target is not None and series.name != group_target.name:
        plot_lbl += f" (vs {group_target.name})"
    
    plot_container_tag = parent # 플롯과 메시지가 추가될 부모 그룹

    # 1. 초기 데이터 정제
    clean_s = series.replace([np.inf, -np.inf], np.nan).dropna()

    if clean_s.empty:
        if dpg.does_item_exist(plot_container_tag):
            dpg.add_text("No valid data points for plot.", parent=plot_container_tag, color=(255, 200, 0))
        return

    # 2. 시리즈 타입 및 그룹화 정보 결정
    s1_type_str = str(series.dtype) # 기본값
    if u_funcs and 'main_app_callbacks' in u_funcs:
        s1_types_func = u_funcs['main_app_callbacks'].get('get_column_analysis_types', lambda: {})
        s1_types_dict = s1_types_func()
        s1_type_str = s1_types_dict.get(series.name, str(series.dtype))

    is_binary_numeric = "Numeric (Binary)" in s1_type_str
    treat_as_categorical = is_binary_numeric or \
                           an_override == "ForceCategoricalForBinaryNumeric" or \
                           any(k in s1_type_str for k in ["Categorical", "Text (", "Potentially Sensitive"]) or \
                           (clean_s.nunique() < 5)
    is_grouped = group_target is not None

    # 3. 플롯 틀 생성
    plot_item_tag, x_axis_tag, y_axis_tag, legend_item_tag = utils.create_dpg_plot_scaffold(
        plot_container_tag, plot_lbl, series.name, "Density/Freq",
        h=290, legend=is_grouped
    )
    if legend_item_tag and not is_grouped:
        dpg.hide_item(legend_item_tag)
    if dpg.does_item_exist(X_AXIS_ROTATED_TICKS_THEME_TAG):
        dpg.bind_item_theme(x_axis_tag, X_AXIS_ROTATED_TICKS_THEME_TAG)

    # --- 4. 플로팅 로직 ---
    # 수치형 데이터
    if pd.api.types.is_numeric_dtype(series.dtype) and not treat_as_categorical:
        num_unique_clean = clean_s.nunique()

        if is_grouped: # 그룹화된 수치형
            unique_groups = sorted(group_target.dropna().unique())[:7]
            for group_val in unique_groups:
                group_data_s = clean_s[group_target == group_val].dropna() # clean_s 에서 그룹 데이터 추출
                if len(group_data_s) < 2: continue

                grp_lbl = f"{str(group_val)[:15]}"
                # "어제 코드"처럼, group_data_s가 이미 숫자형이라고 가정하고 바로 tolist()
                # 단, KDE는 여전히 float을 명시적으로 요구할 수 있음
                final_group_data_list = group_data_s.tolist() 
                
                if not final_group_data_list: continue

                try:
                    if grp_plot_pref == "KDE" and pd.Series(final_group_data_list).nunique() >= 2:
                        # KDE는 float 타입 데이터 필요
                        kde_input_data = pd.Series(final_group_data_list).astype(float).replace([np.inf,-np.inf],np.nan).dropna()
                        if len(kde_input_data) >= 2 and kde_input_data.nunique() >= 2:
                            kde = stats.gaussian_kde(kde_input_data)
                            min_val, max_val = kde_input_data.min(), kde_input_data.max()
                            if min_val < max_val:
                                x_vals = np.linspace(min_val, max_val, 100)
                                utils.add_dpg_line_series(y_axis_tag, x_vals.tolist(), kde(x_vals).tolist(), grp_lbl)
                    elif grp_plot_pref == "Histogram":
                        # "어제 코드"처럼 density=True, bins=-1 사용
                        utils.add_dpg_histogram_series(y_axis_tag, final_group_data_list, grp_lbl, density=True, bins=-1)
                except SystemError as se_grp:
                    print(f"!!! SVA PLOT (GROUPED) SystemError for group '{grp_lbl}' of '{series.name}': {se_grp} !!!")
                except Exception as e_grp:
                    print(f"SVA PLOT (GROUPED) Exception for group '{grp_lbl}' of '{series.name}': {e_grp}")
        
        else: # 그룹화되지 않은 수치형
            if num_unique_clean == 1:
                the_value = clean_s.iloc[0]
                count = len(clean_s)
                msg = f"All {count} data points are: {the_value:.3f}"
                print(f"SVA PLOT INFO ({series.name}): {msg}")
                if dpg.does_item_exist(plot_container_tag):
                    dpg.add_text(msg, parent=plot_container_tag, color=(200, 200, 0))
            
            elif num_unique_clean > 1:
                # "어제 코드"처럼, clean_s가 이미 숫자형이라고 가정하고 바로 tolist()
                final_data_list = clean_s.tolist()

                if not final_data_list: # 혹시 모를 경우 대비
                    if dpg.does_item_exist(plot_container_tag):
                        dpg.add_text(f"No data for histogram for {series.name}.", parent=plot_container_tag, color=(255,150,0))
                    return

                # 히스토그램 시도 ("어제 코드"처럼 density=True, bins=-1)
                try:
                    print(f"SVA PLOT Attempting histogram for '{series.name}' (len: {len(final_data_list)}) with density=True, bins=-1")
                    utils.add_dpg_histogram_series(y_axis_tag, final_data_list, "Histogram", density=True, bins=-1)
                except SystemError as se_hist:
                    print(f"!!! SVA PLOT: SYSTEM ERROR during add_histogram_series for '{series.name}' !!!")
                    print(f"    Parameters: density=True, bins=-1")
                    print(f"    Data len: {len(final_data_list)}, Min: {min(final_data_list) if final_data_list else 'N/A'}, Max: {max(final_data_list) if final_data_list else 'N/A'}")
                    print(f"    Exception: {se_hist}")
                    if dpg.does_item_exist(plot_container_tag): # UI 오류 메시지는 최소화
                         dpg.add_text(f"Failed to draw histogram for {series.name}.", parent=plot_container_tag, color=(255,0,0))
                except Exception as e_hist_other:
                    print(f"SVA PLOT: Other Exception for '{series.name}' (Histogram): {e_hist_other}")
                    if dpg.does_item_exist(plot_container_tag):
                         dpg.add_text(f"Error drawing histogram for '{series.name}'.", parent=plot_container_tag, color=(255,0,0))

                # KDE 시도 (히스토그램 실패와 무관하게 시도)
                if len(final_data_list) >= 2 and pd.Series(final_data_list).nunique() >= 2:
                    try:
                        # KDE는 float 타입 데이터 필요
                        kde_input_data = pd.Series(final_data_list).astype(float).replace([np.inf,-np.inf],np.nan).dropna()
                        if len(kde_input_data) >= 2 and kde_input_data.nunique() >= 2:
                            min_val, max_val = kde_input_data.min(), kde_input_data.max()
                            if min_val < max_val: # 분산 확인
                                kde = stats.gaussian_kde(kde_input_data)
                                x_vals = np.linspace(min_val, max_val, 150)
                                utils.add_dpg_line_series(y_axis_tag, x_vals.tolist(), kde(x_vals).tolist(), "KDE")
                    except SystemError as se_kde:
                         print(f"!!! SVA PLOT: SYSTEM ERROR during KDE for '{series.name}' !!! Exception: {se_kde}")
                    except Exception as e_kde_other:
                        print(f"SVA PLOT: Other Exception for '{series.name}' (KDE): {e_kde_other}")
    
    # 범주형 데이터
    else: 
        value_counts = clean_s.value_counts(dropna=False).nlargest(10)
        if value_counts.empty:
            if dpg.does_item_exist(plot_container_tag):
                dpg.add_text("No data to display in bar chart.", parent=plot_container_tag, color=(255, 200, 0))
            return

        cat_labels = [str(c)[:20] for c in value_counts.index.tolist()]
        x_pos = np.arange(len(cat_labels))

        if is_grouped:
            unique_groups = sorted(group_target.dropna().unique())[:7]
            num_grps = len(unique_groups)
            if num_grps > 0:
                bar_total_w = 0.8; bar_single_w = bar_total_w / num_grps
                for i, group_val in enumerate(unique_groups):
                    grp_s_data = clean_s[group_target == group_val]
                    grp_counts = grp_s_data.value_counts(dropna=False)
                    y_vals = [grp_counts.get(cat_orig_val, 0) for cat_orig_val in value_counts.index]
                    current_x_pos = x_pos - (bar_total_w / 2) + (i * bar_single_w) + (bar_single_w / 2)
                    utils.add_dpg_bar_series(y_axis_tag, current_x_pos.tolist(), y_vals, lbl=f"{str(group_val)[:15]}")
        else: # 그룹화되지 않은 범주형
            utils.add_dpg_bar_series(y_axis_tag, x_pos.tolist(), value_counts.values.tolist(), "Frequency")
        
        if cat_labels:
            dpg.set_axis_ticks(x_axis_tag, tuple(zip(cat_labels, x_pos.tolist())))


def _sva_run_analysis(callbacks: dict):
    df = callbacks['get_current_df']()
    u_funcs = {**_sva_util_funcs, 'main_app_callbacks': callbacks} 
    target = callbacks['get_selected_target_variable']()

    if not utils.show_dpg_progress_modal("Processing SVA", "Analyzing variables...", modal_tag=TAG_SVA_MODULE_PROGRESS_MODAL, text_tag=TAG_SVA_MODULE_PROGRESS_TEXT): return

    if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW): dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
    else: utils.hide_dpg_progress_modal(TAG_SVA_MODULE_PROGRESS_MODAL); utils.show_dpg_alert_modal("UI Error", "SVA result area missing.", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT); return
    
    if df is None: dpg.add_text("Load data for SVA.", parent=TAG_SVA_RESULTS_CHILD_WINDOW); utils.hide_dpg_progress_modal(TAG_SVA_MODULE_PROGRESS_MODAL); return

    strength = dpg.get_value(TAG_SVA_FILTER_STRENGTH_RADIO) if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO) else "Weak (Exclude obvious non-analytical)"
    grp_flag = dpg.get_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX) if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX) else False
    grp_plot_pref = dpg.get_value(TAG_SVA_GROUPED_PLOT_TYPE_RADIO) if grp_flag and dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO) else "KDE"
    
    target_s_grp = _sva_validate_grouping(df, target, callbacks) if grp_flag else None
    if target_s_grp is None and grp_flag: grp_flag = False
        
    dpg.set_value(TAG_SVA_MODULE_PROGRESS_TEXT, "SVA: Filtering variables..."); 
    if dpg.is_dearpygui_running(): dpg.split_frame()
    filt_cols, act_filter = _sva_get_filtered_variables(df, strength, callbacks, target)
    
    if not filt_cols: dpg.add_text(f"No vars for filter: '{act_filter}'", parent=TAG_SVA_RESULTS_CHILD_WINDOW); utils.hide_dpg_progress_modal(TAG_SVA_MODULE_PROGRESS_MODAL); return
        
    total_vars = len(filt_cols)
    for i, col_name in enumerate(filt_cols):
        if not dpg.is_dearpygui_running(): break
        dpg.set_value(TAG_SVA_MODULE_PROGRESS_TEXT, f"SVA: Analyzing {col_name} ({i+1}/{total_vars})"); 
        if dpg.is_dearpygui_running(): dpg.split_frame()
        _sva_create_section(i, col_name, df, target_s_grp if grp_flag else None, grp_plot_pref, u_funcs, act_filter)
    utils.hide_dpg_progress_modal(TAG_SVA_MODULE_PROGRESS_MODAL)


def _sva_validate_grouping(df: pd.DataFrame, target_var: str, callbacks: dict) -> Optional[pd.Series]:
    if not target_var or target_var not in df.columns: 
        utils.show_dpg_alert_modal("Grouping Info", "Target invalid. Grouping disabled.", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
        return None
    unique_n = df[target_var].nunique(dropna=False)
    if not (2 <= unique_n <= 7):
        utils.show_dpg_alert_modal("Grouping Warning", f"Target '{target_var}' has {unique_n} unique values (need 2-7).", modal_tag=TAG_SVA_MODULE_ALERT_MODAL, text_tag=TAG_SVA_MODULE_ALERT_TEXT)
        if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX): dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
        if dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO): dpg.configure_item(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, show=False)
        return None
    return df[target_var]

def _sva_create_section(idx: int, col_name: str, df: pd.DataFrame, target_s: Optional[pd.Series], plot_pref: str, u_funcs: dict, filt_applied: str):
    sec_tag = f"{TAG_SVA_VARIABLE_SECTION_GROUP_PREFIX}{''.join(filter(str.isalnum, col_name))}_{idx}"
    s1_types_func = u_funcs['main_app_callbacks'].get('get_column_analysis_types', lambda: {})
    s1_types_dict = s1_types_func() # 함수를 호출하여 딕셔너리를 얻음
    col_type = s1_types_dict.get(col_name, str(df[col_name].dtype))
    an_override = "ForceCategoricalForBinaryNumeric" if filt_applied == "None (All variables)" and "Numeric (Binary)" in col_type else None
    with dpg.group(tag=sec_tag, parent=TAG_SVA_RESULTS_CHILD_WINDOW):
        res_w = (dpg.get_item_width(TAG_SVA_RESULTS_CHILD_WINDOW) or 900) 
        head_wrap = res_w - 30 if res_w > 50 else 500
        dpg.add_text(f"Var: {u_funcs['format_text_for_display'](col_name, 60)} ({idx+1})", color=(255,255,0), wrap=head_wrap)
        dpg.add_text(f"Type: {col_type} (Dtype: {str(df[col_name].dtype)})", wrap=head_wrap)
        if an_override: dpg.add_text("Display: Treated as Categorical", color=(200,200,0), wrap=head_wrap)
        dpg.add_spacer(height=5)
        with dpg.group(horizontal=True):
            item_w = max(200, int(res_w * 0.28))
            with dpg.group(width=item_w): _sva_create_basic_stats_table(dpg.last_item(), df[col_name], u_funcs, an_override)
            dpg.add_spacer(width=10)
            with dpg.group(width=item_w): _sva_create_advanced_relations_table(dpg.last_item(), df[col_name], df, u_funcs, item_w)
            dpg.add_spacer(width=10)
            with dpg.group(): _sva_create_plot(dpg.last_item(), df[col_name], target_s, an_override, plot_pref, u_funcs)
        dpg.add_separator(); dpg.add_spacer(height=10)

def _sva_group_by_target_cb(sender, app_data, user_data):
    if dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO): dpg.configure_item(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, show=app_data)
    if not app_data and dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO): dpg.set_value(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, "KDE")

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _sva_main_app_callbacks, _sva_util_funcs
    _sva_main_app_callbacks = main_callbacks
    _sva_util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    _sva_setup_axis_themes() 
    # main_callbacks['register_step_group_tag'](step_name, TAG_SVA_STEP_GROUP)
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
                dpg.add_radio_button(items=["KDE", "Histogram"], tag=TAG_SVA_GROUPED_PLOT_TYPE_RADIO, default_value="KDE", horizontal=True, show=False)
                dpg.add_spacer(height=10)
                dpg.add_button(label="Run Single Variable Analysis", tag=TAG_SVA_RUN_BUTTON, callback=lambda: _sva_run_analysis(main_callbacks), width=-1, height=30)
        dpg.add_separator()
        with dpg.child_window(tag=TAG_SVA_RESULTS_CHILD_WINDOW, border=True):
            dpg.add_text("Select filter options and click 'Run Single Variable Analysis'.")
    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui(current_df: pd.DataFrame, main_callbacks: dict):
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_SVA_STEP_GROUP): return
    global _sva_main_app_callbacks, _sva_util_funcs
    _sva_main_app_callbacks = main_callbacks
    _sva_util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    if current_df is None and dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
        dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
        dpg.add_text("Load data to perform Single Variable Analysis.", parent=TAG_SVA_RESULTS_CHILD_WINDOW)

def reset_sva_ui_defaults():
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_SVA_FILTER_STRENGTH_RADIO): dpg.set_value(TAG_SVA_FILTER_STRENGTH_RADIO, "Weak (Exclude obvious non-analytical)")
    if dpg.does_item_exist(TAG_SVA_GROUP_BY_TARGET_CHECKBOX): dpg.set_value(TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
    if dpg.does_item_exist(TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
        dpg.set_value(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, "KDE")
        dpg.configure_item(TAG_SVA_GROUPED_PLOT_TYPE_RADIO, show=False)
    if dpg.does_item_exist(TAG_SVA_RESULTS_CHILD_WINDOW):
        dpg.delete_item(TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
        dpg.add_text("Select filter options and click 'Run Single Variable Analysis'.", parent=TAG_SVA_RESULTS_CHILD_WINDOW)