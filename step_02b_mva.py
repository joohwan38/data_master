# step_02b_mva.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from scipy import stats
# import traceback # 직접 사용 안하면 제거 가능
from typing import Dict, List, Tuple, Optional, Any
import warnings
import utils

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- MVA UI 태그 정의 ---
TAG_MVA_STEP_GROUP = "mva_step_group"
TAG_MVA_MAIN_TAB_BAR = "mva_main_tab_bar" # MVA 내 하위 탭들을 위한 바

TAG_MVA_CORR_TAB = "mva_corr_tab"
TAG_MVA_CORR_RUN_BUTTON = "mva_corr_run_button"
TAG_MVA_CORR_RESULTS_GROUP = "mva_corr_results_group"

TAG_MVA_PAIRPLOT_TAB = "mva_pairplot_tab"
TAG_MVA_PAIRPLOT_VAR_SELECTOR = "mva_pairplot_var_selector"
TAG_MVA_PAIRPLOT_HUE_COMBO = "mva_pairplot_hue_combo"
TAG_MVA_PAIRPLOT_RUN_BUTTON = "mva_pairplot_run_button"
TAG_MVA_PAIRPLOT_RESULTS_GROUP = "mva_pairplot_results_group"

TAG_MVA_TARGET_TAB = "mva_target_vs_feature_tab" # 좀 더 명확한 이름으로 변경
TAG_MVA_TARGET_INFO_TEXT = "mva_target_info_text"
TAG_MVA_TARGET_FEATURE_COMBO = "mva_target_feature_combo"
TAG_MVA_TARGET_RUN_BUTTON = "mva_target_run_button"
TAG_MVA_TARGET_RESULTS_GROUP = "mva_target_results_group"

TAG_MVA_CAT_EDA_TAB = "mva_cat_corr_tab" # 좀 더 명확한 이름으로 변경
TAG_MVA_CAT_EDA_VAR_SELECTOR = "mva_cat_eda_var_selector"
TAG_MVA_CAT_EDA_RUN_BUTTON = "mva_cat_eda_run_button"
TAG_MVA_CAT_EDA_RESULTS_GROUP = "mva_cat_eda_results_group"

# MVA 모듈 전용 프로그레스/알림 태그 정의 (utils의 기본 태그와 다르게 하여 충돌 방지)
TAG_MVA_MODULE_ALERT_MODAL = "mva_module_specific_alert_modal"
TAG_MVA_MODULE_ALERT_TEXT = "mva_module_specific_alert_text"
# TAG_MVA_MODULE_PROGRESS_MODAL = "mva_module_specific_progress_modal" # 필요시 정의
# TAG_MVA_MODULE_PROGRESS_TEXT = "mva_module_specific_progress_text"   # 필요시 정의


MVA_X_AXIS_ROTATED_TICKS_THEME_TAG = "mva_x_axis_rotated_theme" # MVA용 X축 테마

# --- MVA 전역 변수 ---
_mva_main_app_callbacks: Dict[str, Any] = {}
_mva_util_funcs: Dict[str, Any] = {}

# --- MVA 헬퍼 함수 ---
def get_mva_settings_for_saving() -> Dict[str, Any]:
    """MVA 탭의 현재 설정을 저장용 딕셔너리로 반환합니다."""
    settings = {}
    # Correlation Tab - 특별히 저장할 사용자 선택 UI가 없음 (버튼만 존재)
    settings['corr_tab'] = {} 

    # Pair Plot Tab
    pairplot_settings = {}
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_VAR_SELECTOR):
        pairplot_settings['selected_variables'] = dpg.get_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR)
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_HUE_COMBO):
        pairplot_settings['hue_variable'] = dpg.get_value(TAG_MVA_PAIRPLOT_HUE_COMBO)
    settings['pairplot_tab'] = pairplot_settings

    # Target vs Feature Tab
    target_vs_feature_settings = {}
    if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO):
        target_vs_feature_settings['selected_feature'] = dpg.get_value(TAG_MVA_TARGET_FEATURE_COMBO)
    settings['target_vs_feature_tab'] = target_vs_feature_settings
    
    # Categorical EDA Tab
    cat_eda_settings = {}
    if dpg.does_item_exist(TAG_MVA_CAT_EDA_VAR_SELECTOR):
        cat_eda_settings['selected_variables'] = dpg.get_value(TAG_MVA_CAT_EDA_VAR_SELECTOR)
    settings['cat_eda_tab'] = cat_eda_settings
    
    if dpg.does_item_exist(TAG_MVA_MAIN_TAB_BAR): # 현재 활성화된 탭 저장
        settings['active_mva_tab_label'] = None
        for child in dpg.get_item_children(TAG_MVA_MAIN_TAB_BAR, 1): # slot 1 is for tabs
            if dpg.get_item_configuration(child)["show"]: # This might not be the best way to find active tab, DPG might have a direct way
                 # A more robust way would be to store active tab on change if DPG doesn't provide a getter
                settings['active_mva_tab_label'] = dpg.get_item_label(child) # Assuming label is unique and can be used to find it back
                break # MVA 메인 탭바에서 현재 활성 탭의 레이블을 저장 (복원 시 이 레이블로 탭을 찾아 활성화 필요)

    return settings

def apply_mva_settings_from_loaded(settings: Dict[str, Any], current_df: Optional[pd.DataFrame], main_callbacks: Dict[str, Any]):
    """저장된 설정을 MVA UI에 적용합니다."""
    if not dpg.is_dearpygui_running():
        return

    # Pair Plot Tab 복원
    pairplot_settings = settings.get('pairplot_tab', {})
    if 'selected_variables' in pairplot_settings and dpg.does_item_exist(TAG_MVA_PAIRPLOT_VAR_SELECTOR):
        # Listbox 아이템이 먼저 설정되어 있어야 값을 제대로 선택할 수 있음
        # update_ui 함수에서 아이템 목록을 채우므로, 해당 함수가 먼저 호출되거나 여기서 목록을 다시 채워야 함.
        # 여기서는 update_ui가 먼저 실행되었다고 가정하고 값만 설정.
        num_cols = _mva_util_funcs.get('_get_numeric_cols', lambda df: [])(current_df) if _mva_util_funcs else []
        dpg.configure_item(TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=num_cols) # 아이템 먼저 설정
        dpg.set_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR, pairplot_settings['selected_variables'])
        
    if 'hue_variable' in pairplot_settings and dpg.does_item_exist(TAG_MVA_PAIRPLOT_HUE_COMBO):
        cat_cols_hue = [""] + (_mva_util_funcs.get('_get_categorical_cols', lambda df, max_u, mc: [])(current_df, 10, main_callbacks) if _mva_util_funcs else [])
        dpg.configure_item(TAG_MVA_PAIRPLOT_HUE_COMBO, items=cat_cols_hue) # 아이템 먼저 설정
        dpg.set_value(TAG_MVA_PAIRPLOT_HUE_COMBO, pairplot_settings['hue_variable'])

    # Target vs Feature Tab 복원
    target_vs_feature_settings = settings.get('target_vs_feature_tab', {})
    if 'selected_feature' in target_vs_feature_settings and dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO):
        all_cols = current_df.columns.tolist() if current_df is not None else []
        target_var = main_callbacks['get_selected_target_variable']()
        feature_items = [c for c in all_cols if c != target_var]
        dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=feature_items) # 아이템 먼저 설정
        dpg.set_value(TAG_MVA_TARGET_FEATURE_COMBO, target_vs_feature_settings['selected_feature'])
        
    # Categorical EDA Tab 복원
    cat_eda_settings = settings.get('cat_eda_tab', {})
    if 'selected_variables' in cat_eda_settings and dpg.does_item_exist(TAG_MVA_CAT_EDA_VAR_SELECTOR):
        all_cat_cols = _mva_util_funcs.get('_get_categorical_cols', lambda df, max_u, mc: [])(current_df, 30, main_callbacks) if _mva_util_funcs else []
        dpg.configure_item(TAG_MVA_CAT_EDA_VAR_SELECTOR, items=all_cat_cols) # 아이템 먼저 설정
        dpg.set_value(TAG_MVA_CAT_EDA_VAR_SELECTOR, cat_eda_settings['selected_variables'])

    # 활성 MVA 탭 복원 (주의: DPG에서 프로그래밍 방식으로 탭을 활성화하는 직접적인 방법이 제한적일 수 있음)
    # DPG 1.10+ 에서는 dpg.set_value(tab_bar_tag, tab_tag_to_activate) 로 가능
    active_tab_label = settings.get('active_mva_tab_label')
    if active_tab_label and dpg.does_item_exist(TAG_MVA_MAIN_TAB_BAR):
        # 레이블을 기반으로 탭 태그를 찾아야 함. create_ui에서 탭 생성 시 label과 tag를 동일하게 하거나 매핑 필요.
        # 아래는 레이블과 태그가 동일하다고 가정 (예: TAG_MVA_CORR_TAB의 레이블이 "Correlation (Numeric)")
        # 실제로는 탭 생성 시 label과 tag를 잘 관리해야 함.
        # 여기서는 간단히 첫번째 탭을 활성화하는 것으로 대체하거나, 또는
        # create_ui에서 각 탭에 고유 tag를 부여하고 해당 tag를 active_mva_tab_tag로 저장하는 것이 더 좋음
        # 현재는 TAG_MVA_CORR_TAB, TAG_MVA_PAIRPLOT_TAB 등을 사용하고 있으므로, 이 태그를 직접 저장하고 사용 가능
        # get_mva_settings_for_saving에서 active_mva_tab_tag를 저장하도록 수정하는 것이 좋음
        # 예시: settings['active_mva_tab_tag'] = dpg.get_value(TAG_MVA_MAIN_TAB_BAR) # 현재 활성 탭의 태그를 가져옴
        # if 'active_mva_tab_tag' in settings and dpg.does_item_exist(settings['active_mva_tab_tag']):
        #    dpg.set_value(TAG_MVA_MAIN_TAB_BAR, settings['active_mva_tab_tag'])
        pass # 탭 활성화는 DPG 버전에 따라 구현 방식이 다를 수 있어 일단 pass. 필요시 DPG 문서를 참조하여 정확한 방법으로 구현.


    # 각 탭의 결과 창 초기화
    for area_tag in [TAG_MVA_CORR_RESULTS_GROUP, TAG_MVA_PAIRPLOT_RESULTS_GROUP, TAG_MVA_TARGET_RESULTS_GROUP, TAG_MVA_CAT_EDA_RESULTS_GROUP]:
        if dpg.does_item_exist(area_tag):
            dpg.delete_item(area_tag, children_only=True)
            dpg.add_text("Settings loaded. Click the relevant 'Run/Generate' button to update results.", parent=area_tag)
            
    # update_ui를 호출하여 전체적인 MVA UI 상태를 갱신
    update_ui(current_df, main_callbacks)


def _mva_setup_axis_themes():
    if not dpg.does_item_exist(MVA_X_AXIS_ROTATED_TICKS_THEME_TAG):
        with dpg.theme(tag=MVA_X_AXIS_ROTATED_TICKS_THEME_TAG):
            with dpg.theme_component(dpg.mvXAxis): pass # 실제 스타일은 사용자가 이 태그에 직접 정의


def _mva_run_correlation_analysis(df: pd.DataFrame, u_funcs: dict, callbacks: dict):
    results_group = TAG_MVA_CORR_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group): return
    dpg.delete_item(results_group, children_only=True)
    if df is None: dpg.add_text("Load data first.", parent=results_group); return
    
    num_cols = utils._get_numeric_cols(df) # utils 함수 사용
    if len(num_cols) < 2: dpg.add_text("Not enough numeric columns (need at least 2).", parent=results_group); return

    target_var, max_heatmap_vars = callbacks['get_selected_target_variable'](), 10

    if len(num_cols) <= max_heatmap_vars:
        dpg.add_text(f"Correlation Matrix for {len(num_cols)} Numeric Variables:", parent=results_group, color=(255,255,0))
        try:
            corr_mat = df[num_cols].corr(method='pearson')
            utils.create_dpg_heatmap_plot(results_group, corr_mat, "Overall Correlation Heatmap (Pearson)")
        except Exception as e: dpg.add_text(f"Error creating overall heatmap: {e}", parent=results_group, color=(255,0,0))
    else:
        dpg.add_text(f"Num numeric vars ({len(num_cols)}) > {max_heatmap_vars}. Showing targeted (max {max_heatmap_vars} vars/heatmap):", parent=results_group, color=(255,255,0))
        _mva_analyze_highly_correlated(df, num_cols, results_group, max_heatmap_vars) # 내부 헬퍼
        dpg.add_separator(parent=results_group)
        if target_var and target_var in df.columns:
            _mva_analyze_target_correlation(df, target_var, num_cols, max_heatmap_vars, results_group) # 내부 헬퍼
        else: dpg.add_text("No target selected or target not in DF. Target correlation skipped.", parent=results_group)
    _mva_create_correlation_pairs_table(df, num_cols, results_group, u_funcs) # 내부 헬퍼

def _mva_analyze_highly_correlated(df: pd.DataFrame, num_cols: List[str], parent: str, max_vars: int):
    dpg.add_text(f"Analysis 1: Vars with Pairwise |Corr| >= 0.6 (Max {max_vars} vars for heatmap)", parent=parent, color=(200,200,0))
    if len(num_cols) < 2: dpg.add_text("Not enough numeric cols.", parent=parent); return
    try: corr_mat = df[num_cols].corr(method='pearson')
    except Exception as e: dpg.add_text(f"Error calc corr matrix: {e}", parent=parent, color=(255,0,0)); return

    high_corr_set = set()
    for i in range(len(corr_mat.columns)):
        for j in range(i + 1, len(corr_mat.columns)):
            if pd.notna(val := corr_mat.iloc[i, j]) and abs(val) >= 0.6:
                high_corr_set.add(corr_mat.columns[i]); high_corr_set.add(corr_mat.columns[j])
    
    vars_hm = sorted(list(high_corr_set))
    if len(vars_hm) > max_vars:
        dpg.add_text(f"> {max_vars} vars with |corr| >= 0.6. Displaying first {max_vars}.", parent=parent, color=(200,200,100), wrap=-1)
        vars_hm = vars_hm[:max_vars]
    if len(vars_hm) >= 2:
        try: corr_mat_high = df[vars_hm].corr(method='pearson')
        except Exception as e: dpg.add_text(f"Error calc sub-matrix: {e}", parent=parent, color=(255,0,0)); return
        if not corr_mat_high.empty: utils.create_dpg_heatmap_plot(parent, corr_mat_high, f"Heatmap: Highly Correlated (>=0.6, Top {len(vars_hm)})")
        else: dpg.add_text("Could not generate heatmap (matrix empty/invalid).", parent=parent)
    else: dpg.add_text("No var pairs with |corr| >= 0.6 or not enough vars (min 2) for heatmap.", parent=parent)

def _mva_analyze_target_correlation(df: pd.DataFrame, target: str, num_cols: List[str], max_total_vars: int, parent: str):
    dpg.add_text(f"Analysis 2: Top {max_total_vars} Vars (incl. target) Correlated with Target '{target}'", parent=parent, color=(200,200,0))
    if not pd.api.types.is_numeric_dtype(df[target].dtype):
        dpg.add_text(f"Target '{target}' not numeric. Requires numeric target.", parent=parent, color=(255,100,100)); return
    
    top_n_others = max_total_vars - 1
    if top_n_others < 1: dpg.add_text(f"Not enough slots (max_total_vars={max_total_vars}) for other vars.", parent=parent); return
    
    # utils._get_top_n_correlated_with_target 사용
    top_corr_others = utils._get_top_n_correlated_with_target(df, target, num_cols, top_n_others)
    vars_hm = list(dict.fromkeys([target] + [v for v in top_corr_others if v != target]))

    if len(vars_hm) >= 2:
        try:
            corr_mat_target = df[vars_hm].corr(method='pearson')
            utils.create_dpg_heatmap_plot(parent, corr_mat_target, f"Heatmap: Target '{target}' & Top {len(vars_hm)-1} Correlated")
        except Exception as e: dpg.add_text(f"Error creating target corr heatmap: {e}", parent=parent, color=(255,0,0))
    else: dpg.add_text(f"Not enough vars (min 2, found {len(vars_hm)}) for target corr heatmap.", parent=parent)

def _mva_create_correlation_pairs_table(df: pd.DataFrame, num_cols: List[str], parent: str, u_funcs: dict):
    dpg.add_text("Highly Correlated Numeric Pairs (|Correlation| > 0.7):", parent=parent)
    if len(num_cols) < 2: dpg.add_text("Need at least 2 numeric columns.", parent=parent); return
    corr_mat = df[num_cols].corr(method='pearson')
    pairs = [{'Var1': corr_mat.columns[i], 'Var2': corr_mat.columns[j], 'Corr': f"{corr_mat.iloc[i,j]:.3f}"}
             for i in range(len(corr_mat.columns)) for j in range(i+1, len(corr_mat.columns)) if abs(corr_mat.iloc[i,j]) > 0.7]
    if pairs:
        df_pairs, tbl_tag = pd.DataFrame(pairs), dpg.generate_uuid()
        with dpg.table(header_row=True, tag=tbl_tag, parent=parent, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                      scrollY=True, height=200, borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
            u_funcs['create_table_with_data'](tbl_tag, df_pairs, parent_df_for_widths=df_pairs)
    else: dpg.add_text("No pairs with |correlation| > 0.7 found.", parent=parent)


def _mva_run_pair_plot_analysis(df: pd.DataFrame, sel_vars: list, hue_var_name: str, u_funcs: dict, callbacks: dict):
    res_group = TAG_MVA_PAIRPLOT_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(res_group): return
    dpg.delete_item(res_group, children_only=True)
    if df is None: dpg.add_text("Load data first.", parent=res_group); return
    
    num_cols = utils._get_numeric_cols(df)
    if not num_cols: dpg.add_text("No numeric variables for Pair Plot.", parent=res_group); return
    
    vars_plot = _mva_select_vars_for_pairplot(df, sel_vars, num_cols, callbacks, res_group)
    if not vars_plot or len(vars_plot) < 2:
        dpg.add_text("Not enough valid numeric vars for Pair Plot (need min 2).", parent=res_group); return
        
    hue_s, hue_cats, actual_hue_name = _mva_validate_hue_variable(df, hue_var_name, callbacks)
    _mva_create_pair_plot_grid(df, vars_plot, hue_s, hue_cats, actual_hue_name, res_group)

def _mva_select_vars_for_pairplot(df: pd.DataFrame, sel_vars: list, num_cols: List[str], callbacks: dict, parent: str) -> List[str]:
    target, max_v, vars_plot, msg = callbacks['get_selected_target_variable'](), 7, [], ""
    if sel_vars:
        vars_plot = [v for v in sel_vars if v in num_cols]
        if not vars_plot: msg = "Selected vars are not valid numeric."
        elif len(vars_plot) < 2: msg = "Select at least two valid numeric vars."
    else:
        if len(num_cols) <= max_v: vars_plot, msg = num_cols, f"Using all {len(num_cols)} numeric vars."
        else:
            if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target].dtype):
                msg = f"Using top {max_v} vars correlated with target '{target}'."
                vars_plot = utils._get_top_n_correlated_with_target(df, target, num_cols, max_v) # utils 함수 사용
                if target in num_cols and target not in vars_plot:
                    vars_plot = (vars_plot + [target]) if len(vars_plot) < max_v else (vars_plot[:-1] + [target])
                vars_plot = list(dict.fromkeys(vars_plot))
            else: msg, vars_plot = f"Using first {max_v} numeric vars.", num_cols[:max_v]
    if msg: dpg.add_text(msg, parent=parent, wrap=(dpg.get_item_width(parent) or 600) -10)
    if len(vars_plot) > max_v:
        utils.show_dpg_alert_modal("Pair Plot Limit", f"Plotting first {max_v} of {len(vars_plot)} vars.", modal_tag=TAG_MVA_MODULE_ALERT_MODAL, text_tag=TAG_MVA_MODULE_ALERT_TEXT)
        vars_plot = vars_plot[:max_v]
    return vars_plot

def _mva_validate_hue_variable(df: pd.DataFrame, hue_var_name: str, callbacks: dict) -> Tuple[Optional[pd.Series], Optional[List[str]], Optional[str]]:
    if not hue_var_name or hue_var_name not in df.columns: return None, None, None
    hue_s = df[hue_var_name]
    # utils._get_categorical_cols 사용
    cat_cols_hue = utils._get_categorical_cols(df[[hue_var_name]], max_unique_for_cat=10, main_callbacks=callbacks)
    if hue_var_name in cat_cols_hue:
        return hue_s, sorted(hue_s.astype(str).dropna().unique()), hue_var_name
    else:
        utils.show_dpg_alert_modal("Hue Var Warning", f"Hue var '{hue_var_name}' has >10 unique vals or not suitable. Hue disabled.", modal_tag=TAG_MVA_MODULE_ALERT_MODAL, text_tag=TAG_MVA_MODULE_ALERT_TEXT)
        return None, None, None

def _mva_create_pair_plot_grid(df: pd.DataFrame, vars_plot: List[str], hue_s: Optional[pd.Series], 
                              hue_cats: Optional[List[str]], hue_var_name: Optional[str], parent: str):
    n = len(vars_plot)
    p_width = dpg.get_item_width(parent) or 800
    cell_w = max(150, min(250, int(p_width / n) - 15 if n > 0 else 200))
    
    title = f"Pair Plot: {', '.join(vars_plot)}" + (f" (Hue: {hue_var_name})" if hue_var_name else "")
    dpg.add_text(title, parent=parent)
    
    with dpg.child_window(parent=parent, border=False, autosize_x=True, height = n * (cell_w + 5) + 30 if n > 0 else 100):
        for i in range(n):
            with dpg.group(horizontal=True):
                for j in range(n):
                    _mva_create_pair_plot_cell(df, vars_plot, i, j, cell_w, cell_w, hue_s, hue_cats, n)

def _mva_create_pair_plot_cell(df: pd.DataFrame, vars_list: List[str], r_idx: int, c_idx: int,
                              w: int, h: int, hue_s: Optional[pd.Series],
                              hue_cats: Optional[List[str]], n_vars: int):
    var_y, var_x = vars_list[r_idx], vars_list[c_idx]
    lbl = f"{var_y[:10]} vs {var_x[:10]}" if r_idx != c_idx else f"Dist: {var_x[:10]}"
    show_leg = (hue_s is not None and r_idx != c_idx and hue_cats and len(hue_cats) <=5)

    # 수정 전: _, x_ax_tag, y_ax_tag, leg_tag_cell = utils.create_dpg_plot_scaffold(dpg.last_item(), lbl, var_x if (r_idx == n_vars - 1) else "", var_y if (c_idx == 0) else "", width=w, height=h, show_legend=show_leg)
    # 수정 후:
    _, x_ax_tag, y_ax_tag, leg_tag_cell = utils.create_dpg_plot_scaffold(
        dpg.last_item(), lbl, var_x if (r_idx == n_vars - 1) else "", var_y if (c_idx == 0) else "",
        w=w, h=h, legend=show_leg # 파라미터명 w, h, legend 로 수정
    )
    dpg.configure_item(x_ax_tag, no_tick_labels=not (r_idx == n_vars - 1))
    dpg.configure_item(y_ax_tag, no_tick_labels=not (c_idx == 0))
    if dpg.does_item_exist(MVA_X_AXIS_ROTATED_TICKS_THEME_TAG) and (r_idx == n_vars -1):
         dpg.bind_item_theme(x_ax_tag, MVA_X_AXIS_ROTATED_TICKS_THEME_TAG)
    if leg_tag_cell and not show_leg: dpg.hide_item(leg_tag_cell)

    if r_idx == c_idx:
        diag_s = df[var_x].dropna()
        if not diag_s.empty and diag_s.nunique() >= 1:
            if diag_s.nunique() == 1:
                utils.add_dpg_bar_series(y_ax_tag, [0], [len(diag_s)], str(diag_s.iloc[0])[:10])
                dpg.set_axis_ticks(x_ax_tag, [(str(diag_s.iloc[0])[:10], 0)])
            else: utils.add_dpg_histogram_series(y_ax_tag, diag_s.tolist(), "Hist", density=True, bins=-1)
    else:
        s_x, s_y = df[var_x], df[var_y]
        if hue_s is not None and hue_cats is not None:
            for cat_val in hue_cats:
                mask = (hue_s.astype(str) == cat_val)
                df_scatter = pd.concat([s_x[mask], s_y[mask]], axis=1).dropna()
                if not df_scatter.empty: utils.add_dpg_scatter_series(y_ax_tag, df_scatter.iloc[:,0].tolist(), df_scatter.iloc[:,1].tolist(), str(cat_val)[:15])
        else:
            df_scatter = pd.concat([s_x, s_y], axis=1).dropna()
            if not df_scatter.empty: utils.add_dpg_scatter_series(y_ax_tag, df_scatter.iloc[:,0].tolist(), df_scatter.iloc[:,1].tolist())

def _mva_run_cat_corr_analysis(df: pd.DataFrame, u_funcs: dict, callbacks: dict):
    res_group = TAG_MVA_CAT_EDA_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(res_group): return
    dpg.delete_item(res_group, children_only=True)
    if df is None: dpg.add_text("Load data first.", parent=res_group); return
    
    sel_vars = dpg.get_value(TAG_MVA_CAT_EDA_VAR_SELECTOR) if dpg.does_item_exist(TAG_MVA_CAT_EDA_VAR_SELECTOR) else []
    # utils._get_categorical_cols 사용
    all_cat_cols = utils._get_categorical_cols(df, max_unique_for_cat=30, main_callbacks=callbacks)
    
    vars_analyze = ([v for v in sel_vars if v in all_cat_cols] if sel_vars else all_cat_cols)[:20]
    if not sel_vars and len(all_cat_cols) > 20 : dpg.add_text(f"Using first 20 cat vars for Cramer's V.", parent=res_group, color=(200,200,0))
    elif sel_vars and len(vars_analyze) < len(sel_vars) : dpg.add_text(f"Using {len(vars_analyze)} valid cat vars from selection.", parent=res_group, color=(200,200,0))

    if len(vars_analyze) < 2: dpg.add_text("Not enough cat vars for Cramer's V (need min 2).", parent=res_group); return
        
    dpg.add_text(f"Cramer's V Matrix for: {', '.join(vars_analyze)}", parent=res_group)
    cramers_v_mat = pd.DataFrame(np.zeros((len(vars_analyze), len(vars_analyze))), columns=vars_analyze, index=vars_analyze)
    for i in range(len(vars_analyze)):
        for j in range(i, len(vars_analyze)):
            v1, v2 = vars_analyze[i], vars_analyze[j]
            # utils.calculate_cramers_v 사용
            cv_val = 1.0 if v1 == v2 else utils.calculate_cramers_v(df[v1], df[v2])
            cramers_v_mat.iloc[i,j], cramers_v_mat.iloc[j,i] = cv_val, cv_val
    utils.create_dpg_heatmap_plot(res_group, cramers_v_mat, "Cramer's V Heatmap (Categorical Associations)", cmap=dpg.mvPlotColormap_Viridis)

def _mva_run_target_vs_feature(df: pd.DataFrame, target: str, target_type_str: str, feature: str, u_funcs: dict, callbacks: dict): # 이름 변경
    res_group = TAG_MVA_TARGET_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(res_group): return
    dpg.delete_item(res_group, children_only=True)
    if df is None or not target or target not in df.columns or not feature or feature not in df.columns or target == feature:
        dpg.add_text("Select valid & distinct target and feature.", parent=res_group); return
        
    dpg.add_text(f"Analysis: Feature '{feature}' vs Target '{target}' (Type: {target_type_str})", parent=res_group); dpg.add_separator(parent=res_group)
    
    s_target, s_feature = df[target], df[feature]
    s1_types = callbacks.get('get_column_analysis_types', lambda: {})()
    feat_s1_type = s1_types.get(feature, str(s_feature.dtype))
    is_feat_num = ("Numeric" in feat_s1_type and "Binary" not in feat_s1_type) or \
                  (pd.api.types.is_numeric_dtype(s_feature.dtype) and s_feature.nunique() > 2)
    
    plot_parent = dpg.add_group(parent=res_group) # 플롯들을 담을 그룹

    if target_type_str == "Continuous":
        if is_feat_num: _mva_cont_target_num_feature(df,target,feature,s_target,s_feature,plot_parent,u_funcs)
        else: _mva_cont_target_cat_feature(df,target,feature,s_target,s_feature,plot_parent,u_funcs)
    elif target_type_str == "Categorical":
        if is_feat_num: _mva_cat_target_num_feature(df,target,feature,s_target,s_feature,plot_parent,u_funcs)
        else: _mva_cat_target_cat_feature(df,target,feature,s_target,s_feature,plot_parent,u_funcs)
    else: dpg.add_text(f"Analysis for target type '{target_type_str}' not implemented.", parent=plot_parent)

def _mva_cont_target_num_feature(df,target,feature,s_target,s_feature,parent,u_funcs):
    aligned = pd.concat([s_feature, s_target], axis=1).dropna()
    if not aligned.empty and len(aligned) >= 2:
        corr = aligned.iloc[:,0].corr(aligned.iloc[:,1])
        dpg.add_text(f"Pearson Corr: {corr:.3f}" if pd.notna(corr) else "Corr: N/A", parent=parent)
        _, x_ax, y_ax, _ = utils.create_dpg_plot_scaffold(parent, f"Scatter: '{feature}' by '{target}'", feature, target, h=350)
        if dpg.does_item_exist(MVA_X_AXIS_ROTATED_TICKS_THEME_TAG): dpg.bind_item_theme(x_ax, MVA_X_AXIS_ROTATED_TICKS_THEME_TAG)
        utils.add_dpg_scatter_series(y_ax, aligned.iloc[:,0].tolist(), aligned.iloc[:,1].tolist())
    else: dpg.add_text("Not enough common data points.", parent=parent)

def _mva_cont_target_cat_feature(df,target,feature,s_target,s_feature,parent,u_funcs):
    dpg.add_text("Grouped Stats (Feature Cats vs Cont Target):", parent=parent)
    try:
        feat_cat = s_feature.astype(str) if s_feature.nunique(dropna=False) > 20 else s_feature
        if feat_cat.nunique(dropna=False) > 20: dpg.add_text(f"Feature has >20 cats ({feat_cat.nunique(dropna=False)}). Max 20.", parent=parent); return

        grp_stats = df.groupby(feat_cat)[target].agg(['mean','median','std','count','min','max']).reset_index()
        grp_stats.columns = [str(c) for c in grp_stats.columns]
        tbl_tag, tbl_h = dpg.generate_uuid(), min(200, len(grp_stats)*25+40)
        with dpg.table(header_row=True,tag=tbl_tag,parent=parent,resizable=True,height=int(tbl_h),scrollY=True, borders_innerH=True,borders_outerH=True,borders_innerV=True,borders_outerV=True):
            u_funcs['create_table_with_data'](tbl_tag, grp_stats.round(3), parent_df_for_widths=grp_stats.round(3))
        
        dpg.add_text("Distribution comparison (KDE):", parent=parent)
        unique_cats = feat_cat.dropna().unique()[:7]
        if len(unique_cats) >= 1:
            _, x_ax, y_ax, _ = utils.create_dpg_plot_scaffold(parent, f"Dist of '{target}' by '{feature}'", target, "Density", h=350, legend=True)
            if dpg.does_item_exist(MVA_X_AXIS_ROTATED_TICKS_THEME_TAG): dpg.bind_item_theme(x_ax, MVA_X_AXIS_ROTATED_TICKS_THEME_TAG)
            for cat_val in unique_cats:
                subset = s_target[feat_cat == cat_val].dropna()
                if len(subset) > 1 and subset.nunique() > 1:
                    try:
                        kde = stats.gaussian_kde(subset.astype(float))
                        x_v = np.linspace(subset.min(), subset.max(), 100)
                        utils.add_dpg_line_series(y_ax, x_v.tolist(), kde(x_v).tolist(), label=f"{feature}={str(cat_val)[:15]}")
                    except Exception: pass
    except Exception as e: dpg.add_text(f"Error grouping: {e}", parent=parent)

def _mva_cat_target_num_feature(df,target,feature,s_target,s_feature,parent,u_funcs):
    dpg.add_text("Grouped Stats (Target Cats vs Num Feature):", parent=parent)
    try:
        target_cat = s_target.astype(str) if s_target.nunique(dropna=False) > 20 else s_target
        if target_cat.nunique(dropna=False) > 20: dpg.add_text(f"Target has >20 cats ({target_cat.nunique(dropna=False)}). Max 20.", parent=parent); return
            
        grp_stats = df.groupby(target_cat)[feature].agg(['mean','median','std','count','min','max']).reset_index()
        grp_stats.columns = [str(c) for c in grp_stats.columns]
        tbl_tag, tbl_h = dpg.generate_uuid(), min(200, len(grp_stats)*25+40)
        with dpg.table(header_row=True,tag=tbl_tag,parent=parent,resizable=True,height=int(tbl_h),scrollY=True, borders_innerH=True,borders_outerH=True,borders_innerV=True,borders_outerV=True):
            u_funcs['create_table_with_data'](tbl_tag, grp_stats.round(3), parent_df_for_widths=grp_stats.round(3))
        
        dpg.add_text("Density Plots of Feature by Target Cats:", parent=parent)
        unique_targets = target_cat.dropna().unique()[:7]
        if len(unique_targets) >= 1:
            _, x_ax, y_ax, _ = utils.create_dpg_plot_scaffold(parent, f"Density of '{feature}' by '{target}'", feature, "Density", h=350, legend=True)
            if dpg.does_item_exist(MVA_X_AXIS_ROTATED_TICKS_THEME_TAG): dpg.bind_item_theme(x_ax, MVA_X_AXIS_ROTATED_TICKS_THEME_TAG)
            for cat_val in unique_targets:
                subset = s_feature[target_cat == cat_val].dropna()
                if len(subset) > 1 and subset.nunique() > 1:
                    try:
                        kde = stats.gaussian_kde(subset.astype(float))
                        x_v = np.linspace(subset.min(), subset.max(), 100)
                        utils.add_dpg_line_series(y_ax, x_v.tolist(), kde(x_v).tolist(), label=f"{target}={str(cat_val)[:15]}")
                    except Exception: pass
    except Exception as e: dpg.add_text(f"Error grouping: {e}", parent=parent)

def _mva_cat_target_cat_feature(df,target,feature,s_target,s_feature,parent,u_funcs):
    dpg.add_text("Crosstab (Feature vs Target):", parent=parent)
    try:
        ct_feat, ct_target = s_feature.astype(str), s_target.astype(str)
        if ct_feat.nunique(dropna=False) > 20 or ct_target.nunique(dropna=False) > 20:
            dpg.add_text(">20 cats. Showing top 20 combinations.", parent=parent)
            counts_sum = df.groupby([ct_feat.name, ct_target.name]).size().reset_index(name='counts').nlargest(20, 'counts')
            tbl_tag, tbl_h = dpg.generate_uuid(), min(250, len(counts_sum)*25+40)
            with dpg.table(header_row=True,tag=tbl_tag,parent=parent,resizable=True,height=int(tbl_h),scrollY=True, borders_innerH=True,borders_outerH=True,borders_innerV=True,borders_outerV=True):
                u_funcs['create_table_with_data'](tbl_tag, counts_sum, parent_df_for_widths=counts_sum)
        else:
            ct_abs = pd.crosstab(ct_feat, ct_target, dropna=False)
            chi2, p, _, _ = stats.chi2_contingency(ct_abs)
            c_v = utils.calculate_cramers_v(s_feature, s_target) # utils 함수 사용
            dpg.add_text(f"Chi2: {chi2:.2f}, p: {p:.3f}, Cramér's V: {c_v:.3f}", parent=parent)
            dpg.add_text("Counts:", parent=parent)
            tbl_abs_tag, tbl_abs_h = dpg.generate_uuid(), min(180, len(ct_abs.reset_index())*25+40)
            with dpg.table(header_row=True,tag=tbl_abs_tag,parent=parent,resizable=True,height=int(tbl_abs_h),scrollY=True, policy=dpg.mvTable_SizingFixedFit, borders_innerH=True,borders_outerH=True,borders_innerV=True,borders_outerV=True):
                u_funcs['create_table_with_data'](tbl_abs_tag, ct_abs.reset_index(), parent_df_for_widths=ct_abs.reset_index())
    except Exception as e: dpg.add_text(f"Error crosstab: {e}", parent=parent)

# --- MVA UI 생성 및 업데이트 ---
def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _mva_main_app_callbacks, _mva_util_funcs
    _mva_main_app_callbacks = main_callbacks
    _mva_util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    _mva_setup_axis_themes()

    # main_callbacks['register_step_group_tag'](step_name, TAG_MVA_STEP_GROUP)
    with dpg.group(tag=TAG_MVA_STEP_GROUP, parent=parent_container_tag):
        dpg.add_text(f"--- {step_name} ---"); dpg.add_separator()
        with dpg.tab_bar(tag=TAG_MVA_MAIN_TAB_BAR):
            with dpg.tab(label="Correlation (Numeric)", tag=TAG_MVA_CORR_TAB):
                dpg.add_button(label="Run Correlation Analysis", tag=TAG_MVA_CORR_RUN_BUTTON,
                             callback=lambda: _mva_run_correlation_analysis(main_callbacks['get_current_df'](), _mva_util_funcs, main_callbacks))
                dpg.add_child_window(tag=TAG_MVA_CORR_RESULTS_GROUP, border=True)
            with dpg.tab(label="Pair Plot (Numeric)", tag=TAG_MVA_PAIRPLOT_TAB):
                dpg.add_text("Select numeric vars (max 7 recom.). If none, auto-selects.", wrap=-1)
                dpg.add_listbox(tag=TAG_MVA_PAIRPLOT_VAR_SELECTOR, width=-1, num_items=6) # 아이템 수 조절
                dpg.add_combo(label="Hue (Optional Cat Var, <10 Unique)", tag=TAG_MVA_PAIRPLOT_HUE_COMBO, width=-1) # 너비 최대로
                dpg.add_button(label="Generate Pair Plot", tag=TAG_MVA_PAIRPLOT_RUN_BUTTON,
                             callback=lambda: _mva_run_pair_plot_analysis(main_callbacks['get_current_df'](), dpg.get_value(TAG_MVA_PAIRPLOT_VAR_SELECTOR),
                                                                    dpg.get_value(TAG_MVA_PAIRPLOT_HUE_COMBO), _mva_util_funcs, main_callbacks))
                dpg.add_child_window(tag=TAG_MVA_PAIRPLOT_RESULTS_GROUP, border=True)
            with dpg.tab(label="Target vs Feature", tag=TAG_MVA_TARGET_TAB):
                dpg.add_text("Analyze relationship between features and the selected target.", tag=TAG_MVA_TARGET_INFO_TEXT, wrap=-1)
                with dpg.group(horizontal=True):
                    dpg.add_combo(label="Feature Variable", tag=TAG_MVA_TARGET_FEATURE_COMBO, width=-1) # 너비 최대로
                    dpg.add_button(label="Analyze vs Target", tag=TAG_MVA_TARGET_RUN_BUTTON,
                                 callback=lambda: _mva_run_target_vs_feature(main_callbacks['get_current_df'](), main_callbacks['get_selected_target_variable'](),
                                                                              main_callbacks['get_selected_target_variable_type'](), dpg.get_value(TAG_MVA_TARGET_FEATURE_COMBO),
                                                                              _mva_util_funcs, main_callbacks))
                dpg.add_separator()
                dpg.add_child_window(tag=TAG_MVA_TARGET_RESULTS_GROUP, border=True)
            with dpg.tab(label="Correlation (Categorical)", tag=TAG_MVA_CAT_EDA_TAB):
                dpg.add_text("Cramer's V for categorical associations (Max 20 vars).", wrap=-1)
                dpg.add_listbox(tag=TAG_MVA_CAT_EDA_VAR_SELECTOR, width=-1, num_items=6) # 아이템 수 조절
                dpg.add_button(label="Run Categorical Association Analysis", tag=TAG_MVA_CAT_EDA_RUN_BUTTON,
                             callback=lambda: _mva_run_cat_corr_analysis(main_callbacks['get_current_df'](), _mva_util_funcs, main_callbacks))
                dpg.add_child_window(tag=TAG_MVA_CAT_EDA_RESULTS_GROUP, border=True)
    main_callbacks['register_module_updater'](step_name, update_ui)


def update_ui(current_df: pd.DataFrame, main_callbacks: dict):
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_MVA_STEP_GROUP): return
    global _mva_main_app_callbacks, _mva_util_funcs
    _mva_main_app_callbacks = main_callbacks
    _mva_util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    
    all_cols = current_df.columns.tolist() if current_df is not None else []
    num_cols = utils._get_numeric_cols(current_df) if current_df is not None else [] # utils 함수 사용
    cat_cols_hue = [""] + (utils._get_categorical_cols(current_df, 10, main_callbacks) if current_df is not None else []) # utils 함수 사용
    all_cat_cols = utils._get_categorical_cols(current_df, 30, main_callbacks) if current_df is not None else [] # utils 함수 사용
    
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_VAR_SELECTOR): dpg.configure_item(TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=num_cols)
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_HUE_COMBO):
        curr_hue = dpg.get_value(TAG_MVA_PAIRPLOT_HUE_COMBO)
        dpg.configure_item(TAG_MVA_PAIRPLOT_HUE_COMBO, items=cat_cols_hue)
        dpg.set_value(TAG_MVA_PAIRPLOT_HUE_COMBO, curr_hue if curr_hue in cat_cols_hue else "")

    target_var = main_callbacks['get_selected_target_variable']()
    if dpg.does_item_exist(TAG_MVA_TARGET_INFO_TEXT):
        if current_df is not None and target_var and target_var in all_cols:
            dpg.set_value(TAG_MVA_TARGET_INFO_TEXT, f"Target: '{target_var}' (Type: {main_callbacks['get_selected_target_variable_type']()})")
            if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO): dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=[c for c in all_cols if c != target_var])
        else:
            dpg.set_value(TAG_MVA_TARGET_INFO_TEXT, "Load data & select target variable.")
            if dpg.does_item_exist(TAG_MVA_TARGET_FEATURE_COMBO): dpg.configure_item(TAG_MVA_TARGET_FEATURE_COMBO, items=[]); dpg.set_value(TAG_MVA_TARGET_FEATURE_COMBO, "")
    if dpg.does_item_exist(TAG_MVA_CAT_EDA_VAR_SELECTOR): dpg.configure_item(TAG_MVA_CAT_EDA_VAR_SELECTOR, items=all_cat_cols)
    
    if current_df is None:
        for area, msg in [(TAG_MVA_CORR_RESULTS_GROUP, "Load data."), (TAG_MVA_PAIRPLOT_RESULTS_GROUP, "Load data."),
                           (TAG_MVA_TARGET_RESULTS_GROUP, "Load data & select target."), (TAG_MVA_CAT_EDA_RESULTS_GROUP, "Load data.")]:
            if dpg.does_item_exist(area): dpg.delete_item(area, children_only=True); dpg.add_text(msg, parent=area)

def reset_mva_ui_defaults(): # 함수 이름 변경
    if not dpg.is_dearpygui_running(): return
    for selector, default_val_type in [(TAG_MVA_PAIRPLOT_VAR_SELECTOR, list), (TAG_MVA_PAIRPLOT_HUE_COMBO, str),
                                       (TAG_MVA_TARGET_FEATURE_COMBO, str), (TAG_MVA_CAT_EDA_VAR_SELECTOR, list)]:
        if dpg.does_item_exist(selector):
            dpg.configure_item(selector, items=[] if default_val_type == list else [""]) # 빈 리스트/문자열로 초기화
            dpg.set_value(selector, [] if default_val_type == list else "")
    for area, msg in [(TAG_MVA_CORR_RESULTS_GROUP, "Run analysis."), (TAG_MVA_PAIRPLOT_RESULTS_GROUP, "Run analysis."),
                       (TAG_MVA_TARGET_RESULTS_GROUP, "Load data & select target."), (TAG_MVA_CAT_EDA_RESULTS_GROUP, "Run analysis.")]:
        if dpg.does_item_exist(area): dpg.delete_item(area, children_only=True); dpg.add_text(msg, parent=area)