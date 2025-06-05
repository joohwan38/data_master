# step_02b_mva.py
import matplotlib
matplotlib.use('Agg')
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
# utils 임포트 방식 변경 또는 추가
import utils # AI 분석 함수 및 기타 유틸리티 사용
import functools # partial 함수 사용을 위해 추가

import seaborn as sns
import matplotlib.pyplot as plt
import umap
import io
from PIL import Image
import traceback
# ollama_analyzer는 utils.py에서 임포트하여 사용하므로 여기서는 직접 임포트 필요 없음

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='umap')
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')

# --- MVA UI 태그 정의 ---
TAG_MVA_STEP_GROUP = "mva_step_group"
TAG_MVA_MAIN_TAB_BAR = "mva_main_tab_bar"
TAG_MVA_CORR_TAB = "mva_corr_tab"
TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX = "mva_corr_umap_group_by_target_checkbox"
TAG_MVA_CORR_RUN_BUTTON = "mva_corr_run_button"
TAG_MVA_CORR_RESULTS_GROUP = "mva_corr_results_group"
TAG_MVA_PAIRPLOT_TAB = "mva_pairplot_tab"
TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX = "mva_pairplot_group_by_target_checkbox"
TAG_MVA_PAIRPLOT_RUN_BUTTON = "mva_pairplot_run_button"
TAG_MVA_PAIRPLOT_RESULTS_GROUP = "mva_pairplot_results_group"
TAG_MVA_CAT_EDA_TAB = "mva_cat_corr_tab"
TAG_MVA_CAT_EDA_RUN_BUTTON = "mva_cat_eda_run_button"
TAG_MVA_CAT_EDA_RESULTS_GROUP = "mva_cat_eda_results_group"
# TAG_MVA_MODULE_ALERT_MODAL = "mva_module_specific_alert_modal" # 사용되지 않으면 제거 가능
# TAG_MVA_MODULE_ALERT_TEXT = "mva_module_specific_alert_text" # 사용되지 않으면 제거 가능

_mva_main_app_callbacks: Dict[str, Any] = {}
_mva_util_funcs: Dict[str, Any] = {} # 이 변수는 이제 utils.py의 함수를 직접 호출하므로 필요성이 줄어들 수 있음

def get_mva_settings_for_saving() -> Dict[str, Any]:
    settings = {'corr_tab': {}, 'pairplot_tab': {}, 'cat_eda_tab': {}}
    if dpg.is_dearpygui_running(): # DPG 실행 중일 때만 UI 요소 접근
        if dpg.does_item_exist(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX):
            settings['corr_tab']['umap_group_by_target'] = dpg.get_value(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX)
        if dpg.does_item_exist(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX):
            settings['pairplot_tab']['group_by_target'] = dpg.get_value(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX)
        
        if dpg.does_item_exist(TAG_MVA_MAIN_TAB_BAR):
            settings['active_mva_tab_label'] = None # 기본값
            try:
                active_tab_tag = dpg.get_value(TAG_MVA_MAIN_TAB_BAR)
                if active_tab_tag and dpg.does_item_exist(active_tab_tag):
                    settings['active_mva_tab_tag'] = active_tab_tag
            except Exception: # DPG 관련 오류 발생 시 대체 로직
                for child in dpg.get_item_children(TAG_MVA_MAIN_TAB_BAR, 1): # type: ignore
                    item_config = dpg.get_item_configuration(child)
                    if item_config and item_config.get("show", False):
                        settings['active_mva_tab_label'] = item_config.get("label")
                        break
    return settings

def apply_mva_settings_from_loaded(settings: Dict[str, Any], current_df: Optional[pd.DataFrame], main_callbacks: Dict[str, Any]):
    if not dpg.is_dearpygui_running(): return

    corr_tab_settings = settings.get('corr_tab', {})
    if 'umap_group_by_target' in corr_tab_settings and dpg.does_item_exist(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX):
        dpg.set_value(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX, corr_tab_settings['umap_group_by_target'])

    pairplot_tab_settings = settings.get('pairplot_tab', {})
    if 'group_by_target' in pairplot_tab_settings and dpg.does_item_exist(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX):
        dpg.set_value(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX, pairplot_tab_settings['group_by_target'])
        
    active_tab_tag_setting = settings.get('active_mva_tab_tag')
    if active_tab_tag_setting and dpg.does_item_exist(TAG_MVA_MAIN_TAB_BAR) and dpg.does_item_exist(active_tab_tag_setting):
        try: dpg.set_value(TAG_MVA_MAIN_TAB_BAR, active_tab_tag_setting)
        except Exception: # DPG 관련 오류 발생 시 대체 로직
            active_tab_label_setting = settings.get('active_mva_tab_label')
            if active_tab_label_setting:
                 for child_iter in dpg.get_item_children(TAG_MVA_MAIN_TAB_BAR, 1): # type: ignore
                    cfg = dpg.get_item_configuration(child_iter)
                    if cfg and cfg.get("label") == active_tab_label_setting:
                        try: dpg.set_value(TAG_MVA_MAIN_TAB_BAR, child_iter); break
                        except: pass 
    
    for area_tag in [TAG_MVA_CORR_RESULTS_GROUP, TAG_MVA_PAIRPLOT_RESULTS_GROUP, TAG_MVA_CAT_EDA_RESULTS_GROUP]:
        if dpg.does_item_exist(area_tag):
            dpg.delete_item(area_tag, children_only=True)
            dpg.add_text("Settings loaded. Click 'Run/Generate' button.", parent=area_tag)
    update_ui(current_df, main_callbacks)


def _plot_to_dpg_texture_data(fig: plt.Figure, desired_dpi: int = 100) -> Tuple[Optional[str], int, int, Optional[bytes]]:
    img_data_buf = io.BytesIO()
    img_bytes_data = None
    texture_tag = None
    img_width, img_height = 0, 0
    TEXTURE_REGISTRY_TAG = "primary_texture_registry"

    if not dpg.is_dearpygui_running(): # DPG 실행 중이 아니면 아무것도 하지 않음
        plt.close(fig) # 리소스 해제
        return None, 0, 0, None

    if not dpg.does_item_exist(TEXTURE_REGISTRY_TAG):
        print(f"Warning: Texture registry '{TEXTURE_REGISTRY_TAG}' not found. Creating it now in step_02b_mva.")
        dpg.add_texture_registry(tag=TEXTURE_REGISTRY_TAG)
        
    try:
        fig.savefig(img_data_buf, format="png", bbox_inches='tight', dpi=desired_dpi)
        img_data_buf.seek(0)
        img_bytes_data = img_data_buf.getvalue()

        pil_image = Image.open(io.BytesIO(img_bytes_data))
        if pil_image.mode != 'RGBA': pil_image = pil_image.convert('RGBA')
        img_width, img_height = pil_image.size

        if img_width == 0 or img_height == 0:
            print(f"Error: Plot image has zero dimension ({img_width}x{img_height}). Cannot create texture.")
            return None, 0, 0, img_bytes_data

        texture_data_np = np.array(pil_image).astype(np.float32) / 255.0
        texture_data_flat_list = texture_data_np.ravel().tolist()
        texture_tag = dpg.generate_uuid()

        dpg.add_static_texture(
            width=img_width,
            height=img_height,
            default_value=texture_data_flat_list,
            tag=texture_tag,
            parent=TEXTURE_REGISTRY_TAG
        )
        return texture_tag, img_width, img_height, img_bytes_data
    except SystemError as se:
        print(f"SystemError converting plot to DPG texture: {se}")
        print(f"Traceback (SystemError in _plot_to_dpg_texture_data): {traceback.format_exc()}")
        return texture_tag, img_width, img_height, img_bytes_data
    except Exception as e:
        print(f"General error converting plot to DPG texture: {e}")
        print(f"Traceback (General Error in _plot_to_dpg_texture_data): {traceback.format_exc()}")
        return None, 0, 0, img_bytes_data
    finally:
        plt.close(fig)

def _display_dpg_image(parent_group: str, texture_tag: Optional[str], tex_w: int, tex_h: int, max_w: int = 850):
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(parent_group): return

    if texture_tag and tex_w > 0 and tex_h > 0:
        display_w, display_h = tex_w, tex_h
        if tex_w > max_w:
            display_w = max_w
            display_h = int(tex_h * (max_w / tex_w))
        dpg.add_image(texture_tag, parent=parent_group, width=int(display_w), height=int(display_h))
    elif texture_tag: # tex_w 또는 tex_h가 0인 경우
        dpg.add_text("Failed: Image has zero dimension.", parent=parent_group, color=(255,100,0))
    else: # texture_tag가 None인 경우
        dpg.add_text("Failed: Image texture not generated.", parent=parent_group, color=(255,0,0))


def _mva_run_correlation_analysis(df: pd.DataFrame, u_funcs: dict, callbacks: dict):
    global _mva_main_app_callbacks
    results_group = TAG_MVA_CORR_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group): return
    dpg.delete_item(results_group, children_only=True)
    if df is None: dpg.add_text("Load data first.", parent=results_group); return
    
    num_cols = utils._get_numeric_cols(df)
    if len(num_cols) < 2:
        dpg.add_text("Need at least 2 numeric columns.", parent=results_group); return

    MAX_VARS_CM = 15
    target_var = callbacks['get_selected_target_variable']()
    target_var_type = callbacks['get_selected_target_variable_type']()
    corr_abs_mat_full = df[num_cols].corr().abs()

    # --- Clustermap 1: 상호 높은 상관관계 변수 ---
    dpg.add_text(f"Clustermap 1: Top {MAX_VARS_CM} Numeric Variables - Highest Pairwise Correlations", parent=results_group, color=(255,255,0))
    img_bytes_cm1 = None
    tex_tag1, w1, h1 = None, 0, 0
    vars_cm1 = []

    try:
        vars_cm1 = num_cols if len(num_cols) <= MAX_VARS_CM else \
                     [v for v, _ in sorted({col: corr_abs_mat_full.loc[col, corr_abs_mat_full.columns != col].max() if not corr_abs_mat_full.loc[col, corr_abs_mat_full.columns != col].empty else 0 for col in num_cols}.items(), key=lambda item: item[1], reverse=True)[:MAX_VARS_CM]]

        if len(vars_cm1) >= 2:
            sub_corr1 = df[vars_cm1].corr().fillna(0).replace([np.inf, -np.inf], 0)
            n_vars1 = len(vars_cm1); fs1 = max(0.7, 1.2 - n_vars1 * 0.02); fsize1 = (max(7, n_vars1 * 0.8), max(6, n_vars1 * 0.7))
            sns.set_theme(style="whitegrid", font_scale=fs1)
            cm1_plot_obj = sns.clustermap(sub_corr1, annot=True, cmap="RdYlBu_r", fmt=".2f", linewidths=.5, vmin=-1, vmax=1, figsize=fsize1, dendrogram_ratio=0.12, cbar_kws={'shrink': .65})
            cm1_plot_obj.fig.suptitle(f"Top {n_vars1} Inter-correlated Variables", fontsize=15 if fs1 > 0.6 else 13, y=1.02)
            
            plot_result_cm1 = _plot_to_dpg_texture_data(cm1_plot_obj.fig, desired_dpi=95)
            if plot_result_cm1 and len(plot_result_cm1) == 4:
                tex_tag1, w1, h1, img_bytes_cm1 = plot_result_cm1
            
            with dpg.group(horizontal=False, parent=results_group):
                _display_dpg_image(dpg.last_item(), tex_tag1, w1, h1, max_w=700)
                if img_bytes_cm1 and tex_tag1:
                    chart_name_cm1 = f"Clustermap1_Top_{len(vars_cm1)}_InterCorrelated"
                    ai_button_cm1_tag = dpg.generate_uuid()
                    action_for_cm1_button_partial = functools.partial(
                        utils.confirm_and_run_ai_analysis,
                        img_bytes_cm1, 
                        chart_name_cm1, 
                        ai_button_cm1_tag, 
                        _mva_main_app_callbacks 
                    )
                    dpg.add_button(
                        label="💡 Analyze with AI", 
                        tag=ai_button_cm1_tag, 
                        callback=lambda sender, app_data, user_data: action_for_cm1_button_partial() # 이미 올바르게 되어 있음
                    )
                    dpg.add_spacer(height=5)
        else:
            dpg.add_text("Not enough variables for this clustermap.", parent=results_group)
    except Exception as e:
        dpg.add_text(f"Error (CM1): {e}", parent=results_group,color=(255,0,0))
        print(f"Error (CM1 traceback): {traceback.format_exc()}")
    dpg.add_separator(parent=results_group)

    # --- Clustermap 2: 타겟 연관 변수 ---
    clustermap2_title_text = f"Clustermap 2: Top {MAX_VARS_CM} Numeric Variables"
    img_bytes_cm2 = None
    tex_tag2, w2, h2 = None, 0, 0
    vars_for_clustermap2 = []
    selection_method_description = "Not determined"

    if target_var and target_var in df.columns:
        if target_var_type == "Continuous" and target_var in num_cols:
            selection_method_description = f"based on Pearson correlation with Continuous target '{target_var}'"
            other_numeric_cols_for_cont_target = [col for col in num_cols if col != target_var]
            if other_numeric_cols_for_cont_target:
                relevance_scores_cont = utils.calculate_feature_target_relevance(df, target_var, target_var_type, other_numeric_cols_for_cont_target, callbacks)
                top_other_vars = [var_name for var_name, score in relevance_scores_cont[:MAX_VARS_CM - 1]]
                vars_for_clustermap2 = [target_var] + top_other_vars
                vars_for_clustermap2 = list(dict.fromkeys(vars_for_clustermap2))[:MAX_VARS_CM]
            else: vars_for_clustermap2 = [target_var]; selection_method_description = f"target '{target_var}' is the only numeric variable."
        elif target_var_type == "Categorical":
            selection_method_description = f"based on ANOVA F-statistic with Categorical target '{target_var}'"
            target_categories = df[target_var].dropna().unique()
            if 2 <= len(target_categories) <= 10:
                features_to_check_anova = [col for col in num_cols if col != target_var]
                if features_to_check_anova:
                    relevance_scores_cat = utils.calculate_feature_target_relevance(df, target_var, target_var_type, features_to_check_anova, callbacks)
                    vars_for_clustermap2 = [var_name for var_name, score in relevance_scores_cat[:MAX_VARS_CM]]
                else: selection_method_description = "no numeric features to analyze with categorical target."
            else: selection_method_description = f"target '{target_var}' has {len(target_categories)} categories (requires 2-10 for ANOVA selection)."
        else: selection_method_description = f"target '{target_var}' (type: '{target_var_type}') not suitable for selection."
        clustermap2_title_text += f" {selection_method_description}"
    else:
        clustermap2_title_text += " (Skipped: No target variable selected)"
        selection_method_description = "No target selected."
    
    dpg.add_text(clustermap2_title_text, parent=results_group, color=(255,255,0) if len(vars_for_clustermap2) >=2 else (200,200,0) )

    if len(vars_for_clustermap2) >= 2:
        try:
            sub_corr2 = df[vars_for_clustermap2].corr().fillna(0).replace([np.inf, -np.inf], 0)
            n_vars2 = len(vars_for_clustermap2); fs2 = max(0.7, 1.2 - n_vars2 * 0.02); fsize2 = (max(7, n_vars2 * 0.8), max(6, n_vars2 * 0.7))
            sns.set_theme(style="whitegrid", font_scale=fs2)
            cm2_plot_obj = sns.clustermap(sub_corr2, annot=True, cmap="RdYlBu_r", fmt=".2f", linewidths=.5, vmin=-1, vmax=1, figsize=fsize2, dendrogram_ratio=0.12, cbar_kws={'shrink': .65})
            cm2_plot_obj.fig.suptitle(f"Clustermap of Top {n_vars2} Target-Associated Numeric Variables", fontsize=15 if fs2 > 0.7 else 12, y=1.03)

            plot_result_cm2 = _plot_to_dpg_texture_data(cm2_plot_obj.fig, desired_dpi=95)
            if plot_result_cm2 and len(plot_result_cm2) == 4:
                tex_tag2, w2, h2, img_bytes_cm2 = plot_result_cm2

            with dpg.group(horizontal=False, parent=results_group):
                _display_dpg_image(dpg.last_item(), tex_tag2, w2, h2, max_w=700)
                if img_bytes_cm2 and tex_tag2:
                    chart_name_cm2 = f"Clustermap2_Target_{target_var if target_var else 'N_A'}"
                    ai_button_cm2_tag = dpg.generate_uuid()
                    action_for_cm2_button = functools.partial(
                        utils.confirm_and_run_ai_analysis,
                        img_bytes_cm2, chart_name_cm2, ai_button_cm2_tag, _mva_main_app_callbacks
                    )
                    dpg.add_button(label="💡 Analyze with AI", tag=ai_button_cm2_tag, width=150, height=25,
                                   # 수정된 부분: 람다 함수로 감싸기
                                   callback=lambda sender, app_data, user_data: action_for_cm2_button())
                    dpg.add_spacer(height=5)
        except Exception as e_cm2_render:
            dpg.add_text(f"Error rendering Clustermap 2: {e_cm2_render}", parent=results_group, color=(255,0,0)); print(traceback.format_exc())
    elif target_var and target_var in df.columns :
        if not (selection_method_description == "No target selected." or "not suitable for this selection" in selection_method_description or "not suitable." in selection_method_description):
             dpg.add_text(f"-> Not enough numeric variables found based on '{target_var}' for Clustermap 2.", parent=results_group, color=(200,200,0))
    dpg.add_separator(parent=results_group)

    # --- UMAP ---
    dpg.add_text("UMAP 2D Visualization of All Numeric Variables:", parent=results_group, color=(255,255,0))
    img_bytes_umap = None
    tex_tag_umap, w_umap, h_umap = None, 0, 0

    try:
        umap_prep_df = df[num_cols].copy()
        for col in umap_prep_df.columns:
            if umap_prep_df[col].isnull().any() and pd.api.types.is_numeric_dtype(umap_prep_df[col]):
                umap_prep_df[col] = umap_prep_df[col].fillna(umap_prep_df[col].median())
        if umap_prep_df.shape[0] < 2 or umap_prep_df.shape[1] < 2:
            dpg.add_text("Not enough data/features for UMAP.", parent=results_group)
        else:
            group_umap_cb = dpg.get_value(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX) if dpg.does_item_exist(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX) else False
            umap_hue_values = None; umap_legend_hndls = None; actual_umap_hue_var = None
            cmap_for_umap = 'Spectral'; n_cats_umap = 0
            if group_umap_cb and target_var and target_var in df.columns:
                target_s_for_hue = df[target_var].copy()
                if target_s_for_hue.isnull().any(): target_s_for_hue = target_s_for_hue.astype(str).fillna("Missing")
                unique_hue_count = target_s_for_hue.nunique(dropna=False)
                MAX_HUE_CATS_UMAP = 10
                if 2 <= unique_hue_count <= MAX_HUE_CATS_UMAP:
                    actual_umap_hue_var = target_var
                    aligned_hue_series = target_s_for_hue.loc[umap_prep_df.index.intersection(target_s_for_hue.index)]
                    umap_hue_values = aligned_hue_series.astype('category').cat.codes
                    cats_umap = aligned_hue_series.astype('category').cat.categories
                    n_cats_umap = len(cats_umap)
                    if n_cats_umap > 0 : cmap_for_umap = plt.colormaps.get_cmap('Spectral').resampled(n_cats_umap) # type: ignore
                    umap_legend_hndls = [plt.Line2D([0],[0], marker='o', color='w', label=str(c)[:15], markerfacecolor=cmap_for_umap(i) if n_cats_umap > 0 else 'gray', markersize=6) for i, c in enumerate(cats_umap)] # type: ignore
                    dpg.add_text(f"UMAP: Grouping by target '{target_var}'.", parent=results_group, color=(180,180,180))
                else:
                    dpg.add_text(f"UMAP Hue: Target '{target_var}' has {unique_hue_count} unique values. Hue disabled (requires 2-{MAX_HUE_CATS_UMAP}).", parent=results_group, color=(200,200,0))
            n_neigh = min(15, umap_prep_df.shape[0] - 1) if umap_prep_df.shape[0] > 1 else 1
            if n_neigh <= 0: n_neigh = 1
            reducer_umap = umap.UMAP(n_neighbors=n_neigh, n_components=2, random_state=42, min_dist=0.05, spread=1.0)
            umap_embedding = reducer_umap.fit_transform(umap_prep_df)
            plt.style.use('seaborn-v0_8-whitegrid')
            fig_umap_plot = plt.figure(figsize=(8, 6.5))
            scatter_kwargs_umap = {'s': 15, 'alpha': 0.7}
            if umap_hue_values is not None:
                scatter_kwargs_umap['c'] = umap_hue_values
                scatter_kwargs_umap['cmap'] = cmap_for_umap
            plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], **scatter_kwargs_umap)

            if umap_legend_hndls and umap_hue_values is not None:
                plt.legend(handles=umap_legend_hndls, title=str(actual_umap_hue_var)[:15], fontsize=7.5, loc='best', frameon=True, shadow=True)
            plt.title("UMAP Projection of Numeric Variables", fontsize=11); plt.xlabel("UMAP Dimension 1", fontsize=9); plt.ylabel("UMAP Dimension 2", fontsize=9)
            plt.xticks(fontsize=8.5); plt.yticks(fontsize=7.5); plt.tight_layout()
            
            plot_result_umap = _plot_to_dpg_texture_data(fig_umap_plot, desired_dpi=100)
            if plot_result_umap and len(plot_result_umap) == 4:
                tex_tag_umap, w_umap, h_umap, img_bytes_umap = plot_result_umap
            plt.style.use('default')

            with dpg.group(horizontal=False, parent=results_group):
                _display_dpg_image(dpg.last_item(), tex_tag_umap, w_umap, h_umap, max_w=750)
                if img_bytes_umap and tex_tag_umap:
                    chart_name_umap = "UMAP_Projection_Numeric_Variables"
                    ai_button_umap_tag = dpg.generate_uuid()
                    action_for_umap_button = functools.partial(
                        utils.confirm_and_run_ai_analysis,
                        img_bytes_umap, chart_name_umap, ai_button_umap_tag, _mva_main_app_callbacks
                    )
                    dpg.add_button(label="💡 Analyze with AI", tag=ai_button_umap_tag, width=150, height=25,
                                   # 수정된 부분: 람다 함수로 감싸기
                                   callback=lambda sender, app_data, user_data: action_for_umap_button())
                    dpg.add_spacer(height=5)
    except ImportError:
        dpg.add_text("UMAP-learn not installed.",parent=results_group,color=(255,100,0))
    except Exception as e:
        dpg.add_text(f"Error (UMAP): {e}",parent=results_group,color=(255,0,0))
        print(f"Error (UMAP traceback): {traceback.format_exc()}")

def _mva_run_pair_plot_analysis(df: pd.DataFrame, u_funcs: dict, callbacks: dict):
    # Pair Plot은 현재 AI 분석 버튼이 없으므로, 기존 로직 유지 또는 필요시 추가
    res_group = TAG_MVA_PAIRPLOT_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(res_group): return
    dpg.delete_item(res_group, children_only=True)
    if df is None: dpg.add_text("Load data first.", parent=res_group); return

    MAX_VARS_PP = 5
    num_cols_all = utils._get_numeric_cols(df)
    if len(num_cols_all) < 2:
        dpg.add_text("Need at least 2 numeric columns for Pair Plot.", parent=res_group); return
    
    vars_for_pp = []
    if len(num_cols_all) <= MAX_VARS_PP:
        vars_for_pp = num_cols_all
        dpg.add_text(f"Pair Plot: Using all {len(num_cols_all)} available numeric variables.", parent=res_group, color=(180,180,180))
    else:
        dpg.add_text(f"Pair Plot: Selecting top {MAX_VARS_PP} numeric variables based on highest pairwise correlations.", parent=res_group, color=(180,180,180))
        corr_abs_matrix_pp = df[num_cols_all].corr().abs()
        max_corrs_per_var_pp = {}
        for col_pp in num_cols_all:
            other_cols_corr_pp = corr_abs_matrix_pp.loc[col_pp, corr_abs_matrix_pp.columns != col_pp]
            max_corrs_per_var_pp[col_pp] = other_cols_corr_pp.max() if not other_cols_corr_pp.empty else 0
        sorted_vars_by_max_corr_pp = sorted(max_corrs_per_var_pp.items(), key=lambda item: item[1], reverse=True)
        vars_for_pp = [var_name for var_name, _ in sorted_vars_by_max_corr_pp[:MAX_VARS_PP]]

    if len(vars_for_pp) < 2:
        dpg.add_text("Not enough numeric variables selected/available for Pair Plot after filtering.", parent=res_group); return
        
    hue_var_pp = None
    group_by_target_pp_cb = dpg.get_value(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX) if dpg.does_item_exist(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX) else False
    target_var_for_hue = callbacks['get_selected_target_variable']()

    if group_by_target_pp_cb and target_var_for_hue and target_var_for_hue in df.columns:
        unique_target_pp_count = df[target_var_for_hue].nunique(dropna=True)
        if 2 <= unique_target_pp_count <= 7: # Pairplot은 hue 카테고리 수가 적을 때 유용
            hue_var_pp = target_var_for_hue
            dpg.add_text(f"Pair Plot: Using target '{target_var_for_hue}' for Hue.", parent=res_group, color=(180,180,180))
        else:
            dpg.add_text(f"Pair Plot Hue: Target '{target_var_for_hue}' has {unique_target_pp_count} unique values. Hue disabled (requires 2-7).", parent=res_group, color=(200,200,0))
    
    title_pp_str = f"Pair Plot (Top {len(vars_for_pp)} Vars): {', '.join(vars_for_pp)}" + (f" (Hue: {hue_var_pp})" if hue_var_pp else "")
    dpg.add_text(title_pp_str, parent=res_group, color=(255,255,0))
    try:
        pp_df = df.copy() # 원본 DataFrame 복사
        n_vars_pp_plot = len(vars_for_pp)
        height_per_subplot = max(1.0, min(2.8, 13.0 / n_vars_pp_plot if n_vars_pp_plot > 0 else 2.8))
        font_scale_val_pp = max(0.7, 1.0 - n_vars_pp_plot * 0.03)

        sns.set_theme(style="ticks", font_scale=font_scale_val_pp)
        
        cols_for_seaborn_pp_grid = vars_for_pp[:] 
        if hue_var_pp and hue_var_pp not in cols_for_seaborn_pp_grid:
            cols_for_seaborn_pp_grid.append(hue_var_pp)
        
        pp_subset_df_for_grid = pp_df[cols_for_seaborn_pp_grid].copy()

        if hue_var_pp and hue_var_pp in pp_subset_df_for_grid.columns:
             if not pd.api.types.is_string_dtype(pp_subset_df_for_grid[hue_var_pp]) and \
                not pd.api.types.is_categorical_dtype(pp_subset_df_for_grid[hue_var_pp]):
                try: 
                    if pp_subset_df_for_grid[hue_var_pp].nunique(dropna=False) > 10: 
                        pp_subset_df_for_grid[hue_var_pp] = pp_subset_df_for_grid[hue_var_pp].astype(str)
                    else: 
                        pp_subset_df_for_grid[hue_var_pp] = pd.Categorical(pp_subset_df_for_grid[hue_var_pp])
                except: 
                    pp_subset_df_for_grid[hue_var_pp] = pp_subset_df_for_grid[hue_var_pp].astype(str)
        
        pp_subset_df_for_grid.dropna(subset=vars_for_pp, inplace=True) 
        if pp_subset_df_for_grid.empty or len(pp_subset_df_for_grid) < 2:
             dpg.add_text("Not enough data after NaN handling for Pair Plot.", parent=res_group, color=(255,100,0)); return

        g = sns.PairGrid(
            data=pp_subset_df_for_grid, 
            vars=vars_for_pp, 
            hue=hue_var_pp if hue_var_pp in pp_subset_df_for_grid.columns else None, 
            height=height_per_subplot, 
            aspect=1.2, 
            dropna=True 
        )
        g.map_upper(sns.scatterplot, s=12 if n_vars_pp_plot <=7 else 8, alpha=0.55, edgecolor=None)
        def kdeplot_lower_wrapper(x, y, **kwargs): # type: ignore
            if x.nunique() >= 2 and y.nunique() >= 2 and len(x) >=2 :
                try: sns.kdeplot(x=x, y=y, levels=4, fill=True, alpha=0.45, linewidths=0.9, **kwargs)
                except Exception: pass 
        g.map_lower(kdeplot_lower_wrapper)
        def kdeplot_diag_wrapper(x, **kwargs): # type: ignore
            if x.nunique() >= 2 and len(x) >=2:
                try: sns.kdeplot(x=x, fill=True, alpha=0.55, linewidth=1.1, **kwargs)
                except Exception: pass 
        g.map_diag(kdeplot_diag_wrapper)
        
        if hue_var_pp and hue_var_pp in pp_subset_df_for_grid.columns:
            g.add_legend(title=str(hue_var_pp)[:15], fontsize=11 if n_vars_pp_plot <=7 else 10, title_fontsize=11 if n_vars_pp_plot <=7 else 10)
        g.fig.suptitle(f"Pair Plot (Top {len(vars_for_pp)} Vars)" + (f" (Hue: {hue_var_pp})" if hue_var_pp and hue_var_pp in pp_subset_df_for_grid.columns else ""), y=1.01, fontsize=13)
        
        plot_result_pp = _plot_to_dpg_texture_data(g.fig, desired_dpi=80)
        tex_tag_pp_img, w_pp, h_pp, img_bytes_pp = None, 0, 0, None
        if plot_result_pp and len(plot_result_pp) == 4:
            tex_tag_pp_img, w_pp, h_pp, img_bytes_pp = plot_result_pp
        
        with dpg.group(horizontal=False, parent=res_group):
            _display_dpg_image(dpg.last_item(), tex_tag_pp_img, w_pp, h_pp, max_w=850)
            # --- Pair Plot AI 분석 버튼 활성화 ---
            if img_bytes_pp and tex_tag_pp_img:
                chart_name_pp = f"PairPlot_Top_{len(vars_for_pp)}_Vars" + (f"_Hue_{hue_var_pp}" if hue_var_pp else "")
                ai_button_pp_tag = dpg.generate_uuid()
                action_for_pp_button = functools.partial(
                    utils.confirm_and_run_ai_analysis,
                    img_bytes_pp, chart_name_pp, ai_button_pp_tag, _mva_main_app_callbacks
                )
                dpg.add_button(label="💡 Analyze Pair Plot", tag=ai_button_pp_tag, 
                               callback=lambda sender, app_data, user_data: action_for_pp_button()) # 람다로 감싸기
                dpg.add_spacer(height=5)

    except ImportError:
        dpg.add_text("Seaborn or Matplotlib is not installed.", parent=res_group, color=(255,100,0))
    except Exception as e:
        dpg.add_text(f"Error creating Pair Plot: {e}", parent=res_group, color=(255,0,0))
        print(traceback.format_exc())
        
def _mva_run_cat_corr_analysis(df: pd.DataFrame, u_funcs: dict, callbacks: dict):
    res_group = TAG_MVA_CAT_EDA_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(res_group): return
    dpg.delete_item(res_group, children_only=True)
    if df is None: dpg.add_text("Load data first.", parent=res_group); return
    
    MAX_VARS_CV = 15
    all_cat_cols_cv = utils._get_categorical_cols(df, max_unique_for_cat=35, main_callbacks=callbacks)
    if len(all_cat_cols_cv) < 2:
        dpg.add_text("Need at least 2 categorical columns.", parent=res_group); return

    vars_for_cv_final = []
    if len(all_cat_cols_cv) <= MAX_VARS_CV: vars_for_cv_final = all_cat_cols_cv
    else:
        dpg.add_text(f"Selecting top {MAX_VARS_CV} categorical vars by max pairwise Cramer's V.", parent=res_group, color=(180,180,180))
        cv_pairs = {}
        for i in range(len(all_cat_cols_cv)):
            for j in range(i + 1, len(all_cat_cols_cv)):
                v1_cv, v2_cv = all_cat_cols_cv[i], all_cat_cols_cv[j]
                val = utils.calculate_cramers_v(df[v1_cv], df[v2_cv])
                if pd.notna(val): cv_pairs[(v1_cv,v2_cv)] = val
        max_cv_per_var = {var: 0.0 for var in all_cat_cols_cv}
        for (v_a, v_b), v_cv in cv_pairs.items():
            max_cv_per_var[v_a] = max(max_cv_per_var[v_a], v_cv)
            max_cv_per_var[v_b] = max(max_cv_per_var[v_b], v_cv)
        vars_for_cv_final = [var for var, _ in sorted(max_cv_per_var.items(), key=lambda item: item[1], reverse=True)[:MAX_VARS_CV]]

    if len(vars_for_cv_final) < 2:
        dpg.add_text("Not enough variables for Cramer's V clustermap.", parent=res_group); return
        
    dpg.add_text(f"Cramer's V Clustermap (Top {len(vars_for_cv_final)} Associated Variables)", parent=res_group, color=(255,255,0))
    try:
        cv_mat_final = pd.DataFrame(np.zeros((len(vars_for_cv_final), len(vars_for_cv_final))), columns=vars_for_cv_final, index=vars_for_cv_final)
        for r, r_name in enumerate(vars_for_cv_final):
            for c, c_name in enumerate(vars_for_cv_final):
                cv_mat_final.iloc[r,c] = 1.0 if r == c else (utils.calculate_cramers_v(df[r_name], df[c_name]) or 0)

        if cv_mat_final.shape[0] < 2: dpg.add_text("Not enough data for clustermap.", parent=res_group); return
        cv_mat_final = cv_mat_final.replace([np.inf, -np.inf], 0).fillna(0)
        n_cv = len(vars_for_cv_final)
        fs_cv = max(0.9, 1.4 - n_cv * 0.02)
        fsize_cv = (max(9, n_cv * 0.9), max(9, n_cv * 0.9))

        sns.set_theme(style="white", font_scale=fs_cv)
        cm_cv_plot_obj = sns.clustermap(
            cv_mat_final, annot=True, cmap="Blues", fmt=".2f", linewidths=.5,
            vmin=0, vmax=1, figsize=fsize_cv, dendrogram_ratio=0.08,
            cbar_kws={'shrink': .6, 'ticks': [0, 0.25, 0.5, 0.75, 1]}
        )
        cm_cv_plot_obj.fig.suptitle("Cramer's V Association Clustermap", fontsize=15 if fs_cv > 0.8 else 12, y=1.02)
        
        plot_result_cv = _plot_to_dpg_texture_data(cm_cv_plot_obj.fig, desired_dpi=95)
        tex_tag_cv, w_cv, h_cv, img_bytes_cv = None,0,0, None
        if plot_result_cv and len(plot_result_cv) == 4:
             tex_tag_cv, w_cv, h_cv, img_bytes_cv = plot_result_cv

        with dpg.group(horizontal=False, parent=res_group):
            _display_dpg_image(dpg.last_item(), tex_tag_cv, w_cv, h_cv, max_w=700)
            # --- Cramer's V Clustermap AI 분석 버튼 활성화 ---
            if img_bytes_cv and tex_tag_cv:
                chart_name_cv = f"CramersV_Clustermap_Top_{len(vars_for_cv_final)}_Vars"
                ai_button_cv_tag = dpg.generate_uuid()
                action_for_cv_button = functools.partial(
                    utils.confirm_and_run_ai_analysis,
                    img_bytes_cv, chart_name_cv, ai_button_cv_tag, _mva_main_app_callbacks
                )
                dpg.add_button(label="💡 Analyze Cramer's V", tag=ai_button_cv_tag, 
                               callback=lambda sender, app_data, user_data: action_for_cv_button()) # 람다로 감싸기
                dpg.add_spacer(height=5)

    except ImportError: dpg.add_text("Seaborn not installed.",parent=res_group,color=(255,100,0))
    except Exception as e: dpg.add_text(f"Error (CramerV CM): {e}",parent=res_group,color=(255,0,0)); print(traceback.format_exc())


def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _mva_main_app_callbacks, _mva_util_funcs
    _mva_main_app_callbacks = main_callbacks
    # _mva_util_funcs는 이제 main_callbacks['get_util_funcs']()를 통해 utils.py의 함수를 사용하므로,
    # 이 모듈 내에서 별도로 유지할 필요가 줄어듭니다.
    # 필요하다면 _mva_util_funcs = main_callbacks.get('get_util_funcs', lambda: {})() 로 유지할 수 있습니다.

    with dpg.group(tag=TAG_MVA_STEP_GROUP, parent=parent_container_tag): # show=False 기본값으로 시작
        with dpg.tab_bar(tag=TAG_MVA_MAIN_TAB_BAR):
            with dpg.tab(label="Correlation & UMAP (Numeric)", tag=TAG_MVA_CORR_TAB):
                dpg.add_text("Displays clustermaps of numeric correlations and UMAP projection.", wrap=-1)
                dpg.add_checkbox(label="Group UMAP by Target (if Target suitable: 2-10 unique values)", tag=TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX, default_value=False)
                dpg.add_button(label="Run Correlation Analysis & UMAP", tag=TAG_MVA_CORR_RUN_BUTTON, width=-1, height=30,
                             callback=lambda: _mva_run_correlation_analysis(
                                 _mva_main_app_callbacks['get_current_df'](), 
                                 _mva_main_app_callbacks.get('get_util_funcs', lambda: {})(), # u_funcs 전달
                                 _mva_main_app_callbacks # callbacks 전달
                                 )
                            )
                dpg.add_child_window(tag=TAG_MVA_CORR_RESULTS_GROUP, border=True)
            
            with dpg.tab(label="Pair Plot (Numeric)", tag=TAG_MVA_PAIRPLOT_TAB):
                dpg.add_text("Generates pair plots for relevant numeric variables (auto-selected, max 8).", wrap=-1) # MAX_VARS_PP 값 반영
                dpg.add_checkbox(label="Group Pair Plot by Target (if Target suitable: 2-7 unique values)", tag=TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX, default_value=False)
                dpg.add_button(label="Generate Pair Plot", tag=TAG_MVA_PAIRPLOT_RUN_BUTTON, width=-1, height=30,
                             callback=lambda: _mva_run_pair_plot_analysis(
                                 _mva_main_app_callbacks['get_current_df'](),
                                 _mva_main_app_callbacks.get('get_util_funcs', lambda: {})(), # u_funcs 전달
                                 _mva_main_app_callbacks # callbacks 전달
                                 )
                            )
                dpg.add_child_window(tag=TAG_MVA_PAIRPLOT_RESULTS_GROUP, border=True)

            with dpg.tab(label="Association (Categorical)", tag=TAG_MVA_CAT_EDA_TAB):
                dpg.add_text("Shows Cramer's V clustermap for associated categorical variables (auto-selected, max 15).", wrap=-1)
                dpg.add_button(label="Run Categorical Association Analysis", tag=TAG_MVA_CAT_EDA_RUN_BUTTON, width=-1, height=30,
                             callback=lambda: _mva_run_cat_corr_analysis(
                                 _mva_main_app_callbacks['get_current_df'](),
                                 _mva_main_app_callbacks.get('get_util_funcs', lambda: {})(), # u_funcs 전달
                                 _mva_main_app_callbacks # callbacks 전달
                                 )
                            )
                dpg.add_child_window(tag=TAG_MVA_CAT_EDA_RESULTS_GROUP, border=True)
                
    main_callbacks['register_module_updater'](step_name, update_ui) # SVA_STEP_KEY 대신 step_name 사용

def update_ui(current_df: Optional[pd.DataFrame], main_callbacks: Dict[str, Any]):
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_MVA_STEP_GROUP): return
    global _mva_main_app_callbacks # _mva_util_funcs 는 이제 크게 필요 없을 수 있음
    _mva_main_app_callbacks = main_callbacks
    
    is_df_valid = current_df is not None and not current_df.empty

    # 각 결과 그룹 초기화 (데이터가 없으면 메시지 표시)
    for area_tag, msg_if_no_data, run_button_tag in [
        (TAG_MVA_CORR_RESULTS_GROUP, "Run analysis after loading data.", TAG_MVA_CORR_RUN_BUTTON),
        (TAG_MVA_PAIRPLOT_RESULTS_GROUP, "Generate plot after loading data.", TAG_MVA_PAIRPLOT_RUN_BUTTON),
        (TAG_MVA_CAT_EDA_RESULTS_GROUP, "Run categorical analysis after loading data.", TAG_MVA_CAT_EDA_RUN_BUTTON)
    ]:
        if dpg.does_item_exist(area_tag):
            dpg.delete_item(area_tag, children_only=True) # 기존 내용 삭제
            if not is_df_valid:
                dpg.add_text(msg_if_no_data, parent=area_tag)
            else:
                # 데이터가 유효하면, "분석 실행" 등의 안내 메시지를 초기에 보여줄 수 있음
                # 또는, apply_mva_settings_from_loaded에서처럼 "Settings loaded. Click 'Run/Generate' button."
                dpg.add_text(f"Click the button above to generate results.", parent=area_tag)

        # 데이터 유효성에 따라 실행 버튼 활성화/비활성화
        if dpg.does_item_exist(run_button_tag):
            dpg.configure_item(run_button_tag, enabled=is_df_valid)
            
    # 특정 체크박스들의 활성화 여부도 데이터 유효성에 따라 결정 (선택적)
    for checkbox_tag in [TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX, TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX]:
        if dpg.does_item_exist(checkbox_tag):
             dpg.configure_item(checkbox_tag, enabled=is_df_valid and bool(main_callbacks['get_selected_target_variable']()))


def reset_mva_ui_defaults():
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX): dpg.set_value(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX, False)
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX): dpg.set_value(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX, False)
    
    # UI 초기화 시, 각 결과 그룹에 기본 안내 메시지 설정
    for area_tag, msg in [
        (TAG_MVA_CORR_RESULTS_GROUP, "Run 'Correlation Analysis & UMAP' to see results."), 
        (TAG_MVA_PAIRPLOT_RESULTS_GROUP, "Click 'Generate Pair Plot' to see results."), 
        (TAG_MVA_CAT_EDA_RESULTS_GROUP, "Run 'Categorical Association Analysis' to see results.")
    ]:
        if dpg.does_item_exist(area_tag): 
            dpg.delete_item(area_tag, children_only=True)
            dpg.add_text(msg, parent=area_tag)

    # 버튼들도 초기 상태 (예: 데이터 로드 전이면 비활성화)로 되돌릴 수 있음 (update_ui에서 처리)