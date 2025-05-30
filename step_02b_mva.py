# step_02b_mva.py
import matplotlib # Matplotlib 백엔드 설정을 위해 가장 먼저 임포트
matplotlib.use('Agg') # !!! GUI 백엔드 비활성화, 파일 출력 및 이미지 데이터 생성용 !!!
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
import utils

import seaborn as sns
import matplotlib.pyplot as plt # matplotlib.use('Agg') 이후에 임포트
import umap
import io
from PIL import Image
import traceback # traceback 임포트

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='umap')
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')

# --- MVA UI 태그 정의 ---
TAG_MVA_STEP_GROUP = "mva_step_group"
TAG_MVA_MAIN_TAB_BAR = "mva_main_tab_bar"

TAG_MVA_CORR_TAB = "mva_corr_tab"
TAG_MVA_CORR_RUN_BUTTON = "mva_corr_run_button"
TAG_MVA_CORR_RESULTS_GROUP = "mva_corr_results_group"

TAG_MVA_PAIRPLOT_TAB = "mva_pairplot_tab"
TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX = "mva_pairplot_group_by_target_checkbox"
TAG_MVA_PAIRPLOT_RUN_BUTTON = "mva_pairplot_run_button"
TAG_MVA_PAIRPLOT_RESULTS_GROUP = "mva_pairplot_results_group"

TAG_MVA_CAT_EDA_TAB = "mva_cat_corr_tab"
TAG_MVA_CAT_EDA_RUN_BUTTON = "mva_cat_eda_run_button"
TAG_MVA_CAT_EDA_RESULTS_GROUP = "mva_cat_eda_results_group"

TAG_MVA_MODULE_ALERT_MODAL = "mva_module_specific_alert_modal"
TAG_MVA_MODULE_ALERT_TEXT = "mva_module_specific_alert_text"

_mva_main_app_callbacks: Dict[str, Any] = {}
_mva_util_funcs: Dict[str, Any] = {}

def get_mva_settings_for_saving() -> Dict[str, Any]:
    settings = {}
    settings['corr_tab'] = {}

    pairplot_settings = {}
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX):
        pairplot_settings['group_by_target'] = dpg.get_value(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX)
    settings['pairplot_tab'] = pairplot_settings
    
    settings['cat_eda_tab'] = {}
    
    if dpg.does_item_exist(TAG_MVA_MAIN_TAB_BAR):
        settings['active_mva_tab_label'] = None
        try:
            active_tab_tag = dpg.get_value(TAG_MVA_MAIN_TAB_BAR)
            if active_tab_tag and dpg.does_item_exist(active_tab_tag):
                 settings['active_mva_tab_tag'] = active_tab_tag
        except Exception:
            for child in dpg.get_item_children(TAG_MVA_MAIN_TAB_BAR, 1):
                item_config = dpg.get_item_configuration(child)
                if item_config and item_config.get("show", False): # Check if item_config is not None
                    settings['active_mva_tab_label'] = item_config.get("label")
                    break
    return settings

def apply_mva_settings_from_loaded(settings: Dict[str, Any], current_df: Optional[pd.DataFrame], main_callbacks: Dict[str, Any]):
    if not dpg.is_dearpygui_running(): return

    pairplot_settings = settings.get('pairplot_tab', {})
    if 'group_by_target' in pairplot_settings and dpg.does_item_exist(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX):
        dpg.set_value(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX, pairplot_settings['group_by_target'])
        
    active_tab_tag_setting = settings.get('active_mva_tab_tag')
    if active_tab_tag_setting and dpg.does_item_exist(TAG_MVA_MAIN_TAB_BAR) and dpg.does_item_exist(active_tab_tag_setting):
        try:
            dpg.set_value(TAG_MVA_MAIN_TAB_BAR, active_tab_tag_setting)
        except Exception:
            active_tab_label_setting = settings.get('active_mva_tab_label')
            if active_tab_label_setting:
                 for child_tab_tag_iter in dpg.get_item_children(TAG_MVA_MAIN_TAB_BAR, 1):
                    item_config = dpg.get_item_configuration(child_tab_tag_iter)
                    if item_config and item_config.get("label") == active_tab_label_setting:
                        try:
                            dpg.set_value(TAG_MVA_MAIN_TAB_BAR, child_tab_tag_iter)
                        except Exception as e_set_tab:
                            print(f"Error setting active tab by label fallback: {e_set_tab}")
                        break
    
    for area_tag in [TAG_MVA_CORR_RESULTS_GROUP, TAG_MVA_PAIRPLOT_RESULTS_GROUP, TAG_MVA_CAT_EDA_RESULTS_GROUP]:
        if dpg.does_item_exist(area_tag):
            dpg.delete_item(area_tag, children_only=True)
            dpg.add_text("Settings loaded. Click the 'Run/Generate' button to update results.", parent=area_tag)
            
    update_ui(current_df, main_callbacks)

def _plot_to_dpg_texture_data(fig: plt.Figure, desired_dpi: int = 90) -> Tuple[Optional[str], int, int]:
    img_data_buf = io.BytesIO()
    try:
        fig.savefig(img_data_buf, format="png", bbox_inches='tight', dpi=desired_dpi)
        img_data_buf.seek(0)
        pil_image = Image.open(img_data_buf)
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        
        img_width, img_height = pil_image.size
        texture_data_np = np.array(pil_image).astype(np.float32) / 255.0
        texture_data_flat_list = texture_data_np.ravel().tolist()

        texture_tag = dpg.generate_uuid()
        with dpg.texture_registry(show=False):
            dpg.add_static_texture(width=img_width, height=img_height, default_value=texture_data_flat_list, tag=texture_tag)
        return texture_tag, img_width, img_height
    except Exception as e:
        print(f"Error converting plot to DPG texture: {e}")
        print(traceback.format_exc())
        return None, 0, 0
    finally:
        plt.close(fig)

def _display_dpg_image(parent_group: str, texture_tag: Optional[str], tex_w: int, tex_h: int, max_w: int = 750):
    if texture_tag and tex_w > 0 and tex_h > 0:
        # DPG에 표시될 이미지 크기 조절 (가로 또는 세로 기준)
        if tex_w > max_w : # 너비가 최대 너비 초과시
            display_w = max_w
            display_h = int(tex_h * (max_w / tex_w))
        elif tex_h > max_w * 0.8 : # 높이가 너무 클 경우 (최대 너비의 80% 기준)
            display_h = int(max_w * 0.8)
            display_w = int(tex_w * (display_h / tex_h))
        else: # 원본 크기 유지
            display_w = tex_w
            display_h = tex_h
        
        dpg.add_image(texture_tag, parent=parent_group, width=int(display_w), height=int(display_h))
    elif texture_tag:
        dpg.add_text("Failed: Image has zero width or height.", parent=parent_group, color=(255,100,0))
    else:
        dpg.add_text("Failed to generate image texture.", parent=parent_group, color=(255,0,0))


def _mva_run_correlation_analysis(df: pd.DataFrame, u_funcs: dict, callbacks: dict):
    results_group = TAG_MVA_CORR_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group): return
    dpg.delete_item(results_group, children_only=True)
    if df is None: dpg.add_text("Load data first.", parent=results_group); return
    
    num_cols = utils._get_numeric_cols(df)
    if len(num_cols) < 2:
        dpg.add_text("Need at least 2 numeric columns for correlation analysis.", parent=results_group); return

    MAX_VARS_FOR_HEATMAP = 15
    target_var = callbacks['get_selected_target_variable']()
    target_var_type = callbacks['get_selected_target_variable_type']()
    
    corr_abs_matrix_full = df[num_cols].corr().abs() 

    # 1. Clustermap: 상관계수가 높은 변수 쌍 기준
    dpg.add_text(f"Clustermap 1: Top {MAX_VARS_FOR_HEATMAP} Numeric Variables with Highest Pairwise Correlations", parent=results_group, color=(255,255,0))
    try:
        if len(num_cols) <= 2:
            vars_for_clustermap1 = num_cols
        elif len(num_cols) <= MAX_VARS_FOR_HEATMAP :
            vars_for_clustermap1 = num_cols
        else:
            max_corrs_per_var = {}
            for col in num_cols:
                other_cols_corr = corr_abs_matrix_full.loc[col, corr_abs_matrix_full.columns != col]
                max_corrs_per_var[col] = other_cols_corr.max() if not other_cols_corr.empty else 0
            
            sorted_vars_by_max_corr = sorted(max_corrs_per_var.items(), key=lambda item: item[1], reverse=True)
            vars_for_clustermap1 = [var for var, _ in sorted_vars_by_max_corr[:MAX_VARS_FOR_HEATMAP]]

        if len(vars_for_clustermap1) >= 2:
            sub_corr_mat1 = df[vars_for_clustermap1].corr() # 원본 상관계수 (절대값 아님)
            
            num_vars_cm1 = len(vars_for_clustermap1)
            font_scale_cm1 = max(0.5, 1.0 - num_vars_cm1 * 0.03) # 폰트 크기 조절
            figsize_cm1 = (max(6, num_vars_cm1 * 0.7), max(5.5, num_vars_cm1 * 0.6))

            sns.set_theme(style="whitegrid", font_scale=font_scale_cm1)
            cluster_cm1 = sns.clustermap(
                sub_corr_mat1, annot=True, cmap="RdYlBu_r", fmt=".2f", linewidths=.5,
                vmin=-1, vmax=1, figsize=figsize_cm1, dendrogram_ratio=0.15,
                cbar_kws={'shrink': .7}
            )
            cluster_cm1.fig.suptitle(f"Top {num_vars_cm1} Inter-correlated Variables", fontsize=10, y=1.02)
            
            texture_tag_cm1, tex_w_cm1, tex_h_cm1 = _plot_to_dpg_texture_data(cluster_cm1.fig)
            _display_dpg_image(results_group, texture_tag_cm1, tex_w_cm1, tex_h_cm1, max_w=700)
        else:
            dpg.add_text("Not enough variables for this clustermap.", parent=results_group)
    except Exception as e_cm1: 
        dpg.add_text(f"Error for Clustermap 1: {e_cm1}", parent=results_group, color=(255,0,0)); print(traceback.format_exc())
    dpg.add_separator(parent=results_group)

    # 2. Clustermap: 타겟 변수와 관련 높은 변수 기준
    dpg.add_text(f"Clustermap 2: Top {MAX_VARS_FOR_HEATMAP} Numeric Variables Correlated with Target '{target_var}'", parent=results_group, color=(255,255,0))
    if target_var and target_var in num_cols and target_var_type == "Continuous":
        try:
            other_num_cols_for_target = [col for col in num_cols if col != target_var]
            if not other_num_cols_for_target:
                 dpg.add_text("No other numeric variables to correlate with the target.", parent=results_group)
            else:
                target_corrs_series = df[other_num_cols_for_target].corrwith(df[target_var]).abs().sort_values(ascending=False)
                vars_for_clustermap2 = [target_var] + target_corrs_series.head(MAX_VARS_FOR_HEATMAP - 1).index.tolist()
                vars_for_clustermap2 = list(dict.fromkeys(vars_for_clustermap2))

                if len(vars_for_clustermap2) >=2:
                    sub_corr_mat2 = df[vars_for_clustermap2].corr()
                    num_vars_cm2 = len(vars_for_clustermap2)
                    font_scale_cm2 = max(0.5, 1.0 - num_vars_cm2 * 0.03)
                    figsize_cm2 = (max(6, num_vars_cm2 * 0.7), max(5.5, num_vars_cm2 * 0.6))

                    sns.set_theme(style="whitegrid", font_scale=font_scale_cm2)
                    cluster_cm2 = sns.clustermap(
                        sub_corr_mat2, annot=True, cmap="RdYlBu_r", fmt=".2f", linewidths=.5,
                        vmin=-1, vmax=1, figsize=figsize_cm2, dendrogram_ratio=0.15,
                        cbar_kws={'shrink': .7}
                    )
                    cluster_cm2.fig.suptitle(f"Top {num_vars_cm2} Variables Correlated with Target '{target_var}'", fontsize=10, y=1.02)

                    texture_tag_cm2, tex_w_cm2, tex_h_cm2 = _plot_to_dpg_texture_data(cluster_cm2.fig)
                    _display_dpg_image(results_group, texture_tag_cm2, tex_w_cm2, tex_h_cm2, max_w=700)
                else:
                    dpg.add_text("Not enough variables correlated with target for clustermap.", parent=results_group)
        except Exception as e_cm2:
            dpg.add_text(f"Error for Clustermap 2 (Target): {e_cm2}", parent=results_group, color=(255,0,0)); print(traceback.format_exc())
    else:
        dpg.add_text(f"Skipped: Target '{target_var}' not selected, not numeric, or not continuous.", parent=results_group, color=(200,200,0))
    dpg.add_separator(parent=results_group)

    # 3. UMAP
    dpg.add_text("UMAP 2D Visualization of All Numeric Variables:", parent=results_group, color=(255,255,0))
    try:
        umap_df_prepared = df[num_cols].copy()
        for col in umap_df_prepared.columns:
            if umap_df_prepared[col].isnull().any() and pd.api.types.is_numeric_dtype(umap_df_prepared[col]):
                umap_df_prepared[col] = umap_df_prepared[col].fillna(umap_df_prepared[col].median())
        
        if umap_df_prepared.shape[0] < 2: dpg.add_text("Not enough samples for UMAP.", parent=results_group)
        elif umap_df_prepared.shape[1] < 2: dpg.add_text("Not enough numeric features for UMAP.", parent=results_group)
        else:
            n_neighbors_val = min(15, umap_df_prepared.shape[0] - 1) if umap_df_prepared.shape[0] > 1 else 1 
            if n_neighbors_val <= 0 : n_neighbors_val = 1 # Defensive
            reducer = umap.UMAP(n_neighbors=n_neighbors_val, n_components=2, random_state=42, min_dist=0.05, spread=1.0)
            embedding = reducer.fit_transform(umap_df_prepared)
            
            plt.style.use('seaborn-v0_8-whitegrid') # UMAP에 어울리는 스타일
            fig_umap = plt.figure(figsize=(6, 4.5))
            plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.6, cmap='viridis') # 점 크기, 투명도, 색상맵 변경
            plt.title("UMAP Projection", fontsize=10)
            plt.xlabel("UMAP Dim 1", fontsize=8); plt.ylabel("UMAP Dim 2", fontsize=8)
            plt.xticks(fontsize=7); plt.yticks(fontsize=7)
            # plt.grid(True, linestyle=':', alpha=0.5) # seaborn-v0_8-whitegrid 에 이미 그리드 포함
            plt.tight_layout()
            
            texture_tag_umap, tex_w_umap, tex_h_umap = _plot_to_dpg_texture_data(fig_umap)
            _display_dpg_image(results_group, texture_tag_umap, tex_w_umap, tex_h_umap, max_w=600)
            plt.style.use('default') # 스타일 복원
    except ImportError:
         dpg.add_text("UMAP-learn is not installed.", parent=results_group, color=(255,100,0))
    except Exception as e_umap:
        dpg.add_text(f"Error creating UMAP plot: {e_umap}", parent=results_group, color=(255,0,0)); print(traceback.format_exc())

def _mva_run_pair_plot_analysis(df: pd.DataFrame, u_funcs: dict, callbacks: dict):
    res_group = TAG_MVA_PAIRPLOT_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(res_group): return
    dpg.delete_item(res_group, children_only=True)
    if df is None: dpg.add_text("Load data first.", parent=res_group); return
    
    MAX_VARS_PAIRPLOT = 15
    num_cols_all = utils._get_numeric_cols(df)
    if len(num_cols_all) < 2:
        dpg.add_text("Need at least 2 numeric columns for Pair Plot.", parent=res_group); return

    target_var = callbacks['get_selected_target_variable']()
    target_var_type = callbacks['get_selected_target_variable_type']()
    vars_for_pairplot = []

    if target_var and target_var in num_cols_all and target_var_type == "Continuous":
        dpg.add_text(f"Pair Plot: Selecting variables based on correlation with target '{target_var}'.", parent=res_group, color=(180,180,180))
        other_num_cols = [col for col in num_cols_all if col != target_var]
        if other_num_cols:
            target_corrs = df[other_num_cols].corrwith(df[target_var]).abs().sort_values(ascending=False)
            vars_for_pairplot = [target_var] + target_corrs.head(MAX_VARS_PAIRPLOT - 1).index.tolist()
        else: vars_for_pairplot = [target_var]
    
    if not vars_for_pairplot or len(vars_for_pairplot) < 2 :
        dpg.add_text(f"Pair Plot: Using first {min(MAX_VARS_PAIRPLOT, len(num_cols_all))} available numeric variables.", parent=res_group, color=(180,180,180))
        vars_for_pairplot = num_cols_all[:MAX_VARS_PAIRPLOT]
    
    vars_for_pairplot = list(dict.fromkeys(vars_for_pairplot))[:MAX_VARS_PAIRPLOT]

    if len(vars_for_pairplot) < 2:
        dpg.add_text("Not enough numeric variables for Pair Plot.", parent=res_group); return
        
    hue_for_plot = None
    group_by_target_cb_val = dpg.get_value(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX) if dpg.does_item_exist(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX) else False

    if group_by_target_cb_val and target_var and target_var in df.columns:
        unique_target_count = df[target_var].nunique(dropna=True)
        if 2 <= unique_target_count <= 7:
            hue_for_plot = target_var
            dpg.add_text(f"Pair Plot: Using target '{target_var}' for Hue.", parent=res_group, color=(180,180,180))
        else:
            dpg.add_text(f"Pair Plot: Target '{target_var}' has {unique_target_count} unique values. Hue disabled (requires 2-7).", parent=res_group, color=(200,200,0))
    
    title_text_pp = f"Pair Plot: {', '.join(vars_for_pairplot)}" + (f" (Hue: {hue_for_plot})" if hue_for_plot else "")
    dpg.add_text(title_text_pp, parent=res_group, color=(255,255,0))
    try:
        pp_df_copy = df.copy()
        num_vars_for_pp = len(vars_for_pairplot)
        plot_size_val_pp = max(1.0, min(2.0, 7 / num_vars_for_pp if num_vars_for_pp > 0 else 2))
        font_scale_val_pp = max(0.45, min(0.85, 1.0 - num_vars_for_pp * 0.045))

        sns.set_theme(style="ticks", font_scale=font_scale_val_pp)

        cols_for_pp_df = vars_for_pairplot + ([hue_for_plot] if hue_for_plot else [])
        pp_df_subset_final = pp_df_copy[cols_for_pp_df].copy()

        if hue_for_plot and hue_for_plot in pp_df_subset_final.columns:
             if not pd.api.types.is_string_dtype(pp_df_subset_final[hue_for_plot]) and \
                not pd.api.types.is_categorical_dtype(pp_df_subset_final[hue_for_plot]):
                try: pp_df_subset_final[hue_for_plot] = pd.Categorical(pp_df_subset_final[hue_for_plot])
                except: pp_df_subset_final[hue_for_plot] = pp_df_subset_final[hue_for_plot].astype(str)
        
        pp_df_subset_final.dropna(subset=vars_for_pairplot, inplace=True) # 수치형 변수 기준 NaN 제거
        if pp_df_subset_final.empty or len(pp_df_subset_final) < 2:
             dpg.add_text("Not enough data after NaN handling for Pair Plot.", parent=res_group, color=(255,100,0)); return

        pair_plot_seaborn_g = sns.pairplot(
            pp_df_subset_final, vars=vars_for_pairplot, hue=hue_for_plot, 
            diag_kind='kde', corner=False, height=plot_size_val_pp,
            plot_kws={'alpha': 0.55, 's': 12 if num_vars_for_pp <=7 else 7, 'edgecolor':'none'},
            diag_kws={'fill': True, 'alpha': 0.45, 'linewidth': 0.8}
        )
        pair_plot_seaborn_g.fig.suptitle(f"Pair Plot" + (f" (Hue: {hue_for_plot})" if hue_for_plot else ""), y=1.01, fontsize=10)
        
        texture_tag_pp_img, tex_w_pp_img, tex_h_pp_img = _plot_to_dpg_texture_data(pair_plot_seaborn_g.fig, desired_dpi=70) # DPI 더 낮춤 (복잡한 플롯)
        _display_dpg_image(res_group, texture_tag_pp_img, tex_w_pp_img, tex_h_pp_img, max_w=750)
    except ImportError:
        dpg.add_text("Seaborn or Matplotlib is not installed.", parent=res_group, color=(255,100,0))
    except Exception as e_pp_final:
        dpg.add_text(f"Error creating Pair Plot: {e_pp_final}", parent=res_group, color=(255,0,0)); print(traceback.format_exc())

def _mva_run_cat_corr_analysis(df: pd.DataFrame, u_funcs: dict, callbacks: dict):
    res_group = TAG_MVA_CAT_EDA_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(res_group): return
    dpg.delete_item(res_group, children_only=True)
    if df is None: dpg.add_text("Load data first.", parent=res_group); return
    
    MAX_VARS_CRAMER = 15
    all_cat_cols = utils._get_categorical_cols(df, max_unique_for_cat=35, main_callbacks=callbacks) # 고유값 제한 약간 늘림
    if len(all_cat_cols) < 2:
        dpg.add_text("Need at least 2 categorical columns for Cramer's V.", parent=res_group); return

    vars_for_cramer_final = []
    if len(all_cat_cols) <= MAX_VARS_CRAMER:
        vars_for_cramer_final = all_cat_cols
        dpg.add_text(f"Using all {len(all_cat_cols)} categorical variables.", parent=res_group, color=(180,180,180))
    else:
        dpg.add_text(f"Selecting top {MAX_VARS_CRAMER} categorical variables by max pairwise Cramer's V.", parent=res_group, color=(180,180,180))
        cramer_pairs_values = {}
        for i in range(len(all_cat_cols)):
            for j in range(i + 1, len(all_cat_cols)):
                v1, v2 = all_cat_cols[i], all_cat_cols[j]
                c_v_val = utils.calculate_cramers_v(df[v1], df[v2])
                if pd.notna(c_v_val): cramer_pairs_values[(v1,v2)] = c_v_val
        
        max_cramer_for_each_var = {var: 0.0 for var in all_cat_cols}
        for (v_a, v_b), val_cv in cramer_pairs_values.items():
            max_cramer_for_each_var[v_a] = max(max_cramer_for_each_var[v_a], val_cv)
            max_cramer_for_each_var[v_b] = max(max_cramer_for_each_var[v_b], val_cv)
            
        sorted_vars_by_cramer = sorted(max_cramer_for_each_var.items(), key=lambda item: item[1], reverse=True)
        vars_for_cramer_final = [var for var, _ in sorted_vars_by_cramer[:MAX_VARS_CRAMER]]

    if len(vars_for_cramer_final) < 2:
        dpg.add_text("Not enough categorical variables for Cramer's V clustermap.", parent=res_group); return
        
    dpg.add_text(f"Cramer's V Clustermap (Top {len(vars_for_cramer_final)} Associated Variables)", parent=res_group, color=(255,255,0))
    try:
        cramer_v_matrix_final = pd.DataFrame(np.zeros((len(vars_for_cramer_final), len(vars_for_cramer_final))), columns=vars_for_cramer_final, index=vars_for_cramer_final)
        for i_idx, v1_name in enumerate(vars_for_cramer_final):
            for j_idx, v2_name in enumerate(vars_for_cramer_final):
                if i_idx == j_idx: cv_res = 1.0
                else: cv_res = utils.calculate_cramers_v(df[v1_name], df[v2_name])
                cramer_v_matrix_final.iloc[i_idx,j_idx] = cv_res if pd.notna(cv_res) else 0

        if cramer_v_matrix_final.shape[0] < 2: # Clustermap은 최소 2x2 필요
             dpg.add_text("Not enough data (or only 1 variable) for Cramer's V clustermap.", parent=res_group); return

        num_cv_vars = len(vars_for_cramer_final)
        font_scale_cv = max(0.5, 1.0 - num_cv_vars * 0.03)
        figsize_cv = (max(5.5, num_cv_vars * 0.6), max(5, num_cv_vars * 0.55))

        sns.set_theme(style="white", font_scale=font_scale_cv)
        # Cramer's V는 0-1 범위이고 대칭적이므로, standard_scale=1 (행 기준 정규화) 같은 옵션은 부적절.
        # row_cluster, col_cluster를 False로 하면 덴드로그램 없이 순서대로 나옴.
        cluster_cv_plot = sns.clustermap(
            cramer_v_matrix_final, annot=True, cmap="Blues", fmt=".2f", linewidths=.5,
            vmin=0, vmax=1, figsize=figsize_cv, dendrogram_ratio=0.1,
            cbar_kws={'shrink': .6, 'ticks': [0, 0.5, 1]} # 컬러바 눈금 조절
        )
        cluster_cv_plot.fig.suptitle("Cramer's V Association Clustermap", fontsize=10, y=1.02)
        
        texture_tag_cv, tex_w_cv, tex_h_cv = _plot_to_dpg_texture_data(cluster_cv_plot.fig, desired_dpi=75)
        _display_dpg_image(res_group, texture_tag_cv, tex_w_cv, tex_h_cv, max_w=1000)
    except ImportError:
        dpg.add_text("Seaborn or Matplotlib is not installed.", parent=res_group, color=(255,100,0))
    except Exception as e_cv_final:
        dpg.add_text(f"Error creating Cramer's V clustermap: {e_cv_final}", parent=res_group, color=(255,0,0)); print(traceback.format_exc())

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _mva_main_app_callbacks, _mva_util_funcs
    _mva_main_app_callbacks = main_callbacks
    _mva_util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()

    with dpg.group(tag=TAG_MVA_STEP_GROUP, parent=parent_container_tag):
        with dpg.tab_bar(tag=TAG_MVA_MAIN_TAB_BAR):
            with dpg.tab(label="Correlation (Numeric)", tag=TAG_MVA_CORR_TAB):
                dpg.add_text("Displays clustermaps of numeric variable correlations and UMAP projection.", wrap=-1)
                dpg.add_button(label="Run Correlation Analysis & UMAP", tag=TAG_MVA_CORR_RUN_BUTTON, width=-1, height=30,
                             callback=lambda: _mva_run_correlation_analysis(_mva_main_app_callbacks['get_current_df'](), _mva_util_funcs, _mva_main_app_callbacks))
                dpg.add_child_window(tag=TAG_MVA_CORR_RESULTS_GROUP, border=True)
            
            with dpg.tab(label="Pair Plot (Numeric)", tag=TAG_MVA_PAIRPLOT_TAB):
                dpg.add_text("Generates pair plots for highly correlated/relevant numeric variables (auto-selected, max 15).", wrap=-1)
                dpg.add_checkbox(label="Group by Target (if Target has 2-7 unique values & is categorical/suitable)", tag=TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX, default_value=False)
                dpg.add_button(label="Generate Pair Plot", tag=TAG_MVA_PAIRPLOT_RUN_BUTTON, width=-1, height=30,
                             callback=lambda: _mva_run_pair_plot_analysis(_mva_main_app_callbacks['get_current_df'](),_mva_util_funcs, _mva_main_app_callbacks))
                dpg.add_child_window(tag=TAG_MVA_PAIRPLOT_RESULTS_GROUP, border=True)

            with dpg.tab(label="Association (Categorical)", tag=TAG_MVA_CAT_EDA_TAB):
                dpg.add_text("Shows Cramer's V clustermap for highly associated categorical variables (auto-selected, max 15).", wrap=-1)
                dpg.add_button(label="Run Categorical Association Analysis", tag=TAG_MVA_CAT_EDA_RUN_BUTTON, width=-1, height=30,
                             callback=lambda: _mva_run_cat_corr_analysis(_mva_main_app_callbacks['get_current_df'](), _mva_util_funcs, _mva_main_app_callbacks))
                dpg.add_child_window(tag=TAG_MVA_CAT_EDA_RESULTS_GROUP, border=True)
                
    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui(current_df: Optional[pd.DataFrame], main_callbacks: Dict[str, Any]):
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_MVA_STEP_GROUP): return
    
    global _mva_main_app_callbacks, _mva_util_funcs
    _mva_main_app_callbacks = main_callbacks
    _mva_util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    
    if current_df is None:
        for area_tag, initial_msg in [
            (TAG_MVA_CORR_RESULTS_GROUP, "Load data to run correlation analysis."),
            (TAG_MVA_PAIRPLOT_RESULTS_GROUP, "Load data to generate pair plots."),
            (TAG_MVA_CAT_EDA_RESULTS_GROUP, "Load data to run categorical association analysis.")
        ]:
            if dpg.does_item_exist(area_tag):
                dpg.delete_item(area_tag, children_only=True)
                dpg.add_text(initial_msg, parent=area_tag)

def reset_mva_ui_defaults():
    if not dpg.is_dearpygui_running(): return
    
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX):
        dpg.set_value(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX, False)

    initial_messages = {
        TAG_MVA_CORR_RESULTS_GROUP: "Run correlation analysis to see results.",
        TAG_MVA_PAIRPLOT_RESULTS_GROUP: "Click 'Generate Pair Plot' to see results.",
        TAG_MVA_CAT_EDA_RESULTS_GROUP: "Run categorical association analysis to see results."
    }
    for area_tag, msg in initial_messages.items():
        if dpg.does_item_exist(area_tag):
            dpg.delete_item(area_tag, children_only=True)
            dpg.add_text(msg, parent=area_tag)