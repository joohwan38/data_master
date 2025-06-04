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
import ollama_analyzer

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='umap')
warnings.filterwarnings('ignore', category=UserWarning, module='seaborn')
warnings.filterwarnings('ignore', category=FutureWarning, module='seaborn')

# --- MVA UI 태그 정의 ---
TAG_MVA_STEP_GROUP = "mva_step_group"
TAG_MVA_MAIN_TAB_BAR = "mva_main_tab_bar"

TAG_MVA_CORR_TAB = "mva_corr_tab"
TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX = "mva_corr_umap_group_by_target_checkbox" # UMAP 그룹핑용
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
    settings = {'corr_tab': {}, 'pairplot_tab': {}, 'cat_eda_tab': {}}
    if dpg.does_item_exist(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX):
        settings['corr_tab']['umap_group_by_target'] = dpg.get_value(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX)
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX):
        settings['pairplot_tab']['group_by_target'] = dpg.get_value(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX)
    
    if dpg.does_item_exist(TAG_MVA_MAIN_TAB_BAR):
        settings['active_mva_tab_label'] = None
        try:
            active_tab_tag = dpg.get_value(TAG_MVA_MAIN_TAB_BAR)
            if active_tab_tag and dpg.does_item_exist(active_tab_tag):
                 settings['active_mva_tab_tag'] = active_tab_tag
        except Exception:
            for child in dpg.get_item_children(TAG_MVA_MAIN_TAB_BAR, 1):
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
        except Exception:
            active_tab_label_setting = settings.get('active_mva_tab_label')
            if active_tab_label_setting:
                 for child_iter in dpg.get_item_children(TAG_MVA_MAIN_TAB_BAR, 1):
                    cfg = dpg.get_item_configuration(child_iter)
                    if cfg and cfg.get("label") == active_tab_label_setting:
                        try: dpg.set_value(TAG_MVA_MAIN_TAB_BAR, child_iter); break
                        except: pass # Failsafe
    
    for area_tag in [TAG_MVA_CORR_RESULTS_GROUP, TAG_MVA_PAIRPLOT_RESULTS_GROUP, TAG_MVA_CAT_EDA_RESULTS_GROUP]:
        if dpg.does_item_exist(area_tag):
            dpg.delete_item(area_tag, children_only=True)
            dpg.add_text("Settings loaded. Click 'Run/Generate' button.", parent=area_tag)
    update_ui(current_df, main_callbacks)

def _plot_to_dpg_texture_data(fig: plt.Figure, desired_dpi: int = 100) -> Tuple[Optional[str], int, int, Optional[bytes]]:
    img_data_buf = io.BytesIO()
    img_bytes_data = None
    texture_tag = None # 초기화
    img_width, img_height = 0, 0 # 초기화

    TEXTURE_REGISTRY_TAG = "primary_texture_registry"
    if not dpg.does_item_exist(TEXTURE_REGISTRY_TAG):
        if dpg.is_dearpygui_running(): # DPG 컨텍스트가 활성화된 경우에만 추가 시도
            print(f"Warning: Texture registry '{TEXTURE_REGISTRY_TAG}' not found. Creating it now in step_02b_mva.")
            dpg.add_texture_registry(tag=TEXTURE_REGISTRY_TAG)
        else:
            print(f"Error: DPG not running, cannot create texture registry '{TEXTURE_REGISTRY_TAG}'.")
            return None, 0, 0, None


    try:
        fig.savefig(img_data_buf, format="png", bbox_inches='tight', dpi=desired_dpi)
        img_data_buf.seek(0)
        img_bytes_data = img_data_buf.getvalue()

        pil_image = Image.open(io.BytesIO(img_bytes_data))
        if pil_image.mode != 'RGBA': pil_image = pil_image.convert('RGBA')
        img_width, img_height = pil_image.size

        if img_width == 0 or img_height == 0:
            print(f"Error: Plot image has zero dimension ({img_width}x{img_height}). Cannot create texture.")
            return None, 0, 0, img_bytes_data # 바이트 데이터는 반환 가능

        texture_data_np = np.array(pil_image).astype(np.float32) / 255.0
        texture_data_flat_list = texture_data_np.ravel().tolist()
        texture_tag = dpg.generate_uuid() # 고유 태그 생성

        # dpg.add_static_texture 호출 시 parent를 명시적으로 지정합니다.
        dpg.add_static_texture(
            width=img_width,
            height=img_height,
            default_value=texture_data_flat_list,
            tag=texture_tag,
            parent=TEXTURE_REGISTRY_TAG # 명시적 부모 지정
        )
        return texture_tag, img_width, img_height, img_bytes_data
    except SystemError as se: # 구체적인 SystemError를 먼저 캐치
        print(f"SystemError converting plot to DPG texture: {se}")
        print(f"Traceback (SystemError in _plot_to_dpg_texture_data): {traceback.format_exc()}")
        # 오류 발생 시에도 생성된 태그(실패했다면 None), 크기, 바이트 데이터 반환 시도
        return texture_tag, img_width, img_height, img_bytes_data
    except Exception as e:
        print(f"General error converting plot to DPG texture: {e}")
        print(f"Traceback (General Error in _plot_to_dpg_texture_data): {traceback.format_exc()}")
        return None, 0, 0, img_bytes_data # 일반 오류 시 None 반환
    finally:
        plt.close(fig)

def _display_dpg_image(parent_group: str, texture_tag: Optional[str], tex_w: int, tex_h: int, max_w: int = 850): # max_w 상향
    if texture_tag and tex_w > 0 and tex_h > 0:
        display_w, display_h = tex_w, tex_h
        if tex_w > max_w:
            display_w = max_w
            display_h = int(tex_h * (max_w / tex_w))

        dpg.add_image(texture_tag, parent=parent_group, width=int(display_w), height=int(display_h))
    elif texture_tag:
        dpg.add_text("Failed: Image has zero dimension.", parent=parent_group, color=(255,100,0))
    else:
        dpg.add_text("Failed: Image texture not generated.", parent=parent_group, color=(255,0,0))

def _mva_run_correlation_analysis(df: pd.DataFrame, u_funcs: dict, callbacks: dict):
    results_group = TAG_MVA_CORR_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(results_group): return
    dpg.delete_item(results_group, children_only=True)
    if df is None: dpg.add_text("Load data first.", parent=results_group); return
    
    num_cols = utils._get_numeric_cols(df)
    if len(num_cols) < 2:
        dpg.add_text("Need at least 2 numeric columns.", parent=results_group); return

    MAX_VARS_CM = 15 # Clustermap 최대 변수
    target_var = callbacks['get_selected_target_variable']()
    target_var_type = callbacks['get_selected_target_variable_type']()
    corr_abs_mat_full = df[num_cols].corr().abs()

    # --- Clustermap 1: 상호 높은 상관관계 변수 ---
    dpg.add_text(f"Clustermap 1: Top {MAX_VARS_CM} Numeric Variables - Highest Pairwise Correlations", parent=results_group, color=(255,255,0))
    img_bytes_cm1 = None # AI 분석용 이미지 바이트 저장 변수
    tex_tag1, w1, h1 = None, 0, 0 # 이미지 표시용 변수들 초기화

    try:
        vars_cm1 = num_cols if len(num_cols) <= MAX_VARS_CM else \
                     [v for v, _ in sorted({col: corr_abs_mat_full.loc[col, corr_abs_mat_full.columns != col].max() if not corr_abs_mat_full.loc[col, corr_abs_mat_full.columns != col].empty else 0 for col in num_cols}.items(), key=lambda item: item[1], reverse=True)[:MAX_VARS_CM]]
        if len(vars_cm1) >= 2:
            sub_corr1 = df[vars_cm1].corr().fillna(0).replace([np.inf, -np.inf], 0)
            n_vars1 = len(vars_cm1)
            fs1 = max(0.7, 1.2 - n_vars1 * 0.02)
            fsize1 = (max(7, n_vars1 * 0.8), max(6, n_vars1 * 0.7))

            sns.set_theme(style="whitegrid", font_scale=fs1)
            cm1 = sns.clustermap(sub_corr1, annot=True, cmap="RdYlBu_r", fmt=".2f", linewidths=.5, vmin=-1, vmax=1, figsize=fsize1, dendrogram_ratio=0.12, cbar_kws={'shrink': .65})
            cm1.fig.suptitle(f"Top {n_vars1} Inter-correlated Variables", fontsize=15 if fs1 > 0.6 else 13, y=1.02)

            # _plot_to_dpg_texture_data 함수를 한 번만 호출합니다.
            plot_result_cm1 = _plot_to_dpg_texture_data(cm1.fig, desired_dpi=95)

            if plot_result_cm1 and len(plot_result_cm1) == 4:
                tex_tag1, w1, h1, img_bytes_cm1_temp = plot_result_cm1
                img_bytes_cm1 = img_bytes_cm1_temp # AI 분석용 이미지 바이트 저장
            else:
                print("Warning: _plot_to_dpg_texture_data did not return 4 values for Clustermap 1.")
                # 오류 발생 시 tex_tag1 등이 None으로 유지되어 아래 _display_dpg_image에서 처리가능

            # DPG 이미지 및 AI 분석 버튼을 담을 그룹
            # _display_dpg_image 호출은 tex_tag1 등이 올바르게 할당된 후에 이루어져야 합니다.
            with dpg.group(horizontal=False, parent=results_group): # 수직 그룹
                _display_dpg_image(dpg.last_item(), tex_tag1, w1, h1, max_w=700)

                if img_bytes_cm1 and tex_tag1 : # 이미지와 바이트가 모두 성공적으로 생성되었을 때만 버튼 추가
                    chart_name_cm1 = f"Clustermap1_Top_{len(vars_cm1)}_InterCorrelated"
                    # AI 분석 버튼 콜백 정의 (analyze_cm1_callback, confirm_and_run_cm1_analysis)
                    # (이전 답변에서 제공된 AI 분석 버튼 및 콜백 로직은 여기에 위치합니다)
                    def analyze_cm1_callback():
                        loading_indicator_tag = f"loading_cm1_{dpg.generate_uuid()}"
                        if dpg.does_item_exist(ai_button_cm1_tag):
                            dpg.configure_item(ai_button_cm1_tag, enabled=False, label="Analyzing...")
                            dpg.add_loading_indicator(tag=loading_indicator_tag, parent=ai_button_cm1_tag, style=0, radius=7, color=[255,255,0,255])
                        try:
                            analysis_result = ollama_analyzer.analyze_image_with_llava(img_bytes_cm1, chart_name_cm1)
                            if 'add_ai_log' in callbacks: # main_app_callbacks 에서 'add_ai_log' 가져오기
                                callbacks['add_ai_log'](analysis_result, chart_name_cm1)
                            else:
                                print(f"AI Log ({chart_name_cm1}):\n{analysis_result}")
                        except Exception as e_analysis:
                            err_msg = f"Error during AI analysis for {chart_name_cm1}: {e_analysis}"
                            print(err_msg)
                            if 'add_ai_log' in callbacks:
                                callbacks['add_ai_log'](err_msg, chart_name_cm1)
                        finally:
                            if dpg.does_item_exist(loading_indicator_tag):
                                dpg.delete_item(loading_indicator_tag)
                            if dpg.does_item_exist(ai_button_cm1_tag):
                                dpg.configure_item(ai_button_cm1_tag, enabled=True, label="💡 Analyze with AI")


                    def confirm_and_run_cm1_analysis():
                        # callbacks 딕셔너리가 제대로 전달되었는지, 그 안에 'get_util_funcs'가 있는지 확인
                        if 'get_util_funcs' in callbacks:
                            util_funcs_dict = callbacks['get_util_funcs']() # 딕셔너리 반환
                            # util_funcs_dict 안에 'show_confirmation_modal'이 있는지 확인
                            if 'show_confirmation_modal' in util_funcs_dict:
                                util_funcs_dict['show_confirmation_modal'](
                                    title="AI Analysis Confirmation",
                                    message=f"Proceed with AI analysis for '{chart_name_cm1}'?\n(This may take a few moments)",
                                    yes_callback=analyze_cm1_callback
                                )
                            else:
                                print("Confirmation modal function not found in utils dictionary. Running analysis directly.")
                                analyze_cm1_callback()
                        else:
                            print("Util functions ('get_util_funcs') not available in callbacks. Running analysis directly.")
                            analyze_cm1_callback() # 유틸 함수 없으면 바로 분석 (디버깅용)

                    ai_button_cm1_tag = dpg.generate_uuid()
                    # 버튼의 콜백으로 confirm_and_run_cm1_analysis를 연결해야 합니다.
                    dpg.add_button(label="💡 Analyze with AI", tag=ai_button_cm1_tag, width=150, height=25,
                                   callback=confirm_and_run_cm1_analysis) # 여기가 중요!
                    dpg.add_spacer(height=5) # 버튼과 다음 요소 간 간격
        else:
            dpg.add_text("Not enough variables for this clustermap.", parent=results_group)
    except Exception as e:
        dpg.add_text(f"Error (CM1): {e}", parent=results_group,color=(255,0,0))
        print(f"Error (CM1 traceback): {traceback.format_exc()}") # traceback 출력
    dpg.add_separator(parent=results_group)

    # --- Clustermap 2: 타겟 연관 변수 ---
    dpg.add_text(f"Clustermap 2: Top {MAX_VARS_CM} Numeric Variables - Correlated with Target '{target_var}'", parent=results_group, color=(255,255,0))
    img_bytes_cm2 = None # AI 분석용
    tex_tag2, w2, h2 = None, 0, 0
    vars_for_clustermap2 = []  # 최종적으로 Clustermap에 사용될 변수 리스트 초기화
    selection_method_description = "Not determined" # 변수 선택 방법에 대한 설명 초기화

    # 먼저 Clustermap 2의 제목을 표시할지 여부를 결정하기 위해 기본 텍스트 설정
    clustermap2_title_text = f"Clustermap 2: Top {MAX_VARS_CM} Numeric Variables" # 기본 제목

    if target_var and target_var in df.columns: # 타겟 변수가 유효하게 선택되었는지 확인
        if target_var_type == "Continuous" and target_var in num_cols:
            # --- 타겟이 연속형 수치 변수인 경우 ---
            selection_method_description = f"based on Pearson correlation with Continuous target '{target_var}'"
            other_numeric_cols_for_cont_target = [col for col in num_cols if col != target_var]

            if other_numeric_cols_for_cont_target:
                relevance_scores_cont = utils.calculate_feature_target_relevance(
                    df, target_var, target_var_type, other_numeric_cols_for_cont_target, callbacks
                )
                # 연관성 높은 (MAX_VARS_CM - 1)개의 다른 변수 선택
                top_other_vars = [var_name for var_name, score in relevance_scores_cont[:MAX_VARS_CM - 1]]
                # 타겟 변수를 맨 앞에 추가
                vars_for_clustermap2 = [target_var] + top_other_vars
                # 중복 제거(이론상 없을 것이나 안전장치) 및 최종 개수 제한
                vars_for_clustermap2 = list(dict.fromkeys(vars_for_clustermap2))[:MAX_VARS_CM]
            else: # 타겟 외 다른 수치형 변수가 없는 경우
                vars_for_clustermap2 = [target_var] # 타겟 변수만 포함 (Clustermap 생성 조건 len >=2 에 걸릴 것임)
                selection_method_description = f"target '{target_var}' is the only numeric variable."

        elif target_var_type == "Categorical":
            # --- 타겟이 범주형 변수인 경우 ---
            selection_method_description = f"based on ANOVA F-statistic with Categorical target '{target_var}'"
            target_categories = df[target_var].dropna().unique()

            if 2 <= len(target_categories) <= 10: # ANOVA에 적합한 카테고리 수 (예: 2-10개)
                # 범주형 타겟과 연관성이 높은 '수치형' 변수들을 선택
                # num_cols (수치형 변수 리스트)에 대해 연관성 계산
                features_to_check_anova = [col for col in num_cols if col != target_var] # 타겟 자신은 제외 (수치형이라도)
                
                if features_to_check_anova:
                    relevance_scores_cat = utils.calculate_feature_target_relevance(
                        df, target_var, target_var_type, features_to_check_anova, callbacks
                    )
                    # 연관성 높은 상위 MAX_VARS_CM 개의 수치형 변수 선택
                    vars_for_clustermap2 = [var_name for var_name, score in relevance_scores_cat[:MAX_VARS_CM]]
                else:
                    selection_method_description = "no numeric features to analyze with categorical target."
            else:
                selection_method_description = f"target '{target_var}' has {len(target_categories)} categories (requires 2-10 for ANOVA selection)."
        
        else: # 타겟 타입이 "Continuous"도 "Categorical"도 아니거나, 다른 조건 불충족
            selection_method_description = f"target '{target_var}' (type: '{target_var_type}') not suitable for selection."

        # Clustermap 2 제목 업데이트
        clustermap2_title_text += f" {selection_method_description}"

    else: # 타겟 변수가 아예 선택되지 않은 경우
        clustermap2_title_text += " (Skipped: No target variable selected)"
        selection_method_description = "No target selected."

    # Clustermap 2 제목 최종 표시 (선택된 변수가 있거나, 스킵 사유가 명확할 때)
    dpg.add_text(clustermap2_title_text, parent=results_group, color=(255,255,0) if len(vars_for_clustermap2) >=2 else (200,200,0) )

    # --- 최종 선택된 변수들로 Clustermap 생성 ---
    if len(vars_for_clustermap2) >= 2:
        try:
            sub_corr2 = df[vars_for_clustermap2].corr().fillna(0).replace([np.inf, -np.inf], 0)
            n_vars2 = len(vars_for_clustermap2)
            fs2 = max(0.7, 1.2 - n_vars2 * 0.02) # 이전 폰트 조정값
            fsize2 = (max(7, n_vars2 * 0.8), max(6, n_vars2 * 0.7)) # 이전 figsize값

            sns.set_theme(style="whitegrid", font_scale=fs2)
            cm2 = sns.clustermap(sub_corr2, annot=True, cmap="RdYlBu_r", fmt=".2f", linewidths=.5, vmin=-1, vmax=1, figsize=fsize2, dendrogram_ratio=0.12, cbar_kws={'shrink': .65})
            # 제목을 좀 더 일반적이게, selection_method_description은 이미 위에 텍스트로 표시됨
            cm2.fig.suptitle(f"Clustermap of Top {n_vars2} Target-Associated Numeric Variables", fontsize=15 if fs2 > 0.7 else 12, y=1.03) # 이전 폰트 조정값
            
            plot_result_cm2 = _plot_to_dpg_texture_data(cm2.fig, desired_dpi=95) # 한 번 호출
            if plot_result_cm2 and len(plot_result_cm2) == 4:
                tex_tag2, w2, h2, img_bytes_cm2_temp = plot_result_cm2
                img_bytes_cm2 = img_bytes_cm2_temp
            else:
                print("Warning: _plot_to_dpg_texture_data did not return 4 values for Clustermap 2.")

            with dpg.group(horizontal=False, parent=results_group):
                _display_dpg_image(dpg.last_item(), tex_tag2, w2, h2, max_w=700)
                if img_bytes_cm2 and tex_tag2:
                    # 여기에 Clustermap 2를 위한 AI 분석 버튼 로직 추가
                    pass # (Clustermap 1의 패턴 참고)
        except Exception as e_cm2_render:
            dpg.add_text(f"Error rendering Clustermap 2: {e_cm2_render}", parent=results_group, color=(255,0,0)); print(traceback.format_exc())
    elif target_var and target_var in df.columns : # 타겟은 있었으나 최종 선택된 변수가 2개 미만인 경우
        # 위에서 이미 dpg.add_text로 제목과 함께 스킵 사유가 표시되었으므로, 추가 메시지는 생략하거나 간결하게.
        if not (selection_method_description == "No target selected." or "not suitable for this selection" in selection_method_description or "not suitable." in selection_method_description):
             dpg.add_text(f"-> Not enough numeric variables found based on '{target_var}' for Clustermap 2.", parent=results_group, color=(200,200,0))
    # 타겟 자체가 없어서 스킵된 경우는 이미 제목에 표시됨

    dpg.add_separator(parent=results_group) # Clustermap 2와 UMAP 사이 구분선

    # --- UMAP ---
    dpg.add_text("UMAP 2D Visualization of All Numeric Variables:", parent=results_group, color=(255,255,0))
    img_bytes_umap = None # AI 분석용 이미지 바이트 저장 변수
    tex_tag_umap, w_umap, h_umap = None, 0, 0 # 이미지 표시용 변수들 초기화

    try:
        umap_prep_df = df[num_cols].copy()
        for col in umap_prep_df.columns: # Median imputation for numeric UMAP data
            if umap_prep_df[col].isnull().any() and pd.api.types.is_numeric_dtype(umap_prep_df[col]):
                umap_prep_df[col] = umap_prep_df[col].fillna(umap_prep_df[col].median())

        if umap_prep_df.shape[0] < 2 or umap_prep_df.shape[1] < 2:
            dpg.add_text("Not enough data/features for UMAP.", parent=results_group)
            # UMAP 생성 불가 시 try 블록의 나머지 부분 실행하지 않도록 return 또는 다른 처리 필요시 추가
        else:
            group_umap_cb = dpg.get_value(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX) if dpg.does_item_exist(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX) else False
            umap_hue_values = None
            umap_legend_hndls = None
            actual_umap_hue_var = None
            cmap_for_umap = 'Spectral' # 기본값
            n_cats_umap = 0 # 기본값

            if group_umap_cb and target_var and target_var in df.columns:
                target_s_for_hue = df[target_var].copy()
                if target_s_for_hue.isnull().any():
                     target_s_for_hue = target_s_for_hue.astype(str).fillna("Missing")

                unique_hue_count = target_s_for_hue.nunique(dropna=False)
                MAX_HUE_CATS_UMAP = 10

                if 2 <= unique_hue_count <= MAX_HUE_CATS_UMAP:
                    actual_umap_hue_var = target_var
                    # umap_prep_df 인덱스와 target_s_for_hue 인덱스 일치 확인 및 조정 필요
                    # 여기서는 인덱스가 호환된다고 가정합니다.
                    aligned_hue_series = target_s_for_hue.loc[umap_prep_df.index.intersection(target_s_for_hue.index)]
                    umap_hue_values = aligned_hue_series.astype('category').cat.codes

                    cats_umap = aligned_hue_series.astype('category').cat.categories
                    n_cats_umap = len(cats_umap)
                    if n_cats_umap > 0 :
                         cmap_for_umap = plt.colormaps.get_cmap('Spectral').resampled(n_cats_umap)
                    umap_legend_hndls = [plt.Line2D([0],[0], marker='o', color='w', label=str(c)[:15], markerfacecolor=cmap_for_umap(i) if n_cats_umap > 0 else 'gray', markersize=6) for i, c in enumerate(cats_umap)]
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
            else:
                scatter_kwargs_umap['cmap'] = 'viridis'

            plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], **scatter_kwargs_umap)

            if umap_legend_hndls and umap_hue_values is not None:
                plt.legend(handles=umap_legend_hndls, title=str(actual_umap_hue_var)[:15], fontsize=7.5, loc='best', frameon=True, shadow=True)

            plt.title("UMAP Projection of Numeric Variables", fontsize=11)
            plt.xlabel("UMAP Dimension 1", fontsize=9); plt.ylabel("UMAP Dimension 2", fontsize=9)
            plt.xticks(fontsize=8.5); plt.yticks(fontsize=7.5)
            plt.tight_layout()

            # _plot_to_dpg_texture_data 함수를 한 번만 호출합니다.
            plot_result_umap = _plot_to_dpg_texture_data(fig_umap_plot, desired_dpi=100) # fig_umap_plot 사용

            if plot_result_umap and len(plot_result_umap) == 4:
                tex_tag_umap, w_umap, h_umap, img_bytes_umap_temp = plot_result_umap
                img_bytes_umap = img_bytes_umap_temp # AI 분석용 이미지 바이트 저장
            else:
                print("Warning: _plot_to_dpg_texture_data did not return 4 values for UMAP.")
                # 오류 발생 시 tex_tag_umap 등이 None으로 유지

            plt.style.use('default') # 스타일 복원 (matplotlib의 기본 스타일로)

            # DPG 이미지 및 AI 분석 버튼을 담을 그룹
            with dpg.group(horizontal=False, parent=results_group): # 수직 그룹
                _display_dpg_image(dpg.last_item(), tex_tag_umap, w_umap, h_umap, max_w=750)

                if img_bytes_umap and tex_tag_umap : # 이미지와 바이트가 모두 성공적으로 생성되었을 때만 버튼 추가
                    chart_name_umap = "UMAP_Projection"
                    # AI 분석 버튼 콜백 정의
                    def analyze_umap_callback():
                        loading_indicator_tag_umap = f"loading_umap_{dpg.generate_uuid()}"
                        if dpg.does_item_exist(ai_button_umap_tag):
                            dpg.configure_item(ai_button_umap_tag, enabled=False, label="Analyzing...")
                            dpg.add_loading_indicator(tag=loading_indicator_tag_umap, parent=ai_button_umap_tag, style=0, radius=7, color=[255,255,0,255])
                        try:
                            analysis_result = ollama_analyzer.analyze_image_with_llava(img_bytes_umap, chart_name_umap)
                            if 'add_ai_log' in callbacks:
                                callbacks['add_ai_log'](analysis_result, chart_name_umap)
                            else:
                                print(f"AI Log ({chart_name_umap}):\n{analysis_result}")
                        except Exception as e_analysis:
                            err_msg = f"Error during AI analysis for {chart_name_umap}: {e_analysis}"
                            print(err_msg)
                            if 'add_ai_log' in callbacks:
                                callbacks['add_ai_log'](err_msg, chart_name_umap)
                        finally:
                            if dpg.does_item_exist(loading_indicator_tag_umap):
                                dpg.delete_item(loading_indicator_tag_umap)
                            if dpg.does_item_exist(ai_button_umap_tag):
                                dpg.configure_item(ai_button_umap_tag, enabled=True, label="💡 Analyze with AI")

                    def confirm_and_run_umap_analysis():
                        if 'get_util_funcs' in callbacks:
                            util_funcs_dict = callbacks['get_util_funcs']()
                            if 'show_confirmation_modal' in util_funcs_dict:
                                util_funcs_dict['show_confirmation_modal'](
                                    title="AI Analysis Confirmation",
                                    message=f"Proceed with AI analysis for '{chart_name_umap}'?\n(This may take a few moments)",
                                    yes_callback=analyze_umap_callback
                                )
                            else:
                                print("Confirmation modal function not found in utils. Running UMAP analysis directly.")
                                analyze_umap_callback()
                        else:
                            print("Util functions not available for UMAP. Running analysis directly.")
                            analyze_umap_callback()

                    ai_button_umap_tag = dpg.generate_uuid()
                    dpg.add_button(label="💡 Analyze with AI", tag=ai_button_umap_tag, width=150, height=25,
                                   callback=confirm_and_run_umap_analysis)
                    dpg.add_spacer(height=5)
    except ImportError:
        dpg.add_text("UMAP-learn not installed.",parent=results_group,color=(255,100,0))
    except Exception as e:
        dpg.add_text(f"Error (UMAP): {e}",parent=results_group,color=(255,0,0))
        print(f"Error (UMAP traceback): {traceback.format_exc()}")

def _mva_run_pair_plot_analysis(df: pd.DataFrame, u_funcs: dict, callbacks: dict):
    res_group = TAG_MVA_PAIRPLOT_RESULTS_GROUP
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(res_group): return
    dpg.delete_item(res_group, children_only=True)
    if df is None: dpg.add_text("Load data first.", parent=res_group); return

    MAX_VARS_PP = 8
    num_cols_all = utils._get_numeric_cols(df)
    if len(num_cols_all) < 2:
        dpg.add_text("Need at least 2 numeric columns for Pair Plot.", parent=res_group); return

    vars_for_pp = []
    # --- 변수 선택 로직 (Clustermap 1 방식) ---
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
    
    # --- Hue 옵션 처리를 위해 target_var 정의 ---
    target_var_for_hue = callbacks['get_selected_target_variable']() # <--- 이 위치로 이동 및 변수명 변경

    if group_by_target_pp_cb and target_var_for_hue and target_var_for_hue in df.columns:
        unique_target_pp_count = df[target_var_for_hue].nunique(dropna=True)
        if 2 <= unique_target_pp_count <= 7:
            hue_var_pp = target_var_for_hue # 실제 hue에 사용될 변수명
            dpg.add_text(f"Pair Plot: Using target '{target_var_for_hue}' for Hue.", parent=res_group, color=(180,180,180))
        else:
            dpg.add_text(f"Pair Plot Hue: Target '{target_var_for_hue}' has {unique_target_pp_count} unique values. Hue disabled (requires 2-7).", parent=res_group, color=(200,200,0))
    
    title_pp_str = f"Pair Plot (Top {len(vars_for_pp)} Vars): {', '.join(vars_for_pp)}" + (f" (Hue: {hue_var_pp})" if hue_var_pp else "")
    dpg.add_text(title_pp_str, parent=res_group, color=(255,255,0))
    try:
        pp_df = df.copy()
        n_vars_pp_plot = len(vars_for_pp)
        # 폰트 크기 조정 (이전 답변의 폰트 크기 조정 로직 유지)
        height_per_subplot = max(1.0, min(2.8, 13.0 / n_vars_pp_plot if n_vars_pp_plot > 0 else 2.8))
        font_scale_val_pp = max(0.7, 1.0 - n_vars_pp_plot * 0.03) # 이전 답변의 폰트 스케일 조정값

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
                except: pp_subset_df_for_grid[hue_var_pp] = pp_subset_df_for_grid[hue_var_pp].astype(str)
        
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

        def kdeplot_lower_wrapper(x, y, **kwargs):
            if x.nunique() >= 2 and y.nunique() >= 2 and len(x) >=2 :
                try: sns.kdeplot(x=x, y=y, levels=4, fill=True, alpha=0.45, linewidths=0.9, **kwargs)
                except Exception: pass
        g.map_lower(kdeplot_lower_wrapper)

        def kdeplot_diag_wrapper(x, **kwargs):
            if x.nunique() >= 2 and len(x) >=2:
                try: sns.kdeplot(x=x, fill=True, alpha=0.55, linewidth=1.1, **kwargs)
                except Exception: pass
        g.map_diag(kdeplot_diag_wrapper)
        
        if hue_var_pp and hue_var_pp in pp_subset_df_for_grid.columns:
            # 폰트 크기 조정 (이전 답변의 폰트 크기 조정 로직 유지)
            g.add_legend(title=str(hue_var_pp)[:15], 
                         fontsize=11 if n_vars_pp_plot <=7 else 10,
                         title_fontsize=11 if n_vars_pp_plot <=7 else 10)
        # 폰트 크기 조정 (이전 답변의 폰트 크기 조정 로직 유지)
        g.fig.suptitle(f"Pair Plot (Top {len(vars_for_pp)} Vars)" + (f" (Hue: {hue_var_pp})" if hue_var_pp and hue_var_pp in pp_subset_df_for_grid.columns else ""), y=1.01, fontsize=13)
        
        plot_result_pp = _plot_to_dpg_texture_data(g.fig, desired_dpi=80)
        tex_tag_pp_img, w_pp, h_pp, _ = None, 0, 0, None # 기본값 설정
        if plot_result_pp and len(plot_result_pp) == 4:
            tex_tag_pp_img, w_pp, h_pp, img_bytes_pp = plot_result_pp # 4개의 값을 모두 받음
            # img_bytes_pp 변수는 이 함수에서 당장 사용하지 않더라도 받아줘야 합니다.
            # 필요하다면 나중에 이 페어플롯에 대한 AI 분석 기능을 추가할 때 사용할 수 있습니다.
        else:
            # _plot_to_dpg_texture_data가 예상과 다른 값을 반환했을 경우에 대한 로깅 또는 처리
            print("Warning: _plot_to_dpg_texture_data did not return 4 values for Pair Plot.")
        _display_dpg_image(res_group, tex_tag_pp_img, w_pp, h_pp, max_w=850)
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
    
    MAX_VARS_CV = 15 # Cramer's V 최대 변수
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
        cv_mat_final = cv_mat_final.replace([np.inf, -np.inf], 0)
        n_cv = len(vars_for_cv_final)
        fs_cv = max(0.9, 1.4 - n_cv * 0.02)
        fsize_cv = (max(9, n_cv * 0.9), max(9, n_cv * 0.9)) # figsize 늘림

        sns.set_theme(style="white", font_scale=fs_cv)
        cm_cv = sns.clustermap(
            cv_mat_final, annot=True, cmap="Blues", fmt=".2f", linewidths=.5,
            vmin=0, vmax=1, figsize=fsize_cv, dendrogram_ratio=0.08, # 덴드로그램 비율 줄임
            cbar_kws={'shrink': .6, 'ticks': [0, 0.25, 0.5, 0.75, 1]} # 컬러바 눈금 더 자세히
        )
        cm_cv.fig.suptitle("Cramer's V Association Clustermap", fontsize=15 if fs_cv > 0.8 else 12, y=1.02)
        
        tex_tag_cv, w_cv, h_cv = _plot_to_dpg_texture_data(cm_cv.fig, desired_dpi=95)
        _display_dpg_image(res_group, tex_tag_cv, w_cv, h_cv, max_w=700) # max_w 늘림
    except ImportError: dpg.add_text("Seaborn not installed.",parent=res_group,color=(255,100,0))
    except Exception as e: dpg.add_text(f"Error (CramerV CM): {e}",parent=res_group,color=(255,0,0)); print(traceback.format_exc())

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _mva_main_app_callbacks, _mva_util_funcs
    _mva_main_app_callbacks = main_callbacks
    _mva_util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()

    with dpg.group(tag=TAG_MVA_STEP_GROUP, parent=parent_container_tag):
        with dpg.tab_bar(tag=TAG_MVA_MAIN_TAB_BAR):
            with dpg.tab(label="Correlation & UMAP (Numeric)", tag=TAG_MVA_CORR_TAB): # 탭 이름 변경
                dpg.add_text("Displays clustermaps of numeric correlations and UMAP projection.", wrap=-1)
                dpg.add_checkbox(label="Group UMAP by Target (if Target suitable: 2-10 unique values)", tag=TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX, default_value=False) # UMAP 그룹핑 체크박스
                dpg.add_button(label="Run Correlation Analysis & UMAP", tag=TAG_MVA_CORR_RUN_BUTTON, width=-1, height=30,
                             callback=lambda: _mva_run_correlation_analysis(_mva_main_app_callbacks['get_current_df'](), _mva_util_funcs, _mva_main_app_callbacks))
                dpg.add_child_window(tag=TAG_MVA_CORR_RESULTS_GROUP, border=True)
            
            with dpg.tab(label="Pair Plot (Numeric)", tag=TAG_MVA_PAIRPLOT_TAB):
                dpg.add_text("Generates pair plots for relevant numeric variables (auto-selected, max 15).", wrap=-1)
                dpg.add_checkbox(label="Group Pair Plot by Target (if Target suitable: 2-7 unique values)", tag=TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX, default_value=False)
                dpg.add_button(label="Generate Pair Plot", tag=TAG_MVA_PAIRPLOT_RUN_BUTTON, width=-1, height=30,
                             callback=lambda: _mva_run_pair_plot_analysis(_mva_main_app_callbacks['get_current_df'](),_mva_util_funcs, _mva_main_app_callbacks))
                dpg.add_child_window(tag=TAG_MVA_PAIRPLOT_RESULTS_GROUP, border=True)

            with dpg.tab(label="Association (Categorical)", tag=TAG_MVA_CAT_EDA_TAB):
                dpg.add_text("Shows Cramer's V clustermap for associated categorical variables (auto-selected, max 15).", wrap=-1)
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
        for area_tag, msg in [(TAG_MVA_CORR_RESULTS_GROUP, "Load data."), (TAG_MVA_PAIRPLOT_RESULTS_GROUP, "Load data."), (TAG_MVA_CAT_EDA_RESULTS_GROUP, "Load data.")]:
            if dpg.does_item_exist(area_tag): dpg.delete_item(area_tag, children_only=True); dpg.add_text(msg, parent=area_tag)

def reset_mva_ui_defaults():
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX): dpg.set_value(TAG_MVA_CORR_UMAP_GROUP_BY_TARGET_CHECKBOX, False)
    if dpg.does_item_exist(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX): dpg.set_value(TAG_MVA_PAIRPLOT_GROUP_BY_TARGET_CHECKBOX, False)
    for area_tag, msg in [(TAG_MVA_CORR_RESULTS_GROUP, "Run analysis."), (TAG_MVA_PAIRPLOT_RESULTS_GROUP, "Generate plot."), (TAG_MVA_CAT_EDA_RESULTS_GROUP, "Run analysis.")]:
        if dpg.does_item_exist(area_tag): dpg.delete_item(area_tag, children_only=True); dpg.add_text(msg, parent=area_tag)