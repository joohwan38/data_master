# step_05_outlier_treatment.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import io
from PIL import Image
import traceback
import matplotlib as plt

# 하위 모듈 임포트 (실제 파일 생성 후 상대 경로로 수정 필요시 수정)
import step_05a_univariate_outliers as uni_module
import step_05b_multivariate_outliers as mva_module

# --- DPG Tags for Step 5 (Parent) ---
TAG_OT_STEP_GROUP = "step5_outlier_treatment_group"
TAG_OT_LOG_TEXT_AREA = "step5_ot_log_text_area"
TAG_OT_LOG_TEXT = "step5_ot_log_text"

# 기본 텍스처 태그 (하위 모듈에서 문자열로 참조 가능)
# 이 태그들은 create_ui에서 생성됩니다.
TAG_OT_DEFAULT_PLOT_TEXTURE_UNI = "step5_ot_default_plot_texture_uni"
TAG_OT_DEFAULT_PLOT_TEXTURE_MVA = "step5_ot_default_plot_texture_mva"
TAG_OT_MVA_SHAP_DEFAULT_TEXTURE = "step5_ot_mva_shap_default_texture" # SHAP용 기본 텍스처 태그
TAG_OT_MVA_GROUP_DIST_DEFAULT_TEXTURE = "step5_ot_mva_group_dist_default_texture"
TAG_OT_MVA_GROUP_FREQ_DEFAULT_TEXTURE = "step5_ot_mva_group_freq_default_texture"
TAG_OT_MVA_UMAP_DEFAULT_TEXTURE = "step5_ot_mva_umap_default_texture"
TAG_OT_MVA_PCA_DEFAULT_TEXTURE = "step5_ot_mva_pca_default_texture"
TAG_OT_DEFAULT_PLOT_TEXTURE_UNI = "step5_ot_default_plot_texture_uni"


# --- Module State Variables (Parent) ---
_main_app_callbacks_parent: Optional[Dict[str, Any]] = None
_util_funcs_parent: Optional[Dict[str, Any]] = None
_current_df_for_this_step_parent: Optional[pd.DataFrame] = None
_step_05_shared_utilities: Optional[Dict[str, Any]] = None


def _s5_plot_to_dpg_texture_parent(fig: 'plt.Figure', desired_dpi: int = 90) -> Tuple[Optional[str], int, int]:
    """
    Matplotlib figure를 DPG 텍스처로 변환합니다.
    이 함수는 이제 step_05_outlier_treatment.py (부모 모듈)에 위치하며,
    하위 모듈 (uni_module, mva_module)에 전달되어 사용됩니다.
    """
    img_data_buf = io.BytesIO()
    try:
        # Matplotlib Figure는 함수 호출 시점에 plt 객체를 직접 참조하지 않도록 fig를 인자로 받음
        fig.savefig(img_data_buf, format="png", bbox_inches='tight', dpi=desired_dpi)
        img_data_buf.seek(0)
        pil_image = Image.open(img_data_buf)
        if pil_image.mode != 'RGBA': pil_image = pil_image.convert('RGBA')
        img_width, img_height = pil_image.size
        if img_width == 0 or img_height == 0:
            _log_message_parent(f"Error: Plot image has zero dimension ({img_width}x{img_height}). Cannot create texture.")
            # plt.close(fig) # fig는 외부에서 관리
            return None, 0, 0
        texture_data_np = np.array(pil_image).astype(np.float32) / 255.0
        texture_data_flat_list = texture_data_np.ravel().tolist()
        
        # 텍스처 레지스트리는 create_ui에서 한 번만 생성되도록 보장
        if not dpg.does_item_exist("texture_registry"):
             # 이 경우는 create_ui에서 처리되므로, 실제로는 거의 발생하지 않음
            dpg.add_texture_registry(tag="texture_registry", show=False)

        texture_tag = dpg.generate_uuid()
        dpg.add_static_texture(width=img_width, height=img_height, default_value=texture_data_flat_list, tag=texture_tag, parent="texture_registry")
        return texture_tag, img_width, img_height
    except Exception as e:
        _log_message_parent(f"Error converting plot to DPG texture: {e}\n{traceback.format_exc()}")
        return None, 0, 0
    # finally: # fig는 외부에서 관리하므로 여기서 close 하지 않음
        # plt.close(fig)

def _log_message_parent(message: str):
    """
    Step 5의 공통 로그 영역에 메시지를 기록합니다.
    이 함수는 부모 모듈에 있으며, 하위 모듈에 전달됩니다.
    """
    if not dpg.is_dearpygui_running(): return

    if dpg.does_item_exist(TAG_OT_LOG_TEXT):
        current_log = dpg.get_value(TAG_OT_LOG_TEXT)
        max_log_entries = 200
        log_lines = current_log.splitlines()
        if len(log_lines) >= max_log_entries:
            log_lines = log_lines[-(max_log_entries-1):]

        new_log_entry = f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}"
        new_log = "\n".join(log_lines) + "\n" + new_log_entry if current_log else new_log_entry

        dpg.set_value(TAG_OT_LOG_TEXT, new_log.strip())

        if dpg.does_item_exist(TAG_OT_LOG_TEXT_AREA):
            # Scroll to bottom - DPG의 버전에 따라 완벽하게 작동하지 않을 수 있음
            # dpg.set_y_scroll(TAG_OT_LOG_TEXT_AREA, dpg.get_y_scroll_max(TAG_OT_LOG_TEXT_AREA))
            # 좀 더 안정적인 방법은 item_info를 통해 child window인지 확인 후 y_scroll을 -1.0으로 설정
            item_config = dpg.get_item_configuration(TAG_OT_LOG_TEXT_AREA)
            is_shown = item_config.get('show', True)
            item_info = dpg.get_item_info(TAG_OT_LOG_TEXT_AREA)
            is_child_window = item_info['type'] == "mvAppItemType::mvChildWindow" if item_info else False

            if is_shown and is_child_window:
                 dpg.set_y_scroll(TAG_OT_LOG_TEXT_AREA, -1.0) # 음수 값은 맨 아래로 스크롤


def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _main_app_callbacks_parent, _util_funcs_parent, _step_05_shared_utilities
    _main_app_callbacks_parent = main_callbacks
    if 'get_util_funcs' in main_callbacks:
        _util_funcs_parent = main_callbacks['get_util_funcs']()

    # _step_05_shared_utilities 딕셔너리 구성 시 새로운 기본 텍스처 태그 추가
    _step_05_shared_utilities = {
        "main_app_callbacks": _main_app_callbacks_parent,
        "util_funcs_common": _util_funcs_parent,
        "log_message_func": _log_message_parent,
        "plot_to_dpg_texture_func": _s5_plot_to_dpg_texture_parent,
        "get_current_df_func": lambda: _current_df_for_this_step_parent,
        "default_uni_plot_texture_tag": TAG_OT_DEFAULT_PLOT_TEXTURE_UNI, # 이 부분이 중요!
        "default_umap_texture_tag": TAG_OT_MVA_UMAP_DEFAULT_TEXTURE,
        "default_pca_texture_tag": TAG_OT_MVA_PCA_DEFAULT_TEXTURE,
        "default_shap_plot_texture_tag": TAG_OT_MVA_SHAP_DEFAULT_TEXTURE,
        "default_mva_plot_texture_tag": TAG_OT_DEFAULT_PLOT_TEXTURE_MVA,
        "default_group_dist_plot_texture_tag": TAG_OT_MVA_GROUP_DIST_DEFAULT_TEXTURE, # <--- 추가
        "default_group_freq_plot_texture_tag": TAG_OT_MVA_GROUP_FREQ_DEFAULT_TEXTURE, # <--- 추가
    }

    if not dpg.does_item_exist("texture_registry"):
        dpg.add_texture_registry(tag="texture_registry", show=False)

    # 기본 텍스처 생성
    default_textures_to_create = {
        TAG_OT_DEFAULT_PLOT_TEXTURE_UNI: (10,10),
        TAG_OT_DEFAULT_PLOT_TEXTURE_MVA: (10,10), # UMAP DPG 네이티브 플롯으로 대체되었지만, 혹시 다른 곳 참조 가능성
        TAG_OT_MVA_UMAP_DEFAULT_TEXTURE: (10,10), # 예시 크기
        TAG_OT_MVA_PCA_DEFAULT_TEXTURE: (10,10),   # 예시 크기
        TAG_OT_MVA_SHAP_DEFAULT_TEXTURE: (10,10), # 이미 존재할 수 있음
        TAG_OT_MVA_GROUP_DIST_DEFAULT_TEXTURE: (10,10), # <--- 추가
        TAG_OT_MVA_GROUP_FREQ_DEFAULT_TEXTURE: (10,10), # <--- 추가
    }
    for tag, (w, h) in default_textures_to_create.items():
        if not dpg.does_item_exist(tag):
            dpg.add_static_texture(width=w, height=h, default_value=[0.0]*(w*h*4), tag=tag, parent="texture_registry")
            _log_message_parent(f"Created default texture: {tag}")

    main_callbacks['register_step_group_tag'](step_name, TAG_OT_STEP_GROUP)
    with dpg.group(tag=TAG_OT_STEP_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---"); dpg.add_separator()
        # ... (탭 바 및 하위 모듈 UI 생성 호출 등)
        with dpg.tab_bar(tag="step5_outlier_tab_bar"):
            uni_module.create_univariate_ui("step5_outlier_tab_bar", _step_05_shared_utilities)
            mva_module.create_multivariate_ui("step5_outlier_tab_bar", _step_05_shared_utilities)

        dpg.add_separator()
        dpg.add_text("Processing Log (Common for Step 5)", color=[255, 255, 0])
        with dpg.child_window(tag=TAG_OT_LOG_TEXT_AREA, height=200, border=True):
            # dpg.add_text("Logs will appear here...", tag=TAG_OT_LOG_TEXT, wrap=-1)
            dpg.add_input_text(tag=TAG_OT_LOG_TEXT, default_value="Logs will appear here...", multiline=True, readonly=True, width=-1, height=-1) # 변경

    main_callbacks['register_module_updater'](step_name, update_ui)


def update_ui(df_input_for_step: Optional[pd.DataFrame], main_callbacks: dict):
    global _main_app_callbacks_parent, _util_funcs_parent, _current_df_for_this_step_parent, _step_05_shared_utilities

    if not dpg.is_dearpygui_running(): return

    # 콜백 및 유틸리티 최신 상태로 유지 (필요시)
    if not _main_app_callbacks_parent: _main_app_callbacks_parent = main_callbacks
    if not _util_funcs_parent and 'get_util_funcs' in _main_app_callbacks_parent:
        _util_funcs_parent = _main_app_callbacks_parent['get_util_funcs']()
    
    # _step_05_shared_utilities 업데이트 (콜백 등이 변경될 수 있으므로)
    if _step_05_shared_utilities: # 이미 생성되었다면 업데이트
        _step_05_shared_utilities["main_app_callbacks"] = _main_app_callbacks_parent
        _step_05_shared_utilities["util_funcs_common"] = _util_funcs_parent
        _step_05_shared_utilities["get_current_df_func"] = lambda: _current_df_for_this_step_parent


    is_new_data = _current_df_for_this_step_parent is not df_input_for_step
    _current_df_for_this_step_parent = df_input_for_step

    if not dpg.does_item_exist(TAG_OT_STEP_GROUP): return

    # 데이터가 새로 로드되거나 변경된 경우, 하위 모듈의 상태도 리셋 필요
    # 또는 각 하위 모듈의 update_ui에서 df_input_for_step이 None이거나 new_data일 때 자체적으로 리셋하도록 유도
    if _current_df_for_this_step_parent is None or is_new_data:
        # 부모 레벨에서의 메시지 로깅
        msg_parent = "Data loaded for Step 5. Configure detection." if _current_df_for_this_step_parent is not None else "Load data for Step 5 or previous step output is missing."
        _log_message_parent(msg_parent)
        # 하위 모듈의 reset 함수를 호출하여 각자 상태를 초기화하도록 할 수 있음
        # 또는 update_ui에 is_new_data 플래그를 전달하여 처리
    
    # 하위 모듈의 UI 업데이트 함수 호출
    if hasattr(uni_module, 'update_univariate_ui'):
        uni_module.update_univariate_ui(df_input_for_step, _step_05_shared_utilities, is_new_data)
    
    if hasattr(mva_module, 'update_multivariate_ui'):
        mva_module.update_multivariate_ui(df_input_for_step, _step_05_shared_utilities, is_new_data)


def reset_outlier_treatment_state():
    # 이 함수는 main_app에서 호출될 때 Step 5 전체의 상태를 리셋
    global _current_df_for_this_step_parent
    
    _current_df_for_this_step_parent = None # 부모의 현재 DF 참조도 초기화
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_OT_LOG_TEXT):
        dpg.set_value(TAG_OT_LOG_TEXT, "Step 5 state reset. Logs cleared.")

    if hasattr(uni_module, 'reset_univariate_state'):
        uni_module.reset_univariate_state()
    if hasattr(mva_module, 'reset_multivariate_state'):
        mva_module.reset_multivariate_state()
    
    _log_message_parent("Step 5: Outlier Treatment state (Univariate & Multivariate) has been reset via parent.")

def get_outlier_treatment_settings_for_saving() -> dict:
    settings = {}
    if hasattr(uni_module, 'get_univariate_settings'):
        settings["univariate"] = uni_module.get_univariate_settings()
    else:
        settings["univariate"] = {}
        
    if hasattr(mva_module, 'get_multivariate_settings'):
        settings["multivariate"] = mva_module.get_multivariate_settings()
    else:
        settings["multivariate"] = {}
        
    return settings

def apply_outlier_treatment_settings_and_process(df_input: pd.DataFrame, settings: dict, main_callbacks: dict):
    # 이 함수는 저장된 설정을 불러와 각 하위 모듈에 전달하여 상태를 복원하는 역할
    # 실제 "처리(process)"는 사용자가 버튼을 눌러 수동으로 진행하는 현재의 흐름을 유지
    global _current_df_for_this_step_parent, _main_app_callbacks_parent, _step_05_shared_utilities

    if not _main_app_callbacks_parent: _main_app_callbacks_parent = main_callbacks # 콜백 설정

    _current_df_for_this_step_parent = df_input # 입력 DF 설정

    # _step_05_shared_utilities가 최신 콜백과 df getter를 참조하도록 보장
    if _step_05_shared_utilities:
        _step_05_shared_utilities["main_app_callbacks"] = _main_app_callbacks_parent
        _step_05_shared_utilities["get_current_df_func"] = lambda: _current_df_for_this_step_parent
    else: # 초기화되지 않은 경우를 대비 (create_ui가 먼저 호출되는 것이 일반적)
         _step_05_shared_utilities = {
            "main_app_callbacks": _main_app_callbacks_parent,
            "util_funcs_common": _main_app_callbacks_parent['get_util_funcs']() if 'get_util_funcs' in _main_app_callbacks_parent else {},
            "log_message_func": _log_message_parent,
            "plot_to_dpg_texture_func": _s5_plot_to_dpg_texture_parent,
            "get_current_df_func": lambda: _current_df_for_this_step_parent,
            "default_uni_plot_texture_tag": TAG_OT_DEFAULT_PLOT_TEXTURE_UNI,
            "default_mva_plot_texture_tag": TAG_OT_DEFAULT_PLOT_TEXTURE_MVA,
        }


    uni_settings = settings.get("univariate", {})
    if hasattr(uni_module, 'apply_univariate_settings'):
        uni_module.apply_univariate_settings(df_input, uni_settings, _step_05_shared_utilities)
    
    mva_settings = settings.get("multivariate", {})
    if hasattr(mva_module, 'apply_multivariate_settings'):
        # 다변량 분석은 단변량 분석 결과에 영향을 받을 수 있으나,
        # 현재 설정 적용은 각자 독립적으로 UI와 파라미터를 복원하는 개념
        mva_module.apply_multivariate_settings(df_input, mva_settings, _step_05_shared_utilities)

    # update_ui를 호출하여 전체 UI를 반영된 설정으로 새로고침
    # apply_..._settings 함수 내부에서 이미 UI 업데이트가 이루어졌다면 중복일 수 있음
    # 각 하위 모듈의 apply_..._settings가 자신의 update_ui를 호출하도록 하거나, 여기서 한 번만 호출
    update_ui(df_input, main_callbacks) # 부모의 update_ui를 통해 하위 모듈 update_ui 호출

    _log_message_parent("Step 5 Outlier Treatment settings applied from saved state. UI reflects parameters. Please run detection and treatment manually if needed.")