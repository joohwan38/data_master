# main_app.py
import faulthandler
faulthandler.enable()
import dearpygui.dearpygui as dpg
import pandas as pd
import os
import utils
from typing import Optional, Dict, List, Any, Callable
import step_01_data_loading
import step_02a_sva
import step_02b_mva
import step_03_preprocessing
import step_04_missing_values
import step_05_outlier_treatment
import traceback
import hashlib
import json
import ollama_integration
from io import BytesIO
import platform
import threading

STEP_03_SAVE_LOAD_ENABLED = True

class AppState:
    def __init__(self):
        self.current_df = None
        self.original_df = None
        self.df_after_step1 = None
        self.df_after_step3 = None
        self.df_after_step4 = None
        self.df_after_step5 = None
        self.loaded_file_path = None
        self.selected_target_variable = None
        self.selected_target_variable_type = "Continuous"
        self.active_step_name = None
        self.active_settings = {}
        self.step_group_tags = {}
        self.module_ui_updaters = {}
        self.image_data_cache: Dict[str, BytesIO] = {}
        self.callback_queue: List[List[Any]] = []

        # Ollama 분석 관련 상태 추가
        self.is_ollama_analysis_in_progress: bool = False
        self.ollama_confirm_user_data: Optional[Dict] = None # 이미지 제목, 텍스처 태그 임시 저장

app_state = AppState()

SETTINGS_DIR_NAME = ".app_file_settings"
SESSION_INFO_FILE = os.path.join(SETTINGS_DIR_NAME, "session_info.json")

SVA_STEP_KEY = "2a. Single Variable Analysis"
MVA_STEP_KEY = "2b. Multivariate Analysis"
TARGET_VARIABLE_TYPE_RADIO_TAG = "main_target_variable_type_radio"
TARGET_VARIABLE_TYPE_LABEL_TAG = "main_target_variable_type_label"
TARGET_VARIABLE_COMBO_TAG = "main_target_variable_combo"
MAIN_FILE_PATH_DISPLAY_TAG = "main_file_path_display_text"

# 기존 POPUP_TAG 대신 새로운 태그 사용 또는 구분 필요
# POPUP_TAG = "ollama_analysis_popup_v2" # 이 태그는 ollama_integration.py의 레거시 팝업용이므로 유지하거나 삭제. 여기서는 새 창 사용.

ANALYSIS_STEPS = [
    "1. Data Loading & Overview",
    "2. Exploratory Data Analysis (EDA)",
    "3. Preprocessing (Node Editor)",
    "4. Missing Value Treatment",
    "5. Outlier Treatment",
]

_MODAL_ID_SIMPLE_MESSAGE = "main_simple_modal_message_id"

# --- Ollama 분석 확인 팝업 및 결과창 태그 ---
OLLAMA_CONFIRMATION_POPUP_TAG = "ollama_confirmation_popup"
OLLAMA_RESULT_WINDOW_TAG = "ollama_result_window"
OLLAMA_RESULT_WINDOW_TITLE_TEXT_TAG = "ollama_result_window_title_text"
OLLAMA_RESULT_WINDOW_CONTENT_TEXT_TAG = "ollama_result_window_content_text"
OLLAMA_RESULT_WINDOW_STATUS_TEXT_TAG = "ollama_result_window_status_text"
OLLAMA_RESULT_WINDOW_LOADING_SPINNER_TAG = "ollama_result_window_loading_spinner"


# --- Ollama 연동 콜백 함수들 (main_app 관점) ---
def get_cached_image_bytes_for_ollama(texture_tag: str) -> Optional[BytesIO]:
    return app_state.image_data_cache.get(texture_tag)

def cache_image_data_for_ollama(texture_tag: str, image_bytes_io: BytesIO):
    app_state.image_data_cache[texture_tag] = image_bytes_io
    print(f"이미지 캐시됨: {texture_tag}, 크기: {image_bytes_io.getbuffer().nbytes} bytes")

def add_job_to_main_app_queue(job: List[Any]):
    app_state.callback_queue.append(job)

def _ollama_analysis_confirmed_callback(sender, app_data, user_data):
    """Ollama 분석 확인 팝업에서 '예'를 클릭했을 때 호출되는 함수"""
    dpg.configure_item(OLLAMA_CONFIRMATION_POPUP_TAG, show=False)
    if app_state.ollama_confirm_user_data:
        if app_state.is_ollama_analysis_in_progress:
            _show_simple_modal_message("알림", "이미 다른 이미지 분석 작업이 진행 중입니다.")
            return

        # main_app_callbacks를 통해 ollama_integration 모듈의 함수를 호출
        # 이 때, 분석 시작을 알리고, 결과창을 띄우는 로직도 함께 실행
        if 'initiate_ollama_analysis_with_window' in main_app_callbacks:
             main_app_callbacks['initiate_ollama_analysis_with_window'](app_state.ollama_confirm_user_data)
        else:
            _show_simple_modal_message("오류", "Ollama 분석 시작 함수를 찾을 수 없습니다.")
        app_state.ollama_confirm_user_data = None # 사용 후 초기화
    else:
        _show_simple_modal_message("오류", "Ollama 분석을 위한 사용자 데이터가 없습니다.")

def _ollama_analysis_cancelled_callback(sender, app_data, user_data):
    """Ollama 분석 확인 팝업에서 '아니오'를 클릭했을 때 호출되는 함수"""
    dpg.configure_item(OLLAMA_CONFIRMATION_POPUP_TAG, show=False)
    app_state.ollama_confirm_user_data = None # 사용하지 않았으므로 초기화

def _show_ollama_confirmation_popup(user_data_for_analysis: Dict):
    """Ollama 분석 진행 여부를 묻는 팝업창을 표시합니다."""
    if app_state.is_ollama_analysis_in_progress:
        _show_simple_modal_message("알림", "이미 다른 이미지 분석 작업이 진행 중입니다.\n완료 후 다시 시도해주세요.")
        return

    app_state.ollama_confirm_user_data = user_data_for_analysis # 분석 시작 시 사용할 데이터 저장
    image_title = user_data_for_analysis.get("title", "해당 이미지")

    if dpg.does_item_exist(OLLAMA_CONFIRMATION_POPUP_TAG):
        dpg.configure_item(OLLAMA_CONFIRMATION_POPUP_TAG, label=f"'{image_title}' 분석 요청")
        if dpg.does_item_exist(OLLAMA_CONFIRMATION_POPUP_TAG + "_message"):
            dpg.set_value(OLLAMA_CONFIRMATION_POPUP_TAG + "_message", f"'{image_title}'에 대해 AI 분석을 시작하시겠습니까?\n(Ollama {ollama_integration.OLLAMA_MODEL_NAME} 모델 사용)")
        dpg.configure_item(OLLAMA_CONFIRMATION_POPUP_TAG, show=True)
    else:
        vp_w, vp_h = dpg.get_viewport_width(), dpg.get_viewport_height()
        popup_width, popup_height = 450, 180 # 팝업 크기 조정
        with dpg.window(label=f"'{image_title}' 분석 요청", modal=True, show=True, tag=OLLAMA_CONFIRMATION_POPUP_TAG,
                        no_close=True, width=popup_width, height=popup_height, no_saved_settings=True,
                        pos=[(vp_w - popup_width) // 2, (vp_h - popup_height) // 2]):
            dpg.add_text(f"'{image_title}'에 대해 AI 분석을 시작하시겠습니까?\n(Ollama {ollama_integration.OLLAMA_MODEL_NAME} 모델 사용)", tag=OLLAMA_CONFIRMATION_POPUP_TAG + "_message", wrap=popup_width - 20)
            dpg.add_spacer(height=15)
            with dpg.group(horizontal=True):
                btn_width = 100
                spacer_w = (popup_width - (btn_width * 2) - (dpg.get_item_configuration(dpg.last_item())["item_spacing"][0] * 3) - 30) / 2 # 여백 고려
                if spacer_w < 0 : spacer_w = 0
                dpg.add_spacer(width=int(spacer_w))
                dpg.add_button(label="예", width=btn_width, callback=_ollama_analysis_confirmed_callback)
                dpg.add_spacer(width=10)
                dpg.add_button(label="아니오", width=btn_width, callback=_ollama_analysis_cancelled_callback)

def create_ollama_result_window():
    """Ollama 분석 결과를 표시할 별도의 윈도우 (초기에는 숨김)"""
    if not dpg.does_item_exist(OLLAMA_RESULT_WINDOW_TAG):
        with dpg.window(label="Ollama 이미지 분석 결과", show=False, tag=OLLAMA_RESULT_WINDOW_TAG,
                        width=500, height=350, autosize=False, no_collapse=True,
                        no_saved_settings=True, pos=[100, 100],
                        on_close=lambda: dpg.configure_item(OLLAMA_RESULT_WINDOW_TAG, show=False)): # 닫기 버튼 누르면 숨김
            dpg.add_text("제목: ", tag=OLLAMA_RESULT_WINDOW_TITLE_TEXT_TAG)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_loading_indicator(tag=OLLAMA_RESULT_WINDOW_LOADING_SPINNER_TAG, show=False, style=0, radius=3.5, color=(50,150,255,255))
                dpg.add_spacer(width=5)
                dpg.add_text("상태: 대기 중", tag=OLLAMA_RESULT_WINDOW_STATUS_TEXT_TAG, color=(200,200,100))
            dpg.add_input_text(tag=OLLAMA_RESULT_WINDOW_CONTENT_TEXT_TAG, multiline=True, readonly=True, width=-1, height=-1, default_value="분석 결과를 기다리는 중...")
            # 윈도우 닫기 버튼은 윈도우 자체 기능 사용 (on_close)

def show_and_update_ollama_result_window(title: str, initial_status: str = "요청 준비 중...", is_loading: bool = True):
    """Ollama 결과 창을 보이게 하고 초기 메시지 및 상태 설정"""
    if not dpg.does_item_exist(OLLAMA_RESULT_WINDOW_TAG):
        create_ollama_result_window() # 없으면 생성

    dpg.configure_item(OLLAMA_RESULT_WINDOW_TAG, label=f"Ollama 분석: {title[:30]}{'...' if len(title)>30 else ''}", show=True)
    if dpg.does_item_exist(OLLAMA_RESULT_WINDOW_TITLE_TEXT_TAG):
        dpg.set_value(OLLAMA_RESULT_WINDOW_TITLE_TEXT_TAG, f"분석 대상: {title}")
    if dpg.does_item_exist(OLLAMA_RESULT_WINDOW_STATUS_TEXT_TAG):
        dpg.set_value(OLLAMA_RESULT_WINDOW_STATUS_TEXT_TAG, f"상태: {initial_status}")
    if dpg.does_item_exist(OLLAMA_RESULT_WINDOW_LOADING_SPINNER_TAG):
        dpg.configure_item(OLLAMA_RESULT_WINDOW_LOADING_SPINNER_TAG, show=is_loading)
    if dpg.does_item_exist(OLLAMA_RESULT_WINDOW_CONTENT_TEXT_TAG):
        dpg.set_value(OLLAMA_RESULT_WINDOW_CONTENT_TEXT_TAG, "분석을 시작합니다..." if is_loading else "오류 또는 정보 없음")

def update_ollama_result_window_content_from_job(sender, app_data, user_data: Dict[str, Any]):
    """ollama_integration에서 보낸 작업으로 결과 창 업데이트 (메인 스레드에서 실행)"""
    title = user_data.get("title", "분석 결과") # 이 title은 이미지의 원본 제목
    content = user_data.get("content", "내용 없음")
    is_loading = user_data.get("loading", False)
    status_message = user_data.get("status_message", "완료" if not is_loading else "처리 중...")
    error_occurred = user_data.get("error", False)

    if not dpg.is_dearpygui_running(): return
    if not dpg.does_item_exist(OLLAMA_RESULT_WINDOW_TAG): return # 창이 없으면 아무것도 안 함

    # 창이 숨겨져 있다면 보이도록 설정 (예: 오류 발생 시에도 결과창을 띄워 알림)
    if not dpg.is_item_shown(OLLAMA_RESULT_WINDOW_TAG):
        dpg.configure_item(OLLAMA_RESULT_WINDOW_TAG, show=True)

    dpg.configure_item(OLLAMA_RESULT_WINDOW_TAG, label=f"Ollama 분석: {title[:30]}{'...' if len(title)>30 else ''}")
    if dpg.does_item_exist(OLLAMA_RESULT_WINDOW_TITLE_TEXT_TAG):
         dpg.set_value(OLLAMA_RESULT_WINDOW_TITLE_TEXT_TAG, f"분석 대상: {title}")
    if dpg.does_item_exist(OLLAMA_RESULT_WINDOW_CONTENT_TEXT_TAG):
        formatted_content = content.replace("\\n", "\n")
        dpg.set_value(OLLAMA_RESULT_WINDOW_CONTENT_TEXT_TAG, formatted_content)
    if dpg.does_item_exist(OLLAMA_RESULT_WINDOW_LOADING_SPINNER_TAG):
        dpg.configure_item(OLLAMA_RESULT_WINDOW_LOADING_SPINNER_TAG, show=is_loading)
    if dpg.does_item_exist(OLLAMA_RESULT_WINDOW_STATUS_TEXT_TAG):
        final_status_message = f"상태: {status_message}"
        if error_occurred: final_status_message += " (오류 발생)"
        dpg.set_value(OLLAMA_RESULT_WINDOW_STATUS_TEXT_TAG, final_status_message)

    # 분석이 완료되었거나 오류가 발생했고, 로딩 중이 아니라면 상태 플래그 해제
    if not is_loading:
        set_ollama_analysis_status(False) # 여기서 호출하거나, ollama_integration에서 명시적으로 호출

def set_ollama_analysis_status(status: bool):
    app_state.is_ollama_analysis_in_progress = status
    print(f"Ollama 분석 상태 변경: {'진행 중' if status else '종료됨'}")

# --- (이전 이벤트 핸들러 관련 코드 삭제) ---

def _show_simple_modal_message(title: str, message: str, width: int = 450, height: int = 200):
    if dpg.does_item_exist(_MODAL_ID_SIMPLE_MESSAGE): dpg.delete_item(_MODAL_ID_SIMPLE_MESSAGE)
    vp_w = dpg.get_viewport_width(); vp_h = dpg.get_viewport_height()
    modal_x, modal_y = (vp_w - width) // 2, (vp_h - height) // 2
    with dpg.window(label=title, modal=True, show=True, id=_MODAL_ID_SIMPLE_MESSAGE, no_close=True,
                    width=width, height=height, pos=[modal_x, modal_y], no_saved_settings=True, autosize=False):
        dpg.add_text(message, wrap=width - 20); dpg.add_spacer(height=15)
        with dpg.group(horizontal=True):
            item_spacing_x = 8.0
            button_width = 100; spacer_w = (width - button_width - (item_spacing_x * 2)) / 2
            if spacer_w < 0: spacer_w = 0
            dpg.add_spacer(width=int(spacer_w))
            dpg.add_button(label="OK", width=button_width, callback=lambda: dpg.configure_item(_MODAL_ID_SIMPLE_MESSAGE, show=False))

def setup_korean_font():
    font_path, font_size, os_type = None, 17, platform.system()
    font_paths = {
        "Darwin": ["fonts/NanumBarunGothic.otf","/System/Library/Fonts/AppleSDGothicNeo.ttc", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"],
        "Windows": ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc"],
        "Linux": ["NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
    }
    for p in font_paths.get(os_type, []):
        if os.path.exists(p): font_path = p; break
    if font_path:
        try:
            with dpg.font_registry():
                font_id = dpg.add_font(font_path, font_size, tag="korean_font_for_app")
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean, parent=font_id)
                dpg.bind_font(font_id)
            print(f"Korean font for comments/internal use bound: {font_path}")
        except Exception as e: print(f"Font error: {e}"); traceback.print_exc()
    else: print("Korean font not found. Using default.")


def update_target_variable_combo():
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        df_for_combo = app_state.current_df
        items = [""] + list(df_for_combo.columns) if df_for_combo is not None and not df_for_combo.empty else [""]
        dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=items)
        current_val = app_state.selected_target_variable
        if current_val and df_for_combo is not None and current_val in df_for_combo.columns:
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, current_val)
        else:
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
            app_state.selected_target_variable = None
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)

def step1_processing_complete(processed_df: pd.DataFrame):
    if processed_df is None:
        print("Step 1 returned no DataFrame.")
        _show_simple_modal_message("Step 1 Error", "Data processing in Step 1 did not return a valid dataset.")
        app_state.df_after_step1 = None
        app_state.current_df = None
    else:
        print("Step 1 processing complete. Updating app_state.df_after_step1 and current_df.")
        app_state.df_after_step1 = processed_df.copy()
        app_state.current_df = app_state.df_after_step1.copy()
    app_state.df_after_step3 = None
    app_state.df_after_step4 = None
    app_state.df_after_step5 = None
    if app_state.selected_target_variable and \
       (app_state.df_after_step1 is None or app_state.selected_target_variable not in app_state.df_after_step1.columns):
        app_state.selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
    update_target_variable_combo()
    if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0 and ANALYSIS_STEPS[0] in app_state.module_ui_updaters:
        print(f"Explicitly triggering UI update for Step 1: {ANALYSIS_STEPS[0]}")
        trigger_specific_module_update(ANALYSIS_STEPS[0])
    trigger_all_module_updates_except_step1()

def step3_processing_complete(processed_df: pd.DataFrame):
    if processed_df is None:
        print("Step 3 (Node Editor) returned no DataFrame.")
        app_state.current_df = app_state.df_after_step1.copy() if app_state.df_after_step1 is not None else None
        app_state.df_after_step3 = None
    else:
        print("Step 3 (Node Editor) processing complete. Updating app_state.")
        app_state.df_after_step3 = processed_df.copy()
        app_state.current_df = app_state.df_after_step3.copy()
    app_state.df_after_step4 = None
    app_state.df_after_step5 = None
    if app_state.selected_target_variable and \
       (app_state.current_df is None or app_state.selected_target_variable not in app_state.current_df.columns):
        app_state.selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
    update_target_variable_combo()
    if len(ANALYSIS_STEPS) > 3 and ANALYSIS_STEPS[3] in app_state.module_ui_updaters:
        print(f"Triggering UI update for subsequent step: {ANALYSIS_STEPS[3]}")
        trigger_specific_module_update(ANALYSIS_STEPS[3])

def step4_missing_value_processing_complete(processed_df: pd.DataFrame):
    if processed_df is None:
        print("Step 4 (Missing Values) returned no DataFrame.")
        if app_state.df_after_step3 is not None:
            app_state.current_df = app_state.df_after_step3.copy()
        elif app_state.df_after_step1 is not None:
            app_state.current_df = app_state.df_after_step1.copy()
        else:
            app_state.current_df = None
        app_state.df_after_step4 = None
    else:
        print("Step 4 (Missing Values) processing complete. Updating app_state.")
        app_state.df_after_step4 = processed_df.copy()
        app_state.current_df = app_state.df_after_step4.copy()
    app_state.df_after_step5 = None
    if app_state.selected_target_variable and \
       (app_state.current_df is None or app_state.selected_target_variable not in app_state.current_df.columns):
        app_state.selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
    update_target_variable_combo()
    if len(ANALYSIS_STEPS) > 4 and ANALYSIS_STEPS[4] in app_state.module_ui_updaters:
        print(f"Triggering UI update for subsequent step: {ANALYSIS_STEPS[4]}")
        trigger_specific_module_update(ANALYSIS_STEPS[4])

def step5_outlier_treatment_complete(processed_df: pd.DataFrame):
    if processed_df is None:
        print("Step 5 (Outlier Treatment) returned no DataFrame.")
        if app_state.df_after_step4 is not None: app_state.current_df = app_state.df_after_step4.copy()
        elif app_state.df_after_step3 is not None: app_state.current_df = app_state.df_after_step3.copy()
        elif app_state.df_after_step1 is not None: app_state.current_df = app_state.df_after_step1.copy()
        else: app_state.current_df = None
        app_state.df_after_step5 = None
    else:
        print("Step 5 (Outlier Treatment) processing complete. Updating app_state.")
        app_state.df_after_step5 = processed_df.copy()
        app_state.current_df = app_state.df_after_step5.copy()
    if app_state.selected_target_variable and \
       (app_state.current_df is None or app_state.selected_target_variable not in app_state.current_df.columns):
        app_state.selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
    update_target_variable_combo()

def trigger_specific_module_update(module_name_key: str):
    df_to_use_for_module: Optional[pd.DataFrame] = None
    if module_name_key == ANALYSIS_STEPS[0]:
        if module_name_key in app_state.module_ui_updaters:
            updater = app_state.module_ui_updaters[module_name_key]
            if hasattr(step_01_data_loading, 'update_ui'):
                updater(app_state.df_after_step1, app_state.original_df, util_functions_for_modules, app_state.loaded_file_path)
                print(f"Module UI updated for: '{module_name_key}'")
        return
    elif module_name_key == ANALYSIS_STEPS[1]:
        df_to_use_for_module = app_state.current_df
        if SVA_STEP_KEY in app_state.module_ui_updaters:
            app_state.module_ui_updaters[SVA_STEP_KEY](df_to_use_for_module, main_app_callbacks)
        if MVA_STEP_KEY in app_state.module_ui_updaters:
            app_state.module_ui_updaters[MVA_STEP_KEY](df_to_use_for_module, main_app_callbacks)
        print(f"Module UI updated for: '{module_name_key}' and its sub-modules (SVA/MVA)")
        return
    elif module_name_key == ANALYSIS_STEPS[2]:
        df_to_use_for_module = app_state.df_after_step1
        if module_name_key in app_state.module_ui_updaters:
            updater = app_state.module_ui_updaters[module_name_key]
            if hasattr(step_03_preprocessing, 'update_ui'):
                 updater(df_to_use_for_module, main_app_callbacks)
                 print(f"Module UI updated for: '{module_name_key}'")
        return
    elif module_name_key == ANALYSIS_STEPS[3]:
        df_to_use_for_module = app_state.df_after_step3 if app_state.df_after_step3 is not None else app_state.df_after_step1
        if module_name_key in app_state.module_ui_updaters:
            updater = app_state.module_ui_updaters[module_name_key]
            if hasattr(step_04_missing_values, 'update_ui'):
                 updater(df_to_use_for_module, main_app_callbacks)
                 print(f"Module UI updated for: '{module_name_key}'")
        return
    elif module_name_key == ANALYSIS_STEPS[4]:
        if app_state.df_after_step4 is not None: df_to_use_for_module = app_state.df_after_step4
        elif app_state.df_after_step3 is not None: df_to_use_for_module = app_state.df_after_step3
        else: df_to_use_for_module = app_state.df_after_step1
        if module_name_key in app_state.module_ui_updaters:
            updater = app_state.module_ui_updaters[module_name_key]
            if hasattr(step_05_outlier_treatment, 'update_ui'):
                 updater(df_to_use_for_module, main_app_callbacks)
                 print(f"Module UI updated for: '{module_name_key}'")
        return
    if module_name_key in app_state.module_ui_updaters:
        df_to_use_for_module = app_state.current_df
        updater = app_state.module_ui_updaters[module_name_key]
        try:
            updater(df_to_use_for_module, main_app_callbacks)
            print(f"Module UI updated for: '{module_name_key}' (using generic current_df)")
        except TypeError:
             print(f"Warning: Could not update UI for '{module_name_key}' due to argument mismatch or missing df. Using current_df: {app_state.current_df is not None}")

def trigger_all_module_updates():
    print("Updating all module UIs...")
    for step_key_or_name in ANALYSIS_STEPS:
        trigger_specific_module_update(step_key_or_name)
    print("All module UIs update process finished.")

def trigger_all_module_updates_except_step1():
    print("Updating all module UIs except Step 1...")
    for step_name_iter in ANALYSIS_STEPS:
        if step_name_iter == ANALYSIS_STEPS[0]:
            continue
        trigger_specific_module_update(step_name_iter)
    print("Finished updating module UIs (except Step 1).")

def load_data_from_file(file_path: str) -> bool:
    success = False
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".parquet": app_state.original_df = pd.read_parquet(file_path)
        elif file_extension == ".csv":
            try: app_state.original_df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                try: app_state.original_df = pd.read_csv(file_path, encoding='cp949')
                except Exception as e_csv: raise Exception(f"CSV encoding error (UTF-8 and CP949 failed): {e_csv}")
            except Exception as e_csv_other: raise Exception(f"Error reading CSV: {e_csv_other}")
        else:
            _show_simple_modal_message("File Type Error", f"Unsupported file type: {file_extension}\nPlease select a .parquet or .csv file.")
            return False
        app_state.current_df = None; app_state.df_after_step1 = None; app_state.df_after_step3 = None
        app_state.df_after_step4 = None; app_state.df_after_step5 = None
        app_state.loaded_file_path = file_path
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG):
            dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, f"File: {os.path.basename(file_path)} (Shape: {app_state.original_df.shape if app_state.original_df is not None else 'N/A'})")
        success = True
    except Exception as e:
        app_state.current_df = None; app_state.original_df = None; app_state.loaded_file_path = None
        app_state.df_after_step1 = None; app_state.df_after_step3 = None; app_state.df_after_step4 = None; app_state.df_after_step5 = None
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG):
            dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, f"Error loading file: {os.path.basename(file_path)}")
        _show_simple_modal_message("File Load Error", f"Failed to load data from '{os.path.basename(file_path)}'.\nError: {e}")
        print(f"Error loading raw data: {e}"); traceback.print_exc()
        success = False
    if success:
        if hasattr(step_01_data_loading, 'process_newly_loaded_data') and app_state.original_df is not None:
            step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks)
        elif app_state.original_df is not None:
            main_app_callbacks['step1_processing_complete'](app_state.original_df.copy())
        if app_state.selected_target_variable and \
           (app_state.original_df is None or app_state.selected_target_variable not in app_state.original_df.columns):
            app_state.selected_target_variable = None
            if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
        update_target_variable_combo()
        return True
    else:
        reset_application_state(clear_df_completely=True)
        trigger_all_module_updates()
        return False

def target_variable_type_changed_callback(sender, app_data, user_data):
    new_type = app_data
    prev_valid_type = app_state.selected_target_variable_type
    df_for_type_check = app_state.current_df
    if df_for_type_check is None: df_for_type_check = app_state.df_after_step1
    if new_type == "Continuous" and app_state.selected_target_variable and df_for_type_check is not None:
        s1_col_types = main_app_callbacks.get('get_column_analysis_types', lambda: {})()
        analysis_type_s1 = s1_col_types.get(app_state.selected_target_variable)
        block_continuous_selection = False
        if analysis_type_s1:
            if analysis_type_s1.startswith("Categorical") or \
               analysis_type_s1.startswith("Text (") or \
               analysis_type_s1.startswith("Potentially Sensitive") or \
               analysis_type_s1 == "Datetime" or \
               analysis_type_s1 == "Timedelta" or \
               "Numeric (Binary" in analysis_type_s1:
                block_continuous_selection = True
        elif app_state.selected_target_variable in df_for_type_check.columns:
                dtype_in_current_df = df_for_type_check[app_state.selected_target_variable].dtype
                if pd.api.types.is_object_dtype(dtype_in_current_df) or \
                   pd.api.types.is_string_dtype(dtype_in_current_df) or \
                   pd.api.types.is_categorical_dtype(dtype_in_current_df) or \
                   pd.api.types.is_bool_dtype(dtype_in_current_df):
                    block_continuous_selection = True
        if block_continuous_selection:
            var_name = app_state.selected_target_variable
            s1_type_info = f"(Step 1 Type: {analysis_type_s1})" if analysis_type_s1 else \
                           (f"(Actual Dtype: {df_for_type_check[var_name].dtype})" if var_name in df_for_type_check else "")
            df_source_msg = "current data"
            err_msg = f"Variable '{var_name}' {s1_type_info} in {df_source_msg} may not be suitable for 'Continuous' type.\n" \
                      f"Please consider 'Categorical' or verify its type and processing in previous steps."
            _show_simple_modal_message("Type Selection Warning", err_msg, width=500, height=220)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, prev_valid_type)
            return
    app_state.selected_target_variable_type = new_type
    if app_state.active_settings: app_state.active_settings['selected_target_variable_type'] = new_type
    if app_state.active_step_name: trigger_specific_module_update(app_state.active_step_name)

def target_variable_selected_callback(sender, app_data, user_data):
    new_target_variable_name = app_data
    df_for_guessing_type = app_state.df_after_step1
    if df_for_guessing_type is None: df_for_guessing_type = app_state.original_df
    if not new_target_variable_name:
        app_state.selected_target_variable = None
        app_state.selected_target_variable_type = "Continuous"
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, "Continuous")
        if app_state.active_settings:
            app_state.active_settings['selected_target_variable'] = None
            app_state.active_settings['selected_target_variable_type'] = "Continuous"
    else:
        app_state.selected_target_variable = new_target_variable_name
        guessed_type_for_target = "Continuous"
        if df_for_guessing_type is not None and new_target_variable_name in df_for_guessing_type.columns:
            s1_col_types_for_guess = main_app_callbacks.get('get_column_analysis_types', lambda: {})()
            guessed_type_for_target = utils._guess_target_type(df_for_guessing_type, new_target_variable_name, s1_col_types_for_guess)
        app_state.selected_target_variable_type = guessed_type_for_target
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, guessed_type_for_target)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)
        if app_state.active_settings:
            app_state.active_settings['selected_target_variable'] = new_target_variable_name
            app_state.active_settings['selected_target_variable_type'] = guessed_type_for_target
    if app_state.active_step_name: trigger_specific_module_update(app_state.active_step_name)

def switch_step_view(sender, app_data, user_data_step_name: str):
    for step_name_iter, group_tag in app_state.step_group_tags.items():
        if dpg.does_item_exist(group_tag):
            is_active_step = (step_name_iter == user_data_step_name)
            dpg.configure_item(group_tag, show=is_active_step)
            if is_active_step:
                app_state.active_step_name = step_name_iter
                if step_name_iter == ANALYSIS_STEPS[0]: app_state.current_df = app_state.df_after_step1 if app_state.df_after_step1 is not None else app_state.original_df
                elif step_name_iter == ANALYSIS_STEPS[1]:
                    if app_state.df_after_step5 is not None: app_state.current_df = app_state.df_after_step5
                    elif app_state.df_after_step4 is not None: app_state.current_df = app_state.df_after_step4
                    elif app_state.df_after_step3 is not None: app_state.current_df = app_state.df_after_step3
                    elif app_state.df_after_step1 is not None: app_state.current_df = app_state.df_after_step1
                    else: app_state.current_df = app_state.original_df
                elif step_name_iter == ANALYSIS_STEPS[2]: app_state.current_df = app_state.df_after_step1
                elif step_name_iter == ANALYSIS_STEPS[3]: app_state.current_df = app_state.df_after_step3 if app_state.df_after_step3 is not None else app_state.df_after_step1
                elif step_name_iter == ANALYSIS_STEPS[4]:
                    if app_state.df_after_step4 is not None: app_state.current_df = app_state.df_after_step4
                    elif app_state.df_after_step3 is not None: app_state.current_df = app_state.df_after_step3
                    else: app_state.current_df = app_state.df_after_step1
                else: app_state.current_df = app_state.original_df
                print(f"Switched to {step_name_iter}. AppState.current_df (shape: {app_state.current_df.shape if app_state.current_df is not None else 'None'}).")
                trigger_specific_module_update(step_name_iter)
                update_target_variable_combo()

def file_load_callback(sender, app_data):
    new_file_path = app_data.get('file_path_name')
    if not new_file_path: return
    if app_state.loaded_file_path and app_state.active_settings :
        old_settings_path = get_settings_filepath(app_state.loaded_file_path)
        if old_settings_path:
            current_live_settings = gather_current_settings()
            save_json_settings(old_settings_path, current_live_settings)
            print(f"Saved settings for old file: {app_state.loaded_file_path}")
    reset_application_state(clear_df_completely=False)
    if not load_data_from_file(new_file_path): return
    new_settings_path = get_settings_filepath(app_state.loaded_file_path)
    loaded_specific_settings = load_json_settings(new_settings_path)
    if loaded_specific_settings:
        print(f"Applying settings for new file: {app_state.loaded_file_path}")
        apply_settings(loaded_specific_settings)
    else:
        app_state.active_settings = {}
        print(f"No specific settings for: {app_state.loaded_file_path}. Defaults.")
    if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0:
        first_step_name = ANALYSIS_STEPS[0]
        if first_step_name in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step_name]):
            if app_state.active_step_name != first_step_name or app_state.active_step_name is None:
                 switch_step_view(None, None, first_step_name)
        else:
             app_state.active_step_name = first_step_name
             print(f"Warning: First step group '{first_step_name}' not ready.")

def reset_application_state(clear_df_completely=True):
    if clear_df_completely:
        if hasattr(app_state, 'image_data_cache') and app_state.image_data_cache:
            app_state.image_data_cache.clear(); print("Image data cache cleared.")
        else: app_state.image_data_cache = {}; print("Image data cache created/cleared.")
        app_state.original_df = None; app_state.loaded_file_path = None
        app_state.active_settings = {}
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG): dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, "No data loaded.")
    app_state.current_df = None; app_state.df_after_step1 = None; app_state.df_after_step3 = None
    app_state.df_after_step4 = None; app_state.df_after_step5 = None
    app_state.selected_target_variable = None; app_state.selected_target_variable_type = "Continuous"

    # Ollama 상태 초기화 추가
    app_state.is_ollama_analysis_in_progress = False
    app_state.ollama_confirm_user_data = None
    if dpg.is_dearpygui_running() and dpg.does_item_exist(OLLAMA_RESULT_WINDOW_TAG): # 결과 창이 존재하면 숨김
        dpg.configure_item(OLLAMA_RESULT_WINDOW_TAG, show=False)


    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
        dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False); dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, "Continuous")
    if hasattr(step_01_data_loading, 'reset_step1_state'): step_01_data_loading.reset_step1_state()
    if hasattr(step_02a_sva, 'reset_sva_ui_defaults'): step_02a_sva.reset_sva_ui_defaults()
    if hasattr(step_02b_mva, 'reset_mva_ui_defaults'): step_02b_mva.reset_mva_ui_defaults()
    if hasattr(step_03_preprocessing, 'reset_preprocessing_state'): step_03_preprocessing.reset_preprocessing_state()
    if hasattr(step_04_missing_values, 'reset_missing_values_state'): step_04_missing_values.reset_missing_values_state()
    if hasattr(step_05_outlier_treatment, 'reset_outlier_treatment_state'): step_05_outlier_treatment.reset_outlier_treatment_state()
    if clear_df_completely:
        app_state.active_step_name = None
        if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0:
            first_step_name = ANALYSIS_STEPS[0]
            if first_step_name in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step_name]):
                print(f"Full reset: Switching to {first_step_name}")
                switch_step_view(None, None, first_step_name)
            else: app_state.active_step_name = first_step_name; print(f"Warning: First step '{first_step_name}' not ready.")
        trigger_all_module_updates()
    elif app_state.original_df is not None:
        if hasattr(step_01_data_loading, 'process_newly_loaded_data'):
            step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks)
        else: main_app_callbacks['step1_processing_complete'](app_state.original_df.copy())
        if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0: switch_step_view(None, None, ANALYSIS_STEPS[0])
        trigger_all_module_updates()
    if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0:
        current_active = app_state.active_step_name; first_step = ANALYSIS_STEPS[0]
        if not current_active or current_active not in app_state.step_group_tags or \
           not dpg.does_item_exist(app_state.step_group_tags.get(current_active, "")):
            if first_step in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step]):
                if app_state.active_step_name != first_step: switch_step_view(None, None, first_step)
            else: app_state.active_step_name = first_step

def export_to_parquet_callback():
    df_to_export = None; export_source_step = ""
    if app_state.df_after_step5 is not None: df_to_export = app_state.df_after_step5; export_source_step = "Step 5 (Outlier Treatment)"
    elif app_state.df_after_step4 is not None: df_to_export = app_state.df_after_step4; export_source_step = "Step 4 (Missing Values)"
    elif app_state.df_after_step3 is not None: df_to_export = app_state.df_after_step3; export_source_step = "Step 3 (Node Editor)"
    elif app_state.df_after_step1 is not None: df_to_export = app_state.df_after_step1; export_source_step = "Step 1 (Load/Overview)"
    if df_to_export is None: _show_simple_modal_message("Export Info", "No processed data to export."); return
    if not app_state.loaded_file_path: _show_simple_modal_message("Export Error", "Original file path not found."); return
    try:
        original_dir = os.path.dirname(app_state.loaded_file_path); original_basename = os.path.basename(app_state.loaded_file_path)
        original_filename_no_ext, _ = os.path.splitext(original_basename)
        export_suffix = f"_processed_after_{export_source_step.replace(' ', '_').replace('(', '').replace(')', '').lower()}"
        export_filename = f"{original_filename_no_ext}{export_suffix}.parquet"
        export_path = os.path.join(original_dir, export_filename)
        df_to_export.to_parquet(export_path, index=False)
        _show_simple_modal_message("Export Successful", f"Data from {export_source_step} exported to:\n{export_path}")
        print(f"Data exported to {export_path}")
    except Exception as e: _show_simple_modal_message("Export Error", f"Failed to export data.\nError: {e}"); print(f"Error exporting: {e}"); traceback.print_exc()

def get_settings_filepath(original_data_filepath: str) -> Optional[str]:
    if not original_data_filepath: return None
    try:
        safe_original_path = "".join(c if c.isalnum() or c in [' ', '.', '_', '-'] else '_' for c in original_data_filepath)
        filename_base = hashlib.md5(safe_original_path.encode('utf-8')).hexdigest()
    except Exception: filename_base = "default_settings"
    filename = filename_base + ".json"; return os.path.join(SETTINGS_DIR_NAME, filename)

def load_json_settings(settings_filepath: str) -> Optional[dict]:
    if settings_filepath and os.path.exists(settings_filepath):
        try:
            with open(settings_filepath, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception as e: print(f"Error loading '{settings_filepath}': {e}"); traceback.print_exc()
    return None

def save_json_settings(settings_filepath: str, settings_dict: dict):
    if not settings_filepath or not settings_dict: return
    try:
        if not os.path.exists(SETTINGS_DIR_NAME): os.makedirs(SETTINGS_DIR_NAME)
        with open(settings_filepath, 'w', encoding='utf-8') as f: json.dump(settings_dict, f, indent=4, ensure_ascii=False)
        print(f"Settings saved to {settings_filepath}")
    except Exception as e: print(f"Error saving to '{settings_filepath}': {e}"); traceback.print_exc()

def gather_current_settings() -> dict:
    settings = {'selected_target_variable': app_state.selected_target_variable, 'selected_target_variable_type': app_state.selected_target_variable_type,
                'active_step_name': app_state.active_step_name, 'step_01_settings': {}, 'step_02a_sva_settings': {}, 'step_02b_mva_settings': {},
                'step_03_preprocessing_settings': {}, 'step_04_missing_values_settings': {}, 'step_05_outlier_treatment_settings': {}}
    if hasattr(step_01_data_loading, 'get_step1_settings_for_saving'): settings['step_01_settings'] = step_01_data_loading.get_step1_settings_for_saving()
    if hasattr(step_02a_sva, 'get_sva_settings_for_saving'): settings['step_02a_sva_settings'] = step_02a_sva.get_sva_settings_for_saving()
    if hasattr(step_02b_mva, 'get_mva_settings_for_saving'): settings['step_02b_mva_settings'] = step_02b_mva.get_mva_settings_for_saving()
    if STEP_03_SAVE_LOAD_ENABLED and hasattr(step_03_preprocessing, 'get_preprocessing_settings_for_saving'):
        settings['step_03_preprocessing_settings'] = step_03_preprocessing.get_preprocessing_settings_for_saving()
    else: settings['step_03_preprocessing_settings'] = {}
    if hasattr(step_04_missing_values, 'get_missing_values_settings_for_saving'):
        settings['step_04_missing_values_settings'] = step_04_missing_values.get_missing_values_settings_for_saving()
    if hasattr(step_05_outlier_treatment, 'get_outlier_treatment_settings_for_saving'):
        settings['step_05_outlier_treatment_settings'] = step_05_outlier_treatment.get_outlier_treatment_settings_for_saving()
    return settings

def apply_settings(settings_dict: dict):
    if app_state.original_df is None: _show_simple_modal_message("Error", "Cannot apply: No original data."); return
    app_state.active_settings = settings_dict
    app_state.selected_target_variable = settings_dict.get('selected_target_variable')
    app_state.selected_target_variable_type = settings_dict.get('selected_target_variable_type', "Continuous")
    s01_settings = settings_dict.get('step_01_settings', {})
    if hasattr(step_01_data_loading, 'apply_step1_settings_and_process'):
        step_01_data_loading.apply_step1_settings_and_process(app_state.original_df, s01_settings, main_app_callbacks)
    update_target_variable_combo()
    if app_state.selected_target_variable and app_state.df_after_step1 is not None and app_state.selected_target_variable in app_state.df_after_step1.columns:
        s1_types_from_settings = s01_settings.get('type_selections', {})
        guessed_type = utils._guess_target_type(app_state.df_after_step1, app_state.selected_target_variable, s1_types_from_settings)
        final_target_type = settings_dict.get('selected_target_variable_type', guessed_type)
        app_state.selected_target_variable_type = final_target_type
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, app_state.selected_target_variable_type)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)
    else:
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False); dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, "Continuous")
    s02a_settings = settings_dict.get('step_02a_sva_settings', {}); s02b_settings = settings_dict.get('step_02b_mva_settings', {})
    if hasattr(step_02a_sva, 'apply_sva_settings_from_loaded') and app_state.df_after_step1 is not None: step_02a_sva.apply_sva_settings_from_loaded(s02a_settings, app_state.df_after_step1, main_app_callbacks)
    if hasattr(step_02b_mva, 'apply_mva_settings_from_loaded') and app_state.df_after_step1 is not None: step_02b_mva.apply_mva_settings_from_loaded(s02b_settings, app_state.df_after_step1, main_app_callbacks)
    app_state.df_after_step3 = None
    if STEP_03_SAVE_LOAD_ENABLED:
        s03_settings = settings_dict.get('step_03_preprocessing_settings', {})
        if s03_settings and hasattr(step_03_preprocessing, 'apply_preprocessing_settings_and_process') and app_state.df_after_step1 is not None:
            step_03_preprocessing.apply_preprocessing_settings_and_process(app_state.df_after_step1, s03_settings, main_app_callbacks)
    app_state.df_after_step4 = None
    s04_settings = settings_dict.get('step_04_missing_values_settings', {})
    df_input_for_step4 = app_state.df_after_step3 if app_state.df_after_step3 is not None else app_state.df_after_step1
    if s04_settings and hasattr(step_04_missing_values, 'apply_missing_values_settings_and_process') and df_input_for_step4 is not None:
        step_04_missing_values.apply_missing_values_settings_and_process(df_input_for_step4, s04_settings, main_app_callbacks)
    app_state.df_after_step5 = None
    s05_settings = settings_dict.get('step_05_outlier_treatment_settings', {})
    df_input_for_step5 = app_state.df_after_step4 if app_state.df_after_step4 is not None else \
                         (app_state.df_after_step3 if app_state.df_after_step3 is not None else app_state.df_after_step1)
    if s05_settings and hasattr(step_05_outlier_treatment, 'apply_outlier_treatment_settings_and_process') and df_input_for_step5 is not None:
        step_05_outlier_treatment.apply_outlier_treatment_settings_and_process(df_input_for_step5, s05_settings, main_app_callbacks)
    trigger_all_module_updates()
    restored_active_step = settings_dict.get('active_step_name', ANALYSIS_STEPS[0] if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0 else None)
    if restored_active_step and restored_active_step in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[restored_active_step]):
        switch_step_view(None, None, restored_active_step)
    elif ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0 : switch_step_view(None, None, ANALYSIS_STEPS[0])

def initial_load_on_startup():
    session_info = load_json_settings(SESSION_INFO_FILE)
    last_file_path = session_info.get('last_opened_original_file') if session_info else None
    if last_file_path and os.path.exists(last_file_path):
        try:
            if not load_data_from_file(last_file_path): raise Exception("Failed to load data from last session.")
            settings_for_file_path = get_settings_filepath(last_file_path)
            specific_file_settings = load_json_settings(settings_for_file_path)
            if specific_file_settings: apply_settings(specific_file_settings)
            else:
                app_state.active_settings = {}
                if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0:
                    first_step_name = ANALYSIS_STEPS[0]
                    if app_state.active_step_name != first_step_name:
                        if first_step_name in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step_name]):
                            switch_step_view(None, None, first_step_name)
                        else: app_state.active_step_name = first_step_name
            return
        except Exception as e: _show_simple_modal_message("Session Restore Error", f"Could not restore: {os.path.basename(last_file_path)}\nError: {e}")
    reset_application_state(clear_df_completely=True)

def save_state_on_exit():
    if app_state.loaded_file_path and os.path.exists(app_state.loaded_file_path):
        current_settings = gather_current_settings()
        settings_filepath = get_settings_filepath(app_state.loaded_file_path)
        if settings_filepath:
            save_json_settings(settings_filepath, current_settings)
            save_json_settings(SESSION_INFO_FILE, {'last_opened_original_file': app_state.loaded_file_path})
    elif os.path.exists(SESSION_INFO_FILE): save_json_settings(SESSION_INFO_FILE, {})

util_functions_for_modules = {
    'create_table_with_data': utils.create_table_with_data,
    'calculate_column_widths': utils.calculate_column_widths,
    'format_text_for_display': utils.format_text_for_display,
    'get_safe_text_size': utils.get_safe_text_size,
    '_show_simple_modal_message': _show_simple_modal_message,
    'show_dpg_alert_modal': utils.show_dpg_alert_modal,
    'get_numeric_cols': utils._get_numeric_cols,
    'get_categorical_cols': utils._get_categorical_cols,
    'calculate_cramers_v': utils.calculate_cramers_v,
    'create_analyzable_image_widget': utils.create_analyzable_image_widget,
    'add_job_to_callback_queue': lambda job: app_state.callback_queue.append(job)
}

main_app_callbacks = {
    'get_current_df': lambda: app_state.current_df,
    'get_original_df': lambda: app_state.original_df,
    'get_df_after_step1': lambda: app_state.df_after_step1,
    'get_df_after_step3': lambda: app_state.df_after_step3,
    'get_df_after_step4': lambda: app_state.df_after_step4,
    'get_df_after_step5': lambda: app_state.df_after_step5,
    'get_loaded_file_path': lambda: app_state.loaded_file_path,
    'get_util_funcs': lambda: util_functions_for_modules,
    'show_file_dialog': lambda: dpg.show_item("file_dialog_id"),
    'register_step_group_tag': lambda name, tag: app_state.step_group_tags.update({name: tag}),
    'register_module_updater': lambda name, func: app_state.module_ui_updaters.update({name: func}),
    'trigger_module_update': trigger_specific_module_update,
    'reset_current_df_to_original': lambda: (reset_application_state(clear_df_completely=False)),
    'trigger_all_module_updates': trigger_all_module_updates,
    'get_selected_target_variable': lambda: app_state.selected_target_variable,
    'get_selected_target_variable_type': lambda: app_state.selected_target_variable_type,
    'update_target_variable_combo_items': update_target_variable_combo,
    'get_column_analysis_types': lambda: step_01_data_loading._type_selections.copy() if hasattr(step_01_data_loading, '_type_selections') else {},
    'step1_processing_complete': step1_processing_complete,
    'step3_processing_complete': step3_processing_complete,
    'step4_missing_value_processing_complete': step4_missing_value_processing_complete,
    'step5_outlier_treatment_complete': step5_outlier_treatment_complete,

    # Ollama 관련 콜백 변경 및 추가
    'ask_for_ollama_confirmation': _show_ollama_confirmation_popup, # utils.py에서 사용
    'initiate_ollama_analysis_with_window': ollama_integration.request_image_analysis, # 확인 후 호출될 함수
    'get_cached_image_data_func': get_cached_image_bytes_for_ollama,
    'cache_image_data_func': cache_image_data_for_ollama,
    'get_main_callback_queue': lambda: app_state.callback_queue,
    'is_ollama_analysis_in_progress': lambda: app_state.is_ollama_analysis_in_progress,
    'set_ollama_analysis_status': set_ollama_analysis_status,
    'show_ollama_result_window_and_set_initial': show_and_update_ollama_result_window,
    'update_ollama_result_window_job': update_ollama_result_window_content_from_job
}



# --- Dear PyGUI Setup ---
dpg.create_context()
dpg.configure_app(manual_callback_management=True)

with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback, id="file_dialog_id", width=700, height=400, modal=True):
    dpg.add_file_extension(".parquet")
    dpg.add_file_extension(".csv")
    dpg.add_file_extension(".*")

setup_korean_font()
create_ollama_result_window() # Ollama 결과창 미리 생성 (숨김 상태)

with dpg.window(label="Data Analysis Platform", tag="main_window"):
    with dpg.group(horizontal=True):
        dpg.add_button(label="Open File", callback=lambda: dpg.show_item("file_dialog_id"), width=130, height=30)
        dpg.add_button(label="Reset All (to Step 1 Types)", user_data=main_app_callbacks, width=210, height=30,
                       callback=lambda s, a, u: u['reset_current_df_to_original']())
        dpg.add_button(label="Export to Parquet", callback=export_to_parquet_callback, width=160, height=30)
        dpg.add_text("No data loaded.", tag=MAIN_FILE_PATH_DISPLAY_TAG, wrap=-1)
    dpg.add_separator()
    with dpg.group(horizontal=True):
        with dpg.child_window(width=280, tag="navigation_panel", border=True):
            dpg.add_text("Target Variable (y):")
            dpg.add_combo(items=[""], tag=TARGET_VARIABLE_COMBO_TAG, width=-1, callback=target_variable_selected_callback)
            dpg.add_text("Target Variable Type:", tag=TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            dpg.add_radio_button(items=["Categorical", "Continuous"], tag=TARGET_VARIABLE_TYPE_RADIO_TAG, horizontal=True,
                                 default_value=app_state.selected_target_variable_type, callback=target_variable_type_changed_callback, show=False)
            dpg.add_separator(); dpg.add_spacer(height=5)
            dpg.add_text("Analysis Steps", color=[255,255,0]); dpg.add_separator()
            for step_name_nav in ANALYSIS_STEPS:
                dpg.add_button(label=step_name_nav, callback=switch_step_view, user_data=step_name_nav, width=-1, height=30)
        with dpg.child_window(tag="content_area", border=True):
            for step_name_create in ANALYSIS_STEPS:
                if step_name_create == ANALYSIS_STEPS[0]:
                    if hasattr(step_01_data_loading, 'create_ui'):
                        step_01_data_loading.create_ui(step_name_create, "content_area", main_app_callbacks)
                elif step_name_create == ANALYSIS_STEPS[1]:
                    eda_step_group_tag = "eda_step_main_group"
                    main_app_callbacks['register_step_group_tag'](step_name_create, eda_step_group_tag)
                    with dpg.group(tag=eda_step_group_tag, parent="content_area", show=False):
                        dpg.add_text(f"--- {step_name_create} ---"); dpg.add_separator()
                        with dpg.tab_bar(tag="eda_tab_bar"):
                            with dpg.tab(label="Single Variable Analysis", tag="eda_sva_tab"):
                                if hasattr(step_02a_sva, 'create_ui'):
                                    step_02a_sva.create_ui(SVA_STEP_KEY, "eda_sva_tab", main_app_callbacks)
                            with dpg.tab(label="Multivariate Analysis", tag="eda_mva_tab"):
                                if hasattr(step_02b_mva, 'create_ui'):
                                    step_02b_mva.create_ui(MVA_STEP_KEY, "eda_mva_tab", main_app_callbacks)
                elif step_name_create == ANALYSIS_STEPS[2]:
                    if hasattr(step_03_preprocessing, 'create_ui'):
                        step_03_preprocessing.create_ui(step_name_create, "content_area", main_app_callbacks)
                elif step_name_create == ANALYSIS_STEPS[3]:
                    if hasattr(step_04_missing_values, 'create_ui'):
                        step_04_missing_values.create_ui(step_name_create, "content_area", main_app_callbacks)
                elif step_name_create == ANALYSIS_STEPS[4]:
                    if hasattr(step_05_outlier_treatment, 'create_ui'):
                        step_05_outlier_treatment.create_ui(step_name_create, "content_area", main_app_callbacks)

            if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0 and not app_state.active_step_name:
                first_step = ANALYSIS_STEPS[0]
                if first_step in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step]):
                    switch_step_view(None, None, first_step)
                else: app_state.active_step_name = first_step

dpg.create_viewport(title='Data Analysis Platform', width=1600, height=1000)
dpg.set_exit_callback(save_state_on_exit)
dpg.setup_dearpygui()

# --- Ollama Integration 초기화 ---
ollama_integration.initialize_ollama_integration(
    get_image_func=main_app_callbacks['get_cached_image_data_func'],
    add_job_func=main_app_callbacks['get_main_callback_queue'], # 메인 큐 전달
    # 새로운 콜백 함수들 전달
    is_analysis_in_progress_func=main_app_callbacks['is_ollama_analysis_in_progress'],
    set_analysis_status_func=main_app_callbacks['set_ollama_analysis_status'],
    show_result_window_func=main_app_callbacks['show_ollama_result_window_and_set_initial'],
    update_result_window_job_func_name='update_ollama_result_window_job' # 문자열로 전달 (ollama_integration에서 main_app_callbacks 통해 가져옴)
)

initial_load_on_startup()

dpg.show_viewport()
if dpg.does_item_exist("main_window"):
    dpg.set_primary_window("main_window", True)

while dpg.is_dearpygui_running():
    jobs_from_dpg = dpg.get_callback_queue()
    if jobs_from_dpg is None: jobs_from_dpg = []
    
    all_jobs_to_run = []
    if app_state.callback_queue:
        all_jobs_to_run.extend(app_state.callback_queue)
        app_state.callback_queue.clear()
    
    if jobs_from_dpg:
        all_jobs_to_run.extend(jobs_from_dpg)

    if all_jobs_to_run:
        dpg.run_callbacks(all_jobs_to_run)
        all_jobs_to_run.clear()

    dpg.render_dearpygui_frame()

dpg.destroy_context()