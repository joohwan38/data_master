# main_app.py
import dearpygui.dearpygui as dpg
import pandas as pd
import platform
import os
import utils
from typing import Optional, Dict
import step_01_data_loading
import step_02a_sva
import step_02b_mva
import step_03_preprocessing 
import step_04_missing_values
import step_05_outlier_treatment
import step_06_standardization
import step_07_feature_engineering
import step_08_derivation
import step_09_data_viewer
import step_10_advanced_analysis
import step_11_ml_modeling
import traceback
import hashlib
import json
import datetime
import subprocess
import re


STEP_03_SAVE_LOAD_ENABLED = False 

class AppState:
    def __init__(self):
        self.current_df = None
        self.original_df = None
        self.original_base_name = None
        self.df_after_step1 = None
        self.df_after_step3 = None 
        self.df_after_step4 = None 
        self.df_after_step5 = None
        self.df_after_step6 = None
        self.df_after_step7 = None
        self.derived_dfs: Dict[str, pd.DataFrame] = {} # << ADDED
        self.loaded_file_path = None
        self.selected_target_variable = None
        self.selected_target_variable_type = "Continuous"
        self.active_step_name = None
        self.active_settings = {} 
        self.step_group_tags = {}
        self.module_ui_updaters = {}
        self.ai_analysis_log = ""

app_state = AppState()

SETTINGS_DIR_NAME = ".app_file_settings"
SESSION_INFO_FILE = os.path.join(SETTINGS_DIR_NAME, "session_info.json")

SVA_STEP_KEY = "2a. Single Variable Analysis"
MVA_STEP_KEY = "2b. Multivariate Analysis"
TARGET_VARIABLE_TYPE_RADIO_TAG = "main_target_variable_type_radio"
TARGET_VARIABLE_TYPE_LABEL_TAG = "main_target_variable_type_label"
TARGET_VARIABLE_COMBO_TAG = "main_target_variable_combo"
MAIN_FILE_PATH_DISPLAY_TAG = "main_file_path_display_text"


ANALYSIS_STEPS = [
    "1. Data Loading & Overview",
    "2. Exploratory Data Analysis (EDA)",
    "3. Preprocessing (Node Editor)",
    "4. Missing Value Treatment",         
    "5. Outlier Treatment",
    "6. Standardization", 
    "7. Feature Engineering",
    "8. Derive DataFrames",
    "9. DataFrame Viewer",
    "10. Advanced Analysis",
    "11. ML Modeling & AI",
    ]

_MODAL_ID_SIMPLE_MESSAGE = "main_simple_modal_message_id"
try:
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except FileNotFoundError:
    print("Warning: 'ollama' command not found. AI analysis will not be available.")


def _get_base_filename(file_path: str) -> str:
    """Extracts the base name from a file path, stripping _YYYYMMDD_vN suffixes."""
    filename = os.path.basename(file_path)
    filename_no_ext, _ = os.path.splitext(filename)
    # _YYYYMMDD 또는 _YYYYMMDD_v123 같은 패턴을 제거하는 정규식
    base_name = re.sub(r'_(\d{8})(_v\d+)?$', '', filename_no_ext)
    return base_name

def _show_simple_modal_message(title: str, message: str, width: int = 450, height: int = 200):
    """간단한 메시지를 표시하는 모달 창을 띄웁니다."""
    if dpg.does_item_exist(_MODAL_ID_SIMPLE_MESSAGE): dpg.delete_item(_MODAL_ID_SIMPLE_MESSAGE)
    
    vp_w = dpg.get_viewport_width() if dpg.is_dearpygui_running() else 1000
    vp_h = dpg.get_viewport_height() if dpg.is_dearpygui_running() else 700
    modal_x, modal_y = (vp_w - width) // 2, (vp_h - height) // 2
    
    with dpg.window(label=title, modal=True, show=True, id=_MODAL_ID_SIMPLE_MESSAGE, no_close=True,
                    width=width, height=height, pos=[modal_x, modal_y], no_saved_settings=True, autosize=False): 
        dpg.add_text(message, wrap=width - 20) 
        dpg.add_spacer(height=15)
        with dpg.group(horizontal=True):
            item_spacing_x = 8.0 
            button_width = 100
            spacer_w = (width - button_width - (item_spacing_x * 2)) / 2 
            if spacer_w < 0: spacer_w = 0

            dpg.add_spacer(width=int(spacer_w))
            dpg.add_button(label="OK", width=button_width, callback=lambda: dpg.configure_item(_MODAL_ID_SIMPLE_MESSAGE, show=False))

def add_ai_log_message(message: str, chart_context: str = "", mode: str = "new_log_entry"):
    # 디버그 print문 모두 제거
    if not dpg.is_dearpygui_running():
        return

    log_panel_tag = "ai_analysis_log_panel_text"
    log_container_window_tag = "ai_analysis_log_panel"

    if not dpg.does_item_exist(log_panel_tag):
        return
    if not dpg.does_item_exist(log_container_window_tag) or \
       dpg.get_item_info(log_container_window_tag)['type'] != "mvAppItemType::mvChildWindow":
        return

    current_full_log = dpg.get_value(log_panel_tag)
    default_placeholder_msg = "AI 분석 결과가 여기에 표시됩니다.\n"
    
    is_log_empty_or_default = (current_full_log == default_placeholder_msg or not current_full_log.strip())
    if is_log_empty_or_default:
        current_full_log = "" 
        
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_log = ""
    
    standard_separator = "\n\n" + "-"*30 + "\n"

    if mode == "stream_start_entry" or mode == "new_log_entry":
        context_str = f"[{chart_context}] " if chart_context else ""
        entry_header = f"[{timestamp}] {context_str}\n"
        new_entry_content = entry_header + message
        
        if current_full_log:
            updated_log = current_full_log + standard_separator + new_entry_content
        else:
            updated_log = new_entry_content
        
        dpg.set_value(log_panel_tag, updated_log)
        dpg.set_y_scroll(log_container_window_tag, dpg.get_y_scroll_max(log_container_window_tag))

    elif mode == "stream_chunk_append":
        if is_log_empty_or_default: 
            add_ai_log_message(message, chart_context, mode="stream_start_entry")
            return
        
        updated_log = current_full_log + message 
        
        dpg.set_value(log_panel_tag, updated_log)
        dpg.set_y_scroll(log_container_window_tag, dpg.get_y_scroll_max(log_container_window_tag))
    
    # Log Truncation (remove from the TOP - oldest entries)
    max_log_length = 30000
    if len(updated_log) > max_log_length:
        chars_to_remove = len(updated_log) - max_log_length
        
        cutoff_search_start_index = max(0, chars_to_remove - len("...(오래된 로그 일부 잘림)...\n"))
        cutoff_point = updated_log.find(standard_separator, cutoff_search_start_index)
        
        if cutoff_point != -1 and cutoff_point > 0 :
            updated_log = "...(오래된 로그 일부 잘림)..." + updated_log[cutoff_point + len(standard_separator):]
        else: 
            updated_log = "...(오래된 로그 일부 잘림)...\n" + updated_log[-max_log_length:]
            
        dpg.set_value(log_panel_tag, updated_log)

def setup_korean_font():
    """시스템에 맞는 한글 폰트를 설정합니다. UI는 영어로 유지되므로, 주석 등 내부용입니다."""
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
                dpg.bind_font("korean_font_for_app")
            print(f"Korean font for comments/internal use bound: {font_path}")
        except Exception as e: print(f"Font error: {e}"); traceback.print_exc()
    else: print("Korean font not found for system text. Using default. UI elements are in English.")


def update_target_variable_combo():
    """타겟 변수 선택 콤보박스의 아이템을 현재 DataFrame 기준으로 업데이트합니다."""
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        df_for_combo = app_state.current_df # Use current_df which reflects the latest step
        
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


def trigger_all_module_updates():
    """등록된 모든 모듈의 UI를 업데이트합니다."""
    print("Updating all module UIs...")
    for step_key_or_name in ANALYSIS_STEPS:
        trigger_specific_module_update(step_key_or_name)
    print("All module UIs update process finished.")

def _update_pipeline_state(step_name: str, result_df: Optional[pd.DataFrame]):
    """
    파이프라인의 상태를 중앙에서 업데이트하는 헬퍼 함수.
    current_df와 df_after_step*을 업데이트하고 모든 UI를 새로고침합니다.
    """
    if result_df is None:
        print(f"Processing for {step_name} returned None. State not updated.")
        return

    print(f"Updating pipeline state after: {step_name}")

    # 각 단계에 맞는 app_state 변수에 결과 저장
    if step_name == ANALYSIS_STEPS[0]: # "1. Data Loading & Overview"
        app_state.df_after_step1 = result_df.copy(deep=True)
        app_state.current_df = app_state.df_after_step1.copy(deep=True)
    elif step_name == ANALYSIS_STEPS[3]: # "4. Missing Value Treatment"
        app_state.df_after_step4 = result_df.copy(deep=True)
        app_state.current_df = app_state.df_after_step4.copy(deep=True)
    elif step_name == ANALYSIS_STEPS[4]: # "5. Outlier Treatment"
        app_state.df_after_step5 = result_df.copy(deep=True)
        app_state.current_df = app_state.df_after_step5.copy(deep=True)
    elif step_name == ANALYSIS_STEPS[5]: # "6. Standardization"
        app_state.df_after_step6 = result_df.copy(deep=True)
        app_state.current_df = app_state.df_after_step6.copy(deep=True)
    elif step_name == ANALYSIS_STEPS[6]: # "7. Feature Engineering"
        app_state.df_after_step7 = result_df.copy(deep=True)
        app_state.current_df = app_state.df_after_step7.copy(deep=True)

    # 이 단계 이후의 모든 후속 단계 상태는 초기화
    step_index = ANALYSIS_STEPS.index(step_name)
    if step_index < 3: app_state.df_after_step4 = None
    if step_index < 4: app_state.df_after_step5 = None
    if step_index < 5: app_state.df_after_step6 = None
    if step_index < 6: app_state.df_after_step7 = None

    # 타겟 변수 콤보박스와 모든 모듈 UI 업데이트
    update_target_variable_combo()
    trigger_all_module_updates()


def step1_processing_complete(processed_df: pd.DataFrame):
    _update_pipeline_state(ANALYSIS_STEPS[0], processed_df)

def step4_missing_value_processing_complete(processed_df: pd.DataFrame):
    _update_pipeline_state(ANALYSIS_STEPS[3], processed_df)

def step5_outlier_treatment_complete(processed_df: pd.DataFrame):
    _update_pipeline_state(ANALYSIS_STEPS[4], processed_df)

def step6_standardization_complete(processed_df: pd.DataFrame, method_name: str):
    """
    Step 6 완료 시 호출. 이제 사용된 메소드 이름을 받아 파생 DF의 이름을 생성합니다.
    """
    if processed_df is None:
        return

    # 1. 기본 이름 생성 (예: "Standardized_MinMaxScaler")
    base_name = f"Standardized_{method_name}"
    derived_df_name = base_name
    
    # 2. 이름 중복 방지를 위한 카운터 추가
    counter = 1
    while derived_df_name in app_state.derived_dfs:
        derived_df_name = f"{base_name}_{counter}"
        counter += 1

    # 3. 파생 DF 목록에 추가 및 UI 업데이트 (기존 로직과 유사)
    app_state.derived_dfs[derived_df_name] = processed_df
    print(f"Derived DataFrame '{derived_df_name}' created from Standardization.")

    if ANALYSIS_STEPS[7] in app_state.module_ui_updaters:
        trigger_specific_module_update(ANALYSIS_STEPS[7])
    if len(ANALYSIS_STEPS) > 8 and ANALYSIS_STEPS[8] in app_state.module_ui_updaters:
        trigger_specific_module_update(ANALYSIS_STEPS[8])

    # 4. 사용자에게 생성된 이름으로 알림
    _show_simple_modal_message("Standardization Complete",
                               f"The standardized data has been saved as a new derived DataFrame named:\n'{derived_df_name}'\n\nYou can view it in Step 8 or 9.")

def step7_feature_engineering_complete(processed_df: pd.DataFrame):
    _update_pipeline_state(ANALYSIS_STEPS[6], processed_df)

# << ADDED >>: Step 8에서 파생 DF 생성이 완료되었을 때 호출되는 콜백
def step8_derivation_complete(name: str, df: pd.DataFrame):
    """Step 8 (Derive DataFrames) 처리 완료 시 호출되는 콜백입니다."""
    app_state.derived_dfs[name] = df
    print(f"Derived DataFrame '{name}' (shape: {df.shape}) added/updated.")
    # Step 8 UI를 다시 그려서 목록을 갱신하도록 트리거
    if ANALYSIS_STEPS[7] in app_state.module_ui_updaters:
        trigger_specific_module_update(ANALYSIS_STEPS[7])

    # << NEW >>: Step 9 (DataFrame Viewer) UI도 함께 업데이트하도록 신호 추가
    if len(ANALYSIS_STEPS) > 8 and ANALYSIS_STEPS[8] in app_state.module_ui_updaters:
        print(f"Triggering UI update for DataFrame Viewer: {ANALYSIS_STEPS[8]}")
        trigger_specific_module_update(ANALYSIS_STEPS[8])


    # 타겟 변수 관련 UI 업데이트
    if app_state.selected_target_variable and \
       (app_state.current_df is None or app_state.selected_target_variable not in app_state.current_df.columns):
        app_state.selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)

    update_target_variable_combo()
    
    # 만약 8단계가 있다면, 여기서 UI 업데이트를 트리거합니다.
    # if len(ANALYSIS_STEPS) > 7 and ANALYSIS_STEPS[7] in app_state.module_ui_updaters:
    #     print(f"Triggering UI update for subsequent step: {ANALYSIS_STEPS[7]}")
    #     trigger_specific_module_update(ANALYSIS_STEPS[7])


def trigger_specific_module_update(module_name_key: str):
    """특정 모듈의 UI를 업데이트합니다. 모듈별로 필요한 DataFrame을 전달합니다."""
    if not dpg.is_dearpygui_running() or module_name_key not in app_state.module_ui_updaters:
        return

    updater = app_state.module_ui_updaters[module_name_key]
    df_to_use = app_state.current_df # 기본적으로 모든 단계는 current_df를 사용합니다.

    # --- 각 단계별로 호출 방식 및 전달 데이터 재정의 ---
    try:
        if module_name_key == ANALYSIS_STEPS[0]: # Step 1 (Data Loading)
            # Step 1은 original_df와 df_after_step1을 모두 필요로 하는 특별한 케이스
            updater(app_state.df_after_step1, app_state.original_df, util_functions_for_modules, app_state.loaded_file_path)

        elif module_name_key in [ANALYSIS_STEPS[7], ANALYSIS_STEPS[8], ANALYSIS_STEPS[9], ANALYSIS_STEPS[10]]: # << 수정: Step 10 추가
            # 이 단계들의 update_ui는 인자를 받지 않음
            updater()

        else: # Step 2, 4, 5, 6, 7 및 기타 모든 일반 단계
            # 항상 최신 상태인 current_df를 전달
            updater(df_to_use, main_app_callbacks)

        print(f"Module UI updated successfully for: '{module_name_key}'")

    except TypeError as e:
        # TypeError 발생 시 상세 로그 출력
        print(f"ERROR: Could not update UI for '{module_name_key}' due to TypeError.")
        print(f"       Function signature might be incorrect. Error: {e}")
        print(f"       Traceback: {traceback.format_exc()}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while updating UI for '{module_name_key}': {e}")
        print(f"       Traceback: {traceback.format_exc()}")

def trigger_all_module_updates():
    """등록된 모든 모듈의 UI를 업데이트합니다."""
    print("Updating all module UIs...")
    for step_key_or_name in ANALYSIS_STEPS:
        trigger_specific_module_update(step_key_or_name)
    print("All module UIs update process finished.")


def load_data_from_file(file_path: str) -> bool:
    """지정된 경로에서 데이터를 로드하고 애플리케이션 상태를 업데이트합니다."""
    success = False
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".parquet":
            app_state.original_df = pd.read_parquet(file_path)
        elif file_extension == ".csv":
            try:
                app_state.original_df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                try:
                    app_state.original_df = pd.read_csv(file_path, encoding='cp949')
                except Exception as e_csv:
                    raise Exception(f"CSV encoding error (UTF-8 and CP949 failed): {e_csv}")
            except Exception as e_csv_other:
                 raise Exception(f"Error reading CSV: {e_csv_other}")
        else:
            _show_simple_modal_message("File Type Error", f"Unsupported file type: {file_extension}\nPlease select a .parquet or .csv file.")
            return False
            
        app_state.current_df = None
        app_state.df_after_step1 = None
        app_state.df_after_step3 = None
        app_state.df_after_step4 = None
        app_state.df_after_step5 = None
        app_state.df_after_step6 = None 
        app_state.df_after_step7 = None 
        app_state.derived_dfs.clear() 
        app_state.loaded_file_path = file_path
        app_state.original_base_name = _get_base_filename(file_path)
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG):
            dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, f"File: {os.path.basename(file_path)} (Shape: {app_state.original_df.shape if app_state.original_df is not None else 'N/A'})")
        success = True
    except Exception as e:
        app_state.current_df = None; app_state.original_df = None; app_state.loaded_file_path = None
        app_state.df_after_step1 = None; app_state.df_after_step3 = None; app_state.df_after_step4 = None; app_state.df_after_step5 = None; app_state.df_after_step6 = None; app_state.df_after_step7 = None
        app_state.derived_dfs.clear()
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
    """타겟 변수 타입 라디오 버튼 변경 시 호출되는 콜백입니다."""
    new_type = app_data
    prev_valid_type = app_state.selected_target_variable_type
    
    df_for_type_check = app_state.current_df
    if df_for_type_check is None: df_for_type_check = app_state.df_after_step1 # Fallback for guessing

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
    
    if app_state.active_step_name: # Update current step if active
        trigger_specific_module_update(app_state.active_step_name)
    # else: # Or update all if no specific step is active (e.g. initial load)
    #    trigger_all_module_updates() # This might be too much, usually a step is active


def target_variable_selected_callback(sender, app_data, user_data):
    """타겟 변수 콤보박스 선택 변경 시 호출되는 콜백입니다."""
    new_target_variable_name = app_data
    
    df_for_guessing_type = app_state.df_after_step1 # Base type guess on Step 1 processed data
    if df_for_guessing_type is None:
        df_for_guessing_type = app_state.original_df # Fallback to original if step 1 not done

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

    if app_state.active_step_name:
        trigger_specific_module_update(app_state.active_step_name)
    
    sva_ui_updater = main_app_callbacks.get('register_sva_ui_updater')
    if sva_ui_updater and callable(sva_ui_updater):
        try:
            sva_ui_updater()
        except Exception as e:
            print(f"Error explicitly calling SVA UI updater: {e}")


def switch_step_view(sender, app_data, user_data_step_name: str):
    """
    분석 단계를 전환하고 해당 단계의 UI를 표시합니다.
    이 함수는 더 이상 current_df의 상태를 직접 변경하지 않습니다.
    """
    for step_name_iter, group_tag in app_state.step_group_tags.items():
        if dpg.does_item_exist(group_tag):
            is_active_step = (step_name_iter == user_data_step_name)
            # 1. 해당 단계의 UI 그룹을 보여주거나 숨깁니다.
            dpg.configure_item(group_tag, show=is_active_step)

            if is_active_step:
                # 2. 현재 활성화된 스텝 이름만 기록합니다.
                app_state.active_step_name = step_name_iter
                print(f"Switched to view: {step_name_iter}. Triggering UI update with existing current_df.")

                # 3. 해당 스텝의 UI가 최신 current_df를 기반으로 내용을 새로고침하도록 신호를 보냅니다.
                trigger_specific_module_update(step_name_iter)
                update_target_variable_combo()


def file_load_callback(sender, app_data):
    """파일 다이얼로그에서 파일 선택 시 호출되는 콜백입니다."""
    new_file_path = app_data.get('file_path_name')
    if not new_file_path: return

    if app_state.loaded_file_path and app_state.active_settings :
        old_settings_path = get_settings_filepath(app_state.loaded_file_path)
        if old_settings_path:
            current_live_settings = gather_current_settings()
            save_json_settings(old_settings_path, current_live_settings)
            print(f"Saved settings for old file: {app_state.loaded_file_path}")

    reset_application_state(clear_df_completely=False)
    
    if not load_data_from_file(new_file_path):
        return 

    new_settings_path = get_settings_filepath(app_state.loaded_file_path)
    loaded_specific_settings = load_json_settings(new_settings_path)
    
    if loaded_specific_settings:
        print(f"Applying settings for new file: {app_state.loaded_file_path}")
        apply_settings(loaded_specific_settings)
    else: 
        app_state.active_settings = {}
        print(f"No specific settings found for: {app_state.loaded_file_path}. Starting with defaults.")

    if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0:
        first_step_name = ANALYSIS_STEPS[0]
        if first_step_name in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step_name]):
            if app_state.active_step_name != first_step_name or app_state.active_step_name is None:
                 switch_step_view(None, None, first_step_name)
        else: 
             app_state.active_step_name = first_step_name
             print(f"Warning: First step group '{first_step_name}' not fully ready during file load callback. Set as active.")


def reset_application_state(clear_df_completely=True):
    """
    애플리케이션의 주요 상태 변수들을 초기화합니다. (최종 수정 버전)
    중앙 상태 저장소(app_state.active_settings)를 직접 제어하여 확실한 초기화를 보장합니다.
    """
    if clear_df_completely:
        # 파일이 완전히 닫힐 때의 전체 리셋 로직 (기존과 동일)
        app_state.original_df = None
        app_state.loaded_file_path = None
        app_state.original_base_name = None
        app_state.active_settings = {}
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG):
            dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, "No data loaded.")
    else:
        # "Reset All" 버튼을 눌렀을 때의 '부분 리셋' 로직
        
        # 1. [핵심] Step 1 설정을 제외한 모든 설정을 메모리에서 제거합니다.
        step1_settings = app_state.active_settings.get('step_01_settings', {})
        
        clean_settings = {
            # 유지해야 할 최소한의 정보
            'selected_target_variable': app_state.selected_target_variable,
            'selected_target_variable_type': app_state.selected_target_variable_type,
            'active_step_name': ANALYSIS_STEPS[0], # 첫 단계로 활성 스텝 지정
            
            # Step 1 설정은 보존
            'step_01_settings': step1_settings,
            
            # 나머지 모든 후속 단계 설정은 빈 딕셔너리로 초기화
            'step_02a_sva_settings': {},
            'step_02b_mva_settings': {},
            'step_03_preprocessing_settings': {},
            'step_04_missing_values_settings': {},
            'step_05_outlier_treatment_settings': {},
            'step_06_standardization_settings': {},
            'step_07_feature_engineering_settings': {},
            'step_09_data_viewer_settings': {},
            # Step 10 설정도 초기화 (필요 시)
            'step_10_advanced_analysis_settings': {},
        }
        # 중앙 상태 저장소를 정리된 설정으로 즉시 덮어씁니다.
        app_state.active_settings = clean_settings

    # 2. 모든 파생/후속 단계 DataFrame 초기화
    app_state.current_df = None
    app_state.df_after_step3 = None
    app_state.df_after_step4 = None
    app_state.df_after_step5 = None
    app_state.df_after_step6 = None
    app_state.df_after_step7 = None
    app_state.derived_dfs.clear()

    # 3. Step 1 처리 완료 콜백을 다시 호출하여 기본 DataFrame을 생성
    if app_state.original_df is not None:
        if 'step1_processing_complete' in main_app_callbacks:
            # 보존된 Step 1 설정을 사용하여 df_after_step1을 다시 생성
            step_01_data_loading.apply_step1_settings_and_process(
                app_state.original_df, 
                app_state.active_settings.get('step_01_settings', {}), 
                main_app_callbacks
            )
    else:
        if 'step1_processing_complete' in main_app_callbacks:
            main_app_callbacks['step1_processing_complete'](None)
    
    # 4. 모든 모듈의 UI를 업데이트하여 리셋된 상태를 반영
    trigger_all_module_updates()

    # << 추가: Step 10의 상태도 초기화 >>
    if hasattr(step_10_advanced_analysis, 'reset_state'):
        step_10_advanced_analysis.reset_state()

    # 5. 첫 번째 스텝 뷰로 강제 전환
    if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0:
        switch_step_view(None, None, ANALYSIS_STEPS[0])


def export_to_parquet_callback():
    """처리된 데이터를 Parquet 파일로 내보냅니다. (파일명_YYYYMMDD_vN 형식)"""
    df_to_export = None
    export_source_step = ""

    # (내보낼 데이터프레임 선택 로직은 기존과 동일)
    if app_state.df_after_step7 is not None:
        df_to_export = app_state.df_after_step7
        export_source_step = "Step 7 (Feature Engineering)"
    elif app_state.df_after_step6 is not None:
        df_to_export = app_state.df_after_step6
        export_source_step = "Step 6 (Standardization)"
    elif app_state.df_after_step5 is not None:
        df_to_export = app_state.df_after_step5
        export_source_step = "Step 5 (Outlier Treatment)"
    elif app_state.df_after_step4 is not None:
        df_to_export = app_state.df_after_step4
        export_source_step = "Step 4 (Missing Values)"
    elif app_state.df_after_step1 is not None:
        df_to_export = app_state.df_after_step1
        export_source_step = "Step 1 (Load/Overview)"
    
    if df_to_export is None:
        _show_simple_modal_message("Export Info", "No processed data available to export. Please complete at least Step 1.")
        return

    # `loaded_file_path` 와 `original_base_name` 모두 확인
    if not app_state.loaded_file_path or not app_state.original_base_name:
        _show_simple_modal_message("Export Error", "Original file path not found. Cannot determine export location.")
        return

    try:
        original_dir = os.path.dirname(app_state.loaded_file_path)
        base_name = app_state.original_base_name # 저장된 기본 이름 사용
        
        # 오늘 날짜로 파일명 생성 (YYYYMMDD 형식)
        current_date_str = datetime.datetime.now().strftime("%Y%m%d")
        
        # 1. 기본 파일명 확인 (e.g., my_data_20250609.parquet)
        export_filename_base = f"{base_name}_{current_date_str}"
        export_path = os.path.join(original_dir, f"{export_filename_base}.parquet")
        
        # 2. 파일이 존재하면 버전 관리 (v1, v2, ...)
        version = 1
        while os.path.exists(export_path):
            export_filename_versioned = f"{export_filename_base}_v{version}"
            export_path = os.path.join(original_dir, f"{export_filename_versioned}.parquet")
            version += 1
            
        # 최종 결정된 경로로 파일 저장
        df_to_export.to_parquet(export_path, index=False)
        final_filename = os.path.basename(export_path)
        _show_simple_modal_message("Export Successful", f"Data (from {export_source_step}) has been exported to:\n{final_filename}")
        print(f"Data exported to {export_path}")

    except Exception as e:
        _show_simple_modal_message("Export Error", f"Failed to export data to Parquet.\nError: {e}")
        print(f"Error exporting data: {e}"); traceback.print_exc()

def save_current_state():
    """
    현재 애플리케이션의 라이브 상태를 즉시 파일에 저장합니다.
    주로 리셋 직후 호출되어 초기화된 상태를 덮어쓰는 데 사용됩니다.
    """
    if app_state.loaded_file_path and os.path.exists(app_state.loaded_file_path):
        # 각 모듈에서 현재 설정값을 수집
        current_settings = gather_current_settings()
        settings_filepath = get_settings_filepath(app_state.loaded_file_path)
        if settings_filepath:
            # 수집된 설정값을 파일에 저장
            save_json_settings(settings_filepath, current_settings)
            print("Application state has been successfully saved after reset.")
    else:
        # 저장할 수 없는 경우 (데이터 파일이 로드되지 않음)
        print("Could not save state: No data file is currently loaded.")

def reset_and_save_callback(sender, app_data, user_data):
    """
    'Reset' 버튼을 위한 새로운 콜백 함수입니다.
    상태를 초기화하고, 그 결과를 즉시 파일에 저장합니다.
    """
    # 1. user_data로 전달된 main_app_callbacks를 이용해 기존 리셋 로직 호출
    main_app_callbacks = user_data
    if 'reset_current_df_to_original' in main_app_callbacks:
        main_app_callbacks['reset_current_df_to_original']()

    # 2. 초기화된 상태를 즉시 저장
    save_current_state()

    # 3. 사용자에게 리셋 및 저장이 완료되었음을 알림
    _show_simple_modal_message("State Reset", "The application state has been successfully reset and saved.")


def get_settings_filepath(original_data_filepath: str) -> Optional[str]:
    """원본 데이터 파일 경로를 기반으로 설정 파일 경로를 생성합니다."""
    if not original_data_filepath: return None
    try:
        safe_original_path = "".join(c if c.isalnum() or c in [' ', '.', '_', '-'] else '_' for c in original_data_filepath)
        filename_base = hashlib.md5(safe_original_path.encode('utf-8')).hexdigest()
    except Exception: 
        filename_base = "default_settings"
    filename = filename_base + ".json"
    return os.path.join(SETTINGS_DIR_NAME, filename)

def load_json_settings(settings_filepath: str) -> Optional[dict]:
    """JSON 설정 파일을 로드합니다."""
    if settings_filepath and os.path.exists(settings_filepath):
        try:
            with open(settings_filepath, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception as e: print(f"Error loading settings '{settings_filepath}': {e}"); traceback.print_exc()
    return None

def save_json_settings(settings_filepath: str, settings_dict: dict):
    """JSON 설정 파일을 저장합니다."""
    if not settings_filepath or not settings_dict: return
    try:
        if not os.path.exists(SETTINGS_DIR_NAME): os.makedirs(SETTINGS_DIR_NAME)
        with open(settings_filepath, 'w', encoding='utf-8') as f: json.dump(settings_dict, f, indent=4, ensure_ascii=False)
        print(f"Settings saved to {settings_filepath}")
    except Exception as e: print(f"Error saving settings to '{settings_filepath}': {e}"); traceback.print_exc()

def gather_current_settings() -> dict:
    """
    현재 애플리케이션의 모든 설정을 수집합니다. (최종 수정 버전)
    [수정] 이제 중앙 관리되는 app_state.active_settings를 주된 정보 소스로 사용하며,
    각 모듈의 get... 함수를 다시 호출하지 않습니다.
    이를 통해 리셋된 상태가 정확히 저장되도록 보장합니다.
    """
    # 1. 중앙 관리 상태를 기본으로 사용
    settings = app_state.active_settings.copy()

    # 2. 사용자가 UI와 직접 상호작용하여 즉시 바뀌는 최상위 상태 값들 갱신
    settings['selected_target_variable'] = app_state.selected_target_variable
    settings['selected_target_variable_type'] = app_state.selected_target_variable_type
    settings['active_step_name'] = app_state.active_step_name
    
    # 3. 각 모듈의 UI에서 직접 현재 설정값을 가져와 덮어쓰기
    if hasattr(step_01_data_loading, 'get_step1_settings_for_saving'):
        settings['step_01_settings'] = step_01_data_loading.get_step1_settings_for_saving()

    # --- ▼ [수정] 아래 로직 추가/변경 ▼ ---
    if hasattr(step_02a_sva, 'get_sva_settings_for_saving'):
        sva_settings = step_02a_sva.get_sva_settings_for_saving()
        # settings 딕셔너리에 'step_02a_sva_settings' 키가 없는 경우를 대비하여 update 사용
        settings.setdefault('step_02a_sva_settings', {}).update(sva_settings)

    if hasattr(step_02b_mva, 'get_mva_settings_for_saving'):
        mva_settings = step_02b_mva.get_mva_settings_for_saving()
        settings.setdefault('step_02b_mva_settings', {}).update(mva_settings)

    #
    # [핵심] 나머지 단계(Step 2, 4, 5, 6 등)의 설정은 settings 변수 안에 있는 값을 그대로 신뢰하고 반환합니다.
    # 이 값들은 리셋 시점에 이미 빈 딕셔너리로 초기화되었으므로, 더 이상 각 모듈의 get...saving 함수를 호출하지 않습니다.
    #
    return settings

def apply_settings(settings_dict: dict):
    """저장된 설정을 애플리케이션에 적용합니다."""
    if app_state.original_df is None:
        _show_simple_modal_message("Error", "Cannot apply settings: No original data loaded.")
        return
    app_state.active_settings = settings_dict

    app_state.selected_target_variable = settings_dict.get('selected_target_variable')
    app_state.selected_target_variable_type = settings_dict.get('selected_target_variable_type', "Continuous")

    s01_settings = settings_dict.get('step_01_settings', {})
    if hasattr(step_01_data_loading, 'apply_step1_settings_and_process'):
        step_01_data_loading.apply_step1_settings_and_process(app_state.original_df, s01_settings, main_app_callbacks)

    update_target_variable_combo()
    if app_state.selected_target_variable and \
       app_state.df_after_step1 is not None and \
       app_state.selected_target_variable in app_state.df_after_step1.columns:

        s1_types_from_settings = s01_settings.get('type_selections', {})
        guessed_type = utils._guess_target_type(app_state.df_after_step1, app_state.selected_target_variable, s1_types_from_settings)

        final_target_type = settings_dict.get('selected_target_variable_type', guessed_type)
        app_state.selected_target_variable_type = final_target_type

        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, app_state.selected_target_variable_type)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)
    else:
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, "Continuous")

    s02a_settings = settings_dict.get('step_02a_sva_settings', {})
    if hasattr(step_02a_sva, 'apply_sva_settings_from_loaded') and app_state.df_after_step1 is not None:
        step_02a_sva.apply_sva_settings_from_loaded(s02a_settings, app_state.df_after_step1, main_app_callbacks)

    s02b_settings = settings_dict.get('step_02b_mva_settings', {})
    if hasattr(step_02b_mva, 'apply_mva_settings_from_loaded') and app_state.df_after_step1 is not None:
        step_02b_mva.apply_mva_settings_from_loaded(s02b_settings, app_state.df_after_step1, main_app_callbacks)

    app_state.df_after_step3 = None # Step 3 상태는 항상 None

    app_state.df_after_step4 = None
    s04_settings = settings_dict.get('step_04_missing_values_settings', {})
    df_input_for_step4 = app_state.df_after_step1 # Step 3 의존성 제거
    if s04_settings and hasattr(step_04_missing_values, 'apply_missing_values_settings_and_process') and df_input_for_step4 is not None:
        step_04_missing_values.apply_missing_values_settings_and_process(df_input_for_step4, s04_settings, main_app_callbacks)

    app_state.df_after_step5 = None
    s05_settings = settings_dict.get('step_05_outlier_treatment_settings', {})
    df_input_for_step5 = app_state.df_after_step4 if app_state.df_after_step4 is not None else app_state.df_after_step1 # Step 3 의존성 제거
    if s05_settings and hasattr(step_05_outlier_treatment, 'apply_outlier_treatment_settings_and_process') and df_input_for_step5 is not None:
        step_05_outlier_treatment.apply_outlier_treatment_settings_and_process(df_input_for_step5, s05_settings, main_app_callbacks)

    app_state.df_after_step6 = None
    s06_settings = settings_dict.get('step_06_standardization_settings', {})
    df_input_for_step6 = app_state.df_after_step5 if app_state.df_after_step5 is not None else \
                         (app_state.df_after_step4 if app_state.df_after_step4 is not None else app_state.df_after_step1) # Step 3 의존성 제거
    if s06_settings and hasattr(step_06_standardization, 'apply_step6_settings_and_process') and df_input_for_step6 is not None:
        step_06_standardization.apply_step6_settings_and_process(df_input_for_step6, s06_settings, main_app_callbacks)

    app_state.df_after_step7 = None
    s07_settings = settings_dict.get('step_07_feature_engineering_settings', {})
    df_input_for_step7 = app_state.df_after_step6 if app_state.df_after_step6 is not None else \
                         (app_state.df_after_step5 if app_state.df_after_step5 is not None else \
                         (app_state.df_after_step4 if app_state.df_after_step4 is not None else app_state.df_after_step1)) # Step 3 의존성 제거

    app_state.derived_dfs.clear()
    if s07_settings and hasattr(step_07_feature_engineering, 'apply_settings_and_process') and df_input_for_step7 is not None:
        step_07_feature_engineering.apply_settings_and_process(df_input_for_step7, s07_settings, main_app_callbacks)

    trigger_all_module_updates()

    s09_settings = settings_dict.get('step_09_data_viewer_settings', {})
    if s09_settings and hasattr(step_09_data_viewer, 'apply_settings_and_process'):
         step_09_data_viewer.apply_settings_and_process(s09_settings, main_app_callbacks)

    restored_active_step = settings_dict.get('active_step_name', ANALYSIS_STEPS[0] if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0 else None)
    if restored_active_step and restored_active_step in app_state.step_group_tags and \
       dpg.does_item_exist(app_state.step_group_tags[restored_active_step]):
        switch_step_view(None, None, restored_active_step)
    elif ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0 :
        switch_step_view(None, None, ANALYSIS_STEPS[0])

def get_all_available_dfs_callback() -> Dict[str, pd.DataFrame]:
    """Step 8, 9 모듈에서 사용할 수 있는 모든 DF 소스를 딕셔너리로 반환합니다."""
    available_dfs = {}

    # 0. 원본 데이터
    if app_state.original_df is not None:
        available_dfs['0. Original Data'] = app_state.original_df

    # ★★★ 추가된 부분: current_df를 최상단에 추가 ★★★
    if app_state.current_df is not None:
        # 다른 DF와 이름이 겹치지 않도록 고유한 이름 부여
        available_dfs['★ Current Working DF'] = app_state.current_df

    # 1. 단계별 데이터
    if app_state.df_after_step1 is not None:
        available_dfs['1. After Type Conversion'] = app_state.df_after_step1
    # Step 3은 제거되었으므로 df_after_step3 참조는 없음
    if app_state.df_after_step4 is not None:
        available_dfs['4. After Missing Value Imputation'] = app_state.df_after_step4
    if app_state.df_after_step5 is not None:
        available_dfs['5. After Outlier Treatment'] = app_state.df_after_step5
    if app_state.df_after_step6 is not None:
        available_dfs['6. After Standardization'] = app_state.df_after_step6
    if app_state.df_after_step7 is not None:
        available_dfs['7. After Feature Engineering'] = app_state.df_after_step7

    # 2. 파생 데이터
    for name, df in app_state.derived_dfs.items():
        available_dfs[f"Derived: {name}"] = df

    return available_dfs

def add_or_update_derived_df_callback(name: str, df: pd.DataFrame):
    app_state.derived_dfs[name] = df

def delete_derived_df_callback(name: str):
    """지정된 이름의 파생 데이터프레임을 삭제하고, 관련 UI를 새로고침합니다."""
    if name in app_state.derived_dfs:
        del app_state.derived_dfs[name]
        print(f"Derived DataFrame '{name}' deleted.")
        
        # UI 새로고침 트리거
        # Step 8 (Derive DataFrames) UI 업데이트
        if ANALYSIS_STEPS[7] in app_state.module_ui_updaters:
            trigger_specific_module_update(ANALYSIS_STEPS[7])
        # Step 9 (DataFrame Viewer) UI 업데이트
        if len(ANALYSIS_STEPS) > 8 and ANALYSIS_STEPS[8] in app_state.module_ui_updaters:
            trigger_specific_module_update(ANALYSIS_STEPS[8])
    else:
        print(f"Warning: Attempted to delete non-existent derived DataFrame: '{name}'")


def initial_load_on_startup():
    """애플리케이션 시작 시 이전 세션 정보를 로드합니다."""
    session_info = load_json_settings(SESSION_INFO_FILE)
    last_file_path = session_info.get('last_opened_original_file') if session_info else None
    
    if last_file_path and os.path.exists(last_file_path):
        try:
            if not load_data_from_file(last_file_path):
                raise Exception("Failed to load data from last session file.")

            settings_for_file_path = get_settings_filepath(last_file_path)
            specific_file_settings = load_json_settings(settings_for_file_path)
            
            if specific_file_settings:
                apply_settings(specific_file_settings)
            else: 
                app_state.active_settings = {}
                if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0:
                    first_step_name = ANALYSIS_STEPS[0]
                    if app_state.active_step_name != first_step_name:
                        if first_step_name in app_state.step_group_tags and \
                           dpg.does_item_exist(app_state.step_group_tags[first_step_name]):
                            switch_step_view(None, None, first_step_name)
                        else: 
                             app_state.active_step_name = first_step_name
            return 
        except Exception as e:
            _show_simple_modal_message("Session Restore Error", f"Could not fully restore session for: {os.path.basename(last_file_path)}\nError: {e}")
    
    reset_application_state(clear_df_completely=True)


def save_state_on_exit():
    """애플리케이션 종료 시 현재 상태를 저장합니다."""
    if app_state.loaded_file_path and os.path.exists(app_state.loaded_file_path):
        current_settings = gather_current_settings()
        settings_filepath = get_settings_filepath(app_state.loaded_file_path)
        if settings_filepath:
            save_json_settings(settings_filepath, current_settings)
            save_json_settings(SESSION_INFO_FILE, {'last_opened_original_file': app_state.loaded_file_path})
    elif os.path.exists(SESSION_INFO_FILE):
        save_json_settings(SESSION_INFO_FILE, {})


util_functions_for_modules = {
    'create_table_with_data': utils.create_table_with_data,
    'calculate_column_widths': utils.calculate_column_widths,
    'format_text_for_display': utils.format_text_for_display,
    'get_safe_text_size': utils.get_safe_text_size,
    '_show_simple_modal_message': _show_simple_modal_message, # main_app.py 내 정의된 함수
    'show_dpg_alert_modal': utils.show_dpg_alert_modal, # utils.py 에 정의된 함수
    'show_confirmation_modal': utils.show_confirmation_modal,
    'get_numeric_cols': utils._get_numeric_cols,
    'get_categorical_cols': utils._get_categorical_cols,
    'calculate_cramers_v': utils.calculate_cramers_v,
    'calculate_feature_target_relevance': utils.calculate_feature_target_relevance,
    'plot_to_dpg_texture': utils.plot_to_dpg_texture, # 시각화 유틸리티 함수 추가
    'create_table_with_large_data_preview': utils.create_table_with_large_data_preview,
    # << 추가: Step 10을 위한 유틸리티 함수 >>
    'show_dpg_progress_modal': utils.show_dpg_progress_modal,
    'hide_dpg_progress_modal': utils.hide_dpg_progress_modal,
}

main_app_callbacks = {
    'get_current_df': lambda: app_state.current_df,
    'get_original_df': lambda: app_state.original_df,
    'get_df_after_step1': lambda: app_state.df_after_step1,
    'get_df_after_step3': lambda: app_state.df_after_step3,
    'get_df_after_step4': lambda: app_state.df_after_step4,
    'get_df_after_step5': lambda: app_state.df_after_step5,
    'get_df_after_step6': lambda: app_state.df_after_step6,
    'get_df_after_step7': lambda: app_state.df_after_step7, # << ADDED
    'get_derived_dfs': lambda: app_state.derived_dfs.copy(), # << ADDED
    'get_all_available_dfs': get_all_available_dfs_callback, # << ADDED

    'get_loaded_file_path': lambda: app_state.loaded_file_path,
    'get_util_funcs': lambda: util_functions_for_modules,
    'show_file_dialog': lambda: dpg.show_item("file_dialog_id"),
    'register_step_group_tag': lambda name, tag: app_state.step_group_tags.update({name: tag}),
    'register_module_updater': lambda name, func: app_state.module_ui_updaters.update({name: func}),
    'trigger_module_update': trigger_specific_module_update,
    'reset_current_df_to_original': lambda: (
        reset_application_state(clear_df_completely=False)
    ),
    'trigger_all_module_updates': trigger_all_module_updates,
    'get_selected_target_variable': lambda: app_state.selected_target_variable,
    'get_selected_target_variable_type': lambda: app_state.selected_target_variable_type,
    'update_target_variable_combo_items': update_target_variable_combo,
    'get_column_analysis_types': lambda: step_01_data_loading._type_selections.copy() if hasattr(step_01_data_loading, '_type_selections') else {},
    'step1_processing_complete': step1_processing_complete,
    'step4_missing_value_processing_complete': step4_missing_value_processing_complete,
    'step5_outlier_treatment_complete': step5_outlier_treatment_complete,
    'step6_standardization_complete': step6_standardization_complete,
    'step7_feature_engineering_complete': step7_feature_engineering_complete, # << ADDED
    'step8_derivation_complete': step8_derivation_complete, # << ADDED
    'delete_derived_df': delete_derived_df_callback,
    'add_ai_log': add_ai_log_message,
}

dpg.create_context()

TEXTURE_REGISTRY_TAG = "primary_texture_registry"
if not dpg.does_item_exist(TEXTURE_REGISTRY_TAG):
    dpg.add_texture_registry(tag=TEXTURE_REGISTRY_TAG, show=False) # show=False 로 변경

with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback, id="file_dialog_id", width=700, height=400, modal=True):
    dpg.add_file_extension(".parquet")
    dpg.add_file_extension(".csv")
    dpg.add_file_extension(".*")
with dpg.file_dialog(directory_selector=False, show=False, id="sva_export_file_dialog_id", width=700, height=400, modal=True):
    dpg.add_file_extension(".html", color=(0, 255, 0, 255))
    dpg.add_file_extension(".xlsx", color=(0, 255, 0, 255))
    dpg.add_file_extension(".*")

setup_korean_font()

with dpg.window(label="Data Analysis Platform", tag="main_window"):
    with dpg.group(horizontal=True):
        dpg.add_button(label="Open File", callback=lambda: dpg.show_item("file_dialog_id"), width=130, height=30)
        dpg.add_button(label="Reset All (to Step 1 Types)", user_data=main_app_callbacks, width=210, height=30,
                       callback=reset_and_save_callback) # <<< [수정] 콜백 함수를 새로 만든 함수로 변경합니다.
        dpg.add_button(label="Export to Parquet", callback=export_to_parquet_callback, width=160, height=30)
        dpg.add_text("No data loaded.", tag=MAIN_FILE_PATH_DISPLAY_TAG, wrap=-1)
    dpg.add_separator()
    
    with dpg.group(horizontal=True, tag="main_layout_group"): # 전체를 감싸는 가로 그룹
        with dpg.child_window(width=280, tag="navigation_panel", border=True, parent="main_layout_group"):
            dpg.add_text("Target Variable (y):")
            dpg.add_combo(items=[""], tag=TARGET_VARIABLE_COMBO_TAG, width=-1, callback=target_variable_selected_callback)
            dpg.add_text("Target Variable Type:", tag=TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            dpg.add_radio_button(items=["Categorical", "Continuous"], tag=TARGET_VARIABLE_TYPE_RADIO_TAG, horizontal=True,
                                 default_value=app_state.selected_target_variable_type, callback=target_variable_type_changed_callback, show=False)
            dpg.add_separator(); dpg.add_spacer(height=5)
            dpg.add_text("Analysis Steps", color=[255,255,0]); dpg.add_separator()
            for step_name_nav in ANALYSIS_STEPS:
                dpg.add_button(label=step_name_nav, callback=switch_step_view, user_data=step_name_nav, width=-1, height=30)

        ai_log_panel_width = 300
        content_area_width = -(ai_log_panel_width)


        with dpg.child_window(tag="content_area", border=True, parent="main_layout_group", width=content_area_width):
            for step_name_create in ANALYSIS_STEPS:
                if step_name_create == ANALYSIS_STEPS[0]:
                    if hasattr(step_01_data_loading, 'create_ui'):
                        step_01_data_loading.create_ui(step_name_create, "content_area", main_app_callbacks)
                elif step_name_create == ANALYSIS_STEPS[1]:
                    eda_step_group_tag = "eda_step_main_group"
                    main_app_callbacks['register_step_group_tag'](step_name_create, eda_step_group_tag)
                    with dpg.group(tag=eda_step_group_tag, parent="content_area", show=False):
                        dpg.add_text(f"--- {step_name_create} ---")
                        dpg.add_separator()
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
                
                elif step_name_create == ANALYSIS_STEPS[5]: # Step 6 UI 생성
                    if hasattr(step_06_standardization, 'create_ui'):
                        step_06_standardization.create_ui(step_name_create, "content_area", main_app_callbacks)

                elif step_name_create == ANALYSIS_STEPS[6]: # << ADDED: Step 7 UI 생성
                    if hasattr(step_07_feature_engineering, 'create_ui'):
                        step_07_feature_engineering.create_ui(step_name_create, "content_area", main_app_callbacks)

                elif step_name_create == ANALYSIS_STEPS[7]: # Step 8 UI 생성
                    if hasattr(step_08_derivation, 'create_ui'):
                        step_08_derivation.create_ui(step_name_create, "content_area", main_app_callbacks)

                elif step_name_create == ANALYSIS_STEPS[8]: # << NEW >>: Step 9 UI 생성
                    if hasattr(step_09_data_viewer, 'create_ui'):
                        step_09_data_viewer.create_ui(step_name_create, "content_area", main_app_callbacks)
                
                elif step_name_create == ANALYSIS_STEPS[9]: # << 추가: Step 10 UI 생성
                    if hasattr(step_10_advanced_analysis, 'create_ui'):
                        step_10_advanced_analysis.create_ui(step_name_create, "content_area", main_app_callbacks)
                elif step_name_create == ANALYSIS_STEPS[10]:  # Step 11
                    if hasattr(step_11_ml_modeling, 'create_ui'):
                        step_11_ml_modeling.create_ui(step_name_create, "content_area", main_app_callbacks)

            if ANALYSIS_STEPS and len(ANALYSIS_STEPS) > 0 and not app_state.active_step_name:
                first_step = ANALYSIS_STEPS[0]
                if first_step in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step]):
                    switch_step_view(None, None, first_step)
                else: 
                    app_state.active_step_name = first_step

        with dpg.child_window(tag="ai_analysis_log_panel", width=ai_log_panel_width, border=True, parent="main_layout_group"):
            dpg.add_text("💡 AI Analysis Log", color=[255, 255, 0])
            dpg.add_separator()
            dpg.add_text("AI 분석 결과가 여기에 표시됩니다.\n", tag="ai_analysis_log_panel_text", wrap=ai_log_panel_width)
            
dpg.create_viewport(title='Data Analysis Platform', width=1800, height=1100)
dpg.set_exit_callback(save_state_on_exit)
dpg.setup_dearpygui()

initial_load_on_startup()

dpg.show_viewport()
dpg.set_primary_window("main_window", True)

# 명시적인 메인 루프 시작
while dpg.is_dearpygui_running():
    # step_11의 비동기 결과 확인 함수를 매 프레임마다 호출
    if hasattr(step_11_ml_modeling, '_check_for_updates') and callable(step_11_ml_modeling._check_for_updates):
        step_11_ml_modeling._check_for_updates()
    
    # DPG 프레임 렌더링
    dpg.render_dearpygui_frame()

dpg.destroy_context()