# main_app.py
import dearpygui.dearpygui as dpg
import pandas as pd
import platform
import os
import utils
from typing import Optional
import step_01_data_loading
import step_02a_sva
import step_02b_mva
import step_03_preprocessing_editor # 이전 step_03_preprocessing.py (이름 변경 가정)
import step_04_missing_values       # <<< [새 모듈 임포트]
import traceback
import hashlib
import json

# --- Step 03 (노드 에디터) 저장/불러오기 기능 활성화 플래그 ---
STEP_03_SAVE_LOAD_ENABLED = False 

class AppState:
    def __init__(self):
        self.current_df = None
        self.original_df = None
        self.df_after_step1 = None
        self.df_after_step3 = None # 노드 에디터(Step 3) 결과
        self.df_after_step4 = None # 결측치 처리(Step 4) 결과 <<< [새로운 상태 추가]
        self.loaded_file_path = None
        self.selected_target_variable = None
        self.selected_target_variable_type = "Continuous"
        self.active_step_name = None
        self.active_settings = {}
        self.step_group_tags = {}
        self.module_ui_updaters = {}

app_state = AppState()

SETTINGS_DIR_NAME = ".app_file_settings"
SESSION_INFO_FILE = os.path.join(SETTINGS_DIR_NAME, "session_info.json")

SVA_STEP_KEY = "2a. Single Variable Analysis (SVA)"
MVA_STEP_KEY = "2b. Multivariate Analysis (MVA)"
TARGET_VARIABLE_TYPE_RADIO_TAG = "main_target_variable_type_radio"
TARGET_VARIABLE_TYPE_LABEL_TAG = "main_target_variable_type_label"
TARGET_VARIABLE_COMBO_TAG = "main_target_variable_combo"
MAIN_FILE_PATH_DISPLAY_TAG = "main_file_path_display_text"

ANALYSIS_STEPS = [
    "1. Data Loading and Overview",
    "2. Exploratory Data Analysis (EDA)",
    "3. Preprocessing (Pipeline Editor)", # 이전 Step 3 이름 변경 제안 반영
    "4. Missing Value Treatment",         # <<< [새로운 단계 추가]
]

_MODAL_ID_SIMPLE_MESSAGE = "main_simple_modal_message_id"

def _show_simple_modal_message(title: str, message: str, width: int = 450, height: int = 200):
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
            spacer_w = (width - 100 - (item_spacing_x * 2)) // 2
            dpg.add_spacer(width=max(0, int(spacer_w)))
            dpg.add_button(label="OK", width=100, callback=lambda: dpg.configure_item(_MODAL_ID_SIMPLE_MESSAGE, show=False))

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
                dpg.bind_font("korean_font_for_app")
            print(f"Font bound: {font_path}")
        except Exception as e: print(f"Font error: {e}"); traceback.print_exc()
    else: print("Korean font not found. Using default.")


def update_target_variable_combo():
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        df_for_combo = app_state.df_after_step1 if app_state.df_after_step1 is not None else app_state.original_df
        items = [""] + list(df_for_combo.columns) if df_for_combo is not None and not df_for_combo.empty else [""]
        dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=items)
        
        current_val = app_state.selected_target_variable
        if current_val and df_for_combo is not None and current_val in df_for_combo.columns:
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, current_val)
        else: 
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
            app_state.selected_target_variable = None 


def step1_processing_complete(processed_df: pd.DataFrame):
    if processed_df is None: print("Step 1 returned no DataFrame."); return
    app_state.df_after_step1 = processed_df.copy()
    app_state.current_df = app_state.df_after_step1.copy() 
    app_state.df_after_step3 = None # Step 1이 다시 완료되면 이후 단계 결과는 초기화
    app_state.df_after_step4 = None # <<< [추가] Step 4 결과도 초기화

    if app_state.selected_target_variable and (app_state.current_df is None or app_state.selected_target_variable not in app_state.current_df.columns):
        app_state.selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
    
    update_target_variable_combo()
    trigger_all_module_updates_except_step1() 

def step3_processing_complete(processed_df: pd.DataFrame): # 이 함수는 step_03_preprocessing_editor.py 에서 호출
    if processed_df is None:
        print("Step 3 (Editor) returned no DataFrame.")
        # Step 3 결과가 None이면, current_df를 이전 단계(Step1 완료 후)로 되돌릴 수 있음
        app_state.current_df = app_state.df_after_step1.copy() if app_state.df_after_step1 is not None else None
        app_state.df_after_step3 = None
    else:
        app_state.df_after_step3 = processed_df.copy()
        app_state.current_df = app_state.df_after_step3.copy() # 이후 단계를 위해 current_df 업데이트
    
    app_state.df_after_step4 = None # <<< [추가] Step 3 완료 시 Step 4 결과 초기화
    # ... (타겟 변수 UI 업데이트 로직은 동일) ...
    update_target_variable_combo() 
    # Step 3 이후의 다른 모듈이 있다면 해당 모듈 업데이트 트리거 (예: Step 4)
    if ANALYSIS_STEPS[3] in app_state.module_ui_updaters: # "4. Missing Value Treatment"
        trigger_specific_module_update(ANALYSIS_STEPS[3])

def step4_missing_value_processing_complete(processed_df: pd.DataFrame):
    """Step 4 (결측치 처리) 완료 시 호출되는 콜백"""
    if processed_df is None:
        print("Step 4 (Missing Values) returned no DataFrame.")
        # Step 4 결과가 None이면, current_df를 이전 단계 결과로 설정
        if app_state.df_after_step3 is not None:
            app_state.current_df = app_state.df_after_step3.copy()
        elif app_state.df_after_step1 is not None:
            app_state.current_df = app_state.df_after_step1.copy()
        else:
            app_state.current_df = None
        app_state.df_after_step4 = None
    else:
        app_state.df_after_step4 = processed_df.copy()
        app_state.current_df = app_state.df_after_step4.copy() # 이후 단계를 위해 current_df 업데이트
    
    # 타겟 변수 UI는 current_df 기준으로 업데이트/검증 필요 (컬럼 변경 가능성)
    if app_state.selected_target_variable and \
       (app_state.current_df is None or app_state.selected_target_variable not in app_state.current_df.columns):
        app_state.selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
    
    update_target_variable_combo() # 컬럼 목록이 변경되었을 수 있으므로 업데이트
    
    # Step 4 이후의 다른 모듈(예: Step 5 이상치 처리)이 있다면 해당 모듈 업데이트 트리거
    # if ANALYSIS_STEPS[4] in app_state.module_ui_updaters: # "5. Outlier Treatment"
    #     trigger_specific_module_update(ANALYSIS_STEPS[4])

def trigger_specific_module_update(module_name_key: str):
    df_to_use = app_state.current_df # 기본적으로 현재 활성화된 df 사용

    # 각 스텝별로 UI 업데이트에 필요한 DataFrame을 결정하는 로직 개선
    if module_name_key == ANALYSIS_STEPS[0]: # "1. Data Loading and Overview"
        # Step1은 original_df와 df_after_step1을 모두 볼 수 있음
        if module_name_key in app_state.module_ui_updaters:
            updater = app_state.module_ui_updaters[module_name_key]
            if hasattr(step_01_data_loading, 'update_ui'):
                # Step1 update_ui는 original_df와 df_after_step1을 모두 받을 수 있도록 설계됨
                updater(app_state.df_after_step1, app_state.original_df, util_functions_for_modules, app_state.loaded_file_path)
                print(f"Module UI updated for: '{module_name_key}'")
        return

    elif module_name_key == ANALYSIS_STEPS[1]: # "2. Exploratory Data Analysis (EDA)"
        # EDA는 Step1 결과(df_after_step1) 또는 Step3(Editor) 결과(df_after_step3) 또는 Step4(결측치) 결과(df_after_step4) 등을 볼 수 있어야 함
        # 여기서는 가장 마지막 처리된 df를 current_df로 사용한다고 가정
        df_for_eda = app_state.current_df 
        if SVA_STEP_KEY in app_state.module_ui_updaters:
            # ... (SVA 업데이트 로직, df_for_eda 사용)
            app_state.module_ui_updaters[SVA_STEP_KEY](df_for_eda, main_app_callbacks)
            print(f"Module UI updated for: '{SVA_STEP_KEY}' (as part of EDA)")
        if MVA_STEP_KEY in app_state.module_ui_updaters:
            # ... (MVA 업데이트 로직, df_for_eda 사용)
            app_state.module_ui_updaters[MVA_STEP_KEY](df_for_eda, main_app_callbacks)
            print(f"Module UI updated for: '{MVA_STEP_KEY}' (as part of EDA)")
        return

    elif module_name_key == ANALYSIS_STEPS[2]: # "3. Preprocessing (Pipeline Editor)"
        # Step3(Editor)은 Step1의 결과(df_after_step1)를 입력으로 사용
        df_for_step3_editor = app_state.df_after_step1 
        if module_name_key in app_state.module_ui_updaters:
            updater = app_state.module_ui_updaters[module_name_key]
            if hasattr(step_03_preprocessing_editor, 'update_ui'): # 모듈명 변경 가정
                 updater(df_for_step3_editor, main_app_callbacks)
                 print(f"Module UI updated for: '{module_name_key}'")
        return
        
    elif module_name_key == ANALYSIS_STEPS[3]: # "4. Missing Value Treatment" <<< [새로운 단계 로직 추가]
        # Step4(결측치 처리)는 Step3(Editor)의 결과(df_after_step3)가 있으면 그것을, 없으면 Step1의 결과(df_after_step1)를 입력으로 사용
        df_for_step4 = app_state.df_after_step3 if app_state.df_after_step3 is not None else app_state.df_after_step1
        if module_name_key in app_state.module_ui_updaters:
            updater = app_state.module_ui_updaters[module_name_key]
            if hasattr(step_04_missing_values, 'update_ui'):
                 updater(df_for_step4, main_app_callbacks)
                 print(f"Module UI updated for: '{module_name_key}'")
        return

    # 이후 추가될 다른 모듈들 (일반적인 경우)
    if module_name_key in app_state.module_ui_updaters:
        updater = app_state.module_ui_updaters[module_name_key]
        # 일반적인 updater는 가장 최근 처리된 current_df와 main_callbacks를 받는다고 가정
        updater(app_state.current_df, main_app_callbacks)
        print(f"Module UI updated for: '{module_name_key}'")


def trigger_all_module_updates():
    print("Updating all module UIs...")
    for step_key in list(app_state.module_ui_updaters.keys()):
        trigger_specific_module_update(step_key)

def trigger_all_module_updates_except_step1():
    print("Updating all module UIs except Step 1...")
    for step_name_iter in ANALYSIS_STEPS:
        if step_name_iter == ANALYSIS_STEPS[0]: 
            continue
        
        if step_name_iter == ANALYSIS_STEPS[1]:
            if SVA_STEP_KEY in app_state.module_ui_updaters:
                trigger_specific_module_update(SVA_STEP_KEY)
            if MVA_STEP_KEY in app_state.module_ui_updaters:
                trigger_specific_module_update(MVA_STEP_KEY)
        elif step_name_iter in app_state.module_ui_updaters: 
            trigger_specific_module_update(step_name_iter)


def load_data_from_file(file_path: str) -> bool:
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
                    raise Exception(f"CSV encoding error: {e_csv}")
            except Exception as e_csv_other:
                 raise Exception(f"Error reading CSV: {e_csv_other}")
        else:
            _show_simple_modal_message("File Type Error", f"Unsupported file type: {file_extension}\nPlease select a .parquet or .csv file.")
            return False
            
        app_state.df_after_step1 = None; app_state.df_after_step3 = None; app_state.df_after_step4 = None # <<< [추가]
        app_state.loaded_file_path = file_path
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG):
            dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, f"File: {file_path} (Shape: {app_state.original_df.shape})")
        success = True
    except Exception as e:
        app_state.current_df = None; app_state.original_df = None; app_state.loaded_file_path = None
        app_state.df_after_step1 = None; app_state.df_after_step3 = None; app_state.df_after_step4 = None # <<< [추가]
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG):
            dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, f"Error loading file: {os.path.basename(file_path)}")
        _show_simple_modal_message("File Load Error", f"Failed to load data from '{os.path.basename(file_path)}'.\nError: {e}")
        print(f"Error loading raw data: {e}"); traceback.print_exc()
        success = False

    if success:
        if hasattr(step_01_data_loading, 'reset_step1_state_for_new_file'): 
            step_01_data_loading.reset_step1_state_for_new_file()
        elif hasattr(step_01_data_loading, '_type_selections'): step_01_data_loading._type_selections.clear()

        if app_state.selected_target_variable and (app_state.original_df is None or app_state.selected_target_variable not in app_state.original_df.columns):
            app_state.selected_target_variable = None
            if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
        
        update_target_variable_combo() 

        if ANALYSIS_STEPS and ANALYSIS_STEPS[0] in app_state.module_ui_updaters:
             trigger_specific_module_update(ANALYSIS_STEPS[0]) 
        return True
    else: 
        reset_application_state(clear_df_completely=True) 
        trigger_all_module_updates() 
        return False


def target_variable_type_changed_callback(sender, app_data, user_data):
    new_type = app_data
    prev_valid_type = app_state.selected_target_variable_type
    
    df_for_type_check = app_state.current_df 
    if df_for_type_check is None: 
        df_for_type_check = app_state.df_after_step1

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
        else:
            if app_state.selected_target_variable in df_for_type_check.columns:
                dtype_in_current_df = df_for_type_check[app_state.selected_target_variable].dtype
                if pd.api.types.is_object_dtype(dtype_in_current_df) or \
                   pd.api.types.is_string_dtype(dtype_in_current_df) or \
                   pd.api.types.is_categorical_dtype(dtype_in_current_df) or \
                   pd.api.types.is_bool_dtype(dtype_in_current_df):
                    block_continuous_selection = True
        
        if block_continuous_selection:
            var_name = app_state.selected_target_variable
            current_s1_type_msg = f"(Step1 type: {analysis_type_s1})" if analysis_type_s1 else "(Step1 type: N/A)"
            df_source_msg = "current data" 
            if df_for_type_check is app_state.df_after_step3: df_source_msg = "data after Step 3"
            elif df_for_type_check is app_state.df_after_step1: df_source_msg = "data after Step 1"

            err_msg = f"Variable '{var_name}' {current_s1_type_msg} in {df_source_msg} is not suitable for 'Continuous' type.\n" \
                      f"Please use 'Categorical' or verify its type in Step 1."
            _show_simple_modal_message("Type Selection Warning", err_msg, width=500, height=220)
            
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, prev_valid_type)
            return
            
    app_state.selected_target_variable_type = new_type
    if app_state.active_settings: app_state.active_settings['selected_target_variable_type'] = new_type
    
    if app_state.active_step_name:
        trigger_specific_module_update(app_state.active_step_name)


def target_variable_selected_callback(sender, app_data, user_data):
    new_target = app_data
    df_for_guess = app_state.current_df if app_state.current_df is not None else app_state.df_after_step1
    
    if not new_target: 
        app_state.selected_target_variable = None
        app_state.selected_target_variable_type = "Continuous" 
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, "Continuous")
        if app_state.active_settings: app_state.active_settings['selected_target_variable'] = None; app_state.active_settings['selected_target_variable_type'] = None
    else: 
        app_state.selected_target_variable = new_target
        if df_for_guess is not None and new_target in df_for_guess.columns:
            s1_types = main_app_callbacks.get('get_column_analysis_types', lambda: {})()
            guessed_type = utils._guess_target_type(df_for_guess, new_target, s1_types)
            app_state.selected_target_variable_type = guessed_type
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, guessed_type)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)
            if app_state.active_settings: app_state.active_settings['selected_target_variable'] = new_target; app_state.active_settings['selected_target_variable_type'] = guessed_type

    if app_state.active_step_name:
        trigger_specific_module_update(app_state.active_step_name)
    else: 
        trigger_all_module_updates()


def switch_step_view(sender, app_data, user_data_step_name: str):
    for step_name_iter, group_tag in app_state.step_group_tags.items():
        if dpg.does_item_exist(group_tag):
            is_active_step = (step_name_iter == user_data_step_name)
            dpg.configure_item(group_tag, show=is_active_step)
            if is_active_step:
                app_state.active_step_name = step_name_iter
                # current_df를 현재 활성화된 스텝에 맞게 설정 (중요!)
                if step_name_iter == ANALYSIS_STEPS[0]: # Data Loading
                    app_state.current_df = app_state.df_after_step1 
                elif step_name_iter == ANALYSIS_STEPS[1]: # EDA
                    # EDA는 가장 최근 처리된 결과를 봐야 함
                    if app_state.df_after_step4 is not None: app_state.current_df = app_state.df_after_step4
                    elif app_state.df_after_step3 is not None: app_state.current_df = app_state.df_after_step3
                    else: app_state.current_df = app_state.df_after_step1
                elif step_name_iter == ANALYSIS_STEPS[2]: # Preprocessing (Editor)
                    # 에디터는 Step1 결과를 입력으로 받음
                    app_state.current_df = app_state.df_after_step1
                elif step_name_iter == ANALYSIS_STEPS[3]: # Missing Value Treatment <<< [새로운 단계 로직 추가]
                    # 결측치 처리는 에디터 결과(df_after_step3) 또는 Step1 결과(df_after_step1)를 입력으로 받음
                    if app_state.df_after_step3 is not None: app_state.current_df = app_state.df_after_step3
                    else: app_state.current_df = app_state.df_after_step1
                # ... (이후 단계에 대한 current_df 설정 규칙 추가) ...
                else: # 기본적으로는 가장 최신 처리된 df를 사용 (가장 마지막 단계의 df_after_stepX)
                      # 이 부분은 단계가 많아질수록 복잡해지므로, 각 스텝 진입 시 명확한 입력 df를 지정하는 것이 좋음
                    if app_state.df_after_step4 is not None: app_state.current_df = app_state.df_after_step4
                    elif app_state.df_after_step3 is not None: app_state.current_df = app_state.df_after_step3
                    elif app_state.df_after_step1 is not None: app_state.current_df = app_state.df_after_step1
                    else: app_state.current_df = app_state.original_df
                
                print(f"Switched to {step_name_iter}. Input 'current_df' for this step is now set.")
                trigger_specific_module_update(step_name_iter) # 여기서 current_df를 사용하지 않고, trigger_specific_module_update 내부에서 다시 df_to_use를 결정


def file_load_callback(sender, app_data):
    new_file_path = app_data.get('file_path_name')
    if not new_file_path: return

    if app_state.loaded_file_path and app_state.active_settings: 
        old_settings_path = get_settings_filepath(app_state.loaded_file_path)
        current_live_settings = gather_current_settings()
        save_json_settings(old_settings_path, current_live_settings)

    reset_application_state(clear_df_completely=False) 
    
    if not load_data_from_file(new_file_path): 
        return

    new_settings_path = get_settings_filepath(app_state.loaded_file_path)
    loaded_specific_settings = load_json_settings(new_settings_path)
    
    if loaded_specific_settings: 
        apply_settings(loaded_specific_settings) 
    else: 
        app_state.active_settings = {} 
        if hasattr(step_01_data_loading, 'process_newly_loaded_data') and app_state.original_df is not None:
            step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks)
        else: 
            trigger_specific_module_update(ANALYSIS_STEPS[0])

    trigger_all_module_updates()

    if ANALYSIS_STEPS: switch_step_view(None, None, ANALYSIS_STEPS[0])


def reset_application_state(clear_df_completely=True):
    if clear_df_completely:
        app_state.original_df = None
        app_state.loaded_file_path = None
        app_state.active_settings = {}
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG): dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, "No data loaded.")
    
    app_state.current_df = None
    app_state.df_after_step1 = None
    app_state.df_after_step3 = None
    app_state.df_after_step4 = None

    app_state.selected_target_variable = None
    app_state.selected_target_variable_type = "Continuous"
    
    if hasattr(step_01_data_loading, 'reset_step1_state'): step_01_data_loading.reset_step1_state()
    if hasattr(step_02a_sva, 'reset_sva_ui_defaults'): step_02a_sva.reset_sva_ui_defaults()
    if hasattr(step_02b_mva, 'reset_mva_ui_defaults'): step_02b_mva.reset_mva_ui_defaults()
    if hasattr(step_03_preprocessing, 'reset_preprocessing_state'): step_03_preprocessing.reset_preprocessing_state()
    if hasattr(step_04_missing_values, 'reset_missing_values_state'): step_04_missing_values.reset_missing_values_state() # <<< [추가]


    if clear_df_completely: 
        app_state.active_step_name = None 
        if ANALYSIS_STEPS and ANALYSIS_STEPS[0] in app_state.step_group_tags and \
           dpg.does_item_exist(app_state.step_group_tags[ANALYSIS_STEPS[0]]):
            print(f"Resetting to first step: {ANALYSIS_STEPS[0]}")
            switch_step_view(None, None, ANALYSIS_STEPS[0])
        else: 
            app_state.active_step_name = ANALYSIS_STEPS[0] if ANALYSIS_STEPS else None
            print(f"Warning: First step group not ready during reset, setting active_step_name to {app_state.active_step_name}")
        trigger_all_module_updates() 
    elif app_state.original_df is not None: 
        if hasattr(step_01_data_loading, 'process_newly_loaded_data'):
            step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks)
        trigger_all_module_updates()
    
    if ANALYSIS_STEPS:
        current_active = app_state.active_step_name
        first_step = ANALYSIS_STEPS[0]
        if not current_active or current_active not in app_state.step_group_tags or \
           not dpg.does_item_exist(app_state.step_group_tags.get(current_active, "")): 
            if first_step in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step]):
                switch_step_view(None, None, first_step)


def export_to_parquet_callback():
    df_to_export = None
    export_suffix = "_processed"

    if app_state.df_after_step3 is not None:
        df_to_export = app_state.df_after_step3
        export_suffix = "_preprocessed"
    elif app_state.df_after_step1 is not None:
        df_to_export = app_state.df_after_step1
        export_suffix = "_step1_processed"
    elif app_state.original_df is not None: 
        _show_simple_modal_message("Export Info", "No processed data to export. Please complete at least Step 1.")
        return
    
    if df_to_export is None:
        _show_simple_modal_message("Export Error", "No data to export. Please load and process data first.")
        return

    if not app_state.loaded_file_path:
        _show_simple_modal_message("Export Error", "Original file path not found. Cannot determine export location.")
        return

    try:
        original_dir = os.path.dirname(app_state.loaded_file_path)
        original_basename = os.path.basename(app_state.loaded_file_path)
        original_filename_no_ext, original_ext = os.path.splitext(original_basename)
        
        export_filename = f"{original_filename_no_ext}{export_suffix}.parquet"
        export_path = os.path.join(original_dir, export_filename)
        
        df_to_export.to_parquet(export_path, index=False)
        _show_simple_modal_message("Export Successful", f"Processed data has been exported to:\n{export_path}")
        print(f"Data exported to {export_path}")
    except Exception as e:
        _show_simple_modal_message("Export Error", f"Failed to export data to Parquet.\nError: {e}")
        print(f"Error exporting data: {e}"); traceback.print_exc()


def get_settings_filepath(original_data_filepath: str) -> Optional[str]:
    if not original_data_filepath: return None
    filename = hashlib.md5(original_data_filepath.encode('utf-8')).hexdigest() + ".json"
    return os.path.join(SETTINGS_DIR_NAME, filename)

def load_json_settings(settings_filepath: str) -> Optional[dict]:
    if settings_filepath and os.path.exists(settings_filepath):
        try:
            with open(settings_filepath, 'r') as f: return json.load(f)
        except Exception as e: print(f"Error loading settings '{settings_filepath}': {e}"); traceback.print_exc()
    return None

def save_json_settings(settings_filepath: str, settings_dict: dict):
    if not settings_filepath or not settings_dict: return
    try:
        if not os.path.exists(SETTINGS_DIR_NAME): os.makedirs(SETTINGS_DIR_NAME)
        with open(settings_filepath, 'w') as f: json.dump(settings_dict, f, indent=4)
        print(f"Settings saved to {settings_filepath}")
    except Exception as e: print(f"Error saving settings to '{settings_filepath}': {e}"); traceback.print_exc()

def gather_current_settings() -> dict:
    settings = {
        'selected_target_variable': app_state.selected_target_variable,
        'selected_target_variable_type': app_state.selected_target_variable_type,
        'active_step_name': app_state.active_step_name,
        'step_01_settings': {},
        'step_02a_sva_settings': {}, 
        'step_02b_mva_settings': {}, 
        'step_03_preprocessing_settings': {}, 
    }
    if hasattr(step_01_data_loading, 'get_step1_settings_for_saving'):
        settings['step_01_settings'] = step_01_data_loading.get_step1_settings_for_saving()
    
    if hasattr(step_02a_sva, 'get_sva_settings_for_saving'):
        settings['step_02a_sva_settings'] = step_02a_sva.get_sva_settings_for_saving()
    
    if hasattr(step_02b_mva, 'get_mva_settings_for_saving'):
        settings['step_02b_mva_settings'] = step_02b_mva.get_mva_settings_for_saving()
    
    # --- Step 03 설정 저장 조건 처리 ---
    if STEP_03_SAVE_LOAD_ENABLED:
        if hasattr(step_03_preprocessing, 'get_preprocessing_settings_for_saving'):
            settings['step_03_preprocessing_settings'] = step_03_preprocessing.get_preprocessing_settings_for_saving()
    else:
        # STEP_03_SAVE_LOAD_ENABLED가 False이면 빈 딕셔너리를 저장하거나,
        # 이 키 자체를 저장하지 않을 수도 있습니다. 여기서는 빈 딕셔너리로 저장합니다.
        settings['step_03_preprocessing_settings'] = {}
    
    return settings

def apply_settings(settings_dict: dict):
    if app_state.original_df is None: 
        _show_simple_modal_message("Error", "Cannot apply settings without data.")
        return
    app_state.active_settings = settings_dict

    app_state.selected_target_variable = settings_dict.get('selected_target_variable')
    app_state.selected_target_variable_type = settings_dict.get('selected_target_variable_type', "Continuous")
    
    s01_settings = settings_dict.get('step_01_settings', {})
    if hasattr(step_01_data_loading, 'apply_step1_settings_and_process'):
        step_01_data_loading.apply_step1_settings_and_process(app_state.original_df, s01_settings, main_app_callbacks)
    else: 
        app_state.df_after_step1 = app_state.original_df.copy() 
        app_state.current_df = app_state.df_after_step1.copy()
        main_app_callbacks['step1_processing_complete'](app_state.df_after_step1) 

    update_target_variable_combo() 
    if app_state.selected_target_variable and \
       app_state.df_after_step1 is not None and \
       app_state.selected_target_variable in app_state.df_after_step1.columns:
        
        s1_types_loaded = s01_settings.get('type_selections', {}) 
        guessed_type_on_load = utils._guess_target_type(app_state.df_after_step1, app_state.selected_target_variable, s1_types_loaded)
        final_target_type = app_state.selected_target_variable_type if app_state.selected_target_variable_type else guessed_type_on_load
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
            if not settings_dict.get('selected_target_variable'):
                 dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, "Continuous")

    s02a_settings = settings_dict.get('step_02a_sva_settings', {})
    if hasattr(step_02a_sva, 'apply_sva_settings_from_loaded') and app_state.df_after_step1 is not None:
        step_02a_sva.apply_sva_settings_from_loaded(s02a_settings, app_state.df_after_step1, main_app_callbacks)
        
    s02b_settings = settings_dict.get('step_02b_mva_settings', {})
    if hasattr(step_02b_mva, 'apply_mva_settings_from_loaded') and app_state.df_after_step1 is not None:
        step_02b_mva.apply_mva_settings_from_loaded(s02b_settings, app_state.df_after_step1, main_app_callbacks)

    # --- Step 03 설정 적용 조건 처리 ---
    if STEP_03_SAVE_LOAD_ENABLED:
        s03_settings = settings_dict.get('step_03_preprocessing_settings', {})
        # 저장된 설정이 비어있지 않은 경우에만 적용 시도
        if s03_settings and hasattr(step_03_preprocessing, 'apply_preprocessing_settings_and_process') and app_state.df_after_step1 is not None:
            step_03_preprocessing.apply_preprocessing_settings_and_process(app_state.df_after_step1, s03_settings, main_app_callbacks)
        else: 
            app_state.df_after_step3 = None 
            # STEP_03_SAVE_LOAD_ENABLED가 True이지만 저장된 설정이 없거나 적용할 수 없는 경우,
            # Step 03 UI는 리셋하거나 현재 상태를 유지할 수 있습니다. 여기서는 리셋합니다.
            if hasattr(step_03_preprocessing, 'reset_preprocessing_state'):
                step_03_preprocessing.reset_preprocessing_state()
            # current_df는 step1 결과로 유지됩니다.
    else:
        # STEP_03_SAVE_LOAD_ENABLED가 False이면, Step 03 관련 상태를 로드하지 않고 UI를 초기화합니다.
        app_state.df_after_step3 = None 
        if hasattr(step_03_preprocessing, 'reset_preprocessing_state'):
            step_03_preprocessing.reset_preprocessing_state()
        # current_df는 step1 결과로 유지됩니다.


    trigger_all_module_updates() 
    
    restored_active_step = settings_dict.get('active_step_name', ANALYSIS_STEPS[0] if ANALYSIS_STEPS else None)
    if restored_active_step and restored_active_step in app_state.step_group_tags and \
       dpg.does_item_exist(app_state.step_group_tags[restored_active_step]):
        switch_step_view(None, None, restored_active_step)
    elif ANALYSIS_STEPS: 
        switch_step_view(None, None, ANALYSIS_STEPS[0])


def initial_load_on_startup():
    session_info = load_json_settings(SESSION_INFO_FILE)
    last_file_path = session_info.get('last_opened_original_file') if session_info else None
    success_loading_session = False

    if last_file_path and os.path.exists(last_file_path):
        try:
            if not load_data_from_file(last_file_path):
                raise Exception("Failed to load data from file.") 

            settings_for_file_path = get_settings_filepath(last_file_path)
            specific_file_settings = load_json_settings(settings_for_file_path)
            
            if specific_file_settings:
                apply_settings(specific_file_settings) 
            else: 
                app_state.active_settings = {}
                if hasattr(step_01_data_loading, 'process_newly_loaded_data') and app_state.original_df is not None:
                    step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks)
                if ANALYSIS_STEPS and ANALYSIS_STEPS[0] in app_state.step_group_tags and \
                   dpg.does_item_exist(app_state.step_group_tags[ANALYSIS_STEPS[0]]):
                    switch_step_view(None, None, ANALYSIS_STEPS[0])
                else: 
                    app_state.active_step_name = ANALYSIS_STEPS[0] if ANALYSIS_STEPS else None

            trigger_all_module_updates() 
            
            active_step_to_set = app_state.active_settings.get('active_step_name', ANALYSIS_STEPS[0] if ANALYSIS_STEPS else None)
            if active_step_to_set and active_step_to_set in app_state.step_group_tags and \
               dpg.does_item_exist(app_state.step_group_tags[active_step_to_set]):
                 switch_step_view(None, None, active_step_to_set)
            elif ANALYSIS_STEPS and ANALYSIS_STEPS[0] in app_state.step_group_tags and \
                 dpg.does_item_exist(app_state.step_group_tags[ANALYSIS_STEPS[0]]): 
                 switch_step_view(None, None, ANALYSIS_STEPS[0])

            success_loading_session = True
            return True 
        except Exception as e:
            _show_simple_modal_message("Session Restore Error",f"Could not restore: {os.path.basename(last_file_path)}\n{e}")
    
    if not success_loading_session:
        reset_application_state(clear_df_completely=True) 
    return False


def save_state_on_exit():
    if app_state.loaded_file_path and os.path.exists(app_state.loaded_file_path):
        current_settings = gather_current_settings()
        settings_filepath = get_settings_filepath(app_state.loaded_file_path)
        save_json_settings(settings_filepath, current_settings)
        save_json_settings(SESSION_INFO_FILE, {'last_opened_original_file': app_state.loaded_file_path})

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
}

main_app_callbacks = {
    'get_current_df': lambda: app_state.current_df,
    'get_original_df': lambda: app_state.original_df,
    'get_df_after_step1': lambda: app_state.df_after_step1,
    'get_df_after_step3': lambda: app_state.df_after_step3,
    'get_df_after_step4': lambda: app_state.df_after_step4,
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
    'step3_processing_complete': step3_processing_complete,
    'step4_missing_value_processing_complete': step4_missing_value_processing_complete, # <<< [추가]

}

dpg.create_context()
with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback, id="file_dialog_id", width=700, height=400, modal=True):
    dpg.add_file_extension(".parquet")
    dpg.add_file_extension(".csv")
    dpg.add_file_extension(".*")

setup_korean_font()

with dpg.window(label="Data Analysis Platform", tag="main_window"):
    with dpg.group(horizontal=True):
        dpg.add_button(label="Open File", callback=lambda: dpg.show_item("file_dialog_id"), width=130, height=30)
        dpg.add_button(label="Reset All (Back to Step1)", user_data=main_app_callbacks, width=200, height=30,
                       callback=lambda s, a, u: u['reset_current_df_to_original']()) 
        dpg.add_button(label="Export to Parquet", callback=export_to_parquet_callback, width=150, height=30)
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
                if step_name_create == ANALYSIS_STEPS[0]: # 1. Data Loading
                    if hasattr(step_01_data_loading, 'create_ui'):
                        step_01_data_loading.create_ui(step_name_create, "content_area", main_app_callbacks)

                elif step_name_create == ANALYSIS_STEPS[1]: # 2. EDA
                    eda_step_group_tag = "eda_step_main_group"
                    main_app_callbacks['register_step_group_tag'](step_name_create, eda_step_group_tag)

                elif step_name_create == ANALYSIS_STEPS[2]: # 3. Preprocessing (Editor)
                    if hasattr(step_03_preprocessing_editor, 'create_ui'): # 모듈명 변경 가정
                        step_03_preprocessing_editor.create_ui(step_name_create, "content_area", main_app_callbacks)
                
                elif step_name_create == ANALYSIS_STEPS[3]: # 4. Missing Value Treatment <<< [새로운 단계 UI 생성]
                    if hasattr(step_04_missing_values, 'create_ui'):
                        step_04_missing_values.create_ui(step_name_create, "content_area", main_app_callbacks)

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
                
                elif step_name_create == ANALYSIS_STEPS[2]: # 3. Preprocessing
                    if hasattr(step_03_preprocessing, 'create_ui'):
                        step_03_preprocessing.create_ui(step_name_create, "content_area", main_app_callbacks)

            if ANALYSIS_STEPS and not app_state.active_step_name:
                first_step = ANALYSIS_STEPS[0]
                if first_step in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step]):
                    switch_step_view(None, None, first_step)

dpg.create_viewport(title='Modular Data Analysis Platform GUI', width=1600, height=1000)
dpg.set_exit_callback(save_state_on_exit)
dpg.setup_dearpygui()

initial_load_on_startup()

dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()