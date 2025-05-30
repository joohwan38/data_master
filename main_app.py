# main_app.py
import dearpygui.dearpygui as dpg
import pandas as pd
import platform
import os
import utils
from typing import Optional
import step_01_data_loading
import step_02a_sva # 수정: SVA 모듈 임포트
import step_02b_mva # 수정: MVA 모듈 임포트
# import step_03_preprocessing # 추후 추가될 모듈 (예시)
import traceback
import hashlib
import json

class AppState:
    def __init__(self):
        self.current_df = None
        self.original_df = None
        self.df_after_step1 = None
        self.loaded_file_path = None
        self.selected_target_variable = None
        self.selected_target_variable_type = "Continuous"
        self.active_step_name = None
        # self._eda_outlier_settings_applied_once = False # 삭제: Outlier 기능 제거
        self.active_settings = {} # 현재 로드된 파일에 대한 전체 설정
        self.step_group_tags = {}
        self.module_ui_updaters = {}

app_state = AppState()

SETTINGS_DIR_NAME = ".app_file_settings"
SESSION_INFO_FILE = os.path.join(SETTINGS_DIR_NAME, "session_info.json")

# --- UI 태그 정의 (main_app 용도) ---
SVA_STEP_KEY = "2a. Single Variable Analysis (SVA)" # 이전 스텝 이름 유지
MVA_STEP_KEY = "2b. Multivariate Analysis (MVA)" # 이전 스텝 이름 유지
TARGET_VARIABLE_TYPE_RADIO_TAG = "main_target_variable_type_radio"
TARGET_VARIABLE_TYPE_LABEL_TAG = "main_target_variable_type_label"
TARGET_VARIABLE_COMBO_TAG = "main_target_variable_combo"
MAIN_FILE_PATH_DISPLAY_TAG = "main_file_path_display_text" # 파일 경로 표시용 새 태그

# 수정: ANALYSIS_STEPS 업데이트
ANALYSIS_STEPS = [
    "1. Data Loading and Overview",
    "2. Exploratory Data Analysis (EDA)",
    # "3. Data Preprocessing & Feature Engineering", # 추후 추가될 스텝 (예시)
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
            # 아이템 x축 간격을 8.0으로 고정 (사용자 제안 반영)
            item_spacing_x = 8.0  
            
            # spacer_w 계산 시 고정된 item_spacing_x 사용
            spacer_w = (width - 100 - (item_spacing_x * 2)) // 2
            
            dpg.add_spacer(width=max(0, int(spacer_w))) # spacer_w가 float일 수 있으므로 int로 변환
            dpg.add_button(label="OK", width=100, callback=lambda: dpg.configure_item(_MODAL_ID_SIMPLE_MESSAGE, show=False))

def setup_korean_font():
    font_path, font_size, os_type = None, 17, platform.system()
    font_paths = {
        "Darwin": ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"],
        "Windows": ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc"],
        "Linux": ["NanumGothic.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"] # Bundled font 우선
    }
    for p in font_paths.get(os_type, []):
        if os.path.exists(p): font_path = p; break
    
    if font_path:
        try:
            with dpg.font_registry():
                dpg.add_font(font_path, font_size, tag="korean_font_for_app")
                dpg.bind_font("korean_font_for_app") # 기본 폰트로 바인딩
            print(f"Font bound: {font_path}")
        except Exception as e: print(f"Font error: {e}"); traceback.print_exc()
    else: print("Korean font not found. Using default.")


def update_target_variable_combo():
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        items = [""] + list(app_state.current_df.columns) if app_state.current_df is not None and not app_state.current_df.empty else [""]
        dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=items)
        current_val = app_state.selected_target_variable if app_state.selected_target_variable and app_state.current_df is not None and app_state.selected_target_variable in app_state.current_df.columns else ""
        dpg.set_value(TARGET_VARIABLE_COMBO_TAG, current_val)

def step1_processing_complete(processed_df: pd.DataFrame):
    if processed_df is None: print("Step 1 returned no DataFrame."); return
    app_state.df_after_step1 = processed_df.copy()
    app_state.current_df = app_state.df_after_step1.copy() # EDA용 데이터는 Step1 완료된 데이터로 시작
    
    if app_state.selected_target_variable and (app_state.current_df is None or app_state.selected_target_variable not in app_state.current_df.columns):
        app_state.selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
    
    update_target_variable_combo()
    trigger_all_module_updates() 

def trigger_specific_module_update(module_name_key: str):
    if module_name_key == ANALYSIS_STEPS[1]: 
        if SVA_STEP_KEY in app_state.module_ui_updaters:
            sva_updater = app_state.module_ui_updaters[SVA_STEP_KEY]
            if hasattr(step_02a_sva, 'update_ui'): 
                sva_updater(app_state.current_df, main_app_callbacks)
                print(f"Module UI updated for: '{SVA_STEP_KEY}' (as part of EDA)")

        if MVA_STEP_KEY in app_state.module_ui_updaters:
            mva_updater = app_state.module_ui_updaters[MVA_STEP_KEY]
            if hasattr(step_02b_mva, 'update_ui'): 
                mva_updater(app_state.current_df, main_app_callbacks)
                print(f"Module UI updated for: '{MVA_STEP_KEY}' (as part of EDA)")
        return 

    if module_name_key in app_state.module_ui_updaters:
        updater = app_state.module_ui_updaters[module_name_key]
        if module_name_key == ANALYSIS_STEPS[0]: 
            if hasattr(step_01_data_loading, 'update_ui'):
                updater(app_state.current_df, app_state.original_df, util_functions_for_modules, app_state.loaded_file_path)
        else: 
            print(f"Updater for '{module_name_key}' called with default signature (df, callbacks).")
        print(f"Module UI updated for: '{module_name_key}'")

def trigger_all_module_updates():
    print("Updating all module UIs...")
    for step_key in list(app_state.module_ui_updaters.keys()): 
        trigger_specific_module_update(step_key)

def load_data_from_file(file_path: str) -> bool:
    success = False
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".parquet":
            app_state.original_df = pd.read_parquet(file_path)
        elif file_extension == ".csv":
            # CSV 파일 로드 시 인코딩 문제 발생 가능성 있음, 우선 utf-8 시도
            try:
                app_state.original_df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                try:
                    app_state.original_df = pd.read_csv(file_path, encoding='cp949') # 한국어 환경 고려
                except Exception as e_csv:
                    raise Exception(f"CSV encoding error: {e_csv}") # 원본 예외 포함
            except Exception as e_csv_other:
                 raise Exception(f"Error reading CSV: {e_csv_other}")
        else:
            _show_simple_modal_message("File Type Error", f"Unsupported file type: {file_extension}\nPlease select a .parquet or .csv file.")
            return False
            
        app_state.current_df = None 
        app_state.df_after_step1 = None
        app_state.loaded_file_path = file_path
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG):
            dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, f"File: {file_path} (Shape: {app_state.original_df.shape})")
        success = True
    except Exception as e:
        app_state.current_df = None; app_state.original_df = None; app_state.loaded_file_path = None; app_state.df_after_step1 = None
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG):
            dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, f"Error loading file: {os.path.basename(file_path)}") # basename 사용
        _show_simple_modal_message("File Load Error", f"Failed to load data from '{os.path.basename(file_path)}'.\nError: {e}")
        print(f"Error loading raw data: {e}"); traceback.print_exc()
        success = False

    if success:
        if hasattr(step_01_data_loading, '_type_selections'): step_01_data_loading._type_selections.clear()
        if hasattr(step_01_data_loading, '_imputation_selections'): step_01_data_loading._imputation_selections.clear()
        if hasattr(step_01_data_loading, '_custom_nan_input_value'): step_01_data_loading._custom_nan_input_value = ""
        
        if app_state.selected_target_variable and (app_state.original_df is None or app_state.selected_target_variable not in app_state.original_df.columns):
            app_state.selected_target_variable = None
            if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
        
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

    if new_type == "Continuous" and app_state.selected_target_variable and app_state.current_df is not None:
        s1_col_types = main_app_callbacks.get('get_column_analysis_types', lambda: {})()
        analysis_type_s1 = s1_col_types.get(app_state.selected_target_variable)

        block_continuous_selection = False # Continuous 선택을 막아야 하는지 여부

        if analysis_type_s1:  # Step 1에서 정의된 분석 타입이 있는 경우
            if analysis_type_s1.startswith("Categorical") or \
               analysis_type_s1.startswith("Text (") or \
               analysis_type_s1.startswith("Potentially Sensitive") or \
               analysis_type_s1 == "Datetime" or \
               analysis_type_s1 == "Timedelta" or \
               "Numeric (Binary" in analysis_type_s1:  # "Numeric (Binary)" 또는 "Numeric (Binary from Text)" 포함
                block_continuous_selection = True
        else:  # Step 1 분석 타입 정보가 없는 경우 (예: _type_selections에 해당 변수 정보가 아직 없을 때)
               # current_df의 pandas dtype으로 판단
            if app_state.selected_target_variable in app_state.current_df.columns:
                dtype_in_current_df = app_state.current_df[app_state.selected_target_variable].dtype
                if pd.api.types.is_object_dtype(dtype_in_current_df) or \
                   pd.api.types.is_string_dtype(dtype_in_current_df) or \
                   pd.api.types.is_categorical_dtype(dtype_in_current_df) or \
                   pd.api.types.is_bool_dtype(dtype_in_current_df): # boolean 타입도 연속형으로 보기 어려움
                    block_continuous_selection = True
        
        if block_continuous_selection:
            var_name = app_state.selected_target_variable
            current_s1_type_msg = f"(Step1 type: {analysis_type_s1})" if analysis_type_s1 else "(Step1 type: N/A)"
            err_msg = f"Variable '{var_name}' {current_s1_type_msg} is not suitable for 'Continuous' type.\n" \
                      f"Please use 'Categorical' or verify its type in Step 1."
            _show_simple_modal_message("Type Selection Warning", err_msg, width=500, height=220) # 메시지 창 크기 약간 조정
            
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, prev_valid_type) # UI를 이전 값으로 되돌림
            return # app_state.selected_target_variable_type 업데이트 방지
            
    app_state.selected_target_variable_type = new_type
    if app_state.active_settings: app_state.active_settings['selected_target_variable_type'] = new_type
    
    if len(ANALYSIS_STEPS) > 1 and app_state.active_step_name == ANALYSIS_STEPS[1]: # "2. Exploratory Data Analysis (EDA)"
        trigger_specific_module_update(ANALYSIS_STEPS[1]) 


def target_variable_selected_callback(sender, app_data, user_data):
    new_target = app_data
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
        if app_state.current_df is not None and new_target in app_state.current_df.columns:
            s1_types = main_app_callbacks.get('get_column_analysis_types', lambda: {})()
            guessed_type = utils._guess_target_type(app_state.current_df, new_target, s1_types)
            app_state.selected_target_variable_type = guessed_type
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, guessed_type)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)
            if app_state.active_settings: app_state.active_settings['selected_target_variable'] = new_target; app_state.active_settings['selected_target_variable_type'] = guessed_type
    trigger_all_module_updates()


def switch_step_view(sender, app_data, user_data_step_name: str):
    for step_name_iter, group_tag in app_state.step_group_tags.items():
        if dpg.does_item_exist(group_tag):
            is_active_step = (step_name_iter == user_data_step_name)
            dpg.configure_item(group_tag, show=is_active_step)
            if is_active_step:
                app_state.active_step_name = step_name_iter
                trigger_specific_module_update(step_name_iter)

def file_load_callback(sender, app_data):
    new_file_path = app_data.get('file_path_name')
    if not new_file_path: return

    if app_state.loaded_file_path and app_state.active_settings: 
        old_settings_path = get_settings_filepath(app_state.loaded_file_path)
        current_live_settings = gather_current_settings()
        save_json_settings(old_settings_path, current_live_settings)

    reset_application_state(clear_df_completely=False) 
    
    if not load_data_from_file(new_file_path): 
        # _show_simple_modal_message("File Load Error", f"Failed to load {os.path.basename(new_file_path)}") # load_data_from_file에서 이미 메시지 표시
        return

    new_settings_path = get_settings_filepath(app_state.loaded_file_path)
    loaded_specific_settings = load_json_settings(new_settings_path)
    
    if loaded_specific_settings: 
        apply_settings(loaded_specific_settings)
    else: 
        app_state.active_settings = {} 
        if hasattr(step_01_data_loading, 'process_newly_loaded_data'):
            step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks)
        else: 
            trigger_specific_module_update(ANALYSIS_STEPS[0])

    if ANALYSIS_STEPS: switch_step_view(None, None, ANALYSIS_STEPS[0])

def reset_application_state(clear_df_completely=True):
    if clear_df_completely:
        app_state.current_df = None; app_state.original_df = None; app_state.df_after_step1 = None
        app_state.loaded_file_path = None; app_state.active_settings = {}
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG): dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, "No data loaded.")
    else: 
        app_state.current_df = None; app_state.df_after_step1 = None

    app_state.selected_target_variable = None
    app_state.selected_target_variable_type = "Continuous"
    
    if hasattr(step_01_data_loading, 'reset_step1_state'): step_01_data_loading.reset_step1_state()
    
    if hasattr(step_02a_sva, 'reset_sva_ui_defaults'): step_02a_sva.reset_sva_ui_defaults()
    if hasattr(step_02b_mva, 'reset_mva_ui_defaults'): step_02b_mva.reset_mva_ui_defaults()

    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=[""]); dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False); dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, "Continuous")

    if not clear_df_completely and app_state.original_df is not None:
        trigger_specific_module_update(ANALYSIS_STEPS[0])
    else: 
        trigger_all_module_updates()
    
    if ANALYSIS_STEPS:
        current_active = app_state.active_step_name
        first_step = ANALYSIS_STEPS[0]
        if not current_active or current_active not in app_state.step_group_tags or \
           not dpg.does_item_exist(app_state.step_group_tags.get(current_active, "")): 
            if first_step in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step]):
                switch_step_view(None, None, first_step)

# --- 파일 저장 관련 함수 ---
def export_to_parquet_callback():
    if app_state.current_df is None:
        _show_simple_modal_message("Export Error", "No data to export. Please load and process data first.")
        return

    if not app_state.loaded_file_path:
        # 파일명이 없는 경우 (예: 프로그램 시작 직후 데이터를 생성한 경우 - 현재 시나리오에서는 발생 어려움)
        # 우선 간단히 에러 처리. 추후 파일 저장 대화상자 구현 시 이 부분 개선.
        _show_simple_modal_message("Export Error", "Original file path not found. Cannot determine export location.")
        return

    try:
        original_dir = os.path.dirname(app_state.loaded_file_path)
        original_basename = os.path.basename(app_state.loaded_file_path)
        original_filename_no_ext, original_ext = os.path.splitext(original_basename)
        
        # 새 파일명 생성 (예: myfile.parquet -> myfile_processed.parquet)
        # 또는 (예: myfile.csv -> myfile_csv_processed.parquet)
        if original_ext.lower() == ".csv":
             export_filename = f"{original_filename_no_ext}_csv_processed.parquet"
        else:
            export_filename = f"{original_filename_no_ext}_processed.parquet"

        export_path = os.path.join(original_dir, export_filename)
        
        app_state.current_df.to_parquet(export_path, index=False)
        _show_simple_modal_message("Export Successful", f"Processed data has been exported to:\n{export_path}")
        print(f"Data exported to {export_path}")
    except Exception as e:
        _show_simple_modal_message("Export Error", f"Failed to export data to Parquet.\nError: {e}")
        print(f"Error exporting data: {e}"); traceback.print_exc()

# --- 설정 저장/로드 함수 ---
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
    }
    if hasattr(step_01_data_loading, 'get_step1_settings_for_saving'):
        settings['step_01_settings'] = step_01_data_loading.get_step1_settings_for_saving()
    
    if hasattr(step_02a_sva, 'get_sva_settings_for_saving'):
        settings['step_02a_sva_settings'] = step_02a_sva.get_sva_settings_for_saving()
    
    if hasattr(step_02b_mva, 'get_mva_settings_for_saving'):
        settings['step_02b_mva_settings'] = step_02b_mva.get_mva_settings_for_saving()
    
    return settings

def apply_settings(settings_dict: dict):
    if app_state.original_df is None: 
        _show_simple_modal_message("Error", "Cannot apply settings without data.")
        return
    app_state.active_settings = settings_dict # 전체 설정을 active_settings에 저장

    app_state.selected_target_variable = settings_dict.get('selected_target_variable')
    app_state.selected_target_variable_type = settings_dict.get('selected_target_variable_type', "Continuous")
    
    s01_settings = settings_dict.get('step_01_settings', {})
    if hasattr(step_01_data_loading, 'apply_step1_settings_and_process'):
        # 이 함수 내부에서 step1_processing_complete를 호출하여 current_df 등이 설정될 것임
        step_01_data_loading.apply_step1_settings_and_process(app_state.original_df, s01_settings, main_app_callbacks)
    else: # Fallback: Step1 처리 없이 원본을 사용하거나, 최소한의 처리만 가정
        if app_state.df_after_step1 is None: app_state.df_after_step1 = app_state.original_df.copy()
        app_state.current_df = app_state.df_after_step1.copy()

    # --- ▼ [추가된 UI 업데이트 로직] 타겟 변수 타입 UI 상태 복원 ▼ ---
    if app_state.selected_target_variable and \
       app_state.current_df is not None and \
       app_state.selected_target_variable in app_state.current_df.columns:
        
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, app_state.selected_target_variable_type)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)
    else: # 선택된 타겟 변수가 없거나, current_df에 없거나, current_df가 None인 경우
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
            # 설정에서 타겟 변수가 실제로 없는 것으로 로드되었다면 라디오 버튼 값도 기본값으로 리셋
            if not settings_dict.get('selected_target_variable'): # app_state.selected_target_variable이 None일 때
                 dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, "Continuous")
    # --- ▲ [추가된 UI 업데이트 로직] 타겟 변수 타입 UI 상태 복원 ▲ ---

    update_target_variable_combo() # current_df가 설정된 후 호출되어야 하며, 콤보 아이템 및 선택값 업데이트

    # SVA 설정 적용 (step_02a_sva 모듈에 해당 함수가 있다고 가정)
    s02a_settings = settings_dict.get('step_02a_sva_settings', {})
    if hasattr(step_02a_sva, 'apply_sva_settings_from_loaded') and app_state.current_df is not None:
        step_02a_sva.apply_sva_settings_from_loaded(s02a_settings, app_state.current_df, main_app_callbacks)
        
    # MVA 설정 적용 (step_02b_mva 모듈에 해당 함수가 있다고 가정)
    s02b_settings = settings_dict.get('step_02b_mva_settings', {})
    if hasattr(step_02b_mva, 'apply_mva_settings_from_loaded') and app_state.current_df is not None:
        step_02b_mva.apply_mva_settings_from_loaded(s02b_settings, app_state.current_df, main_app_callbacks)

    restored_active_step = settings_dict.get('active_step_name', ANALYSIS_STEPS[0] if ANALYSIS_STEPS else None)
    trigger_all_module_updates() # 모든 모듈 UI 업데이트 (SVA, MVA 포함)
    if restored_active_step and restored_active_step in app_state.step_group_tags and \
       dpg.does_item_exist(app_state.step_group_tags[restored_active_step]):
        switch_step_view(None, None, restored_active_step)



def initial_load_on_startup():
    session_info = load_json_settings(SESSION_INFO_FILE)
    last_file_path = session_info.get('last_opened_original_file') if session_info else None

    if last_file_path and os.path.exists(last_file_path):
        try:
            if not load_data_from_file(last_file_path): 
                reset_application_state(clear_df_completely=True)
                if ANALYSIS_STEPS: switch_step_view(None, None, ANALYSIS_STEPS[0])
                return False

            settings_for_file_path = get_settings_filepath(last_file_path)
            specific_file_settings = load_json_settings(settings_for_file_path)
            
            if specific_file_settings:
                apply_settings(specific_file_settings) 
            else: 
                app_state.active_settings = {}
                if hasattr(step_01_data_loading, 'process_newly_loaded_data'):
                    step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks)
            trigger_all_module_updates() 
            
            if not app_state.active_step_name and ANALYSIS_STEPS:
                 switch_step_view(None, None, ANALYSIS_STEPS[0])
            return True
        except Exception as e:
            _show_simple_modal_message("Session Restore Error",f"Could not restore: {os.path.basename(last_file_path)}\n{e}")
            reset_application_state(clear_df_completely=True)
            if ANALYSIS_STEPS: switch_step_view(None, None, ANALYSIS_STEPS[0])
            return False
    else: 
        reset_application_state(clear_df_completely=True)
        if ANALYSIS_STEPS: switch_step_view(None, None, ANALYSIS_STEPS[0])
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
    'get_loaded_file_path': lambda: app_state.loaded_file_path,
    'get_util_funcs': lambda: util_functions_for_modules,
    'show_file_dialog': lambda: dpg.show_item("file_dialog_id"),
    'register_step_group_tag': lambda name, tag: app_state.step_group_tags.update({name: tag}),
    'register_module_updater': lambda name, func: app_state.module_ui_updaters.update({name: func}),
    'trigger_module_update': trigger_specific_module_update,
    'reset_current_df_to_original': lambda: ( 
        step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks)
        if app_state.original_df is not None and hasattr(step_01_data_loading, 'process_newly_loaded_data')
        else print("Cannot reset: Original DF not loaded or Step1 process func missing.")
    ),
    'trigger_all_module_updates': trigger_all_module_updates,
    'get_selected_target_variable': lambda: app_state.selected_target_variable,
    'get_selected_target_variable_type': lambda: app_state.selected_target_variable_type,
    'update_target_variable_combo_items': update_target_variable_combo,
    'get_column_analysis_types': lambda: step_01_data_loading._type_selections.copy() if hasattr(step_01_data_loading, '_type_selections') else {},
    'step1_processing_complete': step1_processing_complete,
}

dpg.create_context()
with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback, id="file_dialog_id", width=700, height=400, modal=True):
    dpg.add_file_extension(".parquet")
    dpg.add_file_extension(".csv") # .csv 확장자 추가
    dpg.add_file_extension(".*")
setup_korean_font()

with dpg.window(label="Data Analysis Platform", tag="main_window"):
    with dpg.group(horizontal=True):
        dpg.add_button(label="Open File", callback=lambda: dpg.show_item("file_dialog_id"), width=130, height=30) # 라벨 변경 및 너비 조정
        dpg.add_button(label="Reset Original (Step1)", user_data=main_app_callbacks, width=180, height=30, # 라벨 및 너비 조정
                       callback=lambda s, a, u: u['reset_current_df_to_original']()) 
        dpg.add_button(label="Export to Parquet", callback=export_to_parquet_callback, width=150, height=30) # 내보내기 버튼 추가
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
                        dpg.add_text(f"--- {step_name_create} ---")
                        dpg.add_separator()
                        with dpg.tab_bar(tag="eda_tab_bar"):
                            sva_tab_tag = "sva_tab_content_in_eda" 
                            with dpg.tab(label="Single Variable Analysis", tag="eda_sva_tab"):
                                if hasattr(step_02a_sva, 'create_ui'):
                                    with dpg.group(tag=sva_tab_tag): 
                                        step_02a_sva.create_ui(SVA_STEP_KEY, sva_tab_tag, main_app_callbacks)

                            mva_tab_tag = "mva_tab_content_in_eda" 
                            with dpg.tab(label="Multivariate Analysis", tag="eda_mva_tab"):
                                if hasattr(step_02b_mva, 'create_ui'):
                                    with dpg.group(tag=mva_tab_tag): 
                                        step_02b_mva.create_ui(MVA_STEP_KEY, mva_tab_tag, main_app_callbacks)

            if ANALYSIS_STEPS and not app_state.active_step_name:
                first_step = ANALYSIS_STEPS[0]
                if first_step in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step]):
                    switch_step_view(None, None, first_step)

dpg.create_viewport(title='Modular Data Analysis Platform GUI', width=1600, height=1000)
dpg.set_exit_callback(save_state_on_exit)
dpg.setup_dearpygui()
initial_load_on_startup() 
if ANALYSIS_STEPS and app_state.step_group_tags.get(ANALYSIS_STEPS[0]):
    app_state.active_step_name = "" 
    switch_step_view(None, None, ANALYSIS_STEPS[0])
else:
    print("Warning: Could not switch to the first analysis step on startup.")

dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()