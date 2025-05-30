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
            spacer_w = (width - 100 - (dpg.get_style_item_spacing()[0]*2 if dpg.is_dearpygui_running() else 16)) // 2
            dpg.add_spacer(width=max(0, spacer_w))
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
                # 특정 범위 힌트 추가 (선택적)
                # dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean, parent="korean_font_for_app")
                # dpg.add_font_range_hint(dpg.mvFontRangeHint_Default, parent="korean_font_for_app")
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
    
    # 타겟 변수 유효성 검사 및 UI 업데이트
    if app_state.selected_target_variable and (app_state.current_df is None or app_state.selected_target_variable not in app_state.current_df.columns):
        app_state.selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
    
    update_target_variable_combo()
    trigger_all_module_updates() # 모든 모듈 UI 업데이트 (SVA, MVA 등)

def trigger_specific_module_update(module_name_key: str):
    # module_name_key이 "2. Exploratory Data Analysis (EDA)" 로 들어올 경우,
    # SVA와 MVA 업데이터를 모두 호출해야 함.
    if module_name_key == ANALYSIS_STEPS[1]: # "2. Exploratory Data Analysis (EDA)"
        if SVA_STEP_KEY in app_state.module_ui_updaters:
            sva_updater = app_state.module_ui_updaters[SVA_STEP_KEY]
            if hasattr(step_02a_sva, 'update_ui'): # SVA 모듈에 update_ui가 있는지 확인
                sva_updater(app_state.current_df, main_app_callbacks)
                print(f"Module UI updated for: '{SVA_STEP_KEY}' (as part of EDA)")

        if MVA_STEP_KEY in app_state.module_ui_updaters:
            mva_updater = app_state.module_ui_updaters[MVA_STEP_KEY]
            if hasattr(step_02b_mva, 'update_ui'): # MVA 모듈에 update_ui가 있는지 확인
                mva_updater(app_state.current_df, main_app_callbacks)
                print(f"Module UI updated for: '{MVA_STEP_KEY}' (as part of EDA)")
        return # EDA의 경우 여기서 종료

    # 기존 로직 (Data Loading 등 다른 스텝들)
    if module_name_key in app_state.module_ui_updaters:
        updater = app_state.module_ui_updaters[module_name_key]
        if module_name_key == ANALYSIS_STEPS[0]: # Data Loading
            if hasattr(step_01_data_loading, 'update_ui'):
                updater(app_state.current_df, app_state.original_df, util_functions_for_modules, app_state.loaded_file_path)
        else: # 기타 확장 모듈 (필요시)
            print(f"Updater for '{module_name_key}' called with default signature (df, callbacks).")

        print(f"Module UI updated for: '{module_name_key}'")

def trigger_all_module_updates():
    print("Updating all module UIs...")
    for step_key in list(app_state.module_ui_updaters.keys()): # 키 리스트 복사 후 순회
        trigger_specific_module_update(step_key)

def load_data_from_file(file_path: str) -> bool:
    success = False
    try:
        app_state.original_df = pd.read_parquet(file_path)
        app_state.current_df = None # Step 1 처리 전이므로 current_df는 None
        app_state.df_after_step1 = None
        app_state.loaded_file_path = file_path
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG):
            # 수정: os.path.basename(file_path) 대신 file_path 전체를 사용
            dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, f"File: {file_path} (Shape: {app_state.original_df.shape})")
        success = True
    except Exception as e:
        app_state.current_df = None; app_state.original_df = None; app_state.loaded_file_path = None; app_state.df_after_step1 = None
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG):
            # 수정: 오류 메시지에도 전체 파일 경로 또는 파일명 포함 가능
            dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, f"Error loading file: {file_path}")
        print(f"Error loading raw data: {e}"); traceback.print_exc()
        success = False

    if success:
        # Step 1 관련 상태 초기화 (step_01_data_loading 모듈 내 변수들)
        if hasattr(step_01_data_loading, '_type_selections'): step_01_data_loading._type_selections.clear()
        if hasattr(step_01_data_loading, '_imputation_selections'): step_01_data_loading._imputation_selections.clear()
        if hasattr(step_01_data_loading, '_custom_nan_input_value'): step_01_data_loading._custom_nan_input_value = ""
        
        # 타겟 변수 유효성 검사 (original_df 기준)
        if app_state.selected_target_variable and (app_state.original_df is None or app_state.selected_target_variable not in app_state.original_df.columns):
            app_state.selected_target_variable = None
            if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
        
        # Step 1 UI 업데이트 트리거
        if ANALYSIS_STEPS and ANALYSIS_STEPS[0] in app_state.module_ui_updaters:
             trigger_specific_module_update(ANALYSIS_STEPS[0]) # current_df가 None인 상태로 전달됨
        return True
    else: # 로드 실패 시
        reset_application_state(clear_df_completely=True) # 모든 데이터 상태 초기화
        trigger_all_module_updates() # 모든 모듈 UI 업데이트 (초기 상태로)
        return False

def target_variable_type_changed_callback(sender, app_data, user_data):
    new_type = app_data
    prev_valid_type = app_state.selected_target_variable_type # 현재 app_state에 저장된 유효한 타입

    if new_type == "Continuous" and app_state.selected_target_variable and app_state.current_df is not None:
        s1_col_types = main_app_callbacks.get('get_column_analysis_types', lambda: {})()
        analysis_type_s1 = s1_col_types.get(app_state.selected_target_variable)
        is_text_based = False
        if analysis_type_s1 and any(k in analysis_type_s1 for k in ["Text (", "Potentially Sensitive", "Categorical (High Cardinality)"]):
            is_text_based = True
        elif app_state.selected_target_variable in app_state.current_df.columns and \
             (pd.api.types.is_object_dtype(app_state.current_df[app_state.selected_target_variable].dtype) or \
              pd.api.types.is_string_dtype(app_state.current_df[app_state.selected_target_variable].dtype)):
            if analysis_type_s1 is None or not any(k in analysis_type_s1 for k in ["Numeric", "Datetime", "Timedelta"]):
                is_text_based = True
        if is_text_based:
            err_msg = f"Var '{app_state.selected_target_variable}' is Text-based/High Cardinality.\nCannot be reliably 'Continuous'.\nUse 'Categorical' or verify in Step 1."
            _show_simple_modal_message("Type Selection Warning", err_msg)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, prev_valid_type) # UI를 이전 값으로 되돌림
            return # app_state.selected_target_variable_type 업데이트 방지
            
    app_state.selected_target_variable_type = new_type
    if app_state.active_settings: app_state.active_settings['selected_target_variable_type'] = new_type
    
    # 현재 활성화된 SVA 또는 MVA 탭 UI 업데이트
    if app_state.active_step_name == ANALYSIS_STEPS[1]: trigger_specific_module_update(ANALYSIS_STEPS[1]) # SVA
    elif app_state.active_step_name == ANALYSIS_STEPS[2]: trigger_specific_module_update(ANALYSIS_STEPS[2]) # MVA


def target_variable_selected_callback(sender, app_data, user_data):
    new_target = app_data
    if not new_target: # 타겟 선택 해제
        app_state.selected_target_variable = None
        app_state.selected_target_variable_type = "Continuous" # 기본값으로 리셋
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, "Continuous")
        if app_state.active_settings: app_state.active_settings['selected_target_variable'] = None; app_state.active_settings['selected_target_variable_type'] = None
    else: # 새 타겟 선택
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

    if app_state.loaded_file_path and app_state.active_settings: # 기존 파일 설정 저장
        old_settings_path = get_settings_filepath(app_state.loaded_file_path)
        current_live_settings = gather_current_settings()
        save_json_settings(old_settings_path, current_live_settings)

    reset_application_state(clear_df_completely=False) # current_df, df_after_step1 등은 초기화, original_df는 유지될 수 있으나 load_data_from_file에서 덮어씀
    
    if not load_data_from_file(new_file_path): # 새 파일 로드 (실패 시 메시지 표시됨)
        _show_simple_modal_message("File Load Error", f"Failed to load {os.path.basename(new_file_path)}")
        return

    new_settings_path = get_settings_filepath(app_state.loaded_file_path)
    loaded_specific_settings = load_json_settings(new_settings_path)
    
    if loaded_specific_settings: # 저장된 설정 적용
        apply_settings(loaded_specific_settings)
    else: # 새 파일이거나 설정 없을 시
        app_state.active_settings = {} # 활성 설정 초기화
        if hasattr(step_01_data_loading, 'process_newly_loaded_data'):
            # Step 1 모듈에 원본 데이터를 전달하여 기본 처리 실행 (타입 추론 등)
            step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks)
        else: # process_newly_loaded_data가 없다면 Step 1 UI만 업데이트
            trigger_specific_module_update(ANALYSIS_STEPS[0])

    if ANALYSIS_STEPS: switch_step_view(None, None, ANALYSIS_STEPS[0]) # 첫 번째 스텝으로 자동 전환

def reset_application_state(clear_df_completely=True):
    if clear_df_completely:
        app_state.current_df = None; app_state.original_df = None; app_state.df_after_step1 = None
        app_state.loaded_file_path = None; app_state.active_settings = {}
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG): dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, "No data loaded.") # 이 부분은 그대로 두거나 "File: None" 등으로 변경 가능
    else: # 파일 변경 시, original_df는 새 파일로 교체되므로 current_df와 df_after_step1만 초기화
        app_state.current_df = None; app_state.df_after_step1 = None

    app_state.selected_target_variable = None
    app_state.selected_target_variable_type = "Continuous"
    
    # Step 1 상태 초기화
    if hasattr(step_01_data_loading, 'reset_step1_state'): step_01_data_loading.reset_step1_state()
    
    # SVA, MVA UI 초기화 (각 모듈에 reset 함수가 있다면 호출)
    if hasattr(step_02a_sva, 'reset_sva_ui_defaults'): step_02a_sva.reset_sva_ui_defaults()
    if hasattr(step_02b_mva, 'reset_mva_ui_defaults'): step_02b_mva.reset_mva_ui_defaults()
    # Outlier 관련 UI 리셋은 Outlier 기능 제거로 불필요

    # 타겟 변수 UI 초기화
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=[""]); dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False); dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, "Continuous")

    # clear_df_completely=False는 파일 로드 시 사용. 이때는 original_df가 있으므로 Step1 UI 업데이트.
    if not clear_df_completely and app_state.original_df is not None:
        trigger_specific_module_update(ANALYSIS_STEPS[0])
    else: # 완전 초기화 시 모든 모듈 업데이트
        trigger_all_module_updates()
    
    # 현재 활성 스텝이 유효하지 않으면 첫 스텝으로 전환
    if ANALYSIS_STEPS:
        current_active = app_state.active_step_name
        first_step = ANALYSIS_STEPS[0]
        if not current_active or current_active not in app_state.step_group_tags or \
           not dpg.does_item_exist(app_state.step_group_tags.get(current_active, "")): # 기본값으로 빈 문자열
            if first_step in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step]):
                switch_step_view(None, None, first_step)


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
        'step_02a_sva_settings': {}, # SVA 설정용 (추후 각 모듈에 getter 구현)
        'step_02b_mva_settings': {}, # MVA 설정용 (추후 각 모듈에 getter 구현)
        # 'step_03_preprocessing_settings': {}, # 추후 추가될 스텝 설정
    }
    if hasattr(step_01_data_loading, 'get_step1_settings_for_saving'):
        settings['step_01_settings'] = step_01_data_loading.get_step1_settings_for_saving()
    
    # SVA 설정 가져오기 (step_02a_sva 모듈에 해당 함수가 있다고 가정)
    if hasattr(step_02a_sva, 'get_sva_settings_for_saving'):
        settings['step_02a_sva_settings'] = step_02a_sva.get_sva_settings_for_saving()
    
    # MVA 설정 가져오기 (step_02b_mva 모듈에 해당 함수가 있다고 가정)
    if hasattr(step_02b_mva, 'get_mva_settings_for_saving'):
        settings['step_02b_mva_settings'] = step_02b_mva.get_mva_settings_for_saving()
    
    # Outlier 설정 수집 부분은 삭제됨
    return settings

def apply_settings(settings_dict: dict):
    if app_state.original_df is None: _show_simple_modal_message("Error", "Cannot apply settings without data."); return
    app_state.active_settings = settings_dict # 전체 설정을 active_settings에 저장

    app_state.selected_target_variable = settings_dict.get('selected_target_variable')
    app_state.selected_target_variable_type = settings_dict.get('selected_target_variable_type', "Continuous")
    
    s01_settings = settings_dict.get('step_01_settings', {})
    if hasattr(step_01_data_loading, 'apply_step1_settings_and_process'):
        step_01_data_loading.apply_step1_settings_and_process(app_state.original_df, s01_settings, main_app_callbacks)
        # apply_step1_settings_and_process 내부에서 step1_processing_complete를 호출하여 current_df 등이 설정될 것임
    else: # Fallback: Step1 처리 없이 원본을 사용하거나, 최소한의 처리만 가정
        if app_state.df_after_step1 is None: app_state.df_after_step1 = app_state.original_df.copy()
        app_state.current_df = app_state.df_after_step1.copy()

    update_target_variable_combo() # current_df가 설정된 후 호출되어야 함

    # SVA 설정 적용 (step_02a_sva 모듈에 해당 함수가 있다고 가정)
    s02a_settings = settings_dict.get('step_02a_sva_settings', {})
    if hasattr(step_02a_sva, 'apply_sva_settings_from_loaded') and app_state.current_df is not None:
        step_02a_sva.apply_sva_settings_from_loaded(s02a_settings, app_state.current_df, main_app_callbacks)
        
    # MVA 설정 적용 (step_02b_mva 모듈에 해당 함수가 있다고 가정)
    s02b_settings = settings_dict.get('step_02b_mva_settings', {})
    if hasattr(step_02b_mva, 'apply_mva_settings_from_loaded') and app_state.current_df is not None:
        step_02b_mva.apply_mva_settings_from_loaded(s02b_settings, app_state.current_df, main_app_callbacks)

    # Outlier 설정 적용 부분은 삭제됨

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
            # load_data_from_file 함수 내부에서 MAIN_FILE_PATH_DISPLAY_TAG가 업데이트 됨
            if not load_data_from_file(last_file_path): # 파일 로드 실패 시
                reset_application_state(clear_df_completely=True)
                if ANALYSIS_STEPS: switch_step_view(None, None, ANALYSIS_STEPS[0])
                return False

            settings_for_file_path = get_settings_filepath(last_file_path)
            specific_file_settings = load_json_settings(settings_for_file_path)
            
            if specific_file_settings:
                apply_settings(specific_file_settings) # 여기서 step1 처리 및 current_df 설정됨
            else: # 설정 없을 시 기본 처리
                app_state.active_settings = {}
                if hasattr(step_01_data_loading, 'process_newly_loaded_data'):
                    step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks)
                # else: trigger_specific_module_update(ANALYSIS_STEPS[0]) # process_newly_loaded_data가 step1_processing_complete를 호출할 것이므로 중복 호출 방지

            # apply_settings 또는 process_newly_loaded_data 이후 current_df가 설정되었으므로,
            # 타겟 변수 UI가 올바르게 업데이트되도록 한 번 더 호출.
            # (apply_settings 내부에서 update_target_variable_combo가 호출되지만, SVA/MVA update_ui도 필요)
            trigger_all_module_updates() 
            
            # active_step_name은 apply_settings에서 복원될 수 있음.
            # 만약 복원되지 않았다면 첫 번째 스텝으로.
            if not app_state.active_step_name and ANALYSIS_STEPS:
                 switch_step_view(None, None, ANALYSIS_STEPS[0])
            return True
        except Exception as e:
            _show_simple_modal_message("Session Restore Error",f"Could not restore: {os.path.basename(last_file_path)}\n{e}")
            reset_application_state(clear_df_completely=True)
            if ANALYSIS_STEPS: switch_step_view(None, None, ANALYSIS_STEPS[0])
            return False
    else: # 마지막 세션 없음
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
    '_show_simple_modal_message': _show_simple_modal_message, # main_app의 함수를 전달
    'show_dpg_alert_modal': utils.show_dpg_alert_modal, # utils의 함수 전달
    'get_numeric_cols': utils._get_numeric_cols, # utils의 함수 전달
    'get_categorical_cols': utils._get_categorical_cols, # utils의 함수 전달
    'calculate_cramers_v': utils.calculate_cramers_v, # utils의 함수 전달
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
    # 'reset_current_df_to_original' -> reset_application_state(clear_df_completely=False) 와 유사하나, Step1 처리 재실행 여부가 다름.
    # reset_application_state(False)는 Step1 UI를 초기화하고 original_df로 Step1을 다시 트리거하는 효과.
    # 만약 단순히 current_df를 df_after_step1로 되돌리는 기능이 필요하면 별도 콜백 필요.
    # 여기서는 파일 재로드 시 사용되는 reset_application_state(False)를 사용하거나,
    # 또는 Step1의 "Reset to Original Data" 버튼의 콜백에서 step1_processing_complete(app_state.original_df.copy())를 직접 호출하는 방식도 고려 가능.
    # 현재는 reset_application_state(False)가 Step 1의 UI 상태를 초기화하므로, 기존 버튼의 의도와 약간 다를 수 있음.
    # Step 1 모듈에서 'original_df로 Step1 재처리' 하는 콜백을 직접 호출하는 것이 더 명확할 수 있음.
    # 여기서는 기존 콜백명을 유지하고, reset_application_state(False)는 파일 로드 시 사용.
    # Step1의 "Reset to Original Data" 버튼 콜백은 step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks) 호출로 변경하는 것이 적절.
    'reset_current_df_to_original': lambda: ( # Step1의 "Reset to Original Data" 버튼용
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
    # Outlier 관련 콜백 삭제
}

dpg.create_context()
with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback, id="file_dialog_id", width=700, height=400, modal=True):
    dpg.add_file_extension(".parquet"); dpg.add_file_extension(".*")
setup_korean_font()

with dpg.window(label="Data Analysis Platform", tag="main_window"):
    # --- 파일 로드/리셋 및 경로 표시 UI (최상단) ---
    with dpg.group(horizontal=True):
        dpg.add_button(label="Open Parquet File", callback=lambda: dpg.show_item("file_dialog_id"), width=150, height=30)
        dpg.add_button(label="Reset Original Data (Step1)", user_data=main_app_callbacks, width=200, height=30,
                       callback=lambda s, a, u: u['reset_current_df_to_original']()) # 수정된 콜백 사용
        dpg.add_text("No data loaded.", tag=MAIN_FILE_PATH_DISPLAY_TAG, wrap=-1) # wrap=-1로 가로 공간 다 쓰도록
    dpg.add_separator()
    # --- 메인 레이아웃 (네비게이션 + 콘텐츠) ---
    with dpg.group(horizontal=True):
        with dpg.child_window(width=280, tag="navigation_panel", border=True): # 네비게이션 패널
            dpg.add_text("Target Variable (y):")
            dpg.add_combo(items=[""], tag=TARGET_VARIABLE_COMBO_TAG, width=-1, callback=target_variable_selected_callback)
            dpg.add_text("Target Variable Type:", tag=TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            dpg.add_radio_button(items=["Categorical", "Continuous"], tag=TARGET_VARIABLE_TYPE_RADIO_TAG, horizontal=True,
                                 default_value=app_state.selected_target_variable_type, callback=target_variable_type_changed_callback, show=False)
            dpg.add_separator(); dpg.add_spacer(height=5)
            dpg.add_text("Analysis Steps", color=[255,255,0]); dpg.add_separator()
            for step_name_nav in ANALYSIS_STEPS:
                dpg.add_button(label=step_name_nav, callback=switch_step_view, user_data=step_name_nav, width=-1, height=30)
        
        with dpg.child_window(tag="content_area", border=True): # 콘텐츠 영역

            for step_name_create in ANALYSIS_STEPS:
                if step_name_create == ANALYSIS_STEPS[0]: # "1. Data Loading and Overview"
                    if hasattr(step_01_data_loading, 'create_ui'):
                        step_01_data_loading.create_ui(step_name_create, "content_area", main_app_callbacks)

                elif step_name_create == ANALYSIS_STEPS[1]: # "2. Exploratory Data Analysis (EDA)"
                    eda_step_group_tag = "eda_step_main_group"
                    main_app_callbacks['register_step_group_tag'](step_name_create, eda_step_group_tag) # EDA 스텝 그룹 등록

                    with dpg.group(tag=eda_step_group_tag, parent="content_area", show=False): # 기본적으로 숨김
                        dpg.add_text(f"--- {step_name_create} ---")
                        dpg.add_separator()
                        with dpg.tab_bar(tag="eda_tab_bar"):
                            # SVA 탭
                            sva_tab_tag = "sva_tab_content_in_eda" # SVA 내용이 그려질 부모 태그
                            with dpg.tab(label="Single Variable Analysis", tag="eda_sva_tab"):
                                if hasattr(step_02a_sva, 'create_ui'):
                                    with dpg.group(tag=sva_tab_tag): # SVA UI를 담을 그룹
                                        step_02a_sva.create_ui(SVA_STEP_KEY, sva_tab_tag, main_app_callbacks)

                            # MVA 탭
                            mva_tab_tag = "mva_tab_content_in_eda" # MVA 내용이 그려질 부모 태그
                            with dpg.tab(label="Multivariate Analysis", tag="eda_mva_tab"):
                                if hasattr(step_02b_mva, 'create_ui'):
                                    # MVA 모듈도 SVA와 동일한 방식으로 MVA_STEP_KEY를 사용.
                                    with dpg.group(tag=mva_tab_tag): # MVA UI를 담을 그룹
                                        step_02b_mva.create_ui(MVA_STEP_KEY, mva_tab_tag, main_app_callbacks)

            if ANALYSIS_STEPS and not app_state.active_step_name:
                first_step = ANALYSIS_STEPS[0]
                if first_step in app_state.step_group_tags and dpg.does_item_exist(app_state.step_group_tags[first_step]):
                    switch_step_view(None, None, first_step)

dpg.create_viewport(title='Modular Data Analysis Platform GUI', width=1600, height=1000)
dpg.set_exit_callback(save_state_on_exit)
dpg.setup_dearpygui()
initial_load_on_startup() # 데이터 로드 및 UI 업데이트 시도
if ANALYSIS_STEPS and app_state.step_group_tags.get(ANALYSIS_STEPS[0]):
    # active_step_name을 설정 파일에서 읽어왔더라도 여기서 강제로 첫번째 스텝으로 설정
    app_state.active_step_name = "" # switch_step_view가 확실히 동작하도록 현재 활성 스텝을 비움
    switch_step_view(None, None, ANALYSIS_STEPS[0])
else:
    print("Warning: Could not switch to the first analysis step on startup.")

# maximize_viewport()는 사용자 경험에 따라 선택
# dpg.maximize_viewport()
dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()