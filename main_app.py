# main_app.py
import dearpygui.dearpygui as dpg
import pandas as pd
import platform
import os

import utils
import step_01_data_loading
import step_02_exploratory_data_analysis
import traceback
import hashlib # 설정 파일명 생성을 위해 추가
import shutil
import json

SETTINGS_DIR_NAME = ".app_file_settings"
SESSION_INFO_FILE = os.path.join(SETTINGS_DIR_NAME, "session_info.json")
selected_target_variable_type: str = "Continuous" # 기본값 또는 None, "Categorical" 또는 "Continuous" 저장
TARGET_VARIABLE_TYPE_RADIO_TAG = "target_variable_type_radio"
TARGET_VARIABLE_TYPE_LABEL_TAG = "target_variable_type_label"

current_df: pd.DataFrame = None
original_df: pd.DataFrame = None
loaded_file_path: str = None

step_group_tags = {}
module_ui_updaters = {} # {module_name_key: update_function}
active_step_name: str = None
selected_target_variable: str = None
TARGET_VARIABLE_COMBO_TAG = "target_variable_combo"

_eda_sva_initialized = False # EDA 탭의 SVA가 초기 실행되었는지 여부 플래그

ANALYSIS_STEPS = [
    "1. Data Loading and Overview",
    "2. Exploratory Data Analysis (EDA)",
    # "3. Data Quality Assessment",
    # "4. Data Preprocessing",
]
active_settings = {}


# 폰트 설정 함수 (이전과 동일하게 유지)
def setup_korean_font():
    font_path = None
    font_size = 17 # macOS에서는 17 정도가 적당할 수 있습니다.
    os_type = platform.system()
    print(f"--- Font Setup Initiated ---")
    print(f"Operating System: {os_type}")

    if os_type == "Darwin":  # macOS
        potential_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        ]
        print("macOS: Checking potential font paths...")
        for idx, p_path in enumerate(potential_paths):
            print(f"  Checking path [{idx+1}]: {p_path}")
            if os.path.exists(p_path):
                font_path = p_path
                print(f"  SUCCESS: Font found at {font_path}")
                break
            else:
                print(f"  FAIL: Font not found at {p_path}")
        
        if font_path:
            print(f"macOS: Selected font for use: {font_path}")
        else:
            print("macOS: ERROR - No AppleGothic or AppleSDGothicNeo font found.")
    elif os_type == "Windows":
        # Windows 폰트 경로 (이전 코드 참조)
        potential_paths = ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc"]
        for p in potential_paths:
            if os.path.exists(p): font_path = p; break
        if font_path: print(f"Windows: Selected font {font_path}")
        else: print("Windows: Malgun Gothic or Gulim not found.")
    elif os_type == "Linux":
        # Linux 폰트 경로 (이전 코드 참조)
        potential_paths = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
        bundled_font_path = "NanumGothic.ttf"
        if os.path.exists(bundled_font_path): font_path = bundled_font_path
        else:
            for p in potential_paths:
                if os.path.exists(p): font_path = p; break
        if font_path: print(f"Linux: Selected font {font_path}")
        else: print("Linux: NanumGothic not found.")
    else:
        print(f"Unsupported OS for specific font setup: {os_type}.")

    if font_path and os.path.exists(font_path):
        print(f"Attempting to load and bind font: '{font_path}' with size {font_size}")
        try:
            font_registry_tag = "global_font_registry_unique" 
            font_to_bind_tag = "korean_font_for_app" # 태그명 변경 가능

            if not dpg.does_item_exist(font_registry_tag):
                dpg.add_font_registry(tag=font_registry_tag)
                print(f"Font registry '{font_registry_tag}' created.")
            else:
                print(f"Font registry '{font_registry_tag}' already exists.")

            if not dpg.does_item_exist(font_to_bind_tag):
                dpg.add_font(
                    file=font_path,
                    size=font_size,
                    tag=font_to_bind_tag,
                    parent=font_registry_tag
                )
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean, parent=font_to_bind_tag)
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default, parent=font_to_bind_tag)
                print(f"Font '{font_path}' added with tag '{font_to_bind_tag}' to registry '{font_registry_tag}'.")
            else:
                print(f"Font with tag '{font_to_bind_tag}' already exists. Attempting to use existing.")

            dpg.bind_font(font_to_bind_tag)
            print(f"Successfully attempted to bind font '{font_to_bind_tag}' (from {font_path}).")
        except Exception as e:
            print(f"Error during explicit font processing for '{font_path}': {e}. DPG default font will be used.")
            traceback.print_exc()
    elif font_path and not os.path.exists(font_path): # font_path는 있었으나 파일이 없는 경우
        print(f"Font path '{font_path}' was determined, but the file does not exist. DPG default font will be used.")
    else: # 적절한 font_path를 찾지 못한 경우
        print("No suitable Korean font path was found. DPG default font will be used.")
    print(f"--- Font Setup Finished ---")


# 타겟 변수 콤보박스 아이템 업데이트 함수 (이전과 동일)
def update_target_variable_combo():
    global current_df, selected_target_variable, TARGET_VARIABLE_COMBO_TAG
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        if current_df is not None and not current_df.empty:
            items = [""] + list(current_df.columns) 
            dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=items)
            if selected_target_variable and selected_target_variable in current_df.columns:
                dpg.set_value(TARGET_VARIABLE_COMBO_TAG, selected_target_variable)
            else:
                dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        else:
            dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=[""])
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")


# 데이터 로드 함수 (이전과 거의 동일, _eda_sva_initialized 플래그 초기화 추가)
def load_data_from_file(file_path: str) -> bool:
    global current_df, original_df, loaded_file_path, selected_target_variable, _eda_sva_initialized
    success = False
    try:
        current_df = pd.read_parquet(file_path)
        original_df = current_df.copy()
        loaded_file_path = file_path
        print(f"Data loaded successfully: {file_path}, Shape: {current_df.shape}")
        success = True
    except Exception as e:
        current_df = None; original_df = None; loaded_file_path = None
        print(f"Error loading data: {e}")
        if dpg.does_item_exist(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT):
            dpg.set_value(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT, f"Error: {e}")
        success = False

    if success:
        if hasattr(step_01_data_loading, '_type_selections'): step_01_data_loading._type_selections.clear()
        if hasattr(step_01_data_loading, '_imputation_selections'): step_01_data_loading._imputation_selections.clear()
        
        _eda_sva_initialized = False # 새 데이터 로드 시 SVA 초기화 플래그 리셋
        update_target_variable_combo()
        # 타겟 변수 선택 및 타입 추론은 target_variable_selected_callback에서 처리되므로 여기서 직접 호출 불필요
        # (콤보박스 값 변경 시 콜백 자동 호출)
        if selected_target_variable and current_df is not None and selected_target_variable not in current_df.columns:
            selected_target_variable = None # 이전 타겟이 새 데이터에 없으면 초기화
            if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)

        trigger_all_module_updates()
        return True
    else:
        _eda_sva_initialized = False # 실패 시에도 리셋
        update_target_variable_combo() # 콤보박스 아이템은 업데이트
        selected_target_variable = None # 선택된 타겟 변수 초기화
        global selected_target_variable_type
        selected_target_variable_type = "Continuous" # 기본값으로
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): 
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type)

        trigger_all_module_updates() 
        return False

def target_variable_type_changed_callback(sender, app_data, user_data):
    global selected_target_variable_type, active_settings, current_df, selected_target_variable

    newly_selected_type = app_data # "Categorical" 또는 "Continuous"
    previous_type = selected_target_variable_type # 변경 전 값 저장

    # 유효성 검사: 텍스트 기반 변수를 "Continuous"로 설정하려는 경우 방지
    if newly_selected_type == "Continuous" and selected_target_variable and current_df is not None:
        s1_column_types = main_app_callbacks.get('get_column_analysis_types', lambda: {})()
        analysis_type_from_s1 = s1_column_types.get(selected_target_variable)

        is_text_based = False
        if analysis_type_from_s1:
            # step_01_data_loading._infer_series_type 에서 정의된 텍스트 관련 타입들
            text_keywords = ["Text (", "Potentially Sensitive"] 
            if any(keyword in analysis_type_from_s1 for keyword in text_keywords):
                is_text_based = True
        
        # Step 1 정보가 없거나 Text가 아니면, Pandas Dtype으로 한 번 더 간단히 체크 (매우 엄격하진 않음)
        elif selected_target_variable in current_df.columns and \
             (pd.api.types.is_object_dtype(current_df[selected_target_variable].dtype) or \
              pd.api.types.is_string_dtype(current_df[selected_target_variable].dtype)):
            # 이 경우, 거의 모든 값이 고유한 ID성 문자열인지 등으로 추가 판단 가능하나,
            # 여기서는 Step 1의 타입 지정을 더 우선시함.
            # 만약 Step 1에서 Numeric 등으로 변환했다면 그 타입을 따름.
            # 순수 Object/String인데 Step 1 타입 정보가 없다면 Text로 간주 가능성 높음.
            if analysis_type_from_s1 is None: # Step 1 타입 정보가 아예 없을 때만 이 조건 고려
                is_text_based = True


        if is_text_based:
            dpg.alert_dialog(title="Type Selection Error",
                             message=f"Variable '{selected_target_variable}' is identified as Text-based.\nIt cannot be set to 'Continuous'.\n\nPlease use 'Categorical' or change the variable type in 'Step 1. Data Loading and Overview' if it's misclassified.",
                             width=450, height=200)
            # 라디오 버튼 값을 이전 값으로 되돌림
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, previous_type)
            # selected_target_variable_type은 변경하지 않고 종료
            return

    # 유효성 검사 통과 또는 "Categorical" 선택 시
    selected_target_variable_type = newly_selected_type
    print(f"Target variable type explicitly set to: {selected_target_variable_type}")

    if active_settings: # 현재 파일에 대한 설정이 있다면 업데이트
        active_settings['selected_target_variable_type'] = selected_target_variable_type

    # 목표 변수 유형 변경 시 관련 분석(예: SVA)을 다시 트리거 할 수 있음
    if active_step_name == ANALYSIS_STEPS[1]: # "2. Exploratory Data Analysis (EDA)"
         _eda_sva_initialized = False # SVA 재실행 필요
         trigger_specific_module_update(ANALYSIS_STEPS[1])


def target_variable_selected_callback(sender, app_data, user_data):
    global selected_target_variable, selected_target_variable_type, _eda_sva_initialized, current_df, active_settings

    new_target = app_data

    # 실제 변경이 없으면 종료하지 않음. 타입 추론은 항상 실행될 수 있도록.
    # if new_target == selected_target_variable:
    #     return

    selected_target_variable = new_target
    print(f"Target variable selected: {selected_target_variable}")

    if selected_target_variable and current_df is not None and selected_target_variable in current_df.columns:
        # Step 1의 타입 선택 정보 가져오기 (콜백 사용)
        s1_type_selections = main_app_callbacks.get('get_column_analysis_types', lambda: {})()

        guessed_type = utils._guess_target_type(current_df, selected_target_variable, s1_type_selections)
        selected_target_variable_type = guessed_type # 전역 변수 업데이트

        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)

        if active_settings:
            active_settings['selected_target_variable'] = selected_target_variable
            active_settings['selected_target_variable_type'] = selected_target_variable_type
    else: # 목표 변수 선택 해제 시
        selected_target_variable_type = "Continuous" # 기본값으로 초기화
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type) # 기본값으로 설정

        if active_settings:
            active_settings['selected_target_variable'] = None
            active_settings['selected_target_variable_type'] = None # 또는 기본값 selected_target_variable_type

    _eda_sva_initialized = False
    trigger_all_module_updates()

def get_settings_filepath(original_data_filepath: str) -> str:
    """원본 데이터 파일 경로로부터 고유한 설정 파일 경로를 생성합니다."""
    if not original_data_filepath:
        return None
    filename = hashlib.md5(original_data_filepath.encode('utf-8')).hexdigest() + ".json"
    return os.path.join(SETTINGS_DIR_NAME, filename)

def load_json_settings(settings_filepath: str) -> dict:
    """JSON 설정 파일을 로드합니다."""
    if settings_filepath and os.path.exists(settings_filepath):
        try:
            with open(settings_filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings from {settings_filepath}: {e}")
            traceback.print_exc()
    return None

def save_json_settings(settings_filepath: str, settings_dict: dict):
    """설정 딕셔너리를 JSON 파일로 저장합니다."""
    if not settings_filepath or not settings_dict:
        return
    try:
        if not os.path.exists(SETTINGS_DIR_NAME):
            os.makedirs(SETTINGS_DIR_NAME)
        with open(settings_filepath, 'w') as f:
            json.dump(settings_dict, f, indent=4)
        print(f"Settings saved to {settings_filepath}")
    except Exception as e:
        print(f"Error saving settings to {settings_filepath}: {e}")
        traceback.print_exc()

def gather_current_settings() -> dict:
    """현재 애플리케이션의 모든 관련 설정을 수집합니다."""
    global selected_target_variable, selected_target_variable_type, active_step_name, _eda_sva_initialized
    
    settings = {
        'selected_target_variable': selected_target_variable,
        'selected_target_variable_type': selected_target_variable_type, # 추가
        'active_step_name': active_step_name,
        '_eda_sva_initialized': _eda_sva_initialized,
        'step_01_settings': {},
        'step_02_settings': {}
    }

    # Step 01 설정 수집
    if hasattr(step_01_data_loading, '_type_selections'):
        settings['step_01_settings']['type_selections'] = step_01_data_loading._type_selections.copy()
    if hasattr(step_01_data_loading, '_imputation_selections'):
        settings['step_01_settings']['imputation_selections'] = step_01_data_loading._imputation_selections.copy()
    if hasattr(step_01_data_loading, '_custom_nan_input_value'):
        settings['step_01_settings']['custom_nan_input_value'] = step_01_data_loading._custom_nan_input_value
    
    # Step 02 EDA 설정 수집 (DPG 아이템 값 가져오기)
    try:
        if dpg.is_dearpygui_running():
            s02_set = settings['step_02_settings']
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO):
                s02_set['sva_filter_strength'] = dpg.get_value(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO)
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
                s02_set['sva_group_by_target'] = dpg.get_value(step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX)
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
                s02_set['sva_grouped_plot_type'] = dpg.get_value(step_02_exploratory_data_analysis.TAG_SVA_GROUPED_PLOT_TYPE_RADIO)
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR):
                s02_set['mva_pairplot_vars'] = dpg.get_value(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR)
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO):
                s02_set['mva_pairplot_hue'] = dpg.get_value(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO)
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO):
                s02_set['mva_target_feature'] = dpg.get_value(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO)
    except Exception as e:
        print(f"Error gathering DPG settings: {e}")
        
    return settings

def apply_settings(settings_dict: dict):
    global current_df, original_df, selected_target_variable, selected_target_variable_type, \
           active_step_name, _eda_sva_initialized, active_settings

    if original_df is None:
        print("Error in apply_settings: original_df is None. Cannot apply settings.")
        return
    if not settings_dict:
        print("apply_settings: No specific settings provided. Using original_df as current_df.")
        current_df = original_df.copy()
        selected_target_variable = None
        selected_target_variable_type = "Continuous" # 기본값
        _eda_sva_initialized = False
        if hasattr(step_01_data_loading, '_type_selections'): step_01_data_loading._type_selections.clear()
        if hasattr(step_01_data_loading, '_imputation_selections'): step_01_data_loading._imputation_selections.clear()
        if hasattr(step_01_data_loading, '_custom_nan_input_value'): step_01_data_loading._custom_nan_input_value = ""
        active_settings = gather_current_settings() 
        # UI 업데이트는 호출부에서
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): 
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type)
        return

    print("Applying settings...")
    active_settings = settings_dict
    current_df = original_df.copy()

    selected_target_variable = settings_dict.get('selected_target_variable', None)
    selected_target_variable_type = settings_dict.get('selected_target_variable_type', "Continuous") # 추가 및 기본값
    _eda_sva_initialized = settings_dict.get('_eda_sva_initialized', False)

    s01_settings = settings_dict.get('step_01_settings', {})
    if hasattr(step_01_data_loading, '_type_selections'):
        step_01_data_loading._type_selections = s01_settings.get('type_selections', {}).copy()
    if hasattr(step_01_data_loading, '_imputation_selections'):
        step_01_data_loading._imputation_selections = s01_settings.get('imputation_selections', {}).copy()
    custom_nan_val_s01 = s01_settings.get('custom_nan_input_value', "")
    if hasattr(step_01_data_loading, '_custom_nan_input_value'):
        step_01_data_loading._custom_nan_input_value = custom_nan_val_s01
    
    if custom_nan_val_s01:
        if hasattr(step_01_data_loading, '_apply_custom_nans'):
            print("  Applying custom NaNs...")
            step_01_data_loading._apply_custom_nans(main_app_callbacks, custom_nan_val_s01)
    if step_01_data_loading._type_selections:
        if hasattr(step_01_data_loading, '_apply_type_changes'):
            print("  Applying type changes...")
            step_01_data_loading._apply_type_changes(main_app_callbacks)
    if step_01_data_loading._imputation_selections:
        if hasattr(step_01_data_loading, '_apply_missing_value_treatments'):
            print("  Applying missing value treatments...")
            step_01_data_loading._apply_missing_value_treatments(main_app_callbacks)

    print("  Restoring UI widget states...")
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        update_target_variable_combo()
        if selected_target_variable and current_df is not None and selected_target_variable in current_df.columns:
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, selected_target_variable)
            # 타겟 변수 선택 콜백을 여기서 직접 호출하여 타입 라디오 버튼 상태도 복원
            # target_variable_selected_callback(TARGET_VARIABLE_COMBO_TAG, selected_target_variable, None)
            # -> 위 방식 대신 아래처럼 직접 UI 상태 복원
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
                dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)
        elif current_df is not None:
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)


    if dpg.does_item_exist(step_01_data_loading.TAG_DL_CUSTOM_NAN_INPUT):
        dpg.set_value(step_01_data_loading.TAG_DL_CUSTOM_NAN_INPUT, custom_nan_val_s01)

    s02_ui_settings = settings_dict.get('step_02_settings', {})
    # ... (Step 02 UI 복원 로직은 기존과 거의 동일하게 유지) ...
    sva_filter = s02_ui_settings.get('sva_filter_strength')
    if sva_filter and dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO):
        dpg.set_value(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO, sva_filter)

    sva_group_target = s02_ui_settings.get('sva_group_by_target')
    if sva_group_target is not None and dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
        dpg.set_value(step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX, sva_group_target)
        if hasattr(step_02_exploratory_data_analysis, '_sva_group_by_target_callback'):
            step_02_exploratory_data_analysis._sva_group_by_target_callback(
                step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX, 
                sva_group_target, 
                main_app_callbacks
            )
    
    sva_grouped_plot_type = s02_ui_settings.get('sva_grouped_plot_type')
    if sva_grouped_plot_type and dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
        dpg.set_value(step_02_exploratory_data_analysis.TAG_SVA_GROUPED_PLOT_TYPE_RADIO, sva_grouped_plot_type)

    if current_df is not None:
        numeric_cols_for_mva = step_02_exploratory_data_analysis._get_numeric_cols(current_df)
        categorical_cols_for_mva_hue = [""] + step_02_exploratory_data_analysis._get_categorical_cols(current_df, max_unique_for_cat=10)
        all_columns = current_df.columns.tolist()

        mva_pp_vars = s02_ui_settings.get('mva_pairplot_vars')
        if mva_pp_vars is not None and dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR):
            valid_mva_pp_vars = [v for v in mva_pp_vars if v in numeric_cols_for_mva]
            dpg.configure_item(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=numeric_cols_for_mva)
            dpg.set_value(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR, valid_mva_pp_vars)

        mva_pp_hue = s02_ui_settings.get('mva_pairplot_hue')
        if mva_pp_hue is not None and dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO):
            dpg.configure_item(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO, items=categorical_cols_for_mva_hue)
            if mva_pp_hue in categorical_cols_for_mva_hue:
                dpg.set_value(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO, mva_pp_hue)
            elif categorical_cols_for_mva_hue:
                dpg.set_value(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO, categorical_cols_for_mva_hue[0])

        mva_target_feat = s02_ui_settings.get('mva_target_feature')
        if mva_target_feat is not None and dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO) and selected_target_variable:
            feature_candidates_for_mva = [col for col in all_columns if col != selected_target_variable]
            dpg.configure_item(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO, items=feature_candidates_for_mva)
            if mva_target_feat in feature_candidates_for_mva:
                dpg.set_value(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO, mva_target_feat)
            elif feature_candidates_for_mva:
                dpg.set_value(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO, feature_candidates_for_mva[0])
    
    restored_active_step = settings_dict.get('active_step_name', ANALYSIS_STEPS[0] if ANALYSIS_STEPS else None)
    if restored_active_step:
        if restored_active_step in step_group_tags and dpg.does_item_exist(step_group_tags[restored_active_step]):
            switch_step_view(None, None, restored_active_step) 
            active_step_name = restored_active_step
            print(f"  Restored active step: {active_step_name}")
        else:
            print(f"  Could not restore active step '{restored_active_step}'. Defaulting to first.")
            if ANALYSIS_STEPS and ANALYSIS_STEPS[0] in step_group_tags and dpg.does_item_exist(step_group_tags[ANALYSIS_STEPS[0]]):
                switch_step_view(None, None, ANALYSIS_STEPS[0])
                active_step_name = ANALYSIS_STEPS[0]


def reset_application_state(clear_df=True):
    global current_df, original_df, loaded_file_path, selected_target_variable, \
           selected_target_variable_type, active_step_name, _eda_sva_initialized, active_settings
    
    print("Resetting application state...")
    if clear_df:
        current_df = None
        original_df = None
        loaded_file_path = None
    
    selected_target_variable = None
    selected_target_variable_type = "Continuous" # 기본값으로 리셋
    _eda_sva_initialized = False
    active_settings = {}

    if hasattr(step_01_data_loading, '_type_selections'): step_01_data_loading._type_selections.clear()
    if hasattr(step_01_data_loading, '_imputation_selections'): step_01_data_loading._imputation_selections.clear()
    if hasattr(step_01_data_loading, '_custom_nan_input_value'): step_01_data_loading._custom_nan_input_value = ""

    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=[""])
        dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
    
    # 타겟 변수 타입 UI 초기화
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
        dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
        dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
        dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type) # 기본값 설정

    if dpg.does_item_exist(step_01_data_loading.TAG_DL_CUSTOM_NAN_INPUT):
        dpg.set_value(step_01_data_loading.TAG_DL_CUSTOM_NAN_INPUT, "")
    
    if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO):
        dpg.set_value(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO, "Weak (Exclude obvious non-analytical)")
    if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
        dpg.set_value(step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
        if hasattr(step_02_exploratory_data_analysis, '_sva_group_by_target_callback'):
            step_02_exploratory_data_analysis._sva_group_by_target_callback(
                step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False, main_app_callbacks)
    if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR):
        dpg.configure_item(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=[])
    if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO):
        dpg.configure_item(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO, items=[""])
        dpg.set_value(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO, "")
    if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO):
        dpg.configure_item(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO, items=[])

    trigger_all_module_updates() 
    
    if ANALYSIS_STEPS:
        if not active_step_name or active_step_name not in step_group_tags:
             first_step = ANALYSIS_STEPS[0]
             if first_step in step_group_tags and dpg.does_item_exist(step_group_tags[first_step]):
                switch_step_view(None, None, first_step)

def file_load_callback(sender, app_data):
    global loaded_file_path, original_df, current_df, active_settings

    new_file_selected_path = app_data.get('file_path_name')
    if not new_file_selected_path:
        print("File selection cancelled.")
        return

    if loaded_file_path and active_settings:
        print(f"Saving settings for previously loaded file: {loaded_file_path}")
        old_settings_filepath = get_settings_filepath(loaded_file_path)
        current_live_settings = gather_current_settings()
        save_json_settings(old_settings_filepath, current_live_settings)

    reset_application_state(clear_df=False) 

    loaded_file_path = new_file_selected_path
    print(f"Loading new file: {loaded_file_path}")
    try:
        original_df = pd.read_parquet(loaded_file_path)
        current_df = original_df.copy()
        if dpg.does_item_exist(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT):
             dpg.set_value(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT, f"File: {loaded_file_path}, Shape: {current_df.shape}")
        print(f"Data loaded successfully: {loaded_file_path}, Shape: {current_df.shape}")
    except Exception as e:
        original_df = None; current_df = None; loaded_file_path = None
        print(f"Error loading data from {new_file_selected_path}: {e}")
        traceback.print_exc()
        if dpg.does_item_exist(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT):
            dpg.set_value(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT, f"Error loading file: {e}")
        trigger_all_module_updates()
        return

    new_settings_filepath = get_settings_filepath(loaded_file_path)
    loaded_specific_settings = load_json_settings(new_settings_filepath)
    
    if loaded_specific_settings:
        print(f"Found existing settings for {loaded_file_path}. Applying them.")
        apply_settings(loaded_specific_settings) 
    else:
        print(f"No settings found for {loaded_file_path}. Using default settings.")
        active_settings = gather_current_settings() 
        # 타겟 변수 UI 초기화 (reset_application_state에서 이미 처리)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)


    update_target_variable_combo()
    trigger_all_module_updates()

    if not active_step_name and ANALYSIS_STEPS:
        switch_step_view(None, None, ANALYSIS_STEPS[0])


def initial_load_on_startup():
    global loaded_file_path, active_settings, original_df, current_df, selected_target_variable_type

    print("Attempting initial load on startup...")
    session_info = load_json_settings(SESSION_INFO_FILE)
    last_file = None
    if session_info:
        last_file = session_info.get('last_opened_original_file')

    if last_file and os.path.exists(last_file):
        print(f"Restoring last session for file: {last_file}")
        reset_application_state(clear_df=False) 

        loaded_file_path = last_file
        try:
            original_df = pd.read_parquet(loaded_file_path) 
            current_df = original_df.copy()

            if dpg.does_item_exist(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT):
                 dpg.set_value(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT, f"File: {loaded_file_path}, Shape: {current_df.shape}")
            print(f"Data for last session loaded: {loaded_file_path}")

            settings_filepath = get_settings_filepath(last_file)
            specific_settings = load_json_settings(settings_filepath)
            if specific_settings:
                apply_settings(specific_settings) 
            else: 
                active_settings = gather_current_settings() 
                # 타겟 변수 UI는 apply_settings 또는 reset_application_state에 의해 관리됨
                # apply_settings({})를 호출하면 기본값으로 설정됨
                if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
                if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)

            update_target_variable_combo() # 아이템 채우기
            # apply_settings 이후에 selected_target_variable 값에 따라 target_variable_selected_callback이
            # 직접 호출되지는 않으므로, UI 상태를 명시적으로 한 번 더 동기화
            if selected_target_variable and selected_target_variable in current_df.columns:
                if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                    dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type)
                if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
                    dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
                if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                    dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)
            else:
                if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
                if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)

            trigger_all_module_updates()

            if not active_step_name and ANALYSIS_STEPS:
                 switch_step_view(None, None, ANALYSIS_STEPS[0])
            return True

        except Exception as e:
            print(f"Error restoring last session for {last_file}: {e}")
            traceback.print_exc()
            reset_application_state(clear_df=True) 
            trigger_all_module_updates() 
            return False
    else:
        print("No last session info found or file does not exist. Starting fresh.")
        reset_application_state(clear_df=True) 
        active_settings = gather_current_settings() 
        trigger_all_module_updates() 
        if ANALYSIS_STEPS: 
            switch_step_view(None, None, ANALYSIS_STEPS[0])
        return False

def save_state_on_exit():
    global loaded_file_path

    print("Saving state on exit...")
    if loaded_file_path: 
        current_settings_on_exit = gather_current_settings()
        settings_filepath = get_settings_filepath(loaded_file_path)
        save_json_settings(settings_filepath, current_settings_on_exit)
        
        session_info_to_save = {'last_opened_original_file': loaded_file_path}
        save_json_settings(SESSION_INFO_FILE, session_info_to_save)
        print(f"Last session info saved for: {loaded_file_path}")
    else:
        print("No active file loaded. Nothing to save as last session.")


def reset_current_df_to_original_data():
    global current_df, original_df, selected_target_variable, selected_target_variable_type, _eda_sva_initialized
    if original_df is not None:
        current_df = original_df.copy()
        print("Data reset to original.")
        if hasattr(step_01_data_loading, '_type_selections'): step_01_data_loading._type_selections.clear()
        if hasattr(step_01_data_loading, '_imputation_selections'): step_01_data_loading._imputation_selections.clear()
        
        _eda_sva_initialized = False 
        update_target_variable_combo() # 콤보 아이템 리프레시

        # 타겟 변수 선택 및 타입 UI도 초기화 또는 재추론
        if selected_target_variable and selected_target_variable in current_df.columns:
            # 선택된 타겟 변수가 여전히 유효하면, 해당 변수에 대해 타입 재추론 및 UI 업데이트
            s1_selections = main_app_callbacks.get('get_column_analysis_types', lambda: {})()
            guessed_type = utils._guess_target_type(current_df, selected_target_variable, s1_selections)
            selected_target_variable_type = guessed_type
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
                dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)
        elif selected_target_variable: # 선택된 타겟은 있었으나, 리셋 후 current_df에 없거나 문제가 생긴 경우
            selected_target_variable = None
            selected_target_variable_type = "Continuous" # 기본값으로
            if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): 
                dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
                dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type)

        trigger_all_module_updates()
    else:
        print("No original data to reset to.")


def switch_step_view(sender, app_data, user_data_step_name: str):
    global active_step_name, _eda_sva_initialized
    print(f"Attempting to switch to step: {user_data_step_name}")

    for step_name_iter, group_tag in step_group_tags.items():
        if dpg.does_item_exist(group_tag):
            if step_name_iter == user_data_step_name:
                dpg.show_item(group_tag)
                active_step_name = step_name_iter
                # EDA 탭으로 전환 시 SVA가 초기화되지 않았다면 실행하지 않음 (Run 버튼으로 실행 유도)
                # if active_step_name == ANALYSIS_STEPS[1] and not _eda_sva_initialized:
                # print(f"Switched to EDA tab. SVA will run upon clicking 'Run SVA'.")
                # else:
                trigger_specific_module_update(step_name_iter)
            else:
                dpg.hide_item(group_tag)
    print(f"Active step: {active_step_name}")


def trigger_specific_module_update(module_name_key: str):
    if module_name_key in module_ui_updaters:
        updater = module_ui_updaters[module_name_key]
        if module_name_key == "2. Exploratory Data Analysis (EDA)":
            updater(current_df, main_app_callbacks) 
        else: 
             updater(current_df, original_df, util_functions_for_modules, loaded_file_path)
        print(f"Module UI updated for: '{module_name_key}'")
    else:
        print(f"Warning: No UI updater found for '{module_name_key}'.")


def trigger_all_module_updates():
    global _eda_sva_initialized
    print("Updating all module UIs...")
    for step_name_key in list(module_ui_updaters.keys()): 
        trigger_specific_module_update(step_name_key)


util_functions_for_modules = {
    'create_table_with_data': utils.create_table_with_data,
    'calculate_column_widths': utils.calculate_column_widths,
    'format_text_for_display': utils.format_text_for_display,
    'get_safe_text_size': utils.get_safe_text_size,
}


def get_column_analysis_types_from_user_settings():
    if hasattr(step_01_data_loading, '_type_selections') and \
       isinstance(step_01_data_loading._type_selections, dict):
        return step_01_data_loading._type_selections.copy() 
    else:
        print("Warning: User-defined analysis types (_type_selections) not found in step_01_data_loading.")
        if current_df is not None:
             return {col: step_02_exploratory_data_analysis._infer_series_type(current_df[col])[0] for col in current_df.columns}
        return {}


main_app_callbacks = {
    'get_current_df': lambda: current_df,
    'get_original_df': lambda: original_df,
    'get_loaded_file_path': lambda: loaded_file_path,
    'get_util_funcs': lambda: util_functions_for_modules,
    'show_file_dialog': lambda: dpg.show_item("file_dialog_id"),
    'register_step_group_tag': lambda name, tag: step_group_tags.update({name: tag}),
    'register_module_updater': lambda name, func: module_ui_updaters.update({name: func}),
    'trigger_module_update': trigger_specific_module_update,
    'reset_current_df_to_original': reset_current_df_to_original_data,
    'trigger_all_module_updates': trigger_all_module_updates, 
    'get_selected_target_variable': lambda: selected_target_variable,
    'get_selected_target_variable_type': lambda: selected_target_variable_type, # 수정됨
    'update_target_variable_combo_items': update_target_variable_combo,
    'get_column_analysis_types': get_column_analysis_types_from_user_settings,
}


dpg.create_context()

with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback,
                     id="file_dialog_id", width=700, height=400, modal=True):
    dpg.add_file_extension(".parquet"); dpg.add_file_extension(".*")

setup_korean_font()

with dpg.window(label="Data Analysis Platform", tag="main_window"):
    with dpg.group(horizontal=True):
        with dpg.child_window(width=280, tag="navigation_panel", border=True):
            dpg.add_text("Target Variable (y):")
            dpg.add_combo(items=[""], tag=TARGET_VARIABLE_COMBO_TAG, width=-1,
                        callback=target_variable_selected_callback) # 콜백 연결

            dpg.add_text("Target Variable Type:", tag=TARGET_VARIABLE_TYPE_LABEL_TAG, show=False) # 초기 숨김
            dpg.add_radio_button(
                items=["Categorical", "Continuous"],
                tag=TARGET_VARIABLE_TYPE_RADIO_TAG,
                horizontal=True,
                default_value=selected_target_variable_type, # 전역 변수 기본값 사용
                callback=target_variable_type_changed_callback, # 콜백 연결
                show=False # 초기 숨김
            )

            dpg.add_separator()
            dpg.add_spacer(height=5)
            dpg.add_text("Analysis Steps", color=[255, 255, 0]); dpg.add_separator()
            for step_name_nav in ANALYSIS_STEPS:
                dpg.add_button(label=step_name_nav, callback=switch_step_view,
                              user_data=step_name_nav, width=-1, height=30)
        
        with dpg.child_window(tag="content_area", border=True):
            module_map = {
                "1. Data Loading and Overview": step_01_data_loading,
                "2. Exploratory Data Analysis (EDA)": step_02_exploratory_data_analysis,
            }
            for step_name_create in ANALYSIS_STEPS:
                module = module_map.get(step_name_create)
                if module and hasattr(module, 'create_ui'):
                    module.create_ui(step_name_create, "content_area", main_app_callbacks)
                    print(f"UI created for '{step_name_create}'.")
                else: 
                    fallback_tag = f"{step_name_create.lower().replace(' ', '_').replace('.', '').replace('&', 'and')}_fallback_group"
                    if not dpg.does_item_exist(fallback_tag):
                        main_app_callbacks['register_step_group_tag'](step_name_create, fallback_tag)
                        with dpg.group(tag=fallback_tag, parent="content_area", show=False):
                            dpg.add_text(f"--- {step_name_create} ---"); dpg.add_separator()
                            dpg.add_text(f"UI for '{step_name_create}' will be configured here.")
                        main_app_callbacks['register_module_updater'](step_name_create, lambda *args, **kwargs: None)

            if ANALYSIS_STEPS:
                first_step = ANALYSIS_STEPS[0]
                if first_step in step_group_tags and dpg.does_item_exist(step_group_tags[first_step]):
                     switch_step_view(None, None, first_step)
                else:
                    print(f"Warning: First step '{first_step}' UI group not found immediately after creation.")

dpg.create_viewport(title='Modular Data Analysis Platform GUI', width=1440, height=1000)
dpg.set_exit_callback(save_state_on_exit)
dpg.setup_dearpygui()
initial_load_on_startup() # 여기서 UI 상태 복원 시도
dpg.maximize_viewport()
dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()