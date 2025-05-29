# main_app.py
import dearpygui.dearpygui as dpg
import pandas as pd
import platform
import os
import utils
import step_01_data_loading
import step_02_exploratory_data_analysis
import traceback
import hashlib
import json

# 전역 변수 관리를 위한 클래스
class AppState:
    def __init__(self):
        self.current_df = None
        self.original_df = None
        self.df_after_step1 = None
        self.loaded_file_path = None
        self.selected_target_variable = None
        self.selected_target_variable_type = "Continuous"
        self.active_step_name = None
        self._eda_sva_initialized = False
        self._eda_outlier_settings_applied_once = False
        self.active_settings = {}
        self.step_group_tags = {}
        self.module_ui_updaters = {}

# 전역 상태 인스턴스
app_state = AppState()

SETTINGS_DIR_NAME = ".app_file_settings"
SESSION_INFO_FILE = os.path.join(SETTINGS_DIR_NAME, "session_info.json")
TARGET_VARIABLE_TYPE_RADIO_TAG = "target_variable_type_radio"
TARGET_VARIABLE_TYPE_LABEL_TAG = "target_variable_type_label"
TARGET_VARIABLE_COMBO_TAG = "target_variable_combo"

ANALYSIS_STEPS = [
    "1. Data Loading and Overview",
    "2. Exploratory Data Analysis (EDA)",
]

_MODAL_ID_SIMPLE_MESSAGE = "simple_modal_message_id"

def _show_simple_modal_message(title: str, message: str, width: int = 450, height: int = 200):
    """간단한 모달 메시지 표시"""
    if dpg.does_item_exist(_MODAL_ID_SIMPLE_MESSAGE):
        dpg.delete_item(_MODAL_ID_SIMPLE_MESSAGE)

    viewport_width = dpg.get_viewport_width() if dpg.is_dearpygui_running() else 1000
    viewport_height = dpg.get_viewport_height() if dpg.is_dearpygui_running() else 700
    
    modal_pos_x = (viewport_width - width) // 2
    modal_pos_y = (viewport_height - height) // 2
    
    with dpg.window(label=title, modal=True, show=True, id=_MODAL_ID_SIMPLE_MESSAGE,
                    no_close=True, width=width, height=height, pos=[modal_pos_x, modal_pos_y],
                    no_saved_settings=True, autosize=False):
        dpg.add_text(message, wrap=width - 20)
        dpg.add_spacer(height=15)
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=(width - 100 - 30) // 2)
            dpg.add_button(label="OK", width=100, 
                          callback=lambda: dpg.configure_item(_MODAL_ID_SIMPLE_MESSAGE, show=False))

def setup_korean_font():
    """한국어 폰트 설정"""
    font_path = None
    font_size = 17
    os_type = platform.system()
    print(f"--- Font Setup Initiated ---")
    print(f"Operating System: {os_type}")

    font_paths = {
        "Darwin": [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        ],
        "Windows": [
            "C:/Windows/Fonts/malgun.ttf", 
            "C:/Windows/Fonts/gulim.ttc"
        ],
        "Linux": [
            "NanumGothic.ttf",  # Bundled font
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        ]
    }
    
    potential_paths = font_paths.get(os_type, [])
    for p_path in potential_paths:
        if os.path.exists(p_path):
            font_path = p_path
            break
    
    if font_path and os.path.exists(font_path):
        print(f"Attempting to load font: '{font_path}' with size {font_size}")
        try:
            with dpg.font_registry():
                dpg.add_font(font_path, font_size, tag="korean_font_for_app")
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean, parent="korean_font_for_app")
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default, parent="korean_font_for_app")
            
            dpg.bind_font("korean_font_for_app")
            print(f"Successfully bound font.")
        except Exception as e:
            print(f"Error during font processing: {e}")
            traceback.print_exc()
    else:
        print("No suitable Korean font found. Using default font.")
    print(f"--- Font Setup Finished ---")

def update_target_variable_combo():
    """타겟 변수 콤보박스 업데이트"""
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        if app_state.current_df is not None and not app_state.current_df.empty:
            items = [""] + list(app_state.current_df.columns)
            dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=items)
            if app_state.selected_target_variable and app_state.selected_target_variable in app_state.current_df.columns:
                dpg.set_value(TARGET_VARIABLE_COMBO_TAG, app_state.selected_target_variable)
            else:
                dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        else:
            dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=[""])
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")

def step1_processing_complete(processed_df: pd.DataFrame):
    """Step 1 처리 완료 콜백"""
    if processed_df is None:
        print("Step 1 returned no DataFrame. Cannot proceed to EDA.")
        return

    app_state.df_after_step1 = processed_df.copy()
    app_state.current_df = app_state.df_after_step1.copy()
    print(f"DataFrame after Step 1 processing received. Shape: {app_state.current_df.shape}")

    app_state._eda_sva_initialized = False
    app_state._eda_outlier_settings_applied_once = False

    # 타겟 변수 유효성 검사
    if app_state.selected_target_variable and app_state.current_df is not None and \
       app_state.selected_target_variable not in app_state.current_df.columns:
        print(f"Warning: Target variable '{app_state.selected_target_variable}' not in new DataFrame.")
        app_state.selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
    
    update_target_variable_combo()
    
    # 아웃라이어 설정 적용
    if app_state.active_settings and 'step_02_settings' in app_state.active_settings and \
       'outlier_config' in app_state.active_settings['step_02_settings']:
        if hasattr(step_02_exploratory_data_analysis, 'apply_outlier_treatment_from_settings'):
            print("Applying loaded outlier settings...")
            app_state.current_df, applied_now = step_02_exploratory_data_analysis.apply_outlier_treatment_from_settings(
                app_state.current_df, 
                app_state.active_settings['step_02_settings']['outlier_config'], 
                main_app_callbacks
            )
            app_state._eda_outlier_settings_applied_once = applied_now
            if applied_now:
                print("Outlier settings applied to current_df.")
    
    trigger_all_module_updates()

def trigger_specific_module_update(module_name_key: str):
    """특정 모듈 UI 업데이트 - 수정된 버전"""
    if module_name_key in app_state.module_ui_updaters:
        updater = app_state.module_ui_updaters[module_name_key]
        
        if module_name_key == ANALYSIS_STEPS[0]:  # Data Loading
            if hasattr(step_01_data_loading, 'update_ui'):
                # Step 1은 4개의 인자를 받음
                updater(
                    app_state.current_df,
                    app_state.original_df,
                    util_functions_for_modules,
                    app_state.loaded_file_path
                )
        elif module_name_key == ANALYSIS_STEPS[1]:  # EDA
            if hasattr(step_02_exploratory_data_analysis, 'update_ui'):
                # Step 2는 2개의 인자를 받음
                updater(
                    app_state.current_df,
                    main_app_callbacks
                )
        else:
            # 기타 모듈 (향후 확장 시)
            updater(app_state.current_df, main_app_callbacks)
        
        print(f"Module UI updated for: '{module_name_key}'")
    else:
        print(f"Warning: No UI updater found for '{module_name_key}'.")

def trigger_all_module_updates():
    """모든 모듈 UI 업데이트"""
    print("Updating all module UIs...")
    for step_name_key in list(app_state.module_ui_updaters.keys()):
        trigger_specific_module_update(step_name_key)

def load_data_from_file(file_path: str) -> bool:
    """파일에서 데이터 로드"""
    success = False
    try:
        app_state.original_df = pd.read_parquet(file_path)
        app_state.current_df = None
        app_state.df_after_step1 = None
        app_state.loaded_file_path = file_path
        print(f"Raw data loaded successfully: {file_path}, Shape: {app_state.original_df.shape}")
        success = True
    except Exception as e:
        app_state.current_df = None
        app_state.original_df = None
        app_state.loaded_file_path = None
        app_state.df_after_step1 = None
        print(f"Error loading raw data: {e}")
        traceback.print_exc()
        if dpg.does_item_exist(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT):
            dpg.set_value(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT, f"Error loading file: {e}")
        success = False

    if success:
        # Step 1 설정 초기화
        if hasattr(step_01_data_loading, '_type_selections'):
            step_01_data_loading._type_selections.clear()
        if hasattr(step_01_data_loading, '_imputation_selections'):
            step_01_data_loading._imputation_selections.clear()
        if hasattr(step_01_data_loading, '_custom_nan_input_value'):
            step_01_data_loading._custom_nan_input_value = ""
        
        app_state._eda_sva_initialized = False
        app_state._eda_outlier_settings_applied_once = False
        
        # 타겟 변수 유효성 검사
        if app_state.selected_target_variable and app_state.original_df is not None and \
           app_state.selected_target_variable not in app_state.original_df.columns:
            app_state.selected_target_variable = None
            if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
                dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
                dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
        
        if ANALYSIS_STEPS and ANALYSIS_STEPS[0] in app_state.module_ui_updaters:
            trigger_specific_module_update(ANALYSIS_STEPS[0])
        
        return True
    else:
        reset_application_state(clear_df_completely=True)
        trigger_all_module_updates()
        return False

def target_variable_type_changed_callback(sender, app_data, user_data):
    """타겟 변수 타입 변경 콜백"""
    newly_selected_type = app_data
    previous_type = app_state.selected_target_variable_type

    # 타입 검증
    if newly_selected_type == "Continuous" and app_state.selected_target_variable and \
       app_state.current_df is not None:
        s1_column_types = main_app_callbacks.get('get_column_analysis_types', lambda: {})()
        analysis_type_from_s1 = s1_column_types.get(app_state.selected_target_variable)

        is_text_based = False
        if analysis_type_from_s1:
            text_keywords = ["Text (", "Potentially Sensitive", "Categorical (High Cardinality)"]
            if any(keyword in analysis_type_from_s1 for keyword in text_keywords):
                is_text_based = True
        elif app_state.selected_target_variable in app_state.current_df.columns and \
             (pd.api.types.is_object_dtype(app_state.current_df[app_state.selected_target_variable].dtype) or \
              pd.api.types.is_string_dtype(app_state.current_df[app_state.selected_target_variable].dtype)):
            if analysis_type_from_s1 is None:
                is_text_based = True

        if is_text_based:
            error_message = (
                f"Variable '{app_state.selected_target_variable}' is identified as Text-based or high cardinality categorical.\n"
                f"It cannot be reliably treated as 'Continuous' for most analyses.\n\n"
                f"Please use 'Categorical' or verify the variable type in\n"
                f"'Step 1. Data Loading and Overview' if it's misclassified."
            )
            _show_simple_modal_message("Type Selection Warning", error_message)
    
    app_state.selected_target_variable_type = newly_selected_type
    print(f"Target variable type set to: {app_state.selected_target_variable_type}")

    if app_state.active_settings:
        app_state.active_settings['selected_target_variable_type'] = app_state.selected_target_variable_type

    app_state._eda_sva_initialized = False
    if app_state.active_step_name == ANALYSIS_STEPS[1]:
        trigger_specific_module_update(ANALYSIS_STEPS[1])

def target_variable_selected_callback(sender, app_data, user_data):
    """타겟 변수 선택 콜백"""
    new_target = app_data
    if not new_target:
        app_state.selected_target_variable = None
        app_state.selected_target_variable_type = "Continuous"
        print("Target variable selection cleared.")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, app_state.selected_target_variable_type)
        if app_state.active_settings:
            app_state.active_settings['selected_target_variable'] = None
            app_state.active_settings['selected_target_variable_type'] = None
    else:
        app_state.selected_target_variable = new_target
        print(f"Target variable selected: {app_state.selected_target_variable}")

        if app_state.selected_target_variable and app_state.current_df is not None and \
           app_state.selected_target_variable in app_state.current_df.columns:
            s1_type_selections = main_app_callbacks.get('get_column_analysis_types', lambda: {})()
            guessed_type = utils._guess_target_type(
                app_state.current_df, 
                app_state.selected_target_variable, 
                s1_type_selections
            )
            app_state.selected_target_variable_type = guessed_type

            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, app_state.selected_target_variable_type)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
                dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)

            if app_state.active_settings:
                app_state.active_settings['selected_target_variable'] = app_state.selected_target_variable
                app_state.active_settings['selected_target_variable_type'] = app_state.selected_target_variable_type

    app_state._eda_sva_initialized = False
    trigger_all_module_updates()

def switch_step_view(sender, app_data, user_data_step_name: str):
    """스텝 뷰 전환"""
    print(f"Switching to step: {user_data_step_name}")

    for step_name_iter, group_tag in app_state.step_group_tags.items():
        if dpg.does_item_exist(group_tag):
            if step_name_iter == user_data_step_name:
                dpg.show_item(group_tag)
                app_state.active_step_name = step_name_iter
                trigger_specific_module_update(step_name_iter)
            else:
                dpg.hide_item(group_tag)
    print(f"Active step: {app_state.active_step_name}")

def file_load_callback(sender, app_data):
    """파일 로드 콜백"""
    new_file_selected_path = app_data.get('file_path_name')
    if not new_file_selected_path:
        print("File selection cancelled.")
        return

    # 현재 파일 설정 저장
    if app_state.loaded_file_path and app_state.active_settings:
        print(f"Saving settings for: {app_state.loaded_file_path}")
        old_settings_filepath = get_settings_filepath(app_state.loaded_file_path)
        current_live_settings = gather_current_settings()
        save_json_settings(old_settings_filepath, current_live_settings)

    # 상태 초기화
    reset_application_state(clear_df_completely=False)

    # 새 파일 로드
    print(f"Loading new file: {new_file_selected_path}")
    success = load_data_from_file(new_file_selected_path)
    
    if not success:
        _show_simple_modal_message(
            "File Load Error", 
            f"Failed to load {os.path.basename(new_file_selected_path)}"
        )
        return

    # 새 파일 설정 로드
    new_settings_filepath = get_settings_filepath(app_state.loaded_file_path)
    loaded_specific_settings = load_json_settings(new_settings_filepath)
    
    if loaded_specific_settings:
        print(f"Found existing settings for {app_state.loaded_file_path}")
        apply_settings(loaded_specific_settings)
    else:
        print(f"No settings found for {app_state.loaded_file_path}")
        app_state.active_settings = {}
        if hasattr(step_01_data_loading, 'process_newly_loaded_data'):
            step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks)
        else:
            trigger_specific_module_update(ANALYSIS_STEPS[0])

    # 첫 번째 스텝으로 전환
    if ANALYSIS_STEPS:
        switch_step_view(None, None, ANALYSIS_STEPS[0])

def reset_application_state(clear_df_completely=True):
    """애플리케이션 상태 초기화"""
    print("Resetting application state...")
    
    if clear_df_completely:
        app_state.current_df = None
        app_state.original_df = None
        app_state.df_after_step1 = None
        app_state.loaded_file_path = None
        app_state.active_settings = {}
    else:
        app_state.current_df = None
        app_state.df_after_step1 = None

    app_state.selected_target_variable = None
    app_state.selected_target_variable_type = "Continuous"
    app_state._eda_sva_initialized = False
    app_state._eda_outlier_settings_applied_once = False

    # Step 1 상태 초기화
    if hasattr(step_01_data_loading, 'reset_step1_state'):
        step_01_data_loading.reset_step1_state()
    else:
        if hasattr(step_01_data_loading, '_type_selections'):
            step_01_data_loading._type_selections.clear()
        if hasattr(step_01_data_loading, '_imputation_selections'):
            step_01_data_loading._imputation_selections.clear()
        if hasattr(step_01_data_loading, '_custom_nan_input_value'):
            step_01_data_loading._custom_nan_input_value = ""

    # UI 초기화
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=[""])
        dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
        dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
        dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
        dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, app_state.selected_target_variable_type)

    # Step 2 EDA UI 초기화
    if hasattr(step_02_exploratory_data_analysis, 'reset_eda_ui_defaults'):
        step_02_exploratory_data_analysis.reset_eda_ui_defaults()

    if not clear_df_completely and app_state.original_df is not None:
        print("Triggering Step 1 update with original_df.")
        trigger_specific_module_update(ANALYSIS_STEPS[0])
    else:
        trigger_all_module_updates()
    
    if ANALYSIS_STEPS:
        current_active_step = app_state.active_step_name
        if not current_active_step or current_active_step not in app_state.step_group_tags or \
           not dpg.does_item_exist(app_state.step_group_tags.get(current_active_step)):
            first_step = ANALYSIS_STEPS[0]
            if first_step in app_state.step_group_tags and \
               dpg.does_item_exist(app_state.step_group_tags[first_step]):
                switch_step_view(None, None, first_step)

def get_settings_filepath(original_data_filepath: str) -> str:
    """설정 파일 경로 생성"""
    if not original_data_filepath:
        return None
    filename = hashlib.md5(original_data_filepath.encode('utf-8')).hexdigest() + ".json"
    return os.path.join(SETTINGS_DIR_NAME, filename)

def load_json_settings(settings_filepath: str) -> dict:
    """JSON 설정 로드"""
    if settings_filepath and os.path.exists(settings_filepath):
        try:
            with open(settings_filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
            traceback.print_exc()
    return None

def save_json_settings(settings_filepath: str, settings_dict: dict):
    """JSON 설정 저장"""
    if not settings_filepath or not settings_dict:
        return
    try:
        if not os.path.exists(SETTINGS_DIR_NAME):
            os.makedirs(SETTINGS_DIR_NAME)
        with open(settings_filepath, 'w') as f:
            json.dump(settings_dict, f, indent=4)
        print(f"Settings saved to {settings_filepath}")
    except Exception as e:
        print(f"Error saving settings: {e}")
        traceback.print_exc()

def gather_current_settings() -> dict:
    """현재 설정 수집"""
    settings = {
        'selected_target_variable': app_state.selected_target_variable,
        'selected_target_variable_type': app_state.selected_target_variable_type,
        'active_step_name': app_state.active_step_name,
        'step_01_settings': {},
        'step_02_settings': {
            'sva_config': {},
            'mva_config': {},
            'outlier_config': {}
        }
    }

    # Step 1 설정
    if hasattr(step_01_data_loading, '_type_selections'):
        settings['step_01_settings']['type_selections'] = step_01_data_loading._type_selections.copy()
    if hasattr(step_01_data_loading, '_imputation_selections'):
        settings['step_01_settings']['imputation_selections'] = step_01_data_loading._imputation_selections.copy()
    if hasattr(step_01_data_loading, '_custom_nan_input_value'):
        settings['step_01_settings']['custom_nan_input_value'] = step_01_data_loading._custom_nan_input_value
    
    # Step 2 EDA 설정
    try:
        if dpg.is_dearpygui_running():
            s02_set = settings['step_02_settings']
            
            # SVA 설정
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO):
                s02_set['sva_config']['sva_filter_strength'] = dpg.get_value(
                    step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO
                )
            
            # MVA 설정
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR):
                s02_set['mva_config']['mva_pairplot_vars'] = dpg.get_value(
                    step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR
                )
            
            # Outlier 설정
            if hasattr(step_02_exploratory_data_analysis, 'get_outlier_settings_for_saving'):
                s02_set['outlier_config'].update(
                    step_02_exploratory_data_analysis.get_outlier_settings_for_saving()
                )
            s02_set['outlier_config']['_eda_outlier_settings_applied_once'] = \
                app_state._eda_outlier_settings_applied_once

    except Exception as e:
        print(f"Error gathering EDA settings: {e}")
        traceback.print_exc()
        
    return settings

def apply_settings(settings_dict: dict):
    """설정 적용"""
    if app_state.original_df is None:
        print("Error: No original data loaded.")
        _show_simple_modal_message("Error", "Cannot apply settings without data.")
        return
    
    print("Applying settings...")
    app_state.active_settings = settings_dict

    # 일반 상태 복원
    app_state.selected_target_variable = settings_dict.get('selected_target_variable', None)
    app_state.selected_target_variable_type = settings_dict.get('selected_target_variable_type', "Continuous")
    
    # Step 1 설정 적용
    s01_settings = settings_dict.get('step_01_settings', {})
    if hasattr(step_01_data_loading, 'apply_step1_settings_and_process'):
        print("  Applying Step 1 settings...")
        step_01_data_loading.apply_step1_settings_and_process(
            app_state.original_df, s01_settings, main_app_callbacks
        )
        if app_state.df_after_step1 is not None and app_state.current_df is None:
            app_state.current_df = app_state.df_after_step1.copy()
    else:
        print("  Warning: Step 1 cannot reprocess from settings.")
        if app_state.df_after_step1 is None:
            app_state.df_after_step1 = app_state.original_df.copy()
        app_state.current_df = app_state.df_after_step1.copy()

    # UI 업데이트
    update_target_variable_combo()
    
    # Step 2 설정 적용
    s02_settings = settings_dict.get('step_02_settings', {})
    outlier_conf = s02_settings.get('outlier_config', {})
    app_state._eda_outlier_settings_applied_once = outlier_conf.get('_eda_outlier_settings_applied_once', False)

    if hasattr(step_02_exploratory_data_analysis, 'apply_outlier_treatment_from_settings'):
        if app_state.current_df is not None:
            print("  Applying outlier settings...")
            modified_df, applied_this_time = step_02_exploratory_data_analysis.apply_outlier_treatment_from_settings(
                app_state.current_df.copy(),
                outlier_conf,
                main_app_callbacks
            )
            if applied_this_time:
                app_state.current_df = modified_df
                print("  Outlier treatment applied.")
                app_state._eda_outlier_settings_applied_once = True

    # 활성 스텝 복원
    restored_active_step = settings_dict.get('active_step_name', ANALYSIS_STEPS[0] if ANALYSIS_STEPS else None)
    if restored_active_step:
        trigger_all_module_updates()
        if restored_active_step in app_state.step_group_tags and \
           dpg.does_item_exist(app_state.step_group_tags[restored_active_step]):
            switch_step_view(None, None, restored_active_step)
            app_state.active_step_name = restored_active_step
            print(f"  Restored active step: {app_state.active_step_name}")

    print("Settings application finished.")

def initial_load_on_startup():
    """시작 시 초기 로드"""
    print("Attempting initial load on startup...")
    session_info = load_json_settings(SESSION_INFO_FILE)
    last_file_path = None
    if session_info:
        last_file_path = session_info.get('last_opened_original_file')

    if last_file_path and os.path.exists(last_file_path):
        print(f"Restoring last session for: {last_file_path}")
        
        try:
            app_state.original_df = pd.read_parquet(last_file_path)
            app_state.loaded_file_path = last_file_path
            print(f"Last session data loaded: shape {app_state.original_df.shape}")

            if dpg.does_item_exist(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT):
                dpg.set_value(
                    step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT,
                    f"File: {os.path.basename(app_state.loaded_file_path)}, Shape: {app_state.original_df.shape}"
                )

            settings_for_file = get_settings_filepath(last_file_path)
            specific_file_settings = load_json_settings(settings_for_file)
            
            if specific_file_settings:
                app_state.active_settings = specific_file_settings
                print("Applying settings from last session...")
                apply_settings(specific_file_settings)
            else:
                print("No specific settings found. Processing with defaults.")
                app_state.active_settings = {}
                if hasattr(step_01_data_loading, 'process_newly_loaded_data'):
                    step_01_data_loading.process_newly_loaded_data(app_state.original_df, main_app_callbacks)
                else:
                    trigger_specific_module_update(ANALYSIS_STEPS[0])

            if not app_state.active_step_name and ANALYSIS_STEPS:
                switch_step_view(None, None, ANALYSIS_STEPS[0])
            return True

        except Exception as e:
            print(f"Error restoring last session: {e}")
            traceback.print_exc()
            _show_simple_modal_message(
                "Session Restore Error",
                f"Could not restore session for {os.path.basename(last_file_path)}"
            )
            reset_application_state(clear_df_completely=True)
            trigger_all_module_updates()
            if ANALYSIS_STEPS:
                switch_step_view(None, None, ANALYSIS_STEPS[0])
            return False
    else:
        print("No last session to restore.")
        reset_application_state(clear_df_completely=True)
        app_state.active_settings = {}
        trigger_all_module_updates()
        if ANALYSIS_STEPS:
            switch_step_view(None, None, ANALYSIS_STEPS[0])
        return False

def save_state_on_exit():
    """종료 시 상태 저장"""
    print("Saving state on exit...")
    if app_state.loaded_file_path and os.path.exists(app_state.loaded_file_path):
        current_settings = gather_current_settings()
        settings_filepath = get_settings_filepath(app_state.loaded_file_path)
        save_json_settings(settings_filepath, current_settings)
        
        session_info = {'last_opened_original_file': app_state.loaded_file_path}
        save_json_settings(SESSION_INFO_FILE, session_info)
        print(f"Session saved for: {app_state.loaded_file_path}")

# 유틸리티 함수 정의
util_functions_for_modules = {
    'create_table_with_data': utils.create_table_with_data,
    'calculate_column_widths': utils.calculate_column_widths,
    'format_text_for_display': utils.format_text_for_display,
    'get_safe_text_size': utils.get_safe_text_size,
    '_show_simple_modal_message': _show_simple_modal_message,
}

# 콜백 함수 정의
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
    'reset_current_df_to_original': lambda: reset_application_state(clear_df_completely=False),
    'trigger_all_module_updates': trigger_all_module_updates,
    'get_selected_target_variable': lambda: app_state.selected_target_variable,
    'get_selected_target_variable_type': lambda: app_state.selected_target_variable_type,
    'update_target_variable_combo_items': update_target_variable_combo,
    'get_column_analysis_types': lambda: step_01_data_loading._type_selections.copy() if hasattr(step_01_data_loading, '_type_selections') else {},
    'step1_processing_complete': step1_processing_complete,
    'notify_eda_df_changed': lambda new_eda_df: setattr(app_state, 'current_df', new_eda_df),
    'get_eda_outlier_applied_flag': lambda: app_state._eda_outlier_settings_applied_once,
    'set_eda_outlier_applied_flag': lambda flag_val: setattr(app_state, '_eda_outlier_settings_applied_once', flag_val),
}

# DearPyGui 초기화
dpg.create_context()

with dpg.file_dialog(
    directory_selector=False, show=False, callback=file_load_callback,
    id="file_dialog_id", width=700, height=400, modal=True
):
    dpg.add_file_extension(".parquet")
    dpg.add_file_extension(".*")

setup_korean_font()

# 메인 윈도우 생성
with dpg.window(label="Data Analysis Platform", tag="main_window"):
    with dpg.group(horizontal=True):
        # 네비게이션 패널
        with dpg.child_window(width=280, tag="navigation_panel", border=True):
            dpg.add_text("Target Variable (y):")
            dpg.add_combo(
                items=[""], tag=TARGET_VARIABLE_COMBO_TAG, width=-1,
                callback=target_variable_selected_callback
            )

            dpg.add_text("Target Variable Type:", tag=TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            dpg.add_radio_button(
                items=["Categorical", "Continuous"],
                tag=TARGET_VARIABLE_TYPE_RADIO_TAG,
                horizontal=True,
                default_value=app_state.selected_target_variable_type,
                callback=target_variable_type_changed_callback,
                show=False
            )

            dpg.add_separator()
            dpg.add_spacer(height=5)
            dpg.add_text("Analysis Steps", color=[255, 255, 0])
            dpg.add_separator()
            
            for step_name_nav in ANALYSIS_STEPS:
                dpg.add_button(
                    label=step_name_nav, callback=switch_step_view,
                    user_data=step_name_nav, width=-1, height=30
                )
        
        # 콘텐츠 영역
        with dpg.child_window(tag="content_area", border=True):
            module_map = {
                ANALYSIS_STEPS[0]: step_01_data_loading,
                ANALYSIS_STEPS[1]: step_02_exploratory_data_analysis,
            }
            
            for step_name_create in ANALYSIS_STEPS:
                module = module_map.get(step_name_create)
                if module and hasattr(module, 'create_ui'):
                    module.create_ui(step_name_create, "content_area", main_app_callbacks)
                    print(f"UI created for '{step_name_create}'.")
                else:
                    fallback_tag = f"{step_name_create.lower().replace(' ', '_').replace('.', '').replace('&', 'and').replace('(', '').replace(')', '')}_fallback_group"
                    if not dpg.does_item_exist(fallback_tag):
                        main_app_callbacks['register_step_group_tag'](step_name_create, fallback_tag)
                        with dpg.group(tag=fallback_tag, parent="content_area", show=False):
                            dpg.add_text(f"--- {step_name_create} ---")
                            dpg.add_separator()
                            dpg.add_text(f"UI for '{step_name_create}' (fallback) will be configured here.")
                        main_app_callbacks['register_module_updater'](
                            step_name_create,
                            lambda df, mc, sn=step_name_create: print(f"Dummy updater for {sn} called.")
                        )

            if ANALYSIS_STEPS:
                first_step = ANALYSIS_STEPS[0]
                if not app_state.active_step_name:
                    if first_step in app_state.step_group_tags and \
                       dpg.does_item_exist(app_state.step_group_tags[first_step]):
                        switch_step_view(None, None, first_step)

# 뷰포트 설정 및 실행
dpg.create_viewport(title='Modular Data Analysis Platform GUI', width=1600, height=1000)
dpg.set_exit_callback(save_state_on_exit)
dpg.setup_dearpygui()
initial_load_on_startup()
dpg.maximize_viewport()
dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()