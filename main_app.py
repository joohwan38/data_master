# main_app.py
import dearpygui.dearpygui as dpg
import pandas as pd
import platform # Moved to top
import os       # Moved to top

import utils
import step_01_data_loading
import step_02_exploratory_data_analysis

current_df: pd.DataFrame = None
original_df: pd.DataFrame = None
loaded_file_path: str = None

step_group_tags = {}
module_ui_updaters = {}
active_step_name: str = None
selected_target_variable: str = None # 전역 타겟 변수
TARGET_VARIABLE_COMBO_TAG = "target_variable_combo"

ANALYSIS_STEPS = [
    "1. Data Loading and Overview",
    "2. Exploratory Data Analysis (EDA)",
    # "3. Data Quality Assessment", # To be added later
    # "4. Data Preprocessing",   # To be added later
]

def setup_korean_font():
    font_path = None
    font_size = 17
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
    # (다른 OS 로직은 생략 또는 유지)
    elif os_type == "Windows":
        print("Windows: OS detected, skipping macOS/Linux font search for this focused fix.")
    elif os_type == "Linux":
        print("Linux: OS detected, skipping macOS/Windows font search for this focused fix.")
    else:
        print(f"Unsupported OS for specific font setup: {os_type}.")

    if font_path and os.path.exists(font_path):
        print(f"Attempting to load and bind font: '{font_path}' with size {font_size}")
        try:
            font_registry_tag = "global_font_registry_unique" # 태그를 조금 더 유니크하게 변경 (선택 사항)
            font_to_bind_tag = "korean_apple_font"      # 태그를 조금 더 유니크하게 변경 (선택 사항)

            # 1. 폰트 레지스트리가 없으면 생성
            if not dpg.does_item_exist(font_registry_tag):
                dpg.add_font_registry(tag=font_registry_tag)
                print(f"Font registry '{font_registry_tag}' created.")
            else:
                print(f"Font registry '{font_registry_tag}' already exists.")

            # 2. 폰트 추가 (with 구문 없이, parent 인자 사용)
            #    폰트 태그가 이미 존재하면 오류가 발생하므로, 없을 때만 추가하거나, 삭제 후 추가해야 합니다.
            #    이 함수는 앱 시작 시 한 번만 호출되는 것이 가장 이상적입니다.
            if not dpg.does_item_exist(font_to_bind_tag):
                dpg.add_font(
                    file=font_path,
                    size=font_size,
                    tag=font_to_bind_tag,
                    parent=font_registry_tag # 명시적으로 부모 레지스트리 지정
                )
                # 3. 추가된 폰트에 대해 글리프 범위 설정 (parent 인자 사용)
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean, parent=font_to_bind_tag)
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default, parent=font_to_bind_tag)
                print(f"Font '{font_path}' added with tag '{font_to_bind_tag}' to registry '{font_registry_tag}'.")
            else:
                print(f"Font with tag '{font_to_bind_tag}' already exists. Attempting to use existing.")

            # 4. 폰트 바인딩
            dpg.bind_font(font_to_bind_tag)
            print(f"Successfully attempted to bind font '{font_to_bind_tag}' (from {font_path}).")
        except Exception as e:
            # 여기서 발생하는 예외는 DPG 내부 C++ 레벨 오류와 다를 수 있습니다.
            print(f"Error during explicit font processing for '{font_path}': {e}. DPG default font will be used.")
            import traceback
            traceback.print_exc() # 파이썬 예외의 전체 스택 트레이스 출력
    elif font_path and not os.path.exists(font_path):
        print(f"Font path '{font_path}' was determined, but the file does not exist. DPG default font will be used.")
    else:
        print("No suitable Korean font path was found for macOS. DPG default font will be used.")
    
    print(f"--- Font Setup Finished ---")


def load_data_from_file(file_path: str) -> bool:
    global current_df, original_df, loaded_file_path, selected_target_variable # selected_target_variable 추가
    success = False # 성공 여부 플래그 초기화
    try:
        current_df = pd.read_parquet(file_path)
        original_df = current_df.copy()
        loaded_file_path = file_path
        print(f"Data loaded successfully: {file_path}, Shape: {current_df.shape}")
        success = True # 데이터 로드 성공
    except Exception as e:
        current_df = None
        original_df = None
        loaded_file_path = None
        print(f"Error loading data: {e}")
        if dpg.does_item_exist(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT): # step_01_data_loading 모듈의 태그 사용
            dpg.set_value(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT, f"Error: {e}")
        success = False # 데이터 로드 실패

    # try-except 블록이 끝난 후, 성공 여부에 따라 후속 처리
    if success:
        # 데이터 로드 성공 시 step_01_data_loading 모듈의 전역 변수 초기화
        if hasattr(step_01_data_loading, '_type_selections'):
            step_01_data_loading._type_selections.clear()
        if hasattr(step_01_data_loading, '_imputation_selections'):
            step_01_data_loading._imputation_selections.clear()
        
        # _custom_nan_input_value 초기화 및 UI 업데이트 (필요시 주석 해제)
        # if hasattr(step_01_data_loading, '_custom_nan_input_value'):
        #     step_01_data_loading._custom_nan_input_value = "" 
        # if dpg.does_item_exist(step_01_data_loading.TAG_DL_CUSTOM_NAN_INPUT):
        #    dpg.set_value(step_01_data_loading.TAG_DL_CUSTOM_NAN_INPUT, "")

        # 타겟 변수 콤보 박스 아이템 업데이트 (새로운 컬럼 목록으로)
        if callable(update_target_variable_combo): # 함수 존재 여부 확인
            update_target_variable_combo()
        
        # 이전에 선택된 타겟 변수가 새 데이터프레임에 여전히 존재하는지 확인
        if selected_target_variable and current_df is not None and selected_target_variable not in current_df.columns:
            print(f"Previously selected target variable '{selected_target_variable}' not found in the new data. Resetting target variable.")
            selected_target_variable = None
            # UI의 콤보박스 값도 초기화 (선택 안 함 상태로)
            if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
                dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        
        trigger_all_module_updates() # 모든 모듈 UI 업데이트
        return True
    else:
        # 데이터 로드 실패 시에도 타겟 변수 콤보 박스는 빈 목록으로 업데이트
        if callable(update_target_variable_combo): # 함수 존재 여부 확인
            update_target_variable_combo()
        # 실패 시에도 selected_target_variable 초기화
        selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
             dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")

        trigger_all_module_updates() 
        return False
    
def target_variable_selected_callback(sender, app_data, user_data):
    """타겟 변수 콤보박스 선택 시 호출될 콜백"""
    global selected_target_variable
    selected_target_variable = app_data # app_data가 선택된 컬럼명
    print(f"Target variable selected: {selected_target_variable}")
    # 타겟 변수 변경에 따른 UI 및 분석 업데이트 트리거
    # 예: trigger_specific_module_update("2. Exploratory Data Analysis (EDA)")
    # 또는 trigger_all_module_updates()
    trigger_all_module_updates()

def update_target_variable_combo():
    """데이터프레임 로드/변경 시 타겟 변수 콤보박스 아이템 업데이트"""
    global current_df
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        if current_df is not None and not current_df.empty:
            items = [""] + list(current_df.columns) # 첫 번째는 "선택 안 함" 옵션
            dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=items)
            # 이전에 선택된 타겟이 현재 컬럼 목록에 여전히 존재하면 그 값을 유지
            if selected_target_variable and selected_target_variable in current_df.columns:
                dpg.set_value(TARGET_VARIABLE_COMBO_TAG, selected_target_variable)
            else: # 그렇지 않으면 "선택 안 함" 또는 첫 번째 컬럼 등으로 초기화
                dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "") # "선택 안 함"
        else:
            dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=[""])
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")

def file_load_callback(sender, app_data):
    selected_file_path = app_data.get('file_path_name')
    if selected_file_path:
        load_data_from_file(selected_file_path)
    else:
        print("File selection cancelled.")

def reset_current_df_to_original_data():
    global current_df, original_df, selected_target_variable
    if original_df is not None:
        current_df = original_df.copy()
        print("Data reset to original.")
        # ... (기존 _type_selections 등 초기화 로직) ...
        update_target_variable_combo() # 컬럼 목록이 원본으로 돌아갔으므로 호출
        # 원본 데이터에 이전 타겟이 여전히 있는지 확인
        if selected_target_variable and selected_target_variable not in current_df.columns:
            selected_target_variable = None
            print("Previously selected target variable not found in original data after reset. Resetting target.")
            if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
                dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")

        trigger_all_module_updates()
    else:
        print("No original data to reset to.")

def switch_step_view(sender, app_data, user_data_step_name: str):
    global active_step_name
    print(f"Attempting to switch to step: {user_data_step_name}")

    for step_name, group_tag in step_group_tags.items():
        if dpg.does_item_exist(group_tag):
            if step_name == user_data_step_name:
                dpg.show_item(group_tag)
                active_step_name = step_name
                trigger_specific_module_update(step_name) 
            else:
                dpg.hide_item(group_tag)
    print(f"Active step: {active_step_name}")

def trigger_specific_module_update(module_name_key: str):
    if module_name_key in module_ui_updaters:
        updater = module_ui_updaters[module_name_key]
        updater(current_df, original_df, util_functions_for_modules, loaded_file_path)
        print(f"Module UI updated for: '{module_name_key}'")
    else:
        print(f"Warning: No UI updater found for '{module_name_key}'.")

def trigger_all_module_updates():
    print("Updating all module UIs...")
    for step_name_key in module_ui_updaters.keys():
        trigger_specific_module_update(step_name_key)

util_functions_for_modules = {
    'create_table_with_data': utils.create_table_with_data,
    'calculate_column_widths': utils.calculate_column_widths,
    'format_text_for_display': utils.format_text_for_display,
    'get_safe_text_size': utils.get_safe_text_size,
}

main_app_callbacks = {
    'get_current_df': lambda: current_df,
    'get_original_df': lambda: original_df,
    'get_loaded_file_path': lambda: loaded_file_path,
    'get_util_funcs': lambda: util_functions_for_modules,
    'show_file_dialog': lambda: dpg.show_item("file_dialog_id"),
    'register_step_group_tag': lambda name, tag: step_group_tags.update({name: tag}),
    'register_module_updater': lambda name, func: module_ui_updaters.update({name: func}),
    'trigger_module_update': trigger_specific_module_update,
    'reset_current_df_to_original': reset_current_df_to_original_data, # 이 부분 추가
    'trigger_all_module_updates': trigger_all_module_updates, # 모든 모듈 업데이트 트리거 (추가 또는 확인)
    'get_selected_target_variable': lambda: selected_target_variable,
    'update_target_variable_combo_items': update_target_variable_combo
}

dpg.create_context()

# >>>>> 중요: 여기에 setup_korean_font() 함수 호출 추가 <<<<<
setup_korean_font()
# >>>>> 여기까지 <<<<<


with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback,
                     id="file_dialog_id", width=700, height=400, modal=True):
    dpg.add_file_extension(".parquet")
    dpg.add_file_extension(".*")

with dpg.window(label="Data Analysis Platform", tag="main_window"):
    # 전체 창을 가로로 나누는 메인 그룹
    with dpg.group(horizontal=True):
        # --- 수정된 Navigation Panel ---
        with dpg.child_window(width=280, tag="navigation_panel", border=True):
            # 1. 타겟 변수 선택 섹션 (패널 최상단)
            dpg.add_text("Target Variable (y):") # 1행: 레이블
            dpg.add_combo(items=[""], # 초기 아이템은 update_target_variable_combo에서 채워짐
                          tag=TARGET_VARIABLE_COMBO_TAG, 
                          width=-1, # 부모 너비에 맞춤
                          callback=target_variable_selected_callback) # 2행: 드롭다운
            
            # (선택 사항) 타겟 변수 클리어 버튼
            # dpg.add_button(label="Clear Target", width=-1, 
            #                callback=lambda: (
            #                    globals().update(selected_target_variable=None), 
            #                    dpg.set_value(TARGET_VARIABLE_COMBO_TAG, ""), 
            #                    trigger_all_module_updates()
            #                ))
            
            dpg.add_separator() # 3행: 구분자
            dpg.add_spacer(height=5) # 구분자와 다음 텍스트 사이 간격

            # 4. 기존 Analysis Steps 섹션
            dpg.add_text("Analysis Steps", color=[255, 255, 0])
            dpg.add_separator()
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
                    main_app_callbacks['register_step_group_tag'](step_name_create, fallback_tag)
                    with dpg.group(tag=fallback_tag, parent="content_area", show=False):
                        dpg.add_text(f"--- {step_name_create} ---")
                        dpg.add_separator()
                        dpg.add_text(f"UI for '{step_name_create}' will be configured here.")
                    main_app_callbacks['register_module_updater'](step_name_create, lambda *args, **kwargs: None)

            if ANALYSIS_STEPS:
                first_step = ANALYSIS_STEPS[0]
                if first_step in step_group_tags: 
                     switch_step_view(None, None, first_step)
                else: 
                    print(f"Warning: First step '{first_step}' not immediately found in step_group_tags during init. May show after full DPG loop.")


dpg.create_viewport(title='Modular Data Analysis Platform GUI', width=1440, height=1200) # Height corrected from your original 900 to 1200 as per my earlier code, adjust if needed
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()