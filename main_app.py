# main_app.py
import dearpygui.dearpygui as dpg
import pandas as pd
import platform
import os

import utils
import step_01_data_loading
import step_02_exploratory_data_analysis
import traceback



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
        if selected_target_variable and current_df is not None and selected_target_variable not in current_df.columns:
            selected_target_variable = None
            if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        trigger_all_module_updates()
        return True
    else:
        _eda_sva_initialized = False # 실패 시에도 리셋
        update_target_variable_combo()
        selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        trigger_all_module_updates() 
        return False

# 타겟 변수 선택 콜백 (이전과 동일)
def target_variable_selected_callback(sender, app_data, user_data):
    global selected_target_variable, _eda_sva_initialized
    selected_target_variable = app_data
    print(f"Target variable selected: {selected_target_variable}")
    _eda_sva_initialized = False # 타겟 변경 시 SVA 다시 초기 실행하도록 유도 가능 (선택사항)
    trigger_all_module_updates()

# 파일 로드 콜백 (이전과 동일)
def file_load_callback(sender, app_data):
    selected_file_path = app_data.get('file_path_name')
    if selected_file_path: load_data_from_file(selected_file_path)
    else: print("File selection cancelled.")

# 원본 데이터로 리셋 함수 (이전과 거의 동일, _eda_sva_initialized 플래그 초기화 추가)
def reset_current_df_to_original_data():
    global current_df, original_df, selected_target_variable, _eda_sva_initialized
    if original_df is not None:
        current_df = original_df.copy()
        print("Data reset to original.")
        if hasattr(step_01_data_loading, '_type_selections'): step_01_data_loading._type_selections.clear()
        if hasattr(step_01_data_loading, '_imputation_selections'): step_01_data_loading._imputation_selections.clear()
        
        _eda_sva_initialized = False # 리셋 시 SVA 초기화 플래그 리셋
        update_target_variable_combo()
        if selected_target_variable and current_df is not None and selected_target_variable not in current_df.columns:
            selected_target_variable = None
            if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        trigger_all_module_updates()
    else:
        print("No original data to reset to.")

# 스텝 UI 전환 함수 (SVA 초기 실행 로직 추가)
def switch_step_view(sender, app_data, user_data_step_name: str):
    global active_step_name, _eda_sva_initialized
    print(f"Attempting to switch to step: {user_data_step_name}")

    for step_name_iter, group_tag in step_group_tags.items(): # step_name 변수명 변경
        if dpg.does_item_exist(group_tag):
            if step_name_iter == user_data_step_name:
                dpg.show_item(group_tag)
                active_step_name = step_name_iter
                trigger_specific_module_update(step_name_iter) # 일반적인 모듈 업데이트 호출

                # EDA 탭으로 처음 전환되거나, SVA가 아직 초기화되지 않았을 때 SVA 자동 실행
                # if step_name_iter == "2. Exploratory Data Analysis (EDA)" and not _eda_sva_initialized:
                #     print("EDA tab selected: Triggering initial SVA run.")
                #     if hasattr(step_02_exploratory_data_analysis, '_apply_sva_filters_and_run'):
                #         # main_app_callbacks를 통해 EDA 모듈의 함수를 직접 호출
                #         # 또는 EDA 모듈의 update_ui 함수가 이 초기 실행 로직을 포함하도록 할 수도 있음
                #         # 여기서는 EDA 모듈의 함수를 직접 호출한다고 가정 (main_callbacks 전달)
                #         step_02_exploratory_data_analysis._apply_sva_filters_and_run(main_app_callbacks)
                #         _eda_sva_initialized = True
            else:
                dpg.hide_item(group_tag)
    print(f"Active step: {active_step_name}")

# 특정 모듈 UI 업데이트 (이전과 동일)
def trigger_specific_module_update(module_name_key: str):
    if module_name_key in module_ui_updaters:
        updater = module_ui_updaters[module_name_key]
        # updater는 (current_df, main_callbacks) 또는 (current_df, original_df, util_funcs, file_path, main_callbacks) 등
        # 모듈의 update_ui 시그니처에 맞춰서 호출해야 함.
        # step_02_exploratory_data_analysis.update_ui는 (current_df, main_callbacks)를 받음
        if module_name_key == "2. Exploratory Data Analysis (EDA)":
            updater(current_df, main_app_callbacks) # EDA 모듈의 update_ui 호출 방식
        else: # 다른 모듈은 기존 방식 (step_01_data_loading 등)
             updater(current_df, original_df, util_functions_for_modules, loaded_file_path)
        print(f"Module UI updated for: '{module_name_key}'")
    else:
        print(f"Warning: No UI updater found for '{module_name_key}'.")

# 모든 모듈 UI 업데이트 (이전과 동일)
def trigger_all_module_updates():
    global _eda_sva_initialized # 모든 모듈 업데이트 시 SVA 초기화 플래그도 리셋 고려
    # _eda_sva_initialized = False # 데이터가 크게 변경될 수 있으므로 리셋
    print("Updating all module UIs...")
    for step_name_key in list(module_ui_updaters.keys()): # 키 리스트 복사 후 반복
        trigger_specific_module_update(step_name_key)


# --- 유틸리티 함수 딕셔너리 (이전과 동일) ---
util_functions_for_modules = {
    'create_table_with_data': utils.create_table_with_data,
    'calculate_column_widths': utils.calculate_column_widths,
    'format_text_for_display': utils.format_text_for_display,
    'get_safe_text_size': utils.get_safe_text_size,
}

# --- 사용자 지정 분석용 타입 정보 가져오는 함수 ---
def get_column_analysis_types_from_user_settings():
    """
    step_01_data_loading 모듈에서 사용자가 최종 선택한 
    각 컬럼의 "분석용 타입" 정보를 반환합니다.
    실제 구현에서는 step_01_data_loading._type_selections를 참조해야 합니다.
    """
    if hasattr(step_01_data_loading, '_type_selections') and \
       isinstance(step_01_data_loading._type_selections, dict):
        # print(f"DEBUG: Returning analysis types from _type_selections: {step_01_data_loading._type_selections}")
        return step_01_data_loading._type_selections.copy() 
    else:
        print("Warning: User-defined analysis types (_type_selections) not found in step_01_data_loading.")
        # 타입 편집 전이거나 정보가 없는 경우, 현재 df의 Dtype을 기본으로 사용하거나 빈 dict 반환
        # EDA 필터에서 dtypes를 fallback으로 사용하도록 유도
        if current_df is not None:
            # 임시로 Dtype을 반환 (이상적으로는 step_01에서 타입 편집 완료 후 EDA 진입)
            # return {col: str(current_df[col].dtype) for col in current_df.columns}
            # 또는 _infer_series_type을 모든 컬럼에 적용한 결과를 반환할 수도 있으나,
            # 이는 사용자의 명시적 타입 지정을 무시할 수 있음.
            # 여기서는 빈 dict를 반환하여, _get_filtered_variables 내부에서 dtypes를 사용하도록 함.
             return {col: step_02_exploratory_data_analysis._infer_series_type(current_df[col])[0] for col in current_df.columns} # 추론 타입 사용
        return {}


# --- 메인 앱 콜백 딕셔너리 ---
main_app_callbacks = {
    'get_current_df': lambda: current_df,
    'get_original_df': lambda: original_df,
    'get_loaded_file_path': lambda: loaded_file_path,
    'get_util_funcs': lambda: util_functions_for_modules,
    'show_file_dialog': lambda: dpg.show_item("file_dialog_id"),
    'register_step_group_tag': lambda name, tag: step_group_tags.update({name: tag}),
    'register_module_updater': lambda name, func: module_ui_updaters.update({name: func}),
    'trigger_module_update': trigger_specific_module_update, # 특정 모듈 업데이트 요청
    'reset_current_df_to_original': reset_current_df_to_original_data,
    'trigger_all_module_updates': trigger_all_module_updates, 
    'get_selected_target_variable': lambda: selected_target_variable,
    'update_target_variable_combo_items': update_target_variable_combo,
    'get_column_analysis_types': get_column_analysis_types_from_user_settings, # 추가된 콜백
}


# --- DPG 초기화 및 실행 ---
dpg.create_context()
setup_korean_font() # 폰트 설정

# File Dialog (이전과 동일)
with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback,
                     id="file_dialog_id", width=700, height=400, modal=True):
    dpg.add_file_extension(".parquet"); dpg.add_file_extension(".*")

# Main Window (이전과 동일, 타겟 변수 UI 위치 수정 반영)
with dpg.window(label="Data Analysis Platform", tag="main_window"):
    with dpg.group(horizontal=True):
        with dpg.child_window(width=280, tag="navigation_panel", border=True):
            dpg.add_text("Target Variable (y):")
            dpg.add_combo(items=[""], tag=TARGET_VARIABLE_COMBO_TAG, width=-1,
                          callback=target_variable_selected_callback)
            dpg.add_separator(); dpg.add_spacer(height=5)
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
                    # EDA 모듈의 create_ui는 main_callbacks만 받도록 시그니처 통일 시도
                    # 현재 step_02_exploratory_data_analysis.create_ui는 (step_name, parent_tag, main_callbacks)를 받음
                    module.create_ui(step_name_create, "content_area", main_app_callbacks)
                    print(f"UI created for '{step_name_create}'.")
                else: # Fallback
                    fallback_tag = f"{step_name_create.lower().replace(' ', '_').replace('.', '').replace('&', 'and')}_fallback_group"
                    if not dpg.does_item_exist(fallback_tag): # 중복 생성 방지
                        main_app_callbacks['register_step_group_tag'](step_name_create, fallback_tag)
                        with dpg.group(tag=fallback_tag, parent="content_area", show=False):
                            dpg.add_text(f"--- {step_name_create} ---"); dpg.add_separator()
                            dpg.add_text(f"UI for '{step_name_create}' will be configured here.")
                        main_app_callbacks['register_module_updater'](step_name_create, lambda *args, **kwargs: None)

            if ANALYSIS_STEPS:
                first_step = ANALYSIS_STEPS[0]
                # Ensure first step UI is shown after all UIs are created
                # dpg.split_frame() # <--- 이 줄을 주석 처리하거나 삭제합니다.
                if first_step in step_group_tags and dpg.does_item_exist(step_group_tags[first_step]):
                     switch_step_view(None, None, first_step)
                else:
                    print(f"Warning: First step '{first_step}' UI group not found immediately after creation.") #

dpg.create_viewport(title='Modular Data Analysis Platform GUI', width=1440, height=1000) #
dpg.setup_dearpygui() #
dpg.maximize_viewport()
dpg.show_viewport() #
dpg.set_primary_window("main_window", True) #
dpg.start_dearpygui() #
dpg.destroy_context() #