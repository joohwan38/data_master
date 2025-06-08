# main_app.py (수정 완료된 코드)

import dearpygui.dearpygui as dpg
import pandas as pd
import platform
import os
import traceback
import hashlib
import json
import datetime
import subprocess

# 유틸리티 및 각 Step 모듈 임포트
import utils
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
from app_state_manager import app_state, AppState, BaseNode

# --- 상수 및 헬퍼 함수들 ---
SETTINGS_DIR_NAME = ".app_file_settings"
SESSION_INFO_FILE = os.path.join(SETTINGS_DIR_NAME, "session_info.json")
SVA_STEP_KEY = "2a. Single Variable Analysis"
MVA_STEP_KEY = "2b. Multivariate Analysis"
TARGET_VARIABLE_TYPE_RADIO_TAG = "main_target_variable_type_radio"
TARGET_VARIABLE_TYPE_LABEL_TAG = "main_target_variable_type_label"
TARGET_VARIABLE_COMBO_TAG = "main_target_variable_combo"
MAIN_FILE_PATH_DISPLAY_TAG = "main_file_path_display_text"
_MODAL_ID_SIMPLE_MESSAGE = "main_simple_modal_message_id"

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
]

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
            button_width = 100
            spacer_w = (width - button_width - 16) / 2
            dpg.add_spacer(width=int(spacer_w if spacer_w > 0 else 0))
            dpg.add_button(label="OK", width=button_width, callback=lambda: dpg.configure_item(_MODAL_ID_SIMPLE_MESSAGE, show=False))

def add_ai_log_message(message: str, chart_context: str = "", mode: str = "new_log_entry"):
    if not dpg.is_dearpygui_running(): return
    log_panel_tag = "ai_analysis_log_panel_text"
    log_container_window_tag = "ai_analysis_log_panel"
    if not dpg.does_item_exist(log_panel_tag) or not dpg.does_item_exist(log_container_window_tag): return
    current_full_log = dpg.get_value(log_panel_tag)
    default_placeholder_msg = "AI 분석 결과가 여기에 표시됩니다.\n"
    if current_full_log == default_placeholder_msg or not current_full_log.strip():
        current_full_log = ""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    standard_separator = "\n\n" + "-"*30 + "\n"
    if mode == "stream_start_entry" or mode == "new_log_entry":
        entry_header = f"[{timestamp}] [{chart_context}]\n" if chart_context else f"[{timestamp}]\n"
        updated_log = current_full_log + standard_separator + entry_header + message if current_full_log else entry_header + message
    else: # stream_chunk_append
        updated_log = current_full_log + message
    dpg.set_value(log_panel_tag, updated_log)
    max_log_length = 30000
    if len(updated_log) > max_log_length:
        cutoff_point = updated_log.find(standard_separator, len(updated_log) - max_log_length)
        if cutoff_point != -1:
            updated_log = "...(오래된 로그 일부 잘림)..." + updated_log[cutoff_point:]
        else:
            updated_log = "...(오래된 로그 일부 잘림)...\n" + updated_log[-max_log_length:]
        dpg.set_value(log_panel_tag, updated_log)
    dpg.set_y_scroll(log_container_window_tag, -1.0)

def setup_korean_font():
    font_path, font_size, os_type = None, 17, platform.system()
    font_paths = {
        "Darwin": ["/System/Library/Fonts/AppleSDGothicNeo.ttc"],
        "Windows": ["C:/Windows/Fonts/malgun.ttf"],
        "Linux": ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
    }
    for p in font_paths.get(os_type, []):
        if os.path.exists(p):
            font_path = p
            break
    if font_path:
        with dpg.font_registry():
            font_id = dpg.add_font(font_path, font_size, tag="korean_font_for_app")
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean, parent=font_id)
            dpg.bind_font("korean_font_for_app")

# <<< NEW: 파이프라인 관리 클래스 >>>
class PipelineManager:
    def __init__(self, state: AppState):
        self.state = state

    def execute_pipeline_to_node(self, target_node_id: int):
        print(f"--- Pipeline execution requested for node: {target_node_id} ---")
        try:
            execution_order = self._get_execution_order(target_node_id)
            print(f"Execution Order: {execution_order}")
        except Exception as e:
            _show_simple_modal_message("Pipeline Error", f"Failed to determine execution order: {e}")
            return

        for node_id in execution_order:
            node = self.state.nodes.get(node_id)
            if not node: continue

            inputs = {}
            # BaseNode 클래스로 변경되면서 input_connections 구조가 바뀜.
            # 각 노드 클래스에서 필요한 입력을 정의해야 함 (예: 'DataFrame In')
            for input_name, source_node_id in node.input_connections.items():
                if source_node_id in self.state.node_outputs:
                    inputs[input_name] = self.state.node_outputs[source_node_id]
                else:
                    msg = f"Input from node {source_node_id} is not available for node {node_id}."
                    _show_simple_modal_message("Pipeline Error", msg)
                    return
            try:
                if node.type == "data_input":
                    if self.state.original_df is None:
                        raise ValueError("Original DataFrame is not loaded.")
                    output_df = self.state.original_df
                else:
                    output_df = node.process(inputs)
                self.state.node_outputs[node_id] = output_df
                print(f"Successfully processed Node {node_id} ({node.type}). Output shape: {output_df.shape}")
            except Exception as e:
                msg = f"Error processing Node {node_id} ({node.type}):\n{e}"
                traceback.print_exc()
                _show_simple_modal_message("Pipeline Execution Error", msg, width=500, height=250)
                return

        _show_simple_modal_message("Success", f"Pipeline executed successfully up to Node {target_node_id}.")

    def _get_execution_order(self, target_node_id: int) -> list[int]:
        order = []
        visited = set()
        recursion_stack = set()

        def visit(node_id):
            if node_id in recursion_stack:
                raise Exception("Circular dependency detected in the node graph.")
            if node_id in visited:
                return
            
            recursion_stack.add(node_id)
            node = self.state.nodes.get(node_id)
            if not node:
                raise ValueError(f"Node with ID {node_id} not found in graph.")

            for parent_id in node.input_connections.values():
                visit(parent_id)

            recursion_stack.remove(node_id)
            visited.add(node_id)
            order.append(node_id)

        visit(target_node_id)
        return order

app_state = AppState()
pipeline_manager = PipelineManager(app_state)


def load_data_from_file(file_path: str) -> bool:
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_extension == ".csv":
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='cp949')
        else:
            _show_simple_modal_message("File Type Error", "Unsupported file type: .parquet or .csv only.")
            return False

        reset_application_state(clear_df_completely=True)
        app_state.original_df = df
        app_state.loaded_file_path = file_path

        dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, f"File: {os.path.basename(file_path)} (Shape: {df.shape})")
        
        # 여기서 step_03_preprocessing의 `_add_node_to_editor("data_input", ...)`를 직접 호출하여
        # UI에 자동으로 입력 노드를 추가하고, app_state.node_outputs에 원본 DF를 넣는 것이 이상적입니다.
        # 지금은 step3 모듈이 UI를 그릴 때 초기 노드를 생성하도록 의존합니다.
        
        trigger_specific_module_update(ANALYSIS_STEPS[0]) # Update Step 1 summary
        update_target_variable_combo()

        return True
    except Exception as e:
        reset_application_state(clear_df_completely=True)
        _show_simple_modal_message("File Load Error", f"Failed to load data: {e}")
        traceback.print_exc()
        return False

def reset_application_state(clear_df_completely=True):
    if clear_df_completely:
        app_state.original_df = None
        app_state.loaded_file_path = None
        if dpg.does_item_exist(MAIN_FILE_PATH_DISPLAY_TAG):
            dpg.set_value(MAIN_FILE_PATH_DISPLAY_TAG, "No data loaded.")

    app_state.nodes.clear()
    app_state.node_outputs.clear()
    app_state.links.clear()
    app_state.node_id_counter = 0
    app_state.selected_node_id = None
    app_state.selected_target_variable = None
    app_state.selected_target_variable_type = "Continuous"
    
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)

    # 각 모듈의 UI 초기화
    if hasattr(step_01_data_loading, 'reset_step1_state'): step_01_data_loading.reset_step1_state()
    if hasattr(step_02a_sva, 'reset_sva_ui_defaults'): step_02a_sva.reset_sva_ui_defaults()
    if hasattr(step_02b_mva, 'reset_mva_ui_defaults'): step_02b_mva.reset_mva_ui_defaults()
    if hasattr(step_03_preprocessing, 'reset_preprocessing_state'): step_03_preprocessing.reset_preprocessing_state()
    if hasattr(step_04_missing_values, 'reset_missing_values_state'): step_04_missing_values.reset_missing_values_state()
    if hasattr(step_05_outlier_treatment, 'reset_outlier_treatment_state'): step_05_outlier_treatment.reset_outlier_treatment_state()
    if hasattr(step_06_standardization, 'reset_step6_state'): step_06_standardization.reset_step6_state()
    if hasattr(step_07_feature_engineering, 'reset_state'): step_07_feature_engineering.reset_state()
    if hasattr(step_08_derivation, 'reset_state'): step_08_derivation.reset_state()
    if hasattr(step_09_data_viewer, 'reset_state'): step_09_data_viewer.reset_state()
    
    # 기본 뷰를 노드 에디터로 설정
    switch_step_view(None, None, ANALYSIS_STEPS[2])


def update_target_variable_combo():
    # 타겟 변수 목록은 항상 원본 데이터프레임 기준
    df_for_combo = app_state.original_df
    items = [""] + list(df_for_combo.columns) if df_for_combo is not None else [""]
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=items)
        if app_state.selected_target_variable in items:
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, app_state.selected_target_variable)
        else:
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")


def target_variable_selected_callback(sender, app_data, user_data):
    app_state.selected_target_variable = app_data
    if app_data:
        # 타입 추측은 원본 DF 기준
        df_for_guessing = app_state.original_df
        guessed_type = "Continuous"
        if df_for_guessing is not None and app_data in df_for_guessing.columns:
             guessed_type = utils._guess_target_type(df_for_guessing, app_data)
        app_state.selected_target_variable_type = guessed_type
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, guessed_type)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)
    else:
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)


def target_variable_type_changed_callback(sender, app_data, user_data):
    app_state.selected_target_variable_type = app_data


def switch_step_view(sender, app_data, user_data_step_name: str):
    for step_name_iter, group_tag in app_state.step_group_tags.items():
        if dpg.does_item_exist(group_tag):
            is_active_step = (step_name_iter == user_data_step_name)
            dpg.configure_item(group_tag, show=is_active_step)
            if is_active_step:
                app_state.active_step_name = step_name_iter
                trigger_specific_module_update(step_name_iter)

def trigger_specific_module_update(module_name_key: str):
    if module_name_key in app_state.module_ui_updaters:
        updater = app_state.module_ui_updaters[module_name_key]
        try:
            # 이제 모든 모듈의 update_ui는 node_id를 받거나, app_state를 통해
            # 필요한 정보를 스스로 찾아야 합니다.
            # 지금은 main_callbacks를 통해 app_state에 접근하도록 설계.
            if module_name_key == ANALYSIS_STEPS[0]: # Step 1은 특별 취급
                 updater(app_state.original_df, app_state.node_outputs.get(app_state.selected_node_id), util_functions_for_modules, app_state.loaded_file_path)
            else:
                 updater() # 다른 모듈들은 콜백을 통해 필요한 정보를 얻음
        except Exception as e:
            print(f"Error updating UI for {module_name_key}: {e}")
            traceback.print_exc()

util_functions_for_modules = {
    'create_table_with_data': utils.create_table_with_data,
    'calculate_column_widths': utils.calculate_column_widths,
    'format_text_for_display': utils.format_text_for_display,
    'get_safe_text_size': utils.get_safe_text_size,
    '_show_simple_modal_message': _show_simple_modal_message,
    'show_dpg_alert_modal': utils.show_dpg_alert_modal,
    'show_confirmation_modal': utils.show_confirmation_modal,
    'get_numeric_cols': utils._get_numeric_cols,
    'get_categorical_cols': utils._get_categorical_cols,
    'calculate_cramers_v': utils.calculate_cramers_v,
    'calculate_feature_target_relevance': utils.calculate_feature_target_relevance,
    'plot_to_dpg_texture': utils.plot_to_dpg_texture,
    'create_table_with_large_data_preview': utils.create_table_with_large_data_preview,
}

main_app_callbacks = {
    'get_app_state': lambda: app_state,
    'get_pipeline_manager': lambda: pipeline_manager,
    'get_util_funcs': lambda: util_functions_for_modules,
    'register_step_group_tag': lambda name, tag: app_state.step_group_tags.update({name: tag}),
    'register_module_updater': lambda name, func: app_state.module_ui_updaters.update({name: func}),
    'trigger_module_update': trigger_specific_module_update,
    'get_selected_target_variable': lambda: app_state.selected_target_variable,
    'get_selected_target_variable_type': lambda: app_state.selected_target_variable_type,
    'add_ai_log': add_ai_log_message,
}

dpg.create_context()
dpg.add_texture_registry(tag="primary_texture_registry", show=False)
with dpg.file_dialog(directory_selector=False, show=False, callback=load_data_from_file, id="file_dialog_id", width=700, height=400, modal=True):
    dpg.add_file_extension(".parquet"); dpg.add_file_extension(".csv")

setup_korean_font()

with dpg.window(label="Data Analysis Platform", tag="main_window"):
    with dpg.group(horizontal=True):
        dpg.add_button(label="Open File", callback=lambda: dpg.show_item("file_dialog_id"), width=130, height=30)
        dpg.add_button(label="Reset All State", callback=lambda: reset_application_state(clear_df_completely=True), width=150, height=30)
        dpg.add_text("No data loaded.", tag=MAIN_FILE_PATH_DISPLAY_TAG, wrap=-1)
    dpg.add_separator()
    
    with dpg.group(horizontal=True, tag="main_layout_group"):
        with dpg.child_window(width=280, tag="navigation_panel", border=True):
            dpg.add_text("Target Variable (y):")
            dpg.add_combo(items=[""], tag=TARGET_VARIABLE_COMBO_TAG, width=-1, callback=target_variable_selected_callback)
            dpg.add_text("Target Variable Type:", tag=TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            dpg.add_radio_button(items=["Categorical", "Continuous"], tag=TARGET_VARIABLE_TYPE_RADIO_TAG, horizontal=True,
                                 default_value=app_state.selected_target_variable_type, callback=target_variable_type_changed_callback, show=False)
            dpg.add_separator(); dpg.add_spacer(height=5)
            dpg.add_text("Analysis Steps / Views", color=[255,255,0]); dpg.add_separator()
            for step_name_nav in ANALYSIS_STEPS:
                dpg.add_button(label=step_name_nav, callback=switch_step_view, user_data=step_name_nav, width=-1, height=30)

        with dpg.child_window(tag="content_area", border=True, width=-(400)):
            # 각 모듈 UI 생성
            step_01_data_loading.create_ui(ANALYSIS_STEPS[0], "content_area", main_app_callbacks)
            
            # EDA 탭 그룹 생성
            with dpg.group(tag="eda_step_main_group", parent="content_area", show=False):
                dpg.add_text(f"--- {ANALYSIS_STEPS[1]} ---"); dpg.add_separator()
                with dpg.tab_bar():
                    with dpg.tab(label="Single Variable"): step_02a_sva.create_ui(SVA_STEP_KEY, dpg.last_item(), main_app_callbacks)
                    with dpg.tab(label="Multi-Variable"): step_02b_mva.create_ui(MVA_STEP_KEY, dpg.last_item(), main_app_callbacks)
            
            step_03_preprocessing.create_ui(ANALYSIS_STEPS[2], "content_area", main_app_callbacks)
            step_04_missing_values.create_ui(ANALYSIS_STEPS[3], "content_area", main_app_callbacks)
            step_05_outlier_treatment.create_ui(ANALYSIS_STEPS[4], "content_area", main_app_callbacks)
            step_06_standardization.create_ui(ANALYSIS_STEPS[5], "content_area", main_app_callbacks)
            step_07_feature_engineering.create_ui(ANALYSIS_STEPS[6], "content_area", main_app_callbacks)
            step_08_derivation.create_ui(ANALYSIS_STEPS[7], "content_area", main_app_callbacks)
            step_09_data_viewer.create_ui(ANALYSIS_STEPS[8], "content_area", main_app_callbacks)

        with dpg.child_window(tag="ai_analysis_log_panel", width=400, border=True):
            dpg.add_text("💡 AI Analysis Log", color=[255, 255, 0])
            dpg.add_separator()
            dpg.add_text("AI 분석 결과가 여기에 표시됩니다.\n", tag="ai_analysis_log_panel_text", wrap=380)

dpg.create_viewport(title='Data Analysis Platform', width=1700, height=1000)
dpg.setup_dearpygui()

reset_application_state(clear_df_completely=True)

dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()