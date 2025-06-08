# step_03_preprocessing.py (핸들러 오류 최종 수정 완료)

import dearpygui.dearpygui as dpg
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import traceback
import json

# --- 필요한 모듈 및 클래스 임포트 ---
from app_state_manager import app_state, BaseNode
from step_01_data_loading import InputNode, TypeCasterNode
from step_04_missing_values import MissingValueNode
from step_05_outlier_treatment import OutlierNode
from step_06_standardization import StandardizationNode
from step_07_feature_engineering import BinningNode
from step_08_derivation import DerivationNode

# --- UI Tags ---
TAG_PREPROC_STEP_GROUP = "step3_preprocessing_group"
TAG_NODE_EDITOR = "step3_node_editor"
TAG_PROPERTIES_PANEL_WINDOW = "step3_properties_panel_window"
TAG_RUN_PIPELINE_BUTTON = "step3_run_pipeline_button"

# --- 노드 설정 ---
NODE_TYPE_TO_CLASS = {
    "data_input": InputNode,
    "data_output": BaseNode,
    "type_caster": TypeCasterNode,
    "missing_value_imputer": MissingValueNode,
    "outlier_handler": OutlierNode,
    "standardization_scaler": StandardizationNode,
    "binning_node": BinningNode,
    "derivation_node": DerivationNode,
}

NODE_TYPES_CONFIG = {
    "data_input": {"label": "Input Data", "inputs": 0, "outputs": 1, "output_names": ["DataFrame Out"]},
    "data_output": {"label": "Output Data", "inputs": 1, "outputs": 0, "input_names": ["DataFrame In"]},
    "type_caster": {"label": "Type Caster", "inputs": 1, "outputs": 1, "input_names": ["DataFrame In"], "output_names": ["DataFrame Out"]},
    "missing_value_imputer": {"label": "Missing Value Imputer", "inputs": 1, "outputs": 1, "input_names": ["DataFrame In"], "output_names": ["DataFrame Out"]},
    "outlier_handler": {"label": "Outlier Handler", "inputs": 1, "outputs": 1, "input_names": ["DataFrame In"], "output_names": ["DataFrame Out"]},
    "standardization_scaler": {"label": "Scaler (Standardization)", "inputs": 1, "outputs": 1, "input_names": ["DataFrame In"], "output_names": ["DataFrame Out"]},
    "binning_node": {"label": "Binning", "inputs": 1, "outputs": 1, "input_names": ["DataFrame In"], "output_names": ["DataFrame Out"]},
    "derivation_node": {"label": "Derive DataFrame", "inputs": 1, "outputs": 1, "input_names": ["DataFrame In"], "output_names": ["DataFrame Out"]},
}

# --- 콜백 함수들 ---
# (이하 콜백 및 헬퍼 함수들은 이전 버전과 동일)

def _link_callback(sender, app_data):
    output_attr_id, input_attr_id = app_data
    app_state.links.append((output_attr_id, input_attr_id))
    target_node_id, target_input_name, source_node_id = None, None, None
    for node_id, node in app_state.nodes.items():
        if output_attr_id in node.ui_info.get('output_attr_ids', []):
            source_node_id = node_id
        for pin_name, pin_id in node.ui_info.get('input_attr_map', {}).items():
            if input_attr_id == pin_id:
                target_node_id, target_input_name = node_id, pin_name
                break
    if all([target_node_id, target_input_name, source_node_id is not None]):
        app_state.nodes[target_node_id].input_connections[target_input_name] = source_node_id

def _delink_callback(sender, app_data_link_id):
    print(f"Link {app_data_link_id} deleted.")

def _node_selection_callback(sender, app_data):
    main_app_callbacks = app_state.module_ui_updaters.get("get_main_callbacks", lambda: {})()
    selected_nodes = dpg.get_selected_nodes(TAG_NODE_EDITOR)
    new_selection = selected_nodes[0] if selected_nodes else None
    if app_state.selected_node_id != new_selection:
        app_state.selected_node_id = new_selection
        _update_properties_panel(app_state.selected_node_id)
        if 'trigger_specific_module_update' in main_app_callbacks:
            main_app_callbacks['trigger_specific_module_update'](ANALYSIS_STEPS[0])
            if app_state.active_step_name:
                main_app_callbacks['trigger_specific_module_update'](app_state.active_step_name)

def _node_double_clicked_callback(sender, app_data):
    node_id = app_data[1]
    if node_id not in app_state.nodes: return
    node = app_state.nodes[node_id]

    main_app_callbacks = app_state.module_ui_updaters.get("get_main_callbacks", lambda: {})()
    if 'get_util_funcs' not in main_app_callbacks or 'switch_step_view' not in main_app_callbacks: return
    util_funcs = main_app_callbacks['get_util_funcs']()
    
    step_map = { "type_caster": ANALYSIS_STEPS[0], "missing_value_imputer": ANALYSIS_STEPS[3], "outlier_handler": ANALYSIS_STEPS[4], "standardization_scaler": ANALYSIS_STEPS[5], "binning_node": ANALYSIS_STEPS[6], "derivation_node": ANALYSIS_STEPS[7] }
    step_name = step_map.get(node.type)
    if not step_name:
        if '_show_simple_modal_message' in util_funcs:
            util_funcs['_show_simple_modal_message']("Info", f"Node '{node.label}' does not have a detailed editor view.")
        return

    main_app_callbacks['switch_step_view'](sender, app_data, step_name)
    updaters = main_app_callbacks.get('get_module_ui_updaters', lambda: {})()
    updater = updaters.get(step_name)
    if updater: updater(node_id_to_load=node_id)

def _add_node_to_editor(node_type: str, pos: Optional[List[int]] = None):
    app_state.node_id_counter += 1
    node_id = app_state.node_id_counter
    node_class = NODE_TYPE_TO_CLASS.get(node_type)
    if not node_class: return
    node_instance = node_class(node_id=node_id)
    app_state.nodes[node_id] = node_instance
    node_config = NODE_TYPES_CONFIG.get(node_type, {})
    node_pos = pos or [50 + (len(app_state.nodes) % 5) * 220, 50 + (len(app_state.nodes) // 5) * 130]
    with dpg.node(label=node_instance.label, tag=node_id, parent=TAG_NODE_EDITOR, pos=node_pos):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static, label=""): dpg.add_text("")
        input_attr_map, output_attr_ids = {}, []
        for pin_name in node_config.get("input_names", []):
            with dpg.node_attribute(label=pin_name, attribute_type=dpg.mvNode_Attr_Input) as attr_id:
                dpg.add_text(pin_name); input_attr_map[pin_name] = attr_id
        for pin_name in node_config.get("output_names", []):
            with dpg.node_attribute(label=pin_name, attribute_type=dpg.mvNode_Attr_Output) as attr_id:
                dpg.add_text(pin_name); output_attr_ids.append(attr_id)
        node_instance.ui_info = {"input_attr_map": input_attr_map, "output_attr_ids": output_attr_ids}

def _update_properties_panel(node_id: Optional[int]):
    dpg.delete_item(TAG_PROPERTIES_PANEL_WINDOW, children_only=True)
    if node_id is None or node_id not in app_state.nodes:
        dpg.add_text("No node selected.", parent=TAG_PROPERTIES_PANEL_WINDOW)
        return
    node = app_state.nodes[node_id]
    dpg.add_text(f"Properties: {node.label}", parent=TAG_PROPERTIES_PANEL_WINDOW)

def _run_pipeline_callback():
    main_app_callbacks = app_state.module_ui_updaters.get("get_main_callbacks", lambda: {})()
    pipeline_manager = main_app_callbacks.get('get_pipeline_manager')()
    if not pipeline_manager: return
    output_nodes = [nid for nid, node in app_state.nodes.items() if node.type == 'data_output']
    if not output_nodes:
        util_funcs = main_app_callbacks.get('get_util_funcs', lambda: {})()
        if '_show_simple_modal_message' in util_funcs:
            util_funcs['_show_simple_modal_message']("Error", "Please add a 'Data Output' node.")
        return
    pipeline_manager.execute_pipeline_to_node(output_nodes[0])
    _update_properties_panel(app_state.selected_node_id)


# <<< create_ui 함수가 핵심 수정 대상 >>>
def create_ui(step_name: str, parent_container_tag: str, main_callbacks_ref: dict):
    app_state.module_ui_updaters["get_main_callbacks"] = lambda: main_callbacks_ref
    
    # ANALYSIS_STEPS 상수를 main_app에서 가져오도록 수정
    global ANALYSIS_STEPS
    ANALYSIS_STEPS = main_callbacks_ref.get("ANALYSIS_STEPS", [])

    with dpg.group(tag=TAG_PREPROC_STEP_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---"); dpg.add_separator()
        with dpg.group(horizontal=True):
            with dpg.child_window(width=220, menubar=True):
                with dpg.menu_bar(): dpg.add_text("Node Palette")
                for node_type, config in NODE_TYPES_CONFIG.items():
                    dpg.add_button(label=config['label'], width=-1, callback=lambda s, a, ut=node_type: _add_node_to_editor(ut))

            with dpg.group():
                dpg.add_button(label="▶️ Run Pipeline to Output Node", tag=TAG_RUN_PIPELINE_BUTTON, width=-1, height=30, callback=_run_pipeline_callback)
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    with dpg.node_editor(tag=TAG_NODE_EDITOR, callback=_link_callback, delink_callback=_delink_callback,
                                         height=650, minimap=True, minimap_location=dpg.mvNodeMiniMap_Location_BottomRight) as editor_tag:
                        
                        # <<< CHANGED: 올바른 이벤트 핸들러 등록 방식 >>>
                        # 1. 핸들러 등록소를 먼저 만듭니다.
                        with dpg.handler_registry() as registry_tag:
                            # 2. 등록소 안에 더블클릭 핸들러를 추가합니다.
                            dpg.add_item_double_clicked_handler(callback=_node_double_clicked_callback)
                        
                        # 3. 생성한 등록소를 노드 에디터(editor_tag)에 연결(바인딩)합니다.
                        dpg.bind_item_handler_registry(editor_tag, registry_tag)
                        
                        # 노드 선택 콜백은 노드 에디터 자체에 직접 설정합니다.
                        dpg.set_item_callback(editor_tag, _node_selection_callback)
                    
                    with dpg.child_window(tag=TAG_PROPERTIES_PANEL_WINDOW, width=320):
                        dpg.add_text("Select a node to see its properties.")

def update_ui():
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_NODE_EDITOR): return
    _node_selection_callback(None, None)

def reset_preprocessing_state():
    if dpg.does_item_exist(TAG_NODE_EDITOR):
        dpg.delete_item(TAG_NODE_EDITOR, children_only=True)
    app_state.nodes.clear(); app_state.links.clear(); app_state.node_outputs.clear()
    app_state.selected_node_id = None; app_state.node_id_counter = 0
    _update_properties_panel(None)
    _add_node_to_editor("data_input", pos=[50, 50])
    _add_node_to_editor("data_output", pos=[400, 150])