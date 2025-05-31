# step_03_preprocessing.py
import dearpygui.dearpygui as dpg
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import traceback # 예외 출력을 위해 추가

# --- Module-specific Tags ---
TAG_PREPROC_STEP_GROUP = "step3_preprocessing_group"
TAG_NODE_EDITOR = "step3_node_editor"
TAG_NODE_PALETTE_WINDOW = "step3_node_palette_window"
TAG_PROPERTIES_PANEL_WINDOW = "step3_properties_panel_window"
TAG_RUN_PIPELINE_BUTTON = "step3_run_pipeline_button"
NODE_CONTENT_WIDTH = 180
NODE_MIN_HEIGHT = 80

# --- Module State ---
_main_app_callbacks: Optional[Dict[str, Any]] = None
_util_funcs: Optional[Dict[str, Any]] = None
_current_pipeline_nodes: Dict[int, Dict[str, Any]] = {}
_current_pipeline_links: List[Tuple[int, int]] = []
_selected_node_id: Optional[int] = None
_node_id_counter = 0

NODE_TYPES_CONFIG = {
    "data_input": {"inputs": 0, "outputs": 1, "output_names": ["DataFrame Out"], "params": {}},
    "data_output": {"inputs": 1, "outputs": 0, "input_names": ["DataFrame In"], "params": {}},
    "simple_imputer": {
        "inputs": 1, "outputs": 1,
        "input_names": ["DataFrame In"], "output_names": ["DataFrame Out"],
        "params": {"strategy": "mean", "columns": [], "fill_value": None}
    },
    "datetime_feature_extractor": {
        "inputs": 1, "outputs": 1,
        "input_names": ["DataFrame In"], "output_names": ["DataFrame Out"],
        "params": {"target_column": None, "features_to_extract": ["year", "month", "day"]}
    },
}

def _link_callback(sender, app_data):
    output_attr_id = app_data[0]
    input_attr_id = app_data[1]
    print(f"Link created: Output Attr {output_attr_id} -> Input Attr {input_attr_id}")
    _current_pipeline_links.append((output_attr_id, input_attr_id))

def _delink_callback(sender, app_data):
    link_id = app_data
    print(f"Link deleted: Link ID {link_id}")
    _current_pipeline_links.clear()
    if dpg.does_item_exist(TAG_NODE_EDITOR):
        all_links_in_editor = dpg.get_links(TAG_NODE_EDITOR)
        if all_links_in_editor:
            _current_pipeline_links.extend(all_links_in_editor)

def _node_selected_callback(sender, app_data, user_data):
    pass # update_ui에서 처리

def _add_node_to_editor(node_type: str, pos: Optional[List[int]] = None):
    global _node_id_counter, _current_pipeline_nodes, _util_funcs
    _node_id_counter += 1
    node_id = _node_id_counter

    node_config = NODE_TYPES_CONFIG.get(node_type)
    if not node_config:
        print(f"Error: Unknown node type '{node_type}'")
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Node Error", f"Unknown node type: {node_type}")
        return

    if not dpg.does_item_exist(TAG_NODE_EDITOR):
        print("Error: Node editor canvas not found.")
        return

    format_text = (_util_funcs.get('format_text_for_display', lambda text, max_len: str(text)[:max_len] + "..." if len(str(text)) > max_len else str(text))
                   if _util_funcs else lambda text, max_len: str(text)[:max_len] + "..." if len(str(text)) > max_len else str(text))

    node_label_display = format_text(f"{node_type.replace('_', ' ').title()}", 20) + f" ({node_id})"

    palette_width = 220
    editor_canvas_width = dpg.get_item_width(TAG_NODE_EDITOR) if dpg.does_item_exist(TAG_NODE_EDITOR) and dpg.get_item_width(TAG_NODE_EDITOR) else 800
    nodes_per_row = max(1, (editor_canvas_width - palette_width) // 200)
    node_x_pos = 50 + (_node_id_counter % nodes_per_row) * 220
    node_y_pos = 50 + (_node_id_counter // nodes_per_row) * 130
    
    current_node_pos = pos or [node_x_pos, node_y_pos]

    try:
        with dpg.node(label=node_label_display,
                       tag=node_id, parent=TAG_NODE_EDITOR,
                       pos=current_node_pos):

            # --- [최종 해결책 적용] ---
            # 노드 내부에 항상 존재하는 정적 속성을 추가합니다.
            # 이 속성의 label을 명시적인 문자열로 설정하여 노드 확장 버그를 방지합니다.
            # show_label 키워드는 사용하지 않아 SystemError를 피합니다.
            with dpg.node_attribute(tag=dpg.generate_uuid(), 
                                     attribute_type=dpg.mvNode_Attr_Static, 
                                     label=""): # 명시적인 (비어있지 않은) 라벨 제공
                dpg.add_text("") # 이 텍스트는 내부적으로만 존재, 보이지 않음 (선택 사항)

            # --- 기존 입력(Input) 속성 생성 ---
            input_attrs = []
            for i in range(node_config["inputs"]):
                attr_name_raw = node_config["input_names"][i] if "input_names" in node_config and i < len(node_config["input_names"]) else f"In {i+1}"
                attr_label_display = format_text(attr_name_raw, 12)
                if not attr_label_display and attr_name_raw: attr_label_display = attr_name_raw[0:2] + ".."

                attr_id = dpg.add_node_attribute(label=attr_label_display, attribute_type=dpg.mvNode_Attr_Input)
                input_attrs.append(attr_id)
                if attr_label_display != attr_name_raw and dpg.does_item_exist(attr_id):
                    with dpg.tooltip(parent=attr_id):
                        dpg.add_text(attr_name_raw)

            # --- 기존 출력(Output) 속성 생성 ---
            output_attrs = []
            for i in range(node_config["outputs"]):
                attr_name_raw = node_config["output_names"][i] if "output_names" in node_config and i < len(node_config["output_names"]) else f"Out {i+1}"
                attr_label_display = format_text(attr_name_raw, 12)
                if not attr_label_display and attr_name_raw: attr_label_display = attr_name_raw[0:2] + ".."

                attr_id = dpg.add_node_attribute(label=attr_label_display, attribute_type=dpg.mvNode_Attr_Output)
                output_attrs.append(attr_id)
                if attr_label_display != attr_name_raw and dpg.does_item_exist(attr_id):
                    with dpg.tooltip(parent=attr_id):
                        dpg.add_text(attr_name_raw)

            _current_pipeline_nodes[node_id] = {
                "id": node_id,
                "type": node_type,
                "label": node_label_display,
                "params": node_config.get("params", {}).copy(),
                "input_attribute_ids": input_attrs,
                "output_attribute_ids": output_attrs,
            }
    except Exception as e:
        print(f"ERROR during _add_node_to_editor for node_type '{node_type}', id '{node_id}': {e}")
        traceback.print_exc()

# --- (이하 _update_properties_panel, 콜백 함수들, _run_pipeline_from_nodes, create_ui, update_ui, reset_preprocessing_state, get_preprocessing_settings_for_saving, apply_preprocessing_settings_and_process 함수들은 이전과 동일하게 유지합니다) ---
# (이하 동일한 코드는 생략하고, 필요한 경우 이전 답변의 코드를 참조하십시오.)
def _update_properties_panel(node_id: Optional[int]):
    global _main_app_callbacks
    if not dpg.does_item_exist(TAG_PROPERTIES_PANEL_WINDOW):
        return
    dpg.delete_item(TAG_PROPERTIES_PANEL_WINDOW, children_only=True)

    if node_id is None or node_id not in _current_pipeline_nodes:
        dpg.add_text("No node selected.", parent=TAG_PROPERTIES_PANEL_WINDOW)
        return

    node_data = _current_pipeline_nodes[node_id]
    dpg.add_text(f"Properties: {node_data['label']}", parent=TAG_PROPERTIES_PANEL_WINDOW)
    dpg.add_separator(parent=TAG_PROPERTIES_PANEL_WINDOW)

    df_for_columns = _main_app_callbacks['get_df_after_step1']() if _main_app_callbacks else None
    available_columns = list(df_for_columns.columns) if df_for_columns is not None else ["No data available"]

    if node_data["type"] == "simple_imputer":
        params = node_data["params"]
        
        dpg.add_text("Target Columns:", parent=TAG_PROPERTIES_PANEL_WINDOW)
        valid_selected_cols = [col for col in params.get("columns", []) if col in available_columns]
        
        dpg.add_listbox(
            items=available_columns, 
            default_value=valid_selected_cols, 
            label="Columns", 
            num_items=min(8, len(available_columns) +1 ), 
            parent=TAG_PROPERTIES_PANEL_WINDOW,
            user_data={"node_id": node_id, "param_name": "columns"},
            callback=_on_node_param_change_listbox, 
            width=-1, 
            multiple=True 
        )

        strategies = ["mean", "median", "most_frequent", "constant"]
        current_strategy = params.get("strategy", "mean")
        dpg.add_radio_button(
            strategies, 
            label="Strategy", 
            default_value=current_strategy, 
            parent=TAG_PROPERTIES_PANEL_WINDOW,
            user_data={"node_id": node_id, "param_name": "strategy"},
            callback=_on_node_param_change_generic, 
            horizontal=True
        )

        if current_strategy == "constant":
            current_fill_value = params.get("fill_value", "")
            dpg.add_input_text(
                label="Fill Value (if constant)", 
                default_value=str(current_fill_value) if current_fill_value is not None else "", 
                parent=TAG_PROPERTIES_PANEL_WINDOW,
                user_data={"node_id": node_id, "param_name": "fill_value"},
                callback=_on_node_param_change_generic, 
                width=200
            )
        else: 
            if "fill_value" in params:
                params["fill_value"] = None
    
    elif node_data["type"] == "datetime_feature_extractor":
        params = node_data["params"]
        datetime_cols = [col for col in available_columns if df_for_columns is not None and pd.api.types.is_datetime64_any_dtype(df_for_columns[col])]
        if not datetime_cols: datetime_cols = ["No datetime columns found"]

        current_target_col = params.get("target_column")
        if current_target_col not in datetime_cols: current_target_col = datetime_cols[0] if datetime_cols else None

        dpg.add_combo(
            datetime_cols,
            label="Target Datetime Column",
            default_value=current_target_col or "",
            parent=TAG_PROPERTIES_PANEL_WINDOW,
            user_data={"node_id": node_id, "param_name": "target_column"},
            callback=_on_node_param_change_generic,
            width=-1
        )
        dpg.add_text("Features to Extract:", parent=TAG_PROPERTIES_PANEL_WINDOW)
        possible_dt_features = ["year", "month", "day", "hour", "minute", "second", "dayofweek", "dayofyear", "weekofyear", "quarter"]
        current_features = params.get("features_to_extract", [])
        for feat in possible_dt_features:
            dpg.add_checkbox(
                label=feat.capitalize(),
                default_value=feat in current_features,
                parent=TAG_PROPERTIES_PANEL_WINDOW,
                user_data={"node_id": node_id, "param_name": "features_to_extract", "feature_value": feat},
                callback=_on_datetime_feature_checkbox_change
            )

def _on_node_param_change_generic(sender, app_data, user_data):
    node_id = user_data["node_id"]
    param_name = user_data["param_name"]
    
    if node_id in _current_pipeline_nodes:
        _current_pipeline_nodes[node_id]["params"][param_name] = app_data
        print(f"Node {node_id} param '{param_name}' changed to: {app_data}")
        if param_name == "strategy" and _current_pipeline_nodes[node_id]["type"] == "simple_imputer":
            _update_properties_panel(node_id) 

def _on_node_param_change_listbox(sender, app_data, user_data):
    node_id = user_data["node_id"]
    param_name = user_data["param_name"]
    
    if node_id in _current_pipeline_nodes:
        _current_pipeline_nodes[node_id]["params"][param_name] = app_data 
        print(f"Node {node_id} param '{param_name}' (listbox) changed to: {app_data}")

def _on_datetime_feature_checkbox_change(sender, app_data, user_data):
    node_id = user_data["node_id"]
    param_name = user_data["param_name"] 
    feature_value = user_data["feature_value"] 
    is_checked = app_data 

    if node_id in _current_pipeline_nodes:
        current_list = _current_pipeline_nodes[node_id]["params"].get(param_name, [])
        if not isinstance(current_list, list): current_list = [] 

        if is_checked:
            if feature_value not in current_list:
                current_list.append(feature_value)
        else:
            if feature_value in current_list:
                current_list.remove(feature_value)
        
        _current_pipeline_nodes[node_id]["params"][param_name] = current_list
        print(f"Node {node_id} param '{param_name}' updated: {current_list}")

def _run_pipeline_from_nodes():
    global _main_app_callbacks, _util_funcs
    if not _main_app_callbacks or not _util_funcs:
        print("Error: Callbacks not initialized for pipeline run.")
        return

    df_input_initial = _main_app_callbacks['get_df_after_step1']()
    if df_input_initial is None:
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Pipeline Error", "Input data (from Step 1) is not available.")
        _main_app_callbacks['step3_processing_complete'](None) 
        return

    print("Attempting to run pipeline from nodes...")
    print(f"Nodes: {_current_pipeline_nodes}")
    print(f"Links: {_current_pipeline_links}")
    
    processed_df = df_input_initial.copy() 
    
    if _util_funcs and '_show_simple_modal_message' in _util_funcs:
        _util_funcs['_show_simple_modal_message']("Pipeline Execution", "Pipeline execution logic is still a TODO.\nData is passed through without transformation for now.")
    
    _main_app_callbacks['step3_processing_complete'](processed_df)

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _main_app_callbacks, _util_funcs
    _main_app_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()

    main_callbacks['register_step_group_tag'](step_name, TAG_PREPROC_STEP_GROUP)

    with dpg.group(tag=TAG_PREPROC_STEP_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()

        with dpg.group(horizontal=True):
            with dpg.child_window(tag=TAG_NODE_PALETTE_WINDOW, width=220, menubar=True): 
                with dpg.menu_bar():
                    dpg.add_text("Node Palette") 
                
                dpg.add_button(label="Input Data", width=-1, callback=lambda: _add_node_to_editor("data_input"))
                dpg.add_button(label="Output/Preview Data", width=-1, callback=lambda: _add_node_to_editor("data_output"))
                dpg.add_separator()
                dpg.add_text("Imputation:")
                dpg.add_button(label="Simple Imputer", width=-1, callback=lambda: _add_node_to_editor("simple_imputer"))
                dpg.add_separator()
                dpg.add_text("Feature Engineering:")
                dpg.add_button(label="Datetime Extractor", width=-1, callback=lambda: _add_node_to_editor("datetime_feature_extractor"))

            with dpg.group(): 
                dpg.add_button(label="Run Preprocessing Pipeline", tag=TAG_RUN_PIPELINE_BUTTON, width=-1, height=30, callback=_run_pipeline_from_nodes)
                dpg.add_separator()
                
                with dpg.group(horizontal=True):
                    with dpg.node_editor(tag=TAG_NODE_EDITOR, callback=_link_callback, delink_callback=_delink_callback, height=650, width=-1, minimap=True, minimap_location=dpg.mvNodeMiniMap_Location_BottomRight):
                        _add_node_to_editor("data_input", pos=[50,50])
                        _add_node_to_editor("data_output", pos=[400,150])
                    
                    with dpg.child_window(tag=TAG_PROPERTIES_PANEL_WINDOW, width=320): 
                        dpg.add_text("Select a node to see its properties.")

    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui(df_input: Optional[pd.DataFrame], main_callbacks: dict):
    global _selected_node_id, _main_app_callbacks 
    if not _main_app_callbacks: _main_app_callbacks = main_callbacks 

    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_PREPROC_STEP_GROUP):
        return
    
    current_selected_in_editor = None
    if dpg.does_item_exist(TAG_NODE_EDITOR):
         selected_nodes_list = dpg.get_selected_nodes(TAG_NODE_EDITOR)
         if selected_nodes_list:
             current_selected_in_editor = selected_nodes_list[0]

    if _selected_node_id != current_selected_in_editor:
        _selected_node_id = current_selected_in_editor
        _update_properties_panel(_selected_node_id)
    elif _selected_node_id is not None : 
        _update_properties_panel(_selected_node_id) 
    else: 
        _update_properties_panel(None)

def reset_preprocessing_state():
    global _current_pipeline_nodes, _current_pipeline_links, _selected_node_id, _node_id_counter
    if dpg.does_item_exist(TAG_NODE_EDITOR):
        node_ids_to_delete = list(_current_pipeline_nodes.keys())
        for node_id_val in node_ids_to_delete:
            if dpg.does_item_exist(node_id_val):
                dpg.delete_item(node_id_val) 
    
    _current_pipeline_nodes.clear()
    _current_pipeline_links.clear()
    _selected_node_id = None
    _node_id_counter = 0 
    
    _update_properties_panel(None)

    if dpg.does_item_exist(TAG_NODE_EDITOR): 
        _add_node_to_editor("data_input", pos=[50,50])
        _add_node_to_editor("data_output", pos=[400,150])
    print("Preprocessing step state has been reset.")

def get_preprocessing_settings_for_saving() -> dict:
    nodes_to_save = []
    for node_id, node_data in _current_pipeline_nodes.items():
        node_copy = node_data.copy()
        if dpg.does_item_exist(node_id):
            node_copy['pos'] = dpg.get_item_pos(node_id)
        node_copy.pop('input_attribute_ids', None) 
        node_copy.pop('output_attribute_ids', None)
        nodes_to_save.append(node_copy)
    return {
        "nodes": nodes_to_save,
        "links_by_attr_ids": _current_pipeline_links, 
        "node_id_counter": _node_id_counter 
    }

def apply_preprocessing_settings_and_process(df_input: Optional[pd.DataFrame], settings: dict, main_callbacks: dict):
    global _current_pipeline_nodes, _current_pipeline_links, _node_id_counter, _main_app_callbacks, _util_funcs
    
    if not dpg.is_dearpygui_running(): return
    
    _main_app_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()

    reset_preprocessing_state() 

    _node_id_counter = settings.get("node_id_counter", 0)
    saved_nodes_data = settings.get("nodes", [])
    
    for node_setting in saved_nodes_data:
        node_type = node_setting.get("type")
        node_pos = node_setting.get("pos")
        node_params = node_setting.get("params", {})

        if node_type:
            _add_node_to_editor(node_type, pos=node_pos)
            newly_created_node_id = _node_id_counter 

            if newly_created_node_id in _current_pipeline_nodes:
                _current_pipeline_nodes[newly_created_node_id]["params"] = node_params.copy()
                print(f"Restored node: type={node_type}, new_id={newly_created_node_id}, params={node_params}")
            else:
                print(f"Error: Failed to find newly created node {newly_created_node_id} in _current_pipeline_nodes after attempting to restore type {node_type}")

    print("Preprocessing settings partially applied (Nodes and params restored. Links need complex ID mapping and are not automatically restored in this version).")

    if df_input is None: 
         _main_app_callbacks['step3_processing_complete'](None)