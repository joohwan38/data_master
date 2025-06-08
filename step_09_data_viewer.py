# step_09_data_viewer.py (리팩토링 완료)

import dearpygui.dearpygui as dpg
import pandas as pd
from typing import Dict, Optional, Any

# main_app의 핵심 객체 및 콜백을 가져옵니다.
from app_state_manager import BaseNode, app_state

# --- DPG Tags ---
TAG_S9_GROUP = "step9_data_viewer_group"
TAG_S9_DF_LISTBOX = "step9_df_listbox"
TAG_S9_DF_SEARCH_INPUT = "step9_df_search_input"
TAG_S9_PREVIEW_TABLE = "step9_preview_table"
TAG_S9_PREVIEW_TEXT = "step9_preview_text"

def _on_df_select(sender, app_data_selected_label, user_data):
    """왼쪽 리스트에서 노드 출력 선택 시 호출되는 콜백."""
    # 선택된 레이블에서 노드 ID를 파싱
    try:
        node_id_str = app_data_selected_label.split("#")[-1].strip()
        node_id = int(node_id_str)
    except (ValueError, IndexError):
        print(f"Error parsing node ID from label: {app_data_selected_label}")
        return

    df_to_preview = app_state.node_outputs.get(node_id)
    
    util_funcs = main_app_callbacks.get('get_util_funcs', lambda: {})()
    
    if df_to_preview is not None and 'create_table_with_large_data_preview' in util_funcs:
        dpg.set_value(TAG_S9_PREVIEW_TEXT, f"Preview of '{app_data_selected_label}' (Shape: {df_to_preview.shape})")
        util_funcs['create_table_with_large_data_preview'](TAG_S9_PREVIEW_TABLE, df_to_preview)
    else:
        dpg.set_value(TAG_S9_PREVIEW_TEXT, f"Output for '{app_data_selected_label}' not found or no data.")
        if 'create_table_with_data' in util_funcs:
            util_funcs['create_table_with_data'](TAG_S9_PREVIEW_TABLE, pd.DataFrame())

def _update_df_list():
    """표시할 DF 목록 (모든 노드 출력)을 업데이트합니다."""
    search_term = dpg.get_value(TAG_S9_DF_SEARCH_INPUT).lower()
    
    # app_state에서 모든 노드 출력 목록을 가져옴
    available_outputs = []
    for node_id, df in app_state.node_outputs.items():
        node = app_state.nodes.get(node_id)
        if node:
            # 노드의 레이블을 목록에 표시 (예: "Missing Value Imputer #2")
            label = node.label
            if search_term in label.lower():
                available_outputs.append(label)
    
    dpg.configure_item(TAG_S9_DF_LISTBOX, items=available_outputs)

def create_ui(step_name: str, parent_container_tag: str, main_callbacks_ref: dict):
    with dpg.group(tag=TAG_S9_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_text("View the output DataFrame of any executed node in the pipeline.")
        dpg.add_separator()

        with dpg.group(horizontal=True):
            # 왼쪽: DF 목록 패널
            with dpg.child_window(width=300):
                dpg.add_input_text(label="Search", tag=TAG_S9_DF_SEARCH_INPUT, callback=_update_df_list, width=-1)
                dpg.add_listbox(items=[], tag=TAG_S9_DF_LISTBOX, callback=_on_df_select, num_items=-1)

            # 오른쪽: 미리보기 패널
            with dpg.child_window(border=True):
                dpg.add_text("Select a node output from the list to preview.", tag=TAG_S9_PREVIEW_TEXT)
                dpg.add_separator()
                with dpg.table(tag=TAG_S9_PREVIEW_TABLE, header_row=True, resizable=True,
                               policy=dpg.mvTable_SizingFixedFit, scrollY=True, scrollX=True):
                    pass

def update_ui():
    """데이터 뷰어 UI를 최신 상태로 업데이트합니다."""
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_S9_GROUP):
        return
        
    # 파이프라인이 실행될 때마다 이 함수가 호출되어 목록을 갱신
    _update_df_list()
    
    # 목록이 업데이트된 후, 현재 선택된 항목이 여전히 유효한지 확인하고
    # 아니라면 선택을 해제하거나 첫 항목을 선택
    items = dpg.get_item_configuration(TAG_S9_DF_LISTBOX)['items']
    if not items:
        util_funcs = main_app_callbacks.get('get_util_funcs', lambda: {})()
        dpg.set_value(TAG_S9_PREVIEW_TEXT, "No node outputs available to display.")
        if 'create_table_with_data' in util_funcs:
            util_funcs['create_table_with_data'](TAG_S9_PREVIEW_TABLE, pd.DataFrame())