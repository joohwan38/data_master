# << NEW FILE >>: step_09_data_viewer.py

import dearpygui.dearpygui as dpg
import pandas as pd
from typing import Dict, Optional, Any, List

# --- DPG Tags for Step 9 ---
TAG_S9_GROUP = "step9_data_viewer_group"
TAG_S9_DF_LISTBOX = "step9_df_listbox"
TAG_S9_DF_SEARCH_INPUT = "step9_df_search_input"
TAG_S9_PREVIEW_TABLE = "step9_preview_table"
TAG_S9_PREVIEW_TEXT = "step9_preview_text"

# --- Module State ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_available_dfs: Dict[str, pd.DataFrame] = {}


def _on_df_select(sender, app_data, user_data):
    """왼쪽 리스트에서 DF 선택 시 호출되는 콜백."""
    selected_df_name = app_data
    if selected_df_name in _available_dfs:
        df_to_preview = _available_dfs[selected_df_name]
        dpg.set_value(TAG_S9_PREVIEW_TEXT, f"Preview of '{selected_df_name}' (Shape: {df_to_preview.shape})")
        # 대용량 데이터를 위한 미리보기 함수 사용
        _util_funcs['create_table_with_large_data_preview'](TAG_S9_PREVIEW_TABLE, df_to_preview)
    else:
        dpg.set_value(TAG_S9_PREVIEW_TEXT, f"'{selected_df_name}' was not found.")
        _util_funcs['create_table_with_data'](TAG_S9_PREVIEW_TABLE, pd.DataFrame())


def _update_df_list():
    """DF 목록을 필터링하여 리스트박스에 표시합니다."""
    search_term = dpg.get_value(TAG_S9_DF_SEARCH_INPUT).lower()
    
    # 사용 가능한 모든 DF의 이름 목록
    df_names = list(_available_dfs.keys())
    
    # 검색어에 따라 필터링
    if search_term:
        filtered_names = [name for name in df_names if search_term in name.lower()]
    else:
        filtered_names = df_names
    
    dpg.configure_item(TAG_S9_DF_LISTBOX, items=filtered_names)


def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """Step 9의 UI를 생성합니다."""
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    main_callbacks['register_step_group_tag'](step_name, TAG_S9_GROUP)

    with dpg.group(tag=TAG_S9_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        dpg.add_text("View all available DataFrames from previous steps and derivations.")
        dpg.add_spacer(height=5)

        with dpg.group(horizontal=True):
            # --- 왼쪽: DF 목록 패널 ---
            with dpg.child_window(width=300, border=True):
                dpg.add_input_text(label="Search", tag=TAG_S9_DF_SEARCH_INPUT, callback=_update_df_list, width=-1)
                dpg.add_listbox(items=[], tag=TAG_S9_DF_LISTBOX, callback=_on_df_select, num_items=-1)

            # --- 오른쪽: 미리보기 패널 ---
            with dpg.child_window(border=True):
                dpg.add_text("Select a DataFrame to preview.", tag=TAG_S9_PREVIEW_TEXT)
                dpg.add_separator()
                with dpg.table(tag=TAG_S9_PREVIEW_TABLE, header_row=True, resizable=True,
                               policy=dpg.mvTable_SizingFixedFit, scrollY=True, scrollX=True):
                    pass

    main_callbacks['register_module_updater'](step_name, update_ui)


def update_ui():
    """Step 9 UI의 내용을 최신 상태로 업데이트합니다."""
    global _available_dfs
    if not _module_main_callbacks: return

    # AppState에서 모든 DF를 가져옴
    _available_dfs = _module_main_callbacks['get_all_available_dfs']()
    
    # 목록 업데이트 및 첫 번째 항목 미리보기
    _update_df_list()
    
    items = dpg.get_item_configuration(TAG_S9_DF_LISTBOX)['items']
    if items:
        # 현재 선택된 값이 목록에 없으면 첫 번째 항목을 선택
        current_value = dpg.get_value(TAG_S9_DF_LISTBOX)
        if current_value not in items:
            dpg.set_value(TAG_S9_DF_LISTBOX, items[0])
            _on_df_select(None, items[0], None)
    else:
        # 목록이 비었으면 미리보기 초기화
        dpg.set_value(TAG_S9_PREVIEW_TEXT, "No DataFrames available.")
        _util_funcs['create_table_with_data'](TAG_S9_PREVIEW_TABLE, pd.DataFrame())


def reset_state():
    """모듈의 상태를 초기화합니다."""
    global _available_dfs
    _available_dfs = {}
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_S9_GROUP):
        dpg.set_value(TAG_S9_DF_SEARCH_INPUT, "")
        dpg.configure_item(TAG_S9_DF_LISTBOX, items=[])
        update_ui()