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
TAG_S9_DF_TABLE = "step9_df_selectable_table" # <-- 이 태그를 추가하세요.

# --- Module State ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_available_dfs: Dict[str, pd.DataFrame] = {}


def _on_df_select(sender, app_data, user_data):
    """테이블에서 DF 선택 시 호출되는 콜백 (수정됨)."""

    # app_data는 이제 True(선택됨) 또는 False(선택해제됨)입니다.
    # 선택이 해제될 때는 아무 작업도 하지 않도록 합니다.
    if not app_data:
        # 사용자가 이미 선택된 항목을 다시 클릭하여 선택 해제하는 것을 방지하려면
        # 아래 한 줄의 주석을 해제하세요.
        # dpg.set_value(sender, True)
        return

    # 1. 단일 선택 로직 구현
    # 현재 테이블의 모든 행(selectable)을 가져옵니다.
    if dpg.does_item_exist(TAG_S9_DF_TABLE):
        rows = dpg.get_item_children(TAG_S9_DF_TABLE, 1)
        for row in rows:
            # 각 행의 자식인 selectable 위젯을 찾습니다.
            selectable = dpg.get_item_children(row, 1)[0]
            # 방금 클릭한 항목(sender)이 아니라면 모두 선택 해제(False)합니다.
            if selectable != sender:
                dpg.set_value(selectable, False)

    # 2. 올바른 인자 사용
    # 데이터프레임 이름은 이제 app_data가 아닌 user_data에서 가져옵니다.
    selected_df_name = user_data

    if selected_df_name in _available_dfs:
        df_to_preview = _available_dfs[selected_df_name]
        dpg.set_value(TAG_S9_PREVIEW_TEXT, f"Preview of '{selected_df_name}' (Shape: {df_to_preview.shape})")
        if _util_funcs and 'create_table_with_large_data_preview' in _util_funcs:
            _util_funcs['create_table_with_large_data_preview'](TAG_S9_PREVIEW_TABLE, df_to_preview)
    else:
        # 이 부분은 이제 거의 실행되지 않아야 합니다.
        dpg.set_value(TAG_S9_PREVIEW_TEXT, f"'{selected_df_name}' was not found in available list.")
        if _util_funcs and 'create_table_with_data' in _util_funcs:
            _util_funcs['create_table_with_data'](TAG_S9_PREVIEW_TABLE, pd.DataFrame())

def _update_df_list():
    """DF 목록을 필터링하여 테이블에 표시합니다."""
    search_term = dpg.get_value(TAG_S9_DF_SEARCH_INPUT).lower()

    # 사용 가능한 모든 DF의 이름 목록
    df_names = list(_available_dfs.keys())

    # 검색어에 따라 필터링
    if search_term:
        filtered_names = [name for name in df_names if search_term in name.lower()]
    else:
        filtered_names = df_names

    # 테이블의 모든 자식(행)을 삭제하여 목록을 초기화
    if dpg.does_item_exist(TAG_S9_DF_TABLE):
        dpg.delete_item(TAG_S9_DF_TABLE, children_only=True)
        dpg.add_table_column(parent=TAG_S9_DF_TABLE) # 컬럼을 다시 추가해야 합니다.

    # 필터링된 이름으로 테이블 행을 다시 채움
    for name in filtered_names:
        with dpg.table_row(parent=TAG_S9_DF_TABLE):
            # 각 행에 선택 가능한 텍스트(selectable)를 추가
            dpg.add_selectable(label=name, user_data=name, callback=_on_df_select, span_columns=True)


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
            with dpg.child_window(width=200, border=True): # 너비는 400으로 수정했다고 가정
                dpg.add_input_text(label="Search", tag=TAG_S9_DF_SEARCH_INPUT, callback=_update_df_list, width=100)
                # 리스트박스 대신 테이블 사용
                with dpg.table(header_row=False, tag=TAG_S9_DF_TABLE,
                            policy=dpg.mvTable_SizingStretchProp, # 너비를 꽉 채우는 정책
                            scrollY=True, borders_innerV=False, borders_outerH=False, borders_outerV=False):
                    dpg.add_table_column() 

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

    # 목록 업데이트 (이 함수는 이제 테이블의 행을 채웁니다)
    _update_df_list()

    # --- 수정된 부분 ---
    # 더 이상 존재하지 않는 LISTBOX 관련 로직을 모두 삭제하고,
    # 테이블의 자식(행)이 있는지 여부만 판단하여 미리보기를 초기화합니다.
    table_items = []
    if dpg.does_item_exist(TAG_S9_DF_TABLE):
        # 테이블의 자식 아이템(행)들의 목록을 가져옵니다.
        table_items = dpg.get_item_children(TAG_S9_DF_TABLE, 1)

    if not table_items:
        # 테이블에 표시할 항목이 없으면 미리보기 초기화
        dpg.set_value(TAG_S9_PREVIEW_TEXT, "No DataFrames available.")
        if _util_funcs and 'create_table_with_data' in _util_funcs:
            _util_funcs['create_table_with_data'](TAG_S9_PREVIEW_TABLE, pd.DataFrame())


def reset_state():
    """모듈의 상태를 초기화합니다."""
    global _available_dfs
    _available_dfs = {}
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_S9_GROUP):
        # 검색창 초기화
        if dpg.does_item_exist(TAG_S9_DF_SEARCH_INPUT):
            dpg.set_value(TAG_S9_DF_SEARCH_INPUT, "")

        # --- 수정된 부분: 테이블 내용 초기화 ---
        # 더 이상 사용하지 않는 listbox 설정 코드를 삭제하고,
        # 테이블의 자식(모든 행)을 삭제하는 코드로 변경합니다.
        if dpg.does_item_exist(TAG_S9_DF_TABLE):
            dpg.delete_item(TAG_S9_DF_TABLE, children_only=True)
            # 자식을 모두 지운 후에는 테이블에 컬럼을 다시 추가해주어야 합니다.
            dpg.add_table_column(parent=TAG_S9_DF_TABLE)

        # UI의 다른 부분도 초기화
        update_ui()