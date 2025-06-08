# step_01_data_loading.py (Node 클래스로 리팩토링 완료)

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any

# main_app의 BaseNode와 핵심 상태/콜백 객체를 가져옵니다.
from app_state_manager import BaseNode, app_state


# --- DPG Tags ---
TAG_DL_GROUP = "step1_data_loading_group"
TAG_DL_SHAPE_TEXT = "step1_df_summary_shape_text"
TAG_DL_INFO_TABLE = "step1_df_summary_info_table"
TAG_DL_DESCRIBE_TABLE = "step1_df_summary_describe_table"
TAG_DL_HEAD_TABLE = "step1_df_summary_head_table"
TAG_DL_TYPE_EDITOR_TABLE = "step1_type_editor_table" # TypeCasterNode 설정용

# --- Node Class Definitions ---

class InputNode(BaseNode):
    """파이프라인의 시작점. 원본 데이터를 그대로 출력합니다."""
    NODE_TYPE = "data_input"

    def __init__(self, node_id: int, params: dict | None = None):
        super().__init__(node_id=node_id, node_type=self.NODE_TYPE, params=params or {})
        self.label = f"Input Data #{self.id}"

    def process(self, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        # 이 노드는 특별히 app_state에서 원본 데이터를 직접 가져옵니다.
        if app_state.original_df is None:
            raise FileNotFoundError("Original data is not loaded in the application.")
        return app_state.original_df


class TypeCasterNode(BaseNode):
    """사용자가 지정한 타입으로 컬럼 타입을 변환하는 노드입니다."""
    NODE_TYPE = "type_caster"

    def __init__(self, node_id: int, params: dict | None = None):
        default_params = {
            "type_selections": {} # { "column_name": "new_type_key", ... }
        }
        if params:
            default_params.update(params)
        super().__init__(node_id=node_id, node_type=self.NODE_TYPE, params=default_params)
        self.label = f"Type Caster #{self.id}"

    def process(self, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = self.get_input_df(inputs) # 부모 클래스의 헬퍼 함수로 입력 DF 가져오기
        
        type_selections = self.params.get("type_selections", {})
        if not type_selections:
            return df # 변경 사항 없으면 그대로 반환

        for col_name, new_type_key in type_selections.items():
            if col_name in df.columns:
                try:
                    df[col_name] = self._convert_series_type(df[col_name], new_type_key)
                except Exception as e:
                    print(f"Node {self.id}: Failed to convert column '{col_name}' to '{new_type_key}'. Error: {e}")
        return df

    def _convert_series_type(self, series: pd.Series, new_type_key: str) -> pd.Series:
        # 기존 step_01의 타입 변환 로직
        if new_type_key == "Numeric (int)": return pd.to_numeric(series, errors='coerce').astype('Int64')
        if new_type_key == "Numeric (float)": return pd.to_numeric(series, errors='coerce').astype(float)
        if new_type_key.startswith("Categorical"): return series.astype('category')
        if new_type_key.startswith("Datetime"): return pd.to_datetime(series, errors='coerce')
        if new_type_key.startswith("Timedelta"): return pd.to_timedelta(series, errors='coerce')
        if new_type_key.startswith("Text") or new_type_key.startswith("Potentially Sensitive"): return series.astype(str)
        return series # "Original" 또는 알 수 없는 타입은 그대로 반환

# --- UI Functions ---

def _populate_summary_tables(df: Optional[pd.DataFrame]):
    """선택된 노드의 출력 DF로 데이터 요약 테이블들을 채웁니다."""
    util_funcs = main_app_callbacks.get('get_util_funcs', lambda: {})()
    create_tbl_func = util_funcs.get('create_table_with_data')
    
    # Shape 텍스트 업데이트
    if dpg.does_item_exist(TAG_DL_SHAPE_TEXT):
        dpg.set_value(TAG_DL_SHAPE_TEXT, f"Shape: {df.shape}" if df is not None else "Shape: N/A")

    # 테이블 초기화 또는 채우기
    tables_to_update = [TAG_DL_INFO_TABLE, TAG_DL_DESCRIBE_TABLE, TAG_DL_HEAD_TABLE]
    if df is None or not create_tbl_func:
        for tbl in tables_to_update:
            if dpg.does_item_exist(tbl):
                dpg.delete_item(tbl, children_only=True)
                dpg.add_table_column(label="Info", parent=tbl)
                with dpg.table_row(parent=tbl): dpg.add_text("Select a node to inspect its output data.")
        return

    # Info 테이블
    info_df = pd.DataFrame({
        "Column": df.columns, "Dtype": df.dtypes.astype(str),
        "Missing": df.isnull().sum(), "Unique": df.nunique()
    }).reset_index(drop=True)
    create_tbl_func(TAG_DL_INFO_TABLE, info_df, parent_df_for_widths=info_df)

    # Describe 테이블
    num_df = df.select_dtypes(include=np.number)
    desc_df = num_df.describe().reset_index().rename(columns={'index': 'Statistic'}) if not num_df.empty else pd.DataFrame({"Info": ["No numeric columns"]})
    create_tbl_func(TAG_DL_DESCRIBE_TABLE, desc_df, utils_format_numeric=True, parent_df_for_widths=desc_df)

    # Head 테이블
    create_tbl_func(TAG_DL_HEAD_TABLE, df.head(), parent_df_for_widths=df)

def _on_type_selection_change(sender, app_data, user_data):
    """Type Caster 노드의 타입 설정을 변경합니다."""
    node_id = user_data["node_id"]
    col_name = user_data["col_name"]
    node = app_state.nodes.get(node_id)
    if node and isinstance(node, TypeCasterNode):
        node.params["type_selections"][col_name] = app_data

def _populate_type_editor_table(node: Optional[TypeCasterNode]):
    """Type Caster 노드의 상세 설정 UI를 채웁니다."""
    if not dpg.does_item_exist(TAG_DL_TYPE_EDITOR_TABLE): return
    dpg.delete_item(TAG_DL_TYPE_EDITOR_TABLE, children_only=True)

    if not isinstance(node, TypeCasterNode):
        dpg.add_table_column(label="Info", parent=TAG_DL_TYPE_EDITOR_TABLE)
        with dpg.table_row(parent=TAG_DL_TYPE_EDITOR_TABLE):
            dpg.add_text("Select a 'Type Caster' node to edit types.")
        return
        
    # 노드의 입력 데이터를 가져옴
    input_df = None
    if node.input_connections:
        source_id = list(node.input_connections.values())[0]
        input_df = app_state.node_outputs.get(source_id)

    if input_df is None:
        dpg.add_table_column(label="Info", parent=TAG_DL_TYPE_EDITOR_TABLE)
        with dpg.table_row(parent=TAG_DL_TYPE_EDITOR_TABLE):
            dpg.add_text("Connect an input to the Type Caster node.")
        return
        
    headers = ["Column", "Input Dtype", "Selected Type"]
    for header in headers: dpg.add_table_column(label=header, parent=TAG_DL_TYPE_EDITOR_TABLE)
    
    available_types = ["Original", "Numeric (int)", "Numeric (float)", "Categorical", "Datetime", "Text (General)"]
    type_selections = node.params.get("type_selections", {})

    for col_name in input_df.columns:
        with dpg.table_row(parent=TAG_DL_TYPE_EDITOR_TABLE):
            dpg.add_text(col_name)
            dpg.add_text(str(input_df[col_name].dtype))
            dpg.add_combo(available_types, default_value=type_selections.get(col_name, "Original"),
                          callback=_on_type_selection_change, width=-1,
                          user_data={"node_id": node.id, "col_name": col_name})

def create_ui(step_name: str, parent_container_tag: str, main_callbacks_ref: dict):
    """"Data Loading & Overview" 탭의 UI를 생성합니다."""
    with dpg.group(tag=TAG_DL_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text("--- Data Inspector ---")
        dpg.add_text("Select any node in the 'Preprocessing' editor to inspect its output data here.", wrap=-1)
        dpg.add_separator()
        
        dpg.add_text("Shape: N/A", tag=TAG_DL_SHAPE_TEXT)
        dpg.add_separator()
        
        with dpg.collapsing_header(label="Column Info", default_open=True):
            with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                           scrollY=True, height=200, tag=TAG_DL_INFO_TABLE, scrollX=True): pass
        
        with dpg.collapsing_header(label="Descriptive Statistics", default_open=True):
            with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                           scrollY=True, height=200, tag=TAG_DL_DESCRIBE_TABLE, scrollX=True): pass
        
        with dpg.collapsing_header(label="Data Head", default_open=True):
             with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp,
                           scrollY=True, height=150, tag=TAG_DL_HEAD_TABLE, scrollX=True): pass

def update_ui():
    """선택된 노드에 따라 데이터 인스펙터 UI를 업데이트합니다."""
    selected_node_id = app_state.selected_node_id
    df_to_display = None
    if selected_node_id and selected_node_id in app_state.node_outputs:
        df_to_display = app_state.node_outputs[selected_node_id]
    _populate_summary_tables(df_to_display)

# --- 더 이상 사용되지 않는 함수들 ---
# _infer_series_type, _apply_type_changes_and_process, _infer_all_types_and_populate 등은
# 이제 TypeCasterNode 내부 로직이나 별도의 유틸리티 함수로 흡수될 수 있습니다.
# 지금은 혼란을 피하기 위해 파일에서 제거하거나 주석 처리하는 것이 좋습니다.
# 여기서는 일단 삭제했습니다.