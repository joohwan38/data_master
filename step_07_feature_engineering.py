# step_07_feature_engineering.py (Node 클래스 기반으로 리팩토링 완료)

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

# main_app의 BaseNode와 핵심 상태/콜백 객체를 가져옵니다.
from app_state_manager import BaseNode, app_state

# --- UI Tags ---
TAG_S7_MAIN_GROUP = "step7_main_group"
# (다른 UI 태그들은 이제 각 노드의 상세 설정 UI에서 지역적으로 관리됨)

# --- Node Class Definitions ---

class BinningNode(BaseNode):
    """연속형 변수를 구간으로 나누는(Binning) 노드입니다."""
    NODE_TYPE = "binning_node"

    def __init__(self, node_id: int, params: dict | None = None):
        default_params = {
            "target_column": None,
            "new_column_name": "binned_column",
            "method": "qcut",  # 'qcut' (Equal Frequency) or 'cut' (Equal Width)
            "bins": 4,         # 정수 또는 리스트 형태의 경계값
            "labels": None,    # 리스트 또는 False
        }
        if params:
            default_params.update(params)

        super().__init__(node_id=node_id, node_type=self.NODE_TYPE, params=default_params)
        self.label = f"Binning #{self.id}"

    def process(self, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = self.get_input_df(inputs)
        
        target_col = self.params.get("target_column")
        new_col_name = self.params.get("new_column_name", f"{target_col}_binned")
        method = self.params.get("method", "qcut")
        bins = self.params.get("bins", 4)
        labels = self.params.get("labels") # UI에서 문자열로 받았다면 리스트로 변환 필요

        if not target_col or target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in input DataFrame.")

        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise TypeError(f"Binning target column '{target_col}' must be numeric.")

        # 레이블 파라미터 처리 (UI에서 문자열 "a,b,c"로 오면 리스트로 변환)
        processed_labels = None
        if isinstance(labels, str) and labels.strip():
            processed_labels = [label.strip() for label in labels.split(',')]
        elif isinstance(labels, list):
            processed_labels = labels

        # bins 파라미터 처리 (UI에서 문자열 "[0,10,20]"로 오면 리스트로 변환)
        processed_bins = bins
        if isinstance(bins, str) and bins.startswith('[') and bins.endswith(']'):
            try:
                processed_bins = list(map(float, bins.strip('[]').split(',')))
            except ValueError:
                raise ValueError("Bin edges list is in an incorrect format.")

        try:
            if method == "qcut":
                df[new_col_name] = pd.qcut(df[target_col], q=processed_bins, labels=processed_labels, duplicates='drop')
            elif method == "cut":
                df[new_col_name] = pd.cut(df[target_col], bins=processed_bins, labels=processed_labels, right=False)
            else:
                raise ValueError(f"Unknown binning method: {method}")
        except Exception as e:
            raise RuntimeError(f"Failed to apply binning on '{target_col}': {e}")

        return df

# --- 다른 Feature Engineering 노드들 (향후 추가) ---
# class ArithmeticNode(BaseNode): ...
# class ConditionalNode(BaseNode): ...
# class DateTimeFeatureNode(BaseNode): ...

# --- UI Functions ---

def _on_binning_param_change(sender, app_data, user_data):
    """BinningNode의 파라미터를 UI에서 변경 시 호출되는 콜백."""
    node_id = user_data.get("node_id")
    param_name = user_data.get("param_name")
    
    node = app_state.nodes.get(node_id)
    if isinstance(node, BinningNode):
        node.params[param_name] = app_data

def _create_binning_node_editor(parent_tag, node: BinningNode):
    """BinningNode 전용 상세 설정 UI를 생성합니다."""
    # 노드의 입력 데이터를 가져와서 숫자형 컬럼 목록을 만듦
    input_df = None
    if node.input_connections:
        source_id = list(node.input_connections.values())[0]
        input_df = app_state.node_outputs.get(source_id)
    
    numeric_cols = input_df.select_dtypes(include=np.number).columns.tolist() if input_df is not None else []

    # UI 요소 생성 및 현재 노드 파라미터 값으로 설정
    dpg.add_text("Target Column for Binning:", parent=parent_tag)
    dpg.add_combo(numeric_cols, label="Target", default_value=node.params.get("target_column", ""),
                  parent=parent_tag, width=-1,
                  user_data={"node_id": node.id, "param_name": "target_column"},
                  callback=_on_binning_param_change)

    dpg.add_input_text(label="New Column Name", default_value=node.params.get("new_column_name", ""),
                       parent=parent_tag, width=-1,
                       user_data={"node_id": node.id, "param_name": "new_column_name"},
                       callback=_on_binning_param_change)
    
    dpg.add_radio_button(label="Method", items=["qcut (Equal Frequency)", "cut (Equal Width)"],
                         default_value=node.params.get("method", "qcut"), horizontal=True,
                         parent=parent_tag,
                         user_data={"node_id": node.id, "param_name": "method"},
                         callback=_on_binning_param_change)

    dpg.add_input_text(label="Bins / Quantiles", hint="e.g., 4 or [0, 25, 50, 100]",
                       default_value=str(node.params.get("bins", 4)),
                       parent=parent_tag, width=-1,
                       user_data={"node_id": node.id, "param_name": "bins"},
                       callback=_on_binning_param_change)

    dpg.add_input_text(label="Labels (optional)", hint="e.g., low,med,high",
                       default_value=node.params.get("labels", "") or "",
                       parent=parent_tag, width=-1,
                       user_data={"node_id": node.id, "param_name": "labels"},
                       callback=_on_binning_param_change)

def create_ui(step_name: str, parent_container_tag: str, main_callbacks_ref: dict):
    """Feature Engineering 탭의 UI를 생성합니다. (이제 상세 설정 화면 역할)"""
    with dpg.group(tag=TAG_S7_MAIN_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_text("Configure parameters for the selected Feature Engineering node.", wrap=-1)
        dpg.add_separator()

        # 이 그룹은 선택된 노드의 타입에 따라 내용이 동적으로 채워집니다.
        with dpg.group(tag="feature_engineering_node_editor_group"):
            dpg.add_text("Select a feature engineering node (e.g., Binning, Arithmetic) in the editor.",
                         tag="fe_node_select_prompt")

def update_ui(node_id_to_load: Optional[int] = None):
    """선택된 노드에 맞춰 UI를 업데이트합니다."""
    editor_group = "feature_engineering_node_editor_group"
    prompt_tag = "fe_node_select_prompt"
    
    if not dpg.does_item_exist(editor_group): return
    dpg.delete_item(editor_group, children_only=True)
    
    node = app_state.nodes.get(node_id_to_load)
    
    # 선택된 노드 타입에 맞는 상세 에디터 UI를 생성
    if isinstance(node, BinningNode):
        _create_binning_node_editor(editor_group, node)
    # elif isinstance(node, ArithmeticNode):
    #     _create_arithmetic_node_editor(editor_group, node)
    else:
        dpg.add_text("Select a feature engineering node (e.g., Binning) in the editor.", 
                     parent=editor_group, tag=prompt_tag)