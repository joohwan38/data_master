# step_06_standardization.py (Node 클래스로 리팩토링 완료)

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# main_app의 BaseNode와 핵심 상태/콜백 객체를 가져옵니다.
from app_state_manager import BaseNode, app_state

# --- UI Tags ---
TAG_S6_MAIN_GROUP = "step6_main_group"
TAG_S6_SCALER_RADIO = "step6_scaler_method_radio"
TAG_S6_APPLY_BUTTON = "step6_apply_button" # 이 버튼은 이제 시각화 생성용으로 역할 변경
TAG_S6_VISUALIZATION_GROUP = "step6_visualization_group"

# --- Node Class Definition ---

class StandardizationNode(BaseNode):
    NODE_TYPE = "standardization_scaler"

    def __init__(self, node_id: int, params: dict | None = None):
        default_params = {
            "scaler_method": "StandardScaler", # "StandardScaler", "MinMaxScaler", "RobustScaler"
        }
        if params:
            default_params.update(params)
        
        super().__init__(node_id=node_id, node_type=self.NODE_TYPE, params=default_params)
        self.label = f"Scaler #{self.id}"

    def process(self, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """선택된 스케일러를 사용하여 숫자형 특성을 변환합니다."""
        df = self.get_input_df(inputs)
        
        scaler_method = self.params.get("scaler_method", "StandardScaler")
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            print(f"Node {self.id}: No numeric columns to scale.")
            return df # 숫자형 컬럼이 없으면 아무것도 하지 않음

        scaler = None
        if scaler_method == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_method == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaler_method == "RobustScaler":
            scaler = RobustScaler()

        if scaler:
            try:
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                print(f"Node {self.id}: Applied {scaler_method} to {len(numeric_cols)} numeric columns.")
            except Exception as e:
                print(f"Node {self.id}: Error during scaling with {scaler_method}. Error: {e}")
        
        return df

# --- UI Functions ---

def _on_scaler_changed(sender, app_data_method: str, user_data: dict):
    """UI에서 스케일러 메소드 변경 시 노드의 파라미터를 업데이트합니다."""
    node_id = user_data.get("node_id")
    node = app_state.nodes.get(node_id)
    if isinstance(node, StandardizationNode):
        node.params["scaler_method"] = app_data_method
        # 파라미터 변경 시, 시각화는 다시 생성해야 하므로 기존 시각화는 지움
        _clear_visualizations()

def _generate_comparison_plots(sender, app_data, user_data: dict):
    """선택된 노드의 입력과 출력을 비교하는 시각화를 생성합니다."""
    node_id = user_data.get("node_id")
    node = app_state.nodes.get(node_id)
    if not isinstance(node, StandardizationNode): return

    # 입력 데이터 가져오기
    input_df = None
    if node.input_connections:
        source_id = list(node.input_connections.values())[0]
        input_df = app_state.node_outputs.get(source_id)
    
    # 출력 데이터 가져오기 (이미 실행되었다면)
    output_df = app_state.node_outputs.get(node_id)

    if input_df is None or output_df is None:
        util_funcs = main_app_callbacks.get('get_util_funcs', lambda: {})()
        if '_show_simple_modal_message' in util_funcs:
            util_funcs['_show_simple_modal_message']("Info", "Input or Output data for this node is not available. Please run the pipeline first.")
        return
        
    _clear_visualizations()

    util_funcs = main_app_callbacks.get('get_util_funcs', lambda: {})()
    plot_func = util_funcs.get('plot_to_dpg_texture')
    if not plot_func: return

    numeric_cols = input_df.select_dtypes(include=np.number).columns.tolist()
    cols_to_plot = [col for col in numeric_cols if input_df[col].nunique() > 10][:10]

    if not cols_to_plot:
        dpg.add_text("No suitable columns to visualize.", parent=TAG_S6_VISUALIZATION_GROUP)
        return
        
    for col in cols_to_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=100)
        sns.histplot(input_df[col].dropna(), kde=True, ax=ax1, color='skyblue', bins=30)
        ax1.set_title(f'Before: {col}', fontsize=10)
        
        sns.histplot(output_df[col].dropna(), kde=True, ax=ax2, color='salmon', bins=30)
        ax2.set_title(f'After: {node.params["scaler_method"]}', fontsize=10)
        
        plt.tight_layout()
        
        try:
            tex_tag, w, h, _ = plot_func(fig)
            if tex_tag and dpg.does_item_exist(TAG_S6_VISUALIZATION_GROUP):
                dpg.add_image(tex_tag, width=w, height=h, parent=TAG_S6_VISUALIZATION_GROUP)
                dpg.add_separator(parent=TAG_S6_VISUALIZATION_GROUP)
        finally:
            plt.close(fig)

def _clear_visualizations():
    """기존 시각화 결과물을 삭제합니다."""
    if dpg.does_item_exist(TAG_S6_VISUALIZATION_GROUP):
        dpg.delete_item(TAG_S6_VISUALIZATION_GROUP, children_only=True)
        dpg.add_text("Click 'Generate Previews' to see before/after plots.", parent=TAG_S6_VISUALIZATION_GROUP)

def create_ui(step_name: str, parent_container_tag: str, main_callbacks_ref: dict):
    with dpg.group(tag=TAG_S6_MAIN_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_text("Configure scaler for the selected 'Scaler' node.")
        dpg.add_separator()
        
        with dpg.group(tag="scaler_node_content_group"):
            dpg.add_text("Select a method to standardize (scale) numeric features.")
            dpg.add_radio_button(
                items=["StandardScaler", "MinMaxScaler", "RobustScaler"],
                tag=TAG_S6_SCALER_RADIO,
                horizontal=True
            )
            dpg.add_button(
                label="Generate Before/After Previews",
                tag=TAG_S6_APPLY_BUTTON,
                width=-1, height=30
            )
            dpg.add_separator()
            dpg.add_text("Comparison Plots", color=[255, 255, 0])
            with dpg.child_window(tag=TAG_S6_VISUALIZATION_GROUP, border=True, height=-1):
                dpg.add_text("Click 'Generate Previews' to see results.")

        dpg.add_text("Select a 'Scaler' node in the editor to see options.", 
                     tag="scaler_node_select_prompt", show=True)

def update_ui(node_id_to_load: Optional[int] = None):
    """선택된 노드에 맞춰 UI를 업데이트합니다."""
    prompt_tag = "scaler_node_select_prompt"
    content_tag = "scaler_node_content_group"
    
    node = app_state.nodes.get(node_id_to_load) if node_id_to_load else None
    is_scaler_node = isinstance(node, StandardizationNode)
    
    if dpg.does_item_exist(prompt_tag): dpg.configure_item(prompt_tag, show=not is_scaler_node)
    if dpg.does_item_exist(content_tag): dpg.configure_item(content_tag, show=is_scaler_node)
    
    if is_scaler_node:
        # UI 요소들의 콜백과 값을 현재 노드에 맞게 설정
        dpg.set_value(TAG_S6_SCALER_RADIO, node.params.get("scaler_method", "StandardScaler"))
        dpg.set_item_callback(TAG_S6_SCALER_RADIO, _on_scaler_changed)
        dpg.set_item_user_data(TAG_S6_SCALER_RADIO, {"node_id": node_id_to_load})
        
        dpg.set_item_callback(TAG_S6_APPLY_BUTTON, _generate_comparison_plots)
        dpg.set_item_user_data(TAG_S6_APPLY_BUTTON, {"node_id": node_id_to_load})
        
        # 노드가 바뀌면 이전 시각화는 초기화
        _clear_visualizations()