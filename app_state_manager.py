# app_state_manager.py (새 파일)

import pandas as pd
from typing import Dict, Any

# --- Node Base Class Definition ---
class BaseNode:
    """모든 처리 노드의 부모 클래스입니다."""
    NODE_TYPE = "Base"

    def __init__(self, node_id: int, params: dict | None = None):
        self.id = node_id
        self.type = self.NODE_TYPE
        self.params = params if params is not None else {}
        self.label = f"{self.type.replace('_', ' ').title()} #{self.id}"
        self.input_connections: dict[str, int] = {}
        # UI 정보를 저장할 딕셔너리 추가
        self.ui_info: dict[str, Any] = {}

    def process(self, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        raise NotImplementedError(f"`process` method not implemented for node type {self.type}")

    def get_input_df(self, inputs: dict[str, pd.DataFrame], input_name: str = "DataFrame In") -> pd.DataFrame:
        input_df = inputs.get(input_name)
        if input_df is None:
            raise ValueError(f"'{input_name}' is missing for Node {self.id} ({self.type})")
        return input_df.copy()

# --- Global Application State ---
class AppState:
    def __init__(self):
        self.original_df: pd.DataFrame | None = None
        self.loaded_file_path: str | None = None
        self.selected_target_variable: str | None = None
        self.selected_target_variable_type: str = "Continuous"
        
        self.nodes: dict[int, BaseNode] = {}
        self.node_outputs: dict[int, pd.DataFrame] = {}
        self.links: list[tuple[int, int]] = []
        
        self.step_group_tags: dict[str, str] = {}
        self.module_ui_updaters: dict[str, callable] = {}
        self.active_step_name: str | None = None
        self.selected_node_id: int | None = None
        self.ai_analysis_log: str = ""
        self.node_id_counter = 0

# --- Singleton Instance ---
# 애플리케이션 전체에서 단 하나의 app_state 인스턴스를 공유합니다.
app_state = AppState()