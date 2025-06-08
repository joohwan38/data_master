# step_04_missing_values.py (Node 클래스로 리팩토링 완료)

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# main_app에서 정의한 BaseNode를 임포트 (실제로는 별도 파일로 분리하는 것이 좋음)
from app_state_manager import BaseNode, app_state

# --- DPG Tags ---
TAG_MV_STEP_GROUP = "step4_missing_values_group"
TAG_MV_METHOD_SELECTION_TABLE = "step4_mv_method_selection_table"
# (다른 UI 태그들은 create_ui 내에서 지역적으로 사용되거나 필요시 여기에 정의)

# <<< NEW: MissingValueNode 클래스 정의 >>>
class MissingValueNode(BaseNode):
    NODE_TYPE = "missing_value_imputer"

    def __init__(self, node_id: int, params: dict | None = None):
        # 이 노드의 기본 파라미터 구조 정의
        default_params = {
            "imputation_methods": {} # { "column_name": ("method", "custom_value"), ... }
        }
        if params:
            default_params.update(params)

        super().__init__(node_id=node_id, node_type=self.NODE_TYPE, params=default_params)
        self.label = f"Missing Value Imputer #{self.id}"


    def process(self, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """결측치 처리 로직을 실행합니다."""
        df_to_process = self.get_input_df(inputs) # BaseNode의 헬퍼 함수 사용
        
        df_processed = df_to_process.copy()
        imputation_applied_any = False
        
        # 현재 노드의 파라미터에서 처리 방법들을 가져옴
        imputation_methods = self.params.get("imputation_methods", {})
        
        # Iterative Imputer를 사용할 컬럼이 있는지 확인
        use_iterative_imputer = any(method == "Iterative Imputer (MICE)" for method, _ in imputation_methods.values())

        if use_iterative_imputer:
            numeric_cols = [col for col, (method, _) in imputation_methods.items() if method == "Iterative Imputer (MICE)" and pd.api.types.is_numeric_dtype(df_processed[col].dtype)]
            
            if numeric_cols:
                print(f"Node {self.id}: Applying IterativeImputer to columns: {numeric_cols}")
                df_numeric_for_imputation = df_processed[numeric_cols].copy()
                
                try:
                    iter_imputer = IterativeImputer(random_state=0, max_iter=10)
                    imputed_values = iter_imputer.fit_transform(df_numeric_for_imputation)
                    df_processed[numeric_cols] = imputed_values
                    imputation_applied_any = True
                except Exception as e:
                    print(f"Node {self.id}: Error during IterativeImputer: {e}")

        # 개별 컬럼 처리
        for col_name, (method, custom_value) in imputation_methods.items():
            if col_name not in df_processed.columns or method in ["Do Not Impute", "Iterative Imputer (MICE)"]:
                continue

            if df_processed[col_name].isnull().sum() > 0:
                imputation_applied_any = True
                if method == "Remove Rows with Missing":
                    df_processed.dropna(subset=[col_name], inplace=True)
                else:
                    imputer = self._get_imputer_for_method(method, custom_value, df_processed[col_name].dtype)
                    if imputer:
                        try:
                            df_processed[[col_name]] = imputer.fit_transform(df_processed[[col_name]])
                        except Exception as e:
                            print(f"Node {self.id}: Could not impute '{col_name}' with method '{method}'. Error: {e}")

        if not imputation_applied_any:
            print(f"Node {self.id}: No effective imputation methods were applied.")
            
        return df_processed

    def _get_imputer_for_method(self, method: str, custom_value: Any, dtype: Any) -> Optional[SimpleImputer]:
        """메소드 문자열에 따라 적절한 Scikit-learn Imputer 객체를 반환합니다."""
        if method == "Impute with Mean" and pd.api.types.is_numeric_dtype(dtype):
            return SimpleImputer(strategy='mean')
        if method == "Impute with Median" and pd.api.types.is_numeric_dtype(dtype):
            return SimpleImputer(strategy='median')
        if method == "Impute with Mode":
            return SimpleImputer(strategy='most_frequent')
        if method == "Impute with 0":
            return SimpleImputer(strategy='constant', fill_value=0)
        if method == "Impute with Custom Value":
            fill_val_typed = custom_value
            try: # 원본 데이터 타입에 맞게 커스텀 값 변환 시도
                if pd.api.types.is_integer_dtype(dtype): fill_val_typed = int(custom_value)
                elif pd.api.types.is_float_dtype(dtype): fill_val_typed = float(custom_value)
            except (ValueError, TypeError): pass
            return SimpleImputer(strategy='constant', fill_value=fill_val_typed)
        return None

# --- UI 함수들 ---

def _on_imputation_method_change(sender, app_data_method: str, user_data: Dict):
    """UI에서 처리 방법 변경 시 호출되는 콜백. 선택된 노드의 파라미터를 업데이트합니다."""
    node_id = user_data["node_id"]
    col_name = user_data["col_name"]
    
    node = app_state.nodes.get(node_id)
    if not node: return

    # 노드의 파라미터 직접 수정
    current_selections = node.params.setdefault("imputation_methods", {})
    current_method, current_custom_val = current_selections.get(col_name, ("Do Not Impute", ""))
    
    current_selections[col_name] = (app_data_method, current_custom_val)
    
    # UI 다시 그리기 (파라미터 변경에 따른 UI 상태 변화 반영)
    update_ui(node_id_to_load=node_id)

def _on_custom_fill_value_change(sender, app_data_fill_value: str, user_data: Dict):
    """UI에서 커스텀 값 입력 시 호출되는 콜백."""
    node_id = user_data["node_id"]
    col_name = user_data["col_name"]
    
    node = app_state.nodes.get(node_id)
    if not node: return

    current_selections = node.params.setdefault("imputation_methods", {})
    current_method, _ = current_selections.get(col_name, ("Impute with Custom Value", ""))
    
    current_selections[col_name] = (current_method, app_data_fill_value)

def _populate_method_selection_table(df: Optional[pd.DataFrame], node: Optional[MissingValueNode]):
    """선택된 노드의 입력 데이터와 파라미터를 기반으로 UI 테이블을 채웁니다."""
    if not dpg.does_item_exist(TAG_MV_METHOD_SELECTION_TABLE): return
    dpg.delete_item(TAG_MV_METHOD_SELECTION_TABLE, children_only=True)

    if df is None or node is None:
        dpg.add_table_column(label="Info", parent=TAG_MV_METHOD_SELECTION_TABLE)
        with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
            dpg.add_text("Node's input data not available. Connect a parent node.")
        return

    # 헤더 생성
    headers = ["Column", "Dtype", "Missing", "Imputation Method", "Custom Value"]
    for header in headers: dpg.add_table_column(label=header, parent=TAG_MV_METHOD_SELECTION_TABLE)

    imputation_options_base = ["Do Not Impute", "Impute with Mean", "Impute with Median", "Impute with Mode", "Impute with 0", "Impute with Custom Value", "Remove Rows with Missing", "Iterative Imputer (MICE)"]
    node_selections = node.params.get("imputation_methods", {})

    # 결측치가 있는 컬럼만 표시
    cols_with_missing = [col for col in df.columns if df[col].isnull().sum() > 0]
    for col_name in cols_with_missing:
        with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
            dpg.add_text(col_name)
            dpg.add_text(str(df[col_name].dtype))
            dpg.add_text(str(df[col_name].isnull().sum()))
            
            # 옵션 필터링
            is_numeric = pd.api.types.is_numeric_dtype(df[col_name].dtype)
            options = [opt for opt in imputation_options_base if is_numeric or opt not in ["Impute with Mean", "Impute with Median", "Iterative Imputer (MICE)"]]

            method, custom_val = node_selections.get(col_name, ("Do Not Impute", ""))
            
            dpg.add_combo(options, default_value=method, width=-1,
                          user_data={"node_id": node.id, "col_name": col_name},
                          callback=_on_imputation_method_change)
            
            dpg.add_input_text(default_value=str(custom_val), width=-1,
                               show=(method == "Impute with Custom Value"),
                               user_data={"node_id": node.id, "col_name": col_name},
                               callback=_on_custom_fill_value_change)

def create_ui(step_name: str, parent_container_tag: str, main_callbacks_ref: dict):
    """결측치 처리 탭의 UI를 생성합니다."""
    # 이제 main_callbacks_ref는 전역 변수가 아닌 참조용으로만 사용
    with dpg.group(tag=TAG_MV_STEP_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_text("Set imputation methods for the selected 'Missing Value Imputer' node.")
        dpg.add_separator()
        
        with dpg.table(tag=TAG_MV_METHOD_SELECTION_TABLE, header_row=True, resizable=True,
                       policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=-1,
                       borders_outerH=True, borders_innerV=True, borders_innerH=True):
            dpg.add_table_column(label="Info", parent=TAG_MV_METHOD_SELECTION_TABLE)
            with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
                dpg.add_text("Select a 'Missing Value Imputer' node in the Node Editor.")

def update_ui(node_id_to_load: Optional[int] = None):
    """선택된 노드에 맞춰 UI를 업데이트합니다."""
    # 이제 이 함수는 'main_app'에서 특정 노드를 더블클릭했을 때 호출됩니다.
    if node_id_to_load is None:
        _populate_method_selection_table(None, None)
        return

    node = app_state.nodes.get(node_id_to_load)
    if not isinstance(node, MissingValueNode):
        _populate_method_selection_table(None, None)
        return

    # 노드의 입력 데이터 가져오기
    input_df = None
    if node.input_connections:
        # 간단히 첫 번째 입력을 사용한다고 가정
        source_node_id = list(node.input_connections.values())[0]
        # 입력 노드가 실행되었는지 확인
        if source_node_id in app_state.node_outputs:
            input_df = app_state.node_outputs[source_node_id]
        else:
            # 입력 데이터가 없으면 사용자에게 파이프라인 실행을 유도
            util_funcs = main_app_callbacks.get('get_util_funcs', lambda: {})()
            if '_show_simple_modal_message' in util_funcs:
                util_funcs['_show_simple_modal_message']("Info", f"Input for Node #{node.id} is not ready.\nPlease run the pipeline up to its parent node first.")

    _populate_method_selection_table(input_df, node)

# --- 기존의 reset, save, apply 함수들은 제거됩니다. ---
# 이 모든 기능은 이제 main_app.py의 PipelineManager와 Node 객체 자체에서 관리합니다.