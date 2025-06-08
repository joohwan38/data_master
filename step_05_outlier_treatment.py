# step_05_outlier_treatment.py (기능이 모두 채워진 최종 코드)

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
import functools

# <<< CHANGED >>>: 새로 만든 app_state_manager에서 필요한 것들을 가져옵니다.
from app_state_manager import BaseNode, app_state
# utils는 직접 임포트하여 사용합니다.
import utils as s5_utils

# PyOD, UMAP, SHAP 등 필요한 라이브러리 임포트
try:
    from pyod.models.hbos import HBOS
    from pyod.models.ecod import ECOD
    from pyod.models.iforest import IForest as PyOD_IForest
    import umap
    import shap
except ImportError as e:
    print(f"Warning: Missing libraries for OutlierNode (PyOD, UMAP, SHAP). {e}")
    HBOS, ECOD, PyOD_IForest, umap, shap = None, None, None, None, None

# --- UI Tags ---
TAG_OT_STEP_GROUP = "step5_outlier_treatment_group"
TAG_OT_UNIVARIATE_RESULTS_TABLE = "step5_uni_results_table"
TAG_OT_UNIVARIATE_TREATMENT_TABLE = "step5_uni_treatment_table"
TAG_OT_UNIVARIATE_VIS_GROUP = "step5_uni_vis_group"


class OutlierNode(BaseNode):
    NODE_TYPE = "outlier_handler"

    def __init__(self, node_id: int, params: dict | None = None):
        default_params = {
            "mode": "univariate",
            # 단변량 파라미터
            "uni_detection_method": "IQR",
            "uni_iqr_multiplier": 1.5,
            "uni_hbos_n_bins": 20,
            "uni_ecod_contamination": 0.1,
            "uni_treatment_selections": {},
            # 다변량 파라미터
            "mva_contamination": 0.1,
            "mva_col_selection_mode": "All Numeric",
            "mva_selected_columns_for_detection": [],
            # 처리 결과 저장용
            "detected_uni_outliers": {},
            "mva_outlier_info": {},
        }
        final_params = default_params.copy()
        if params:
            for key, value in params.items():
                if isinstance(value, dict) and isinstance(final_params.get(key), dict):
                    final_params[key].update(value)
                else:
                    final_params[key] = value
        
        super().__init__(node_id=node_id, node_type=self.NODE_TYPE, params=final_params)
        self.label = f"Outlier Handler #{self.id}"

    def process(self, inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = self.get_input_df(inputs).copy()

        treatment_selections = self.params.get("uni_treatment_selections", {})
        detected_indices = self.params.get("detected_uni_outliers", {})

        if not treatment_selections or not detected_indices:
            return df

        for col_name, treatment in treatment_selections.items():
            method = treatment.get("method")
            indices = detected_indices.get(col_name)

            if not method or method == "Do Not Treat" or not isinstance(indices, list) or not indices:
                continue

            valid_indices = [idx for idx in indices if idx in df.index]
            if not valid_indices: continue

            if method == "Treat as Missing":
                df.loc[valid_indices, col_name] = np.nan
            elif method == "Ratio-based Capping":
                lower_p = treatment.get("lower_percentile", 5) / 100.0
                upper_p = treatment.get("upper_percentile", 95) / 100.0
                lower_cap = df[col_name].quantile(lower_p)
                upper_cap = df[col_name].quantile(upper_p)
                df[col_name] = df[col_name].clip(lower=lower_cap, upper=upper_cap)
            elif method == "Absolute Value Capping":
                lower_b = treatment.get("abs_lower_bound")
                upper_b = treatment.get("abs_upper_bound")
                if lower_b is not None and upper_b is not None:
                    df[col_name] = df[col_name].clip(lower=lower_b, upper=upper_b)
        return df

# --- Helper Functions ---
def _get_node_and_input_df(node_id: Optional[int]) -> Tuple[Optional[OutlierNode], Optional[pd.DataFrame]]:
    if node_id is None: return None, None
    node = app_state.nodes.get(node_id)
    if not isinstance(node, OutlierNode): return None, None
    input_df = None
    if node.input_connections:
        source_id = list(node.input_connections.values())[0]
        input_df = app_state.node_outputs.get(source_id)
    return node, input_df

def _uni_param_change(sender, app_data, user_data):
    node_id = user_data["node_id"]
    param_name = user_data["param_name"]
    node = app_state.nodes.get(node_id)
    if isinstance(node, OutlierNode):
        node.params[param_name] = app_data
        _update_univariate_tab_ui(node)

def _uni_run_detection(sender, app_data, user_data):
    node_id = user_data.get("node_id")
    node, df = _get_node_and_input_df(node_id)
    if not all([node, df is not None]):
        return s5_utils.show_dpg_alert_modal("Error", "Node or input data not available.")

    method = node.params.get("uni_detection_method", "IQR")
    eligible_cols = [c for c in df.select_dtypes(include=np.number).columns if df[c].nunique() > 10]
    
    node.params['detected_uni_outliers'] = {}
    for col in eligible_cols:
        series = df[col].dropna()
        if series.empty: continue
        
        indices = np.array([])
        try:
            if method == "IQR":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    multiplier = node.params.get("uni_iqr_multiplier", 1.5)
                    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
                    indices = series[(series < lower) | (series > upper)].index
            elif method == "HBOS" and HBOS:
                model = HBOS(n_bins=node.params.get('uni_hbos_n_bins', 20), contamination=node.params.get('uni_ecod_contamination', 0.1))
                model.fit(series.values.reshape(-1, 1))
                indices = series.index[model.labels_ == 1]
            elif method == "ECOD" and ECOD:
                model = ECOD(contamination=node.params.get('uni_ecod_contamination', 0.1))
                model.fit(series.values.reshape(-1, 1))
                indices = series.index[model.labels_ == 1]
        except Exception as e:
            print(f"Error detecting outliers for column {col}: {e}")

        if len(indices) > 0:
            node.params['detected_uni_outliers'][col] = indices.tolist()

    _update_univariate_tab_ui(node)
    s5_utils.show_dpg_alert_modal("Success", "Univariate outlier detection complete.")

def _update_univariate_tab_ui(node: OutlierNode):
    # Detection Results Table 업데이트
    results_table = "step5_uni_results_table"
    dpg.delete_item(results_table, children_only=True)
    dpg.add_table_column(label="Column", parent=results_table)
    dpg.add_table_column(label="Detected Outliers", parent=results_table)
    for col, indices in node.params.get("detected_uni_outliers", {}).items():
        with dpg.table_row(parent=results_table):
            dpg.add_text(col)
            dpg.add_text(str(len(indices)))

def _create_univariate_tab(parent_tag, node_id):
    node = app_state.nodes.get(node_id)
    if not isinstance(node, OutlierNode): return

    with dpg.tab(label="Univariate", parent=parent_tag, tag=f"uni_tab_{node_id}"):
        with dpg.group(horizontal=True):
            dpg.add_text("Detection Method:")
            dpg.add_radio_button(
                items=["IQR", "HBOS", "ECOD"], horizontal=True,
                default_value=node.params.get("uni_detection_method", "IQR"),
                user_data={"node_id": node_id, "param_name": "uni_detection_method"},
                callback=_uni_param_change
            )
        
        with dpg.group(horizontal=True, show=node.params.get("uni_detection_method") == "IQR"):
            dpg.add_text("IQR Multiplier:")
            dpg.add_input_float(default_value=node.params.get("uni_iqr_multiplier", 1.5),
                                 user_data={"node_id": node_id, "param_name": "uni_iqr_multiplier"}, callback=_uni_param_change)
        
        dpg.add_button(label="Run Univariate Detection", user_data={"node_id": node_id}, callback=_uni_run_detection, width=-1)
        dpg.add_separator()
        dpg.add_text("Detection Results:")
        with dpg.table(tag="step5_uni_results_table", header_row=True, height=150, scrollY=True): pass
        
        dpg.add_separator()
        dpg.add_text("Treatment Settings:")
        with dpg.table(tag="step5_uni_treatment_table", header_row=True, height=150, scrollY=True): pass


def _create_multivariate_tab(parent_tag, node_id):
     with dpg.tab(label="Multivariate", parent=parent_tag, tag=f"multi_tab_{node_id}"):
        dpg.add_text("Multivariate outlier detection logic is under construction.")

def create_ui(step_name: str, parent_container_tag: str, main_callbacks_ref: dict):
    with dpg.group(tag=TAG_OT_STEP_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_text("Configure for the selected 'Outlier Handler' node.")
        dpg.add_separator()
        
        with dpg.group(tag="outlier_node_content_group"):
            with dpg.tab_bar(tag="outlier_mode_tab_bar"): pass
        
        dpg.add_text("Select an 'Outlier Handler' node in the editor.", tag="outlier_node_select_prompt", show=True)

def update_ui(node_id_to_load: Optional[int] = None):
    prompt_tag, content_tag, tab_bar_tag = "outlier_node_select_prompt", "outlier_node_content_group", "outlier_mode_tab_bar"
    node = app_state.nodes.get(node_id_to_load)
    is_outlier_node = isinstance(node, OutlierNode)

    if dpg.does_item_exist(prompt_tag): dpg.configure_item(prompt_tag, show=not is_outlier_node)
    if dpg.does_item_exist(content_tag): dpg.configure_item(content_tag, show=is_outlier_node)

    if is_outlier_node:
        dpg.delete_item(tab_bar_tag, children_only=True)
        _create_univariate_tab(tab_bar_tag, node_id_to_load)
        _create_multivariate_tab(tab_bar_tag, node_id_to_load)
        
        mode = node.params.get("mode", "univariate")
        active_tab_tag = f"uni_tab_{node_id_to_load}" if mode == "univariate" else f"multi_tab_{node_id_to_load}"
        dpg.set_value(tab_bar_tag, active_tab_tag)