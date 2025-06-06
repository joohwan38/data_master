# step_07_feature_engineering.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import traceback
import re

# --- Module State ---
_main_app_callbacks: Optional[Dict[str, Any]] = None
_util_funcs: Optional[Dict[str, Any]] = None
_current_df_input: Optional[pd.DataFrame] = None
_execution_list: List[Dict[str, Any]] = [] 
_available_columns_df: Optional[pd.DataFrame] = None 

# --- UI Tags ---
TAG_S7_MAIN_GROUP = "step7_main_group"
TAG_S7_SIDEBAR_WINDOW = "step7_sidebar_window"
TAG_S7_COLUMN_TABLE = "step7_column_info_table"
TAG_S7_EXECUTION_TABLE = "step7_execution_list_table"
TAG_S7_RUN_PIPELINE_BUTTON = "step7_run_pipeline_button"
TAG_S7_SOURCE_DF_INFO_TEXT = "step7_source_df_info_text"

# Tab 1: Arithmetic
# TAG_S7_ARITH_NEW_COL_INPUT  <- 삭제됨
TAG_S7_ARITH_FORMULA_TEXT = "step7_arith_formula_text"
TAG_S7_ARITH_OPERATOR_COMBO = "step7_arith_operator_combo"

# Tab 4: Advanced Syntax
TAG_S7_ADV_SYNTAX_INPUT = "step7_adv_syntax_input"

# --- Helper Functions ---

def _log(message: str):
    print(f"[Step7 FeatureEngineering] {message}")

def _refresh_sidebar():
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_S7_COLUMN_TABLE):
        return
    dpg.delete_item(TAG_S7_COLUMN_TABLE, children_only=True)
    if _available_columns_df is None or _available_columns_df.empty:
        dpg.add_table_column(label="Info", parent=TAG_S7_COLUMN_TABLE)
        with dpg.table_row(parent=TAG_S7_COLUMN_TABLE):
            dpg.add_text("No data available.")
        return
    headers = ["Variable", "Dtype"]
    for header in headers:
        dpg.add_table_column(label=header, parent=TAG_S7_COLUMN_TABLE)
    for _, row in _available_columns_df.iterrows():
        with dpg.table_row(parent=TAG_S7_COLUMN_TABLE):
            dpg.add_selectable(label=str(row["Variable"]), span_columns=True, callback=_on_sidebar_variable_double_clicked, user_data=str(row["Variable"]))
            dpg.add_text(str(row["Dtype"]))

def _on_sidebar_variable_double_clicked(sender, app_data, user_data):
    variable_name = f"df['{user_data}']"
    if dpg.is_item_visible(TAG_S7_ARITH_FORMULA_TEXT):
        current_formula = dpg.get_value(TAG_S7_ARITH_FORMULA_TEXT)
        operator = dpg.get_value(TAG_S7_ARITH_OPERATOR_COMBO)
        if not current_formula or current_formula.strip().endswith(('+', '-', '*', '/')):
            new_formula = f"{current_formula} {variable_name}"
        else:
            new_formula = f"{current_formula} {operator} {variable_name}"
        dpg.set_value(TAG_S7_ARITH_FORMULA_TEXT, new_formula.strip())
    elif dpg.is_item_visible(TAG_S7_ADV_SYNTAX_INPUT):
        dpg.set_value(TAG_S7_ADV_SYNTAX_INPUT, f"df['new_variable'] = {variable_name}")

def _add_to_execution_list(output_col: str, method: str, syntax: str):
    global _available_columns_df
    if not output_col:
        _util_funcs['_show_simple_modal_message']("Error", "Output variable name cannot be empty.")
        return
    if output_col in _available_columns_df["Variable"].tolist():
        _util_funcs['_show_simple_modal_message']("Error", f"Variable '{output_col}' already exists.")
        return
    op_details = {"output_col": output_col, "method": method, "syntax": syntax}
    _execution_list.append(op_details)
    _update_execution_list_table()
    new_row = pd.DataFrame([{"Variable": output_col, "Dtype": "derived", "Unique": "N/A"}])
    _available_columns_df = pd.concat([_available_columns_df, new_row], ignore_index=True)
    _refresh_sidebar()

# ** NEW: 실행 목록 테이블 내에서 변수명 변경 시 호출되는 콜백 **
def _on_output_var_name_changed(sender, new_name, user_data):
    global _available_columns_df
    index = user_data["index"]
    old_name = user_data["old_name"]

    if not new_name.isidentifier():
        _util_funcs['_show_simple_modal_message']("Error", f"'{new_name}' is not a valid variable name.")
        dpg.set_value(sender, old_name)
        return

    if new_name == old_name: return

    all_other_names = _available_columns_df['Variable'][_available_columns_df['Variable'] != old_name].tolist()
    if new_name in all_other_names:
        _util_funcs['_show_simple_modal_message']("Error", f"Variable name '{new_name}' already exists.")
        dpg.set_value(sender, old_name)
        return

    op = _execution_list[index]
    op['output_col'] = new_name
    formula_part = op['syntax'].split('=', 1)[1]
    op['syntax'] = f"df['{new_name}'] = {formula_part.strip()}"

    _available_columns_df.loc[_available_columns_df['Variable'] == old_name, 'Variable'] = new_name
    _refresh_sidebar()
    dpg.configure_item(sender, user_data={"index": index, "old_name": new_name})
    _update_execution_list_table()


def _update_execution_list_table():
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_S7_EXECUTION_TABLE):
        return
    dpg.delete_item(TAG_S7_EXECUTION_TABLE, children_only=True)
    headers = ["Output Variable (Editable)", "Method", "Definition (Pandas Syntax)", "Action"]
    for header in headers:
        dpg.add_table_column(label=header, parent=TAG_S7_EXECUTION_TABLE)

    for i, op in enumerate(_execution_list):
        with dpg.table_row(parent=TAG_S7_EXECUTION_TABLE):
            # ** CHANGED: 텍스트 대신 입력 필드를 사용해 수정 가능하게 변경 **
            dpg.add_input_text(
                default_value=op.get("output_col", "N/A"),
                callback=_on_output_var_name_changed,
                user_data={"index": i, "old_name": op.get("output_col")},
                width=-1,
                on_enter=True # 엔터키로 수정 완료
            )
            dpg.add_text(op.get("method", "N/A"))
            dpg.add_text(op.get("syntax", "N/A"))
            dpg.add_button(label="Delete", user_data=i, callback=_delete_operation)

def _delete_operation(sender, app_data, user_data):
    global _available_columns_df
    index_to_delete = user_data
    deleted_op = _execution_list.pop(index_to_delete)
    _available_columns_df = _available_columns_df[_available_columns_df["Variable"] != deleted_op["output_col"]]
    _update_execution_list_table()
    _refresh_sidebar()

def _add_arithmetic_op():
    # ** REFACTORED: 변수명 자동 생성 로직 **
    formula = dpg.get_value(TAG_S7_ARITH_FORMULA_TEXT)
    operator = dpg.get_value(TAG_S7_ARITH_OPERATOR_COMBO)

    if not formula:
        _util_funcs['_show_simple_modal_message']("Input Error", "Please build a formula first.")
        return
        
    op_map = {'+': 'sum', '-': 'sub', '*': 'mul', '/': 'div'}
    base_name = op_map.get(operator, 'op') + '_var'
    
    i = 1
    all_column_names = _available_columns_df['Variable'].tolist()
    while True:
        new_col = f"{base_name}{i}"
        if new_col not in all_column_names:
            break
        i += 1
    
    syntax = f"df['{new_col}'] = {formula}"
    
    _add_to_execution_list(new_col, "Arithmetic", syntax)
    dpg.set_value(TAG_S7_ARITH_FORMULA_TEXT, "")

def _add_advanced_syntax_op():
    syntax = dpg.get_value(TAG_S7_ADV_SYNTAX_INPUT)
    if not syntax or 'df[' not in syntax or '=' not in syntax:
        _util_funcs['_show_simple_modal_message']("Input Error", "Please provide valid pandas syntax (e.g., df['new_col'] = ...).")
        return
    try:
        new_col = re.search(r"df\[['\"](.+?)['\"]\]", syntax.split('=')[0]).group(1)
    except Exception:
        _util_funcs['_show_simple_modal_message']("Syntax Error", "Could not automatically determine the new variable name.\nPlease ensure it is in the format df['your_new_variable_name'] = ...")
        return
    _add_to_execution_list(new_col, "Advanced Syntax", syntax)
    dpg.set_value(TAG_S7_ADV_SYNTAX_INPUT, "")

def _run_pipeline():
    if not _execution_list:
        _util_funcs['_show_simple_modal_message']("Info", "No operations to run.")
        return
    if _current_df_input is None:
        _util_funcs['_show_simple_modal_message']("Error", "Input data is not available.")
        return
    df = _current_df_input.copy()
    _log(f"Starting pipeline with {_execution_list}")
    try:
        exec_scope = {'df': df, 'pd': pd, 'np': np}
        for op in _execution_list:
            syntax = op.get('syntax')
            if not syntax:
                print(f"WARNING: Skipping operation with no syntax: {op}")
                continue
            print(f"Executing: {syntax}")
            exec(syntax, exec_scope)
        processed_df = exec_scope['df']
        _log("Pipeline execution successful.")
        _main_app_callbacks['step7_feature_engineering_complete'](processed_df)
    except Exception as e:
        error_msg = f"Pipeline execution failed: {e}\n\n{traceback.format_exc()}"
        _log(error_msg)
        _util_funcs['_show_simple_modal_message']("Pipeline Error", error_msg, width=600, height=400)
        _main_app_callbacks['step7_feature_engineering_complete'](None)


# --- Main UI Creation and Update Functions ---

def create_ui(step_name: str, parent_tag: str, main_app_callbacks: Dict[str, Any]):
    global _main_app_callbacks, _util_funcs
    _main_app_callbacks = main_app_callbacks
    _util_funcs = main_app_callbacks.get('get_util_funcs', lambda: {})()
    main_app_callbacks['register_step_group_tag'](step_name, TAG_S7_MAIN_GROUP)
    main_app_callbacks['register_module_updater'](step_name, update_ui)
    
    with dpg.group(tag=TAG_S7_MAIN_GROUP, parent=parent_tag, show=False):
        dpg.add_text(f"--- {step_name} ---", color=[255, 255, 0])
        dpg.add_separator()
        with dpg.group(horizontal=True):
            with dpg.child_window(tag=TAG_S7_SIDEBAR_WINDOW, width=280):
                dpg.add_text("Variable Explorer")
                dpg.add_button(label="Refresh List", width=-1, callback=_refresh_sidebar)
                dpg.add_separator()
                with dpg.table(tag=TAG_S7_COLUMN_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True): pass
            with dpg.group():
                dpg.add_text("Source DataFrame: N/A", tag=TAG_S7_SOURCE_DF_INFO_TEXT)
                dpg.add_button(label="▶️ Run Feature Engineering", tag=TAG_S7_RUN_PIPELINE_BUTTON, width=-1, height=30, callback=_run_pipeline)
                dpg.add_text("Execution List:")
                with dpg.table(tag=TAG_S7_EXECUTION_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=200): pass
                dpg.add_separator()
                with dpg.tab_bar():
                    with dpg.tab(label="Arithmetic Operations"):
                        # ** CHANGED: "New Variable Name" 입력란 삭제 **
                        dpg.add_text("Formula Builder (Double-click variable on left):")
                        with dpg.group(horizontal=True):
                           dpg.add_input_text(tag=TAG_S7_ARITH_FORMULA_TEXT, readonly=True, width=-100)
                           dpg.add_combo(items=['+', '-', '*', '/'], tag=TAG_S7_ARITH_OPERATOR_COMBO, default_value='+', width=80)
                        dpg.add_button(label="Add to List", width=-1, callback=_add_arithmetic_op)
                    
                    with dpg.tab(label="Advanced Syntax"):
                        dpg.add_text("Enter full pandas syntax below.", wrap=500)
                        dpg.add_text("Example: df['new_col'] = df['col_A'] / (df['col_B'] + 1)", color=[200, 200, 200])
                        dpg.add_input_text(tag=TAG_S7_ADV_SYNTAX_INPUT, multiline=True, width=-1, height=120)
                        dpg.add_button(label="Add Syntax to List", width=-1, callback=_add_advanced_syntax_op)

def update_ui(df_input: Optional[pd.DataFrame], main_app_callbacks: Dict[str, Any]):
    global _current_df_input, _main_app_callbacks, _available_columns_df
    _main_app_callbacks = main_app_callbacks
    if df_input is None: return
    _current_df_input = df_input
    
    input_cols_df = pd.DataFrame({"Variable": df_input.columns, "Dtype": [str(d) for d in df_input.dtypes], "Unique": df_input.nunique().values})
    current_cols_set = set(input_cols_df['Variable'])
    
    valid_derived_ops = []
    for op in _execution_list:
        if op['output_col'] not in current_cols_set:
            valid_derived_ops.append(op)
    _execution_list[:] = valid_derived_ops
    
    derived_cols_df = pd.DataFrame([{"Variable": op["output_col"], "Dtype": "derived", "Unique": "N/A"} for op in _execution_list])
    
    if not derived_cols_df.empty:
        _available_columns_df = pd.concat([input_cols_df, derived_cols_df], ignore_index=True)
    else:
        _available_columns_df = input_cols_df

    if dpg.is_dearpygui_running():
        dpg.set_value(TAG_S7_SOURCE_DF_INFO_TEXT, f"Source DataFrame: (Shape: {df_input.shape})")
        _refresh_sidebar()
        _update_execution_list_table()

def reset_state():
    global _current_df_input, _execution_list, _available_columns_df
    _current_df_input = None
    _execution_list = []
    _available_columns_df = None
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_S7_ARITH_FORMULA_TEXT): dpg.set_value(TAG_S7_ARITH_FORMULA_TEXT, "")
        if dpg.does_item_exist(TAG_S7_ADV_SYNTAX_INPUT): dpg.set_value(TAG_S7_ADV_SYNTAX_INPUT, "")
        _update_execution_list_table()
        _refresh_sidebar()
    _log("State reset.")

def get_settings_for_saving() -> dict:
    return {"execution_list": _execution_list}

def apply_settings_and_process(df_input: pd.DataFrame, settings: dict, main_app_callbacks: Dict[str, Any]):
    global _execution_list
    _execution_list = settings.get("execution_list", [])
    update_ui(df_input, main_app_callbacks)
    _log("Settings applied. User needs to run the pipeline manually.")