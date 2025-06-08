# step_07_feature_engineering.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import traceback
import re
import json
import tkinter as tk
from tkinter import filedialog

try:
    import ollama_analyzer
except ImportError:
    ollama_analyzer = None

# --- Module State ---
_main_app_callbacks: Optional[Dict[str, Any]] = None
_util_funcs: Optional[Dict[str, Any]] = None
_current_df_input: Optional[pd.DataFrame] = None
_execution_list: List[Dict[str, Any]] = []
_available_columns_list: List[str] = []
_derived_columns: List[str] = []
_if_conditions_list: List[Dict[str, Any]] = []
_categorical_mapping_state: Dict[str, str] = {}
_categorical_input_tags: Dict[str, int] = {}

# --- UI Tags ---
TAG_S7_MAIN_GROUP = "step7_main_group"
TAG_S7_SIDEBAR_LISTBOX = "step7_sidebar_listbox"
TAG_S7_EXECUTION_TABLE = "step7_execution_list_table"
TAG_S7_SOURCE_DF_INFO_TEXT = "step7_source_df_info_text"
TAG_S7_MAIN_TAB_BAR = "step7_main_tab_bar"
_MODAL_ID_PREVIEW = "step7_preview_modal"
_EXCEL_IMPORT_DIALOG_ID = "step7_excel_import_file_dialog"
_EXEC_PREVIEW_MODAL_ID = "step7_exec_preview_modal" # NEW

TAG_S7_ARITH_TAB = "step7_arith_tab"
TAG_S7_ARITH_FORMULA_TEXT = "step7_arith_formula_text"
TAG_S7_ARITH_OPERATOR_COMBO = "step7_arith_operator_combo"
TAG_S7_BINNING_TAB = "step7_binning_tab"
TAG_S7_BINNING_GROUP = "step7_binning_group"
TAG_S7_BINNING_SELECTED_VAR_TEXT = "step7_binning_selected_var_text"
TAG_S7_BINNING_STATS_TEXT = "step7_binning_stats_text"
TAG_S7_BINNING_PLOT = "step7_binning_plot"
TAG_S7_BINNING_PLOT_XAXIS = "step7_binning_plot_xaxis"
TAG_S7_BINNING_PLOT_YAXIS = "step7_binning_plot_yaxis"
TAG_S7_BINNING_NEW_COL_INPUT = "step7_binning_new_col_input"
TAG_S7_BINNING_METHOD_RADIO = "step7_binning_method_radio"
TAG_S7_BINNING_BINS_INPUT = "step7_binning_bins_input"
TAG_S7_BINNING_LABELS_INPUT = "step7_binning_labels_input"
TAG_S7_IF_TAB = "step7_if_tab"
TAG_S7_IF_GROUP = "step7_if_group"
TAG_S7_IF_SELECTED_VAR_TEXT = "step7_if_selected_var_text"
TAG_S7_IF_STATS_TEXT = "step7_if_stats_text"
TAG_S7_IF_PLOT = "step7_if_plot"
TAG_S7_IF_PLOT_XAXIS = "step7_if_plot_xaxis"
TAG_S7_IF_PLOT_YAXIS = "step7_if_plot_yaxis"
TAG_S7_IF_NEW_COL_INPUT = "step7_if_new_col_input"
TAG_S7_IF_BUILDER_TABLE = "step7_if_builder_table"
TAG_S7_IF_VAR_COMBO = "step7_if_var_combo"
TAG_S7_IF_OP_COMBO = "step7_if_op_combo"
TAG_S7_IF_VAL_INPUT = "step7_if_val_input"
TAG_S7_IF_RES_INPUT = "step7_if_res_input"
TAG_S7_IF_DEFAULT_INPUT = "step7_if_default_input"
TAG_S7_DATETIME_TAB = "step7_datetime_tab"
TAG_S7_DATETIME_GROUP = "step7_datetime_group"
TAG_S7_DATETIME_SELECTED_VAR_TEXT = "step7_datetime_selected_var_text"
TAG_S7_CATGROUP_TAB = "step7_catgroup_tab"
TAG_S7_CATGROUP_GROUP = "step7_catgroup_group"
TAG_S7_CATGROUP_SELECTED_VAR_TEXT = "step7_catgroup_selected_var_text"
TAG_S7_CATGROUP_NEW_COL_INPUT = "step7_catgroup_new_col_input"
TAG_S7_CATGROUP_MAPPING_TABLE = "step7_catgroup_mapping_table"
TAG_S7_ADV_SYNTAX_TAB = "step7_adv_syntax_tab"
TAG_S7_ADV_SYNTAX_INPUT = "step7_adv_syntax_input"
TAG_S7_LISTBOX_HANDLER_REG = "s7_listbox_handler_registry" # NEW


def _log(message: str): print(f"[Step7 FE] {message}")

# --- NEW: Export Categorical Mapping ---
def _export_categorical_mapping():
    var = dpg.get_value(TAG_S7_CATGROUP_SELECTED_VAR_TEXT)
    if not var:
        _util_funcs['_show_simple_modal_message']("Error", "No variable selected to export.")
        return
    if not _categorical_mapping_state:
        _util_funcs['_show_simple_modal_message']("Info", "There is no mapping data to export.")
        return

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
        title="Save Categorical Mapping",
        initialfile=f"{var}_mapping.xlsx"
    )

    if not file_path:
        return # User cancelled

    try:
        mapping_list = list(_categorical_mapping_state.items())
        df_to_export = pd.DataFrame(mapping_list, columns=['original_value', 'category'])
        df_to_export['var_name'] = var
        df_to_export = df_to_export[['var_name', 'original_value', 'category']]
        
        df_to_export.to_excel(file_path, index=False)
        _util_funcs['_show_simple_modal_message']("Success", f"Mapping exported successfully to:\n{file_path}")
    except Exception as e:
        _util_funcs['_show_simple_modal_message']("Export Error", f"Failed to export mapping:\n{e}", width=450, height=200)

# --- NEW: Execution List Preview ---
def _show_execution_preview_popup(sender, app_data, user_data_index):
    if _current_df_input is None: return

    try:
        temp_df = _current_df_input.copy()
        exec_scope = {'df': temp_df, 'pd': pd, 'np': np}
        
        for i in range(user_data_index + 1):
            exec(_execution_list[i]['syntax'], exec_scope)
        
        final_df = exec_scope['df']
        output_col = _execution_list[user_data_index]['output_col']

        if output_col not in final_df.columns:
            raise ValueError(f"Column '{output_col}' not found after executing step.")

        preview_data = final_df[[output_col]].head(10)
        
        if dpg.does_item_exist(_EXEC_PREVIEW_MODAL_ID):
            dpg.delete_item(_EXEC_PREVIEW_MODAL_ID)

        with dpg.window(label=f"Preview for '{output_col}'", modal=True, show=True, id=_EXEC_PREVIEW_MODAL_ID, no_close=True, autosize=True):
            dpg.add_text(f"Showing first 10 rows of '{output_col}':")
            with dpg.table(header_row=True, borders_innerV=True, borders_outerH=True, resizable=True, policy=dpg.mvTable_SizingStretchProp):
                dpg.add_table_column(label="Index")
                dpg.add_table_column(label=output_col)
                for index, row in preview_data.iterrows():
                    with dpg.table_row():
                        dpg.add_text(str(index))
                        dpg.add_text(str(row[output_col]))
            dpg.add_button(label="Close", width=-1, callback=lambda: dpg.configure_item(_EXEC_PREVIEW_MODAL_ID, show=False))

    except Exception as e:
        _util_funcs['_show_simple_modal_message']("Preview Error", f"Could not generate preview for step {user_data_index + 1}.\n\nError: {e}", width=500, height=250)


# --- MODIFIED: Execution List Table Update ---
def _update_execution_list_table():
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_S7_EXECUTION_TABLE): return
    dpg.delete_item(TAG_S7_EXECUTION_TABLE, children_only=True)
    
    dpg.add_table_column(label="Output Variable", parent=TAG_S7_EXECUTION_TABLE, width_stretch=True, init_width_or_weight=0.20)
    dpg.add_table_column(label="Method", parent=TAG_S7_EXECUTION_TABLE, width_stretch=True, init_width_or_weight=0.15)
    dpg.add_table_column(label="Definition", parent=TAG_S7_EXECUTION_TABLE, width_stretch=True, init_width_or_weight=0.50)
    dpg.add_table_column(label="Preview", parent=TAG_S7_EXECUTION_TABLE, width_fixed=True, init_width_or_weight=70)
    dpg.add_table_column(label="Delete", parent=TAG_S7_EXECUTION_TABLE, width_fixed=True, init_width_or_weight=65)

    for i, op in enumerate(_execution_list):
        with dpg.table_row(parent=TAG_S7_EXECUTION_TABLE):
            dpg.add_text(op.get("output_col", "N/A"))
            dpg.add_text(op.get("method", "N/A"))
            dpg.add_text(op.get("syntax", "N/A"))
            dpg.add_button(label="Preview", user_data=i, callback=_show_execution_preview_popup)
            dpg.add_button(label="Delete", user_data=i, callback=_delete_operation)

# --- MODIFIED: Create UI ---
def create_ui(step_name: str, parent_tag: str, main_app_callbacks: Dict[str, Any]):
    global _main_app_callbacks, _util_funcs
    _main_app_callbacks, _util_funcs = main_app_callbacks, main_app_callbacks.get('get_util_funcs', lambda: {})()
    main_app_callbacks['register_step_group_tag'](step_name, TAG_S7_MAIN_GROUP)
    main_app_callbacks['register_module_updater'](step_name, update_ui)

    # *** 수정 사항: 더블 클릭 이벤트를 처리할 핸들러 레지스트리를 먼저 생성합니다. ***
    with dpg.item_handler_registry(tag=TAG_S7_LISTBOX_HANDLER_REG):
        dpg.add_item_double_clicked_handler(callback=_on_sidebar_variable_double_clicked)
    
    # Excel 가져오기를 위한 파일 다이얼로그
    with dpg.file_dialog(
        directory_selector=False, show=False, callback=_process_excel_file, 
        id=_EXCEL_IMPORT_DIALOG_ID, width=700, height=400, modal=True
    ):
        dpg.add_file_extension(".xlsx")

    with dpg.group(tag=TAG_S7_MAIN_GROUP, parent=parent_tag, show=False):
        dpg.add_text(f"--- {step_name} ---", color=[255, 255, 0]); dpg.add_separator()
        with dpg.group(horizontal=True):
            with dpg.child_window(width=200):
                dpg.add_text("Variable Explorer")
                dpg.add_button(label="Refresh List", width=-1, callback=_refresh_sidebar)
                dpg.add_radio_button(
                    items=["All", "Numeric", "Non-Numeric"], 
                    horizontal=False, 
                    default_value="All",
                    tag="sidebar_filter_radio",
                    callback=lambda: _filter_sidebar_list()
                )
                
                # 콜백 없이 리스트박스를 생성합니다.
                dpg.add_listbox(
                    items=[], 
                    tag=TAG_S7_SIDEBAR_LISTBOX, 
                    num_items=25, 
                    width=-1
                )
                
                # *** 수정 사항: 위에서 만든 핸들러 레지스트리를 리스트박스에 연결(바인딩)합니다. ***
                dpg.bind_item_handler_registry(TAG_S7_SIDEBAR_LISTBOX, TAG_S7_LISTBOX_HANDLER_REG)

            with dpg.group(width=-1):
                dpg.add_text("Source DataFrame: N/A", tag=TAG_S7_SOURCE_DF_INFO_TEXT)
                dpg.add_button(label="▶️ Run Feature Engineering Pipeline", width=-1, height=30, callback=_run_pipeline)
                dpg.add_text("Execution List (Pipeline Steps):")
                dpg.add_table(tag=TAG_S7_EXECUTION_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=180, borders_innerV=True)
                dpg.add_separator()

                with dpg.tab_bar(tag=TAG_S7_MAIN_TAB_BAR):
                    # (이하 탭 부분은 이전 답변과 동일)
                    with dpg.tab(label="Arithmetic", tag=TAG_S7_ARITH_TAB):
                        dpg.add_text("Build formula by double-clicking variables."); 
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(tag=TAG_S7_ARITH_FORMULA_TEXT, hint="Formula appears here...", width=-150)
                            dpg.add_combo(items=[' + ', ' - ', ' * ', ' / '], tag=TAG_S7_ARITH_OPERATOR_COMBO, default_value=' + ', width=60)
                            dpg.add_button(label="Add to List", callback=_add_arithmetic_op)

                    with dpg.tab(label="Binning", tag=TAG_S7_BINNING_TAB):
                        with dpg.group(tag=TAG_S7_BINNING_GROUP, show=False):
                            with dpg.group(horizontal=True):
                                with dpg.group(width=200):
                                    dpg.add_text(tag=TAG_S7_BINNING_SELECTED_VAR_TEXT); dpg.add_spacer(height=5)
                                    dpg.add_text(tag=TAG_S7_BINNING_STATS_TEXT)
                                with dpg.plot(label="Distribution", height=150, width=-1, tag=TAG_S7_BINNING_PLOT):
                                    dpg.add_plot_axis(dpg.mvXAxis, tag=TAG_S7_BINNING_PLOT_XAXIS); dpg.add_plot_axis(dpg.mvYAxis, tag=TAG_S7_BINNING_PLOT_YAXIS)
                            dpg.add_input_text(label="New Variable Name", tag=TAG_S7_BINNING_NEW_COL_INPUT)
                            dpg.add_radio_button(label="Method", items=["Equal Frequency (qcut)", "Equal Width (cut)"], tag=TAG_S7_BINNING_METHOD_RADIO, horizontal=True, default_value="Equal Frequency (qcut)")
                            dpg.add_input_text(label="Bins / Quantiles", hint="e.g., 4 or [0,25,50,100]", tag=TAG_S7_BINNING_BINS_INPUT, callback=_auto_generate_bin_labels)
                            dpg.add_input_text(label="Labels (optional)", hint="Auto-generated if empty", tag=TAG_S7_BINNING_LABELS_INPUT)
                            dpg.add_button(label="Add Binning Op to List", width=-1, callback=_add_binning_op)

                    with dpg.tab(label="If Builder", tag=TAG_S7_IF_TAB):
                        with dpg.group(tag=TAG_S7_IF_GROUP, show=False):
                            with dpg.group(horizontal=True):
                                with dpg.group(width=200):
                                    dpg.add_text(tag=TAG_S7_IF_SELECTED_VAR_TEXT); dpg.add_spacer(height=5)
                                    dpg.add_text(tag=TAG_S7_IF_STATS_TEXT)
                                with dpg.plot(label="Distribution", height=120, width=-1, tag=TAG_S7_IF_PLOT):
                                    dpg.add_plot_axis(dpg.mvXAxis, tag=TAG_S7_IF_PLOT_XAXIS); dpg.add_plot_axis(dpg.mvYAxis, tag=TAG_S7_IF_PLOT_YAXIS)
                            dpg.add_input_text(label="New Variable Name", tag=TAG_S7_IF_NEW_COL_INPUT)
                            with dpg.table(tag=TAG_S7_IF_BUILDER_TABLE, header_row=True, borders_innerV=True): pass
                            with dpg.group(horizontal=True):
                                dpg.add_combo(items=[], tag=TAG_S7_IF_VAR_COMBO, width=120, label="If"); dpg.add_combo(items=['>', '<', '>=', '<=', '==', '!='], tag=TAG_S7_IF_OP_COMBO, width=60, default_value='>')
                                dpg.add_input_text(hint="Value", tag=TAG_S7_IF_VAL_INPUT, width=100); dpg.add_text("then"); dpg.add_input_text(hint="Result", tag=TAG_S7_IF_RES_INPUT, width=150)
                                dpg.add_button(label="Add", callback=_add_if_condition_row)
                            dpg.add_input_text(label="Else (Default)", tag=TAG_S7_IF_DEFAULT_INPUT); dpg.add_button(label="Add 'If' Op to List", width=-1, callback=_add_if_builder_op)
                    
                    with dpg.tab(label="Datetime", tag=TAG_S7_DATETIME_TAB):
                        with dpg.group(tag=TAG_S7_DATETIME_GROUP, show=False):
                            dpg.add_text(tag=TAG_S7_DATETIME_SELECTED_VAR_TEXT)
                            dpg.add_text("Select features to extract:")
                            with dpg.group(horizontal=True):
                                features = ['Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'IsWeekend']
                                for feat in features: dpg.add_checkbox(label=feat, tag=f"dt_check_{feat}")
                            dpg.add_button(label="Add Datetime Features to List", width=-1, callback=_add_datetime_features_op)
                    
                    with dpg.tab(label="Categorical", tag=TAG_S7_CATGROUP_TAB):
                        with dpg.group(tag=TAG_S7_CATGROUP_GROUP, show=False):
                            dpg.add_text(tag=TAG_S7_CATGROUP_SELECTED_VAR_TEXT); dpg.add_input_text(label="New Variable Name", tag=TAG_S7_CATGROUP_NEW_COL_INPUT)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="AI Auto-Categorization", callback=_run_ai_categorization, height=30)
                                dpg.add_button(label="Import from Excel", callback=lambda: dpg.show_item(_EXCEL_IMPORT_DIALOG_ID), height=30)
                                dpg.add_button(label="Export to Excel", callback=_export_categorical_mapping, height=30)
                            with dpg.table(tag=TAG_S7_CATGROUP_MAPPING_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=250): pass
                            dpg.add_button(label="Add Mapping Op to List", width=-1, callback=_add_categorical_grouping_op)

                    with dpg.tab(label="transform", tag="step7_transform_tab"):
                        with dpg.group(tag="step7_transform_group", show=False):
                            dpg.add_text(tag="step7_transform_selected_var_text")
                            dpg.add_separator()
                            dpg.add_text("적용할 변환을 선택하세요:")
                            
                            # 변환 버튼들을 테이블 형태로 깔끔하게 정렬
                            with dpg.table(header_row=False):
                                dpg.add_table_column()
                                dpg.add_table_column()
                                dpg.add_table_column()

                                with dpg.table_row():
                                    dpg.add_button(label="Log (x+1)", width=-1, callback=lambda: _add_transform_op('log1p'))
                                    dpg.add_button(label="Square Root (√x)", width=-1, callback=lambda: _add_transform_op('sqrt'))
                                    dpg.add_button(label="Square (x²)", width=-1, callback=lambda: _add_transform_op('square'))
                                
                                with dpg.table_row():
                                    dpg.add_button(label="Reciprocal (1/x)", width=-1, callback=lambda: _add_transform_op('reciprocal'))
                                    dpg.add_button(label="Absolute (abs)", width=-1, callback=lambda: _add_transform_op('abs'))
                                    dpg.add_button(label="Sign (-1, 0, 1)", width=-1, callback=lambda: _add_transform_op('sign'))

                                with dpg.table_row():
                                    dpg.add_button(label="Z-Score Scaling", width=-1, callback=lambda: _add_transform_op('zscore'))
                                    dpg.add_button(label="Min-Max Scaling", width=-1, callback=lambda: _add_transform_op('minmax'))
                                    dpg.add_button(label="Rank", width=-1, callback=lambda: _add_transform_op('rank'))

                    
                    with dpg.tab(label="Advanced Syntax", tag=TAG_S7_ADV_SYNTAX_TAB):
                        dpg.add_button(label="Import Pipeline to Code", width=-1, callback=_import_pipeline_to_code)
                        dpg.add_input_text(tag=TAG_S7_ADV_SYNTAX_INPUT, multiline=True, width=-1, height=390, hint="Visually built pipeline appears here for fine-tuning...")

def _add_transform_op(op_type: str):
    """선택된 수치형 변수에 수학적 변환을 적용하는 작업을 파이프라인에 추가합니다."""
    # 현재 선택된 변수 이름에서 'Selected: ' 부분을 제외하고 가져옵니다.
    selected_text = dpg.get_value("step7_transform_selected_var_text")
    if not selected_text.startswith("Selected: "):
        return
    var = selected_text.replace("Selected: ", "")

    new_col = f"{var}_{op_type}"
    syntax = ""
    epsilon = 1e-6 # 0으로 나누는 것을 방지하기 위한 아주 작은 값

    transform_map = {
        'log1p': f"df['{new_col}'] = np.log1p(df['{var}'])",
        'sqrt': f"df['{new_col}'] = np.sqrt(df.loc[df['{var}'] >= 0, '{var}']) # 음수는 NaN으로 처리",
        'square': f"df['{new_col}'] = df['{var}'] ** 2",
        'reciprocal': f"df['{new_col}'] = 1 / (df['{var}'] + {epsilon})",
        'abs': f"df['{new_col}'] = df['{var}'].abs()",
        'sign': f"df['{new_col}'] = np.sign(df['{var}'])",
        'zscore': f"df['{new_col}'] = (df['{var}'] - df['{var}'].mean()) / df['{var}'].std(ddof=0)",
        'minmax': f"df['{new_col}'] = (df['{var}'] - df['{var}'].min()) / (df['{var}'].max() - df['{var}'].min())",
        'rank': f"df['{new_col}'] = df['{var}'].rank()"
    }
    
    syntax = transform_map.get(op_type)

    if syntax:
        _add_to_execution_list(new_col, f"Transform ({op_type})", syntax, show_preview=True)

def _auto_generate_bin_labels(sender, app_data, user_data):
    """'Bins / Quantiles' 입력값과 선택된 메소드에 따라 실제 데이터 구간을 반영한 라벨을 자동으로 생성합니다."""
    # 1. 필요한 모든 UI의 현재 값을 가져옵니다.
    bin_count_str = app_data
    var = dpg.get_value(TAG_S7_BINNING_SELECTED_VAR_TEXT)
    method = dpg.get_value(TAG_S7_BINNING_METHOD_RADIO)

    if _current_df_input is None or not var:
        return

    try:
        bin_count = int(bin_count_str)
        if bin_count <= 0:
            dpg.set_value(TAG_S7_BINNING_LABELS_INPUT, "")
            return
    except (ValueError, TypeError):
        dpg.set_value(TAG_S7_BINNING_LABELS_INPUT, "")
        return

    labels = []
    
    # 2. 'Equal Frequency (qcut)' 로직
    if method == 'Equal Frequency (qcut)':
        quantiles = np.linspace(0, 1, bin_count + 1)
        for i in range(bin_count):
            start_p = int(quantiles[i] * 100)
            end_p = int(quantiles[i+1] * 100)
            # ★★★ 수정사항: 라벨에 따옴표를 직접 추가하지 않습니다.
            label = f'q{i+1}_{start_p}-{end_p}%'
            labels.append(label)

    # 3. 'Equal Width (cut)' 로직
    elif method == 'Equal Width (cut)':
        series = _current_df_input[var].dropna()
        if series.empty:
            return

        min_val = series.min()
        max_val = series.max()
        bins = np.linspace(min_val, max_val, bin_count + 1)
        
        for i in range(bin_count):
            start_v = bins[i]
            end_v = bins[i+1]
            # ★★★ 수정사항: 라벨에 따옴표를 직접 추가하지 않습니다.
            if i == bin_count - 1:
                label = f'c{i+1}_{int(start_v)}_{int(end_v)}'
            else:
                label = f'c{i+1}_{int(start_v)}_{int(end_v - 1)}'
            labels.append(label)

    # 4. 생성된 라벨이 있으면 UI에 적용합니다.
    if labels:
        # 따옴표 없이 쉼표로만 구분된 문자열을 생성합니다.
        labels_str = ", ".join(labels)
        dpg.set_value(TAG_S7_BINNING_LABELS_INPUT, labels_str)

def _run_ai_categorization():
    if not ollama_analyzer:
        _util_funcs['_show_simple_modal_message']("AI Error", "Ollama analyzer module not found.")
        return
    var = dpg.get_value(TAG_S7_CATGROUP_SELECTED_VAR_TEXT)
    if not var or _current_df_input is None:
        _util_funcs['_show_simple_modal_message']("Error", "Please select a categorical variable first.")
        return

    unique_values = _current_df_input[var].unique().tolist()
    unique_values_str = ", ".join([f'"{str(v)}"' for v in unique_values[:100]]) # Limit to 100 values for prompt
    
    prompt = f"""
You are a professional data analyst. The following is a list of unique values from a categorical column: [{unique_values_str}].
Your task is to group these values into a smaller number of meaningful categories.
Please provide your response as a single, valid JSON object.
The JSON object should have keys that are the original values from the list, and values that are the new category names you have assigned.
Example: {{ "value_a": "Group 1", "value_b": "Group 1", "value_c": "Group 2" }}
Do not provide any other text, explanation, or markdown formatting. Just the raw JSON object.
"""
    
    _main_app_callbacks['add_ai_log'](f"Asking AI to categorize '{var}'...", chart_context="AI Categorization")
    
    try:
        full_response = ""
        for chunk in ollama_analyzer.analyze_text_with_ollama(prompt, chart_name=f"Categorize_{var}"):
            full_response += chunk
        
        _main_app_callbacks['add_ai_log']("AI Response Received. Parsing and applying.", chart_context="AI Categorization", mode="stream_chunk_append")
        
        json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
        if not json_match:
            raise json.JSONDecodeError("No JSON object found in AI response.", full_response, 0)
        
        json_str = json_match.group(0)
        mapping = json.loads(json_str)

        for original_val, new_cat in mapping.items():
            if original_val in _categorical_input_tags:
                tag = _categorical_input_tags[original_val]
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, new_cat)
                    _categorical_mapping_state[original_val] = new_cat
        
        _main_app_callbacks['add_ai_log']("\nSuccessfully applied AI-generated categories.", chart_context="AI Categorization", mode="stream_chunk_append")

    except Exception as e:
        error_msg = f"Failed to get or apply AI categorization:\n{e}\n\nFull Response:\n{full_response[:500]}"
        _util_funcs['_show_simple_modal_message']("AI Error", error_msg, width=500, height=300)
        _main_app_callbacks['add_ai_log'](f"\nError during AI categorization: {e}", chart_context="AI Categorization", mode="stream_chunk_append")

def _process_excel_file(sender, app_data):
    file_path = app_data.get('file_path_name')
    if not file_path: return

    var = dpg.get_value(TAG_S7_CATGROUP_SELECTED_VAR_TEXT)
    if not var or _current_df_input is None:
        _util_funcs['_show_simple_modal_message']("Error", "Please select a target variable before importing.")
        return

    try:
        excel_df = pd.read_excel(file_path)
        required_cols = ['var_name', 'original_value', 'category']
        if not all(col in excel_df.columns for col in required_cols):
            _util_funcs['_show_simple_modal_message']("Excel Error", f"Excel file must contain columns: {', '.join(required_cols)}")
            return

        mapping_df = excel_df[excel_df['var_name'] == var]
        if mapping_df.empty:
            _util_funcs['_show_simple_modal_message']("Info", f"No mappings found for variable '{var}' in the Excel file.")
            return

        source_values = set(_current_df_input[var].astype(str).unique())
        excel_values = set(mapping_df['original_value'].astype(str))
        
        extra_in_excel = excel_values - source_values
        if extra_in_excel:
            _util_funcs['_show_simple_modal_message']("Excel Error", f"The following values from Excel do not exist in the source data for '{var}':\n{', '.join(list(extra_in_excel)[:10])}")
            return
            
        mapping_dict = pd.Series(mapping_df.category.values, index=mapping_df.original_value.astype(str)).to_dict()

        for original_val, new_cat in mapping_dict.items():
            if original_val in _categorical_input_tags:
                tag = _categorical_input_tags[original_val]
                if dpg.does_item_exist(tag):
                    dpg.set_value(tag, new_cat)
                    _categorical_mapping_state[original_val] = new_cat
        _util_funcs['_show_simple_modal_message']("Success", "Successfully applied mapping from Excel file.")

    except Exception as e:
        _util_funcs['_show_simple_modal_message']("Excel Error", f"Failed to process Excel file:\n{e}")

def _add_categorical_grouping_op():
    var = dpg.get_value(TAG_S7_CATGROUP_SELECTED_VAR_TEXT)
    if not var or _current_df_input is None: return

    new_col = dpg.get_value(TAG_S7_CATGROUP_NEW_COL_INPUT)
    mapping_dict = {k: v for k, v in _categorical_mapping_state.items() if v}
    
    if not mapping_dict:
        _util_funcs['_show_simple_modal_message']("Info", "No new categories have been defined.")
        return

    displayed_values = set(_categorical_mapping_state.keys())
    mapped_values = set(mapping_dict.keys())
    unmapped_values = displayed_values - mapped_values

    def proceed_with_nan():
        syntax = f"mapping_dict = {mapping_dict}\ndf['{new_col}'] = df['{var}'].astype(str).map(mapping_dict)"
        _add_to_execution_list(new_col, "Cat. Grouping (NaN)", syntax, show_preview=True)

    def proceed_with_fill():
        syntax = f"mapping_dict = {mapping_dict}\ndf['{new_col}'] = df['{var}'].astype(str).map(mapping_dict).fillna(df['{var}'])"
        _add_to_execution_list(new_col, "Cat. Grouping", syntax, show_preview=True)

    if unmapped_values:
        _util_funcs['show_confirmation_modal'](
            title="미지정 카테고리",
            message="지정되지 않은 카테고리 값이 있습니다. 전부 NaN 으로 채울까요?",
            yes_callback=proceed_with_nan,
        )
    else:
        proceed_with_fill()

def _on_sidebar_variable_double_clicked(sender, app_data, user_data):
    selected_item_str = dpg.get_value(TAG_S7_SIDEBAR_LISTBOX)
    
    # get_value()는 숫자 ID를 반환합니다.
    active_tab_id = dpg.get_value(TAG_S7_MAIN_TAB_BAR)

    if not selected_item_str:
        return
    
    try:
        var_name, var_type_str = selected_item_str.rsplit(' (', 1)
        var_type = var_type_str[:-1]
    except ValueError:
        return
        
    is_derived = var_name in _derived_columns
    
    # ★★★ 최종 수정사항 ★★★
    # 숫자 ID를 문자열 별명으로 변환하여, 문자열끼리 비교합니다.
    if dpg.get_item_alias(active_tab_id) == TAG_S7_ARITH_TAB:
        current_formula = dpg.get_value(TAG_S7_ARITH_FORMULA_TEXT)
        operator = dpg.get_value(TAG_S7_ARITH_OPERATOR_COMBO)
        
        if not current_formula or current_formula.strip().endswith(('+', '-', '*', '/')):
            new_formula = f"{current_formula} df['{var_name}']"
        else:
            new_formula = f"{current_formula}{operator}df['{var_name}']"
        
        dpg.set_value(TAG_S7_ARITH_FORMULA_TEXT, new_formula.strip())
        return

    # --- 이하 다른 탭 로직 (기존과 동일) ---
    # if is_derived:
    #     _util_funcs['_show_simple_modal_message']("Info", f"'{var_name}' is a derived variable.\nIt can be used in Arithmetic formulas, but not as an input for other operations.")
    #     return
    
    numeric_types = ['int64', 'float64', 'int32', 'float32']
    datetime_types = ['datetime', 'timestamp']
    is_numeric = any(ntype in var_type for ntype in numeric_types)
    is_datetime = any(dtype in var_type for dtype in datetime_types)

    dpg.configure_item(TAG_S7_BINNING_GROUP, show=is_numeric)
    dpg.configure_item(TAG_S7_IF_GROUP, show=is_numeric)
    dpg.configure_item(TAG_S7_DATETIME_GROUP, show=is_datetime)
    dpg.configure_item(TAG_S7_CATGROUP_GROUP, show=not (is_numeric or is_datetime))
    dpg.configure_item("step7_transform_group", show=is_numeric)


    if is_numeric:
        dpg.set_value(TAG_S7_BINNING_SELECTED_VAR_TEXT, var_name)
        dpg.set_value(TAG_S7_BINNING_NEW_COL_INPUT, f"{var_name}_binned")
        _update_stats_and_plot('step7_binning', var_name, _current_df_input)
        dpg.set_value(TAG_S7_IF_SELECTED_VAR_TEXT, var_name)
        dpg.set_value(TAG_S7_IF_NEW_COL_INPUT, f"{var_name}_if")
        _update_stats_and_plot('step7_if', var_name, _current_df_input)
        dpg.set_value("step7_transform_selected_var_text", f"Selected: {var_name}")
        all_numeric_cols = [c.split(' (')[0] for c in _available_columns_list if any(ntype in c for ntype in numeric_types)]
        dpg.configure_item(TAG_S7_IF_VAR_COMBO, items=all_numeric_cols, default_value=var_name)
    elif is_datetime:
        dpg.set_value(TAG_S7_DATETIME_SELECTED_VAR_TEXT, var_name)
    else: # Categorical
        dpg.set_value(TAG_S7_CATGROUP_SELECTED_VAR_TEXT, var_name)
        dpg.set_value(TAG_S7_CATGROUP_NEW_COL_INPUT, f"{var_name}_cat1")
        _update_categorical_mapping_table()


def _update_categorical_mapping_table():
    global _categorical_mapping_state, _categorical_input_tags
    var = dpg.get_value(TAG_S7_CATGROUP_SELECTED_VAR_TEXT)
    if not var or _current_df_input is None: return
    
    dpg.delete_item(TAG_S7_CATGROUP_MAPPING_TABLE, children_only=True)
    
    value_counts = _current_df_input[var].value_counts()
    total_unique = len(value_counts)
    
    value_counts_top50 = value_counts.head(50).reset_index()
    value_counts_top50.columns = ['value', 'count']
    
    if dpg.does_item_exist("cat_info_text"): dpg.delete_item("cat_info_text")
    dpg.add_text(f"Showing Top 50 of {total_unique} unique values.", parent=TAG_S7_CATGROUP_GROUP, tag="cat_info_text", before=TAG_S7_CATGROUP_MAPPING_TABLE)

    dpg.add_table_column(label="Original Value", parent=TAG_S7_CATGROUP_MAPPING_TABLE)
    dpg.add_table_column(label="Count", parent=TAG_S7_CATGROUP_MAPPING_TABLE, width_fixed=True, init_width_or_weight=80)
    dpg.add_table_column(label="New Category (Editable)", parent=TAG_S7_CATGROUP_MAPPING_TABLE)
    
    _categorical_mapping_state = {str(row['value']): "" for _, row in value_counts_top50.iterrows()}
    _categorical_input_tags = {}
    for _, row in value_counts_top50.iterrows():
        original_value_str = str(row['value'])
        with dpg.table_row(parent=TAG_S7_CATGROUP_MAPPING_TABLE):
            dpg.add_text(original_value_str); dpg.add_text(str(row['count']))
            input_tag = dpg.add_input_text(default_value="", user_data={'raw_value': original_value_str}, callback=lambda s, a, u: _categorical_mapping_state.update({u['raw_value']: a}), width=-1)
            _categorical_input_tags[original_value_str] = input_tag

def _generate_unique_col_name(base_name: str) -> str:
    all_names = [item.split(' (')[0] for item in _available_columns_list]
    clean_base_name = re.sub(r'[^A-Za-z0-9_]+', '', base_name)
    if not clean_base_name: clean_base_name = "new_var"
    if clean_base_name not in all_names: return clean_base_name
    i = 1
    while f"{clean_base_name}_{i}" in all_names: i += 1
    return f"{clean_base_name}_{i}"

def _show_preview_modal(df_preview: pd.DataFrame, input_vars: List[str], output_var: str):
    if dpg.does_item_exist(_MODAL_ID_PREVIEW): dpg.delete_item(_MODAL_ID_PREVIEW)
    cols_to_show = list(set(input_vars + [output_var]))
    preview_data = df_preview[cols_to_show]
    with dpg.window(label="Operation Preview", modal=True, show=True, id=_MODAL_ID_PREVIEW, no_close=True, autosize=True):
        dpg.add_text("The first 5 rows of the result will look like this:")
        with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, height=250, scrollY=True, borders_innerV=True, borders_outerH=True, borders_innerH=True, borders_outerV=True):
            for col in preview_data.columns: dpg.add_table_column(label=col)
            for _, row in preview_data.head().iterrows():
                with dpg.table_row():
                    for item in row: dpg.add_text(str(item))
        dpg.add_button(label="Close", width=-1, callback=lambda: dpg.configure_item(_MODAL_ID_PREVIEW, show=False))

def _add_to_execution_list(output_col: str, method: str, syntax: str, show_preview: bool = False):
    global _available_columns_list, _derived_columns
    unique_output_col = _generate_unique_col_name(output_col)
    if unique_output_col != output_col:
        syntax = syntax.replace(f"df['{output_col}']", f"df['{unique_output_col}']", 1)
    _execution_list.append({"output_col": unique_output_col, "method": method, "syntax": syntax})
    _refresh_sidebar(); _update_execution_list_table()
    if show_preview and _current_df_input is not None:
        temp_df = _current_df_input.copy()
        exec(syntax, {'df': temp_df, 'pd': pd, 'np': np})
        input_vars = re.findall(r"df\[['\"](.*?)['\"]\]", syntax.split('=', 1)[1])
        _show_preview_modal(temp_df, input_vars, unique_output_col)

def _delete_operation(sender, app_data, user_data):
    _execution_list.pop(user_data); _refresh_sidebar(); _update_execution_list_table()
def _filter_sidebar_list():
    if not dpg.is_dearpygui_running(): return
    filter_type = dpg.get_value("sidebar_filter_radio")
    numeric_types = ['int64', 'float64', 'int32', 'float32']
    if filter_type == "All": filtered_list = _available_columns_list
    elif filter_type == "Numeric": filtered_list = [item for item in _available_columns_list if any(ntype in item for ntype in numeric_types)]
    else: filtered_list = [item for item in _available_columns_list if not any(ntype in item for ntype in numeric_types)]
    dpg.configure_item(TAG_S7_SIDEBAR_LISTBOX, items=filtered_list)
def _refresh_sidebar():
    global _available_columns_list, _derived_columns
    if _current_df_input is None: _available_columns_list, _derived_columns = [], []
    else:
        input_cols = [f"{col} ({dtype})" for col, dtype in _current_df_input.dtypes.items()]
        exec_scope = {'df': _current_df_input.copy(), 'pd': pd, 'np': np}
        derived_cols_info, new_derived_columns = [], []
        for op in _execution_list:
            try:
                exec(op['syntax'], exec_scope)
                col_name, dtype = op['output_col'], exec_scope['df'][op['output_col']].dtype
                derived_cols_info.append(f"{col_name} ({dtype})"); new_derived_columns.append(col_name)
            except Exception: derived_cols_info.append(f"{op['output_col']} (error)"); new_derived_columns.append(op['output_col'])
        _available_columns_list = input_cols + derived_cols_info; _derived_columns = new_derived_columns
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_S7_SIDEBAR_LISTBOX):
        dpg.configure_item(TAG_S7_SIDEBAR_LISTBOX, items=_available_columns_list)

def _update_stats_and_plot(group_tag, selected_var, df):
    stats_text_tag, plot_tag = f"{group_tag}_stats_text", f"{group_tag}_plot"
    plot_xaxis_tag, plot_yaxis_tag = f"{plot_tag}_xaxis", f"{plot_tag}_yaxis"
    stats = df[selected_var].describe()
    stats_str = f"Count: {stats['count']:.0f}, Mean: {stats['mean']:.2f}\nStd Dev: {stats['std']:.2f}, Min: {stats['min']:.2f}\n25%: {stats['25%']:.2f}, 50%: {stats['50%']:.2f}\n75%: {stats['75%']:.2f}, Max: {stats['max']:.2f}"
    dpg.set_value(stats_text_tag, stats_str)
    if dpg.does_item_exist(plot_yaxis_tag):
        dpg.delete_item(plot_yaxis_tag, children_only=True)
        dpg.add_histogram_series(df[selected_var].dropna().tolist(), bins=-1, parent=plot_yaxis_tag)
        dpg.fit_axis_data(plot_xaxis_tag); dpg.fit_axis_data(plot_yaxis_tag)
def _add_arithmetic_op():
    formula = dpg.get_value(TAG_S7_ARITH_FORMULA_TEXT)
    if not formula.strip(): return
    _add_to_execution_list("arith_var", "Arithmetic", f"df['arith_var'] = {formula}", show_preview=True); dpg.set_value(TAG_S7_ARITH_FORMULA_TEXT, "")
def _add_binning_op():
    var, new_col, method, bins_input, user_labels = (
        dpg.get_value(TAG_S7_BINNING_SELECTED_VAR_TEXT),
        dpg.get_value(TAG_S7_BINNING_NEW_COL_INPUT),
        dpg.get_value(TAG_S7_BINNING_METHOD_RADIO),
        dpg.get_value(TAG_S7_BINNING_BINS_INPUT),
        dpg.get_value(TAG_S7_BINNING_LABELS_INPUT),
    )
    if not var:
        return

    try:
        if not bins_input.strip():
            raise ValueError("Bins/Quantiles input cannot be empty.")
        
        labels_str = ""
        # 라벨이 입력되었을 경우, 자동으로 따옴표를 붙여주는 처리
        if user_labels.strip():
            # 사용자가 'a, b, c' 형식으로 입력해도 ['"a"', '"b"', '"c"'] 형태로 변환
            labels_list = [f'"{label.strip()}"' for label in user_labels.split(',')]
            labels_str = f"labels=[{', '.join(labels_list)}]"

        # Equal Frequency (qcut) 또는 Equal Width (cut)에 대한 구문 생성
        if method == 'Equal Frequency (qcut)':
            syntax = f"df['{new_col}'] = pd.qcut(df['{var}'], q={bins_input}, {labels_str}, duplicates='drop')"
        else:
            syntax = f"df['{new_col}'] = pd.cut(df['{var}'], bins={bins_input}, {labels_str}, right=False)"
        
        _add_to_execution_list(new_col, f"Binning ({method.split(' ')[0]})", syntax, show_preview=True)

    except Exception as e:
        _util_funcs['_show_simple_modal_message']("Error", f"Failed to create binning operation:\n{e}")

def _add_if_condition_row():
    var, op, val, res = dpg.get_value(TAG_S7_IF_VAR_COMBO), dpg.get_value(TAG_S7_IF_OP_COMBO), dpg.get_value(TAG_S7_IF_VAL_INPUT), dpg.get_value(TAG_S7_IF_RES_INPUT)
    if not all([var, op, str(val), res]): _util_funcs['_show_simple_modal_message']("Error", "All fields for a condition are required."); return
    new_condition = {'var': var, 'op': op, 'val': val, 'res': res}
    if new_condition in _if_conditions_list: _util_funcs['_show_simple_modal_message']("Error", "This exact condition already exists."); return
    _if_conditions_list.append(new_condition); _update_if_builder_table()
def _delete_if_condition_row(sender, app_data, user_data): _if_conditions_list.pop(user_data); _update_if_builder_table()
def _update_if_builder_table():
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_S7_IF_BUILDER_TABLE): return
    dpg.delete_item(TAG_S7_IF_BUILDER_TABLE, children_only=True)
    headers = ["Condition", "Result", "Action"]
    for h in headers: dpg.add_table_column(label=h, parent=TAG_S7_IF_BUILDER_TABLE)
    for i, item in enumerate(_if_conditions_list):
        with dpg.table_row(parent=TAG_S7_IF_BUILDER_TABLE):
            dpg.add_text(f"df['{item['var']}'] {item['op']} {item['val']}"); dpg.add_text(f"'{item['res']}'"); dpg.add_button(label="Delete", user_data=i, callback=_delete_if_condition_row)
def _add_if_builder_op():
    var, new_col, default_val = dpg.get_value(TAG_S7_IF_SELECTED_VAR_TEXT), dpg.get_value(TAG_S7_IF_NEW_COL_INPUT), dpg.get_value(TAG_S7_IF_DEFAULT_INPUT)
    if not var or not _if_conditions_list: return
    try:
        conditions, results, default_val_str = [f"(df['{item['var']}'] {item['op']} {item['val']})" for item in _if_conditions_list], [f"'{item['res']}'" for item in _if_conditions_list], f"'{default_val}'"
        syntax = f"conditions = [{', '.join(conditions)}]\nresults = [{', '.join(results)}]\ndf['{new_col}'] = np.select(conditions, results, default={default_val_str})"
        _add_to_execution_list(new_col, "If Builder", syntax, show_preview=True)
        _if_conditions_list.clear(); _update_if_builder_table()
    except Exception as e: _util_funcs['_show_simple_modal_message']("Error", f"Failed to create 'if' operation:\n{e}")
def _add_datetime_features_op():
    var = dpg.get_value(TAG_S7_DATETIME_SELECTED_VAR_TEXT)
    if not var: return
    features = {'Year':'.dt.year', 'Month':'.dt.month', 'Day':'.dt.day', 'Hour':'.dt.hour', 'DayOfWeek':'.dt.dayofweek', 'IsWeekend':'.dt.dayofweek >= 5'}
    for feature, syntax_part in features.items():
        if dpg.get_value(f"dt_check_{feature}"):
            new_col_name = f"{var}_{feature.lower()}"
            syntax = f"df['{new_col_name}'] = pd.to_datetime(df['{var}']).dt.dayofweek >= 5" if feature == 'IsWeekend' else f"df['{new_col_name}'] = pd.to_datetime(df['{var}']).dt.{feature.lower()}"
            _add_to_execution_list(new_col_name, "Datetime Feature", syntax)
    _util_funcs['_show_simple_modal_message']("Success", "Datetime features added to the execution list.")
def _import_pipeline_to_code():
    if not _execution_list: _util_funcs['_show_simple_modal_message']("Info", "Execution list is empty."); return
    full_code = "# Generated from visual pipeline\nimport pandas as pd\nimport numpy as np\n\n"
    for op in _execution_list: full_code += f"# Method: {op['method']}\n{op['syntax']}\n\n"
    dpg.set_value(TAG_S7_ADV_SYNTAX_INPUT, full_code)
def _run_pipeline():
    if _current_df_input is None: return
    df, exec_scope = _current_df_input.copy(), {'df': _current_df_input.copy(), 'pd': pd, 'np': np}
    try:
        active_tab = dpg.get_value(TAG_S7_MAIN_TAB_BAR)
        if active_tab == dpg.get_item_alias(TAG_S7_ADV_SYNTAX_TAB):
            syntax_from_editor = dpg.get_value(TAG_S7_ADV_SYNTAX_INPUT)
            if not syntax_from_editor.strip(): _util_funcs['_show_simple_modal_message']("Info", "Syntax editor is empty."); return
            exec(syntax_from_editor, exec_scope)
        else:
            if not _execution_list: _util_funcs['_show_simple_modal_message']("Info", "No operations in execution list."); return
            for op in _execution_list: exec(op.get('syntax', ''), exec_scope)
        _main_app_callbacks['step7_feature_engineering_complete'](exec_scope['df'])
        _util_funcs['_show_simple_modal_message']("Success", "Pipeline execution completed successfully.")
    except Exception as e: _util_funcs['_show_simple_modal_message']("Pipeline Error", f"Pipeline execution failed:\n\n{traceback.format_exc()}", width=600, height=400)
def update_ui(df_input: Optional[pd.DataFrame], main_app_callbacks: Dict[str, Any]):
    global _current_df_input, _main_app_callbacks
    _main_app_callbacks, _current_df_input = main_app_callbacks, df_input
    if df_input is None: reset_state(); return
    if dpg.is_dearpygui_running():
        dpg.set_value(TAG_S7_SOURCE_DF_INFO_TEXT, f"Source DataFrame: (Shape: {df_input.shape})")
        _refresh_sidebar(); _update_execution_list_table()
        _if_conditions_list.clear(); _update_if_builder_table()
        dpg.configure_item(TAG_S7_BINNING_GROUP, show=False); dpg.configure_item(TAG_S7_IF_GROUP, show=False); dpg.configure_item(TAG_S7_DATETIME_GROUP, show=False); dpg.configure_item(TAG_S7_CATGROUP_GROUP, show=False)
def reset_state():
    global _execution_list, _if_conditions_list, _available_columns_list, _derived_columns, _categorical_mapping_state, _categorical_input_tags
    _execution_list, _if_conditions_list, _available_columns_list, _derived_columns = [], [], [], []
    _categorical_mapping_state, _categorical_input_tags = {}, {}
    if dpg.is_dearpygui_running(): _refresh_sidebar(); _update_execution_list_table(); _update_if_builder_table()
def get_settings_for_saving() -> dict: return {"execution_list": _execution_list}

def apply_settings_and_process(df_input: pd.DataFrame, settings: dict, main_app_callbacks: Dict[str, Any]):
    global _execution_list, _main_app_callbacks

    # main_app_callbacks가 설정되지 않았다면 설정
    if not _main_app_callbacks: _main_app_callbacks = main_app_callbacks

    _execution_list = settings.get("execution_list", [])
    update_ui(df_input, main_app_callbacks)

    # --- 추가된 부분: 파이프라인 강제 실행 ---
    print("Force-running Step 7 processing after applying settings...")
    if _execution_list: # 실행할 파이프라인이 있을 경우에만 실행
        _run_pipeline()
    else: # 실행할 파이프라인이 없으면, 입력받은 df를 그대로 완료 콜백으로 전달
        _main_app_callbacks['step7_feature_engineering_complete'](df_input)