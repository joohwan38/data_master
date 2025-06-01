# step_04_missing_values.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer 

# --- Unique DPG Tags for this module ---
TAG_MV_STEP_GROUP = "step4_missing_values_group"
TAG_MV_PREDEFINED_NAN_GROUP = "step4_mv_predefined_nan_group"
TAG_MV_CONVERT_PREDEFINED_NANS_BUTTON = "step4_mv_convert_predefined_nans_button"
TAG_MV_METHOD_SELECTION_TABLE = "step4_mv_method_selection_table"
TAG_MV_EXECUTE_IMPUTATION_BUTTON = "step4_mv_execute_imputation_button"
TAG_MV_RESULTS_GROUP = "step4_mv_results_group" 
TAG_MV_LOG_TEXT_AREA = "step4_mv_log_text_area"
TAG_MV_LOG_TEXT = "step4_mv_log_text"

# --- Module State Variables ---
_main_app_callbacks: Optional[Dict[str, Any]] = None
_util_funcs: Optional[Dict[str, Any]] = None
_current_df_for_this_step: Optional[pd.DataFrame] = None 
_df_after_nan_conversion: Optional[pd.DataFrame] = None 
_df_after_imputation: Optional[pd.DataFrame] = None 

_imputation_method_selections: Dict[str, Tuple[str, Optional[str]]] = {} 

PREDEFINED_NAN_STRINGS = ["N/A", "NA", "null", "Null", "NULL", "-", "?", "Missing", "missing", "not available", "", " "] 
_selected_predefined_nans: Dict[str, bool] = {nan_str: False for nan_str in PREDEFINED_NAN_STRINGS}


def _on_predefined_nan_checkbox_change(sender, app_data: bool, user_data_nan_str: str):
    _selected_predefined_nans[user_data_nan_str] = app_data
    print(f"Step 4 MV: Predefined NaN string '{user_data_nan_str}' selection changed to {app_data}")

def _convert_selected_to_nans_logic():
    global _current_df_for_this_step, _df_after_nan_conversion, _util_funcs, _main_app_callbacks, _df_after_imputation
    
    df_input_for_conversion = _df_after_imputation if _df_after_imputation is not None else \
                              (_df_after_nan_conversion if _df_after_nan_conversion is not None else _current_df_for_this_step)

    if df_input_for_conversion is None:
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Error", "No data loaded to convert NaNs.")
        return

    df_to_convert = df_input_for_conversion.copy() 
    strings_to_replace = [nan_str for nan_str, selected in _selected_predefined_nans.items() if selected]
    log_messages = [f"--- Converting Predefined Strings to NaN (Step 4) ---"]

    if not strings_to_replace:
        log_messages.append("No predefined NaN strings selected for conversion.")
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Info", "No predefined NaN strings were selected for conversion.")
        _df_after_nan_conversion = df_to_convert 
        _df_after_imputation = None 
    else:
        log_messages.append(f"Selected strings for NaN conversion: {', '.join(repr(s) for s in strings_to_replace)}")
        converted_any = False
        for col in df_to_convert.columns:
            original_missing = df_to_convert[col].isnull().sum()
            df_to_convert[col] = df_to_convert[col].replace(strings_to_replace, np.nan)
            new_missing = df_to_convert[col].isnull().sum()
            if new_missing > original_missing:
                log_messages.append(f"  Column '{col}': Converted {new_missing - original_missing} additional values to NaN.")
                converted_any = True
        
        if converted_any:
            log_messages.append("Conversion complete. DataFrame updated for this step.")
            if _util_funcs and '_show_simple_modal_message' in _util_funcs:
                _util_funcs['_show_simple_modal_message']("Success", "Selected strings converted to NaN for further processing in this step.")
        else:
            log_messages.append("No values were converted to NaN based on selected strings.")
            if _util_funcs and '_show_simple_modal_message' in _util_funcs:
                _util_funcs['_show_simple_modal_message']("Info", "No values matched the selected strings for NaN conversion.")
        _df_after_nan_conversion = df_to_convert 
        _df_after_imputation = None 

    if _main_app_callbacks:
        update_ui(_df_after_nan_conversion, _main_app_callbacks) 

    if dpg.does_item_exist(TAG_MV_LOG_TEXT):
        dpg.set_value(TAG_MV_LOG_TEXT, "\n".join(log_messages))

def _on_imputation_method_change(sender, app_data_method: str, user_data: Dict):
    col_name = user_data["col_name"]
    custom_value_input_tag = user_data["custom_value_input_tag"]
    
    current_custom_value = ""
    if dpg.does_item_exist(custom_value_input_tag):
        current_custom_value = dpg.get_value(custom_value_input_tag)
    
    _imputation_method_selections[col_name] = (app_data_method, current_custom_value if app_data_method == "Impute with Custom Value" else None)
    
    if dpg.does_item_exist(custom_value_input_tag):
        dpg.configure_item(custom_value_input_tag, show=(app_data_method == "Impute with Custom Value"))
    print(f"Step 4 MV: Column '{col_name}' imputation method: '{app_data_method}'")

def _on_custom_fill_value_change(sender, app_data_fill_value: str, user_data_col_name: str):
    if user_data_col_name in _imputation_method_selections:
        method, _ = _imputation_method_selections[user_data_col_name]
        if method == "Impute with Custom Value":
            _imputation_method_selections[user_data_col_name] = (method, app_data_fill_value)
    else:
         _imputation_method_selections[user_data_col_name] = ("Impute with Custom Value", app_data_fill_value)
    print(f"Step 4 MV: Column '{user_data_col_name}' custom fill value: '{app_data_fill_value}'")

def _execute_imputation_logic():
    global _main_app_callbacks, _df_after_nan_conversion, _current_df_for_this_step, _util_funcs, _df_after_imputation
    
    df_to_process = _df_after_nan_conversion if _df_after_nan_conversion is not None else _current_df_for_this_step

    if not _main_app_callbacks or df_to_process is None:
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Error", "No data or main callbacks available for imputation.")
        if _main_app_callbacks: 
             _main_app_callbacks['step4_missing_value_processing_complete'](None)
        return

    log_messages = ["--- Executing Selected Imputations (Step 4) ---"]
    df_processed = df_to_process.copy() 
    imputation_applied_any = False
    
    use_iterative_imputer = any(method == "Iterative Imputer (MICE)" for col, (method, _) in _imputation_method_selections.items())
    
    if use_iterative_imputer:
        log_messages.append("Iterative Imputer (MICE) selected for one or more columns.")
        log_messages.append("  └─ Applying IterativeImputer to all numeric columns with missing values.")
        
        numeric_cols_original_order = [col for col in df_processed.columns if pd.api.types.is_numeric_dtype(df_processed[col])]
        
        if not numeric_cols_original_order:
            log_messages.append("  └─ No numeric columns found to apply IterativeImputer.")
        else:
            df_numeric_for_imputation = df_processed[numeric_cols_original_order].copy()
            missing_in_numeric_before_iter = df_numeric_for_imputation.isnull().sum().sum()

            if missing_in_numeric_before_iter > 0:
                try:
                    iter_imputer = IterativeImputer(random_state=0, max_iter=10) 
                    df_numeric_imputed_values = iter_imputer.fit_transform(df_numeric_for_imputation)
                    df_numeric_imputed = pd.DataFrame(df_numeric_imputed_values, columns=numeric_cols_original_order, index=df_numeric_for_imputation.index)
                    
                    for col_name_iter in numeric_cols_original_order:
                        df_processed[col_name_iter] = df_numeric_imputed[col_name_iter]

                    missing_in_numeric_after_iter = df_processed[numeric_cols_original_order].isnull().sum().sum()
                    imputed_count_iter = missing_in_numeric_before_iter - missing_in_numeric_after_iter
                    log_messages.append(f"  └─ IterativeImputer successfully applied to numeric columns. {imputed_count_iter} missing values imputed.")
                    imputation_applied_any = True
                except Exception as e_iter:
                    log_messages.append(f"  └─ Error during IterativeImputer: {e_iter}")
            else:
                log_messages.append("  └─ No missing values found in numeric columns for IterativeImputer.")

    for col_name, (method, custom_value_str) in _imputation_method_selections.items():
        if col_name not in df_processed.columns:
            continue
        
        if method == "Iterative Imputer (MICE)" and pd.api.types.is_numeric_dtype(df_processed[col_name].dtype):
            log_messages.append(f"Column '{col_name}' was processed by Iterative Imputer (MICE).")
            continue 
        elif method == "Iterative Imputer (MICE)" and not pd.api.types.is_numeric_dtype(df_processed[col_name].dtype):
            log_messages.append(f"Column '{col_name}': Iterative Imputer (MICE) selected but column is non-numeric. Skipped by MICE.")
            continue

        if method == "Do Not Impute":
            continue

        log_messages.append(f"Processing Column '{col_name}' with method '{method}':")
        original_missing_count_col = df_processed[col_name].isnull().sum()

        try:
            if method == "Remove Rows with Missing":
                if original_missing_count_col > 0:
                    rows_before = len(df_processed)
                    df_processed.dropna(subset=[col_name], inplace=True) 
                    rows_after = len(df_processed)
                    log_messages.append(f"  └─ Removed {rows_before - rows_after} rows with missing values in '{col_name}'.")
                    imputation_applied_any = True
                else:
                    log_messages.append(f"  └─ No missing values to remove in '{col_name}'.")
            
            elif original_missing_count_col > 0: 
                series_to_impute_df = df_processed[[col_name]] 
                imputer = None

                if method == "Impute with 0":
                    imputer = SimpleImputer(strategy='constant', fill_value=0)
                elif method == "Impute with Mean":
                    if pd.api.types.is_numeric_dtype(df_processed[col_name].dtype):
                        imputer = SimpleImputer(strategy='mean')
                    else:
                        log_messages.append(f"  └─ Skipped: Mean imputation is for numeric columns only. '{col_name}' is {df_processed[col_name].dtype}.")
                        continue
                elif method == "Impute with Median":
                    if pd.api.types.is_numeric_dtype(df_processed[col_name].dtype):
                        imputer = SimpleImputer(strategy='median')
                    else:
                        log_messages.append(f"  └─ Skipped: Median imputation is for numeric columns only. '{col_name}' is {df_processed[col_name].dtype}.")
                        continue
                elif method == "Impute with Mode":
                    imputer = SimpleImputer(strategy='most_frequent')
                elif method == "Impute with Custom Value":
                    fill_val_typed = custom_value_str 
                    try:
                        original_col_dtype = df_to_process[col_name].dtype 
                        if pd.api.types.is_integer_dtype(original_col_dtype):
                            fill_val_typed = int(custom_value_str)
                        elif pd.api.types.is_float_dtype(original_col_dtype):
                            fill_val_typed = float(custom_value_str)
                        elif pd.api.types.is_bool_dtype(original_col_dtype):
                            if custom_value_str.lower() == 'true': fill_val_typed = True
                            elif custom_value_str.lower() == 'false': fill_val_typed = False
                            else: raise ValueError("Boolean custom value must be 'true' or 'false'")
                        elif pd.api.types.is_datetime64_any_dtype(original_col_dtype):
                            fill_val_typed = pd.to_datetime(custom_value_str)
                    except ValueError as ve:
                        log_messages.append(f"  └─ Warning: Custom value '{custom_value_str}' for column '{col_name}' (dtype: {original_col_dtype}) caused conversion error: {ve}. Using as string if applicable, or imputation might fail.")
                    imputer = SimpleImputer(strategy='constant', fill_value=fill_val_typed)

                if imputer:
                    transformed_col = imputer.fit_transform(series_to_impute_df)
                    df_processed[col_name] = transformed_col 
                    imputation_applied_any = True
                    
                    applied_stat_info = ""
                    if hasattr(imputer, 'statistics_') and len(imputer.statistics_) > 0:
                        stat_val = imputer.statistics_[0]
                        if isinstance(stat_val, float): applied_stat_info = f" (applied: {stat_val:.2f})"
                        else: applied_stat_info = f" (applied: '{stat_val}')"
                    log_messages.append(f"  └─ Imputed {original_missing_count_col} missing values{applied_stat_info}.")
            else: 
                 log_messages.append(f"  └─ No missing values to impute in '{col_name}'.")
        except Exception as e:
            log_messages.append(f"  └─ Error during {method} for '{col_name}': {e}")

    if not imputation_applied_any:
        log_messages.append("No imputation methods selected or applied.")
    else:
        log_messages.append("--- Imputation execution finished. ---")

    if dpg.does_item_exist(TAG_MV_LOG_TEXT):
        dpg.set_value(TAG_MV_LOG_TEXT, "\n".join(log_messages))
    
    _df_after_imputation = df_processed.copy() 

    if dpg.does_item_exist(TAG_MV_METHOD_SELECTION_TABLE) and _df_after_imputation is not None:
        _populate_method_selection_table(_df_after_imputation) 

    _show_simple_modal_message_func = _util_funcs.get('_show_simple_modal_message') if _util_funcs else None
    if _show_simple_modal_message_func:
        if imputation_applied_any:
             _show_simple_modal_message_func("Imputation Complete", "Selected missing value treatments have been applied.\nCheck the 'Select Imputation Method & View Status' table and logs for details.")
        else:
             _show_simple_modal_message_func("Notice", "No imputation methods were selected or applied.")
    
    _main_app_callbacks['step4_missing_value_processing_complete'](_df_after_imputation)


def _populate_method_selection_table(df: pd.DataFrame):
    """결측치 처리 방법 선택 테이블을 채웁니다. 현재 df에서 결측치가 있는 컬럼만 표시합니다.""" # 설명 수정
    if not dpg.does_item_exist(TAG_MV_METHOD_SELECTION_TABLE) or df is None or df.empty:
        if dpg.does_item_exist(TAG_MV_METHOD_SELECTION_TABLE):
            dpg.delete_item(TAG_MV_METHOD_SELECTION_TABLE, children_only=True)
            dpg.add_table_column(label="Info", parent=TAG_MV_METHOD_SELECTION_TABLE)
            with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
                dpg.add_text("No data available or no columns with missing values.") # 메시지 수정
        return
    
    dpg.delete_item(TAG_MV_METHOD_SELECTION_TABLE, children_only=True)
    
    imputation_options_base = [
        "Do Not Impute", 
        "Impute with 0", "Impute with Mean", "Impute with Median", "Impute with Mode", 
        "Impute with Custom Value", 
        "Remove Rows with Missing",
        "Iterative Imputer (MICE)" 
    ]

    dpg.add_table_column(label="Column Name", parent=TAG_MV_METHOD_SELECTION_TABLE, width_fixed=True, init_width_or_weight=200)
    dpg.add_table_column(label="Data Type", parent=TAG_MV_METHOD_SELECTION_TABLE, width_fixed=True, init_width_or_weight=120)
    dpg.add_table_column(label="Current Missing Count", parent=TAG_MV_METHOD_SELECTION_TABLE, width_fixed=True, init_width_or_weight=150) 
    dpg.add_table_column(label="Imputation Method", parent=TAG_MV_METHOD_SELECTION_TABLE, width_stretch=True) 
    dpg.add_table_column(label="Custom Value", parent=TAG_MV_METHOD_SELECTION_TABLE, width_fixed=True, init_width_or_weight=180)

    # 결측치가 있는 컬럼만 필터링하여 표시
    cols_with_missing = [col for col in df.columns if df[col].isnull().sum() > 0]

    if not cols_with_missing:
        with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
            dpg.add_text("No columns with missing values found in the current dataset.")
            # 빈 셀 추가 (컬럼 헤더 유지용)
            # 컬럼 수를 가져오는 방법 수정
            num_columns = 0
            if dpg.does_item_exist(TAG_MV_METHOD_SELECTION_TABLE):
                table_config = dpg.get_item_configuration(TAG_MV_METHOD_SELECTION_TABLE)
                num_columns = table_config.get('columns', 1) # 'columns' 키로 컬럼 수 가져오기 (기본값 1)
            
            for _ in range(max(0, num_columns - 1)): # 컬럼 수가 1보다 작을 경우 대비
                 dpg.add_text("")
        return

    for col_name in cols_with_missing: # 결측치가 있는 컬럼에 대해서만 반복
        with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
            dpg.add_text(col_name)
            dpg.add_text(str(df[col_name].dtype))
            missing_count = df[col_name].isnull().sum() # 이 값은 항상 > 0
            dpg.add_text(str(missing_count))
            
            current_method_tuple = _imputation_method_selections.get(col_name)
            current_method = current_method_tuple[0] if current_method_tuple else "Do Not Impute"
            
            custom_fill_input_tag = f"custom_fill_input_s4_{''.join(filter(str.isalnum, col_name))}"

            is_numeric = pd.api.types.is_numeric_dtype(df[col_name].dtype)
            current_col_options = imputation_options_base[:] 
            
            if not is_numeric:
                if "Impute with Mean" in current_col_options: current_col_options.remove("Impute with Mean")
                if "Impute with Median" in current_col_options: current_col_options.remove("Impute with Median")
                if "Iterative Imputer (MICE)" in current_col_options: current_col_options.remove("Iterative Imputer (MICE)")
            
            # 결측치가 있는 컬럼이므로 항상 콤보박스 표시
            dpg.add_combo(current_col_options, default_value=current_method, width=-1,
                          callback=_on_imputation_method_change, 
                          user_data={"col_name": col_name, "custom_value_input_tag": custom_fill_input_tag})
            
            current_custom_value = current_method_tuple[1] if current_method_tuple and current_method_tuple[1] is not None else ""
            dpg.add_input_text(tag=custom_fill_input_tag, default_value=current_custom_value, width=-1,
                               show=(current_method == "Impute with Custom Value"),
                               callback=_on_custom_fill_value_change, user_data=col_name)


def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _main_app_callbacks, _util_funcs
    _main_app_callbacks = main_callbacks
    if 'get_util_funcs' in main_callbacks:
        _util_funcs = main_callbacks['get_util_funcs']()

    main_callbacks['register_step_group_tag'](step_name, TAG_MV_STEP_GROUP)

    with dpg.group(tag=TAG_MV_STEP_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---") 
        dpg.add_separator()

        dpg.add_text("1. Convert Predefined Strings to NaN (Optional)", color=[255, 255, 0])
        dpg.add_text("Select common string representations to treat as missing values (affects current step only):")
        with dpg.group(tag=TAG_MV_PREDEFINED_NAN_GROUP, horizontal=True, horizontal_spacing=15):
            num_nans = len(PREDEFINED_NAN_STRINGS)
            cols = 3 
            items_per_col = (num_nans + cols - 1) // cols
            for col_idx in range(cols):
                with dpg.group():
                    start_idx = col_idx * items_per_col
                    end_idx = min((col_idx + 1) * items_per_col, num_nans)
                    for i in range(start_idx, end_idx):
                        nan_str = PREDEFINED_NAN_STRINGS[i]
                        display_label = f"'{nan_str}'" if nan_str else "'Empty String'" 
                        dpg.add_checkbox(label=display_label, tag=f"nan_cb_s4_{i}", 
                                         default_value=_selected_predefined_nans[nan_str],
                                         user_data=nan_str, callback=_on_predefined_nan_checkbox_change)
        dpg.add_button(label="Convert Selected Strings to NaN (Current Step)", tag=TAG_MV_CONVERT_PREDEFINED_NANS_BUTTON,
                       callback=_convert_selected_to_nans_logic, width=-1)
        dpg.add_spacer(height=10)

        dpg.add_text("2. Select Imputation Method & View Status", color=[255, 255, 0])
        with dpg.table(tag=TAG_MV_METHOD_SELECTION_TABLE, header_row=True, resizable=True, 
                       policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=350, scrollX=True, 
                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            dpg.add_table_column(label="Info", parent=TAG_MV_METHOD_SELECTION_TABLE)
            with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
                dpg.add_text("Data needed to select methods.")
        dpg.add_spacer(height=10)
        
        dpg.add_button(label="Apply Selected Imputations", tag=TAG_MV_EXECUTE_IMPUTATION_BUTTON, width=-1, height=30,
                       callback=_execute_imputation_logic)
        dpg.add_spacer(height=15)

        with dpg.group(tag=TAG_MV_RESULTS_GROUP): 
            dpg.add_text("3. Processing Log", color=[255, 255, 0])
            dpg.add_separator()
            with dpg.child_window(tag=TAG_MV_LOG_TEXT_AREA, height=150, border=True): 
                 dpg.add_text("Press 'Convert Selected Strings' or 'Apply Imputations' to see logs.", tag=TAG_MV_LOG_TEXT, wrap=-1)

    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui(df_input_for_step: Optional[pd.DataFrame], main_callbacks: dict):
    global _main_app_callbacks, _util_funcs, _current_df_for_this_step, _df_after_imputation
    
    if not _main_app_callbacks: _main_app_callbacks = main_callbacks
    if not _util_funcs and 'get_util_funcs' in _main_app_callbacks:
        _util_funcs = _main_app_callbacks['get_util_funcs']()

    _current_df_for_this_step = df_input_for_step 
    
    df_for_display_in_table = _current_df_for_this_step 
    if _df_after_imputation is not None:
        df_for_display_in_table = _df_after_imputation
    elif _df_after_nan_conversion is not None:
        df_for_display_in_table = _df_after_nan_conversion

    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_MV_STEP_GROUP):
        return
    
    if df_for_display_in_table is not None and not df_for_display_in_table.empty:
        _populate_method_selection_table(df_for_display_in_table)
    else:
        if dpg.does_item_exist(TAG_MV_METHOD_SELECTION_TABLE): 
            dpg.delete_item(TAG_MV_METHOD_SELECTION_TABLE, children_only=True)
            dpg.add_table_column(label="Info", parent=TAG_MV_METHOD_SELECTION_TABLE)
            with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
                dpg.add_text("No data to select methods for.")

    if dpg.does_item_exist(TAG_MV_LOG_TEXT):
        current_log = dpg.get_value(TAG_MV_LOG_TEXT)
        if not current_log or "Executing" not in current_log and "Converting" not in current_log :
             dpg.set_value(TAG_MV_LOG_TEXT, "Select actions and execute to see logs.")
    
def reset_missing_values_state():
    global _imputation_method_selections, _current_df_for_this_step, _selected_predefined_nans, _df_after_nan_conversion, _df_after_imputation
    _imputation_method_selections.clear()
    _current_df_for_this_step = None
    _df_after_nan_conversion = None 
    _df_after_imputation = None 
    
    for nan_str in PREDEFINED_NAN_STRINGS:
        _selected_predefined_nans[nan_str] = False
        if dpg.is_dearpygui_running():
            for i, predefined_str_iter in enumerate(PREDEFINED_NAN_STRINGS):
                if predefined_str_iter == nan_str:
                    cb_tag = f"nan_cb_s4_{i}"
                    if dpg.does_item_exist(cb_tag):
                        dpg.set_value(cb_tag, False)
                    break
    
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_MV_LOG_TEXT):
            dpg.set_value(TAG_MV_LOG_TEXT, "Select actions and execute to see logs.")
        
        if dpg.does_item_exist(TAG_MV_METHOD_SELECTION_TABLE):
            dpg.delete_item(TAG_MV_METHOD_SELECTION_TABLE, children_only=True)
            dpg.add_table_column(label="Info", parent=TAG_MV_METHOD_SELECTION_TABLE)
            with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
                dpg.add_text("Load data to select methods.")

    print("Step 4: Missing Values state has been reset.")

def get_missing_values_settings_for_saving() -> dict:
    return {
        "imputation_selections": _imputation_method_selections.copy(),
        "selected_predefined_nans_for_step4": _selected_predefined_nans.copy() 
    }

def apply_missing_values_settings_and_process(df_input: pd.DataFrame, settings: dict, main_callbacks: dict):
    global _imputation_method_selections, _selected_predefined_nans, _main_app_callbacks, _current_df_for_this_step, _df_after_nan_conversion, _df_after_imputation
    
    if not _main_app_callbacks: _main_app_callbacks = main_callbacks
    
    _current_df_for_this_step = df_input 
    _df_after_nan_conversion = None 
    _df_after_imputation = None 

    loaded_imputation_selections = settings.get("imputation_selections", {})
    _imputation_method_selections = loaded_imputation_selections.copy()

    loaded_predefined_nans = settings.get("selected_predefined_nans_for_step4", {})
    _selected_predefined_nans = loaded_predefined_nans.copy()

    if dpg.is_dearpygui_running():
        for i, nan_str_iter in enumerate(PREDEFINED_NAN_STRINGS):
            cb_tag = f"nan_cb_s4_{i}"
            if dpg.does_item_exist(cb_tag):
                dpg.set_value(cb_tag, _selected_predefined_nans.get(nan_str_iter, False))
    
    update_ui(df_input, main_callbacks)
    print("Step 4 Missing Values settings applied. UI updated. User needs to click 'Convert' or 'Apply' manually.")

