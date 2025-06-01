# step_01_data_loading.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import utils

# --- DPG Tags for Step 1 ---
TAG_DL_GROUP = "step1_data_loading_group"
TAG_DL_SHAPE_TEXT = "step1_df_summary_shape_text"
TAG_DL_INFO_TABLE = "step1_df_summary_info_table"
TAG_DL_DESCRIBE_TABLE = "step1_df_summary_describe_table"
TAG_DL_HEAD_TABLE = "step1_df_summary_head_table"
TAG_DL_OVERVIEW_TAB_BAR = "step1_overview_tab_bar"
TAG_DL_TYPE_EDITOR_TABLE = "step1_type_editor_table"
TAG_DL_APPLY_TYPE_CHANGES_BUTTON = "step1_apply_type_changes_button"
TAG_DL_INFER_TYPES_BUTTON = "step1_infer_types_button"
# Removed: TAG_DL_CUSTOM_NAN_INPUT, TAG_DL_APPLY_CUSTOM_NAN_BUTTON, TAG_DL_MISSING_HANDLER_TABLE, TAG_DL_APPLY_MISSING_TREATMENTS_BUTTON

_type_selections: Dict[str, str] = {}
# Removed: _imputation_selections, _custom_nan_input_value
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None # Added to store util funcs locally

SENSITIVE_KEYWORDS = ['name', 'email', 'phone', 'ssn', '주민', '전번', '이멜', '이름']
DATE_FORMATS = ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d-%m-%Y", "%Y%m%d%H%M%S", "%Y.%m.%d"]
TIMEDELTA_KEYWORDS = ["duration", "period", "interval", "lead_time", "경과", "기간"]

# _infer_series_type and its helpers are kept as they were in your original file
# (Assuming they were working well for type inference as you mentioned)
def _infer_series_type(series: pd.Series) -> Tuple[str, Optional[str], bool]:
    if series.empty: return "Unknown", None, False
    s_valid = series.dropna()
    if s_valid.empty: return "All Missing", None, False
    sample_size = min(len(s_valid), 500) 
    sample = s_valid.sample(n=sample_size, random_state=1) if len(s_valid) > sample_size else s_valid

    kind, name_low = series.dtype.kind, str(series.name).lower()
    if any(kw in name_low for kw in SENSITIVE_KEYWORDS):
        return ("Potentially Sensitive (Email?)" if sample.astype(str).str.contains('@').any() else "Potentially Sensitive (Name/Phone?)"), "Review PII", False

    is_bin_num = False
    if kind in 'iuf':
        unique_vals = s_valid.unique()
        if len(unique_vals) == 2 and all(val in [0, 1, 0.0, 1.0] for val in unique_vals):
            is_bin_num = True
            return "Numeric (Binary)", "Can be Categorical", is_bin_num
    if kind in 'iufc': return "Numeric", None, is_bin_num

    if kind == 'O' or pd.api.types.is_datetime64_any_dtype(series.dtype):
        if pd.api.types.is_datetime64_any_dtype(series.dtype): return "Datetime", "Original Dtype", False
        fmt, _ = _find_best_date_format(sample.astype(str))
        if fmt: return "Datetime", f"Format ~{fmt}", False
    
    if _check_timedelta_type(sample, name_low, kind):
        return "Timedelta", ("From Numeric" if kind in 'iuf' else "From Text"), False
    
    if kind == 'O':
        num_type, is_bin = _check_numeric_conversion(sample)
        if num_type: return num_type, None, is_bin
        
    return _classify_categorical_or_text(s_valid)

def _find_best_date_format(sample: pd.Series) -> Tuple[Optional[str], float]:
    best_fmt, max_rate = None, 0.0
    if sample.empty: return best_fmt, max_rate
    for fmt in DATE_FORMATS:
        try:
            converted_dates = pd.to_datetime(sample.astype(str), format=fmt, errors='coerce')
            rate = converted_dates.notna().mean()
            if rate > max_rate and rate > 0.85:
                max_rate, best_fmt = rate, fmt
        except (ValueError, TypeError):
            continue
    return best_fmt, max_rate

def _check_timedelta_type(sample: pd.Series, name_low: str, kind: str) -> bool:
    if sample.empty: return False
    try:
        if kind == 'O' and pd.to_timedelta(sample.astype(str), errors='coerce').notna().mean() > 0.8: return True
        if any(kw in name_low for kw in TIMEDELTA_KEYWORDS) and kind in 'iuf': return True
    except Exception: 
        pass
    return False

def _check_numeric_conversion(sample: pd.Series) -> Tuple[Optional[str], bool]:
    if sample.empty: return None, False
    try:
        num_conv = pd.to_numeric(sample.astype(str), errors='coerce')
        if num_conv.notna().mean() > 0.95:
            unique_numeric_vals = num_conv.dropna().unique()
            is_bin = len(unique_numeric_vals) == 2 and all(val in [0, 1, 0.0, 1.0] for val in unique_numeric_vals)
            return ("Numeric (Binary from Text)" if is_bin else "Numeric (from Text)"), is_bin
    except Exception:
        pass
    return None, False

def _classify_categorical_or_text(s_valid: pd.Series) -> Tuple[str, Optional[str], bool]:
    if s_valid.empty: return "Unknown", None, False
    n_uniq, len_val = s_valid.nunique(), len(s_valid)
    avg_len = s_valid.astype(str).str.len().mean() if len_val > 0 else 0
    
    if n_uniq == 2: return "Categorical (Binary Text)", None, False
    if n_uniq / len_val > 0.8 and n_uniq > min(1000, 0.5 * len_val) and avg_len < 50 : return "Text (ID/Code)", None, False
    if n_uniq < max(30, len_val * 0.05) and n_uniq < 200 : return "Categorical", None, False
    if avg_len > 100 and n_uniq / len_val > 0.5 : return "Text (Long/Free)", None, False
    return "Text (General)", None, False

def _apply_type_changes_and_process():
    """Applies selected type changes based on _type_selections and calls step1_processing_complete."""
    global _module_main_callbacks, _type_selections, _util_funcs
    if not _module_main_callbacks:
        print("Error: Main callbacks not set in step_01_data_loading for apply_type_changes.")
        return

    # In Step 1, type changes are applied to the original_df to create df_after_step1
    original_df = _module_main_callbacks.get('get_original_df', lambda: None)()
    
    if original_df is None:
        if _util_funcs and '_show_simple_modal_message' in _util_funcs: # Check _util_funcs
            _util_funcs['_show_simple_modal_message']("Error", "Original data not available to apply type changes.")
        else:
            print("Error: Original data not available to apply type changes.")
        _module_main_callbacks['step1_processing_complete'](None) # Pass None if no data
        return

    df_processed = original_df.copy() # Start with a fresh copy of original_df

    if not _type_selections:
        print("No type changes selected in _type_selections. Passing a copy of original_df.")

        _module_main_callbacks['step1_processing_complete'](df_processed)
        return

    for col_name, new_type_key in _type_selections.items():
        if col_name not in df_processed.columns:
            continue
        try:
            df_processed[col_name] = _convert_column_type(original_df[col_name].copy(), new_type_key, original_df) # Apply to a copy of the original series
        except Exception as e:
            error_message = f"Error converting column '{col_name}' to type '{new_type_key}': {e}"
            print(error_message)
            if _util_funcs and '_show_simple_modal_message' in _util_funcs:
                 _util_funcs['_show_simple_modal_message']("Type Conversion Error", error_message)
            # Keep the column as is in df_processed if conversion fails for this column
            df_processed[col_name] = original_df[col_name].copy()


    _module_main_callbacks['step1_processing_complete'](df_processed)
    print("Type changes applied from Step 1 and step1_processing_complete called.")

def _convert_column_type(series: pd.Series, new_type_key: str, original_df_for_context: Optional[pd.DataFrame]) -> pd.Series:
    if new_type_key == "Numeric (int)":
        return pd.to_numeric(series, errors='coerce').astype('Int64')
    elif new_type_key == "Numeric (float)":
        return pd.to_numeric(series, errors='coerce').astype(float)
    elif new_type_key.startswith("Categorical"):
        return series.astype('category')
    elif new_type_key.startswith("Datetime"):
        try:
            return pd.to_datetime(series, errors='coerce')
        except Exception:
            return pd.to_datetime(series, errors='coerce')
    elif new_type_key.startswith("Timedelta"):
        return pd.to_timedelta(series, errors='coerce')
    elif new_type_key.startswith("Text (") or new_type_key.startswith("Potentially Sensitive"):
        return series.astype(pd.StringDtype())
    elif new_type_key == "Original":
        if original_df_for_context is not None and series.name in original_df_for_context.columns:
            return series.astype(original_df_for_context[series.name].dtype)
        return series 
    else:
        return series.astype(str)

def _populate_type_editor_table(df_original: Optional[pd.DataFrame], df_processed_after_step1: Optional[pd.DataFrame]):
    global _type_selections, _util_funcs
    
    if not dpg.does_item_exist(TAG_DL_TYPE_EDITOR_TABLE): return
    dpg.delete_item(TAG_DL_TYPE_EDITOR_TABLE, children_only=True)

    if df_original is None:
        dpg.add_table_column(label="Info", parent=TAG_DL_TYPE_EDITOR_TABLE)
        with dpg.table_row(parent=TAG_DL_TYPE_EDITOR_TABLE):
            dpg.add_text("No data loaded to edit types.")
        return

    headers = ["Column Name", "Original Dtype", "Applied Dtype", "Selected Type", "Unique Values", "Sample Values (Top 3)"]
    for header_label in headers:
        dpg.add_table_column(label=header_label, parent=TAG_DL_TYPE_EDITOR_TABLE)

    available_types_for_combo = [
        "Original", "Numeric (int)", "Numeric (float)", "Categorical", 
        "Datetime", "Timedelta", "Text (General)", "Text (ID/Code)", 
        "Text (Long/Free)", "Potentially Sensitive (Review Needed)"
    ]
    
    format_text_func = _util_funcs.get('format_text_for_display', lambda t, m: str(t)[:m]) if _util_funcs else lambda t, m: str(t)[:m]

    for col_name in df_original.columns:
        with dpg.table_row(parent=TAG_DL_TYPE_EDITOR_TABLE):
            dpg.add_text(format_text_func(col_name, 30))
            dpg.add_text(str(df_original[col_name].dtype))
            
            applied_dtype_str = "N/A" # Default if df_processed_after_step1 is None
            if df_processed_after_step1 is not None and col_name in df_processed_after_step1.columns:
                applied_dtype_str = str(df_processed_after_step1[col_name].dtype)
            dpg.add_text(applied_dtype_str)

            current_selection = _type_selections.get(col_name, "Original")
            combo_tag = f"type_combo_step1_{''.join(filter(str.isalnum, str(col_name)))}"
            dpg.add_combo(items=available_types_for_combo, default_value=current_selection, 
                          tag=combo_tag, width=-1, user_data=col_name,
                          callback=lambda sender, app_data, user_data: _type_selections.update({user_data: app_data}))
            
            dpg.add_text(str(df_original[col_name].nunique()))
            
            sample_values_series = df_original[col_name].dropna().unique()[:3]
            sample_values_str = ", ".join([str(val) for val in sample_values_series])
            dpg.add_text(format_text_func(sample_values_str, 50))

def _infer_all_types_and_populate():
    global _module_main_callbacks, _type_selections
    if not _module_main_callbacks: return
    
    original_df = _module_main_callbacks.get('get_original_df', lambda: None)()
    df_after_step1 = _module_main_callbacks.get('get_df_after_step1', lambda: None)()
    
    if original_df is None:
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("Info", "No data loaded to infer types.")
        return

    _type_selections.clear()
    for col_name in original_df.columns:
        inferred_type_tuple = _infer_series_type(original_df[col_name])
        inferred_key = inferred_type_tuple[0]
        
        if "Numeric (Binary" in inferred_key: mapped_type = "Numeric (int)"
        elif "Numeric (from Text)" == inferred_key: mapped_type = "Numeric (float)"
        elif "Numeric" == inferred_key: mapped_type = "Numeric (float)"
        elif "Categorical" in inferred_key: mapped_type = "Categorical"
        elif "Datetime" == inferred_key: mapped_type = "Datetime"
        elif "Timedelta" == inferred_key: mapped_type = "Timedelta"
        elif "Text (ID/Code)" == inferred_key: mapped_type = "Text (ID/Code)"
        elif "Text (Long/Free)" == inferred_key: mapped_type = "Text (Long/Free)"
        elif "Text (General)" == inferred_key: mapped_type = "Text (General)"
        elif "Potentially Sensitive" in inferred_key: mapped_type = "Potentially Sensitive (Review Needed)"
        else: mapped_type = "Original"
        _type_selections[col_name] = mapped_type
        
    _populate_type_editor_table(original_df, df_after_step1) # Pass df_after_step1 for "Applied Dtype"
    print("Inferred types for all columns and updated editor table selections.")


def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    if 'get_util_funcs' in main_callbacks:
        _util_funcs = main_callbacks['get_util_funcs']()

    main_callbacks['register_step_group_tag'](step_name, TAG_DL_GROUP)

    with dpg.group(tag=TAG_DL_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---") 
        dpg.add_separator()
        
        dpg.add_text("--- Data Details & Type Options ---") # Modified title
        dpg.add_separator()

        with dpg.tab_bar(tag=TAG_DL_OVERVIEW_TAB_BAR):
            with dpg.tab(label="Data Summary"):
                dpg.add_button(label="Refresh DataFrame Info", width=-1, height=30,
                               callback=lambda: update_ui(
                                   _module_main_callbacks.get('get_df_after_step1', lambda: None)(),
                                   _module_main_callbacks.get('get_original_df', lambda: None)(),
                                   _util_funcs,
                                   _module_main_callbacks.get('get_loaded_file_path', lambda: None)()
                                ))
                dpg.add_text("Shape: N/A", tag=TAG_DL_SHAPE_TEXT)
                dpg.add_separator()
                dpg.add_text("Column Info (Type, Missing, Unique):")
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit, 
                               scrollY=True, height=200, tag=TAG_DL_INFO_TABLE, scrollX=True,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                    pass 
                dpg.add_separator()
                dpg.add_text("Descriptive Statistics (Numeric Columns):")
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                               scrollY=True, height=200, tag=TAG_DL_DESCRIBE_TABLE, scrollX=True,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                    pass 
                dpg.add_separator()
                dpg.add_text("Data Head (First 5 Rows):")
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                               scrollY=True, height=150, tag=TAG_DL_HEAD_TABLE, scrollX=True,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                    pass

            with dpg.tab(label="Variable Type Editor"):
                dpg.add_text("Infer and set data types for analysis.")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Infer All Types", tag=TAG_DL_INFER_TYPES_BUTTON,
                                   callback=_infer_all_types_and_populate)
                    dpg.add_button(label="Apply Type Changes", tag=TAG_DL_APPLY_TYPE_CHANGES_BUTTON,
                                   callback=_apply_type_changes_and_process)
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                               scrollY=True, height=450, tag=TAG_DL_TYPE_EDITOR_TABLE, scrollX=True,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                    pass 
            
            # Missing Value Handler Tab and its creation function call are removed.

    main_callbacks['register_module_updater'](step_name, update_ui)

# file_path parameter is now optional as it's not always used by update_ui directly for all tables
def update_ui(current_df_after_step1: Optional[pd.DataFrame], 
              original_df: Optional[pd.DataFrame], 
              util_funcs_from_main: dict, 
              file_path: Optional[str] = None): 
    global _util_funcs # Ensure global _util_funcs is used/updated
    
    _util_funcs = util_funcs_from_main # Always update from main_app, as it holds the central util_funcs
    
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_DL_GROUP):
        return

    df_display_for_summary = current_df_after_step1 if current_df_after_step1 is not None else original_df

    if dpg.does_item_exist(TAG_DL_SHAPE_TEXT):
        dpg.set_value(TAG_DL_SHAPE_TEXT, f"Shape: {df_display_for_summary.shape}" if df_display_for_summary is not None else "Shape: N/A")
    
    create_tbl_func = _util_funcs.get('create_table_with_data') if _util_funcs else None

    if df_display_for_summary is not None and create_tbl_func:
        info_data = pd.DataFrame({
            "Column Name": df_display_for_summary.columns.astype(str),
            "Dtype": [str(d) for d in df_display_for_summary.dtypes],
            "Missing": df_display_for_summary.isnull().sum().values,
            "Unique": df_display_for_summary.nunique().values
        })
        create_tbl_func(TAG_DL_INFO_TABLE, info_data, parent_df_for_widths=info_data)
        
        num_df = df_display_for_summary.select_dtypes(include=np.number)
        if not num_df.empty:
            desc_df = num_df.describe().reset_index().rename(columns={'index': 'Statistic'})
            create_tbl_func(TAG_DL_DESCRIBE_TABLE, desc_df, utils_format_numeric=True, parent_df_for_widths=desc_df)
        else:
            create_tbl_func(TAG_DL_DESCRIBE_TABLE, pd.DataFrame({"Info": ["No numeric columns to describe."]}))
        
        create_tbl_func(TAG_DL_HEAD_TABLE, df_display_for_summary.head(), parent_df_for_widths=df_display_for_summary)
    else:
        for table_tag in [TAG_DL_INFO_TABLE, TAG_DL_DESCRIBE_TABLE, TAG_DL_HEAD_TABLE]:
            if dpg.does_item_exist(table_tag):
                dpg.delete_item(table_tag, children_only=True)
                dpg.add_table_column(label="Info", parent=table_tag)
                with dpg.table_row(parent=table_tag):
                    dpg.add_text("No data to display or table utility missing.")

    _populate_type_editor_table(original_df, current_df_after_step1) # Pass both for full context


def process_newly_loaded_data(original_df: pd.DataFrame, main_callbacks: dict):

    global _type_selections, _module_main_callbacks
    if not _module_main_callbacks: _module_main_callbacks = main_callbacks

    if original_df is None:
        main_callbacks['step1_processing_complete'](None)
        return
    
    _type_selections.clear() 

    main_callbacks['step1_processing_complete'](original_df.copy())
    print("New data loaded. Types are original. Use 'Infer/Apply Type Changes' in Step 1.")

def get_step1_settings_for_saving() -> dict:
    global _type_selections
    return {
        'type_selections': _type_selections.copy(),
    }

def apply_step1_settings_and_process(original_df: pd.DataFrame, settings: dict, main_callbacks: dict):

    global _type_selections, _module_main_callbacks
    if not _module_main_callbacks: _module_main_callbacks = main_callbacks
    
    if original_df is None: 
        main_callbacks['step1_processing_complete'](None)
        return

    loaded_type_selections = settings.get('type_selections', {}).copy()
    _type_selections = loaded_type_selections # Restore selections

    # If settings are applied, it's reasonable to assume the types from settings should be applied
    # to create the initial df_after_step1 for this loaded session.
    df_processed_on_settings_load = original_df.copy()
    if _type_selections:
        for col_name, col_type_key in _type_selections.items():
            if col_name in df_processed_on_settings_load.columns:
                try:
                    df_processed_on_settings_load[col_name] = _convert_column_type(
                        original_df[col_name].copy(), col_type_key, original_df
                    )
                except Exception as e:
                    print(f"Error applying loaded type '{col_type_key}' to column '{col_name}': {e}")
                    df_processed_on_settings_load[col_name] = original_df[col_name].copy() # Revert to original on error
    
    main_callbacks['step1_processing_complete'](df_processed_on_settings_load)
    print("Step 1 settings (type selections) applied and data processed accordingly.")

def reset_step1_state():
    global _type_selections
    _type_selections.clear()
    print("Step 1 state (type selections) has been reset.")

# Added for clarity if main_app.py needs to reset more from outside (e.g. new file)
def reset_step1_state_for_new_file():
    reset_step1_state() # For now, same as general reset