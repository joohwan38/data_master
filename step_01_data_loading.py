# step_01_data_loading.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
# from collections import Counter # 직접 사용 안하면 제거
from typing import Dict, Tuple, Optional, Any, List
# import traceback # 직접 사용 안하면 제거
import utils # utils.py의 함수들을 사용하기 위함

TAG_DL_GROUP = "step1_data_loading_group"
TAG_DL_SHAPE_TEXT = "step1_df_summary_shape_text"
TAG_DL_INFO_TABLE = "step1_df_summary_info_table"
TAG_DL_DESCRIBE_TABLE = "step1_df_summary_describe_table"
TAG_DL_HEAD_TABLE = "step1_df_summary_head_table"
TAG_DL_OVERVIEW_TAB_BAR = "step1_overview_tab_bar"
TAG_DL_TYPE_EDITOR_TABLE = "step1_type_editor_table"
TAG_DL_APPLY_TYPE_CHANGES_BUTTON = "step1_apply_type_changes_button"
TAG_DL_INFER_TYPES_BUTTON = "step1_infer_types_button"
TAG_DL_CUSTOM_NAN_INPUT = "step1_custom_nan_input"
TAG_DL_APPLY_CUSTOM_NAN_BUTTON = "step1_apply_custom_nan_button"
TAG_DL_MISSING_HANDLER_TABLE = "step1_missing_handler_table"
TAG_DL_APPLY_MISSING_TREATMENTS_BUTTON = "step1_apply_missing_treatments_button"

_type_selections: Dict[str, str] = {}
_imputation_selections: Dict[str, Tuple[str, Optional[str]]] = {}
_custom_nan_input_value: str = ""
_module_main_callbacks: Optional[Dict] = None

SENSITIVE_KEYWORDS = ['name', 'email', 'phone', 'ssn', '주민', '전번', '이멜', '이름']
DATE_FORMATS = ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d-%m-%Y", "%Y%m%d%H%M%S", "%Y.%m.%d"]
TIMEDELTA_KEYWORDS = ["duration", "period", "interval", "lead_time", "경과", "기간"]

def _infer_series_type(series: pd.Series) -> Tuple[str, Optional[str], bool]:
    if series.empty: return "Unknown", None, False
    s_valid = series.dropna()
    if s_valid.empty: return "All Missing", None, False
    sample = s_valid.head(min(max(1, int(len(s_valid) * 0.1)), 1000))
    if sample.empty: return "Effectively All Missing", None, False

    kind, name_low = series.dtype.kind, str(series.name).lower()
    if any(kw in name_low for kw in SENSITIVE_KEYWORDS):
        return ("Potentially Sensitive (Email?)" if sample.astype(str).str.contains('@').any() else "Potentially Sensitive (Name/Phone?)"), "Review PII", False

    is_bin_num = False
    if kind in 'iuf' and set(s_valid.astype(str).unique()) in [{'0','1'}, {'0.0','1.0'}, {'0.0','1'}, {'0','1.0'}]:
        is_bin_num = True; return "Numeric (Binary)", "Can be Categorical", is_bin_num
    if kind in 'iufc': return "Numeric", None, is_bin_num

    if kind == 'O' or pd.api.types.is_datetime64_any_dtype(series.dtype):
        if pd.api.types.is_datetime64_any_dtype(series.dtype): return "Datetime", "Original Dtype", False
        fmt, _ = _find_best_date_format(sample)
        if fmt: return "Datetime", f"Format ~{fmt}", False
    if _check_timedelta_type(sample, name_low, kind):
        return "Timedelta", ("From Numeric" if kind in 'iuf' else "From Text"), False
    if kind == 'O':
        num_type, is_bin = _check_numeric_conversion(sample)
        if num_type: return num_type, None, is_bin
    return _classify_categorical_or_text(s_valid)

def _find_best_date_format(sample: pd.Series) -> Tuple[Optional[str], float]:
    best_fmt, max_rate = None, 0.0
    for fmt in DATE_FORMATS:
        try:
            rate = pd.to_datetime(sample, format=fmt, errors='coerce').notna().mean()
            if rate > max_rate and rate > 0.85: max_rate, best_fmt = rate, fmt
        except (ValueError, TypeError): continue
    return best_fmt, max_rate

def _check_timedelta_type(sample: pd.Series, name_low: str, kind: str) -> bool:
    try:
        if kind == 'O' and pd.to_timedelta(sample, errors='coerce').notna().mean() > 0.8: return True
        if any(kw in name_low for kw in TIMEDELTA_KEYWORDS) and kind in 'iuf': return True
    except Exception: pass
    return False

def _check_numeric_conversion(sample: pd.Series) -> Tuple[Optional[str], bool]:
    try:
        num_conv = pd.to_numeric(sample.astype(str), errors='coerce')
        if num_conv.notna().mean() > 0.95:
            is_bin = set(map(str, num_conv.dropna().unique())) in [{'0','1'}, {'0.0','1.0'}, {'0.0','1'}, {'0','1.0'}]
            return ("Numeric (Binary from Text)" if is_bin else "Numeric (from Text)"), is_bin
    except Exception: pass
    return None, False

def _classify_categorical_or_text(s_valid: pd.Series) -> Tuple[str, Optional[str], bool]:
    n_uniq, len_val = s_valid.nunique(), len(s_valid)
    avg_len = s_valid.astype(str).str.len().mean() if len_val > 0 else 0
    if n_uniq == 2: return "Categorical (Binary Text)", None, False
    if n_uniq / len_val > 0.8 and n_uniq > 1000 and avg_len < 50: return "Text (ID/Code)", None, False
    if n_uniq < max(30, len_val * 0.05): return "Categorical", None, False
    if avg_len > 100: return "Text (Long/Free)", None, False
    return "Text (General)", None, False

def _apply_type_changes(main_callbacks: dict):
    df_after_s1, orig_df = main_callbacks['get_df_after_step1'](), main_callbacks['get_original_df']()
    df_proc = (df_after_s1.copy() if df_after_s1 is not None else (orig_df.copy() if orig_df is not None else None))
    if df_proc is None or not _type_selections: return
    for col, new_type in _type_selections.items():
        if col not in df_proc.columns: continue
        try: df_proc[col] = _convert_column_type(df_proc[col], new_type, orig_df)
        except Exception as e: print(f"Err converting '{col}': {e}")
    if df_proc is not None: main_callbacks['step1_processing_complete'](df_proc)

def _convert_column_type(series: pd.Series, new_type: str, orig_df: pd.DataFrame) -> pd.Series:
    if new_type == "Numeric (int)": return pd.to_numeric(series, errors='coerce').astype('Int64')
    if new_type == "Numeric (float)": return pd.to_numeric(series, errors='coerce').astype(float)
    if new_type.startswith("Categorical"): return series.astype('category')
    if new_type.startswith("Datetime"): return pd.to_datetime(series, errors='coerce')
    if new_type.startswith("Timedelta"): return pd.to_timedelta(series, errors='coerce')
    if new_type.startswith("Text (") or new_type=="Original Text" or new_type.startswith("Potentially Sensitive"):
        return series.astype(pd.StringDtype())
    if new_type == "Original" and series.name in orig_df.columns: return series.astype(orig_df[series.name].dtype)
    return series

def _apply_custom_nans(main_callbacks: dict, custom_nan_str: str):
    df = main_callbacks['get_df_after_step1']() if main_callbacks['get_df_after_step1']() is not None else main_callbacks['get_original_df']()
    if df is None or not custom_nan_str.strip(): return
    df_copy, nan_vals = df.copy(), [s.strip() for s in custom_nan_str.split(',')]
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object' or pd.api.types.is_string_dtype(df_copy[col].dtype):
            df_copy[col].replace(nan_vals, np.nan, inplace=True)
    main_callbacks['step1_processing_complete'](df_copy)

def _apply_missing_value_treatments(main_callbacks: dict):
    df = main_callbacks['get_df_after_step1']() if main_callbacks['get_df_after_step1']() is not None else main_callbacks['get_original_df']()
    if df is None or not _imputation_selections: return
    df_copy = df.copy()
    for col, (method, fill_val_str) in _imputation_selections.items():
        if col not in df_copy.columns: continue
        try: df_copy = _apply_imputation_method(df_copy, col, method, fill_val_str)
        except Exception as e: print(f"Err imputing '{col}': {e}")
    _imputation_selections.clear()
    main_callbacks['step1_processing_complete'](df_copy)

def _apply_imputation_method(df: pd.DataFrame, col: str, method: str, fill_val_str: Optional[str]) -> pd.DataFrame:
    s = df[col]
    if method == "drop_rows": return df.dropna(subset=[col])
    if method == "fill_mean" and pd.api.types.is_numeric_dtype(s): s.fillna(s.mean(), inplace=True)
    elif method == "fill_median" and pd.api.types.is_numeric_dtype(s): s.fillna(s.median(), inplace=True)
    elif method == "fill_mode": s.fillna(s.mode().iloc[0] if not s.mode().empty else np.nan, inplace=True)
    elif method == "fill_custom" and fill_val_str is not None:
        try: val = float(fill_val_str) if pd.api.types.is_numeric_dtype(s) else fill_val_str
        except ValueError: val = fill_val_str
        s.fillna(val, inplace=True)
    elif method == "as_category_missing":
        if not pd.api.types.is_categorical_dtype(s): s = s.astype('category')
        if "Missing" not in s.cat.categories: s = s.cat.add_categories("Missing")
        s.fillna("Missing", inplace=True)
    df[col] = s
    return df

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _module_main_callbacks
    _module_main_callbacks = main_callbacks
    main_callbacks['register_step_group_tag'](step_name, TAG_DL_GROUP)
    with dpg.group(tag=TAG_DL_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---"); dpg.add_separator()
        # 파일 로드/리셋 버튼, 파일 요약 텍스트 UI는 main_app.py에서 관리하므로 여기서는 생성 안 함
        dpg.add_text("--- Data Details & Preprocessing Options ---"); dpg.add_separator()
        with dpg.tab_bar(tag=TAG_DL_OVERVIEW_TAB_BAR):
            _create_data_summary_tab()
            _create_type_editor_tab(main_callbacks)
            _create_missing_value_handler_tab(main_callbacks)
    main_callbacks['register_module_updater'](step_name, update_ui)

def _trigger_step1_ui_update():
    if not _module_main_callbacks: return
    curr_df = _module_main_callbacks.get('get_current_df', lambda: None)()
    orig_df = _module_main_callbacks.get('get_original_df', lambda: None)()
    u_funcs = _module_main_callbacks.get('get_util_funcs', lambda: {})()
    update_ui(curr_df, orig_df, u_funcs) # file_path 인자 제거

def _create_data_summary_tab():
    with dpg.tab(label="Data Summary"):
        dpg.add_button(label="Refresh DataFrame Info", width=-1, height=30, callback=_trigger_step1_ui_update)
        dpg.add_text("Shape: N/A", tag=TAG_DL_SHAPE_TEXT)
        dpg.add_separator()
        dpg.add_text("Column Info (Type, Missing, Unique):")
        with dpg.table(header_row=True,resizable=True,policy=dpg.mvTable_SizingFixedFit,scrollY=True,height=200,
                      borders_outerH=True,borders_innerV=True,borders_innerH=True,borders_outerV=True,
                      tag=TAG_DL_INFO_TABLE, scrollX=True): pass
        dpg.add_separator()
        dpg.add_text("Descriptive Statistics (Numeric Columns):")
        with dpg.table(header_row=True,resizable=True,policy=dpg.mvTable_SizingFixedFit,scrollY=True,height=200,
                      borders_outerH=True,borders_innerV=True,borders_innerH=True,borders_outerV=True,
                      tag=TAG_DL_DESCRIBE_TABLE, scrollX=True): pass
        dpg.add_separator()
        dpg.add_text("Data Head (First 5 Rows):")
        with dpg.table(header_row=True,resizable=True,policy=dpg.mvTable_SizingFixedFit,scrollY=True,height=150,
                      borders_outerH=True,borders_innerV=True,borders_innerH=True,borders_outerV=True,
                      tag=TAG_DL_HEAD_TABLE, scrollX=True): pass

def _create_type_editor_tab(main_callbacks: dict):
    with dpg.tab(label="Variable Type Editor"):
        dpg.add_text("Infer and set data types for analysis.")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Infer All Types",tag=TAG_DL_INFER_TYPES_BUTTON, callback=lambda: _populate_type_editor_table(main_callbacks, infer_all=True))
            dpg.add_button(label="Apply Type Changes",tag=TAG_DL_APPLY_TYPE_CHANGES_BUTTON, callback=lambda: _apply_type_changes(main_callbacks))
        with dpg.table(header_row=True,resizable=True,policy=dpg.mvTable_SizingFixedFit,scrollY=True,height=400,
                      borders_outerH=True,borders_innerV=True,borders_innerH=True,borders_outerV=True,
                      tag=TAG_DL_TYPE_EDITOR_TABLE, scrollX=True): pass

def _create_missing_value_handler_tab(main_callbacks: dict):
    with dpg.tab(label="Missing Value Handler"):
        dpg.add_text("Define and handle missing values.")
        with dpg.group(horizontal=True):
            global _custom_nan_input_value
            dpg.add_text("Custom NaN strings (comma-separated):")
            # 콜백에서 직접 _custom_nan_input_value를 수정하는 대신, Apply 버튼 클릭 시 값을 가져오도록 변경
            dpg.add_input_text(tag=TAG_DL_CUSTOM_NAN_INPUT, width=300, default_value=_custom_nan_input_value)
            dpg.add_button(label="Convert Custom to NaN", tag=TAG_DL_APPLY_CUSTOM_NAN_BUTTON,
                          callback=lambda: _apply_custom_nans(main_callbacks, dpg.get_value(TAG_DL_CUSTOM_NAN_INPUT)))
        dpg.add_button(label="Apply Selected Imputations", tag=TAG_DL_APPLY_MISSING_TREATMENTS_BUTTON,
                      callback=lambda: _apply_missing_value_treatments(main_callbacks))
        with dpg.table(header_row=True,resizable=True,policy=dpg.mvTable_SizingFixedFit,scrollY=True,height=400,
                      borders_outerH=True,borders_innerV=True,borders_innerH=True,borders_outerV=True,
                      tag=TAG_DL_MISSING_HANDLER_TABLE, scrollX=True): pass

def _populate_type_editor_table(main_callbacks: dict, infer_all: bool = False):
    global _type_selections
    orig_df, df_after_s1 = main_callbacks['get_original_df'](), main_callbacks['get_df_after_step1']()
    u_funcs = main_callbacks['get_util_funcs']()
    if not dpg.does_item_exist(TAG_DL_TYPE_EDITOR_TABLE): return
    dpg.delete_item(TAG_DL_TYPE_EDITOR_TABLE, children_only=True)
    if orig_df is None:
        dpg.add_table_column(label="Info", parent=TAG_DL_TYPE_EDITOR_TABLE);
        with dpg.table_row(parent=TAG_DL_TYPE_EDITOR_TABLE): dpg.add_text("Load data.")
        return
    cols = ["Col Name", "Original Dtype", "Applied Dtype", "Selected Type", "Unique", "Sample Values"]
    for lbl in cols: dpg.add_table_column(label=lbl, parent=TAG_DL_TYPE_EDITOR_TABLE)
    avail_types = ["Original","Numeric (int)","Numeric (float)","Categorical","Datetime","Timedelta",
                   "Text (ID/Code)","Text (Long/Free)","Text (General)","Potentially Sensitive (Review Needed)"]
    for col_name in orig_df.columns:
        with dpg.table_row(parent=TAG_DL_TYPE_EDITOR_TABLE):
            dpg.add_text(u_funcs['format_text_for_display'](col_name, 30))
            dpg.add_text(str(orig_df[col_name].dtype))
            applied_dtype = str(df_after_s1[col_name].dtype) if df_after_s1 is not None and col_name in df_after_s1.columns else str(orig_df[col_name].dtype)
            dpg.add_text(applied_dtype)
            curr_sel = _type_selections.get(col_name, "Original")
            if infer_all and col_name not in _type_selections: # infer_all 시에만, 그리고 아직 사용자 선택이 없을 때만 추론값으로 _type_selections 업데이트
                inferred_type_tuple = _infer_series_type(orig_df[col_name])
                curr_sel = _map_inferred_to_available_type(inferred_type_tuple[0], inferred_type_tuple[2])
                _type_selections[col_name] = curr_sel # 추론된 타입으로 내부 상태 업데이트
            
            combo_tag = f"type_combo_{''.join(filter(str.isalnum, col_name))}"
            dpg.add_combo(items=avail_types, default_value=curr_sel, tag=combo_tag, width=-1, user_data=col_name,
                          callback=lambda s, a, u: _type_selections.update({u: a})) # 사용자 선택 시 _type_selections 업데이트
            dpg.add_text(str(orig_df[col_name].nunique()))
            samples = ", ".join(orig_df[col_name].dropna().head(3).astype(str).tolist())
            dpg.add_text(u_funcs['format_text_for_display'](samples, 50))

def _map_inferred_to_available_type(inferred: str, is_binary: bool) -> str:
    if inferred.startswith("Numeric"): return "Numeric (int)" if is_binary or "Binary" in inferred else "Numeric (float)"
    if inferred == "Datetime": return "Datetime"
    if inferred.startswith("Categorical"): return "Categorical"
    if inferred == "Text (ID/Code)": return "Text (ID/Code)"
    if inferred == "Text (Long/Free)": return "Text (Long/Free)"
    if inferred.startswith("Text"): return "Text (General)"
    if inferred == "Timedelta": return "Timedelta"
    if inferred.startswith("Potentially Sensitive"): return "Potentially Sensitive (Review Needed)"
    return "Original"

def _populate_missing_handler_table(main_callbacks: dict):
    global _imputation_selections
    df = main_callbacks['get_df_after_step1']() if main_callbacks['get_df_after_step1']() is not None else main_callbacks['get_original_df']()
    u_funcs = main_callbacks['get_util_funcs']()
    if not dpg.does_item_exist(TAG_DL_MISSING_HANDLER_TABLE): return
    dpg.delete_item(TAG_DL_MISSING_HANDLER_TABLE, children_only=True)
    if df is None:
        dpg.add_table_column(label="Info", parent=TAG_DL_MISSING_HANDLER_TABLE)
        with dpg.table_row(parent=TAG_DL_MISSING_HANDLER_TABLE): dpg.add_text("Load data.")
        return
    cols = ["Col Name", "Missing Count", "Missing %", "Imputation Method", "Custom Fill Value"]
    for lbl in cols: dpg.add_table_column(label=lbl, parent=TAG_DL_MISSING_HANDLER_TABLE)
    impute_methods = ["Keep Missing","Drop Rows with Missing","Fill with Mean","Fill with Median",
                      "Fill with Mode","Fill with Custom Value","As 'Missing' Category"]
    for col_name in df.columns:
        missing_count = df[col_name].isnull().sum()
        if missing_count == 0: continue
        with dpg.table_row(parent=TAG_DL_MISSING_HANDLER_TABLE):
            dpg.add_text(u_funcs['format_text_for_display'](col_name, 30))
            dpg.add_text(str(missing_count))
            dpg.add_text(f"{missing_count / len(df) * 100:.2f}%")
            curr_method, curr_fill_val = "Keep Missing", ""
            if col_name in _imputation_selections:
                curr_method = _convert_method_tag_to_display(_imputation_selections[col_name][0])
                curr_fill_val = _imputation_selections[col_name][1] or ""
            method_tag, fill_tag = f"impute_method_{''.join(filter(str.isalnum,col_name))}", f"impute_value_{''.join(filter(str.isalnum,col_name))}"
            dpg.add_combo(items=impute_methods, default_value=curr_method, tag=method_tag, width=-1,
                         callback=lambda s, a, u=(col_name, fill_tag): _update_imputation_selection(u[0], a, u[1]))
            dpg.add_input_text(tag=fill_tag, default_value=curr_fill_val, width=-1,
                              callback=lambda s, a, u=(col_name, method_tag): _update_imputation_value(u[0], a, u[1]))

def _convert_method_tag_to_display(tag: str) -> str:
    return {"keep_missing":"Keep Missing","drop_rows":"Drop Rows with Missing","fill_mean":"Fill with Mean",
            "fill_median":"Fill with Median","fill_mode":"Fill with Mode","fill_custom":"Fill with Custom Value",
            "as_category_missing":"As 'Missing' Category"}.get(tag, "Keep Missing")

def _convert_display_to_method_tag(display: str) -> str:
    return {v: k for k, v in {"keep_missing":"Keep Missing","drop_rows":"Drop Rows with Missing",
                              "fill_mean":"Fill with Mean","fill_median":"Fill with Median",
                              "fill_mode":"Fill with Mode","fill_custom":"Fill with Custom Value",
                              "as_category_missing":"As 'Missing' Category"}.items()}.get(display, "keep_missing")

def _update_imputation_selection(col_name: str, method_display: str, fill_input_tag: str):
    global _imputation_selections
    fill_val = dpg.get_value(fill_input_tag) if dpg.does_item_exist(fill_input_tag) else ""
    _imputation_selections[col_name] = (_convert_display_to_method_tag(method_display), fill_val)

def _update_imputation_value(col_name: str, fill_value: str, method_combo_tag: str):
    global _imputation_selections
    method_disp = dpg.get_value(method_combo_tag) if dpg.does_item_exist(method_combo_tag) else "Keep Missing"
    _imputation_selections[col_name] = (_convert_display_to_method_tag(method_disp), fill_value)

def update_ui(current_df: Optional[pd.DataFrame], original_df: Optional[pd.DataFrame], 
             util_funcs: dict, file_path: Optional[str] = None): # file_path는 이제 main_app에서 관리
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_DL_GROUP): return
    
    df_display = current_df if current_df is not None else original_df
    if dpg.does_item_exist(TAG_DL_SHAPE_TEXT):
        dpg.set_value(TAG_DL_SHAPE_TEXT, f"Shape: {df_display.shape}" if df_display is not None else "Shape: N/A")
    
    create_tbl_func = util_funcs.get('create_table_with_data', lambda *a, **k: None)
    if df_display is not None:
        info_data = {"Col Name":df_display.columns.astype(str), "Dtype":[str(d) for d in df_display.dtypes],
                     "Missing":df_display.isnull().sum().values, "Unique":df_display.nunique().values}
        create_tbl_func(TAG_DL_INFO_TABLE, pd.DataFrame(info_data), parent_df_for_widths=pd.DataFrame(info_data))
        num_df = df_display.select_dtypes(include=np.number)
        if not num_df.empty:
            desc_df = num_df.describe().reset_index().rename(columns={'index': 'Statistic'})
            create_tbl_func(TAG_DL_DESCRIBE_TABLE, desc_df, utils_format_numeric=True, parent_df_for_widths=desc_df)
        else: create_tbl_func(TAG_DL_DESCRIBE_TABLE, pd.DataFrame({"Info": ["No numeric columns."]}))
        create_tbl_func(TAG_DL_HEAD_TABLE, df_display.head(), parent_df_for_widths=df_display)
    else: create_tbl_func(TAG_DL_INFO_TABLE, None); create_tbl_func(TAG_DL_DESCRIBE_TABLE, None); create_tbl_func(TAG_DL_HEAD_TABLE, None)
    
    temp_cb = {'get_original_df': lambda: original_df, 'get_df_after_step1': lambda: current_df, 'get_util_funcs': lambda: util_funcs}
    _populate_type_editor_table(temp_cb, infer_all=False)
    _populate_missing_handler_table(temp_cb)

def process_newly_loaded_data(original_df: pd.DataFrame, main_callbacks: dict):
    global _type_selections, _imputation_selections, _custom_nan_input_value
    if original_df is None: return
    _type_selections.clear(); _imputation_selections.clear(); _custom_nan_input_value = ""
    if dpg.does_item_exist(TAG_DL_CUSTOM_NAN_INPUT): dpg.set_value(TAG_DL_CUSTOM_NAN_INPUT, "")
    for col in original_df.columns:
        inferred, _, is_bin = _infer_series_type(original_df[col])
        _type_selections[col] = _map_inferred_to_available_type(inferred, is_bin)
    main_callbacks['step1_processing_complete'](original_df.copy())

def get_step1_settings_for_saving() -> dict:
    """Step1의 현재 설정을 저장용 딕셔너리로 반환합니다."""
    global _type_selections, _imputation_selections, _custom_nan_input_value
    return {
        'type_selections': _type_selections.copy(),
        'imputation_selections': _imputation_selections.copy(),
        'custom_nan_input_value': _custom_nan_input_value
    }

def apply_step1_settings_and_process(original_df: pd.DataFrame, settings: dict, main_callbacks: dict):
    global _type_selections, _imputation_selections, _custom_nan_input_value
    if original_df is None: return
    _type_selections = settings.get('type_selections', {}).copy()
    _imputation_selections = settings.get('imputation_selections', {}).copy()
    _custom_nan_input_value = settings.get('custom_nan_input_value', "")
    if dpg.does_item_exist(TAG_DL_CUSTOM_NAN_INPUT): dpg.set_value(TAG_DL_CUSTOM_NAN_INPUT, _custom_nan_input_value)
    df_processed = original_df.copy()
    if _custom_nan_input_value.strip():
        nan_vals = [s.strip() for s in _custom_nan_input_value.split(',')]
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object' or pd.api.types.is_string_dtype(df_processed[col].dtype):
                df_processed[col].replace(nan_vals, np.nan, inplace=True)
    for col_name, col_type in _type_selections.items():
        if col_name in df_processed.columns:
            try: df_processed[col_name] = _convert_column_type(df_processed[col_name], col_type, original_df)
            except Exception as e: print(f"Err converting '{col_name}': {e}")
    for col_name, (method, fill_value) in _imputation_selections.items():
        if col_name in df_processed.columns:
            try: df_processed = _apply_imputation_method(df_processed, col_name, method, fill_value)
            except Exception as e: print(f"Err imputing '{col_name}': {e}")
    main_callbacks['step1_processing_complete'](df_processed)

def reset_step1_state():
    global _type_selections, _imputation_selections, _custom_nan_input_value
    _type_selections.clear(); _imputation_selections.clear(); _custom_nan_input_value = ""
    if dpg.does_item_exist(TAG_DL_CUSTOM_NAN_INPUT): dpg.set_value(TAG_DL_CUSTOM_NAN_INPUT, "")