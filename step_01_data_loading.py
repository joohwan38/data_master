# step_01_data_loading.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, Tuple, Optional, Any, List
import traceback

# UI 태그 정의
TAG_DL_GROUP = "step1_data_loading_group"
TAG_DL_FILE_SUMMARY_TEXT = "step1_file_summary_text"
TAG_DL_SHAPE_TEXT = "step1_df_summary_shape_text"
TAG_DL_INFO_TABLE = "step1_df_summary_info_table"
TAG_DL_DESCRIBE_TABLE = "step1_df_summary_describe_table"
TAG_DL_HEAD_TABLE = "step1_df_summary_head_table"
TAG_DL_PROCESSED_TABLE = "step1_df_processed_table"
TAG_DL_OVERVIEW_TAB_BAR = "step1_overview_tab_bar"
TAG_DL_RESET_BUTTON = "step1_reset_data_button"
TAG_DL_TYPE_EDITOR_TABLE = "step1_type_editor_table"
TAG_DL_APPLY_TYPE_CHANGES_BUTTON = "step1_apply_type_changes_button"
TAG_DL_INFER_TYPES_BUTTON = "step1_infer_types_button"
TAG_DL_CUSTOM_NAN_INPUT = "step1_custom_nan_input"
TAG_DL_APPLY_CUSTOM_NAN_BUTTON = "step1_apply_custom_nan_button"
TAG_DL_MISSING_HANDLER_TABLE = "step1_missing_handler_table"
TAG_DL_APPLY_MISSING_TREATMENTS_BUTTON = "step1_apply_missing_treatments_button"
TAG_DL_COLUMN_CONFIG_TABLE_GROUP = "dl_column_config_table_group_tag"

# 전역 변수 (모듈 내 상태 관리)
_type_selections: Dict[str, str] = {}
_imputation_selections: Dict[str, Tuple[str, Optional[str]]] = {}
_custom_nan_input_value: str = ""

# 타입 추론 관련 상수
SENSITIVE_KEYWORDS = ['name', 'email', 'phone', 'ssn', '주민', '전번', '이멜', '이름']
DATE_FORMATS = [
    "%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y",
    "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
    "%d-%m-%Y", "%Y%m%d%H%M%S", "%Y.%m.%d"
]
TIMEDELTA_KEYWORDS = ["duration", "period", "interval", "lead_time", "경과", "기간"]

def _infer_series_type(series: pd.Series) -> Tuple[str, Optional[str], bool]:
    """
    Pandas Series를 분석하여 데이터 타입을 추론
    Returns: (추론타입, 힌트, 이진숫자형여부)
    """
    if series.empty:
        return "Unknown", None, False

    series_valid = series.dropna()
    if series_valid.empty:
        return "All Missing", None, False

    sample_size = min(max(1, int(len(series_valid) * 0.1)), 1000)
    series_sample = series_valid.head(sample_size)
    if series_sample.empty:
        return "Effectively All Missing (after sample)", None, False

    current_dtype_kind = series.dtype.kind
    series_name_lower = str(series.name).lower()

    # 민감 정보 체크
    if any(keyword in series_name_lower for keyword in SENSITIVE_KEYWORDS):
        if series_sample.astype(str).str.contains('@').any():
            return "Potentially Sensitive (Email?)", "Review for PII", False
        return "Potentially Sensitive (Name/Phone?)", "Review for PII", False

    # 이진 숫자 체크
    is_binary_numeric = False
    if current_dtype_kind in 'iuf':
        str_unique_values = set(series_valid.astype(str).unique())
        binary_sets = [{'0', '1'}, {'0.0', '1.0'}, {'0.0', '1'}, {'0', '1.0'}]
        if str_unique_values in binary_sets:
            is_binary_numeric = True
            return "Numeric (Binary)", "Can be Categorical", is_binary_numeric

    # 숫자형 체크
    if current_dtype_kind in 'iufc':
        return "Numeric", None, is_binary_numeric

    # 날짜/시간형 체크
    if current_dtype_kind == 'O' or pd.api.types.is_datetime64_any_dtype(series.dtype):
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return "Datetime", "Original Dtype was Datetime", False

        # 날짜 형식 추론
        best_date_format, max_conversion_rate = _find_best_date_format(series_sample)
        if best_date_format:
            return "Datetime", f"Format ~{best_date_format}", False

    # Timedelta 체크
    if _check_timedelta_type(series, series_sample, series_name_lower, current_dtype_kind):
        return "Timedelta", "From Numeric (unit needed)" if current_dtype_kind in 'iuf' else "From Text", False

    # Object 타입의 숫자 변환 시도
    if current_dtype_kind == 'O':
        numeric_type, is_binary = _check_numeric_conversion(series_sample)
        if numeric_type:
            return numeric_type, None, is_binary

    # 범주형 또는 텍스트형 판단
    return _classify_categorical_or_text(series_valid)

def _find_best_date_format(series_sample: pd.Series) -> Tuple[Optional[str], float]:
    """날짜 형식 찾기"""
    best_format = None
    max_conversion_rate = 0
    
    for fmt in DATE_FORMATS:
        try:
            converted_dates = pd.to_datetime(series_sample, format=fmt, errors='coerce')
            conversion_rate = converted_dates.notna().sum() / len(series_sample)
            if conversion_rate > max_conversion_rate and conversion_rate > 0.85:
                max_conversion_rate = conversion_rate
                best_format = fmt
        except (ValueError, TypeError):
            continue
    
    return best_format, max_conversion_rate

def _check_timedelta_type(series: pd.Series, series_sample: pd.Series, 
                         series_name_lower: str, dtype_kind: str) -> bool:
    """Timedelta 타입 체크"""
    try:
        if dtype_kind == 'O':
            converted_td = pd.to_timedelta(series_sample, errors='coerce')
            if converted_td.notna().sum() / len(series_sample) > 0.8:
                return True
        if any(keyword in series_name_lower for keyword in TIMEDELTA_KEYWORDS) and dtype_kind in 'iuf':
            return True
    except Exception:
        pass
    return False

def _check_numeric_conversion(series_sample: pd.Series) -> Tuple[Optional[str], bool]:
    """Object 타입의 숫자 변환 체크"""
    try:
        numeric_converted = pd.to_numeric(series_sample.astype(str), errors='coerce')
        conversion_rate = numeric_converted.notna().sum() / len(series_sample)
        if conversion_rate > 0.95:
            unique_vals = set(map(str, numeric_converted.dropna().unique()))
            binary_sets = [{'0', '1'}, {'0.0', '1.0'}, {'0.0', '1'}, {'0', '1.0'}]
            if unique_vals in binary_sets:
                return "Numeric (Binary from Text)", True
            return "Numeric (from Text)", False
    except Exception:
        pass
    return None, False

def _classify_categorical_or_text(series_valid: pd.Series) -> Tuple[str, Optional[str], bool]:
    """범주형 또는 텍스트형 분류"""
    num_unique = series_valid.nunique()
    len_valid = len(series_valid)
    avg_str_len = series_valid.astype(str).str.len().mean() if len_valid > 0 else 0

    if num_unique == 2:
        return "Categorical (Binary Text)", None, False
    
    if num_unique / len_valid > 0.8 and num_unique > 1000 and avg_str_len < 50:
        return "Text (ID/Code)", None, False
    
    if num_unique < max(30, len_valid * 0.05):
        return "Categorical", None, False
    
    if avg_str_len > 100:
        return "Text (Long/Free)", None, False
    
    return "Text (General)", None, False

def _apply_type_changes(main_callbacks: dict):
    """타입 변경 적용"""
    print("--- _apply_type_changes called ---")
    
    util_funcs = main_callbacks['get_util_funcs']() # 오류 처리 등에 여전히 필요할 수 있음

    print(f"Applying type changes with _type_selections at function start: {_type_selections}")

    df_after_s1 = main_callbacks['get_df_after_step1']()
    original_df = main_callbacks['get_original_df']()
    df_to_process = None

    if df_after_s1 is not None:
        print("Using df_after_step1 for processing.")
        df_to_process = df_after_s1.copy()
    elif original_df is not None:
        print("Using original_df for processing as df_after_step1 was None.")
        df_to_process = original_df.copy()
    
    if df_to_process is None:
        print("Error: No DataFrame available. Cannot apply type changes.")
        return

    if not _type_selections:
        print("Warning: _type_selections is empty. No type changes to apply.")
        return
    
    print(f"Proceeding with type changes for {len(_type_selections)} columns. DataFrame to process shape: {df_to_process.shape}")
    
    conversion_errors_occurred = False # 오류 발생 여부 플래그
    for col_name, new_type in _type_selections.items():
        if col_name not in df_to_process.columns:
            print(f"Warning: Column '{col_name}' not found. Skipping.")
            continue
        
        try:
            print(f"  Converting '{col_name}' (current: {df_to_process[col_name].dtype}) to '{new_type}'...")
            converted_series = _convert_column_type(df_to_process[col_name], new_type, 
                                                    main_callbacks['get_original_df'](), util_funcs)
            df_to_process[col_name] = converted_series
            print(f"  Success for '{col_name}'. New dtype: {df_to_process[col_name].dtype}")
        except Exception as e:
            conversion_errors_occurred = True
            print(f"  Critical Error during conversion of '{col_name}': {e}")
           
    print("Type conversion loop finished. Calling step1_processing_complete...")
    if df_to_process is not None:
        df_to_process.info()
        main_callbacks['step1_processing_complete'](df_to_process)
        print("step1_processing_complete called successfully.")
    else:
        print("Error: df_to_process became None unexpectedly.")


def _convert_column_type(series: pd.Series, new_type: str, original_df: pd.DataFrame, util_funcs: dict = None) -> pd.Series:
    """컬럼 타입 변환"""
    if new_type == "Numeric (int)":
        return pd.to_numeric(series, errors='coerce').astype('Int64')
    elif new_type == "Numeric (float)":
        return pd.to_numeric(series, errors='coerce').astype(float)
    elif new_type == "Categorical" or new_type.startswith("Categorical ("):
        return series.astype('category')
    elif new_type.startswith("Datetime"):
        return pd.to_datetime(series, errors='coerce')
    elif new_type.startswith("Timedelta"):
        return pd.to_timedelta(series, errors='coerce')
    elif new_type.startswith("Text (") or new_type == "Original Text":
        return series.astype(pd.StringDtype())
    elif new_type == "Original":
        if series.name in original_df.columns:
            return series.astype(original_df[series.name].dtype)
    elif new_type.startswith("Potentially Sensitive"):
        if series.dtype == 'object':
            return series.astype(pd.StringDtype())
    
    return series

def _apply_custom_nans(main_callbacks: dict, custom_nan_str: str):
    """사용자 정의 NaN 값 적용"""
    df = main_callbacks['get_df_after_step1']()
    if df is None:
        df = main_callbacks['get_original_df']()
    if df is None:
        print("No data loaded to apply custom NaN.")
        return
    
    if not custom_nan_str.strip():
        print("No custom NaN values specified.")
        return
    
    df = df.copy()
    nan_values = [s.strip() for s in custom_nan_str.split(',')]
    print(f"Applying custom NaN values: {nan_values}")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].replace(nan_values, np.nan, inplace=True)
    
    main_callbacks['step1_processing_complete'](df)
    print("Custom NaN values applied.")

def _apply_missing_value_treatments(main_callbacks: dict):
    """결측치 처리 적용"""
    df = main_callbacks['get_df_after_step1']()
    if df is None:
        df = main_callbacks['get_original_df']()
    if df is None:
        print("No data loaded for missing value treatment.")
        return
    
    df = df.copy()
    print("Applying missing value treatments...")
    
    for col_name, (method, fill_value_str) in _imputation_selections.items():
        if col_name not in df.columns:
            continue
        
        try:
            print(f"  Treating '{col_name}' with method: {method}")
            df = _apply_imputation_method(df, col_name, method, fill_value_str)
            print(f"  Treatment completed for '{col_name}'")
        except Exception as e:
            print(f"  Error treating '{col_name}': {e}")
            traceback.print_exc()
    
    _imputation_selections.clear()
    main_callbacks['step1_processing_complete'](df)
    print("Missing value treatments applied.")

def _apply_imputation_method(df: pd.DataFrame, col_name: str, 
                            method: str, fill_value_str: str) -> pd.DataFrame:
    """개별 결측치 처리 방법 적용"""
    if method == "drop_rows":
        return df.dropna(subset=[col_name])
    elif method == "fill_mean" and pd.api.types.is_numeric_dtype(df[col_name]):
        df[col_name].fillna(df[col_name].mean(), inplace=True)
    elif method == "fill_median" and pd.api.types.is_numeric_dtype(df[col_name]):
        df[col_name].fillna(df[col_name].median(), inplace=True)
    elif method == "fill_mode":
        mode_val = df[col_name].mode().iloc[0] if not df[col_name].mode().empty else np.nan
        df[col_name].fillna(mode_val, inplace=True)
    elif method == "fill_custom" and fill_value_str:
        try:
            # 타입에 맞게 변환 시도
            if pd.api.types.is_numeric_dtype(df[col_name]):
                converted_value = float(fill_value_str)
            else:
                converted_value = fill_value_str
            df[col_name].fillna(converted_value, inplace=True)
        except ValueError:
            df[col_name].fillna(fill_value_str, inplace=True)
    elif method == "as_category_missing":
        if not pd.api.types.is_categorical_dtype(df[col_name]):
            df[col_name] = df[col_name].astype('category')
        if "Missing" not in df[col_name].cat.categories:
            df[col_name] = df[col_name].cat.add_categories("Missing")
        df[col_name].fillna("Missing", inplace=True)
    
    return df

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """UI 생성"""
    global _module_main_callbacks
    _module_main_callbacks = main_callbacks
    main_callbacks['register_step_group_tag'](step_name, TAG_DL_GROUP)
    
    with dpg.group(tag=TAG_DL_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        
        # 파일 로드 버튼
        with dpg.group(horizontal=True):
            dpg.add_button(label="Open Parquet File", 
                          callback=main_callbacks['show_file_dialog'])
            dpg.add_button(label="Reset to Original Data", 
                          tag=TAG_DL_RESET_BUTTON,
                          callback=lambda: main_callbacks['reset_current_df_to_original']())
        
        dpg.add_text("No data loaded.", tag=TAG_DL_FILE_SUMMARY_TEXT, wrap=700)
        dpg.add_spacer(height=10)
        
        dpg.add_text("--- Data Details & Preprocessing ---")
        dpg.add_separator()
        
        # 탭 구성
        with dpg.tab_bar(tag=TAG_DL_OVERVIEW_TAB_BAR):
            _create_data_summary_tab()
            _create_type_editor_tab(main_callbacks)
            _create_missing_value_handler_tab(main_callbacks)
    
    main_callbacks['register_module_updater'](step_name, update_ui)

def _trigger_step1_ui_update():
    """Data Summary 탭의 UI 업데이트를 트리거하는 내부 함수"""
    if not _module_main_callbacks:
        print("Error: main_callbacks not initialized for Step 1 UI update.")
        return

    # main_callbacks를 통해 필요한 데이터 가져오기
    current_df = _module_main_callbacks.get('get_current_df', lambda: None)()
    original_df = _module_main_callbacks.get('get_original_df', lambda: None)()
    util_funcs = _module_main_callbacks.get('get_util_funcs', lambda: {})()
    file_path = _module_main_callbacks.get('get_loaded_file_path', lambda: None)()
    
    # update_ui 함수 호출
    # update_ui 함수 시그니처: update_ui(current_df: pd.DataFrame, original_df: pd.DataFrame, util_funcs: dict, file_path: str = None)
    update_ui(current_df, original_df, util_funcs, file_path)
    print("Data Summary tab refreshed.")

def _create_data_summary_tab():
    """데이터 요약 탭 생성"""
    with dpg.tab(label="Data Summary"):
        dpg.add_button(label="Refresh DataFrame Info", width=-1, height=30,
                      callback=lambda: _trigger_step1_ui_update())
        dpg.add_text("Shape: N/A (No data)", tag=TAG_DL_SHAPE_TEXT)
        dpg.add_separator()
        
        dpg.add_text("Column Info (Type, Missing, Unique):")
        with dpg.table(header_row=True, resizable=True, 
                      policy=dpg.mvTable_SizingFixedFit,
                      borders_outerH=True, borders_innerV=True, 
                      borders_innerH=True, borders_outerV=True,
                      tag=TAG_DL_INFO_TABLE, scrollX=True, 
                      scrollY=True, height=200, freeze_columns=0):
            pass
        
        dpg.add_separator()
        dpg.add_text("Descriptive Statistics (Numeric Columns):")
        with dpg.table(header_row=True, resizable=True, 
                      policy=dpg.mvTable_SizingFixedFit,
                      borders_outerH=True, borders_innerV=True, 
                      borders_innerH=True, borders_outerV=True,
                      tag=TAG_DL_DESCRIBE_TABLE, scrollX=True, 
                      scrollY=True, height=200, freeze_columns=0):
            pass
        
        dpg.add_separator()
        dpg.add_text("Data Head (First 5 Rows):")
        with dpg.table(header_row=True, resizable=True, 
                      policy=dpg.mvTable_SizingFixedFit,
                      borders_outerH=True, borders_innerV=True, 
                      borders_innerH=True, borders_outerV=True,
                      tag=TAG_DL_HEAD_TABLE, scrollX=True, 
                      scrollY=True, height=150, freeze_columns=0):
            pass

def _create_type_editor_tab(main_callbacks: dict):
    """타입 편집 탭 생성"""
    with dpg.tab(label="Variable Type Editor"):
        dpg.add_text("Infer and set data types for analysis.")
        with dpg.group(horizontal=True):
            dpg.add_button(label="Infer All Types (Rule-based)", 
                          tag=TAG_DL_INFER_TYPES_BUTTON,
                          callback=lambda: _populate_type_editor_table(main_callbacks, infer_all=True))
            dpg.add_button(label="Apply Type Changes", 
                          tag=TAG_DL_APPLY_TYPE_CHANGES_BUTTON,
                          callback=lambda: _apply_type_changes(main_callbacks))
        
        with dpg.table(header_row=True, resizable=True, 
                      policy=dpg.mvTable_SizingFixedFit,
                      tag=TAG_DL_TYPE_EDITOR_TABLE, scrollX=True, 
                      scrollY=True, height=400,
                      borders_outerH=True, borders_innerV=True, 
                      borders_innerH=True, borders_outerV=True):
            pass

def _create_missing_value_handler_tab(main_callbacks: dict):
    """결측값 처리 탭 생성"""
    with dpg.tab(label="Missing Value Handler"):
        dpg.add_text("Define and handle missing values.")
        
        with dpg.group(horizontal=True):
            global _custom_nan_input_value
            dpg.add_text("Custom NaN strings (comma-separated):")
            dpg.add_input_text(tag=TAG_DL_CUSTOM_NAN_INPUT, width=300, 
                             default_value=_custom_nan_input_value,
                             callback=lambda s, a: globals().update(_custom_nan_input_value=a))
            dpg.add_button(label="Convert to NaN", 
                          tag=TAG_DL_APPLY_CUSTOM_NAN_BUTTON,
                          callback=lambda: _apply_custom_nans(main_callbacks, _custom_nan_input_value))
        
        dpg.add_button(label="Apply Missing Value Treatments", 
                      tag=TAG_DL_APPLY_MISSING_TREATMENTS_BUTTON,
                      callback=lambda: _apply_missing_value_treatments(main_callbacks))
        
        with dpg.table(header_row=True, resizable=True, 
                      policy=dpg.mvTable_SizingFixedFit,
                      tag=TAG_DL_MISSING_HANDLER_TABLE, scrollX=True, 
                      scrollY=True, height=400,
                      borders_outerH=True, borders_innerV=True, 
                      borders_innerH=True, borders_outerV=True):
            pass

def _populate_type_editor_table(main_callbacks: dict, infer_all: bool = False):
    global _type_selections
    
    original_df = main_callbacks['get_original_df']()
    # df_after_step1은 Step 1의 모든 처리(타입 변경 포함)가 완료된 후의 DataFrame입니다.
    df_after_step1 = main_callbacks['get_df_after_step1']() 
    util_funcs = main_callbacks['get_util_funcs']()
    
    if not dpg.does_item_exist(TAG_DL_TYPE_EDITOR_TABLE):
        print(f"Error: Table with tag '{TAG_DL_TYPE_EDITOR_TABLE}' does not exist.")
        return
    
    dpg.delete_item(TAG_DL_TYPE_EDITOR_TABLE, children_only=True)
    
    if original_df is None: # 원본 데이터프레임 기준으로 테이블을 구성하므로 original_df를 체크
        dpg.add_table_column(label="Info", parent=TAG_DL_TYPE_EDITOR_TABLE)
        with dpg.table_row(parent=TAG_DL_TYPE_EDITOR_TABLE):
            dpg.add_text("Load data to edit types.")
        # 모달 대신 콘솔에만 출력하거나, 상태바가 있다면 그곳에 표시 가능
        print("Info: Please load data first to infer or edit types.")
        return
    
    # 테이블 헤더 수정: "Applied Dtype" 추가
    columns = ["Column Name", "Original Dtype", "Applied Dtype", 
               "Selected Type", "Unique Count", "Sample Values"]
    for col_label in columns: # 변수명 변경 (col -> col_label)
        dpg.add_table_column(label=col_label, parent=TAG_DL_TYPE_EDITOR_TABLE)
    
    available_types = [
        "Original", "Numeric (int)", "Numeric (float)",
        "Categorical", "Datetime", "Timedelta",
        "Text (ID/Code)", "Text (Long/Free)", "Text (General)",
        "Potentially Sensitive (Review Needed)"
    ]
    
    for col_name in original_df.columns:
        with dpg.table_row(parent=TAG_DL_TYPE_EDITOR_TABLE):
            # 컬럼명
            dpg.add_text(util_funcs['format_text_for_display'](col_name, max_chars=30))
            
            # 원본 dtype (original_df 기준)
            dpg.add_text(str(original_df[col_name].dtype))
            
            # Applied Dtype (df_after_step1 기준, 없으면 original_df 기준)
            applied_dtype_str = ""
            # 고유 태그 대신, 이 텍스트는 매번 새로 그려지므로 직접 값을 설정합니다.
            if df_after_step1 is not None and col_name in df_after_step1.columns:
                applied_dtype_str = str(df_after_step1[col_name].dtype)
            elif original_df is not None and col_name in original_df.columns: 
                # df_after_step1이 None일 경우 (예: 최초 로드 후 아직 Apply 안 함) original_df의 dtype 표시
                applied_dtype_str = str(original_df[col_name].dtype)
            dpg.add_text(applied_dtype_str)
            
            # 타입 선택 콤보박스
            # current_selection은 _type_selections에 저장된 값 (사용자 선택/추론)을 따름
            current_selection = _type_selections.get(col_name, "Original")
            if infer_all and col_name not in _type_selections: # infer_all 시 _type_selections 업데이트
                inferred_series_type_tuple = _infer_series_type(original_df[col_name]) # util_funcs 전달 불필요 시 제거
                current_selection = _map_inferred_to_available_type(
                    inferred_series_type_tuple[0], inferred_series_type_tuple[2] # inferred_type, is_binary
                )
                _type_selections[col_name] = current_selection
            
            combo_tag = f"type_combo_{col_name.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')}" # 태그용 이름 정제
            dpg.add_combo(items=available_types, default_value=current_selection, 
                          tag=combo_tag, width=-1, user_data=col_name,
                          callback=lambda s, a, u: _type_selections.update({u: a}))
            
            # 고유값 수 (original_df 기준)
            dpg.add_text(str(original_df[col_name].nunique()))
            
            # 샘플 값 (original_df 기준)
            sample_vals = original_df[col_name].dropna().head(3).astype(str).tolist()
            dpg.add_text(util_funcs['format_text_for_display'](
                ", ".join(sample_vals), max_chars=50
            ))

def _map_inferred_to_available_type(inferred_type: str, is_binary: bool) -> str:
    """추론된 타입을 사용 가능한 타입으로 매핑"""
    if inferred_type.startswith("Numeric"):
        if is_binary or "Binary" in inferred_type:
            return "Numeric (int)"
        return "Numeric (float)"
    elif inferred_type == "Datetime":
        return "Datetime"
    elif inferred_type.startswith("Categorical"):
        return "Categorical"
    elif inferred_type == "Text (ID/Code)":
        return "Text (ID/Code)"
    elif inferred_type == "Text (Long/Free)":
        return "Text (Long/Free)"
    elif inferred_type.startswith("Text"):
        return "Text (General)"
    elif inferred_type == "Timedelta":
        return "Timedelta"
    elif inferred_type.startswith("Potentially Sensitive"):
        return "Potentially Sensitive (Review Needed)"
    else:
        return "Original"

def _populate_missing_handler_table(main_callbacks: dict):
    """결측값 처리 테이블 채우기"""
    global _imputation_selections
    
    df = main_callbacks['get_original_df']()
    if main_callbacks['get_df_after_step1']() is not None:
        df = main_callbacks['get_df_after_step1']()
    
    util_funcs = main_callbacks['get_util_funcs']()
    
    if not dpg.does_item_exist(TAG_DL_MISSING_HANDLER_TABLE):
        return
    
    dpg.delete_item(TAG_DL_MISSING_HANDLER_TABLE, children_only=True)
    
    if df is None:
        dpg.add_table_column(label="Info", parent=TAG_DL_MISSING_HANDLER_TABLE)
        with dpg.table_row(parent=TAG_DL_MISSING_HANDLER_TABLE):
            dpg.add_text("Load data to handle missing values.")
        return
    
    # 테이블 헤더
    columns = ["Column Name", "Missing Count", "Missing %", 
              "Imputation Method", "Custom Fill Value"]
    for col in columns:
        dpg.add_table_column(label=col, parent=TAG_DL_MISSING_HANDLER_TABLE)
    
    imputation_methods = [
        "Keep Missing", "Drop Rows with Missing", "Fill with Mean",
        "Fill with Median", "Fill with Mode", "Fill with Custom Value",
        "As 'Missing' Category"
    ]
    
    # 각 컬럼에 대한 행 생성
    for col_name in df.columns:
        missing_count = df[col_name].isnull().sum()
        if missing_count == 0:
            continue
        
        with dpg.table_row(parent=TAG_DL_MISSING_HANDLER_TABLE):
            # 컬럼명
            dpg.add_text(util_funcs['format_text_for_display'](col_name, max_chars=30))
            
            # 결측치 수
            dpg.add_text(str(missing_count))
            
            # 결측치 비율
            dpg.add_text(f"{missing_count / len(df) * 100:.2f}%")
            
            # 처리 방법 선택
            current_method = "Keep Missing"
            current_fill_value = ""
            if col_name in _imputation_selections:
                current_method = _convert_method_tag_to_display(
                    _imputation_selections[col_name][0]
                )
                current_fill_value = _imputation_selections[col_name][1] or ""
            
            method_combo_tag = f"impute_method_{col_name}"
            fill_input_tag = f"impute_value_{col_name}"
            
            dpg.add_combo(items=imputation_methods, default_value=current_method,
                         tag=method_combo_tag, width=-1,
                         callback=lambda s, a, u=(col_name, fill_input_tag): 
                             _update_imputation_selection(u[0], a, u[1]))
            
            # 커스텀 값 입력
            dpg.add_input_text(tag=fill_input_tag, default_value=current_fill_value,
                             width=-1,
                             callback=lambda s, a, u=(col_name, method_combo_tag):
                                 _update_imputation_value(u[0], a, u[1]))

def _convert_method_tag_to_display(method_tag: str) -> str:
    """메서드 태그를 표시용 텍스트로 변환"""
    mapping = {
        "keep_missing": "Keep Missing",
        "drop_rows": "Drop Rows with Missing",
        "fill_mean": "Fill with Mean",
        "fill_median": "Fill with Median",
        "fill_mode": "Fill with Mode",
        "fill_custom": "Fill with Custom Value",
        "as_category_missing": "As 'Missing' Category"
    }
    return mapping.get(method_tag, "Keep Missing")

def _convert_display_to_method_tag(display_text: str) -> str:
    """표시용 텍스트를 메서드 태그로 변환"""
    mapping = {
        "Keep Missing": "keep_missing",
        "Drop Rows with Missing": "drop_rows",
        "Fill with Mean": "fill_mean",
        "Fill with Median": "fill_median",
        "Fill with Mode": "fill_mode",
        "Fill with Custom Value": "fill_custom",
        "As 'Missing' Category": "as_category_missing"
    }
    return mapping.get(display_text, "keep_missing")

def _update_imputation_selection(col_name: str, method_display: str, 
                                fill_input_tag: str):
    """결측치 처리 선택 업데이트"""
    method_tag = _convert_display_to_method_tag(method_display)
    fill_value = ""
    if dpg.does_item_exist(fill_input_tag):
        fill_value = dpg.get_value(fill_input_tag)
    _imputation_selections[col_name] = (method_tag, fill_value)

def _update_imputation_value(col_name: str, fill_value: str, 
                            method_combo_tag: str):
    """결측치 처리 값 업데이트"""
    method_display = "Keep Missing"
    if dpg.does_item_exist(method_combo_tag):
        method_display = dpg.get_value(method_combo_tag)
    method_tag = _convert_display_to_method_tag(method_display)
    _imputation_selections[col_name] = (method_tag, fill_value)

def update_ui(current_df: pd.DataFrame, original_df: pd.DataFrame, 
             util_funcs: dict, file_path: str = None):
    """UI 업데이트"""
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_DL_GROUP):
        return
    
    # 파일 요약 업데이트
    if dpg.does_item_exist(TAG_DL_FILE_SUMMARY_TEXT):
        if file_path:
            dpg.set_value(TAG_DL_FILE_SUMMARY_TEXT, f"File: {file_path}")
        elif original_df is None:
            dpg.set_value(TAG_DL_FILE_SUMMARY_TEXT, "No data loaded.")
    
    # 작업할 DataFrame 결정
    df = current_df if current_df is not None else original_df
    
    # Shape 업데이트
    if dpg.does_item_exist(TAG_DL_SHAPE_TEXT):
        shape_text = f"Shape: {df.shape}" if df is not None else "Shape: N/A (No data)"
        dpg.set_value(TAG_DL_SHAPE_TEXT, shape_text)
    
    # 테이블 업데이트
    create_table_func = util_funcs.get('create_table_with_data', lambda *args, **kwargs: None)
    
    if df is not None:
        # Info 테이블
        info_data = {
            "Column Name": df.columns.astype(str),
            "Original Dtype": [str(dtype) for dtype in df.dtypes],
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        }
        info_df = pd.DataFrame(info_data)
        create_table_func(TAG_DL_INFO_TABLE, info_df, parent_df_for_widths=info_df)
        
        # Describe 테이블
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            desc_df = numeric_df.describe().reset_index().rename(columns={'index': 'Statistic'})
            create_table_func(TAG_DL_DESCRIBE_TABLE, desc_df, 
                            utils_format_numeric=True, parent_df_for_widths=desc_df)
        else:
            create_table_func(TAG_DL_DESCRIBE_TABLE, 
                            pd.DataFrame({"Info": ["No numeric columns found."]}))
        
        # Head 테이블
        create_table_func(TAG_DL_HEAD_TABLE, df.head(), parent_df_for_widths=df)
    else:
        # 빈 테이블
        create_table_func(TAG_DL_INFO_TABLE, None)
        create_table_func(TAG_DL_DESCRIBE_TABLE, None)
        create_table_func(TAG_DL_HEAD_TABLE, None)
    
    # 타입 편집 및 결측값 처리 테이블 업데이트
    temp_callbacks = {
        'get_original_df': lambda: original_df,
        'get_df_after_step1': lambda: current_df,
        'get_util_funcs': lambda: util_funcs
    }
    _populate_type_editor_table(temp_callbacks, infer_all=False)
    _populate_missing_handler_table(temp_callbacks)

def process_newly_loaded_data(original_df: pd.DataFrame, main_callbacks: dict):
    """새로 로드된 데이터 처리"""
    if original_df is None:
        return
    
    print("Processing newly loaded data with default settings...")
    
    # 타입 추론 및 설정
    global _type_selections
    _type_selections.clear()
    
    for col in original_df.columns:
        inferred_type, _, is_binary = _infer_series_type(original_df[col])
        _type_selections[col] = _map_inferred_to_available_type(inferred_type, is_binary)
    
    # 처리 완료를 main_app에 알림
    main_callbacks['step1_processing_complete'](original_df.copy())

def apply_step1_settings_and_process(original_df: pd.DataFrame, 
                                   settings: dict, main_callbacks: dict):
    """저장된 설정을 적용하여 데이터 처리"""
    if original_df is None:
        return
    
    print("Applying Step 1 settings and processing data...")
    
    # 설정 복원
    global _type_selections, _imputation_selections, _custom_nan_input_value
    
    _type_selections = settings.get('type_selections', {}).copy()
    _imputation_selections = settings.get('imputation_selections', {}).copy()
    _custom_nan_input_value = settings.get('custom_nan_input_value', '')
    
    # UI 업데이트
    if dpg.does_item_exist(TAG_DL_CUSTOM_NAN_INPUT):
        dpg.set_value(TAG_DL_CUSTOM_NAN_INPUT, _custom_nan_input_value)
    
    # 데이터 처리
    df = original_df.copy()
    
    # 타입 변환 적용
    for col_name, col_type in _type_selections.items():
        if col_name in df.columns:
            try:
                df[col_name] = _convert_column_type(df[col_name], col_type, original_df)
            except Exception as e:
                print(f"Error converting '{col_name}': {e}")
    
    # 결측치 처리 적용
    for col_name, (method, fill_value) in _imputation_selections.items():
        if col_name in df.columns:
            try:
                df = _apply_imputation_method(df, col_name, method, fill_value)
            except Exception as e:
                print(f"Error treating missing values in '{col_name}': {e}")
    
    # 처리 완료를 main_app에 알림
    main_callbacks['step1_processing_complete'](df)

def reset_step1_state():
    """Step 1 상태 초기화"""
    global _type_selections, _imputation_selections, _custom_nan_input_value
    
    _type_selections.clear()
    _imputation_selections.clear()
    _custom_nan_input_value = ""
    
    if dpg.does_item_exist(TAG_DL_CUSTOM_NAN_INPUT):
        dpg.set_value(TAG_DL_CUSTOM_NAN_INPUT, "")
    
    if dpg.does_item_exist(TAG_DL_COLUMN_CONFIG_TABLE_GROUP):
        dpg.delete_item(TAG_DL_COLUMN_CONFIG_TABLE_GROUP, children_only=True)
        dpg.add_text("Load data to configure.", parent=TAG_DL_COLUMN_CONFIG_TABLE_GROUP)