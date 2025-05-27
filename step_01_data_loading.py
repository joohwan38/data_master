# step_01_data_loading.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from collections import Counter

# --- 기존 TAG 정의에 새로운 UI 요소들 추가 ---
TAG_DL_GROUP = "step1_data_loading_group"
TAG_DL_FILE_SUMMARY_TEXT = "step1_file_summary_text"
TAG_DL_SHAPE_TEXT = "step1_df_summary_shape_text"
TAG_DL_INFO_TABLE = "step1_df_summary_info_table"
TAG_DL_DESCRIBE_TABLE = "step1_df_summary_describe_table"
TAG_DL_HEAD_TABLE = "step1_df_summary_head_table"
TAG_DL_PROCESSED_TABLE = "step1_df_processed_table"
TAG_DL_OVERVIEW_TAB_BAR = "step1_overview_tab_bar"

# 새로운 UI 요소 태그
TAG_DL_RESET_BUTTON = "step1_reset_data_button"
TAG_DL_TYPE_EDITOR_TABLE = "step1_type_editor_table"
TAG_DL_APPLY_TYPE_CHANGES_BUTTON = "step1_apply_type_changes_button"
TAG_DL_INFER_TYPES_BUTTON = "step1_infer_types_button"

TAG_DL_CUSTOM_NAN_INPUT = "step1_custom_nan_input"
TAG_DL_APPLY_CUSTOM_NAN_BUTTON = "step1_apply_custom_nan_button"
TAG_DL_MISSING_HANDLER_TABLE = "step1_missing_handler_table"
TAG_DL_APPLY_MISSING_TREATMENTS_BUTTON = "step1_apply_missing_treatments_button"

# --- 전역 변수 (모듈 내에서 타입/결측치 선택 상태 임시 저장용) ---
# 이 값들은 "적용" 버튼 클릭 시 current_df에 반영되고, UI 업데이트 시 다시 로드됩니다.
_type_selections = {} # {col_name: selected_type_tag}
_imputation_selections = {} # {col_name: (method_tag, value_tag or None)}
_custom_nan_input_value = ""


# --- Helper Functions for Type Inference and Conversion ---
# step_01_data_loading.py 내의 _infer_series_type 함수 수정

def _infer_series_type(series: pd.Series):
    """
    Pandas Series를 분석하여 데이터 타입을 추론하고, 추가 정보(힌트)와 이진 숫자형 여부를 반환.
    반환: (추론타입_문자열, 힌트_문자열_또는_None, 이진숫자형_bool)
    추론타입 종류: "Numeric", "Numeric (Binary)", "Datetime", "Timedelta",
                   "Categorical", "Categorical (Binary Text)", "Text (ID/Code)", "Text (Long/Free)",
                   "Potentially Sensitive (Review Needed)", "Unknown", "All Missing"
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

    # 0. 민감 정보 패턴 (컬럼명 기반의 매우 간단한 휴리스틱 - 주의 필요!)
    sensitive_keywords = ['name', 'email', 'phone', 'ssn', ' 주민', ' 전번', ' 이멜', ' 이름']
    if any(keyword in series_name_lower for keyword in sensitive_keywords):
        if series_sample.astype(str).str.contains('@').any():
             return "Potentially Sensitive (Email?)", "Review for PII", False
        return "Potentially Sensitive (Name/Phone?)", "Review for PII", False

    # 1. 0/1 값 처리 (기본을 수치형으로) - Boolean 타입 추론 로직은 여기서 제외됨
    is_binary_numeric_flag = False
    if current_dtype_kind in 'iuf': 
        str_unique_values = set(series_valid.astype(str).unique())
        if str_unique_values == {'0', '1'} or \
           str_unique_values == {'0.0', '1.0'} or \
           str_unique_values == {'0.0', '1'} or \
           str_unique_values == {'0', '1.0'}:
            is_binary_numeric_flag = True
            return "Numeric (Binary)", "Can be Categorical", is_binary_numeric_flag

    # 2. 현재 Dtype이 이미 숫자(정수, 부동소수점, 복소수)인 경우
    if current_dtype_kind in 'iufc':
        return "Numeric", None, is_binary_numeric_flag


    # 3. Dtype이 Object(문자열 등) 또는 다른 타입인 경우
    # 3-1. 날짜/시간형 시도
    if current_dtype_kind == 'O' or pd.api.types.is_datetime64_any_dtype(series.dtype):
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
             return "Datetime", "Original Dtype was Datetime", False

        date_formats_to_try = ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y",
                               "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                               "%d-%m-%Y", "%Y%m%d%H%M%S", "%Y.%m.%d"]
        best_date_format = None
        max_conversion_rate_date = 0
        
        for fmt in date_formats_to_try:
            try:
                converted_dates = pd.to_datetime(series_sample, format=fmt, errors='coerce')
                conversion_rate = converted_dates.notna().sum() / len(series_sample)
                if conversion_rate > max_conversion_rate_date and conversion_rate > 0.85:
                    max_conversion_rate_date = conversion_rate
                    best_date_format = fmt
            except (ValueError, TypeError):
                continue
        if best_date_format:
            return "Datetime", f"Format ~{best_date_format}", False

    # 3-2. Timedelta 시도
    if current_dtype_kind in 'iuf' or current_dtype_kind == 'O':
        try:
            if current_dtype_kind == 'O':
                converted_td_sample = pd.to_timedelta(series_sample, errors='coerce')
                if converted_td_sample.notna().sum() / len(series_sample) > 0.8:
                    return "Timedelta", "From Text", False
            timedelta_keywords = ["duration", "period", "interval", "lead_time", "경과", "기간"]
            if any(keyword in series_name_lower for keyword in timedelta_keywords) and current_dtype_kind in 'iuf':
                return "Timedelta", "From Numeric (unit needed)", False
        except Exception:
            pass

    # 3-3. Object 타입의 숫자 변환 시도
    if current_dtype_kind == 'O':
        try:
            numeric_converted_sample = pd.to_numeric(series_sample.astype(str), errors='coerce')
            conversion_rate_numeric = numeric_converted_sample.notna().sum() / len(series_sample)
            if conversion_rate_numeric > 0.95:
                unique_vals_after_conversion = numeric_converted_sample.dropna().unique()
                str_unique_vals_after_conversion = set(map(str, unique_vals_after_conversion))
                if str_unique_vals_after_conversion == {'0', '1'} or \
                   str_unique_vals_after_conversion == {'0.0', '1.0'} or \
                   str_unique_vals_after_conversion == {'0.0', '1'} or \
                   str_unique_vals_after_conversion == {'0', '1.0'}:
                    is_binary_numeric_flag = True # 이 플래그는 Numeric (Binary from Text)에만 해당
                    return "Numeric (Binary from Text)", "Can be Categorical", is_binary_numeric_flag
                return "Numeric (from Text)", None, False # 일반 숫자 변환은 is_binary_numeric_flag가 False
        except Exception:
            pass

    # 4. 범주형 또는 텍스트형으로 최종 판단
    num_unique_values = series_valid.nunique()
    len_valid_series = len(series_valid)
    avg_str_len = series_valid.astype(str).str.len().mean() if len_valid_series > 0 else 0

    # 이진값 텍스트 (예: "A", "B" 또는 "True", "False" 문자열 - Boolean 타입 추론이 제거되었으므로 여기서 처리)
    if num_unique_values == 2:
        return "Categorical (Binary Text)", None, False

    if num_unique_values / len_valid_series > 0.8 and num_unique_values > 1000 and avg_str_len < 50:
        return "Text (ID/Code)", None, False
    
    if num_unique_values < max(30, len_valid_series * 0.05):
        return "Categorical", None, False
    
    if avg_str_len > 100 :
        return "Text (Long/Free)", None, False
        
    return "Text (General)", None, False


def _apply_type_changes(main_callbacks: dict):
    current_df = main_callbacks['get_current_df']()
    if current_df is None: return
    print("Applying type changes...")

    for col_name, new_type_info in _type_selections.items():
        if col_name not in current_df.columns: continue
        
        original_series = current_df[col_name].copy()
        new_type_str = new_type_info
        
        try:
            print(f" Column '{col_name}': Changing type to '{new_type_str}'...")
            if new_type_str == "Numeric (int)":
                current_df[col_name] = pd.to_numeric(current_df[col_name], errors='coerce').astype('Int64')
            elif new_type_str == "Numeric (float)":
                current_df[col_name] = pd.to_numeric(current_df[col_name], errors='coerce').astype(float)
            elif new_type_str == "Categorical" or new_type_str.startswith("Categorical ("):
                current_df[col_name] = current_df[col_name].astype('category')
            # Boolean 타입 변환 로직 제거됨
            elif new_type_str.startswith("Datetime"):
                current_df[col_name] = pd.to_datetime(current_df[col_name], errors='coerce')
            elif new_type_str.startswith("Timedelta"):
                current_df[col_name] = pd.to_timedelta(current_df[col_name], errors='coerce')
            elif new_type_str.startswith("Text (") or new_type_str == "Original Text":
                current_df[col_name] = current_df[col_name].astype(pd.StringDtype())
            elif new_type_str == "Original":
                original_dtype = main_callbacks['get_original_df']()[col_name].dtype
                current_df[col_name] = current_df[col_name].astype(original_dtype)
            elif new_type_str == "Potentially Sensitive (Review Needed)" or new_type_str.startswith("Potentially Sensitive ("):
                print(f" Column '{col_name}' marked as Potentially Sensitive. No type change applied by default. Convert to Text (string) if needed for masking.")
                if current_df[col_name].dtype == 'object':
                     current_df[col_name] = current_df[col_name].astype(pd.StringDtype())

            print(f" Column '{col_name}' type change successful. New Dtype: {current_df[col_name].dtype}, Missing: {current_df[col_name].isnull().sum()}")
        except Exception as e:
            print(f"Error changing type for column '{col_name}' to '{new_type_str}': {e}. Reverting to original series for this column.")
            current_df[col_name] = original_series
            import traceback
            traceback.print_exc()

    main_callbacks['trigger_all_module_updates']()
    print("Type changes application process finished.")


def _apply_custom_nans(main_callbacks: dict, custom_nan_str: str):
    """사용자가 입력한 문자열들을 NaN으로 변환합니다."""
    current_df = main_callbacks['get_current_df']()
    if current_df is None:
        print("데이터가 로드되지 않아 Custom NaN을 적용할 수 없습니다.")
        return
    
    if not custom_nan_str.strip():
        print("Custom NaN으로 처리할 문자열이 입력되지 않았습니다.")
        return

    nan_values_to_replace = [s.strip() for s in custom_nan_str.split(',')]
    print(f"Custom NaN 적용 시작: {nan_values_to_replace}")
    
    for col in current_df.columns:
        # object 타입 컬럼에 대해서만 문자열 대체를 시도 (숫자형 컬럼은 숫자형 결측값으로 처리)
        if current_df[col].dtype == 'object':
            current_df[col].replace(nan_values_to_replace, np.nan, inplace=True)
    
    main_callbacks['trigger_all_module_updates']()
    print("Custom NaN 적용 완료.")

def _apply_missing_value_treatments(main_callbacks:dict):
    """_imputation_selections에 저장된 사용자 선택 결측치 처리 방법을 current_df에 적용합니다."""
    current_df = main_callbacks['get_current_df']()
    if current_df is None:
        print("데이터가 로드되지 않아 결측치 처리를 적용할 수 없습니다.")
        return

    print("결측치 처리 적용 시작...")
    for col_name, selection in _imputation_selections.items():
        if col_name not in current_df.columns or not selection:
            continue

        method, fill_value_str = selection # method는 태그값, fill_value_str는 입력된 문자열
        
        try:
            print(f" 컬럼 '{col_name}' 결측치 처리 시도 (방법: {method})...")
            if method == "drop_rows":
                current_df.dropna(subset=[col_name], inplace=True)
            elif method == "fill_mean":
                if pd.api.types.is_numeric_dtype(current_df[col_name]):
                    current_df[col_name].fillna(current_df[col_name].mean(), inplace=True)
            elif method == "fill_median":
                if pd.api.types.is_numeric_dtype(current_df[col_name]):
                    current_df[col_name].fillna(current_df[col_name].median(), inplace=True)
            elif method == "fill_mode":
                current_df[col_name].fillna(current_df[col_name].mode().iloc[0] if not current_df[col_name].mode().empty else np.nan, inplace=True)
            elif method == "fill_custom":
                # fill_value_str을 해당 컬럼 타입에 맞게 변환 시도 필요
                try:
                    # TODO: 컬럼 타입에 따른 적절한 변환 로직 필요
                    converted_value = float(fill_value_str) # 임시로 float 변환
                    current_df[col_name].fillna(converted_value, inplace=True)
                except ValueError:
                    current_df[col_name].fillna(fill_value_str, inplace=True) # 문자열로 채우기
            elif method == "as_category_missing":
                if pd.api.types.is_categorical_dtype(current_df[col_name]):
                    if "Missing" not in current_df[col_name].cat.categories:
                        current_df[col_name] = current_df[col_name].cat.add_categories("Missing")
                else: # 범주형이 아니면 먼저 범주형으로 변환
                    current_df[col_name] = current_df[col_name].astype('category')
                    if "Missing" not in current_df[col_name].cat.categories:
                         current_df[col_name] = current_df[col_name].cat.add_categories("Missing")
                current_df[col_name].fillna("Missing", inplace=True)
            print(f" 컬럼 '{col_name}' 결측치 처리 완료.")
        except Exception as e:
            print(f"오류: 컬럼 '{col_name}' 결측치 처리 중 오류: {e}")
    
    _imputation_selections.clear() # 적용 후 선택 초기화
    main_callbacks['trigger_all_module_updates']()
    print("결측치 처리 적용 완료.")

# --- Main UI Creation and Update Functions ---
def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """기존 UI 생성 함수 확장: 데이터 타입 편집 및 결측치 처리 탭 추가."""
    main_callbacks['register_step_group_tag'](step_name, TAG_DL_GROUP)
    
    with dpg.group(tag=TAG_DL_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Open Parquet File", callback=main_callbacks['show_file_dialog'])
            dpg.add_button(label="Reset to Original Data", tag=TAG_DL_RESET_BUTTON,
                           callback=lambda: (
                               main_callbacks['reset_current_df_to_original'](), # main_app.py에 이 함수 필요
                               main_callbacks['trigger_all_module_updates']()
                           ))
        dpg.add_text("No data loaded.", tag=TAG_DL_FILE_SUMMARY_TEXT, wrap=700)
        dpg.add_spacer(height=10)
        
        dpg.add_text("--- Data Details & Preprocessing ---")
        dpg.add_separator()
        
        with dpg.tab_bar(tag=TAG_DL_OVERVIEW_TAB_BAR):
            # --- 1. 원본 데이터 탭 (기존 내용 유지) ---
            with dpg.tab(label="Data Summary"):
                dpg.add_button(label="Refresh DataFrame Info", width=-1, height=30,
                               callback=lambda: main_callbacks['trigger_module_update'](step_name))
                dpg.add_text("Shape: N/A (No data)", tag=TAG_DL_SHAPE_TEXT)
                dpg.add_separator()
                dpg.add_text("Column Info (Type, Missing, Unique):")
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True,
                               tag=TAG_DL_INFO_TABLE, scrollX=True, scrollY=True, height=200, freeze_columns=0): pass
                dpg.add_separator()
                dpg.add_text("Descriptive Statistics (Numeric Columns):")
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True,
                               tag=TAG_DL_DESCRIBE_TABLE, scrollX=True, scrollY=True, height=200, freeze_columns=0): pass
                dpg.add_separator()
                dpg.add_text("Data Head (First 5 Rows):")
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True,
                               tag=TAG_DL_HEAD_TABLE, scrollX=True, scrollY=True, height=150, freeze_columns=0): pass
            
            # --- 2. 변수 타입 편집 탭 ---
            with dpg.tab(label="Variable Type Editor"):
                dpg.add_text("Infer and set data types for analysis.")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Infer All Types (Rule-based)", tag=TAG_DL_INFER_TYPES_BUTTON,
                                   callback=lambda: _populate_type_editor_table(main_callbacks, infer_all=True))
                    dpg.add_button(label="Apply Type Changes", tag=TAG_DL_APPLY_TYPE_CHANGES_BUTTON,
                                   callback=lambda: _apply_type_changes(main_callbacks))
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                               tag=TAG_DL_TYPE_EDITOR_TABLE, scrollX=True, scrollY=True, height=400,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                    # 헤더는 _populate_type_editor_table 에서 동적으로 추가
                    pass

            # --- 3. 결측값 처리 탭 ---
            with dpg.tab(label="Missing Value Handler"):
                dpg.add_text("Define and handle missing values.")
                with dpg.group(horizontal=True):
                    global _custom_nan_input_value # dpg.add_input_text 콜백에서 사용하기 위함
                    dpg.add_text("Custom NaN strings (comma-separated):")
                    dpg.add_input_text(tag=TAG_DL_CUSTOM_NAN_INPUT, width=300, default_value=_custom_nan_input_value,
                                       callback=lambda sender, app_data: globals().update(_custom_nan_input_value=app_data))
                    dpg.add_button(label="Convert Custom NaNs to np.nan", tag=TAG_DL_APPLY_CUSTOM_NAN_BUTTON,
                                   callback=lambda: _apply_custom_nans(main_callbacks, _custom_nan_input_value))
                
                dpg.add_button(label="Apply Selected Missing Value Treatments", tag=TAG_DL_APPLY_MISSING_TREATMENTS_BUTTON,
                               callback=lambda: _apply_missing_value_treatments(main_callbacks))
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                               tag=TAG_DL_MISSING_HANDLER_TABLE, scrollX=True, scrollY=True, height=400,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                    # 헤더는 _populate_missing_handler_table 에서 동적으로 추가
                    pass

    main_callbacks['register_module_updater'](step_name, update_ui)
    update_ui(main_callbacks['get_current_df'](), 
              main_callbacks['get_original_df'](), 
              main_callbacks['get_util_funcs'](),
              main_callbacks['get_loaded_file_path']())


def _populate_type_editor_table(main_callbacks: dict, infer_all=False):
    """변수 타입 편집 테이블을 데이터프레임 정보로 채웁니다."""
    global _type_selections
    current_df = main_callbacks['get_current_df']()
    util_funcs = main_callbacks['get_util_funcs']()
    
    if not dpg.does_item_exist(TAG_DL_TYPE_EDITOR_TABLE): return
    dpg.delete_item(TAG_DL_TYPE_EDITOR_TABLE, children_only=True)

    if current_df is None:
        dpg.add_table_column(label="Info", parent=TAG_DL_TYPE_EDITOR_TABLE)
        with dpg.table_row(parent=TAG_DL_TYPE_EDITOR_TABLE):
            dpg.add_text("Load data to edit types.")
        return

    dpg.add_table_column(label="Column Name", parent=TAG_DL_TYPE_EDITOR_TABLE)
    dpg.add_table_column(label="Original Dtype", parent=TAG_DL_TYPE_EDITOR_TABLE)
    dpg.add_table_column(label="Inferred Type", parent=TAG_DL_TYPE_EDITOR_TABLE)
    dpg.add_table_column(label="Selected Type", parent=TAG_DL_TYPE_EDITOR_TABLE, width=200)
    dpg.add_table_column(label="Unique Count", parent=TAG_DL_TYPE_EDITOR_TABLE)
    dpg.add_table_column(label="Sample Values", parent=TAG_DL_TYPE_EDITOR_TABLE, width=300)

    available_types = ["Original", "Numeric (int)", "Numeric (float)", 
                       "Categorical", # "Categorical (Ordered)"는 UI 지원 시 추가
                       "Datetime", "Timedelta", 
                       "Text (ID/Code)", "Text (Long/Free)", "Text (General)", # pd.StringDtype()으로 통합 가능
                       "Potentially Sensitive (Review Needed)"]

    for col_name in current_df.columns:
        with dpg.table_row(parent=TAG_DL_TYPE_EDITOR_TABLE):
            dpg.add_text(util_funcs['format_text_for_display'](col_name, max_chars=30))
            dpg.add_text(str(current_df[col_name].dtype))
            
            # _infer_series_type는 이제 3개의 값을 반환합니다.
            inferred_type_str, inferred_hint, is_col_binary_numeric = _infer_series_type(current_df[col_name])
            
            inferred_type_display = inferred_type_str
            if inferred_hint: 
                inferred_type_display = f"{inferred_type_str} ({inferred_hint})"
            dpg.add_text(inferred_type_display)

            current_selection = _type_selections.get(col_name)
            if infer_all and not current_selection: 
                if inferred_type_str.startswith("Numeric"):
                    current_selection = "Numeric (int)" if is_col_binary_numeric else "Numeric (float)" 
                elif inferred_type_str == "Datetime":
                    current_selection = "Datetime"
                elif inferred_type_str.startswith("Categorical"): # "Categorical", "Categorical (Binary Text)" 등
                    current_selection = "Categorical" # 대표 타입으로 설정
                # Boolean 관련 로직은 이미 _infer_series_type에서 Numeric(Binary) 등으로 처리되므로 별도 조건 불필요
                elif inferred_type_str.startswith("Text (ID/Code)"):
                    current_selection = "Text (ID/Code)"
                elif inferred_type_str.startswith("Text (Long/Free)"):
                    current_selection = "Text (Long/Free)"
                elif inferred_type_str.startswith("Text"): # "Text (General)" 등
                    current_selection = "Text (General)"
                elif inferred_type_str.startswith("Timedelta"):
                    current_selection = "Timedelta"
                elif inferred_type_str.startswith("Potentially Sensitive"):
                    current_selection = "Potentially Sensitive (Review Needed)"
                else: # Unknown, All Missing 등
                    current_selection = "Original" # 기본값으로 Original 또는 Text (General) 고려
                _type_selections[col_name] = current_selection
            elif not current_selection:
                 current_selection = "Original"

            combo_tag = f"type_combo_{col_name}"
            dpg.add_combo(items=available_types, default_value=current_selection, tag=combo_tag, width=-1,
                          user_data=col_name, callback=lambda sender, app_data, user_data: _type_selections.update({user_data: app_data}))
            
            dpg.add_text(str(current_df[col_name].nunique()))
            sample_vals = current_df[col_name].dropna().head(3).astype(str).tolist()
            dpg.add_text(util_funcs['format_text_for_display'](", ".join(sample_vals), max_chars=50))


def _populate_missing_handler_table(main_callbacks: dict):
    """결측값 처리 테이블을 데이터프레임 정보로 채웁니다."""
    global _imputation_selections
    current_df = main_callbacks['get_current_df']()
    util_funcs = main_callbacks['get_util_funcs']()

    if not dpg.does_item_exist(TAG_DL_MISSING_HANDLER_TABLE): return
    dpg.delete_item(TAG_DL_MISSING_HANDLER_TABLE, children_only=True)

    if current_df is None:
        dpg.add_table_column(label="Info", parent=TAG_DL_MISSING_HANDLER_TABLE)
        with dpg.table_row(parent=TAG_DL_MISSING_HANDLER_TABLE):
            dpg.add_text("Load data to handle missing values.")
        return

    dpg.add_table_column(label="Column Name", parent=TAG_DL_MISSING_HANDLER_TABLE)
    dpg.add_table_column(label="Missing Count", parent=TAG_DL_MISSING_HANDLER_TABLE)
    dpg.add_table_column(label="Missing %", parent=TAG_DL_MISSING_HANDLER_TABLE)
    dpg.add_table_column(label="Imputation Method", parent=TAG_DL_MISSING_HANDLER_TABLE, width=200)
    dpg.add_table_column(label="Custom Fill Value", parent=TAG_DL_MISSING_HANDLER_TABLE, width=150)

    imputation_methods = ["Keep Missing", "Drop Rows with Missing", "Fill with Mean", 
                          "Fill with Median", "Fill with Mode", "Fill with Custom Value", 
                          "As 'Missing' Category"]

    for col_name in current_df.columns:
        missing_count = current_df[col_name].isnull().sum()
        if missing_count == 0: continue # 결측치 없는 컬럼은 스킵 (선택사항)

        with dpg.table_row(parent=TAG_DL_MISSING_HANDLER_TABLE):
            dpg.add_text(util_funcs['format_text_for_display'](col_name, max_chars=30))
            dpg.add_text(str(missing_count))
            dpg.add_text(f"{missing_count / len(current_df) * 100:.2f}%")

            # 이전에 선택된 값으로 초기화
            current_method_selection, current_fill_value = "Keep Missing", ""
            if col_name in _imputation_selections:
                current_method_selection = _imputation_selections[col_name][0]
                current_fill_value = _imputation_selections[col_name][1] if _imputation_selections[col_name][1] is not None else ""


            method_combo_tag = f"impute_method_combo_{col_name}"
            fill_value_input_tag = f"impute_value_input_{col_name}"

            # 콜백에서 현재 컬럼명, 메소드 콤보 태그, 값 입력 태그를 알아야 함
            callback_user_data = {"col": col_name, "m_tag": method_combo_tag, "v_tag": fill_value_input_tag}

            dpg.add_combo(items=imputation_methods, default_value=current_method_selection, 
                          tag=method_combo_tag, width=-1, user_data=callback_user_data,
                          callback=lambda s, a, u: _imputation_selections.update(
                              {u["col"]: (a, dpg.get_value(u["v_tag"]) if dpg.does_item_exist(u["v_tag"]) else "")}
                          ))
            
            dpg.add_input_text(tag=fill_value_input_tag, default_value=current_fill_value, width=-1, user_data=callback_user_data,
                               callback=lambda s, a, u: _imputation_selections.update(
                                   {u["col"]: (dpg.get_value(u["m_tag"]), a)}
                               ))


def update_ui(current_df: pd.DataFrame, original_df: pd.DataFrame, util_funcs: dict, file_path: str = None):
    """기존 UI 업데이트 함수 확장: 새로운 탭들의 테이블도 업데이트."""
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_DL_GROUP):
        return

    # --- 1. 파일 요약 및 원본 데이터 탭 업데이트 (기존 로직) ---
    create_table_func = util_funcs.get('create_table_with_data', lambda *args, **kwargs: None)
    if dpg.does_item_exist(TAG_DL_FILE_SUMMARY_TEXT):
        if file_path: dpg.set_value(TAG_DL_FILE_SUMMARY_TEXT, f"File: {file_path}")
        elif current_df is None: dpg.set_value(TAG_DL_FILE_SUMMARY_TEXT, "No data loaded.")
    if dpg.does_item_exist(TAG_DL_SHAPE_TEXT):
        dpg.set_value(TAG_DL_SHAPE_TEXT, f"Shape: {current_df.shape}" if current_df is not None else "Shape: N/A (No data)")
    
    if current_df is not None:
        info_df_data = {
            "Column Name": current_df.columns.astype(str),
            "Original Dtype": [str(dtype) for dtype in current_df.dtypes],
            "Missing Values": current_df.isnull().sum().values,
            "Unique Values": current_df.nunique().values
        }
        info_df = pd.DataFrame(info_df_data)
        create_table_func(TAG_DL_INFO_TABLE, info_df, parent_df_for_widths=info_df)
        
        numeric_df = current_df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            descriptive_stats_df = numeric_df.describe().reset_index().rename(columns={'index': 'Statistic'})
            create_table_func(TAG_DL_DESCRIBE_TABLE, descriptive_stats_df, utils_format_numeric=True, parent_df_for_widths=descriptive_stats_df)
        else:
            create_table_func(TAG_DL_DESCRIBE_TABLE, pd.DataFrame({"Info": ["No numeric columns found."]}))
        create_table_func(TAG_DL_HEAD_TABLE, current_df.head(), parent_df_for_widths=current_df)
    else:
        create_table_func(TAG_DL_INFO_TABLE, None)
        create_table_func(TAG_DL_DESCRIBE_TABLE, None)
        create_table_func(TAG_DL_HEAD_TABLE, None)

    # --- 2. 변수 타입 편집 탭 테이블 업데이트 ---
    # main_callbacks를 여기서 직접 구성하거나, update_ui 호출 시 main_callbacks 자체를 넘겨받아야 함.
    # 지금은 main_callbacks가 없으므로, _populate_type_editor_table 호출을 위한 임시 main_callbacks 생성
    temp_main_callbacks_for_type_editor = {
        'get_current_df': lambda: current_df,
        'get_original_df': lambda: original_df, # 원본 Dtype 복원 등에 필요할 수 있음
        'get_util_funcs': lambda: util_funcs
        # 'trigger_all_module_updates' 등은 여기서 직접 호출하지 않음
    }
    _populate_type_editor_table(temp_main_callbacks_for_type_editor, infer_all=False) # UI 로드 시에는 자동추론 결과만 보여줌

    # --- 3. 결측값 처리 탭 테이블 업데이트 ---
    temp_main_callbacks_for_missing_handler = {
         'get_current_df': lambda: current_df,
         'get_util_funcs': lambda: util_funcs
    }
    _populate_missing_handler_table(temp_main_callbacks_for_missing_handler)