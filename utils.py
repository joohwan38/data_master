# utils.py

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np

# Constants for text display and column width calculation
MIN_COL_WIDTH = 50  # Min width for a table column
MAX_COL_WIDTH = 300 # Max width for a table column
CELL_PADDING = 20   # Padding for cell content width
TARGET_DATA_CHARS = 25 # Target characters to display before truncating
ELLIPSIS = "..."

def _guess_target_type(df: pd.DataFrame, column_name: str,
                       step1_type_selections: dict = None) -> str:
    if not column_name or df is None or column_name not in df.columns:
        return "Continuous" # 기본값

    series = df[column_name]

    # Step 1에서 사용자가 지정한 타입을 우선 고려
    if step1_type_selections and column_name in step1_type_selections:
        s1_type = step1_type_selections[column_name] # step_01_data_loading 모듈에서 정의된 타입 문자열
        
        # "Text", "Potentially Sensitive" 계열은 Categorical
        if any(keyword in s1_type for keyword in ["Text (", "Potentially Sensitive"]):
            return "Categorical"
        
        # "Categorical", "Categorical (Binary Text)" 계열은 Categorical
        if "Categorical" in s1_type: # "Categorical" 또는 "Categorical (Binary Text)" 등
            return "Categorical"

        # "Numeric" 계열 처리
        if "Numeric" in s1_type:
            # "Numeric (Binary)" 또는 "Numeric (Binary from Text)"는 고유값 2개이므로 Categorical
            if "Binary" in s1_type: # e.g., "Numeric (Binary)", "Numeric (Binary from Text)"
                return "Categorical"
            # 그 외 "Numeric (int)", "Numeric (float)", "Numeric (from Text)", "Numeric" 등
            # 사용자가 Numeric으로 지정한 경우, 고유값이 매우 적지 않으면 (예: 5개 초과) Continuous로 간주 (Ordinal 포함)
            if series.nunique(dropna=False) > 5: # 임계값 5 (조절 가능)
                return "Continuous"
            else: # 고유값이 5개 이하인 Numeric은 Categorical (예: 0, 1, 2 클래스 레이블)
                return "Categorical"
        
        # "Datetime", "Timedelta"는 분석 목적상 Categorical로 일단 분류 (주로 범주형으로 활용되거나 feature engineering 대상)
        if "Datetime" in s1_type or "Timedelta" in s1_type:
            return "Categorical"

    # Pandas dtype 및 고유값 기반 추론 (Step 1 정보 없거나 위에서 결정되지 않았을 시)
    if pd.api.types.is_categorical_dtype(series.dtype) or \
       pd.api.types.is_bool_dtype(series.dtype):
        return "Categorical"
    if pd.api.types.is_object_dtype(series.dtype) or \
       pd.api.types.is_string_dtype(series.dtype):
        # 문자열 타입은 Categorical
        return "Categorical"
    if pd.api.types.is_numeric_dtype(series.dtype):
        # 순수 숫자형: 고유값이 적으면 (예: 10개 이하) Categorical, 아니면 Continuous
        return "Categorical" if series.nunique(dropna=False) <= 10 else "Continuous"
    if pd.api.types.is_datetime64_any_dtype(series.dtype) or \
       pd.api.types.is_timedelta64_dtype(series.dtype):
        return "Categorical"

    return "Continuous" # 모든 조건 불일치 시 기본값


def get_safe_text_size(text: str, font=None, wrap_width: float = -1.0) -> tuple[int, int]:
    """
    텍스트 크기를 안전하게 가져옵니다.
    DPG가 실행 중이 아니거나 텍스트 크기 측정에 실패할 경우, 기본 추정값을 반환합니다.
    """
    # 기본 추정 크기: 글자당 가로 8픽셀, 기본 높이 16픽셀
    # 실제 폰트와 렌더링 환경에 따라 다를 수 있으므로, 이는 오류 방지를 위한 값입니다.
    default_width = len(str(text)) * 8
    default_height = 16
    default_size = (default_width, default_height)

    if not dpg.is_dearpygui_running():
        return default_size
    
    try:
        # dpg.get_text_size는 (width, height) 튜플을 반환합니다.
        size = dpg.get_text_size(text, font=font, wrap_width=wrap_width)
        
        # 간혹 size가 None으로 반환될 경우를 대비
        if size is None or not isinstance(size, (tuple, list)) or len(size) != 2:
            return default_size
        
        # 정수형으로 변환하여 반환
        return int(size[0]), int(size[1])
    except Exception:
        # dpg.get_text_size 호출 중 예외 발생 시 (예: 폰트 문제, 초기화 미완료 등)
        return default_size

def format_text_for_display(text_val, max_chars=TARGET_DATA_CHARS) -> str:
    """Formats text for display, truncating if necessary."""
    s_text = str(text_val)
    if len(s_text) > max_chars:
        return s_text[:max_chars] + ELLIPSIS
    return s_text

def calculate_column_widths(df: pd.DataFrame, 
                            min_width=MIN_COL_WIDTH, 
                            max_width=MAX_COL_WIDTH,
                            cell_padding=CELL_PADDING,
                            max_sample_rows=20) -> dict:
    """Calculates optimal column widths based on content."""
    if df is None or df.empty:
        return {}
    
    col_widths = {}
    for col_name in df.columns:
        header_display_text = format_text_for_display(str(col_name))
        max_px_width = get_safe_text_size(header_display_text)[0]
        
        if not df[col_name].empty:
            sample = df[col_name].dropna().head(max_sample_rows)
            if not sample.empty:
                formatted_sample = sample.astype(str).apply(lambda x: format_text_for_display(x))
                max_data_px_width = 0
                if not formatted_sample.empty:
                     max_data_px_width = max(get_safe_text_size(item)[0] for item in formatted_sample)
                max_px_width = max(max_px_width, max_data_px_width)
        
        final_width = max(min_width, min(max_px_width + cell_padding, max_width))
        col_widths[col_name] = int(final_width)
    
    return col_widths

def create_table_with_data(table_tag: str, df: pd.DataFrame, 
                           utils_format_numeric=False, parent_df_for_widths: pd.DataFrame = None):
    """
    Clears and populates a DPG table with data from a Pandas DataFrame.
    Uses parent_df_for_widths to calculate column widths if provided, otherwise uses df.
    """
    if not dpg.does_item_exist(table_tag):
        print(f"Error: Table with tag '{table_tag}' does not exist.")
        return
        
    dpg.delete_item(table_tag, children_only=True)
    
    if df is None or df.empty:
        dpg.add_table_column(label="Info", parent=table_tag, init_width_or_weight=300)
        with dpg.table_row(parent=table_tag):
            dpg.add_text("No data to display." if df is None else "DataFrame is empty.")
        return
    
    df_for_widths = parent_df_for_widths if parent_df_for_widths is not None and not parent_df_for_widths.empty else df
    col_widths = calculate_column_widths(df_for_widths)
    
    for col_name in df.columns: 
        dpg.add_table_column(label=str(col_name), 
                           parent=table_tag, 
                           init_width_or_weight=col_widths.get(str(col_name), MIN_COL_WIDTH))
    
    for row_idx in range(len(df)): # Use integer-based indexing for safety with mixed types
        with dpg.table_row(parent=table_tag):
            for col_name in df.columns:
                val = df.iat[row_idx, df.columns.get_loc(col_name)] # Use iat for positional access
                text_to_display = ""
                if pd.isna(val):
                    text_to_display = "NaN"
                elif utils_format_numeric and isinstance(val, (float, np.floating)):
                    text_to_display = f"{val:.3f}"
                else:
                    text_to_display = str(val)
                
                dpg.add_text(format_text_for_display(text_to_display))