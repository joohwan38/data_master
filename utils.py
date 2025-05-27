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

def get_safe_text_size(text: str) -> tuple[float, float]:
    """Safely gets text size, handling empty strings and non-running DPG."""
    text_str = str(text)
    if not text_str:
        return (0.0, 0.0)
    
    if not dpg.is_dearpygui_running():
        # Estimate size when DPG is not running (e.g., during setup)
        return (len(text_str) * 7.0, 16.0) 
    
    size = dpg.get_text_size(text_str)
    return size if size != (0.0, 0.0) else (len(text_str) * 7.0, 16.0)

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