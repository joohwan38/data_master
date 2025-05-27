import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np

# Global variables
current_df = None
original_df = None
step_group_tags = {}

# Constants
MIN_COL_WIDTH = 10
MAX_COL_WIDTH = 250
CELL_PADDING = 10
TARGET_DATA_CHARS = 20
ELLIPSIS = "..."

def get_safe_text_size(text: str) -> tuple[float, float]:
    text_str = str(text)
    if not text_str:
        return (0.0, 0.0)
    
    if not dpg.is_dearpygui_running():
        return (len(text_str) * 6.0, 16.0)
    
    size = dpg.get_text_size(text_str)
    return size if size != (0.0, 0.0) else (len(text_str) * 6.0, 16.0)

def format_text_for_display(text_val, max_chars=TARGET_DATA_CHARS) -> str:
    s_text = str(text_val)
    return s_text[:max_chars] + ELLIPSIS if len(s_text) > max_chars else s_text

def calculate_column_widths(df: pd.DataFrame, min_width=MIN_COL_WIDTH, max_width=MAX_COL_WIDTH) -> dict:
    if df is None or df.empty:
        return {}
    
    col_widths = {}
    for col_name in df.columns:
        display_header = format_text_for_display(str(col_name))
        max_width_px = get_safe_text_size(display_header)[0]
        
        sample = df[col_name].dropna().head(20).astype(str).apply(format_text_for_display)
        max_width_px = max(max_width_px, max(get_safe_text_size(item)[0] for item in sample))
        
        final_width = max(min_width, min(max_width_px + CELL_PADDING, max_width))
        col_widths[col_name] = int(final_width)
    
    return col_widths

def load_data_from_file(file_path: str) -> bool:
    global current_df, original_df
    try:
        current_df = pd.read_parquet(file_path)
        original_df = current_df.copy()
        print(f"Data loaded: {file_path}, shape: {current_df.shape}")
        
        if dpg.does_item_exist("data_summary_text"):
            dpg.set_value("data_summary_text", 
                         f"File: {file_path}")
        
        update_dataframe_summary_ui()
        return True
    except Exception as e:
        print(f"Data load error: {e}")
        current_df = None
        original_df = None
        if dpg.does_item_exist("data_summary_text"):
            dpg.set_value("data_summary_text", f"Error: {e}")
        update_dataframe_summary_ui()
        return False

def file_load_callback(sender, app_data):
    load_data_from_file(app_data['file_path_name'])

def create_table_with_data(table_tag: str, df: pd.DataFrame, format_numeric=False):
    dpg.delete_item(table_tag, children_only=True)
    
    if df is None or df.empty:
        dpg.add_table_column(label="Data", parent=table_tag, init_width_or_weight=300)
        with dpg.table_row(parent=table_tag):
            dpg.add_text("No data to display.")
        return
    
    col_widths = calculate_column_widths(df)
    
    for col_name in df.columns:
        dpg.add_table_column(label=str(col_name), parent=table_tag, 
                           init_width_or_weight=col_widths.get(col_name, MIN_COL_WIDTH))
    
    for row_data in df.itertuples(index=False):
        with dpg.table_row(parent=table_tag):
            for val in row_data:
                text = f"{val:.3f}" if format_numeric and isinstance(val, (float, np.floating)) else str(val)
                dpg.add_text(format_text_for_display(text))

def update_dataframe_summary_ui():
    global current_df
    
    if dpg.does_item_exist("df_summary_shape_text"):
        shape_text = f"Shape: {current_df.shape}" if current_df is not None else "Shape: N/A (No data)"
        dpg.set_value("df_summary_shape_text", shape_text)
    
    if dpg.does_item_exist("df_summary_info_table") and current_df is not None:
        info_df = pd.DataFrame({
            "Column Name": current_df.columns.astype(str),
            "Data Type": [str(dtype) for dtype in current_df.dtypes],
            "Missing Values": current_df.isnull().sum().values
        })
        create_table_with_data("df_summary_info_table", info_df)
    
    if dpg.does_item_exist("df_summary_describe_table") and current_df is not None:
        try:
            numeric_df = current_df.select_dtypes(include=np.number)
            create_table_with_data("df_summary_describe_table", 
                                 numeric_df.describe().reset_index() if not numeric_df.empty else None, 
                                 format_numeric=True)
        except Exception as e:
            print(f"Descriptive stats error: {e}")
            create_table_with_data("df_summary_describe_table", None)
    
    if dpg.does_item_exist("df_summary_head_table") and current_df is not None:
        create_table_with_data("df_summary_head_table", current_df.head())

def create_data_loading_and_overview_ui(step_name: str, parent_container_tag: str):
    step_tag = f"{step_name.lower().replace(' ', '_').replace('.', '').replace('&', 'and')}_group"
    step_group_tags[step_name] = step_tag
    
    with dpg.group(tag=step_tag, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        
        # Data Loading Section
        dpg.add_button(label="Open Parquet File", callback=lambda: dpg.show_item("file_dialog_id"))
        dpg.add_text("No data loaded.", tag="data_summary_text", wrap=500)
        dpg.add_spacer(height=10)
        
        # Data Overview Section with Sub-tabs
        dpg.add_text("--- Data Overview ---")
        dpg.add_separator()
        
        with dpg.tab_bar(tag="overview_tab_bar"):
            with dpg.tab(label="Original Data"):
                dpg.add_button(label="Refresh DataFrame Info", callback=update_dataframe_summary_ui, width=-1, height=30)
                dpg.add_text("Shape: N/A (No data)", tag="df_summary_shape_text")
                dpg.add_separator()
                
                dpg.add_text("Column Info (Type, Missing Values):")
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True,
                               tag="df_summary_info_table", scrollX=True, scrollY=True, height=250, freeze_columns=1):
                    pass
                
                dpg.add_separator()
                dpg.add_text("Descriptive Statistics (Numeric Columns):")
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True,
                               tag="df_summary_describe_table", scrollX=True, scrollY=True, height=250, freeze_columns=1):
                    pass
                
                dpg.add_separator()
                dpg.add_text("Data Head (First 5 Rows):")
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True,
                               tag="df_summary_head_table", scrollX=True, scrollY=True, height=150, freeze_columns=1):
                    pass
            
            with dpg.tab(label="Processed Data"):
                dpg.add_text("Processed data will be displayed here after preprocessing steps.")
                # Placeholder for future processed data tables
                with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit,
                               borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True,
                               tag="df_processed_table", scrollX=True, scrollY=True, height=450, freeze_columns=1):
                    dpg.add_table_column(label="Data", init_width_or_weight=300)
                    with dpg.table_row():
                        dpg.add_text("No processed data available yet.")

def create_step_ui(step_name: str, parent_container_tag: str):
    step_tag = f"{step_name.lower().replace(' ', '_').replace('.', '').replace('&', 'and')}_group"
    step_group_tags[step_name] = step_tag
    
    with dpg.group(tag=step_tag, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        
        if "2." in step_name:
            dpg.add_button(label="View Basic Info (Console Output)")
        elif "3." in step_name:
            dpg.add_text("UI for missing value analysis, duplicate checks, etc. will be displayed here.")
        else:
            dpg.add_text(f"UI for {step_name} will be configured here.")
        
        dpg.add_spacer(height=10)

def switch_step_view(sender, app_data, user_data_step_name: str):
    for step_name, group_tag in step_group_tags.items():
        if dpg.does_item_exist(group_tag):
            dpg.show_item(group_tag) if step_name == user_data_step_name else dpg.hide_item(group_tag)
            if step_name == user_data_step_name and "1. Data Loading and Overview" in step_name:
                update_dataframe_summary_ui()
    print(f"Switched to: {user_data_step_name}")

# Initialize GUI
dpg.create_context()

analysis_steps = [
    "1. Data Loading and Overview", "2. Exploratory Data Analysis (EDA)",
    "3. Data Quality Assessment", "4. Data Preprocessing", "5. Feature Engineering", "6. Modeling",
    "7. Model Evaluation", "8. Result Interpretation & Visualization", "9. Model Deployment & Monitoring",
    "10. Reporting & Documentation"
]

ui_creation_dispatch = {
    "1. Data Loading and Overview": create_data_loading_and_overview_ui
}
for step in analysis_steps:
    ui_creation_dispatch.setdefault(step, create_step_ui)

with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback,
                     id="file_dialog_id", width=700, height=400, modal=True):
    dpg.add_file_extension(".parquet")
    dpg.add_file_extension(".*")

with dpg.window(label="Data Analysis Platform", tag="main_window", width=1200, height=800):
    with dpg.group(horizontal=True):
        with dpg.child_window(width=300, tag="navigation_panel"):
            dpg.add_text("Analysis Steps", color=[255, 255, 0])
            dpg.add_separator()
            for step_name in analysis_steps:
                dpg.add_button(label=step_name, callback=switch_step_view,
                              user_data=step_name, width=-1, height=30)
        
        with dpg.child_window(tag="content_area"):
            for step_name in analysis_steps:
                ui_creation_dispatch[step_name](step_name, "content_area")
            
            if analysis_steps and (first_step := analysis_steps[0]) in step_group_tags:
                if dpg.does_item_exist(step_group_tags[first_step]):
                    dpg.show_item(step_group_tags[first_step])
                    if "1. Data Loading and Overview" in first_step:
                        update_dataframe_summary_ui()

dpg.create_viewport(title='Data Analysis Platform GUI', width=1440, height=1200)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()