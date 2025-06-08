# << FINAL & STABLE >> : step_08_derivation.py 파일의 전체 내용을 아래 코드로 교체합니다.

import dearpygui.dearpygui as dpg
import pandas as pd
from typing import Dict, Optional, Any, List
import traceback

# --- DPG Tags ---
TAG_S8_GROUP = "step8_derivation_group"
TAG_S8_VAR_LIST_WINDOW = "step8_var_list_window"
TAG_S8_VAR_SEARCH_INPUT = "step8_var_search_input"
TAG_S8_SHELF_INDEX_CW = "step8_shelf_index_child_window"
TAG_S8_SHELF_COLUMNS_CW = "step8_shelf_columns_child_window"
TAG_S8_SHELF_VALUES_CW = "step8_shelf_values_child_window"
TAG_S8_SHELF_INDEX_GROUP = "step8_shelf_index_group"
TAG_S8_SHELF_COLUMNS_GROUP = "step8_shelf_columns_group"
TAG_S8_SHELF_VALUES_GROUP = "step8_shelf_values_group"
TAG_S8_PREVIEW_TABLE = "step8_preview_table"
TAG_S8_PREVIEW_TEXT = "step8_preview_text"
TAG_S8_OUTPUT_NAME_INPUT = "step8_output_name_input"
TAG_S8_PIVOT_FILLNA_CHECK = "step8_pivot_fillna_check"
TAG_S8_PIVOT_MARGINS_CHECK = "step8_pivot_margins_check"
TAG_S8_PIVOT_DROPNA_CHECK = "step8_pivot_dropna_check"

# --- Module State ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_input_df: Optional[pd.DataFrame] = None
_s1_col_types: Dict = {}
_last_previewed_df: Optional[pd.DataFrame] = None
_index_cols: List[str] = []
_column_cols: List[str] = []
_value_cols: List[str] = []
_value_agg_funcs: Dict[str, str] = {}


def _create_pill(parent_shelf_group: str, col_name: str, shelf_type: str):
    s1_type = _s1_col_types.get(col_name, "")
    is_numeric = "Numeric" in s1_type or (_input_df is not None and pd.api.types.is_numeric_dtype(_input_df[col_name].dtype))
    color = [100, 200, 255] if is_numeric else [150, 255, 150]

    with dpg.group(horizontal=True, parent=parent_shelf_group):
        agg_str = f"({_value_agg_funcs.get(col_name, 'sum')}) " if shelf_type == 'values' else ""
        pill_button = dpg.add_button(label=f" {agg_str}{col_name} ", small=True)
        with dpg.theme() as pill_theme:
            with dpg.theme_component():
                dpg.add_theme_color(dpg.mvThemeCol_Button, color)
                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0))
        dpg.bind_item_theme(pill_button, pill_theme)
        
        with dpg.drag_payload(parent=pill_button, payload_type="COLUMN_DRAG", drag_data={'col': col_name, 'source_shelf': shelf_type}):
            dpg.add_text(f"Moving '{col_name}'")

        if shelf_type == 'values':
            with dpg.popup(pill_button, mousebutton=dpg.mvMouseButton_Left):
                agg_funcs = ['sum', 'mean', 'count', 'nunique', 'min', 'max', 'std', 'var']
                for func in agg_funcs:
                    dpg.add_selectable(label=func, user_data={'col': col_name, 'func': func}, callback=_change_agg_func, span_columns=True)

        dpg.add_button(label="X", small=True, user_data={'col': col_name, 'type': shelf_type}, callback=_remove_pill_callback)

def _handle_drop(target_shelf: str, payload: Any):
    if target_shelf == 'values' and len(_value_cols) >= 1:
        # 이동하는 경우는 예외 (기존 것을 지우고 새로 추가하므로)
        if isinstance(payload, str) or (isinstance(payload, dict) and payload.get('source_shelf') != 'values'):
            _util_funcs['_show_simple_modal_message']("Info", "The 'Values' shelf can only contain one variable.")
            return

    if isinstance(payload, str):
        col_name = payload
        all_cols = _index_cols + _column_cols + _value_cols
        if col_name in all_cols:
            _util_funcs['_show_simple_modal_message']("Info", f"'{col_name}' is already on a shelf.\nDrag the pill to move it.")
            return
    elif isinstance(payload, dict):
        col_name, source_shelf = payload['col'], payload['source_shelf']
        if source_shelf == target_shelf: return
        _remove_pill_from_state(col_name, source_shelf)
    else: return

    if target_shelf in ['index', 'columns']:
        nunique = _input_df[col_name].nunique()
        if nunique > 500:
            def proceed_action():
                _add_pill_to_state(col_name, target_shelf); _update_ui_shelves()
            _util_funcs['show_confirmation_modal'](title="High Cardinality Warning", message=f"'{col_name}' has {nunique} unique values.\nThis may cause performance issues.\n\nDo you want to proceed?", yes_callback=proceed_action)
            return
    _add_pill_to_state(col_name, target_shelf)
    _update_ui_shelves()

def _on_drop_index(sender, app_data, user_data): _handle_drop('index', app_data)
def _on_drop_columns(sender, app_data, user_data): _handle_drop('columns', app_data)
def _on_drop_values(sender, app_data, user_data): _handle_drop('values', app_data)

def _preview_callback():
    global _last_previewed_df
    _last_previewed_df = None

    if not _index_cols and not _value_cols:
        _util_funcs['_show_simple_modal_message']("Info", "Please drag variables to shelves to generate a preview."); return
    
    dpg.set_value(TAG_S8_PREVIEW_TEXT, "Processing...")
    
    try:
        df, result_df = _input_df.copy(), None
        
        if not _column_cols:
            if not _index_cols: _util_funcs['_show_simple_modal_message']("Error", "GroupBy requires at least one variable on the 'Index' shelf."); return
            agg_dict = {k: v for k, v in _value_agg_funcs.items() if k in _value_cols}
            result_df = df.groupby(_index_cols, as_index=False, observed=False).agg(agg_dict) if agg_dict else df.groupby(_index_cols, as_index=False, observed=False).size().reset_index(name='count')
        else:
            if not _index_cols or not _value_cols:
                _util_funcs['_show_simple_modal_message']("Error", "Pivot Table requires variables on 'Index', 'Columns', and 'Values' shelves."); return
            
            pivot_kwargs = {
                'values': _value_cols[0],
                'index': _index_cols,
                'columns': _column_cols,
                'aggfunc': _value_agg_funcs.get(_value_cols[0], 'mean'),
                'dropna': dpg.get_value(TAG_S8_PIVOT_DROPNA_CHECK)
            }
            if dpg.get_value(TAG_S8_PIVOT_FILLNA_CHECK):
                pivot_kwargs['fill_value'] = 0
            if dpg.get_value(TAG_S8_PIVOT_MARGINS_CHECK):
                pivot_kwargs['margins'] = True

            result_df = pd.pivot_table(df, **pivot_kwargs)
            
            if isinstance(result_df.columns, pd.MultiIndex):
                result_df.columns = result_df.columns.map(lambda col: '_'.join(map(str, col)).strip())

            result_df = result_df.reset_index()

        _last_previewed_df = result_df
        dpg.set_value(TAG_S8_PREVIEW_TEXT, f"Preview (Shape: {result_df.shape})")
        _util_funcs['create_table_with_large_data_preview'](TAG_S8_PREVIEW_TABLE, result_df)

    except Exception as e:
        dpg.set_value(TAG_S8_PREVIEW_TEXT, "Error during processing.")
        _util_funcs['_show_simple_modal_message']("Processing Error", f"An error occurred:\n{e}")
        traceback.print_exc()

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks, _util_funcs = main_callbacks, main_callbacks.get('get_util_funcs', lambda: {})()
    main_callbacks['register_step_group_tag'](step_name, TAG_S8_GROUP)

    with dpg.group(tag=TAG_S8_GROUP, parent=parent_container_tag, show=False):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=200, border=True):
                dpg.add_text("Variables(Drag&Drop)")
                dpg.add_input_text(tag=TAG_S8_VAR_SEARCH_INPUT, hint="Search variables...", callback=lambda: _update_ui_variable_list(), width=-1)
                dpg.add_separator()
                with dpg.child_window(tag=TAG_S8_VAR_LIST_WINDOW): pass
            
            with dpg.child_window(border=True):
                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_text("Index:   ", color=[200, 200, 200])
                        with dpg.child_window(tag=TAG_S8_SHELF_INDEX_CW, height=45, border=True, drop_callback=_on_drop_index, payload_type="COLUMN_DRAG"):
                            dpg.add_group(tag=TAG_S8_SHELF_INDEX_GROUP, horizontal=True)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Columns:", color=[200, 200, 200])
                        with dpg.child_window(tag=TAG_S8_SHELF_COLUMNS_CW, height=45, border=True, drop_callback=_on_drop_columns, payload_type="COLUMN_DRAG"):
                            dpg.add_group(tag=TAG_S8_SHELF_COLUMNS_GROUP, horizontal=True)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Values:  ", color=[200, 200, 200])
                        with dpg.child_window(tag=TAG_S8_SHELF_VALUES_CW, height=45, border=True, drop_callback=_on_drop_values, payload_type="COLUMN_DRAG"):
                            dpg.add_group(tag=TAG_S8_SHELF_VALUES_GROUP, horizontal=True)

                with dpg.group(horizontal=True):
                    dpg.add_text("Pivot Options: ")
                    dpg.add_checkbox(label="fill_value=0", tag=TAG_S8_PIVOT_FILLNA_CHECK)
                    with dpg.tooltip(dpg.last_item()):
                        dpg.add_text("Replace missing values (NaN) with 0.")
                    dpg.add_checkbox(label="margins=True", tag=TAG_S8_PIVOT_MARGINS_CHECK)
                    with dpg.tooltip(dpg.last_item()):
                        dpg.add_text("Add all row / column subtotals (margins).")
                    dpg.add_checkbox(label="dropna=True", tag=TAG_S8_PIVOT_DROPNA_CHECK, default_value=True)
                    with dpg.tooltip(dpg.last_item()):
                        dpg.add_text("Do not include columns whose entries are all NaN.")

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Preview", callback=_preview_callback)
                    dpg.add_button(label="Clear All Shelves", callback=_clear_shelves)
                    dpg.add_spacer(width=50)
                    dpg.add_text("Save as:")
                    dpg.add_input_text(tag=TAG_S8_OUTPUT_NAME_INPUT, width=200)
                    dpg.add_button(label="Save Derived DF", callback=_save_derived_df)
                dpg.add_separator()
                dpg.add_text("Preview Area", tag=TAG_S8_PREVIEW_TEXT)
                with dpg.child_window(border=False):
                    with dpg.table(tag=TAG_S8_PREVIEW_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit, scrollY=True, scrollX=True): pass
    
    main_callbacks['register_module_updater'](step_name, update_ui)

def _change_agg_func(sender, app_data, user_data):
    _value_agg_funcs[user_data['col']] = user_data['func']
    _update_ui_shelves()

def _remove_pill_from_state(col_to_remove: str, shelf_type: str):
    lists_map = {'index': _index_cols, 'columns': _column_cols, 'values': _value_cols}
    if shelf_type in lists_map and col_to_remove in lists_map[shelf_type]:
        lists_map[shelf_type].remove(col_to_remove)
        if shelf_type == 'values' and col_to_remove in _value_agg_funcs: del _value_agg_funcs[col_to_remove]

def _remove_pill_callback(sender, app_data, user_data):
    _remove_pill_from_state(user_data['col'], user_data['type'])
    _update_ui_shelves()

def _add_pill_to_state(col_name: str, shelf_type: str):
    target_list = None
    if shelf_type == 'index': target_list = _index_cols
    elif shelf_type == 'columns': target_list = _column_cols
    elif shelf_type == 'values': target_list = _value_cols
    if target_list is not None and col_name not in target_list:
        target_list.append(col_name)
        if shelf_type == 'values' and col_name not in _value_agg_funcs:
            _value_agg_funcs[col_name] = 'sum' if (_input_df is not None and pd.api.types.is_numeric_dtype(_input_df[col_name].dtype)) else 'count'

def _update_ui_variable_list():
    dpg.delete_item(TAG_S8_VAR_LIST_WINDOW, children_only=True)
    if _input_df is None: dpg.add_text("No data available.", parent=TAG_S8_VAR_LIST_WINDOW); return
    
    search_val = dpg.get_value(TAG_S8_VAR_SEARCH_INPUT)
    search_term = search_val.lower() if search_val else ""
    
    for col_name in _input_df.columns:
        if search_term and search_term not in col_name.lower(): continue
        s1_type = _s1_col_types.get(col_name, str(_input_df[col_name].dtype)); is_numeric = "Numeric" in s1_type or pd.api.types.is_numeric_dtype(_input_df[col_name].dtype)
        icon, color = ("#", (100, 200, 255)) if is_numeric else ("A", (150, 255, 150))
        with dpg.group(horizontal=True, parent=TAG_S8_VAR_LIST_WINDOW) as var_group:
            dpg.add_text(f"({icon})", color=color); dpg.add_text(col_name)
        with dpg.drag_payload(parent=var_group, payload_type="COLUMN_DRAG", drag_data=col_name):
            dpg.add_text(f"Adding '{col_name}' to shelf...")
            with dpg.theme() as payload_theme:
                 with dpg.theme_component(dpg.mvDragPayload):
                    dpg.add_theme_color(dpg.mvThemeCol_Border, (255, 255, 0), category=dpg.mvThemeCat_Core)
            dpg.bind_item_theme(dpg.last_item(), payload_theme)

def _update_ui_shelves():
    for shelf_group_tag, cols, shelf_type in [(TAG_S8_SHELF_INDEX_GROUP, _index_cols, 'index'), (TAG_S8_SHELF_COLUMNS_GROUP, _column_cols, 'columns'), (TAG_S8_SHELF_VALUES_GROUP, _value_cols, 'values')]:
        if dpg.does_item_exist(shelf_group_tag):
            dpg.delete_item(shelf_group_tag, children_only=True)
            for col in cols: _create_pill(shelf_group_tag, col, shelf_type)

def _clear_shelves():
    global _last_previewed_df; _index_cols.clear(); _column_cols.clear(); _value_cols.clear(); _value_agg_funcs.clear(); _last_previewed_df = None
    _update_ui_shelves()
    dpg.set_value(TAG_S8_PREVIEW_TEXT, "Drag variables to shelves and click 'Preview'.")
    _util_funcs['create_table_with_data'](TAG_S8_PREVIEW_TABLE, pd.DataFrame())

def _save_derived_df():
    output_df_name = dpg.get_value(TAG_S8_OUTPUT_NAME_INPUT).strip()
    if not output_df_name: _util_funcs['_show_simple_modal_message']("Input Error", "Please enter a name for the derived DataFrame."); return
    if _last_previewed_df is None: _util_funcs['_show_simple_modal_message']("Data Error", "No data to save. Please generate a preview first."); return
    if _module_main_callbacks and 'step8_derivation_complete' in _module_main_callbacks:
        _module_main_callbacks['step8_derivation_complete'](output_df_name, _last_previewed_df.copy())
        _util_funcs['_show_simple_modal_message']("Success", f"Derived DataFrame '{output_df_name}' has been saved.")
        dpg.set_value(TAG_S8_OUTPUT_NAME_INPUT, "")
    else: _util_funcs['_show_simple_modal_message']("Callback Error", "Could not save DataFrame due to a missing application callback.")

def update_ui():
    global _input_df, _s1_col_types
    if not _module_main_callbacks: return
    all_dfs = _module_main_callbacks['get_all_available_dfs']()
    if not all_dfs: _input_df = None
    else:
        df_keys_priority = ['From Step 7 (Features Engineered)', 'From Step 6 (Standardized)','From Step 5 (Outliers Treated)', 'From Step 4 (Missing Imputed)','From Step 3 (Preprocessed)', 'From Step 1 (Types Applied)']
        found_df = False
        for key in df_keys_priority:
            if key in all_dfs: _input_df = all_dfs[key]; found_df = True; break
        if not found_df: _input_df = next(iter(all_dfs.values()))
    _s1_col_types = _module_main_callbacks['get_column_analysis_types']()
    _update_ui_variable_list(); _update_ui_shelves()
    dpg.set_value(TAG_S8_PREVIEW_TEXT, "Drag variables to shelves and click 'Preview'.")
    _util_funcs['create_table_with_data'](TAG_S8_PREVIEW_TABLE, pd.DataFrame())
    
def reset_state():
    global _input_df, _s1_col_types, _last_previewed_df
    _input_df = None; _s1_col_types = {}; _last_previewed_df = None
    _clear_shelves()
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_S8_VAR_SEARCH_INPUT):
            dpg.set_value(TAG_S8_VAR_SEARCH_INPUT, "")
        if dpg.does_item_exist(TAG_S8_GROUP):
            update_ui()