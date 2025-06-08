# step_08_derivation.py (KeyError 수정 최종 버전)

import dearpygui.dearpygui as dpg
import pandas as pd
from typing import Dict, Optional, Any, List
import traceback
import datetime

# --- DPG Tags ---
TAG_S8_GROUP = "step8_derivation_group"
TAG_S8_VAR_LIST_WINDOW = "step8_var_list_window"
TAG_S8_VAR_SEARCH_INPUT = "step8_var_search_input"
TAG_S8_SHELF_FILTERS_CW = "step8_shelf_filters_child_window"
TAG_S8_SHELF_FILTERS_GROUP = "step8_shelf_filters_group"
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
TAG_S8_FILTER_MODAL = "step8_filter_modal"
TAG_S8_FILTER_ERROR_TEXT = "step8_filter_error_text"
TAG_S8_FILTER_TAB_BAR = "step8_filter_tab_bar"

# --- Module State ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_input_df: Optional[pd.DataFrame] = None
_s1_col_types: Dict = {}
_last_previewed_df: Optional[pd.DataFrame] = None
_filter_cols: List[str] = []
_index_cols: List[str] = []
_column_cols: List[str] = []
_value_cols: List[str] = []
_value_agg_funcs: Dict[str, str] = {}
_filter_conditions: Dict[str, Dict] = {}


def _create_pill(parent_shelf_group: str, col_name: str, shelf_type: str):
    s1_type = _s1_col_types.get(col_name, "")
    is_numeric = "Numeric" in s1_type or (_input_df is not None and pd.api.types.is_numeric_dtype(_input_df[col_name].dtype))
    is_datetime = "Datetime" in s1_type or (_input_df is not None and pd.api.types.is_datetime64_any_dtype(_input_df[col_name].dtype))

    if is_numeric: color = [100, 200, 255]
    elif is_datetime: color = [255, 220, 150]
    else: color = [150, 255, 150]

    with dpg.group(horizontal=True, parent=parent_shelf_group):
        pill_label = ""
        if shelf_type == 'values': pill_label = f" ({_value_agg_funcs.get(col_name, 'sum')}) {col_name} "
        elif shelf_type == 'filters': pill_label = f" {col_name}{' ✔' if col_name in _filter_conditions and _filter_conditions[col_name] else ''} "
        else: pill_label = f" {col_name} "

        pill_button = dpg.add_button(label=pill_label, small=True, user_data={'col': col_name}, callback=_open_filter_modal if shelf_type == 'filters' else None)

        with dpg.theme() as pill_theme:
            with dpg.theme_component(): dpg.add_theme_color(dpg.mvThemeCol_Button, color); dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0))
        dpg.bind_item_theme(pill_button, pill_theme)
        
        with dpg.drag_payload(parent=pill_button, payload_type="COLUMN_DRAG", drag_data={'col': col_name, 'source_shelf': shelf_type}):
            dpg.add_text(f"Moving '{col_name}'")

        if shelf_type == 'values':
            with dpg.popup(pill_button, mousebutton=dpg.mvMouseButton_Left):
                for func in ['sum', 'mean', 'count', 'nunique', 'min', 'max', 'std', 'var']:
                    dpg.add_selectable(label=func, user_data={'col': col_name, 'func': func}, callback=_change_agg_func, span_columns=True)

        dpg.add_button(label="X", small=True, user_data={'col': col_name, 'type': shelf_type}, callback=_remove_pill_callback)

def _handle_drop(target_shelf: str, payload: Any):
    if isinstance(payload, str):
        col_name = payload
        if target_shelf == 'filters':
            if col_name in _filter_cols: _util_funcs['_show_simple_modal_message']("Info", f"'{col_name}' is already on the Filters shelf."); return
        else:
            if col_name in (_index_cols + _column_cols + _value_cols): _util_funcs['_show_simple_modal_message']("Info", f"'{col_name}' is already on a pivot shelf."); return
    elif isinstance(payload, dict):
        col_name, source_shelf = payload['col'], payload['source_shelf']
        if source_shelf == target_shelf: return
        _remove_pill_from_state(col_name, source_shelf)
    else: return

    if target_shelf == 'values' and len(_value_cols) >= 1: _util_funcs['_show_simple_modal_message']("Info", "The 'Values' shelf can only contain one variable."); return
    if target_shelf in ['index', 'columns'] and _input_df[col_name].nunique() > 500:
        def proceed(): _add_pill_to_state(col_name, target_shelf); _update_ui_shelves()
        _util_funcs['show_confirmation_modal'](title="High Cardinality", message=f"'{col_name}' has high cardinality.\nProceed?", yes_callback=proceed)
        return
        
    _add_pill_to_state(col_name, target_shelf)
    _update_ui_shelves()

def _on_drop_filter(s, a, u): _handle_drop('filters', a)
def _on_drop_index(s, a, u): _handle_drop('index', a)
def _on_drop_columns(s, a, u): _handle_drop('columns', a)
def _on_drop_values(s, a, u): _handle_drop('values', a)

def _preview_callback():
    global _last_previewed_df
    _last_previewed_df = None
    if not _index_cols and not _value_cols:
        _util_funcs['_show_simple_modal_message']("Info", "Please drag variables to shelves to generate a preview."); return
    
    dpg.set_value(TAG_S8_PREVIEW_TEXT, "Processing...")
    try:
        # 1. 필터링 적용 (기존과 동일)
        df = _apply_filters_to_df(_input_df)

        # --- [수정] 범주형 필터가 0으로 보이는 문제 해결 ---
        # 집계/피봇에 사용될 컬럼들에서, 필터링 후 사용되지 않는 카테고리 정보를 제거합니다.
        pivot_and_group_cols = _index_cols + _column_cols
        for col_name in pivot_and_group_cols:
            if col_name in df.columns and pd.api.types.is_categorical_dtype(df[col_name].dtype):
                df[col_name] = df[col_name].cat.remove_unused_categories()
        # --- 수정 끝 ---
        
        result_df = None
        if not _column_cols: # GroupBy 로직
            if not _index_cols: 
                _util_funcs['_show_simple_modal_message']("Error", "GroupBy requires at least one variable on the 'Index' shelf."); return
            
            agg_dict = {k: v for k, v in _value_agg_funcs.items() if k in _value_cols}
            
            if agg_dict:
                 result_df = df.groupby(_index_cols, as_index=False, observed=False).agg(agg_dict)
            else: # 집계(Values) 없이 카운트만 할 때
                 # --- [수정] GroupBy 오류(TypeError, ValueError) 해결 ---
                 result_df = df.groupby(_index_cols, observed=False).size().to_frame(name='count').reset_index()
                 # --- 수정 끝 ---

        else: # Pivot Table 로직
            if not _index_cols or not _value_cols:
                _util_funcs['_show_simple_modal_message']("Error", "Pivot Table requires variables on all shelves."); return
            
            pivot_kwargs = {'values': _value_cols[0], 'index': _index_cols, 'columns': _column_cols, 'aggfunc': _value_agg_funcs.get(_value_cols[0], 'mean'), 'dropna': dpg.get_value(TAG_S8_PIVOT_DROPNA_CHECK)}
            if dpg.get_value(TAG_S8_PIVOT_FILLNA_CHECK): pivot_kwargs['fill_value'] = 0
            if dpg.get_value(TAG_S8_PIVOT_MARGINS_CHECK): pivot_kwargs['margins'] = True
            
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
                dpg.add_text("Variables(Drag&Drop)"); dpg.add_input_text(tag=TAG_S8_VAR_SEARCH_INPUT, hint="Search...", callback=lambda: _update_ui_variable_list(), width=-1)
                dpg.add_separator(); dpg.add_child_window(tag=TAG_S8_VAR_LIST_WINDOW)
            with dpg.child_window(border=True):
                with dpg.group(horizontal=True):
                    with dpg.group(width=220):
                        dpg.add_text("Filters:", color=[200, 200, 200])
                        with dpg.child_window(tag=TAG_S8_SHELF_FILTERS_CW, height=140, border=True, drop_callback=_on_drop_filter, payload_type="COLUMN_DRAG"):
                            dpg.add_group(tag=TAG_S8_SHELF_FILTERS_GROUP, horizontal=False)
                    with dpg.group():
                        for name in ["Index", "Columns", "Values"]:
                            with dpg.group(horizontal=True):
                                dpg.add_text(f"{name+':':<8}", color=[200, 200, 200])
                                with dpg.child_window(tag=globals()[f'TAG_S8_SHELF_{name.upper()}_CW'], height=45, border=True, drop_callback=globals()[f'_on_drop_{name.lower()}'], payload_type="COLUMN_DRAG"):
                                    dpg.add_group(tag=globals()[f'TAG_S8_SHELF_{name.upper()}_GROUP'], horizontal=True)
                with dpg.group(horizontal=True):
                    dpg.add_text("Pivot Options: "); [dpg.add_checkbox(label=l, tag=t) for l, t in [("fill_value=0", TAG_S8_PIVOT_FILLNA_CHECK), ("margins=True", TAG_S8_PIVOT_MARGINS_CHECK)]]; dpg.add_checkbox(label="dropna=True", tag=TAG_S8_PIVOT_DROPNA_CHECK, default_value=True)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Preview", callback=_preview_callback); dpg.add_button(label="Clear All Shelves", callback=_clear_shelves); dpg.add_spacer(width=50); dpg.add_text("Save as:"); dpg.add_input_text(tag=TAG_S8_OUTPUT_NAME_INPUT, width=200); dpg.add_button(label="Save Derived DF", callback=_save_derived_df)
                dpg.add_separator()
                dpg.add_text("Preview Area", tag=TAG_S8_PREVIEW_TEXT)
                with dpg.child_window(border=False):
                    with dpg.table(tag=TAG_S8_PREVIEW_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit, scrollY=True, scrollX=True): pass
    main_callbacks['register_module_updater'](step_name, update_ui)

def _change_agg_func(s, a, u): _value_agg_funcs[u['col']] = u['func']; _update_ui_shelves()
def _remove_pill_from_state(col, shelf): shelf_map = {'filters': _filter_cols, 'index': _index_cols, 'columns': _column_cols, 'values': _value_cols}; shelf_map[shelf].remove(col) if col in shelf_map[shelf] else None; _value_agg_funcs.pop(col, None) if shelf == 'values' else None; _filter_conditions.pop(col, None) if shelf == 'filters' else None
def _remove_pill_callback(s, a, u): _remove_pill_from_state(u['col'], u['type']); _update_ui_shelves()
def _add_pill_to_state(col, shelf): shelf_map = {'filters': _filter_cols, 'index': _index_cols, 'columns': _column_cols, 'values': _value_cols}; shelf_map[shelf].append(col) if col not in shelf_map[shelf] else None; _value_agg_funcs.setdefault(col, 'sum' if pd.api.types.is_numeric_dtype(_input_df[col].dtype) else 'count') if shelf == 'values' else None

def _update_ui_variable_list():
    dpg.delete_item(TAG_S8_VAR_LIST_WINDOW, children_only=True)
    if _input_df is None: return
    term = dpg.get_value(TAG_S8_VAR_SEARCH_INPUT).lower()
    for col in _input_df.columns:
        if term and term not in col.lower(): continue
        s1_type = _s1_col_types.get(col, str(_input_df[col].dtype)); is_num, is_dt = "Numeric" in s1_type or pd.api.types.is_numeric_dtype(_input_df[col].dtype), "Datetime" in s1_type or pd.api.types.is_datetime64_any_dtype(_input_df[col].dtype)
        icon, color = ("#", (100, 200, 255)) if is_num else (("T", (255, 220, 150)) if is_dt else ("A", (150, 255, 150)))
        with dpg.group(horizontal=True, parent=TAG_S8_VAR_LIST_WINDOW) as vg:
            dpg.add_text(f"({icon})", color=color); dpg.add_text(col)
        with dpg.drag_payload(parent=vg, payload_type="COLUMN_DRAG", drag_data=col): dpg.add_text(f"Adding '{col}'...")

def _update_ui_shelves():
    configs = [(TAG_S8_SHELF_FILTERS_GROUP, _filter_cols, 'filters'), (TAG_S8_SHELF_INDEX_GROUP, _index_cols, 'index'), (TAG_S8_SHELF_COLUMNS_GROUP, _column_cols, 'columns'), (TAG_S8_SHELF_VALUES_GROUP, _value_cols, 'values')]
    for group, cols, type in configs:
        if dpg.does_item_exist(group):
            dpg.delete_item(group, children_only=True)
            for col in cols: _create_pill(group, col, type)

def _clear_shelves(s=None, a=None, u=None):
    global _last_previewed_df
    _filter_cols.clear(); _index_cols.clear(); _column_cols.clear(); _value_cols.clear(); _filter_conditions.clear(); _value_agg_funcs.clear(); _last_previewed_df = None
    _update_ui_shelves(); dpg.set_value(TAG_S8_PREVIEW_TEXT, "Drag variables and click 'Preview'."); _util_funcs['create_table_with_data'](TAG_S8_PREVIEW_TABLE, pd.DataFrame())

def _save_derived_df():
    name = dpg.get_value(TAG_S8_OUTPUT_NAME_INPUT).strip()
    if not name: _util_funcs['_show_simple_modal_message']("Input Error", "Please enter a name."); return
    if _last_previewed_df is None: _util_funcs['_show_simple_modal_message']("Data Error", "No data to save."); return
    if _module_main_callbacks and 'step8_derivation_complete' in _module_main_callbacks:
        _module_main_callbacks['step8_derivation_complete'](name, _last_previewed_df.copy()); _util_funcs['_show_simple_modal_message']("Success", f"DataFrame '{name}' saved."); dpg.set_value(TAG_S8_OUTPUT_NAME_INPUT, "")

def update_ui():
    global _input_df, _s1_col_types
    if not _module_main_callbacks: return
    all_dfs = _module_main_callbacks['get_all_available_dfs']()
    if not all_dfs: _input_df = None
    else:
        priority = ['★ Current Working DF', '7. After Feature Engineering', '6. After Standardization', '5. After Outlier Treatment', '4. After Missing Value Imputation', '1. After Type Conversion', '0. Original Data']
        _input_df = next((all_dfs[k] for k in priority if k in all_dfs), next(iter(all_dfs.values()), None))
    _s1_col_types = _module_main_callbacks['get_column_analysis_types'](); _clear_shelves(); _update_ui_variable_list()
    
def reset_state():
    global _input_df, _s1_col_types, _last_previewed_df
    _input_df, _s1_col_types, _last_previewed_df = None, {}, None; _clear_shelves()
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_S8_GROUP): update_ui()

def _apply_filters_to_df(df: pd.DataFrame) -> pd.DataFrame:
    if not _filter_conditions or df is None: return df
    df_filtered = df.copy()
    for col, cond in _filter_conditions.items():
        if not cond or col not in df_filtered.columns: continue
        try:
            if cond.get('type') == 'in': df_filtered = df_filtered[df_filtered[col].isin(cond['values'])]
            elif cond.get('type') == 'range': df_filtered = df_filtered[df_filtered[col].between(cond['min'], cond['max'])]
            elif cond.get('type') == 'compare':
                op, val = cond['op'], float(cond['val'])
                ops = {'>': df_filtered[col] > val, '>=': df_filtered[col] >= val, '<': df_filtered[col] < val, '<=': df_filtered[col] <= val, '==': df_filtered[col] == val, '!=': df_filtered[col] != val}
                df_filtered = df_filtered[ops[op]]
            elif cond.get('type') == 'datetime_range':
                start, end = pd.to_datetime(cond['start']), pd.to_datetime(cond['end']).replace(hour=23, minute=59, second=59)
                df_filtered = df_filtered[df_filtered[col].between(start, end)]
        except Exception as e: _util_funcs['_show_simple_modal_message']("Filter Error", f"Error on '{col}':\n{e}")
    return df_filtered

def _set_error_message(message: str):
    if dpg.does_item_exist(TAG_S8_FILTER_ERROR_TEXT): dpg.set_value(TAG_S8_FILTER_ERROR_TEXT, message); dpg.configure_item(TAG_S8_FILTER_ERROR_TEXT, show=True, color=(255, 80, 80))
def _clear_error_message():
    if dpg.does_item_exist(TAG_S8_FILTER_ERROR_TEXT): dpg.set_value(TAG_S8_FILTER_ERROR_TEXT, ""); dpg.configure_item(TAG_S8_FILTER_ERROR_TEXT, show=False)

def _save_filter_condition(sender, app_data, user_data):
    try:
        _clear_error_message()
        col, new_cond = user_data['col'], {}
        
        cond_type = user_data.get('type')
        if not cond_type and 'tab_bar' in user_data:
            active_tab = dpg.get_value(user_data['tab_bar'])
            if active_tab == user_data.get('compare_tab'): cond_type = 'compare'
            elif active_tab == user_data.get('range_tab'): cond_type = 'range'
        if not cond_type: _set_error_message("Could not determine filter type."); return
        
        new_cond['type'] = cond_type

        if cond_type == 'datetime_range':
            start, end = dpg.get_value(user_data['start_picker']), dpg.get_value(user_data['end_picker'])
            sel_start, sel_end = datetime.date(start['year']+1900, start['month']+1, start['month_day']), datetime.date(end['year']+1900, end['month']+1, end['month_day'])
            valid_min, valid_max = user_data['valid_min'].date(), user_data['valid_max'].date()
            if not (valid_min <= sel_start <= valid_max): _set_error_message(f"Start date out of range ({valid_min} ~ {valid_max})"); return
            if not (valid_min <= sel_end <= valid_max): _set_error_message(f"End date out of range ({valid_min} ~ {valid_max})"); return
            if sel_start > sel_end: _set_error_message("Start date cannot be after end date."); return
            new_cond.update({'start': sel_start.isoformat(), 'end': sel_end.isoformat()})
        elif cond_type == 'in':
            values = [dpg.get_item_user_data(cb) for cb in user_data['cbs'] if dpg.get_value(cb)]
            if not values: new_cond = {}
            else: new_cond.update({'values': values, 'type': 'in'})
        elif cond_type == 'range':
            new_cond.update({'min': dpg.get_value(user_data['drag'])[0], 'max': dpg.get_value(user_data['drag'])[1]})
        elif cond_type == 'compare':
            new_cond.update({'op': dpg.get_value(user_data['op']), 'val': dpg.get_value(user_data['val'])})
        
        _filter_conditions[col] = new_cond if new_cond and 'type' in new_cond else {}
        dpg.configure_item(TAG_S8_FILTER_MODAL, show=False); _update_ui_shelves()
    except Exception:
        _set_error_message(f"Unexpected error:\n{traceback.format_exc()}")

def _update_selected_date_display(s, a, u):
    if dpg.does_item_exist(u['tag']): dpg.set_value(u['tag'], f"Selected: {a['year']+1900}-{a['month']+1:02d}-{a['month_day']:02d}")
def _update_range_text(s, a, u):
    if dpg.does_item_exist(u['tag']): dpg.set_value(u['tag'], f"Selected: {a[0]:.3f}  to  {a[1]:.3f}")

def _open_filter_modal(sender, app_data, user_data):
    col = user_data['col']
    series = _input_df[col].dropna()
    if series.empty:
        _util_funcs['_show_simple_modal_message']("Info", f"'{col}' has no valid data.")
        return
    if dpg.does_item_exist(TAG_S8_FILTER_MODAL):
        dpg.delete_item(TAG_S8_FILTER_MODAL)
    
    prev = _filter_conditions.get(col, {})
    with dpg.window(label=f"Filter: {col}", modal=True, show=True, tag=TAG_S8_FILTER_MODAL, width=300, no_close=True):
        dpg.add_text("", tag=TAG_S8_FILTER_ERROR_TEXT, show=False, wrap=400)
        
        # 새로운 규칙: 고유값 20개 이하이면 무조건 체크박스
        if series.nunique() <= 20:
            g = dpg.add_group()
            dpg.add_text("Include values:", parent=g)
            with dpg.child_window(height=min(250, len(series.unique()) * 35), parent=g, border=True):
                unique_values = sorted(list(series.unique()))
                cbs = [dpg.add_checkbox(label=str(v), user_data=v, default_value=v in prev.get('values', [])) for v in unique_values]
            dpg.add_button(label="Apply", width=-1, parent=g, user_data={'col': col, 'type':'in', 'cbs': cbs}, callback=_save_filter_condition)
        
        # 고유값이 20개를 초과할 때 타입별 처리
        else:
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                # (이전과 동일)
                g = dpg.add_group(); min_dt, max_dt = series.min(), series.max()
                start_val, end_val = pd.to_datetime(prev.get('start', min_dt)).date(), pd.to_datetime(prev.get('end', max_dt)).date()
                dpg_min, dpg_max = {'month_day': start_val.day, 'month': start_val.month-1, 'year': start_val.year-1900}, {'month_day': end_val.day, 'month': end_val.month-1, 'year': end_val.year-1900}
                dpg.add_text(f"Valid range: {min_dt.date()} to {max_dt.date()}", parent=g, color=(255, 255, 0))
                start_display = dpg.add_text(f"Selected: {start_val.isoformat()}", parent=g)
                s_picker = dpg.add_date_picker(label="Start", parent=g, default_value=dpg_min, callback=_update_selected_date_display, user_data={'tag': start_display})
                end_display = dpg.add_text(f"Selected: {end_val.isoformat()}", parent=g)
                e_picker = dpg.add_date_picker(label="End", parent=g, default_value=dpg_max, callback=_update_selected_date_display, user_data={'tag': end_display})
                ud = {'col': col, 'type':'datetime_range', 'start_picker': s_picker, 'end_picker': e_picker, 'valid_min': min_dt, 'valid_max': max_dt}
                dpg.add_button(label="Apply", width=-1, parent=g, user_data=ud, callback=_save_filter_condition)

            elif pd.api.types.is_numeric_dtype(series.dtype):
                # (이전과 동일)
                min_v, max_v = float(series.min()), float(series.max())
                dpg.add_text(f"Valid range: {min_v:.3f} to {max_v:.3f}", color=(255, 255, 0))
                with dpg.tab_bar(tag=TAG_S8_FILTER_TAB_BAR) as tab_bar:
                    with dpg.tab(label="Comparison") as compare_tab:
                        g = dpg.add_group(); 
                        op = dpg.add_combo(['>', '>=', '<', '<=', '==', '!='], label="Operator", parent=g, default_value=prev.get('op', '>'))
                        val = dpg.add_input_float(label="Value", parent=g, default_value=float(prev.get('val', (min_v + max_v) / 2)))
                    with dpg.tab(label="Range") as range_tab:
                        g = dpg.add_group(); 
                        range_val = [prev.get('min', min_v), prev.get('max', max_v)]
                        text = dpg.add_text(f"Selected: {range_val[0]:.3f}  to  {range_val[1]:.3f}", parent=g)
                        drag = dpg.add_drag_floatx(label="Min/Max", size=2, parent=g, default_value=range_val, callback=_update_range_text, user_data={'tag': text}, speed=0.01 * (max_v - min_v) if (max_v - min_v) > 0 else 0.1)
                if prev.get('type') == 'range': dpg.set_value(tab_bar, range_tab)
                else: dpg.set_value(tab_bar, compare_tab)
                dpg.add_separator()
                ud_numeric = {'col': col, 'tab_bar': tab_bar, 'compare_tab': compare_tab, 'range_tab': range_tab, 'op': op, 'val': val, 'drag': drag}
                dpg.add_button(label="Apply Filter", width=-1, user_data=ud_numeric, callback=_save_filter_condition)

            else: # 고유값 21~100개인 범주형/문자열 처리
                g = dpg.add_group()
                if series.nunique() > 100:
                    dpg.add_text("Filter is disabled.", parent=g, color=(255, 200, 0))
                    dpg.add_text("This column has more than 100 unique values.", parent=g)
                else:
                    dpg.add_text("Search and select values to include:", parent=g)
                    all_items_map = {str(v): v for v in sorted(series.unique())}
                    
                    def search_callback(s, a, u):
                        cbs_map = u['cbs_map']
                        term = a.lower()
                        for label, cb_tag in cbs_map.items():
                            dpg.configure_item(cb_tag, show=(term in label.lower()))
                    
                    searcher = dpg.add_input_text(label="Search", parent=g, width=-1)
                    with dpg.child_window(height=250, parent=g, border=True):
                        cbs_map = {str(v): dpg.add_checkbox(label=str(v), user_data=v, default_value=v in prev.get('values', [])) for v in all_items_map.values()}
                    
                    dpg.set_item_callback(searcher, search_callback)
                    dpg.set_item_user_data(searcher, {'cbs_map': cbs_map})
                    
                    dpg.add_button(label="Apply", width=-1, parent=g, user_data={'col': col, 'type':'in', 'cbs': list(cbs_map.values())}, callback=_save_filter_condition)
        
        dpg.add_separator()
        dpg.add_button(label="Cancel", width=-1, callback=lambda: dpg.configure_item(TAG_S8_FILTER_MODAL, show=False))
