# << REWRITTEN >> : step_08_derivation.py 파일의 전체 내용을 아래 코드로 교체합니다.

import dearpygui.dearpygui as dpg
import pandas as pd
from typing import Dict, Optional, Any, List

# --- DPG Tags ---
TAG_S8_GROUP = "step8_derivation_group"
TAG_S8_VAR_LIST_WINDOW = "step8_var_list_window"
TAG_S8_VAR_SEARCH_INPUT = "step8_var_search_input"
TAG_S8_SHELF_INDEX = "step8_shelf_index"
TAG_S8_SHELF_COLUMNS = "step8_shelf_columns"
TAG_S8_SHELF_VALUES = "step8_shelf_values"
TAG_S8_PREVIEW_TABLE = "step8_preview_table"
TAG_S8_PREVIEW_TEXT = "step8_preview_text"
TAG_S8_OUTPUT_NAME_INPUT = "step8_output_name_input"

# --- Module State ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_input_df: Optional[pd.DataFrame] = None
_s1_col_types: Dict = {}

# 선반에 놓인 변수들을 관리하는 리스트
_index_cols: List[str] = []
_column_cols: List[str] = []
_value_cols: List[str] = []
# Value 필의 집계함수를 관리하는 딕셔너리
_value_agg_funcs: Dict[str, str] = {}


def _create_pill(parent_shelf: str, col_name: str, shelf_type: str):
    """선반 위에 놓일 '필(Pill)' UI를 생성합니다."""
    with dpg.group(horizontal=True, parent=parent_shelf) as pill_group:
        dpg.add_button(label=f" {col_name} ", small=True)
        # TODO: 필 자체를 드래그 가능하게 하여 순서 변경 기능 추가
        
        # Value 선반 필인 경우, 집계 함수 변경 기능 제공
        if shelf_type == 'values':
            current_agg = _value_agg_funcs.get(col_name, 'sum')
            with dpg.popup(dpg.last_item(), mousebutton=dpg.mvMouseButton_Left) as popup:
                dpg.add_text(f"'{col_name}' Aggregation")
                dpg.add_separator()
                agg_funcs = ['sum', 'mean', 'count', 'nunique', 'min', 'max']
                for func in agg_funcs:
                    dpg.add_selectable(label=func, user_data={'col': col_name, 'func': func}, 
                                       callback=_change_agg_func, span_columns=True)

        dpg.add_button(label="X", small=True, user_data={'col': col_name, 'type': shelf_type},
                       callback=_remove_pill_callback)

def _change_agg_func(sender, app_data, user_data):
    """Value 필의 집계 함수를 변경합니다."""
    col = user_data['col']
    func = user_data['func']
    _value_agg_funcs[col] = func
    _update_ui_shelves()

def _remove_pill_callback(sender, app_data, user_data):
    """선반에서 필을 제거합니다."""
    col_to_remove = user_data['col']
    shelf_type = user_data['type']

    if shelf_type == 'index' and col_to_remove in _index_cols:
        _index_cols.remove(col_to_remove)
    elif shelf_type == 'columns' and col_to_remove in _column_cols:
        _column_cols.remove(col_to_remove)
    elif shelf_type == 'values' and col_to_remove in _value_cols:
        _value_cols.remove(col_to_remove)
        if col_to_remove in _value_agg_funcs:
            del _value_agg_funcs[col_to_remove]
            
    _update_ui_shelves()

def _on_drop(sender, app_data, user_data):
    """선반에 변수가 드롭되었을 때 호출되는 콜백."""
    col_name = dpg.get_value(app_data)
    shelf_type = user_data
    
    target_list = None
    if shelf_type == 'index':
        target_list = _index_cols
    elif shelf_type == 'columns':
        target_list = _column_cols
    elif shelf_type == 'values':
        target_list = _value_cols

    if target_list is None or col_name in target_list:
        return

    # 고유값(Cardinality) 체크
    if shelf_type in ['index', 'columns']:
        nunique = _input_df[col_name].nunique()
        if nunique > 100:
            def proceed_action():
                target_list.append(col_name)
                if shelf_type == 'values' and col_name not in _value_agg_funcs:
                    _value_agg_funcs[col_name] = 'sum' # 기본값
                _update_ui_shelves()

            _util_funcs['show_confirmation_modal'](
                title="High Cardinality Warning",
                message=f"'{col_name}' has {nunique} unique values.\nThis may cause performance issues or long processing times.\n\nDo you want to proceed?",
                yes_callback=proceed_action
            )
            return

    target_list.append(col_name)
    if shelf_type == 'values' and col_name not in _value_agg_funcs:
        _value_agg_funcs[col_name] = 'sum' # 기본값
    
    _update_ui_shelves()

def _update_ui_variable_list():
    """(좌측) 변수 목록 UI를 새로고침합니다."""
    dpg.delete_item(TAG_S8_VAR_LIST_WINDOW, children_only=True)
    
    if _input_df is None:
        dpg.add_text("No data available.", parent=TAG_S8_VAR_LIST_WINDOW)
        return

    search_term = dpg.get_value(TAG_S8_VAR_SEARCH_INPUT).lower()

    for col_name in _input_df.columns:
        if search_term and search_term not in col_name.lower():
            continue

        # 차원/측정값 시각적 구분
        s1_type = _s1_col_types.get(col_name, str(_input_df[col_name].dtype))
        is_numeric = "Numeric" in s1_type or pd.api.types.is_numeric_dtype(_input_df[col_name].dtype)
        
        # 아이콘 또는 색상으로 구분
        icon = "#" if is_numeric else "A"
        color = [100, 200, 255] if is_numeric else [150, 255, 150]
        
        with dpg.group(horizontal=True, parent=TAG_S8_VAR_LIST_WINDOW):
            dpg.add_text(f"({icon})", color=color)
            dpg.add_text(col_name)
        
        with dpg.drag_payload(parent=dpg.last_item(), payload_type="COLUMN_DRAG", drag_data=col_name):
            dpg.add_text(f"Dragging '{col_name}'")

def _update_ui_shelves():
    """(상단) 선반 UI를 현재 상태에 맞게 새로고침합니다."""
    for shelf_tag, cols, shelf_type in [(TAG_S8_SHELF_INDEX, _index_cols, 'index'), 
                                       (TAG_S8_SHELF_COLUMNS, _column_cols, 'columns'), 
                                       (TAG_S8_SHELF_VALUES, _value_cols, 'values')]:
        dpg.delete_item(shelf_tag, children_only=True)
        for col in cols:
            _create_pill(shelf_tag, col, shelf_type)
            
def _clear_shelves():
    """모든 선반을 비웁니다."""
    _index_cols.clear()
    _column_cols.clear()
    _value_cols.clear()
    _value_agg_funcs.clear()
    _update_ui_shelves()
    
def _preview_callback():
    """'Preview' 버튼 클릭 시 데이터 처리 및 미리보기를 실행합니다."""
    if not _index_cols and not _value_cols:
        _util_funcs['_show_simple_modal_message']("Info", "Please drag variables to 'Index' and 'Values' shelves to generate a preview.")
        return

    dpg.set_value(TAG_S8_PREVIEW_TEXT, "Processing...")
    
    try:
        df = _input_df.copy()
        result_df = None
        
        # 1. GroupBy (Columns 선반이 비어있을 때)
        if not _column_cols:
            if not _index_cols:
                 _util_funcs['_show_simple_modal_message']("Error", "GroupBy requires at least one variable on the 'Index' shelf.")
                 return
            
            if _value_cols:
                result_df = df.groupby(_index_cols, as_index=False).agg(_value_agg_funcs)
            else: # Values가 없으면 size 계산
                result_df = df.groupby(_index_cols, as_index=False).size()

        # 2. Pivot Table (Columns 선반이 채워져 있을 때)
        else:
            if not _index_cols or not _value_cols:
                _util_funcs['_show_simple_modal_message']("Error", "Pivot Table requires variables on 'Index', 'Columns', and 'Values' shelves.")
                return
            
            # pivot_table은 단일 value, 단일 aggfunc에 최적화. 여러개일 경우 pivot_table을 여러번 호출하거나 로직 복잡화 필요
            # 여기서는 첫번째 value와 그 aggfunc을 사용
            value_col = _value_cols[0]
            agg_func = _value_agg_funcs.get(value_col, 'mean')
            
            result_df = pd.pivot_table(df, values=value_col, index=_index_cols,
                                       columns=_column_cols, aggfunc=agg_func).reset_index()

        dpg.set_value(TAG_S8_PREVIEW_TEXT, f"Preview (Shape: {result_df.shape})")
        _util_funcs['create_table_with_large_data_preview'](TAG_S8_PREVIEW_TABLE, result_df)

    except Exception as e:
        dpg.set_value(TAG_S8_PREVIEW_TEXT, "Error during processing.")
        _util_funcs['_show_simple_modal_message']("Processing Error", f"An error occurred:\n{e}")
        traceback.print_exc()

def _save_derived_df():
    """처리된 DF를 AppState에 저장합니다."""
    # TODO: _preview_callback에서 생성된 result_df를 임시 저장했다가 여기서 사용
    _util_funcs['_show_simple_modal_message']("Info", "Save functionality is not yet implemented in this UI.")


def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    main_callbacks['register_step_group_tag'](step_name, TAG_S8_GROUP)

    with dpg.group(tag=TAG_S8_GROUP, parent=parent_container_tag, show=False):
        with dpg.group(horizontal=True):
            # --- (좌측) 변수 목록 패널 ---
            with dpg.child_window(width=250, border=True):
                dpg.add_text("Variables")
                dpg.add_input_text(tag=TAG_S8_VAR_SEARCH_INPUT, hint="Search variables...", 
                                   callback=lambda: _update_ui_variable_list(), width=-1)
                dpg.add_separator()
                with dpg.child_window(tag=TAG_S8_VAR_LIST_WINDOW):
                    pass # 동적으로 채워짐

            # --- (우측) 메인 패널 ---
            with dpg.child_window(border=True):
                # --- (상단) 선반 영역 ---
                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_text("Index:   ", color=[200, 200, 200])
                        with dpg.child_window(tag=TAG_S8_SHELF_INDEX, height=40, horizontal_scrollbar=True):
                             dpg.add_drop_callback(parent=dpg.last_item(), drop_callback=_on_drop, user_data='index')
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("Columns:", color=[200, 200, 200])
                        with dpg.child_window(tag=TAG_S8_SHELF_COLUMNS, height=40, horizontal_scrollbar=True):
                             dpg.add_drop_callback(parent=dpg.last_item(), drop_callback=_on_drop, user_data='columns')

                    with dpg.group(horizontal=True):
                        dpg.add_text("Values:  ", color=[200, 200, 200])
                        with dpg.child_window(tag=TAG_S8_SHELF_VALUES, height=40, horizontal_scrollbar=True):
                            dpg.add_drop_callback(parent=dpg.last_item(), drop_callback=_on_drop, user_data='values')

                # --- 실행 버튼 영역 ---
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Preview", callback=_preview_callback)
                    dpg.add_button(label="Clear All Shelves", callback=_clear_shelves)
                    dpg.add_spacer(width=50)
                    dpg.add_text("Save as:")
                    dpg.add_input_text(tag=TAG_S8_OUTPUT_NAME_INPUT, width=200)
                    dpg.add_button(label="Save Derived DF", callback=_save_derived_df)

                dpg.add_separator()
                
                # --- (하단) 미리보기 영역 ---
                dpg.add_text("Preview Area", tag=TAG_S8_PREVIEW_TEXT)
                with dpg.child_window(border=False):
                    with dpg.table(tag=TAG_S8_PREVIEW_TABLE, header_row=True, resizable=True, 
                                   policy=dpg.mvTable_SizingFixedFit, scrollY=True, scrollX=True):
                        pass

    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui():
    """전체 Step 8 UI를 현재 상태에 맞게 새로고침합니다."""
    global _input_df, _s1_col_types
    if not _module_main_callbacks: return
    
    # 이 단계에서는 항상 최신 DF를 사용
    all_dfs = _module_main_callbacks['get_all_available_dfs']()
    if not all_dfs: 
        _input_df = None
    else: # 가장 마지막 단계의 DF를 우선적으로 선택
        df_keys_priority = [
            'From Step 7 (Features Engineered)', 'From Step 6 (Standardized)',
            'From Step 5 (Outliers Treated)', 'From Step 4 (Missing Imputed)',
            'From Step 3 (Preprocessed)', 'From Step 1 (Types Applied)'
        ]
        found_df = False
        for key in df_keys_priority:
            if key in all_dfs:
                _input_df = all_dfs[key]
                found_df = True
                break
        if not found_df:
            _input_df = next(iter(all_dfs.values()))

    _s1_col_types = _module_main_callbacks['get_column_analysis_types']()

    # UI 컴포넌트들 업데이트
    _update_ui_variable_list()
    _update_ui_shelves()
    
    # 테이블 초기화
    dpg.set_value(TAG_S8_PREVIEW_TEXT, "Drag variables to shelves and click 'Preview'.")
    _util_funcs['create_table_with_data'](TAG_S8_PREVIEW_TABLE, pd.DataFrame())


def reset_state():
    """모듈의 상태를 초기화합니다."""
    global _input_df, _s1_col_types
    _input_df = None
    _s1_col_types = {}
    _clear_shelves()
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_S8_GROUP):
        dpg.set_value(TAG_S8_VAR_SEARCH_INPUT, "")
        update_ui()