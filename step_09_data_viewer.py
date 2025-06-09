# << MODIFIED FILE >>: step_09_data_viewer.py

import dearpygui.dearpygui as dpg
import pandas as pd
from typing import Dict, Optional, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import numpy as np
import traceback
import utils 

# --- DPG Tags for Step 9 ---
TAG_S9_GROUP = "step9_data_viewer_group"
TAG_S9_MAIN_TAB_BAR = "step9_main_tab_bar"
TAG_S9_DATA_VIEWER_TAB = "step9_data_viewer_tab"
TAG_S9_DF_TABLE = "step9_df_selectable_table"
TAG_S9_DF_SEARCH_INPUT = "step9_df_search_input"
TAG_S9_PREVIEW_TABLE = "step9_preview_table"
TAG_S9_PREVIEW_TEXT = "step9_preview_text"
TAG_S9_CHART_CONFIG_TAB = "step9_chart_config_tab"
TAG_S9_CHART_CONFIG_TABLE = "step9_chart_config_table"
TAG_S9_CHART_DISPLAY_TAB = "step9_chart_display_tab"
TAG_S9_CHART_DISPLAY_WINDOW = "step9_chart_display_window"

# --- Module State ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_available_dfs: Dict[str, pd.DataFrame] = {}
_chart_configs: List[Dict[str, Any]] = []
_texture_tags: List[str] = []

# --- Charting Constants ---
CHART_TYPES = ["lineplot", "scatterplot", "barplot", "histplot", "countplot", "boxplot", "violinplot"]
PLOT_CONFIG_DEFAULTS = {
    "df_source": "", "chart_type": "scatterplot", "x_axis": "", "y_axis": "",
    "hue": "", "col_facet": "", "width": -1, "height": 350, "label_by": "None", "id": ""
}
CHART_UI_RULES = {
    "lineplot":    {"y_axis": True, "hue": True, "col_facet": True, "label_by": True},
    "scatterplot": {"y_axis": True, "hue": True, "col_facet": True, "label_by": True},
    "barplot":     {"y_axis": True, "hue": True, "col_facet": True, "label_by": False},
    "histplot":    {"y_axis": False, "hue": True, "col_facet": True, "label_by": False},
    "countplot":   {"y_axis": False, "hue": True, "col_facet": True, "label_by": False},
    "boxplot":     {"y_axis": True, "hue": True, "col_facet": True, "label_by": False},
    "violinplot":  {"y_axis": True, "hue": True, "col_facet": True, "label_by": False},
}

# --- HELPER & CALLBACK FUNCTIONS ---

def _on_df_select(sender, app_data, user_data):
    if not app_data: return
    if dpg.does_item_exist(TAG_S9_DF_TABLE):
        rows = dpg.get_item_children(TAG_S9_DF_TABLE, 1)
        for row in rows:
            selectable = dpg.get_item_children(row, 1)[0]
            if selectable != sender: dpg.set_value(selectable, False)
    
    selected_df_name = user_data
    if selected_df_name in _available_dfs:
        df_to_preview = _available_dfs[selected_df_name]
        dpg.set_value(TAG_S9_PREVIEW_TEXT, f"Preview of '{selected_df_name}' (Shape: {df_to_preview.shape})")
        if _util_funcs and 'create_table_with_large_data_preview' in _util_funcs:
            _util_funcs['create_table_with_large_data_preview'](TAG_S9_PREVIEW_TABLE, df_to_preview)

def _update_df_list():
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_S9_DF_SEARCH_INPUT): return
    search_term = dpg.get_value(TAG_S9_DF_SEARCH_INPUT).lower()
    df_names = list(_available_dfs.keys())
    filtered_names = [name for name in df_names if search_term in name.lower()] if search_term else df_names
    
    if dpg.does_item_exist(TAG_S9_DF_TABLE):
        dpg.delete_item(TAG_S9_DF_TABLE, children_only=True)
        
        # 테이블 컬럼을 2개로 설정: 이름(가변), 삭제 버튼(고정)
        dpg.add_table_column(parent=TAG_S9_DF_TABLE, width_stretch=True)
        dpg.add_table_column(parent=TAG_S9_DF_TABLE, width_fixed=True, init_width_or_weight=45)

        for name in filtered_names:
            with dpg.table_row(parent=TAG_S9_DF_TABLE):
                # 이름 부분 (기존과 유사하나 span_columns=False로 변경)
                dpg.add_selectable(label=name, user_data=name, callback=_on_df_select, span_columns=False)
                
                # 삭제 버튼 부분
                if name.startswith("Derived: "):
                    # "Derived: " 접두사를 제거한 실제 DF 이름을 user_data로 전달
                    actual_df_name = name.replace("Derived: ", "", 1)
                    dpg.add_button(
                        label=" X ", 
                        user_data=actual_df_name, 
                        small=True,
                        callback=lambda s, a, u: _module_main_callbacks['delete_derived_df'](u)
                    )
                else:
                    # 파생 DF가 아닌 경우 빈 공간
                    dpg.add_text("")

def _clear_and_generate_charts():
    global _texture_tags
    for tag in _texture_tags:
        if dpg.does_item_exist(tag): dpg.delete_item(tag)
    _texture_tags.clear()
    
    if not dpg.does_item_exist(TAG_S9_CHART_DISPLAY_WINDOW): return
    dpg.delete_item(TAG_S9_CHART_DISPLAY_WINDOW, children_only=True)

    if not _chart_configs:
        dpg.add_text("차트가 구성되지 않았습니다. 'Chart Configuration' 탭에서 차트를 추가하세요.", parent=TAG_S9_CHART_DISPLAY_WINDOW)
        return

    plot_func = _util_funcs.get('plot_to_dpg_texture')
    if not plot_func:
        dpg.add_text("오류: 플롯 유틸리티 함수를 사용할 수 없습니다.", parent=TAG_S9_CHART_DISPLAY_WINDOW, color=(255,0,0))
        return

    for i, config in enumerate(_chart_configs):
        with dpg.group(parent=TAG_S9_CHART_DISPLAY_WINDOW):
            try:
                # --- 차트 생성 로직 (이전과 동일) ---
                df_name, chart_type = config.get("df_source"), config.get("chart_type")
                x, y, hue, col_facet = (config.get(k) or None for k in ["x_axis", "y_axis", "hue", "col_facet"])
                if not df_name or df_name not in _available_dfs: raise ValueError(f"DF '{df_name}' 없음")
                df = _available_dfs[df_name].copy()
                if not x or x not in df.columns: raise ValueError("X축 지정 필요")
                title = f"Chart {i+1}: {chart_type.capitalize()} on '{df_name}'"
                plot_function_map = { "lineplot": sns.pointplot, "scatterplot": sns.scatterplot, "barplot": sns.barplot, "histplot": sns.histplot, "countplot": sns.countplot, "boxplot": sns.boxplot, "violinplot": sns.violinplot }
                plot_function = plot_function_map.get(chart_type)
                if not plot_function: raise ValueError(f"알 수 없는 차트 타입: {chart_type}")
                plot_kwargs = {'data': df, 'x': x}
                if y and CHART_UI_RULES.get(chart_type,{})["y_axis"]: plot_kwargs['y'] = y
                if hue: plot_kwargs['hue'] = hue
                if col_facet and col_facet in df.columns:
                    g = sns.FacetGrid(df, col=col_facet, hue=hue, col_wrap=4, sharex=False, sharey=False, height=3, aspect=1.2)
                    map_args = [x]; 
                    if y and CHART_UI_RULES.get(chart_type,{})["y_axis"]: map_args.append(y)
                    g.map(plot_function, *map_args); final_fig = g.fig
                else:
                    fig, ax = plt.subplots(figsize=(8, 5)); plot_function(ax=ax, **plot_kwargs); final_fig = fig
                final_fig.suptitle(title, y=1.02); plt.tight_layout(pad=1.0)
                # --- 차트 생성 로직 끝 ---
                
                dpg.add_text(title)
                dpg.add_spacer(height=5)
                
                tex_tag, w, h, img_bytes = plot_func(final_fig)
                
                if tex_tag:
                    _texture_tags.append(tex_tag)
                    
                    # [최종 수정] 너비에만 맞추고, 높이는 비율에 따라 자유롭게 결정되도록 합니다.
                    # 부모 창의 스크롤바가 넘치는 높이를 처리해 줄 것입니다.
                    container_w = dpg.get_item_width(TAG_S9_CHART_DISPLAY_WINDOW) or 800
                    max_w = container_w - 30 # 좌우 여백

                    display_w = min(w, max_w)
                    display_h = int(h * (display_w / w)) if w > 0 else h

                    dpg.add_image(tex_tag, width=display_w, height=display_h)
                    
                    if img_bytes:
                        ai_button_tag = f"s9_ai_analyze_btn_{config.get('id', uuid.uuid4())}"
                        user_data_for_btn = {'bytes': img_bytes, 'name': title, 'btn_tag': ai_button_tag}
                        action_callback = lambda s, a, u: utils.confirm_and_run_ai_analysis(
                            image_bytes=u['bytes'], chart_name=u['name'],
                            ai_button_tag=u['btn_tag'], main_callbacks=_module_main_callbacks
                        )
                        dpg.add_button(label="💡 Analyze with AI", tag=ai_button_tag,
                                       user_data=user_data_for_btn, callback=action_callback)
            except Exception as e:
                dpg.add_text(f"차트 {i+1} 생성 실패.\n오류: {traceback.format_exc()}", color=(255, 0, 0))
            finally:
                plt.close('all')
                dpg.add_separator()

def _update_chart_config_state(sender, app_data, user_data):
    config_id, field = user_data["id"], user_data["field"]
    config = next((c for c in _chart_configs if c["id"] == config_id), None)
    if config:
        config[field] = app_data
        if field in ["df_source", "chart_type"]:
            if field == "df_source": config.update({"x_axis": "", "y_axis": "", "hue": "", "col_facet": "", "label_by": "None"})
            _populate_chart_config_table()

def _delete_chart_config_row(sender, app_data, user_data):
    config_id_to_delete = user_data
    _chart_configs[:] = [config for config in _chart_configs if config["id"] != config_id_to_delete]
    _populate_chart_config_table()

def _populate_chart_config_table():
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_S9_CHART_CONFIG_TABLE): return
    dpg.delete_item(TAG_S9_CHART_CONFIG_TABLE, children_only=True)
    headers = ["DF Source", "Chart Type", "X-Axis", "Y-Axis", "Hue", "Col Facet", "Width", "Height", "Action"]
    for header in headers: dpg.add_table_column(label=header, parent=TAG_S9_CHART_CONFIG_TABLE)

    for config in _chart_configs:
        with dpg.table_row(parent=TAG_S9_CHART_CONFIG_TABLE):
            dpg.add_combo(list(_available_dfs.keys()), default_value=config["df_source"], width=-1, callback=_update_chart_config_state, user_data={"id": config["id"], "field": "df_source"})
            cols = list(_available_dfs[config["df_source"]].columns) if config["df_source"] in _available_dfs else []
            all_cols = [""] + cols
            dpg.add_combo(CHART_TYPES, default_value=config["chart_type"], width=-1, callback=_update_chart_config_state, user_data={"id": config["id"], "field": "chart_type"})
            
            rules = CHART_UI_RULES.get(config["chart_type"], {})
            fields_ui = {"x_axis": all_cols, "y_axis": all_cols, "hue": all_cols, "col_facet": all_cols}
            for field, items in fields_ui.items():
                is_enabled = rules.get(field, True) if field != "x_axis" else True
                dpg.add_combo(items, default_value=config.get(field, ""), width=-1, enabled=is_enabled, callback=_update_chart_config_state, user_data={"id": config["id"], "field": field})

            dpg.add_input_int(default_value=config["width"], width=-1, step=0, callback=_update_chart_config_state, user_data={"id": config["id"], "field": "width"})
            dpg.add_input_int(default_value=config["height"], width=-1, step=0, callback=_update_chart_config_state, user_data={"id": config["id"], "field": "height"})
            dpg.add_button(label="Del", user_data=config["id"], callback=_delete_chart_config_row)

def _add_chart_config_row():
    new_config = PLOT_CONFIG_DEFAULTS.copy()
    new_config["id"] = str(uuid.uuid4())
    current_df_key = '★ Current Working DF'
    if current_df_key in _available_dfs:
        new_config["df_source"] = current_df_key
    if _module_main_callbacks and 'get_selected_target_variable' in _module_main_callbacks:
        target_var = _module_main_callbacks['get_selected_target_variable']()
        if target_var and new_config["df_source"] and target_var in _available_dfs[new_config["df_source"]].columns:
            new_config["y_axis"] = target_var
    _chart_configs.append(new_config)
    _populate_chart_config_table()

# --- MAIN UI FUNCTIONS ---

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    main_callbacks['register_step_group_tag'](step_name, TAG_S9_GROUP)

    with dpg.group(tag=TAG_S9_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---"); dpg.add_separator()
        with dpg.tab_bar(tag=TAG_S9_MAIN_TAB_BAR):
            with dpg.tab(label="Data Viewer", tag=TAG_S9_DATA_VIEWER_TAB):
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=250, border=True):
                        dpg.add_input_text(label="Search", tag=TAG_S9_DF_SEARCH_INPUT, callback=_update_df_list, width=-1)
                        with dpg.table(header_row=False, tag=TAG_S9_DF_TABLE, policy=dpg.mvTable_SizingStretchProp, scrollY=True): dpg.add_table_column()
                    with dpg.child_window(border=True):
                        dpg.add_text("Select a DataFrame to preview.", tag=TAG_S9_PREVIEW_TEXT); dpg.add_separator()
                        with dpg.table(tag=TAG_S9_PREVIEW_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit, scrollY=True, scrollX=True): pass
            with dpg.tab(label="Chart Configuration", tag=TAG_S9_CHART_CONFIG_TAB):
                dpg.add_button(label="Add Chart", width=-1, callback=_add_chart_config_row)
                with dpg.child_window(border=True, height=500):
                    with dpg.table(tag=TAG_S9_CHART_CONFIG_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True): pass
                dpg.add_button(label="Generate All Charts", width=-1, height=30, callback=_clear_and_generate_charts)
            with dpg.tab(label="Chart Display", tag=TAG_S9_CHART_DISPLAY_TAB):
                with dpg.child_window(tag=TAG_S9_CHART_DISPLAY_WINDOW, border=True, no_scrollbar=False):
                    dpg.add_text("Click 'Generate All Charts' in the configuration tab.")

    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui():
    global _available_dfs
    if not _module_main_callbacks: return
    _available_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    _update_df_list()
    if dpg.does_item_exist(TAG_S9_DF_TABLE) and not dpg.get_item_children(TAG_S9_DF_TABLE, 1):
        dpg.set_value(TAG_S9_PREVIEW_TEXT, "No DataFrames available.")
        if _util_funcs and 'create_table_with_data' in _util_funcs:
            _util_funcs['create_table_with_data'](TAG_S9_PREVIEW_TABLE, pd.DataFrame())
    _populate_chart_config_table()

def reset_state():
    global _available_dfs, _chart_configs, _texture_tags
    _available_dfs, _chart_configs, _texture_tags = {}, [], []
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_S9_GROUP):
        if dpg.does_item_exist(TAG_S9_DF_SEARCH_INPUT): dpg.set_value(TAG_S9_DF_SEARCH_INPUT, "")
        _populate_chart_config_table()
        if dpg.does_item_exist(TAG_S9_CHART_DISPLAY_WINDOW):
            dpg.delete_item(TAG_S9_CHART_DISPLAY_WINDOW, children_only=True)
            dpg.add_text("Click 'Generate All Charts' in the configuration tab.", parent=TAG_S9_CHART_DISPLAY_WINDOW)
        update_ui()

# step_09_data_viewer.py 파일에 추가

def get_settings_for_saving() -> dict:
    """현재 차트 구성을 저장용 딕셔너리로 반환합니다."""
    return {"chart_configs": _chart_configs}

def apply_settings_and_process(settings: dict, main_callbacks: dict):
    """불러온 설정을 적용합니다."""
    global _chart_configs, _module_main_callbacks
    if not _module_main_callbacks:
        _module_main_callbacks = main_callbacks

    # 설정 파일에서 차트 구성 목록을 복원합니다.
    _chart_configs = settings.get("chart_configs", [])

    # 불러온 구성을 UI 테이블에 반영합니다.
    if dpg.is_dearpygui_running():
        _populate_chart_config_table()