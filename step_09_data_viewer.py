# << MODIFIED FILE >>: step_09_data_viewer.py

import dearpygui.dearpygui as dpg
import pandas as pd
from typing import Dict, Optional, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import numpy as np
import traceback

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
        dpg.add_table_column(parent=TAG_S9_DF_TABLE)
        for name in filtered_names:
            with dpg.table_row(parent=TAG_S9_DF_TABLE):
                dpg.add_selectable(label=name, user_data=name, callback=_on_df_select, span_columns=True)

def _clear_and_generate_charts():
    global _texture_tags
    for tag in _texture_tags:
        if dpg.does_item_exist(tag): dpg.delete_item(tag)
    _texture_tags.clear()
    
    if not dpg.does_item_exist(TAG_S9_CHART_DISPLAY_WINDOW): return
    dpg.delete_item(TAG_S9_CHART_DISPLAY_WINDOW, children_only=True)

    if not _chart_configs:
        dpg.add_text("No charts configured.", parent=TAG_S9_CHART_DISPLAY_WINDOW); return

    plot_func = _util_funcs.get('plot_to_dpg_texture')
    if not plot_func:
        dpg.add_text("Error: Plotting utility function is not available.", parent=TAG_S9_CHART_DISPLAY_WINDOW, color=(255,0,0))
        return

    for i, config in enumerate(_chart_configs):
        with dpg.group(parent=TAG_S9_CHART_DISPLAY_WINDOW):
            try:
                df_name, chart_type = config.get("df_source"), config.get("chart_type")
                x, y, hue, col_facet = (config.get(k) or None for k in ["x_axis", "y_axis", "hue", "col_facet"])
                width, height = config.get("width", -1), config.get("height", 350)
                
                if not df_name or df_name not in _available_dfs: raise ValueError(f"DF '{df_name}' not found.")
                df = _available_dfs[df_name].copy()
                if not x or x not in df.columns: raise ValueError("X-axis not specified.")
                if CHART_UI_RULES.get(chart_type, {})["y_axis"] and (not y or y not in df.columns): raise ValueError("Y-axis required.")
                if hue and df[hue].nunique() > 10: raise ValueError(f"Hue ('{hue}') has > 10 unique values.")
                
                title = f"Chart {i+1}: {chart_type.capitalize()} on '{df_name}'"
                dpg.add_text(title)
                dpg.add_spacer(height=5, parent=TAG_S9_CHART_DISPLAY_WINDOW)

                # --- Seaborn을 사용한 통합 플롯 로직 ---
                plot_function_map = {
                    "lineplot": sns.lineplot, "scatterplot": sns.scatterplot,
                    "barplot": sns.barplot, "histplot": sns.histplot,
                    "countplot": sns.countplot, "boxplot": sns.boxplot,
                    "violinplot": sns.violinplot
                }
                plot_function = plot_function_map.get(chart_type)
                if not plot_function: raise ValueError(f"Unknown chart type: {chart_type}")

                plot_kwargs = {'data': df, 'x': x}
                if y and CHART_UI_RULES.get(chart_type,{})["y_axis"]: plot_kwargs['y'] = y
                if hue: plot_kwargs['hue'] = hue

                # FacetGrid를 사용해야 하는 경우
                if col_facet and col_facet in df.columns:
                    g = sns.FacetGrid(df, col=col_facet, hue=hue, col_wrap=4, sharex=False, sharey=False)
                    # FacetGrid.map()은 y축 인자를 명시적으로 요구할 수 있음
                    map_args = [x]
                    if y and CHART_UI_RULES.get(chart_type,{})["y_axis"]: map_args.append(y)
                    g.map(plot_function, *map_args)
                    final_fig = g.fig
                else: # 단일 차트
                    fig_w = 8 if width == -1 else max(5, width / 80)
                    fig_h = 6 if height == -1 else max(4, height / 80)
                    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                    plot_function(ax=ax, **plot_kwargs)
                    final_fig = fig

                final_fig.suptitle(title, y=1.02)
                plt.tight_layout()
                
                tex_tag, w, h, _ = plot_func(final_fig)
                if tex_tag:
                    _texture_tags.append(tex_tag)
                    dpg.add_image(tex_tag, width=w, height=h)
                
            except Exception as e:
                dpg.add_text(f"Failed to generate Chart {i+1}.\nError: {traceback.format_exc()}", color=(255, 0, 0))
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
                with dpg.child_window(border=True, height=450):
                    with dpg.table(tag=TAG_S9_CHART_CONFIG_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True): pass
                dpg.add_button(label="Generate All Charts", width=-1, height=30, callback=_clear_and_generate_charts)
            with dpg.tab(label="Chart Display", tag=TAG_S9_CHART_DISPLAY_TAB):
                with dpg.child_window(tag=TAG_S9_CHART_DISPLAY_WINDOW, border=True):
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