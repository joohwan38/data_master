# << MODIFIED FILE >>: step_09_data_viewer.py

import dearpygui.dearpygui as dpg
import pandas as pd
from typing import Dict, Optional, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

# --- DPG Tags for Step 9 ---
TAG_S9_GROUP = "step9_data_viewer_group"
TAG_S9_MAIN_TAB_BAR = "step9_main_tab_bar"

# Tab 1: Data Viewer
TAG_S9_DATA_VIEWER_TAB = "step9_data_viewer_tab"
TAG_S9_DF_TABLE = "step9_df_selectable_table"
TAG_S9_DF_SEARCH_INPUT = "step9_df_search_input"
TAG_S9_PREVIEW_TABLE = "step9_preview_table"
TAG_S9_PREVIEW_TEXT = "step9_preview_text"

# Tab 2: Chart Configuration
TAG_S9_CHART_CONFIG_TAB = "step9_chart_config_tab"
TAG_S9_CHART_CONFIG_TABLE = "step9_chart_config_table"

# Tab 3: Chart Display
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
    "hue": "", "col_facet": "", "label_by": "None", "id": ""
}
# 차트 타입별 UI 요소 활성화/비활성화 규칙
CHART_UI_RULES = {
    "lineplot":    {"y_axis": True,  "hue": True,  "col_facet": True, "label_by": True},
    "scatterplot": {"y_axis": True,  "hue": True,  "col_facet": True, "label_by": True},
    "barplot":     {"y_axis": True,  "hue": True,  "col_facet": True, "label_by": False},
    "histplot":    {"y_axis": False, "hue": True,  "col_facet": True, "label_by": False},
    "countplot":   {"y_axis": False, "hue": True,  "col_facet": True, "label_by": False},
    "boxplot":     {"y_axis": True,  "hue": True,  "col_facet": True, "label_by": False},
    "violinplot":  {"y_axis": True,  "hue": True,  "col_facet": True, "label_by": False},
}


def _on_df_select(sender, app_data, user_data):
    """(탭 1) 테이블에서 DF 선택 시 호출되는 콜백."""
    if not app_data: return

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
    """(탭 1) DF 목록을 필터링하여 테이블에 표시합니다."""
    search_term = dpg.get_value(TAG_S9_DF_SEARCH_INPUT).lower()
    df_names = list(_available_dfs.keys())
    filtered_names = [name for name in df_names if search_term in name.lower()] if search_term else df_names

    if dpg.does_item_exist(TAG_S9_DF_TABLE):
        dpg.delete_item(TAG_S9_DF_TABLE, children_only=True)
        dpg.add_table_column(parent=TAG_S9_DF_TABLE)
        for name in filtered_names:
            with dpg.table_row(parent=TAG_S9_DF_TABLE):
                dpg.add_selectable(label=name, user_data=name, callback=_on_df_select, span_columns=True)

def _add_chart_config_row():
    """(탭 2) 차트 설정 테이블에 새로운 행을 추가합니다."""
    new_config = PLOT_CONFIG_DEFAULTS.copy()
    new_config["id"] = str(uuid.uuid4())
    _chart_configs.append(new_config)
    _populate_chart_config_table()

def _delete_chart_config_row(sender, app_data, user_data):
    """(탭 2) 차트 설정 행을 삭제합니다."""
    config_id_to_delete = user_data
    _chart_configs[:] = [config for config in _chart_configs if config["id"] != config_id_to_delete]
    _populate_chart_config_table()
    
def _update_chart_config_state(sender, app_data, user_data):
    """(탭 2) UI 변경사항을 _chart_configs 상태에 반영합니다."""
    config_id = user_data["id"]
    field = user_data["field"]
    config = next((c for c in _chart_configs if c["id"] == config_id), None)
    if config:
        config[field] = app_data
        # DF 소스나 차트 타입이 변경되면 UI 규칙을 다시 적용
        if field in ["df_source", "chart_type"]:
            _populate_chart_config_table()

def _populate_chart_config_table():
    """(탭 2) _chart_configs 상태를 기반으로 설정 테이블 UI를 다시 그립니다."""
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_S9_CHART_CONFIG_TABLE):
        return

    dpg.delete_item(TAG_S9_CHART_CONFIG_TABLE, children_only=True)

    headers = ["DF Source", "Chart Type", "X-Axis", "Y-Axis", "Hue", "Col Facet", "Label By", "Action"]
    widths = [0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05]
    for i, header in enumerate(headers):
        dpg.add_table_column(label=header, parent=TAG_S9_CHART_CONFIG_TABLE, width_stretch=(widths[i] > 0), init_width_or_weight=widths[i])

    df_source_list = list(_available_dfs.keys())

    for i, config in enumerate(_chart_configs):
        with dpg.table_row(parent=TAG_S9_CHART_CONFIG_TABLE):
            # DF 소스
            dpg.add_combo(df_source_list, default_value=config["df_source"], width=-1,
                          callback=_update_chart_config_state, user_data={"id": config["id"], "field": "df_source"})

            # 컬럼 목록 (선택된 DF 기준)
            cols = list(_available_dfs[config["df_source"]].columns) if config["df_source"] in _available_dfs else []
            empty_option = [""]
            all_cols = empty_option + cols

            # 차트 타입
            dpg.add_combo(CHART_TYPES, default_value=config["chart_type"], width=-1,
                          callback=_update_chart_config_state, user_data={"id": config["id"], "field": "chart_type"})

            # UI 규칙 가져오기
            rules = CHART_UI_RULES.get(config["chart_type"], CHART_UI_RULES["scatterplot"])

            # X, Y, Hue, Col, Label By
            fields_ui = {
                "x_axis": {"items": all_cols, "enabled": True},
                "y_axis": {"items": all_cols, "enabled": rules["y_axis"]},
                "hue":    {"items": all_cols, "enabled": rules["hue"]},
                "col_facet": {"items": all_cols, "enabled": rules["col_facet"]},
                "label_by": {"items": ["None"] + cols, "enabled": rules["label_by"]}
            }
            for field, ui_props in fields_ui.items():
                dpg.add_combo(ui_props["items"], default_value=config[field], width=-1, enabled=ui_props["enabled"],
                              callback=_update_chart_config_state, user_data={"id": config["id"], "field": field})

            # Action
            dpg.add_button(label="Del", user_data=config["id"], callback=_delete_chart_config_row)

def _clear_and_generate_charts():
    """(탭 3) 차트 표시 영역을 비우고 새로 생성합니다."""
    global _texture_tags
    # 이전 텍스처 삭제
    for tag in _texture_tags:
        if dpg.does_item_exist(tag): dpg.delete_item(tag)
    _texture_tags.clear()

    # 차트 표시 창 비우기
    if dpg.does_item_exist(TAG_S9_CHART_DISPLAY_WINDOW):
        dpg.delete_item(TAG_S9_CHART_DISPLAY_WINDOW, children_only=True)

    if not _chart_configs:
        dpg.add_text("No charts configured.", parent=TAG_S9_CHART_DISPLAY_WINDOW)
        return

    plot_func = _util_funcs.get('plot_to_dpg_texture') if _util_funcs else None
    if not plot_func:
        dpg.add_text("Plotting utility not found.", parent=TAG_S9_CHART_DISPLAY_WINDOW)
        return

    # 각 설정에 따라 차트 생성
    for i, config in enumerate(_chart_configs):
        try:
            df_name = config.get("df_source")
            if not df_name or df_name not in _available_dfs:
                dpg.add_text(f"Chart {i+1}: DataFrame '{df_name}' not found. Skipping.", parent=TAG_S9_CHART_DISPLAY_WINDOW)
                continue

            df = _available_dfs[df_name]
            chart_type = config.get("chart_type")
            x = config.get("x_axis") or None
            y = config.get("y_axis") or None
            hue = config.get("hue") or None
            col_facet = config.get("col_facet") or None
            label_by = config.get("label_by") if config.get("label_by") != "None" else None

            if not x:
                dpg.add_text(f"Chart {i+1}: X-axis not specified. Skipping.", parent=TAG_S9_CHART_DISPLAY_WINDOW)
                continue

            # Seaborn FacetGrid 사용
            g = sns.FacetGrid(df, col=col_facet, hue=hue, sharex=False, sharey=False, col_wrap=4 if col_facet else None)

            # 차트 종류에 따른 매핑
            if chart_type == "lineplot": g.map(sns.lineplot, x, y)
            elif chart_type == "scatterplot": g.map(sns.scatterplot, x, y, alpha=0.7)
            elif chart_type == "barplot": g.map(sns.barplot, x, y)
            elif chart_type == "histplot": g.map(sns.histplot, x, kde=True)
            elif chart_type == "countplot": g.map(sns.countplot, x)
            elif chart_type == "boxplot": g.map(sns.boxplot, x, y)
            elif chart_type == "violinplot": g.map(sns.violinplot, x, y)
            
            # 라벨링 기능 (scatterplot, lineplot만 지원)
            if label_by and chart_type in ["scatterplot", "lineplot"]:
                def annotate_points(x_data, y_data, label_data, **kwargs):
                    df_subset = pd.DataFrame({'x': x_data, 'y': y_data, 'label': label_data}).dropna().head(50)
                    for idx, point in df_subset.iterrows():
                        plt.text(point['x'], point['y'], str(point['label']), fontsize=8, alpha=0.8)

                g.map(annotate_points, x, y, label_by)

            g.add_legend()
            g.fig.suptitle(f"Chart {i+1}: {chart_type.capitalize()} on '{df_name}'", y=1.03)
            plt.tight_layout()

            # DPG 텍스처로 변환 및 표시
            tex_tag, w, h, _ = plot_func(g.fig)
            if tex_tag:
                _texture_tags.append(tex_tag)
                dpg.add_image(tex_tag, width=w, height=h, parent=TAG_S9_CHART_DISPLAY_WINDOW)
                dpg.add_separator(parent=TAG_S9_CHART_DISPLAY_WINDOW)

        except Exception as e:
            error_text = f"Failed to generate Chart {i+1} ({config.get('chart_type')} on '{config.get('df_source')}').\nError: {str(e)}"
            dpg.add_text(error_text, color=(255, 0, 0), parent=TAG_S9_CHART_DISPLAY_WINDOW)
            dpg.add_separator(parent=TAG_S9_CHART_DISPLAY_WINDOW)
        finally:
            plt.close('all') # 모든 Matplotlib figure 닫기


def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """Step 9의 UI를 생성합니다."""
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    main_callbacks['register_step_group_tag'](step_name, TAG_S9_GROUP)

    with dpg.group(tag=TAG_S9_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        
        with dpg.tab_bar(tag=TAG_S9_MAIN_TAB_BAR):
            # --- 탭 1: 데이터 뷰어 ---
            with dpg.tab(label="Data Viewer", tag=TAG_S9_DATA_VIEWER_TAB):
                with dpg.group(horizontal=True):
                    with dpg.child_window(width=250, border=True):
                        dpg.add_input_text(label="Search", tag=TAG_S9_DF_SEARCH_INPUT, callback=_update_df_list, width=-1)
                        with dpg.table(header_row=False, tag=TAG_S9_DF_TABLE, policy=dpg.mvTable_SizingStretchProp, scrollY=True):
                            dpg.add_table_column()
                    with dpg.child_window(border=True):
                        dpg.add_text("Select a DataFrame to preview.", tag=TAG_S9_PREVIEW_TEXT)
                        dpg.add_separator()
                        with dpg.table(tag=TAG_S9_PREVIEW_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingFixedFit, scrollY=True, scrollX=True):
                            pass

            # --- 탭 2: 차트 설정 ---
            with dpg.tab(label="Chart Configuration", tag=TAG_S9_CHART_CONFIG_TAB):
                dpg.add_text("Define charts to be generated. Click 'Generate All Charts' to view them in the 'Chart Display' tab.")
                dpg.add_button(label="Add Chart Configuration", width=-1, callback=_add_chart_config_row)
                with dpg.child_window(border=True, height=400):
                    with dpg.table(tag=TAG_S9_CHART_CONFIG_TABLE, header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, scrollY=True):
                        pass
                dpg.add_button(label="Generate All Charts", width=-1, height=30, callback=_clear_and_generate_charts)

            # --- 탭 3: 차트 표시 ---
            with dpg.tab(label="Chart Display", tag=TAG_S9_CHART_DISPLAY_TAB):
                dpg.add_text("Generated charts will appear here. Scroll to see all charts.")
                with dpg.child_window(tag=TAG_S9_CHART_DISPLAY_WINDOW, border=True):
                    dpg.add_text("Click 'Generate All Charts' in the configuration tab.")

    main_callbacks['register_module_updater'](step_name, update_ui)

def update_ui():
    """Step 9 UI의 내용을 최신 상태로 업데이트합니다."""
    global _available_dfs
    if not _module_main_callbacks: return
    _available_dfs = _module_main_callbacks['get_all_available_dfs']()
    
    # 데이터 뷰어 탭 업데이트
    _update_df_list()
    table_items = dpg.get_item_children(TAG_S9_DF_TABLE, 1) if dpg.does_item_exist(TAG_S9_DF_TABLE) else []
    if not table_items:
        dpg.set_value(TAG_S9_PREVIEW_TEXT, "No DataFrames available.")
        if _util_funcs and 'create_table_with_data' in _util_funcs:
            _util_funcs['create_table_with_data'](TAG_S9_PREVIEW_TABLE, pd.DataFrame())

    # 차트 설정 탭 업데이트
    _populate_chart_config_table()

def reset_state():
    """모듈의 상태를 초기화합니다."""
    global _available_dfs, _chart_configs, _texture_tags
    _available_dfs, _chart_configs, _texture_tags = {}, [], []
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_S9_GROUP):
        if dpg.does_item_exist(TAG_S9_DF_SEARCH_INPUT):
            dpg.set_value(TAG_S9_DF_SEARCH_INPUT, "")
        _populate_chart_config_table() # 빈 테이블로 초기화
        if dpg.does_item_exist(TAG_S9_CHART_DISPLAY_WINDOW):
            dpg.delete_item(TAG_S9_CHART_DISPLAY_WINDOW, children_only=True)
            dpg.add_text("Click 'Generate All Charts' in the configuration tab.", parent=TAG_S9_CHART_DISPLAY_WINDOW)
        update_ui()