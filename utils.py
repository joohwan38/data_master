# utils.py

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from scipy import stats
import traceback 
import functools
import io
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:
    import ollama_analyzer 
except ImportError:
    print("Warning: utils.py - ollama_analyzer module not found. AI analysis features will be unavailable.")
    ollama_analyzer = None

MIN_COL_WIDTH = 50
MAX_COL_WIDTH = 300
CELL_PADDING = 20
TARGET_DATA_CHARS = 25
ELLIPSIS = "..."

UTL_CONFIRMATION_MODAL_TAG = "utl_reusable_confirmation_modal"
UTL_CONFIRMATION_TEXT_TAG = "utl_reusable_confirmation_text"
_yes_callback_storage = None


def plot_to_dpg_texture(fig: Figure, desired_dpi: int = 90) -> Tuple[Optional[str], int, int, Optional[bytes]]:
    """
    Converts a Matplotlib figure to a DearPyGui texture.
    Returns the texture tag, width, height, and raw PNG bytes.
    """
    if not fig:
        return None, 0, 0, None
    try:
        # Save figure to an in-memory buffer
        with io.BytesIO() as buf:
            fig.savefig(buf, format='png', dpi=desired_dpi)
            buf.seek(0)
            img_bytes = buf.getvalue()

        # Use PIL to get dimensions and convert to RGBA
        with Image.open(io.BytesIO(img_bytes)) as img:
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            width, height = img.size
            raw_data = np.array(img, dtype=np.float32) / 255.0
        
        if not dpg.is_dearpygui_running():
            return "dummy_texture_not_created", width, height, img_bytes

        # Generate a unique tag for the texture
        texture_tag = dpg.generate_uuid()
        
        # Add texture to a registry
        dpg.add_raw_texture(
            width=width,
            height=height,
            default_value=raw_data.flatten(), # Flatten the array
            format=dpg.mvFormat_Float_rgba,
            tag=texture_tag,
            parent="primary_texture_registry" # Use the main texture registry
        )
        
        return texture_tag, width, height, img_bytes
    except Exception as e:
        print(f"Error converting plot to DPG texture: {e}")
        traceback.print_exc()
        return None, 0, 0, None


def _internal_yes_callback_handler(sender, app_data, user_data_modal_tag):
    global _yes_callback_storage
    if dpg.does_item_exist(user_data_modal_tag):
        dpg.configure_item(user_data_modal_tag, show=False)
    if _yes_callback_storage:
        _yes_callback_storage()
        _yes_callback_storage = None

def _internal_no_callback_handler(sender, app_data, user_data_modal_tag):
    global _yes_callback_storage
    if dpg.does_item_exist(user_data_modal_tag):
        dpg.configure_item(user_data_modal_tag, show=False)
    _yes_callback_storage = None

def show_confirmation_modal(title: str, message: str,
                            yes_callback: callable,
                            modal_tag: str = UTL_CONFIRMATION_MODAL_TAG,
                            text_tag: str = UTL_CONFIRMATION_TEXT_TAG):
    global _yes_callback_storage
    if not dpg.is_dearpygui_running():
        print(f"CONFIRMATION (Non-DPG): {title} - {message}. Assuming 'Yes'.")
        if yes_callback:
            yes_callback()
        return

    _yes_callback_storage = yes_callback

    vp_w = dpg.get_viewport_width()
    vp_h = dpg.get_viewport_height()
    modal_w = 450

    if not dpg.does_item_exist(modal_tag):
        with dpg.window(label=title, modal=True, show=False, tag=modal_tag,
                        no_close=True, width=modal_w, autosize=True, no_saved_settings=True):
            dpg.add_text("", tag=text_tag, wrap=modal_w - 30)
            dpg.add_spacer(height=15)
            with dpg.group(horizontal=True):
                btn_width = 100
                num_buttons = 2
                default_item_spacing_x = 8.0
                window_padding_x = 8.0 * 2
                available_space_for_buttons_and_spacing = modal_w - window_padding_x
                total_button_width = btn_width * num_buttons
                spacing_between_buttons = default_item_spacing_x * (num_buttons -1) if num_buttons > 1 else 0
                total_side_padding = available_space_for_buttons_and_spacing - total_button_width - spacing_between_buttons
                single_side_spacer_width = total_side_padding / 2
                if single_side_spacer_width < 5: single_side_spacer_width = 5
                dpg.add_spacer(width=int(single_side_spacer_width))
                dpg.add_button(label="Proceed", width=btn_width, user_data=modal_tag,
                               callback=_internal_yes_callback_handler)
                dpg.add_button(label="Cancel", width=btn_width, user_data=modal_tag,
                               callback=_internal_no_callback_handler)
    else:
        dpg.configure_item(modal_tag, label=title)

    dpg.set_value(text_tag, message)
    dpg.configure_item(modal_tag, show=True)
    current_modal_height = dpg.get_item_height(modal_tag) if dpg.does_item_exist(modal_tag) else 200
    pos_x = (vp_w - modal_w) // 2
    pos_y = (vp_h - current_modal_height) // 2 if current_modal_height > 0 else vp_h // 3
    dpg.set_item_pos(modal_tag, [max(0, pos_x), max(0, pos_y)])

# --- AI 분석 관련 유틸리티 함수 ---
def generic_ai_analysis_streaming_handler(
    image_bytes: bytes,
    chart_name: str,
    ai_button_tag: str,
    main_callbacks: dict,
    loading_label: str = "Analyzing...",
    finished_label: str = "💡 Analyze with AI"
):
    """AI 이미지 분석을 요청하고 결과를 스트리밍하여 로그에 기록하는 일반 함수"""
    if not ollama_analyzer:
        error_msg_no_analyzer = f"Ollama analyzer module is not available. Cannot perform AI analysis for '{chart_name}'."
        print(f"Error: {error_msg_no_analyzer}")
        if 'add_ai_log' in main_callbacks:
            main_callbacks['add_ai_log'](error_msg_no_analyzer, chart_name, mode="new_log_entry")
        if dpg.is_dearpygui_running() and dpg.does_item_exist(ai_button_tag):
            dpg.configure_item(ai_button_tag, enabled=True, label=finished_label)
        return

    if not image_bytes:
        no_image_msg = f"분석할 이미지가 없습니다 ({chart_name})."
        if 'add_ai_log' in main_callbacks:
            main_callbacks['add_ai_log'](no_image_msg, chart_name, mode="new_log_entry")
        if dpg.is_dearpygui_running() and dpg.does_item_exist(ai_button_tag):
            dpg.configure_item(ai_button_tag, enabled=True, label=finished_label)
        return

    if dpg.is_dearpygui_running() and dpg.does_item_exist(ai_button_tag):
        dpg.configure_item(ai_button_tag, enabled=False, label=loading_label)

    try:
        first_chunk_processed = False
        for chunk in ollama_analyzer.analyze_image_with_llava(image_bytes, chart_name): # type: ignore
            if not chunk and not first_chunk_processed:
                if 'add_ai_log' in main_callbacks:
                    main_callbacks['add_ai_log']("(AI 응답 스트림 시작...)", chart_name, mode="stream_start_entry")
                first_chunk_processed = True
                if not chunk.strip(): continue

            if not first_chunk_processed:
                if 'add_ai_log' in main_callbacks:
                    main_callbacks['add_ai_log'](chunk if chunk.strip() else "(빈 청크 수신)", chart_name, mode="stream_start_entry")
                first_chunk_processed = True
            else:
                if chunk.strip() and 'add_ai_log' in main_callbacks:
                    main_callbacks['add_ai_log'](chunk, chart_name, mode="stream_chunk_append")
        
        if not first_chunk_processed:
            if 'add_ai_log' in main_callbacks:
                main_callbacks['add_ai_log']("(AI로부터 응답이 없습니다)", chart_name, mode="stream_start_entry")
        
        if 'add_ai_log' in main_callbacks:
            main_callbacks['add_ai_log']("\n(AI 분석 스트림 완료)", chart_name, mode="stream_chunk_append")

    except Exception as e_analysis:
        error_message = f"AI 분석 중 예외 발생 ({chart_name}): {str(e_analysis)}"
        print(f"[Error] {error_message}\n{traceback.format_exc()}") #
        if 'add_ai_log' in main_callbacks:
            main_callbacks['add_ai_log'](error_message, chart_name, mode="new_log_entry")
    finally:
        if dpg.is_dearpygui_running() and dpg.does_item_exist(ai_button_tag):
            dpg.configure_item(ai_button_tag, enabled=True, label=finished_label)

def confirm_and_run_ai_analysis(
    image_bytes: bytes,
    chart_name: str,
    ai_button_tag: str,
    main_callbacks: dict
):
    """사용자에게 AI 분석 실행 여부를 확인하고, 동의 시 일반 분석 핸들러를 호출합니다."""
    if 'get_util_funcs' in main_callbacks:
        util_funcs = main_callbacks['get_util_funcs']()
        if 'show_confirmation_modal' in util_funcs:
            yes_action = functools.partial(
                generic_ai_analysis_streaming_handler,
                image_bytes,
                chart_name,
                ai_button_tag,
                main_callbacks
            ) 
            util_funcs['show_confirmation_modal'](
                title="AI Analysis Confirmation",
                message=f"'{chart_name}'에 대한 AI 분석을 진행하시겠습니까?\n(응답은 스트리밍됩니다)",
                yes_callback=yes_action
            ) 
        else:
            print(f"Warning: Confirmation modal utility not found. Running AI analysis for '{chart_name}' directly.")
            generic_ai_analysis_streaming_handler(image_bytes, chart_name, ai_button_tag, main_callbacks)
    else:
        print(f"Warning: Utility functions not accessible. Running AI analysis for '{chart_name}' directly.")
        generic_ai_analysis_streaming_handler(image_bytes, chart_name, ai_button_tag, main_callbacks)

def icon_button(label: str, icon: str, width: int = -1, height: int = 0, **kwargs):
    button_label = f"{icon} {label}" if label else icon
    return dpg.add_button(label=button_label, width=width, height=height, **kwargs)

def calculate_feature_target_relevance(
    df: pd.DataFrame,
    target_var: str,
    target_var_type: str,
    features_to_analyze: List[str],
    main_app_callbacks: Optional[Dict] = None
) -> List[Tuple[str, float]]:
    if df is None or target_var not in df.columns or not features_to_analyze:
        return []
    scores = []
    s1_analysis_types = {}
    if main_app_callbacks and 'get_column_analysis_types' in main_app_callbacks:
        s1_analysis_types = main_app_callbacks['get_column_analysis_types']()
    for feature_col in features_to_analyze:
        if feature_col == target_var or feature_col not in df.columns:
            continue
        feature_s1_type = s1_analysis_types.get(feature_col, str(df[feature_col].dtype))
        is_feature_numeric = ("Numeric" in feature_s1_type and "Binary" not in feature_s1_type) or \
                             (pd.api.types.is_numeric_dtype(df[feature_col].dtype) and df[feature_col].nunique() > 5)
        score = 0.0
        try:
            target_series_clean = df[target_var].dropna()
            feature_series_clean = df[feature_col].dropna()
            common_index = target_series_clean.index.intersection(feature_series_clean.index)
            if len(common_index) < 20:
                scores.append((feature_col, 0.0))
                continue
            aligned_target = target_series_clean.loc[common_index]
            aligned_feature = feature_series_clean.loc[common_index]
            if target_var_type == "Categorical" and is_feature_numeric: 
                target_categories_local = aligned_target.unique() 
                if len(target_categories_local) >= 2: 
                    grouped_feature_data_for_anova = [
                        aligned_feature[aligned_target == cat]
                        for cat in target_categories_local
                    ] 
                    valid_groups_anova = [g for g in grouped_feature_data_for_anova if len(g) >= 2] 
                    if len(valid_groups_anova) >= 2: 
                        f_val, p_val = stats.f_oneway(*valid_groups_anova) 
                        score = abs(f_val) if pd.notna(f_val) and np.isfinite(f_val) else 0.0 
            elif target_var_type == "Continuous" and is_feature_numeric: 
                if pd.api.types.is_numeric_dtype(aligned_target.dtype): 
                    corr_val = aligned_feature.corr(aligned_target) 
                    score = abs(corr_val) if pd.notna(corr_val) and np.isfinite(corr_val) else 0.0 
        except Exception as e_relevance:
            score = 0.0 
        if score > 1e-6:
            scores.append((feature_col, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _get_numeric_cols(df: pd.DataFrame, main_callbacks: Optional[Dict] = None) -> List[str]: # <<< [수정] main_callbacks 인자 추가
    if df is None: return []
    numeric_cols = []
    
    s1_analysis_types = {}
    if main_callbacks and 'get_column_analysis_types' in main_callbacks:
        s1_analysis_types = main_callbacks['get_column_analysis_types']()

    for col in df.select_dtypes(include=np.number).columns:
        # [핵심] Step 1에서 '분석 제외'로 설정된 타입은 건너뜁니다.
        if col in s1_analysis_types and s1_analysis_types[col] == "분석에서 제외 (Exclude)":
            continue
        numeric_cols.append(col)
        
    return numeric_cols

def _get_categorical_cols(df: pd.DataFrame, max_unique_for_cat: int = 35, main_callbacks: Optional[Dict] = None) -> List[str]: 
    if df is None: return [] 
    categorical_cols = [] 
    s1_analysis_types = {} 
    if main_callbacks and 'get_column_analysis_types' in main_callbacks: 
        s1_analysis_types = main_callbacks['get_column_analysis_types']() 

    for col in df.columns:
        # [핵심] Step 1에서 '분석 제외'로 설정된 타입은 건너뜁니다.
        if col in s1_analysis_types and s1_analysis_types[col] == "분석에서 제외 (Exclude)":
            continue

        if col in s1_analysis_types: 
            s1_type = s1_analysis_types[col]
            # [수정] "Text (" 조건을 좀 더 명확하게 하여 ID/Code 타입이 자동 포함되지 않도록 검토할 수 있습니다.
            # 우선은 기존 로직을 유지하되, 제외 로직이 먼저 실행됩니다.
            if any(cat_keyword in s1_type for cat_keyword in ["Categorical", "Text (", "Potentially Sensitive"]): 
                categorical_cols.append(col); continue 

def _guess_target_type(df: pd.DataFrame, column_name: str, step1_type_selections: dict = None) -> str: 
    if not column_name or df is None or column_name not in df.columns: return "Continuous" 
    series = df[column_name] 
    if step1_type_selections and column_name in step1_type_selections: 
        s1_type = step1_type_selections[column_name] 
        if any(k in s1_type for k in ["Text (", "Potentially Sensitive", "Categorical"]): return "Categorical" 
        if "Numeric" in s1_type: 
            return "Categorical" if "Binary" in s1_type or series.nunique(dropna=False) <= 5 else "Continuous" 
        if any(k in s1_type for k in ["Datetime", "Timedelta"]): return "Categorical" 
    if pd.api.types.is_categorical_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype): return "Categorical" 
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype): return "Categorical" 
    if pd.api.types.is_numeric_dtype(series.dtype): 
        return "Categorical" if series.nunique(dropna=False) <= 10 else "Continuous" 
    if pd.api.types.is_datetime64_any_dtype(series.dtype) or pd.api.types.is_timedelta64_dtype(series.dtype): return "Categorical" 
    return "Continuous" 

def get_safe_text_size(text: str, font=None, wrap_width: float = -1.0) -> tuple[int, int]:
    if not dpg.is_dearpygui_running(): return (len(str(text)) * 8, 16)
    try:
        size = dpg.get_text_size(text, font=font, wrap_width=wrap_width)
        return (int(size[0]), int(size[1])) if size and len(size) == 2 else (len(str(text)) * 8, 16)
    except Exception: return (len(str(text)) * 8, 16)

def format_text_for_display(text_val, max_chars=TARGET_DATA_CHARS) -> str:
    s_text = str(text_val)
    return s_text[:max_chars] + ELLIPSIS if len(s_text) > max_chars else s_text

def calculate_column_widths(df: pd.DataFrame, min_w=MIN_COL_WIDTH, max_w=MAX_COL_WIDTH, pad=CELL_PADDING, rows=20) -> dict:
    if df is None or df.empty: return {}
    col_widths = {}
    for col_name in df.columns:
        max_px = get_safe_text_size(format_text_for_display(str(col_name)))[0]
        sample = df[col_name].dropna().head(rows)
        
        if not sample.empty:
            # << CHANGED >>: 너비 측정 전, 화면에 표시될 값과 동일하게 포맷팅하는 로직 추가
            formatted_sample = []
            for x in sample:
                if pd.isna(x):
                    s_x = "NaN"
                # float, np.floating 타입인 경우 round() 적용
                elif isinstance(x, (float, np.floating)):
                    s_x = str(round(x, 2))
                else:
                    s_x = str(x)
                formatted_sample.append(s_x)
            
            # 포맷팅된 문자열 샘플을 기준으로 최대 너비 계산
            max_px_data = max(get_safe_text_size(format_text_for_display(s_x))[0] for s_x in formatted_sample)
            max_px = max(max_px, max_px_data)

        col_widths[col_name] = int(max(min_w, min(max_px + pad, max_w)))
    return col_widths

def add_dpg_histogram_series(y_tag: str, data: List[float], lbl: str, bins: int=-1, density: bool=False):
    if data: dpg.add_histogram_series(data, label=lbl, bins=bins, density=density, parent=y_tag)

def add_dpg_line_series(y_tag: str, x_data: List[float], y_data: List[float], lbl: str):
    if x_data and y_data: dpg.add_line_series(x_data, y_data, label=lbl, parent=y_tag)

def add_dpg_bar_series(y_tag: str, x_data: List[float], y_data: List[float], lbl: str):
    if x_data and y_data: dpg.add_bar_series(x_data, y_data, label=lbl, parent=y_tag)

def add_dpg_scatter_series(y_tag: str, x_data: List[float], y_data: List[float], lbl: Optional[str]=None):
    if x_data and y_data: dpg.add_scatter_series(x_data, y_data, label=lbl or "", parent=y_tag)

def add_dpg_heat_series(y_tag: str, data_flat: List[float], r: int, c: int, s_min: float, s_max: float, fmt: str='%.2f', b_min: Tuple[float,float]=(0.0,0.0), b_max: Optional[Tuple[float,float]]=None):
    if data_flat: dpg.add_heat_series(data_flat, r, c, scale_min=s_min, scale_max=s_max, format=fmt, parent=y_tag, bounds_min=b_min, bounds_max=b_max or (float(c),float(r)))

def create_dpg_heatmap_plot(parent: str, matrix: pd.DataFrame, title: str, h: int=450, cmap: int=dpg.mvPlotColormap_RdBu):
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(parent): return
    if matrix is None or matrix.empty: dpg.add_text(f"{title}: No data.", parent=parent); return
    r, c = matrix.shape
    if r == 0 or c == 0: dpg.add_text(f"{title}: Empty data (0 rows/cols).", parent=parent); return
    data_np = np.nan_to_num(matrix.values.flatten().astype(float), nan=0.0, posinf=1.0, neginf=-1.0) # type: ignore
    data_flat = data_np.tolist()
    if len(data_flat) != r * c: dpg.add_text(f"{title}: Data size mismatch.", parent=parent, color=(255,0,0)); return
    col_lbls, row_lbls = [str(x) for x in matrix.columns], [str(x) for x in matrix.index]
    p_tag, x_tag, y_tag, _ = create_dpg_plot_scaffold(parent, title, "", "", h=h, eq_asp=(r == c))
    dpg.bind_colormap(p_tag, cmap)
    if col_lbls and c > 0: dpg.set_axis_ticks(x_tag, tuple(zip(col_lbls, [i + 0.5 for i in range(c)])))
    if row_lbls and r > 0: dpg.set_axis_ticks(y_tag, tuple(zip(row_lbls, [i + 0.5 for i in range(r)])))
    s_min, s_max = -1.0, 1.0
    actual_min, actual_max = data_np.min(), data_np.max()
    if actual_min == actual_max: s_min, s_max = actual_min - 0.5 if actual_min != 0 else -0.5, actual_max + 0.5 if actual_max != 0 else 0.5
    elif cmap in [dpg.mvPlotColormap_RdBu, dpg.mvPlotColormap_Spectral, dpg.mvPlotColormap_PiYG, dpg.mvPlotColormap_BrBG]:
        abs_val = max(abs(actual_min), abs(actual_max))
        s_min, s_max = -abs_val if abs_val != 0 else -0.5, abs_val if abs_val != 0 else 0.5
    else: s_min, s_max = actual_min, actual_max
    add_dpg_heat_series(y_tag, data_flat, r, c, float(s_min), float(s_max))

def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float: 
    if x is None or y is None or x.empty or y.empty: return 0.0 
    try:
        temp_df = pd.DataFrame({'x': x, 'y': y}).dropna() 
        if temp_df.empty or temp_df['x'].nunique() < 1 or temp_df['y'].nunique() < 1: return 0.0 
        confusion_matrix = pd.crosstab(temp_df['x'], temp_df['y']) 
        if confusion_matrix.empty or confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2: return 0.0 
        chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0] 
        n = confusion_matrix.sum().sum() 
        if n == 0: return 0.0 
        phi2 = chi2 / n 
        r_rows, k_cols = confusion_matrix.shape 
        phi2corr = max(0, phi2 - ((k_cols - 1) * (r_rows - 1)) / (n - 1 if n > 1 else 1)) 
        rcorr = r_rows - (((r_rows - 1)**2) / (n - 1 if n > 1 else 1) if r_rows > 1 else 0) 
        kcorr = k_cols - (((k_cols - 1)**2) / (n - 1 if n > 1 else 1) if k_cols > 1 else 0) 
        denominator = min((kcorr - 1 if kcorr > 1 else 0), (rcorr - 1 if rcorr > 1 else 0)) 
        return np.sqrt(phi2corr / denominator) if denominator != 0 else 0.0 
    except Exception: return 0.0 

UTL_REUSABLE_ALERT_MODAL_TAG = "utl_reusable_alert_modal"
UTL_REUSABLE_ALERT_TEXT_TAG = "utl_reusable_alert_text"

def show_dpg_alert_modal(title: str, message: str,
                         modal_tag: str = UTL_REUSABLE_ALERT_MODAL_TAG,
                         text_tag: str = UTL_REUSABLE_ALERT_TEXT_TAG):
    if not dpg.is_dearpygui_running(): print(f"ALERT MODAL (Non-DPG): {title} - {message}"); return
    vp_w = dpg.get_viewport_width() if dpg.is_dearpygui_running() else 800
    vp_h = dpg.get_viewport_height() if dpg.is_dearpygui_running() else 600
    modal_w = 450
    if not dpg.does_item_exist(modal_tag):
        with dpg.window(label=title, modal=True, show=False, tag=modal_tag,
                       no_close=True, width=modal_w, autosize=True, no_saved_settings=True):
            dpg.add_text("", tag=text_tag, wrap=modal_w - 30)
            dpg.add_spacer(height=15)
            with dpg.group(horizontal=True):
                btn_w = 100
                default_item_spacing_x = 8.0
                spacer_w = (modal_w - btn_w - (default_item_spacing_x * 2) if dpg.is_dearpygui_running() else modal_w - btn_w - 16) / 2
                dpg.add_spacer(width=max(0, int(spacer_w)))
                dpg.add_button(label="OK", width=btn_w, user_data=modal_tag,
                               callback=lambda s, a, u: dpg.configure_item(u, show=False))
    dpg.set_value(text_tag, message)
    dpg.configure_item(modal_tag, label=title, show=True)
    dpg.set_item_pos(modal_tag, [max(0,(vp_w - modal_w) // 2), max(0, vp_h // 3)])

UTL_PROGRESS_MODAL_TAG = "utl_global_progress_modal"
UTL_PROGRESS_TEXT_TAG = "utl_global_progress_text"

def show_dpg_progress_modal(title: str, message: str,
                            modal_tag: str = UTL_PROGRESS_MODAL_TAG,
                            text_tag: str = UTL_PROGRESS_TEXT_TAG) -> bool:
    if not dpg.is_dearpygui_running(): print(f"PROGRESS MODAL (Non-DPG): {title} - {message}"); return False
    vp_w = dpg.get_viewport_width() if dpg.is_dearpygui_running() else 800
    vp_h = dpg.get_viewport_height() if dpg.is_dearpygui_running() else 600
    modal_width, modal_height = 350, 70
    if not dpg.does_item_exist(modal_tag):
        with dpg.window(label=title, modal=True, show=False, tag=modal_tag,
                       no_close=True, no_title_bar=True, no_saved_settings=True,
                       pos=[(vp_w - modal_width) // 2, (vp_h - modal_height) // 2],
                       width=modal_width, height=modal_height):
            dpg.add_text(message, tag=text_tag)
    dpg.configure_item(modal_tag, show=True, label=title)
    dpg.set_value(text_tag, message)
    if dpg.is_dearpygui_running(): dpg.split_frame()
    return True

def hide_dpg_progress_modal(modal_tag: str = UTL_PROGRESS_MODAL_TAG):
    if dpg.is_dearpygui_running() and dpg.does_item_exist(modal_tag):
        dpg.configure_item(modal_tag, show=False)

def _get_top_n_correlated_with_target(df: pd.DataFrame, target_col: str, numeric_cols: List[str], top_n: int) -> List[str]:
    if df is None or target_col not in df.columns: return []
    if not pd.api.types.is_numeric_dtype(df[target_col].dtype): return []
    if not numeric_cols: return []
    if top_n <= 0: return []
    correlations = []
    valid_numeric_cols = [
        col for col in numeric_cols
        if col != target_col and col in df.columns and pd.api.types.is_numeric_dtype(df[col].dtype)
    ]
    if not valid_numeric_cols: return []
    for col in valid_numeric_cols:
        try:
            temp_df = df[[target_col, col]].dropna()
            if len(temp_df) >= 2:
                corr_val = temp_df[target_col].corr(temp_df[col])
                if pd.notna(corr_val):
                    correlations.append((col, abs(corr_val)))
        except Exception as e: continue
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_n_cols = [col_name for col_name, corr_val in correlations[:top_n]]
    return top_n_cols

def create_table_with_data(table_tag: str, df: pd.DataFrame,
                           utils_format_numeric=True, parent_df_for_widths: Optional[pd.DataFrame] = None):
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(table_tag):
        print(f"Error: Table '{table_tag}' not exist."); return
    dpg.delete_item(table_tag, children_only=True)
    if df is None or df.empty:
        dpg.add_table_column(label="Info", parent=table_tag, init_width_or_weight=300)
        with dpg.table_row(parent=table_tag): dpg.add_text("No data." if df is None else "Empty DF.")
        return
    col_widths = calculate_column_widths(parent_df_for_widths if parent_df_for_widths is not None and not parent_df_for_widths.empty else df)
    for col in df.columns: dpg.add_table_column(label=str(col), parent=table_tag, init_width_or_weight=col_widths.get(str(col), MIN_COL_WIDTH))
    for i in range(len(df)):
        with dpg.table_row(parent=table_tag):
            for col_name in df.columns:
                val = df.iat[i, df.columns.get_loc(col_name)]

                # << CHANGED >>: 숫자 포맷팅 부분을 round() 함수로 변경
                if pd.isna(val):
                    s_val = "NaN"
                elif utils_format_numeric and isinstance(val, (float, np.floating)):
                    # 소수점 둘째자리에서 반올림하여 문자열로 변환
                    s_val = str(round(val, 2))
                else:
                    s_val = str(val)
                
                dpg.add_text(format_text_for_display(s_val))

def create_table_with_large_data_preview(table_tag: str, df: pd.DataFrame, 
                                         max_rows: int = 25, max_cols: int = 20):
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(table_tag):
        print(f"Error: Table '{table_tag}' not exist.")
        return
        
    if df is None:
        create_table_with_data(table_tag, pd.DataFrame({"Info": ["No data to display."]}))
        return
    if df.empty:
        create_table_with_data(table_tag, pd.DataFrame({"Info": ["DataFrame is empty."]}))
        return

    display_df = df

    if len(df) > max_rows:
        head_rows = max_rows // 2
        tail_rows = max_rows - head_rows
        head_df = df.head(head_rows)
        tail_df = df.tail(tail_rows)
        separator_data = {col: '...' for col in df.columns}
        separator_df = pd.DataFrame([separator_data])
        display_df = pd.concat([head_df, separator_df, tail_df], ignore_index=True)

    if len(display_df.columns) > max_cols:
        head_cols_count = max_cols // 2
        tail_cols_count = max_cols - head_cols_count
        head_cols = display_df.iloc[:, :head_cols_count]
        tail_cols = display_df.iloc[:, -tail_cols_count:]
        ellipsis_col = pd.Series(['...'] * len(display_df), name='...')
        display_df = pd.concat([head_cols, ellipsis_col, tail_cols], axis=1)

    create_table_with_data(table_tag, display_df)

# utils.py 파일에 추가할 코드

def create_dpg_plot_scaffold(parent: str, title: str, x_label: str, y_label: str, 
                             h: int = 300, w: int = -1, legend: bool = True,
                             eq_asp: bool = False) -> Tuple[str, str, str, Optional[str]]:
    """
    DearPyGui 플롯의 기본 틀(플롯, 축, 범례)을 생성하고 각 요소의 태그를 반환합니다.
    """
    plot_tag = dpg.generate_uuid()
    x_axis_tag = dpg.generate_uuid()
    y_axis_tag = dpg.generate_uuid()
    legend_tag = dpg.generate_uuid() if legend else None

    with dpg.plot(label=title, height=h, width=w, parent=parent, tag=plot_tag):
        if legend_tag:
            dpg.add_plot_legend(tag=legend_tag)
        
        dpg.add_plot_axis(dpg.mvXAxis, label=x_label, tag=x_axis_tag, auto_fit=True)
        dpg.add_plot_axis(dpg.mvYAxis, label=y_label, tag=y_axis_tag, auto_fit=True)
        
        if eq_asp:
            dpg.set_axis_limits_auto(dpg.last_item())
            dpg.set_axis_limits_auto(x_axis_tag)
            # Aspect ratio is set on the plot itself after creation if needed
            # dpg.set_plot_aspect_ratio(plot_tag, 1.0) # This might be what you want
    
    return plot_tag, x_axis_tag, y_axis_tag, legend_tag