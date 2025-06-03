import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from io import BytesIO
from typing import List, Tuple, Optional, Any, Dict, Callable
from scipy import stats # calculate_feature_target_relevance 에서 사용
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg # ★★★ 추가 ★★★
import traceback

MIN_COL_WIDTH = 50
MAX_COL_WIDTH = 300
CELL_PADDING = 20
TARGET_DATA_CHARS = 25
ELLIPSIS = "..."

def _log_util(message: str, level: str = "DEBUG"): # 로깅 함수 추가 (선택적)
    print(f"[{level} utils] {message}")


def calculate_feature_target_relevance(
    df: pd.DataFrame,
    target_var: str,
    target_var_type: str, # "Continuous" 또는 "Categorical"
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
                    grouped_feature_data_for_anova = [aligned_feature[aligned_target == cat] for cat in target_categories_local]
                    valid_groups_anova = [g for g in grouped_feature_data_for_anova if len(g) >= 2]
                    if len(valid_groups_anova) >= 2:
                        f_val, p_val = stats.f_oneway(*valid_groups_anova)
                        score = abs(f_val) if pd.notna(f_val) and np.isfinite(f_val) else 0.0
            elif target_var_type == "Continuous" and is_feature_numeric:
                if pd.api.types.is_numeric_dtype(aligned_target.dtype):
                    corr_val = aligned_feature.corr(aligned_target)
                    score = abs(corr_val) if pd.notna(corr_val) and np.isfinite(corr_val) else 0.0
        except Exception:
            score = 0.0
        if score > 1e-6:
            scores.append((feature_col, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def create_analyzable_image_widget(
    parent_dpg_tag: Any,
    fig: matplotlib.figure.Figure,
    image_widget_tag: Any,
    image_title: str,
    shared_callbacks: Dict[str, Callable],
    desired_dpi: int = 100,
    default_texture_tag: Optional[Any] = None,
    existing_dpg_texture_id: Optional[Any] = None
) -> Tuple[Optional[str], Optional[str]]:
    _log_util(f"Called: image_title='{image_title}', widget_tag='{str(image_widget_tag)}', parent='{str(parent_dpg_tag)}', existing_tex_id='{existing_dpg_texture_id}'")

    if not all([fig, parent_dpg_tag, image_widget_tag, image_title, shared_callbacks]):
        _log_util("Error: 필수 인자 누락.", "ERROR")
        plt.close(fig) if fig else None
        return None, None

    cache_image_func = shared_callbacks.get('cache_image_data_func')
    initiate_ollama_func = shared_callbacks.get('initiate_ollama_analysis')

    if not callable(cache_image_func) or not callable(initiate_ollama_func):
        _log_util("Error: 필수 콜백 함수 누락.", "ERROR")
        plt.close(fig) if fig else None
        return None, None

    new_dpg_texture_id_str: Optional[str] = None
    img_buf_for_cache: Optional[BytesIO] = None
    width, height = 0, 0
    image_numpy_uint8: Optional[np.ndarray] = None

    try:
        _log_util(f"1. Matplotlib Figure -> DPG 텍스처 변환 시작: {image_title}")

        # ★★★ FigureCanvasAgg를 사용하여 이미지 데이터 추출 ★★★
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # RGBA 버퍼 가져오기 - 변수명 오타 수정
        buf_rgba_raw = canvas.buffer_rgba()
        image_numpy_uint8_temp = np.asarray(buf_rgba_raw)  # ← 수정: buf_rgba_raw 사용
        _log_util(f"   FigureCanvasAgg buffer_rgba() -> np.asarray() shape: {image_numpy_uint8_temp.shape}, dtype: {image_numpy_uint8_temp.dtype}")

        if image_numpy_uint8_temp.ndim == 3 and image_numpy_uint8_temp.shape[2] == 4:
            height, width, _ = image_numpy_uint8_temp.shape
            image_numpy_uint8 = image_numpy_uint8_temp
        elif image_numpy_uint8_temp.ndim == 1 and image_numpy_uint8_temp.size % 4 == 0:
            width_px = int(fig.get_figwidth() * fig.get_dpi())
            height_px = int(fig.get_figheight() * fig.get_dpi())
            if image_numpy_uint8_temp.size == width_px * height_px * 4:
                _log_util(f"   1D buffer_rgba() size matches fig_width*dpi * fig_height*dpi * 4. Reshaping.")
                width, height = width_px, height_px
                image_numpy_uint8 = image_numpy_uint8_temp.reshape((height, width, 4))
            else:
                width_canvas, height_canvas = canvas.get_width_height()
                if image_numpy_uint8_temp.size == width_canvas * height_canvas * 4:
                    _log_util(f"   1D buffer_rgba() size matches canvas_width*canvas_height*4. Reshaping using canvas dimensions.")
                    width, height = width_canvas, height_canvas
                    image_numpy_uint8 = image_numpy_uint8_temp.reshape((height, width, 4))
                else:
                    raise ValueError(f"Cannot determine correct shape for 1D buffer. Buffer size: {image_numpy_uint8_temp.size}, Fig W*H*4: {width_px*height_px*4}, Canvas W*H*4: {width_canvas*height_canvas*4}")
        else:
            raise ValueError(f"Unexpected image data shape from FigureCanvasAgg: {image_numpy_uint8_temp.shape}")

        # 로깅 함수명 오타 수정
        _log_util(f"   Matplotlib Figure 최종 사용 크기: Width={width}, Height={height}")  # ← 수정: _log_util 사용
        if width == 0 or height == 0:
            raise ValueError("Figure width or height is 0 after processing canvas.")

        # float32로 변환하고 0-1 범위로 정규화
        texture_data = image_numpy_uint8.astype(np.float32) / 255.0
        _log_util(f"   최종 DPG 텍스처 데이터 - Shape: {texture_data.shape}, Min: {np.min(texture_data):.2f}, Max: {np.max(texture_data):.2f}, Dtype: {texture_data.dtype}")

        if texture_data.ndim == 3:
            texture_data_flat = texture_data.flatten()
        else:
            texture_data_flat = texture_data
        
        if texture_data_flat.size != width * height * 4:
            _log_util(f"   [ERROR] 최종 texture_data_flat.size ({texture_data_flat.size}) != width*height*4 ({width*height*4}).", "ERROR")
            raise ValueError("Final flattened texture data size mismatch for DPG.")

        temp_dpg_texture_id = dpg.generate_uuid()
        new_dpg_texture_id_str = str(temp_dpg_texture_id)
        _log_util(f"   생성된 새 DPG 텍스처 ID: {new_dpg_texture_id_str} (original int: {temp_dpg_texture_id})")

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=width, height=height, default_value=texture_data_flat,
                                format=dpg.mvFormat_Float_rgba, tag=new_dpg_texture_id_str)
        _log_util(f"   DPG 텍스처 등록 완료: {new_dpg_texture_id_str}")

        img_buf_for_cache = BytesIO()
        fig.savefig(img_buf_for_cache, format='png', dpi=desired_dpi, bbox_inches='tight')
        img_buf_for_cache.seek(0)
        cache_image_func(new_dpg_texture_id_str, img_buf_for_cache)
        _log_util(f"   BytesIO 객체 캐싱 완료 (키: {new_dpg_texture_id_str}).")

    except Exception as e:
        _log_util(f"이미지/텍스처 생성 중 예외 발생: {image_title} - {e}", "ERROR")
        _log_util(traceback.format_exc(), "ERROR")
        if img_buf_for_cache:
            img_buf_for_cache.close()
        if new_dpg_texture_id_str and dpg.does_item_exist(new_dpg_texture_id_str):
            try:
                dpg.delete_item(new_dpg_texture_id_str)
            except Exception as e_del_tex:
                _log_util(f"오류 발생한 텍스처 삭제 중 예외: {e_del_tex}", "ERROR")
        new_dpg_texture_id_str = None
    finally:
        if fig:
            plt.close(fig)
            _log_util(f"Matplotlib Figure 객체 닫힘: {image_title}")

    # ... (나머지 코드는 동일하게 유지)
    str_existing_dpg_texture_id = str(existing_dpg_texture_id) if existing_dpg_texture_id is not None else None
    if str_existing_dpg_texture_id and dpg.does_item_exist(str_existing_dpg_texture_id):
        if new_dpg_texture_id_str is None or (new_dpg_texture_id_str != str_existing_dpg_texture_id):
            _log_util(f"이전 DPG 텍스처 삭제 시도: {str_existing_dpg_texture_id}")
            try:
                dpg.delete_item(str_existing_dpg_texture_id)
                _log_util(f"이전 DPG 텍스처 ({str_existing_dpg_texture_id}) 삭제됨.", "INFO")
            except Exception as e_del:
                _log_util(f"이전 DPG 텍스처 ({str_existing_dpg_texture_id}) 삭제 중 오류: {e_del}", "WARN")
    elif str_existing_dpg_texture_id:
        _log_util(f"이전 DPG 텍스처 ({str_existing_dpg_texture_id})가 존재하지 않아 삭제 건너뜀.")

    str_parent_dpg_tag = str(parent_dpg_tag)
    str_image_widget_tag = str(image_widget_tag)
    str_default_texture_tag = str(default_texture_tag) if default_texture_tag is not None else None

    _log_util(f"DPG 이미지 위젯 생성/업데이트 시작: {str_image_widget_tag}")
    if dpg.does_item_exist(str_parent_dpg_tag):
        if new_dpg_texture_id_str and width > 0 and height > 0:
            _log_util(f"   새 텍스처({new_dpg_texture_id_str})로 이미지 위젯({str_image_widget_tag}) 설정. 부모: {str_parent_dpg_tag}, 크기: {width}x{height}")
            if dpg.does_item_exist(str_image_widget_tag):
                dpg.configure_item(str_image_widget_tag, texture_tag=new_dpg_texture_id_str, width=width, height=height, show=True)
                _log_util(f"   기존 이미지 위젯 '{str_image_widget_tag}' 업데이트 완료.")
            else:
                dpg.add_image(texture_tag=new_dpg_texture_id_str, tag=str_image_widget_tag,
                              width=width, height=height, parent=str_parent_dpg_tag, show=True)
                _log_util(f"   새 이미지 위젯 '{str_image_widget_tag}' 추가 완료.")
            user_data_for_callback = {"title": image_title, "texture_tag_for_ollama": new_dpg_texture_id_str}
            handler_reg_tag = f"{str_image_widget_tag}_hr_{dpg.generate_uuid()}"
            try:
                # DPG 버전 호환성을 위한 안전한 unbind 처리
                if dpg.does_item_exist(str_image_widget_tag):
                    try:
                        if hasattr(dpg, 'unbind_item_handler_registry'):
                            dpg.unbind_item_handler_registry(str_image_widget_tag)
                        elif hasattr(dpg, 'bind_item_handler_registry'):
                            # 구버전에서는 None을 바인딩하여 기존 핸들러 제거
                            dpg.bind_item_handler_registry(str_image_widget_tag, None)
                    except Exception as e_unbind:
                        _log_util(f"기존 핸들러 unbind 중 오류 (무시): {e_unbind}", "WARN")
                
                with dpg.item_handler_registry(tag=str(handler_reg_tag)):
                    dpg.add_item_clicked_handler(callback=initiate_ollama_func, user_data=user_data_for_callback)
                dpg.bind_item_handler_registry(str_image_widget_tag, str(handler_reg_tag))
                _log_util(f"   ItemClickedHandler 설정 완료: {str_image_widget_tag} -> {str(handler_reg_tag)}")
            except Exception as e_handler:
                _log_util(f"ItemClickedHandler 설정 중 오류: {str_image_widget_tag} - {e_handler}", "ERROR")
            return new_dpg_texture_id_str, str_image_widget_tag
        else:
            _log_util(f"새 텍스처 생성 실패({new_dpg_texture_id_str}), 기본 텍스처로 대체: {str_image_widget_tag}", "WARN")
            if dpg.does_item_exist(str_image_widget_tag):
                if str_default_texture_tag and dpg.does_item_exist(str_default_texture_tag):
                    cfg = dpg.get_item_configuration(str_default_texture_tag)
                    dpg.configure_item(str_image_widget_tag, texture_tag=str_default_texture_tag, width=cfg.get('width',100), height=cfg.get('height',30), show=True)
                    try:
                        if hasattr(dpg, 'unbind_item_handler_registry'):
                            dpg.unbind_item_handler_registry(str_image_widget_tag)
                        elif hasattr(dpg, 'bind_item_handler_registry'):
                            dpg.bind_item_handler_registry(str_image_widget_tag, None)
                    except Exception as e_unbind_fallback:
                        _log_util(f"기본 텍스처 대체 시 unbind 오류 (무시): {e_unbind_fallback}", "WARN")
                    dpg.set_item_user_data(str_image_widget_tag, None)
                else:
                    dpg.configure_item(str_image_widget_tag, show=False)
            elif str_default_texture_tag and dpg.does_item_exist(str_default_texture_tag):
                cfg = dpg.get_item_configuration(str_default_texture_tag)
                dpg.add_image(texture_tag=str_default_texture_tag, tag=str_image_widget_tag, width=cfg.get('width',100), height=cfg.get('height',30), parent=str_parent_dpg_tag, show=True)
            return None, (str_image_widget_tag if dpg.does_item_exist(str_image_widget_tag) else None)
    else:
        _log_util(f"부모 DPG 태그 '{parent_dpg_tag}'가 존재하지 않음.", "ERROR")
        if new_dpg_texture_id_str and dpg.does_item_exist(new_dpg_texture_id_str):
            try:
                dpg.delete_item(new_dpg_texture_id_str)
            except Exception as e_del_tex2:
                _log_util(f"부모 부재로 텍스처 삭제 중 예외: {e_del_tex2}", "ERROR")
        return None, None
    
def _get_numeric_cols(df: pd.DataFrame) -> List[str]:
    if df is None: return []
    return df.select_dtypes(include=np.number).columns.tolist()

def _get_categorical_cols(df: pd.DataFrame, max_unique_for_cat: int = 30, main_callbacks: Optional[Dict] = None) -> List[str]:
    if df is None: return []
    categorical_cols = []
    s1_analysis_types = {}
    if main_callbacks and 'get_column_analysis_types' in main_callbacks:
        s1_analysis_types = main_callbacks['get_column_analysis_types']()
    for col in df.columns:
        if col in s1_analysis_types:
            s1_type = s1_analysis_types[col]
            if any(cat_keyword in s1_type for cat_keyword in ["Categorical", "Text (", "Potentially Sensitive"]):
                categorical_cols.append(col); continue
            elif "Numeric (Binary)" in s1_type:
                categorical_cols.append(col); continue
            elif "Numeric" in s1_type: continue
        dtype = df[col].dtype; nunique = df[col].nunique(dropna=False)
        if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            if nunique <= max_unique_for_cat: categorical_cols.append(col)
        elif pd.api.types.is_categorical_dtype(dtype): categorical_cols.append(col)
        elif pd.api.types.is_bool_dtype(dtype): categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            if nunique <= 5:
                if not (col in s1_analysis_types and "Numeric" in s1_analysis_types[col]):
                    categorical_cols.append(col)
        if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
            if nunique <= max_unique_for_cat: categorical_cols.append(col)
    return list(dict.fromkeys(categorical_cols))

def _guess_target_type(df: pd.DataFrame, column_name: str, step1_type_selections: dict = None) -> str:
    if not column_name or df is None or column_name not in df.columns: return "Continuous"
    series = df[column_name]
    if step1_type_selections and column_name in step1_type_selections:
        s1_type = step1_type_selections[column_name]
        if any(k in s1_type for k in ["Text (", "Potentially Sensitive", "Categorical"]): return "Categorical"
        if "Numeric" in s1_type: return "Categorical" if "Binary" in s1_type or series.nunique(dropna=False) <= 5 else "Continuous"
        if any(k in s1_type for k in ["Datetime", "Timedelta"]): return "Categorical"
    if pd.api.types.is_categorical_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype): return "Categorical"
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype): return "Categorical"
    if pd.api.types.is_numeric_dtype(series.dtype): return "Categorical" if series.nunique(dropna=False) <= 10 else "Continuous"
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
            max_px = max(max_px, max(get_safe_text_size(format_text_for_display(str(x)))[0] for x in sample))
        col_widths[col_name] = int(max(min_w, min(max_px + pad, max_w)))
    return col_widths

def create_table_with_data(table_tag: str, df: pd.DataFrame,
                           utils_format_numeric=False, parent_df_for_widths: Optional[pd.DataFrame] = None):
    if not dpg.does_item_exist(table_tag): print(f"Error: Table '{table_tag}' not exist."); return
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
                s_val = "NaN" if pd.isna(val) else (f"{val:.3f}" if utils_format_numeric and isinstance(val, (float, np.floating)) else str(val))
                dpg.add_text(format_text_for_display(s_val))

PLOT_DEF_H = 300; PLOT_DEF_W = -1

def create_dpg_plot_scaffold(parent: str, title: str, x_lbl: str, y_lbl: str, w: int=PLOT_DEF_W, h: int=PLOT_DEF_H, legend: bool=False, eq_asp: bool=False) -> Tuple[str, str, str, Optional[str]]:
    p_tag, x_tag, y_tag = dpg.generate_uuid(), dpg.generate_uuid(), dpg.generate_uuid()
    l_tag = dpg.generate_uuid() if legend else None
    with dpg.plot(label=title, height=h, width=w, parent=parent, tag=p_tag, equal_aspects=eq_asp):
        dpg.add_plot_axis(dpg.mvXAxis, label=x_lbl, tag=x_tag, auto_fit=True)
        dpg.add_plot_axis(dpg.mvYAxis, label=y_lbl, tag=y_tag, auto_fit=True)
        if legend and l_tag: dpg.add_plot_legend(tag=l_tag, parent=p_tag, horizontal=False, location=dpg.mvPlot_Location_NorthEast)
    return p_tag, x_tag, y_tag, l_tag

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

    # 수정: matrix.values.flatten.astype(float) -> matrix.values.flatten().astype(float)
    data_np = np.nan_to_num(matrix.values.flatten().astype(float), nan=0.0, posinf=1.0, neginf=-1.0)

    data_flat = data_np.tolist()
    if len(data_flat) != r * c: dpg.add_text(f"{title}: Data size mismatch.", parent=parent, color=(255,0,0)); return
    col_lbls, row_lbls = [str(x) for x in matrix.columns], [str(x) for x in matrix.index]
    p_tag, x_tag, y_tag, _ = create_dpg_plot_scaffold(parent, title, "", "", h=h, eq_asp=(r == c))
    dpg.bind_colormap(p_tag, cmap)
    if col_lbls and c > 0: dpg.set_axis_ticks(x_tag, tuple(zip(col_lbls, [i + 0.5 for i in range(c)])))
    if row_lbls and r > 0: dpg.set_axis_ticks(y_tag, tuple(zip(row_lbls, [i + 0.5 for i in range(r)])))
    s_min, s_max = -1.0, 1.0; actual_min, actual_max = data_np.min(), data_np.max()
    if actual_min == actual_max: s_min, s_max = actual_min - 0.5 if actual_min != 0 else -0.5, actual_max + 0.5 if actual_max != 0 else 0.5
    elif cmap in [dpg.mvPlotColormap_RdBu, dpg.mvPlotColormap_Spectral, dpg.mvPlotColormap_PiYG, dpg.mvPlotColormap_BrBG]: # RdBu 중복 제거
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
    vp_w = dpg.get_viewport_width(); vp_h = dpg.get_viewport_height(); modal_w = 450
    if not dpg.does_item_exist(modal_tag):
        with dpg.window(label=title, modal=True, show=False, tag=modal_tag,
                       no_close=True, width=modal_w, autosize=True, no_saved_settings=True):
            dpg.add_text("", tag=text_tag, wrap=modal_w - 30); dpg.add_spacer(height=15)
            with dpg.group(horizontal=True):
                btn_w = 100; default_item_spacing_x = 8.0
                spacer_w = (modal_w - btn_w - (default_item_spacing_x * 2)) / 2
                dpg.add_spacer(width=max(0, spacer_w))
                dpg.add_button(label="OK", width=btn_w, user_data=modal_tag,
                               callback=lambda s, a, u: dpg.configure_item(u, show=False))
    dpg.set_value(text_tag, message)
    dpg.configure_item(modal_tag, label=title, show=True)
    dpg.set_item_pos(modal_tag, [(vp_w - modal_w) // 2, vp_h // 3])

UTL_PROGRESS_MODAL_TAG = "utl_global_progress_modal"
UTL_PROGRESS_TEXT_TAG = "utl_global_progress_text"

def show_dpg_progress_modal(title: str, message: str,
                            modal_tag: str = UTL_PROGRESS_MODAL_TAG,
                            text_tag: str = UTL_PROGRESS_TEXT_TAG) -> bool:
    if not dpg.is_dearpygui_running(): print(f"PROGRESS MODAL (Non-DPG): {title} - {message}"); return False
    vp_w = dpg.get_viewport_width(); vp_h = dpg.get_viewport_height()
    if not dpg.does_item_exist(modal_tag):
        modal_width, modal_height = 350, 70
        with dpg.window(label=title, modal=True, show=False, tag=modal_tag,
                       no_close=True, no_title_bar=True, no_saved_settings=True,
                       pos=[(vp_w - modal_width) // 2, (vp_h - modal_height) // 2],
                       width=modal_width, height=modal_height):
            dpg.add_text(message, tag=text_tag)
    dpg.configure_item(modal_tag, show=True, label=title)
    dpg.set_value(text_tag, message); dpg.split_frame()
    return True

def hide_dpg_progress_modal(modal_tag: str = UTL_PROGRESS_MODAL_TAG):
    if dpg.is_dearpygui_running() and dpg.does_item_exist(modal_tag):
        dpg.configure_item(modal_tag, show=False)

def _get_top_n_correlated_with_target(df: pd.DataFrame, target_col: str, numeric_cols: List[str], top_n: int) -> List[str]:
    if df is None or target_col not in df.columns: print(f"경고: 대상 컬럼 '{target_col}' 없음"); return []
    if not pd.api.types.is_numeric_dtype(df[target_col].dtype): print(f"경고: 대상 컬럼 '{target_col}' 수치형 아님"); return []
    if not numeric_cols: print("경고: 비교할 수치형 컬럼 목록 비어있음"); return []
    if top_n <= 0: return []
    correlations = []
    valid_numeric_cols = [col for col in numeric_cols if col != target_col and col in df.columns and pd.api.types.is_numeric_dtype(df[col].dtype)]
    if not valid_numeric_cols: print("경고: 유효한 수치형 비교 대상 컬럼 없음"); return []
    for col in valid_numeric_cols:
        try:
            temp_df = df[[target_col, col]].dropna()
            if len(temp_df) >= 2:
                corr_val = temp_df[target_col].corr(temp_df[col])
                if pd.notna(corr_val): correlations.append((col, abs(corr_val)))
        except Exception as e: print(f"경고: 컬럼 '{col}'과 '{target_col}' 간 상관관계 계산 중 오류: {e}"); continue
    correlations.sort(key=lambda x: x[1], reverse=True)
    return [col_name for col_name, corr_val in correlations[:top_n]]