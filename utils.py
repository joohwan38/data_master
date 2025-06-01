# utils.py

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
from scipy import stats

MIN_COL_WIDTH = 50
MAX_COL_WIDTH = 300
CELL_PADDING = 20
TARGET_DATA_CHARS = 25
ELLIPSIS = "..."

def calculate_feature_target_relevance(
    df: pd.DataFrame,
    target_var: str,
    target_var_type: str, # "Continuous" 또는 "Categorical"
    features_to_analyze: List[str],
    main_app_callbacks: Optional[Dict] = None # S1 타입을 고려하기 위해 추가 (선택적)
) -> List[Tuple[str, float]]:
    """
    주어진 특성들과 타겟 변수 간의 연관성 점수를 계산합니다.
    타겟이 연속형이면 (특성이 수치형일 때) 상관계수 절대값을,
    타겟이 범주형이면 (특성이 수치형일 때) ANOVA F-통계량 절대값을 반환합니다.
    결과는 (특성 변수명, 연관성 점수) 튜플의 리스트로, 점수가 높은 순으로 정렬됩니다.
    """
    if df is None or target_var not in df.columns or not features_to_analyze:
        return []

    scores = []
    
    s1_analysis_types = {}
    if main_app_callbacks and 'get_column_analysis_types' in main_app_callbacks:
        # Step 1에서 정의된 분석 타입을 가져옵니다. (예: main_app_callbacks['get_column_analysis_types']() 호출)
        s1_analysis_types = main_app_callbacks['get_column_analysis_types']()


    for feature_col in features_to_analyze:
        if feature_col == target_var or feature_col not in df.columns:
            continue

        # 분석할 feature가 실제로 수치형인지 확인 (S1 타입 또는 pandas dtype 기준)
        feature_s1_type = s1_analysis_types.get(feature_col, str(df[feature_col].dtype))
        is_feature_numeric = ("Numeric" in feature_s1_type and "Binary" not in feature_s1_type) or \
                             (pd.api.types.is_numeric_dtype(df[feature_col].dtype) and df[feature_col].nunique() > 5) # 고유값 많은 수치형


        score = 0.0
        try:
            # 연관성 계산을 위한 데이터 준비 (결측값 제거)
            # Series로 직접 작업하여 불필요한 DataFrame 복사 방지
            target_series_clean = df[target_var].dropna()
            feature_series_clean = df[feature_col].dropna()
            
            # 공통 인덱스 기준으로 데이터 정렬 및 필터링
            common_index = target_series_clean.index.intersection(feature_series_clean.index)
            if len(common_index) < 20: # 최소 데이터 포인트 수 (예: 20)
                scores.append((feature_col, 0.0))
                continue
                
            aligned_target = target_series_clean.loc[common_index]
            aligned_feature = feature_series_clean.loc[common_index]

            if target_var_type == "Categorical" and is_feature_numeric:
                target_categories_local = aligned_target.unique()
                if len(target_categories_local) >= 2: # ANOVA를 위한 최소 카테고리 수
                    grouped_feature_data_for_anova = [
                        aligned_feature[aligned_target == cat]
                        for cat in target_categories_local
                    ]
                    valid_groups_anova = [g for g in grouped_feature_data_for_anova if len(g) >= 2] # 각 그룹 최소 샘플 수
                    if len(valid_groups_anova) >= 2: # ANOVA를 위한 최소 그룹 수
                        f_val, p_val = stats.f_oneway(*valid_groups_anova)
                        score = abs(f_val) if pd.notna(f_val) and np.isfinite(f_val) else 0.0
            
            elif target_var_type == "Continuous" and is_feature_numeric:
                # 타겟도 수치형이어야 상관계산 의미 있음
                if pd.api.types.is_numeric_dtype(aligned_target.dtype):
                    corr_val = aligned_feature.corr(aligned_target)
                    score = abs(corr_val) if pd.notna(corr_val) and np.isfinite(corr_val) else 0.0
            
            # 다른 조합에 대한 연관성 계산 로직은 필요시 여기에 추가 가능
            
        except Exception as e_relevance:
            # print(f"Relevance calculation error for {feature_col} vs {target_var}: {e_relevance}") # 디버깅용
            score = 0.0 
        
        # 유의미한 점수만 추가 (0점은 제외하거나, 매우 작은 임계값 설정 가능)
        if score > 1e-6: # 매우 작은 값은 사실상 0으로 간주될 수 있으므로 임계값 사용
            scores.append((feature_col, score))
        # else: # 0점인 경우도 추가하고 싶다면 이 부분을 활성화
        #    scores.append((feature_col, 0.0))


    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def _get_numeric_cols(df: pd.DataFrame) -> List[str]:
    """DataFrame에서 수치형 데이터 타입을 가진 컬럼 목록을 반환합니다."""
    if df is None:
        return []
    return df.select_dtypes(include=np.number).columns.tolist()

def _get_categorical_cols(df: pd.DataFrame, max_unique_for_cat: int = 30, main_callbacks: Optional[Dict] = None) -> List[str]:
    """
    DataFrame에서 범주형으로 간주될 수 있는 컬럼 목록을 반환합니다.
    step_01_data_loading에서 정의된 분석 타입을 우선적으로 고려합니다.
    """
    if df is None:
        return []
    
    categorical_cols = []
    s1_analysis_types = {}
    if main_callbacks and 'get_column_analysis_types' in main_callbacks:
        s1_analysis_types = main_callbacks['get_column_analysis_types']() # step_01_data_loading._type_selections

    for col in df.columns:
        # Step 1에서 정의된 타입 확인
        if col in s1_analysis_types:
            s1_type = s1_analysis_types[col]
            if any(cat_keyword in s1_type for cat_keyword in ["Categorical", "Text (", "Potentially Sensitive"]):
                categorical_cols.append(col)
                continue
            elif "Numeric (Binary)" in s1_type: # 수치형 바이너리도 범주형으로 취급 가능
                categorical_cols.append(col)
                continue
            elif "Numeric" in s1_type: # 일반 수치형은 제외
                continue
        
        # Step 1 타입 정보가 없거나, 위 조건에 해당 안 될 경우 dtype 및 고유값 기반으로 판단
        dtype = df[col].dtype
        nunique = df[col].nunique(dropna=False)

        if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            if nunique <= max_unique_for_cat:
                categorical_cols.append(col)
        elif pd.api.types.is_categorical_dtype(dtype):
            categorical_cols.append(col)
        elif pd.api.types.is_bool_dtype(dtype):
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dtype): # 수치형 중 고유값이 매우 적은 경우
            if nunique <= 5: # 매우 적은 고유값을 가진 수치형 (예: 0, 1, 2)
                 # S1 타입에서 명시적으로 Numeric이 아니었다면 범주형으로 볼 여지 있음
                if not (col in s1_analysis_types and "Numeric" in s1_analysis_types[col]):
                    categorical_cols.append(col)
        # Datetime, Timedelta는 보통 범주형으로 직접 사용하지 않으나, 필요시 추가 가능
        if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
            if nunique <= max_unique_for_cat:
                categorical_cols.append(col)

    return list(dict.fromkeys(categorical_cols)) # 중복 제거 후 반환

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
            max_px = max(max_px, max(get_safe_text_size(format_text_for_display(str(x)))[0] for x in sample))
        col_widths[col_name] = int(max(min_w, min(max_px + pad, max_w)))
    return col_widths

# 'parent_df_for_widths' 파라미터 명칭 일관성 유지
def create_table_with_data(table_tag: str, df: pd.DataFrame, 
                           utils_format_numeric=False, parent_df_for_widths: Optional[pd.DataFrame] = None): # 파라미터명 수정
    if not dpg.does_item_exist(table_tag): print(f"Error: Table '{table_tag}' not exist."); return
    dpg.delete_item(table_tag, children_only=True)
    if df is None or df.empty:
        dpg.add_table_column(label="Info", parent=table_tag, init_width_or_weight=300)
        with dpg.table_row(parent=table_tag): dpg.add_text("No data." if df is None else "Empty DF.")
        return
    
    # 'parent_df_for_widths'를 사용하도록 수정
    col_widths = calculate_column_widths(parent_df_for_widths if parent_df_for_widths is not None and not parent_df_for_widths.empty else df)
    for col in df.columns: dpg.add_table_column(label=str(col), parent=table_tag, init_width_or_weight=col_widths.get(str(col), MIN_COL_WIDTH))
    for i in range(len(df)):
        with dpg.table_row(parent=table_tag):
            for col_name in df.columns:
                val = df.iat[i, df.columns.get_loc(col_name)]
                s_val = "NaN" if pd.isna(val) else (f"{val:.3f}" if utils_format_numeric and isinstance(val, (float, np.floating)) else str(val))
                dpg.add_text(format_text_for_display(s_val))

PLOT_DEF_H = 300
PLOT_DEF_W = -1 

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

def create_dpg_heatmap_plot(parent: str, matrix: pd.DataFrame, title: str, h: int=450, cmap: int=dpg.mvPlotColormap_RdBu): # 사용자가 수정한 컬러맵으로 변경
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(parent): return
    if matrix is None or matrix.empty: dpg.add_text(f"{title}: No data.", parent=parent); return
    r, c = matrix.shape
    if r == 0 or c == 0: dpg.add_text(f"{title}: Empty data (0 rows/cols).", parent=parent); return

    data_np = np.nan_to_num(matrix.values.flatten.astype(float), nan=0.0, posinf=1.0, neginf=-1.0)

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
    elif cmap in [dpg.mvPlotColormap_RdBu, dpg.mvPlotColormap_Spectral, dpg.mvPlotColormap_PiYG, dpg.mvPlotColormap_BrBG, dpg.mvPlotColormap_RdBu]: # RdBu 추가
        abs_val = max(abs(actual_min), abs(actual_max))
        s_min, s_max = -abs_val if abs_val != 0 else -0.5, abs_val if abs_val != 0 else 0.5
    else: s_min, s_max = actual_min, actual_max
    add_dpg_heat_series(y_tag, data_flat, r, c, float(s_min), float(s_max))

def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """두 범주형 시리즈 간의 Cramer's V 상관계수를 계산합니다."""
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
        r_rows, k_cols = confusion_matrix.shape # 변수명 변경 (r, k -> r_rows, k_cols)
        phi2corr = max(0, phi2 - ((k_cols - 1) * (r_rows - 1)) / (n - 1 if n > 1 else 1))
        rcorr = r_rows - (((r_rows - 1)**2) / (n - 1 if n > 1 else 1) if r_rows > 1 else 0)
        kcorr = k_cols - (((k_cols - 1)**2) / (n - 1 if n > 1 else 1) if k_cols > 1 else 0)
        denominator = min((kcorr - 1 if kcorr > 1 else 0), (rcorr - 1 if rcorr > 1 else 0))
        return np.sqrt(phi2corr / denominator) if denominator != 0 else 0.0
    except Exception: return 0.0

# --- 범용 UI 헬퍼 함수 ---
# 각 모달/텍스트 태그는 호출하는 쪽에서 고유하게 생성하여 전달하는 것을 권장.
# 여기서는 예시로 기본 태그명을 사용하나, 충돌 방지를 위해 호출 시 고유 태그 전달.

UTL_REUSABLE_ALERT_MODAL_TAG = "utl_reusable_alert_modal"
UTL_REUSABLE_ALERT_TEXT_TAG = "utl_reusable_alert_text"

def show_dpg_alert_modal(title: str, message: str,
                         modal_tag: str = UTL_REUSABLE_ALERT_MODAL_TAG,
                         text_tag: str = UTL_REUSABLE_ALERT_TEXT_TAG):
    """일반화된 알림 모달을 보여줍니다."""
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
                # 버튼 중앙 정렬을 위한 spacer (근사치)
                default_item_spacing_x = 8.0
                spacer_w = (modal_w - btn_w - (default_item_spacing_x * 2) if dpg.is_dearpygui_running() else modal_w - btn_w - 16) / 2
                dpg.add_spacer(width=max(0, spacer_w))
                dpg.add_button(label="OK", width=btn_w, user_data=modal_tag,
                               callback=lambda s, a, u: dpg.configure_item(u, show=False))
    
    dpg.set_value(text_tag, message)
    dpg.configure_item(modal_tag, label=title, show=True)
    # 모달 위치 설정 (뷰포트 중앙 상단)
    # 실제 모달 크기는 autosize 후 알 수 있으므로, 완벽한 중앙 정렬은 한 프레임 뒤에 가능
    # 여기서는 생성 시 위치를 대략적으로 설정
    dpg.set_item_pos(modal_tag, [(vp_w - modal_w) // 2, vp_h // 3])


UTL_PROGRESS_MODAL_TAG = "utl_global_progress_modal"
UTL_PROGRESS_TEXT_TAG = "utl_global_progress_text"

def show_dpg_progress_modal(title: str, message: str,
                            modal_tag: str = UTL_PROGRESS_MODAL_TAG,
                            text_tag: str = UTL_PROGRESS_TEXT_TAG) -> bool:
    """일반화된 진행 표시 모달을 보여줍니다."""
    if not dpg.is_dearpygui_running(): print(f"PROGRESS MODAL (Non-DPG): {title} - {message}"); return False
    
    vp_w = dpg.get_viewport_width() if dpg.is_dearpygui_running() else 800
    vp_h = dpg.get_viewport_height() if dpg.is_dearpygui_running() else 600

    if not dpg.does_item_exist(modal_tag):
        modal_width, modal_height = 350, 70
        with dpg.window(label=title, modal=True, show=False, tag=modal_tag,
                       no_close=True, no_title_bar=True, no_saved_settings=True,
                       pos=[(vp_w - modal_width) // 2, (vp_h - modal_height) // 2],
                       width=modal_width, height=modal_height):
            dpg.add_text(message, tag=text_tag) # 초기 메시지 설정
            
    dpg.configure_item(modal_tag, show=True, label=title) # 모달 표시 전 타이틀 설정 (no_title_bar=True면 효과 없음)
    dpg.set_value(text_tag, message) # 메시지 업데이트
    if dpg.is_dearpygui_running(): dpg.split_frame() # UI 업데이트 강제
    return True

def hide_dpg_progress_modal(modal_tag: str = UTL_PROGRESS_MODAL_TAG):
    """일반화된 진행 표시 모달을 숨깁니다."""
    if dpg.is_dearpygui_running() and dpg.does_item_exist(modal_tag):
        dpg.configure_item(modal_tag, show=False)

def _get_top_n_correlated_with_target(df: pd.DataFrame, target_col: str, numeric_cols: List[str], top_n: int) -> List[str]:

    if df is None or target_col not in df.columns:
        print(f"경고: 대상 컬럼 '{target_col}'이 데이터프레임에 없거나 데이터프레임이 None입니다.")
        return []
    if not pd.api.types.is_numeric_dtype(df[target_col].dtype):
        print(f"경고: 대상 컬럼 '{target_col}'은 수치형이 아닙니다.")
        return []
    if not numeric_cols:
        print("경고: 상관관계를 비교할 수치형 컬럼 목록이 비어있습니다.")
        return []
    if top_n <= 0:
        return []

    correlations = []
    # numeric_cols에서 target_col과 중복되지 않고, 실제 df에 존재하는 수치형 컬럼만 필터링
    valid_numeric_cols = [
        col for col in numeric_cols
        if col != target_col and col in df.columns and pd.api.types.is_numeric_dtype(df[col].dtype)
    ]

    if not valid_numeric_cols:
        print("경고: 대상 컬럼을 제외하고 유효한 수치형 비교 대상 컬럼이 없습니다.")
        return []

    for col in valid_numeric_cols:
        try:
            # 상관관계 계산 시 NaN 값으로 인한 문제 방지
            temp_df = df[[target_col, col]].dropna()
            if len(temp_df) >= 2:  # 최소 2개 이상의 유효한 데이터 쌍이 있어야 상관관계 계산 가능
                corr_val = temp_df[target_col].corr(temp_df[col])
                if pd.notna(corr_val):
                    correlations.append((col, abs(corr_val))) # 절대값으로 상관계수의 크기 비교
            # else:
            #     print(f"정보: 컬럼 '{col}'과 대상 컬럼 '{target_col}' 간 유효 데이터 부족으로 상관관계 계산 건너뜀.")
        except Exception as e:
            print(f"경고: 컬럼 '{col}'과 대상 컬럼 '{target_col}' 간 상관관계 계산 중 오류 발생: {e}")
            continue

    # 상관계수 절대값을 기준으로 내림차순 정렬
    correlations.sort(key=lambda x: x[1], reverse=True)

    # 상위 N개 컬럼명 추출
    top_n_cols = [col_name for col_name, corr_val in correlations[:top_n]]
    return top_n_cols