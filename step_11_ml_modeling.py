# step_11_ml_modeling.py

import dearpygui.dearpygui as dpg
import sys
import pandas as pd

print("--- 스크립트 실행 환경 정보 ---")
print(f"파이썬 실행 경로: {sys.executable}")
print(f"Pandas 버전: {pd.__version__}")
print("---------------------------------")

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import traceback
import datetime
import uuid
import threading
import queue
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import shutil

# [NEW] AutoGluon 라이브러리 임포트
from autogluon.tabular import TabularPredictor
import shap

AUTOGLUON_AVAILABLE = True
SHAP_AVAILABLE = True

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- DPG Tags ---
TAG_S11_GROUP = "step11_ml_modeling_group"
TAG_S11_MAIN_TAB_BAR = "step11_main_tab_bar"
TAG_S11_MODELING_TAB = "step11_modeling_tab"
TAG_S11_INFERENCE_TAB = "step11_inference_tab"

# Modeling Tab UI Tags
TAG_S11_DF_SELECTOR = "step11_df_selector"
TAG_S11_TARGET_SELECTOR = "step11_target_selector"
TAG_S11_TASK_TYPE_TEXT = "s11_detected_task_text"
TAG_S11_RUN_BUTTON = "step11_run_button"
TAG_S11_LEADERBOARD_TABLE = "step11_leaderboard_table"
TAG_S11_DEEP_DIVE_GROUP = "step11_deep_dive_group"
TAG_S11_LOG_TEXT = "step11_log_text"
TAG_S11_LOG_WINDOW = "step11_log_window"
TAG_S11_PROGRESS_BAR = "step11_progress_bar"

# AutoGluon 관련 UI Tags
TAG_S11_PRESET_SELECTOR = "step11_preset_selector"
TAG_S11_TIME_LIMIT_INPUT = "step11_time_limit_input"

# Inference Tags
TAG_S11_INFERENCE_MODEL_DIALOG = "s11_inference_model_file_dialog"
TAG_S11_INFERENCE_DATA_DIALOG = "s11_inference_data_file_dialog"
TAG_S11_INFERENCE_SAVE_DIALOG = "s11_inference_save_file_dialog"

# [수정] Plotting 및 SHAP 관련 Tags
TAG_S11_RESULTS_TAB_BAR = "s11_results_tab_bar"
TAG_S11_LEADERBOARD_TAB = "s11_leaderboard_tab"
TAG_S11_DEEP_DIVE_TAB = "s11_deep_dive_tab"
TAG_S11_PERFORMANCE_PLOT_WINDOW = "s11_performance_plot_window" # 피처 중요도 플롯용
TAG_S11_PERFORMANCE_PLOT_IMAGE = "s11_performance_plot_image"
TAG_S11_SHAP_DASHBOARD_WINDOW = "s11_shap_dashboard_window" # [신규] SHAP 대시보드용
TAG_S11_SHAP_DASHBOARD_PLACEHOLDER = "s11_shap_dashboard_placeholder" # [신규] SHAP 대시보드 플레이스홀더


# --- Module State ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_texture_tags: List[str] = []
_results_queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None

_leaderboard_results: List[Dict] = []
_current_deep_dive_model_name: Optional[str] = None
_current_predictor_path: Optional[str] = None
_current_predictor: Optional[TabularPredictor] = None

_inference_predictor: Optional[TabularPredictor] = None
_inference_df: Optional[pd.DataFrame] = None
_inference_result_df: Optional[pd.DataFrame] = None

# --- 유틸리티 및 백그라운드 작업 함수들 (기존과 거의 동일) ---

def _log_message(message: str, level: str = "INFO"):
    # (기존 코드와 동일)
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_S11_LOG_TEXT): return
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {level}: {message}"
    current_log = dpg.get_value(TAG_S11_LOG_TEXT)
    if "AutoGluon 훈련을 시작하세요." in current_log: current_log = ""
    new_log = f"{current_log}\n{formatted_msg}"
    if len(new_log) > 20000: new_log = new_log[-20000:]
    dpg.set_value(TAG_S11_LOG_TEXT, new_log)
    if dpg.does_item_exist(TAG_S11_LOG_WINDOW): dpg.set_y_scroll(TAG_S11_LOG_WINDOW, dpg.get_y_scroll_max(TAG_S11_LOG_WINDOW))

def _start_background_task(target_func, args=()):
    # (기존 코드와 동일)
    global _worker_thread
    if not AUTOGLUON_AVAILABLE:
        _util_funcs['_show_simple_modal_message']("오류", "AutoGluon이 설치되어 있지 않습니다. AutoML 기능을 사용할 수 없습니다.")
        return
    if _worker_thread and _worker_thread.is_alive():
        _util_funcs['_show_simple_modal_message']("작업 중", "이전 작업이 아직 실행 중입니다. 잠시 후 다시 시도해주세요."); return
    _worker_thread = threading.Thread(target=target_func, args=args, daemon=True)
    _worker_thread.start()
    _update_progress(0.0, "작업을 시작했습니다...")

def _run_autogluon_fit_thread(df: pd.DataFrame, target_name: str, preset: str, time_limit: int):
    # (기존 코드와 동일)
    global _current_predictor_path, _current_predictor
    try:
        _results_queue.put({"type": "progress", "value": 0.05, "log": "AutoGluon 설정 및 데이터 준비 중..."})
        output_dir_base = "autogluon_models"
        if os.path.exists(output_dir_base):
             shutil.rmtree(output_dir_base)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir_base, f"ag_{timestamp}")
        predictor = TabularPredictor(label=target_name, path=save_path, problem_type=None)
        _results_queue.put({"type": "progress", "value": 0.1, "log": f"'{preset}' 프리셋으로 모델 훈련 시작 (시간 제한: {time_limit}초)..."})
        predictor.fit(train_data=df, presets=preset, time_limit=time_limit)
        _results_queue.put({"type": "progress", "value": 0.9, "log": "리더보드 생성 중..."})
        leaderboard_df = predictor.leaderboard(df, silent=True)
        _current_predictor_path = predictor.path
        _current_predictor = predictor
        _results_queue.put({
            "type": "autogluon_result",
            "leaderboard": leaderboard_df.to_dict('records'),
            "predictor_path": predictor.path
        })
    except Exception as e:
        error_msg = f"AutoGluon 훈련 오류: {str(e)}\n{traceback.format_exc()}"
        _results_queue.put({"type": "error", "log": error_msg})

def _run_feature_importance_thread():
    # (기존 코드와 거의 동일, 결과 타입만 변경)
    try:
        if not _current_predictor:
            _results_queue.put({"type": "error", "log": "분석할 Predictor가 로드되지 않았습니다."}); return
        _results_queue.put({"type": "progress", "value": 0.1, "log": "피처 중요도 계산 중..."})
        df_name = dpg.get_value(TAG_S11_DF_SELECTOR)
        all_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
        data_for_analysis = all_dfs.get(df_name)
        if data_for_analysis is None: raise ValueError(f"'{df_name}' 데이터프레임을 현재 상태에서 불러올 수 없습니다.")
        feature_importance_df = _current_predictor.feature_importance(data=data_for_analysis, feature_stage='original')
        _results_queue.put({"type": "progress", "value": 0.8, "log": "피처 중요도 시각화 생성 중..."})
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x=feature_importance_df['importance'].head(20), y=feature_importance_df.index[:20], ax=ax)
        ax.set_title('Feature Importance (Permutation)', fontsize=16); ax.set_xlabel('Importance Score', fontsize=12); ax.set_ylabel('Features', fontsize=12)
        plt.tight_layout()
        texture_tag, width, height, _ = (None, 0, 0, None)
        if _util_funcs and 'plot_to_dpg_texture' in _util_funcs: texture_tag, width, height, _ = _util_funcs['plot_to_dpg_texture'](fig)
        plt.close(fig)
        if texture_tag:
            # [수정] 피처 중요도 플롯 전용 결과 타입을 사용
            _results_queue.put({"type": "performance_plot_result", "texture_tag": texture_tag, "width": width, "height": height})
        else:
            _results_queue.put({"type": "error", "log": "피처 중요도 플롯을 DPG 텍스처로 변환하는 데 실패했습니다."})
    except Exception as e:
        error_msg = f"피처 중요도 분석 오류: {str(e)}\n{traceback.format_exc()}"
        _results_queue.put({"type": "error", "log": error_msg})

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    # (기존 코드와 동일)
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    main_callbacks['register_step_group_tag'](step_name, TAG_S11_GROUP)
    with dpg.file_dialog(directory_selector=True, show=False, callback=_on_model_dir_selected, id=TAG_S11_INFERENCE_MODEL_DIALOG, width=700, height=400, modal=True): pass
    with dpg.file_dialog(directory_selector=False, show=False, callback=_on_inference_data_selected, id=TAG_S11_INFERENCE_DATA_DIALOG, width=700, height=400, modal=True):
        dpg.add_file_extension(".csv"); dpg.add_file_extension(".parquet")
    with dpg.file_dialog(directory_selector=False, show=False, callback=_on_save_inference_result_selected, id=TAG_S11_INFERENCE_SAVE_DIALOG, width=700, height=400, modal=True, default_filename="predictions.csv"):
        dpg.add_file_extension(".csv", color=(0, 255, 0, 255))
    with dpg.group(tag=TAG_S11_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} (Powered by AutoGluon) ---"); dpg.add_separator()
        with dpg.tab_bar(tag=TAG_S11_MAIN_TAB_BAR):
            with dpg.tab(label="모델링 (Modeling)", tag=TAG_S11_MODELING_TAB):
                _create_modeling_tab_ui()
            with dpg.tab(label="추론 (Inference)", tag=TAG_S11_INFERENCE_TAB):
                _create_inference_tab_ui()
    main_callbacks['register_module_updater'](step_name, update_ui)

def _create_modeling_tab_ui():
    with dpg.group(tag="s11_modeling_main_group"):
        with dpg.group(horizontal=True):
            # --- 좌측 설정 패널 ---
            with dpg.group(width=200):
                dpg.add_text("1. 설정 (Setup)", color=(255, 255, 0)); dpg.add_separator()
                dpg.add_text("데이터 소스 선택:"); dpg.add_combo(label="", tag=TAG_S11_DF_SELECTOR, width=-1, callback=_on_df_or_target_selected)
                dpg.add_text("타겟 변수 (y) 선택:"); dpg.add_combo(label="", tag=TAG_S11_TARGET_SELECTOR, width=-1, callback=_on_df_or_target_selected)
                with dpg.group(horizontal=True):
                    dpg.add_text("감지된 태스크 타입:"); dpg.add_text("(데이터 선택)", tag=TAG_S11_TASK_TYPE_TEXT, color=(255, 255, 0))
                dpg.add_separator()
                dpg.add_text("2. AutoGluon 설정", color=(255, 255, 0))
                dpg.add_text("품질 프리셋 (Quality Preset):"); dpg.add_combo(items=['best_quality', 'high_quality', 'good_quality', 'medium_quality'], default_value="medium_quality", width=-1, tag=TAG_S11_PRESET_SELECTOR)
                dpg.add_text("훈련 시간 제한 (초):"); dpg.add_input_int(default_value=60, width=-1, tag=TAG_S11_TIME_LIMIT_INPUT, min_value=10, min_clamped=True)
                dpg.add_spacer(height=10)
                dpg.add_button(label="🚀 AutoGluon 훈련 실행", tag=TAG_S11_RUN_BUTTON, width=-1, height=40, callback=_start_autogluon_fit_callback)
            
            # --- 우측 결과 패널 (탭 구조) ---
            with dpg.group(): 
                with dpg.tab_bar(tag=TAG_S11_RESULTS_TAB_BAR):
                    # --- 리더보드 탭 ---
                    with dpg.tab(label="리더보드", tag=TAG_S11_LEADERBOARD_TAB):
                        dpg.add_text("모델 성능 리더보드", color=(255, 255, 0))
                        dpg.add_table(tag=TAG_S11_LEADERBOARD_TABLE, header_row=True, resizable=True, reorderable=True, borders_innerV=True, borders_outerH=True, height=600, scrollX=True, policy=dpg.mvTable_SizingFixedFit)
                    
                    # --- [수정] 심층 분석 탭 UI 재구성 ---
                    with dpg.tab(label="심층 분석", tag=TAG_S11_DEEP_DIVE_TAB):
                        with dpg.group(tag=TAG_S11_DEEP_DIVE_GROUP, show=False):
                            dpg.add_text("Selected Model: ", tag="s11_deep_dive_model_name"); dpg.add_separator()
                            with dpg.tab_bar():
                                with dpg.tab(label="성능 (Performance)"):
                                    dpg.add_text("모델의 상세 성능 지표입니다.")
                                    with dpg.table(tag="s11_deep_dive_perf_table", header_row=True, height=300):
                                        dpg.add_table_column(label="Metric"); dpg.add_table_column(label="Value")
                                
                                with dpg.tab(label="피처 중요도 (Feature Importance)"):
                                    dpg.add_text("모델이 예측에 중요하게 사용한 피처 목록입니다. (Permutation Importance)", wrap=500)
                                    dpg.add_button(label="피처 중요도 분석 실행", callback=_run_feature_importance_analysis)
                                    dpg.add_separator()
                                    with dpg.child_window(tag=TAG_S11_PERFORMANCE_PLOT_WINDOW, border=True, height=-1):
                                        dpg.add_text("분석을 실행하면 결과가 여기에 표시됩니다.", tag="s11_perf_plot_placeholder")

                                # --- [신규] SHAP 종합 분석 탭 ---
                                with dpg.tab(label="SHAP 종합 분석", show=SHAP_AVAILABLE):
                                    _create_shap_dashboard_tab_ui()

        # --- 하단 로그 패널 ---
        dpg.add_separator()
        with dpg.group():
            dpg.add_text("진행 상황 및 로그", color=(100, 200, 255)); dpg.add_progress_bar(tag=TAG_S11_PROGRESS_BAR, default_value=0.0, width=-1)
            with dpg.child_window(height=150, tag=TAG_S11_LOG_WINDOW, border=True):
                dpg.add_input_text(default_value="AutoGluon 훈련을 시작하세요.", tag=TAG_S11_LOG_TEXT, multiline=True, readonly=True, width=-1, height=-1)

# --- [신규] SHAP 종합 분석 탭 UI 생성 함수 ---
def _create_shap_dashboard_tab_ui():
    dpg.add_text("SHAP(SHapley Additive exPlanations) 분석은 모델의 예측 결과를 설명하는 강력한 기법입니다.", wrap=600)
    dpg.add_separator()
    
    with dpg.group(horizontal=True):
        dpg.add_text("분석 샘플 수:")
        dpg.add_input_int(tag="s11_shap_sample_size_input_dashboard", default_value=500, min_value=10, max_value=500, width=120)
        dpg.add_text("(주의: 값이 크면 분석 시간이 매우 길어집니다)", color=(255, 180, 0))

    dpg.add_button(label="📊 SHAP 종합 분석 실행", callback=_run_comprehensive_shap_analysis_callback, height=30)
    dpg.add_separator()
    
    with dpg.child_window(tag=TAG_S11_SHAP_DASHBOARD_WINDOW, border=True, height=-1):
        dpg.add_text("분석을 실행하면 결과 대시보드가 여기에 표시됩니다.", tag=TAG_S11_SHAP_DASHBOARD_PLACEHOLDER)


def _create_inference_tab_ui():
    # (기존 코드와 동일)
    dpg.add_text("학습된 AutoGluon 모델을 사용하여 새로운 데이터에 대한 예측을 수행합니다.", wrap=500); dpg.add_separator()
    dpg.add_button(label="📂 저장된 모델 폴더 불러오기", callback=lambda: dpg.show_item(TAG_S11_INFERENCE_MODEL_DIALOG)); dpg.add_text("불러온 모델: 없음", tag="s11_inference_model_path")
    dpg.add_separator()
    dpg.add_button(label="📄 예측할 데이터 불러오기 (.csv/.parquet)", callback=lambda: dpg.show_item(TAG_S11_INFERENCE_DATA_DIALOG)); dpg.add_text("예측할 데이터: 없음", tag="s11_inference_data_path")
    dpg.add_table(tag="s11_inference_data_preview", header_row=True, height=150)
    dpg.add_separator()
    dpg.add_button(label="실행", width=-1, callback=_run_inference, height=30, tag="s11_run_inference_button", enabled=False)
    dpg.add_separator(); dpg.add_text("예측 결과")
    with dpg.group(horizontal=True):
        dpg.add_text("", tag="s11_inference_result_count")
        dpg.add_button(label="결과 다운로드", show=False, tag="s11_inference_download_button", callback=_download_inference_result)
    dpg.add_table(tag="s11_inference_result_table", header_row=True, height=200, resizable=True)


# --- 콜백 함수들 ---

def _on_df_or_target_selected(sender, app_data, user_data):
    # (기존 코드와 동일)
    df_name, target_name = dpg.get_value(TAG_S11_DF_SELECTOR), dpg.get_value(TAG_S11_TARGET_SELECTOR)
    all_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    if not df_name or df_name not in all_dfs: return
    df = all_dfs[df_name]
    if sender == TAG_S11_DF_SELECTOR:
        cols = [""] + df.columns.tolist()
        dpg.configure_item(TAG_S11_TARGET_SELECTOR, items=cols); dpg.set_value(TAG_S11_TARGET_SELECTOR, "")
    if target_name and target_name in df.columns:
        y = df[target_name]
        task_type = "Regression" if pd.api.types.is_numeric_dtype(y) and y.nunique() >= 25 else "Classification"
        dpg.set_value(TAG_S11_TASK_TYPE_TEXT, f"{task_type}")
    else:
        dpg.set_value(TAG_S11_TASK_TYPE_TEXT, "(타겟 선택)")

def _start_autogluon_fit_callback():
    # (기존 코드와 동일)
    global _leaderboard_results, _current_predictor, _current_predictor_path
    df_name = dpg.get_value(TAG_S11_DF_SELECTOR)
    target_name = dpg.get_value(TAG_S11_TARGET_SELECTOR)
    if not df_name or not target_name: _util_funcs['_show_simple_modal_message']("설정 오류", "데이터 소스와 타겟 변수를 모두 선택해주세요."); return
    preset = dpg.get_value(TAG_S11_PRESET_SELECTOR)
    time_limit = dpg.get_value(TAG_S11_TIME_LIMIT_INPUT)
    df = _module_main_callbacks.get('get_all_available_dfs')().get(df_name)
    if df is None: _util_funcs['_show_simple_modal_message']("오류", "선택된 데이터프레임을 찾을 수 없습니다."); return
    _leaderboard_results.clear()
    _current_predictor, _current_predictor_path = None, None
    _update_leaderboard_display()
    dpg.configure_item(TAG_S11_DEEP_DIVE_GROUP, show=False)
    if dpg.does_item_exist(TAG_S11_RESULTS_TAB_BAR):
        dpg.set_value(TAG_S11_RESULTS_TAB_BAR, TAG_S11_LEADERBOARD_TAB)
    _start_background_task(_run_autogluon_fit_thread, args=(df.copy(), target_name, preset, time_limit))

def _update_leaderboard_display():
    # (기존 코드와 동일)
    table = TAG_S11_LEADERBOARD_TABLE
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(table): return
    dpg.delete_item(table, children_only=True)
    if not _leaderboard_results:
        dpg.add_table_column(label="알림", parent=table)
        with dpg.table_row(parent=table): dpg.add_text("AutoGluon 훈련을 실행하여 결과를 확인하세요.")
        return
    headers = list(_leaderboard_results[0].keys())
    display_headers = ['model', 'score_val', 'eval_metric', 'pred_time_val', 'fit_time', 'stack_level']
    final_headers = display_headers + [h for h in headers if h not in display_headers]
    for key in final_headers: 
        if key in _leaderboard_results[0]: dpg.add_table_column(label=key, parent=table)
    dpg.add_table_column(label="분석", parent=table, width=100)
    for res in _leaderboard_results:
        with dpg.table_row(parent=table):
            for key in final_headers:
                 if key in res:
                    val = res[key]
                    s_val = f"{val:.4f}" if isinstance(val, (float, np.number)) else str(val)
                    dpg.add_text(s_val)
            with dpg.table_cell():
                dpg.add_button(label="상세", user_data=res.get("model"), callback=_select_model_for_deep_dive)

def _select_model_for_deep_dive(sender, app_data, user_data_model_name):
    # [수정] SHAP 관련 로직 제거, UI 초기화 로직 추가
    global _current_deep_dive_model_name
    if not _current_predictor: _log_message("오류: Predictor가 로드되지 않았습니다.", "ERROR"); return
    _current_deep_dive_model_name = user_data_model_name
    
    dpg.configure_item(TAG_S11_DEEP_DIVE_GROUP, show=True)
    if dpg.does_item_exist(TAG_S11_RESULTS_TAB_BAR): dpg.set_value(TAG_S11_RESULTS_TAB_BAR, TAG_S11_DEEP_DIVE_TAB)
    dpg.set_value("s11_deep_dive_model_name", f"Selected Model: {user_data_model_name}")
    
    perf_table = "s11_deep_dive_perf_table"
    dpg.delete_item(perf_table, children_only=True); dpg.add_table_column(label="Metric", parent=perf_table); dpg.add_table_column(label="Value", parent=perf_table)

    try:
        df_name = dpg.get_value(TAG_S11_DF_SELECTOR)
        all_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
        df = all_dfs.get(df_name)
        if df is None: raise ValueError(f"'{df_name}' 데이터프레임을 현재 상태에서 불러올 수 없습니다.")
        
        model_perf = _current_predictor.evaluate(df, model=user_data_model_name)
        for metric, value in model_perf.items():
            with dpg.table_row(parent=perf_table): dpg.add_text(metric); dpg.add_text(f"{value:.4f}")
    except Exception as e:
        with dpg.table_row(parent=perf_table): dpg.add_text("Error"); dpg.add_text(f"성능 로드 실패: {str(e)}")
        print(f"심층 분석 성능 로드 중 오류 발생: {e}\n{traceback.format_exc()}")

    _log_message(f"'{user_data_model_name}' 모델이 심층 분석 대상으로 선택되었습니다.")
    
    # [수정] 이전 분석 결과(플롯)들을 초기화
    if dpg.does_item_exist(TAG_S11_PERFORMANCE_PLOT_IMAGE): dpg.delete_item(TAG_S11_PERFORMANCE_PLOT_IMAGE)
    if dpg.does_item_exist("s11_perf_plot_placeholder"): dpg.configure_item("s11_perf_plot_placeholder", show=True)
    
    if dpg.does_item_exist(TAG_S11_SHAP_DASHBOARD_WINDOW):
        dpg.delete_item(TAG_S11_SHAP_DASHBOARD_WINDOW, children_only=True)
        dpg.add_text("분석을 실행하면 결과 대시보드가 여기에 표시됩니다.", tag=TAG_S11_SHAP_DASHBOARD_PLACEHOLDER, parent=TAG_S11_SHAP_DASHBOARD_WINDOW)

def _run_feature_importance_analysis():
    if not _current_predictor: _util_funcs['_show_simple_modal_message']("오류", "먼저 모델을 훈련시켜주세요."); return
    if dpg.does_item_exist(TAG_S11_PERFORMANCE_PLOT_IMAGE): dpg.delete_item(TAG_S11_PERFORMANCE_PLOT_IMAGE)
    if dpg.does_item_exist("s11_perf_plot_placeholder"): dpg.configure_item("s11_perf_plot_placeholder", show=True)
    _start_background_task(_run_feature_importance_thread)

def _run_comprehensive_shap_analysis_callback():
    if not _current_predictor or not _current_deep_dive_model_name:
        _util_funcs['_show_simple_modal_message']("오류", "먼저 리더보드에서 분석할 모델을 선택해주세요."); return

    # 분석 시작 시, 대시보드 창을 비우고 "로딩 중" 메시지를 표시합니다.
    if dpg.does_item_exist(TAG_S11_SHAP_DASHBOARD_WINDOW):
        dpg.delete_item(TAG_S11_SHAP_DASHBOARD_WINDOW, children_only=True)
        dpg.add_text("SHAP 종합 분석을 시작합니다. 잠시만 기다려주세요...", parent=TAG_S11_SHAP_DASHBOARD_WINDOW, color=(255,255,0))

    # 백그라운드에서 SHAP 분석 스레드를 시작합니다.
    _start_background_task(_run_comprehensive_shap_thread)

def _run_comprehensive_shap_thread():
    # 이 프린트문이 로그에 보이는지 확인하는 것이 중요합니다.
    print("--- [v3] 문제 코드 완전 제거 버전 실행 확인 ---")
    _log_message("--- [v3] 코드 실행 확인 ---", "DEBUG") # GUI 로그에도 표시

    try:
        # --- 준비 과정 (기존과 동일) ---
        if not _current_predictor or not _current_deep_dive_model_name:
            _results_queue.put({"type": "error", "log": "SHAP: 분석할 모델이 선택되지 않았습니다."}); return
        _results_queue.put({"type": "progress", "value": 0.05, "log": "SHAP 종합 분석 시작: 데이터 준비 중..."})
        df_name = dpg.get_value(TAG_S11_DF_SELECTOR)
        all_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
        data_for_analysis = all_dfs.get(df_name)
        if data_for_analysis is None: raise ValueError(f"'{df_name}' 데이터프레임을 현재 불러올 수 없습니다.")
        sample_size = dpg.get_value("s11_shap_sample_size_input_dashboard")
        if len(data_for_analysis) > sample_size:
            _log_message(f"SHAP: 데이터가 많아 {sample_size}개 행으로 샘플링합니다.", "WARN")
            data_sample = data_for_analysis.sample(n=sample_size, random_state=42)
        else:
            data_sample = data_for_analysis.copy()
        _results_queue.put({"type": "progress", "value": 0.2, "log": "SHAP 값 계산 중... (시간 소요)"})
        X_sample = data_sample.drop(columns=[_current_predictor.label])

        X_sample_for_plotting = X_sample.copy()

        # --- 이하 SHAP 값 계산 및 플롯 생성 (기존과 동일) ---
        X_sample_numeric = X_sample.copy()
        for col in X_sample_numeric.columns:
            if not pd.api.types.is_numeric_dtype(X_sample_numeric[col]):
                X_sample_numeric[col] = X_sample_numeric[col].astype('category').cat.codes

        if _current_predictor.problem_type in ['binary', 'multiclass']:
            def predict_fn(x):
                df_pred = pd.DataFrame(x, columns=X_sample.columns)
                return _current_predictor.predict_proba(df_pred, model=_current_deep_dive_model_name)
        else:
            def predict_fn(x):
                df_pred = pd.DataFrame(x, columns=X_sample.columns)
                return _current_predictor.predict(df_pred, model=_current_deep_dive_model_name)

        masker = shap.maskers.Partition(X_sample_numeric)
        explainer = shap.Explainer(predict_fn, masker)
        shap_values = explainer(X_sample_numeric)
        shap_values.data = X_sample.values

        shap_values_to_plot = shap_values
        if _current_predictor.problem_type == 'binary':
            try:
                positive_class_index = _current_predictor.class_labels.index(_current_predictor.positive_class)
                shap_values_to_plot = shap_values[..., positive_class_index]
            except (ValueError, AttributeError, IndexError):
                shap_values_to_plot = shap_values[..., 1]

        _results_queue.put({"type": "progress", "value": 0.6, "log": "SHAP 값 계산 완료. 플롯 생성 중..."})

        all_plots_data = []
        plt.style.use('seaborn-v0_8-darkgrid')

        try:
            shap.plots.bar(shap_values_to_plot, show=False)
            fig = plt.gcf(); plt.title('SHAP Global Feature Importance'); plt.tight_layout()
            if texture_info := _util_funcs['plot_to_dpg_texture'](fig):
                all_plots_data.append({'title': '전역 피처 중요도 (Bar Plot)', 'texture_info': texture_info})
            plt.close(fig)
        except Exception as e: _log_message(f"SHAP Bar Plot 생성 실패: {e}", "ERROR")

        try:
            shap.summary_plot(shap_values_to_plot, X_sample_for_plotting, show=False, plot_type="dot")
            fig = plt.gcf(); plt.title('SHAP Summary Plot'); plt.tight_layout()
            # [수정 1] 함수 이름의 오타를 수정합니다. ('plot_to_d_texture' -> 'plot_to_dpg_texture')
            if texture_info := _util_funcs['plot_to_dpg_texture'](fig):
                all_plots_data.append({'title': '특성별 영향 요약 (Summary Plot)', 'texture_info': texture_info})
            plt.close(fig)
        except Exception as e: _log_message(f"SHAP Summary Plot 생성 실패: {e}", "ERROR")

        try:
            top_feature_indices = np.argsort(np.abs(shap_values_to_plot.values).mean(0))[-4:]
            top_features = X_sample.columns[top_feature_indices]
            for feature in reversed(top_features):
                # [수정 2] 결측치로 인한 오류를 방지하기 위해 interaction_index를 'auto'에서 None으로 변경합니다.
                shap.dependence_plot(feature, shap_values_to_plot.values, X_sample_for_plotting, show=False, interaction_index=None)
                fig = plt.gcf(); plt.title(f'SHAP Dependence Plot for "{feature}"'); plt.tight_layout()
                if texture_info := _util_funcs['plot_to_dpg_texture'](fig):
                    all_plots_data.append({'title': f'의존성 플롯: {feature}', 'texture_info': texture_info})
                plt.close(fig)
        except Exception as e:
            error_detail = f"SHAP Dependence Plot 생성 중 오류 발생: {e}"
            _log_message(error_detail, "WARN")

        if all_plots_data:
            _results_queue.put({"type": "shap_dashboard_result", "plots": all_plots_data})
        else:
            _results_queue.put({"type": "error", "log": "생성된 SHAP 플롯이 없습니다."})

    except Exception as e:
        error_msg = f"SHAP 종합 분석 오류: {str(e)}\n{traceback.format_exc()}"
        _results_queue.put({"type": "error", "log": error_msg})

# --- [수정] UI 업데이트 처리 함수 ---
def _check_for_updates():
    global _worker_thread, _leaderboard_results, _current_predictor, _current_predictor_path
    try:
        result = _results_queue.get_nowait()
        if result["type"] == "progress": _update_progress(result.get("value", 0), result.get("log", ""))
        elif result["type"] == "error": _log_message(result["log"], "ERROR"); _update_progress(0.0, "작업 실패.")
        elif result["type"] == "autogluon_result":
            _leaderboard_results = result.get("leaderboard", [])
            _current_predictor_path = result.get("predictor_path")
            if _current_predictor_path: _current_predictor = TabularPredictor.load(_current_predictor_path)
            _update_leaderboard_display(); _update_progress(1.0, "AutoML 훈련 완료."); time.sleep(1); _update_progress(0.0)
        
        # [수정] 피처 중요도 플롯 결과 처리
        elif result["type"] == "performance_plot_result":
            if dpg.does_item_exist(TAG_S11_PERFORMANCE_PLOT_IMAGE): dpg.delete_item(TAG_S11_PERFORMANCE_PLOT_IMAGE)
            if dpg.does_item_exist("s11_perf_plot_placeholder"): dpg.configure_item("s11_perf_plot_placeholder", show=False)
            dpg.add_image(result["texture_tag"], width=result["width"], height=result["height"], parent=TAG_S11_PERFORMANCE_PLOT_WINDOW, tag=TAG_S11_PERFORMANCE_PLOT_IMAGE)
            _update_progress(1.0, "피처 중요도 분석 완료."); time.sleep(1); _update_progress(0.0)
            _texture_tags.append(result["texture_tag"])

        # --- [신규] SHAP 대시보드 결과 처리 ---
        elif result["type"] == "shap_dashboard_result":
            plots_data = result.get("plots", [])
            dpg.delete_item(TAG_S11_SHAP_DASHBOARD_WINDOW, children_only=True)
            if not plots_data:
                dpg.add_text("오류: 생성된 SHAP 플롯이 없습니다.", parent=TAG_S11_SHAP_DASHBOARD_WINDOW, color=(255,0,0))
                return

            new_texture_tags = []
            for plot in plots_data:
                title = plot['title']
                texture_tag, width, height, _ = plot['texture_info']
                new_texture_tags.append(texture_tag)
                dpg.add_text(title, parent=TAG_S11_SHAP_DASHBOARD_WINDOW, color=(100, 200, 255))
                dpg.add_image(texture_tag, width=width, height=height, parent=TAG_S11_SHAP_DASHBOARD_WINDOW)
                dpg.add_separator(parent=TAG_S11_SHAP_DASHBOARD_WINDOW)
            
            _texture_tags.extend(new_texture_tags)
            _update_progress(1.0, "SHAP 종합 분석 완료."); time.sleep(1); _update_progress(0.0)

        _results_queue.task_done()
    except queue.Empty: pass
    except Exception as e: print(f"업데이트 확인 중 오류 발생: {e}\n{traceback.format_exc()}")
    if _worker_thread and not _worker_thread.is_alive(): _worker_thread = None


def _update_progress(value: float, message: str = ""):
    # (기존 코드와 동일)
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_S11_PROGRESS_BAR): dpg.set_value(TAG_S11_PROGRESS_BAR, value)
    if message: _log_message(message)

def update_ui():
    # (기존 코드와 동일)
    if not _module_main_callbacks or not dpg.is_dearpygui_running(): return
    all_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    if dpg.does_item_exist(TAG_S11_DF_SELECTOR):
        df_names = [""] + list(all_dfs.keys())
        dpg.configure_item(TAG_S11_DF_SELECTOR, items=df_names)

def reset_state():
    # [수정] 상태 초기화 시 SHAP 대시보드도 초기화
    global _worker_thread, _leaderboard_results, _texture_tags, _current_deep_dive_model_name
    global _current_predictor, _current_predictor_path, _inference_predictor, _inference_df, _inference_result_df
    
    if _worker_thread and _worker_thread.is_alive(): return
    
    _leaderboard_results.clear(); _current_deep_dive_model_name = None
    _current_predictor, _current_predictor_path = None, None
    _inference_predictor, _inference_df, _inference_result_df = None, None, None

    for tag in _texture_tags:
        if dpg.does_item_exist(tag): dpg.delete_item(tag)
    _texture_tags.clear()

    if os.path.exists("autogluon_models"):
        try: shutil.rmtree("autogluon_models"); _log_message("이전 AutoGluon 모델 결과 폴더를 삭제했습니다.", "INFO")
        except Exception as e: _log_message(f"모델 폴더 삭제 실패: {e}", "ERROR")

    if dpg.is_dearpygui_running():
        _update_leaderboard_display()
        dpg.configure_item(TAG_S11_DEEP_DIVE_GROUP, show=False)
        # [수정] 모든 플롯 영역 초기화
        if dpg.does_item_exist(TAG_S11_PERFORMANCE_PLOT_IMAGE): dpg.delete_item(TAG_S11_PERFORMANCE_PLOT_IMAGE)
        if dpg.does_item_exist("s11_perf_plot_placeholder"): dpg.configure_item("s11_perf_plot_placeholder", show=True)
        if dpg.does_item_exist(TAG_S11_SHAP_DASHBOARD_WINDOW):
            dpg.delete_item(TAG_S11_SHAP_DASHBOARD_WINDOW, children_only=True)
            dpg.add_text("분석을 실행하면 결과 대시보드가 여기에 표시됩니다.", tag=TAG_S11_SHAP_DASHBOARD_PLACEHOLDER, parent=TAG_S11_SHAP_DASHBOARD_WINDOW)

        # 추론 탭 초기화 (기존과 동일)
        dpg.set_value("s11_inference_model_path", "불러온 모델: 없음"); dpg.set_value("s11_inference_data_path", "예측할 데이터: 없음")
        if dpg.does_item_exist("s11_inference_data_preview"): dpg.delete_item("s11_inference_data_preview", children_only=True)
        if dpg.does_item_exist("s11_inference_result_table"): dpg.delete_item("s11_inference_result_table", children_only=True)
        dpg.configure_item("s11_inference_download_button", show=False)
        _check_inference_readiness()
        update_ui()
    _log_message("ML 모델링 상태가 초기화되었습니다.", "INFO")

# --- Inference Tab Functions (기존 코드와 동일) ---
def _check_inference_readiness():
    ready = _inference_predictor is not None and _inference_df is not None
    if dpg.does_item_exist("s11_run_inference_button"): dpg.configure_item("s11_run_inference_button", enabled=ready)

def _on_model_dir_selected(sender, app_data):
    global _inference_predictor
    try:
        dir_path = app_data['file_path_name']
        _inference_predictor = TabularPredictor.load(dir_path)
        dpg.set_value("s11_inference_model_path", f"불러온 모델: {os.path.basename(dir_path)}")
        _log_message(f"추론 모델 '{os.path.basename(dir_path)}' 로드 완료."); _check_inference_readiness()
    except Exception as e:
        _inference_predictor = None; _util_funcs['_show_simple_modal_message']("모델 로드 실패", f"AutoGluon 모델 폴더 로드 중 오류 발생:\n{e}")
        dpg.set_value("s11_inference_model_path", "불러온 모델: 없음 (로드 실패)"); _check_inference_readiness()

def _on_inference_data_selected(sender, app_data):
    global _inference_df
    try:
        file_path = app_data['file_path_name']
        if file_path.endswith('.csv'): _inference_df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'): _inference_df = pd.read_parquet(file_path)
        else: raise ValueError("지원하지 않는 파일 형식입니다.")
        if _inference_predictor and _inference_predictor.label in _inference_df.columns:
             _inference_df = _inference_df.drop(columns=[_inference_predictor.label])
             _log_message(f"추론 데이터에서 타겟 변수 '{_inference_predictor.label}'를 자동으로 제거했습니다.", "WARN")
        dpg.set_value("s11_inference_data_path", f"예측할 데이터: {os.path.basename(file_path)} ({_inference_df.shape})")
        if _util_funcs and 'create_table_with_large_data_preview' in _util_funcs: _util_funcs['create_table_with_large_data_preview']("s11_inference_data_preview", _inference_df)
        _log_message(f"추론 데이터 '{os.path.basename(file_path)}' 로드 완료."); _check_inference_readiness()
    except Exception as e:
        _inference_df = None; _util_funcs['_show_simple_modal_message']("데이터 로드 실패", f"데이터 파일 로드 중 오류 발생:\n{e}")
        dpg.set_value("s11_inference_data_path", "예측할 데이터: 없음 (로드 실패)"); _check_inference_readiness()

def _run_inference():
    global _inference_result_df
    if _inference_predictor is None or _inference_df is None: _util_funcs['_show_simple_modal_message']("오류", "모델과 데이터를 모두 불러와야 합니다."); return
    try:
        _log_message("추론 시작...")
        predictions = _inference_predictor.predict(_inference_df)
        result_df = _inference_df.copy(); result_df['prediction'] = predictions
        if _inference_predictor.problem_type != 'regression':
            pred_probas = _inference_predictor.predict_proba(_inference_df)
            for col in pred_probas.columns: result_df[f'proba_{col}'] = pred_probas[col]
        _inference_result_df = result_df
        if _util_funcs and 'create_table_with_large_data_preview' in _util_funcs: _util_funcs['create_table_with_large_data_preview']("s11_inference_result_table", _inference_result_df)
        dpg.set_value("s11_inference_result_count", f"총 {len(_inference_result_df)}개 행 예측 완료"); dpg.configure_item("s11_inference_download_button", show=True)
        _log_message("추론 완료.")
    except Exception as e:
        _util_funcs['_show_simple_modal_message']("추론 오류", f"예측 중 오류 발생:\n{e}\n\n입력 데이터의 컬럼과 형식이 모델 훈련 시와 일치하는지 확인하세요.")
        _log_message(f"추론 오류: {e}", "ERROR")

def _on_save_inference_result_selected(sender, app_data):
    try:
        file_path = app_data['file_path_name']
        if _inference_result_df is not None:
            _inference_result_df.to_csv(file_path, index=False)
            _util_funcs['_show_simple_modal_message']("저장 완료", f"추론 결과가 다음 경로에 저장되었습니다:\n{file_path}")
            _log_message(f"추론 결과가 '{file_path}'에 저장되었습니다.")
    except Exception as e: _util_funcs['_show_simple_modal_message']("저장 실패", f"추론 결과 저장 중 오류 발생:\n{e}")

def _download_inference_result():
    if _inference_result_df is None: _util_funcs['_show_simple_modal_message']("오류", "다운로드할 추론 결과가 없습니다."); return
    dpg.show_item(TAG_S11_INFERENCE_SAVE_DIALOG)