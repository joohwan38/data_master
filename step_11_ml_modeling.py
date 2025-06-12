# step_11_ml_modeling.py - 1단계 구현 (비동기 처리, AutoML, SHAP)

"""
Step 11 ML Modeling & AI 통합 모듈 (Phase 1 Implementation)

강화된 기능:
- 비동기 처리:バックグラウンドでの学習・分析によるUIフリーズ防止
- AutoML: LightAutoML을 이용한 모델 선택, 하이퍼파라미터 튜닝 자동화
- SHAP: 모델 예측에 대한 피처 영향도 및 방향성 분석
- 실시간 모니터링: 학습 과정 로그 및 진행률 표시
"""

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import traceback
import datetime
import pickle
import json
import sys
import uuid
from dataclasses import dataclass, asdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                           mean_squared_error, r2_score, mean_absolute_error, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import threading
import queue
import time
import re

# --- 외부 라이브러리 (설치 필요) ---
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.addons.tabular_interpretation import SSWARM
from lightautoml.tasks import Task
import shap
# from lightautoml.explain.shap import SSWARM

warnings.filterwarnings('ignore')

# --- Classification/Regression Models (기존과 동일) ---
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# ... (기타 모델들은 간결성을 위해 생략)

# --- DPG Tags ---
TAG_S11_GROUP = "step11_ml_modeling_group"
TAG_S11_UPPER_VIZ_WINDOW = "step11_upper_viz_window"
TAG_S11_LOWER_CONTROL_PANEL = "step11_lower_control_panel"
TAG_S11_VIZ_TAB_BAR = "step11_viz_tab_bar"

TAG_S11_EXPERIMENT_TAB = "step11_experiment_tab"
TAG_S11_MONITORING_TAB = "step11_monitoring_tab"
# --- [1단계 추가] ---
TAG_S11_AUTOML_TAB = "step11_automl_tab" 
TAG_S11_AUTOML_CONTROLS_GROUP = "s11_automl_controls_group"
TAG_S11_AUTOML_RESULTS_GROUP = "s11_automl_results_group"
TAG_S11_AUTOML_RUN_BUTTON = "s11_automl_run_button"

TAG_S11_PROGRESS_BAR = "step11_progress_bar"
TAG_S11_LOG_WINDOW = "step11_log_window"
TAG_S11_EXPERIMENT_TABLE = "step11_experiment_table"
TAG_S11_DF_SELECTOR = "step11_df_selector"

# --- ML_ALGORITHMS (기존과 동일) ---
ML_ALGORITHMS = {
    "Classification": {
        "Logistic Regression": {"class": LogisticRegression, "params": {}},
        "Random Forest": {"class": RandomForestClassifier, "params": {"n_estimators": 100}},
    },
    "Regression": {
        "Linear Regression": {"class": LinearRegression, "params": {}},
        "Random Forest": {"class": RandomForestRegressor, "params": {"n_estimators": 100}},
    },
    "Clustering": {} # 간결성 위해 생략
}

# --- Module State 변수 ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_experiments_history: List['ExperimentResult'] = []
_texture_tags: List[str] = []
# --- [1단계 추가] 비동기 처리 관련 변수 ---
_results_queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_detected_task_type: str = "" 


@dataclass
class ExperimentResult:
    """실험 결과를 저장하는 데이터 클래스 (기존과 동일)"""
    id: str
    timestamp: datetime.datetime
    model_name: str
    model_type: str
    algorithm: str
    parameters: Dict[str, Any]
    features: List[str]
    target: str
    metrics: Dict[str, Any]
    training_time: float
    model_object: Any # 학습된 모델 객체 저장
    dataframe_name: str # 어떤 데이터프레임으로 학습했는지 기록
    
    def to_dict(self):
        d = asdict(self)
        d.pop('model_object', None)
        return d
    
def _update_automl_target_combo(df_name: str):
    """(최종 수정) '분석 제외' 컬럼을 UI 목록에서 제거"""
    all_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    if not df_name or df_name not in all_dfs:
        if dpg.does_item_exist("s11_automl_target_combo"): dpg.configure_item("s11_automl_target_combo", items=[])
        if dpg.does_item_exist("s11_automl_detected_task"): dpg.set_value("s11_automl_detected_task", "(select source)")
        return

    df = all_dfs[df_name]
    
    # --- 아래 로직 추가 ---
    # 1. Step 1의 타입 설정을 가져와 '분석 제외' 목록 생성
    step1_type_selections = _module_main_callbacks.get('get_column_analysis_types', lambda: {})()
    cols_to_exclude = {col for col, type_val in step1_type_selections.items() if type_val == "분석에서 제외 (Exclude)"}
    
    # 2. 전체 컬럼 목록에서 제외할 컬럼을 뺀 최종 후보 목록 생성
    candidate_cols = [col for col in df.columns if col not in cols_to_exclude]
    # --- 여기까지 추가 ---
    
    combo_tag = "s11_automl_target_combo"
    if not dpg.does_item_exist(combo_tag): return
        
    # 후보 목록으로 콤보박스 아이템 설정
    dpg.configure_item(combo_tag, items=candidate_cols)
    
    final_target = ""
    if _module_main_callbacks and 'get_selected_target_variable' in _module_main_callbacks:
        global_target = _module_main_callbacks['get_selected_target_variable']()
        # 전역 타겟이 후보 목록에 있을 경우에만 기본값으로 설정
        if global_target and global_target in candidate_cols:
            final_target = global_target
        elif candidate_cols:
            final_target = candidate_cols[0]
            
    if final_target:
        dpg.set_value(combo_tag, final_target)
        _detect_and_update_task_type(df_name, final_target)
    else:
        # 선택할 수 있는 타겟이 하나도 없는 경우
        dpg.set_value(combo_tag, "")
        _detect_and_update_task_type(df_name, "")

def _detect_and_update_task_type(df_name: str, target_name: str):
    """(신설) 선택된 Target 변수를 분석하여 Task 유형을 결정하고 UI를 업데이트"""
    global _detected_task_type

    # 최신 DF 목록을 직접 가져옴
    all_dfs = _module_main_callbacks['get_all_available_dfs']()
    if not df_name or not target_name or df_name not in all_dfs:
        _detected_task_type = ""
        if dpg.does_item_exist("s11_automl_detected_task"):
            dpg.set_value("s11_automl_detected_task", "(error)")
        return

    df = all_dfs[df_name]
    if target_name not in df.columns:
        _detected_task_type = ""
        return

    target_series = df[target_name]
    nunique = target_series.nunique()
    dtype_kind = target_series.dtype.kind
    
    task = 'reg'  # 기본값은 회귀(reg)
    
    # 데이터 타입이 문자열/카테고리이거나, 정수형인데 고유값이 20개 미만인 경우
    if dtype_kind in 'Ocb' or (dtype_kind == 'i' and nunique < 20):
        if nunique == 2:
            task = 'binary'
        else:
            task = 'multiclass'
    
    _detected_task_type = task
    if dpg.does_item_exist("s11_automl_detected_task"):
        dpg.set_value("s11_automl_detected_task", task)

def _on_automl_target_changed(sender, target_name, user_data):
    """(신설) AutoML 타겟 콤보박스 변경 시 콜백"""
    df_name = dpg.get_value(TAG_S11_DF_SELECTOR)
    _detect_and_update_task_type(df_name, target_name)

# --- [1단계] 비동기 작업 래퍼 ---
def _start_background_task(target_func, args):
    """작업 스레드를 시작하는 헬퍼 함수"""
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        _log_message("ERROR: A task is already running.", "ERROR")
        if _util_funcs: _util_funcs['_show_simple_modal_message']("Task Busy", "A previous task is still running. Please wait.")
        return
    _worker_thread = threading.Thread(target=target_func, args=args, daemon=True)
    _worker_thread.start()
    _update_progress(0.0, "Task started in background...")


def _run_automl_in_thread(df: pd.DataFrame, target: str, task_type: str, time_budget: int):
    """(최종 수정) '분석 제외' 컬럼 제거 및 '역할' 명시적 지정을 통해 AutoML 실행"""
    start_time = time.time()
    try:
        if TabularAutoML is None:
            raise ImportError("LightAutoML is not installed. Please run 'pip install lightautoml'.")

        _results_queue.put({"type": "progress", "value": 0.1, "log": "Preparing data for AutoML..."})

        step1_type_selections = _module_main_callbacks.get('get_column_analysis_types', lambda: {})()
        cols_to_exclude = [col for col, type_val in step1_type_selections.items() if type_val == "분석에서 제외 (Exclude)"]
        
        if target in cols_to_exclude:
            cols_to_exclude.remove(target)
            
        if cols_to_exclude:
            df = df.drop(columns=cols_to_exclude, errors='ignore')
            _results_queue.put({"type": "progress", "value": 0.15, "log": f"Excluded {len(cols_to_exclude)} columns from analysis."})

        # --- 아래 로직 추가 ---
        # LightAutoML 호환성을 위해 'category' 타입을 'object' 타입으로 변환
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col].dtype):
                df[col] = df[col].astype('object')
        # --- 여기까지 추가 ---

        original_columns = df.columns.tolist()
        new_columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in original_columns]
        df.columns = new_columns
        try:
            target_index = original_columns.index(target)
            target = new_columns[target_index]
        except ValueError:
            _results_queue.put({"type": "error", "log": f"Target column '{target}' not found."})
            return

        task = Task(task_type)
        categorical_features = []
        numeric_features = []
        
        for col in df.columns:
            if col == target:
                continue
            
            if df[col].dtype.name in ['object', 'category'] or df[col].nunique() < 25:
                categorical_features.append(col)
            else:
                numeric_features.append(col)

        roles = {
            'target': target,
            'category': categorical_features,
            'numeric': numeric_features,
        }
        
        _results_queue.put({"type": "progress", "value": 0.2, "log": f"Roles defined. Categorical: {len(categorical_features)}, Numeric: {len(numeric_features)}"})
        
        automl = TabularAutoML(task=task, timeout=time_budget, cpu_limit=1,
                               general_params={"use_algos": [["lgb"]]}) 

        _results_queue.put({"type": "progress", "value": 0.3, "log": "Fitting AutoML model..."})
        oof_predictions = automl.fit_predict(df, roles=roles, verbose=1)
        
        # (이하 코드 동일)
        training_time = time.time() - start_time
        _results_queue.put({"type": "progress", "value": 0.9, "log": "AutoML fitting complete. Generating report..."})
        
        feature_scores_df = automl.get_feature_scores()
        if not feature_scores_df.empty:
            feature_scores_df['Feature'] = feature_scores_df['Feature'].map(lambda x: original_columns[new_columns.index(x)] if x in new_columns else x)
            feature_scores = feature_scores_df.set_index('Feature')['Importance'].to_dict()
        else:
            feature_scores = {}

        report = {
            "model_name": f"AutoML_{original_columns[new_columns.index(target)]}",
            "algorithm": "LightAutoML",
            "model_type": "Classification" if task_type in ['binary', 'multiclass'] else "Regression",
            "target": original_columns[new_columns.index(target)],
            "training_time": training_time,
            "feature_scores": feature_scores,
            "model_object": automl,
        }
        _results_queue.put({"type": "automl_result", "data": report})

    except Exception as e:
        _results_queue.put({"type": "error", "log": f"AutoML Error: {e}"})
        traceback.print_exc()

# --- [1단계] SHAP 분석 로직 (백그라운드 실행) ---
def _run_shap_in_thread(experiment: ExperimentResult, df: pd.DataFrame):
    """(디버깅 버전) 데이터 처리 단계별로 로그를 추가하여 문제 원인 추적"""
    try:
        if SSWARM is None:
            raise ImportError("SHAP or LightAutoML's SSWARM is not installed.")

        _results_queue.put({"type": "progress", "value": 0.05, "log": "--- Starting SHAP Analysis (Debug Mode) ---"})
        
        model = experiment.model_object
        features = experiment.features
        
        # --- 디버깅 1: 최초 데이터 상태 ---
        try:
            X = df[features].copy()
            _results_queue.put({"type": "progress", "log": f"[DEBUG 1] Initial X shape: {X.shape}, Columns: {X.columns.tolist()}"})
        except Exception as e:
            _results_queue.put({"type": "error", "log": f"[DEBUG 1 FAILED] Could not select features. Error: {e}"})
            return

        # 컬럼 이름 정리
        original_feature_names = X.columns.tolist()
        new_columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in original_feature_names]
        X.columns = new_columns
        # --- 디버깅 2: 컬럼 이름 정리 후 ---
        _results_queue.put({"type": "progress", "log": f"[DEBUG 2] After name sanitization, X shape: {X.shape}"})

        # 타입 변환
        for col in X.columns:
            if pd.api.types.is_categorical_dtype(X[col].dtype):
                X[col] = X[col].astype('object')
        # --- 디버깅 3: 타입 변환 후 ---
        _results_queue.put({"type": "progress", "log": f"[DEBUG 3] After type conversion, X shape: {X.shape}"})

        # 라벨 인코딩
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = pd.Series(le.fit_transform(X[col].astype(str)), index=X.index)
        # --- 디버깅 4: 라벨 인코딩 후 ---
        _results_queue.put({"type": "progress", "log": f"[DEBUG 4] After LabelEncoding, X shape: {X.shape}, Dtypes: {X.dtypes.to_dict()}"})

        # --- 디버깅 5: 최종 데이터 확인 ---
        _results_queue.put({"type": "progress", "value": 0.3, "log": f"[DEBUG 5] Final data check before SHAP. Shape: {X.shape}"})

        # SSWARM 알고리즘을 위한 피처 개수 방어 코드 (여전히 유효)
        if X.shape[1] < 4:
            message = f"SHAP analysis skipped: The final feature count is less than 4 ({X.shape[1]}). Check debug logs for data loss."
            _results_queue.put({"type": "error", "log": message})
            return

        explainer = SSWARM(model)
        _results_queue.put({"type": "progress", "value": 0.6, "log": "Calculating SHAP values..."})
        shap_values = explainer.shap_values(X)
        _results_queue.put({"type": "progress", "value": 0.9, "log": "Generating SHAP summary plot..."})
        
        fig, ax = plt.subplots()
        shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values

        # --- 아래 로직 추가/수정 ---

        # 1. SSWARM이 실제로 사용한 피처 목록 가져오기 (sanitized된 이름)
        used_features_sanitized = explainer.used_feats
        
        # 2. 플롯에 사용할 데이터도 실제 사용된 피처만으로 구성
        X_for_plot = X[used_features_sanitized]

        # 3. 플롯의 축에 표시할 원본 피처 이름 목록 생성
        # sanitized 이름 -> 원본 이름 매핑 생성
        name_map = dict(zip(new_columns, original_feature_names))
        feature_names_for_plot = [name_map.get(sanitized_name, sanitized_name) for sanitized_name in used_features_sanitized]
        
        # 4. summary_plot에 정확히 일치하는 데이터와 이름을 전달
        shap.summary_plot(shap_values_to_plot, X_for_plot, feature_names=feature_names_for_plot, show=False)
        
        # --- 여기까지 추가/수정 ---
        plt.tight_layout()
        
        plot_func = _util_funcs.get('plot_to_dpg_texture')
        if plot_func:
            tex_tag, w, h, _ = plot_func(fig)
            plt.close(fig)
            _results_queue.put({"type": "shap_result", "tex_tag": tex_tag, "width": w, "height": h, "exp_id": experiment.id})

    except Exception as e:
        _results_queue.put({"type": "error", "log": f"SHAP Error: {e}"})
        traceback.print_exc()


# --- UI Creation ---
def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    
    main_callbacks['register_step_group_tag'](step_name, TAG_S11_GROUP)

    with dpg.group(tag=TAG_S11_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        
        with dpg.child_window(height=600, border=True, tag=TAG_S11_UPPER_VIZ_WINDOW):
            dpg.add_text("ML Modeling Dashboard", color=(255, 255, 0))
            dpg.add_separator()
            with dpg.tab_bar(tag=TAG_S11_VIZ_TAB_BAR):
                with dpg.tab(label="🧪 Experiments", tag=TAG_S11_EXPERIMENT_TAB):
                    _create_experiment_tracking_tab()
                # --- [1단계] AutoML 탭 UI 생성 ---
                with dpg.tab(label="🤖 AutoML", tag=TAG_S11_AUTOML_TAB):
                    _create_automl_tab()

        with dpg.child_window(border=True):
             dpg.add_text("Training Progress & Log", color=(100, 200, 255))
             dpg.add_progress_bar(tag=TAG_S11_PROGRESS_BAR, default_value=0.0, width=-1)
             with dpg.child_window(height=-1, border=True, tag=TAG_S11_LOG_WINDOW):
                dpg.add_text("Ready for training...", tag="s11_log_text")
    
    main_callbacks['register_module_updater'](step_name, update_ui)

def _create_automl_tab():
    """(수정) AutoML 탭의 UI를 생성합니다."""
    with dpg.group(horizontal=True):
        # AutoML 제어판
        with dpg.group(width=300, tag=TAG_S11_AUTOML_CONTROLS_GROUP):
            dpg.add_text("AutoML Settings", color=(255, 255, 0))
            dpg.add_separator()
            dpg.add_text("Data Source:")
            dpg.add_combo(label="", tag=TAG_S11_DF_SELECTOR, width=-1, callback=_on_df_selected_automl)
            dpg.add_text("Target Variable:")
            # Target 콤보박스에 콜백 추가
            dpg.add_combo(label="", tag="s11_automl_target_combo", width=-1, callback=_on_automl_target_changed)
            
            # --- 아래 Task Type 라디오 버튼 삭제 ---
            # dpg.add_text("Task Type:")
            # dpg.add_radio_button(items=["binary", "reg"], tag="s11_automl_task_type", horizontal=True, default_value="binary")
            
            # --- 대신 아래 텍스트 위젯 추가 ---
            with dpg.group(horizontal=True):
                dpg.add_text("Detected Task Type:")
                dpg.add_text("(select target)", tag="s11_automl_detected_task", color=(255, 255, 0))

            dpg.add_text("Time Budget (seconds):")
            dpg.add_input_int(default_value=60, width=-1, tag="s11_automl_time_budget")
            dpg.add_separator()
            dpg.add_button(label="🚀 Run AutoML", tag=TAG_S11_AUTOML_RUN_BUTTON, width=-1, height=40,
                           callback=_start_automl_run_callback)
        
        # AutoML 결과 표시 영역
        with dpg.child_window(border=True, tag=TAG_S11_AUTOML_RESULTS_GROUP):
            dpg.add_text("AutoML Results will be shown here.", tag="s11_automl_results_placeholder")

def _create_experiment_tracking_tab():
    """기존 실험 추적 탭 (UI 동일)"""
    with dpg.table(tag=TAG_S11_EXPERIMENT_TABLE, header_row=True, resizable=True, scrollY=True,
                   borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
        dpg.add_table_column(label="Timestamp")
        dpg.add_table_column(label="Model")
        dpg.add_table_column(label="Algorithm")
        dpg.add_table_column(label="Target")
        dpg.add_table_column(label="Actions")


# --- Callbacks ---
def _start_automl_run_callback():
    """(수정) Run AutoML 버튼 콜백"""
    df_name = dpg.get_value(TAG_S11_DF_SELECTOR)
    target = dpg.get_value("s11_automl_target_combo")
    # 라디오 버튼 대신 내부 변수에서 task_type을 가져옴
    task_type = _detected_task_type 
    time_budget = dpg.get_value("s11_automl_time_budget")
    
    if not df_name or not target:
        if _util_funcs: _util_funcs['_show_simple_modal_message']("Error", "Please select DataFrame and Target.")
        return
    
    # task_type이 유효한지 검사
    if not task_type:
        if _util_funcs: _util_funcs['_show_simple_modal_message']("Error", "Could not detect a valid task type.")
        return

    all_dfs = _module_main_callbacks['get_all_available_dfs']()
    df = all_dfs.get(df_name)

    if df is None:
        if _util_funcs: _util_funcs['_show_simple_modal_message']("Error", "DataFrame not found.")
        return
    
    _start_background_task(_run_automl_in_thread, args=(df.copy(), target, task_type, time_budget))

def _start_shap_analysis_callback(sender, app_data, user_data: ExperimentResult):
    """(수정) SHAP 분석 시작 버튼 콜백"""
    exp = user_data
    
    # 실행 시점에 main_app에서 직접 모든 DF 목록을 가져옴
    all_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    df = all_dfs.get(exp.dataframe_name)
    
    if df is None:
        _log_message(f"Error: DataFrame '{exp.dataframe_name}' for SHAP analysis not found.", "ERROR")
        return
    
    _start_background_task(_run_shap_in_thread, args=(exp, df.copy()))

def _on_df_selected_automl(sender, df_name, user_data):
    """AutoML 탭에서 데이터프레임 선택 시 콜백"""
    _update_automl_target_combo(df_name)

# --- [1단계] 비동기 결과 처리 및 UI 업데이트 ---
def _check_for_updates():
    """매 프레임마다 큐를 확인하여 UI를 업데이트하는 함수"""
    global _worker_thread
    try:
        # 큐에서 논블로킹으로 메시지 가져오기
        result = _results_queue.get_nowait()
        
        if result["type"] == "progress":
            _update_progress(result["value"], result["log"])
        
        elif result["type"] == "error":
            _log_message(result["log"], "ERROR")
            _update_progress(0.0) # 에러 시 프로그레스 바 리셋
        
        elif result["type"] == "automl_result":
            _display_automl_results(result["data"])
            _update_progress(1.0, "AutoML process finished.")
            time.sleep(1)
            _update_progress(0.0)

        elif result["type"] == "shap_result":
            _display_shap_results(result)
            _update_progress(1.0, "SHAP analysis finished.")
            time.sleep(1)
            _update_progress(0.0)

        _results_queue.task_done()

    except queue.Empty:
        # 큐가 비어있으면 아무것도 하지 않음
        pass
    except Exception as e:
        print(f"Error checking for updates: {e}")

    # 스레드 종료 확인
    if _worker_thread and not _worker_thread.is_alive():
        _worker_thread = None

# --- 결과 표시 함수 ---
def _display_automl_results(data: dict):
    """AutoML 결과를 UI에 표시"""
    _log_message("Displaying AutoML results...")
    
    # 결과 저장
    exp = ExperimentResult(
        id=str(uuid.uuid4()), timestamp=datetime.datetime.now(),
        model_name=data["model_name"], model_type=data["model_type"],
        algorithm=data["algorithm"], parameters={"time_budget": dpg.get_value("s11_automl_time_budget")},
        features=list(data["feature_scores"].keys()), target=data["target"],
        metrics={}, training_time=data["training_time"],
        model_object=data["model_object"], dataframe_name=dpg.get_value(TAG_S11_DF_SELECTOR)
    )
    _experiments_history.append(exp)
    _update_experiment_table()
    
    # AutoML 결과 탭 비우기
    res_group = TAG_S11_AUTOML_RESULTS_GROUP
    if dpg.does_item_exist(res_group):
        dpg.delete_item(res_group, children_only=True)
    
    # 결과 표시
    dpg.add_text(f"Results for: {exp.model_name}", parent=res_group, color=(255, 255, 0))
    dpg.add_text(f"Training Time: {exp.training_time:.2f} seconds", parent=res_group)
    dpg.add_separator(parent=res_group)
    dpg.add_text("Feature Importances:", parent=res_group, color=(100, 200, 255))
    
    # 피처 중요도 테이블
    with dpg.table(header_row=True, parent=res_group):
        dpg.add_table_column(label="Feature")
        dpg.add_table_column(label="Importance")
        for feature, score in sorted(data["feature_scores"].items(), key=lambda x: x[1], reverse=True):
            with dpg.table_row():
                dpg.add_text(feature)
                dpg.add_text(f"{score:.4f}")

def _display_shap_results(data: dict):
    """SHAP 분석 결과를 UI에 표시"""
    exp_id = data['exp_id']
    result_tab_tag = f"s11_result_tab_{exp_id}"
    
    # 해당 실험의 결과 탭이 열려있는지 확인
    if dpg.does_item_exist(result_tab_tag):
        # SHAP 결과 이미지를 해당 탭에 추가
        dpg.add_separator(parent=result_tab_tag)
        dpg.add_text("SHAP Summary Plot", parent=result_tab_tag, color=(100, 200, 255))
        dpg.add_image(data['tex_tag'], width=data['width'], height=data['height'], parent=result_tab_tag)
        _texture_tags.append(data['tex_tag']) # 나중에 삭제하기 위해 태그 저장
    else:
        _log_message(f"Info: Result tab for experiment {exp_id[:8]} is not open. SHAP plot not displayed.", "INFO")


def _create_results_visualizations(experiment: ExperimentResult):
    """수동 학습 결과 탭 생성 (SHAP 버튼 추가)"""
    tab_name = f"📈 {experiment.model_name}"
    tab_tag = f"s11_result_tab_{experiment.id}"
    
    if dpg.does_item_exist(tab_tag):
        dpg.focus_item(tab_tag)
        return

    with dpg.tab(label=tab_name, parent=TAG_S11_VIZ_TAB_BAR, closable=True, tag=tab_tag):
        dpg.add_text(f"Results for: {experiment.model_name}", color=(255, 255, 0))
        dpg.add_text(f"Algorithm: {experiment.algorithm}")
        dpg.add_separator()
        
        # --- [1단계] SHAP 분석 버튼 추가 ---
        dpg.add_button(label="🔬 Run SHAP Analysis", user_data=experiment, 
                       callback=_start_shap_analysis_callback)
        dpg.add_separator()
        
        # (기존의 Confusion Matrix, ROC Curve 등 시각화 로직은 여기에 위치)

# --- Helper Functions (기존 함수들 일부 수정 및 추가) ---
def _log_message(message: str, level: str = "INFO"):
    if not dpg.is_dearpygui_running(): return
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {level}: {message}"
    if dpg.does_item_exist("s11_log_text"):
        current_log = dpg.get_value("s11_log_text")
        new_log = f"{current_log}\n{formatted_msg}"
        log_lines = new_log.split("\n")
        if len(log_lines) > 100: new_log = "\n".join(log_lines[-100:])
        dpg.set_value("s11_log_text", new_log)
        if dpg.does_item_exist(TAG_S11_LOG_WINDOW):
            dpg.set_y_scroll(TAG_S11_LOG_WINDOW, dpg.get_y_scroll_max(TAG_S11_LOG_WINDOW))

def _update_progress(value: float, message: str = ""):
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_S11_PROGRESS_BAR):
        dpg.set_value(TAG_S11_PROGRESS_BAR, value)
    if message: _log_message(message)

def update_ui():
    """(수정) UI 업데이트"""
    if not _module_main_callbacks: return
    
    # 함수 내에서만 사용할 지역 변수로 DF 목록을 가져옴
    all_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    
    df_selector_tag = TAG_S11_DF_SELECTOR
    if dpg.does_item_exist(df_selector_tag):
        df_names = list(all_dfs.keys())
        dpg.configure_item(df_selector_tag, items=df_names)
        
        current_selection = dpg.get_value(df_selector_tag)
        if not current_selection and df_names:
            current_selection = df_names[0]
            dpg.set_value(df_selector_tag, current_selection)

        if current_selection:
            _update_automl_target_combo(current_selection)

    _update_experiment_table()

def _update_experiment_table():
    """실험 목록 테이블 업데이트"""
    if not dpg.does_item_exist(TAG_S11_EXPERIMENT_TABLE): return
    
    dpg.delete_item(TAG_S11_EXPERIMENT_TABLE, children_only=True)
    # 테이블 헤더 다시 추가
    dpg.add_table_column(label="Timestamp", parent=TAG_S11_EXPERIMENT_TABLE)
    dpg.add_table_column(label="Model", parent=TAG_S11_EXPERIMENT_TABLE)
    dpg.add_table_column(label="Algorithm", parent=TAG_S11_EXPERIMENT_TABLE)
    dpg.add_table_column(label="Target", parent=TAG_S11_EXPERIMENT_TABLE)
    dpg.add_table_column(label="Actions", parent=TAG_S11_EXPERIMENT_TABLE)

    for exp in reversed(_experiments_history):
        with dpg.table_row(parent=TAG_S11_EXPERIMENT_TABLE):
            dpg.add_text(exp.timestamp.strftime("%H:%M:%S"))
            dpg.add_text(exp.model_name)
            dpg.add_text(exp.algorithm)
            dpg.add_text(exp.target)
            with dpg.group(horizontal=True):
                dpg.add_button(label="View", user_data=exp, callback=_view_experiment_details)

def _view_experiment_details(sender, app_data, user_data: ExperimentResult):
    """실험 결과 보기 콜백 (SHAP 버튼이 있는 탭을 생성)"""
    _create_results_visualizations(user_data)


def reset_state():
    """상태 초기화"""
    global _experiments_history, _texture_tags, _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        # 여기서 스레드를 강제 종료하는 것은 위험하므로, 사용자에게 알리는 것이 좋음
        _log_message("Warning: A task is still running. Cannot reset state now.", "WARN")
        return

    _experiments_history.clear()
    for tag in _texture_tags:
        if dpg.does_item_exist(tag): dpg.delete_item(tag)
    _texture_tags.clear()
    
    if dpg.is_dearpygui_running():
        update_ui()
    _log_message("State has been reset.", "INFO")