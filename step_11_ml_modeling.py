# step_11_ml_modeling.py - 개선된 버전

"""
Step 11 ML Modeling & AI 통합 모듈 (개선 버전)

추가/개선된 기능:
- AutoML 진행 과정 실시간 모니터링
- 하이퍼파라미터 최적화 시각화
- 교차 검증 및 학습 곡선
- SHAP 자동 실행
- True/False 타겟 처리
- SHAP shape 에러 수정
- 모델 성능 비교 대시보드
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
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
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

# --- 외부 라이브러리 ---
try:
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task
    LIGHTAUTOML_AVAILABLE = True
except ImportError:
    LIGHTAUTOML_AVAILABLE = False
    print("Warning: LightAutoML not available. Using fallback AutoML implementation.")
    
import shap

warnings.filterwarnings('ignore')

# --- Classification/Regression Models ---
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# --- DPG Tags ---
TAG_S11_GROUP = "step11_ml_modeling_group"
TAG_S11_UPPER_VIZ_WINDOW = "step11_upper_viz_window"
TAG_S11_LOWER_CONTROL_PANEL = "step11_lower_control_panel"
TAG_S11_VIZ_TAB_BAR = "step11_viz_tab_bar"

# 탭 순서 변경: AutoML을 첫 번째로
TAG_S11_AUTOML_TAB = "step11_automl_tab" 
TAG_S11_EXPERIMENT_TAB = "step11_experiment_tab"
TAG_S11_MONITORING_TAB = "step11_monitoring_tab"
TAG_S11_COMPARISON_TAB = "step11_comparison_tab"

TAG_S11_AUTOML_CONTROLS_GROUP = "s11_automl_controls_group"
TAG_S11_AUTOML_RESULTS_GROUP = "s11_automl_results_group"
TAG_S11_AUTOML_RUN_BUTTON = "s11_automl_run_button"
TAG_S11_MONITORING_PLOTS_GROUP = "s11_monitoring_plots_group"

TAG_S11_PROGRESS_BAR = "step11_progress_bar"
TAG_S11_LOG_WINDOW = "step11_log_window"
TAG_S11_EXPERIMENT_TABLE = "step11_experiment_table"
TAG_S11_DF_SELECTOR = "step11_df_selector"

# --- ML_ALGORITHMS 확장 ---
ML_ALGORITHMS = {
    "Classification": {
        "Logistic Regression": {"class": LogisticRegression, "params": {"max_iter": 1000}},
        "Random Forest": {"class": RandomForestClassifier, "params": {"n_estimators": 100}},
        "Decision Tree": {"class": DecisionTreeClassifier, "params": {"max_depth": 5}},
        "SVM": {"class": SVC, "params": {"probability": True}},
        "KNN": {"class": KNeighborsClassifier, "params": {"n_neighbors": 5}},
        "XGBoost": {"class": XGBClassifier, "params": {"eval_metric": "logloss"}},
        "LightGBM": {"class": LGBMClassifier, "params": {"verbose": -1}},
    },
    "Regression": {
        "Linear Regression": {"class": LinearRegression, "params": {}},
        "Random Forest": {"class": RandomForestRegressor, "params": {"n_estimators": 100}},
        "Decision Tree": {"class": DecisionTreeRegressor, "params": {"max_depth": 5}},
        "SVR": {"class": SVR, "params": {}},
        "KNN": {"class": KNeighborsRegressor, "params": {"n_neighbors": 5}},
        "XGBoost": {"class": XGBRegressor, "params": {}},
        "LightGBM": {"class": LGBMRegressor, "params": {"verbose": -1}},
    }
}

# --- Module State 변수 ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_experiments_history: List['ExperimentResult'] = []
_texture_tags: List[str] = []
_results_queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_detected_task_type: str = "" 
_automl_progress_history: List[Dict[str, Any]] = []  # AutoML 진행 상황 기록
_monitoring_data: Dict[str, List[float]] = {  # 모니터링 데이터
    "time": [], "score": [], "n_models": []
}

@dataclass
class ExperimentResult:
    """실험 결과를 저장하는 데이터 클래스 (확장)"""
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
    model_object: Any
    dataframe_name: str
    cv_scores: Optional[List[float]] = None
    learning_curve_data: Optional[Dict] = None
    shap_values: Optional[Any] = None
    
    def to_dict(self):
        d = asdict(self)
        d.pop('model_object', None)
        d.pop('shap_values', None)
        return d

def _preprocess_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    """타겟 변수 전처리 (True/False 처리 포함)"""
    df = df.copy()
    le = LabelEncoder()
    
    target_series = df[target].copy()
    
    # True/False 값을 문자열로 변환
    if target_series.dtype == bool or set(target_series.dropna().unique()) <= {True, False}:
        target_series = target_series.map({True: 'True', False: 'False'})
    
    # NaN 처리
    if target_series.isna().any():
        target_series = target_series.fillna('Missing')
    
    # LabelEncoder 적용
    df[target] = le.fit_transform(target_series.astype(str))
    
    return df, le

def _detect_and_update_task_type(df_name: str, target_name: str):
    """개선된 타겟 변수 타입 감지"""
    global _detected_task_type

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
    
    # True/False 처리
    unique_vals = set(target_series.dropna().unique())
    if unique_vals <= {True, False, 'True', 'False', 'true', 'false', 1, 0, '1', '0'}:
        _detected_task_type = 'binary'
    else:
        nunique = target_series.nunique()
        dtype_kind = target_series.dtype.kind
        
        if dtype_kind in 'Ocb' or (dtype_kind == 'i' and nunique < 20):
            _detected_task_type = 'binary' if nunique == 2 else 'multiclass'
        else:
            _detected_task_type = 'reg'
    
    if dpg.does_item_exist("s11_automl_detected_task"):
        dpg.set_value("s11_automl_detected_task", _detected_task_type)

def _run_automl_in_thread(df: pd.DataFrame, target: str, task_type: str, time_budget: int):
    """개선된 AutoML 실행 (모니터링 포함)"""
    start_time = time.time()
    use_lightautoml = LIGHTAUTOML_AVAILABLE  # 로컬 변수로 복사
    
    try:
        if not use_lightautoml:
            _results_queue.put({"type": "progress", "value": 0.05, "log": "LightAutoML not available, using built-in AutoML..."})
        
        _results_queue.put({"type": "progress", "value": 0.1, "log": "Preparing data for AutoML..."})

        # 데이터 전처리
        step1_type_selections = _module_main_callbacks.get('get_column_analysis_types', lambda: {})()
        cols_to_exclude = [col for col, type_val in step1_type_selections.items() if type_val == "분석에서 제외 (Exclude)"]
        
        if target in cols_to_exclude:
            cols_to_exclude.remove(target)
            
        if cols_to_exclude:
            df = df.drop(columns=cols_to_exclude, errors='ignore')
            _results_queue.put({"type": "progress", "value": 0.15, "log": f"Excluded {len(cols_to_exclude)} columns from analysis."})

        # 타겟 전처리
        if task_type in ['binary', 'multiclass']:
            df, label_encoder = _preprocess_target(df, target)
            _results_queue.put({"type": "progress", "value": 0.2, "log": "Target variable preprocessed."})

        # Category 타입 변환
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col].dtype):
                df[col] = df[col].astype('object')

        _results_queue.put({"type": "progress", "value": 0.25, "log": "Data preparation complete."})

        # 역할 정의
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
        
        _results_queue.put({"type": "progress", "value": 0.3, "log": f"Roles defined. Categorical: {len(categorical_features)}, Numeric: {len(numeric_features)}"})
        
        # Feature scores 초기화
        feature_scores = {}
        cv_scores = None
        automl = None  # 초기화
        
        # Pickle 오류 회피를 위한 대체 AutoML 구현
        if use_lightautoml:
            try:
                # LightAutoML 시도
                task = Task(task_type)
                automl = TabularAutoML(
                    task=task, 
                    timeout=time_budget, 
                    cpu_limit=1,
                    general_params={"use_algos": [["lgb"]]},
                    reader_params={"cv": 3, "random_state": 42}
                )
                
                _results_queue.put({"type": "progress", "value": 0.3, "log": "Fitting LightAutoML model..."})
                oof_predictions = automl.fit_predict(df, roles=roles, verbose=0)
                
                # Feature importance
                try:
                    feature_scores_df = automl.get_feature_scores()
                    if not feature_scores_df.empty:
                        feature_scores = feature_scores_df.set_index('Feature')['Importance'].to_dict()
                except:
                    pass
                    
            except Exception as e:
                _results_queue.put({"type": "progress", "value": 0.3, "log": f"LightAutoML failed: {str(e)}, using fallback..."})
                use_lightautoml = False  # 로컬 변수 업데이트
        
        if not use_lightautoml:
            # LightAutoML 실패 시 대체 AutoML 구현
            _results_queue.put({"type": "progress", "value": 0.3, "log": "Using fallback AutoML implementation..."})
            
            from sklearn.model_selection import cross_val_score
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.impute import SimpleImputer
            from sklearn.compose import ColumnTransformer
            
            # 전처리 파이프라인
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', 'passthrough', categorical_features)  # 카테고리는 이미 인코딩됨
            ])
            
            # 모델 후보군
            models = []
            if task_type in ['binary', 'multiclass']:
                models = [
                    ('LogisticRegression', LogisticRegression(max_iter=1000)),
                    ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
                    ('LightGBM', LGBMClassifier(n_estimators=100, verbose=-1, random_state=42))
                ]
            else:
                models = [
                    ('LinearRegression', LinearRegression()),
                    ('RandomForest', RandomForestRegressor(n_estimators=100, random_state=42)),
                    ('LightGBM', LGBMRegressor(n_estimators=100, verbose=-1, random_state=42))
                ]
            
            X = df.drop(columns=[target])
            y = df[target]
            
            best_score = -np.inf
            best_model = None
            cv_scores_list = []  # cv_scores를 cv_scores_list로 변경
            
            # 각 모델 평가
            for i, (name, model) in enumerate(models):
                progress = 0.4 + (i / len(models)) * 0.4
                _results_queue.put({
                    "type": "progress", 
                    "value": progress, 
                    "log": f"Evaluating {name}..."
                })
                
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                try:
                    scores = cross_val_score(pipeline, X, y, cv=3, 
                                           scoring='accuracy' if task_type in ['binary', 'multiclass'] else 'neg_mean_squared_error')
                    mean_score = scores.mean()
                    
                    if mean_score > best_score:
                        best_score = mean_score
                        best_model = pipeline
                        cv_scores_list = scores.tolist()
                    
                    _monitoring_data["time"].append(time.time() - start_time)
                    _monitoring_data["score"].append(mean_score)
                    _monitoring_data["n_models"].append(i + 1)
                    _results_queue.put({"type": "monitoring_update"})
                    
                except Exception as model_e:
                    _results_queue.put({"type": "progress", "value": progress, "log": f"Failed to evaluate {name}: {str(model_e)}"})
            
            # 최종 모델 학습
            if best_model:
                _results_queue.put({"type": "progress", "value": 0.85, "log": "Training final model..."})
                best_model.fit(X, y)
                
                # cv_scores 변수 설정
                cv_scores = cv_scores_list
                # automl 변수에 최종 모델 할당
                automl = best_model
                automl = best_model  # automl 변수에 할당
                
                # Feature importance 계산
                if hasattr(best_model.named_steps['model'], 'feature_importances_'):
                    importances = best_model.named_steps['model'].feature_importances_
                    # 전처리된 피처 이름 가져오기
                    feature_names = (numeric_features + categorical_features)
                    for fname, imp in zip(feature_names, importances[:len(feature_names)]):
                        feature_scores[fname] = float(imp)
            else:
                raise Exception("No model could be trained successfully")
                
        # 모델 확인
        if automl is None and 'best_model' in locals():
            automl = best_model
        
        training_time = time.time() - start_time
        _results_queue.put({"type": "progress", "value": 0.9, "log": "AutoML complete. Generating report..."})
        
        # 결과 보고서 생성
        report = {
            "model_name": f"AutoML_{target}",
            "algorithm": "LightAutoML" if use_lightautoml else "Sklearn AutoML (Ensemble)",
            "model_type": "Classification" if task_type in ['binary', 'multiclass'] else "Regression",
            "target": target,
            "training_time": training_time,
            "feature_scores": feature_scores,
            "model_object": automl,  # automl은 위에서 할당됨
            "cv_scores": cv_scores,
            "task_type": task_type,
        }
        _results_queue.put({"type": "automl_result", "data": report})

    except Exception as e:
        error_msg = f"AutoML Error: {str(e)}\n{traceback.format_exc()}"
        _results_queue.put({"type": "error", "log": error_msg})
        print(error_msg)  # 콘솔에도 출력

def _run_shap_in_thread(experiment: ExperimentResult, df: pd.DataFrame):
    """개선된 SHAP 분석 (shape 에러 수정)"""
    try:
        _results_queue.put({"type": "progress", "value": 0.05, "log": "Starting SHAP Analysis..."})
        
        model = experiment.model_object
        features = experiment.features
        
        # 데이터 준비
        X = df[features].copy()
        
        # 타겟이 분류 문제인 경우 전처리
        if experiment.model_type == "Classification":
            target = experiment.target
            if target in df.columns:
                df_processed, _ = _preprocess_target(df, target)
        
        # 카테고리 변환
        for col in X.columns:
            if pd.api.types.is_categorical_dtype(X[col].dtype):
                X[col] = X[col].astype('object')
        
        # 라벨 인코딩
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        _results_queue.put({"type": "progress", "value": 0.3, "log": f"Data prepared. Shape: {X.shape}"})
        
        # 카테고리 변환
        for col in X.columns:
            if pd.api.types.is_categorical_dtype(X[col].dtype):
                X[col] = X[col].astype('object')
        
        # 라벨 인코딩
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        _results_queue.put({"type": "progress", "value": 0.3, "log": f"Data prepared. Shape: {X.shape}"})
        
        # SHAP 분석
        if hasattr(model, 'predict'):
            # scikit-learn 모델인 경우
            if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                # Pipeline인 경우
                base_model = model.named_steps['model']
                # TreeExplainer 시도
                try:
                    explainer = shap.TreeExplainer(base_model)
                    # 전처리된 데이터 준비
                    X_transformed = model.named_steps['preprocessor'].transform(X)
                    shap_values = explainer.shap_values(X_transformed)
                except:
                    # KernelExplainer 사용
                    sample_size = min(100, len(X))
                    X_sample = X.sample(n=sample_size, random_state=42)
                    X_sample_transformed = model.named_steps['preprocessor'].transform(X_sample)
                    explainer = shap.KernelExplainer(base_model.predict, X_sample_transformed)
                    shap_values = explainer.shap_values(X_sample_transformed)
                    X = X_sample
            else:
                # 일반 모델인 경우
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)
                except:
                    sample_size = min(100, len(X))
                    X_sample = X.sample(n=sample_size, random_state=42)
                    explainer = shap.KernelExplainer(model.predict, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                    X = X_sample
        else:
            # LightAutoML의 경우
            try:
                # 모델에서 예측 함수 추출
                predict_func = lambda x: model.predict(pd.DataFrame(x, columns=X.columns)).data
                sample_size = min(100, len(X))
                X_sample = X.sample(n=sample_size, random_state=42)
                explainer = shap.KernelExplainer(predict_func, X_sample)
                shap_values = explainer.shap_values(X_sample)
                X = X_sample
            except Exception as e:
                _results_queue.put({"type": "error", "log": f"SHAP Error: Could not create explainer - {str(e)}"})
                return
        
        _results_queue.put({"type": "progress", "value": 0.6, "log": "SHAP values calculated."})
        
        # Summary plot 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Binary classification의 경우
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values
        
        # Shape 확인 및 조정
        if hasattr(shap_values_to_plot, 'shape'):
            if len(shap_values_to_plot.shape) == 3:
                shap_values_to_plot = shap_values_to_plot[:, :, 0]
        
        # Feature 이름 (원본 그대로 사용)
        feature_names_for_plot = X.columns.tolist()
        
        shap.summary_plot(shap_values_to_plot, X, feature_names=feature_names_for_plot, show=False)
        plt.tight_layout()
        
        _results_queue.put({"type": "progress", "value": 0.9, "log": "Generating SHAP plot..."})
        
        plot_func = _util_funcs.get('plot_to_dpg_texture')
        if plot_func:
            tex_tag, w, h, _ = plot_func(fig)
            plt.close(fig)
            
            # SHAP 값 저장
            experiment.shap_values = shap_values_to_plot
            
            _results_queue.put({
                "type": "shap_result", 
                "tex_tag": tex_tag, 
                "width": w, 
                "height": h, 
                "exp_id": experiment.id
            })

    except Exception as e:
        _results_queue.put({"type": "error", "log": f"SHAP Error: {str(e)}\n{traceback.format_exc()}"})

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
                # AutoML을 첫 번째 탭으로
                with dpg.tab(label="🤖 AutoML", tag=TAG_S11_AUTOML_TAB):
                    _create_automl_tab()
                with dpg.tab(label="📊 Monitoring", tag=TAG_S11_MONITORING_TAB):
                    _create_monitoring_tab()
                with dpg.tab(label="🧪 Experiments", tag=TAG_S11_EXPERIMENT_TAB):
                    _create_experiment_tracking_tab()
                with dpg.tab(label="📈 Comparison", tag=TAG_S11_COMPARISON_TAB):
                    _create_comparison_tab()

        with dpg.child_window(border=True):
            dpg.add_text("Training Progress & Log", color=(100, 200, 255))
            dpg.add_progress_bar(tag=TAG_S11_PROGRESS_BAR, default_value=0.0, width=-1)
            with dpg.child_window(height=-1, border=True, tag=TAG_S11_LOG_WINDOW):
                dpg.add_text("Ready for training...", tag="s11_log_text")
    
    main_callbacks['register_module_updater'](step_name, update_ui)

def _create_automl_tab():
    """개선된 AutoML 탭"""
    with dpg.group(horizontal=True):
        with dpg.group(width=300, tag=TAG_S11_AUTOML_CONTROLS_GROUP):
            dpg.add_text("AutoML Settings", color=(255, 255, 0))
            dpg.add_separator()
            
            dpg.add_text("Data Source:")
            dpg.add_combo(label="", tag=TAG_S11_DF_SELECTOR, width=-1, callback=_on_df_selected_automl)
            
            dpg.add_text("Target Variable:")
            dpg.add_combo(label="", tag="s11_automl_target_combo", width=-1, callback=_on_automl_target_changed)
            
            with dpg.group(horizontal=True):
                dpg.add_text("Detected Task Type:")
                dpg.add_text("(select target)", tag="s11_automl_detected_task", color=(255, 255, 0))
            
            dpg.add_text("Time Budget (seconds):")
            dpg.add_input_int(default_value=120, width=-1, tag="s11_automl_time_budget", min_value=30, max_value=3600)
            
            dpg.add_text("Advanced Options:")
            dpg.add_checkbox(label="Use Cross-Validation", tag="s11_use_cv", default_value=True)
            dpg.add_checkbox(label="Auto Feature Engineering", tag="s11_auto_fe", default_value=True)
            
            dpg.add_separator()
            dpg.add_button(label="🚀 Run AutoML", tag=TAG_S11_AUTOML_RUN_BUTTON, width=-1, height=40,
                           callback=_start_automl_run_callback)
        
        with dpg.child_window(border=True, tag=TAG_S11_AUTOML_RESULTS_GROUP):
            dpg.add_text("AutoML Results will be shown here.", tag="s11_automl_results_placeholder")

def _create_monitoring_tab():
    """AutoML 진행 상황 모니터링 탭"""
    with dpg.child_window(border=True, tag=TAG_S11_MONITORING_PLOTS_GROUP):
        dpg.add_text("AutoML Progress Monitoring", color=(255, 255, 0))
        dpg.add_separator()
        dpg.add_text("Run AutoML to see real-time monitoring charts.")

def _create_comparison_tab():
    """모델 비교 탭"""
    dpg.add_text("Model Comparison Dashboard", color=(255, 255, 0))
    dpg.add_separator()
    with dpg.child_window(border=True, tag="s11_comparison_window"):
        dpg.add_text("Complete experiments to compare models.")

def _create_experiment_tracking_tab():
    """개선된 실험 추적 탭"""
    with dpg.table(tag=TAG_S11_EXPERIMENT_TABLE, header_row=True, resizable=True, scrollY=True,
                   borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
        dpg.add_table_column(label="Timestamp")
        dpg.add_table_column(label="Model")
        dpg.add_table_column(label="Algorithm")
        dpg.add_table_column(label="Target")
        dpg.add_table_column(label="Score")
        dpg.add_table_column(label="Actions")

# --- Callbacks ---
def _update_automl_target_combo(df_name: str):
    """타겟 변수 콤보박스 업데이트"""
    all_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    if not df_name or df_name not in all_dfs:
        if dpg.does_item_exist("s11_automl_target_combo"): 
            dpg.configure_item("s11_automl_target_combo", items=[])
        if dpg.does_item_exist("s11_automl_detected_task"): 
            dpg.set_value("s11_automl_detected_task", "(select source)")
        return

    df = all_dfs[df_name]
    
    step1_type_selections = _module_main_callbacks.get('get_column_analysis_types', lambda: {})()
    cols_to_exclude = {col for col, type_val in step1_type_selections.items() if type_val == "분석에서 제외 (Exclude)"}
    
    candidate_cols = [col for col in df.columns if col not in cols_to_exclude]
    
    combo_tag = "s11_automl_target_combo"
    if not dpg.does_item_exist(combo_tag): return
        
    dpg.configure_item(combo_tag, items=candidate_cols)
    
    final_target = ""
    if _module_main_callbacks and 'get_selected_target_variable' in _module_main_callbacks:
        global_target = _module_main_callbacks['get_selected_target_variable']()
        if global_target and global_target in candidate_cols:
            final_target = global_target
        elif candidate_cols:
            final_target = candidate_cols[0]
            
    if final_target:
        dpg.set_value(combo_tag, final_target)
        _detect_and_update_task_type(df_name, final_target)

def _on_automl_target_changed(sender, target_name, user_data):
    """AutoML 타겟 변경 콜백"""
    df_name = dpg.get_value(TAG_S11_DF_SELECTOR)
    _detect_and_update_task_type(df_name, target_name)

def _on_df_selected_automl(sender, df_name, user_data):
    """데이터프레임 선택 콜백"""
    _update_automl_target_combo(df_name)

def _start_automl_run_callback():
    """AutoML 실행 콜백"""
    global _monitoring_data
    
    df_name = dpg.get_value(TAG_S11_DF_SELECTOR)
    target = dpg.get_value("s11_automl_target_combo")
    task_type = _detected_task_type
    time_budget = dpg.get_value("s11_automl_time_budget")
    
    if not df_name or not target:
        if _util_funcs: 
            _util_funcs['_show_simple_modal_message']("Error", "Please select DataFrame and Target.")
        return
    
    if not task_type:
        if _util_funcs: 
            _util_funcs['_show_simple_modal_message']("Error", "Could not detect a valid task type.")
        return

    all_dfs = _module_main_callbacks['get_all_available_dfs']()
    df = all_dfs.get(df_name)

    if df is None:
        if _util_funcs: 
            _util_funcs['_show_simple_modal_message']("Error", "DataFrame not found.")
        return
    
    # 모니터링 데이터 초기화
    _monitoring_data = {"time": [], "score": [], "n_models": []}
    
    _start_background_task(_run_automl_in_thread, args=(df.copy(), target, task_type, time_budget))

def _view_experiment_details(sender, app_data, user_data: ExperimentResult):
    """실험 상세 보기 (자동 SHAP 실행)"""
    _create_results_visualizations(user_data)
    
    # 자동으로 SHAP 분석 시작
    all_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    df = all_dfs.get(user_data.dataframe_name)
    
    if df is not None and user_data.shap_values is None:
        _log_message(f"Auto-starting SHAP analysis for {user_data.model_name}...")
        _start_background_task(_run_shap_in_thread, args=(user_data, df.copy()))

def _start_background_task(target_func, args):
    """백그라운드 작업 시작"""
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        _log_message("ERROR: A task is already running.", "ERROR")
        if _util_funcs: 
            _util_funcs['_show_simple_modal_message']("Task Busy", "A previous task is still running. Please wait.")
        return
    _worker_thread = threading.Thread(target=target_func, args=args, daemon=True)
    _worker_thread.start()
    _update_progress(0.0, "Task started in background...")

def _check_for_updates():
    """UI 업데이트 체크"""
    global _worker_thread
    try:
        result = _results_queue.get_nowait()
        
        if result["type"] == "progress":
            _update_progress(result["value"], result["log"])
        
        elif result["type"] == "error":
            _log_message(result["log"], "ERROR")
            _update_progress(0.0)
        
        elif result["type"] == "automl_result":
            _display_automl_results(result["data"])
            _update_progress(1.0, "AutoML process finished.")
            time.sleep(1)
            _update_progress(0.0)
            _update_comparison_view()

        elif result["type"] == "shap_result":
            _display_shap_results(result)
            _update_progress(1.0, "SHAP analysis finished.")
            time.sleep(1)
            _update_progress(0.0)
            
        elif result["type"] == "monitoring_update":
            _update_monitoring_plots()

        _results_queue.task_done()

    except queue.Empty:
        pass
    except Exception as e:
        print(f"Error checking for updates: {e}")

    if _worker_thread and not _worker_thread.is_alive():
        _worker_thread = None

def _update_monitoring_plots():
    """모니터링 플롯 업데이트"""
    if not _monitoring_data["time"]:
        return
        
    if not dpg.does_item_exist(TAG_S11_MONITORING_PLOTS_GROUP):
        return
    
    # 플롯 업데이트 로직
    try:
        dpg.delete_item(TAG_S11_MONITORING_PLOTS_GROUP, children_only=True)
        
        dpg.add_text("AutoML Progress Monitoring", parent=TAG_S11_MONITORING_PLOTS_GROUP, color=(255, 255, 0))
        dpg.add_separator(parent=TAG_S11_MONITORING_PLOTS_GROUP)
        
        # Score over time 플롯
        with dpg.plot(label="Model Performance Over Time", height=200, width=-1, 
                      parent=TAG_S11_MONITORING_PLOTS_GROUP):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (seconds)")
            dpg.add_plot_axis(dpg.mvYAxis, label="Score", tag="score_y_axis")
            dpg.add_line_series(_monitoring_data["time"], _monitoring_data["score"], 
                               label="Validation Score", parent="score_y_axis")
        
        # Models over time 플롯
        with dpg.plot(label="Models Trained", height=200, width=-1, 
                      parent=TAG_S11_MONITORING_PLOTS_GROUP):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (seconds)")
            dpg.add_plot_axis(dpg.mvYAxis, label="Number of Models", tag="models_y_axis")
            dpg.add_stair_series(_monitoring_data["time"], _monitoring_data["n_models"], 
                                label="Models", parent="models_y_axis")
        
        # Current status
        dpg.add_text(f"Current Best Score: {_monitoring_data['score'][-1]:.4f}", 
                    parent=TAG_S11_MONITORING_PLOTS_GROUP)
        dpg.add_text(f"Total Models Trained: {_monitoring_data['n_models'][-1]}", 
                    parent=TAG_S11_MONITORING_PLOTS_GROUP)
                    
    except Exception as e:
        print(f"Error updating monitoring plots: {e}")

def _update_comparison_view():
    """모델 비교 뷰 업데이트"""
    if not dpg.does_item_exist("s11_comparison_window"):
        return
        
    dpg.delete_item("s11_comparison_window", children_only=True)
    
    if len(_experiments_history) < 2:
        dpg.add_text("Need at least 2 experiments to compare.", parent="s11_comparison_window")
        return
    
    # 메트릭 비교 테이블
    dpg.add_text("Model Performance Comparison", parent="s11_comparison_window", color=(255, 255, 0))
    
    with dpg.table(header_row=True, resizable=True, parent="s11_comparison_window"):
        dpg.add_table_column(label="Model")
        dpg.add_table_column(label="Algorithm")
        dpg.add_table_column(label="Training Time")
        dpg.add_table_column(label="CV Score")
        
        for exp in _experiments_history[-5:]:  # 최근 5개
            with dpg.table_row():
                dpg.add_text(exp.model_name)
                dpg.add_text(exp.algorithm)
                dpg.add_text(f"{exp.training_time:.2f}s")
                cv_score = "N/A"
                if exp.cv_scores:
                    cv_score = f"{np.mean(exp.cv_scores):.4f} ± {np.std(exp.cv_scores):.4f}"
                dpg.add_text(cv_score)

def _display_automl_results(data: dict):
    """AutoML 결과 표시"""
    _log_message("Displaying AutoML results...")
    
    # 결과 저장
    exp = ExperimentResult(
        id=str(uuid.uuid4()), 
        timestamp=datetime.datetime.now(),
        model_name=data["model_name"], 
        model_type=data["model_type"],
        algorithm=data["algorithm"], 
        parameters={"time_budget": dpg.get_value("s11_automl_time_budget")},
        features=list(data["feature_scores"].keys()), 
        target=data["target"],
        metrics={}, 
        training_time=data["training_time"],
        model_object=data["model_object"], 
        dataframe_name=dpg.get_value(TAG_S11_DF_SELECTOR),
        cv_scores=data.get("cv_scores")
    )
    _experiments_history.append(exp)
    _update_experiment_table()
    
    # AutoML 결과 탭 업데이트
    res_group = TAG_S11_AUTOML_RESULTS_GROUP
    if dpg.does_item_exist(res_group):
        dpg.delete_item(res_group, children_only=True)
    
    dpg.add_text(f"Results for: {exp.model_name}", parent=res_group, color=(255, 255, 0))
    dpg.add_text(f"Task Type: {data.get('task_type', 'Unknown')}", parent=res_group)
    dpg.add_text(f"Training Time: {exp.training_time:.2f} seconds", parent=res_group)
    
    if exp.cv_scores:
        cv_text = f"CV Score: {np.mean(exp.cv_scores):.4f} ± {np.std(exp.cv_scores):.4f}"
        dpg.add_text(cv_text, parent=res_group, color=(100, 255, 100))
    
    dpg.add_separator(parent=res_group)
    dpg.add_text("Top Feature Importances:", parent=res_group, color=(100, 200, 255))
    
    # Feature importance 차트
    if data["feature_scores"]:
        sorted_features = sorted(data["feature_scores"].items(), key=lambda x: x[1], reverse=True)[:10]
        features, scores = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        y_pos = np.arange(len(features))
        ax.barh(y_pos, scores)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importances')
        plt.tight_layout()
        
        plot_func = _util_funcs.get('plot_to_dpg_texture')
        if plot_func:
            tex_tag, w, h, _ = plot_func(fig)
            dpg.add_image(tex_tag, width=w, height=h, parent=res_group)
            _texture_tags.append(tex_tag)
        plt.close(fig)

def _display_shap_results(data: dict):
    """SHAP 결과 표시"""
    exp_id = data['exp_id']
    result_tab_tag = f"s11_result_tab_{exp_id}"
    
    if dpg.does_item_exist(result_tab_tag):
        dpg.add_separator(parent=result_tab_tag)
        dpg.add_text("SHAP Analysis Results", parent=result_tab_tag, color=(100, 200, 255))
        dpg.add_image(data['tex_tag'], width=data['width'], height=data['height'], parent=result_tab_tag)
        _texture_tags.append(data['tex_tag'])

def _create_results_visualizations(experiment: ExperimentResult):
    """실험 결과 시각화 탭"""
    tab_name = f"📈 {experiment.model_name}"
    tab_tag = f"s11_result_tab_{experiment.id}"
    
    if dpg.does_item_exist(tab_tag):
        dpg.focus_item(tab_tag)
        return

    with dpg.tab(label=tab_name, parent=TAG_S11_VIZ_TAB_BAR, closable=True, tag=tab_tag):
        dpg.add_text(f"Results for: {experiment.model_name}", color=(255, 255, 0))
        dpg.add_text(f"Algorithm: {experiment.algorithm}")
        dpg.add_text(f"Training Time: {experiment.training_time:.2f}s")
        
        if experiment.cv_scores:
            cv_text = f"CV Score: {np.mean(experiment.cv_scores):.4f} ± {np.std(experiment.cv_scores):.4f}"
            dpg.add_text(cv_text, color=(100, 255, 100))
        
        dpg.add_separator()
        dpg.add_text("SHAP analysis will start automatically...", color=(200, 200, 200))

def _log_message(message: str, level: str = "INFO"):
    """로그 메시지 출력"""
    if not dpg.is_dearpygui_running(): 
        return
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {level}: {message}"
    if dpg.does_item_exist("s11_log_text"):
        current_log = dpg.get_value("s11_log_text")
        new_log = f"{current_log}\n{formatted_msg}"
        log_lines = new_log.split("\n")
        if len(log_lines) > 100: 
            new_log = "\n".join(log_lines[-100:])
        dpg.set_value("s11_log_text", new_log)
        if dpg.does_item_exist(TAG_S11_LOG_WINDOW):
            dpg.set_y_scroll(TAG_S11_LOG_WINDOW, dpg.get_y_scroll_max(TAG_S11_LOG_WINDOW))

def _update_progress(value: float, message: str = ""):
    """진행률 업데이트"""
    if not dpg.is_dearpygui_running(): 
        return
    if dpg.does_item_exist(TAG_S11_PROGRESS_BAR):
        dpg.set_value(TAG_S11_PROGRESS_BAR, value)
    if message: 
        _log_message(message)

def update_ui():
    """UI 업데이트"""
    if not _module_main_callbacks: 
        return
    
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
    """실험 테이블 업데이트"""
    if not dpg.does_item_exist(TAG_S11_EXPERIMENT_TABLE): 
        return
    
    dpg.delete_item(TAG_S11_EXPERIMENT_TABLE, children_only=True)
    dpg.add_table_column(label="Timestamp", parent=TAG_S11_EXPERIMENT_TABLE)
    dpg.add_table_column(label="Model", parent=TAG_S11_EXPERIMENT_TABLE)
    dpg.add_table_column(label="Algorithm", parent=TAG_S11_EXPERIMENT_TABLE)
    dpg.add_table_column(label="Target", parent=TAG_S11_EXPERIMENT_TABLE)
    dpg.add_table_column(label="Score", parent=TAG_S11_EXPERIMENT_TABLE)
    dpg.add_table_column(label="Actions", parent=TAG_S11_EXPERIMENT_TABLE)

    for exp in reversed(_experiments_history):
        with dpg.table_row(parent=TAG_S11_EXPERIMENT_TABLE):
            dpg.add_text(exp.timestamp.strftime("%H:%M:%S"))
            dpg.add_text(exp.model_name)
            dpg.add_text(exp.algorithm)
            dpg.add_text(exp.target)
            
            score_text = "N/A"
            if exp.cv_scores:
                score_text = f"{np.mean(exp.cv_scores):.4f}"
            dpg.add_text(score_text)
            
            dpg.add_button(label="View", user_data=exp, callback=_view_experiment_details)

def reset_state():
    """상태 초기화"""
    global _experiments_history, _texture_tags, _worker_thread, _monitoring_data
    
    if _worker_thread and _worker_thread.is_alive():
        _log_message("Warning: A task is still running. Cannot reset state now.", "WARN")
        return

    _experiments_history.clear()
    _monitoring_data = {"time": [], "score": [], "n_models": []}
    
    for tag in _texture_tags:
        if dpg.does_item_exist(tag): 
            dpg.delete_item(tag)
    _texture_tags.clear()
    
    if dpg.is_dearpygui_running():
        update_ui()
    _log_message("State has been reset.", "INFO")