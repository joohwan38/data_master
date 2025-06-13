# step_11_ml_modeling.py - 2-Track 하이브리드 접근법 구현

"""
Step 11 ML Modeling & AI 통합 모듈 (하이브리드 접근법)

주요 기능:
- 2-Track 하이브리드 접근법:
  - Track 1: 자동화된 모델 탐색 (하이퍼파라미터 튜닝 포함) 및 리더보드
  - Track 2: 심층 분석, 사용자 정의 튜닝 및 인사이트 도출
- 모델 인사이트 랩: SHAP Beeswarm, Dependence Plot 등 다양한 XAI 시각화
- 앙상블: 버튼 클릭으로 상위 모델을 결합하여 성능 극대화
- 모델 서빙: 학습된 모델 파이프라인을 내보내고, 새로운 데이터에 대한 추론 수행
- 고급 훈련 옵션: CV 폴드, 최적화 지표 등 세부 제어
- 딥러닝 확장 기반: MLP 모델 추가를 위한 UI 및 코드 구조 포함
"""

import dearpygui.dearpygui as dpg
import pandas as pd
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
import joblib
import os

# Scikit-learn 및 관련 라이브러리
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                           roc_curve, roc_auc_score,auc, mean_squared_error, r2_score, mean_absolute_error, make_scorer,
                           cohen_kappa_score, matthews_corrcoef)

from sklearn.ensemble import VotingClassifier, VotingRegressor

# 모델 알고리즘
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP 라이브러리를 찾을 수 없습니다. 모델 설명 기능이 비활성화됩니다.")

# 사용자 참고: 딥러닝 모델을 사용하려면 tensorflow 또는 pytorch 설치가 필요합니다.
# try:
#     import tensorflow as tf
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
#     TENSORFLOW_AVAILABLE = True
# except ImportError:
#     TENSORFLOW_AVAILABLE = False
#     print("Warning: TensorFlow 라이브러리를 찾을 수 없습니다. 딥러닝 모델 기능이 비활성화됩니다.")
TENSORFLOW_AVAILABLE = False # 우선 비활성화


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- DPG Tags ---
TAG_S11_GROUP = "step11_ml_modeling_group"
TAG_S11_MAIN_TAB_BAR = "step11_main_tab_bar"
TAG_S11_MODELING_TAB = "step11_modeling_tab"
TAG_S11_INFERENCE_TAB = "step11_inference_tab"

# Modeling Tab UI
TAG_S11_DF_SELECTOR = "step11_df_selector"
TAG_S11_TARGET_SELECTOR = "step11_target_selector"
TAG_S11_TASK_TYPE_TEXT = "s11_detected_task_text"
TAG_S11_RUN_AUTO_DISCOVERY_BUTTON = "step11_run_auto_discovery_button"
TAG_S11_LEADERBOARD_TABLE = "step11_leaderboard_table"
TAG_S11_LEADERBOARD_GROUP = "step11_leaderboard_group"
TAG_S11_ENSEMBLE_BUTTON = "step11_ensemble_button"
TAG_S11_DEEP_DIVE_GROUP = "step11_deep_dive_group"
TAG_S11_LOG_TEXT = "step11_log_text"
TAG_S11_LOG_WINDOW = "step11_log_window"
TAG_S11_PROGRESS_BAR = "step11_progress_bar"

# --- ML 알고리즘 및 하이퍼파라미터 탐색 범위 정의 ---
# RandomizedSearchCV를 위한 파라미터 분포 정의
PARAM_DISTRIBUTIONS = {
    "Classification": {
        "Random Forest": {
            "model_class": RandomForestClassifier,
            "params": {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, 30, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
        },
        "XGBoost": {
            "model_class": XGBClassifier,
            "params": {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7],
                'classifier__subsample': [0.7, 0.8, 0.9]
            }
        },
        "LightGBM": {
            "model_class": LGBMClassifier,
            "params": {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__num_leaves': [20, 31, 40],
                'classifier__max_depth': [-1, 10, 20]
            }
        }
    },
    "Regression": {
        "Random Forest": {
            "model_class": RandomForestRegressor,
            "params": {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__max_depth': [10, 20, 30, None],
                'regressor__min_samples_split': [2, 5, 10],
                'regressor__min_samples_leaf': [1, 2, 4]
            }
        },
        "XGBoost": {
            "model_class": XGBRegressor,
            "params": {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__max_depth': [3, 5, 7],
                'regressor__subsample': [0.7, 0.8, 0.9]
            }
        },
        "LightGBM": {
            "model_class": LGBMRegressor,
            "params": {
                'regressor__n_estimators': [100, 200, 300],
                'regressor__learning_rate': [0.01, 0.1, 0.2],
                'regressor__num_leaves': [20, 31, 40],
                'regressor__max_depth': [-1, 10, 20]
            }
        }
    }
}
# 기본 모델 (튜닝 없이 빠르게 실행)
BASE_MODELS = {
    "Classification": {"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)},
    "Regression": {"Linear Regression": LinearRegression()}
}


# --- Module State ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_texture_tags: List[str] = []
_results_queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_leaderboard_results: List[Dict] = []
_current_deep_dive_model: Optional[Dict] = None

def _calculate_all_metrics(pipeline, X_test, y_test, task_type):
    """모든 평가지표를 계산하는 헬퍼 함수"""
    y_pred = pipeline.predict(X_test)
    # metrics 딕셔너리를 비어있는 상태에서 시작합니다.
    metrics = {}

    if task_type == "Classification":
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
        metrics["AUC"] = roc_auc_score(y_test, y_proba)
        metrics["Recall"] = recall_score(y_test, y_pred, average='macro', zero_division=0)
        metrics["Prec."] = precision_score(y_test, y_pred, average='macro', zero_division=0)
        metrics["F1"] = f1_score(y_test, y_pred, average='macro', zero_division=0)
        metrics["Kappa"] = cohen_kappa_score(y_test, y_pred)
        metrics["MCC"] = matthews_corrcoef(y_test, y_pred)
    else: # Regression
        metrics["MAE"] = mean_absolute_error(y_test, y_pred)
        metrics["MSE"] = mean_squared_error(y_test, y_pred)
        metrics["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics["R2"] = r2_score(y_test, y_pred)

    return metrics


# --- Data Preparation ---
def _prepare_data_for_modeling(df: pd.DataFrame, target_name: str) -> Tuple:
    """
    데이터를 모델링에 맞게 준비. Step 1-10의 전처리 결과를 최대한 존중.
    결측치 처리, 인코딩, 데이터 분할을 수행. 스케일링은 파이프라인의 일부로 남겨둠.
    """
    _log_message("데이터 준비 시작: X, y 분리...")
    X = df.drop(columns=[target_name])
    y = df[target_name]

    task_type = "Regression"
    le_map = {} # 라벨 인코딩 정보 저장
    if y.dtype == 'object' or y.nunique() < 25:
        task_type = "Classification"
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        le_map = dict(zip(le.classes_, le.transform(le.classes_)))
        y = pd.Series(y_encoded, index=y.index, name=y.name)
        _log_message(f"타겟 변수 라벨 인코딩 완료. 매핑: {le_map}")

    _log_message(f"태스크 타입 감지: {task_type}")

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    _log_message(f"숫자형 피처: {len(numeric_features)}개, 범주형 피처: {len(categorical_features)}개")

    # 전처리 파이프라인 정의 (결측치 처리 및 인코딩)
    # 스케일링은 각 모델 파이프라인에 포함하여 하이퍼파라미터 튜닝 시 정보 유출 방지
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if task_type == 'Classification' else None)
    _log_message("데이터 분할 완료 (훈련 75% / 테스트 25%).")

    return X, y, X_train, X_test, y_train, y_test, preprocessor, task_type, le_map

# --- Background Thread Functions ---
def _run_automated_discovery_thread(df: pd.DataFrame, target_name: str, cv_folds: int, optimization_metric: str):
    print("\n" + "="*60)
    print(">>> STEP 11 DEBUG: Modeling function received the following DataFrame:")
    print(df.info())
    print("="*60 + "\n")
    """[Track 1] 자동화된 모델 탐색을 백그라운드에서 수행"""
    try:
        # 1. 데이터 준비
        _results_queue.put({"type": "progress", "value": 0.05, "log": "데이터 준비 중..."})
        X, y, X_train, X_test, y_train, y_test, preprocessor, task_type, le_map = _prepare_data_for_modeling(df, target_name)

        # 2. 탐색할 모델 목록 정의
        models_to_tune = PARAM_DISTRIBUTIONS.get(task_type, {})
        base_models_to_run = BASE_MODELS.get(task_type, {})
        all_models = list(base_models_to_run.keys()) + list(models_to_tune.keys())
        total_steps = len(all_models)
        
        # Scikit-learn의 scoring 이름과 표시 이름을 매핑
        metric_name_map = {
            "accuracy": "Accuracy", "f1": "F1", "recall": "Recall",
            "precision": "Prec.", "roc_auc": "AUC",
            "r2": "R2", "neg_mean_absolute_error": "MAE", "neg_mean_squared_error": "MSE"
        }
        # UI에서 받은 값(e.g., 'F1')을 scikit-learn이 이해하는 이름(e.g., 'f1')으로 변환
        inverse_metric_name_map = {v: k for k, v in metric_name_map.items()}
        sklearn_metric_name = inverse_metric_name_map.get(optimization_metric, optimization_metric.lower())


        # 3. 모델 훈련 및 튜닝 루프
        results = []
        for i, model_name in enumerate(all_models):
            progress = 0.1 + (i / total_steps) * 0.8
            _results_queue.put({"type": "progress", "value": progress, "log": f"({i+1}/{total_steps}) {model_name} 모델 훈련/튜닝 중..."})
            start_time = time.time()

            # 파이프라인 구성
            model_instance = None
            param_dist = None # param_dist 초기화
            if model_name in models_to_tune:  # 튜닝 대상 모델
                model_config = models_to_tune[model_name]
                model_class = model_config["model_class"]
                model_instance = model_class(random_state=42)
                param_dist = model_config["params"]
            else:  # 기본 모델
                model_instance = base_models_to_run[model_name]

            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('scaler', StandardScaler(with_mean=False)),
                ('classifier' if task_type == 'Classification' else 'regressor', model_instance)
            ])

            # RandomizedSearchCV 또는 일반 fit 수행
            best_pipeline = None
            best_params = {}
            cv_score = 0.0

            if model_name in models_to_tune:
                search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=10, cv=cv_folds, scoring=sklearn_metric_name, random_state=42, n_jobs=2)
                search.fit(X, y)
                best_pipeline = search.best_estimator_
                cv_score = search.best_score_
                best_params = search.best_params_
                model_display_name = f"Tuned_{model_name.replace(' ', '')}"
            else:  # 기본 모델은 CV 점수만 계산
                from sklearn.model_selection import cross_val_score
                cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring=sklearn_metric_name)
                cv_score = np.mean(cv_scores)
                pipeline.fit(X_train, y_train)  # 테스트를 위해 fit은 필요
                best_pipeline = pipeline
                best_params = {"default": "default"}
                model_display_name = model_name
            
            training_time = time.time() - start_time

            # 4. 모든 평가지표 계산
            test_metrics = _calculate_all_metrics(best_pipeline, X_test, y_test, task_type)

            # 5. 결과 저장
            results.append({
                "id": str(uuid.uuid4()),
                "name": model_display_name,
                "cv_score": cv_score, # RandomizedSearchCV 또는 cross_val_score 결과
                "primary_metric_key": optimization_metric, # 정렬 기준이 될 키 (e.g., F1)
                "time": training_time,
                "pipeline": best_pipeline,
                "params": best_params,
                "test_metrics": test_metrics, # 모든 테스트 지표가 담긴 딕셔너리
                "task_info": {"type": task_type, "target": target_name, "le_map": le_map,
                              "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
            })

        _results_queue.put({"type": "discovery_result", "data": results})

    except Exception as e:
        error_msg = f"자동 모델 탐색 오류: {str(e)}\n{traceback.format_exc()}"
        _results_queue.put({"type": "error", "log": error_msg})

# --- UI Creation ---
def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()

    main_callbacks['register_step_group_tag'](step_name, TAG_S11_GROUP)

    with dpg.group(tag=TAG_S11_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        
        with dpg.tab_bar(tag=TAG_S11_MAIN_TAB_BAR):
            with dpg.tab(label="모델링 (Modeling)", tag=TAG_S11_MODELING_TAB):
                _create_modeling_tab_ui()
            with dpg.tab(label="추론 (Inference)", tag=TAG_S11_INFERENCE_TAB):
                _create_inference_tab_ui()

    main_callbacks['register_module_updater'](step_name, update_ui)

def _create_modeling_tab_ui():
    # 전체를 감싸는 부모 그룹
    with dpg.group(tag="s11_modeling_main_group"):
        # --- 상단 패널: 좌우 분할 (제어판 + 결과창) ---
        with dpg.group(horizontal=True):
            # --- 좌측 제어판 ---
            with dpg.group(width=400): # 너비를 400으로 소폭 조정
                dpg.add_text("1. 설정 (Setup)", color=(255, 255, 0))
                dpg.add_separator()
                dpg.add_text("데이터 소스 선택:")
                dpg.add_combo(label="", tag=TAG_S11_DF_SELECTOR, width=-1, callback=_on_df_or_target_selected)
                dpg.add_text("타겟 변수 (y) 선택:")
                dpg.add_combo(label="", tag=TAG_S11_TARGET_SELECTOR, width=-1, callback=_on_df_or_target_selected)
                with dpg.group(horizontal=True):
                    dpg.add_text("감지된 태스크 타입:")
                    dpg.add_text("(데이터 선택)", tag=TAG_S11_TASK_TYPE_TEXT, color=(255, 255, 0))

                dpg.add_separator()
                dpg.add_text("2. 자동 탐색 설정 (Automated Discovery)", color=(255, 255, 0))
                dpg.add_text("교차 검증 (CV) 폴드 수:")
                dpg.add_combo(items=["3", "5", "10"], default_value="5", width=-1, tag="s11_cv_folds_selector")
                
                # [수정] 기본 최적화 지표를 선택하는 UI. 실제 표시는 모든 지표가 나옴.
                dpg.add_text("정렬 기준 지표 (Primary Metric):")
                dpg.add_combo(items=["F1", "Accuracy", "Recall", "Precision", "AUC", "Kappa"], default_value="F1", label="분류", width=-1, tag="s11_clf_metric_selector")
                dpg.add_combo(items=["R2", "MAE", "MSE"], default_value="R2", label="회귀", width=-1, tag="s11_reg_metric_selector", show=False)

                dpg.add_spacer(height=10)
                dpg.add_button(label="🚀 최적 모델 자동 탐색 실행", tag=TAG_S11_RUN_AUTO_DISCOVERY_BUTTON, width=-1, height=40,
                               callback=_start_automated_discovery_callback)

            # --- 우측 결과창 ---
            with dpg.group():
                # 리더보드
                with dpg.group(tag=TAG_S11_LEADERBOARD_GROUP):
                    with dpg.group(horizontal=True):
                        dpg.add_text("모델 성능 리더보드", color=(255, 255, 0))
                        dpg.add_spacer()
                        dpg.add_button(label="🔄", callback=_update_leaderboard_display)
                        dpg.add_button(label=" Ensemble", tag=TAG_S11_ENSEMBLE_BUTTON, show=False, callback=_create_ensemble_callback)
                    # 리더보드 테이블 높이를 -1로 설정하여 가변적으로 만듦
                    dpg.add_table(tag=TAG_S11_LEADERBOARD_TABLE, header_row=True, resizable=True,
                                  borders_innerV=True, borders_outerH=True, height=300,
                                  policy=dpg.mvTable_SizingStretchSame)

                dpg.add_separator()
                # 심층 분석 영역 (리더보드 아래에 위치하게 됨)
                with dpg.group(tag=TAG_S11_DEEP_DIVE_GROUP, show=False):
                    dpg.add_text("심층 분석 및 사용자 정의 (Deep Dive & Customization)", color=(255, 255, 0))
                    _create_deep_dive_ui()

        dpg.add_separator()
        # --- 하단 패널: 전체 너비 로그 창 ---
        with dpg.group():
            dpg.add_text("진행 상황 및 로그", color=(100, 200, 255))
            dpg.add_progress_bar(tag=TAG_S11_PROGRESS_BAR, default_value=0.0, width=-1)
            # 로그 창의 높이를 150으로 고정, 너비는 화면 전체
            with dpg.child_window(height=150, tag=TAG_S11_LOG_WINDOW, border=True):
                dpg.add_input_text(default_value="자동 탐색을 시작하세요.", tag=TAG_S11_LOG_TEXT, multiline=True, readonly=True, width=-1, height=-1)


def _create_deep_dive_ui():
    """Track 2: 심층 분석 및 사용자 정의 UI를 생성하는 함수"""
    with dpg.child_window(border=True):
        dpg.add_text("Selected Model: ", tag="s11_deep_dive_model_name")
        dpg.add_separator()
        with dpg.tab_bar():
            with dpg.tab(label="모델 인사이트 (Insights)"):
                with dpg.group(horizontal=True):
                    dpg.add_text("분석할 피처 선택:")
                    dpg.add_combo(tag="s11_insight_feature_selector", width=200)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="SHAP 요약 플롯 (Beeswarm)", callback=lambda: _run_shap_analysis("summary"))
                    dpg.add_button(label="SHAP 의존성 플롯", callback=lambda: _run_shap_analysis("dependence"))
                dpg.add_separator()
                dpg.add_text("개별 예측 분석 (데이터 테이블에서 행 선택 필요)")
                dpg.add_button(label="선택된 데이터 예측 근거 보기 (SHAP Force)", callback=lambda: _run_shap_analysis("force"))

                with dpg.child_window(tag="s11_insight_plots_window", border=True):
                    dpg.add_text("분석 결과를 여기에 표시합니다.")

            with dpg.tab(label="파라미터 튜닝 (Tuning)"):
                dpg.add_text("이곳에서 파라미터를 직접 수정하고 재훈련할 수 있습니다. (구현 예정)")
                # 사용자 정의 튜닝 UI가 여기에 들어갑니다.

            with dpg.tab(label="성능 (Performance)"):
                dpg.add_text("테스트 데이터 성능")
                with dpg.table(tag="s11_deep_dive_perf_table", header_row=True):
                    dpg.add_table_column(label="Metric")
                    dpg.add_table_column(label="Value")
                dpg.add_text("Confusion Matrix 등 시각화가 여기에 표시됩니다. (구현 예정)")

def _create_inference_tab_ui():
    """추론 탭 UI를 생성하는 함수"""
    dpg.add_text("학습된 모델을 사용하여 새로운 데이터에 대한 예측을 수행합니다.", wrap=500)
    dpg.add_separator()
    dpg.add_button(label="💾 저장된 모델 불러오기", callback=_load_model_for_inference)
    dpg.add_text("불러온 모델: 없음", tag="s11_inference_model_path")
    dpg.add_separator()
    dpg.add_button(label="📄 예측할 데이터 불러오기", callback=_load_data_for_inference)
    dpg.add_text("예측할 데이터: 없음", tag="s11_inference_data_path")
    dpg.add_table(tag="s11_inference_data_preview", header_row=True, height=150)
    dpg.add_separator()
    dpg.add_button(label="실행", width=-1, callback=_run_inference, height=30)
    dpg.add_separator()
    dpg.add_text("예측 결과")
    with dpg.group(horizontal=True):
        dpg.add_text("", tag="s11_inference_result_count")
        dpg.add_button(label="결과 다운로드", show=False, tag="s11_inference_download_button", callback=_download_inference_result)
    dpg.add_table(tag="s11_inference_result_table", header_row=True, height=200)

# --- Callbacks & UI Update Functions ---
def _on_df_or_target_selected(sender, app_data, user_data):
    """데이터 소스 또는 타겟 변수 선택 시 UI 업데이트"""
    df_name = dpg.get_value(TAG_S11_DF_SELECTOR)
    target_name = dpg.get_value(TAG_S11_TARGET_SELECTOR)
    all_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    if not df_name or df_name not in all_dfs:
        return

    df = all_dfs[df_name]
    if sender == TAG_S11_DF_SELECTOR:
        cols = [""] + df.columns.tolist()
        dpg.configure_item(TAG_S11_TARGET_SELECTOR, items=cols)
        dpg.set_value(TAG_S11_TARGET_SELECTOR, "")

    if target_name and target_name in df.columns:
        y = df[target_name]
        task_type = "Regression"
        if y.dtype == 'object' or y.nunique() < 25:
             task_type = "Classification"
        dpg.set_value(TAG_S11_TASK_TYPE_TEXT, f"{task_type}")
        # 태스크 타입에 따라 메트릭 콤보박스 표시/숨김
        dpg.configure_item("s11_clf_metric_selector", show=(task_type == "Classification"))
        dpg.configure_item("s11_reg_metric_selector", show=(task_type == "Regression"))
    else:
        dpg.set_value(TAG_S11_TASK_TYPE_TEXT, "(타겟 선택)")

def _start_automated_discovery_callback():
    """'최적 모델 자동 탐색 실행' 버튼 콜백"""
    global _leaderboard_results
    df_name = dpg.get_value(TAG_S11_DF_SELECTOR)
    target_name = dpg.get_value(TAG_S11_TARGET_SELECTOR)
    if not df_name or not target_name:
        _util_funcs['_show_simple_modal_message']("설정 오류", "데이터 소스와 타겟 변수를 모두 선택해주세요.")
        return

    # 설정값 가져오기
    cv_folds = int(dpg.get_value("s11_cv_folds_selector"))
    task_type = dpg.get_value(TAG_S11_TASK_TYPE_TEXT)
    metric_selector = "s11_clf_metric_selector" if task_type == "Classification" else "s11_reg_metric_selector"
    optimization_metric = dpg.get_value(metric_selector)

    all_dfs = _module_main_callbacks.get('get_all_available_dfs')()
    df = all_dfs.get(df_name)
    if df is None:
        _util_funcs['_show_simple_modal_message']("오류", "선택된 데이터프레임을 찾을 수 없습니다.")
        return

    # 상태 초기화
    _leaderboard_results.clear()
    _update_leaderboard_display()
    dpg.configure_item(TAG_S11_DEEP_DIVE_GROUP, show=False)
    dpg.configure_item(TAG_S11_ENSEMBLE_BUTTON, show=False)

    _start_background_task(_run_automated_discovery_thread, args=(df.copy(), target_name, cv_folds, optimization_metric))

def _update_leaderboard_display():
    """리더보드 테이블 UI를 현재 _leaderboard_results 기준으로 새로고침"""
    table = TAG_S11_LEADERBOARD_TABLE
    dpg.delete_item(table, children_only=True)

    if not _leaderboard_results:
        dpg.add_table_column(label="알림", parent=table)
        with dpg.table_row(parent=table):
            dpg.add_text("자동 탐색을 실행하여 결과를 확인하세요.")
        return

    # 정렬 기준(Primary Metric)에 따라 결과 정렬
    primary_metric = _leaderboard_results[0].get("primary_metric_key", "F1")
    sorted_results = sorted(
        _leaderboard_results,
        key=lambda x: x.get("test_metrics", {}).get(primary_metric, 0),
        reverse=True
    )
    
    # 헤더(컬럼명) 생성
    metric_keys = list(sorted_results[0].get("test_metrics", {}).keys())
    header_keys = ['Model'] + metric_keys # 'Model' 컬럼을 맨 앞에 추가
    
    best_scores = {}
    for key in metric_keys:
        scores = [res.get("test_metrics", {}).get(key, -np.inf) for res in sorted_results]
        best_scores[key] = max(scores)

    for key in header_keys:
        dpg.add_table_column(label=key, parent=table)
    dpg.add_table_column(label="분석/저장", parent=table)

    highlight_color = (255, 255, 150, 100)
    
    for row_idx, res in enumerate(sorted_results):
        with dpg.table_row(parent=table):
            # 1. 모델 이름 표시 (수정된 핵심)
            with dpg.table_cell():
                dpg.add_text(res.get("name", "N/A"))
            
            # 2. 나머지 메트릭 표시
            metrics = res.get("test_metrics", {})
            for col_idx, key in enumerate(metric_keys):
                value = metrics.get(key, "N/A")
                with dpg.table_cell():
                    if isinstance(value, float):
                        cell_text = f"{value:.4f}"
                        dpg.add_text(cell_text)
                        if key in best_scores and abs(value - best_scores[key]) < 1e-6:
                            # 모델 이름 컬럼이 빠졌으므로 col_idx는 그대로 사용 가능
                            dpg.highlight_table_cell(table, row_idx, col_idx + 1, color=highlight_color)
                    else:
                        dpg.add_text(str(value))
            
            with dpg.table_cell():
                 with dpg.group(horizontal=True):
                    dpg.add_button(label="상세 분석", user_data=res["id"], callback=_select_model_for_deep_dive)
                    dpg.add_button(label="💾", user_data=res["id"], callback=_export_model_callback)

    can_ensemble = len([r for r in _leaderboard_results if 'Tuned' in r['name']]) >= 2
    dpg.configure_item(TAG_S11_ENSEMBLE_BUTTON, show=can_ensemble)

def _select_model_for_deep_dive(sender, app_data, user_data_model_id):
    """리더보드에서 모델을 선택하여 심층 분석 UI를 활성화"""
    global _current_deep_dive_model
    model_data = next((res for res in _leaderboard_results if res["id"] == user_data_model_id), None)
    if not model_data:
        _log_message(f"오류: 모델 ID {user_data_model_id}를 찾을 수 없습니다.", "ERROR")
        return

    _current_deep_dive_model = model_data
    dpg.configure_item(TAG_S11_DEEP_DIVE_GROUP, show=True)
    dpg.set_value("s11_deep_dive_model_name", f"Selected Model: {model_data['name']}")

    # 인사이트 탭 피처 콤보박스 채우기
    X_train = model_data['task_info']['X_train']
    dpg.configure_item("s11_insight_feature_selector", items=X_train.columns.tolist())

    # 성능 탭 채우기
    perf_table = "s11_deep_dive_perf_table"
    dpg.delete_item(perf_table, children_only=True)
    dpg.add_table_column(label="Metric", parent=perf_table)
    dpg.add_table_column(label="Value", parent=perf_table)
    for metric, value in model_data['test_metrics'].items():
        with dpg.table_row(parent=perf_table):
            dpg.add_text(metric)
            dpg.add_text(f"{value:.4f}")
    
    _log_message(f"'{model_data['name']}' 모델이 심층 분석 대상으로 선택되었습니다.")

def _export_model_callback(sender, app_data, user_data_model_id):
    """모델 내보내기 콜백"""
    model_data = next((res for res in _leaderboard_results if res["id"] == user_data_model_id), None)
    if not model_data:
        _util_funcs['_show_simple_modal_message']("오류", "저장할 모델을 찾을 수 없습니다.")
        return

    # 파일 저장 다이얼로그 (main_app.py에 정의되어 있다고 가정)
    # 여기서는 간단히 파일명을 정하고 저장합니다.
    try:
        save_dir = "trained_models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_data['name']}_{timestamp}.joblib"
        save_path = os.path.join(save_dir, filename)

        joblib.dump(model_data['pipeline'], save_path)
        _util_funcs['_show_simple_modal_message']("저장 완료", f"모델이 다음 경로에 저장되었습니다:\n{save_path}")
        _log_message(f"모델 '{model_data['name']}'이(가) '{save_path}'에 저장되었습니다.")
    except Exception as e:
        _util_funcs['_show_simple_modal_message']("저장 실패", f"모델 저장 중 오류가 발생했습니다:\n{e}")
        _log_message(f"모델 저장 실패: {e}", "ERROR")

# --- 기타 유틸리티 및 헬퍼 함수 ---
def _calculate_metrics(y_true, y_pred, task_type):
    """성능 지표를 계산하는 헬퍼 함수"""
    if task_type == "Classification":
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1 Score (Weighted)": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "Recall (Weighted)": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "Precision (Weighted)": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        }
    else: # Regression
        return {
            "R2 Score": r2_score(y_true, y_pred),
            "Mean Squared Error (MSE)": mean_squared_error(y_true, y_pred),
            "Mean Absolute Error (MAE)": mean_absolute_error(y_true, y_pred),
        }

def _start_background_task(target_func, args):
    """백그라운드 스레드를 시작하는 래퍼 함수"""
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        _util_funcs['_show_simple_modal_message']("작업 중", "이전 작업이 아직 실행 중입니다. 잠시 후 다시 시도해주세요.")
        return
    _worker_thread = threading.Thread(target=target_func, args=args, daemon=True)
    _worker_thread.start()
    _update_progress(0.0, "작업을 시작했습니다...")

def _check_for_updates():
    """메인 스레드에서 주기적으로 호출되어 큐를 확인하고 UI를 업데이트"""
    global _worker_thread, _leaderboard_results
    try:
        result = _results_queue.get_nowait()
        
        if result["type"] == "progress":
            _update_progress(result["value"], result["log"])
        elif result["type"] == "error":
            _log_message(result["log"], "ERROR")
            _update_progress(0.0, "작업 실패.")
        elif result["type"] == "discovery_result":
            _leaderboard_results = result["data"]
            _update_leaderboard_display()
            _update_progress(1.0, "자동 탐색 완료.")
            time.sleep(1)
            _update_progress(0.0)
        
        # 다른 result type (e.g., shap_result) 처리 로직 추가 가능

        _results_queue.task_done()
    except queue.Empty:
        pass
    except Exception as e:
        print(f"업데이트 확인 중 오류 발생: {e}\n{traceback.format_exc()}")

    if _worker_thread and not _worker_thread.is_alive():
        _worker_thread = None

def _update_progress(value: float, message: str = ""):
    """진행 바 및 로그 메시지 업데이트"""
    if not dpg.is_dearpygui_running(): return
    if dpg.does_item_exist(TAG_S11_PROGRESS_BAR):
        dpg.set_value(TAG_S11_PROGRESS_BAR, value)
    if message: 
        _log_message(message)

def _log_message(message: str, level: str = "INFO"):
    """로그 창에 메시지 추가"""
    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_S11_LOG_TEXT): return
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {level}: {message}"
    
    current_log = dpg.get_value(TAG_S11_LOG_TEXT)
    if "자동 탐색을 시작하세요." in current_log:
        current_log = ""

    new_log = f"{current_log}\n{formatted_msg}"
    log_lines = new_log.split("\n")
    if len(log_lines) > 100: 
        new_log = "\n".join(log_lines[-100:])
        
    dpg.set_value(TAG_S11_LOG_TEXT, new_log)
    if dpg.does_item_exist(TAG_S11_LOG_WINDOW):
        dpg.set_y_scroll(TAG_S11_LOG_WINDOW, dpg.get_y_scroll_max(TAG_S11_LOG_WINDOW))

# --- main_app.py와의 인터페이스 함수 ---
def update_ui():
    """main_app에서 호출하여 UI 상태를 업데이트 (e.g., 데이터 소스 목록)"""
    if not _module_main_callbacks: return
    all_dfs = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    if dpg.does_item_exist(TAG_S11_DF_SELECTOR):
        df_names = [""] + list(all_dfs.keys())
        dpg.configure_item(TAG_S11_DF_SELECTOR, items=df_names)

def reset_state():
    """애플리케이션 리셋 시 호출되어 이 모듈의 상태를 초기화"""
    global _worker_thread, _leaderboard_results, _texture_tags, _current_deep_dive_model
    if _worker_thread and _worker_thread.is_alive():
        return # 작업 중 리셋 방지

    _leaderboard_results.clear()
    _current_deep_dive_model = None
    for tag in _texture_tags:
        if dpg.does_item_exist(tag): 
            dpg.delete_item(tag)
    _texture_tags.clear()
    
    if dpg.is_dearpygui_running():
        _update_leaderboard_display()
        dpg.configure_item(TAG_S11_DEEP_DIVE_GROUP, show=False)
        dpg.set_value(TAG_S11_MAIN_TAB_BAR, TAG_S11_MODELING_TAB)
        update_ui()
    
    _log_message("ML 모델링 상태가 초기화되었습니다.", "INFO")

# --- 아래는 심층 분석, 앙상블, 추론 등 추가 구현이 필요한 기능들의 자리입니다 ---

def _create_ensemble_callback():
    _log_message("앙상블 기능은 현재 구현 준비 중입니다.", "INFO")
    # TODO: 상위 모델 2-3개를 선택하여 VotingClassifier/Regressor를 만들고
    # _leaderboard_results에 추가한 후 _update_leaderboard_display() 호출

def _run_shap_analysis(plot_type: str):
    _log_message(f"SHAP 분석 ({plot_type}) 기능은 현재 구현 준비 중입니다.", "INFO")
    # TODO: _current_deep_dive_model을 기반으로 SHAP 분석 스레드 실행
    # 1. Explainer 생성
    # 2. shap_values 계산
    # 3. plot_type에 맞는 플롯 생성 (matplotlib)
    # 4. plot_to_dpg_texture 유틸리티로 DPG 텍스처 변환
    # 5. s11_insight_plots_window에 이미지 추가

def _load_model_for_inference():
    _log_message("모델 불러오기 기능은 파일 다이얼로그 연동 후 구현됩니다.", "INFO")

def _load_data_for_inference():
    _log_message("데이터 불러오기 기능은 파일 다이얼로그 연동 후 구현됩니다.", "INFO")

def _run_inference():
    _log_message("추론 실행 기능은 모델/데이터 로딩 구현 후 활성화됩니다.", "INFO")

def _download_inference_result():
    _log_message("결과 다운로드 기능은 추론 실행 구현 후 활성화됩니다.", "INFO")