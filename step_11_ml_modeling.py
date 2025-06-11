# step_11_ml_modeling.py - ML/AI 모델링 통합 모듈

"""
Step 11 ML Modeling & AI 통합 모듈

ML/NN/AI 모델링 기능을 제공하는 모듈
- 분류, 회귀, 클러스터링
- 모델 학습 및 평가
- 하이퍼파라미터 튜닝
- 결과 시각화
"""

import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import traceback
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# --- DPG Tags ---
TAG_S11_GROUP = "step11_ml_modeling_group"
TAG_S11_UPPER_VIZ_WINDOW = "step11_upper_viz_window"
TAG_S11_LOWER_CONTROL_PANEL = "step11_lower_control_panel"
TAG_S11_VIZ_TAB_BAR = "step11_viz_tab_bar"
TAG_S11_DF_SELECTOR = "step11_df_selector"
TAG_S11_MODEL_TYPE_SELECTOR = "step11_model_type_selector"
TAG_S11_ALGORITHM_SELECTOR = "step11_algorithm_selector"
TAG_S11_FEATURE_LIST = "step11_feature_list"
TAG_S11_TARGET_COMBO = "step11_target_combo"
TAG_S11_TRAIN_BUTTON = "step11_train_button"
TAG_S11_DYNAMIC_PARAMS_AREA = "step11_dynamic_params_area"

# --- Module State ---
_module_main_callbacks: Optional[Dict] = None
_util_funcs: Optional[Dict[str, Any]] = None
_available_dfs: Dict[str, pd.DataFrame] = {}
_selected_df_name: str = ""
_selected_features: List[str] = []
_trained_models: Dict[str, Any] = {}
_model_counter: int = 0

# --- ML 알고리즘 정의 ---
ML_ALGORITHMS = {
    "Classification": {
        "Logistic Regression": {"module": "sklearn.linear_model", "class": "LogisticRegression"},
        "Random Forest": {"module": "sklearn.ensemble", "class": "RandomForestClassifier"},
        "SVM": {"module": "sklearn.svm", "class": "SVC"},
        "KNN": {"module": "sklearn.neighbors", "class": "KNeighborsClassifier"},
        "Decision Tree": {"module": "sklearn.tree", "class": "DecisionTreeClassifier"},
        "Gradient Boosting": {"module": "sklearn.ensemble", "class": "GradientBoostingClassifier"},
        "Neural Network": {"module": "sklearn.neural_network", "class": "MLPClassifier"}
    },
    "Regression": {
        "Linear Regression": {"module": "sklearn.linear_model", "class": "LinearRegression"},
        "Random Forest": {"module": "sklearn.ensemble", "class": "RandomForestRegressor"},
        "SVR": {"module": "sklearn.svm", "class": "SVR"},
        "KNN": {"module": "sklearn.neighbors", "class": "KNeighborsRegressor"},
        "Decision Tree": {"module": "sklearn.tree", "class": "DecisionTreeRegressor"},
        "Gradient Boosting": {"module": "sklearn.ensemble", "class": "GradientBoostingRegressor"},
        "Neural Network": {"module": "sklearn.neural_network", "class": "MLPRegressor"}
    },
    "Clustering": {
        "K-Means": {"module": "sklearn.cluster", "class": "KMeans"},
        "DBSCAN": {"module": "sklearn.cluster", "class": "DBSCAN"},
        "Hierarchical": {"module": "sklearn.cluster", "class": "AgglomerativeClustering"},
        "Gaussian Mixture": {"module": "sklearn.mixture", "class": "GaussianMixture"}
    }
}

# --- 하이퍼파라미터 정의 ---
HYPERPARAMETERS = {
    "LogisticRegression": {
        "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0},
        "max_iter": {"type": "int", "default": 100, "min": 50, "max": 1000}
    },
    "RandomForestClassifier": {
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        "max_depth": {"type": "int", "default": 10, "min": 1, "max": 50},
        "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20}
    },
    "RandomForestRegressor": {
        "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500},
        "max_depth": {"type": "int", "default": 10, "min": 1, "max": 50},
        "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 20}
    },
    "SVC": {
        "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0},
        "kernel": {"type": "combo", "default": "rbf", "options": ["linear", "poly", "rbf", "sigmoid"]}
    },
    "KMeans": {
        "n_clusters": {"type": "int", "default": 3, "min": 2, "max": 20},
        "n_init": {"type": "int", "default": 10, "min": 1, "max": 50}
    },
    "MLPClassifier": {
        "hidden_layer_sizes": {"type": "text", "default": "100,50", "hint": "e.g., 100,50"},
        "activation": {"type": "combo", "default": "relu", "options": ["relu", "tanh", "logistic"]},
        "learning_rate_init": {"type": "float", "default": 0.001, "min": 0.0001, "max": 0.1},
        "max_iter": {"type": "int", "default": 200, "min": 50, "max": 1000}
    }
}

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """Step 11 UI 생성"""
    global _module_main_callbacks, _util_funcs
    _module_main_callbacks = main_callbacks
    _util_funcs = main_callbacks.get('get_util_funcs', lambda: {})()
    
    main_callbacks['register_step_group_tag'](step_name, TAG_S11_GROUP)

    with dpg.group(tag=TAG_S11_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()
        
        # 전체 높이 설정
        total_height = 1000
        upper_height = int(total_height * 0.6)
        lower_height = int(total_height * 0.4)
        
        # 상부 시각화 영역
        with dpg.child_window(height=upper_height, border=True, tag=TAG_S11_UPPER_VIZ_WINDOW):
            dpg.add_text("Model Results", color=(255, 255, 0))
            dpg.add_separator()
            
            with dpg.tab_bar(tag=TAG_S11_VIZ_TAB_BAR):
                with dpg.tab(label="Guide"):
                    _create_guide_tab()
        
        # 하부 컨트롤 패널
        with dpg.child_window(height=lower_height, border=True, tag=TAG_S11_LOWER_CONTROL_PANEL):
            with dpg.group(horizontal=True):
                # 좌측: 데이터/모델 선택
                _create_left_panel()
                
                # 중앙: 변수 선택
                _create_center_panel()
                
                # 우측: 옵션 설정
                _create_right_panel()

    main_callbacks['register_module_updater'](step_name, update_ui)   
    # 2. UI가 처음 생성될 때 DataFrame 목록을 한번 로드합니다.
    update_ui()

def _create_guide_tab():
    """가이드 탭 생성"""
    dpg.add_text("ML/AI Modeling Guide", color=(255, 255, 0))
    dpg.add_separator()
    dpg.add_text("1. Select DataFrame and Model Type")
    dpg.add_text("2. Choose Algorithm")
    dpg.add_text("3. Select Features (and Target for supervised learning)")
    dpg.add_text("4. Configure Hyperparameters")
    dpg.add_text("5. Train Model")
    dpg.add_separator()
    dpg.add_text("Available Model Types:")
    dpg.add_text("• Classification: Predict categories")
    dpg.add_text("• Regression: Predict continuous values")
    dpg.add_text("• Clustering: Find groups in data")

def _create_left_panel():
    """좌측 패널: 데이터/모델 선택"""
    with dpg.child_window(width=250, border=True):
        dpg.add_text("Model Configuration", color=(255, 255, 0))
        dpg.add_separator()
        
        # DataFrame 선택
        dpg.add_text("Data Source:")
        dpg.add_combo(
            label="", 
            tag=TAG_S11_DF_SELECTOR,
            callback=_on_df_selected, 
            width=-1,
            items=[]  # 초기에는 빈 리스트, update_ui에서 채워짐
        )
        
        dpg.add_separator()
        
        # 모델 타입 선택
        dpg.add_text("Model Type:")
        dpg.add_radio_button(
            items=["Classification", "Regression", "Clustering"],
            default_value="Classification",
            tag=TAG_S11_MODEL_TYPE_SELECTOR,
            callback=_on_model_type_changed
        )
        
        dpg.add_separator()
        
        # 알고리즘 선택
        dpg.add_text("Algorithm:")
        dpg.add_combo(
            label="", 
            tag=TAG_S11_ALGORITHM_SELECTOR,
            items=list(ML_ALGORITHMS["Classification"].keys()),
            default_value="Random Forest",
            callback=_on_algorithm_changed,
            width=-1
        )

def _create_center_panel():
    """중앙 패널: 변수 선택"""
    with dpg.child_window(width=300, border=True):
        dpg.add_text("Variable Selection", color=(255, 255, 0))
        dpg.add_separator()
        
        # Features 선택
        dpg.add_text("Features (Multi-select):")
        with dpg.child_window(height=150, border=True, tag=TAG_S11_FEATURE_LIST):
            # 이 함수에서 동적으로 생성되므로 초기 내용은 비워둡니다.
            pass
        
        dpg.add_separator()
        
        # Target 선택 (supervised learning용)
        dpg.add_text("Target Variable:")

        dpg.add_combo(label="", tag=TAG_S11_TARGET_COMBO,
                     items=[""], default_value="", width=-1,
                     callback=_update_variable_lists) 

def _create_right_panel():
    """우측 패널: 옵션 설정"""
    with dpg.child_window(border=True):
        dpg.add_text("Training Options", color=(255, 255, 0))
        dpg.add_separator()
        
        # --- ▼▼▼▼▼ 추가된 부분 ▼▼▼▼▼ ---
        dpg.add_text("Categorical Feature Encoding:")
        dpg.add_radio_button(
            items=["Label Encoding", "One-Hot Encoding"],
            default_value="Label Encoding",
            tag="s11_encoding_method",
            horizontal=True
        )
        dpg.add_separator()
        # --- ▲▲▲▲▲ 추가된 부분 ▲▲▲▲▲ ---

        # Train/Test Split
        with dpg.group(horizontal=True):
            dpg.add_text("Test Size (%):")
            dpg.add_input_int(default_value=20, min_value=10, max_value=50,
                            tag="s11_test_size", width=100)
        
        with dpg.group(horizontal=True):
            dpg.add_text("Random State:")
            dpg.add_input_int(default_value=42, min_value=0, max_value=9999,
                            tag="s11_random_state", width=100)
        
        dpg.add_checkbox(label="Use Cross-Validation", tag="s11_use_cv")
        dpg.add_checkbox(label="Scale Features", default_value=True, tag="s11_scale_features")
        
        dpg.add_separator()
        
        # 하이퍼파라미터 영역
        dpg.add_text("Hyperparameters:", color=(255, 255, 0))
        with dpg.group(tag=TAG_S11_DYNAMIC_PARAMS_AREA):
            pass
        
        dpg.add_separator()
        
        # 실행 버튼
        dpg.add_button(label="Train Model", tag=TAG_S11_TRAIN_BUTTON,
                      callback=_train_model, width=-1, height=30)

def _on_df_selected(sender, app_data, user_data):
    """DataFrame 선택 시"""
    global _selected_df_name, _selected_features
    
    if not app_data:  # app_data가 None이면 리턴
        return
        
    _selected_df_name = app_data
    _selected_features.clear()
    _update_variable_lists()
    
    # 디버그 출력
    print(f"[Step 11] Selected DataFrame: {_selected_df_name}")

def _on_model_type_changed(sender, app_data, user_data):
    """모델 타입 변경 시"""
    model_type = app_data
    
    # 알고리즘 목록 업데이트
    if dpg.does_item_exist(TAG_S11_ALGORITHM_SELECTOR):
        algorithms = list(ML_ALGORITHMS[model_type].keys())
        dpg.configure_item(TAG_S11_ALGORITHM_SELECTOR, items=algorithms)
        if algorithms:
            dpg.set_value(TAG_S11_ALGORITHM_SELECTOR, algorithms[0])
    
    # Target 변수 표시 여부
    if dpg.does_item_exist(TAG_S11_TARGET_COMBO):
        show_target = model_type != "Clustering"
        dpg.configure_item(TAG_S11_TARGET_COMBO, show=show_target)
    
    _on_algorithm_changed(None, None, None)

    _update_variable_lists()


def _on_algorithm_changed(sender, app_data, user_data):
    """알고리즘 변경 시"""
    _create_hyperparameter_ui()

def _toggle_all_features(sender, app_data, user_data):
    """모든 features 선택/해제 (화면에 보이는 필터링된 변수만 대상)"""
    global _selected_features
    
    if not dpg.does_item_exist(TAG_S11_FEATURE_LIST):
        return

    feature_checkboxes = dpg.get_item_children(TAG_S11_FEATURE_LIST, 1)
    
    # --- ▼▼▼▼▼ 문제 3 해결을 위한 수정 ▼▼▼▼▼ ---
    if app_data:  # "Select All"이 체크된 경우
        for checkbox in feature_checkboxes:
            # "Select All" 자신은 제외하고, 변수 체크박스만 처리
            if dpg.get_item_label(checkbox) != "Select All":
                var_name = dpg.get_item_user_data(checkbox)
                if var_name and var_name not in _selected_features:
                    _selected_features.append(var_name)
    else:  # "Select All"이 체크 해제된 경우
        for checkbox in feature_checkboxes:
             if dpg.get_item_label(checkbox) != "Select All":
                var_name = dpg.get_item_user_data(checkbox)
                if var_name and var_name in _selected_features:
                    _selected_features.remove(var_name)
    
    _update_feature_checkboxes()

def _update_variable_lists():
    """변수 목록 업데이트 (필터링 로직 강화)"""
    if not _selected_df_name or _selected_df_name not in _available_dfs:
        if dpg.does_item_exist(TAG_S11_FEATURE_LIST):
            dpg.delete_item(TAG_S11_FEATURE_LIST, children_only=True)
        if dpg.does_item_exist(TAG_S11_TARGET_COMBO):
            dpg.configure_item(TAG_S11_TARGET_COMBO, items=[""])
        return

    df = _available_dfs[_selected_df_name]
    all_columns = list(df.columns)

    model_type = dpg.get_value(TAG_S11_MODEL_TYPE_SELECTOR) if dpg.does_item_exist(TAG_S11_MODEL_TYPE_SELECTOR) else "Classification"
    target_col = dpg.get_value(TAG_S11_TARGET_COMBO) if dpg.does_item_exist(TAG_S11_TARGET_COMBO) else None

    s1_analysis_types = {}
    if _module_main_callbacks:
        s1_analysis_types = _module_main_callbacks.get('get_column_analysis_types', lambda: {})()

    eligible_features = all_columns.copy()

    if target_col and target_col in eligible_features:
        eligible_features.remove(target_col)

    eligible_features = [
        var for var in eligible_features
        if s1_analysis_types.get(var) != "분석에서 제외 (Exclude)"
    ]

    if model_type == "Regression" or model_type == "Clustering":
        numeric_features = []
        for var in eligible_features:
            s1_type = s1_analysis_types.get(var)
            if (s1_type and "Numeric" in s1_type) or pd.api.types.is_numeric_dtype(df[var].dtype):
                numeric_features.append(var)
        eligible_features = numeric_features
    
    # --- ▼▼▼▼▼ 문제 1 해결을 위한 수정 ▼▼▼▼▼ ---
    elif model_type == "Classification":
        classification_features = []
        # 카테고리형 변수로 인정할 최대 고유값 개수 (예: 35개)
        MAX_UNIQUE_FOR_CAT = 35 

        for var in eligible_features:
            s1_type = s1_analysis_types.get(var, "")

            # ID, 긴 텍스트, 민감 정보는 항상 제외
            if "Text (ID/Code)" in s1_type or "Text (Long/Free)" in s1_type or "Potentially Sensitive" in s1_type:
                continue

            # 숫자형 변수는 포함
            if pd.api.types.is_numeric_dtype(df[var].dtype):
                classification_features.append(var)
            # 카테고리형 또는 object 타입이면서 고유값이 적은 경우에만 포함
            elif df[var].nunique() < MAX_UNIQUE_FOR_CAT:
                 classification_features.append(var)
                 
        eligible_features = classification_features
    # --- ▲▲▲▲▲ 수정 완료 ▲▲▲▲▲ ---

    if dpg.does_item_exist(TAG_S11_FEATURE_LIST):
        dpg.delete_item(TAG_S11_FEATURE_LIST, children_only=True)
        # "Select All" 체크박스를 그룹 맨 위에 추가
        dpg.add_checkbox(label="Select All", callback=_toggle_all_features, parent=TAG_S11_FEATURE_LIST)
        dpg.add_separator(parent=TAG_S11_FEATURE_LIST)
        
        for col in eligible_features:
            is_checked = col in _selected_features
            dpg.add_checkbox(label=col,
                           callback=lambda s, a, u: _on_feature_toggle(u, a),
                           user_data=col,
                           parent=TAG_S11_FEATURE_LIST,
                           default_value=is_checked)

    if dpg.does_item_exist(TAG_S11_TARGET_COMBO):
        dpg.configure_item(TAG_S11_TARGET_COMBO, items=[""] + all_columns)
        if target_col and target_col in all_columns:
            dpg.set_value(TAG_S11_TARGET_COMBO, target_col)

def _on_feature_toggle(col_name: str, is_checked: bool):
    """Feature 체크박스 토글"""
    global _selected_features
    
    if is_checked and col_name not in _selected_features:
        _selected_features.append(col_name)
    elif not is_checked and col_name in _selected_features:
        _selected_features.remove(col_name)

def _update_feature_checkboxes():
    """Feature 체크박스 상태 업데이트"""
    if not dpg.does_item_exist(TAG_S11_FEATURE_LIST):
        return
    
    children = dpg.get_item_children(TAG_S11_FEATURE_LIST, 1)
    for child in children:
        if dpg.get_item_type(child) == "mvAppItemType::mvCheckbox":
            col_name = dpg.get_item_user_data(child)
            if col_name:
                dpg.set_value(child, col_name in _selected_features)

def _create_hyperparameter_ui():
    """하이퍼파라미터 UI 생성"""
    if not dpg.does_item_exist(TAG_S11_DYNAMIC_PARAMS_AREA):
        return
    
    dpg.delete_item(TAG_S11_DYNAMIC_PARAMS_AREA, children_only=True)
    
    algorithm = dpg.get_value(TAG_S11_ALGORITHM_SELECTOR) if dpg.does_item_exist(TAG_S11_ALGORITHM_SELECTOR) else None
    if not algorithm:
        return
    
    # 알고리즘 클래스 이름 찾기
    model_type = dpg.get_value(TAG_S11_MODEL_TYPE_SELECTOR) if dpg.does_item_exist(TAG_S11_MODEL_TYPE_SELECTOR) else "Classification"
    class_name = ML_ALGORITHMS[model_type][algorithm]["class"]
    
    # 하이퍼파라미터 가져오기
    params = HYPERPARAMETERS.get(class_name, {})
    
    for param_name, param_info in params.items():
        with dpg.group(horizontal=True, parent=TAG_S11_DYNAMIC_PARAMS_AREA):
            dpg.add_text(f"{param_name}:")
            
            tag = f"s11_param_{param_name}"
            
            if param_info["type"] == "int":
                dpg.add_input_int(
                    default_value=param_info["default"],
                    min_value=param_info["min"],
                    max_value=param_info["max"],
                    tag=tag,
                    width=100
                )
            elif param_info["type"] == "float":
                dpg.add_input_float(
                    default_value=param_info["default"],
                    min_value=param_info["min"],
                    max_value=param_info["max"],
                    tag=tag,
                    width=100,
                    format="%.4f"
                )
            elif param_info["type"] == "combo":
                dpg.add_combo(
                    items=param_info["options"],
                    default_value=param_info["default"],
                    tag=tag,
                    width=100
                )
            elif param_info["type"] == "text":
                dpg.add_input_text(
                    default_value=param_info["default"],
                    hint=param_info.get("hint", ""),
                    tag=tag,
                    width=150
                )

def _train_model():
    """모델 학습"""
    try:
        # Progress modal 표시
        _util_funcs['show_dpg_progress_modal']("Training Model", "Preparing data...")
        
        # 데이터 확인
        if not _selected_df_name:
            raise ValueError("Please select a DataFrame")
            
        if _selected_df_name not in _available_dfs:
            raise ValueError(f"Selected DataFrame '{_selected_df_name}' not found")
            
        if not _selected_features:
            raise ValueError("Please select at least one feature")
        
        df = _available_dfs[_selected_df_name]
        model_type = dpg.get_value(TAG_S11_MODEL_TYPE_SELECTOR)
        algorithm = dpg.get_value(TAG_S11_ALGORITHM_SELECTOR)
        
        # 디버그 출력
        print(f"[Step 11] Training with:")
        print(f"  - DataFrame: {_selected_df_name} (shape: {df.shape})")
        print(f"  - Features: {_selected_features}")
        print(f"  - Model Type: {model_type}")
        print(f"  - Algorithm: {algorithm}")
        
        # Features 추출
        X = df[_selected_features].copy()
        
        encoding_method = dpg.get_value("s11_encoding_method") if dpg.does_item_exist("s11_encoding_method") else "Label Encoding"
        encoders = {} # LabelEncoder 정보를 저장하기 위한 딕셔너리

        if encoding_method == "One-Hot Encoding":
            # 원-핫 인코딩 적용
            print("[Step 11] Applying One-Hot Encoding...")
            # 인코딩할 범주형 변수 식별 (object 타입)
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            if not categorical_cols.empty:
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=float)
        
        else: # Label Encoding (기본값)
            # 레이블 인코딩 적용 (기존 로직)
            print("[Step 11] Applying Label Encoding...")
            for col in X.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna('missing'))
                encoders[col] = le
        
        # Target 추출 (supervised learning)
        y = None
        if model_type != "Clustering":
            target_col = dpg.get_value(TAG_S11_TARGET_COMBO)
            if not target_col:
                raise ValueError("Please select target variable")
            
            y = df[target_col].copy()
            if y.dtype == 'object' or model_type == "Classification":
                le = LabelEncoder()
                y = le.fit_transform(y)
                encoders['target'] = le
        
        # 스케일링
        scaler = None
        if dpg.get_value("s11_scale_features"):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # 모델 생성
        model_info = ML_ALGORITHMS[model_type][algorithm]
        module = __import__(model_info["module"], fromlist=[model_info["class"]])
        model_class = getattr(module, model_info["class"])
        
        # 하이퍼파라미터 수집
        hyperparams = _get_hyperparameters(model_info["class"])
        model = model_class(**hyperparams)
        
        # 학습
        _util_funcs['show_dpg_progress_modal']("Training Model", "Training in progress...")
        
        if model_type == "Clustering":
            # 클러스터링
            labels = model.fit_predict(X_scaled)
            results = _evaluate_clustering(model, X_scaled, labels)
        else:
            # 분류/회귀
            test_size = dpg.get_value("s11_test_size") / 100.0
            random_state = dpg.get_value("s11_random_state")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )
            
            model.fit(X_train, y_train)
            
            if dpg.get_value("s11_use_cv"):
                cv_scores = cross_val_score(model, X_scaled, y, cv=5)
            else:
                cv_scores = None
            
            results = _evaluate_model(model, X_train, X_test, y_train, y_test, 
                                    model_type, cv_scores, encoders)
        
        # 결과 저장
        global _model_counter
        _model_counter += 1
        model_name = f"{algorithm}_{_model_counter}"
        
        _trained_models[model_name] = {
            'model': model,
            'scaler': scaler,
            'encoders': encoders,
            'features': _selected_features,
            'results': results,
            'type': model_type,
            'algorithm': algorithm,
            'X': X,
            'y': y
        }
        
        # 결과 표시
        _create_results_tab(model_name, results)
        
        _util_funcs['_show_simple_modal_message']("Success", 
            f"Model '{model_name}' trained successfully!")
        
    except Exception as e:
        _util_funcs['_show_simple_modal_message']("Training Error", str(e))
        traceback.print_exc()
    
    finally:
        _util_funcs['hide_dpg_progress_modal']()

def _get_hyperparameters(class_name: str) -> dict:
    """UI에서 하이퍼파라미터 값 수집"""
    params = {}
    param_defs = HYPERPARAMETERS.get(class_name, {})
    
    for param_name, param_info in param_defs.items():
        tag = f"s11_param_{param_name}"
        if dpg.does_item_exist(tag):
            value = dpg.get_value(tag)
            
            # 특수 처리
            if param_name == "hidden_layer_sizes" and isinstance(value, str):
                # "100,50" -> (100, 50)
                try:
                    value = tuple(map(int, value.split(',')))
                except:
                    value = (100,)
            
            params[param_name] = value
    
    return params

def _evaluate_model(model, X_train, X_test, y_train, y_test, model_type, cv_scores, encoders):
    """모델 평가"""
    results = {
        'model_type': model_type,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    if model_type == "Classification":
        # 분류 평가
        results['train_accuracy'] = accuracy_score(y_train, y_pred_train)
        results['test_accuracy'] = accuracy_score(y_test, y_pred_test)
        
        # Precision, Recall, F1
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')
        results['precision'] = prec
        results['recall'] = rec
        results['f1_score'] = f1
        
        # 혼동 행렬
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred_test)
        
        # 클래스 이름
        if 'target' in encoders:
            results['class_names'] = encoders['target'].classes_
        
        # 확률 예측 (가능한 경우)
        if hasattr(model, 'predict_proba'):
            results['y_prob_test'] = model.predict_proba(X_test)
    
    else:  # Regression
        # 회귀 평가
        results['train_r2'] = r2_score(y_train, y_pred_train)
        results['test_r2'] = r2_score(y_test, y_pred_test)
        results['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
        results['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred_test))
        results['train_mae'] = mean_absolute_error(y_train, y_pred_train)
        results['test_mae'] = mean_absolute_error(y_test, y_pred_test)
    
    # CV 점수
    if cv_scores is not None:
        results['cv_scores'] = cv_scores
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
    
    # 예측값 저장
    results['y_test'] = y_test
    results['y_pred_test'] = y_pred_test
    
    # Feature importance (가능한 경우)
    if hasattr(model, 'feature_importances_'):
        results['feature_importances'] = model.feature_importances_
        results['feature_names'] = _selected_features
    
    return results

def _evaluate_clustering(model, X, labels):
    """클러스터링 평가"""
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    results = {
        'model_type': 'Clustering',
        'n_samples': len(X),
        'labels': labels
    }
    
    # 클러스터 개수
    n_clusters = len(np.unique(labels[labels != -1]))  # -1은 노이즈
    results['n_clusters'] = n_clusters
    
    if n_clusters > 1:
        # 실루엣 점수
        mask = labels != -1  # 노이즈 제외
        if mask.sum() > 0:
            results['silhouette_score'] = silhouette_score(X[mask], labels[mask])
            results['davies_bouldin_score'] = davies_bouldin_score(X[mask], labels[mask])
    
    # 클러스터별 크기
    unique_labels, counts = np.unique(labels, return_counts=True)
    results['cluster_sizes'] = dict(zip(unique_labels, counts))
    
    # 클러스터 중심 (K-Means의 경우)
    if hasattr(model, 'cluster_centers_'):
        results['cluster_centers'] = model.cluster_centers_
    
    return results

def _create_results_tab(model_name: str, results: Dict[str, Any]):
    """결과 탭 생성"""
    if not dpg.does_item_exist(TAG_S11_VIZ_TAB_BAR):
        return
    
    with dpg.tab(label=model_name, parent=TAG_S11_VIZ_TAB_BAR, closable=True):
        with dpg.child_window(border=False):
            # 모델 정보
            dpg.add_text(f"Model: {model_name}", color=(255, 255, 0))
            dpg.add_separator()
            
            # 성능 지표
            _create_performance_metrics(results)
            dpg.add_separator()
            
            # 시각화
            _create_visualizations(results)
            
            # Export 버튼
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save Model", 
                             callback=lambda: _save_model(model_name))
                dpg.add_button(label="Export Results", 
                             callback=lambda: _export_results(model_name))

def _create_performance_metrics(results: Dict[str, Any]):
    """성능 지표 표시"""
    model_type = results['model_type']
    
    if model_type == "Classification":
        dpg.add_text("Classification Metrics:", color=(255, 255, 0))
        dpg.add_text(f"Train Accuracy: {results['train_accuracy']:.4f}")
        dpg.add_text(f"Test Accuracy: {results['test_accuracy']:.4f}")
        dpg.add_text(f"Precision: {results['precision']:.4f}")
        dpg.add_text(f"Recall: {results['recall']:.4f}")
        dpg.add_text(f"F1-Score: {results['f1_score']:.4f}")
        
        if 'cv_mean' in results:
            dpg.add_text(f"CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
    
    elif model_type == "Regression":
        dpg.add_text("Regression Metrics:", color=(255, 255, 0))
        dpg.add_text(f"Train R²: {results['train_r2']:.4f}")
        dpg.add_text(f"Test R²: {results['test_r2']:.4f}")
        dpg.add_text(f"Train RMSE: {results['train_rmse']:.4f}")
        dpg.add_text(f"Test RMSE: {results['test_rmse']:.4f}")
        dpg.add_text(f"Train MAE: {results['train_mae']:.4f}")
        dpg.add_text(f"Test MAE: {results['test_mae']:.4f}")
        
        if 'cv_mean' in results:
            dpg.add_text(f"CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
    
    elif model_type == "Clustering":
        dpg.add_text("Clustering Metrics:", color=(255, 255, 0))
        dpg.add_text(f"Number of Clusters: {results['n_clusters']}")
        
        if 'silhouette_score' in results:
            dpg.add_text(f"Silhouette Score: {results['silhouette_score']:.4f}")
        if 'davies_bouldin_score' in results:
            dpg.add_text(f"Davies-Bouldin Score: {results['davies_bouldin_score']:.4f}")
        
        dpg.add_text("Cluster Sizes:")
        for label, size in results['cluster_sizes'].items():
            cluster_name = f"Cluster {label}" if label != -1 else "Noise"
            dpg.add_text(f"  {cluster_name}: {size}")

def _create_visualizations(results: Dict[str, Any]):
    """시각화 생성"""
    model_type = results['model_type']
    plot_func = _util_funcs.get('plot_to_dpg_texture')
    
    if not plot_func:
        return
    
    if model_type == "Classification":
        # 혼동 행렬
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        _add_plot_to_ui(fig, plot_func, "Confusion Matrix")
        
        # Feature Importance
        if 'feature_importances' in results:
            fig, ax = plt.subplots(figsize=(10, 6))
            features = results['feature_names']
            importances = results['feature_importances']
            indices = np.argsort(importances)[::-1][:10]  # Top 10
            
            ax.bar(range(len(indices)), importances[indices])
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels([features[i] for i in indices], rotation=45)
            ax.set_title('Top 10 Feature Importances')
            
            _add_plot_to_ui(fig, plot_func, "Feature Importance")
    
    elif model_type == "Regression":
        # 실제값 vs 예측값
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(results['y_test'], results['y_pred_test'], alpha=0.6)
        ax.plot([results['y_test'].min(), results['y_test'].max()], 
               [results['y_test'].min(), results['y_test'].max()], 
               'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        
        _add_plot_to_ui(fig, plot_func, "Actual vs Predicted")
        
        # 잔차 플롯
        fig, ax = plt.subplots(figsize=(8, 6))
        residuals = results['y_test'] - results['y_pred_test']
        ax.scatter(results['y_pred_test'], residuals, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        
        _add_plot_to_ui(fig, plot_func, "Residual Plot")
    
    elif model_type == "Clustering":
        # 클러스터 분포 (2D 투영)
        from sklearn.decomposition import PCA
        
        # PCA로 2D 투영
        X = _trained_models[list(_trained_models.keys())[-1]]['X']
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
        else:
            X_2d = X.values
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], 
                           c=results['labels'], cmap='viridis', 
                           alpha=0.6)
        plt.colorbar(scatter, ax=ax)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_title('Cluster Visualization (2D PCA)')
        
        _add_plot_to_ui(fig, plot_func, "Cluster Visualization")

def _add_plot_to_ui(fig, plot_func, title: str):
    """플롯을 UI에 추가"""
    tex_tag, w, h, img_bytes = plot_func(fig)
    
    if tex_tag:
        dpg.add_text(title)
        dpg.add_image(tex_tag, width=w, height=h)
        
        # AI 분석 버튼
        if img_bytes and _module_main_callbacks:
            import utils
            ai_button_tag = dpg.generate_uuid()
            action_callback = lambda: utils.confirm_and_run_ai_analysis(
                img_bytes, f"ML Model - {title}", ai_button_tag, _module_main_callbacks
            )
            dpg.add_button(label=f"💡 Analyze {title}", tag=ai_button_tag,
                         callback=action_callback)
    
    plt.close(fig)
    dpg.add_separator()

def _save_model(model_name: str):
    """모델 저장"""
    import pickle
    import datetime
    
    try:
        model_data = _trained_models[model_name]
        filename = f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        # 파일 다이얼로그로 저장 위치 선택
        with dpg.file_dialog(
            directory_selector=False, show=True,
            callback=lambda s, a: _do_save_model(a['file_path_name'], model_data),
            default_filename=filename, width=700, height=400, modal=True
        ):
            dpg.add_file_extension(".pkl", color=(0, 255, 0, 255))
            
    except Exception as e:
        _util_funcs['_show_simple_modal_message']("Save Error", str(e))

def _do_save_model(filepath: str, model_data: dict):
    """실제 모델 저장"""
    import pickle
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        _util_funcs['_show_simple_modal_message']("Success", 
            f"Model saved to:\n{filepath}")
    except Exception as e:
        _util_funcs['_show_simple_modal_message']("Save Error", str(e))

def _export_results(model_name: str):
    """결과 내보내기"""
    # Excel로 내보내기 구현
    _util_funcs['_show_simple_modal_message']("Info", 
        "Export functionality coming soon!")

def update_ui():
    """UI 업데이트"""
    global _available_dfs
    
    if not _module_main_callbacks:
        return
    
    # DataFrame 목록 가져오기 - Step 10과 동일한 패턴
    all_dfs_from_main = _module_main_callbacks.get('get_all_available_dfs', lambda: {})()
    
    # Original Data 제외 (Step 10과 동일)
    _available_dfs = {k: v for k, v in all_dfs_from_main.items() if k != '0. Original Data'}
    
    if dpg.does_item_exist(TAG_S11_DF_SELECTOR):
        df_names = list(_available_dfs.keys())
        current = dpg.get_value(TAG_S11_DF_SELECTOR)
        
        dpg.configure_item(TAG_S11_DF_SELECTOR, items=df_names)
        
        # 현재 선택이 유효하지 않으면 첫 번째 항목 선택
        if current not in df_names:
            new_selection = df_names[0] if df_names else ""
            dpg.set_value(TAG_S11_DF_SELECTOR, new_selection)
            if new_selection:
                _on_df_selected(None, new_selection, None)

def reset_state():
    """상태 초기화"""
    global _selected_df_name, _selected_features, _trained_models, _model_counter
    
    _selected_df_name = ""
    _selected_features.clear()
    _trained_models.clear()
    _model_counter = 0
    
    # UI 초기화
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_S11_VIZ_TAB_BAR):
            tabs = dpg.get_item_children(TAG_S11_VIZ_TAB_BAR, 1)
            for tab in tabs[1:]:  # Guide 탭 제외
                if dpg.does_item_exist(tab):
                    dpg.delete_item(tab)
        
        update_ui()