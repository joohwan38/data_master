# step_06_standardization.py
import dearpygui.dearpygui as dpg
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Module-specific state
_main_app_callbacks: Optional[Dict[str, Any]] = None
_current_df_input: Optional[pd.DataFrame] = None
_selected_scaler_method: str = "StandardScaler"
_texture_tags: List[str] = []

# UI Tags
TAG_S6_MAIN_GROUP = "step6_main_group"
TAG_S6_SCALER_RADIO = "step6_scaler_method_radio"
TAG_S6_APPLY_BUTTON = "step6_apply_button"
TAG_S6_VISUALIZATION_GROUP = "step6_visualization_group"
TAG_S6_HELP_BUTTON = "step6_help_button"
TAG_S6_HELP_MODAL = "step6_help_modal"

# --- 도움말 팝업 내용 ---
HELP_TEXT_CONTENT = """
## 언제 표준화를 해야 할까요?

특성(변수)의 스케일(값의 범위)이 서로 크게 다를 때 사용하는 것이 좋습니다. 많은 머신러닝 알고리즘은 스케일에 영향을 받기 때문입니다.

- **거리 기반 알고리즘:** KNN, K-Means, SVM 등은 데이터 포인트 간의 거리를 계산하므로 스케일이 다르면 특정 변수의 영향력이 비정상적으로 커집니다.
- **경사 하강법 기반 알고리즘:** 선형 회귀, 로지스틱 회귀, 신경망 등은 최적의 가중치를 찾는 과정에서 스케일이 맞지 않으면 학습이 불안정하거나 느려질 수 있습니다.
- **차원 축소 (PCA):** 분산이 큰 변수가 주성분을 과도하게 설명하는 것을 방지합니다.
- **정규화 (L1, L2):** 모든 변수에 공평한 패널티를 부여하기 위해 필요합니다.

---

## 언제 표준화가 필요 없거나 피해야 할까요?

- **트리 기반 모델:** Decision Tree, Random Forest, Gradient Boosting(XGBoost, LightGBM) 등은 각 변수를 독립적으로 보고 분기점을 찾으므로 스케일의 영향을 받지 않습니다.
- **변수의 스케일 자체가 중요한 정보일 경우:** 예를 들어, '거래 횟수'와 '거래 금액' 변수가 있을 때, 스케일을 맞추면 모델이 원래의 의미를 잃을 수 있습니다.

---

## 스케일러(Scaler) 별 특징

### StandardScaler (기본값)
- **특징:** 모든 데이터의 평균을 0, 표준편차를 1로 맞춥니다. (Z-score 정규화)
- **장점:** 가장 널리 사용되는 기본적인 스케일러입니다.
- **단점:** 이상치(outlier)에 매우 민감합니다. 이상치가 있으면 평균과 표준편차가 크게 왜곡될 수 있습니다.
- **사용 시기:** 데이터가 정규분포(가우시안 분포)에 가깝고, 이상치가 적을 때 효과적입니다.

### MinMaxScaler
- **특징:** 모든 데이터의 값을 0과 1 사이로 압축합니다. (최소-최대 정규화)
- **장점:** 모든 값의 범위를 명확하게 제한할 수 있습니다. 신경망 등에서 유용할 수 있습니다.
- **단점:** 이상치에 매우 민감합니다. 아주 크거나 작은 이상치 하나가 다른 모든 값의 범위를 왜곡시킬 수 있습니다.
- **사용 시기:** 이상치를 미리 처리했으며, 데이터의 범위를 [0, 1]로 제한해야 할 때 사용합니다. 이미지 데이터(0~255 픽셀) 처리 시 자주 사용됩니다.

### RobustScaler
- **특징:** 평균/표준편차 대신, 중앙값(median)과 사분위 범위(IQR)를 사용하여 스케일링합니다.
- **장점:** 이상치의 영향을 거의 받지 않습니다. 데이터에 이상치가 많을 때 가장 안정적입니다.
- **단점:** 데이터를 0 중심으로 맞추거나 특정 범위로 제한하지는 않습니다.
- **사용 시기:** 데이터에 제거하기 어려운 이상치가 포함되어 있을 때, 이상치의 영향을 최소화하며 스케일링하고 싶을 때 사용합니다.
"""


def _log(message: str):
    """Module-specific logger."""
    print(f"[Step6 Standardization] {message}")

def _show_help_modal(sender, app_data, user_data):
    """Displays the help modal window with detailed explanations."""
    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_S6_HELP_MODAL):
        dpg.delete_item(TAG_S6_HELP_MODAL)

    vp_w = dpg.get_viewport_width() if dpg.is_dearpygui_running() else 1024
    vp_h = dpg.get_viewport_height() if dpg.is_dearpygui_running() else 768
    modal_width = 650
    # Calculate height dynamically or set a max
    modal_height = min(650, vp_h - 100)

    with dpg.window(label="💡 표준화(Standardization) 도움말", modal=True, show=True, tag=TAG_S6_HELP_MODAL,
                    width=modal_width, height=modal_height, pos=[(vp_w - modal_width) // 2, (vp_h - modal_height) // 2]):
        
        with dpg.child_window(border=False, height = -50):
            lines = HELP_TEXT_CONTENT.strip().split('\n')
            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    dpg.add_spacer(height=5)
                    continue

                if stripped_line.startswith('## '):
                    dpg.add_text(stripped_line[3:], color=[255, 255, 0])
                    dpg.add_separator()
                elif stripped_line.startswith('### '):
                    dpg.add_spacer(height=10)
                    dpg.add_text(stripped_line[4:], color=[220, 220, 220])
                elif stripped_line.startswith('- '):
                    dpg.add_text(f"  • {stripped_line[2:]}", wrap=modal_width - 50)
                elif stripped_line == '---':
                    dpg.add_separator()
                else:
                    dpg.add_text(stripped_line, wrap=modal_width - 50)
        
        dpg.add_separator()
        dpg.add_button(label="닫기", width=-1, height=30, callback=lambda: dpg.configure_item(TAG_S6_HELP_MODAL, show=False))


def _get_util_func(func_name: str) -> Optional[callable]:
    """Helper to safely get a utility function from main_app_callbacks."""
    if _main_app_callbacks:
        util_funcs = _main_app_callbacks.get('get_util_funcs', lambda: {})()
        return util_funcs.get(func_name)
    return None

def _clear_visualizations():
    """Clears any existing plot textures and the visualization group."""
    global _texture_tags
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_S6_VISUALIZATION_GROUP):
            dpg.delete_item(TAG_S6_VISUALIZATION_GROUP, children_only=True)
            dpg.add_text("Click 'Apply' to see visualizations.", parent=TAG_S6_VISUALIZATION_GROUP)

        for tag in _texture_tags:
            if dpg.does_item_exist(tag):
                try:
                    dpg.delete_item(tag)
                except Exception as e:
                    _log(f"Error deleting texture {tag}: {e}")
    _texture_tags.clear()

def _on_apply_clicked(sender, app_data, user_data):
    """Callback for the 'Apply Standardization' button."""
    _log("Apply button clicked.")
    if _current_df_input is None:
        modal_func = _get_util_func('_show_simple_modal_message')
        if modal_func:
            modal_func("Error", "No input data available for standardization.")
        return

    _clear_visualizations()

    df_to_process = _current_df_input.copy()
    numeric_cols = df_to_process.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        modal_func = _get_util_func('_show_simple_modal_message')
        if modal_func:
            modal_func("Info", "No numeric columns found to apply standardization.")
        if _main_app_callbacks:
            _main_app_callbacks['step6_standardization_complete'](df_to_process)
        return

    scaler = None
    if _selected_scaler_method == "StandardScaler":
        scaler = StandardScaler()
    elif _selected_scaler_method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif _selected_scaler_method == "RobustScaler":
        scaler = RobustScaler(quantile_range=(25.0, 75.0))

    if scaler:
        _log(f"Applying {_selected_scaler_method} to {len(numeric_cols)} columns.")
        try:
            df_to_process[numeric_cols] = scaler.fit_transform(df_to_process[numeric_cols])
            _log("Scaling successful.")
        except Exception as e:
            _log(f"Error during scaling: {e}")
            modal_func = _get_util_func('_show_simple_modal_message')
            if modal_func:
                modal_func("Scaling Error", f"An error occurred during scaling: {e}")
            if _main_app_callbacks:
                _main_app_callbacks['step6_standardization_complete'](None)
            return

    _generate_comparison_plots(_current_df_input, df_to_process, numeric_cols)

    if _main_app_callbacks:
        _main_app_callbacks['step6_standardization_complete'](df_to_process)
        _log("Processing complete, callback sent to main_app.")

def _generate_comparison_plots(df_before: pd.DataFrame, df_after: pd.DataFrame, numeric_cols: List[str]):
    """Creates and displays side-by-side distribution plots for a sample of columns."""
    plot_func = _get_util_func('plot_to_dpg_texture')
    if not plot_func:
        _log("Plotting utility not found.")
        return

    eligible_cols = [col for col in numeric_cols if df_before[col].nunique() > 10]
    cols_to_plot = eligible_cols[:10]

    if not cols_to_plot:
        _log("No columns with > 10 unique values to plot.")
        if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_S6_VISUALIZATION_GROUP):
            dpg.delete_item(TAG_S6_VISUALIZATION_GROUP, children_only=True)
            dpg.add_text("No columns with > 10 unique values to visualize.", parent=TAG_S6_VISUALIZATION_GROUP)
        return
        
    _log(f"Generating comparison plots for ({len(cols_to_plot)} columns): {cols_to_plot}")

    if dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_S6_VISUALIZATION_GROUP):
        dpg.delete_item(TAG_S6_VISUALIZATION_GROUP, children_only=True)

    for col in cols_to_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=100)
        
        sns.histplot(df_before[col].dropna(), kde=True, ax=ax1, color='skyblue', bins=30)
        ax1.set_title(f'Before: {col}', fontsize=10)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        ax1.tick_params(axis='y', labelsize=8)
        ax1.set_xlabel('')
        
        sns.histplot(df_after[col].dropna(), kde=True, ax=ax2, color='salmon', bins=30)
        ax2.set_title(f'After: {col} ({_selected_scaler_method})', fontsize=10)
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.tick_params(axis='y', labelsize=8)
        ax2.set_xlabel('')

        plt.tight_layout()
        
        try:
            tex_tag, w, h, _ = plot_func(fig)
            if tex_tag and dpg.is_dearpygui_running() and dpg.does_item_exist(TAG_S6_VISUALIZATION_GROUP):
                dpg.add_image(tex_tag, width=w, height=h, parent=TAG_S6_VISUALIZATION_GROUP)
                dpg.add_separator(parent=TAG_S6_VISUALIZATION_GROUP)
                _texture_tags.append(tex_tag)
        except Exception as e:
            _log(f"Error creating plot texture for {col}: {e}")
        finally:
            plt.close(fig)

def _on_scaler_changed(sender, app_data, user_data):
    """Callback for when the scaler radio button is changed."""
    global _selected_scaler_method
    _selected_scaler_method = app_data
    _log(f"Scaler method changed to: {_selected_scaler_method}")
    _clear_visualizations()

def create_ui(step_name: str, parent_tag: str, main_app_callbacks: Dict[str, Any]):
    """Creates the UI for the Standardization step."""
    global _main_app_callbacks
    _main_app_callbacks = main_app_callbacks
    
    group_tag = f"step_group_{step_name.replace(' ', '_').replace('(', '').replace(')', '')}"
    _main_app_callbacks['register_step_group_tag'](step_name, group_tag)
    _main_app_callbacks['register_module_updater'](step_name, update_ui)
    
    with dpg.group(tag=group_tag, parent=parent_tag, show=False) as main_group:
        with dpg.group(horizontal=True):
            dpg.add_text(f"--- {step_name} ---", color=[255, 255, 0])
            dpg.add_spacer()
            dpg.add_button(label="💡 도움말", tag=TAG_S6_HELP_BUTTON, callback=_show_help_modal, small=True)

        dpg.add_separator()
        dpg.add_text("Select a method to standardize (scale) numeric features in the dataset.")
        dpg.add_radio_button(
            items=["StandardScaler", "MinMaxScaler", "RobustScaler"],
            tag=TAG_S6_SCALER_RADIO,
            default_value=_selected_scaler_method,
            callback=_on_scaler_changed,
            horizontal=True
        )
        dpg.add_button(
            label="Apply Standardization and Generate Previews",
            tag=TAG_S6_APPLY_BUTTON,
            callback=_on_apply_clicked,
            width=-1, height=30
        )
        dpg.add_separator()
        dpg.add_text("Before vs. After Comparison", color=[255, 255, 0])
        with dpg.child_window(tag=TAG_S6_VISUALIZATION_GROUP, border=True, height=-1):
            dpg.add_text("Click 'Apply' to see visualizations.")

def update_ui(df_input: Optional[pd.DataFrame], main_app_callbacks: Dict[str, Any]):
    """Updates the UI based on the current DataFrame."""
    global _current_df_input, _main_app_callbacks
    _main_app_callbacks = main_app_callbacks
    _current_df_input = df_input
    
    if not dpg.is_dearpygui_running():
        return

    is_enabled = df_input is not None and not df_input.empty
    if dpg.does_item_exist(TAG_S6_APPLY_BUTTON):
        dpg.configure_item(TAG_S6_APPLY_BUTTON, enabled=is_enabled)
    
    if not is_enabled:
        _clear_visualizations()

def reset_step6_state():
    """Resets the state of the standardization module."""
    global _selected_scaler_method
    _selected_scaler_method = "StandardScaler"
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_S6_SCALER_RADIO):
            dpg.set_value(TAG_S6_SCALER_RADIO, _selected_scaler_method)
        _clear_visualizations()
    _log("State reset.")

def get_step6_settings_for_saving() -> dict:
    """Returns the current settings for saving to a file."""
    return {
        "selected_scaler_method": _selected_scaler_method
    }

def apply_step6_settings_and_process(df_input: pd.DataFrame, settings: dict, main_app_callbacks: Dict[str, Any]):
    """Applies loaded settings and processes the data."""
    global _selected_scaler_method, _main_app_callbacks, _current_df_input
    _main_app_callbacks = main_app_callbacks
    _current_df_input = df_input
    
    _selected_scaler_method = settings.get("selected_scaler_method", "StandardScaler")
    _log(f"Applied loaded settings: scaler is now {_selected_scaler_method}")

    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_S6_SCALER_RADIO):
            dpg.set_value(TAG_S6_SCALER_RADIO, _selected_scaler_method)

    _on_apply_clicked(None, None, None)