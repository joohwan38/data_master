# step_04_missing_values.py
import dearpygui.dearpygui as dpg
import pandas as pd
from typing import Dict, Any, Optional

# --- 고유 태그 정의 ---
TAG_MV_STEP_GROUP = "step4_missing_values_group"
TAG_MV_INFO_TABLE_BEFORE = "step4_mv_info_table_before"  # 처리 전 결측치 현황 테이블
TAG_MV_METHOD_SELECTION_TABLE = "step4_mv_method_selection_table" # 변수별 처리 방법 선택 테이블
TAG_MV_EXECUTE_BUTTON = "step4_mv_execute_button"       # 처리 실행 버튼
TAG_MV_RESULTS_GROUP = "step4_mv_results_group"         # 결과 표시 그룹
TAG_MV_INFO_TABLE_AFTER = "step4_mv_info_table_after"   # 처리 후 결측치 현황 테이블
TAG_MV_LOG_TEXT_AREA = "step4_mv_log_text_area"         # 처리 로그 표시 영역 (Child Window)
TAG_MV_LOG_TEXT = "step4_mv_log_text"                   # 실제 로그 텍스트

# --- 모듈 상태 변수 ---
_main_app_callbacks: Optional[Dict[str, Any]] = None
_util_funcs: Optional[Dict[str, Any]] = None
_current_df_for_this_step: Optional[pd.DataFrame] = None # 이 스텝에서 현재 보여주고 있는 DataFrame

# 사용자가 선택한 처리 방법을 저장 (컬럼명: 방법)
_imputation_method_selections: Dict[str, str] = {}
# "특정 값으로 대체" 선택 시 사용자가 입력한 값을 저장 (컬럼명: 값)
_custom_fill_values: Dict[str, str] = {}


def _on_imputation_method_change(sender, app_data_method: str, user_data_col_name: str):
    """콤보박스에서 처리 방법 변경 시 호출되는 콜백"""
    _imputation_method_selections[user_data_col_name] = app_data_method
    
    # "특정 값으로 대체" 선택 시 해당 컬럼의 "특정 값" 입력 필드 표시/숨김 처리
    custom_fill_input_tag = f"custom_fill_input_{user_data_col_name}"
    if dpg.does_item_exist(custom_fill_input_tag):
        dpg.configure_item(custom_fill_input_tag, show=(app_data_method == "특정 값으로 대체"))
    
    print(f"결측치 처리: 컬럼 '{user_data_col_name}'의 처리 방법이 '{app_data_method}'(으)로 변경됨")

def _on_custom_fill_value_change(sender, app_data_fill_value: str, user_data_col_name: str):
    """ "특정 값" 입력 필드 값 변경 시 호출되는 콜백 """
    _custom_fill_values[user_data_col_name] = app_data_fill_value
    print(f"결측치 처리: 컬럼 '{user_data_col_name}'의 특정 대체 값이 '{app_data_fill_value}'(으)로 변경됨")

def _execute_imputation_placeholder():
    """ "결측치 처리 실행" 버튼 콜백 (현재는 플레이스홀더) """
    global _main_app_callbacks, _current_df_for_this_step, _util_funcs
    if not _main_app_callbacks or _current_df_for_this_step is None:
        msg = "데이터가 로드되지 않았거나 내부 설정 오류입니다."
        if _util_funcs and '_show_simple_modal_message' in _util_funcs:
            _util_funcs['_show_simple_modal_message']("처리 오류", msg)
        else:
            print(f"ERROR: {msg}")
        return

    log_messages = ["--- 결측치 처리 실행 로그 (1-2 단계에서 실제 로직 구현 예정) ---"]
    processed_df_placeholder = _current_df_for_this_step.copy() # 실제로는 이 DataFrame이 변경됨

    for col_name, method in _imputation_method_selections.items():
        if col_name not in processed_df_placeholder.columns:
            continue
        
        log_messages.append(f"컬럼 [{col_name}]: 선택된 처리 방법 = '{method}'")
        if method == "특정 값으로 대체":
            fill_val = _custom_fill_values.get(col_name, "")
            log_messages.append(f"  └─ 특정 대체 값: '{fill_val}' (실제 변환 로직은 추후 구현)")
        # 여기에 다른 처리 방법들에 대한 로그도 추가될 수 있습니다.
    
    # 로그 업데이트
    if dpg.does_item_exist(TAG_MV_LOG_TEXT):
        dpg.set_value(TAG_MV_LOG_TEXT, "\n".join(log_messages) if log_messages else "선택된 처리 작업이 없습니다.")

    # 처리 후 테이블 업데이트 (플레이스홀더 - 실제로는 처리된 df_after_imputation 사용)
    if dpg.does_item_exist(TAG_MV_INFO_TABLE_AFTER) and _util_funcs and 'create_table_with_data' in _util_funcs:
        dpg.delete_item(TAG_MV_INFO_TABLE_AFTER, children_only=True) # 기존 내용 삭제
        
        # 지금은 원본 데이터의 결측치 정보를 다시 보여주지만, 1-2 단계에서는 처리 후의 정보를 보여줍니다.
        missing_summary_after_placeholder = pd.DataFrame({
            '컬럼명': processed_df_placeholder.columns,
            '처리 후 결측 수': processed_df_placeholder.isnull().sum().values,
            '처리 후 결측 비율 (%)': (processed_df_placeholder.isnull().mean() * 100).round(2).values
        })
        _util_funcs['create_table_with_data'](TAG_MV_INFO_TABLE_AFTER, missing_summary_after_placeholder, parent_df_for_widths=missing_summary_after_placeholder)

    if _util_funcs and '_show_simple_modal_message' in _util_funcs:
        _util_funcs['_show_simple_modal_message']("알림", "결측치 처리 실행 로직은 다음 단계에서 구현됩니다.\n현재는 선택된 설정에 대한 로그만 표시됩니다.")
    
    # TODO (1-2 단계에서):
    # 1. 실제 결측치 처리 로직 구현 (pandas 사용)
    # 2. 처리된 DataFrame (df_after_imputation)을 main_app.py의 콜백 (예: step4_processing_complete)으로 전달하여
    #    app_state.current_df 와 app_state.df_after_step4 (가칭) 등을 업데이트.


def _populate_method_selection_table(df: pd.DataFrame):
    """결측치 처리 방법 선택 테이블 내용을 채우는 함수"""
    if not dpg.does_item_exist(TAG_MV_METHOD_SELECTION_TABLE) or df is None or df.empty:
        if dpg.does_item_exist(TAG_MV_METHOD_SELECTION_TABLE):
            dpg.delete_item(TAG_MV_METHOD_SELECTION_TABLE, children_only=True)
            dpg.add_table_column(label="정보", parent=TAG_MV_METHOD_SELECTION_TABLE)
            with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
                dpg.add_text("처리 방법을 선택할 데이터가 없습니다.")
        return
    
    dpg.delete_item(TAG_MV_METHOD_SELECTION_TABLE, children_only=True) # 기존 내용 삭제
    
    # 사용 가능한 처리 방법 목록
    imputation_options = [
        "처리 안 함", "0으로 대체", "평균값으로 대체", 
        "중앙값으로 대체", "최빈값으로 대체", "특정 값으로 대체", "행 제거"
    ]

    # 테이블 헤더 정의
    dpg.add_table_column(label="변수명", parent=TAG_MV_METHOD_SELECTION_TABLE, width_fixed=True, init_width_or_weight=150)
    dpg.add_table_column(label="데이터 타입", parent=TAG_MV_METHOD_SELECTION_TABLE, width_fixed=True, init_width_or_weight=120)
    dpg.add_table_column(label="결측치 수", parent=TAG_MV_METHOD_SELECTION_TABLE, width_fixed=True, init_width_or_weight=100)
    dpg.add_table_column(label="처리 방법", parent=TAG_MV_METHOD_SELECTION_TABLE, width_stretch=True) # 남은 공간 채움
    dpg.add_table_column(label="특정 값 (필요시 입력)", parent=TAG_MV_METHOD_SELECTION_TABLE, width_fixed=True, init_width_or_weight=180)

    for col_name in df.columns:
        with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
            dpg.add_text(col_name)
            dpg.add_text(str(df[col_name].dtype))
            dpg.add_text(str(df[col_name].isnull().sum()))
            
            # 현재 컬럼에 대해 저장된 처리 방법 가져오기 (없으면 "처리 안 함")
            current_method = _imputation_method_selections.get(col_name, "처리 안 함")
            # 데이터 타입에 따라 선택 불가능한 옵션 비활성화 (1-2 단계에서 구현 고려)
            # 예: 범주형 변수에 "평균값으로 대체" 비활성화
            
            # 처리 방법 선택 콤보박스
            dpg.add_combo(imputation_options, default_value=current_method, width=-1,
                          callback=_on_imputation_method_change, user_data=col_name)
            
            # "특정 값으로 대체" 시 사용할 입력 필드
            custom_fill_input_tag = f"custom_fill_input_{col_name}"
            current_fill_value = _custom_fill_values.get(col_name, "")
            dpg.add_input_text(tag=custom_fill_input_tag, default_value=current_fill_value, width=-1,
                               show=(current_method == "특정 값으로 대체"), # 초기 표시 상태 설정
                               callback=_on_custom_fill_value_change, user_data=col_name)

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """이 모듈의 전체 UI를 생성합니다."""
    global _main_app_callbacks, _util_funcs
    _main_app_callbacks = main_callbacks
    # main_callbacks에서 get_util_funcs를 호출하여 _util_funcs를 가져옵니다.
    if 'get_util_funcs' in main_callbacks:
        _util_funcs = main_callbacks['get_util_funcs']()

    main_callbacks['register_step_group_tag'](step_name, TAG_MV_STEP_GROUP)

    with dpg.group(tag=TAG_MV_STEP_GROUP, parent=parent_container_tag, show=False):
        dpg.add_text(f"--- {step_name} ---")
        dpg.add_separator()

        # 1. 처리 전 결측치 현황
        dpg.add_text("1. 현재 데이터 결측치 현황 (처리 전)", color=[255, 255, 0])
        with dpg.table(tag=TAG_MV_INFO_TABLE_BEFORE, header_row=True, resizable=True, 
                       policy=dpg.mvTable_SizingFixedFit, scrollY=True, height=180,
                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            # 내용은 update_ui에서 채워짐
            dpg.add_table_column(label="안내", parent=TAG_MV_INFO_TABLE_BEFORE)
            with dpg.table_row(parent=TAG_MV_INFO_TABLE_BEFORE):
                dpg.add_text("데이터를 불러오거나 이전 단계를 완료하면 여기에 현황이 표시됩니다.")
        dpg.add_spacer(height=15)

        # 2. 변수별 처리 방법 선택
        dpg.add_text("2. 변수별 결측치 처리 방법 선택", color=[255, 255, 0])
        with dpg.table(tag=TAG_MV_METHOD_SELECTION_TABLE, header_row=True, resizable=True, 
                       policy=dpg.mvTable_SizingStretchProp, scrollY=True, height=350,
                       borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
            # 내용은 update_ui에서 채워짐
            dpg.add_table_column(label="안내", parent=TAG_MV_METHOD_SELECTION_TABLE)
            with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
                dpg.add_text("데이터를 불러오거나 이전 단계를 완료하면 여기에 변수 목록이 표시됩니다.")
        dpg.add_spacer(height=15)
        
        # 3. 처리 실행 버튼
        dpg.add_button(label="결측치 처리 실행", tag=TAG_MV_EXECUTE_BUTTON, width=-1, height=30,
                       callback=_execute_imputation_placeholder)
        dpg.add_spacer(height=15)

        # 4. 결과 표시 그룹
        with dpg.group(tag=TAG_MV_RESULTS_GROUP):
            dpg.add_text("3. 결측치 처리 결과", color=[255, 255, 0])
            dpg.add_separator()
            dpg.add_text("처리 로그:")
            # 로그를 표시할 스크롤 가능한 자식 창
            with dpg.child_window(tag=TAG_MV_LOG_TEXT_AREA, height=120, border=True):
                 dpg.add_text("실행 버튼을 누르면 여기에 로그가 표시됩니다.", tag=TAG_MV_LOG_TEXT, wrap=-1)
            
            dpg.add_spacer(height=10)
            dpg.add_text("처리 후 결측치 현황:")
            with dpg.table(tag=TAG_MV_INFO_TABLE_AFTER, header_row=True, resizable=True, 
                           policy=dpg.mvTable_SizingFixedFit, scrollY=True, height=180,
                           borders_outerH=True, borders_innerV=True, borders_innerH=True, borders_outerV=True):
                dpg.add_table_column(label="안내", parent=TAG_MV_INFO_TABLE_AFTER)
                with dpg.table_row(parent=TAG_MV_INFO_TABLE_AFTER):
                    dpg.add_text("실행 버튼을 누르면 여기에 처리 후 현황이 표시됩니다.")

    # 이 모듈의 UI를 업데이트하는 함수를 main_app에 등록
    main_callbacks['register_module_updater'](step_name, update_ui)


def update_ui(df: Optional[pd.DataFrame], main_callbacks: dict):
    """이 모듈의 UI를 현재 데이터프레임 기준으로 업데이트합니다."""
    global _main_app_callbacks, _util_funcs, _current_df_for_this_step
    
    # 콜백 및 유틸 함수 최신화 (create_ui 이후에도 호출될 수 있으므로)
    if not _main_app_callbacks: _main_app_callbacks = main_callbacks
    if not _util_funcs: 
        if 'get_util_funcs' in main_callbacks:
            _util_funcs = main_callbacks['get_util_funcs']()

    _current_df_for_this_step = df # 현재 스텝에서 사용할 DataFrame 저장

    if not dpg.is_dearpygui_running() or not dpg.does_item_exist(TAG_MV_STEP_GROUP):
        return # UI가 아직 준비되지 않았거나 DPG가 실행 중이 아님

    # 1. "처리 전 결측치 현황" 테이블 업데이트
    if dpg.does_item_exist(TAG_MV_INFO_TABLE_BEFORE):
        dpg.delete_item(TAG_MV_INFO_TABLE_BEFORE, children_only=True) # 테이블 내용만 삭제
        if df is not None and not df.empty:
            if _util_funcs and 'create_table_with_data' in _util_funcs:
                missing_summary_before = pd.DataFrame({
                    '컬럼명': df.columns,
                    '데이터 타입': df.dtypes.astype(str).values, # .astype(str) 추가
                    '결측 수': df.isnull().sum().values,
                    '결측 비율 (%)': (df.isnull().mean() * 100).round(2).values
                })
                _util_funcs['create_table_with_data'](TAG_MV_INFO_TABLE_BEFORE, missing_summary_before, parent_df_for_widths=missing_summary_before)
            else: # 유틸 함수 사용 불가 시 대체 표시
                dpg.add_table_column(label="오류", parent=TAG_MV_INFO_TABLE_BEFORE)
                with dpg.table_row(parent=TAG_MV_INFO_TABLE_BEFORE):
                    dpg.add_text("테이블 생성 유틸 함수를 찾을 수 없습니다.")
        else: # df가 None이거나 비어있을 때
            dpg.add_table_column(label="안내", parent=TAG_MV_INFO_TABLE_BEFORE)
            with dpg.table_row(parent=TAG_MV_INFO_TABLE_BEFORE):
                dpg.add_text("표시할 데이터가 없습니다. 파일을 로드하거나 이전 단계를 완료해주세요.")
    
    # 2. "변수별 처리 방법 선택" 테이블 업데이트
    if df is not None and not df.empty:
        _populate_method_selection_table(df) # 이 함수가 내부적으로 테이블 내용을 지우고 다시 그림
    else:
        if dpg.does_item_exist(TAG_MV_METHOD_SELECTION_TABLE):
            dpg.delete_item(TAG_MV_METHOD_SELECTION_TABLE, children_only=True)
            dpg.add_table_column(label="안내", parent=TAG_MV_METHOD_SELECTION_TABLE)
            with dpg.table_row(parent=TAG_MV_METHOD_SELECTION_TABLE):
                dpg.add_text("처리 방법을 선택할 데이터가 없습니다.")

    # 3. 결과 영역 초기화 (로그 및 처리 후 테이블)
    if dpg.does_item_exist(TAG_MV_LOG_TEXT):
        dpg.set_value(TAG_MV_LOG_TEXT, "처리 방법을 선택하고 '결측치 처리 실행' 버튼을 누르세요.")
    
    if dpg.does_item_exist(TAG_MV_INFO_TABLE_AFTER):
        dpg.delete_item(TAG_MV_INFO_TABLE_AFTER, children_only=True)
        dpg.add_table_column(label="안내", parent=TAG_MV_INFO_TABLE_AFTER)
        with dpg.table_row(parent=TAG_MV_INFO_TABLE_AFTER):
            dpg.add_text("실행 버튼을 누르면 여기에 처리 후 현황이 표시됩니다.")


def reset_missing_values_state():
    """이 모듈의 상태를 초기화합니다 (예: 새 파일 로드 시)."""
    global _imputation_method_selections, _custom_fill_values, _current_df_for_this_step
    _imputation_method_selections.clear()
    _custom_fill_values.clear()
    _current_df_for_this_step = None

    # UI 자체는 update_ui가 다시 호출되면서 데이터 없음 상태로 갱신될 것이므로,
    # 여기서는 주로 내부 상태 변수만 초기화합니다.
    # 만약 특정 UI 요소의 값을 직접 초기화해야 한다면 여기서 처리합니다.
    # (예: _populate_method_selection_table(None) 호출 등은 update_ui에서 df가 None일 때 처리됨)
    if dpg.is_dearpygui_running():
        if dpg.does_item_exist(TAG_MV_LOG_TEXT):
            dpg.set_value(TAG_MV_LOG_TEXT, "처리 방법을 선택하고 '결측치 처리 실행' 버튼을 누르세요.")
        if dpg.does_item_exist(TAG_MV_INFO_TABLE_AFTER):
            dpg.delete_item(TAG_MV_INFO_TABLE_AFTER, children_only=True)
            dpg.add_table_column(label="안내", parent=TAG_MV_INFO_TABLE_AFTER)
            with dpg.table_row(parent=TAG_MV_INFO_TABLE_AFTER):
                dpg.add_text("실행 버튼을 누르면 여기에 처리 후 현황이 표시됩니다.")
        # update_ui(None, _main_app_callbacks) # main_app에서 호출해 줄 것이므로 중복 호출 방지

    print("결측치 처리(Step 4) 상태 초기화 완료.")