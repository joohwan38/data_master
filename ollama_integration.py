# ollama_integration.py
import dearpygui.dearpygui as dpg
import requests
import base64
from io import BytesIO
from PIL import Image # Not strictly needed if BytesIO is directly used by DPG or for base64
import threading
import subprocess
import time
from typing import Optional, Callable, Dict, Any, List
import platform
import traceback # For more detailed error logging

# --- Configuration ---
OLLAMA_MODEL_NAME = "qwen2.5vl"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_SERVER_CHECK_TIMEOUT = 3
OLLAMA_MODEL_PULL_TIMEOUT = 300

# 이 태그들은 이제 main_app.py에서 관리하는 결과창의 태그를 사용하게 되므로,
# ollama_integration 모듈 내에서 직접적인 POPUP_TAG 사용은 줄어듭니다.
# POPUP_TAG = "ollama_analysis_popup_v2" # 기존 레거시 팝업용
# POPUP_TITLE_TEXT_TAG = "ollama_popup_title_text_v2"
# POPUP_CONTENT_TEXT_TAG = "ollama_popup_content_text_v2"
# POPUP_LOADING_SPINNER_TAG = "ollama_popup_loading_spinner_v2"
# POPUP_STATUS_TEXT_TAG = "ollama_popup_status_text_v2"

# --- State ---
_is_ollama_server_running = False
_is_target_model_available = False
_get_cached_image_data_func: Optional[Callable[[str], Optional[BytesIO]]] = None
_add_job_to_main_queue_func: Optional[Callable[[List[Any]], None]] = None

# main_app으로부터 전달받을 콜백 함수들
_is_analysis_in_progress_func: Optional[Callable[[], bool]] = None
_set_analysis_status_func: Optional[Callable[[bool], None]] = None
_show_result_window_func: Optional[Callable[[str, str, bool], None]] = None # title, initial_status, is_loading
_update_result_window_job_func_name: Optional[str] = None # main_app_callbacks에서 가져올 콜백 함수 이름


# --- Helper to add jobs to main_app's queue (UI 업데이트용) ---
def _add_gui_job_to_queue(callback_func_name_in_main_app: str, user_data_dict: Dict):
    """GUI 업데이트 작업을 메인 콜백 큐에 추가합니다. 콜백 함수 이름을 직접 사용합니다."""
    if _add_job_to_main_queue_func and dpg.is_dearpygui_running() and callback_func_name_in_main_app:
        # main_app_callbacks에서 실제 함수를 찾아서 실행하도록 main_app.py의 루프에서 처리
        # 여기서는 작업 명세만 전달
        job = [callback_func_name_in_main_app, None, None, user_data_dict]
        _add_job_to_main_queue_func(job)
    elif not dpg.is_dearpygui_running():
        print(f"DPG 미실행, 작업 추가 건너뜀 (메인 큐): {callback_func_name_in_main_app} with {user_data_dict}")
    else:
        print(f"오류: 메인 콜백 큐 추가 함수 또는 콜백 이름이 설정되지 않았습니다.")


# --- Ollama Server & Model Management (기존과 유사) ---
def _update_status_via_gui_job(status_message: str, is_loading: bool = False, error: bool = False, title: Optional[str] = None):
    """결과창의 상태 메시지를 업데이트하기 위한 GUI 작업 추가"""
    if _update_result_window_job_func_name: # 이 이름의 함수가 main_app_callbacks에 있어야 함
        job_data = {
            "title": title or "Ollama 환경 설정", # title이 없으면 기본값 사용
            "content": "" if not error else status_message, # 오류 시 content에 메시지, 아니면 비움
            "loading": is_loading,
            "status_message": status_message,
            "error": error
        }
        _add_gui_job_to_queue(_update_result_window_job_func_name, job_data)

def _check_ollama_server_status() -> bool:
    global _is_ollama_server_running
    try:
        response = requests.get(OLLAMA_BASE_URL, timeout=OLLAMA_SERVER_CHECK_TIMEOUT)
        _is_ollama_server_running = response.status_code == 200 and "Ollama is running" in response.text
        return _is_ollama_server_running
    except requests.exceptions.RequestException:
        _is_ollama_server_running = False
        return False

def _check_target_model_availability() -> bool:
    global _is_target_model_available
    if not _is_ollama_server_running:
        _is_target_model_available = False; return False
    try:
        response = requests.get(OLLAMA_API_TAGS_URL, timeout=OLLAMA_SERVER_CHECK_TIMEOUT)
        response.raise_for_status()
        models = response.json().get("models", [])
        _is_target_model_available = any(m["name"].startswith(OLLAMA_MODEL_NAME.split(':')[0]) for m in models)
        return _is_target_model_available
    except requests.exceptions.RequestException:
        _is_target_model_available = False; return False

def _start_ollama_server_and_pull_model_job():
    global _is_ollama_server_running, _is_target_model_available

    if not _check_ollama_server_status():
        _update_status_via_gui_job("Ollama 서버 시작 시도 중...", is_loading=True, title="Ollama 서버")
        print("Ollama 서버가 실행 중이지 않습니다. 시작을 시도합니다...")
        try:
            cmd = ["ollama", "serve"]
            if platform.system() == "Windows":
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5)
            if not _check_ollama_server_status():
                msg = "Ollama 서버 시작 실패."; print(msg)
                _update_status_via_gui_job(msg, error=True, title="Ollama 서버 오류")
                return
            print("Ollama 서버 시작됨.")
            _update_status_via_gui_job("Ollama 서버 시작됨.", title="Ollama 서버")
        except FileNotFoundError:
            msg = "Ollama CLI를 찾을 수 없습니다. PATH 설정을 확인하세요."; print(msg)
            _update_status_via_gui_job(msg, error=True, title="Ollama 설정 오류")
            return
        except Exception as e:
            msg = f"Ollama 서버 시작 중 오류: {e}"; print(msg)
            _update_status_via_gui_job(msg, error=True, title="Ollama 서버 오류")
            return

    if _is_ollama_server_running and not _check_target_model_availability():
        _update_status_via_gui_job(f"{OLLAMA_MODEL_NAME} 모델 다운로드 중...", is_loading=True, title="Ollama 모델")
        print(f"{OLLAMA_MODEL_NAME} 모델이 로컬에 없습니다. 다운로드를 시도합니다...")
        try:
            process = subprocess.Popen(["ollama", "pull", OLLAMA_MODEL_NAME], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
            start_time = time.time(); last_status_update_time = time.time()
            for line in iter(process.stdout.readline, ''):
                stripped_line = line.strip(); print(f"[Ollama Pull]: {stripped_line}")
                current_time = time.time()
                if "pulling" in stripped_line and "100%" not in stripped_line and (current_time - last_status_update_time > 0.5):
                    _update_status_via_gui_job(f"모델 다운로드 중: {stripped_line.split()[-1]}", is_loading=True, title="Ollama 모델")
                    last_status_update_time = current_time
                if current_time - start_time > OLLAMA_MODEL_PULL_TIMEOUT:
                    process.kill(); msg = "모델 다운로드 시간 초과."
                    print(msg); _update_status_via_gui_job(msg, error=True, title="Ollama 모델 오류"); process.stdout.close(); return
                if process.poll() is not None: break
            process.stdout.close(); return_code = process.wait()
            if return_code == 0:
                msg = f"{OLLAMA_MODEL_NAME} 모델 다운로드 완료."; print(msg); _is_target_model_available = True
                _update_status_via_gui_job("모델 준비 완료.", title="Ollama 모델")
            else:
                stderr_output = process.stderr.read() if process.stderr else "No stderr."
                msg = f"{OLLAMA_MODEL_NAME} 모델 다운로드 실패: {stderr_output[:150]}..."; print(msg); _is_target_model_available = False
                _update_status_via_gui_job(msg, error=True, title="Ollama 모델 오류")
        except FileNotFoundError:
            msg = "Ollama CLI를 찾을 수 없습니다. 모델을 수동으로 다운로드해주세요."; print(msg)
            _update_status_via_gui_job(msg, error=True, title="Ollama 설정 오류")
        except Exception as e:
            msg = f"모델 다운로드 중 오류: {e}"; print(msg)
            _update_status_via_gui_job(msg, error=True, title="Ollama 모델 오류")
    elif _is_ollama_server_running and _is_target_model_available:
         _update_status_via_gui_job("Ollama 환경 준비 완료.", title="Ollama 환경")


def _convert_bytesio_to_base64(image_bytes_io: BytesIO) -> Optional[str]:
    try:
        return base64.b64encode(image_bytes_io.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"이미지 base64 인코딩 오류: {e}"); return None

def _ollama_api_call_job(image_title: str, base64_image: str):
    """ 실제 Ollama API를 호출하고 결과를 처리하는 작업 (스레드에서 실행) """
    error_msg_prefix = f"이미지 '{image_title}' 분석 실패: "
    final_status_message = "API 통신 오류"
    final_content = error_msg_prefix
    error_occurred = True

    if not _is_ollama_server_running or not _is_target_model_available:
        error_detail = ""
        if not _is_ollama_server_running: error_detail += "Ollama 서버가 실행 중이지 않습니다. "
        if not _is_target_model_available: error_detail += f"{OLLAMA_MODEL_NAME} 모델을 사용할 수 없습니다."
        final_content += error_detail.strip()
        final_status_message = "Ollama 환경 오류"
    else:
        prompt = f"다음 이미지는 데이터 분석 과정에서 생성된 시각화 자료입니다. 이미지의 제목은 '{image_title}'입니다. 이 이미지에서 나타나는 주요 특징이나 인사이트를 3가지 핵심 항목으로 요약해주세요. 한국어로 답변해주세요."
        payload = {"model": OLLAMA_MODEL_NAME, "prompt": prompt, "images": [base64_image], "stream": False}
        try:
            response = requests.post(OLLAMA_API_GENERATE_URL, json=payload, timeout=180) # 타임아웃 증가
            response.raise_for_status()
            result_text = response.json().get("response", "Ollama로부터 응답 내용을 받지 못했습니다.")
            final_content = result_text
            final_status_message = "분석 완료"
            error_occurred = False
        except requests.exceptions.Timeout:
            print(f"Ollama API 요청 시간 초과: {image_title}")
            final_content += f"API 요청 시간 초과. 서버가 응답하지 않거나 분석에 너무 오랜 시간이 소요됩니다."
            final_status_message = "API 시간 초과"
        except requests.exceptions.RequestException as e:
            print(f"Ollama API 요청 실패: {e}\n{traceback.format_exc()}")
            final_content += f"API 요청 실패 ({type(e).__name__}). 서버 상태 및 네트워크 연결을 확인하세요."
            final_status_message = "API 통신 오류"
        except Exception as e:
            print(f"Ollama API 처리 중 예외 발생: {e}\n{traceback.format_exc()}")
            final_content += f"처리 중 예외 발생 ({type(e).__name__})."
            final_status_message = "내부 처리 오류"

    # 결과창 업데이트 작업 추가
    if _update_result_window_job_func_name:
        _add_gui_job_to_queue(_update_result_window_job_func_name, {
            "title": image_title, "content": final_content, "loading": False,
            "status_message": final_status_message, "error": error_occurred
        })
    
    # 분석 상태 해제 (성공/실패 무관하게)
    if _set_analysis_status_func:
        _set_analysis_status_func(False)


def initialize_ollama_integration(
    get_image_func: Callable[[str], Optional[BytesIO]],
    add_job_func: Callable[[List[Any]], None],
    is_analysis_in_progress_func: Callable[[], bool], # 누락된 인자 1
    set_analysis_status_func: Callable[[bool], None],   # 누락된 인자 2
    show_result_window_func: Callable[[str, str, bool], None], # 누락된 인자 3
    update_result_window_job_func_name: str # 누락된 인자 4
):
    global _get_cached_image_data_func, _add_job_to_main_queue_func
    global _is_analysis_in_progress_func, _set_analysis_status_func, _show_result_window_func
    global _update_result_window_job_func_name

    _get_cached_image_data_func = get_image_func
    _add_job_to_main_queue_func = add_job_func
    _is_analysis_in_progress_func = is_analysis_in_progress_func
    _set_analysis_status_func = set_analysis_status_func
    _show_result_window_func = show_result_window_func
    _update_result_window_job_func_name = update_result_window_job_func_name

    # 기존의 create_ollama_popup()은 main_app.py에서 결과창을 미리 생성하므로 제거
    # if dpg.is_dearpygui_running():
    #     create_ollama_popup() # 이 부분은 이제 main_app.py에서 관리

    # 서버/모델 확인 스레드는 여전히 시작
    threading.Thread(target=_start_ollama_server_and_pull_model_job, daemon=True).start()


def request_image_analysis(user_data_from_confirm: Dict[str,str]): # 이제 main_app의 확인 후 호출됨
    """ 사용자의 확인 후 실제 분석 요청을 처리하는 함수 """
    image_title = user_data_from_confirm.get("title")
    texture_tag = user_data_from_confirm.get("texture_tag_for_ollama")

    if not image_title or not texture_tag:
        msg = "이미지 분석 요청에 필요한 정보(제목 또는 텍스처 태그)가 부족합니다."
        print(f"[ERROR ollama_integration] {msg}")
        if _show_result_window_func:
             _show_result_window_func(image_title or "오류", msg, False) # 로딩 false
        if _set_analysis_status_func: _set_analysis_status_func(False) # 혹시 모르니 상태 해제
        return

    # 이미 분석 중인지 다시 한번 확인 (이론적으로는 main_app에서 이미 확인했어야 함)
    if _is_analysis_in_progress_func and _is_analysis_in_progress_func():
        msg = "이미 다른 분석 작업이 진행 중입니다. 잠시 후 다시 시도해주세요."
        print(f"[INFO ollama_integration] {msg}")
        if _show_result_window_func: # 결과창에 상태 표시
            _show_result_window_func(image_title, msg, False)
        # 상태는 이미 True일 것이므로 여기서 변경하지 않음
        return

    # 분석 시작 상태 설정 및 결과창 표시
    if _set_analysis_status_func: _set_analysis_status_func(True)
    if _show_result_window_func:
        _show_result_window_func(image_title, "Ollama 환경 확인 중...", True)

    # 서버/모델 가용성 확인
    if not _is_ollama_server_running or not _is_target_model_available:
        status_msg = ""
        if not _is_ollama_server_running : status_msg += "Ollama 서버 연결 불가. "
        if not _is_target_model_available: status_msg += f"{OLLAMA_MODEL_NAME} 모델 준비 안됨."
        if _update_result_window_job_func_name:
            _add_gui_job_to_queue(_update_result_window_job_func_name, {
                "title": image_title, "content": "Ollama 환경 준비가 필요합니다. 잠시 후 다시 시도하거나 상태 메시지를 확인하세요.",
                "loading": False, "status_message": status_msg.strip(), "error": True
            })
        if _set_analysis_status_func: _set_analysis_status_func(False) # 환경 문제 시 상태 해제
        if not _is_ollama_server_running: # 서버가 안 켜져 있으면 다시 시작 시도
             threading.Thread(target=_start_ollama_server_and_pull_model_job, daemon=True).start()
        return

    if not _get_cached_image_data_func:
        msg = "내부 설정 오류: 이미지 로더 함수가 설정되지 않았습니다."
        print(f"오류: {msg}")
        if _update_result_window_job_func_name:
             _add_gui_job_to_queue(_update_result_window_job_func_name, {
                "title": image_title, "content": msg, "loading": False, "status_message": "설정 오류", "error": True
            })
        if _set_analysis_status_func: _set_analysis_status_func(False)
        return

    image_bytes_io = _get_cached_image_data_func(texture_tag)
    if not image_bytes_io:
        msg = f"'{texture_tag}'에 대한 캐시된 이미지 데이터를 찾을 수 없습니다."
        if _update_result_window_job_func_name:
             _add_gui_job_to_queue(_update_result_window_job_func_name, {
                "title": image_title, "content": msg, "loading": False, "status_message": "이미지 데이터 오류", "error": True
            })
        if _set_analysis_status_func: _set_analysis_status_func(False)
        return

    base64_image = _convert_bytesio_to_base64(image_bytes_io)
    if not base64_image:
        msg = "이미지 데이터 변환에 실패했습니다 (Base64 인코딩 오류)."
        if _update_result_window_job_func_name:
            _add_gui_job_to_queue(_update_result_window_job_func_name, {
                "title": image_title, "content": msg, "loading": False, "status_message": "이미지 변환 오류", "error": True
            })
        if _set_analysis_status_func: _set_analysis_status_func(False)
        return

    # API 호출 준비 완료, 상태 메시지 업데이트
    if _update_result_window_job_func_name:
        _add_gui_job_to_queue(_update_result_window_job_func_name, {
            "title": image_title, "content": "Ollama 서버에 분석을 요청합니다...", "loading": True,
            "status_message": "요청 전송 중...", "error": False
        })
    
    # 스레드에서 API 호출 실행
    threading.Thread(target=_ollama_api_call_job, args=(image_title, base64_image), daemon=True).start()