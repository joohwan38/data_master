# ollama_integration.py
import dearpygui.dearpygui as dpg
import requests
import base64
from io import BytesIO
from PIL import Image
import threading
import subprocess
import time
from typing import Optional, Callable, Dict, Any, List
import platform

# --- Configuration ---
OLLAMA_MODEL_NAME = "qwen2.5vl"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_SERVER_CHECK_TIMEOUT = 3
OLLAMA_MODEL_PULL_TIMEOUT = 300 # 모델 다운로드 최대 시간 (초)

POPUP_TAG = "ollama_analysis_popup_v2"
POPUP_TITLE_TEXT_TAG = "ollama_popup_title_text_v2"
POPUP_CONTENT_TEXT_TAG = "ollama_popup_content_text_v2"
POPUP_LOADING_SPINNER_TAG = "ollama_popup_loading_spinner_v2"
POPUP_STATUS_TEXT_TAG = "ollama_popup_status_text_v2"

# --- State ---
_is_ollama_server_running = False
_is_target_model_available = False
_get_cached_image_data_func: Optional[Callable[[str], Optional[BytesIO]]] = None
_add_job_to_main_queue_func: Optional[Callable[[List[Any]], None]] = None


# --- UI Update Functions (called via main_app's callback_queue) ---

def _update_popup_content_from_job(sender, app_data, user_data: Dict[str, Any]): # 이름 변경
    """메인 큐의 작업으로부터 받은 데이터로 팝업을 업데이트합니다."""
    title = user_data.get("title", "분석 결과")
    content = user_data.get("content", "내용 없음")
    is_loading = user_data.get("loading", False)
    status_message = user_data.get("status_message")
    _update_popup_content(title, content, is_loading, status_message)

def _update_popup_status_from_job(sender, app_data, user_data: Dict[str, str]): # 이름 변경
    """메인 큐의 작업으로부터 받은 데이터로 팝업 상태 메시지만 업데이트합니다."""
    status_message = user_data.get("status_message", "")
    if dpg.is_dearpygui_running():
        # 팝업이 현재 화면에 떠 있을 때만 상태 메시지 업데이트 시도
        if dpg.does_item_exist(POPUP_TAG) and dpg.is_item_shown(POPUP_TAG):
            if dpg.does_item_exist(POPUP_STATUS_TEXT_TAG):
                dpg.set_value(POPUP_STATUS_TEXT_TAG, status_message)

# --- Helper to add jobs to main_app's queue ---
def _add_gui_job_to_queue(callback_func: Callable, user_data_dict: Dict):
    """GUI 업데이트 작업을 메인 콜백 큐에 추가합니다."""
    if _add_job_to_main_queue_func and dpg.is_dearpygui_running():
        job = [callback_func, None, None, user_data_dict] # Job 포맷: [callable, sender, app_data, user_data]
        _add_job_to_main_queue_func(job)
    elif not dpg.is_dearpygui_running():
        print(f"DPG 미실행, 작업 추가 건너뜀 (메인 큐): {callback_func.__name__} with {user_data_dict}")
    else:
        print(f"오류: 메인 콜백 큐 추가 함수(_add_job_to_main_queue_func)가 설정되지 않았습니다.")

# --- Ollama Server & Model Management ---

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
        _is_target_model_available = False
        return False
    try:
        response = requests.get(OLLAMA_API_TAGS_URL, timeout=OLLAMA_SERVER_CHECK_TIMEOUT)
        response.raise_for_status()
        models = response.json().get("models", [])
        _is_target_model_available = any(m["name"].startswith(OLLAMA_MODEL_NAME.split(':')[0]) for m in models)
        return _is_target_model_available
    except requests.exceptions.RequestException:
        _is_target_model_available = False
        return False

def _start_ollama_server_and_pull_model_job():
    global _is_ollama_server_running, _is_target_model_available

    if not _check_ollama_server_status():
        _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": "Ollama 서버 시작 시도 중..."})
        print("Ollama 서버가 실행 중이지 않습니다. 시작을 시도합니다...")
        try:
            cmd = ["ollama", "serve"]
            if platform.system() == "Windows":
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(5) # 서버 시작 대기 시간
            if not _check_ollama_server_status():
                msg = "Ollama 서버 시작 실패."
                print(msg)
                _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": msg})
                return
            print("Ollama 서버 시작됨.")
            _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": "Ollama 서버 시작됨."})
        except FileNotFoundError:
            msg = "Ollama CLI를 찾을 수 없습니다. PATH 설정을 확인하세요."
            print(msg)
            _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": msg})
            return
        except Exception as e:
            msg = f"Ollama 서버 시작 중 오류: {e}"
            print(msg)
            _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": msg})
            return

    if _is_ollama_server_running and not _check_target_model_availability():
        _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": f"{OLLAMA_MODEL_NAME} 모델 다운로드 중..."})
        print(f"{OLLAMA_MODEL_NAME} 모델이 로컬에 없습니다. 다운로드를 시도합니다...")
        try:
            process = subprocess.Popen(["ollama", "pull", OLLAMA_MODEL_NAME], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
            start_time = time.time()
            last_status_update_time = time.time()

            for line in iter(process.stdout.readline, ''):
                stripped_line = line.strip()
                print(f"[Ollama Pull]: {stripped_line}") # 콘솔에는 모든 진행률 출력
                current_time = time.time()
                # UI 업데이트는 너무 자주 하지 않도록 시간 간격 조절
                if "pulling" in stripped_line and "100%" not in stripped_line and (current_time - last_status_update_time > 0.5):
                    _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": f"모델 다운로드 중: {stripped_line.split()[-1]}"})
                    last_status_update_time = current_time

                if current_time - start_time > OLLAMA_MODEL_PULL_TIMEOUT:
                    process.kill(); msg = "모델 다운로드 시간 초과."
                    print(msg)
                    _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": msg})
                    process.stdout.close(); return
                if process.poll() is not None: break # 프로세스 종료 시 루프 탈출
            process.stdout.close()
            return_code = process.wait() # 최종 종료 코드 확인

            if return_code == 0:
                msg = f"{OLLAMA_MODEL_NAME} 모델 다운로드 완료."
                print(msg); _is_target_model_available = True
                _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": "모델 준비 완료."})
            else:
                stderr_output = process.stderr.read() if process.stderr else "No stderr output."
                msg = f"{OLLAMA_MODEL_NAME} 모델 다운로드 실패: {stderr_output[:150]}..."
                print(msg); _is_target_model_available = False
                _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": msg})
        except FileNotFoundError:
            msg = "Ollama CLI를 찾을 수 없습니다. 모델을 수동으로 다운로드해주세요."
            print(msg)
            _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": msg})
        except Exception as e:
            msg = f"모델 다운로드 중 오류: {e}"
            print(msg)
            _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": msg})
    elif _is_ollama_server_running and _is_target_model_available:
         _add_gui_job_to_queue(_update_popup_status_from_job, {"status_message": "Ollama 환경 준비 완료."})


def _convert_bytesio_to_base64(image_bytes_io: BytesIO) -> Optional[str]:
    try:
        return base64.b64encode(image_bytes_io.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"이미지 base64 인코딩 오류: {e}")
        return None

def _ollama_api_call_job(image_title: str, base64_image: str):
    error_msg_prefix = f"이미지 '{image_title}' 분석 실패: "
    if not _is_ollama_server_running or not _is_target_model_available:
        error_msg_detail = ""
        if not _is_ollama_server_running: error_msg_detail += "Ollama 서버가 실행 중이지 않습니다. "
        if not _is_target_model_available: error_msg_detail += f"{OLLAMA_MODEL_NAME} 모델을 사용할 수 없습니다."
        _add_gui_job_to_queue(_update_popup_content_from_job, {
            "title": image_title, "content": error_msg_prefix + error_msg_detail.strip(),
            "loading": False, "status_message": "Ollama 환경 오류"
        })
        return

    prompt = f"다음 이미지는 데이터 분석 과정에서 생성된 시각화 자료입니다. 이미지의 제목은 '{image_title}'입니다. 이 이미지에서 나타나는 주요 특징이나 인사이트를 3가지 핵심 항목으로 요약해주세요."
    payload = {"model": OLLAMA_MODEL_NAME, "prompt": prompt, "images": [base64_image], "stream": False}
    try:
        response = requests.post(OLLAMA_API_GENERATE_URL, json=payload, timeout=120)
        response.raise_for_status()
        result_text = response.json().get("response", "Ollama로부터 응답 내용을 받지 못했습니다.")
        _add_gui_job_to_queue(_update_popup_content_from_job, {
            "title": image_title, "content": result_text, "loading": False
        })
    except requests.exceptions.RequestException as e:
        print(f"Ollama API 요청 실패: {e}")
        _add_gui_job_to_queue(_update_popup_content_from_job, {
            "title": image_title, "content": error_msg_prefix + f"API 요청 실패 ({type(e).__name__})",
            "loading": False, "status_message": "API 통신 오류"
        })
    except Exception as e:
        print(f"Ollama API 처리 중 예외 발생: {e}")
        _add_gui_job_to_queue(_update_popup_content_from_job, {
            "title": image_title, "content": error_msg_prefix + f"처리 중 예외 발생 ({type(e).__name__})",
            "loading": False, "status_message": "내부 처리 오류"
        })

def _update_popup_content(title:str, content: str, is_loading: bool, status_message: Optional[str] = None):
    if dpg.does_item_exist(POPUP_TAG):
        dpg.configure_item(POPUP_TAG, label=f"이미지 분석: {title[:30]}{'...' if len(title)>30 else ''}")
        if dpg.does_item_exist(POPUP_CONTENT_TEXT_TAG):
            formatted_content = content.replace("\\n", "\n")
            dpg.set_value(POPUP_CONTENT_TEXT_TAG, formatted_content)
        if dpg.does_item_exist(POPUP_LOADING_SPINNER_TAG):
            dpg.configure_item(POPUP_LOADING_SPINNER_TAG, show=is_loading)
        current_status = status_message if status_message else ("분석 중..." if is_loading else "완료")
        if dpg.does_item_exist(POPUP_STATUS_TEXT_TAG):
            dpg.set_value(POPUP_STATUS_TEXT_TAG, current_status)

def create_ollama_popup():
    if not dpg.does_item_exist(POPUP_TAG):
        with dpg.window(label="이미지 분석", modal=True, show=False, tag=POPUP_TAG,
                        width=450, height=280, no_close=True, autosize=False,
                        no_saved_settings=True, no_collapse=True, no_resize=True) as popup_window_id:
            with dpg.group(horizontal=True):
                dpg.add_loading_indicator(tag=POPUP_LOADING_SPINNER_TAG, show=True, style=0, radius=3.5, color=(50,150,255,255))
                dpg.add_spacer(width=5)
                dpg.add_text("상태: ", tag=POPUP_STATUS_TEXT_TAG, color=(200,200,100))
            dpg.add_separator()
            dpg.add_text("분석 내용 로딩 중...", tag=POPUP_CONTENT_TEXT_TAG, wrap=420)
            dpg.add_spacer(height=15)
            with dpg.group(horizontal=True):
                popup_width_val = dpg.get_item_configuration(popup_window_id)["width"]
                button_width = 80
                item_spacing_x = 8.0
                spacer_width = (popup_width_val - button_width - item_spacing_x * 2) / 2
                if spacer_width < 0 : spacer_width = 0
                dpg.add_spacer(width=int(spacer_width))
                dpg.add_button(label="닫기", width=button_width, callback=lambda: dpg.configure_item(POPUP_TAG, show=False))

def initialize_ollama_integration(
    get_image_func: Callable[[str], Optional[BytesIO]],
    add_job_func: Callable[[List[Any]], None] # ★ main_app의 큐에 작업을 추가하는 함수를 받음
):
    global _get_cached_image_data_func, _add_job_to_main_queue_func
    _get_cached_image_data_func = get_image_func
    _add_job_to_main_queue_func = add_job_func # ★ 수정: 함수 포인터 저장

    if dpg.is_dearpygui_running():
        create_ollama_popup()
        threading.Thread(target=_start_ollama_server_and_pull_model_job, daemon=True).start()
    else:
        print("DPG 미실행 상태 (ollama_integration.initialize): Ollama 팝업 사전 생성 건너뜀. 서버/모델 확인 스레드는 시작됨.")
        threading.Thread(target=_start_ollama_server_and_pull_model_job, daemon=True).start()

def request_image_analysis(sender: Any, app_data: Any, user_data: Dict[str,str]): # 시그니처 변경
    image_title = user_data.get("title")
    texture_tag = user_data.get("texture_tag_for_ollama")
    if not image_title or not texture_tag:
        print("Error: Missing title or texture_tag in user_data for Ollama analysis.")
        return

    image_title = user_data.get("title")
    texture_tag = user_data.get("texture_tag_for_ollama")

    print(f"[DEBUG ollama_integration] request_image_analysis called. Sender: {sender}, AppData: {app_data}, UserData: {user_data}")

    if not image_title or not texture_tag:
        print(f"[ERROR ollama_integration] 필수 user_data 누락: title='{image_title}', texture_tag='{texture_tag}'")
        # 사용자에게 알림 (예: 팝업 메시지)
        _add_gui_job_to_queue(_update_popup_content_from_job, { # _update_popup_content_from_job 사용
            "title": "오류", "content": "이미지 분석 요청에 필요한 정보가 부족합니다.",
            "loading": False, "status_message": "요청 데이터 오류"
        })
        if dpg.is_dearpygui_running() and not dpg.does_item_exist(POPUP_TAG): create_ollama_popup()
        if dpg.is_dearpygui_running() and dpg.does_item_exist(POPUP_TAG): dpg.configure_item(POPUP_TAG, show=True)
        return
    
    if not _get_cached_image_data_func:
        msg = "내부 설정 오류: 이미지 로더 없음."
        print(f"오류: {msg}")
        if not dpg.does_item_exist(POPUP_TAG): create_ollama_popup()
        if dpg.does_item_exist(POPUP_TAG):
             _update_popup_content(image_title, msg, False, "오류 발생")
             dpg.configure_item(POPUP_TAG, show=True)
        return

    if not dpg.does_item_exist(POPUP_TAG): create_ollama_popup()
    _update_popup_content(image_title, "이미지 데이터 준비 중...", True, "서버/모델 확인 중...")
    dpg.configure_item(POPUP_TAG, show=True)

    if not _is_ollama_server_running or not _is_target_model_available:
        status_msg = ""
        if not _is_ollama_server_running : status_msg += "Ollama 서버 연결 불가. "
        if not _is_target_model_available: status_msg += f"{OLLAMA_MODEL_NAME} 모델 준비 안됨."
        _update_popup_content(image_title, "Ollama 환경 준비가 필요합니다. 잠시 후 다시 시도하거나 상태 메시지를 확인하세요.", False, status_msg.strip())
        if not _is_ollama_server_running:
             threading.Thread(target=_start_ollama_server_and_pull_model_job, daemon=True).start()
        return

    image_bytes_io = _get_cached_image_data_func(texture_tag)
    if not image_bytes_io:
        msg = f"'{texture_tag}'에 대한 캐시된 이미지 데이터를 찾을 수 없습니다."
        _update_popup_content(image_title, msg, False, "이미지 데이터 오류")
        return

    base64_image = _convert_bytesio_to_base64(image_bytes_io)
    if not base64_image:
        _update_popup_content(image_title, "이미지 데이터 변환에 실패했습니다.", False, "이미지 변환 오류")
        return

    _update_popup_content(image_title, "Ollama 서버에 분석 요청 중...", True, "요청 전송 중...")
    threading.Thread(target=_ollama_api_call_job, args=(image_title, base64_image), daemon=True).start()
