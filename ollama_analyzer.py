# ollama_analyzer.py
import time
import requests # API 호출을 위한 requests 라이브러리
import base64 # 이미지 인코딩을 위함
import json # JSON 다루기 위함

# Ollama 서버 정보 (이전에 정의한 상수 사용)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b" # 또는 사용하고자 하는 Llava 모델명

def analyze_image_with_llava(image_bytes: bytes, chart_name: str = "Untitled Chart") -> str:
    """
    Ollama Llava 모델을 사용하여 이미지 바이트를 분석하고 텍스트 설명을 반환합니다.
    image_bytes: PNG 이미지 등의 바이트 데이터.
    chart_name: 분석 대상 차트 이름 (로그 및 프롬프트용).
    """
    print(f"[Ollama Analyzer] Received image for '{chart_name}' (size: {len(image_bytes)} bytes). Starting REAL analysis...")

    try:
        # 1. 이미지 바이트를 Base64로 인코딩
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')

        # 2. Ollama API에 전달할 페이로드(payload) 구성
        # 프롬프트는 차트의 종류나 분석 목적에 맞게 상세하게 작성할수록 좋은 결과를 얻을 수 있습니다.
        prompt_text = (
            f"'{chart_name}' 차트를 데이터분석 해주세요. 전반적인 해석은 빼고 핵심분석결과와 활용방안에 대해서만 이야기해주세요."
        )

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt_text,
            "images": [encoded_image],
            "stream": False # True로 설정하면 스트리밍 방식으로 응답을 받지만, 여기서는 False로 전체 응답을 한 번에 받음
        }

        # 3. Ollama API 호출
        # 타임아웃을 적절히 설정하여 너무 오래 기다리지 않도록 합니다 (예: 60초)
        print(f"[Ollama Analyzer] Sending request to Ollama for '{chart_name}'...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status() # HTTP 오류 발생 시 예외를 발생시킴 (4xx, 5xx 상태 코드)

        # 4. 응답 파싱 및 결과 추출
        result_json = response.json()
        analysis_text = result_json.get("response", "Error: No 'response' field in Ollama result.")
        # 모델에 따라 'response' 대신 다른 키를 사용할 수 있으니, 실제 응답을 확인하고 조정하세요.

        print(f"[Ollama Analyzer] Analysis complete for '{chart_name}'. Raw response snippet: {analysis_text[:100]}...")
        return analysis_text

    except requests.exceptions.Timeout:
        error_message = f"Error: Timeout during Ollama API call for '{chart_name}'."
        print(f"[Ollama Analyzer] {error_message}")
        return error_message
    except requests.exceptions.ConnectionError:
        error_message = f"Error: Could not connect to Ollama server at {OLLAMA_API_URL} for '{chart_name}'. Is Ollama running?"
        print(f"[Ollama Analyzer] {error_message}")
        return error_message
    except requests.exceptions.HTTPError as http_err:
        error_message = f"Error: HTTP error during Ollama API call for '{chart_name}'. Status: {http_err.response.status_code}\nResponse: {http_err.response.text[:200]}"
        print(f"[Ollama Analyzer] {error_message}")
        return error_message
    except requests.exceptions.RequestException as req_err: # 기타 requests 관련 예외
        error_message = f"Error: API request failed for '{chart_name}'.\nDetails: {str(req_err)[:100]}"
        print(f"[Ollama Analyzer] {error_message}")
        return error_message
    except json.JSONDecodeError:
        error_message = f"Error: Could not decode JSON response from Ollama for '{chart_name}'."
        print(f"[Ollama Analyzer] {error_message} Raw response: {response.text[:200] if 'response' in locals() else 'N/A'}")
        return error_message
    except Exception as e: # 기타 모든 예외
        error_message = f"Error: An unexpected error occurred during Ollama analysis for '{chart_name}'.\nDetails: {str(e)[:100]}"
        print(f"[Ollama Analyzer] {error_message}")
        import traceback
        print(traceback.format_exc()) # 개발 중 상세 오류 확인용
        return error_message