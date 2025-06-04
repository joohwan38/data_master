# ollama_analyzer.py
import time
import random

# 초기 하드코딩 설정
OLLAMA_API_URL = "http://localhost:11434/api/generate" # 또는 실제 사용 주소
OLLAMA_MODEL = "llava" # llava 또는 사용하고자 하는 비전 모델

def analyze_image_with_llava(image_bytes: bytes, chart_name: str = "Untitled Chart") -> str:
    """
    (초기 더미 버전) Ollama Llava 모델을 사용하여 이미지 바이트를 분석하고 텍스트 설명을 반환합니다.
    image_bytes: PNG 이미지 등의 바이트 데이터.
    chart_name: 분석 대상 차트 이름 (로그용).
    """
    print(f"[Ollama Analyzer] Received image for '{chart_name}' (size: {len(image_bytes)} bytes). Analyzing (dummy)...")

    # --- 실제 API 호출 로직은 여기에 구현될 예정 ---
    # 예시: requests 라이브러리 사용
    # import requests
    # import base64
    # encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    # payload = {
    #     "model": OLLAMA_MODEL,
    #     "prompt": f"Describe this chart image named '{chart_name}' in about 3 concise bullet points, focusing on key insights. Maximum 100 characters.",
    #     "images": [encoded_image],
    #     "stream": False
    # }
    # try:
    #     # response = requests.post(OLLAMA_API_URL, json=payload, timeout=60) # 60초 타임아웃
    #     # response.raise_for_status() # 오류 발생 시 예외 발생
    #     # result_json = response.json()
    #     # analysis_text = result_json.get("response", "Error: No 'response' field in Ollama result.")
    #     # 실제로는 여기서 result_json['response'] 등을 파싱해야 합니다.
    # except requests.exceptions.RequestException as e:
    #     print(f"[Ollama Analyzer] Error calling Ollama API: {e}")
    #     return f"Error: Could not analyze chart '{chart_name}'. API request failed.\nDetails: {str(e)[:100]}"
    # except Exception as e_parse:
    #     print(f"[Ollama Analyzer] Error parsing Ollama response: {e_parse}")
    #     return f"Error: Could not parse analysis for '{chart_name}'.\nDetails: {str(e_parse)[:100]}"
    # ---------------------------------------------

    # 더미 분석 시간 시뮬레이션
    time.sleep(random.uniform(1, 3))

    # 더미 분석 결과
    dummy_insights = [
        f"- {chart_name}: Insight A - Positive trend observed.",
        f"- {chart_name}: Insight B - Key cluster around (X, Y).",
        f"- {chart_name}: Insight C - Potential outlier detected at Z."
    ]
    random.shuffle(dummy_insights) # 순서 섞기
    analysis_text = "\n".join(dummy_insights[:random.randint(2,3)]) # 2~3개 랜덤 선택

    print(f"[Ollama Analyzer] Dummy analysis complete for '{chart_name}'.")
    return analysis_text