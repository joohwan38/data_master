# ollama_analyzer.py
import time
import requests
import base64
import json
import traceback
import re

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:latest"

def analyze_image_with_llava(image_bytes: bytes, chart_name: str = "Untitled Chart"):
    print(f"[Ollama Analyzer] Received image for '{chart_name}' (size: {len(image_bytes)} bytes). Starting analysis (streaming)...")

    try:
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        prompt_text = (
            f"당신은 전문적인 데이터분석가 입니다. '{chart_name}'차트를 데이터분석 하고 핵심분석결과와 활용방안에 대해서 짧고 강력한 인사이트를 주세요."
        )

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt_text,
            "images": [encoded_image],
            "stream": True
        }

        print(f"[Ollama Analyzer] Sending request to Ollama for '{chart_name}' (streaming)...")
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120, stream=True)
        response.raise_for_status()

        print(f"[Ollama Analyzer] Receiving stream for '{chart_name}'...")
        full_response_for_internal_log = ""
        any_chunk_yielded = False

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    json_chunk = json.loads(decoded_line)
                    chunk_text_from_llm = json_chunk.get("response", "")

                    if chunk_text_from_llm:
                        processed_chunk_text = chunk_text_from_llm
                        
                        if processed_chunk_text.startswith('*') and not processed_chunk_text.startswith('\n'):
                            processed_chunk_text = '\n' + processed_chunk_text
                        if processed_chunk_text.startswith('#') and not processed_chunk_text.startswith('\n'):
                            processed_chunk_text = '\n' + processed_chunk_text

                        processed_chunk_text = re.sub(r'(\*\*[^*]+\*\*):(?!\n)', r'\1:\n', processed_chunk_text)                       
                        processed_chunk_text = processed_chunk_text.replace('. ', '.\n')
                        processed_chunk_text = re.sub(r'\.(?=[A-Za-z가-힣])(?!\s*\n)', r'.\n', processed_chunk_text)
                        processed_chunk_text = re.sub(r'\. (?!\n)', r'.\n', processed_chunk_text)
                        processed_chunk_text = re.sub(r'\.(?=[가-힣A-Za-z])(?!\n)', r'.\n', processed_chunk_text)

                        yield processed_chunk_text
                        any_chunk_yielded = True
                        full_response_for_internal_log += processed_chunk_text
                    
                    if json_chunk.get("done"):
                        if not json_chunk.get("error"):
                            print(f"[Ollama Analyzer] Stream finished for '{chart_name}'.")
                        else:
                            error_message_from_ollama = json_chunk.get("error", "Unknown error from Ollama stream 'done' signal.")
                            print(f"[Ollama Analyzer] Error in Ollama stream for '{chart_name}': {error_message_from_ollama}")
                            yield f"\nOllama Error: {error_message_from_ollama}"
                        break
                except json.JSONDecodeError:
                    print(f"[Ollama Analyzer] Warning: Could not decode JSON line from stream: {decoded_line}")
        
        if not any_chunk_yielded:
            print(f"[Ollama Analyzer] Stream for '{chart_name}' was empty or contained no 'response' fields.")
            yield ""

        if full_response_for_internal_log:
            print(f"[Ollama Analyzer] Streaming complete for '{chart_name}'. Total response length: {len(full_response_for_internal_log)}. Snippet: {full_response_for_internal_log[:100]}...")

    except requests.exceptions.Timeout:
        error_message = f"Error: Timeout during Ollama API call for '{chart_name}'."
        print(f"[Ollama Analyzer] {error_message}")
        yield error_message
    except requests.exceptions.ConnectionError:
        error_message = f"Error: Could not connect to Ollama server at {OLLAMA_API_URL} for '{chart_name}'. Is Ollama running?"
        print(f"[Ollama Analyzer] {error_message}")
        yield error_message
    except requests.exceptions.HTTPError as http_err:
        error_message = f"Error: HTTP error during Ollama API call for '{chart_name}'. Status: {http_err.response.status_code}\nResponse: {http_err.response.text[:200]}"
        print(f"[Ollama Analyzer] {error_message}")
        yield error_message
    except requests.exceptions.RequestException as req_err:
        error_message = f"Error: API request failed for '{chart_name}'.\nDetails: {str(req_err)[:100]}"
        print(f"[Ollama Analyzer] {error_message}")
        yield error_message
    except Exception as e:
        error_message = f"Error: An unexpected error occurred during Ollama analysis for '{chart_name}'.\nDetails: {str(e)[:100]}"
        print(f"[Ollama Analyzer] {error_message}")
        print(traceback.format_exc())
        yield error_message