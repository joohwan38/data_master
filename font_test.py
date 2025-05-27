# korean_font_test.py
import dearpygui.dearpygui as dpg
import platform
import os

def setup_minimal_korean_font(font_file_path_to_test, font_size_to_test=20):
    if not os.path.exists(font_file_path_to_test):
        print(f"Test Error: Font file not found: {font_file_path_to_test}")
        return False
    
    print(f"Test Info: Attempting to load font: {font_file_path_to_test}")
    try:
        with dpg.font_registry(): # Use default registry for simplicity in test
            with dpg.font(file=font_file_path_to_test, size=font_size_to_test, tag="my_test_korean_font"):
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean)
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
        
        dpg.bind_font("my_test_korean_font") # Bind the font
        print(f"Test Info: Font '{font_file_path_to_test}' bound with tag 'my_test_korean_font'.")
        return True
    except Exception as e:
        print(f"Test Error: Could not setup test font '{font_file_path_to_test}': {e}")
        return False

dpg.create_context()

# --- 자동으로 테스트할 폰트 경로 결정 ---
# 1. Windows: Malgun Gothic
# 2. macOS: Apple SD Gothic Neo
# 3. Linux: /usr/share/fonts/truetype/nanum/NanumGothic.ttf 또는 현재 폴더의 NanumGothic.ttf
# 이 경로들이 없거나 다른 폰트를 테스트하고 싶다면 test_font_path_manual 값을 직접 설정하세요.

test_font_path_manual = None # 예: "C:/path/to/your/font.ttf" 또는 "/Users/name/font.ttf"

chosen_test_font_path = ""
os_name = platform.system()

if test_font_path_manual and os.path.exists(test_font_path_manual):
    chosen_test_font_path = test_font_path_manual
    print(f"Test Info: Using manually specified font: {chosen_test_font_path}")
else:
    if os_name == "Windows":
        path = "C:/Windows/Fonts/malgun.ttf"
        if os.path.exists(path): chosen_test_font_path = path
    elif os_name == "Darwin":
        path = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
        if os.path.exists(path): chosen_test_font_path = path
    elif os_name == "Linux":
        path_system = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        path_local = "NanumGothic.ttf" # 스크립트와 같은 폴더
        if os.path.exists(path_local):
            chosen_test_font_path = path_local
        elif os.path.exists(path_system):
            chosen_test_font_path = path_system
    
    if chosen_test_font_path:
        print(f"Test Info: Automatically selected font for {os_name}: {chosen_test_font_path}")
    else:
        print(f"Test Warning: Could not automatically find a Korean font for {os_name}.")
        print("Place a Korean .ttf font (e.g., NanumGothic.ttf) in the same directory as this script,")
        print("or set 'test_font_path_manual' in the script to a valid font file path.")


if chosen_test_font_path:
    font_setup_success = setup_minimal_korean_font(chosen_test_font_path)
    if not font_setup_success:
        print("Test Warning: Font setup failed. DPG will use its default font.")
else:
    print("Test Warning: No font path to test. DPG will use its default font.")


with dpg.window(label="Korean Font Minimal Test", width=400, height=200):
    dpg.add_text("Hello (English Test)")
    dpg.add_text("안녕하세요 (한글 테스트)") 
    dpg.add_text("Symbols: ???") # To differentiate from other text

dpg.create_viewport(title='Korean Font Test Viewport', width=500, height=300)
dpg.setup_dearpygui() # setup_dearpygui 전에 폰트 설정이 완료되어야 함
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()