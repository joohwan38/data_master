# step_10_advanced_analysis.py - 통합 파일 (수정된 버전)

"""
Step 10 Advanced Analysis 통합 모듈

이 파일은 main_app.py에서 import하는 메인 파일로,
세 개의 분리된 모듈을 통합하여 관리합니다:
- step_10_ui: UI 담당
- step_10_analysis: 분석 로직 담당  
- step_10_visualization: 시각화 담당
"""

# 모듈 분리를 위한 import 구조
import step_10_ui as ui_module
import step_10_analysis as analysis_module
import step_10_visualization as viz_module

def create_ui(step_name: str, parent_container_tag: str, main_callbacks: dict):
    """Step 10 UI 생성 - UI 모듈로 위임"""
    return ui_module.create_ui(step_name, parent_container_tag, main_callbacks)

def update_ui():
    """UI 업데이트 - UI 모듈로 위임"""
    return ui_module.update_ui()

def reset_state():
    """상태 초기화 - 모든 모듈의 상태 초기화"""
    ui_module.reset_state()
    analysis_module.reset_state()
    viz_module.reset_state()

# 하위 호환성을 위해 기존 함수들을 유지하되, 적절한 모듈로 위임
def run_analysis(params):
    """분석 실행 - 분석 모듈로 위임"""
    return analysis_module.run_analysis(params)

def create_results_tab(tab_bar_tag, results):
    """결과 탭 생성 - 시각화 모듈로 위임"""
    return viz_module.create_results_tab(tab_bar_tag, results)

def export_results(results, format):
    """결과 내보내기 - 분석 모듈로 위임"""
    return analysis_module.export_results(results, format)

# 모듈 정보
__version__ = "2.0.0"
__modules__ = {
    "ui": ui_module,
    "analysis": analysis_module,
    "visualization": viz_module
}

def get_module_info():
    """모듈 정보 반환"""
    return {
        "version": __version__,
        "modules": list(__modules__.keys()),
        "description": "Step 10 Advanced Analysis - Modular Architecture"
    }