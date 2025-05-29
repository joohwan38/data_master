# main_app.py
import dearpygui.dearpygui as dpg
import pandas as pd
import platform
import os
import utils
import step_01_data_loading
import step_02_exploratory_data_analysis
import traceback
import hashlib
import json

SETTINGS_DIR_NAME = ".app_file_settings"
SESSION_INFO_FILE = os.path.join(SETTINGS_DIR_NAME, "session_info.json")
selected_target_variable_type: str = "Continuous"
TARGET_VARIABLE_TYPE_RADIO_TAG = "target_variable_type_radio"
TARGET_VARIABLE_TYPE_LABEL_TAG = "target_variable_type_label"

current_df: pd.DataFrame = None
original_df: pd.DataFrame = None # DataFrame as it was loaded from file, before any step 1 processing
df_after_step1: pd.DataFrame = None # ADDED: DataFrame after Step 1 processing, serves as base for EDA
loaded_file_path: str = None

step_group_tags = {}
module_ui_updaters = {}
active_step_name: str = None
selected_target_variable: str = None
TARGET_VARIABLE_COMBO_TAG = "target_variable_combo"

_eda_sva_initialized = False # This might be specific to SVA, consider if it's still needed globally for EDA
_eda_outlier_settings_applied_once = False # ADDED: Flag to track if outlier settings have been applied

ANALYSIS_STEPS = [
    "1. Data Loading and Overview",
    "2. Exploratory Data Analysis (EDA)",
]
active_settings = {}

# --- Helper function to show a simple modal message ---
_MODAL_ID_SIMPLE_MESSAGE = "simple_modal_message_id"

def _show_simple_modal_message(title: str, message: str, width: int = 450, height: int = 200):
    if dpg.does_item_exist(_MODAL_ID_SIMPLE_MESSAGE):
        dpg.delete_item(_MODAL_ID_SIMPLE_MESSAGE)

    viewport_width = dpg.get_viewport_width() if dpg.is_dearpygui_running() else 1000
    viewport_height = dpg.get_viewport_height() if dpg.is_dearpygui_running() else 700
    
    modal_pos_x = (viewport_width - width) // 2
    modal_pos_y = (viewport_height - height) // 2
    
    with dpg.window(label=title, modal=True, show=True, id=_MODAL_ID_SIMPLE_MESSAGE,
                    no_close=True, width=width, height=height, pos=[modal_pos_x, modal_pos_y],
                    no_saved_settings=True, autosize=False):
        dpg.add_text(message, wrap=width - 20)
        dpg.add_spacer(height=15)
        with dpg.group(horizontal=True):
            dpg.add_spacer(width= (width - 100 - 30) // 2)
            dpg.add_button(label="OK", width=100, callback=lambda: dpg.configure_item(_MODAL_ID_SIMPLE_MESSAGE, show=False))

def setup_korean_font():
    font_path = None
    font_size = 17
    os_type = platform.system()
    print(f"--- Font Setup Initiated ---")
    print(f"Operating System: {os_type}")

    if os_type == "Darwin":
        potential_paths = [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        ]
        for p_path in potential_paths:
            if os.path.exists(p_path): font_path = p_path; break
    elif os_type == "Windows":
        potential_paths = ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc"]
        for p in potential_paths:
            if os.path.exists(p): font_path = p; break
    elif os_type == "Linux":
        potential_paths = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
        bundled_font_path = "NanumGothic.ttf" # Assuming it's bundled
        if os.path.exists(bundled_font_path): font_path = bundled_font_path
        else:
            for p in potential_paths:
                if os.path.exists(p): font_path = p; break
    
    if font_path and os.path.exists(font_path):
        print(f"Attempting to load and bind font: '{font_path}' with size {font_size}")
        try:
            font_registry_tag = "global_font_registry_unique" 
            font_to_bind_tag = "korean_font_for_app"

            if not dpg.does_item_exist(font_registry_tag):
                with dpg.font_registry(tag=font_registry_tag): pass # DEPRECATED: dpg.add_font_registry
            
            # Ensure font registry exists (DPG 1.10+ pattern)
            if not dpg.does_item_exist("korean_font_for_app"): # Check if font itself is registered
                with dpg.font_registry(): # Use the default registry
                    dpg.add_font(font_path, font_size, tag="korean_font_for_app")
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Korean, parent="korean_font_for_app")
                    dpg.add_font_range_hint(dpg.mvFontRangeHint_Default, parent="korean_font_for_app")
            
            dpg.bind_font("korean_font_for_app")
            print(f"Successfully attempted to bind font '{font_to_bind_tag}'.")
        except Exception as e:
            print(f"Error during explicit font processing for '{font_path}': {e}. DPG default font will be used.")
            traceback.print_exc()
    else:
        print("No suitable Korean font path was found or file does not exist. DPG default font will be used.")
    print(f"--- Font Setup Finished ---")


def update_target_variable_combo():
    global current_df, selected_target_variable, TARGET_VARIABLE_COMBO_TAG
    # MODIFIED: Target variable combo should reflect columns from current_df (which might be affected by outlier removal)
    # However, the *selection* of the target is a primary characteristic that shouldn't change due to outlier removal on *other* columns.
    # Let's assume current_df is the correct source for columns.
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        if current_df is not None and not current_df.empty:
            items = [""] + list(current_df.columns) 
            dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=items)
            if selected_target_variable and selected_target_variable in current_df.columns:
                dpg.set_value(TARGET_VARIABLE_COMBO_TAG, selected_target_variable)
            else:
                # If selected_target_variable is no longer in current_df (e.g. due to some operation not here)
                # it should be cleared. For now, this combo just lists available.
                dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "") 
        else:
            dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=[""])
            dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")

# MODIFIED: This function will now be called by Step 1 when its processing is done.
# EDA will use df_after_step1 as its starting point.
# current_df will be a working copy for EDA that can be modified by outlier treatment.
def step1_processing_complete(processed_df: pd.DataFrame):
    global current_df, df_after_step1, _eda_sva_initialized, _eda_outlier_settings_applied_once
    if processed_df is None:
        print("Step 1 returned no DataFrame. Cannot proceed to EDA with new data.")
        # Optionally, clear current_df and df_after_step1 if this implies an error or reset
        # current_df = None 
        # df_after_step1 = None
        return

    df_after_step1 = processed_df.copy()
    current_df = df_after_step1.copy() # EDA works on a copy of Step 1's output
    print(f"DataFrame after Step 1 processing received by main_app. Shape: {current_df.shape}")

    _eda_sva_initialized = False # Reset SVA state for new data
    _eda_outlier_settings_applied_once = False # Reset outlier application state

    # Check if previously selected target variable is still valid
    if selected_target_variable and current_df is not None and selected_target_variable not in current_df.columns:
        print(f"Warning: Previously selected target variable '{selected_target_variable}' not in new DataFrame. Resetting target.")
        selected_target_variable = None
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
    
    update_target_variable_combo() # Update combo with columns from new current_df
    
    # If settings were loaded that included outlier config, try to apply them
    if active_settings and 'step_02_settings' in active_settings and 'outlier_config' in active_settings['step_02_settings']:
        if hasattr(step_02_exploratory_data_analysis, 'apply_outlier_treatment_from_settings'):
            print("Attempting to apply loaded outlier settings...")
            current_df, applied_now = step_02_exploratory_data_analysis.apply_outlier_treatment_from_settings(
                current_df, 
                active_settings['step_02_settings']['outlier_config'], 
                main_app_callbacks
            )
            _eda_outlier_settings_applied_once = applied_now # Update flag based on actual application
            if applied_now:
                print("Outlier settings from session applied to current_df for EDA.")
    
    trigger_all_module_updates() # Update all modules, especially EDA with the new current_df

# ADDED: Function to reset current_df for EDA to the state after Step 1 (and re-apply outliers if configured)
def reset_eda_df_to_after_step1():
    global current_df, df_after_step1, _eda_outlier_settings_applied_once
    if df_after_step1 is None:
        print("Cannot reset EDA DataFrame: No data from Step 1 available.")
        _show_simple_modal_message("Error", "No data from Step 1 to reset to.")
        return
    
    current_df = df_after_step1.copy()
    print("EDA DataFrame reset to the output of Step 1.")
    _eda_outlier_settings_applied_once = False # Reset this flag

    # Re-apply outlier settings if they exist in active_settings and are meant to be active
    if active_settings and 'step_02_settings' in active_settings and 'outlier_config' in active_settings['step_02_settings']:
        outlier_conf = active_settings['step_02_settings']['outlier_config']
        # We need a clear flag if outliers *should* be active, not just configured
        # For now, assume if configured, they should be re-applied after this reset.
        if hasattr(step_02_exploratory_data_analysis, 'apply_outlier_treatment_from_settings'):
            print("Re-applying outlier settings after EDA reset...")
            current_df, applied_now = step_02_exploratory_data_analysis.apply_outlier_treatment_from_settings(
                current_df, 
                outlier_conf, 
                main_app_callbacks
            )
            _eda_outlier_settings_applied_once = applied_now
            if applied_now:
                print("Outlier settings re-applied to EDA DataFrame.")
    
    update_target_variable_combo() # Ensure combo lists columns from the potentially modified current_df
    trigger_specific_module_update(ANALYSIS_STEPS[1]) # Trigger update only for EDA module


def load_data_from_file(file_path: str) -> bool:
    global current_df, original_df, loaded_file_path, selected_target_variable, _eda_sva_initialized, df_after_step1, _eda_outlier_settings_applied_once
    # This function is called by step_01_data_loading when IT loads a file initially.
    # It sets original_df. Step 1 then processes original_df and calls step1_processing_complete().
    success = False
    try:
        # original_df is the raw loaded data.
        original_df = pd.read_parquet(file_path)
        # current_df and df_after_step1 will be set by Step 1's processing.
        # For now, we can make a temporary copy for Step 1 to pick up if it needs one before its UI is fully built.
        # However, the flow should be: this loads original_df, Step 1 module uses original_df.
        current_df = None # Will be set after step 1 processing
        df_after_step1 = None # Will be set after step 1 processing

        loaded_file_path = file_path
        print(f"Raw data loaded successfully: {file_path}, Shape: {original_df.shape}")
        success = True
    except Exception as e:
        current_df = None; original_df = None; loaded_file_path = None; df_after_step1 = None
        print(f"Error loading raw data: {e}")
        traceback.print_exc()
        if dpg.does_item_exist(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT): # step_01_data_loading should handle its own errors
            dpg.set_value(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT, f"Error loading file: {e}")
        success = False

    if success:
        # Clear previous step 1 settings as new raw data is loaded
        if hasattr(step_01_data_loading, '_type_selections'): step_01_data_loading._type_selections.clear()
        if hasattr(step_01_data_loading, '_imputation_selections'): step_01_data_loading._imputation_selections.clear()
        if hasattr(step_01_data_loading, '_custom_nan_input_value'): step_01_data_loading._custom_nan_input_value = ""
        
        _eda_sva_initialized = False 
        _eda_outlier_settings_applied_once = False
        # Target variable related resets if raw data changes significantly
        if selected_target_variable and original_df is not None and selected_target_variable not in original_df.columns:
            selected_target_variable = None 
            if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
        
        # update_target_variable_combo() # This should be based on current_df, which is set after Step 1
        # trigger_all_module_updates() # Step 1 will trigger its own update, then call step1_processing_complete which triggers others
        # Trigger update for Step 1 as it's the one that needs to process the new original_df
        if ANALYSIS_STEPS and ANALYSIS_STEPS[0] in module_ui_updaters:
             trigger_specific_module_update(ANALYSIS_STEPS[0])

        return True
    else: # Failed to load raw data
        _eda_sva_initialized = False 
        _eda_outlier_settings_applied_once = False
        update_target_variable_combo() # Combo will be empty
        selected_target_variable = None 
        global selected_target_variable_type
        selected_target_variable_type = "Continuous" 
        if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG): dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): 
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type)

        trigger_all_module_updates() 
        return False


def target_variable_type_changed_callback(sender, app_data, user_data):
    global selected_target_variable_type, active_settings, current_df, selected_target_variable, _eda_sva_initialized

    newly_selected_type = app_data 
    previous_type = selected_target_variable_type 

    # Validation should use current_df (the one EDA operates on)
    # And s1_column_types (which comes from step_01_data_loading._type_selections)
    if newly_selected_type == "Continuous" and selected_target_variable and current_df is not None:
        # Use the main_app_callbacks to get column types from Step 1
        s1_column_types = main_app_callbacks.get('get_column_analysis_types', lambda: {})()
        analysis_type_from_s1 = s1_column_types.get(selected_target_variable)

        is_text_based = False
        if analysis_type_from_s1: # Check based on Step 1's detailed classification
            text_keywords = ["Text (", "Potentially Sensitive", "Categorical (High Cardinality)"] # Added High Cardinality
            if any(keyword in analysis_type_from_s1 for keyword in text_keywords):
                is_text_based = True
        # Fallback if Step 1 type is not available or not decisive, check dtype
        elif selected_target_variable in current_df.columns and \
             (pd.api.types.is_object_dtype(current_df[selected_target_variable].dtype) or \
              pd.api.types.is_string_dtype(current_df[selected_target_variable].dtype)):
            if analysis_type_from_s1 is None: # Only if S1 didn't classify it otherwise (e.g. as numeric)
                # Consider it text-based if it's object/string and not explicitly numeric/categorical by S1
                is_text_based = True 

        if is_text_based:
            error_message = (f"Variable '{selected_target_variable}' is identified as Text-based or high cardinality categorical.\n"
                             f"It cannot be reliably treated as 'Continuous' for most analyses.\n\n"
                             f"Please use 'Categorical' or verify the variable type in\n"
                             f"'Step 1. Data Loading and Overview' if it's misclassified.")
            _show_simple_modal_message("Type Selection Warning", error_message) # Changed to warning
            
            # Allow the change but with a warning, or revert:
            # For now, let's allow but user is warned. If strict prevention:
            # if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            #     dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, previous_type)
            # return
    
    selected_target_variable_type = newly_selected_type
    print(f"Target variable type explicitly set to: {selected_target_variable_type}")

    if active_settings: 
        active_settings['selected_target_variable_type'] = selected_target_variable_type

    # If EDA is the active step, its state might need to be reset or updated
    _eda_sva_initialized = False # SVA often depends on target type
    if active_step_name == ANALYSIS_STEPS[1]: 
         trigger_specific_module_update(ANALYSIS_STEPS[1]) # Re-render EDA or parts of it


def target_variable_selected_callback(sender, app_data, user_data):
    global selected_target_variable, selected_target_variable_type, _eda_sva_initialized, current_df, active_settings

    new_target = app_data
    if not new_target: # If selection is cleared ("")
        selected_target_variable = None
        selected_target_variable_type = "Continuous" # Reset to default
        print("Target variable selection cleared.")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type)
        if active_settings:
            active_settings['selected_target_variable'] = None
            active_settings['selected_target_variable_type'] = None
    else:
        selected_target_variable = new_target
        print(f"Target variable selected: {selected_target_variable}")

        if selected_target_variable and current_df is not None and selected_target_variable in current_df.columns:
            s1_type_selections = main_app_callbacks.get('get_column_analysis_types', lambda: {})()
            guessed_type = utils._guess_target_type(current_df, selected_target_variable, s1_type_selections) # Util function needs access to current_df
            selected_target_variable_type = guessed_type 

            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
                dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
            if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
                dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)

            if active_settings:
                active_settings['selected_target_variable'] = selected_target_variable
                active_settings['selected_target_variable_type'] = selected_target_variable_type
        # This 'else' for when new_target is not empty but not in current_df should ideally not happen if combo is updated correctly.
        # If it does, it means an invalid state, possibly clear selection. For now, covered by the initial `if not new_target`.

    _eda_sva_initialized = False # SVA state depends on target
    trigger_all_module_updates() # Especially EDA


def get_settings_filepath(original_data_filepath: str) -> str:
    if not original_data_filepath:
        return None
    filename = hashlib.md5(original_data_filepath.encode('utf-8')).hexdigest() + ".json"
    return os.path.join(SETTINGS_DIR_NAME, filename)

def load_json_settings(settings_filepath: str) -> dict:
    if settings_filepath and os.path.exists(settings_filepath):
        try:
            with open(settings_filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading settings from {settings_filepath}: {e}")
            traceback.print_exc()
    return None

def save_json_settings(settings_filepath: str, settings_dict: dict):
    if not settings_filepath or not settings_dict:
        return
    try:
        if not os.path.exists(SETTINGS_DIR_NAME):
            os.makedirs(SETTINGS_DIR_NAME)
        with open(settings_filepath, 'w') as f:
            json.dump(settings_dict, f, indent=4)
        print(f"Settings saved to {settings_filepath}")
    except Exception as e:
        print(f"Error saving settings to {settings_filepath}: {e}")
        traceback.print_exc()

def gather_current_settings() -> dict:
    global selected_target_variable, selected_target_variable_type, active_step_name, \
           _eda_sva_initialized, _eda_outlier_settings_applied_once
    
    settings = {
        'selected_target_variable': selected_target_variable,
        'selected_target_variable_type': selected_target_variable_type, 
        'active_step_name': active_step_name,
        'step_01_settings': {},
        'step_02_settings': { # MODIFIED: EDA settings are now more structured
            'sva_config': {}, # For SVA specific UI like filter strength
            'mva_config': {}, # For MVA UI (correlation, pairplot selectors)
            'outlier_config': {} # For new outlier management settings
        }
    }

    # Step 1 settings
    if hasattr(step_01_data_loading, '_type_selections'):
        settings['step_01_settings']['type_selections'] = step_01_data_loading._type_selections.copy()
    if hasattr(step_01_data_loading, '_imputation_selections'):
        settings['step_01_settings']['imputation_selections'] = step_01_data_loading._imputation_selections.copy()
    if hasattr(step_01_data_loading, '_custom_nan_input_value'):
        settings['step_01_settings']['custom_nan_input_value'] = step_01_data_loading._custom_nan_input_value
    
    # Step 2 EDA settings
    try:
        if dpg.is_dearpygui_running():
            s02_set = settings['step_02_settings']
            sva_conf = s02_set['sva_config']
            mva_conf = s02_set['mva_config']
            outlier_conf = s02_set['outlier_config']

            # SVA settings
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO):
                sva_conf['sva_filter_strength'] = dpg.get_value(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO)
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
                sva_conf['sva_group_by_target'] = dpg.get_value(step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX)
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
                sva_conf['sva_grouped_plot_type'] = dpg.get_value(step_02_exploratory_data_analysis.TAG_SVA_GROUPED_PLOT_TYPE_RADIO)
            sva_conf['_eda_sva_initialized'] = _eda_sva_initialized # Still relevant for SVA part of EDA

            # MVA general settings (selectors that are part of the main EDA UI for MVA)
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR):
                mva_conf['mva_pairplot_vars'] = dpg.get_value(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR)
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO):
                mva_conf['mva_pairplot_hue'] = dpg.get_value(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO)
            if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO): # For Target vs Feature analysis
                mva_conf['mva_target_feature_for_analysis'] = dpg.get_value(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO)
            
            # ADDED: Outlier settings
            if hasattr(step_02_exploratory_data_analysis, 'get_outlier_settings_for_saving'):
                outlier_conf.update(step_02_exploratory_data_analysis.get_outlier_settings_for_saving())
            outlier_conf['_eda_outlier_settings_applied_once'] = _eda_outlier_settings_applied_once


    except Exception as e:
        print(f"Error gathering DPG settings from EDA: {e}")
        traceback.print_exc()
        
    return settings

def apply_settings(settings_dict: dict):
    global current_df, original_df, df_after_step1, selected_target_variable, selected_target_variable_type, \
           active_step_name, _eda_sva_initialized, active_settings, _eda_outlier_settings_applied_once

    if original_df is None: # Raw data must be loaded first
        print("Error in apply_settings: original_df is None. Cannot apply settings without raw data.")
        _show_simple_modal_message("Error", "Cannot apply settings as no raw data (original_df) is loaded. Please load a file.")
        return
    
    print("Applying settings...")
    active_settings = settings_dict # Store the loaded settings globally

    # Restore general app state from settings
    selected_target_variable = settings_dict.get('selected_target_variable', None)
    selected_target_variable_type = settings_dict.get('selected_target_variable_type', "Continuous")
    
    # --- Step 1 Data Application ---
    # current_df and df_after_step1 are critical.
    # The flow: original_df -> Step 1 processing -> df_after_step1 -> EDA current_df (copy of df_after_step1 + outliers)
    
    # 1. Restore Step 1 configurations (type changes, imputations, custom NaNs)
    s01_settings = settings_dict.get('step_01_settings', {})
    if hasattr(step_01_data_loading, 'apply_step1_settings_and_process'):
        # This function in step_01_data_loading should take s01_settings, 
        # apply them to original_df, and then call main_app.step1_processing_complete()
        # which will set df_after_step1 and the initial current_df for EDA.
        print("  Applying Step 1 settings and reprocessing data...")
        step_01_data_loading.apply_step1_settings_and_process(
            original_df, s01_settings, main_app_callbacks
        )
        # After this, df_after_step1 and current_df should be set by step1_processing_complete.
        # If step1_processing_complete doesn't set current_df, we might need to do it here from df_after_step1.
        if df_after_step1 is not None and current_df is None: # Should be handled by step1_processing_complete
             current_df = df_after_step1.copy()
    else:
        # Fallback if step_01_data_loading cannot reprocess from settings directly
        # This implies Step 1 must be manually re-run or its settings are only for UI restoration.
        # For robust state restoration, Step 1 needs to be able to re-apply its transformations.
        print("  Warning: Step 1 module does not have 'apply_step1_settings_and_process'. Step 1 transformations may not be fully restored from settings.")
        # We still need a base for EDA's current_df. If df_after_step1 isn't set by a Step1 re-process,
        # we might have to assume original_df is the base, which is not ideal.
        if df_after_step1 is None:
            df_after_step1 = original_df.copy() # Less ideal, Step 1 changes are lost
            print("  Fallback: df_after_step1 set to original_df. Step 1 changes from settings are not applied to data.")
        current_df = df_after_step1.copy()

    # --- Restore Target Variable UI based on loaded settings and current_df columns ---
    update_target_variable_combo() # Populate with columns from (potentially new) current_df
    if selected_target_variable and current_df is not None and selected_target_variable in current_df.columns:
        dpg.set_value(TARGET_VARIABLE_COMBO_TAG, selected_target_variable)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
            dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=True)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=True)
    elif current_df is not None: # Target not valid or not set
        dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)

    # --- Step 2 EDA Settings Application (UI restoration and data modification for outliers) ---
    s02_settings = settings_dict.get('step_02_settings', {})
    
    # Restore SVA config (mostly UI)
    sva_conf = s02_settings.get('sva_config', {})
    _eda_sva_initialized = sva_conf.get('_eda_sva_initialized', False) # SVA specific flag
    if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO) and 'sva_filter_strength' in sva_conf:
        dpg.set_value(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO, sva_conf['sva_filter_strength'])
    if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX) and 'sva_group_by_target' in sva_conf:
        sva_group_target_val = sva_conf['sva_group_by_target']
        dpg.set_value(step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX, sva_group_target_val)
        if hasattr(step_02_exploratory_data_analysis, '_sva_group_by_target_callback'): # Update dependent UI
            step_02_exploratory_data_analysis._sva_group_by_target_callback(None, sva_group_target_val, main_app_callbacks)
    if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_GROUPED_PLOT_TYPE_RADIO) and 'sva_grouped_plot_type' in sva_conf:
        dpg.set_value(step_02_exploratory_data_analysis.TAG_SVA_GROUPED_PLOT_TYPE_RADIO, sva_conf['sva_grouped_plot_type'])

    # Restore MVA config (UI for selectors)
    mva_conf = s02_settings.get('mva_config', {})
    if current_df is not None: # MVA selectors depend on current_df columns
        numeric_cols_for_mva = step_02_exploratory_data_analysis._get_numeric_cols(current_df) # Helper from EDA
        categorical_cols_for_mva_hue = [""] + step_02_exploratory_data_analysis._get_categorical_cols(current_df, max_unique_for_cat=10)
        all_columns_mva = current_df.columns.tolist()

        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR) and 'mva_pairplot_vars' in mva_conf:
            dpg.configure_item(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=numeric_cols_for_mva)
            valid_vars = [v for v in mva_conf['mva_pairplot_vars'] if v in numeric_cols_for_mva]
            dpg.set_value(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR, valid_vars)
        
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO) and 'mva_pairplot_hue' in mva_conf:
            dpg.configure_item(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO, items=categorical_cols_for_mva_hue)
            if mva_conf['mva_pairplot_hue'] in categorical_cols_for_mva_hue:
                dpg.set_value(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO, mva_conf['mva_pairplot_hue'])
        
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO) and 'mva_target_feature_for_analysis' in mva_conf and selected_target_variable:
            feature_candidates_mva = [col for col in all_columns_mva if col != selected_target_variable]
            dpg.configure_item(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO, items=feature_candidates_mva)
            if mva_conf['mva_target_feature_for_analysis'] in feature_candidates_mva:
                dpg.set_value(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO, mva_conf['mva_target_feature_for_analysis'])

    # ADDED: Apply Outlier settings
    # This is crucial: outlier treatment modifies current_df for EDA.
    # It must happen AFTER current_df is initialized from df_after_step1.
    outlier_conf = s02_settings.get('outlier_config', {})
    _eda_outlier_settings_applied_once = outlier_conf.get('_eda_outlier_settings_applied_once', False) # Restore the flag

    if hasattr(step_02_exploratory_data_analysis, 'apply_outlier_treatment_from_settings'):
        if current_df is not None: # Ensure current_df exists before modifying
            print("  Applying outlier settings to EDA DataFrame...")
            # This function in EDA module should take current_df and outlier_conf,
            # return the modified current_df, and update its own UI for outlier settings.
            modified_df_by_outliers, applied_this_time = step_02_exploratory_data_analysis.apply_outlier_treatment_from_settings(
                current_df.copy(), # Pass a copy to modify
                outlier_conf,
                main_app_callbacks # For EDA to update its own UI components related to outliers
            )
            if applied_this_time: # Only replace current_df if outliers were actually applied
                current_df = modified_df_by_outliers
                print("  Outlier treatment applied based on saved settings. current_df updated.")
                _eda_outlier_settings_applied_once = True # Ensure this flag is correctly set
            else:
                print("  Outlier treatment was configured but not applied (e.g. 'No Removal' or error). current_df unchanged by outliers.")
                # _eda_outlier_settings_applied_once remains as loaded from settings or False
        else:
            print("  Warning: current_df is None, cannot apply outlier settings.")
    else:
        print("  Warning: Step 2 module does not have 'apply_outlier_treatment_from_settings'. Outlier states may not be fully restored.")


    # --- Restore Active Step ---
    restored_active_step = settings_dict.get('active_step_name', ANALYSIS_STEPS[0] if ANALYSIS_STEPS else None)
    if restored_active_step:
        # Ensure the UI for the step (especially EDA) is updated with the potentially modified current_df
        # before switching view. Triggering all updates might be broad but ensures consistency here.
        trigger_all_module_updates() # Crucial after all data and UI states are set

        if restored_active_step in step_group_tags and dpg.does_item_exist(step_group_tags[restored_active_step]):
            switch_step_view(None, None, restored_active_step) 
            active_step_name = restored_active_step # Ensure active_step_name is correctly set
            print(f"  Restored active step: {active_step_name}")
        else: # Fallback to first step if restored one is invalid
            print(f"  Could not restore active step '{restored_active_step}'. Defaulting to first available step.")
            if ANALYSIS_STEPS and ANALYSIS_STEPS[0] in step_group_tags and dpg.does_item_exist(step_group_tags[ANALYSIS_STEPS[0]]):
                switch_step_view(None, None, ANALYSIS_STEPS[0])
                active_step_name = ANALYSIS_STEPS[0]
    else: # No active step in settings, default to first
        trigger_all_module_updates()
        if ANALYSIS_STEPS and ANALYSIS_STEPS[0] in step_group_tags and dpg.does_item_exist(step_group_tags[ANALYSIS_STEPS[0]]):
            switch_step_view(None, None, ANALYSIS_STEPS[0])
            active_step_name = ANALYSIS_STEPS[0]
            print(f"  Defaulted to active step: {active_step_name}")

    print("Settings application process finished.")


def reset_application_state(clear_df_completely=True): # clear_df_completely to also nullify original_df
    global current_df, original_df, df_after_step1, loaded_file_path, selected_target_variable, \
           selected_target_variable_type, active_step_name, _eda_sva_initialized, active_settings, \
           _eda_outlier_settings_applied_once
    
    print("Resetting application state...")
    if clear_df_completely: # Full reset, e.g. loading a brand new file or initial startup with no session
        current_df = None
        original_df = None
        df_after_step1 = None
        loaded_file_path = None
        active_settings = {} # Clear all previously loaded/active settings
    else: # Resetting for a new file load, keep original_df if it's about to be replaced
          # Or, if it's a project reset, original_df might be kept if we want to "revert" to it.
          # For a "soft" reset (like changing a file), loaded_file_path is kept until new one is set.
          # This function is now more about resetting states derived from original_df.
        current_df = None # Will be repopulated after Step 1 processes original_df (or its new version)
        df_after_step1 = None
        # active_settings might be preserved if we are just reloading the same file's settings
        # but for a generic reset, clearing them is safer unless managed carefully by the caller.
        # For now, let's assume reset_application_state implies clearing most derived states.
        # If called from file_load_callback, active_settings will be re-evaluated (loaded or gathered).

    selected_target_variable = None
    selected_target_variable_type = "Continuous" 
    _eda_sva_initialized = False
    _eda_outlier_settings_applied_once = False

    # Reset Step 1 internal states
    if hasattr(step_01_data_loading, 'reset_step1_state'): # Ideal
        step_01_data_loading.reset_step1_state()
    else: # Manual reset of known Step 1 attributes
        if hasattr(step_01_data_loading, '_type_selections'): step_01_data_loading._type_selections.clear()
        if hasattr(step_01_data_loading, '_imputation_selections'): step_01_data_loading._imputation_selections.clear()
        if hasattr(step_01_data_loading, '_custom_nan_input_value'): step_01_data_loading._custom_nan_input_value = ""
        if dpg.does_item_exist(step_01_data_loading.TAG_DL_CUSTOM_NAN_INPUT): # Reset UI
            dpg.set_value(step_01_data_loading.TAG_DL_CUSTOM_NAN_INPUT, "")
        # Add reset for other Step 1 UI elements if necessary (e.g., clear tables, messages)
        if dpg.does_item_exist(step_01_data_loading.TAG_DL_COLUMN_CONFIG_TABLE_GROUP):
             dpg.delete_item(step_01_data_loading.TAG_DL_COLUMN_CONFIG_TABLE_GROUP, children_only=True)
             dpg.add_text("Load data to configure.", parent=step_01_data_loading.TAG_DL_COLUMN_CONFIG_TABLE_GROUP)


    # Reset global target variable UI
    if dpg.does_item_exist(TARGET_VARIABLE_COMBO_TAG):
        dpg.configure_item(TARGET_VARIABLE_COMBO_TAG, items=[""])
        dpg.set_value(TARGET_VARIABLE_COMBO_TAG, "")
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG):
        dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
    if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG):
        dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
        dpg.set_value(TARGET_VARIABLE_TYPE_RADIO_TAG, selected_target_variable_type) 

    # Reset Step 2 EDA UI elements to defaults
    if hasattr(step_02_exploratory_data_analysis, 'reset_eda_ui_defaults'):
        step_02_exploratory_data_analysis.reset_eda_ui_defaults()
    else: # Manual reset if specialized function not available
        # SVA specific
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO):
            dpg.set_value(step_02_exploratory_data_analysis.TAG_SVA_FILTER_STRENGTH_RADIO, "Weak (Exclude obvious non-analytical)")
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX):
            dpg.set_value(step_02_exploratory_data_analysis.TAG_SVA_GROUP_BY_TARGET_CHECKBOX, False)
            if hasattr(step_02_exploratory_data_analysis, '_sva_group_by_target_callback'): # Sync dependent UI
                step_02_exploratory_data_analysis._sva_group_by_target_callback(None, False, main_app_callbacks)
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_GROUPED_PLOT_TYPE_RADIO):
            dpg.set_value(step_02_exploratory_data_analysis.TAG_SVA_GROUPED_PLOT_TYPE_RADIO, "KDE") # Default, show=False handled by callback
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_SVA_RESULTS_CHILD_WINDOW):
             dpg.delete_item(step_02_exploratory_data_analysis.TAG_SVA_RESULTS_CHILD_WINDOW, children_only=True)
             dpg.add_text("Select filter options and click 'Run Single Variable Analysis'.", parent=step_02_exploratory_data_analysis.TAG_SVA_RESULTS_CHILD_WINDOW)


        # MVA specific
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR):
            dpg.configure_item(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_VAR_SELECTOR, items=[])
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO):
            dpg.configure_item(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO, items=[""])
            dpg.set_value(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_HUE_COMBO, "")
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO):
            dpg.configure_item(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO, items=[])
            dpg.set_value(step_02_exploratory_data_analysis.TAG_MVA_TARGET_FEATURE_COMBO, "")
        # Clear MVA result areas
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_CORR_OUTPUT_GROUP):
            dpg.delete_item(step_02_exploratory_data_analysis.TAG_MVA_CORR_OUTPUT_GROUP, children_only=True)
            dpg.add_text("Run Correlation Analysis to see results.", parent=step_02_exploratory_data_analysis.TAG_MVA_CORR_OUTPUT_GROUP)
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_RESULTS_GROUP):
            dpg.delete_item(step_02_exploratory_data_analysis.TAG_MVA_PAIRPLOT_RESULTS_GROUP, children_only=True)
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_TARGET_RESULTS_GROUP):
            dpg.delete_item(step_02_exploratory_data_analysis.TAG_MVA_TARGET_RESULTS_GROUP, children_only=True)
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_CAT_EDA_RESULTS_GROUP): # New Cat EDA
            dpg.delete_item(step_02_exploratory_data_analysis.TAG_MVA_CAT_EDA_RESULTS_GROUP, children_only=True)
        if dpg.does_item_exist(step_02_exploratory_data_analysis.TAG_MVA_OUTLIER_RESULTS_TEXT): # New Outlier
            dpg.set_value(step_02_exploratory_data_analysis.TAG_MVA_OUTLIER_RESULTS_TEXT, "Apply outlier treatment to see effects.")


    if not clear_df_completely and original_df is not None:
        # If not a full clear, and original_df exists (e.g. new file loaded), Step 1 needs to run
        print("Triggering Step 1 update as part of reset (original_df exists).")
        trigger_specific_module_update(ANALYSIS_STEPS[0])
        # step1_processing_complete will be called by Step 1, which then triggers EDA update.
    else:
        trigger_all_module_updates() # If full clear, update all modules to reflect empty state.
    
    if ANALYSIS_STEPS: # Ensure a view is active
        current_active_step = active_step_name
        if not current_active_step or current_active_step not in step_group_tags or not dpg.does_item_exist(step_group_tags.get(current_active_step)):
             first_step = ANALYSIS_STEPS[0]
             if first_step in step_group_tags and dpg.does_item_exist(step_group_tags[first_step]):
                switch_step_view(None, None, first_step) # This also sets active_step_name

def file_load_callback(sender, app_data):
    global loaded_file_path, original_df, current_df, df_after_step1, active_settings

    new_file_selected_path = app_data.get('file_path_name')
    if not new_file_selected_path:
        print("File selection cancelled.")
        return

    # 1. Save settings for the *currently loaded* file (if any)
    if loaded_file_path and active_settings: # active_settings should reflect the live state
        print(f"Saving settings for previously loaded file: {loaded_file_path}")
        old_settings_filepath = get_settings_filepath(loaded_file_path)
        current_live_settings = gather_current_settings() # Gather live state before reset
        save_json_settings(old_settings_filepath, current_live_settings)

    # 2. Reset application state for the new file, but don't clear original_df yet,
    # as it will be replaced by the new file's data.
    reset_application_state(clear_df_completely=False) # Soft reset, preserves loaded_file_path until new one is confirmed

    # 3. Load the new file into original_df
    print(f"Attempting to load new file: {new_file_selected_path}")
    try:
        new_original_df = pd.read_parquet(new_file_selected_path)
        original_df = new_original_df # Successfully loaded, now assign to global
        loaded_file_path = new_file_selected_path # Update global path
        
        # df_after_step1 and current_df will be set after Step 1 processes this new original_df.
        df_after_step1 = None
        current_df = None

        print(f"New raw data loaded successfully: {loaded_file_path}, Shape: {original_df.shape}")
        # Update Step 1's file summary text directly (as it's the first point of contact for the new file)
        if dpg.does_item_exist(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT):
             dpg.set_value(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT, f"File: {os.path.basename(loaded_file_path)}, Shape: {original_df.shape}")

    except Exception as e:
        original_df = None; current_df = None; df_after_step1 = None; # loaded_file_path remains the old one or None
        # Do not update loaded_file_path if new load fails.
        print(f"Error loading data from {new_file_selected_path}: {e}")
        traceback.print_exc()
        _show_simple_modal_message("File Load Error", f"Failed to load {os.path.basename(new_file_selected_path)}:\n{e}")
        # Since new file failed, Step 1 summary should reflect error or revert to "no file loaded"
        if dpg.does_item_exist(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT):
            dpg.set_value(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT, f"Error loading file. Please try again.")
        
        # Restore settings of the *previous* file if new load failed, if possible.
        # This is complex. Simpler: state is now "no file" or "error".
        # Or, try to reload settings for the *previously successfully loaded* file if it exists.
        # For now, a clean reset and trigger updates to reflect failure.
        reset_application_state(clear_df_completely=True) # Full reset on error
        trigger_all_module_updates() # Show error state in UI
        if ANALYSIS_STEPS: switch_step_view(None, None, ANALYSIS_STEPS[0]) # Go to first step
        return

    # 4. Load settings for the *newly loaded* file (if they exist)
    new_settings_filepath = get_settings_filepath(loaded_file_path)
    loaded_specific_settings = load_json_settings(new_settings_filepath)
    
    if loaded_specific_settings:
        print(f"Found existing settings for {loaded_file_path}. Applying them.")
        # apply_settings will handle reprocessing Step 1 and then EDA based on these settings.
        # It internally calls step_01_data_loading.apply_step1_settings_and_process,
        # which then calls step1_processing_complete, setting df_after_step1 and current_df.
        # Then apply_settings continues with EDA settings.
        apply_settings(loaded_specific_settings)
    else:
        print(f"No settings found for {loaded_file_path}. Initializing with default state after loading raw data.")
        active_settings = {} # Clear any old active_settings; new ones will be default generated by Step 1
        # Trigger Step 1 to process the new original_df with its default settings.
        # Step 1's update_ui (called via trigger_specific_module_update) should handle this.
        # It will then call step1_processing_complete.
        if hasattr(step_01_data_loading, 'process_newly_loaded_data'):
            step_01_data_loading.process_newly_loaded_data(original_df, main_app_callbacks)
        else: # Fallback
            trigger_specific_module_update(ANALYSIS_STEPS[0]) # Make Step 1 process the new original_df

        # Default UI state for target var if no settings
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
        if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)
        
        # After Step 1 processes (via above trigger or specific call), df_after_step1 and current_df are set.
        # Then EDA needs to be updated. This might already be covered by apply_settings flow or needs explicit trigger.
        # If no specific settings, a general update after Step 1 has run is good.
        # This is usually handled by step1_processing_complete -> trigger_all_module_updates.


    # 5. Final UI updates
    # update_target_variable_combo() is called within step1_processing_complete now.
    # trigger_all_module_updates() is also called by step1_processing_complete or by apply_settings.

    # 6. Ensure the view is on the first step (Data Loading) after a new file is loaded.
    if ANALYSIS_STEPS:
        switch_step_view(None, None, ANALYSIS_STEPS[0])
        active_step_name = ANALYSIS_STEPS[0] # Explicitly set

def initial_load_on_startup():
    global loaded_file_path, active_settings, original_df, current_df, df_after_step1, \
           selected_target_variable, selected_target_variable_type, _eda_outlier_settings_applied_once

    print("Attempting initial load on startup...")
    session_info = load_json_settings(SESSION_INFO_FILE)
    last_file_path_from_session = None
    if session_info:
        last_file_path_from_session = session_info.get('last_opened_original_file')

    if last_file_path_from_session and os.path.exists(last_file_path_from_session):
        print(f"Restoring last session for file: {last_file_path_from_session}")
        # reset_application_state(clear_df_completely=False) # Soft reset initially

        try:
            original_df = pd.read_parquet(last_file_path_from_session)
            loaded_file_path = last_file_path_from_session # Set global path
            print(f"Raw data for last session loaded: {loaded_file_path}, shape: {original_df.shape}")

            # Update Step 1 file summary
            if dpg.does_item_exist(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT):
                 dpg.set_value(step_01_data_loading.TAG_DL_FILE_SUMMARY_TEXT, f"File: {os.path.basename(loaded_file_path)}, Shape: {original_df.shape}")

            settings_for_file = get_settings_filepath(last_file_path_from_session)
            specific_file_settings = load_json_settings(settings_for_file)
            
            if specific_file_settings:
                active_settings = specific_file_settings # Make them active before applying
                print("Applying settings from last session...")
                apply_settings(specific_file_settings) # This will process Step 1 and then EDA
                # apply_settings internally calls step1_processing_complete, which updates df_after_step1 and current_df,
                # and also handles outlier application if configured.
            else: 
                print("No specific settings found for the last loaded file. Processing with defaults.")
                active_settings = {} # No specific settings to make active.
                # Trigger Step 1 to process original_df with defaults.
                if hasattr(step_01_data_loading, 'process_newly_loaded_data'):
                    step_01_data_loading.process_newly_loaded_data(original_df, main_app_callbacks)
                else:
                    trigger_specific_module_update(ANALYSIS_STEPS[0])
                # UI for target var
                if dpg.does_item_exist(TARGET_VARIABLE_TYPE_LABEL_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_LABEL_TAG, show=False)
                if dpg.does_item_exist(TARGET_VARIABLE_TYPE_RADIO_TAG): dpg.configure_item(TARGET_VARIABLE_TYPE_RADIO_TAG, show=False)

            # _eda_outlier_settings_applied_once should be set by apply_settings or step1_processing_complete
            # selected_target_variable and type are also set by apply_settings.
            # update_target_variable_combo is called within apply_settings or subsequent updates.
            # trigger_all_module_updates is also part of apply_settings.

            # Active step is restored by apply_settings. If not, default.
            if not active_step_name and ANALYSIS_STEPS:
                 switch_step_view(None, None, ANALYSIS_STEPS[0])
            return True

        except Exception as e:
            print(f"Error restoring last session for {last_file_path_from_session}: {e}")
            traceback.print_exc()
            _show_simple_modal_message("Session Restore Error", f"Could not restore session for {os.path.basename(last_file_path_from_session)}:\n{e}")
            reset_application_state(clear_df_completely=True) 
            trigger_all_module_updates() 
            if ANALYSIS_STEPS: switch_step_view(None, None, ANALYSIS_STEPS[0])
            return False
    else:
        if last_file_path_from_session and not os.path.exists(last_file_path_from_session):
            print(f"Last session file '{last_file_path_from_session}' not found. Starting fresh.")
        else:
            print("No last session info found. Starting fresh.")
        reset_application_state(clear_df_completely=True) 
        # active_settings = gather_current_settings() # Gathers empty/default state
        active_settings = {} # Start with no active settings
        trigger_all_module_updates() 
        if ANALYSIS_STEPS: 
            switch_step_view(None, None, ANALYSIS_STEPS[0])
        return False

def save_state_on_exit():
    global loaded_file_path, active_settings # active_settings should be current state

    print("Saving state on exit...")
    if loaded_file_path and os.path.exists(loaded_file_path): # Ensure file still exists
        current_settings_on_exit = gather_current_settings() # Get the live state
        
        # Check if current_settings_on_exit is substantially different from active_settings
        # This is to avoid re-saving if nothing changed. For simplicity, always save for now.
        
        settings_filepath = get_settings_filepath(loaded_file_path)
        save_json_settings(settings_filepath, current_settings_on_exit)
        
        session_info_to_save = {'last_opened_original_file': loaded_file_path}
        save_json_settings(SESSION_INFO_FILE, session_info_to_save)
        print(f"Last session info and current file settings saved for: {loaded_file_path}")
    elif loaded_file_path and not os.path.exists(loaded_file_path):
        print(f"Warning: Loaded file path '{loaded_file_path}' does not exist anymore. Cannot save its settings. Clearing last session info.")
        if os.path.exists(SESSION_INFO_FILE):
            try: os.remove(SESSION_INFO_FILE)
            except OSError as e: print(f"Error removing session file: {e}")
    else:
        print("No active file loaded or file path is invalid. Nothing to save as last session.")
        # Optionally clear session info if desired when exiting with no file
        # if os.path.exists(SESSION_INFO_FILE):
        #     try: os.remove(SESSION_INFO_FILE)
        #     except OSError as e: print(f"Error removing session file: {e}")


def reset_current_df_to_original_data(): # This should be "Reset Step 1 and EDA to use original_df"
    global current_df, original_df, df_after_step1, selected_target_variable, selected_target_variable_type, \
           _eda_sva_initialized, _eda_outlier_settings_applied_once, active_settings
    
    if original_df is None:
        _show_simple_modal_message("Error", "No original data loaded to reset to.")
        print("No original data to reset to.")
        return

    print("Resetting to use original raw data: Step 1 will be reset and reprocessed, then EDA will update.")
    
    # 1. Clear active_settings related to transformations, as we are starting fresh from original_df
    #    Alternatively, one might want to keep target var selection. For a full data reset, clear most.
    active_settings = {
        'selected_target_variable': selected_target_variable, # Preserve user's target choice if any
        'selected_target_variable_type': selected_target_variable_type,
        'active_step_name': active_step_name,
        'step_01_settings': {}, # Clear Step 1 specific settings
        'step_02_settings': { # Clear EDA specific settings like filters, outlier choices
            'sva_config': {},
            'mva_config': {},
            'outlier_config': {}
        }
    }
    # If target var was from a processed column that won't exist in raw original_df, it needs to be cleared.
    if selected_target_variable and selected_target_variable not in original_df.columns:
        selected_target_variable = None
        selected_target_variable_type = "Continuous"
        active_settings['selected_target_variable'] = None
        active_settings['selected_target_variable_type'] = "Continuous"

    # 2. Reset Step 1 module's internal state and UI to defaults
    if hasattr(step_01_data_loading, 'reset_step1_state'):
        step_01_data_loading.reset_step1_state()
    else: # Manual reset
        if hasattr(step_01_data_loading, '_type_selections'): step_01_data_loading._type_selections.clear()
        if hasattr(step_01_data_loading, '_imputation_selections'): step_01_data_loading._imputation_selections.clear()
        if hasattr(step_01_data_loading, '_custom_nan_input_value'): step_01_data_loading._custom_nan_input_value = ""
        if dpg.does_item_exist(step_01_data_loading.TAG_DL_CUSTOM_NAN_INPUT):
            dpg.set_value(step_01_data_loading.TAG_DL_CUSTOM_NAN_INPUT, "")
        # Reset column config table in Step 1 UI
        if dpg.does_item_exist(step_01_data_loading.TAG_DL_COLUMN_CONFIG_TABLE_GROUP):
            dpg.delete_item(step_01_data_loading.TAG_DL_COLUMN_CONFIG_TABLE_GROUP, children_only=True)
            # Step 1's update_ui should rebuild this based on original_df.
            # dpg.add_text("Data reset. Step 1 will re-evaluate.", parent=step_01_data_loading.TAG_DL_COLUMN_CONFIG_TABLE_GROUP)

    # 3. Reset EDA states
    _eda_sva_initialized = False
    _eda_outlier_settings_applied_once = False
    if hasattr(step_02_exploratory_data_analysis, 'reset_eda_ui_defaults'):
        step_02_exploratory_data_analysis.reset_eda_ui_defaults()
        # This should also clear EDA result areas.

    # 4. Trigger Step 1 to process the original_df with default settings
    # This will eventually call step1_processing_complete, which sets df_after_step1, current_df,
    # and then triggers updates for other modules including EDA.
    print("  Triggering Step 1 to re-process original_df with defaults...")
    if hasattr(step_01_data_loading, 'process_newly_loaded_data'): # Ideal function in Step 1
        step_01_data_loading.process_newly_loaded_data(original_df, main_app_callbacks)
    else: # Fallback: trigger general update for Step 1
        trigger_specific_module_update(ANALYSIS_STEPS[0])

    # Target variable UI update will happen as part of the triggered updates once current_df is established.
    # Ensure the view is switched to Step 1
    switch_step_view(None, None, ANALYSIS_STEPS[0])
    print("Data reset to original, Step 1 processing initiated with defaults.")


def switch_step_view(sender, app_data, user_data_step_name: str):
    global active_step_name, _eda_sva_initialized # _eda_sva_initialized might be deprecated or moved into EDA module's state
    print(f"Attempting to switch to step: {user_data_step_name}")

    # Before switching, if leaving Step 1, ensure its current state (transformations) is applied
    # and df_after_step1 is up-to-date. This is now handled by Step 1's "Apply Changes" button
    # which calls step1_processing_complete.

    for step_name_iter, group_tag in step_group_tags.items():
        if dpg.does_item_exist(group_tag):
            if step_name_iter == user_data_step_name:
                dpg.show_item(group_tag)
                active_step_name = step_name_iter
                # When switching TO a step, trigger its UI update.
                # This ensures it has the latest data (e.g. current_df for EDA).
                trigger_specific_module_update(step_name_iter)
            else:
                dpg.hide_item(group_tag)
    print(f"Active step: {active_step_name}")


def trigger_specific_module_update(module_name_key: str):
    global current_df, original_df, df_after_step1
    if module_name_key in module_ui_updaters:
        updater = module_ui_updaters[module_name_key]
        # util_funcs_to_pass는 ANALYSIS_STEPS[0] 또는 util_funcs를 필요로 하는 다른 모듈을 위해 준비합니다.
        util_funcs_to_pass = main_app_callbacks.get('get_util_funcs', lambda: {})()

        if module_name_key == ANALYSIS_STEPS[0]: # "1. Data Loading and Overview"
            # 현재 오류에 따르면, 이 단계의 updater는 3개의 인자를 필요로 합니다.
            updater(original_df, main_app_callbacks, util_funcs_to_pass)
        elif module_name_key == ANALYSIS_STEPS[1]: # "2. Exploratory Data Analysis (EDA)"
            # 이전 오류 및 step_02_exploratory_data_analysis.py 파일 내용에 따르면,
            # 이 단계의 updater(update_ui)는 2개의 인자만 필요로 합니다.
            if current_df is not None:
                 updater(current_df, main_app_callbacks)
            else:
                 updater(None, main_app_callbacks)
        else:
             # 다른 모듈이 있다면, 해당 모듈의 updater 함수 정의에 맞게 호출해야 합니다.
             # 예시로 2개 인자를 받는다고 가정합니다. 필요시 수정하세요.
             updater(current_df, main_app_callbacks)
        print(f"Module UI updated for: '{module_name_key}'")
    else:
        print(f"Warning: No UI updater found for '{module_name_key}'.")


def trigger_all_module_updates():
    # global _eda_sva_initialized # This flag might be managed within EDA module now
    print("Updating all module UIs...")
    for step_name_key in list(module_ui_updaters.keys()): 
        trigger_specific_module_update(step_name_key)


util_functions_for_modules = {
    'create_table_with_data': utils.create_table_with_data,
    'calculate_column_widths': utils.calculate_column_widths,
    'format_text_for_display': utils.format_text_for_display,
    'get_safe_text_size': utils.get_safe_text_size,
    '_show_simple_modal_message': _show_simple_modal_message, # ADDED: Make modal accessible to modules
}

# This function primarily sources type information from Step 1's user-defined settings.
def get_column_analysis_types_from_user_settings():
    if hasattr(step_01_data_loading, '_type_selections') and \
       isinstance(step_01_data_loading._type_selections, dict) and \
       step_01_data_loading._type_selections: # Check if not empty
        return step_01_data_loading._type_selections.copy() 
    else:
        # Fallback: If Step 1 selections are empty or not available,
        # try to infer from current_df (if available and if step_01_data_loading has an inferrer).
        # This is a secondary measure. Primary truth should be user settings from Step 1.
        print("Warning: User-defined analysis types (_type_selections) from Step 1 are not available or empty.")
        if current_df is not None and not current_df.empty:
            if hasattr(step_01_data_loading, 'infer_column_types_for_display'): # Check for a specific inference function in Step 1
                # This function should ideally return a dictionary similar to _type_selections format
                # e.g., {'col_name': 'Numeric (Integer)', ...}
                inferred_types = step_01_data_loading.infer_column_types_for_display(current_df)
                if inferred_types: return inferred_types

            # If specific inference isn't available from Step 1, use basic pandas dtypes as a last resort.
            print("Fallback: Using basic pandas dtypes for column analysis types.")
            return {col: str(current_df[col].dtype) for col in current_df.columns}
        return {}


main_app_callbacks = {
    'get_current_df': lambda: current_df, # EDA uses this (df_after_step1 + outliers)
    'get_original_df': lambda: original_df, # Step 1 uses this
    'get_df_after_step1': lambda: df_after_step1, # EDA might need this to revert outlier changes without full reset
    'get_loaded_file_path': lambda: loaded_file_path,
    'get_util_funcs': lambda: util_functions_for_modules,
    'show_file_dialog': lambda: dpg.show_item("file_dialog_id"),
    'register_step_group_tag': lambda name, tag: step_group_tags.update({name: tag}),
    'register_module_updater': lambda name, func: module_ui_updaters.update({name: func}),
    'trigger_module_update': trigger_specific_module_update, # Specific module
    'reset_current_df_to_original': reset_current_df_to_original_data, # Full reset to raw original_df
    'reset_eda_df_to_after_step1': reset_eda_df_to_after_step1, # ADDED: Reset EDA's current_df to df_after_step1
    'trigger_all_module_updates': trigger_all_module_updates, 
    'get_selected_target_variable': lambda: selected_target_variable,
    'get_selected_target_variable_type': lambda: selected_target_variable_type,
    'update_target_variable_combo_items': update_target_variable_combo, # Usually called internally when df changes
    'get_column_analysis_types': get_column_analysis_types_from_user_settings, # Gets Step 1's column type decisions
    'step1_processing_complete': step1_processing_complete, # Called by Step 1 when it's done
    # ADDED: Callback for EDA to notify main_app of changes to current_df (e.g., after outlier removal)
    'notify_eda_df_changed': lambda new_eda_df: globals().update(current_df=new_eda_df, _eda_outlier_settings_applied_once=True),
    'get_eda_outlier_applied_flag': lambda: _eda_outlier_settings_applied_once, # ADDED
    'set_eda_outlier_applied_flag': lambda flag_val: globals().update(_eda_outlier_settings_applied_once=flag_val), # ADDED
}


dpg.create_context()
# ADDED: Texture registry for seaborn/matplotlib plots
# dpg.add_texture_registry(tag="texture_registry") # DPG < 1.10
# For DPG 1.10+, texture registry is implicitly available, or can be added if needed for specific management
# No explicit add_texture_registry needed for simple add_static_texture / add_dynamic_texture.

with dpg.file_dialog(directory_selector=False, show=False, callback=file_load_callback,
                     id="file_dialog_id", width=700, height=400, modal=True):
    dpg.add_file_extension(".parquet"); dpg.add_file_extension(".*")

setup_korean_font()

with dpg.window(label="Data Analysis Platform", tag="main_window"):
    with dpg.group(horizontal=True):
        with dpg.child_window(width=280, tag="navigation_panel", border=True):
            dpg.add_text("Target Variable (y):")
            dpg.add_combo(items=[""], tag=TARGET_VARIABLE_COMBO_TAG, width=-1,
                        callback=target_variable_selected_callback) 

            dpg.add_text("Target Variable Type:", tag=TARGET_VARIABLE_TYPE_LABEL_TAG, show=False) 
            dpg.add_radio_button(
                items=["Categorical", "Continuous"],
                tag=TARGET_VARIABLE_TYPE_RADIO_TAG,
                horizontal=True,
                default_value=selected_target_variable_type, 
                callback=target_variable_type_changed_callback, 
                show=False 
            )

            dpg.add_separator()
            dpg.add_spacer(height=5)
            dpg.add_text("Analysis Steps", color=[255, 255, 0]); dpg.add_separator()
            for step_name_nav in ANALYSIS_STEPS:
                dpg.add_button(label=step_name_nav, callback=switch_step_view,
                              user_data=step_name_nav, width=-1, height=30)
        
        with dpg.child_window(tag="content_area", border=True):
            module_map = {
                ANALYSIS_STEPS[0]: step_01_data_loading,
                ANALYSIS_STEPS[1]: step_02_exploratory_data_analysis,
            }
            for step_name_create in ANALYSIS_STEPS:
                module = module_map.get(step_name_create)
                if module and hasattr(module, 'create_ui'):
                    module.create_ui(step_name_create, "content_area", main_app_callbacks)
                    print(f"UI created for '{step_name_create}'.")
                else: 
                    fallback_tag = f"{step_name_create.lower().replace(' ', '_').replace('.', '').replace('&', 'and').replace('(', '').replace(')', '')}_fallback_group" # Sanitize name for tag
                    if not dpg.does_item_exist(fallback_tag): # Check before creating
                        main_app_callbacks['register_step_group_tag'](step_name_create, fallback_tag)
                        with dpg.group(tag=fallback_tag, parent="content_area", show=False): # Ensure parent is content_area
                            dpg.add_text(f"--- {step_name_create} ---"); dpg.add_separator()
                            dpg.add_text(f"UI for '{step_name_create}' (fallback) will be configured here.")
                        # Register a dummy updater if no real UI creation
                        main_app_callbacks['register_module_updater'](step_name_create, 
                            lambda df, mc, sn=step_name_create: print(f"Dummy updater for {sn} called."))


            if ANALYSIS_STEPS: # Default to showing the first step
                first_step = ANALYSIS_STEPS[0]
                if not active_step_name : # If no active step set by initial_load (e.g. fresh start)
                    if first_step in step_group_tags and dpg.does_item_exist(step_group_tags[first_step]):
                        switch_step_view(None, None, first_step)
                    else:
                        print(f"Warning: First step '{first_step}' UI group not found immediately after creation for default view.")


dpg.create_viewport(title='Modular Data Analysis Platform GUI', width=1600, height=1000) # Increased width slightly
dpg.set_exit_callback(save_state_on_exit)
dpg.setup_dearpygui()
initial_load_on_startup() # This will load data and settings, and trigger UI updates including active step
dpg.maximize_viewport()
dpg.show_viewport()
dpg.set_primary_window("main_window", True)
dpg.start_dearpygui()
dpg.destroy_context()