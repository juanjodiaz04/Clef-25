import multiprocessing
import os
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import gradio as gr
import librosa
import webview

import birdnet_analyzer.config as cfg
import birdnet_analyzer.localization as loc
import birdnet_analyzer.utils as utils

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # divert stdout & stderr to logs.txt file since we have no console when deployed
    userdir = Path.home()

    if sys.platform == "win32":
        userdir /= "AppData/Roaming"
    elif sys.platform == "linux":
        userdir /= ".local/share"
    elif sys.platform == "darwin":
        userdir /= "Library/Application Support"

    APPDIR = userdir / "BirdNET-Analyzer-GUI"

    APPDIR.mkdir(parents=True, exist_ok=True)

    sys.stderr = sys.stdout = open(str(APPDIR / "logs.txt"), "a")
    cfg.ERROR_LOG_FILE = str(APPDIR / cfg.ERROR_LOG_FILE)
    FROZEN = True
else:
    FROZEN = False

loc.load_local_state()

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ORIGINAL_TRANSLATED_LABELS_PATH = str(Path(SCRIPT_DIR).parent / cfg.TRANSLATED_LABELS_PATH)
LANG_DIR = str(Path(SCRIPT_DIR).parent / "lang")
_CUSTOM_SPECIES = loc.localize("species-list-radio-option-custom-list")
_PREDICT_SPECIES = loc.localize("species-list-radio-option-predict-list")
_CUSTOM_CLASSIFIER = loc.localize("species-list-radio-option-custom-classifier")
_ALL_SPECIES = loc.localize("species-list-radio-option-all")
_WINDOW: webview.Window = None


# Nishant - Following two functions (select_folder andget_files_and_durations) are written for Folder selection
def select_folder(state_key=None):
    if sys.platform == "win32":
        from tkinter import Tk, filedialog

        tk = Tk()
        tk.withdraw()

        initial_dir = loc.get_state(state_key, None) if state_key else None
        folder_selected = filedialog.askdirectory(initialdir=initial_dir)

        tk.destroy()
    else:
        initial_dir = loc.get_state(state_key, "") if state_key else ""
        dirname = _WINDOW.create_file_dialog(webview.FOLDER_DIALOG, directory=initial_dir)
        folder_selected = dirname[0] if dirname else None

    if folder_selected and state_key:
        loc.set_state(state_key, folder_selected)

    return folder_selected


def get_files_and_durations(folder, max_files=None):
    files_and_durations = []
    files = utils.collect_audio_files(folder, max_files=max_files)  # Use the collect_audio_files function

    for file_path in files:
        try:
            duration = format_seconds(librosa.get_duration(filename=file_path))

        except Exception as _:
            duration = "0:00"  # Default value in case of an error

        files_and_durations.append([os.path.relpath(file_path, folder), duration])
    return files_and_durations


def set_window(window):
    global _WINDOW
    _WINDOW = window


def validate(value, msg):
    """Checks if the value ist not falsy.

    If the value is falsy, an error will be raised.

    Args:
        value: Value to be tested.
        msg: Message in case of an error.
    """
    if not value:
        raise gr.Error(msg)


def format_seconds(secs: float):
    """Formats a number of seconds into a string.

    Formats the seconds into the format "h:mm:ss.ms"

    Args:
        secs: Number of seconds.

    Returns:
        A string with the formatted seconds.
    """
    hours, secs = divmod(secs, 3600)
    minutes, secs = divmod(secs, 60)

    return f"{hours:2.0f}:{minutes:02.0f}:{secs:06.3f}"


def select_directory(collect_files=True, max_files=None, state_key=None):
    """Shows a directory selection system dialog.

    Uses the pywebview to create a system dialog.

    Args:
        collect_files: If True, also lists a files inside the directory.

    Returns:
        If collect_files==True, returns (directory path, list of (relative file path, audio length))
        else just the directory path.
        All values will be None of the dialog is cancelled.
    """
    dir_name = select_folder(state_key=state_key)

    if collect_files:
        if not dir_name:
            return None, None

        files = utils.collect_audio_files(dir_name, max_files=max_files)

        return dir_name, [
            [os.path.relpath(file, dir_name), format_seconds(librosa.get_duration(filename=file))] for file in files
        ]

    return dir_name if dir_name else None


def build_header():
    # Custom HTML header with gr.Markdown
    # There has to be another way, but this works for now; paths are weird in gradio
    with gr.Row():
        gr.Markdown(
            f"""
            <div style='display: flex; align-items: center;'>
                <img src='data:image/png;base64,{utils.img2base64(os.path.join(SCRIPT_DIR, "assets/img/birdnet_logo.png"))}' style='width: 50px; height: 50px; margin-right: 10px;'>
                <h2>BirdNET Analyzer</h2>
            </div>
            """
        )


def build_footer():
    with gr.Row():
        gr.Markdown(
            f"""
                <div style='display: flex; justify-content: space-around; align-items: center; padding: 10px; text-align: center'>
                    <div>
                        <div style="display: flex;flex-direction: row;">GUI version:&nbsp<span id="current-version">{os.environ['GUI_VERSION'] if FROZEN else 'main'}</span><span style="display: none" id="update-available"><a>+</a></span></div>
                        <div>Model version: {cfg.MODEL_VERSION}</div>
                    </div>
                    <div>K. Lisa Yang Center for Conservation Bioacoustics<br>Chemnitz University of Technology</div>
                    <div>{loc.localize('footer-help')}:<br><a href='https://birdnet.cornell.edu/analyzer' target='_blank'>birdnet.cornell.edu/analyzer</a></div>
                </div>
                """
        )


def build_settings():
    with gr.Tab(loc.localize("settings-tab-title")) as settings_tab:
        with gr.Row():
            options = [lang.rsplit(".", 1)[0] for lang in os.listdir(LANG_DIR) if lang.endswith(".json")]
            languages_dropdown = gr.Dropdown(
                options,
                value=loc.TARGET_LANGUAGE,
                label=loc.localize("settings-tab-language-dropdown-label"),
                info=loc.localize("settings-tab-language-dropdown-info"),
                interactive=True,
            )

        gr.Markdown(
            """
                If you encounter a bug or error, please provide the error log.\n
                You can submit an issue on our [GitHub](https://github.com/kahst/BirdNET-Analyzer/issues).
                """,
            label=loc.localize("settings-tab-error-log-textbox-label"),
            elem_classes="mh-200",
        )

        error_log_tb = gr.TextArea(
            label=loc.localize("settings-tab-error-log-textbox-label"),
            info=f"{loc.localize('settings-tab-error-log-textbox-info-path')}: {cfg.ERROR_LOG_FILE}",
            interactive=False,
            placeholder=loc.localize("settings-tab-error-log-textbox-placeholder"),
            show_copy_button=True,
        )

        def on_language_change(value):
            loc.set_language(value)

        def on_tab_select(value: gr.SelectData):
            if value.selected and os.path.exists(cfg.ERROR_LOG_FILE):
                with open(cfg.ERROR_LOG_FILE, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    last_100_lines = lines[-100:]
                    return "".join(last_100_lines)

            return ""

        languages_dropdown.input(on_language_change, inputs=languages_dropdown, show_progress=False)

        settings_tab.select(on_tab_select, outputs=error_log_tb, show_progress=False)


def sample_sliders(opened=True):
    """Creates the gradio accordion for the inference settings.

    Args:
        opened: If True the accordion is open on init.

    Returns:
        A tuple with the created elements:
        (Slider (min confidence), Slider (sensitivity), Slider (overlap))
    """
    with gr.Accordion(loc.localize("inference-settings-accordion-label"), open=opened):
        with gr.Group():
            with gr.Row():
                confidence_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.01,
                    label=loc.localize("inference-settings-confidence-slider-label"),
                    info=loc.localize("inference-settings-confidence-slider-info"),
                )
                sensitivity_slider = gr.Slider(
                    minimum=0.5,
                    maximum=1.5,
                    value=1,
                    step=0.01,
                    label=loc.localize("inference-settings-sensitivity-slider-label"),
                    info=loc.localize("inference-settings-sensitivity-slider-info"),
                )
                overlap_slider = gr.Slider(
                    minimum=0,
                    maximum=2.99,
                    value=0,
                    step=0.01,
                    label=loc.localize("inference-settings-overlap-slider-label"),
                    info=loc.localize("inference-settings-overlap-slider-info"),
                )

            with gr.Row():
                fmin_number = gr.Number(
                    cfg.SIG_FMIN,
                    minimum=0,
                    label=loc.localize("inference-settings-fmin-number-label"),
                    info=loc.localize("inference-settings-fmin-number-info"),
                )

                fmax_number = gr.Number(
                    cfg.SIG_FMAX,
                    minimum=0,
                    label=loc.localize("inference-settings-fmax-number-label"),
                    info=loc.localize("inference-settings-fmax-number-info"),
                )

        return confidence_slider, sensitivity_slider, overlap_slider, fmin_number, fmax_number


def locale():
    """Creates the gradio elements for locale selection

    Reads the translated labels inside the checkpoints directory.

    Returns:
        The dropdown element.
    """
    label_files = os.listdir(ORIGINAL_TRANSLATED_LABELS_PATH)
    options = ["EN"] + [label_file.rsplit("_", 1)[-1].split(".")[0].upper() for label_file in label_files]

    return gr.Dropdown(
        options,
        value="EN",
        label=loc.localize("analyze-locale-dropdown-label"),
        info=loc.localize("analyze-locale-dropdown-info"),
    )


def species_list_coordinates():
    with gr.Group():
        lat_number = gr.Slider(
            minimum=-90,
            maximum=90,
            value=0,
            step=1,
            label=loc.localize("species-list-coordinates-lat-number-label"),
            info=loc.localize("species-list-coordinates-lat-number-info"),
        )
        lon_number = gr.Slider(
            minimum=-180,
            maximum=180,
            value=0,
            step=1,
            label=loc.localize("species-list-coordinates-lon-number-label"),
            info=loc.localize("species-list-coordinates-lon-number-info"),
        )
        with gr.Row():
            yearlong_checkbox = gr.Checkbox(
                True, label=loc.localize("species-list-coordinates-yearlong-checkbox-label")
            )
            week_number = gr.Slider(
                minimum=1,
                maximum=48,
                value=1,
                step=1,
                interactive=False,
                label=loc.localize("species-list-coordinates-week-slider-label"),
                info=loc.localize("species-list-coordinates-week-slider-info"),
            )

        sf_thresh_number = gr.Slider(
            minimum=0.01,
            maximum=0.99,
            value=0.03,
            step=0.01,
            label=loc.localize("species-list-coordinates-threshold-slider-label"),
            info=loc.localize("species-list-coordinates-threshold-slider-info"),
        )

    def onChange(use_yearlong):
        return gr.Slider(interactive=(not use_yearlong))

    yearlong_checkbox.change(onChange, inputs=yearlong_checkbox, outputs=week_number, show_progress=False)

    return lat_number, lon_number, week_number, sf_thresh_number, yearlong_checkbox


def select_file(filetypes=(), state_key=None):
    """Creates a file selection dialog.

    Args:
        filetypes: List of filetypes to be filtered in the dialog.

    Returns:
        The selected file or None of the dialog was canceled.
    """
    initial_selection = loc.get_state(state_key, "") if state_key else ""
    files = _WINDOW.create_file_dialog(webview.OPEN_DIALOG, file_types=filetypes, directory=initial_selection)

    if files:
        if state_key:
            loc.set_state(state_key, files[0])

        return files[0]

    return None


def show_species_choice(choice: str):
    """Sets the visibility of the species list choices.

    Args:
        choice: The label of the currently active choice.

    Returns:
        A list of [
            Row update,
            File update,
            Column update,
            Column update,
        ]
    """
    if choice == _CUSTOM_SPECIES:
        return [
            gr.Row(visible=False),
            gr.File(visible=True),
            gr.Column(visible=False),
            gr.Column(visible=False),
        ]
    elif choice == _PREDICT_SPECIES:
        return [
            gr.Row(visible=True),
            gr.File(visible=False),
            gr.Column(visible=False),
            gr.Column(visible=False),
        ]
    elif choice == _CUSTOM_CLASSIFIER:
        return [
            gr.Row(visible=False),
            gr.File(visible=False),
            gr.Column(visible=True),
            gr.Column(visible=False),
        ]

    return [
        gr.Row(visible=False),
        gr.File(visible=False),
        gr.Column(visible=False),
        gr.Column(visible=True),
    ]


def species_lists(opened=True):
    """Creates the gradio accordion for species selection.

    Args:
        opened: If True the accordion is open on init.

    Returns:
        A tuple with the created elements:
        (Radio (choice), File (custom species list), Slider (lat), Slider (lon), Slider (week), Slider (threshold), Checkbox (yearlong?), State (custom classifier))
    """
    with gr.Accordion(loc.localize("species-list-accordion-label"), open=opened):
        with gr.Row():
            species_list_radio = gr.Radio(
                [_CUSTOM_SPECIES, _PREDICT_SPECIES, _CUSTOM_CLASSIFIER, _ALL_SPECIES],
                value=_ALL_SPECIES,
                label=loc.localize("species-list-radio-label"),
                info=loc.localize("species-list-radio-info"),
                elem_classes="d-block",
            )

            with gr.Column(visible=False) as position_row:
                lat_number, lon_number, week_number, sf_thresh_number, yearlong_checkbox = species_list_coordinates()

            species_file_input = gr.File(
                file_types=[".txt"], visible=False, label=loc.localize("species-list-custom-list-file-label")
            )
            empty_col = gr.Column()

            with gr.Column(visible=False) as custom_classifier_selector:
                classifier_selection_button = gr.Button(
                    loc.localize("species-list-custom-classifier-selection-button-label")
                )
                classifier_file_input = gr.Files(file_types=[".tflite"], visible=False, interactive=False)
                selected_classifier_state = gr.State()

                def on_custom_classifier_selection_click():
                    file = select_file(("TFLite classifier (*.tflite)",), state_key="custom_classifier_file")

                    if file:
                        labels = os.path.splitext(file)[0] + "_Labels.txt"

                        return file, gr.File(value=[file, labels], visible=True)

                    return None, None

                classifier_selection_button.click(
                    on_custom_classifier_selection_click,
                    outputs=[selected_classifier_state, classifier_file_input],
                    show_progress=False,
                )

            species_list_radio.change(
                show_species_choice,
                inputs=[species_list_radio],
                outputs=[position_row, species_file_input, custom_classifier_selector, empty_col],
                show_progress=False,
            )

            return (
                species_list_radio,
                species_file_input,
                lat_number,
                lon_number,
                week_number,
                sf_thresh_number,
                yearlong_checkbox,
                selected_classifier_state,
            )


def _get_win_drives():
    from string import ascii_uppercase as UPPER_CASE

    return [f"{drive}:\\" for drive in UPPER_CASE]


def open_window(builder: list[Callable] | Callable):
    multiprocessing.freeze_support()

    with gr.Blocks(
        css=open(os.path.join(SCRIPT_DIR, "assets/gui.css")).read(),
        js=open(os.path.join(SCRIPT_DIR, "assets/gui.js")).read(),
        theme=gr.themes.Default(),
        analytics_enabled=False,
    ) as demo:
        build_header()

        if callable(builder):
            builder()
        elif isinstance(builder, (tuple, set, list)):
            for build in builder:
                build()

        build_settings()
        build_footer()

    url = demo.queue(api_open=False).launch(
        prevent_thread_lock=True,
        quiet=True,
        show_api=False,
        enable_monitoring=False,
        allowed_paths=_get_win_drives() if sys.platform == "win32" else ["/"],
    )[1]
    _WINDOW = webview.create_window("BirdNET-Analyzer", url.rstrip("/") + "?__theme=light", min_size=(1024, 768))
    set_window(_WINDOW)

    with suppress(ModuleNotFoundError):
        import pyi_splash  # type: ignore

        pyi_splash.close()

    webview.start(private_mode=False)
