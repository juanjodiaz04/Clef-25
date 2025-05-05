import os

import gradio as gr

import birdnet_analyzer.audio as audio
import birdnet_analyzer.config as cfg
import birdnet_analyzer.gui.analysis as ga
import birdnet_analyzer.gui.utils as gu
import birdnet_analyzer.localization as loc
import birdnet_analyzer.utils as utils


def runSingleFileAnalysis(
    input_path,
    confidence,
    sensitivity,
    overlap,
    fmin,
    fmax,
    species_list_choice,
    species_list_file,
    lat,
    lon,
    week,
    use_yearlong,
    sf_thresh,
    custom_classifier_file,
    locale,
):
    import csv
    from datetime import timedelta

    if species_list_choice == gu._CUSTOM_SPECIES:
        gu.validate(species_list_file, loc.localize("validation-no-species-list-selected"))

    gu.validate(input_path, loc.localize("validation-no-file-selected"))

    if fmin is None or fmax is None or fmin < cfg.SIG_FMIN or fmax > cfg.SIG_FMAX or fmin > fmax:
        raise gr.Error(f"{loc.localize('validation-no-valid-frequency')} [{cfg.SIG_FMIN}, {cfg.SIG_FMAX}]")

    result_filepath = ga.runAnalysis(
        input_path,
        None,
        confidence,
        sensitivity,
        overlap,
        fmin,
        fmax,
        species_list_choice,
        species_list_file,
        lat,
        lon,
        week,
        use_yearlong,
        sf_thresh,
        custom_classifier_file,
        "csv",
        None,
        "en" if not locale else locale,
        1,
        4,
        None,
        skip_existing=False,
        progress=None,
    )

    # read the result file to return the data to be displayed.
    with open(result_filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
        data = [l[0:-1] for l in data[1:]]  # remove last column (file path) and first row (header)

        for row in data:
            for col_idx in range(2):
                seconds = float(row[col_idx])
                time_str = str(timedelta(seconds=seconds))
                row[col_idx] = time_str
            row.insert(0, "▶")  # add empty column for selection

    return data


def build_single_analysis_tab():
    with gr.Tab(loc.localize("single-tab-title")):
        audio_input = gr.Audio(type="filepath", label=loc.localize("single-audio-label"), sources=["upload"])
        with gr.Group():
            spectogram_output = gr.Plot(
                label=loc.localize("review-tab-spectrogram-plot-label"), visible=False, show_label=False
            )
            generate_spectrogram_cb = gr.Checkbox(
                value=True,
                label=loc.localize("single-tab-spectrogram-checkbox-label"),
                info="Potentially slow for long audio files.",
            )
        audio_path_state = gr.State()

        confidence_slider, sensitivity_slider, overlap_slider, fmin_number, fmax_number = gu.sample_sliders(False)

        (
            species_list_radio,
            species_file_input,
            lat_number,
            lon_number,
            week_number,
            sf_thresh_number,
            yearlong_checkbox,
            selected_classifier_state,
        ) = gu.species_lists(False)
        locale_radio = gu.locale()

        def get_audio_path(i, generate_spectrogram):
            if i:
                return (
                    i["path"],
                    gr.Audio(label=os.path.basename(i["path"])),
                    gr.Plot(visible=True, value=utils.spectrogram_from_file(i["path"], fig_size="auto"))
                    if generate_spectrogram
                    else gr.Plot(visible=False),
                )
            else:
                return None, None, gr.Plot(visible=False)

        def try_generate_spectrogram(audio_path, generate_spectrogram):
            if audio_path and generate_spectrogram:
                return gr.Plot(visible=True, value=utils.spectrogram_from_file(audio_path["path"], fig_size="auto"))
            else:
                return gr.Plot()

        generate_spectrogram_cb.change(
            try_generate_spectrogram,
            inputs=[audio_input, generate_spectrogram_cb],
            outputs=spectogram_output,
            preprocess=False,
        )

        audio_input.change(
            get_audio_path,
            inputs=[audio_input, generate_spectrogram_cb],
            outputs=[audio_path_state, audio_input, spectogram_output],
            preprocess=False,
        )

        inputs = [
            audio_path_state,
            confidence_slider,
            sensitivity_slider,
            overlap_slider,
            fmin_number,
            fmax_number,
            species_list_radio,
            species_file_input,
            lat_number,
            lon_number,
            week_number,
            yearlong_checkbox,
            sf_thresh_number,
            selected_classifier_state,
            locale_radio,
        ]

        output_dataframe = gr.Dataframe(
            type="pandas",
            headers=[
                "",
                loc.localize("single-tab-output-header-start"),
                loc.localize("single-tab-output-header-end"),
                loc.localize("single-tab-output-header-sci-name"),
                loc.localize("single-tab-output-header-common-name"),
                loc.localize("single-tab-output-header-confidence"),
            ],
            elem_classes="matrix-mh-200",
            elem_id="single-file-output",
        )
        single_file_analyze = gr.Button(loc.localize("analyze-start-button-label"), variant="huggingface")
        hidden_segment_audio = gr.Audio(visible=False, autoplay=True, type="numpy")

        def time_to_seconds(time_str):
            try:
                hours, minutes, seconds = map(int, time_str.split(":"))
                total_seconds = hours * 3600 + minutes * 60 + seconds

                return float(total_seconds)
            except ValueError:
                raise ValueError("Input must be in the format hh:mm:ss with numeric values.")

        def play_selected_audio(evt: gr.SelectData, audio_path):
            if evt.row_value[1] and evt.row_value[2]:
                start = time_to_seconds(evt.row_value[1])
                end = time_to_seconds(evt.row_value[2])
                arr, sr = audio.openAudioFile(audio_path, offset=start, duration=end - start)

                return sr, arr

            return None

        output_dataframe.select(play_selected_audio, inputs=audio_path_state, outputs=hidden_segment_audio)
        single_file_analyze.click(runSingleFileAnalysis, inputs=inputs, outputs=output_dataframe)


if __name__ == "__main__":
    gu.open_window(build_single_analysis_tab)
