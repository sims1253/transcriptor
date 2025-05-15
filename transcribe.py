import os
import shutil # Added for moving files
import torch
import torchaudio # Added for pre-loading audio
import argparse
import datetime
import gc # Added for garbage collection
from tqdm import tqdm # Added for transcription progress (optional, can be removed if single call is fast)
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook # Added for diarization progress
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as whisper_pipeline # Re-added transformers
# from faster_whisper import WhisperModel # Removed faster-whisper

# --- Timestamp Formatting ---
def format_timestamp(seconds):
    """Converts seconds to HH:MM:SS.ms format."""
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int(delta.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

# --- Main Transcription and Diarization Function ---
def transcribe_and_diarize(audio_path, num_speakers, output_file, hf_token=None, device=None, batch_size=16, language='en'): # Changed beam_size back to batch_size, kept language
    """
    Transcribes audio using Distil-Whisper (distil-large-v3.5 via Transformers) and performs speaker diarization.

    Args:
        audio_path (str): Path to the audio file.
        num_speakers (int): The predefined number of speakers.
        output_file (str): Path to save the formatted transcription.
        hf_token (str, optional): Hugging Face authentication token for pyannote.
                                  Defaults to None (uses cached login).
        device (str, optional): Device to run models on ('cuda:0' or 'cpu').
                                Defaults to auto-detect.
        batch_size (int): Batch size for Whisper inference (used if pipeline supports batching for long-form). # Changed back to batch_size
        language (str): Language code for transcription (e.g., 'en', 'de', 'fr'). Defaults to 'en'.
    """

    # --- Setup Device ---
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32 # Use torch_dtype again
    print(f"Using device: {device} with dtype: {torch_dtype}")

    # --- Load Audio ---
    print(f"Loading audio file into memory: {audio_path}") # Load into memory for pipeline
    try:
        # Pre-load audio using torchaudio for potentially faster diarization
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate
        print(f"Audio duration: {datetime.timedelta(seconds=duration)}, Sample rate: {sample_rate} Hz")
        # Note: pyannote pipeline handles resampling and channel mixing automatically
        # if the input is a waveform tensor.
    except Exception as e:
        print(f"Error loading audio file with torchaudio: {e}")
        return

    # --- 1. Speaker Diarization (Optional) ---
    speaker_turns = []
    if num_speakers > 1:
        print("Loading speaker diarization pipeline (pyannote/speaker-diarization-3.1)...")
        print("Ensure pyannote.audio >= 3.1 is installed.")
        try:
            # Use authentication token if provided, otherwise rely on CLI login
            diarization_pipeline = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", # Updated model
                use_auth_token=hf_token
            ).to(torch.device(device)) # Move pipeline to the designated device

            print("Performing speaker diarization from memory (with progress)...")
            # Pass waveform and sample_rate dict for in-memory processing
            diarization_input = {"waveform": waveform, "sample_rate": sample_rate}
            # Use ProgressHook for monitoring
            with ProgressHook() as hook:
                diarization = diarization_pipeline(diarization_input,
                                                   num_speakers=num_speakers,
                                                   hook=hook)
            print("\nSpeaker diarization complete.") # Add newline after progress bar

            # Prepare diarization results for easy lookup
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_turns.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            print(f"Found {len(speaker_turns)} speaker turns.")

            # --- Unload Diarization Model ---
            print("Unloading diarization pipeline to free memory...")
            del diarization_pipeline
            del diarization # Remove the results object as well
            del diarization_input # Remove the input dict
            gc.collect() # Trigger garbage collection
            if device.startswith("cuda"): # Check if device is CUDA before clearing cache
                torch.cuda.empty_cache() # Clear CUDA cache if applicable
            print("Diarization model unloaded.")
            # --- End Unload ---

        except Exception as e:
            print(f"Error during speaker diarization: {e}")
            print("Ensure you have accepted pyannote terms on Hugging Face and are logged in.")
            print("Proceeding without speaker diarization.")
            # Ensure speaker_turns is empty if diarization fails mid-way
            speaker_turns = []
    elif num_speakers == 1:
        print("Skipping speaker diarization as num_speakers is 1.")
    else: # num_speakers <= 0 or invalid
        print("Invalid num_speakers provided. Skipping speaker diarization.")
        # Treat as single speaker case for transcription assignment
        num_speakers = 1 # Set to 1 internally to handle assignment logic below


    # --- 2. Whisper Transcription (Transformers Pipeline) ---
    try:
        # Conditionally select model based on language
        if language == 'en':
            model_id = "distil-whisper/distil-large-v3.5"
            print(f"Loading English-optimized model ({model_id} via Transformers)...")
        else:
            model_id = "openai/whisper-large-v3-turbo"
            print(f"Loading multilingual model ({model_id} via Transformers) for language: {language}...")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        print("Initializing transcription pipeline...")
        pipe = whisper_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            batch_size=batch_size, # Use batch_size again
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )

        # --- Run Transcription on Full Audio ---
        print(f"Starting transcription with Transformers pipeline (language={language})...")
        # Prepare audio input for the pipeline
        if isinstance(waveform, torch.Tensor):
             if waveform.ndim > 1 and waveform.shape[0] > 1:
                 print("Audio has multiple channels, converting to mono for transcription.")
                 waveform_mono = torch.mean(waveform, dim=0)
             else:
                 waveform_mono = waveform.squeeze(0)
             audio_input = {"raw": waveform_mono.cpu().numpy(), "sampling_rate": sample_rate}
        else:
             print("Error: Expected waveform to be a torch.Tensor")
             return

        # Perform transcription, passing language via generate_kwargs
        # The specific kwarg might depend on the transformers version, 'language' is common for Whisper
        transcription_result = pipe(audio_input, generate_kwargs={"language": language})


        if not transcription_result or "chunks" not in transcription_result:
             print("Error: Transcription pipeline did not return expected 'chunks' output.")
             if "text" in transcription_result:
                 print(f"Transcription succeeded but only returned text: {transcription_result['text']}")
                 print("Cannot perform speaker assignment without timestamps ('chunks').")
             return

        print(f"Transcription complete. Found {len(transcription_result['chunks'])} segments.")

        # --- Unload Transcription Model ---
        print("Unloading transcription model to free memory...")
        del model
        del processor
        del pipe
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print("Transcription model unloaded.")
        # --- End Unload ---

    except Exception as e:
        print(f"\nError during Transformers pipeline transcription: {e}")
        print("Ensure 'transformers', 'torch', and 'accelerate' are installed.")
        return


    # --- 3. Combine Diarization and Transcription ---
    print("Combining transcription and diarization results...")
    final_transcript = []
    unassigned_segments = 0

    # Iterate through transcribed segments (chunks from the pipeline)
    for segment in transcription_result["chunks"]: # Changed back to pipeline output format
        segment_start, segment_end = segment["timestamp"] # Get timestamp tuple
        segment_text = segment["text"]

        assigned_speaker = "SPEAKER_01" # Default for single speaker case
        if num_speakers > 1 and speaker_turns: # Only assign if diarization ran and succeeded
            # Calculate segment midpoint for speaker assignment
            segment_midpoint = segment_start + (segment_end - segment_start) / 2
            assigned_speaker = "UNKNOWN" # Reset default if we are assigning

            # Find the speaker turn that contains the segment's midpoint
            # This is a simple approach; more sophisticated overlap analysis could be used
            found_speaker = False
            for turn in speaker_turns:
                if turn["start"] <= segment_midpoint < turn["end"]:
                    assigned_speaker = turn["speaker"]
                    found_speaker = True
                    break

            if not found_speaker:
                # Basic fallback: Assign to speaker whose turn is closest to the midpoint
                closest_turn = None
                min_distance = float('inf')
                for turn in speaker_turns:
                    # Distance from midpoint to center of turn
                    turn_midpoint = turn["start"] + (turn["end"] - turn["start"]) / 2
                    distance = abs(segment_midpoint - turn_midpoint)
                    if distance < min_distance:
                        min_distance = distance
                        closest_turn = turn
                if closest_turn:
                     assigned_speaker = closest_turn["speaker"]
                     # Optionally add a threshold for max distance if needed
                # else: # Should not happen if speaker_turns exist
                unassigned_segments += 1 # Count unassigned only when diarization was attempted

        elif num_speakers > 1 and not speaker_turns:
             # Diarization was requested but failed or yielded no turns
             assigned_speaker = "UNKNOWN" # Mark as unknown if diarization failed
             unassigned_segments +=1


        final_transcript.append({
            "start": segment_start,
            "end": segment_end,
            "speaker": assigned_speaker,
            "text": segment_text.strip() # Ensure text is stripped
        })

    if unassigned_segments > 0:
        print(f"Warning: Could not confidently assign a speaker to {unassigned_segments} segments (likely in gaps or near precise turn boundaries). Assigned based on proximity or marked as UNKNOWN.")

    # --- 4. Format and Save Output ---
    print(f"Saving merged transcript to: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            if not final_transcript:
                print("Warning: No transcript segments to write.")
                return # Exit if transcript is empty

            # Initialize the first merged segment
            merged_start = final_transcript[0]['start']
            merged_end = final_transcript[0]['end']
            merged_speaker = final_transcript[0]['speaker']
            merged_text = final_transcript[0]['text']

            for i in range(1, len(final_transcript)):
                current_item = final_transcript[i]
                # Check if speaker is the same as the previous segment
                if current_item['speaker'] == merged_speaker:
                    # Merge: update end time and append text (with space)
                    merged_end = current_item['end']
                    # Add space only if the previous text didn't end with space and current doesn't start with one
                    if merged_text and not merged_text.endswith(' ') and current_item['text'] and not current_item['text'].startswith(' '):
                         merged_text += " " + current_item['text']
                    else:
                         merged_text += current_item['text'] # Handle cases like leading/trailing spaces in words
                else:
                    # Speaker changed: write the previous merged segment
                    start_time = format_timestamp(merged_start)
                    end_time = format_timestamp(merged_end)
                    f.write(f"[{start_time} --> {end_time}] {merged_speaker}:{merged_text.strip()}\n")

                    # Start a new merged segment
                    merged_start = current_item['start']
                    merged_end = current_item['end']
                    merged_speaker = current_item['speaker']
                    merged_text = current_item['text']

            # Write the last merged segment after the loop finishes
            start_time = format_timestamp(merged_start)
            end_time = format_timestamp(merged_end)
            f.write(f"[{start_time} --> {end_time}] {merged_speaker}:{merged_text.strip()}\n")

        print("Done.")
    except Exception as e:
        print(f"Error writing output file: {e}")

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe long audio with Whisper (large-v3 via Transformers) and speaker diarization. Can process a single file or all files in an 'input' directory.")
    parser.add_argument("audio_path", nargs='?', default=None, help="Path to the input audio file (e.g., wav, mp3, flac). If not provided, processes all files in the 'input/' directory.")
    parser.add_argument("num_speakers", type=int, help="The exact number of speakers in the audio (set to 1 to skip diarization).")
    parser.add_argument("-o", "--output_file", default="transcript.txt", help="Path to save the output transcript. For single file mode, defaults to 'output/<input_filename_without_ext>.txt' if not specified. For directory processing, this argument is ignored and outputs are saved in 'output/' with original names and .txt extension.")
    parser.add_argument("--hf_token", default=None, help="Hugging Face authentication token for pyannote (optional, reads from cache if not provided).")
    parser.add_argument("--device", default=None, choices=['cuda:0', 'cpu'], help="Device to use ('cuda:0' or 'cpu'). Auto-detects if not specified.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for transcription pipeline (default: 16). Lower if you encounter memory issues.") # Changed back to batch_size
    parser.add_argument("--language", type=str, default="de", help="Language code for transcription (e.g., 'en', 'de', 'fr', default: 'en').") # Kept language argument

    args = parser.parse_args()

    input_folder = "input"
    output_folder = "output"
    supported_extensions = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.opus', '.aac', '.wma', '.aiff', '.aif')

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    if args.audio_path:
        # --- Process a single audio file ---
        if not os.path.exists(args.audio_path):
            print(f"Error: Audio file not found at {args.audio_path}")
        elif not os.path.isfile(args.audio_path):
            print(f"Error: Provided path {args.audio_path} is not a file.")
        else:
            audio_file_to_process = args.audio_path
            base_name = os.path.basename(audio_file_to_process)
            name_without_ext = os.path.splitext(base_name)[0]

            # Determine output file path for single file mode
            if args.output_file != parser.get_default("output_file"): # User specified a custom output via -o
                output_txt_path = args.output_file
                print(f"Processing single file: {audio_file_to_process}, custom output transcript: {output_txt_path}")
            else:
                # Default behavior: output to 'output/' folder with same name + .txt
                output_txt_path = os.path.join(output_folder, f"{name_without_ext}.txt")
                print(f"Processing single file: {audio_file_to_process}, output transcript: {output_txt_path}")

            transcribe_and_diarize(
                audio_file_to_process,
                args.num_speakers,
                output_txt_path,
                args.hf_token,
                args.device,
                args.batch_size,
                args.language
            )

            # Move the processed input audio file to the output folder
            try:
                # Ensure the source file is not already in the output folder
                # (e.g. if input path was output/somefile.mp3 or a relative path resolving there)
                if os.path.abspath(os.path.dirname(audio_file_to_process)) != os.path.abspath(output_folder):
                    destination_audio_path = os.path.join(output_folder, base_name)
                    # Check if a file with the same name already exists in the destination
                    if os.path.exists(destination_audio_path):
                        print(f"Warning: File {base_name} already exists in {output_folder}. Overwriting.")
                    print(f"Moving processed audio file {audio_file_to_process} to {destination_audio_path}")
                    shutil.move(audio_file_to_process, destination_audio_path)
                    print(f"Successfully moved {base_name} to {output_folder}")
                else:
                    print(f"Audio file {audio_file_to_process} is already in the output folder '{output_folder}'. No move needed.")
            except Exception as e:
                print(f"Error moving file {audio_file_to_process} to {output_folder}: {e}")

    else:
        # --- Process all files in the input folder ---
        print(f"\nNo specific audio file provided. Processing all supported audio files in '{input_folder}' directory...")
        if not os.path.exists(input_folder):
            print(f"Error: Input directory '{input_folder}' does not exist. Please create it and place audio files inside.")
        elif not os.path.isdir(input_folder):
            print(f"Error: '{input_folder}' is not a directory.")
        else:
            audio_files_found = [
                f for f in os.listdir(input_folder)
                if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(input_folder, f))
            ]

            if not audio_files_found:
                print(f"No supported audio files found in '{input_folder}'. Supported extensions: {supported_extensions}")
            else:
                print(f"Found {len(audio_files_found)} audio file(s) to process in '{input_folder}': {', '.join(audio_files_found)}")
                files_processed_count = 0
                for filename in audio_files_found:
                    audio_file_path = os.path.join(input_folder, filename)
                    name_without_ext = os.path.splitext(filename)[0]
                    output_txt_file = os.path.join(output_folder, f"{name_without_ext}.txt")

                    print(f"\n--- Processing: {filename} ({files_processed_count + 1}/{len(audio_files_found)}) ---")
                    print(f"Input audio: {audio_file_path}")
                    print(f"Output transcript: {output_txt_file}")

                    transcribe_and_diarize(
                        audio_file_path,
                        args.num_speakers,
                        output_txt_file,
                        args.hf_token,
                        args.device,
                        args.batch_size,
                        args.language
                    )
                    files_processed_count += 1

                    # Move the processed input audio file to the output folder
                    try:
                        destination_audio_path = os.path.join(output_folder, filename)
                        if os.path.exists(destination_audio_path):
                            print(f"Warning: File {filename} already exists in {output_folder}. Overwriting.")
                        print(f"Moving processed audio file {audio_file_path} to {destination_audio_path}")
                        shutil.move(audio_file_path, destination_audio_path)
                        print(f"Successfully moved {filename} to {output_folder}")
                    except Exception as e:
                        print(f"Error moving file {audio_file_path} to {output_folder}: {e}")
                
                if files_processed_count > 0:
                    print(f"\nFinished processing {files_processed_count} audio file(s) from '{input_folder}'.")
                else: # Should not happen if audio_files_found was populated, but as a safeguard
                    print(f"\nNo audio files were processed from '{input_folder}'.")
