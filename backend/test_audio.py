from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def split_audio(input_file, max_duration=12000):
    # Load the audio file
    audio = AudioSegment.from_file(input_file)
    
    # Convert the audio into chunks based on silence detection
    chunks = split_on_silence(audio, 
                              min_silence_len=1000, # Minimum silence length in ms (1 second)
                              silence_thresh=-40,   # Silence threshold in dBFS
                              keep_silence=500)     # Keep 0.5s of silence at the beginning and end of the chunks

    # List to store audio chunks
    final_chunks = []

    # Iterate over each chunk and ensure it's under the max_duration
    for chunk in chunks:
        start_time = 0
        while start_time < len(chunk):
            # Calculate the end time, ensuring it's not more than max_duration
            end_time = min(start_time + max_duration, len(chunk))
            final_chunks.append(chunk[start_time:end_time])
            start_time = end_time

    # Create an output directory for the chunks
    output_dir = "splitted_audio"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export the chunks as separate audio files
    for idx, chunk in enumerate(final_chunks):
        chunk.export(f"{output_dir}/chunk_{idx + 1}.wav", format="wav")
    
    print(f"Audio has been split into {len(final_chunks)} parts and saved to '{output_dir}'.")

# Call the function with the input file
input_file = "SPEAKER_01_5.wav"  # Provide your input audio file path here
split_audio(input_file)
