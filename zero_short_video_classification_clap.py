from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from transformers import pipeline


# Create a subclip file from bounce_ad.mp4
ffmpeg_extract_subclip("bounce_ad.mp4", 0, 5, "bounce_ad_5s.mp4")

# Load the new subclip
video = VideoFileClip("bounce_ad_5s.mp4")

# Extract the audio stream
audio = video.audio

# Write the audio stream
audio.write_audiofile("bounce_ad_5s.mp3")


from decoder import VideoReader
from PIL import Image

video_reader = VideoReader(video_path)
video = video_reader.get_batch(range(20)).asnumpy()
video = video[:, :, :, ::-1]
video = [Image.fromarray(frame) for frame in video]

from datasets import Dataset, Audio
audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio())
audio_sample = audio_dataset[0]["audio"]["array"]

emotions = []

image_classifier = pipeline(
    model="openai/clip-vit-base-patch32",
    task="zero-shot-image-classification"
)

# Create emotion scores for each video frame
predictions = image_classifier(video, candidate_labels=emotions)
scores = [
    {l['label']: l['score'] for l in prediction}
    for prediction in predictions
]

avg_image_scores = {emotion: sum([s[emotion] for s in scores])/len(scores) for emotion in emotions}
print(f"Average scores: {avg_image_scores}")

# Make an audio classifier pipeline
audio_classifier = pipeline(
    model="laion/clap-htsat-unfused",
    task="zero-shot-audio-classification"
)

audio_scores = audio_classifier(audio_sample, candidate_labels=emotions)
audio_scores = {l['label']: l['score'] for l in audio_scores}

multimodal_scores = {emotion: (avg_image_scores[emotion] + audio_scores[emotion])/2 for emotion in emotions}
print(f"Multimodal scores: {multimodal_scores}")
