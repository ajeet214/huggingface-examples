from transformers import pipeline

pipe = pipeline(task="text-to-audio",
                model="", framework="pt"
                )

generate_kwargs = {"temperature": 0.8, "max_new_tokens": 20}

outputs = pipe("Classic rock riff", generate_kwargs=generate_kwargs)

# UNDER THE HOOD
# MusicgenForConditionalGeneration