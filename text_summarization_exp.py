from transformers import pipeline


summarizer = pipeline(
    task="summarization",
    model="cnicu/t5-small-booksum"
)

text = ""
summary_text = summarizer(text)

# Compare the length
print(f"Original text length: {len(text)}")
print(f"Summary length: {len(summary_text[0]['summary_text'])}")

# --------------------------------------------------------------------

# Create the summarization pipeline
short_summarizer = pipeline(
    task="summarization",
    model="cnicu/t5-small-booksum",
    min_new_tokens=10,
    max_new_tokens=200
)

# Summarize the text
short_summary_text = summarizer(text)

print(f"Summary length: {len(short_summary_text[0]['summary_text'])}")

# --------------------------------------------------------------------
