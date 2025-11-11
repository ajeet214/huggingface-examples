# Import the class for defining Hugging Face pipelines
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# Define the LLM from the Hugging Face model ID
# llm = HuggingFacePipeline.from_model_id(
#     model_id="crumb/nano-mistral",
#     task="text-generation",
#     pipeline_kwargs={"max_new_tokens": 20}
# )


# Create the examples list of dicts
examples = [
  {
    "question": "How many DataCamp courses has Jack completed?",
    "answer": "36"
  },
  {
    "question": "How much XP does Jack have on DataCamp?",
    "answer": "284,320XP"
  },
  {
    "question": "What technology does Jack learn about most on DataCamp?",
    "answer": "Python"
  }
]

# Complete the prompt for formatting answers
example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")

# Create the few-shot prompt
prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

prompt = prompt_template.invoke({"input": "What is Jack's favorite technology on DataCamp?"})
print(prompt.text)

"""
Question: How many DataCamp courses has Jack completed?
36

Question: How much XP does Jack have on DataCamp?
284,320XP

Question: What technology does Jack learn about most on DataCamp?
Python

Question: What is Jack's favorite technology on DataCamp?

"""

llm = HuggingFacePipeline.from_model_id(
    model_id="crumb/nano-mistral",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 20}
)

# Create and invoke the chain
llm_chain = prompt_template | llm
print(llm_chain.invoke({"input": "What is Jack's favorite technology on DataCamp?"}))

# content="Jack's favorite technology on DataCamp is Python." additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 64, 'total_tokens': 74, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_560af6e559', 'finish_reason': 'stop', 'logprobs': None} id='run-3ab4e6d7-9f13-4d5e-aeb8-839069ce850a-0' usage_metadata={'input_tokens': 64, 'output_tokens': 10, 'total_tokens': 74, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}