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