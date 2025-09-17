import requests

response = requests.get("https://huggingface.co/api/tasks")
tasks_data = response.json()

print(tasks_data.keys())