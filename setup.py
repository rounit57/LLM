
from langchain.llms import Ollama

llm = Ollama(
    model = 'deepseek-r1:8b',
    base_url = 'https://localhost:5003',
)

llm.predict('I am happy')