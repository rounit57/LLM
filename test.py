import os
import ollama

# Restrict computation to GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Check if GPU is available
try:
    import torch  # Assuming PyTorch might be used internally by Ollama
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. Please ensure a GPU is installed and accessible.")
except ImportError:
    print("Torch is not installed. Ensure Ollama uses GPU if supported by the library.")

# Model and input configuration
desiredModel = 'deepseek-r1:8b'
question = 'tell me something about IPO'

# Generate response using Ollama
response = ollama.chat(
    model=desiredModel,
    messages=[
        {
            'role': 'user',
            'content': question
        },
    ]
)

# Extract and print the response
OllamaResponse = response['message']['content']
print(OllamaResponse)

# Save the response to a file
with open("output.txt", "w", encoding="utf-8") as text_file:
    text_file.write(OllamaResponse)
