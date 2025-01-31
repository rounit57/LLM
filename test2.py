import ollama

def check_ollama_health():
    """Verify Ollama is running and accessible"""
    try:
        # Check server status
        response = ollama.Client().show('')
        print("Ollama server version:", response.get('version', 'unknown'))
        return True
    except Exception as e:
        print(f"Ollama connection failed: {str(e)}")
        print("Make sure Ollama is running: run 'ollama serve' first!")
        return False

def verify_gpu_acceleration():
    """Check if GPU acceleration is available"""
    try:
        # New proper way to check GPU status
        gpu_info = ollama.Client().show('gpu')
        print("\nGPU Acceleration Status:")
        print(f"Driver: {gpu_info.get('driver', 'N/A')}")
        print(f"Libraries: {gpu_info.get('cuda', 'N/A')}")
        return True
    except Exception as e:
        print("\nGPU Acceleration Not Available:", str(e))
        return False

def main():
    if not check_ollama_health():
        return

    if not verify_gpu_acceleration():
        print("Continuing with CPU...")

    # Configuration
    desired_model = 'deepseek-r1:8b'
    question = "Tell me something about GDP"

    try:
        # Verify model exists
        models = [model['name'] for model in ollama.list()['models']]
        if desired_model not in models:
            print(f"\nModel {desired_model} not found! Available models: {models}")
            print(f"Run 'ollama pull {desired_model}' first")
            return

        # Generate with GPU acceleration
        response = ollama.chat(
            model=desired_model,
            messages=[{'role': 'user', 'content': question}],
            options={
                'stop': ["\n", "User:", "Assistant:", ""],
                'temperature': 0.7,
                'num_gpu': 50  # Percentage of GPU memory to use
            }
        )
        
        ollama_response = response['message']['content']
        print("\nGenerated Response:", ollama_response)
        
        with open("output.txt", "w", encoding="utf-8") as text_file:
            text_file.write(ollama_response)

    except Exception as e:
        print("\nGeneration error:", str(e))

if __name__ == "__main__":
    main()