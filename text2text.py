import os
import gradio as gr
from openai import OpenAI
import time
import tempfile
import atexit

# This list will hold the paths of all generated chat logs for this session.
temp_files_to_clean = []

# --- Function to perform cleanup on exit ---
def cleanup_temp_files():
    """Iterates through the global list and deletes the tracked files."""
    if not temp_files_to_clean:
        return
    print(f"\nCleaning up {len(temp_files_to_clean)} temporary files...")
    for file_path in temp_files_to_clean:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  - Error removing {file_path}: {e}")
    print("Cleanup complete.")

atexit.register(cleanup_temp_files)

# --- Function to read system prompt from a file ---
def load_system_prompt(filepath="system-prompt.txt"):
    """Loads the system prompt from a text file, with a fallback default."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Using a default system prompt.")
        return "You are a helpful assistant."

# --- Function to read the model list from a file ---
def load_models(filepath="models.txt"):
    """Loads the list of models from a text file, with a fallback default list."""
    default_models = [
        "qwen3-max",
        "qwen3.5-flash",
        "qwen3.5-27b",
        "qwen3.5-plus",
    ]
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # Read non-empty lines and strip whitespace
            models = [line.strip() for line in f if line.strip()]
            if not models:
                print(f"Warning: '{filepath}' was empty. Using default model list.")
                return default_models
            return models
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Using default model list.")
        return default_models

print("Temporary chat download files will be saved in the OS's default temp directory.")

# --- Configuration ---
api_key = os.environ.get("QWEN_API_KEY")
if not api_key:
    print("CRITICAL: QWEN_API_KEY environment variable not found.")

# --- Initialize the OpenAI client to connect to Qwen ---

client = OpenAI(
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
)

# --- Updated function signature to accept model_choice ---
def chat_with_openai(message, history, model_choice, instructions,
                     temperature, max_tokens, effort):

    initial_download_update = gr.update(visible=False)

    if not message.strip():
        return history, "", "*No reasoning generated yet...*", initial_download_update

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]

    messages = []
    if instructions.strip():
        messages.append({"role": "system", "content": instructions})

    for m in history:
        if m["role"] == "assistant" and m["content"] == "":
            continue
        messages.append({"role": m["role"], "content": m["content"]})

    try:
        # Use a dictionary for request parameters for conditional logic
        request_params = {
            "model": model_choice, # Use the selected model
            "messages": messages,
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": int(max_tokens),
            "stream": True,
        }

        # OpenAI models use 'effort'
        if "openai/gpt-oss" in model_choice or "openai/gpt-5" in model_choice:
            request_params["extra_body"] = {
                "reasoning": {"effort": effort}
            }
        # Grok model uses a boolean 'enabled'
        elif "x-ai/grok-4-fast" in model_choice:
            # We interpret 'medium' or 'high' from the UI as a request to enable reasoning
            if effort in ["medium", "high"]:
                request_params["extra_body"] = {
                    "reasoning": {"enabled": True}
                }
       
        completion = client.chat.completions.create(**request_params)

        full_content = ""
        reasoning_content = ""
        
        last_yield_time = time.time()
        flush_interval_s = 0.04

        for chunk in completion:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            new_content = getattr(delta, "content", None) or None
            new_reasoning = getattr(delta, "reasoning", None) or None

            if new_content is not None:
                full_content += new_content
                history[-1]["content"] = full_content

            if new_reasoning is not None:
                reasoning_content += new_reasoning
            if "grok-4" in model_choice and not reasoning_content:
                reasoning_content = "*This model does not expose reasoning traces.*"

            now = time.time()
            if now - last_yield_time >= flush_interval_s:
                last_yield_time = now
                yield history, None, reasoning_content, initial_download_update

        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as temp_file:
            output_filepath = temp_file.name
            temp_file.write(full_content)

        temp_files_to_clean.append(output_filepath)
        print(f"Created and tracking temp file: {output_filepath}")

        final_download_update = gr.update(visible=True, value=output_filepath)

        yield history, "", reasoning_content, final_download_update

    except Exception as e:
        error_message = f"❌ An error occurred: {str(e)}"
        history[-1]["content"] = error_message
        yield history, "", f"An error occurred: {e}", initial_download_update

# Load external configuration before building UI ---
model_list = load_models()
initial_system_prompt = load_system_prompt()
# Set the default model to the first one in the list, or None if the list is empty
default_model = model_list[0] if model_list else None

# --- Gradio UI  ---
with gr.Blocks(title="💬 Qwen Chatbot") as demo:
    gr.Markdown("# 💬 Chatbot (Powered by Qwen API)")
    with gr.Row():
        with gr.Column(scale=3):
            # NEW (Gradio 6.0)
            chatbot = gr.Chatbot(height=500, buttons=["copy"])
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1)
            with gr.Row():
                stop_btn = gr.Button("Stop", scale=1)
                clear_btn = gr.Button("Clear Chat", scale=1)
                download_btn = gr.DownloadButton("⬇️ Download Last Response", visible=False, scale=3)

        with gr.Column(scale=1):
            # Models are loaded from models.txt ---
            model_choice = gr.Dropdown(
                label="Choose a Model",
                choices=model_list,
                value=default_model
            )
            
            # System prompt is loaded from system-prompt.txt ---
            instructions = gr.Textbox(
                label="System Instructions", 
                value=initial_system_prompt, 
                lines=3
            )
            
            effort = gr.Radio(
                ["low", "medium", "high"], 
                value="medium", 
                label="Reasoning Control (Effort for OpenAI / On for Grok)"
            )
            temperature = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Temperature")
            max_tokens = gr.Slider(100, 65535, value=8192, step=256, label="Max Tokens")
            thoughts_box = gr.Markdown(label="🧠 Model Thoughts", value="*Reasoning will appear here...*")

    inputs = [msg, chatbot, model_choice, instructions, temperature, max_tokens, effort]
    outputs = [chatbot, msg, thoughts_box, download_btn]

    e_submit = msg.submit(chat_with_openai, inputs, outputs)
    e_click = send_btn.click(chat_with_openai, inputs, outputs)

    stop_btn.click(fn=lambda: None, cancels=[e_submit, e_click], queue=False)

    clear_btn.click(
        lambda: ([], "", "*Reasoning will appear here...*", gr.update(visible=False)),
        outputs=[chatbot, msg, thoughts_box, download_btn],
        queue=False
    )

demo.queue()

if __name__ == "__main__":
    print("Launching Gradio interface... Press Ctrl+C to exit.")
    print("Temporary files for this session will be cleaned up automatically on exit.")
# NEW (Gradio 6.0)
demo.launch(theme=gr.themes.Default())
