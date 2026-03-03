import os
import gradio as gr
from PIL import Image
from io import BytesIO
import tempfile
import atexit
import requests

# --- Import DashScope ---
import dashscope
from dashscope import MultiModalConversation

# --- Temporary File Management ---

# This list will hold the paths of all generated chat logs and temp images for this session.
temp_files_to_clean = []

def cleanup_temp_files():
    """Iterates through the global list and deletes the tracked files."""
    if not temp_files_to_clean:
        return
    print(f"\nCleaning up {len(temp_files_to_clean)} temporary files...")
    for file_path in temp_files_to_clean:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass # File was already deleted or never existed
        except Exception as e:
            print(f"  - Error removing {file_path}: {e}")
    print("Cleanup complete.")

# Register the cleanup function to be called on script exit
atexit.register(cleanup_temp_files)

print("Temporary download files will be saved in the OS's default temp directory and cleaned on exit.")

# --- Function to read system prompt from a file ---
def load_system_prompt(filepath="system-prompt-image.txt"):
    """Loads the system prompt from a text file, with a fallback default."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Using a default system prompt.")
        return "You are a helpful multimodal AI assistant."

# --- Function to read the model list from a file ---
def load_models(filepath="models-image.txt"):
    """Loads the list of models from a text file, with a fallback default list."""
    default_models = [
        "qwen-image-max",
        "qwen-image-edit-max",
        "qwen3-vl-plus",
        "qwen-vl-ocr",
        "qwen3-vl-flash",
        "qwen-image-plus",
        "qwen-image-edit-plus",
    ]
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            models = [line.strip() for line in f if line.strip()]
            if not models:
                print(f"Warning: '{filepath}' was empty. Using default model list.")
                return default_models
            return models
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Using default model list.")
        return default_models

# --- Configuration ---
api_key = os.environ.get("QWEN_API_KEY")
if not api_key:
    print("CRITICAL: QWEN_API_KEY environment variable not found.")

# Set up DashScope globally
dashscope.api_key = api_key
# Setting base URL to Singapore region as requested in previous contexts
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

# --- Core Logic (Unified DashScope approach) ---
def get_multimodal_response(prompt, source_image, model_choice, instructions, max_tokens):
    initial_download_update = gr.update(visible=False)

    if not api_key:
        raise gr.Error("QWEN_API_KEY not set. Please set the environment variable.")
    if not prompt and not source_image:
        raise gr.Error("Please enter a prompt or upload an image.")
    
    # If no prompt is provided with an image, use a default one.
    if source_image and not prompt:
        prompt = "Describe this image in detail."

    # Determine what kind of model is being called to adjust parameters safely
    is_vision_model = "vl" in model_choice.lower() or "ocr" in model_choice.lower()

    api_messages = []
    user_content = []

    try:
        # --- Handle Inputs for DashScope ---
        if source_image:
            # Dashscope needs local files to be saved and referenced with the file:// scheme
            temp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            # Standardize image format to RGB/JPEG
            source_image.convert('RGB').save(temp_img_file, format="JPEG")
            temp_img_file.close()
            
            temp_files_to_clean.append(temp_img_file.name)
            user_content.append({"image": f"file://{temp_img_file.name}"})

        # Apply System Prompt
        if instructions and instructions.strip():
            if is_vision_model:
                api_messages.append({"role": "system", "content": [{"text": instructions}]})
            else:
                # Text-to-Image models generally only accept user roles. Let's merge instructions into prompt.
                prompt = f"{instructions}\n\n{prompt}"

        user_content.append({"text": prompt})
        api_messages.append({"role": "user", "content": user_content})

        # --- Dynamic Arguments ---
        call_kwargs = {
            "model": model_choice,
            "messages": api_messages,
            "result_format": "message"
        }
        
        # Add model-specific parameters to prevent parameter rejection
        if is_vision_model:
             call_kwargs["max_tokens"] = int(max_tokens)
        else:
             call_kwargs["size"] = "1024*1024"  # Default for image models

        # --- Make the DashScope API call ---
        response = MultiModalConversation.call(**call_kwargs)

        # --- Process the response ---
        if response.status_code == 200:
            content_list = response.output.choices[0].message.content
            
            text_response = ""
            image_output = None
            download_update = initial_download_update

            # Qwen returns a list of dictionaries. Parse text and image URLs dynamically.
            for item in content_list:
                if 'text' in item:
                    text_response += item['text']
                elif 'image' in item:
                    image_url = item['image']
                    try:
                        # If a URL is returned, download it to display in Gradio
                        img_data = requests.get(image_url).content
                        image_output = Image.open(BytesIO(img_data))
                    except Exception as e:
                        text_response += f"\n[Error downloading generated image: {e}]"

            # Create standard download markdown if text was generated
            if text_response:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as temp_file:
                    output_filepath = temp_file.name
                    temp_file.write(text_response)
                
                temp_files_to_clean.append(output_filepath)
                download_update = gr.update(visible=True, value=output_filepath)

            # Prettify the status
            if image_output and not text_response.strip():
                text_response = "Image generated successfully."
            
            status_message = f"✅ Success with {model_choice}."
            return (text_response, image_output, status_message, download_update)
            
        else:
            # Catch API rejections cleanly
            error_msg = f"❌ API Error {response.code}: {response.message}"
            print(error_msg)
            return ("", None, error_msg, initial_download_update)

    except Exception as e:
        print(f"An exception occurred: {e}")
        error_message = f"❌ An internal error occurred: {e}"
        return ("", None, error_message, initial_download_update)

# --- Load external configuration before building UI ---
model_list = load_models()
initial_system_prompt = load_system_prompt()
default_model = model_list[0] if model_list else None

# --- Gradio User Interface ---
with gr.Blocks(title="👁️ Multimodal AI Studio") as demo:
    gr.Markdown("# 👁️ Multimodal AI Studio (via QWEN API)")
    gr.Markdown("Provide a text prompt and/or an image to any model and see what happens.")
    with gr.Row():
        # --- LEFT COLUMN ---
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
               label="Choose a Model",
               choices=model_list,
               value=default_model
            )
            input_image = gr.Image(type="pil", label="Upload an Image (Optional)", height=350)
            prompt_box = gr.Textbox(
                label="Your Prompt",
                placeholder="Ask a question or describe an image to generate...",
                lines=5
            )
            with gr.Row():
                clear_btn = gr.Button(value="🗑️ Clear I/O", scale=1)
                run_btn = gr.Button("Run Model", variant="primary", scale=2)
            status_box = gr.Markdown("")
        
        # --- RIGHT COLUMN ---
        with gr.Column(scale=1):
            text_output_box = gr.Textbox(
                label="Model's Text Response",
                lines=5,
                interactive=False,
                buttons=["copy"]
            )
            download_btn = gr.DownloadButton(
                "⬇️ Download Text Response",
                visible=False
            )
            image_output_box = gr.Image(label="Image Output", interactive=False, height=400, buttons=["download"])
            
            # --- ADDED UI ELEMENTS ---
            instructions = gr.Textbox(
                label="System Instructions", 
                value=initial_system_prompt, 
                lines=3
            )
            max_tokens = gr.Slider(
                8192, 
                65535, 
                value=32768, 
                step=256, 
                label="Max Tokens"
            )

    # Define inputs and outputs for the Gradio function
    inputs_list = [prompt_box, input_image, model_choice, instructions, max_tokens]
    outputs_list = [text_output_box, image_output_box, status_box, download_btn]

    run_btn.click(
        fn=get_multimodal_response,
        inputs=inputs_list,
        outputs=outputs_list
    )

    # Configure the clear button
    clear_btn.click(
        fn=lambda: (None, "", "", None, "Inputs and outputs cleared.", gr.update(visible=False)),
        inputs=None,
        outputs=[input_image, prompt_box, text_output_box, image_output_box, status_box, download_btn],
        queue=False
    )

if __name__ == "__main__":
    print("Launching Gradio interface... Press Ctrl+C to exit.")
    demo.launch(theme=gr.themes.Soft())