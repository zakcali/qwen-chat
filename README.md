# qwen-chat

A comprehensive, locally-hosted AI studio providing access to Alibaba's **Qwen** (Tongyi Qianwen) models for text generation, vision analysis, and image creation

## 🚀 Key Features

- **💬 Interactive Chat**: Chat with advanced text models (including `qwen3-max` and `qwen3.5-flash`) via `text2text.py`.
- **🧠 Reasoning & Logic**: Supports models with reasoning traces, featuring adjustable "effort" controls for enhanced logic.
- **🖼️ Multimodal Studio**: Analyze images, perform OCR, or generate high-quality images from text descriptions via `image-analysis-and-generator.py`.
- **⚙️ Dynamic Configuration**: Effortlessly update available models and default system instructions by editing simple text files.
- **🧹 Auto-Cleanup**: Automated session management that deletes temporary downloads and chat logs upon exiting the application.

## 🛠️ Prerequisites

- **Python 3.8+**
- **Qwen API Key**: Obtain your API key from the [Alibaba DashScope Console](https://modelstudio.console.alibabacloud.com/ap-southeast-1/).

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd qwen
    ```

2.  **Set your API Key**:
    Set the `QWEN_API_KEY` environment variable in your terminal:
    - **Windows (PowerShell)**: `$env:QWEN_API_KEY = "your-api-key"`
    - **Linux/macOS**: `export QWEN_API_KEY="your-api-key"`

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

The project is split into two specialized interfaces:

### 1. Text Chat Studio
Run the following to start the text generation and reasoning interface:
```bash
python text2text.py
```
*   **Features**: System prompt control, temperature/max token adjustment, and the ability to download chat logs as Markdown.

### 2. Multimodal AI Studio
Run the following for image-related tasks:
```bash
python image-analysis-and-generator.py
```
*   **Features**: Image-to-Text (Vision/OCR), Text-to-Image (Generation), and Image-to-Image (Editing).

## 📂 Configuration Files

Customize the application by editing these files:

-   `models.txt`: List of text-based models for `text2text.py`.
-   `models-image.txt`: List of multimodal/vision models for `image-analysis-and-generator.py`.
-   `system-prompt.txt`: Default system instructions for text chat.
-   `system-prompt-image.txt`: Default system instructions for image tasks.

## 📄 License


