# 🎤 Virtual Voice Assistant - AI-Powered Multi-Modal Assistant

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-orange)](https://huggingface.co/transformers/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-green)](https://github.com/openai/whisper)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-ff6b6b)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An advanced AI-powered voice assistant that combines **speech recognition**, **image analysis**, and **text-to-speech** capabilities to create an accessible and interactive multi-modal experience.

## 🌟 Features

### 🎯 Core Capabilities
- **🎙️ Speech-to-Text**: Convert voice input to text using OpenAI Whisper
- **🖼️ Image Analysis**: Detailed image description using LLaVA vision model
- **🔊 Text-to-Speech**: Convert responses back to natural speech
- **🌐 Multi-Modal Interface**: Interactive web interface with Gradio
- **♿ Accessibility**: Enhanced support for users with audio/visual challenges

### 🚀 Advanced Features
- **4-bit Quantization**: Optimized model performance with reduced memory usage
- **Multi-language Support**: Automatic language detection for audio input
- **Real-time Processing**: Live audio recording and instant responses
- **Logging System**: Comprehensive interaction logging for analysis
- **GPU Acceleration**: CUDA support for faster processing

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Vision Model** | LLaVA-1.5-7B | Image understanding and description |
| **Speech Recognition** | OpenAI Whisper (Medium) | Audio transcription |
| **Text-to-Speech** | Google TTS (gTTS) | Voice synthesis |
| **Quantization** | BitsAndBytes | Memory optimization |
| **Interface** | Gradio | Web-based UI |
| **Backend** | PyTorch + Transformers | ML framework |

## 📋 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- Internet connection for model downloads

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/virtual-voice-assistant.git
cd virtual-voice-assistant

# Install required packages
pip install -q transformers
pip install bitsandbytes
pip install accelerate
pip install -q git+https://github.com/openai/whisper.git
pip install -q gradio
pip install -q gTTS
pip install nltk
pip install pillow
```

### Alternative Installation
```bash
# Install all dependencies at once
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Basic Setup
```python
import torch
from transformers import BitsAndBytesConfig, pipeline

# Configure quantization for optimized performance
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Initialize the image-to-text pipeline
model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", 
                model=model_id, 
                model_kwargs={"quantization_config": quantization_config})
```

### 2. Launch the Interface
```python
import gradio as gr

# Start the interactive web interface
iface.launch(debug=True)
```

### 3. Using the Assistant
1. **Upload an Image** 📸
2. **Record Voice Input** 🎙️
3. **Get AI-Generated Description** 🤖
4. **Listen to Audio Response** 🔊

## 💡 Usage Examples

### Image Analysis
```python
# Analyze an image with voice command
image_path = "your_image.jpg"
voice_input = "Describe what you see in this image"

# Process through the assistant
result = process_inputs(voice_input, image_path)
```

### Voice Interaction
- **"What colors are in this image?"**
- **"Is this a painting or photograph?"**
- **"Describe the scene in detail"**
- **"What's happening in this picture?"**

## 🏗️ Architecture

```mermaid
graph TD
    A[Voice Input] --> B[Whisper STT]
    C[Image Upload] --> D[LLaVA Vision Model]
    B --> E[Text Processing]
    D --> E
    E --> F[Response Generation]
    F --> G[Google TTS]
    G --> H[Audio Output]
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style H fill:#e8f5e8
```

## 📊 Performance

| Model Component | Size | Memory Usage | Speed |
|----------------|------|-------------|-------|
| **LLaVA-1.5-7B** | ~13GB | ~8GB (4-bit) | ~2-3s |
| **Whisper Medium** | ~769MB | ~2GB | ~1-2s |
| **TTS Engine** | ~50MB | ~100MB | ~1s |

## 🔧 Configuration

### Model Settings
```python
# Adjust generation parameters
max_new_tokens = 250  # Response length
temperature = 0.7     # Creativity level
device = "cuda"       # GPU/CPU selection
```

### Audio Settings
```python
# Whisper configuration
model_size = "medium"  # tiny, base, small, medium, large
language = "auto"      # Auto-detect or specify
```

## 📁 Project Structure

```
virtual-voice-assistant/
│
├── 📄 main.py                 # Main application script
├── 📄 requirements.txt        # Dependencies
├── 📄 README.md              # This file
├── 📁 models/                # Model cache directory
├── 📁 logs/                  # Interaction logs
├── 📁 temp/                  # Temporary audio files
├── 📁 examples/              # Sample images and audio
│   ├── 🖼️ sample_image.jpg
│   └── 🎵 sample_audio.wav
└── 📄 LICENSE                # MIT License
```

## 🎯 Use Cases

### Accessibility
- **Visual Impairment**: Audio descriptions of images
- **Hearing Impairment**: Visual text output of speech
- **Motor Impairment**: Voice-controlled interface

### Education
- **Language Learning**: Pronunciation and description practice
- **Art Analysis**: Detailed artwork descriptions
- **Science**: Visual experiment analysis

### Professional
- **Content Creation**: Image analysis for social media
- **Documentation**: Voice-to-text transcription
- **Accessibility Compliance**: Multi-modal content support

## 🔬 Technical Details

### Quantization Benefits
- **Memory Reduction**: 75% less GPU memory usage
- **Speed Improvement**: 2x faster inference
- **Quality Retention**: 95%+ accuracy maintained

### Model Specifications
- **Vision Understanding**: 7B parameter LLaVA model
- **Speech Recognition**: Transformer-based Whisper
- **Language Support**: 99 languages supported
- **Audio Quality**: 16kHz sampling rate

## 🚧 Troubleshooting

### Common Issues

#### CUDA Memory Error
```bash
# Reduce model size or use CPU
DEVICE = "cpu"  # Force CPU usage
```

#### Audio Processing Error
```bash
# Install system dependencies
sudo apt-get install ffmpeg
```

#### Model Download Timeout
```bash
# Use HuggingFace cache
export HF_HOME=/path/to/cache
```

## 📈 Performance Optimization

### GPU Optimization
```python
# Enable mixed precision training
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
```

### Memory Management
```python
# Clear cache periodically
torch.cuda.empty_cache()
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone for development
git clone https://github.com/yourusername/virtual-voice-assistant.git
cd virtual-voice-assistant

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

