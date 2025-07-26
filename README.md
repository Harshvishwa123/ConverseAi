# 🤖 ConversAI CLI Chatbot

*A Smart Command-Line AI Chatbot Powered by Pre-trained Language Models*

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗-Transformers-orange?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Screenshots](#-screenshots)
- [Installation](#-installation)
- [Usage](#-usage)
- [Supported Models](#-supported-models)
- [Commands](#-commands)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Technical Details](#-technical-details)
- [Contributing](#-contributing)

## 🌟 Overview

ConversAI CLI is an intelligent command-line chatbot that leverages state-of-the-art pre-trained transformer models to engage in natural, human-like conversations. Built with Python and powered by Hugging Face Transformers, it provides a seamless chat experience directly in your terminal.

### Why ConversAI CLI?
- 🚀 **Fast & Lightweight**: Optimized for quick responses
- 🧠 **Context-Aware**: Remembers conversation history
- 🔄 **Multi-Model Support**: Choose from various AI models
- 💾 **Conversation Management**: Save and load chat sessions
- 🛡️ **Robust Fallbacks**: Automatic model switching if issues occur
- 🎯 **Easy to Use**: Simple, intuitive command-line interface

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multiple AI Models** | DialoGPT, Blenderbot, and more |
| **Context Memory** | Maintains conversation flow and context |
| **Smart Fallbacks** | Automatically switches models if one fails |
| **Conversation History** | Keep track of your chat sessions |
| **Save/Load Chats** | Persistent conversation storage |
| **GPU Acceleration** | CUDA support for faster responses |
| **Clean Interface** | User-friendly terminal experience |
| **Cross-Platform** | Works on Windows, macOS, and Linux |

## 📸 Screenshots


### Interactive Conversation
![Conversation Screenshot](screenshots/conversation.png)
*Example conversation showing context-aware responses and natural dialogue*


## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (8GB recommended for larger models)
- Internet connection (for model downloads)
- Optional: CUDA-compatible GPU for faster inference

### Quick Install

#### Option 1: Automatic Setup (Recommended)
```bash
# Clone or download the project
git clone <your-repo-url>
cd conversai-cli

# Run the setup script
chmod +x setup.sh
./setup.sh
```

#### Option 2: Manual Installation
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 3: Direct Install
```bash
# Install core dependencies
pip install torch transformers accelerate
```

### Verify Installation
```bash
python conversai.py --help
```

## 🎮 Usage

### Starting the Chatbot
```bash
python conversai.py
```

### Basic Workflow
1. **Launch**: Run the script to start the chatbot
2. **Select Model**: Choose from available AI models (1-3)
3. **Chat**: Type your messages and get AI responses
4. **Manage**: Use commands to control the conversation
5. **Exit**: Type 'quit' to end the session

### Sample Session
```bash
$ python conversai.py

🤖 ConversAI Chatbot
==================================================
Available models:
1. microsoft/DialoGPT-medium
2. facebook/blenderbot-400M-distill
3. microsoft/DialoGPT-small

Select model (1-3) or press Enter for default: 1

Initializing chatbot with microsoft/DialoGPT-medium...
Model loaded successfully on cpu

🎉 Chatbot ready! Type 'quit' to exit, 'clear' to clear history, 'save' to save conversation.
--------------------------------------------------

You: Hello! How are you today?
🤖 Bot: Hello! I'm doing well, thank you for asking. How are you doing today?

You: I'm working on a programming project
🤖 Bot: That sounds interesting! What kind of programming project are you working on? I'd love to hear more about it.

You: save
💾 Conversation saved to conversation_20241125_143052.json

You: quit
👋 Goodbye!
```

## 🧠 Supported Models

| Model Name | Size | Type | Best For | Memory Usage |
|------------|------|------|----------|--------------|
| **DialoGPT-small** | 117M | Causal LM | Quick responses, testing | ~1GB |
| **DialoGPT-medium** | 345M | Causal LM | Balanced performance | ~2GB |
| **DialoGPT-large** | 774M | Causal LM | High-quality conversations | ~3GB |
| **Blenderbot-400M** | 400M | Seq2Seq | Engaging, diverse responses | ~1.5GB |
| **Blenderbot-1B** | 1B | Seq2Seq | Premium conversation quality | ~4GB |

### Model Selection Guide
- **New users**: Start with DialoGPT-medium
- **Limited resources**: Use DialoGPT-small
- **High quality**: Choose DialoGPT-large or Blenderbot-1B
- **Diverse responses**: Try Blenderbot models

## 🎛️ Commands

| Command | Description | Example |
|---------|-------------|---------|
| `quit` / `exit` / `bye` | Exit the chatbot | `You: quit` |
| `clear` | Clear conversation history | `You: clear` |
| `save` | Save current conversation | `You: save` |
| `<message>` | Send message to AI | `You: Hello there!` |

### Command Examples
```bash
# Clear conversation history
You: clear
🧹 Conversation history cleared!

# Save conversation
You: save
💾 Conversation saved to conversation_20241125_143052.json

# Exit chatbot
You: quit
👋 Goodbye!
```

## 💬 Examples

### Creative Conversation
```
You: Tell me a short story about a robot
🤖 Bot: Once upon a time, there was a little robot named Bolt who dreamed of becoming an artist. Every night, he would sneak into the scrapyard and create beautiful sculptures from discarded metal pieces...

You: What happened next?
🤖 Bot: Bolt's sculptures were discovered by a famous art collector who was amazed by the robot's creativity. The collector organized an exhibition featuring Bolt's work, making him the first robot artist to gain international recognition...
```

### Technical Discussion
```
You: Can you explain machine learning?
🤖 Bot: Machine learning is a subset of artificial intelligence where computers learn patterns from data without being explicitly programmed for each task. It's like teaching a computer to recognize patterns the same way humans do...

You: What are the main types?
🤖 Bot: There are three main types: supervised learning (learning from labeled examples), unsupervised learning (finding hidden patterns), and reinforcement learning (learning through trial and error with rewards)...
```

### Casual Chat
```
You: What's your favorite hobby?
🤖 Bot: I really enjoy having conversations and learning about different topics! If I had to pick, I'd say I love discussing books, science, and creative projects. What about you?

You: I like reading sci-fi novels
🤖 Bot: That's awesome! Sci-fi is such a fascinating genre. Have you read any good books lately? I'd love to hear about your favorites!
```

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Model Loading Errors
```
❌ Error: Could not load model
✅ Solution: Check internet connection, try smaller model
```

#### 2. Out of Memory
```
❌ Error: CUDA out of memory / RuntimeError
✅ Solution: Use DialoGPT-small or close other applications
```

#### 3. Slow Responses
```
❌ Issue: Bot takes too long to respond
✅ Solution: Use GPU if available, try smaller model
```

#### 4. Installation Issues
```
❌ Error: Module not found
✅ Solution: pip install -r requirements.txt
```

### Performance Tips
- **Use GPU**: Install CUDA-compatible PyTorch for speed
- **Manage Memory**: Clear history frequently for long chats
- **Choose Right Model**: Balance quality vs. performance needs
- **Close Programs**: Free up RAM for better performance

### Debug Mode
```bash
# Run with verbose logging
python conversai.py --debug

# Check system resources
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🔬 Technical Details

### Architecture
```
ConversAI CLI
├── Model Manager (Load/Fallback)
├── Conversation Engine (Context + Generation)
├── History Manager (Save/Load/Clear)
└── CLI Interface (User Interaction)
```

### Performance Metrics
- **Startup Time**: 10-30 seconds (first run)
- **Response Time**: 0.5-3 seconds per message
- **Memory Usage**: 1-4GB depending on model
- **Context Length**: Keeps last 10 exchanges



## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute
1. **Bug Reports**: Found an issue? Please report it
2. **Feature Requests**: Suggest new features
3. **Code Contributions**: Submit pull requests
4. **Documentation**: Improve docs and examples
5. **Testing**: Test on different systems

### Development Setup
```bash
# Fork the repository
git clone <your-fork-url>
cd conversai-cli

# Create development branch
git checkout -b feature/your-feature

# Make changes and test
python conversai.py

# Submit pull request
```

### Code Guidelines
- Follow PEP 8 style guidelines
- Add comments for complex logic
- Include error handling
- Test on multiple models
- Update documentation


## 🙏 Acknowledgments

- **[Hugging Face](https://huggingface.co/)** - For the amazing Transformers library
- **[Microsoft](https://microsoft.com/)** - For DialoGPT models
- **[Meta](https://ai.facebook.com/)** - For Blenderbot models
- **[PyTorch](https://pytorch.org/)** - For the deep learning framework
- **Community** - For feedback and contributions



### Useful Links
- 📚 [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/)
- 🔥 [PyTorch Documentation](https://pytorch.org/docs/)
- 🐍 [Python Official Docs](https://docs.python.org/)

---

## 🎉 Ready to Chat?

Get started with just one command:

```bash
python conversai.py
```

**Happy Chatting! 🚀**

---

*Made with ❤️ using Python and AI*
