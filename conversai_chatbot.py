# ConversAI: Smart Chatbot using Pre-trained Language Models
# A comprehensive chatbot implementation with multiple model support

import torch
import warnings
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BlenderbotTokenizer, BlenderbotForConditionalGeneration,
    pipeline
)
import streamlit as st
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

class ConversAI:
    """
    Smart AI Chatbot powered by pre-trained transformer models
    Supports multiple models with fallback mechanisms
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.conversation_history = []
        self.max_history = 10  # Keep last 10 exchanges
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Available models with their configurations
        self.available_models = {
            "microsoft/DialoGPT-small": {"type": "causal", "max_length": 1000},
            "microsoft/DialoGPT-medium": {"type": "causal", "max_length": 1000},
            "microsoft/DialoGPT-large": {"type": "causal", "max_length": 1000},
            "facebook/blenderbot-400M-distill": {"type": "seq2seq", "max_length": 512},
            "facebook/blenderbot-1B-distill": {"type": "seq2seq", "max_length": 512},
            "microsoft/phi-3-mini-4k-instruct": {"type": "causal", "max_length": 2048}
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the specified model with fallback options"""
        try:
            logging.info(f"Loading model: {self.model_name}")
            
            if "blenderbot" in self.model_name.lower():
                # Load Blenderbot model (seq2seq)
                self.tokenizer = BlenderbotTokenizer.from_pretrained(self.model_name)
                self.model = BlenderbotForConditionalGeneration.from_pretrained(self.model_name)
                self.model_type = "seq2seq"
            else:
                # Load DialoGPT or other causal models
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.model_type = "causal"
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logging.error(f"Error loading model {self.model_name}: {e}")
            self.fallback_model()
    
    def fallback_model(self):
        """Fallback to a simpler model if primary model fails"""
        fallback_models = [
            "microsoft/DialoGPT-small",
            "facebook/blenderbot-400M-distill"
        ]
        
        for fallback in fallback_models:
            try:
                logging.info(f"Trying fallback model: {fallback}")
                self.model_name = fallback
                self.load_model()
                return
            except Exception as e:
                logging.error(f"Fallback model {fallback} also failed: {e}")
                continue
        
        # If all models fail, use a simple pipeline
        logging.warning("All models failed, using basic pipeline")
        self.use_pipeline_fallback()
    
    def use_pipeline_fallback(self):
        """Use Hugging Face pipeline as final fallback"""
        try:
            self.pipeline = pipeline(
                "text-generation",
                model="gpt2",
                tokenizer="gpt2",
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_type = "pipeline"
            logging.info("Using pipeline fallback with GPT-2")
        except Exception as e:
            logging.error(f"Pipeline fallback failed: {e}")
            self.model_type = "none"
    
    def preprocess_input(self, user_input: str) -> str:
        """Preprocess user input for better model performance"""
        # Basic preprocessing
        user_input = user_input.strip()
        if not user_input.endswith(('.', '!', '?')):
            user_input += '.'
        return user_input
    
    def generate_response(self, user_input: str) -> str:
        """Generate response based on user input and conversation history"""
        if self.model_type == "none":
            return "Sorry, I'm having trouble loading the AI model. Please try again later."
        
        user_input = self.preprocess_input(user_input)
        
        try:
            if self.model_type == "causal":
                return self._generate_causal_response(user_input)
            elif self.model_type == "seq2seq":
                return self._generate_seq2seq_response(user_input)
            elif self.model_type == "pipeline":
                return self._generate_pipeline_response(user_input)
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your message. Could you try rephrasing?"
    
    def _generate_causal_response(self, user_input: str) -> str:
        """Generate response using causal language models like DialoGPT"""
        # Build context from conversation history
        context = ""
        for exchange in self.conversation_history[-5:]:  # Last 5 exchanges
            context += f"{exchange['user']}{self.tokenizer.eos_token}{exchange['bot']}{self.tokenizer.eos_token}"
        
        # Add current input
        context += user_input + self.tokenizer.eos_token
        
        # Tokenize
        inputs = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                top_k=50,
                top_p=0.9
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _generate_seq2seq_response(self, user_input: str) -> str:
        """Generate response using seq2seq models like Blenderbot"""
        # Tokenize input
        inputs = self.tokenizer(user_input, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=150,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                no_repeat_ngram_size=3
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    
    def _generate_pipeline_response(self, user_input: str) -> str:
        """Generate response using pipeline fallback"""
        prompt = f"Human: {user_input}\nAI:"
        
        result = self.pipeline(
            prompt,
            max_length=len(prompt) + 50,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        
        response = result[0]['generated_text'][len(prompt):].strip()
        # Clean up response
        response = response.split('\n')[0].strip()
        return response if response else "I'm not sure how to respond to that."
    
    def add_to_history(self, user_input: str, bot_response: str):
        """Add exchange to conversation history"""
        self.conversation_history.append({
            "user": user_input,
            "bot": bot_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def chat(self, user_input: str) -> str:
        """Main chat function"""
        response = self.generate_response(user_input)
        self.add_to_history(user_input, response)
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def save_conversation(self, filename: str):
        """Save conversation to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
    
    def load_conversation(self, filename: str):
        """Load conversation from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
        except FileNotFoundError:
            logging.warning(f"Conversation file {filename} not found")


def run_cli_interface():
    """Run command-line interface"""
    print("🤖 ConversAI Chatbot")
    print("=" * 50)
    print("Available models:")
    for i, model in enumerate([
        "microsoft/DialoGPT-medium",
        "facebook/blenderbot-400M-distill",
        "microsoft/DialoGPT-small"
    ], 1):
        print(f"{i}. {model}")
    
    choice = input("\nSelect model (1-3) or press Enter for default: ").strip()
    model_map = {
        "1": "microsoft/DialoGPT-medium",
        "2": "facebook/blenderbot-400M-distill",
        "3": "microsoft/DialoGPT-small"
    }
    
    model_name = model_map.get(choice, "microsoft/DialoGPT-medium")
    
    print(f"\nInitializing chatbot with {model_name}...")
    chatbot = ConversAI(model_name)
    
    print("\n🎉 Chatbot ready! Type 'quit' to exit, 'clear' to clear history, 'save' to save conversation.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("🧹 Conversation history cleared!")
                continue
            elif user_input.lower() == 'save':
                filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                chatbot.save_conversation(filename)
                print(f"💾 Conversation saved to {filename}")
                continue
            elif not user_input:
                continue
            
            print("🤖 Bot: ", end="")
            response = chatbot.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def run_streamlit_interface():
    """Run Streamlit web interface"""
    st.set_page_config(
        page_title="ConversAI Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 ConversAI: Smart Chatbot")
    st.markdown("*Powered by Pre-trained Transformer Models*")
    
    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        model_options = [
            "microsoft/DialoGPT-medium",
            "facebook/blenderbot-400M-distill",
            "microsoft/DialoGPT-small",
            "microsoft/DialoGPT-large"
        ]
        
        selected_model = st.selectbox(
            "Select Model:",
            model_options,
            index=0
        )
        
        if st.button("🔄 Load Model"):
            st.session_state.chatbot = ConversAI(selected_model)
            st.success(f"Model {selected_model} loaded!")
        
        st.markdown("---")
        
        if st.button("🧹 Clear History"):
            if 'chatbot' in st.session_state:
                st.session_state.chatbot.clear_history()
            if 'messages' in st.session_state:
                st.session_state.messages = []
            st.success("History cleared!")
        
        if st.button("💾 Save Conversation"):
            if 'chatbot' in st.session_state:
                filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.session_state.chatbot.save_conversation(filename)
                st.success(f"Saved to {filename}")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading chatbot..."):
            st.session_state.chatbot = ConversAI(selected_model)
    
    # Initialize chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm ConversAI, your smart chatbot assistant. How can I help you today?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.chat(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display model info
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Current Model:** {st.session_state.chatbot.model_name}")
        st.write(f"**Model Type:** {st.session_state.chatbot.model_type}")
        st.write(f"**Device:** {st.session_state.chatbot.device}")
        st.write(f"**Conversation Length:** {len(st.session_state.chatbot.conversation_history)} exchanges")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        # Run Streamlit interface
        run_streamlit_interface()
    else:
        # Run CLI interface
        run_cli_interface()