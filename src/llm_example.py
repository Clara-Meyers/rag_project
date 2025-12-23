"""
LLM Example - Introduction to Gemini API
=========================================

This script demonstrates how to make basic API calls to Google's Gemini model.
It's a starting point to understand how LLMs (Large Language Models) work.

Before running this script:
1. Copy env.example to .env
2. Add your Gemini API key to .env
3. Install dependencies: pip install -r requirements.txt

Run with: python -m src.llm_example
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai


def setup_gemini():
    """
    Configure the Gemini API with your API key.

    The API key is loaded from environment variables for security.
    Never hardcode API keys directly in your code!
    """
    # Load environment variables from .env file
    load_dotenv()

    # Get the API key from environment variables
    api_key = os.getenv("GEMINI_API_KEY")

    # Check if the API key exists
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "Please set your GEMINI_API_KEY in the .env file!\n"
            "Get your key from: https://makersuite.google.com/app/apikey"
        )

    # Configure the Gemini API with our key
    genai.configure(api_key=api_key)

    print("✓ Gemini API configured successfully!")
    return True


def simple_text_generation(prompt: str) -> str:
    """
    Generate text using the Gemini model.

    Args:
        prompt: The text prompt to send to the model

    Returns:
        The generated response text

    This is the most basic way to interact with an LLM.
    You send a prompt (question/instruction) and receive a response.
    """
    # Create a Gemini model instance
    # 'gemini-1.5-flash' is a fast and efficient model, good for learning
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Generate a response
    # This is where the "magic" happens - the API sends your prompt
    # to Google's servers and returns the model's response
    response = model.generate_content(prompt)

    # Extract and return the text from the response
    return response.text


def chat_example():
    """
    Demonstrate a simple chat conversation with the model.

    Unlike single prompts, chat maintains context across messages.
    The model "remembers" what was said earlier in the conversation.
    """
    # Create a model instance
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Start a chat session
    chat = model.start_chat(history=[])

    print("\n--- Chat Example ---")
    print("(The model will remember context from previous messages)\n")

    # First message
    response1 = chat.send_message("Hi! My name is Alex and I'm learning Python.")
    print(f"User: Hi! My name is Alex and I'm learning Python.")
    print(f"Model: {response1.text}\n")

    # Second message - the model should remember the name
    response2 = chat.send_message("What's my name? And what am I learning?")
    print(f"User: What's my name? And what am I learning?")
    print(f"Model: {response2.text}\n")


def generation_with_context(context: str, question: str) -> str:
    """
    Generate a response based on provided context.

    THIS IS THE FOUNDATION OF RAG!

    Instead of relying only on the model's training data,
    we provide specific context (retrieved documents) that
    the model should use to answer the question.

    Args:
        context: Relevant information to help answer the question
        question: The user's question

    Returns:
        The model's response based on the provided context
    """
    # Create a prompt that includes the context
    # This is a simple "prompt template"
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return response.text


def main():
    """
    Main function demonstrating different ways to use the Gemini API.
    """
    print("=" * 60)
    print("Welcome to the Gemini API Tutorial!")
    print("=" * 60)

    # Step 1: Setup the API
    setup_gemini()

    # Step 2: Simple text generation
    print("\n--- Example 1: Simple Text Generation ---")
    prompt = "Explain what Python is in 2 sentences."
    print(f"Prompt: {prompt}")
    response = simple_text_generation(prompt)
    print(f"Response: {response}")

    # Step 3: Chat example (contextual conversation)
    chat_example()

    # Step 4: Generation with context (RAG foundation!)
    print("\n--- Example 3: Generation with Context (RAG Foundation) ---")

    # This is fake context - in a real RAG system, this would come
    # from your retrieved documents!
    context = """
    TechCorp was founded in 2020 in San Francisco.
    The company offers three products: DataFlow Pro, SecureVault, and TaskMaster.
    DataFlow Pro is a data analytics platform starting at $29/month.
    The CEO of TechCorp is Jane Smith.
    """

    question = "What products does TechCorp offer and when was the company founded?"
    print(f"Context: {context}")
    print(f"Question: {question}")
    answer = generation_with_context(context, question)
    print(f"Answer: {answer}")

    # Try a question where the answer is NOT in the context
    print("\n--- Example 4: Question without answer in context ---")
    question2 = "What is TechCorp's phone number?"
    print(f"Question: {question2}")
    answer2 = generation_with_context(context, question2)
    print(f"Answer: {answer2}")

    print("\n" + "=" * 60)
    print("Tutorial complete! You've learned the basics of using Gemini.")
    print("Next step: Build a RAG system using the documents in data/documents/")
    print("=" * 60)


if __name__ == "__main__":
    main()
