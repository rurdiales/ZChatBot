#!/usr/bin/env python3
import os
import argparse
import time
from src.chatbot import IndustrialChatbot
from src.config import AVAILABLE_MODELS, DEFAULT_MODEL

def main():
    parser = argparse.ArgumentParser(description="Industrial Chatbot CLI")
    parser.add_argument("--process", action="store_true", help="Process knowledge base")
    parser.add_argument("--clear", action="store_true", help="Clear knowledge base")
    parser.add_argument("--interactive", action="store_true", help="Start interactive session")
    parser.add_argument("--query", type=str, help="Single query to ask")
    parser.add_argument("--fast", action="store_true", help="Use fast mode for common questions")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        choices=list(AVAILABLE_MODELS.keys()),
                        help=f"Select LLM model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--compare", action="store_true", 
                        help="Compare all available models on the same query")
    
    args = parser.parse_args()
    
    # Create the chatbot with the selected model
    chatbot = IndustrialChatbot(model_name=args.model)
    
    # Process knowledge base if requested
    if args.process:
        chatbot.process_knowledge_base()
    
    # Clear knowledge base if requested
    if args.clear:
        chatbot.clear_knowledge_base()
    
    # Compare models mode
    if args.compare and args.query:
        print(f"Comparing all models for query: {args.query}")
        results = compare_models(args.query, args.fast)
        print("\n=== Model Comparison Results ===")
        for model_name, result in results.items():
            print(f"\nModel: {model_name}")
            print(f"Time: {result['time']:.2f} seconds")
            print(f"Answer: {result['answer']}")
        return
    
    # Interactive mode
    if args.interactive:
        print("Starting interactive session. Type 'exit' to quit.")
        print("Type 'process' to process the knowledge base.")
        print("Type 'clear' to clear the knowledge base.")
        print(f"Type 'switch:<model_name>' to switch models. Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        print("Fast mode: " + ("Enabled" if args.fast else "Disabled"))
        print(f"Current model: {args.model}")
        
        while True:
            query = input("\nYou: ")
            
            if query.lower() == "exit":
                print("Goodbye!")
                break
            elif query.lower() == "process":
                chatbot.process_knowledge_base()
                continue
            elif query.lower() == "clear":
                chatbot.clear_knowledge_base()
                continue
            elif query.lower().startswith("switch:"):
                # Switch model command
                parts = query.split(":", 1)
                if len(parts) == 2:
                    model_name = parts[1].strip()
                    if model_name in AVAILABLE_MODELS:
                        chatbot.switch_model(model_name)
                    else:
                        print(f"Unknown model: {model_name}")
                        print(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
                continue
            
            # Get answer with fast mode if enabled
            answer = chatbot.ask(query, fast_mode=args.fast)
            print(f"\nChatbot: {answer}")
    
    # Single query mode
    elif args.query:
        answer = chatbot.ask(args.query, fast_mode=args.fast)
        print(f"\nChatbot: {answer}")
    
    # If no action specified, show help
    elif not (args.process or args.clear):
        parser.print_help()

def compare_models(query: str, fast_mode: bool = False):
    """Compare all available models on the same query."""
    results = {}
    
    for model_name in AVAILABLE_MODELS.keys():
        print(f"\nTesting model: {model_name}")
        try:
            # Create new chatbot with this model
            chatbot = IndustrialChatbot(model_name=model_name)
            
            # Time the response
            start_time = time.time()
            answer = chatbot.ask(query, fast_mode=fast_mode)
            elapsed_time = time.time() - start_time
            
            results[model_name] = {
                "time": elapsed_time,
                "answer": answer
            }
            
            print(f"Response time: {elapsed_time:.2f} seconds")
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            results[model_name] = {
                "time": 0,
                "answer": f"Error: {str(e)}"
            }
    
    return results

if __name__ == "__main__":
    main() 