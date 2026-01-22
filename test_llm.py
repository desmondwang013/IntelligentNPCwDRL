"""
Test script for LLM integration.

Run this after setting up the model to verify everything works.

Usage:
    python test_llm.py                    # Use Ollama (default)
    python test_llm.py --model qwen3:4b   # Specify Ollama model
    python test_llm.py --backend llama-cpp --model-path path/to/model.gguf
"""
import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="Test LLM integration")
    parser.add_argument(
        "--backend",
        type=str,
        default="ollama",
        choices=["ollama", "llama-cpp"],
        help="LLM backend to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3:4b",
        help="Ollama model name (e.g., qwen3:4b)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/llm/qwen2.5-3b-instruct-q5_k_m.gguf",
        help="Path to GGUF model file (for llama-cpp backend)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LLM Integration Test")
    print("=" * 60)
    print(f"Backend: {args.backend}")
    if args.backend == "ollama":
        print(f"Model: {args.model}")
    else:
        print(f"Model path: {args.model_path}")
    print()

    # Import here to avoid import errors if dependencies not installed
    try:
        from src.llm import IntentParser
    except ImportError as e:
        print(f"ERROR: Failed to import LLM module: {e}")
        print("\nMake sure dependencies are installed:")
        print("  pip install ollama")
        return

    # Initialize parser
    print("Initializing LLM...")
    start = time.time()

    try:
        intent_parser = IntentParser(
            backend=args.backend,
            model_name=args.model,
            model_path=args.model_path,
        )
        intent_parser.load()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return
    except Exception as e:
        print(f"ERROR: Failed to initialize: {e}")
        return

    load_time = time.time() - start
    print(f"Ready in {load_time:.1f}s")
    print()

    # Test cases matching our world
    test_inputs = [
        # Tasks - navigation
        "go to the red circle",
        "walk to the player",
        "move to the large blue square",
        "wait",
        # Tasks - edge cases
        "attack the enemy",          # Unsupported action
        "go to the red thing",       # Ambiguous - no shape specified
        # Conversation
        "good morning",
        "hello there",
        "how are you doing?",
    ]

    print("=" * 60)
    print("Intent Parsing Tests")
    print("=" * 60)

    for i, user_input in enumerate(test_inputs):
        start = time.time()
        # Enable debug for first input to see raw output
        intent = intent_parser.parse(user_input, debug=(i == 0))
        parse_time = time.time() - start

        print(f"\nInput: \"{user_input}\"")
        print(f"  Type: {intent.type}")
        if intent.is_task:
            print(f"  Action: {intent.action}")
            if intent.target_type:
                print(f"  Target Type: {intent.target_type}")
            if intent.color:
                print(f"  Color: {intent.color}")
            if intent.shape:
                print(f"  Shape: {intent.shape}")
            if intent.size:
                print(f"  Size: {intent.size}")
            print(f"  Valid Task: {intent.is_valid_task}")
            print(f"  Command: {intent.to_command()}")
        else:
            print(f"  Response: \"{intent.response}\"")
        print(f"  Time: {parse_time:.2f}s")

    # Test response generation
    print()
    print("=" * 60)
    print("Response Generation Tests")
    print("=" * 60)

    # Test Gateway responses that LLM must handle
    test_results = [
        {
            "user_input": "go to the red thing",
            "result": {"status": "ambiguous", "reason": "multiple matches", "matches": ["red circle", "red square", "red triangle"]}
        },
        {
            "user_input": "go to the blue triangle",
            "result": {"status": "success", "target_id": "obj_3", "steps": 45}
        },
        {
            "user_input": "attack the enemy",
            "result": {"status": "unsupported", "reason": "action not available"}
        },
    ]

    for test in test_results:
        start = time.time()
        response = intent_parser.respond_to_result(
            test["user_input"],
            test["result"]
        )
        gen_time = time.time() - start

        print(f"\nUser: \"{test['user_input']}\"")
        print(f"Result: {test['result']['status']}")
        print(f"Response: \"{response}\"")
        print(f"Time: {gen_time:.2f}s")

    print()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)

    # Cleanup
    intent_parser.unload()


if __name__ == "__main__":
    main()
