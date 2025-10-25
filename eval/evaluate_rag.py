import httpx
import json
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

# Configuration
API_BASE_URL = "http://localhost:8000"
QA_SET_FILE = os.path.join(os.path.dirname(__file__), "eval_qa_set.jsonl")
ASK_ENDPOINT = "/ask"
MAX_CONCURRENT_REQUESTS = 1 # Run sequentially to avoid quota limits

# Function to call the /ask endpoint
async def ask_api(client: httpx.AsyncClient, question: str) -> str:
    """Sends a question to the /ask endpoint and returns the answer."""
    try:
        response = await client.post(ASK_ENDPOINT, json={"question": question}, timeout=60)
        response.raise_for_status() # Raise exception for bad status codes
        data = response.json()
        return data.get("answer", "").lower() # Return answer in lowercase
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error for question '{question}': {e.response.status_code} - {e.response.text}")
        return "[API ERROR]"
    except Exception as e:
        print(f"Error asking question '{question}': {e}")
        return "[SCRIPT ERROR]"

# Function to evaluate a single Q&A pair
async def evaluate_pair(client: httpx.AsyncClient, qa_pair: dict) -> bool:
    """Evaluates a single question-answer pair."""
    question = qa_pair["question"]
    expected_keywords = [kw.lower() for kw in qa_pair["expected_answer_keywords"]]
    
    print(f"Asking: {question}")
    actual_answer = await ask_api(client, question)
    print(f"Received: {actual_answer[:100]}...") # Print truncated answer
    
    if "[API ERROR]" in actual_answer or "[SCRIPT ERROR]" in actual_answer:
        print("--> FAILED (Error during API call)")
        return False
        
    # Check if all expected keywords are in the actual answer
    match = all(keyword in actual_answer for keyword in expected_keywords)
    
    if match:
        print("--> PASSED")
    else:
        print(f"--> FAILED (Missing keywords: {[kw for kw in expected_keywords if kw not in actual_answer]})")
        
    print("-" * 20)
    return match

# Main evaluation function
async def main():
    """Loads QA set, runs evaluation, and prints results."""
    qa_pairs = []
    try:
        with open(QA_SET_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    qa_pairs.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Evaluation file not found at {QA_SET_FILE}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {QA_SET_FILE}: {e}")
        return
        
    if not qa_pairs:
        print("No question-answer pairs found in the evaluation file.")
        return

    print(f"Loaded {len(qa_pairs)} Q&A pairs for evaluation.")
    print("=" * 30)

    results = []
    
    async with httpx.AsyncClient(base_url=API_BASE_URL) as client:
        for pair in qa_pairs:
            result = await evaluate_pair(client, pair)
            results.append(result)
            await asyncio.sleep(1)  # Add delay between requests to avoid quota limits

    # Calculate and print score
    passed_count = sum(results)
    total_count = len(qa_pairs)
    score = (passed_count / total_count) * 100 if total_count > 0 else 0
    
    print("=" * 30)
    print("Evaluation Summary")
    print(f"Total Questions: {total_count}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {total_count - passed_count}")
    print(f"Score: {score:.2f}%")
    print("=" * 30)
    
    # Write one-line score summary to a file as requested
    summary_file = os.path.join(os.path.dirname(__file__), "eval_summary.txt")
    with open(summary_file, 'w') as f:
         f.write(f"RAG Evaluation Score: {score:.2f}% ({passed_count}/{total_count} passed)\n")
    print(f"Summary written to {summary_file}")

if __name__ == "__main__":
    # Check if API is running before starting
    try:
        response = httpx.get(f"{API_BASE_URL}/healthz", timeout=5)
        response.raise_for_status()
        print("API is healthy. Starting evaluation...")
        asyncio.run(main())
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        print(f"API is not running or accessible at {API_BASE_URL}/healthz. Please start the API first.")
        print(f"Error details: {e}")