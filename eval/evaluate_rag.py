import httpx
import json
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

# where our API is running
API_BASE_URL = "http://localhost:8000"
QA_SET_FILE = os.path.join(os.path.dirname(__file__), "eval_qa_set.jsonl")
ASK_ENDPOINT = "/ask"
MAX_CONCURRENT_REQUESTS = 1 # don't spam the API, go one by one

# send a question to our API and get back the answer
async def ask_api(client: httpx.AsyncClient, question: str) -> str:
    """ask the API a question and return what it says"""
    try:
        response = await client.post(ASK_ENDPOINT, json={"question": question}, timeout=60)
        response.raise_for_status() # blow up if we get 4xx/5xx errors
        data = response.json()
        return data.get("answer", "").lower() # lowercase for easier matching
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error for question '{question}': {e.response.status_code} - {e.response.text}")
        return "[API ERROR]"
    except Exception as e:
        print(f"Error asking question '{question}': {e}")
        return "[SCRIPT ERROR]"

# test one question and see if the answer has the right keywords
async def evaluate_pair(client: httpx.AsyncClient, qa_pair: dict) -> bool:
    """ask one question and check if we got the expected keywords back"""
    question = qa_pair["question"]
    expected_keywords = [kw.lower() for kw in qa_pair["expected_answer_keywords"]]
    
    print(f"Asking: {question}")
    actual_answer = await ask_api(client, question)
    print(f"Received: {actual_answer[:100]}...") # just show first 100 chars
    
    if "[API ERROR]" in actual_answer or "[SCRIPT ERROR]" in actual_answer:
        print("--> FAILED (Error during API call)")
        return False
        
    # see if all the keywords we expect are actually in the answer
    match = all(keyword in actual_answer for keyword in expected_keywords)
    
    if match:
        print("--> PASSED")
    else:
        print(f"--> FAILED (Missing keywords: {[kw for kw in expected_keywords if kw not in actual_answer]})")
        
    print("-" * 20)
    return match

# the main function that runs everything
async def main():
    """load questions, test them all, show the results"""
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
            await asyncio.sleep(1)  # wait a bit so we don't hit rate limits

    # figure out how we did
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
    
    # save the score to a file too
    summary_file = os.path.join(os.path.dirname(__file__), "eval_summary.txt")
    with open(summary_file, 'w') as f:
         f.write(f"RAG Evaluation Score: {score:.2f}% ({passed_count}/{total_count} passed)\n")
    print(f"Summary written to {summary_file}")

if __name__ == "__main__":
    # make sure the API is actually running first
    try:
        response = httpx.get(f"{API_BASE_URL}/healthz", timeout=5)
        response.raise_for_status()
        print("API is healthy. Starting evaluation...")
        asyncio.run(main())
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        print(f"API is not running or accessible at {API_BASE_URL}/healthz. Please start the API first.")
        print(f"Error details: {e}")