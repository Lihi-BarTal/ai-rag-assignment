import requests
import pandas as pd

# 1. Load CSV and safely pick a relevant topic
print("Loading CSV to determine a safe test topic...")
try:
    df = pd.read_csv("medium-english-50mb.csv", nrows=5)
    # Extract the first tag from the first article to ensure a meaningful topic
    first_tags = df["tags"].iloc[0]
    topic = first_tags.replace('[', '').replace(']', '').replace("'", '').split(',')[0].strip()
    if not topic:
        topic = "technology" # Fallback if no tags are found
except Exception as e:
    print(f"Could not load CSV for topic extraction. Defaulting to 'technology'. Error: {e}")
    topic = "technology"

print(f"=== Selected Test Topic: '{topic}' ===\n")

# 2. Setup target API URLs
BASE_URL = "http://127.0.0.1:8000"
PROMPT_URL = f"{BASE_URL}/api/prompt"
STATS_URL = f"{BASE_URL}/api/stats"

# 3. Define the assignment test categories
queries = [
    {
        "name": "1. Precise fact retrieval", 
        "q": f"Find an article about {topic}. Provide the title and author."
    },
    {
        "name": "2. Multi-result topic listing", 
        "q": f"List up to 3 articles about {topic}. Return only the titles."
    },
    {
        "name": "3. Key idea summary extraction", 
        "q": f"Find an article about {topic} and summarise its central argument."
    },
    {
        "name": "4. Recommendation with evidence", 
        "q": f"I want advice on {topic}. Which article would you recommend, and why?"
    },
    {
        "name": "5. Out of Domain (Anti-Hallucination)", 
        "q": "What is the capital city of France?"
    }
]

def test_stats_endpoint():
    print(f"\n[TEST]: Checking GET /api/stats endpoint...")
    try:
        response = requests.get(STATS_URL)
        response.raise_for_status()
        data = response.json()
        print("--- STATS RESPONSE ---")
        print(data)
        
        # Verify required keys exist
        required_keys = ["chunk_size", "overlap_ratio", "top_k"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"❌ WARNING: Missing required fields in stats response: {missing_keys}")
        else:
            print("✅ Stats endpoint looks good!")
        print("="*50)
            
    except Exception as e:
        print(f"❌ Error during /api/stats request: {e}")
        print("="*50)

def run_tests():
    print("Starting Agent Tests...\n" + "="*50)
    
    # Run the stats endpoint test first
    test_stats_endpoint()
    
    for test in queries:
        print(f"\n[TEST]: {test['name']}")
        print(f"[QUESTION]: {test['q']}\n")
        
        try:
            # Send query to FastAPI server
            response = requests.post(PROMPT_URL, json={"question": test['q']})
            response.raise_for_status() 
            data = response.json()
            
            agent_response = data.get("response", "No response found")
            print("--- AGENT RESPONSE ---")
            print(agent_response)
            
            # Specific check for Anti-Hallucination requirement
            if "Anti-Hallucination" in test['name']:
                required_fallback = "I don't know based on the provided Medium articles data."
                if required_fallback.lower() in agent_response.lower():
                    print("\n✅ Anti-Hallucination Fallback triggered correctly!")
                else:
                    print(f"\n❌ FAILED Anti-Hallucination test. Expected the exact fallback string.")

            print("\n--- SOURCES USED (From Pinecone) ---")
            for i, ctx in enumerate(data.get("context", [])):
                title = ctx.get('title', 'Unknown')
                score = ctx.get('score', 0)
                print(f"  {i+1}. {title} (Match Score: {score:.4f})")
                
            print("\n" + "="*50)
        
        except Exception as e:
            print(f"❌ Error during request: {e}")

if __name__ == "__main__":
    run_tests()