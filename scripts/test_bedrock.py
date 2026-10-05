import time
import json
import boto3

def test_model(model_id: str, model_name: str, prompt: str, region: str = "us-east-1"):
    print(f"\n{'='*60}")
    print(f"🧠 Testing Model: {model_name}")
    print(f"🆔 Model ID: {model_id}")
    print(f"🌍 Region: {region}")
    print(f"💬 Prompt: \"{prompt}\"")
    print(f"{'='*60}")
    
    try:
        # 1. Initialize Bedrock Runtime Client (uses EC2 IAM Role automatically)
        client = boto3.client("bedrock-runtime", region_name=region)
        
        # 2. Prepare message in standard Converse API format
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]
        
        # 3. Time the API call (Latency)
        start_time = time.time()
        
        response = client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={
                "maxTokens": 300,
                "temperature": 0.7,
            }
        )
        
        latency = time.time() - start_time
        
        # 4. Extract generated text and usage metadata
        output_message = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)
        total_tokens = usage.get("totalTokens", 0)
        
        print("\n✨ Generated Response:")
        print(output_message.strip())
        print(f"\n📊 Metrics:")
        print(f"  ⏱️  Latency       : {latency:.2f} seconds")
        print(f"  📥 Input Tokens  : {input_tokens}")
        print(f"  📤 Output Tokens : {output_tokens}")
        print(f"  🔢 Total Tokens  : {total_tokens}")
        print(f"  ✅ Status        : SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ Error calling {model_name}: {e}")
        return False

if __name__ == "__main__":
    test_prompt = "Hello! In 2 concise sentences, introduce yourself as my personal AI agent and tell me what you can do."
    
    # Test 1: Amazon Nova Lite (Super fast & cheap)
    print("\n--- TEST 1: Amazon Nova Lite ---")
    test_model("us.amazon.nova-lite-v1:0", "Amazon Nova Lite", test_prompt)
    
    # Test 2: Claude 3.5 Sonnet (State of the art reasoning)
    print("\n--- TEST 2: Claude 3.5 Sonnet ---")
    test_model("us.anthropic.claude-3-5-sonnet-20241022-v2:0", "Anthropic Claude 3.5 Sonnet v2", test_prompt)
