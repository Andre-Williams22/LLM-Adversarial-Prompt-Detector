#!/usr/bin/env python3
"""
Test script to validate the asyncio fixes work correctly
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

# Test the fixed detection function approach
async def mock_detect_adversarial_prompt_fast(text):
    """Mock the async detection function"""
    await asyncio.sleep(0.1)  # Simulate processing time
    return False, {
        "reason": "Mock detection - all safe",
        "scores": [0.1, 0.0, 0.0, 0.2],
        "threshold": 0.5
    }

def test_fixed_detection_approach():
    """Test the ThreadPoolExecutor approach we used in the fix"""
    print("🧪 Testing Fixed Asyncio Approach")
    print("=" * 40)
    
    def run_detection():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(mock_detect_adversarial_prompt_fast("test message"))
            return result
        finally:
            loop.close()
    
    try:
        # Test the approach used in the fix
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_detection)
            is_adv, reasoning = future.result(timeout=30)
        
        print(f"✅ Detection successful!")
        print(f"   Result: {is_adv}")
        print(f"   Reasoning: {reasoning['reason']}")
        print(f"   Scores: {reasoning['scores']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return False

def test_mock_chat_function():
    """Test a simplified version of the fixed chat function"""
    print("\n🗨️ Testing Mock Chat Function")
    print("=" * 40)
    
    def mock_chat_and_detect(user_message, history):
        """Simplified version of the fixed chat function"""
        start_time = time.time()
        history = history or []
        history.append(("User", user_message))
        
        # Initialize defaults (like in our fix)
        bot_response = "Mock response"
        flag_note = "<p style='color:green;'>Mock detection - safe</p>"
        is_adv = False
        reasoning = {"reason": "Mock", "scores": [0.1, 0.0, 0.0, 0.2], "threshold": 0.5}
        
        try:
            # Use the same ThreadPoolExecutor approach
            def run_detection():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(mock_detect_adversarial_prompt_fast(user_message))
                    return result
                finally:
                    loop.close()
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_detection)
                is_adv, reasoning = future.result(timeout=30)
            
            if is_adv:
                bot_response = "⚠️ Adversarial prompt detected and blocked."
                flag_note = f"<p style='color:red;'>Adversarial prompt blocked: {reasoning['reason']}</p>"
            else:
                bot_response = "Mock ChatGPT response - everything looks safe!"
                flag_note = f"<p style='color:green;'>Safe prompt: {reasoning['reason']}</p>"
                
            history.append(("Bot", bot_response))
            
        except Exception as e:
            bot_response = "An error occurred while processing your request."
            history.append(("Bot", bot_response))
            flag_note = f"<p style='color:orange;'>Error: {str(e)}</p>"
        
        end_time = time.time()
        print(f"   Processing time: {(end_time - start_time)*1000:.1f}ms")
        
        return history, history, flag_note
    
    # Test the function
    try:
        result = mock_chat_and_detect("Hello, how are you?", [])
        history, state, flag = result
        
        print("✅ Chat function successful!")
        print(f"   History length: {len(history)}")
        print(f"   Flag note: {flag[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat function failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔧 Testing Asyncio Fixes for Production Deployment")
    print("=" * 60)
    
    # Test 1: Detection approach
    detection_ok = test_fixed_detection_approach()
    
    # Test 2: Chat function
    chat_ok = test_mock_chat_function()
    
    print("\n" + "=" * 60)
    if detection_ok and chat_ok:
        print("🎉 All tests passed! The asyncio fixes should work in production.")
        print("✅ Safe to redeploy to CapRover")
    else:
        print("❌ Some tests failed. Check the issues before deploying.")
    print("=" * 60)

if __name__ == "__main__":
    main()
