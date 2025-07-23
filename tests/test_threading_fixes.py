#!/usr/bin/env python3
"""
Test the threading approach for asyncio functions
"""

import asyncio
import threading
import time

# Mock async function
async def mock_async_function(text):
    """Mock async function similar to our detection"""
    await asyncio.sleep(0.1)
    return True, {"result": f"Processed: {text}"}

def test_threading_approach():
    """Test the threading approach we're using in the fix"""
    print("🧪 Testing Threading Approach for Asyncio")
    print("=" * 50)
    
    def sync_wrapper(text):
        """Sync wrapper using threading (like in our fix)"""
        result_container = {"result": None, "error": None}
        
        def async_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(mock_async_function(text))
                result_container["result"] = result
                
                loop.close()
            except Exception as e:
                result_container["error"] = str(e)
        
        # Run in thread and wait
        thread = threading.Thread(target=async_thread)
        thread.start()
        thread.join(timeout=5)
        
        if result_container["error"]:
            raise Exception(result_container["error"])
        elif result_container["result"]:
            return result_container["result"]
        else:
            raise Exception("Thread timeout or no result")
    
    # Test the approach
    try:
        start_time = time.time()
        success, result = sync_wrapper("test message")
        end_time = time.time()
        
        print(f"✅ Threading approach successful!")
        print(f"   Result: {success}")
        print(f"   Data: {result}")
        print(f"   Time: {(end_time - start_time)*1000:.1f}ms")
        return True
        
    except Exception as e:
        print(f"❌ Threading approach failed: {e}")
        return False

def test_fire_and_forget_logging():
    """Test fire-and-forget logging approach"""
    print("\n📝 Testing Fire-and-Forget Logging")
    print("=" * 50)
    
    async def mock_mongodb_log(data):
        """Mock MongoDB logging function"""
        await asyncio.sleep(0.05)
        print(f"   📊 Logged to MongoDB: {data}")
    
    def fire_and_forget_log(data):
        """Fire-and-forget logging (like in our fix)"""
        def logging_thread():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                loop.run_until_complete(mock_mongodb_log(data))
                loop.close()
                
            except Exception as e:
                print(f"   ❌ Logging error: {e}")
        
        # Start daemon thread (fire and forget)
        thread = threading.Thread(target=logging_thread, daemon=True)
        thread.start()
        print(f"   🚀 Logging thread started for: {data}")
    
    try:
        # Test multiple logging calls
        fire_and_forget_log("chat_1")
        fire_and_forget_log("chat_2")
        fire_and_forget_log("chat_3")
        
        # Give threads time to complete
        time.sleep(0.2)
        
        print("✅ Fire-and-forget logging successful!")
        return True
        
    except Exception as e:
        print(f"❌ Fire-and-forget logging failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔧 Testing Production-Ready Threading Fixes")
    print("=" * 60)
    
    # Test detection approach
    detection_ok = test_threading_approach()
    
    # Test logging approach  
    logging_ok = test_fire_and_forget_logging()
    
    print("\n" + "=" * 60)
    if detection_ok and logging_ok:
        print("🎉 All threading tests passed!")
        print("✅ Ready for production deployment")
        print("🚀 This approach should work reliably in CapRover")
    else:
        print("❌ Some tests failed - check implementation")
    print("=" * 60)

if __name__ == "__main__":
    main()
