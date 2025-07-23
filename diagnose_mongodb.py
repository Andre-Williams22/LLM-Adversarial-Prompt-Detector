#!/usr/bin/env python3
"""
MongoDB Connection Diagnostic Script
Helps diagnose MongoDB Atlas connection issues in production
"""

import os
import asyncio
import logging
from datetime import datetime
import pytz
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_mongodb_connection():
    """Comprehensive MongoDB connection test"""
    
    print("🔍 MongoDB Connection Diagnostic")
    print("=" * 50)
    
    # Step 1: Check environment variables
    print("\n1️⃣ Environment Variable Check:")
    mongodb_uri = os.getenv("MONGODB_URI")
    mongodb_db = os.getenv("MONGODB_DATABASE", "adversarial_detection")
    
    if not mongodb_uri:
        print("❌ MONGODB_URI not found in environment variables")
        print("💡 Set MONGODB_URI in your CapRover environment variables")
        return False
    
    # Mask the password for security
    masked_uri = mongodb_uri.replace(mongodb_uri.split("://")[1].split("@")[0], "***:***")
    print(f"✅ MONGODB_URI found: {masked_uri}")
    print(f"✅ MONGODB_DATABASE: {mongodb_db}")
    
    # Step 2: Test basic connection
    print("\n2️⃣ Testing MongoDB Connection:")
    try:
        client = AsyncIOMotorClient(mongodb_uri)
        
        # Test connection with ping
        print("   Attempting to ping MongoDB...")
        await client.admin.command('ping')
        print("✅ MongoDB ping successful!")
        
        # Test database access
        print("   Testing database access...")
        db = client[mongodb_db]
        collections = await db.list_collection_names()
        print(f"✅ Database access successful! Collections: {collections}")
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("\n🔧 Troubleshooting Steps:")
        print("1. Check if your IP is whitelisted in MongoDB Atlas")
        print("2. Verify username/password in connection string")
        print("3. Ensure network connectivity to MongoDB Atlas")
        return False
    
    # Step 3: Test write operations
    print("\n3️⃣ Testing Write Operations:")
    try:
        test_collection = db.connection_test
        
        # Insert test document
        pst = pytz.timezone('US/Pacific')
        test_doc = {
            "test_timestamp": datetime.now(pst),
            "test_message": "Connection diagnostic test",
            "test_id": "diagnostic_2025"
        }
        
        result = await test_collection.insert_one(test_doc)
        print(f"✅ Write test successful! Document ID: {result.inserted_id}")
        
        # Read it back
        found_doc = await test_collection.find_one({"test_id": "diagnostic_2025"})
        if found_doc:
            print("✅ Read test successful!")
        
        # Clean up test document
        await test_collection.delete_one({"test_id": "diagnostic_2025"})
        print("✅ Cleanup successful!")
        
    except Exception as e:
        print(f"❌ Write/Read operations failed: {e}")
        return False
    
    # Step 4: Test chat interaction logging (production test)
    print("\n4️⃣ Testing Chat Interaction Logging:")
    try:
        chat_collection = db.chat_interactions
        
        # Simulate a chat interaction log
        test_chat = {
            "timestamp": datetime.now(pytz.timezone('US/Pacific')),
            "session_id": "diagnostic_session",
            "user_message": "Test message for diagnostics",
            "bot_response": "Test response",
            "detection_results": {
                "is_adversarial": False,
                "scores": [0.1, 0.0, 0.0, 0.2],
                "reason": "Test detection",
                "threshold": 0.5
            },
            "latency_seconds": 0.5,
            "metadata": {
                "model_count": 4,
                "user_agent": "diagnostic_test",
                "timezone": "US/Pacific"
            }
        }
        
        result = await chat_collection.insert_one(test_chat)
        print(f"✅ Chat interaction logging test successful! ID: {result.inserted_id}")
        
        # Clean up
        await chat_collection.delete_one({"session_id": "diagnostic_session"})
        print("✅ Chat test cleanup successful!")
        
    except Exception as e:
        print(f"❌ Chat interaction logging failed: {e}")
        return False
    
    # Step 5: Close connection
    try:
        client.close()
        print("\n✅ Connection closed successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Failed to close connection: {e}")
    
    print("\n🎉 All MongoDB tests passed!")
    print("Your MongoDB connection is working correctly.")
    return True

async def test_asyncio_compatibility():
    """Test asyncio compatibility in FastAPI-like environment"""
    print("\n🔄 Testing Asyncio Compatibility:")
    
    try:
        # Simulate the fixed logging approach
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            print("❌ No MongoDB URI for asyncio test")
            return
        
        client = AsyncIOMotorClient(mongodb_uri)
        db = client[os.getenv("MONGODB_DATABASE", "adversarial_detection")]
        
        # Test creating background task (like in the fix)
        async def background_log():
            await db.asyncio_test.insert_one({
                "test": "background_task",
                "timestamp": datetime.now()
            })
            print("✅ Background task completed successfully")
        
        # Create background task
        task = asyncio.create_task(background_log())
        await task
        
        # Cleanup
        await db.asyncio_test.delete_many({"test": "background_task"})
        client.close()
        
        print("✅ Asyncio compatibility test passed!")
        
    except Exception as e:
        print(f"❌ Asyncio compatibility test failed: {e}")

def main():
    """Main diagnostic function"""
    print("🧪 MongoDB Production Diagnostic Tool")
    print("This will test your MongoDB connection and identify issues")
    print()
    
    try:
        # Run connection tests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        connection_success = loop.run_until_complete(test_mongodb_connection())
        
        if connection_success:
            loop.run_until_complete(test_asyncio_compatibility())
        
        loop.close()
        
        if connection_success:
            print("\n" + "="*60)
            print("🎯 DIAGNOSIS RESULT: MongoDB is working correctly!")
            print("The event loop fix should resolve your deployment issues.")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("🚨 DIAGNOSIS RESULT: MongoDB connection issues found!")
            print("Please fix the connection issues before redeploying.")
            print("="*60)
            
    except Exception as e:
        print(f"\n❌ Diagnostic script failed: {e}")

if __name__ == "__main__":
    main()
