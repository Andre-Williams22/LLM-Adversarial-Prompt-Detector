import os
from dotenv import load_dotenv
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from typing import Dict, Any, Optional
import logging
import pytz

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class MongoDBManager:
    def __init__(self):
        self.connection_string = os.getenv("MONGODB_URI")
        if not self.connection_string:
            raise ValueError("MONGODB_URI environment variable is required")
        
        self.database_name = os.getenv("MONGODB_DATABASE", "adversarial_detection")
        self.client = None
        self.db = None
        
    async def connect(self):
        """Connect to MongoDB Atlas"""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            self.db = self.client[self.database_name]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"✅ Connected to MongoDB Atlas database: {self.database_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            return False
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    async def log_chat_interaction(self, user_message: str, bot_response: str, 
                                 detection_results: Dict[str, Any], 
                                 latency: float, session_id: Optional[str] = None):
        """Log a chat interaction to MongoDB"""
        try:
            # Get current time in PST timezone
            pst = pytz.timezone('US/Pacific')
            pst_time = datetime.now(pst)
            
            document = {
                "timestamp": pst_time,
                "session_id": session_id,
                "prompt": user_message,  # Changed from user_message to prompt
                "bot_response": bot_response,
                "detection_results": {
                    "is_adversarial": detection_results.get("is_adversarial", False),
                    "scores": detection_results.get("scores", {}),
                    "reason": detection_results.get("reason", ""),
                    "threshold": detection_results.get("threshold", 0.0)
                },
                "latency_seconds": latency,
                "metadata": {
                    "model_count": len(detection_results.get("scores", [])),
                    "user_agent": "web_interface",
                    "timezone": "US/Pacific"
                }
            }
            
            result = await self.db.chat_interactions.insert_one(document)
            logger.info(f"Chat interaction logged to MongoDB: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to log chat interaction: {e}")
            return None
    
    async def log_model_performance(self, model_name: str, inference_time: float, 
                                  score: float, input_text: str):
        """Log individual model performance metrics"""
        try:
            # Get current time in PST timezone
            pst = pytz.timezone('US/Pacific')
            pst_time = datetime.now(pst)
            
            document = {
                "timestamp": pst_time,
                "model_name": model_name,
                "inference_time_seconds": inference_time,
                "adversarial_score": score,
                "input_length": len(input_text),
                "input_hash": hash(input_text),  # For privacy while allowing deduplication
                "timezone": "US/Pacific"
            }
            
            result = await self.db.model_performance.insert_one(document)
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to log model performance: {e}")
            return None
    
    async def get_chat_statistics(self, days: int = 7):
        """Get chat statistics for the last N days"""
        try:
            from datetime import timedelta
            # Use PST timezone for date calculations
            pst = pytz.timezone('US/Pacific')
            pst_now = datetime.now(pst)
            start_date = pst_now - timedelta(days=days)
            
            pipeline = [
                {"$match": {"timestamp": {"$gte": start_date}}},
                {"$group": {
                    "_id": None,
                    "total_chats": {"$sum": 1},
                    "adversarial_detected": {
                        "$sum": {"$cond": ["$detection_results.is_adversarial", 1, 0]}
                    },
                    "avg_latency": {"$avg": "$latency_seconds"},
                    "unique_sessions": {"$addToSet": "$session_id"}
                }},
                {"$project": {
                    "total_chats": 1,
                    "adversarial_detected": 1,
                    "adversarial_rate": {
                        "$divide": ["$adversarial_detected", "$total_chats"]
                    },
                    "avg_latency": 1,
                    "unique_sessions_count": {"$size": "$unique_sessions"}
                }}
            ]
            
            result = await self.db.chat_interactions.aggregate(pipeline).to_list(1)
            return result[0] if result else {}
        except Exception as e:
            logger.error(f"Failed to get chat statistics: {e}")
            return {}
    
    async def get_model_performance_stats(self, model_name: str = None, days: int = 7):
        """Get model performance statistics"""
        try:
            from datetime import timedelta
            # Use PST timezone for date calculations
            pst = pytz.timezone('US/Pacific')
            pst_now = datetime.now(pst)
            start_date = pst_now - timedelta(days=days)
            
            match_filter = {"timestamp": {"$gte": start_date}}
            if model_name:
                match_filter["model_name"] = model_name
            
            pipeline = [
                {"$match": match_filter},
                {"$group": {
                    "_id": "$model_name",
                    "total_inferences": {"$sum": 1},
                    "avg_inference_time": {"$avg": "$inference_time_seconds"},
                    "avg_score": {"$avg": "$adversarial_score"},
                    "max_score": {"$max": "$adversarial_score"},
                    "min_score": {"$min": "$adversarial_score"}
                }}
            ]
            
            results = await self.db.model_performance.aggregate(pipeline).to_list(None)
            return results
        except Exception as e:
            logger.error(f"Failed to get model performance stats: {e}")
            return []

# Global MongoDB manager instance
mongodb_manager = MongoDBManager()
