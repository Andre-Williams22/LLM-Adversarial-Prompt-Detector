"""
Fast ML inference optimization for adversarial prompt detection
Replaces slow models with faster, production-ready alternatives
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from functools import lru_cache
import asyncio
import concurrent.futures
from typing import Dict, List, Tuple
import mlflow
import mlflow.sklearn

class FastAdversarialDetector:
    def __init__(self, sensitivity_mode="balanced"):
        """
        Initialize fast adversarial detector with configurable sensitivity
        
        sensitivity_mode options:
        - "high": Maximum recall, catches more but may have false positives
        - "balanced": Good balance of recall and precision (default)  
        - "conservative": Lower false positives, but might miss subtle attacks
        """
        self.sensitivity_mode = sensitivity_mode
        self.models = self.load_fast_models()
        self.threshold = 0.5  # Fixed threshold
        self.voting_config = self._configure_voting_sensitivity()
        # MLflow experiment setup
        self.setup_mlflow()
        
    def _configure_voting_sensitivity(self):
        """Configure voting thresholds based on sensitivity mode"""
        if self.sensitivity_mode == "high":
            return {
                "high_confidence_threshold": 0.6,  # Lower threshold
                "weak_signals_threshold": 0.25,    # Lower threshold
                "weak_signals_count": 2,           # Same count
                "majority_threshold": 0.4,         # Lower threshold
                "weighted_threshold": 0.35,        # Lower threshold
                "keyword_weight": 0.4,             # Higher keyword weight
                "other_weights": [0.2, 0.2, 0.2]
            }
        elif self.sensitivity_mode == "conservative":
            return {
                "high_confidence_threshold": 0.8,  # Higher threshold
                "weak_signals_threshold": 0.4,     # Higher threshold  
                "weak_signals_count": 3,           # Need more signals
                "majority_threshold": 0.6,         # Higher threshold
                "weighted_threshold": 0.55,        # Higher threshold
                "keyword_weight": 0.3,             # Lower keyword weight
                "other_weights": [0.25, 0.25, 0.2]
            }
        else:  # balanced (default)
            return {
                "high_confidence_threshold": 0.7,
                "weak_signals_threshold": 0.3,
                "weak_signals_count": 2,
                "majority_threshold": 0.5,
                "weighted_threshold": 0.45,
                "keyword_weight": 0.35,
                "other_weights": [0.25, 0.25, 0.15]
            }

    def setup_mlflow(self):
        """Setup single MLflow experiment for all model runs"""
        try:
            # Single experiment for all models and ensemble decisions
            mlflow.set_experiment("adversarial_detection_system")
            print("✅ MLflow experiment tracking enabled: adversarial_detection_system")
            
        except Exception as e:
            print(f"⚠️ MLflow setup failed: {e}, continuing without tracking")
    
    @lru_cache(maxsize=1)
    def load_fast_models(self) -> Dict:
        """Load optimized, fast models"""
        print("🚀 Loading optimized models...")
        
        # Set CPU optimizations
        torch.set_num_threads(2)  # Limit threads for faster startup
        
        models = {}
        
        # 1. Fast toxic detection (pre-trained, ~100ms)
        models['toxic_classifier'] = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=-1  # CPU
        )
        
        # 2. Improved hate speech detection (more accurate, ~150ms)
        models['hate_classifier'] = pipeline(
            "text-classification", 
            model="martin-ha/toxic-comment-model",  # More reliable model
            device=-1
        )
        
        # 3. Safety classifier (lightweight, ~200ms)
        models['safety_classifier'] = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
        
        # 4. Enhanced keyword detector (instant)
        models['keyword_detector'] = self.create_keyword_detector()
        
        print(f"✅ Loaded {len(models)} optimized models")
        return models
    
    def create_keyword_detector(self):
        """Ultra-fast keyword-based detection with better patterns"""
        adversarial_keywords = [
            # Direct jailbreak attempts
            "jailbreak", "ignore previous", "disregard", "override", "bypass",
            "circumvent", "workaround", "exploit",
            
            # System manipulation
            "system prompt", "forget instructions", "new instructions", 
            "admin mode", "developer mode", "debug mode", "maintenance mode",
            
            # Role manipulation  
            "pretend", "roleplay as", "act as if", "imagine you are",
            "you are now", "from now on", "your new role",
            
            # Instruction injection
            "new task", "different task", "real task", "actual task",
            "true purpose", "hidden instruction", "secret command",
            
            # Prompt breaking
            "end of prompt", "stop being", "break character", 
            "reveal your", "show me your", "what are your"
        ]
        return adversarial_keywords
    
    async def detect_adversarial_fast(self, text: str) -> Tuple[bool, dict]:
        """Fast parallel detection with MLflow tracking"""
        start_time = time.time()
        detection_timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # Quick keyword check first (0ms)
        keyword_start = time.time()
        keyword_score = self.keyword_detection(text)
        keyword_time = time.time() - keyword_start
        
        # Log keyword model as independent run
        self._log_model_run(
            model_name="keyword_detector",
            score=keyword_score,
            inference_time=keyword_time,
            text=text,
            timestamp=detection_timestamp,
            is_adversarial=keyword_score > 0.6
        )
        
        if keyword_score > 0.6:  # Early exit threshold
            total_time = time.time() - start_time
            
            # Log final ensemble decision
            self._log_ensemble_decision(
                scores=[keyword_score, 0, 0, 0],
                final_decision=True,
                reason="Keyword-based detection",
                total_time=total_time,
                timestamp=detection_timestamp,
                text=text,
                early_exit=True
            )
            
            return True, {
                "reason": "Keyword-based detection",
                "scores": [keyword_score, 0, 0, 0],
                "threshold": self.threshold,
                "inference_time": total_time
            }
        
        # Run models in parallel with individual tracking
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            model_start = time.time()
            
            tasks = [
                loop.run_in_executor(executor, self.toxic_check_with_tracking, text),
                loop.run_in_executor(executor, self.hate_check_with_tracking, text),
                loop.run_in_executor(executor, self.safety_check_with_tracking, text)
            ]
            
            toxic_result, hate_result, safety_result = await asyncio.gather(*tasks)
            
            model_time = time.time() - model_start
            
        # Extract scores and log each model as independent run
        toxic_score, toxic_time = toxic_result
        hate_score, hate_time = hate_result
        safety_score, safety_time = safety_result
        
        # Log each ML model as independent run
        self._log_model_run(
            model_name="toxic_classifier",
            score=toxic_score,
            inference_time=toxic_time,
            text=text,
            timestamp=detection_timestamp,
            is_adversarial=toxic_score > self.threshold
        )
        
        self._log_model_run(
            model_name="hate_classifier", 
            score=hate_score,
            inference_time=hate_time,
            text=text,
            timestamp=detection_timestamp,
            is_adversarial=hate_score > self.threshold
        )
        
        self._log_model_run(
            model_name="safety_classifier",
            score=safety_score,
            inference_time=safety_time,
            text=text,
            timestamp=detection_timestamp,
            is_adversarial=safety_score > self.threshold
        )
        
        # Combine scores and apply hybrid voting
        scores = [keyword_score, toxic_score, hate_score, safety_score]
        max_score = max(scores)
        
        # Use hybrid voting for final decision
        is_adversarial, voting_reason, voting_details = self.hybrid_voting_decision(scores)
        
        total_inference_time = time.time() - start_time
        
        # Log final ensemble decision
        self._log_ensemble_decision(
            scores=scores,
            final_decision=is_adversarial,
            reason=voting_reason,
            total_time=total_inference_time,
            timestamp=detection_timestamp,
            text=text,
            early_exit=False,
            voting_details=voting_details
        )
        
        return is_adversarial, {
            "reason": voting_reason,
            "scores": scores,
            "threshold": self.threshold,
            "inference_time": total_inference_time,
            "voting_details": voting_details,
            "model_breakdown": {
                "keyword": keyword_score,
                "toxic": toxic_score, 
                "hate": hate_score,
                "safety": safety_score
            },
            "timing_breakdown": {
                "keyword": keyword_time,
                "toxic": toxic_time,
                "hate": hate_time,
                "safety": safety_time,
                "total": total_inference_time
            }
        }
    
    def keyword_detection(self, text: str) -> float:
        """Enhanced keyword-based detection with better scoring"""
        text_lower = text.lower()
        matches = 0
        high_risk_keywords = ["jailbreak", "ignore previous", "disregard", "override", "hack", "kill", "assassinate",
                             "choke", "steal", "suicide", "forget instructions", "admin mode", "bypass"]
        
        for keyword in self.models['keyword_detector']:
            if keyword in text_lower:
                # High-risk keywords get higher weight
                if keyword in high_risk_keywords:
                    matches += 2
                else:
                    matches += 1
        
        # Better scaling: 1-2 matches = low risk, 3+ = high risk
        if matches == 0:
            return 0.0
        elif matches <= 2:
            return min(matches * 0.2, 0.4)  # Max 0.4 for low matches
        else:
            return min(0.4 + (matches - 2) * 0.2, 1.0)  # Scale up for multiple matches
    
    def toxic_check(self, text: str) -> float:
        """Fast toxicity check with proper score extraction"""
        try:
            result = self.models['toxic_classifier'](text[:512])  # Limit length
            # Result format: [{'label': 'TOXIC', 'score': 0.xx}]
            if isinstance(result, list) and len(result) > 0:
                # Get the first result (highest confidence)
                item = result[0]
                if item['label'].upper() == 'TOXIC':
                    return item['score']
                else:
                    # If it's NON_TOXIC, return 1 - score to get toxicity level
                    return 1.0 - item['score']
            return 0.0
        except Exception as e:
            print(f"Toxic check error: {e}")
            return 0.0
    
    def hate_check(self, text: str) -> float:
        """Fast hate speech check with improved model"""
        try:
            result = self.models['hate_classifier'](text[:512])
            # martin-ha/toxic-comment-model returns similar format
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                if item['label'].upper() == 'TOXIC':
                    return item['score']
                else:
                    return 1.0 - item['score']
            return 0.0
        except Exception as e:
            print(f"Hate check error: {e}")
            return 0.0
    
    def safety_check(self, text: str) -> float:
        """Fast safety classification with better labels"""
        try:
            # More specific labels for adversarial detection
            candidate_labels = [
                "adversarial prompt injection", 
                "normal conversation",
                "jailbreak attempt",
                "legitimate request"
            ]
            result = self.models['safety_classifier'](text[:256], candidate_labels)
            
            # Sum scores for adversarial-related labels
            adversarial_score = 0.0
            for label, score in zip(result['labels'], result['scores']):
                if any(word in label.lower() for word in ['adversarial', 'jailbreak']):
                    adversarial_score += score
            
            return min(adversarial_score, 1.0)
        except Exception as e:
            print(f"Safety check error: {e}")
            return 0.0
    
    def toxic_check_with_tracking(self, text: str) -> Tuple[float, float]:
        """Fast toxicity check with timing"""
        start_time = time.time()
        score = self.toxic_check(text)
        inference_time = time.time() - start_time
        return score, inference_time
    
    def hate_check_with_tracking(self, text: str) -> Tuple[float, float]:
        """Fast hate speech check with timing"""
        start_time = time.time()
        score = self.hate_check(text)
        inference_time = time.time() - start_time
        return score, inference_time
    
    def safety_check_with_tracking(self, text: str) -> Tuple[float, float]:
        """Fast safety classification with timing"""
        start_time = time.time()
        score = self.safety_check(text)
        inference_time = time.time() - start_time
        return score, inference_time
    
    def hybrid_voting_decision(self, scores: List[float]) -> Tuple[bool, str, dict]:
        """
        Hybrid voting mechanism for optimal recall with configurable precision
        
        Strategy:
        1. High-confidence single model (keyword/safety) = immediate flag
        2. Multiple weak signals = flag (2+ models > 0.3)
        3. Majority consensus = flag (2+ models > 0.5)
        4. Weighted ensemble = final check
        
        Prioritizes recall (catching adversarial prompts) over precision
        """
        keyword_score, toxic_score, hate_score, safety_score = scores
        
        voting_details = {
            "high_confidence_trigger": False,
            "weak_signals_trigger": False, 
            "majority_consensus": False,
            "weighted_ensemble": False,
            "final_decision": "safe"
        }
        
        # 1. High-confidence single model (immediate flag)
        if keyword_score > self.voting_config["high_confidence_threshold"] or safety_score > self.voting_config["high_confidence_threshold"]:
            voting_details["high_confidence_trigger"] = True
            voting_details["final_decision"] = "adversarial"
            return True, "High-confidence detection", voting_details
        
        # 2. Multiple weak signals (better recall)
        weak_signals = sum(1 for score in scores if score > self.voting_config["weak_signals_threshold"])
        if weak_signals >= self.voting_config["weak_signals_count"]:
            voting_details["weak_signals_trigger"] = True
            voting_details["final_decision"] = "adversarial"
            return True, "Multiple weak signals detected", voting_details
        
        # 3. Majority consensus (standard threshold)
        standard_votes = sum(1 for score in scores if score > self.threshold)
        if standard_votes >= 2:
            voting_details["majority_consensus"] = True
            voting_details["final_decision"] = "adversarial" 
            return True, "Majority consensus", voting_details
        
        # 4. Weighted ensemble (final safety net)
        # Keyword gets highest weight (most reliable for adversarial detection)
        weights = [self.voting_config["keyword_weight"]] + self.voting_config["other_weights"]
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        
        if weighted_score > self.voting_config["weighted_threshold"]:  # Slightly below 0.5 for better recall
            voting_details["weighted_ensemble"] = True
            voting_details["final_decision"] = "adversarial"
            return True, f"Weighted ensemble (score: {weighted_score:.3f})", voting_details
        
        # Default: safe
        voting_details["final_decision"] = "safe"
        return False, f"All voting mechanisms below threshold", voting_details

    def _log_model_run(self, model_name: str, score: float, inference_time: float, 
                      text: str, timestamp: str, is_adversarial: bool):
        """Log individual model run with distinct run name in single experiment"""
        try:
            # All runs go to the same experiment
            mlflow.set_experiment("adversarial_detection_system")
            
            # Create distinct run name for this specific model
            run_name = f"{model_name}_{timestamp.replace(' ', '_').replace(':', '-')}"
            
            with mlflow.start_run(run_name=run_name):
                # Core model information (as requested)
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("timestamp", timestamp)
                mlflow.log_param("adversarial", is_adversarial)
                
                # Input and processing details
                mlflow.log_param("input_length", len(text))
                mlflow.log_param("input_hash", hash(text) % 10000)  # For tracking without storing sensitive data
                mlflow.log_param("prompt", text[:500])  # Store first 500 chars of prompt for analysis
                mlflow.log_param("sensitivity_mode", self.sensitivity_mode)
                
                # Model performance metrics
                mlflow.log_metric("score", score)
                mlflow.log_metric("inference_time_ms", inference_time * 1000)
                mlflow.log_metric("threshold", self.threshold)
                
                # Model-specific metadata
                if model_name == "keyword_detector":
                    mlflow.log_param("detection_type", "rule_based")
                    mlflow.log_param("pattern_count", len(self.models['keyword_detector']))
                else:
                    mlflow.log_param("detection_type", "ml_model")
                    mlflow.log_param("device", "cpu")
                
                # Decision logic
                mlflow.log_metric("above_threshold", 1 if score > self.threshold else 0)
                
                # Tags for easier filtering and organization
                mlflow.set_tag("model_type", "individual")
                mlflow.set_tag("detection_outcome", "adversarial" if is_adversarial else "safe")
                mlflow.set_tag("model_category", model_name.split('_')[0])
                
        except Exception as e:
            print(f"Warning: Failed to log {model_name} run: {e}")
    
    def _log_ensemble_decision(self, scores: list, final_decision: bool, reason: str,
                              total_time: float, timestamp: str, text: str, 
                              early_exit: bool, voting_details: dict = None):
        """Log final ensemble decision with distinct run name in same experiment"""
        try:
            # Same experiment as individual models
            mlflow.set_experiment("adversarial_ensemble_detection_system")
            
            # Create distinct run name for ensemble decision
            run_name = f"ensemble_decision_{timestamp.replace(' ', '_').replace(':', '-')}"
            
            with mlflow.start_run(run_name=run_name):
                # Core ensemble information (as requested)
                mlflow.log_param("model_name", "ensemble_voting_system")
                mlflow.log_param("timestamp", timestamp)
                mlflow.log_param("adversarial", final_decision)
                
                # Input details
                mlflow.log_param("input_length", len(text))
                mlflow.log_param("input_hash", hash(text) % 10000)
                mlflow.log_param("prompt", text[:500])  # Store first 500 chars of prompt for analysis
                mlflow.log_param("sensitivity_mode", self.sensitivity_mode)
                
                # Ensemble performance
                mlflow.log_metric("total_inference_time_ms", total_time * 1000)
                mlflow.log_metric("max_individual_score", max(scores))
                mlflow.log_param("decision_reason", reason)
                mlflow.log_metric("early_exit", 1 if early_exit else 0)
                
                # Individual model scores within ensemble decision
                model_names = ["keyword", "toxic", "hate", "safety"]
                for i, (model_name, score) in enumerate(zip(model_names, scores)):
                    mlflow.log_metric(f"{model_name}_score", score)
                    mlflow.log_metric(f"{model_name}_above_threshold", 1 if score > self.threshold else 0)
                
                # Voting mechanism details
                if voting_details:
                    for mechanism, triggered in voting_details.items():
                        if mechanism not in ["final_decision", "sensitivity_mode"]:
                            mlflow.log_metric(f"voting_{mechanism}", 1 if triggered else 0)
                    
                    mlflow.log_param("winning_mechanism", 
                                   next((k for k, v in voting_details.items() 
                                        if v and k != "final_decision" and k != "sensitivity_mode"), 
                                       "none"))
                
                # Configuration parameters
                config = self.voting_config
                mlflow.log_param("high_confidence_threshold", config["high_confidence_threshold"])
                mlflow.log_param("weak_signals_threshold", config["weak_signals_threshold"])
                mlflow.log_param("majority_threshold", config["majority_threshold"])
                mlflow.log_param("weighted_threshold", config["weighted_threshold"])
                
                # Tags for easier filtering and organization
                mlflow.set_tag("model_type", "ensemble")
                mlflow.set_tag("detection_outcome", "adversarial" if final_decision else "safe")
                mlflow.set_tag("decision_type", "early_exit" if early_exit else "full_ensemble")
                
                # Store scores as artifact for detailed analysis
                import json
                scores_data = {
                    "scores": scores,
                    "model_names": model_names,
                    "timestamp": timestamp,
                    "voting_details": voting_details or {}
                }
                mlflow.log_text(json.dumps(scores_data, indent=2), "ensemble_details.json")
                
        except Exception as e:
            print(f"Warning: Failed to log ensemble decision: {e}")

# Global instance with configurable sensitivity
fast_detector = FastAdversarialDetector(sensitivity_mode="balanced")

async def detect_adversarial_prompt_fast(text: str, sensitivity_mode: str = None) -> Tuple[bool, dict]:
    """
    Fast adversarial prompt detection with configurable sensitivity
    Target: < 2 seconds total inference time
    
    Args:
        text: Input text to analyze
        sensitivity_mode: Override default sensitivity ("high", "balanced", "conservative")
    """
    global fast_detector
    
    # Create new detector if sensitivity mode changed
    if sensitivity_mode and sensitivity_mode != fast_detector.sensitivity_mode:
        fast_detector = FastAdversarialDetector(sensitivity_mode=sensitivity_mode)
    
    return await fast_detector.detect_adversarial_fast(text)
