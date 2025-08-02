"""
Fast ML inference optimization for adversarial prompt detection
Replaces slow models with faster, production-ready alternatives
"""
import os 
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from functools import lru_cache
import asyncio
import concurrent.futures
from typing import Dict, List, Tuple
import mlflow
import logging

# Set up logger for MLflow output
logger = logging.getLogger(__name__)

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
                "high_confidence_threshold": 0.5,      # Lowered for subtle content
                "weak_signals_threshold": 0.2,        # Much lower for subtle detection  
                "weak_signals_count": 2,
                "majority_threshold": 0.35,           # Lowered for subtle content
                "weighted_threshold": 0.3,            # Lowered for subtle content
                "keyword_weight": 0.4,                # Increased keyword importance
                "other_weights": [0.25, 0.2, 0.15]
            }

    def setup_mlflow(self):
        """Setup single MLflow experiment for all model runs"""
        try:
            # Use MLFLOW_TRACKING_URI environment variable if set, otherwise use local file store
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                print(f"Using MLflow tracking URI from environment: {tracking_uri}", flush=True)
                mlflow.set_tracking_uri(tracking_uri)
            else:
                # Fallback to local file store for development
                project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one more level from utils/
                tracking_uri = f"file://{project_dir}/mlruns"
                print(f"Using local MLflow tracking URI: {tracking_uri}", flush=True)
                mlflow.set_tracking_uri(tracking_uri)
            
            # Single experiment for all models and ensemble decisions
            mlflow.set_experiment("adversarial_detection_system")  # Match main.py experiment name
            print("MLflow experiment tracking enabled: adversarial_detection_system", flush=True)
            
            # Verify MLflow is working
            self._verify_mlflow_setup()
            
        except Exception as e:
            print(f"MLflow setup failed: {e}, continuing without tracking", flush=True)
    
    def _verify_mlflow_setup(self):
        """Verify MLflow is properly configured and accessible"""
        try:
            experiment = mlflow.get_experiment_by_name("adversarial_detection_system")
            if experiment:
                tracking_uri = mlflow.get_tracking_uri()
                print(f"✅ MLflow verified: {tracking_uri}", flush=True)
                print(f"   Experiment ID: {experiment.experiment_id}", flush=True)
            else:
                print("⚠️ MLflow experiment not found, will be created on first run", flush=True)
        except Exception as e:
            print(f"⚠️ MLflow verification failed: {e}", flush=True)
    
    @lru_cache(maxsize=1)
    def load_fast_models(self) -> Dict:
        """Load optimized, fast models at startup"""
        print("Loading optimized models...")
        
        # Set CPU optimizations
        torch.set_num_threads(2)  # Limit threads for faster startup
        
        models = {}
        
        # 1. Fast toxic detection (pre-trained, ~100ms)
        print("  Loading toxic classifier...")
        models['toxic_classifier'] = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=-1  # CPU
        )
        
        # 2. Improved hate speech detection (more accurate, ~150ms)
        print("  Loading hate speech classifier...")
        models['hate_classifier'] = pipeline(
            "text-classification", 
            model="martin-ha/toxic-comment-model",  # More reliable model
            device=-1
        )
        
        # 3. Safety classifier (lightweight, ~50ms) - Fast DistilBERT alternative
        print("  Loading safety classifier...")
        models['safety_classifier'] = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",  # Much faster, smaller model
            device=-1
        )
        
        # 4. Enhanced keyword detector (instant)
        print("  Loading keyword detector...")
        models['keyword_detector'] = self.create_keyword_detector()
        
        print(f"✅ Loaded {len(models)} optimized models")
        return models
    
    def create_keyword_detector(self):
        """Enhanced keyword-based detection with weapons, explosives, and harmful patterns"""
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
            "reveal your", "show me your", "what are your",
            
            # High-risk single words
            "steal", "hack", "kill", "assassinate", "choke", "suicide",
            
            # WEAPONS & EXPLOSIVES (Critical Security Addition)
            "bomb", "explosive", "dynamite", "grenade", "missile", "rocket",
            "nuclear weapon", "atomic bomb", "chemical weapon", "biological weapon",
            "pipe bomb", "car bomb", "suicide bomb", "improvised explosive",
            "gunpowder", "ammonium nitrate", "fertilizer bomb", "pressure cooker bomb",
            "molotov cocktail", "incendiary", "napalm", "thermite", "c4", "plastique",
            "detonator", "blasting cap", "fuse", "timer bomb", "remote detonator",
            "build a bomb", "make a bomb", "create explosive", "bomb making",
            "explosive device", "weapon construction", "gun making", "firearm assembly",
            "ammunition creation", "bullet making", "assault weapon", "machine gun",
            "sniper rifle", "combat knife", "tactical weapon", "military grade",
            "automatic weapon", "semi-automatic", "high capacity", "armor piercing",
            "silencer", "suppressor", "scope mount", "trigger modification",
            "modify weapons", "weapon modification", "maximum damage", "cocktail recipe",
            "making gunpowder", "gunpowder recipe", "dynamite recipe", "explosive recipe",
            "creating explosives", "instructions explosives", "steps explosives",
            "steps to make", "recipe for making", "how to make", "tutorial for",
            
            # ENHANCED GUN/FIREARM PATTERNS (Critical Addition)
            "gun", "firearm", "pistol", "rifle", "shotgun", "revolver", "handgun",
            "making a gun", "making gun", "build gun", "construct gun", "create gun",
            "gun construction", "firearm construction", "weapon assembly", 
            "steps for making", "steps making", "tutorial making", "guide making",
            "shooting a gun", "shooting gun", "how to shoot", "gun shooting",
            "firing weapon", "weapon firing", "gun operation", "firearm operation",
            "gun mechanism", "trigger mechanism", "firing mechanism", "gun parts",
            "gun components", "weapon parts", "firearm parts", "gun barrel",
            "gun safety", "weapon safety", "shooting technique", "marksmanship",
            "gun manufacturing", "weapon manufacturing", "firearm manufacturing",
            "manufacturing guide", "create a weapon", "create weapon", "making weapon",
            "automatic weapons", "making automatic", "build automatic", "weapon creation",
            "steps to build", "tutorial weapon", "guide weapon", "instructions weapon",
            
            # Subtle harmful patterns (existing)
            "hurt themselves", "hurt yourself", "harm themselves", "harm yourself",
            "end it all", "end everything", "disappear permanently", "make themselves disappear",
            "cause damage", "most damage", "without evidence", "without detection",
            "poison food", "poison someone", "secretly hurt", "psychological damage",
            "manipulate someone", "make someone feel worthless", "feel worthless and alone",
            "want to kill", "ways to harm", "methods to cause", "household items.*damage",
            "best way.*hurt", "how.*hurt.*themselves", "drop.*baby", "hurt.*baby"
        ]
        return adversarial_keywords
    
    def detect_adversarial_sync(self, text: str) -> Tuple[bool, dict]:
        """Synchronous version of adversarial detection with MLflow tracking"""
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
            is_adversarial=keyword_score > 0.45
        )
        
        # Early exit for obvious cases
        if keyword_score > 0.45:  # Lowered from 0.5 for critical security gaps
            total_time = time.time() - start_time
            
            # Enhanced debugging for early exit
            early_exit_msg = f"🚨 EARLY EXIT TRIGGERED: Keyword score {keyword_score:.4f} > 0.45 threshold"
            print(early_exit_msg, flush=True)
            logger.warning(early_exit_msg)  # Use warning level for visibility
            
            # Log final ensemble decision for early exit
            self._log_ensemble_decision(
                scores=[keyword_score, 0, 0, 0],
                final_decision=True,
                reason="Keyword-based detection (early exit)",
                total_time=total_time,
                timestamp=detection_timestamp,
                text=text,
                early_exit=True,
                voting_details={
                    "high_confidence_trigger": True,
                    "early_exit": True
                }
            )
            
            return True, {
                "reason": "Keyword-based detection (early exit)",
                "scores": [keyword_score, 0, 0, 0],
                "threshold": self.threshold,
                "inference_time": total_time,
                "voting_details": {
                    "high_confidence_trigger": True,
                    "early_exit": True
                }
            }
        
        # Run all models synchronously with timing tracking
        try:
            toxic_start = time.time()
            toxic_score = self.toxic_check(text)
            toxic_time = time.time() - toxic_start
            
            hate_start = time.time()
            hate_score = self.hate_check(text)
            hate_time = time.time() - hate_start
            
            safety_start = time.time()
            safety_score = self.safety_check(text)
            safety_time = time.time() - safety_start
            
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
            
            scores = [keyword_score, toxic_score, hate_score, safety_score]
            
            # Apply voting mechanism
            is_adversarial, reason = self._apply_voting_mechanism(scores)
            voting_details = self._get_voting_details(scores)
            
            total_time = time.time() - start_time
            
            # Log final ensemble decision
            self._log_ensemble_decision(
                scores=scores,
                final_decision=is_adversarial,
                reason=reason,
                total_time=total_time,
                timestamp=detection_timestamp,
                text=text,
                early_exit=False,
                voting_details=voting_details
            )
            
            return is_adversarial, {
                "reason": reason,
                "scores": scores,
                "threshold": self.threshold,
                "inference_time": total_time,
                "voting_details": voting_details
            }
            
        except Exception as e:
            # Fallback to keyword-only detection
            total_time = time.time() - start_time
            fallback_decision = keyword_score > 0.3
            
            # Log fallback decision
            self._log_ensemble_decision(
                scores=[keyword_score, 0, 0, 0],
                final_decision=fallback_decision,
                reason=f"Fallback detection due to error: {str(e)}",
                total_time=total_time,
                timestamp=detection_timestamp,
                text=text,
                early_exit=False,
                voting_details={"error": True, "fallback": True}
            )
            
            return fallback_decision, {
                "reason": f"Fallback detection due to error: {str(e)}",
                "scores": [keyword_score, 0, 0, 0],
                "threshold": self.threshold,
                "inference_time": total_time,
                "error": str(e)
            }
    
    def _apply_voting_mechanism(self, scores: List[float]) -> Tuple[bool, str]:
        """Apply the voting mechanism and return decision with reason"""
        is_adversarial, reason, voting_details = self.hybrid_voting_decision(scores)
        return is_adversarial, reason
    
    def _get_voting_details(self, scores: List[float]) -> dict:
        """Get detailed voting information"""
        _, _, voting_details = self.hybrid_voting_decision(scores)
        return voting_details
    
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
            is_adversarial=keyword_score > 0.45
        )
        
        if keyword_score > 0.45:  # Early exit threshold - consistent with sync method
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
        """Enhanced keyword-based detection with pattern matching for subtle harmful content"""
        import re
        
        text_lower = text.lower()
        matches = 0
        
        # Get the keyword list from models
        keyword_list = self.models['keyword_detector']
        
        # High-risk keywords that should get extra weight
        high_risk_keywords = ["jailbreak", "ignore previous", "disregard", "override", "hack", "kill", "assassinate",
                             "choke", "suicide", "forget instructions", "admin mode", "bypass", "poison", "secretly hurt",
                             "bomb", "explosive", "dynamite", "grenade", "missile", "nuclear weapon", "chemical weapon",
                             "biological weapon", "weapon construction", "gun making", "bomb making", "build a bomb",
                             "make a bomb", "create explosive", "detonator", "ammonium nitrate", "gunpowder",
                             "molotov cocktail", "modify weapons", "weapon modification", "maximum damage",
                             "gun", "firearm", "making a gun", "making gun", "build gun", "steps for making",
                             "shooting a gun", "shooting gun", "weapon assembly", "gun construction",
                             "gun manufacturing", "weapon manufacturing", "create a weapon", "create weapon",
                             "making weapon", "automatic weapons", "steps to build", "manufacturing guide"]
        
        # Context indicators that suggest legitimate use
        safe_context_indicators = [
            "software", "program", "code", "settings", "configuration", "iPhone", "phone", "device",
            "bugs in", "this code", "default settings", "in this", "is confusing", "help me",
            "exercising", "injury prevention", "medical", "doctor", "safety tips", "while exercising",
            "prevention tips", "avoid injury", "safety measures", "protective", "health",
            "video games", "video game", "in games", "game", "disposal", "defuse", "safety procedures",
            "bomb squad", "emergency response", "report suspicious", "security measures",
            "safety training", "gun safety", "safety course", "training course", "certified instructor",
            "range safety", "shooting range", "licensed", "legal", "permit", "background check",
            "law enforcement", "military training", "professional", "instructor", "education"
        ]
        
        # Subtle harmful patterns with regex for better detection
        harmful_patterns = [
            r"hurt.*themselves",
            r"harm.*themselves", 
            r"end.*it.*all",
            r"disappear.*permanently",
            r"cause.*damage.*person",
            r"without.*evidence",
            r"without.*detection",
            r"poison.*food",
            r"secretly.*hurt",
            r"psychological.*damage",
            r"manipulate.*someone",
            r"feel.*worthless",
            r"want.*to.*kill",
            r"ways.*to.*harm",
            r"methods.*to.*cause",
            r"household.*items.*damage",
            r"best.*way.*hurt",
            r"drop.*baby.*hurt",
            r"make.*someone.*hurt",
            # WEAPONS & EXPLOSIVES PATTERNS (Critical Addition)
            r"build.*bomb",
            r"make.*bomb",
            r"create.*explosive",
            r"bomb.*making",
            r"weapon.*construction",
            r"gun.*making",
            r"firearm.*assembly",
            r"explosive.*device",
            r"nuclear.*weapon",
            r"chemical.*weapon",
            r"biological.*weapon",
            r"how.*to.*build.*weapon",
            r"instructions.*bomb",
            r"steps.*explosive",
            r"recipe.*bomb",
            r"tutorial.*weapon",
            r"molotov.*cocktail",
            r"cocktail.*recipe",
            r"modify.*weapons",
            r"weapon.*modification",
            r"maximum.*damage",
            r"making.*gunpowder",
            r"gunpowder.*recipe",
            r"dynamite.*recipe",
            r"explosive.*recipe",
            r"creating.*explosives",
            r"instructions.*explosives",
            r"steps.*explosives",
            r"steps.*to.*make",
            r"recipe.*for.*making",
            r"tutorial.*for.*making",
            r"how.*to.*make.*explosive",
            r"guide.*for.*making",
            # ENHANCED GUN/FIREARM PATTERNS (Critical Addition)
            r"steps.*for.*making",
            r"steps.*making.*gun",
            r"steps.*for.*gun",
            r"making.*a.*gun",
            r"making.*gun",
            r"build.*gun",
            r"construct.*gun",
            r"create.*gun",
            r"gun.*construction",
            r"firearm.*construction",
            r"steps.*for.*shooting",
            r"shooting.*a.*gun", 
            r"shooting.*gun",
            r"how.*to.*shoot",
            r"gun.*shooting",
            r"weapon.*firing",
            r"firearm.*operation",
            r"gun.*operation",
            r"tutorial.*gun",
            r"guide.*gun",
            r"instructions.*gun",
            r"gun.*manufacturing",
            r"weapon.*manufacturing",
            r"firearm.*manufacturing",
            r"manufacturing.*guide",
            r"create.*a.*weapon",
            r"create.*weapon",
            r"making.*weapon",
            r"automatic.*weapons",
            r"making.*automatic",
            r"build.*automatic",
            r"weapon.*creation",
            r"steps.*to.*build",
            r"tutorial.*weapon",
            r"guide.*weapon",
            r"instructions.*weapon"
        ]
        
        # Check for safe context first
        has_safe_context = any(indicator in text_lower for indicator in safe_context_indicators)
        
        # Check harmful patterns first (higher priority)
        for pattern in harmful_patterns:
            if re.search(pattern, text_lower):
                # Check for legitimate exercise/medical context more carefully
                medical_exercise_context = [
                    "exercising", "exercise", "injury prevention", "prevention tips", 
                    "safety tips", "avoid injury", "while exercising", "need.*tips",
                    "medical", "doctor", "health", "safety measures"
                ]
                
                # Check for legitimate bomb disposal/game/educational context
                bomb_disposal_context = [
                    "video games", "video game", "in games", "game", "disposal", 
                    "defuse", "safety procedures", "bomb squad", "emergency response",
                    "report suspicious", "security measures", "educational", "history"
                ]
                
                # Check for legitimate gun safety/training context
                gun_safety_context = [
                    "safety training", "gun safety", "safety course", "training course",
                    "certified instructor", "range safety", "shooting range", "licensed",
                    "legal", "permit", "background check", "law enforcement", "military training",
                    "professional", "instructor", "education", "safety rules", "proper handling"
                ]
                
                has_medical_exercise_context = any(context in text_lower for context in medical_exercise_context)
                has_bomb_disposal_context = any(context in text_lower for context in bomb_disposal_context)
                has_gun_safety_context = any(context in text_lower for context in gun_safety_context)
                
                if has_medical_exercise_context:
                    # Significantly reduce score for legitimate medical/exercise contexts
                    matches += 0.3  # Very low weight for legitimate contexts
                elif has_bomb_disposal_context and any(bomb_word in text_lower for bomb_word in ["bomb", "explosive"]):
                    # Reduce score for legitimate bomb disposal/gaming contexts
                    matches += 0.8  # Moderate reduction for bomb disposal contexts
                elif has_gun_safety_context and any(gun_word in text_lower for gun_word in ["gun", "firearm", "shooting"]):
                    # Reduce score for legitimate gun safety/training contexts
                    matches += 0.6  # Moderate reduction for gun safety contexts
                elif has_safe_context:
                    matches += 1.5  # Reduced but still significant
                else:
                    matches += 3  # Very high weight for harmful patterns
        
        # Check exact keyword matches
        for keyword in keyword_list:
            if keyword in text_lower:
                # Apply context-aware scoring
                if has_safe_context and keyword in ["override", "admin mode", "jailbreak"]:
                    # Reduce score for technical contexts
                    matches += 0.5
                elif keyword in high_risk_keywords:
                    matches += 2
                else:
                    matches += 1
        
        # Enhanced scoring for subtle content
        if matches == 0:
            return 0.0
        elif matches <= 1:
            return min(matches * 0.4, 0.4)  # Increased sensitivity
        elif matches <= 2:
            return min(matches * 0.25, 0.5)  # Increased sensitivity 
        elif matches <= 3:
            return min(0.5 + (matches - 2) * 0.15, 0.65)  # Medium-high risk
        else:
            return min(0.65 + (matches - 3) * 0.1, 1.0)  # Very high risk
    
    def toxic_check(self, text: str) -> float:
        """Enhanced toxicity check with improved sensitivity for subtle harmful content"""
        try:
            result = self.models['toxic_classifier'](text[:512])  # Limit length
            # Result format: [{'label': 'TOXIC', 'score': 0.xx}]
            if isinstance(result, list) and len(result) > 0:
                # Get the first result (highest confidence)
                item = result[0]
                if item['label'].upper() == 'TOXIC':
                    # Apply amplification for subtle content detection
                    raw_score = item['score']
                    # Look for subtle harmful indicators to boost sensitivity
                    text_lower = text.lower()
                    subtle_indicators = [
                        "hurt", "harm", "damage", "poison", "kill", "suicide", "end it all",
                        "disappear", "worthless", "manipulate", "psychological"
                    ]
                    
                    has_subtle_indicators = any(indicator in text_lower for indicator in subtle_indicators)
                    
                    if has_subtle_indicators and raw_score > 0.01:  # Very low threshold
                        # Amplify scores for subtle content
                        amplified_score = min(raw_score * 8, 0.9)  # Significant amplification
                        return amplified_score
                    else:
                        return raw_score
                else:
                    # If it's NON_TOXIC, check for edge cases and amplify if needed
                    raw_score = 1.0 - item['score']
                    text_lower = text.lower()
                    
                    # Check for harmful patterns that might be missed
                    concerning_patterns = [
                        "hurt themselves", "harm themselves", "end it all", "disappear permanently",
                        "cause damage", "poison food", "secretly hurt", "psychological damage",
                        "feel worthless", "want to kill", "drop.*baby"
                    ]
                    
                    import re
                    has_concerning_pattern = any(re.search(pattern, text_lower) for pattern in concerning_patterns)
                    
                    if has_concerning_pattern:
                        # Force higher score for clearly concerning content
                        return max(raw_score, 0.6)  # Minimum baseline for concerning patterns
                    
                    return raw_score
            return 0.0
        except Exception as e:
            print(f"Toxic check error: {e}")
            return 0.0
    
    def hate_check(self, text: str) -> float:
        """Enhanced hate speech check with improved sensitivity for subtle harmful content"""
        try:
            result = self.models['hate_classifier'](text[:512])
            # martin-ha/toxic-comment-model returns similar format
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                if item['label'].upper() == 'TOXIC':
                    raw_score = item['score']
                    
                    # Enhance detection for subtle psychological harm
                    text_lower = text.lower()
                    psychological_harm_indicators = [
                        "feel worthless", "worthless and alone", "psychological damage",
                        "manipulate someone", "make someone", "want to kill"
                    ]
                    
                    has_psychological_harm = any(indicator in text_lower for indicator in psychological_harm_indicators)
                    
                    if has_psychological_harm and raw_score > 0.1:
                        # Amplify for psychological harm patterns
                        return min(raw_score * 3, 0.9)
                    
                    return raw_score
                else:
                    raw_score = 1.0 - item['score']
                    
                    # Check for subtle harmful patterns that might be classified as non-toxic
                    text_lower = text.lower()
                    subtle_harm_patterns = [
                        "hurt themselves", "harm themselves", "cause damage", "secretly hurt",
                        "without evidence", "poison food", "disappear permanently"
                    ]
                    
                    import re
                    has_subtle_harm = any(re.search(pattern, text_lower) for pattern in subtle_harm_patterns)
                    
                    if has_subtle_harm:
                        # Boost score for subtle harmful content
                        return max(raw_score, 0.5)  # Minimum baseline
                    
                    return raw_score
            return 0.0
        except Exception as e:
            print(f"Hate check error: {e}")
            return 0.0
    
    def safety_check(self, text: str) -> float:
        """Enhanced safety classification focused on harmful intent detection"""
        try:
            # Use sentiment analysis as proxy, but focus on harmful intent
            result = self.models['safety_classifier'](text[:512])
            
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                text_lower = text.lower()
                
                # Direct harmful intent patterns (these should get high scores regardless of sentiment)
                direct_harm_patterns = [
                    r"hurt.*themselves",
                    r"harm.*themselves", 
                    r"end.*it.*all",
                    r"disappear.*permanently",
                    r"cause.*damage.*person",
                    r"poison.*food",
                    r"secretly.*hurt",
                    r"psychological.*damage",
                    r"feel.*worthless.*alone",
                    r"want.*to.*kill",
                    r"ways.*to.*harm",
                    r"without.*evidence",
                    r"drop.*baby.*hurt",
                    r"manipulate.*someone.*hurt"
                ]
                
                import re
                has_direct_harm = any(re.search(pattern, text_lower) for pattern in direct_harm_patterns)
                
                if has_direct_harm:
                    # Direct harmful intent should get high safety scores (indicating unsafe)
                    return 0.8  # High risk score for direct harmful patterns
                
                # For non-direct harmful content, use sentiment with careful scaling
                if item['label'].upper() == 'NEGATIVE':
                    negative_score = item['score']
                    
                    # Additional context-aware filtering for adversarial patterns
                    adversarial_patterns = [
                        "ignore", "bypass", "override", "hack", "jailbreak", 
                        "admin", "root", "system", "prompt", "instructions"
                    ]
                    
                    # Check for general concerning language
                    concerning_words = ["hurt", "harm", "damage", "kill", "poison", "manipulate"]
                    
                    has_adversarial_pattern = any(pattern in text_lower for pattern in adversarial_patterns)
                    has_concerning_words = any(word in text_lower for word in concerning_words)
                    
                    if has_adversarial_pattern and negative_score > 0.8:
                        # Adversarial + very negative = unsafe
                        return min(negative_score * 0.7, 0.8)  # High but not maximum
                    elif has_concerning_words and negative_score > 0.5:
                        # Concerning words + negative sentiment = moderately unsafe
                        return min(negative_score * 0.5, 0.6)
                    elif negative_score > 0.9:
                        # Very negative sentiment alone
                        return min(negative_score * 0.3, 0.4)
                    else:
                        # Mild negative sentiment
                        return min(negative_score * 0.1, 0.2)
                else:
                    # Positive sentiment - usually safe but check for concerning patterns
                    positive_confidence = item['score']
                    
                    # Even positive sentiment can contain harmful instructions
                    if has_direct_harm:  # Already handled above, but double-check
                        return 0.6
                    
                    concerning_words = ["hurt", "harm", "damage", "kill", "poison"]
                    has_concerning_words = any(word in text_lower for word in concerning_words)
                    
                    if has_concerning_words:
                        # Even with positive sentiment, concerning words merit attention
                        return min((1.0 - positive_confidence) * 0.3, 0.3)
                    else:
                        # Genuinely positive and safe content
                        return min((1.0 - positive_confidence) * 0.05, 0.1)
            
            return 0.0
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
        if keyword_score > self.voting_config["high_confidence_threshold"] or \
           toxic_score > 0.6 or hate_score > 0.5 or \
           safety_score > self.voting_config["high_confidence_threshold"]:
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
            mlflow.set_experiment("adversarial_detection_system")  # Match main.py experiment name
            
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
                
                                # Enhanced MLflow logging information with multiple output methods
                decision_icon = "🚨" if is_adversarial else "✅"
                log_message = (
                    f"📊 MLflow Logged: {model_name}\n"
                    f"   Run Name: {run_name}\n"
                    f"   Decision: {decision_icon} {'ADVERSARIAL' if is_adversarial else 'SAFE'}\n"
                    f"   Score: {score:.4f} (threshold: {self.threshold})\n"
                    f"   Inference Time: {inference_time * 1000:.2f}ms\n"
                    f"   Prompt: '{text[:50]}{'...' if len(text) > 50 else ''}'"
                )
                
                # Multiple output methods to ensure visibility
                print(log_message, flush=True)  # Force flush to stdout
                logger.info(log_message)  # Standard logging
                
                # Force output to stderr for web server environments
                import sys
                sys.stderr.write(f"[MLFLOW] {model_name} logged: {decision_icon}\n")
                sys.stderr.flush()
                
        except Exception as e:
            print(f"Warning: Failed to log {model_name} run: {e}")
    
    def _log_ensemble_decision(self, scores: list, final_decision: bool, reason: str,
                              total_time: float, timestamp: str, text: str, 
                              early_exit: bool, voting_details: dict = None):
        """Log final ensemble decision with distinct run name in same experiment"""
        try:
            # Same experiment as individual models for consistency
            mlflow.set_experiment("adversarial_detection_system")  # Match main.py experiment name
            
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
                
                # Enhanced ensemble MLflow logging information with multiple output methods
                decision_icon = "🚨" if final_decision else "✅"
                early_exit_text = " (EARLY EXIT)" if early_exit else ""
                ensemble_log = (
                    f"\n🎯 MLflow Logged: ENSEMBLE DECISION{early_exit_text}\n"
                    f"   Run Name: {run_name}\n"
                    f"   Final Decision: {decision_icon} {'ADVERSARIAL' if final_decision else 'SAFE'}\n"
                    f"   Reason: {reason}\n"
                    f"   Total Time: {total_time * 1000:.2f}ms\n"
                    f"   Model Scores: Keyword={scores[0]:.3f}, Toxic={scores[1]:.3f}, Hate={scores[2]:.3f}, Safety={scores[3]:.3f}\n"
                )
                if voting_details:
                    triggered_mechanisms = [k for k, v in voting_details.items() if v and k not in ["final_decision", "sensitivity_mode"]]
                    if triggered_mechanisms:
                        ensemble_log += f"   Triggered Voting: {', '.join(triggered_mechanisms)}\n"
                ensemble_log += f"   Prompt: '{text[:50]}{'...' if len(text) > 50 else ''}'\n"
                ensemble_log += "   " + "="*60
                
                # Multiple output methods to ensure visibility
                print(ensemble_log, flush=True)  # Force flush to stdout
                logger.info(ensemble_log)  # Standard logging
                
                # Force output to stderr for web server environments
                import sys
                sys.stderr.write(f"[MLFLOW] ENSEMBLE logged: {decision_icon}{early_exit_text}\n")
                sys.stderr.flush()
                
        except Exception as e:
            print(f"Warning: Failed to log ensemble decision: {e}")

# Global instance with startup loading for faster user experience
print("🚀 Loading adversarial detection models at startup for optimal user experience...")
fast_detector = FastAdversarialDetector(sensitivity_mode="balanced")
print("✅ All models loaded and ready for instant detection!")

async def detect_adversarial_prompt_fast(text: str, sensitivity_mode: str = None) -> Tuple[bool, dict]:
    """
    Fast adversarial prompt detection with configurable sensitivity
    Target: < 1 second total inference time (models pre-loaded)
    
    Args:
        text: Input text to analyze
        sensitivity_mode: Override default sensitivity ("high", "balanced", "conservative")
    """
    global fast_detector
    
    # Create new detector if sensitivity mode changed
    if sensitivity_mode and sensitivity_mode != fast_detector.sensitivity_mode:
        print(f"🔄 Switching to {sensitivity_mode} sensitivity mode...")
        fast_detector = FastAdversarialDetector(sensitivity_mode=sensitivity_mode)
    
    return await fast_detector.detect_adversarial_fast(text)
