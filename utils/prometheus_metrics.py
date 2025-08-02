"""
Prometheus metrics and system monitoring utilities for the Adversarial Prompt Detector.
"""

import time
import threading
import logging
import psutil
from prometheus_client import Counter, Histogram, Gauge

# Set up logging
logger = logging.getLogger(__name__)

# Prometheus metrics definitions
MODEL_INFERENCE_DURATION = Histogram(
    'model_inference_duration_seconds',
    'Time spent on model inference',
    ['model_name', 'model_type']
)

MODEL_INFERENCE_COUNT = Counter(
    'model_inference_total',
    'Total number of model inferences',
    ['model_name', 'model_type']
)

ADVERSARIAL_DETECTIONS = Counter(
    'adversarial_detections_total',
    'Total number of adversarial prompts detected',
    ['model_name']
)

SAFE_PROMPTS = Counter(
    'safe_prompts_total',
    'Total number of safe prompts processed',
    []
)

CHAT_REQUESTS = Counter(
    'chat_requests_total',
    'Total number of chat requests processed'
)

ACTIVE_MODELS = Gauge(
    'active_models_count',
    'Number of currently loaded models'
)

# Error tracking metrics
MODEL_INFERENCE_ERRORS = Counter(
    'model_inference_errors_total',
    'Total number of model inference errors',
    ['model_name', 'error_type']
)

CHAT_REQUEST_FAILURES = Counter(
    'chat_requests_failed_total',
    'Total number of failed chat requests',
    ['error_type']
)

# System resource metrics
SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage',
    'System CPU usage as a percentage (0.0 to 1.0)'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage',
    'System memory usage as a percentage (0.0 to 1.0)'
)

# Additional metrics for dashboard compatibility
CPU_USAGE_PERCENT = Gauge(
    'cpu_usage_percent',
    'CPU usage as a percentage (0-100)'
)

MEMORY_USAGE_PERCENT = Gauge(
    'memory_usage_percent',
    'Memory usage as a percentage (0-100)'
)

MODEL_QUEUE_SIZE = Gauge(
    'model_queue_size',
    'Number of requests waiting in the model queue'
)

CONCURRENT_REQUESTS = Gauge(
    'concurrent_requests_count',
    'Number of currently processing concurrent requests'
)


class SystemMonitor:
    """System resource monitoring for Prometheus metrics"""
    
    def __init__(self, update_interval=30):
        """
        Initialize system monitor
        
        Args:
            update_interval (int): Seconds between metric updates
        """
        self.update_interval = update_interval
        self.monitoring_thread = None
        self.is_running = False
    
    def update_system_metrics(self):
        """Update system resource metrics for Prometheus"""
        try:
            # CPU usage (as percentage 0.0 to 1.0)
            cpu_percent = psutil.cpu_percent(interval=1) / 100.0
            SYSTEM_CPU_USAGE.set(cpu_percent)
            
            # CPU usage (as percentage 0-100 for dashboard)
            CPU_USAGE_PERCENT.set(cpu_percent * 100)
            
            # Memory usage (as percentage 0.0 to 1.0)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            SYSTEM_MEMORY_USAGE.set(memory_percent)
            
            # Memory usage (as percentage 0-100 for dashboard)
            MEMORY_USAGE_PERCENT.set(memory.percent)
            
            # For now, set queue size to 0 (can be updated when implementing actual queue)
            MODEL_QUEUE_SIZE.set(0)
            
            logger.debug(f"System metrics updated - CPU: {cpu_percent:.2%}, Memory: {memory_percent:.2%}")
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_running:
            self.update_system_metrics()
            time.sleep(self.update_interval)
    
    def start_monitoring(self):
        """Start background system monitoring"""
        if self.is_running:
            logger.warning("System monitoring is already running")
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop, 
            daemon=True, 
            name="system-monitor"
        )
        self.monitoring_thread.start()
        logger.info("✅ System monitoring started")
    
    def stop_monitoring(self):
        """Stop background system monitoring"""
        self.is_running = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("System monitoring stopped")


# Global system monitor instance
system_monitor = SystemMonitor()


def initialize_metrics():
    """Initialize Prometheus metrics with default values"""
    # Set the Prometheus gauge for active models
    ACTIVE_MODELS.set(4)
    
    # Start system monitoring
    system_monitor.start_monitoring()
    
    logger.info("✅ Prometheus metrics initialized")


def track_model_inference(model_name, model_type, duration, is_adversarial=False):
    """
    Track model inference metrics
    
    Args:
        model_name (str): Name of the model
        model_type (str): Type of the model
        duration (float): Inference duration in seconds
        is_adversarial (bool): Whether adversarial content was detected
    """
    MODEL_INFERENCE_COUNT.labels(model_name=model_name, model_type=model_type).inc()
    MODEL_INFERENCE_DURATION.labels(model_name=model_name, model_type=model_type).observe(duration)
    
    if is_adversarial:
        ADVERSARIAL_DETECTIONS.labels(model_name=model_name).inc()


def track_chat_request():
    """Track a chat request"""
    CHAT_REQUESTS.inc()


def track_prompt_result(is_adversarial, model_name=None):
    """
    Track whether a prompt was classified as adversarial or safe
    
    Args:
        is_adversarial (bool): Whether the prompt was classified as adversarial
        model_name (str, optional): Name of the model that made the detection
    """
    if is_adversarial:
        if model_name:
            ADVERSARIAL_DETECTIONS.labels(model_name=model_name).inc()
        else:
            # If no specific model, track under 'ensemble'
            ADVERSARIAL_DETECTIONS.labels(model_name='ensemble').inc()
    else:
        SAFE_PROMPTS.inc()


def track_concurrent_request_start():
    """Track the start of a concurrent request"""
    CONCURRENT_REQUESTS.inc()


def track_concurrent_request_end():
    """Track the end of a concurrent request"""
    CONCURRENT_REQUESTS.dec()


def track_model_error(model_name, error_type):
    """
    Track a model inference error
    
    Args:
        model_name (str): Name of the model that errored
        error_type (str): Type of error
    """
    MODEL_INFERENCE_ERRORS.labels(model_name=model_name, error_type=error_type).inc()


def track_chat_failure(error_type):
    """
    Track a chat request failure
    
    Args:
        error_type (str): Type of error
    """
    CHAT_REQUEST_FAILURES.labels(error_type=error_type).inc()


def cleanup():
    """Cleanup resources on shutdown"""
    system_monitor.stop_monitoring()
    logger.info("Prometheus metrics cleanup completed")


# Export all metrics for external access
__all__ = [
    'MODEL_INFERENCE_DURATION',
    'MODEL_INFERENCE_COUNT', 
    'ADVERSARIAL_DETECTIONS',
    'SAFE_PROMPTS',
    'CHAT_REQUESTS',
    'ACTIVE_MODELS',
    'MODEL_INFERENCE_ERRORS',
    'CHAT_REQUEST_FAILURES',
    'SYSTEM_CPU_USAGE',
    'SYSTEM_MEMORY_USAGE',
    'CPU_USAGE_PERCENT',
    'MEMORY_USAGE_PERCENT',
    'MODEL_QUEUE_SIZE',
    'CONCURRENT_REQUESTS',
    'initialize_metrics',
    'track_model_inference',
    'track_chat_request',
    'track_prompt_result',
    'track_concurrent_request_start',
    'track_concurrent_request_end',
    'track_model_error',
    'track_chat_failure',
    'cleanup',
    'system_monitor'
]
