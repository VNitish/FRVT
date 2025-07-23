"""
Performance monitoring utilities for FRT service
"""
import time
import psutil
import threading
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_memory_used: Optional[float]
    request_count: int
    avg_response_time: float
    cache_hit_rate: float

class PerformanceMonitor:
    """Real-time performance monitoring for the FRT service"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.request_times: deque = deque(maxlen=100)  # Last 100 requests
        self.request_count = 0
        self.cache_hits = 0
        self.cache_requests = 0
        self.lock = threading.RLock()
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self, interval: float = 5.0):
        """Start background performance monitoring"""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background performance monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                metrics = self._collect_metrics()
                with self.lock:
                    self.metrics_history.append(metrics)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU memory (if available)
        gpu_memory_used = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_memory_used = gpus[0].memoryUsed / gpus[0].memoryTotal * 100
        except ImportError:
            pass
        
        # Request metrics
        with self.lock:
            avg_response_time = np.mean(self.request_times) if self.request_times else 0.0
            cache_hit_rate = (self.cache_hits / max(1, self.cache_requests)) * 100
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_memory_used=gpu_memory_used,
            request_count=self.request_count,
            avg_response_time=avg_response_time,
            cache_hit_rate=cache_hit_rate
        )
    
    def record_request(self, response_time: float):
        """Record a request completion time"""
        with self.lock:
            self.request_count += 1
            self.request_times.append(response_time)
    
    def record_cache_hit(self):
        """Record a cache hit"""
        with self.lock:
            self.cache_hits += 1
            self.cache_requests += 1
    
    def record_cache_miss(self):
        """Record a cache miss"""
        with self.lock:
            self.cache_requests += 1
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics"""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, minutes: int = 5) -> Dict:
        """Get performance summary for the last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
        
        if not recent_metrics:
            return {}
        
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        response_times = list(self.request_times)
        
        summary = {
            "time_window_minutes": minutes,
            "sample_count": len(recent_metrics),
            "cpu": {
                "avg": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values)
            },
            "memory": {
                "avg": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values)
            },
            "requests": {
                "total": self.request_count,
                "avg_response_time": np.mean(response_times) if response_times else 0,
                "p95_response_time": np.percentile(response_times, 95) if response_times else 0,
                "p99_response_time": np.percentile(response_times, 99) if response_times else 0
            },
            "cache": {
                "hit_rate": (self.cache_hits / max(1, self.cache_requests)) * 100,
                "total_requests": self.cache_requests,
                "hits": self.cache_hits
            }
        }
        
        # Add GPU metrics if available
        gpu_values = [m.gpu_memory_used for m in recent_metrics if m.gpu_memory_used is not None]
        if gpu_values:
            summary["gpu"] = {
                "avg_memory": np.mean(gpu_values),
                "max_memory": np.max(gpu_values),
                "min_memory": np.min(gpu_values)
            }
        
        return summary
    
    def log_performance_summary(self, minutes: int = 5):
        """Log a performance summary"""
        summary = self.get_metrics_summary(minutes)
        if summary:
            logger.info(f"Performance Summary ({minutes}min):")
            logger.info(f"  CPU: {summary['cpu']['avg']:.1f}% avg, {summary['cpu']['max']:.1f}% max")
            logger.info(f"  Memory: {summary['memory']['avg']:.1f}% avg, {summary['memory']['max']:.1f}% max")
            logger.info(f"  Requests: {summary['requests']['total']} total, {summary['requests']['avg_response_time']:.3f}s avg")
            logger.info(f"  Cache: {summary['cache']['hit_rate']:.1f}% hit rate")
            if 'gpu' in summary:
                logger.info(f"  GPU Memory: {summary['gpu']['avg_memory']:.1f}% avg")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()
