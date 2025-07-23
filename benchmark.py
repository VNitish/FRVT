#!/usr/bin/env python3
"""
Performance benchmark script for FRT service optimizations
"""
import asyncio
import time
import requests
import base64
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
import argparse

class FRTBenchmark:
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def create_test_image_b64(self) -> str:
        """Create a simple test image encoded as base64"""
        # This would normally be a real face image
        # For demo purposes, using a small placeholder
        import io
        from PIL import Image
        import numpy as np
        
        # Create a simple test image (normally would be a face)
        img = Image.new('RGB', (224, 224), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    def benchmark_verification(self, num_requests: int = 100) -> dict:
        """Benchmark face verification endpoint"""
        test_img_b64 = self.create_test_image_b64()
        
        times = []
        errors = 0
        
        print(f"Running verification benchmark with {num_requests} requests...")
        
        for i in range(num_requests):
            start_time = time.time()
            try:
                response = self.session.post(
                    f"{self.base_url}/verify",
                    json={
                        "img1": test_img_b64,
                        "img2": test_img_b64,
                        "threshold": 0.25
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    times.append(time.time() - start_time)
                else:
                    errors += 1
                    
            except Exception as e:
                print(f"Request {i+1} failed: {e}")
                errors += 1
            
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{num_requests} requests")
        
        if times:
            return {
                "total_requests": num_requests,
                "successful_requests": len(times),
                "errors": errors,
                "avg_response_time": statistics.mean(times),
                "median_response_time": statistics.median(times),
                "p95_response_time": statistics.quantiles(times, n=20)[18] if len(times) > 20 else max(times),
                "p99_response_time": statistics.quantiles(times, n=100)[98] if len(times) > 100 else max(times),
                "min_response_time": min(times),
                "max_response_time": max(times),
                "requests_per_second": len(times) / sum(times) if sum(times) > 0 else 0
            }
        else:
            return {"error": "No successful requests"}
    
    def benchmark_concurrent(self, num_concurrent: int = 10, requests_per_thread: int = 10) -> dict:
        """Benchmark concurrent requests"""
        test_img_b64 = self.create_test_image_b64()
        
        def worker():
            times = []
            errors = 0
            
            for _ in range(requests_per_thread):
                start_time = time.time()
                try:
                    response = self.session.post(
                        f"{self.base_url}/verify",
                        json={
                            "img1": test_img_b64,
                            "img2": test_img_b64,
                            "threshold": 0.25
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        times.append(time.time() - start_time)
                    else:
                        errors += 1
                        
                except Exception:
                    errors += 1
            
            return times, errors
        
        print(f"Running concurrent benchmark: {num_concurrent} threads x {requests_per_thread} requests...")
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(worker) for _ in range(num_concurrent)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        
        all_times = []
        total_errors = 0
        
        for times, errors in results:
            all_times.extend(times)
            total_errors += errors
        
        total_requests = num_concurrent * requests_per_thread
        
        if all_times:
            return {
                "concurrent_threads": num_concurrent,
                "requests_per_thread": requests_per_thread,
                "total_requests": total_requests,
                "successful_requests": len(all_times),
                "errors": total_errors,
                "total_time": total_time,
                "avg_response_time": statistics.mean(all_times),
                "median_response_time": statistics.median(all_times),
                "p95_response_time": statistics.quantiles(all_times, n=20)[18] if len(all_times) > 20 else max(all_times),
                "throughput_rps": len(all_times) / total_time
            }
        else:
            return {"error": "No successful requests"}
    
    def get_metrics(self) -> dict:
        """Get current service metrics"""
        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            return {"error": f"Failed to get metrics: {e}"}
        
        return {}

def main():
    parser = argparse.ArgumentParser(description="FRT Service Performance Benchmark")
    parser.add_argument("--url", default="http://localhost:5001", help="Service URL")
    parser.add_argument("--sequential", type=int, default=50, help="Number of sequential requests")
    parser.add_argument("--concurrent", type=int, default=5, help="Number of concurrent threads")
    parser.add_argument("--per-thread", type=int, default=10, help="Requests per thread")
    
    args = parser.parse_args()
    
    benchmark = FRTBenchmark(args.url)
    
    print("=== FRT Service Performance Benchmark ===\n")
    
    # Test service availability
    try:
        response = requests.get(f"{args.url}/ping", timeout=5)
        if response.status_code != 200:
            print("❌ Service not available")
            return
        print("✅ Service is available\n")
    except Exception as e:
        print(f"❌ Service not available: {e}")
        return
    
    # Sequential benchmark
    print("1. Sequential Request Benchmark")
    print("-" * 40)
    seq_results = benchmark.benchmark_verification(args.sequential)
    
    if "error" not in seq_results:
        print(f"Total requests: {seq_results['total_requests']}")
        print(f"Successful: {seq_results['successful_requests']}")
        print(f"Errors: {seq_results['errors']}")
        print(f"Average response time: {seq_results['avg_response_time']:.3f}s")
        print(f"Median response time: {seq_results['median_response_time']:.3f}s")
        print(f"95th percentile: {seq_results['p95_response_time']:.3f}s")
        print(f"99th percentile: {seq_results['p99_response_time']:.3f}s")
        print(f"Requests per second: {seq_results['requests_per_second']:.2f}")
    else:
        print(f"❌ {seq_results['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # Concurrent benchmark
    print("2. Concurrent Request Benchmark")
    print("-" * 40)
    conc_results = benchmark.benchmark_concurrent(args.concurrent, args.per_thread)
    
    if "error" not in conc_results:
        print(f"Concurrent threads: {conc_results['concurrent_threads']}")
        print(f"Requests per thread: {conc_results['requests_per_thread']}")
        print(f"Total requests: {conc_results['total_requests']}")
        print(f"Successful: {conc_results['successful_requests']}")
        print(f"Errors: {conc_results['errors']}")
        print(f"Total time: {conc_results['total_time']:.3f}s")
        print(f"Average response time: {conc_results['avg_response_time']:.3f}s")
        print(f"Median response time: {conc_results['median_response_time']:.3f}s")
        print(f"95th percentile: {conc_results['p95_response_time']:.3f}s")
        print(f"Throughput: {conc_results['throughput_rps']:.2f} RPS")
    else:
        print(f"❌ {conc_results['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # Service metrics
    print("3. Service Metrics")
    print("-" * 40)
    metrics = benchmark.get_metrics()
    
    if metrics and "error" not in metrics:
        if "cpu" in metrics:
            print(f"CPU Usage: {metrics['cpu']['avg']:.1f}% avg, {metrics['cpu']['max']:.1f}% max")
        if "memory" in metrics:
            print(f"Memory Usage: {metrics['memory']['avg']:.1f}% avg, {metrics['memory']['max']:.1f}% max")
        if "requests" in metrics:
            print(f"Total Requests: {metrics['requests']['total']}")
            print(f"Average Response Time: {metrics['requests']['avg_response_time']:.3f}s")
        if "cache" in metrics:
            print(f"Cache Hit Rate: {metrics['cache']['hit_rate']:.1f}%")
        if "gpu" in metrics:
            print(f"GPU Memory: {metrics['gpu']['avg_memory']:.1f}% avg")
    else:
        print("❌ Could not retrieve metrics")
    
    print("\n=== Benchmark Complete ===")

if __name__ == "__main__":
    main()
