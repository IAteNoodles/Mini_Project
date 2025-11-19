#!/usr/bin/env python3
"""
Performance monitoring script for the parallel news fetcher.
This script provides insights into the performance gains from parallelization.
"""

import time
import psutil
import threading
from collections import defaultdict, deque
import os

class PerformanceMonitor:
    """Monitor system performance during scraping."""
    
    def __init__(self):
        self.start_time = None
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.process_count = deque(maxlen=100)
        self.monitoring = False
        self.stats = defaultdict(int)
        
    def start_monitoring(self):
        """Start performance monitoring in a separate thread."""
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print("🔍 Performance monitoring started...")
        
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
        print("📊 Performance monitoring stopped.")
        
    def _monitor_loop(self):
        """Monitor system resources continuously."""
        while self.monitoring:
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu_percent)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.append(memory.percent)
                
                # Count active processes
                process_count = len([p for p in psutil.process_iter() if p.is_running()])
                self.process_count.append(process_count)
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in monitoring: {e}")
                
    def log_article_processed(self, topic):
        """Log that an article was processed for a topic."""
        self.stats[f'articles_{topic}'] += 1
        self.stats['total_articles'] += 1
        
    def log_process_started(self, process_id):
        """Log that a new process started."""
        self.stats[f'process_{process_id}_started'] = time.time()
        
    def log_process_finished(self, process_id):
        """Log that a process finished."""
        start_time = self.stats.get(f'process_{process_id}_started')
        if start_time:
            duration = time.time() - start_time
            self.stats[f'process_{process_id}_duration'] = duration
            
    def get_summary(self):
        """Get performance summary."""
        if not self.start_time:
            return "Monitoring not started"
            
        total_time = time.time() - self.start_time
        
        # Calculate averages
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        max_cpu = max(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        max_memory = max(self.memory_usage) if self.memory_usage else 0
        
        # Calculate throughput
        articles_per_minute = (self.stats['total_articles'] / total_time * 60) if total_time > 0 else 0
        
        summary = f"""
📊 PERFORMANCE SUMMARY
{'='*50}
⏱️  Total Runtime: {total_time:.2f} seconds ({total_time/60:.1f} minutes)
📰 Articles Processed: {self.stats['total_articles']}
⚡ Throughput: {articles_per_minute:.1f} articles/minute

💻 CPU Usage:
   Average: {avg_cpu:.1f}%
   Peak: {max_cpu:.1f}%

🧠 Memory Usage:
   Average: {avg_memory:.1f}%
   Peak: {max_memory:.1f}%

🔄 Process Information:
   CPU Cores Available: {psutil.cpu_count()}
   CPU Cores (Logical): {psutil.cpu_count(logical=True)}
   
📈 Efficiency Metrics:
   CPU Utilization: {'High' if avg_cpu > 60 else 'Moderate' if avg_cpu > 30 else 'Low'}
   Memory Pressure: {'High' if avg_memory > 80 else 'Moderate' if avg_memory > 60 else 'Low'}
        """
        
        return summary

def estimate_performance_gain():
    """Estimate the performance gain from parallelization."""
    
    print("🔢 PARALLEL PROCESSING PERFORMANCE ESTIMATION")
    print("=" * 50)
    
    # Get system specs
    cpu_count = psutil.cpu_count()
    cpu_count_logical = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"🖥️  System Specifications:")
    print(f"   Physical CPU Cores: {cpu_count}")
    print(f"   Logical CPU Cores: {cpu_count_logical}")
    print(f"   Total RAM: {memory_gb:.1f} GB")
    
    # Estimate scraping times
    avg_article_scrape_time = 5  # seconds per article
    articles_per_topic = 20  # estimated articles per topic
    total_topics = 76  # from TOPICS list
    
    # Sequential processing time
    sequential_time = total_topics * articles_per_topic * avg_article_scrape_time
    
    # Parallel processing time (assuming 16 cores, some overhead)
    parallel_efficiency = 0.8  # 80% efficiency due to overhead
    parallel_time = sequential_time / (16 * parallel_efficiency)
    
    # Browser window parallelization within each process
    browser_parallel_efficiency = 0.7  # 70% efficiency for browser parallelization
    browser_parallel_time = parallel_time / (4 * browser_parallel_efficiency)
    
    print(f"\n⏱️  Estimated Processing Times:")
    print(f"   Sequential (1 core): {sequential_time/3600:.1f} hours")
    print(f"   Parallel (16 cores): {parallel_time/60:.1f} minutes")
    print(f"   + Browser parallel: {browser_parallel_time/60:.1f} minutes")
    
    speedup = sequential_time / browser_parallel_time
    print(f"\n🚀 Estimated Speedup: {speedup:.1f}x faster")
    
    print(f"\n💡 Optimization Tips:")
    if cpu_count >= 16:
        print("   ✅ System has enough cores for 16 parallel processes")
    else:
        print(f"   ⚠️  System only has {cpu_count} cores, consider reducing NUM_CORES")
    
    if memory_gb >= 8:
        print("   ✅ Sufficient RAM for parallel browser instances")
    else:
        print("   ⚠️  Low RAM may limit parallel browser performance")
    
    return speedup

def monitor_scraping_session():
    """Monitor a scraping session and provide real-time stats."""
    
    monitor = PerformanceMonitor()
    
    print("🚀 STARTING PERFORMANCE MONITORING")
    print("=" * 50)
    print("Press Ctrl+C to stop monitoring and see results...")
    
    monitor.start_monitoring()
    
    try:
        # Simulate monitoring (in real usage, this would run alongside the scraper)
        start_time = time.time()
        while True:
            time.sleep(10)  # Update every 10 seconds
            
            current_time = time.time() - start_time
            if current_time > 60:  # Stop after 1 minute for demo
                break
                
            # In real usage, the scraper would call monitor.log_article_processed()
            # For demo, we'll simulate some activity
            monitor.stats['total_articles'] += 2
            
            print(f"⏰ Running for {current_time:.0f}s - Articles: {monitor.stats['total_articles']}")
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring interrupted by user")
    
    monitor.stop_monitoring()
    print(monitor.get_summary())

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_scraping_session()
    else:
        estimate_performance_gain()
        print("\n" + "=" * 50)
        print("To monitor a live scraping session, run:")
        print("python performance_monitor.py monitor")
