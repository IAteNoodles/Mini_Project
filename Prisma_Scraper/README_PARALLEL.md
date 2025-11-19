# 🚀 Parallel News Scraper

Ultra-fast news scraping with **16-core parallel processing** for bias analysis and sentiment classification.

## ⚡ Performance Features

- **16 parallel processes** using all CPU cores
- **4 concurrent browser windows** per process (64 total browsers)
- **Up to 10x faster** than sequential scraping
- **Intelligent load balancing** across topics
- **Real-time performance monitoring**

## 🏗️ Architecture

```
Main Process
├── Process 1 (Topics 1-5)
│   ├── Browser Window 1 → Article scraping
│   ├── Browser Window 2 → Article scraping
│   ├── Browser Window 3 → Article scraping
│   └── Browser Window 4 → Article scraping
├── Process 2 (Topics 6-10)
│   └── 4 Browser Windows...
├── ...
└── Process 16 (Topics 71-76)
    └── 4 Browser Windows...
```

## 🚀 Quick Start

### 1. Auto Setup (Recommended)
```bash
python start_parallel_scraper.py
```
This will:
- Check all requirements
- Install missing packages
- Configure Playwright
- Run tests
- Start scraping automatically

### 2. Manual Setup

#### Install Dependencies
```bash
pip install requests python-dotenv playwright trafilatura pymongo psutil
python -m playwright install chromium
```

#### Configure Environment
Create `.env` file:
```
MONGO_URI=your_mongodb_connection_string
NEWS_API_KEY=your_newsapi_key
```

#### Run Tests
```bash
python test_parallel_scraper.py
```

#### Start Scraping
```bash
python news_fetcher.py
```

## ⚙️ Configuration

### Auto-Configuration
The scraper automatically adjusts settings based on your system:

```python
from config_parallel import auto_configure
auto_configure()  # Optimizes for your CPU/RAM
```

### Manual Configuration
Edit `config_parallel.py`:

```python
NUM_CORES = 16              # Parallel processes
MAX_BROWSER_WINDOWS = 4     # Browsers per process  
MIN_CONTENT_LENGTH = 200    # Article minimum length
SLEEP_TIME_SECONDS = 900    # 15 minutes between cycles
```

### System Requirements

| Specification | Minimum | Recommended |
|---------------|---------|-------------|
| CPU Cores     | 4       | 16+         |
| RAM           | 4GB     | 16GB+       |
| Storage       | 1GB     | 10GB+       |
| Internet      | 10Mbps  | 100Mbps+    |

## 📊 Performance Monitoring

### Real-time Stats
```bash
python performance_monitor.py monitor
```

### Performance Estimates
```bash
python performance_monitor.py
```

Example output:
```
🔢 PARALLEL PROCESSING PERFORMANCE ESTIMATION
==================================================
🖥️  System Specifications:
   Physical CPU Cores: 16
   Logical CPU Cores: 32
   Total RAM: 32.0 GB

⏱️  Estimated Processing Times:
   Sequential (1 core): 21.1 hours
   Parallel (16 cores): 98.4 minutes
   + Browser parallel: 24.6 minutes

🚀 Estimated Speedup: 51.5x faster
```

## 🔧 Troubleshooting

### Common Issues

#### High Memory Usage
- Reduce `MAX_BROWSER_WINDOWS` in config
- Close other applications
- Add more RAM

#### Browser Crashes
```python
# In config_parallel.py, use conservative settings:
MAX_BROWSER_WINDOWS = 2
BROWSER_ARGS = [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-gpu',
    '--disable-images'  # Faster scraping
]
```

#### MongoDB Connection Issues
- Check connection string format
- Verify network connectivity
- Test with MongoDB Compass

#### Rate Limiting
- Increase delays between requests
- Reduce number of parallel processes
- Use VPN/proxy rotation

### Debug Mode
```bash
# Run with debug output
python -u news_fetcher.py | tee scraper.log
```

## 📈 Performance Optimization Tips

### 1. CPU Optimization
```python
# Match your CPU cores
import psutil
NUM_CORES = psutil.cpu_count()
```

### 2. Memory Optimization
```python
# Conservative browser settings
BROWSER_ARGS = [
    '--no-sandbox',
    '--disable-dev-shm-usage', 
    '--disable-gpu',
    '--disable-images',
    '--disable-javascript'  # If not needed
]
```

### 3. Network Optimization
- Use wired connection instead of WiFi
- Close bandwidth-heavy applications
- Consider CDN/proxy for better speeds

### 4. Storage Optimization
- Use SSD instead of HDD
- Ensure sufficient free space
- Regular database maintenance

## 🧪 Testing

### Full Test Suite
```bash
python test_parallel_scraper.py
```

Tests include:
- ✅ Configuration validation
- ✅ MongoDB connection
- ✅ News API access
- ✅ Playwright browser
- ✅ Parallel processing

### Individual Tests
```bash
# Test MongoDB only
python -c "from test_parallel_scraper import test_mongodb_connection; test_mongodb_connection()"

# Test News API only  
python -c "from test_parallel_scraper import test_news_api; test_news_api()"
```

## 📋 Monitoring & Logs

### Real-time Monitoring
The scraper provides detailed real-time feedback:

```
🚀 Starting scraping cycle #1
⚙️  Configuration: 16 processes × 4 browser windows
====================================================================
📊 Processing 76 topics across 5 batches
📋 Topics per process: 5
🔄 Starting 16 parallel processes...

📊 Cycle #1 Summary:
⏱️  Duration: 127.3 seconds (2.1 minutes)
📰 Articles processed: 342
⚡ Throughput: 161.2 articles/minute
🎯 Average per process: 21.4 articles
```

### Performance Logs
- CPU usage tracking
- Memory consumption monitoring
- Article processing rates
- Error rate statistics

## 🔄 Scaling

### Horizontal Scaling (Multiple Machines)
```python
# Machine 1: Topics 1-25
TOPICS = TOPICS[:25]

# Machine 2: Topics 26-50  
TOPICS = TOPICS[25:50]

# Machine 3: Topics 51-76
TOPICS = TOPICS[50:]
```

### Vertical Scaling (More Resources)
```python
# For 32-core machines
NUM_CORES = 32
MAX_BROWSER_WINDOWS = 8  # 256 total browsers!
```

## 🛡️ Production Deployment

### Docker Setup
```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    chromium \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN playwright install chromium

COPY . .
CMD ["python", "news_fetcher.py"]
```

### Process Management
```bash
# Using supervisord
sudo apt install supervisor

# Create /etc/supervisor/conf.d/news-scraper.conf
[program:news-scraper]
command=python /path/to/news_fetcher.py
directory=/path/to/scraper
user=scraper
autostart=true
autorestart=true
```

### Health Monitoring
```bash
# Check if scraper is running
ps aux | grep news_fetcher.py

# Monitor resource usage
htop

# Check database growth
mongo --eval "db.articles.count()"
```

## 📊 Analytics Integration

The scraped data is ready for:
- **Bias Analysis** → `classifier/bias_analysis.py`
- **Sentiment Classification** → `classifier/process_articles.py` 
- **Topic Modeling** → Ready for ML pipelines
- **Real-time Dashboards** → Connect to visualization tools

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**🚀 Happy Scraping!** 

For issues or questions, please check the troubleshooting section or create an issue in the repository.
