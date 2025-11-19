# Prisma Scraper LLM Framework

A robust framework for analyzing articles from MongoDB using Large Language Models (LLMs) with bias classification capabilities.

## Features

- **Flexible LLM Support**: Use local models (optimized for 6GB VRAM) or OpenAI API
- **Robust Processing**: Retry mechanisms, fallback classification, and error handling
- **MongoDB Integration**: Seamless connection to Prisma database with batch processing
- **Bias Classification**: Detects Political, Gender, Cultural, and Ideology bias
- **Structured Output**: Uses Instructor and Pydantic for validated responses
- **Multiple Analysis Modes**: Sample testing, full processing, and bias reporting

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (create `.env` file):
```env
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=Prisma
MONGODB_COLLECTION=articles

# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
MODEL_NAME=microsoft/DialoGPT-medium
MAX_LENGTH=512
TEMPERATURE=0.7
BATCH_SIZE=5
```

## Usage

### Command Line Interface

#### Test Database Connection
```bash
python main.py --mode test
```

#### Run Sample Analysis (10 articles)
```bash
python main.py --mode sample --sample-size 10
```

#### Full Analysis with Local Model
```bash
python main.py --mode full --limit 100 --batch-size 5
```

#### Full Analysis with OpenAI
```bash
python main.py --mode full --openai --limit 50
```

#### Generate Bias Analysis Report
```bash
python main.py --mode report
```

### Programmatic Usage

```python
import asyncio
from processor import AnalysisRunner

# Run sample analysis
async def analyze_sample():
    result = await AnalysisRunner.run_sample_analysis(
        sample_size=5,
        use_local_model=True
    )
    print(result)

asyncio.run(analyze_sample())
```

## Architecture

### Core Components

1. **LLM Manager** (`llm_manager.py`)
   - Handles both local and OpenAI models
   - Memory-efficient loading for 6GB VRAM
   - Structured output parsing with fallback

2. **Database Manager** (`database.py`)
   - MongoDB connection and operations
   - Batch processing and error handling
   - Article status tracking

3. **Processor** (`processor.py`)
   - Main processing pipeline
   - Statistics tracking
   - Multiple analysis runners

4. **Models** (`models.py`)
   - Pydantic models for type safety
   - Structured input/output validation
   - Bias classification schemas

### Data Flow

```
MongoDB Articles → ArticleInput → LLM Analysis → ArticleOutput → MongoDB (Updated)
```

### Bias Classification

The framework detects four types of bias:

- **Political Bias**: Partisan language, political favoritism
- **Gender Bias**: Gender stereotypes, discriminatory language
- **Cultural Bias**: Cultural superiority, discriminatory practices
- **Ideology Bias**: Ideological stance affecting objectivity

## Configuration

### Model Configuration

```python
# config.py
class ModelConfig:
    model_name = "microsoft/DialoGPT-medium"  # Local model
    max_length = 512
    temperature = 0.7
    load_in_8bit = True  # Memory optimization
```

### Processing Configuration

```python
class ProcessingConfig:
    batch_size = 5
    delay_between_batches = 1.0  # seconds
    max_retries = 3
```

## Memory Optimization

For 6GB VRAM constraints:

- 8-bit quantization enabled by default
- Batch processing to prevent memory overflow
- Efficient model loading with `device_map="auto"`
- Fallback to regex classification if LLM fails

## Fallback Mechanisms

1. **LLM Failure**: Falls back to regex-based bias detection
2. **Memory Issues**: Automatic batch size reduction
3. **Connection Problems**: Retry with exponential backoff
4. **Parse Errors**: Graceful degradation with partial results

## Output Examples

### Sample Analysis Result
```json
{
    "processed": 10,
    "successful": 8,
    "failed": 2,
    "success_rate": "80.00%",
    "duration_seconds": 45.2,
    "sample_results": [
        {
            "url": "https://example.com/article1",
            "summary_length": 234,
            "bias_detected": true,
            "confidence": 0.85
        }
    ]
}
```

### Bias Analysis Report
```json
{
    "total_processed_articles": 500,
    "bias_percentages": {
        "political": 25.4,
        "gender": 12.8,
        "cultural": 18.2,
        "ideology": 31.6
    },
    "average_confidence": 0.742,
    "high_confidence_articles": 380,
    "low_confidence_articles": 45
}
```

## Error Handling

- Comprehensive logging with `loguru`
- Graceful degradation on failures
- Detailed error reporting and statistics
- Retry mechanisms with exponential backoff

## Contributing

1. Follow the existing code structure
2. Add comprehensive error handling
3. Include type hints and docstrings
4. Test with both local and OpenAI models

## Requirements

- Python 3.8+
- 6GB+ VRAM for local models (or use OpenAI API)
- MongoDB instance with article collection
- Internet connection for model downloads

## License

MIT License - see LICENSE file for details.
