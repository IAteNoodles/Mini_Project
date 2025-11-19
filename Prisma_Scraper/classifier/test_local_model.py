#!/usr/bin/env python3
"""
Test script to verify the local Mistral model is working correctly
"""

from local_model_summarizer import LocalModelSummarizer

def test_local_model():
    """Test the local model with a sample article"""
    
    print("🧪 Testing Local Mistral Model")
    print("=" * 50)
    
    # Initialize summarizer
    summarizer = LocalModelSummarizer()
    
    # Connect to MongoDB
    if not summarizer.connect_to_mongodb():
        print("❌ Failed to connect to MongoDB. Exiting.")
        return
    
    # Try to load local model
    print(f"\n🔄 Loading local Mistral model...")
    model_success = summarizer.load_local_model()
    
    if not model_success:
        print("❌ Local model failed to load")
        return
    
    print(f"✅ Local model loaded successfully!")
    
    # Test with a sample article
    sample_article = """
    Apple Inc. announced today that its new iPhone 15 will feature significant improvements in battery life and camera quality. 
    The device, set to launch in September 2025, will include a 48-megapixel main camera and support for wireless charging up to 25W. 
    CEO Tim Cook stated that the company expects strong demand for the new device, particularly in international markets.
    Apple's stock price rose 3% following the announcement during the company's quarterly earnings call.
    """
    
    print(f"\n📝 Testing with sample article ({len(sample_article)} characters)")
    print(f"📖 Sample text: {sample_article[:100]}...")
    
    # Test the retry generation method
    summary, method = summarizer.retry_summary_generation(sample_article, "test-url")
    
    print(f"\n🎯 Test Results:")
    print(f"📄 Summary: {summary}")
    print(f"🤖 Method Used: {method}")
    print(f"📏 Summary Length: {len(summary)} characters")
    
    # Test quality validation
    is_quality = summarizer.validate_summary_quality(summary, min_length=50, is_local_model=True)
    print(f"✅ Quality Check: {'PASSED' if is_quality else 'FAILED'}")
    
    # Test bias classification
    bias = summarizer.classify_bias(summary)
    bias_detected = [k for k, v in bias.items() if v == 1]
    print(f"🏷️  Bias Detection: {bias_detected if bias_detected else ['None']}")
    
    print(f"\n🏁 Local model test completed!")

if __name__ == "__main__":
    test_local_model()
