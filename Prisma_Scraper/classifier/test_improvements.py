#!/usr/bin/env python3
"""
Test script to verify the improved summary cleaning and dynamic confidence scoring
"""

from local_model_summarizer import LocalModelSummarizer

def test_improvements():
    """Test the summary cleaning and dynamic confidence features"""
    
    print("🧪 Testing Summary Cleaning and Dynamic Confidence")
    print("=" * 60)
    
    # Initialize summarizer
    summarizer = LocalModelSummarizer()
    
    # Test summary cleaning
    print("\n🧹 Testing Summary Cleaning:")
    
    test_summaries = [
        "This is a good summary about [INSERT] technology [/INSERT] and innovation.",
        "### ShowHide Instructions [SHOW] Apple announced new products [/SHOW] today.",
        "SUMMARY: The company [ANSWER] reported strong earnings [/ANSWER] this quarter.",
        "Here is a [comprehensive] summary of the [article] content.",
        "The [politician] made [statements] about policy changes."
    ]
    
    for i, test_summary in enumerate(test_summaries, 1):
        cleaned = summarizer.clean_summary(test_summary)
        print(f"   Test {i}:")
        print(f"     Original: {test_summary}")
        print(f"     Cleaned:  {cleaned}")
        print()
    
    # Test dynamic confidence scoring
    print("\n📊 Testing Dynamic Confidence Scoring:")
    
    test_cases = [
        {
            "summary": "Apple announced new iPhone 15 with improved camera and battery life. The device launches in September 2025 with 48MP camera. CEO Tim Cook expects strong demand. Stock price rose 3%.",
            "article": "Apple Inc. announced today significant improvements...",
            "method": "local_mistral"
        },
        {
            "summary": "Short summary.",
            "article": "This is a test article...",
            "method": "openai_gpt3.5"
        },
        {
            "summary": "Error generating summary failed",
            "article": "Test article content...",
            "method": "regex_extractive"
        },
        {
            "summary": "Very comprehensive and detailed summary that covers all aspects of the news article including specific names like John Smith and Jane Doe, important dates like January 15, 2025, significant numbers like 25% increase and $1.2 billion revenue, providing excellent coverage of the Who, What, When, Where, Why and How of the story with proper context and background information.",
            "article": "Long detailed article...",
            "method": "local_mistral"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        confidence = summarizer.calculate_dynamic_confidence(
            case["summary"], 
            case["article"], 
            case["method"]
        )
        print(f"   Test Case {i}: {case['method']}")
        print(f"     Summary: {case['summary'][:80]}...")
        print(f"     Dynamic Confidence: {confidence:.3f}")
        print()
    
    print("\n✅ All tests completed!")
    print("\nKey Improvements:")
    print("  🧹 Removes [anything in brackets] from summaries")
    print("  📊 Dynamic confidence based on content quality")
    print("  🎯 No more static confidence scores")
    print("  🤖 Optional model self-evaluation")

if __name__ == "__main__":
    test_improvements()
