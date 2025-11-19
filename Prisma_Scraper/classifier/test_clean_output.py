#!/usr/bin/env python3
"""
Test script to verify the improved cleaning function removes prompt artifacts
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from local_model_summarizer import ComprehensiveArticleSummarizer

def test_cleaning():
    """Test the cleaning function with problematic examples"""
    
    summarizer = ComprehensiveArticleSummarizer()
    
    # Test cases with problematic outputs
    test_cases = [
        {
            "name": "Prompt artifact example 1",
            "input": "📄 Summary: Answer & Explanation Summarize this news article clearly in 3-4 sentences. Survivors of sexual violence at universities struggle with reporting due to institutional barriers and social stigma.",
            "expected_clean": "Survivors of sexual violence at universities struggle with reporting due to institutional barriers and social stigma."
        },
        {
            "name": "Prompt artifact example 2", 
            "input": "Answer & Explanation Solution: Survivors of sexual violence at universities struggle with reporting incidents due to institutional barriers and social stigma. Many victims face re-traumatization through inadequate support systems. Universities are implementing new policies to address these challenges.",
            "expected_clean": "Survivors of sexual violence at universities struggle with reporting incidents due to institutional barriers and social stigma. Many victims face re-traumatization through inadequate support systems. Universities are implementing new policies to address these challenges."
        },
        {
            "name": "Bracket removal test",
            "input": "The [comprehensive] analysis shows that [INSERT] climate change is accelerating. Scientists warn about [urgent] action needed.",
            "expected_clean": "The analysis shows that climate change is accelerating. Scientists warn about action needed."
        },
        {
            "name": "Instruction removal test",
            "input": "Summarize this news article in 3-4 clear sentences. The economy is showing signs of recovery. Stock markets are rising. Employment rates are improving.",
            "expected_clean": "The economy is showing signs of recovery. Stock markets are rising. Employment rates are improving."
        }
    ]
    
    print("🧪 Testing Enhanced Cleaning Function")
    print("=" * 50)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test['name']}")
        print(f"📥 Input:  {test['input']}")
        
        cleaned = summarizer.clean_summary(test['input'])
        
        print(f"📤 Output: {cleaned}")
        print(f"✅ Expected: {test['expected_clean']}")
        
        # Check if cleaning was successful
        is_clean = cleaned == test['expected_clean']
        print(f"🎯 Result: {'✅ PASS' if is_clean else '❌ NEEDS IMPROVEMENT'}")
        
        if not is_clean:
            print(f"🔍 Analysis:")
            print(f"   Length difference: {len(cleaned)} vs {len(test['expected_clean'])}")
            if len(cleaned) != len(test['expected_clean']):
                print(f"   Extra characters: {repr(cleaned[len(test['expected_clean']):]) if len(cleaned) > len(test['expected_clean']) else 'None'}")

if __name__ == "__main__":
    test_cleaning()
