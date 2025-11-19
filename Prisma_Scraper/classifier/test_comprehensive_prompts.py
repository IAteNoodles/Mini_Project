#!/usr/bin/env python3
"""
Test script to verify the expanded comprehensive prompts work correctly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from local_model_summarizer import LocalModelSummarizer

def test_comprehensive_prompts():
    """Test the new comprehensive prompt generation"""
    
    summarizer = LocalModelSummarizer()
    
    # Sample article text for testing
    sample_article = """
    The Federal Reserve announced a 0.25% interest rate cut today, marking the third reduction this year as policymakers respond to slowing economic growth and inflation concerns. Fed Chairman Jerome Powell stated during the press conference that "we are taking measured steps to support economic stability while monitoring global market conditions closely." 

    The decision was unanimous among voting members, with analysts noting this reflects consensus about the need for monetary stimulus. Financial markets responded positively, with the S&P 500 gaining 1.2% in after-hours trading. Bond yields fell to 3.8%, the lowest level since March.

    Economic data released earlier this week showed unemployment remaining steady at 3.9%, while consumer spending increased 0.3% last month. However, manufacturing output declined for the second consecutive quarter, raising concerns about industrial sector weakness.

    Treasury Secretary Janet Yellen praised the Fed's decision, calling it "appropriate given current economic headwinds." She emphasized the administration's commitment to supporting job growth while maintaining price stability.

    Looking ahead, Powell indicated the Fed will remain "data-dependent" in future policy decisions, with the next meeting scheduled for December 18-19. Market participants are now pricing in a 60% probability of another rate cut before year-end.
    """
    
    print("🧪 Testing Comprehensive Prompt Generation")
    print("=" * 60)
    
    # Test local model prompt
    print("\n📋 LOCAL MODEL PROMPT:")
    print("-" * 40)
    local_prompt = summarizer.create_summarization_prompt(sample_article)
    print(local_prompt)
    
    # Test OpenAI prompt structure (simulate)
    print("\n📋 OPENAI PROMPT STRUCTURE:")
    print("-" * 40)
    openai_prompt_preview = f"""Create a comprehensive and detailed summary of this complete news article. Your summary should:

1. Include ALL key information, facts, and important details from the article
2. Capture essential quotes, names, dates, locations, and statistics mentioned
3. Cover the main story, background context, and any developments or implications
4. Maintain chronological order and logical flow of events
5. Include relevant stakeholder perspectives and expert opinions cited
6. Be thorough and informative - use as many sentences as needed to cover all important aspects (typically 6-12 sentences)
7. Write in clear, professional journalism style
8. Focus ONLY on the content - provide just the summary without explanations

Article to summarize:
{sample_article[:200]}...

Comprehensive Summary:"""
    
    print(openai_prompt_preview)
    
    # Test cleaning function with comprehensive content
    print("\n🧹 TESTING CLEANING FUNCTION:")
    print("-" * 40)
    
    test_comprehensive_output = """The Federal Reserve announced a 0.25% interest rate cut today, marking the third reduction this year as policymakers respond to slowing economic growth and inflation concerns. Fed Chairman Jerome Powell stated during the press conference that "we are taking measured steps to support economic stability while monitoring global market conditions closely." The decision was unanimous among voting members, with analysts noting this reflects consensus about the need for monetary stimulus. Financial markets responded positively, with the S&P 500 gaining 1.2% in after-hours trading, while bond yields fell to 3.8%, the lowest level since March. Economic data showed unemployment remaining steady at 3.9% and consumer spending increased 0.3% last month, though manufacturing output declined for the second consecutive quarter. Treasury Secretary Janet Yellen praised the Fed's decision, calling it "appropriate given current economic headwinds." Powell indicated the Fed will remain "data-dependent" in future policy decisions, with the next meeting scheduled for December 18-19, and market participants are now pricing in a 60% probability of another rate cut before year-end."""
    
    cleaned = summarizer.clean_summary(test_comprehensive_output)
    
    print(f"📥 Input length: {len(test_comprehensive_output)} characters")
    print(f"📤 Output length: {len(cleaned)} characters")
    print(f"🔍 Content preserved: {'✅ Yes' if len(cleaned) > len(test_comprehensive_output) * 0.9 else '❌ Too much removed'}")
    print(f"📄 Cleaned summary preview: {cleaned[:300]}...")
    
    # Test quality validation
    print("\n✅ TESTING QUALITY VALIDATION:")
    print("-" * 40)
    
    # Test comprehensive summary validation
    is_valid_local = summarizer.validate_summary_quality(cleaned, min_length=80, is_local_model=True)
    is_valid_standard = summarizer.validate_summary_quality(cleaned, min_length=120, is_local_model=False)
    
    print(f"Local model validation (min 80 chars): {'✅ PASS' if is_valid_local else '❌ FAIL'}")
    print(f"Standard validation (min 120 chars): {'✅ PASS' if is_valid_standard else '❌ FAIL'}")
    
    # Count sentences
    sentences = cleaned.split('.')
    valid_sentences = [s for s in sentences if len(s.strip()) > 15]
    print(f"Sentence count: {len(valid_sentences)} (expecting 4+ for comprehensive)")
    print(f"Sentence structure: {'✅ GOOD' if len(valid_sentences) >= 4 else '⚠️  NEEDS MORE'}")
    
    # Test confidence calculation
    print("\n🎯 TESTING CONFIDENCE CALCULATION:")
    print("-" * 40)
    
    confidence_local = summarizer.calculate_dynamic_confidence(cleaned, sample_article, "local_mistral")
    confidence_openai = summarizer.calculate_dynamic_confidence(cleaned, sample_article, "openai_gpt3.5")
    confidence_regex = summarizer.calculate_dynamic_confidence(cleaned, sample_article, "regex_extractive")
    
    print(f"Local Mistral confidence: {confidence_local:.3f}")
    print(f"OpenAI confidence: {confidence_openai:.3f}")
    print(f"Regex confidence: {confidence_regex:.3f}")
    
    print(f"\n🎉 Comprehensive Prompt Testing Complete!")
    print(f"📊 Summary: Prompts expanded for detailed coverage, quality validation updated for longer summaries")

if __name__ == "__main__":
    test_comprehensive_prompts()
