#!/usr/bin/env python3
"""
Enhanced Article Processing Script with Default Batch Processing
Features:
- Process ALL articles by default (unless specific limit specified)
- Skip already processed articles to avoid reprocessing
- Summary generation BEFORE bias classification
- Comprehensive error handling and logging
"""

import sys
import argparse
from local_model_summarizer import LocalModelSummarizer

def main():
    """Main function with enhanced default processing behavior"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process articles with comprehensive summarization')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Number of articles to process (default: ALL articles)')
    parser.add_argument('--no-skip', action='store_true',
                        help='Process all articles including already processed ones')
    parser.add_argument('--stats-only', '-s', action='store_true', 
                        help='Show statistics only, no processing')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced Article Processing System")
    print("🤖 Mistral 7B + Comprehensive Analysis + Bias Classification")
    print("=" * 70)
    
    # Initialize comprehensive summarizer
    summarizer = LocalModelSummarizer()
    
    # Connect to MongoDB
    if not summarizer.connect_to_mongodb():
        print("❌ Failed to connect to MongoDB. Exiting.")
        return
    
    # Get statistics
    stats = summarizer.get_collection_stats()
    if "error" not in stats:
        print(f"\n📊 Collection Statistics:")
        print(f"📰 Original Collection:")
        print(f"   Total articles: {stats['original_collection']['total_articles']}")
        print(f"   Processed: {stats['original_collection']['processed']} ({stats['original_collection']['processed_percentage']})")
        print(f"   Unprocessed: {stats['original_collection']['unprocessed']}")
        print(f"🗃️  Processed Articles Collection:")
        print(f"   Total comprehensive summaries: {stats['processed_articles_collection']['total_processed']}")
    
    if args.stats_only:
        print(f"\n📊 Statistics only requested. Exiting.")
        return
    
    # Configure processing parameters
    skip_processed = not args.no_skip  # Default to skip processed unless --no-skip is used
    
    if args.limit is None:
        # Process ALL articles by default
        total_to_process = stats['original_collection']['unprocessed'] if skip_processed else stats['original_collection']['total_articles']
        print(f"\n🔄 DEFAULT BEHAVIOR: Processing ALL articles ({total_to_process} articles)")
    else:
        total_to_process = args.limit
        print(f"\n🔄 Processing {total_to_process} articles (custom limit)")
    
    print(f"� Processing Configuration:")
    print(f"   • Skip already processed: {'✅ Yes' if skip_processed else '❌ No'}")
    print(f"   • Processing order: Summary → Bias Classification")
    print(f"   • Quality validation: ✅ Enabled")
    
    # Try to load local model
    print(f"\n🔄 Attempting to load local Mistral model...")
    model_success = summarizer.load_local_model()
    
    if not model_success:
        print("\n💡 Local model unavailable. Will use fallback methods:")
        print("   1. OpenAI API (if key available)")
        print("   2. Regex-based extractive summarization")
    
    # Process articles with enhanced parameters
    print(f"\n🚀 Starting comprehensive processing...")
    result = summarizer.process_articles(
        limit=args.limit,  # None means process all
        skip_processed=skip_processed
    )
    
    print(f"\n🎯 Processing Results:")
    print(f"📝 Processed: {result.get('processed', 0)}")
    print(f"✅ Successful: {result.get('successful', 0)}")
    print(f"❌ Failed: {result.get('failed', 0)}")
    print(f"📊 Success Rate: {result.get('success_rate', '0%')}")
    print(f"🤖 Method Used: {result.get('method_used', 'unknown')}")
    
    # Get final statistics
    final_stats = summarizer.get_collection_stats()
    if "error" not in final_stats:
        print(f"\n📊 Final Statistics:")
        print(f"📰 Original Collection:")
        print(f"   Total articles: {final_stats['original_collection']['total_articles']}")
        print(f"   Processed: {final_stats['original_collection']['processed']} ({final_stats['original_collection']['processed_percentage']})")
        print(f"   Unprocessed: {final_stats['original_collection']['unprocessed']}")
        print(f"🗃️  Processed Articles Collection:")
        print(f"   Total comprehensive summaries: {final_stats['processed_articles_collection']['total_processed']}")
    
    print(f"\n🏁 Processing completed!")

if __name__ == "__main__":
    main()
