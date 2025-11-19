"""
Main entry point for the Prisma Scraper LLM Framework
"""
import asyncio
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from processor import AnalysisRunner
from database import MongoDBManager

async def main():
    """Main function to run the analysis"""
    
    parser = argparse.ArgumentParser(description="Prisma Scraper LLM Analysis Framework")
    parser.add_argument("--mode", choices=["sample", "full", "report", "test"], 
                       default="sample", help="Analysis mode to run")
    parser.add_argument("--limit", type=int, help="Limit number of articles to process")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for processing")
    parser.add_argument("--local-model", action="store_true", default=True, 
                       help="Use local LLM model (default: True)")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI API instead of local model")
    parser.add_argument("--sample-size", type=int, default=10, help="Sample size for testing")
    
    args = parser.parse_args()
    
    # Determine model type
    use_local_model = not args.openai
    
    print(f"🚀 Starting Prisma Scraper LLM Framework")
    print(f"Mode: {args.mode}")
    print(f"Model: {'Local LLM' if use_local_model else 'OpenAI API'}")
    print("-" * 50)
    
    try:
        if args.mode == "test":
            # Test database connection
            print("🔗 Testing database connection...")
            db_manager = MongoDBManager()
            success = await db_manager.test_connection()
            if success:
                print("✅ Database connection successful!")
                
                # Get collection stats
                stats = await db_manager.get_collection_stats()
                print(f"📊 Collection stats: {stats}")
            else:
                print("❌ Database connection failed!")
                return
                
        elif args.mode == "sample":
            # Run sample analysis
            print(f"🧪 Running sample analysis ({args.sample_size} articles)...")
            result = await AnalysisRunner.run_sample_analysis(
                sample_size=args.sample_size,
                use_local_model=use_local_model
            )
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print("✅ Sample analysis completed!")
                print(f"📈 Processed: {result['processed']} articles")
                print(f"🎯 Success rate: {result['success_rate']}")
                print(f"⏱️  Duration: {result['duration_seconds']:.2f} seconds")
                
                if "sample_results" in result:
                    print("\n📋 Sample Results:")
                    for i, sample in enumerate(result["sample_results"], 1):
                        print(f"  {i}. URL: {sample['url'][:60]}...")
                        print(f"     Summary length: {sample['summary_length']} chars")
                        print(f"     Bias detected: {sample['bias_detected']}")
                        print(f"     Confidence: {sample['confidence']:.2f}")
                        print()
                        
        elif args.mode == "full":
            # Run full analysis
            print(f"🔄 Running full analysis...")
            if args.limit:
                print(f"📝 Limiting to {args.limit} articles")
                
            result = await AnalysisRunner.run_full_analysis(
                use_local_model=use_local_model,
                limit=args.limit,
                batch_size=args.batch_size
            )
            
            print("✅ Full analysis completed!")
            print(f"📈 Total articles: {result['total_articles']}")
            print(f"✅ Processed: {result['processed']}")
            print(f"🎯 Success rate: {result['success_rate']}")
            print(f"⏱️  Duration: {result['duration_seconds']:.2f} seconds")
            print(f"🚀 Speed: {result['articles_per_second']:.2f} articles/second")
            
        elif args.mode == "report":
            # Generate bias analysis report
            print("📊 Generating bias analysis report...")
            result = await AnalysisRunner.run_bias_analysis_report(limit=args.limit)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print("✅ Bias analysis report generated!")
                print(f"📰 Total processed articles: {result['total_processed_articles']}")
                print(f"🎯 Average confidence: {result['average_confidence']:.3f}")
                print(f"💯 High confidence articles: {result['high_confidence_articles']}")
                print(f"⚠️  Low confidence articles: {result['low_confidence_articles']}")
                
                print("\n🏷️  Bias Detection Results:")
                for bias_type, percentage in result['bias_percentages'].items():
                    count = result['bias_counts'][bias_type]
                    print(f"  {bias_type.capitalize()}: {count} articles ({percentage:.1f}%)")
    
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Analysis complete!")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
