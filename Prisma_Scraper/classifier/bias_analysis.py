#!/usr/bin/env python3
"""
Comprehensive Bias Analysis and Verification Script
Features:
- Analyze articles, summaries, and bias classifications
- Verify regex patterns against both summaries and articles
- Calculate enhanced confidence scores
- Detect false positives and missed detections
"""

import re
import json
from typing import Dict, List, Tuple
from local_model_summarizer import LocalModelSummarizer


class BiasAnalyzer:
    def __init__(self):
        self.summarizer = LocalModelSummarizer()
        
        # Enhanced regex patterns for better detection
        self.bias_patterns = {
            "political": [
                r'\b(conservative|liberal|democrat|republican|left-wing|right-wing|government|policy|politics|political|election|voting|campaign|congress|senate|house|minister|president|prime minister|parliament|legislation|bill|law|candidate|ballot|referendum)\b',
                r'\b(trump|biden|harris|obama|clinton|bush|reagan|pentagon|white house|capitol|supreme court|judicial|executive|legislative|partisan|bipartisan)\b'
            ],
            "gender": [
                r'\b(gender|sex|women|men|woman|man|female|male|masculine|feminine|feminist|sexist|discrimination|equality|harassment|assault|domestic violence|reproductive|maternity|paternity)\b',
                r'\b(she|her|hers|he|him|his|girls|boys|ladies|gentlemen|mother|father|wife|husband|daughter|son|sister|brother|workplace harassment|glass ceiling)\b'
            ],
            "cultural": [
                r'\b(culture|cultural|ethnic|ethnicity|race|racial|religion|religious|faith|belief|tradition|traditional|heritage|community|minority|majority|immigrant|migration|foreigner|native|indigenous)\b',
                r'\b(muslim|islamic|christian|jewish|hindu|buddhist|catholic|protestant|church|mosque|temple|synagogue|prayer|worship|festival|celebration|ceremony|racism|xenophobia)\b'
            ],
            "ideology": [
                r'\b(ideology|ideological|belief|opinion|perspective|viewpoint|agenda|propaganda|biased|partisan|slanted|framing|narrative|spin|activism|movement|radical|extremist|moderate)\b',
                r'\b(capitalism|socialism|communism|fascism|democracy|authoritarian|libertarian|progressive|traditionalist|nationalist|manifesto|doctrine)\b'
            ]
        }
    
    def analyze_text_for_bias(self, text: str) -> Dict[str, any]:
        """Analyze text for bias patterns with detailed matching"""
        text_lower = text.lower()
        results = {}
        
        for bias_type, patterns in self.bias_patterns.items():
            matches = []
            total_matches = 0
            
            for pattern in patterns:
                found_matches = re.findall(pattern, text_lower, re.IGNORECASE)
                matches.extend(found_matches)
                total_matches += len(found_matches)
            
            results[bias_type] = {
                "detected": 1 if total_matches > 0 else 0,
                "match_count": total_matches,
                "matches": list(set(matches))  # Remove duplicates
            }
        
        return results
    
    def calculate_enhanced_confidence(self, summary_analysis: Dict, article_analysis: Dict, 
                                    original_bias: Dict, verification_results: Dict = None) -> float:
        """Calculate enhanced confidence score based on multiple factors"""
        confidence = 0.5  # Base confidence
        
        # Factor 1: Summary-Article consistency
        for bias_type in ["political", "gender", "cultural", "ideology"]:
            summary_detected = summary_analysis[bias_type]["detected"]
            article_detected = article_analysis[bias_type]["detected"]
            original_detected = original_bias.get(bias_type, 0)
            
            # Perfect match: all agree
            if summary_detected == article_detected == original_detected:
                confidence += 0.1
            
            # Good match: summary and article agree
            elif summary_detected == article_detected:
                confidence += 0.05
            
            # Mismatch penalty
            elif summary_detected != article_detected:
                confidence -= 0.05
        
        # Factor 2: Evidence strength in article
        total_article_evidence = sum(analysis["match_count"] for analysis in article_analysis.values())
        if total_article_evidence >= 10:
            confidence += 0.15  # Strong evidence
        elif total_article_evidence >= 5:
            confidence += 0.1   # Moderate evidence
        elif total_article_evidence >= 2:
            confidence += 0.05  # Weak evidence
        
        # Factor 3: Summary quality (evidence vs claims)
        total_summary_evidence = sum(analysis["match_count"] for analysis in summary_analysis.values())
        detected_biases = sum(analysis["detected"] for analysis in summary_analysis.values())
        
        if detected_biases > 0:
            evidence_ratio = total_summary_evidence / detected_biases
            if evidence_ratio >= 3:
                confidence += 0.1   # Good evidence-to-claim ratio
            elif evidence_ratio >= 1:
                confidence += 0.05  # Acceptable ratio
            else:
                confidence -= 0.1   # Insufficient evidence for claims
        
        # Factor 4: Verification results (if available)
        if verification_results:
            for bias_type, verification in verification_results.items():
                if verification.get("detected_in_summary", False):
                    if verification.get("verified_in_article", False):
                        confidence += 0.05  # Verified detection
                    else:
                        confidence -= 0.1   # False positive
        
        return max(0.1, min(1.0, confidence))
    
    def analyze_recent_articles(self, limit: int = 10) -> List[Dict]:
        """Analyze recent processed articles for bias accuracy"""
        if not self.summarizer.connect_to_mongodb():
            return []
        
        # Get recent articles
        recent_articles = list(
            self.summarizer.processed_collection
            .find()
            .sort('processed_at', -1)
            .limit(limit)
        )
        
        results = []
        
        for i, article in enumerate(recent_articles, 1):
            print(f"\n{'='*60}")
            print(f"ANALYZING ARTICLE {i}/{len(recent_articles)}")
            print(f"{'='*60}")
            
            url = article.get('url', 'Unknown')
            summary = article.get('summary', '')
            article_text = article.get('text', '')
            original_bias = article.get('bias', {})
            verification_results = article.get('bias_verification', {})
            original_confidence = article.get('confidence_score', 0)
            
            print(f"URL: {url[:80]}...")
            print(f"Original Confidence: {original_confidence:.2f}")
            
            # Analyze summary for bias
            print(f"\n🔍 ANALYZING SUMMARY:")
            print(f"Summary: {summary[:200]}...")
            summary_analysis = self.analyze_text_for_bias(summary)
            
            # Analyze article for bias
            print(f"\n📰 ANALYZING ARTICLE ({len(article_text)} chars):")
            print(f"Article preview: {article_text[:200]}...")
            article_analysis = self.analyze_text_for_bias(article_text)
            
            # Compare results
            print(f"\n📊 BIAS COMPARISON:")
            print(f"{'Bias Type':<12} {'Original':<8} {'Summary':<8} {'Article':<8} {'Summary Matches':<15} {'Article Matches'}")
            print("-" * 80)
            
            for bias_type in ["political", "gender", "cultural", "ideology"]:
                original_det = original_bias.get(bias_type, 0)
                summary_det = summary_analysis[bias_type]["detected"]
                article_det = article_analysis[bias_type]["detected"]
                summary_matches = summary_analysis[bias_type]["match_count"]
                article_matches = article_analysis[bias_type]["match_count"]
                
                print(f"{bias_type:<12} {original_det:<8} {summary_det:<8} {article_det:<8} {summary_matches:<15} {article_matches}")
                
                # Show actual matches if detected
                if summary_det or article_det:
                    if summary_analysis[bias_type]["matches"]:
                        print(f"  Summary matches: {summary_analysis[bias_type]['matches'][:5]}")
                    if article_analysis[bias_type]["matches"]:
                        print(f"  Article matches: {article_analysis[bias_type]['matches'][:5]}")
            
            # Calculate enhanced confidence
            enhanced_confidence = self.calculate_enhanced_confidence(
                summary_analysis, article_analysis, original_bias, verification_results
            )
            
            print(f"\n🎯 CONFIDENCE ANALYSIS:")
            print(f"Original Confidence: {original_confidence:.2f}")
            print(f"Enhanced Confidence: {enhanced_confidence:.2f}")
            print(f"Confidence Change: {enhanced_confidence - original_confidence:+.2f}")
            
            # Identify issues
            issues = []
            for bias_type in ["political", "gender", "cultural", "ideology"]:
                original_det = original_bias.get(bias_type, 0)
                summary_det = summary_analysis[bias_type]["detected"]
                article_det = article_analysis[bias_type]["detected"]
                
                if original_det and not article_det:
                    issues.append(f"⚠️  {bias_type}: Detected in summary but not supported by article")
                elif not original_det and article_det and summary_det:
                    issues.append(f"❌ {bias_type}: Missed detection - present in both article and summary")
                elif original_det and summary_det and not article_det:
                    issues.append(f"🔍 {bias_type}: Summary bias not supported by article content")
            
            if issues:
                print(f"\n🚨 ISSUES FOUND:")
                for issue in issues:
                    print(f"  {issue}")
            else:
                print(f"\n✅ No bias classification issues found")
            
            # Store results
            result = {
                'url': url,
                'summary': summary,
                'article_length': len(article_text),
                'original_bias': original_bias,
                'summary_analysis': summary_analysis,
                'article_analysis': article_analysis,
                'original_confidence': original_confidence,
                'enhanced_confidence': enhanced_confidence,
                'issues': issues,
                'verification_results': verification_results
            }
            
            results.append(result)
        
        return results
    
    def generate_report(self, results: List[Dict]) -> None:
        """Generate summary report of bias analysis"""
        print(f"\n{'='*80}")
        print(f"BIAS ANALYSIS SUMMARY REPORT")
        print(f"{'='*80}")
        
        total_articles = len(results)
        total_issues = sum(len(result['issues']) for result in results)
        avg_original_confidence = sum(r['original_confidence'] for r in results) / total_articles
        avg_enhanced_confidence = sum(r['enhanced_confidence'] for r in results) / total_articles
        
        print(f"📊 OVERALL STATISTICS:")
        print(f"  Articles Analyzed: {total_articles}")
        print(f"  Total Issues Found: {total_issues}")
        print(f"  Average Original Confidence: {avg_original_confidence:.2f}")
        print(f"  Average Enhanced Confidence: {avg_enhanced_confidence:.2f}")
        print(f"  Confidence Improvement: {avg_enhanced_confidence - avg_original_confidence:+.2f}")
        
        # Issue breakdown
        issue_types = {}
        for result in results:
            for issue in result['issues']:
                issue_type = issue.split(':')[0].strip()
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        if issue_types:
            print(f"\n🚨 ISSUE BREAKDOWN:")
            for issue_type, count in sorted(issue_types.items()):
                print(f"  {issue_type}: {count} occurrences")
        
        # Bias type accuracy
        bias_accuracy = {}
        for bias_type in ["political", "gender", "cultural", "ideology"]:
            correct = 0
            total = 0
            for result in results:
                original = result['original_bias'].get(bias_type, 0)
                article = result['article_analysis'][bias_type]['detected']
                if original == article:
                    correct += 1
                total += 1
            bias_accuracy[bias_type] = (correct / total) * 100
        
        print(f"\n📈 BIAS DETECTION ACCURACY:")
        for bias_type, accuracy in bias_accuracy.items():
            print(f"  {bias_type}: {accuracy:.1f}%")


def main():
    """Main function to run bias analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze bias classification accuracy')
    parser.add_argument('--limit', '-l', type=int, default=10,
                        help='Number of recent articles to analyze (default: 10)')
    parser.add_argument('--detailed', '-d', action='store_true',
                        help='Show detailed match information')
    
    args = parser.parse_args()
    
    print("🔍 Comprehensive Bias Analysis System")
    print("📊 Analyzing Articles, Summaries, and Classifications")
    print("=" * 60)
    
    analyzer = BiasAnalyzer()
    results = analyzer.analyze_recent_articles(args.limit)
    
    if results:
        analyzer.generate_report(results)
        
        # Save detailed results
        with open('bias_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: bias_analysis_results.json")
    else:
        print("❌ No articles found or database connection failed")


if __name__ == "__main__":
    main()
