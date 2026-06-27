#!/usr/bin/env python3
"""
Paper Analysis Tool
Analyzes search results to identify papers with improvement opportunities
"""

import json
import argparse
from typing import List, Dict
import re


class PaperAnalyzer:
    """Analyzes papers to find improvement opportunities"""
    
    # Keywords that indicate future work or improvement opportunities
    FUTURE_WORK_KEYWORDS = [
        'future work', 'future research', 'future direction', 'future study',
        'limitation', 'limited', 'challenge', 'challenging', 'difficult',
        'can be extended', 'can be improved', 'could be improved',
        'further research', 'further investigation', 'further study',
        'open problem', 'open question', 'remains to be', 'yet to be',
        'not yet', 'unexplored', 'under-explored', 'underexplored',
        'potential improvement', 'room for improvement',
        'would be interesting', 'worth investigating', 'worth exploring'
    ]
    
    # Keywords indicating available resources
    RESOURCE_KEYWORDS = [
        'code available', 'code is available', 'publicly available',
        'github', 'open source', 'dataset', 'benchmark',
        'we release', 'we provide', 'we make available'
    ]
    
    # Methodology keywords from APAI requirements
    METHODOLOGY_KEYWORDS = {
        'self-supervised': ['self-supervised', 'self supervised', 'contrastive learning', 
                           'pretext task', 'pre-training', 'pretraining'],
        'knowledge-distillation': ['knowledge distillation', 'teacher-student', 
                                   'model compression', 'distillation'],
        'continual-learning': ['continual learning', 'lifelong learning', 
                              'incremental learning', 'catastrophic forgetting'],
        'meta-learning': ['meta learning', 'meta-learning', 'few-shot', 
                         'learning to learn', 'MAML'],
        'vlm-mllm': ['vision language', 'vision-language', 'multimodal', 
                    'CLIP', 'BLIP', 'visual language', 'VLM', 'MLLM']
    }
    
    def __init__(self, papers_file: str):
        """Load papers from JSON file"""
        with open(papers_file, 'r', encoding='utf-8') as f:
            self.papers = json.load(f)
        print(f"📚 Loaded {len(self.papers)} papers from {papers_file}")
    
    def analyze_future_work_potential(self, paper: Dict) -> Dict:
        """Analyze a paper for future work potential"""
        abstract = paper.get('abstract', '').lower()
        title = paper.get('title', '').lower()
        text = abstract + ' ' + title
        
        # Count future work indicators
        future_work_score = 0
        found_keywords = []
        
        for keyword in self.FUTURE_WORK_KEYWORDS:
            if keyword in text:
                future_work_score += 1
                found_keywords.append(keyword)
        
        # Check for resource availability
        has_resources = any(kw in text for kw in self.RESOURCE_KEYWORDS)
        
        # Identify methodology
        methodologies = []
        for method, keywords in self.METHODOLOGY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                methodologies.append(method)
        
        return {
            'future_work_score': future_work_score,
            'future_work_keywords': found_keywords,
            'has_resources': has_resources,
            'methodologies': methodologies,
            'recency_score': paper.get('year', 0),
            'citation_score': paper.get('citations', 0)
        }
    
    def rank_papers(self) -> List[Dict]:
        """Rank papers by improvement potential"""
        ranked = []
        
        for paper in self.papers:
            analysis = self.analyze_future_work_potential(paper)
            
            # Calculate overall score
            # Weights: future_work (40%), recency (30%), citations (20%), resources (10%)
            overall_score = (
                analysis['future_work_score'] * 4.0 +
                (analysis['recency_score'] - 2020) * 3.0 +  # Normalize year
                min(analysis['citation_score'] / 10, 10) * 2.0 +  # Cap citations impact
                (10 if analysis['has_resources'] else 0) * 1.0
            )
            
            ranked.append({
                'paper': paper,
                'analysis': analysis,
                'overall_score': overall_score
            })
        
        # Sort by overall score
        ranked.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return ranked
    
    def print_top_papers(self, ranked_papers: List[Dict], top_n: int = 10):
        """Print top N papers with analysis"""
        print(f"\n{'='*100}")
        print(f"🏆 TOP {top_n} PAPERS WITH IMPROVEMENT POTENTIAL")
        print(f"{'='*100}\n")
        
        for i, item in enumerate(ranked_papers[:top_n], 1):
            paper = item['paper']
            analysis = item['analysis']
            score = item['overall_score']
            
            print(f"{i}. {paper['title']}")
            print(f"   {'─'*95}")
            print(f"   📊 Overall Score: {score:.1f}")
            print(f"   📅 Year: {paper['year']} | 📚 Citations: {paper['citations']} | 🔗 Source: {paper['source']}")
            print(f"   🎯 Methodologies: {', '.join(analysis['methodologies']) if analysis['methodologies'] else 'Not specified'}")
            print(f"   💡 Future Work Indicators: {analysis['future_work_score']}")
            if analysis['future_work_keywords']:
                print(f"      Keywords found: {', '.join(analysis['future_work_keywords'][:5])}")
            print(f"   📦 Resources Available: {'✅ Yes' if analysis['has_resources'] else '❌ Not mentioned'}")
            print(f"   🔖 DOI: {paper.get('doi', 'Not available')}")
            print(f"   🔗 URL: {paper['url']}")
            print(f"   📝 Abstract Preview: {paper['abstract'][:250]}...")
            print()
    
    def generate_report(self, output_file: str = 'paper_analysis_report.txt'):
        """Generate a detailed analysis report"""
        ranked = self.rank_papers()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("PAPER ANALYSIS REPORT - IMPROVEMENT OPPORTUNITIES\n")
            f.write("="*100 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-"*100 + "\n")
            f.write(f"Total papers analyzed: {len(self.papers)}\n")
            
            # Count by methodology
            methodology_counts = {}
            for item in ranked:
                for method in item['analysis']['methodologies']:
                    methodology_counts[method] = methodology_counts.get(method, 0) + 1
            
            f.write(f"\nPapers by methodology:\n")
            for method, count in sorted(methodology_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  - {method}: {count}\n")
            
            # Papers with resources
            with_resources = sum(1 for item in ranked if item['analysis']['has_resources'])
            f.write(f"\nPapers with available resources: {with_resources}\n")
            
            # Papers with high future work potential
            high_potential = sum(1 for item in ranked if item['analysis']['future_work_score'] >= 3)
            f.write(f"Papers with high future work potential (score ≥ 3): {high_potential}\n")
            
            f.write("\n" + "="*100 + "\n\n")
            
            # Top papers
            f.write("TOP 20 PAPERS WITH IMPROVEMENT POTENTIAL\n")
            f.write("="*100 + "\n\n")
            
            for i, item in enumerate(ranked[:20], 1):
                paper = item['paper']
                analysis = item['analysis']
                score = item['overall_score']
                
                f.write(f"{i}. {paper['title']}\n")
                f.write(f"   {'-'*95}\n")
                f.write(f"   Overall Score: {score:.1f}\n")
                f.write(f"   Year: {paper['year']} | Citations: {paper['citations']} | Source: {paper['source']}\n")
                f.write(f"   Authors: {', '.join(paper['authors'][:5])}\n")
                f.write(f"   Venue: {paper['venue']}\n")
                f.write(f"   Methodologies: {', '.join(analysis['methodologies']) if analysis['methodologies'] else 'Not specified'}\n")
                f.write(f"   Future Work Score: {analysis['future_work_score']}\n")
                if analysis['future_work_keywords']:
                    f.write(f"   Future Work Keywords: {', '.join(analysis['future_work_keywords'])}\n")
                f.write(f"   Resources Available: {'Yes' if analysis['has_resources'] else 'Not mentioned'}\n")
                f.write(f"   DOI: {paper.get('doi', 'Not available')}\n")
                f.write(f"   URL: {paper['url']}\n")
                f.write(f"   Keywords Found: {', '.join(paper['keywords_found'])}\n")
                f.write(f"\n   Abstract:\n   {paper['abstract']}\n")
                f.write("\n" + "="*100 + "\n\n")
        
        print(f"\n📄 Detailed report saved to: {output_file}")
    
    def filter_by_methodology(self, methodology: str) -> List[Dict]:
        """Filter papers by specific methodology"""
        ranked = self.rank_papers()
        filtered = [
            item for item in ranked 
            if methodology.lower() in [m.lower() for m in item['analysis']['methodologies']]
        ]
        return filtered
    
    def print_methodology_summary(self):
        """Print summary of papers by methodology"""
        ranked = self.rank_papers()
        
        print(f"\n{'='*100}")
        print(f"📊 PAPERS BY METHODOLOGY")
        print(f"{'='*100}\n")
        
        for method_key, method_keywords in self.METHODOLOGY_KEYWORDS.items():
            filtered = self.filter_by_methodology(method_key)
            
            if filtered:
                print(f"\n{method_key.upper().replace('-', ' ')}: {len(filtered)} papers")
                print(f"{'-'*100}")
                
                for i, item in enumerate(filtered[:5], 1):
                    paper = item['paper']
                    print(f"  {i}. {paper['title'][:80]}...")
                    print(f"     Year: {paper['year']} | Citations: {paper['citations']} | Score: {item['overall_score']:.1f}")
                
                if len(filtered) > 5:
                    print(f"  ... and {len(filtered) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze papers to identify improvement opportunities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze papers and show top 10
  python analyze_papers.py papers_results.json
  
  # Show top 20 papers
  python analyze_papers.py papers_results.json --top 20
  
  # Generate detailed report
  python analyze_papers.py papers_results.json --report analysis_report.txt
  
  # Show papers by methodology
  python analyze_papers.py papers_results.json --by-methodology
        """
    )
    
    parser.add_argument('papers_file', help='JSON file with paper search results')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top papers to display (default: 10)')
    parser.add_argument('--report', type=str,
                       help='Generate detailed report to specified file')
    parser.add_argument('--by-methodology', action='store_true',
                       help='Show papers grouped by methodology')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = PaperAnalyzer(args.papers_file)
    
    # Rank papers
    ranked = analyzer.rank_papers()
    
    # Print top papers
    analyzer.print_top_papers(ranked, args.top)
    
    # Show by methodology if requested
    if args.by_methodology:
        analyzer.print_methodology_summary()
    
    # Generate report if requested
    if args.report:
        analyzer.generate_report(args.report)
    
    print(f"\n{'='*100}")
    print(f"💡 RECOMMENDATIONS")
    print(f"{'='*100}")
    print(f"1. Focus on papers with high 'Future Work Score' (≥ 3)")
    print(f"2. Prioritize recent papers (2023-2024) for cutting-edge topics")
    print(f"3. Look for papers with available code/datasets")
    print(f"4. Read the full paper, especially:")
    print(f"   - Introduction (for problem motivation)")
    print(f"   - Conclusion (for limitations and future work)")
    print(f"   - Experiments (for datasets and baselines)")
    print(f"5. Verify the methodology complexity matches APAI requirements")
    print(f"6. Discuss your top 3 choices with your teacher")
    print(f"{'='*100}\n")


if __name__ == '__main__':
    main()
