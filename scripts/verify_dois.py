#!/usr/bin/env python3
"""
DOI Verification Tool
Verifies DOIs in paper search results to ensure paper authenticity
"""

import json
import requests
import sys
import argparse
from typing import Tuple


def verify_doi(doi: str) -> Tuple[bool, str]:
    """
    Verify if DOI is valid by checking if it resolves
    
    Args:
        doi: DOI string to verify
        
    Returns:
        Tuple of (is_valid, message)
    """
    if not doi or doi == 'N/A' or doi is None:
        return False, "No DOI available"
    
    url = f"https://doi.org/{doi}"
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        if response.status_code == 200:
            final_url = response.url
            # Extract domain from final URL
            domain = final_url.split('/')[2] if len(final_url.split('/')) > 2 else 'unknown'
            return True, f"Valid - resolves to {domain}"
        else:
            return False, f"Invalid - HTTP status {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout - server not responding"
    except requests.exceptions.ConnectionError:
        return False, "Connection error - check internet"
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description='Verify DOIs in paper search results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify all DOIs in search results
  python verify_dois.py papers_results.json
  
  # Verify and save report
  python verify_dois.py papers_results.json --report doi_verification.txt
  
  # Verify only top N papers
  python verify_dois.py papers_results.json --top 10
        """
    )
    
    parser.add_argument('papers_file', help='JSON file with paper search results')
    parser.add_argument('--top', type=int, help='Only verify top N papers')
    parser.add_argument('--report', type=str, help='Save verification report to file')
    
    args = parser.parse_args()
    
    # Load papers
    try:
        with open(args.papers_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File '{args.papers_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in '{args.papers_file}'")
        sys.exit(1)
    
    # Limit to top N if specified
    if args.top:
        papers = papers[:args.top]
    
    print(f"{'='*100}")
    print(f"🔍 DOI VERIFICATION REPORT")
    print(f"{'='*100}")
    print(f"Verifying DOIs for {len(papers)} papers from {args.papers_file}\n")
    
    # Verify each paper
    results = []
    valid_count = 0
    no_doi_count = 0
    invalid_count = 0
    
    for i, paper in enumerate(papers, 1):
        doi = paper.get('doi')
        title = paper.get('title', 'Unknown')
        source = paper.get('source', 'Unknown')
        year = paper.get('year', 'Unknown')
        
        print(f"{i}. {title[:70]}...")
        print(f"   {'─'*95}")
        
        is_valid, message = verify_doi(doi)
        
        if is_valid:
            status = "✅ VALID"
            valid_count += 1
        elif doi and doi != 'N/A':
            status = "❌ INVALID"
            invalid_count += 1
        else:
            status = "⚠️  NO DOI"
            no_doi_count += 1
        
        print(f"   Status: {status}")
        print(f"   DOI: {doi if doi else 'Not available'}")
        print(f"   Message: {message}")
        print(f"   Source: {source} | Year: {year}")
        
        if doi and doi != 'N/A':
            print(f"   Verify at: https://doi.org/{doi}")
        
        print()
        
        results.append({
            'title': title,
            'doi': doi,
            'is_valid': is_valid,
            'message': message,
            'source': source,
            'year': year
        })
    
    # Print summary
    print(f"{'='*100}")
    print(f"📊 VERIFICATION SUMMARY")
    print(f"{'='*100}")
    print(f"Total papers checked: {len(papers)}")
    print(f"✅ Valid DOIs: {valid_count} ({valid_count/len(papers)*100:.1f}%)")
    print(f"⚠️  No DOI: {no_doi_count} ({no_doi_count/len(papers)*100:.1f}%)")
    print(f"❌ Invalid DOIs: {invalid_count} ({invalid_count/len(papers)*100:.1f}%)")
    print(f"{'='*100}\n")
    
    # Recommendations
    print(f"💡 RECOMMENDATIONS")
    print(f"{'─'*100}")
    
    if valid_count > 0:
        print(f"✅ {valid_count} papers have valid DOIs - these are officially published and safe to use")
    
    if no_doi_count > 0:
        print(f"⚠️  {no_doi_count} papers have no DOI - these might be preprints (arXiv)")
        print(f"   → Check if they have arXiv IDs or PubMed IDs")
        print(f"   → Verify quality before using for your project")
    
    if invalid_count > 0:
        print(f"❌ {invalid_count} papers have invalid DOIs - avoid these!")
        print(f"   → DOI doesn't resolve to a valid page")
        print(f"   → Might be fake or incorrectly recorded")
    
    print(f"{'─'*100}\n")
    
    # Save report if requested
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write("="*100 + "\n")
            f.write("DOI VERIFICATION REPORT\n")
            f.write("="*100 + "\n\n")
            f.write(f"Source file: {args.papers_file}\n")
            f.write(f"Papers checked: {len(papers)}\n")
            f.write(f"Valid DOIs: {valid_count}\n")
            f.write(f"No DOI: {no_doi_count}\n")
            f.write(f"Invalid DOIs: {invalid_count}\n\n")
            f.write("="*100 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {result['title']}\n")
                f.write(f"   {'─'*95}\n")
                f.write(f"   DOI: {result['doi'] if result['doi'] else 'Not available'}\n")
                f.write(f"   Status: {'Valid' if result['is_valid'] else 'Invalid/Missing'}\n")
                f.write(f"   Message: {result['message']}\n")
                f.write(f"   Source: {result['source']} | Year: {result['year']}\n")
                if result['doi'] and result['doi'] != 'N/A':
                    f.write(f"   Verify at: https://doi.org/{result['doi']}\n")
                f.write("\n")
        
        print(f"📄 Detailed report saved to: {args.report}\n")


if __name__ == '__main__':
    main()
