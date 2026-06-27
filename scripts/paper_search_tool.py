#!/usr/bin/env python3
"""
Academic Paper Search Tool for APAI Project
Searches for papers based on keywords, year range, and filters by abstract/title content
Supports multiple data sources: arXiv, Semantic Scholar, and PubMed
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import argparse
from dataclasses import dataclass, asdict
import csv


@dataclass
class Paper:
    """Data class to store paper information"""
    title: str
    authors: List[str]
    year: int
    abstract: str
    url: str
    venue: str
    citations: int
    source: str
    keywords_found: List[str]
    doi: Optional[str] = None  # Digital Object Identifier for verification
    
    def to_dict(self):
        return asdict(self)


class PaperSearcher:
    """Main class for searching academic papers"""
    
    def __init__(self, year_start: int, year_end: int, keywords: List[str]):
        self.year_start = year_start
        self.year_end = year_end
        self.keywords = [kw.lower() for kw in keywords]
        self.papers = []
        
    def search_arxiv(self, query: str, max_results: int = 100) -> List[Paper]:
        """
        Search arXiv for papers
        API: http://export.arxiv.org/api/query
        """
        print(f"\n🔍 Searching arXiv for: {query}")
        base_url = "http://export.arxiv.org/api/query"
        
        papers = []
        start = 0
        batch_size = 50  # arXiv limits to 50 per request
        
        while start < max_results:
            params = {
                'search_query': query,
                'start': start,
                'max_results': min(batch_size, max_results - start),
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse XML response
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                
                # Define namespaces
                ns = {
                    'atom': 'http://www.w3.org/2005/Atom',
                    'arxiv': 'http://arxiv.org/schemas/atom'
                }
                
                entries = root.findall('atom:entry', ns)
                if not entries:
                    break
                
                for entry in entries:
                    title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
                    abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
                    
                    # Extract year from published date
                    published = entry.find('atom:published', ns).text
                    year = int(published[:4])
                    
                    # Filter by year
                    if not (self.year_start <= year <= self.year_end):
                        continue
                    
                    # Check if keywords are in title or abstract
                    text_to_search = (title + ' ' + abstract).lower()
                    keywords_found = [kw for kw in self.keywords if kw in text_to_search]
                    
                    if not keywords_found:
                        continue
                    
                    # Extract authors
                    authors = [author.find('atom:name', ns).text 
                              for author in entry.findall('atom:author', ns)]
                    
                    # Get URL
                    url = entry.find('atom:id', ns).text
                    
                    # Extract DOI if available
                    doi = None
                    doi_elem = entry.find('arxiv:doi', ns)
                    if doi_elem is not None:
                        doi = doi_elem.text
                    
                    paper = Paper(
                        title=title,
                        authors=authors,
                        year=year,
                        abstract=abstract,
                        url=url,
                        venue='arXiv',
                        citations=0,  # arXiv doesn't provide citation count
                        source='arXiv',
                        keywords_found=keywords_found,
                        doi=doi
                    )
                    papers.append(paper)
                
                start += batch_size
                time.sleep(1)  # Be nice to the API
                
            except Exception as e:
                print(f"❌ Error searching arXiv: {e}")
                break
        
        print(f"✅ Found {len(papers)} papers from arXiv")
        return papers
    
    def search_semantic_scholar(self, query: str, max_results: int = 100) -> List[Paper]:
        """
        Search Semantic Scholar for papers
        API: https://api.semanticscholar.org/
        """
        print(f"\n🔍 Searching Semantic Scholar for: {query}")
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        papers = []
        offset = 0
        batch_size = 100  # Max per request
        
        while offset < max_results:
            params = {
                'query': query,
                'offset': offset,
                'limit': min(batch_size, max_results - offset),
                'fields': 'title,authors,year,abstract,url,venue,citationCount,publicationDate,externalIds',
                'year': f'{self.year_start}-{self.year_end}'
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'data' not in data or not data['data']:
                    break
                
                for item in data['data']:
                    # Skip if no abstract
                    if not item.get('abstract'):
                        continue
                    
                    title = item.get('title', '')
                    abstract = item.get('abstract', '')
                    
                    # Check if keywords are in title or abstract
                    text_to_search = (title + ' ' + abstract).lower()
                    keywords_found = [kw for kw in self.keywords if kw in text_to_search]
                    
                    if not keywords_found:
                        continue
                    
                    authors = [author.get('name', '') for author in item.get('authors', [])]
                    year = item.get('year', 0)
                    
                    # Extract DOI from externalIds
                    doi = None
                    external_ids = item.get('externalIds', {})
                    if external_ids:
                        doi = external_ids.get('DOI')
                    
                    paper = Paper(
                        title=title,
                        authors=authors,
                        year=year,
                        abstract=abstract,
                        url=item.get('url', ''),
                        venue=item.get('venue', 'Unknown'),
                        citations=item.get('citationCount', 0),
                        source='Semantic Scholar',
                        keywords_found=keywords_found,
                        doi=doi
                    )
                    papers.append(paper)
                
                offset += batch_size
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error searching Semantic Scholar: {e}")
                break
        
        print(f"✅ Found {len(papers)} papers from Semantic Scholar")
        return papers
    
    def search_pubmed(self, query: str, max_results: int = 100) -> List[Paper]:
        """
        Search PubMed for medical/health papers
        API: https://www.ncbi.nlm.nih.gov/books/NBK25501/
        """
        print(f"\n🔍 Searching PubMed for: {query}")
        
        # First, search for paper IDs
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'pub_date',
            'mindate': self.year_start,
            'maxdate': self.year_end
        }
        
        papers = []
        
        try:
            response = requests.get(search_url, params=search_params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            id_list = data.get('esearchresult', {}).get('idlist', [])
            
            if not id_list:
                print("✅ No papers found in PubMed")
                return papers
            
            # Fetch details for each paper
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            
            # Process in batches of 20
            for i in range(0, len(id_list), 20):
                batch_ids = id_list[i:i+20]
                fetch_params = {
                    'db': 'pubmed',
                    'id': ','.join(batch_ids),
                    'retmode': 'xml'
                }
                
                response = requests.get(fetch_url, params=fetch_params, timeout=30)
                response.raise_for_status()
                
                # Parse XML
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                
                for article in root.findall('.//PubmedArticle'):
                    try:
                        # Extract title
                        title_elem = article.find('.//ArticleTitle')
                        title = title_elem.text if title_elem is not None else ''
                        
                        # Extract abstract
                        abstract_texts = article.findall('.//AbstractText')
                        abstract = ' '.join([a.text for a in abstract_texts if a.text])
                        
                        if not abstract:
                            continue
                        
                        # Check keywords
                        text_to_search = (title + ' ' + abstract).lower()
                        keywords_found = [kw for kw in self.keywords if kw in text_to_search]
                        
                        if not keywords_found:
                            continue
                        
                        # Extract year
                        year_elem = article.find('.//PubDate/Year')
                        year = int(year_elem.text) if year_elem is not None else 0
                        
                        # Extract authors
                        author_elems = article.findall('.//Author')
                        authors = []
                        for author in author_elems:
                            lastname = author.find('LastName')
                            forename = author.find('ForeName')
                            if lastname is not None:
                                name = lastname.text
                                if forename is not None:
                                    name = f"{forename.text} {name}"
                                authors.append(name)
                        
                        # Get PMID for URL
                        pmid_elem = article.find('.//PMID')
                        pmid = pmid_elem.text if pmid_elem is not None else ''
                        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ''
                        
                        # Get journal name
                        journal_elem = article.find('.//Journal/Title')
                        venue = journal_elem.text if journal_elem is not None else 'Unknown'
                        
                        # Extract DOI
                        doi = None
                        doi_elems = article.findall('.//ArticleId[@IdType="doi"]')
                        if doi_elems:
                            doi = doi_elems[0].text
                        
                        paper = Paper(
                            title=title,
                            authors=authors,
                            year=year,
                            abstract=abstract,
                            url=url,
                            venue=venue,
                            citations=0,  # PubMed doesn't provide citation count directly
                            source='PubMed',
                            keywords_found=keywords_found,
                            doi=doi
                        )
                        papers.append(paper)
                        
                    except Exception as e:
                        print(f"⚠️  Error parsing article: {e}")
                        continue
                
                time.sleep(0.5)  # Rate limiting
                
        except Exception as e:
            print(f"❌ Error searching PubMed: {e}")
        
        print(f"✅ Found {len(papers)} papers from PubMed")
        return papers
    
    def search_all(self, query: str, max_per_source: int = 100) -> List[Paper]:
        """Search all sources and combine results"""
        all_papers = []
        
        # Search arXiv
        all_papers.extend(self.search_arxiv(query, max_per_source))
        
        # Search Semantic Scholar
        all_papers.extend(self.search_semantic_scholar(query, max_per_source))
        
        # Search PubMed (especially for health-related topics)
        all_papers.extend(self.search_pubmed(query, max_per_source))
        
        # Remove duplicates based on title similarity
        unique_papers = self._remove_duplicates(all_papers)
        
        # Sort by year (newest first) and citations
        unique_papers.sort(key=lambda p: (p.year, p.citations), reverse=True)
        
        self.papers = unique_papers
        return unique_papers
    
    def _remove_duplicates(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers based on title similarity"""
        unique = []
        seen_titles = set()
        
        for paper in papers:
            # Normalize title for comparison
            normalized_title = paper.title.lower().strip()
            
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique.append(paper)
        
        return unique
    
    def save_results(self, filename: str = 'papers_results.json'):
        """Save results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([p.to_dict() for p in self.papers], f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {filename}")
    
    def save_to_csv(self, filename: str = 'papers_results.csv'):
        """Save results to CSV file"""
        if not self.papers:
            print("⚠️  No papers to save")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Title', 'Authors', 'Year', 'Venue', 'Citations', 
                           'Source', 'Keywords Found', 'DOI', 'URL', 'Abstract'])
            
            for paper in self.papers:
                writer.writerow([
                    paper.title,
                    '; '.join(paper.authors),
                    paper.year,
                    paper.venue,
                    paper.citations,
                    paper.source,
                    ', '.join(paper.keywords_found),
                    paper.doi if paper.doi else 'N/A',
                    paper.url,
                    paper.abstract[:500] + '...' if len(paper.abstract) > 500 else paper.abstract
                ])
        
        print(f"💾 Results saved to {filename}")
    
    def print_summary(self):
        """Print a summary of found papers"""
        if not self.papers:
            print("\n❌ No papers found matching your criteria")
            return
        
        print(f"\n{'='*80}")
        print(f"📊 SEARCH SUMMARY")
        print(f"{'='*80}")
        print(f"Total papers found: {len(self.papers)}")
        print(f"Year range: {self.year_start} - {self.year_end}")
        print(f"Keywords: {', '.join(self.keywords)}")
        
        # Group by source
        by_source = {}
        for paper in self.papers:
            by_source[paper.source] = by_source.get(paper.source, 0) + 1
        
        print(f"\nPapers by source:")
        for source, count in by_source.items():
            print(f"  - {source}: {count}")
        
        # Group by year
        by_year = {}
        for paper in self.papers:
            by_year[paper.year] = by_year.get(paper.year, 0) + 1
        
        print(f"\nPapers by year:")
        for year in sorted(by_year.keys(), reverse=True):
            print(f"  - {year}: {by_year[year]}")
        
        print(f"\n{'='*80}")
        print(f"📄 TOP 10 PAPERS (by citations and recency)")
        print(f"{'='*80}\n")
        
        for i, paper in enumerate(self.papers[:10], 1):
            print(f"{i}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
            print(f"   Year: {paper.year} | Venue: {paper.venue} | Citations: {paper.citations}")
            print(f"   Source: {paper.source} | Keywords: {', '.join(paper.keywords_found)}")
            print(f"   DOI: {paper.doi if paper.doi else 'Not available'}")
            print(f"   URL: {paper.url}")
            print(f"   Abstract: {paper.abstract[:200]}...")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='Search for academic papers based on keywords and year range',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search for image segmentation papers in medical domain from 2022-2024
  python paper_search_tool.py --keywords "image segmentation" "medical" "deep learning" --year-start 2022 --year-end 2024
  
  # Search for self-supervised learning papers
  python paper_search_tool.py --keywords "self-supervised learning" "computer vision" --year-start 2023 --year-end 2024 --max-results 50
  
  # Search for continual learning papers in health domain
  python paper_search_tool.py --keywords "continual learning" "medical imaging" --year-start 2022 --year-end 2024
        """
    )
    
    parser.add_argument('--keywords', nargs='+', required=True,
                       help='Keywords to search for in title and abstract')
    parser.add_argument('--year-start', type=int, required=True,
                       help='Start year for paper search')
    parser.add_argument('--year-end', type=int, required=True,
                       help='End year for paper search')
    parser.add_argument('--max-results', type=int, default=100,
                       help='Maximum number of results per source (default: 100)')
    parser.add_argument('--output-json', type=str, default='papers_results.json',
                       help='Output JSON filename (default: papers_results.json)')
    parser.add_argument('--output-csv', type=str, default='papers_results.csv',
                       help='Output CSV filename (default: papers_results.csv)')
    
    args = parser.parse_args()
    
    # Create search query from keywords
    query = ' '.join(args.keywords)
    
    print(f"{'='*80}")
    print(f"🔬 ACADEMIC PAPER SEARCH TOOL")
    print(f"{'='*80}")
    print(f"Query: {query}")
    print(f"Year range: {args.year_start} - {args.year_end}")
    print(f"Max results per source: {args.max_results}")
    print(f"{'='*80}\n")
    
    # Create searcher and run search
    searcher = PaperSearcher(args.year_start, args.year_end, args.keywords)
    papers = searcher.search_all(query, args.max_results)
    
    # Print summary
    searcher.print_summary()
    
    # Save results
    searcher.save_results(args.output_json)
    searcher.save_to_csv(args.output_csv)
    
    print(f"\n✅ Search complete! Found {len(papers)} papers.")
    print(f"📁 Results saved to {args.output_json} and {args.output_csv}")


if __name__ == '__main__':
    main()
