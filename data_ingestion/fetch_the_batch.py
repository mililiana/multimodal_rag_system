import requests
from bs4 import BeautifulSoup
import json
import os
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import time
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TheBatchScraper:
    # I created this Scraper for The Batch newsletter from DeepLearning.AI 
    # to extracts articles, text content, and associated images
    
    def __init__(self, delay: float = 1.0):
        self.base_url = "https://www.deeplearning.ai/the-batch/"
        self.delay = delay  
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_batch_issues(self, max_issues: int = 10) -> List[Dict[str, Any]]:
        # let`s fetch The Batch newsletter issues
        issues = []
        
        try:
            logger.info("Fetching The Batch main page...")
            response = self.session.get(self.base_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for issue links - I discovered The Batch typically has issue numbers or dates
            issue_links = self._extract_issue_links(soup)
            
            logger.info(f"Found {len(issue_links)} potential issue links")
            
            for i, link in enumerate(issue_links[:max_issues]):
                if self.delay > 0:
                    time.sleep(self.delay)
                
                try:
                    logger.info(f"Processing issue {i+1}/{min(len(issue_links), max_issues)}: {link}")
                    issue_data = self._scrape_issue(link)
                    if issue_data:
                        issues.append(issue_data)
                        logger.info(f"Successfully scraped issue: {issue_data['title'][:50]}...")
                
                except Exception as e:
                    logger.error(f"Failed to scrape issue {link}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.error(f"Failed to fetch main page: {str(e)}")
        
        return issues
    
    def _extract_issue_links(self, soup: BeautifulSoup) -> List[str]: # Extract links to individual Batch issues
       
        links = []
        
        # I will try different selectors that might contain issue links
        selectors = [
            'a[href*="batch"]',
            'a[href*="issue"]',
            'a[href*="newsletter"]',
            '.post-title a',
            '.entry-title a',
            'h2 a',
            'h3 a'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            for elem in elements:
                href = elem.get('href')
                if href:
                    full_url = urljoin(self.base_url, href)
                    if self._is_valid_issue_url(full_url) and full_url not in links:
                        links.append(full_url)
        
        # Also, I look for any links that might be issue-related
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link.get('href', '')
            text = link.get_text().strip().lower()
            
            # Check if link text or href suggests it's an issue
            if any(keyword in text for keyword in ['issue', 'batch', 'newsletter']) or \
               any(keyword in href.lower() for keyword in ['batch', 'issue', 'newsletter']):
                full_url = urljoin(self.base_url, href)
                if self._is_valid_issue_url(full_url) and full_url not in links:
                    links.append(full_url)
        
        return links[:20]  # I limited to the first 20 potential links
    
    def _is_valid_issue_url(self, url: str) -> bool:  # to check if URL looks like a valid Batch issue
    
        parsed = urlparse(url)
        
        # Must be from the same domain
        if 'deeplearning.ai' not in parsed.netloc.lower():
            return False
        
        # Should contain relevant keywords or patterns
        path_lower = parsed.path.lower()
        return any(keyword in path_lower for keyword in ['batch', 'issue', 'newsletter'])
    
    def _scrape_issue(self, url: str) -> Optional[Dict[str, Any]]:
        # let`s scrape a single Batch issue
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract main content
            content = self._extract_content(soup)
            
            # Extract articles/sections
            articles = self._extract_articles(soup)
            
            # Extract images
            images = self._extract_images(soup, url)
            
            # Extract metadata
            published_date = self._extract_date(soup)
            
            return {
                'id': self._generate_id(url),
                'title': title,
                'articles': articles,
                'images': images,
                'url': url,
                'published_date': published_date,
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(content.split()) if content else 0
            }
        
        except Exception as e:
            logger.error(f"Error scraping issue {url}: {str(e)}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str: # here I  try to extract the title of the issue
        # and try different title selectors
        selectors = ['h1', '.entry-title', '.post-title', 'title', '.page-title']
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text().strip()
                if title and len(title) > 5:  # Reasonable title length
                    return title
        
        return "The Batch - Issue"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        # Extract main content from the page
        # don`t forget to remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # here I try to find main content area
        content_selectors = [
            '.entry-content',
            '.post-content',
            '.content',
            'main',
            'article',
            '.newsletter-content'
        ]
        
        content_text = ""
        
        for selector in content_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                # Extract text from paragraphs and headings
                elements = content_div.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])
                texts = [elem.get_text().strip() for elem in elements if elem.get_text().strip()]
                content_text = ' '.join(texts)
                break
        
        # If no specific content area found, extract from body
        if not content_text:
            body = soup.find('body')
            if body:
                paragraphs = body.find_all('p')
                texts = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
                content_text = ' '.join(texts)
        
        # Clean up the text
        content_text = re.sub(r'\s+', ' ', content_text)
        return content_text.strip()
    
    def _extract_articles(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        # Extract individual articles or sections
        articles = []
        
        # Look for article sections
        article_selectors = [
            'article',
            '.article',
            '.story',
            '.news-item',
            '.section',
            'section'
        ]
        
        for selector in article_selectors:
            elements = soup.select(selector)
            for elem in elements:
                title_elem = elem.find(['h1', 'h2', 'h3', 'h4'])
                title = title_elem.get_text().strip() if title_elem else ""
                
                # Get content (excluding the title)
                if title_elem:
                    title_elem.decompose()
                
                content = elem.get_text().strip()
                content = re.sub(r'\s+', ' ', content)
                
                if len(content) > 50:  # Only include substantial content
                    articles.append({
                        'title': title,
                        'content': content
                    })
        
        # If no articles found, let`s try to split by headings
        if not articles:
            headings = soup.find_all(['h2', 'h3'])
            for heading in headings:
                title = heading.get_text().strip()
                content_parts = []
                
                # Get content after heading until next heading
                next_elem = heading.next_sibling
                while next_elem and next_elem.name not in ['h1', 'h2', 'h3']:
                    if hasattr(next_elem, 'get_text'):
                        text = next_elem.get_text().strip()
                        if text:
                            content_parts.append(text)
                    next_elem = next_elem.next_sibling
                
                content = ' '.join(content_parts)
                if len(content) > 50:
                    articles.append({
                        'title': title,
                        'content': content
                    })
        
        return articles[:10]  # Limit to 10 articles per issue
    

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]: # to extract and download images
        images = []
        img_tags = soup.find_all('img')
        
        for img in img_tags[:5]:  # Limit to 5 images per issue
            src = img.get('src') or img.get('data-src')
            if not src:
                continue
            
            # Make URL absolute
            img_url = urljoin(base_url, src)
            
            # Skip very small images (likely icons)
            width = img.get('width')
            height = img.get('height')
            if width and height:
                try:
                    if int(width) < 50 or int(height) < 50:
                        continue
                except ValueError:
                    pass
            
            try:
                img_data = self._download_image(img_url)
                if img_data:
                    images.append({
                        'url': img_url,
                        'alt': img.get('alt', ''),
                        'data': img_data,
                        'caption': self._extract_image_caption(img)
                    })
            except Exception as e:
                logger.warning(f"Failed to download image {img_url}: {str(e)}")
                continue
        
        return images
    import base64



    
    def _download_image(self, url: str) -> Optional[str]: # to download and encode image as base64
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Open image and resize if needed
            img = Image.open(BytesIO(response.content))
            
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Resize if too large
            if img.width > 800 or img.height > 600:
                img.thumbnail((800, 600), Image.Resampling.LANCZOS)
            
            # Save as JPEG
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            
            # Encode as base64
            img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_data
        
        except Exception as e:
            logger.warning(f"Failed to process image {url}: {str(e)}")
            return None
    
    def _extract_image_caption(self, img_tag) -> str: # to extract caption for an image
        # Look for caption in nearby elements
        parent = img_tag.parent
        if parent:
            caption_elem = parent.find(['figcaption', '.caption', '.image-caption'])
            if caption_elem:
                return caption_elem.get_text().strip()
        
        return img_tag.get('alt', '')
    
    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:  # to extract publication date
        date_selectors = [
            'time[datetime]',
            '.published',
            '.date',
            '.post-date',
            '[datetime]'
        ]
        
        for selector in date_selectors:
            elem = soup.select_one(selector)
            if elem:
                datetime_attr = elem.get('datetime')
                if datetime_attr:
                    return datetime_attr
                
                text = elem.get_text().strip()
                if text:
                    return text
        
        return None
    
    def _generate_id(self, url: str) -> str: # to generate unique ID for the issue
        return f"batch_{abs(hash(url)) % 10000}"
    
    
    def save_data(self, issues: List[Dict[str, Any]], filename: str = 'batch_issues.json'): # to save scraped data to JSON file
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(issues, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(issues)} issues to {filepath}")
        return filepath

def main():
    scraper = TheBatchScraper(delay=1.0)  # 1 second delay between requests
    
    logger.info("Starting The Batch scraping...")
    issues = scraper.fetch_batch_issues(max_issues=5)
    
    if issues:
        filepath = scraper.save_data(issues)
        logger.info(f"Successfully scraped and saved {len(issues)} issues")
        
        for issue in issues:
            print(f"\nTitle: {issue['title']}")
            print(f"Articles: {len(issue['articles'])}")
            print(f"Images: {len(issue['images'])}")
            print(f"Word count: {issue['word_count']}")
    else:
        logger.error("No issues were scraped")

if __name__ == "__main__":
    main()