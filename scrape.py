import requests
import argparse
import os
from bs4 import BeautifulSoup
import time
import random
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import sys
sys.path.append("helpers")
from logger import create_logger

logger = create_logger(__name__)


PRODUCT_BASE_URL = "https://www.kddc.com/product/"
JOBS_API_URL = "https://career.kddc.com/api/career/employers/2930/jobs?pageSize=1000&page=1"


def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def get_categories():
    """Get all product category links from the base page"""
    session = create_session()
    
    try:
        print("Fetching product categories...")
        logger.info(f"Fetching categories from {PRODUCT_BASE_URL}")
        response = session.get(PRODUCT_BASE_URL)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find all links with the specific class
        button_links = soup.find_all("a", class_="elementor-button elementor-button-link elementor-size-sm elementor-animation-grow")
        
        # Extract hrefs and names
        product_categories = []
        for a in button_links:
            if "href" in a.attrs:
                category_name = a.text.strip()
                category_name = ' '.join(category_name.split()[2:])
                product_categories.append({
                    "name": category_name,
                    "url": a["href"]
                })
        
        print(f"Found {len(product_categories)} product categories")
        logger.info(f"Found {len(product_categories)} categories")
        return product_categories
    
    except Exception as e:
        print(f"Error fetching categories: {str(e)}")
        logger.error(f"Error fetching categories: {str(e)}")
        return []


def get_all_subpage_links(category_url):
    """Get all pagination subpage links for a category"""
    session = create_session()
    
    try:
        logger.info(f"Fetching subpages for {category_url}")
        res = session.get(category_url)
        res.raise_for_status()
        
        soup = BeautifulSoup(res.content, "html.parser")
        
        subpage_urls = {category_url}
        
        pagination = soup.find("ul", class_="pagination")
        if pagination:
            for li in pagination.find_all("li"):
                a = li.find("a")
                if a and a.get("href") and "page=" in a["href"]:
                    full_url = a["href"]
                    subpage_urls.add(full_url)
            
            logger.info(f"Found {len(subpage_urls)} subpages")
        else:
            logger.info("No pagination found - only one page")
            
        return list(subpage_urls)
    
    except Exception as e:
        logger.error(f"Error fetching subpages for {category_url}: {str(e)}")
        return [category_url]  # Return at least the main page


def get_all_product_links(category_url, category_name):
    """Get all product links from a category and its subpages"""
    session = create_session()
    print(f"Getting subpages for category: {category_name}")
    subpages = get_all_subpage_links(category_url)
    product_links = []
    
    try:
        for url in tqdm(subpages, desc=f"Scraping subpages in '{category_name}'"):
            logger.info(f"Scraping products from: {url}")
            res = session.get(url)
            res.raise_for_status()
            
            soup = BeautifulSoup(res.content, "html.parser")
            
            products = soup.find_all("h4", class_="protitle")
            for p in products:
                a = p.find("a")
                if a and a.get("href"):
                    product_links.append({
                        "name": a.text.strip(),
                        "url": a["href"],
                        "category": category_name
                    })
            time.sleep(random.uniform(1, 2))
        print(f"Found {len(product_links)} products in category '{category_name}'")
        logger.info(f"Found {len(product_links)} products in category '{category_name}'")
        return product_links
    
    except Exception as e:
        print(f"Error fetching product links for {category_name}: {str(e)}")
        logger.error(f"Error fetching product links for {category_url}: {str(e)}")
        return []


def scrape_product_details(product_info):
    """Scrape information about a product"""
    session = create_session()
    product_url = product_info["url"]
    
    try:
        logger.info(f"Scraping details for: {product_info['name']}")
        res = session.get(product_url)
        res.raise_for_status()
        
        soup = BeautifulSoup(res.content, "html.parser") 
        # description
        desc = ""
        desc_div = soup.find('div', id='tab-description')
    
        # Nutrition table
        nutrition_dict = {}
        nutrition_div = desc_div.find("div", class_="NutritionFact")
        if nutrition_div:
            # Extract nutrition facts
            table = nutrition_div.find("table", class_="NutritionFacts")
            if table:
                rows = table.find_all("tr")[1:]  # Skip header row
                for row in rows:
                    cols = row.find_all("td")
                    if len(cols) == 2:
                        key = cols[0].text.strip()
                        value = cols[1].text.strip()
                        nutrition_dict[key] = value
            if nutrition_dict:
                desc = desc_div.get_text(strip=True).replace(nutrition_div.get_text(strip=True), '').strip()
                # Construct nutrition facts string
                nutrition_text = "\n\nNutrition Facts (per 100 ml):\n"
                for key, value in nutrition_dict.items():
                    nutrition_text += f"- {key}: {value}\n"
                
                # Combine description with nutrition facts
                desc += nutrition_text
            else:
                desc = desc_div.get_text(strip=True).strip()
        else:
            # use original description if no nutrition dev
            desc = desc_div.get_text(strip=True).strip()
    
        # Reviews
        product_id = None
        hidden_input = soup.find("input", {"name": "product_id"})
        if hidden_input:
            product_id = hidden_input.get("value", "").strip()
        review_count = 0
        review_texts = []
        review_anchor = soup.find("a", id="ratecount")
        if review_anchor:
            count_text = review_anchor.text.strip().split()[0]
            if count_text.isdigit():
                review_count = int(count_text)
        if review_count > 0 and product_id:
            review_api = f"https://eshop.kddc.com/index.php?route=product/product/review&product_id={product_id}"
            review_res = session.get(review_api)
            if review_res.ok:
                review_soup = BeautifulSoup(review_res.content, "html.parser")
                review_divs = review_soup.find_all("div", class_="custreview")
                for div in review_divs:
                    p = div.find("p")
                    if p:
                        review_texts.append(p.text.strip())
        if not review_texts:
            review_texts = ["no review on this product"]
        # Price
        price = ""
        price_tag = soup.find("h2", class_="price")
        if price_tag:
            price = price_tag.text.strip()
        
        # Availability
        availability = ""
        stock_tag = soup.find("li", class_="stock_bg")
        if stock_tag:
            availability = stock_tag.text.strip().replace("Availability:", "").strip()
        
        product_data = {
            'product_id': product_id,
            "title": product_info["name"],
            "description": desc,
            "price": price,
            "review count": review_count,
            "reviews": review_texts,
            "availability": availability,
            "url": product_url,
            "category": product_info["category"],
        }
        
        return product_data
    
    except Exception as e:
        logger.error(f"Error scraping product {product_url}: {str(e)}")
        return {
            "title": product_info["name"],
            "url": product_url,
            "category": product_info["category"],
            "error": str(e)
        }


def scrape_with_threading(product_links, max_workers=5):
    """Scrape product details using multi-threading for better performance"""
    results = []
    
    print(f"Starting multi-threaded scraping with {max_workers} workers for {len(product_links)} products")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in tqdm(executor.map(scrape_product_details, product_links), 
                          total=len(product_links), 
                          desc="Scraping product details"):
            results.append(result)
    
    return results


def save_to_csv(items, filename, item_type="products"):
    """Save all scraped data to a CSV file"""
   
    df = pd.DataFrame(items)
    try:
        df.to_csv(filename, index=False)
        print(f"Successfully saved {len(items)} {item_type} to {filename}")
        logger.info(f"Successfully saved {len(items)} {item_type} to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")
        logger.error(f"Error saving to CSV: {str(e)}")


def scrape_products():
    start_time = time.time()
    print("Starting product scraping process...")
    
    categories = get_categories()[:2]
    print(f"Will be scraping only 2 categories {categories[0]['name']} & {categories[1]['name']}")
    if not categories:
        print("No categories found. Exiting product scraping.")
        logger.error("No categories found. Exiting.")
        return []
    
    all_product_links = []
    for category in tqdm(categories, desc="Processing categories"):
        print(f"\nProcessing category: {category['name']}")
        logger.info(f"Processing category: {category['name']}")
        category_products = get_all_product_links(category['url'], category['name'])
        all_product_links.extend(category_products)
    logger.info(f"Total product links found: {len(all_product_links)}")
    print(f"Total product links found: {len(all_product_links)}")
    
    print(f"Starting detailed product information scraping...")
    all_products = scrape_with_threading(all_product_links)
    execution_time = time.time() - start_time
    print(f"Product scraping completed in {execution_time:.2f} seconds")
    print(f"Scraped {len(all_products)} products from {len(categories)} categories")
    logger.info(f"Product scraping completed in {execution_time:.2f} seconds")
    logger.info(f"Scraped {len(all_products)} products from {len(categories)} categories")
    
    return all_products


def scrape_jobs():
    start_time = time.time()
    session = create_session()
    try:
        print("Starting job scraping process...")
        logger.info(f"Fetching jobs from {JOBS_API_URL}")
        response = session.get(JOBS_API_URL)
        response.raise_for_status()
        
        data = response.json()
        jobs = data.get("data", [])
        total_jobs = data.get("total", 0)
        
        print(f"Found {len(jobs)} jobs out of {total_jobs} total jobs")
        logger.info(f"Found {len(jobs)} jobs out of {total_jobs} total jobs")
        
        processed_jobs = []
        
        print(f"Processing job details...")
        for job in tqdm(jobs, desc="Processing job listings"):
            job_entry = {
                "job_id": job.get("id"),
                "title": job.get("title"),
                "job_code": job.get("jobcode"),
                "industry": job.get("industry"),
                "country": job.get("country"),
                "city": job.get("city"),
                "state": job.get("state"),
                "min_experience": job.get("minexp"),
                "max_experience": job.get("maxexp"),
                "job_type": job.get("jobtype"),
                "department": job.get("department"),
                "job_category": job.get("jobcategory"),
                "remote": "Yes" if job.get("is_remote_job") == "1" else "No",
                "zipcode": job.get("zipcode"),
                "date_posted": job.get("dt"),
                "description": clean_html_content(job.get("jobdescription"))
            }
            
            processed_jobs.append(job_entry)
        
        print(f"Successfully processed {len(processed_jobs)} job listings")
        logger.info(f"Successfully processed {len(processed_jobs)} job listings")
        return processed_jobs
    
    except Exception as e:
        print(f"Error scraping jobs: {str(e)}")
        logger.error(f"Error scraping jobs: {str(e)}")
        return []
    finally:
        execution_time = time.time() - start_time
        print(f"Jobs scraping completed in {execution_time:.2f} seconds")
        logger.info(f"Jobs scraping completed in {execution_time:.2f} seconds")


def clean_html_content(html_content):
    if not html_content:
        return ""
    
    soup = BeautifulSoup(html_content, "html.parser")
    # Replace <div> and <p> tags with newlines
    for tag in soup.find_all(['div', 'p']):
        tag.append('\n')
    # Replace <li> tags with bullet points
    for tag in soup.find_all('li'):
        tag.insert_before('• ')
        tag.append('\n')
    # Replace <br> tags with newlines
    for tag in soup.find_all('br'):
        tag.replace_with('\n')
    # Get text and clean up extra spaces/newlines
    text = soup.get_text()
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)
    
    return text


def main():
    parser = argparse.ArgumentParser(description="Index CSV data in Elasticsearch using LangChain")
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True,
        help="Directory to save CSV files after scraping"
    )
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*50)
    print("## KDDC SCRAPER STARTED ##")
    print("="*50 + "\n")
    
    logger.info("Starting KDDC product scraper")
    products = scrape_products()
    if products:
        save_to_csv(products, f'{args.output_dir}/kddc_products.csv', "products")
    
    print("\n" + "="*50)
    logger.info("Starting KDDC job scraper")
    jobs = scrape_jobs()
    if jobs:
        save_to_csv(jobs, f'{args.output_dir}/kddc_jobs.csv', "jobs")
    
    print("\n" + "="*50)
    print("SCRAPING COMPLETED SUCCESSFULLY")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()