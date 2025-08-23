# deep_research_agent.py
from .base_agent import BaseAgent
from typing import Dict, Any, Optional, AsyncGenerator
import time
import asyncio
import os
import re
from datetime import datetime
from dateutil import parser
from ddgs import DDGS
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

class DeepResearchAgent(BaseAgent):
    """Agent specialized in deep research with web data"""
    
    def __init__(self, llm_client):
        super().__init__(llm_client)
        self.search_client = DDGS()
        self.specializations = [
            "web research",
            "comprehensive analysis",
            "data synthesis",
            "trend analysis",
            "market research",
            "financial insights",
            "pdf_report_generation",
            "yahoo_finance_metrics",
            "html_dashboard_generation",
            "financial_charts",
            "data_visualization"
        ]
        
        # Create reports directory if it doesn't exist
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def validate_and_clean_urls(self, sources: list) -> list:
        """Validate and clean URLs in sources to ensure they are properly formatted"""
        cleaned_sources = []
        
        for i, source in enumerate(sources):
            # Look for URL patterns in the source
            url_pattern = r'URL: (https?://[^\s\n]+)'
            url_matches = re.findall(url_pattern, source)
            
            if url_matches:
                # Clean and validate each URL
                cleaned_urls = []
                for url in url_matches:
                    # Remove any trailing punctuation or whitespace
                    cleaned_url = url.strip().rstrip('.,;:!?')
                    # Ensure URL starts with http:// or https://
                    if not cleaned_url.startswith(('http://', 'https://')):
                        cleaned_url = 'https://' + cleaned_url
                    cleaned_urls.append(cleaned_url)
                
                # Replace the original URLs with cleaned ones
                cleaned_source = source
                for j, url in enumerate(url_matches):
                    cleaned_source = cleaned_source.replace(f'URL: {url}', f'URL: {cleaned_urls[j]}')
                
                cleaned_sources.append(cleaned_source)
                print(f"🔗 [{self.agent_name}] Source {i+1} URLs cleaned: {url_matches} -> {cleaned_urls}")
            else:
                cleaned_sources.append(source)
                print(f"⚠️ [{self.agent_name}] Source {i+1} has no URLs")
        
        return cleaned_sources

    def rank_sources(self, query: str, sources: list) -> list:
        """Rank sources by relevance, credibility, and freshness"""
        ranked_sources = []
        print(f"🏆 [{self.agent_name}] Ranking {len(sources)} sources for query: '{query}'")
        
        for i, source in enumerate(sources):
            score = 0
            content = source.lower()
            keywords = query.lower().split()
            
            # 1. Relevance: count query keywords in content
            relevance = sum(1 for word in keywords if word in content)
            relevance_score = relevance * 5  # weight = 5
            score += relevance_score
            
            # 2. Credibility: high score for trusted domains and URLs
            trusted_domains = [
                'yahoo.com', 'bloomberg.com', 'reuters.com', 'nasdaq.com', 'investing.com',
                'marketwatch.com', 'cnbc.com', 'wsj.com', 'ft.com', 'economist.com',
                'seekingalpha.com', 'morningstar.com', 'fool.com', 'barrons.com',
                'financialtimes.com', 'businessinsider.com', 'forbes.com', 'fortune.com'
            ]
            credibility_score = 0
            
            # Check for trusted domains
            if any(domain in source for domain in trusted_domains):
                credibility_score += 10
            
            # Bonus points for having a URL (source verification)
            if 'URL:' in source:
                credibility_score += 5
                print(f"🔗 [{self.agent_name}] Source {i+1} has URL - credibility bonus +5")
            
            score += credibility_score
            
            # 3. Freshness: more recent articles score higher
            freshness_score = 0
            # Look for date patterns in the source text (e.g., 'Aug 21, 2025', '2025-08-21')
            date_patterns = [
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b',  # Aug 21, 2025
                r'\b\d{4}-\d{2}-\d{2}\b'  # 2025-08-21
            ]
            found_dates = []
            for pattern in date_patterns:
                matches = re.findall(pattern, source)
                for m in matches:
                    try:
                        dt = parser.parse(m)
                        found_dates.append(dt)
                    except:
                        continue
            
            if found_dates:
                latest_date = max(found_dates)
                days_diff = (datetime.now() - latest_date).days
                # Freshness scoring: max 10 points if today, decreases over 30 days
                freshness_score = max(0, 10 - (days_diff / 3))  # 0 points if older than ~30 days
                score += freshness_score
            
            # Log scoring details for transparency
            source_preview = source[:100] + "..." if len(source) > 100 else source
            has_url = "🔗 YES" if 'URL:' in source else "❌ NO"
            print(f"📊 [{self.agent_name}] Source {i+1} scoring: Relevance={relevance_score}, Credibility={credibility_score}, Freshness={freshness_score:.1f}, Total={score}, URL: {has_url}")
            print(f"📄 [{self.agent_name}] Source {i+1} preview: {source_preview}")
            
            ranked_sources.append((score, source))
        
        # Sort sources by score descending
        ranked_sources.sort(reverse=True, key=lambda x: x[0])
        
        # Log final ranking
        print(f"🏆 [{self.agent_name}] Final ranking (top 5):")
        for i, (score, source) in enumerate(ranked_sources[:5]):
            source_preview = source[:80] + "..." if len(source) > 80 else source
            print(f"   {i+1}. Score {score}: {source_preview}")
        
        return [s[1] for s in ranked_sources]

    def _safe_json_serialize(self, obj):
        """Convert pandas/numpy objects to JSON-serializable types"""
        import pandas as pd
        import numpy as np
        
        try:
            if isinstance(obj, pd.Timestamp):
                return obj.strftime('%Y-%m-%d')
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(key): self._safe_json_serialize(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [self._safe_json_serialize(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # For any other types, convert to string
                return str(obj)
        except Exception as e:
            print(f"⚠️ [{self.agent_name}] Error in _safe_json_serialize: {str(e)} for object type {type(obj)}")
            return str(obj) if obj is not None else None

    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process deep research queries with web data and Yahoo Finance metrics"""
        start_time = self._log_start("deep research", query)
        
        # Clear cache and ensure fresh data
        self._clear_cache_and_refresh()
        
        # Extract parameters
        generate_report = kwargs.get('generate_report', False)
        generate_dashboard = kwargs.get('generate_dashboard', False)
        
        print(f"🔍 [{self.agent_name}] Parameters - generate_report: {generate_report}, generate_dashboard: {generate_dashboard}")
        
        # Perform web search and Yahoo Finance fetch sequentially
        print(f"🚀 [{self.agent_name}] Starting sequential tasks...")
        print(f"🔍 [{self.agent_name}] Starting web search...")
        web_data = self._perform_web_search(query)
        print(f"🔍 [{self.agent_name}] Starting Yahoo Finance fetch...")
        yahoo_data = self._fetch_yahoo_finance_metrics(query)
        print(f"✅ [{self.agent_name}] Both tasks completed successfully")
        print(f"📊 [{self.agent_name}] Web data: {len(web_data)} chars")
        print(f"📈 [{self.agent_name}] Yahoo data type: {type(yahoo_data)}")
        if isinstance(yahoo_data, dict):
            print(f"📈 [{self.agent_name}] Yahoo data keys: {list(yahoo_data.keys())}")
        
        # Generate comprehensive response
        response = self._generate_research_response(query, web_data)
        
        # Generate reports if requested
        pdf_path = None
        dashboard_path = None
        
        if generate_report:
            pdf_path = self._generate_pdf_report(query, response, web_data)
            print(f"📄 [{self.agent_name}] PDF report generated: {pdf_path}")
        
        if generate_dashboard:
            print(f"🌐 [{self.agent_name}] Generating HTML dashboard...")
            dashboard_path = self._generate_html_dashboard(query, response, web_data, yahoo_data)
            print(f"🌐 [{self.agent_name}] Dashboard generated: {dashboard_path}")
        
        # Log completion
        total_time = time.time() - start_time
        self._log_completion("deep research", total_time)
        
        # Ensure all data is JSON-serializable
        safe_yahoo_data = self._safe_json_serialize(yahoo_data)
        
        # Create the response dictionary
        response_dict = {
            "response": response,
            "agent": "deep_research",
            "web_data": web_data,
            "specializations": self.specializations,
            "processing_time": total_time,
            "pdf_report_path": pdf_path,
            "dashboard_path": dashboard_path,
            "report_generated": generate_report,
            "dashboard_generated": generate_dashboard
        }
        
        # Ensure the entire response is JSON-serializable
        safe_response = self._safe_json_serialize(response_dict)
        
        # Test JSON serialization to catch any remaining issues
        try:
            import json
            json.dumps(safe_response)
            print("✅ [DeepResearchAgent] Response is JSON-serializable")
        except Exception as e:
            print(f"❌ [DeepResearchAgent] JSON serialization test failed: {str(e)}")
            # Fallback: convert everything to strings
            safe_response = {str(k): str(v) if v is not None else None for k, v in safe_response.items()}
        
        return safe_response
    
    def _perform_web_search(self, query: str) -> str:
        """Perform web search using DuckDuckGo"""
        print(f"🌐 [{self.agent_name}] Searching web for: '{query}'")
        start_time = time.time()
        
        try:
            web_data = self._duckduckgo_search(query)
            search_time = time.time() - start_time
            print(f"📊 [{self.agent_name}] Web search completed in {search_time:.2f}s")
            print(f"📄 [{self.agent_name}] Web data length: {len(web_data)} characters")
            return web_data
        except Exception as e:
            print(f"❌ [{self.agent_name}] Web search failed: {str(e)}")
            return f"Search error: {str(e)}"
    
    def _fetch_yahoo_finance_metrics(self, query: str) -> Dict[str, Any]:
        """Fetch essential financial metrics from Yahoo Finance"""
        print(f"📈 [{self.agent_name}] Fetching Yahoo Finance metrics for: '{query}'")
        start_time = time.time()
        
        try:
            # Extract tickers from query
            tickers = self._extract_tickers_from_query(query)
            print(f"🔍 [{self.agent_name}] Tickers found: {tickers}")
            
            if not tickers:
                return {"error": "No ticker symbols found in query", "data": {}}
            
            # Fetch data for all tickers
            results = {}
            for ticker in tickers:
                print(f"🔍 [{self.agent_name}] Fetching data for {ticker}...")
                result = self._fetch_ticker_data(ticker)
                results[ticker] = result
            
            fetch_time = time.time() - start_time
            print(f"📊 [{self.agent_name}] Yahoo Finance fetch completed in {fetch_time:.2f}s")
            
            return {
                "tickers": tickers,
                "data": results,
                "fetch_time": fetch_time
            }
            
        except Exception as e:
            print(f"❌ [{self.agent_name}] Yahoo Finance fetch failed: {str(e)}")
            return {"error": str(e), "data": {}}
    
    def _extract_tickers_from_query(self, query: str) -> list:
        """Extract ticker symbols from query text using LLM"""
        print(f"🔍 [{self.agent_name}] Extracting tickers from query: '{query}'")
        
        try:
            # Use LLM to generate Yahoo Finance query
            llm_prompt = f"""
You are a financial data expert. Given the user's research query, identify the most relevant ticker symbols for Yahoo Finance.

User Query: "{query}"

Your task:
1. Identify specific company tickers (e.g., AAPL, GOOGL, MSFT)
2. Identify market indices (e.g., ^GSPC, ^NSEI, ^DJI)
3. Return ONLY the ticker symbols, separated by commas
4. Limit to maximum 5 tickers
5. If no specific tickers found, return empty list

Examples:
- "Apple stock performance" → AAPL
- "S&P 500 and tech stocks" → ^GSPC, AAPL, GOOGL, MSFT
- "Indian market analysis" → ^NSEI, ^BSESN
- "Tesla and electric vehicles" → TSLA

Return only the ticker symbols, nothing else:"""

            print(f"🤖 [{self.agent_name}] Calling LLM to extract tickers...")
            llm_response = self.llm_client(llm_prompt)
            print(f"🤖 [{self.agent_name}] LLM response: {llm_response}")
            
            # Parse LLM response to extract tickers
            if llm_response and isinstance(llm_response, str):
                # Clean the response and extract tickers
                response_clean = llm_response.strip().upper()
                
                # Look for patterns like ^NSEI, ^GSPC, etc. (market indices)
                index_pattern = r'\^[A-Z]+'
                indices = re.findall(index_pattern, response_clean)
                
                # Look for company tickers (1-5 uppercase letters)
                ticker_pattern = r'\b[A-Z]{1,5}\b'
                tickers = re.findall(ticker_pattern, response_clean)
                
                # Filter out common words and combine
                common_words = {'THE', 'AND', 'OR', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT', 'WILL', 'CAN', 'ARE', 'WAS', 'HAS', 'HAD', 'NOT', 'BUT', 'YOU', 'ALL', 'ANY', 'HER', 'HIS', 'ITS', 'OUR', 'THEY', 'THEM', 'WHAT', 'WHEN', 'WHERE', 'WHO', 'WHY', 'HOW', 'IN', 'ON', 'AT', 'TO', 'OF', 'BY', 'FOR', 'WITH', 'ABOUT', 'AGAINST', 'BETWEEN', 'INTO', 'THROUGH', 'DURING', 'BEFORE', 'AFTER', 'ABOVE', 'BELOW', 'FROM', 'UP', 'DOWN', 'IN', 'OUT', 'ON', 'OFF', 'OVER', 'UNDER', 'AGAIN', 'FURTHER', 'THEN', 'ONCE', 'HERE', 'THERE', 'WHEN', 'WHERE', 'WHY', 'HOW', 'ALL', 'ANY', 'BOTH', 'EACH', 'FEW', 'MORE', 'MOST', 'OTHER', 'SOME', 'SUCH', 'NO', 'NOR', 'NOT', 'ONLY', 'OWN', 'SAME', 'SO', 'THAN', 'TOO', 'VERY', 'CAN', 'WILL', 'JUST', 'SHOULD', 'NOW'}
                
                filtered_tickers = [ticker for ticker in tickers if ticker not in common_words]
                all_tickers = indices + filtered_tickers
                
                # Remove duplicates and limit to 5
                unique_tickers = list(dict.fromkeys(all_tickers))[:5]
                
                print(f"🔍 [{self.agent_name}] LLM extracted tickers: {unique_tickers}")
                return unique_tickers
            
            else:
                print(f"⚠️ [{self.agent_name}] LLM response invalid, falling back to regex extraction")
                return self._fallback_ticker_extraction(query)
                
        except Exception as e:
            print(f"❌ [{self.agent_name}] LLM ticker extraction failed: {str(e)}, using fallback")
            return self._fallback_ticker_extraction(query)
    
    def _fallback_ticker_extraction(self, query: str) -> list:
        """Fallback method for ticker extraction using regex patterns"""
        print(f"🔍 [{self.agent_name}] Using fallback ticker extraction for: '{query}'")
        
        # Look for patterns like ^NSEI, ^GSPC, etc. (market indices)
        index_pattern = r'\^[A-Z]+'
        indices = re.findall(index_pattern, query.upper())
        
        # Look for company tickers (1-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        tickers = re.findall(ticker_pattern, query.upper())
        
        # Filter out common words
        common_words = {'THE', 'AND', 'OR', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT', 'WILL', 'CAN', 'ARE', 'WAS', 'HAS', 'HAD', 'NOT', 'BUT', 'YOU', 'ALL', 'ANY', 'HER', 'HIS', 'ITS', 'OUR', 'THEY', 'THEM', 'WHAT', 'WHEN', 'WHERE', 'WHO', 'WHY', 'HOW', 'IN', 'ON', 'AT', 'TO', 'OF', 'BY', 'FOR', 'WITH', 'ABOUT', 'AGAINST', 'BETWEEN', 'INTO', 'THROUGH', 'DURING', 'BEFORE', 'AFTER', 'ABOVE', 'BELOW', 'FROM', 'UP', 'DOWN', 'IN', 'OUT', 'ON', 'OFF', 'OVER', 'UNDER', 'AGAIN', 'FURTHER', 'THEN', 'ONCE', 'HERE', 'THERE', 'WHEN', 'WHERE', 'WHY', 'HOW', 'ALL', 'ANY', 'BOTH', 'EACH', 'FEW', 'MORE', 'MOST', 'OTHER', 'SOME', 'SUCH', 'NO', 'NOR', 'NOT', 'ONLY', 'OWN', 'SAME', 'SO', 'THAN', 'TOO', 'VERY', 'CAN', 'WILL', 'JUST', 'SHOULD', 'NOW'}
        
        filtered_tickers = [ticker for ticker in tickers if ticker not in common_words]
        
        # Combine indices and filtered tickers
        all_tickers = indices + filtered_tickers
        print(f"🔍 [{self.agent_name}] Fallback extracted tickers: {all_tickers[:5]}")
        
        return all_tickers[:5]  # Limit to 5 tickers
    
    def _fetch_ticker_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch essential data for a single ticker"""
        try:
            print(f"🔍 [{self.agent_name}] Fetching data for {ticker}...")
            
            ticker_obj = yf.Ticker(ticker)
            
            # Fetch essential data
            info = ticker_obj.info
            history = ticker_obj.history(period="1mo", interval="1d")
            
            # Extract key metrics
            key_metrics = {
                "current_price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
                "market_cap": info.get("marketCap", "N/A"),
                "pe_ratio": info.get("trailingPE", "N/A")
            }
            
            # Process historical data
            if not history.empty:
                history = history.tail(30)  # Last 30 days
                if history.index.name == 'Date':
                    history = history.reset_index()
                if 'Date' in history.columns:
                    history['Date'] = history['Date'].dt.strftime('%Y-%m-%d')
                
                # Ensure numeric columns are properly formatted
                numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in numeric_columns:
                    if col in history.columns:
                        history[col] = pd.to_numeric(history[col], errors='coerce')
            
            # Create result dictionary
            result = {
                "ticker": ticker,
                "key_metrics": key_metrics,
                "history": history.to_dict('records') if not history.empty else []
            }
            
            print(f"✅ [{self.agent_name}] Successfully fetched data for {ticker}")
            return self._safe_json_serialize(result)
            
        except Exception as e:
            print(f"❌ [{self.agent_name}] Error fetching {ticker}: {str(e)}")
            return {"error": str(e), "ticker": ticker}
    
    def _duckduckgo_search(self, query: str) -> str:
        """Use DDGS to fetch search results and rank them by relevance, credibility, and freshness"""
        try:
            all_results = []
            with self.search_client as ddgs:
                # Fetch 10 sources instead of 5
                for i, r in enumerate(ddgs.text(query, max_results=10)):
                    print(f"🔍 [{self.agent_name}] Raw result {i+1} type: {type(r)}, content: {str(r)[:100]}...")
                    
                    # Extract relevant text content from search result
                    if isinstance(r, dict):
                        # Handle dictionary format from DDGS
                        title = r.get('title', '')
                        body = r.get('body', '')
                        url = r.get('link', '')
                        if title and body:
                            formatted_result = f"Title: {title}\nContent: {body}\nURL: {url}\n"
                            all_results.append(formatted_result)
                            print(f"✅ [{self.agent_name}] Result {i+1} formatted with URL: {url}")
                        else:
                            print(f"⚠️ [{self.agent_name}] Result {i+1} missing title or body: {r}")
                    elif isinstance(r, str):
                        # Handle string format - try to extract URL if present
                        result_text = str(r)
                        # Look for URLs in the string
                        url_match = re.search(r'https?://[^\s]+', result_text)
                        if url_match:
                            url = url_match.group(0)
                            formatted_result = f"Content: {result_text}\nURL: {url}\n"
                            all_results.append(formatted_result)
                            print(f"✅ [{self.agent_name}] Result {i+1} string format with extracted URL: {url}")
                        else:
                            # No URL found, add as is
                            all_results.append(result_text)
                            print(f"⚠️ [{self.agent_name}] Result {i+1} string format without URL")
                    else:
                        # Handle other formats
                        result_text = str(r)
                        all_results.append(result_text)
                        print(f"⚠️ [{self.agent_name}] Result {i+1} unexpected format: {type(r)}")
            
            if all_results:
                print(f"🔍 [{self.agent_name}] Found {len(all_results)} total search results")
                
                # Debug: Check if URLs are present
                urls_found = sum(1 for result in all_results if 'URL:' in result)
                print(f"🔗 [{self.agent_name}] Results with URLs: {urls_found}/{len(all_results)}")
                
                # Validate and clean URLs in all results
                print(f"🔗 [{self.agent_name}] Validating and cleaning URLs...")
                cleaned_results = self.validate_and_clean_urls(all_results)
                
                # Rank sources and select top 5
                ranked_sources = self.rank_sources(query, cleaned_results)
                top_5_sources = ranked_sources[:5]
                
                print(f"🏆 [{self.agent_name}] Selected top 5 sources after ranking")
                
                # Debug: Check URLs in top 5
                top_5_urls = sum(1 for result in top_5_sources if 'URL:' in result)
                print(f"🔗 [{self.agent_name}] Top 5 sources with URLs: {top_5_urls}/5")
                
                # Final validation: ensure URLs are properly formatted
                final_sources = []
                for i, source in enumerate(top_5_sources):
                    if 'URL:' in source:
                        # Extract and validate the URL
                        url_match = re.search(r'URL: (https?://[^\s\n]+)', source)
                        if url_match:
                            url = url_match.group(1)
                            # Ensure URL is clean
                            clean_url = url.strip().rstrip('.,;:!?')
                            if clean_url.startswith(('http://', 'https://')):
                                final_sources.append(source)
                                print(f"✅ [{self.agent_name}] Source {i+1} has valid URL: {clean_url}")
                            else:
                                print(f"⚠️ [{self.agent_name}] Source {i+1} has invalid URL format: {url}")
                                final_sources.append(source)  # Keep it anyway
                        else:
                            print(f"⚠️ [{self.agent_name}] Source {i+1} has URL: but couldn't extract it")
                            final_sources.append(source)
                    else:
                        print(f"⚠️ [{self.agent_name}] Source {i+1} has no URL")
                        final_sources.append(source)
                
                # Format the top 5 sources
                formatted_results = "\n---\n".join(final_sources)
                print(f"📄 [{self.agent_name}] Top 5 sources formatted, length: {len(formatted_results)} characters")
                
                # Log a sample of the formatted results to verify URLs
                print(f"🔍 [{self.agent_name}] Sample of formatted results:")
                for i, source in enumerate(final_sources[:2]):  # Show first 2 sources
                    url_match = re.search(r'URL: (https?://[^\s\n]+)', source)
                    if url_match:
                        print(f"   Source {i+1} URL: {url_match.group(1)}")
                    else:
                        print(f"   Source {i+1}: No URL found")
                
                return formatted_results
            else:
                return "No search results found."
        except Exception as e:
            print(f"❌ [{self.agent_name}] Error during search: {str(e)}")
            return f"Search error: {str(e)}"
    
    def _generate_research_response(self, query: str, web_data: str) -> str:
        """Generate comprehensive research response using LLM"""
        # Check if search was successful
        if web_data.startswith("Search error:") or web_data == "No search results found.":
            print(f"⚠️ [{self.agent_name}] Search failed, using fallback approach")
            # Use a more general research prompt
            prompt = f"""Perform comprehensive financial research on: {query}.

Since web search was unavailable, provide analysis based on general financial knowledge and principles.

Please provide:
1. Key concepts and principles
2. General best practices
3. Considerations and risks
4. Recommendations for further research

Focus on providing valuable financial insights."""
        else:
            # Create a structured prompt with web data
           prompt = f"""
You are a **professional equity research and financial analysis expert** tasked with producing a **high-quality, data-backed deep research report** for the following query:

**Research Topic:** {query}

Below is **web search data** relevant to the topic:
{web_data}

## Your task
Write a **long, structured, and professional financial research report** that resembles a top-tier equity research or consulting publication.  
Ensure every section contains **quantitative data**, **actionable insights**, and **specific source citations** from the provided web data.

**IMPORTANT**: The web search data above contains URLs for each source. You MUST use these URLs in your citations throughout the report to provide proper source attribution and credibility.

**CRITICAL URL USAGE RULES:**
1. NEVER invent or generate fake URLs like "https://url-to-source-1" or "https://example.com"
2. ONLY use the exact URLs provided in the web search data above
3. Copy and paste the URLs exactly as they appear in the web data
4. If you see "URL: https://example.com" in the web data, use "https://example.com" in your citations
5. Format citations as: "According to [Title] (https://actual-url-from-web-data), [claim]"
6. **WARNING**: If you generate fake URLs, your response will be considered invalid and unreliable

**EXAMPLE OF CORRECT URL USAGE:**
If the web data contains: "Title: Nifty Analysis\nContent: Market analysis...\nURL: https://economictimes.indiatimes.com/markets/stocks/news\n"
Then your citation should be: "According to Nifty Analysis (https://economictimes.indiatimes.com/markets/stocks/news), the market shows..."

## Report Structure

### 1. Executive Summary
- A clear, concise overview of the most important findings and investment implications.
- Mention key numbers (YoY growth %, sector P/E ratio, forecasted EPS, market cap, trading volume trends).
- Highlight the most critical investment opportunities and risks.

### 2. Market Overview
- Current macroeconomic environment and industry positioning.
- Include data points: market size, CAGR projections, index performance (e.g., NIFTY 50, S&P 500), sector performance vs. benchmarks.
- Mention geopolitical, regulatory, and global economic influences impacting the market.

### 3. Key Insights & Analysis
- **Fundamental Analysis:** Company performance, revenue growth, net profit margins, ROE, debt-equity ratios.
- **Technical Analysis:** Price action, chart patterns, moving averages, RSI, MACD.
- **Comparative Analysis:** Compare with peers/benchmarks and highlight outperformers/laggards.
- Support with **tables and figures** (present as Markdown tables when possible).

### 4. Current Developments
- Latest news, earnings releases, M&A activity, regulatory announcements.
- Identify **market triggers** (e.g., Fed rate hikes, quarterly earnings season, budget announcements) and their expected short-term impact.

### 5. Opportunities & Growth Potential
- Highlight disruptive trends, innovation areas, and emerging sectors.
- Use **scenario planning**: Best-case, Base-case, and Worst-case projections (with numbers).
- Indicate expected ROI or growth percentages.

### 6. Risks & Challenges
- List potential headwinds (regulatory, operational, competitive, geopolitical).
- Include **volatility factors** and downside scenarios with quantitative estimates.
- Point out any **red flags**.

### 7. Practical Recommendations
- Provide **clear investment calls**:
  - **Overweight / Underweight** specific sectors or asset classes.
  - Suggested entry/exit price ranges and target time horizons.
  - Risk management or hedging strategies.
- Differentiate between **short-term tactical** and **long-term strategic** positions.

### 8. Conclusion
- Reinforce the main strategic outlook.
- State the highest-priority action items for investors or businesses.

### 9. Sources & References
- **CRITICAL**: Use the **exact URLs** provided in the web search data above for source citations.
- For each major claim or data point, cite the specific source URL from the web data.
- If a URL is provided in the web data, you MUST use it in your citations.
- Format citations as: "According to [Source Name] ([URL]), [claim/data]..."
- If URLs aren’t available, summarize the source credibility.

---

**Formatting & Style Guidelines:**
- Use **clear section headings**.
- Always include **numerical evidence** where available.
- Keep tone factual, analytical, and investment-oriented.
- Avoid generic statements — always connect insights to **data-backed reasoning**.

"""
        
        print(f"🤖 [{self.agent_name}] Calling LLM for comprehensive response...")
        print(f"📝 [{self.agent_name}] Prompt length: {len(prompt)} characters")
        
        llm_start = time.time()
        response = self.llm_client(prompt)
        response = self._log_llm_response(response, llm_start)
        
        return response
    
    def _generate_pdf_report(self, query: str, response: str, web_data: str) -> str:
        """Generate a professional PDF report with nicely formatted paragraphs."""
        print(f"📄 [{self.agent_name}] Generating PDF report...")

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_query = safe_query.replace(' ', '_')[:50]  # Limit length
        filename = f"financial_research_{safe_query}_{timestamp}.pdf"
        filepath = os.path.join(self.reports_dir, filename)

        try:
            # Create PDF document
            doc = SimpleDocTemplate(filepath, pagesize=A4)
            story = []

            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.darkblue
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=20,
                textColor=colors.darkblue
            )
            normal_style = styles['Normal']

            # Helper function to split text into paragraphs
            def split_text_to_paragraphs(text, max_chars=300):
                import re
                sentences = re.split(r'(?<=[.!?]) +', text)
                paragraphs = []
                current_para = ""
                for sentence in sentences:
                    if len(current_para) + len(sentence) <= max_chars:
                        current_para += sentence + " "
                    else:
                        paragraphs.append(current_para.strip())
                        current_para = sentence + " "
                if current_para:
                    paragraphs.append(current_para.strip())
                return paragraphs

            # Title
            story.append(Paragraph("Financial Research Report", title_style))
            story.append(Spacer(1, 20))

            # Query
            story.append(Paragraph(f"<b>Research Query:</b> {query}", heading_style))
            story.append(Spacer(1, 15))

            # Executive Summary
            story.append(Paragraph("<b>Executive Summary</b>", heading_style))
            summary = response[:500] + "..." if len(response) > 500 else response
            for para in split_text_to_paragraphs(summary):
                story.append(Paragraph(para, normal_style))
                story.append(Spacer(1, 10))

            # Key Findings
            story.append(Paragraph("<b>Key Findings</b>", heading_style))
            for para in split_text_to_paragraphs(response):
                story.append(Paragraph(para, normal_style))
                story.append(Spacer(1, 10))

            # Research Sources (if available)
            if web_data and not web_data.startswith("Search error:") and web_data != "No search results found.":
                story.append(Paragraph("<b>Research Sources</b>", heading_style))
                story.append(Paragraph("This analysis is based on the following web sources:", normal_style))
                story.append(Spacer(1, 10))

                sources = web_data.split("---")
                for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
                    if source.strip():
                        story.append(Paragraph(f"<b>Source {i}:</b>", normal_style))
                        # Break source into paragraphs
                        for para in split_text_to_paragraphs(source.strip(), max_chars=400):
                            story.append(Paragraph(para, normal_style))
                            story.append(Spacer(1, 5))
                        story.append(Spacer(1, 10))

            # Report metadata
            story.append(Spacer(1, 20))
            story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", normal_style))
            story.append(Paragraph(f"<b>Agent:</b> Deep Research Agent", normal_style))

            # Build PDF
            doc.build(story)

            print(f"✅ [{self.agent_name}] PDF report generated: {filename}")
            return filepath

        except Exception as e:
            print(f"❌ [{self.agent_name}] Error generating PDF: {str(e)}")
            return None

    def _generate_html_dashboard(self, query: str, response: str, web_data: str, yahoo_data: Dict[str, Any]) -> str:
        """Generate an interactive HTML dashboard with charts and tables"""
        print(f"🌐 [{self.agent_name}] Generating HTML dashboard...")
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_query = safe_query.replace(' ', '_')[:50]  # Limit length
        filename = f"financial_dashboard_{safe_query}_{timestamp}.html"
        filepath = os.path.join(self.reports_dir, filename)
        
        try:
            # Generate charts and tables
            charts_html = self._generate_charts_html(yahoo_data)
            tables_html = self._generate_tables_html(yahoo_data)
            
            # Create the HTML dashboard
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Research Dashboard - {query}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
            font-size: 1.1em;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #3498db;
        }}
        .section h2 {{
            color: #2c3e50;
            margin-top: 0;
            font-size: 1.8em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
            border-top: 4px solid #3498db;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .table-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin: 20px 0;
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e8f4fd;
        }}
        .summary {{
            background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            border-left: 5px solid #17a2b8;
        }}
        .footer {{
            background: #2c3e50;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            .header h1 {{
                font-size: 2em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Financial Research Dashboard</h1>
            <p>Comprehensive Analysis: {query}</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📈 Executive Summary</h2>
                <div class="summary">
                    <p>{response[:1000]}{'...' if len(response) > 1000 else ''}</p>
                </div>
            </div>
            
            {charts_html}
            
            {tables_html}
            
            <div class="section">
                <h2>🔍 Research Sources</h2>
                <div class="summary">
                    {self._format_web_sources(web_data)}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by Deep Research Agent | Financial Analysis Dashboard</p>
        </div>
    </div>
</body>
</html>
            """
            
            # Write HTML file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"✅ [{self.agent_name}] HTML dashboard generated: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ [{self.agent_name}] Error generating HTML dashboard: {str(e)}")
            return None
    
    def _generate_charts_html(self, yahoo_data: Dict[str, Any]) -> str:
        """Generate HTML for interactive charts"""
        if not yahoo_data or yahoo_data.get("error"):
            return '<div class="section"><h2>📊 Charts</h2><p>No financial data available for charts.</p></div>'
        
        print(f"🔍 [{self.agent_name}] Generating charts for yahoo_data: {type(yahoo_data)}")
        if isinstance(yahoo_data, dict):
            print(f"🔍 [{self.agent_name}] Yahoo data keys: {list(yahoo_data.keys())}")
            if "data" in yahoo_data:
                print(f"🔍 [{self.agent_name}] Data keys: {list(yahoo_data['data'].keys())}")
        
        charts_html = '<div class="section"><h2>📊 Financial Charts & Visualizations</h2>'
        
        for ticker, data in yahoo_data.get("data", {}).items():
            print(f"🔍 [{self.agent_name}] Processing ticker {ticker}: {type(data)}")
            if isinstance(data, dict) and not data.get("error"):
                charts_html += f'<div class="chart-container"><h3>{ticker} - Key Metrics</h3>'
                
                # Debug: Print the data structure
                print(f"🔍 [{self.agent_name}] {ticker} data keys: {list(data.keys())}")
                print(f"🔍 [{self.agent_name}] {ticker} key_metrics: {data.get('key_metrics', {})}")
                print(f"🔍 [{self.agent_name}] {ticker} history type: {type(data.get('history'))}")
                if data.get("history"):
                    print(f"🔍 [{self.agent_name}] {ticker} history length: {len(data['history'])}")
                    if len(data['history']) > 0:
                        print(f"🔍 [{self.agent_name}] {ticker} first history record: {data['history'][0]}")
                
                # Price chart if historical data available
                if data.get("history"):
                    # Safely extract and format date and price data
                    try:
                        dates = []
                        prices = []
                        for d in data['history']:
                            date_val = d.get('Date', '')
                            close_val = d.get('Close', 0)
                            
                            # Convert Timestamp to string if needed
                            if hasattr(date_val, 'strftime'):
                                date_val = date_val.strftime('%Y-%m-%d')
                            elif not isinstance(date_val, str):
                                date_val = str(date_val)
                            
                            # Ensure price is numeric
                            if isinstance(close_val, (int, float)):
                                prices.append(close_val)
                                dates.append(date_val)
                        
                        if dates and prices:
                            charts_html += f'''
                            <div id="price-chart-{ticker}" style="height: 400px;"></div>
                            <script>
                                var trace = {{
                                    x: {json.dumps(dates)},
                                    y: {json.dumps(prices)},
                                    type: 'scatter',
                                    mode: 'lines+markers',
                                    name: '{ticker} Price',
                                    line: {{color: '#3498db', width: 3}},
                                    marker: {{size: 6}}
                                }};
                                
                                var layout = {{
                                    title: '{ticker} Stock Price (Last 30 Days)',
                                    xaxis: {{title: 'Date'}},
                                    yaxis: {{title: 'Price ($)'}},
                                    hovermode: 'closest',
                                    plot_bgcolor: 'rgba(0,0,0,0)',
                                    paper_bgcolor: 'rgba(0,0,0,0)'
                                }};
                                
                                Plotly.newPlot('price-chart-{ticker}', [trace], layout);
                            </script>
                            '''
                    except Exception as e:
                        print(f"⚠️ [{self.agent_name}] Error generating chart for {ticker}: {str(e)}")
                        charts_html += f'<p>Chart generation failed for {ticker}: {str(e)}</p>'
                
                # Key metrics cards
                metrics = data.get("key_metrics", {})
                if metrics:
                    charts_html += '<div class="metrics-grid">'
                    for key, value in metrics.items():
                        if value != "N/A" and value is not None:
                            # Skip data_freshness in the metrics grid, show it separately
                            if key == "data_freshness":
                                continue
                            formatted_value = self._format_metric_value(key, value)
                            charts_html += f'''
                            <div class="metric-card">
                                <div class="metric-label">{key.replace('_', ' ').title()}</div>
                                <div class="metric-value">{formatted_value}</div>
                            </div>
                            '''
                    charts_html += '</div>'
                    
                    # Show data freshness separately
                    if "data_freshness" in metrics:
                        charts_html += f'''
                        <div style="text-align: center; margin: 20px 0; padding: 15px; background: #e8f4fd; border-radius: 8px; border-left: 4px solid #17a2b8;">
                            <strong>📅 Data Last Updated:</strong> {metrics["data_freshness"]}
                        </div>
                        '''
                
                charts_html += '</div>'
        
        charts_html += '</div>'
        return charts_html
    
    def _generate_tables_html(self, yahoo_data: Dict[str, Any]) -> str:
        """Generate HTML for data tables"""
        if not yahoo_data or yahoo_data.get("error"):
            return '<div class="section"><h2>📋 Data Tables</h2><p>No financial data available for tables.</p></div>'
        
        tables_html = '<div class="section"><h2>📋 Financial Data Tables</h2>'
        
        for ticker, data in yahoo_data.get("data", {}).items():
            if isinstance(data, dict) and not data.get("error"):
                tables_html += f'<div class="table-container"><h3>{ticker} - Financial Data</h3>'
                
                # Key metrics table
                metrics = data.get("key_metrics", {})
                if metrics:
                    tables_html += '<h4>Key Metrics</h4><table><thead><tr>'
                    tables_html += '<th>Metric</th><th>Value</th></tr></thead><tbody>'
                    for key, value in metrics.items():
                        if value != "N/A" and value is not None:
                            formatted_value = self._format_metric_value(key, value)
                            tables_html += f'<tr><td>{key.replace("_", " ").title()}</td><td>{formatted_value}</td></tr>'
                    tables_html += '</tbody></table>'
                
                # Historical data table
                if data.get("history"):
                    tables_html += '<h4>Recent Price History</h4><table><thead><tr>'
                    tables_html += '<th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th></tr></thead><tbody>'
                    
                    # Get the last 10 records and ensure proper formatting
                    history_records = data["history"][-10:] if len(data["history"]) > 10 else data["history"]
                    
                    for record in history_records:
                        if isinstance(record, dict):
                            date_val = record.get("Date", "")
                            open_val = record.get("Open", "")
                            high_val = record.get("High", "")
                            low_val = record.get("Low", "")
                            close_val = record.get("Close", "")
                            volume_val = record.get("Volume", "")
                            
                            # Format numeric values
                            def format_price(val):
                                if isinstance(val, (int, float)) and val != 0:
                                    return f"${val:.2f}"
                                return str(val) if val else "N/A"
                            
                            def format_volume(val):
                                if isinstance(val, (int, float)) and val != 0:
                                    return f"{val:,.0f}"
                                return str(val) if val else "N/A"
                            
                            tables_html += f'<tr><td>{date_val}</td><td>{format_price(open_val)}</td><td>{format_price(high_val)}</td><td>{format_price(low_val)}</td><td>{format_price(close_val)}</td><td>{format_volume(volume_val)}</td></tr>'
                    
                    tables_html += '</tbody></table>'
                
                tables_html += '</div>'
        
        tables_html += '</div>'
        return tables_html
    
    def _format_metric_value(self, key: str, value: Any) -> str:
        """Format metric values for display"""
        if isinstance(value, (int, float)):
            if key in ['market_cap']:
                return f"${value:,.0f}"
            elif key in ['volume', 'avg_volume']:
                return f"{value:,.0f}"  # Volume is a count, not a price
            elif key in ['pe_ratio', 'forward_pe', 'price_to_book', 'beta']:
                return f"{value:.2f}"
            elif key == 'dividend_yield':
                return f"{value:.2%}" if value > 0 else "N/A"
            elif key in ['fifty_two_week_high', 'fifty_two_week_low', 'current_price']:
                return f"${value:.2f}"
            else:
                return f"{value:,.0f}"
        return str(value)
    
    def _format_web_sources(self, web_data: str) -> str:
        """Format web sources for HTML display"""
        if not web_data or web_data.startswith("Search error:") or web_data == "No search results found.":
            return "<p>No web sources available for this analysis.</p>"
        
        sources = web_data.split("---")
        formatted_sources = ""
        for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
            if source.strip():
                formatted_sources += f"<p><strong>Source {i}:</strong> {source.strip()}</p>"
        
        return formatted_sources if formatted_sources else "<p>Web sources processed but no specific details available.</p>"
    
    def _clear_cache_and_refresh(self):
        """Clear any cached data and force fresh fetches"""
        try:
            # Clear any potential yfinance cache
            import yfinance as yf
            if hasattr(yf, 'cache'):
                yf.cache.clear()
            print(f"🧹 [{self.agent_name}] Cleared yfinance cache")
        except Exception as e:
            print(f"⚠️ [{self.agent_name}] Could not clear cache: {str(e)}")
    
    def _get_capabilities(self) -> list:
        """Get deep research agent capabilities"""
        return [
            "query_processing",
            "web_research",
            "yahoo_finance_metrics",
            "data_synthesis",
            "comprehensive_analysis",
            "trend_analysis",
            "pdf_report_generation",
            "html_dashboard_generation",
            "financial_charts",
            "data_visualization"
        ]

    