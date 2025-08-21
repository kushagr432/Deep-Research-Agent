# deep_research_agent.py
from .base_agent import BaseAgent
from typing import Dict, Any, Optional, AsyncGenerator
import time
import asyncio
import os
from datetime import datetime
from ddgs import DDGS
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

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
            "pdf_report_generation"
        ]
        
        # Create reports directory if it doesn't exist
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
    
    async def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process deep research queries with web data"""
        start_time = self._log_start("deep research", query)
        
        # Extract parameters
        generate_report = kwargs.get('generate_report', False)
        
        # Perform web search
        web_data = await self._perform_web_search(query)
        
        # Generate comprehensive response
        response = await self._generate_research_response(query, web_data)
        
        # Generate PDF report if requested
        pdf_path = None
        if generate_report:
            pdf_path = await self._generate_pdf_report(query, response, web_data)
        
        # Log completion
        total_time = time.time() - start_time
        self._log_completion("deep research", total_time)
        
        return {
            "response": response,
            "agent": "deep_research",
            "web_data": web_data,
            "specializations": self.specializations,
            "processing_time": total_time,
            "pdf_report_path": pdf_path,
            "report_generated": generate_report
        }
    
    async def process_stream(self, query: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Process deep research queries with streaming updates"""
        start_time = self._log_start("deep research", query)
        
        # Extract parameters
        generate_report = kwargs.get('generate_report', False)
        
        # Stream: Starting research
        yield {
            "type": "status",
            "message": "Starting deep financial research...",
            "step": "start"
        }
        
        # Stream: Web search
        yield {
            "type": "status",
            "message": "Searching the web for current information...",
            "step": "web_search"
        }
        
        # Perform web search
        web_data = await self._perform_web_search(query)
        
        # Stream: Search results
        if web_data and not web_data.startswith("Search error:") and web_data != "No search results found.":
            yield {
                "type": "status",
                "message": f"Found {len(web_data.split('---'))} research sources",
                "step": "search_complete"
            }
        else:
            yield {
                "type": "status",
                "message": "Using general financial knowledge for analysis",
                "step": "search_fallback"
            }
        
        # Stream: Generating response
        yield {
            "type": "status",
            "message": "Analyzing research data and generating insights...",
            "step": "analysis"
        }
        
        # Generate comprehensive response
        response = await self._generate_research_response(query, web_data)
        
        # Stream: Response chunks (simulate streaming by splitting into sentences)
        sentences = response.split('. ')
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                yield {
                    "type": "response_chunk",
                    "content": sentence.strip() + ('.' if i < len(sentences) - 1 else ''),
                    "chunk_index": i,
                    "total_chunks": len(sentences)
                }
                await asyncio.sleep(0.1)  # Small delay for realistic streaming effect
        
        # Generate PDF report if requested
        pdf_path = None
        if generate_report:
            yield {
                "type": "status",
                "message": "Generating PDF report...",
                "step": "pdf_generation"
            }
            pdf_path = await self._generate_pdf_report(query, response, web_data)
            
            if pdf_path:
                yield {
                    "type": "status",
                    "message": "PDF report generated successfully!",
                    "step": "pdf_complete"
                }
        
        # Log completion
        total_time = time.time() - start_time
        self._log_completion("deep research", total_time)
        
        # Stream: Final result
        yield {
            "type": "final_result",
            "response": response,
            "agent": "deep_research",
            "web_data": web_data,
            "specializations": self.specializations,
            "processing_time": total_time,
            "pdf_report_path": pdf_path,
            "report_generated": generate_report
        }
    
    async def _perform_web_search(self, query: str) -> str:
        """Perform web search using DuckDuckGo"""
        print(f"🌐 [{self.agent_name}] Searching web for: '{query}'")
        start_time = time.time()
        
        try:
            web_data = await asyncio.to_thread(self._duckduckgo_search, query)
            search_time = time.time() - start_time
            print(f"📊 [{self.agent_name}] Web search completed in {search_time:.2f}s")
            print(f"📄 [{self.agent_name}] Web data length: {len(web_data)} characters")
            return web_data
        except Exception as e:
            print(f"❌ [{self.agent_name}] Web search failed: {str(e)}")
            return f"Search error: {str(e)}"
    
    def _duckduckgo_search(self, query: str) -> str:
        """Use DDGS to fetch search results"""
        try:
            results = []
            with self.search_client as ddgs:
                for i, r in enumerate(ddgs.text(query, max_results=5)):
                    print(f"🔍 [{self.agent_name}] Raw result {i+1} type: {type(r)}, content: {str(r)[:100]}...")
                    
                    # Extract relevant text content from search result
                    if isinstance(r, dict):
                        # Handle dictionary format from DDGS
                        title = r.get('title', '')
                        body = r.get('body', '')
                        url = r.get('link', '')
                        if title and body:
                            results.append(f"Title: {title}\nContent: {body}\nURL: {url}\n")
                        else:
                            print(f"⚠️ [{self.agent_name}] Result {i+1} missing title or body: {r}")
                    elif isinstance(r, str):
                        # Handle string format
                        results.append(r)
                    else:
                        # Handle other formats
                        print(f"⚠️ [{self.agent_name}] Result {i+1} unexpected format: {type(r)}")
                        results.append(str(r))
            
            if results:
                formatted_results = "\n---\n".join(results)
                print(f"🔍 [{self.agent_name}] Found {len(results)} search results")
                print(f"📄 [{self.agent_name}] Formatted results length: {len(formatted_results)} characters")
                return formatted_results
            else:
                return "No search results found."
        except Exception as e:
            print(f"❌ [{self.agent_name}] Error during search: {str(e)}")
            return f"Search error: {str(e)}"
    
    async def _generate_research_response(self, query: str, web_data: str) -> str:
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
You are a professional financial research analyst tasked with preparing a **comprehensive deep research report** on the following query:

**Research Topic:** {query}

Below is relevant web search data that you must carefully analyze:
{web_data[:3000]}

Your task is to create a **long, detailed, and structured financial research report**. 
Ensure the tone is professional, analytical, and precise, similar to top-tier equity research or consulting reports.

The report must include:

1. **Executive Summary**
   - A concise overview of the key findings, conclusions, and recommendations.
   - Highlight critical investment insights or risks upfront.

2. **Market Overview**
   - Current market trends, macroeconomic factors, and industry outlook related to the query.
   - Geopolitical, regulatory, or global economic influences.

3. **Key Insights & Analysis**
   - Fundamental analysis (company performance, financial ratios, revenue, margins, debt, etc., if applicable).
   - Technical analysis (chart patterns, price action, indicators, market sentiment).
   - Comparative analysis (peers, sector performance, benchmarks).
   - Institutional or expert views, if available.

4. **Current Developments**
   - Latest news, events, and updates impacting the query (earnings reports, regulatory changes, global macro events).
   - Short-term and long-term implications.

5. **Opportunities & Growth Potential**
   - Emerging trends, innovation, or disruptions in the sector.
   - Investment opportunities and scenarios where growth is possible.

6. **Risks & Challenges**
   - Downside risks, volatility factors, competitive threats, macroeconomic pressures.
   - Potential red flags investors should watch out for.

7. **Practical Recommendations**
   - Actionable investment or business strategies.
   - Short-term vs. long-term recommendations.
   - Potential entry/exit points, allocation suggestions, or hedging strategies (if investment-related).

8. **Conclusion**
   - Summarize the strategic outlook.
   - Reinforce the most critical insight(s).

9. **Sources and References**
   - Provide a structured list of references used from the web data.
   - If exact links are available, cite them. If not, summarize source credibility.

Formatting Guidelines:
- Use **clear section headings**.
- Write in a structured, report-style format.
- Ensure clarity, depth, and actionable insights.

Your final output should read like a **professional financial deep research report**, not just a summary.
"""
        
        print(f"🤖 [{self.agent_name}] Calling LLM for comprehensive response...")
        print(f"📝 [{self.agent_name}] Prompt length: {len(prompt)} characters")
        
        llm_start = time.time()
        response = self.llm_client(prompt)
        response = self._log_llm_response(response, llm_start)
        
        return response
    
    async def _generate_pdf_report(self, query: str, response: str, web_data: str) -> str:
        """Generate a professional PDF report"""
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
            
            # Title
            story.append(Paragraph("Financial Research Report", title_style))
            story.append(Spacer(1, 20))
            
            # Query
            story.append(Paragraph(f"<b>Research Query:</b> {query}", heading_style))
            story.append(Spacer(1, 15))
            
            # Executive Summary
            story.append(Paragraph("<b>Executive Summary</b>", heading_style))
            summary = response[:500] + "..." if len(response) > 500 else response
            story.append(Paragraph(summary, normal_style))
            story.append(Spacer(1, 20))
            
            # Key Findings
            story.append(Paragraph("<b>Key Findings</b>", heading_style))
            story.append(Paragraph(response, normal_style))
            story.append(Spacer(1, 20))
            
            # Research Sources (if available)
            if web_data and not web_data.startswith("Search error:") and web_data != "No search results found.":
                story.append(Paragraph("<b>Research Sources</b>", heading_style))
                story.append(Paragraph("This analysis is based on the following web sources:", normal_style))
                story.append(Spacer(1, 10))
                
                # Split web data into sources
                sources = web_data.split("---")
                for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
                    if source.strip():
                        story.append(Paragraph(f"<b>Source {i}:</b>", normal_style))
                        story.append(Paragraph(source.strip(), normal_style))
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
    
    def _get_capabilities(self) -> list:
        """Get deep research agent capabilities"""
        return [
            "query_processing",
            "web_research",
            "data_synthesis",
            "comprehensive_analysis",
            "trend_analysis",
            "pdf_report_generation"
        ]
