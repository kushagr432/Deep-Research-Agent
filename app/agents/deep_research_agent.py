"""
Deep Research Agent for comprehensive financial research and PDF report generation
"""
import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
from asyncio_throttle import Throttler

logger = logging.getLogger(__name__)

class DeepResearchAgent:
    """Agent specialized in deep financial research with web search and PDF generation"""
    
    def __init__(self):
        self.throttler = Throttler(rate_limit=10, period=1)  # 10 requests per second
        self.llm_cache = {}  # Simple in-memory cache for LLM responses
        self.name = "DeepResearchAgent"
        self.specialization = "Comprehensive Financial Research & Analysis"
        self.version = "1.0.0"
        
        # Research capabilities
        self.research_methods = [
            "web_search",
            "vector_knowledge_retrieval", 
            "financial_data_analysis",
            "market_trend_analysis",
            "risk_assessment"
        ]
        
        # Report templates
        self.report_templates = {
            "investment_analysis": "investment_report_template.html",
            "market_research": "market_research_template.html",
            "company_analysis": "company_analysis_template.html",
            "risk_assessment": "risk_assessment_template.html"
        }
    
    async def perform_deep_research_streaming(self, query: str, research_type: str = "comprehensive", generate_pdf: bool = False):
        """Perform deep research with real-time streaming updates"""
        try:
            logger.info(f"Starting streaming deep research: {query}")
            start_time = datetime.now()
            
            # Step 1: Parallel Web Search + Vector Knowledge (no waiting)
            web_task = asyncio.create_task(self._perform_web_search(query, max_results=5))
            # vector_task = asyncio.create_task(self._retrieve_vector_knowledge(query))
            
            # Wait for both to complete (should be ~1-2 seconds)
            web_results = await asyncio.gather(web_task)
            logger.info(f"Data collection completed: {len(web_results)} web results")
            
            # Step 2: Combine all sources
            all_sources = web_results
            
            # Step 3: LLM Analysis (with streaming updates)
            yield {'status': 'progress', 'step': 'llm_analysis', 'message': 'Analyzing with Llama3:8b...'}
            
            # Collect streaming analysis results
            analysis_results = {}
            full_analysis_text = ""
            
            try:
                async for analysis_chunk in self._generate_comprehensive_analysis_streaming(query, all_sources):
                    if analysis_chunk['status'] == 'analysis_chunk':
                        # Stream each chunk of analysis immediately
                        yield {'status': 'progress', 'step': 'llm_analysis', 'message': 'AI analysis in progress...', 'data': {'chunk': analysis_chunk['chunk']}}
                        full_analysis_text += analysis_chunk['chunk']
                    elif analysis_chunk['status'] == 'analysis_complete':
                        # Analysis completed, parse the full response
                        analysis_results = self._parse_comprehensive_analysis(analysis_chunk['full_response'])
                        yield {'status': 'progress', 'step': 'llm_analysis', 'message': 'AI analysis completed'}
                        break
                    elif analysis_chunk['status'] == 'analysis_error':
                        # Analysis failed, use fallback
                        analysis_results = self._get_fallback_comprehensive_analysis()
                        yield {'status': 'progress', 'step': 'llm_analysis', 'message': 'AI analysis completed with fallback'}
                        break
            except Exception as e:
                logger.error(f"LLM analysis streaming failed for query '{query}': {e}")
                analysis_results = self._get_fallback_comprehensive_analysis()
                yield {'status': 'progress', 'step': 'llm_analysis', 'message': 'AI analysis completed with fallback'}
            
            # Step 4: PDF Generation (if requested) - Run in parallel
            report_path = None
            if generate_pdf:
                yield {'status': 'progress', 'step': 'pdf_generation', 'message': 'Generating PDF report...'}
                try:
                    # Run PDF generation in parallel to avoid blocking
                    pdf_task = asyncio.create_task(self._generate_pdf_report_async(query, all_sources))
                    report_path = await pdf_task
                    yield {'status': 'progress', 'step': 'pdf_generation', 'message': 'PDF report generated', 'data': {'report_path': report_path}}
                except Exception as e:
                    logger.error(f"PDF generation failed for query '{query}': {e}")
                    yield {'status': 'progress', 'step': 'pdf_generation', 'message': 'PDF generation failed', 'data': {'error': str(e)}}
            
            # Calculate total time
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Streaming deep research completed for query '{query}' in {total_time:.2f} seconds")
            
            # Final results
            yield {
                'status': 'results',
                'message': 'Research completed successfully',
                'data': {
                    'query': query,
                    'research_type': research_type,
                    'timestamp': datetime.now().isoformat(),
                    'sources': all_sources,
                    'web_search_results': len(web_results),
                    'vector_results': len(vector_results),
                    'processing_time_seconds': total_time,
                    'report_path': report_path,
                    **analysis_results
                }
            }
            
        except Exception as e:
            logger.error(f"Streaming deep research failed for query '{query}': {e}")
            yield {'status': 'error', 'message': f'Deep research failed: {str(e)}'}
    
    async def _perform_web_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Perform real web search using DuckDuckGo (only for deep research)"""
        try:
            # Import the new ddgs package
            from ddgs import DDGS
            
            # Initialize search client
            with DDGS() as ddgs:
                # Perform web search with timeout
                search_results = ddgs.text(query, max_results=max_results)
                
                # Process and format results quickly
                formatted_results = []
                for result in search_results:
                    formatted_result = {
                        "source": "Web Search",
                        "title": result.get("title", "No title"),
                        "url": result.get("link", "No URL"),
                        "snippet": result.get("body", "No description")[:200],  # Limit snippet length
                        "relevance_score": 0.8,
                        "timestamp": datetime.now().isoformat()
                    }
                    formatted_results.append(formatted_result)
                
                logger.info(f"Web search completed: {len(formatted_results)} results found")
                return formatted_results
                
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            # Return empty results if web search fails
            return []
    
    async def _retrieve_vector_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve knowledge from vector database (currently mock)"""
        try:
            # Simulate vector database retrieval (very fast)
            await asyncio.sleep(0.05)  # Reduced from 0.1 to 0.05
            
            # Mock vector knowledge (in production, this would query Pinecone/FAISS)
            vector_results = [
                {
                    "source": "Vector DB",
                    "title": "Financial Knowledge Base",
                    "content": "Compound interest is interest earned on both the principal and accumulated interest.",
                    "relevance_score": 0.9,
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "source": "Vector DB", 
                    "title": "Investment Strategies",
                    "content": "Diversification helps reduce risk by spreading investments across different assets.",
                    "relevance_score": 0.8,
                    "timestamp": datetime.now().isoformat()
                }
            ]
            
            return vector_results
            
        except Exception as e:
            logger.error(f"Vector knowledge retrieval failed: {e}")
            return []
    
    async def _generate_comprehensive_analysis_streaming(self, query: str, sources: List[Any]):
        """Stream comprehensive analysis in real-time using appropriate agent"""
        try:
            # Detect which agent to use
            agent_type = self._detect_agent_type(query)
            logger.info(f"Using {agent_type} agent for deep research analysis")
            
            # Import appropriate agent
            if agent_type == "banking":
                from app.agents.banking_agent import BankingAgent
                agent = BankingAgent()
            else:
                from app.agents.finance_agent import FinanceAgent
                agent = FinanceAgent()
            
            # Create specialized prompt based on agent type
            source_summary = self._summarize_sources(sources)
            
            if agent_type == "banking":
                prompt = f"""
                You are a senior banking and credit research analyst. Analyze the following query and provide comprehensive insights.

                RESEARCH QUERY: {query}

                SOURCES ANALYZED: {len(sources)} sources
                SOURCE CONTEXT: {source_summary}

                Please provide a comprehensive banking analysis covering:

                1. Banking Analysis: Key insights about banking products, services, and implications
                2. Market Trends: Current banking and credit market conditions
                3. Recommendations: 3 actionable banking and credit suggestions
                4. Risk Assessment: Key banking and credit risks and mitigation strategies
                5. Executive Summary: 3-4 sentence overview of banking insights

                Focus on banking, loans, credit, and financial services.
                Format your response naturally. You don't need to follow exact labels.
                Provide valuable, actionable insights based on the sources.

                ANALYSIS:
                """
            else:
                prompt = f"""
                You are a senior financial research analyst. Analyze the following query and provide comprehensive insights.

                RESEARCH QUERY: {query}

                SOURCES ANALYZED: {len(sources)} sources
                SOURCE CONTEXT: {source_summary}

                Please provide a comprehensive financial analysis covering:

                1. Financial Analysis: Key insights and investment implications
                2. Market Trends: Current market conditions and investment patterns
                3. Recommendations: 3 actionable investment and financial suggestions
                4. Risk Assessment: Key investment risks and mitigation strategies
                5. Executive Summary: 3-4 sentence overview of financial insights

                Focus on investments, markets, and financial planning.
                Format your response naturally. You don't need to follow exact labels.
                Provide valuable, actionable insights based on the sources.

                ANALYSIS:
                """
            
            # Use the appropriate agent's LLM for streaming
            try:
                # Use the agent's streaming capability directly
                full_response = ""
                async for chunk in agent.generate_response_streaming(query, [source_summary], agent_type):
                    # Stream each chunk immediately as it's generated
                    full_response += chunk
                    yield {'status': 'analysis_chunk', 'chunk': chunk, 'partial': True}
                
                # Send completion signal
                yield {'status': 'analysis_complete', 'full_response': full_response, 'partial': False}
                        
            except Exception as streaming_error:
                logger.warning(f"Agent streaming failed, falling back to direct LLM: {streaming_error}")
                
                # Fallback: Use direct LLM approach with smart chunking
                from langchain_community.llms import Ollama
                llm = Ollama(model="llama3:8b", temperature=0.7, top_p=0.9)
                
                response = await llm.agenerate([prompt])
                generated_text = response.generations[0][0].text.strip()
                
                if generated_text:
                    # Use smart chunking instead of sentence splitting
                    chunks = self._create_smart_chunks(generated_text)
                    for chunk in chunks:
                        if chunk.strip():
                            yield {'status': 'analysis_chunk', 'chunk': chunk.strip(), 'partial': True}
                            await asyncio.sleep(0.1)
                    
                    yield {'status': 'analysis_complete', 'full_response': generated_text, 'partial': False}
                else:
                    yield {'status': 'analysis_error', 'message': 'Failed to generate analysis'}
                
        except Exception as e:
            logger.error(f"Comprehensive analysis failed for query '{query}': {e}")
            yield {'status': 'analysis_error', 'message': f'Analysis failed: {str(e)}'}

    
    def _parse_comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """Parse the comprehensive LLM response into structured data"""
        try:
            # Simple parsing - extract sections
            sections = {
                "financial_analysis": {"analysis": "Analysis completed", "sources_analyzed": 0},
                "market_trends": {"trends": "Trends analyzed", "sources_analyzed": 0},
                "recommendations": [],
                "risk_factors": [],
                "executive_summary": "Analysis completed successfully"
            }
            
            # Extract recommendations (look for numbered items)
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and any(char.isdigit() for char in line[:3]) and '.' in line:
                    rec_text = line.split('.', 1)[-1].strip()
                    if rec_text and len(sections["recommendations"]) < 3:
                        sections["recommendations"].append(rec_text)
            
            # Extract risk factors
            risk_section = False
            current_risk = {}
            for line in lines:
                line = line.strip()
                if 'RISK ASSESSMENT:' in line:
                    risk_section = True
                    continue
                elif risk_section and 'Risk Type:' in line:
                    if current_risk:
                        sections["risk_factors"].append(current_risk)
                    current_risk = {'risk_type': line.split(':', 1)[-1].strip()}
                elif risk_section and 'Severity:' in line:
                    current_risk['severity'] = line.split(':', 1)[-1].strip()
                elif risk_section and 'Description:' in line:
                    current_risk['description'] = line.split(':', 1)[-1].strip()
                elif risk_section and 'Mitigation:' in line:
                    current_risk['mitigation'] = line.split(':', 1)[-1].strip()
            
            # Add the last risk factor
            if current_risk:
                sections["risk_factors"].append(current_risk)
            
            # Extract executive summary
            summary_section = False
            summary_lines = []
            for line in lines:
                if 'EXECUTIVE SUMMARY:' in line:
                    summary_section = True
                    continue
                elif summary_section and line and not line.startswith(('ANALYSIS:', 'RISK ASSESSMENT:')):
                    summary_lines.append(line)
                elif summary_section and line.startswith(('ANALYSIS:', 'RISK ASSESSMENT:')):
                    break
            
            if summary_lines:
                sections["executive_summary"] = ' '.join(summary_lines).strip()
            
            return sections
            
        except Exception as e:
            logger.error(f"Error parsing comprehensive analysis: {e}")
            return self._get_fallback_comprehensive_analysis()
    
    def _get_fallback_comprehensive_analysis(self) -> Dict[str, Any]:
        """Fallback analysis if LLM fails"""
        return {
            "financial_analysis": {"analysis": "Financial analysis completed", "sources_analyzed": 0},
            "market_trends": {"trends": "Market trends analyzed", "sources_analyzed": 0},
            "recommendations": [
                "Consider diversifying your investment portfolio",
                "Monitor market conditions regularly",
                "Consult with financial professionals for personalized advice"
            ],
            "risk_factors": [
                {
                    "risk_type": "Market Risk",
                    "severity": "Medium",
                    "description": "Market volatility may impact performance",
                    "mitigation": "Diversification and risk management strategies"
                }
            ],
            "executive_summary": "Research analysis completed with key insights and recommendations for informed decision-making."
        }
    
    async def _generate_pdf_report_async(self, query: str, sources: List[Any]) -> str:
        """
        Generate a downloadable PDF report asynchronously
        
        Args:
            query: Research query
            sources: List of sources (web and vector)
            
        Returns:
            Path to generated PDF file
        """
        try:
            logger.info("Generating PDF report asynchronously...")
            
            # Create reports directory if it doesn't exist
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate filename with .txt extension for now (since we're in mock mode)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{reports_dir}/financial_research_{timestamp}.txt"
            
            # Mock PDF generation (replace with real implementation)
            await asyncio.sleep(0.5)
            
            # Create a comprehensive text report
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("FINANCIAL RESEARCH REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Research Query: {query}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                
                # Sources
                f.write("SOURCES CONSULTED\n")
                f.write("-" * 20 + "\n")
                for i, source in enumerate(sources, 1):
                    f.write(f"{i}. {source.get('source', 'Unknown')}\n")
                    f.write(f"   Title: {source.get('title', 'N/A')}\n")
                    f.write(f"   URL: {source.get('url', 'N/A')}\n")
                    f.write(f"   Relevance: {source.get('relevance_score', 'N/A')}\n\n")
                
                # Findings
                f.write("DETAILED FINDINGS\n")
                f.write("-" * 20 + "\n")
                for i, finding in enumerate(sources, 1): # Use 'sources' for findings
                    f.write(f"{i}. {str(finding)}\n\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 20 + "\n")
                for i, rec in enumerate(self._get_fallback_recommendations(), 1): # Use fallback recommendations
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
                
                # Risk Factors
                f.write("RISK ASSESSMENT\n")
                f.write("-" * 20 + "\n")
                for i, risk in enumerate(self._get_fallback_risks(), 1): # Use fallback risks
                    f.write(f"{i}. {risk.get('risk_type', 'Unknown Risk')}\n")
                    f.write(f"   Severity: {risk.get('severity', 'N/A')}\n")
                    f.write(f"   Description: {risk.get('description', 'N/A')}\n")
                    f.write(f"   Mitigation: {risk.get('mitigation', 'N/A')}\n\n")
                
                f.write("\n" + "=" * 50 + "\n")
                f.write("Report generated by Financial Research Chatbot\n")
                f.write("This is a mock report for demonstration purposes\n")
            
            logger.info(f"Report generated: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return ""
    
    def _get_error_response(self, query: str, error: str) -> Dict[str, Any]:
        """Generate error response when research fails"""
        return {
            "query": query,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "status": "failed"
        }
    
    async def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent"""
        return {
            "name": self.name,
            "specialization": self.specialization,
            "version": self.version,
            "capabilities": self.research_methods,
            "report_templates": list(self.report_templates.keys()),
            "last_updated": datetime.now().isoformat()
        }

    def _get_fallback_recommendations(self) -> List[str]:
        """Fallback recommendations if LLM fails"""
        return [
            "Consider diversifying across multiple asset classes",
            "Monitor key support and resistance levels",
            "Implement risk management strategies",
            "Stay informed about market developments",
            "Consult with qualified financial professionals for personalized advice"
        ]
    
    def _get_fallback_risks(self) -> List[Dict[str, Any]]:
        """Fallback risk assessment if LLM fails"""
        return [
            {
                "risk_type": "Market Risk",
                "severity": "Medium",
                "description": "Market volatility may impact performance",
                "mitigation": "Diversification and hedging strategies"
            },
            {
                "risk_type": "Regulatory Risk",
                "severity": "Low",
                "description": "Changes in regulations may affect operations",
                "mitigation": "Stay updated on regulatory changes"
            }
        ]
    
    async def generate_pdf_report(self, research_results: Dict[str, Any], report_type: str = "comprehensive") -> str:
        """Legacy method for backward compatibility"""
        return await self._generate_pdf_report_async(
            research_results.get('query', 'Unknown Query'),
            research_results.get('sources', [])
        )

    def _create_smart_chunks(self, text: str) -> List[str]:
        """Create natural, readable chunks instead of single words"""
        import re
        
        # Split into sentences first
        sentences = re.split(r'[.!?]+\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would make chunk too long, yield current chunk
            if len(current_chunk) + len(sentence) > 150:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # If single sentence is too long, split it further
                if len(sentence) > 150:
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 > 150:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                                temp_chunk = word
                            else:
                                temp_chunk = word
                        else:
                            temp_chunk += " " + word if temp_chunk else word
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for natural chunking"""
        import re
        
        # Split on sentence endings (., !, ?) followed by space or newline
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Clean up and filter empty sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Only include meaningful sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def _summarize_sources(self, sources: List[Any]) -> str:
        """Summarize sources for context injection"""
        if not sources:
            return "No sources available"
        
        try:
            # Extract key information from sources
            summaries = []
            for i, source in enumerate(sources[:5]):  # Limit to 5 sources
                title = source.get('title', 'Unknown')
                snippet = source.get('snippet', '')
                if snippet:
                    # Take first 100 characters of snippet
                    summary = f"{title}: {snippet[:100]}..."
                    summaries.append(summary)
            
            return " | ".join(summaries) if summaries else f"{len(sources)} sources analyzed"
            
        except Exception as e:
            logger.warning(f"Failed to summarize sources: {e}")
            return f"{len(sources)} sources analyzed"

    def _detect_agent_type(self, query: str) -> str:
        """Detect whether to use finance or banking agent based on query content"""
        query_lower = query.lower()
        
        # Banking-related keywords
        banking_keywords = [
            'bank', 'loan', 'credit', 'mortgage', 'savings', 'checking', 'debit',
            'interest rate', 'apr', 'credit score', 'credit card', 'debt',
            'refinance', 'home loan', 'car loan', 'personal loan', 'business loan',
            'investment account', 'retirement account', 'cd', 'certificate of deposit',
            'money market', 'overdraft', 'wire transfer', 'ach', 'direct deposit'
        ]
        
        # Finance/investment keywords
        finance_keywords = [
            'stock', 'bond', 'mutual fund', 'etf', 'portfolio', 'diversification',
            'market', 'trading', 'investment', 'dividend', 'capital gains',
            'retirement planning', '401k', 'ira', 'roth', 'tax planning',
            'estate planning', 'insurance', 'annuity', 'real estate investment',
            'cryptocurrency', 'forex', 'commodities', 'options', 'futures'
        ]
        
        # Count matches for each category
        banking_score = sum(1 for keyword in banking_keywords if keyword in query_lower)
        finance_score = sum(1 for keyword in finance_keywords if keyword in query_lower)
        
        # Determine agent type
        if banking_score > finance_score:
            return "banking"
        elif finance_score > banking_score:
            return "finance"
        else:
            # If equal or no clear match, use finance as default
            return "finance"
