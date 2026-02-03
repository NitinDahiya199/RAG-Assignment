"""
Query Engine Module
Handles user queries, context retrieval, and response generation using Gemini API.
"""

import re
import json
from typing import Dict, List, Optional, Any
import logging

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Google Generative AI not available")

from src.utils import get_api_key, setup_logging
from src.function_calling import ArxivFunctionCalling
from src.conversation_manager import ConversationManager
from src.response_cache import ResponseCache
from src.error_handler import ErrorHandler
from src.rate_limiter import RateLimiter
from src.input_validator import InputValidator

logger = logging.getLogger(__name__)
setup_logging()


class QueryEngine:
    """
    Processes user queries and generates responses using Gemini API.
    Supports:
    - Direct content lookup
    - Summarization
    - Evaluation results extraction
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        vector_store=None,
        model_name: str = "gemini-1.5-flash",
        enable_function_calling: bool = True,
        enable_caching: bool = True,
        enable_rate_limiting: bool = True,
        enable_conversation: bool = True
    ):
        """
        Initialize the QueryEngine with enterprise features.
        
        Args:
            api_key: Google Gemini API key (optional, can be set via env var)
            vector_store: VectorStore instance for context retrieval
            model_name: Gemini model to use (default: gemini-1.5-flash)
            enable_function_calling: Enable Arxiv function calling (default: True)
            enable_caching: Enable response caching (default: True)
            enable_rate_limiting: Enable rate limiting (default: True)
            enable_conversation: Enable conversation history (default: True)
        """
        self.api_key = api_key or get_api_key()
        self.vector_store = vector_store
        self.model_name = model_name
        self.enable_function_calling = enable_function_calling
        self.fallback_models = []  # Initialize fallback models list
        
        # Initialize enterprise features
        self.error_handler = ErrorHandler()
        self.input_validator = InputValidator()
        
        # Initialize caching
        if enable_caching:
            self.cache = ResponseCache(ttl=3600)
        else:
            self.cache = None
        
        # Initialize rate limiting
        if enable_rate_limiting:
            self.rate_limiter = RateLimiter(max_requests=60, time_window=60)
        else:
            self.rate_limiter = None
        
        # Initialize conversation management
        if enable_conversation:
            self.conversation_manager = ConversationManager(max_history=10)
        else:
            self.conversation_manager = None
        
        # Initialize Arxiv function calling
        if enable_function_calling:
            self.arxiv_func = ArxivFunctionCalling()
            self.function_schema = self.arxiv_func.get_function_schema()
        else:
            self.arxiv_func = None
            self.function_schema = None
        
        if not self.api_key:
            logger.warning("Gemini API key not found. Query engine will have limited functionality.")
            self.model = None
        elif GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=self.api_key)
                
                # First, try to list available models
                available_models = []
                try:
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            model_id = m.name.split('/')[-1] if '/' in m.name else m.name
                            available_models.append(model_id)
                            logger.debug(f"Found available model: {model_id}")
                except Exception as list_error:
                    logger.warning(f"Could not list models: {str(list_error)}")
                
                # Try multiple model names as fallback
                fallback_models = [
                    model_name,
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                    "gemini-pro",
                    "gemini-2.5-flash",
                    "gemini-1.5-flash-002",
                    "gemini-1.5-pro-002"
                ]
                
                # If we found available models, prioritize them
                if available_models:
                    # Reorder fallback_models to prioritize available ones
                    prioritized_models = [m for m in available_models if m in fallback_models]
                    prioritized_models.extend([m for m in fallback_models if m not in prioritized_models])
                    fallback_models = prioritized_models
                    logger.info(f"Prioritizing available models: {prioritized_models[:3]}")
                
                model_initialized = False
                self.fallback_models = []  # Store fallback model instances
                
                for model_to_try in fallback_models:
                    try:
                        # Initialize model with function calling if enabled
                        if enable_function_calling and self.function_schema:
                            test_model = genai.GenerativeModel(
                                model_to_try,
                                tools=[self.function_schema]
                            )
                        else:
                            test_model = genai.GenerativeModel(model_to_try)
                        
                        # Store first successfully initialized model as primary
                        if not model_initialized:
                            self.model = test_model
                            self.model_name = model_to_try
                            logger.info(f"Initialized QueryEngine with model: {model_to_try}")
                            model_initialized = True
                        else:
                            # Store additional models as fallbacks
                            self.fallback_models.append(test_model)
                            logger.debug(f"Added fallback model: {model_to_try}")
                    except Exception as model_error:
                        logger.debug(f"Failed to initialize model {model_to_try}: {str(model_error)}")
                        continue
                
                if not model_initialized:
                    logger.error(f"Failed to initialize any of the attempted models: {fallback_models}")
                    self.model = None
                    self.fallback_models = []
                else:
                    # Initialize fallback_models list if not already done
                    if not hasattr(self, 'fallback_models'):
                        self.fallback_models = []
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {str(e)}")
                self.model = None
        else:
            self.model = None
            logger.warning("Google Generative AI package not available")
    
    def query(
        self,
        question: str,
        context: Optional[List[Dict]] = None,
        top_k: int = 5,
        user_id: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict:
        """
        Process a user query and generate a response with enterprise features.
        
        Args:
            question: User's question
            context: Optional context documents from vector search
            top_k: Number of context chunks to retrieve if context not provided
            user_id: Optional user identifier for rate limiting and conversation tracking
            use_cache: Whether to use response cache
            
        Returns:
            Dictionary containing answer and metadata
        """
        # Input validation
        is_valid, error_msg = self.input_validator.validate_query(question)
        if not is_valid:
            error_response = self.error_handler.handle_query_error(
                ValueError(error_msg),
                question
            )
            return self.error_handler.create_safe_response(error_response)
        
        # Sanitize query
        question = self.input_validator.sanitize_query(question)
        
        # Rate limiting
        if self.rate_limiter:
            is_allowed, rate_limit_msg = self.rate_limiter.is_allowed(user_id, "query")
            if not is_allowed:
                return {
                    "answer": rate_limit_msg,
                    "sources": [],
                    "confidence": 0.0,
                    "rate_limited": True
                }
        
        # Check cache
        if use_cache and self.cache:
            cached_response = self.cache.get(question)
            if cached_response:
                logger.info("Returning cached response")
                return cached_response
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Get conversation context if available
            conversation_context = None
            if self.conversation_manager:
                conversation_context = self.conversation_manager.get_relevant_context(question)
            
            # Classify query intent
            intent = self._classify_intent(question)
            logger.info(f"Detected intent: {intent}")
            
            # Retrieve context if not provided
            if context is None and self.vector_store:
                context = self._retrieve_context(question, top_k=top_k)
            
            # Generate response based on intent
            if intent == "direct_lookup":
                response = self._handle_direct_lookup(question, context)
            elif intent == "summarization":
                response = self._handle_summarization(question, context)
            elif intent == "metrics_extraction":
                response = self._handle_metrics_extraction(question, context)
            else:
                response = self._handle_general_query(question, context)
            
            # Add conversation context if available
            if conversation_context and "answer" in response:
                # Could enhance response with conversation context
                pass
            
            # Cache response
            if use_cache and self.cache:
                self.cache.set(question, response)
            
            # Add to conversation history
            if self.conversation_manager:
                self.conversation_manager.add_turn(question, response, {"user_id": user_id})
            
            return response
            
        except Exception as e:
            # Handle errors gracefully
            error_response = self.error_handler.handle_query_error(e, question)
            return self.error_handler.create_safe_response(error_response)
    
    def _classify_intent(self, query: str) -> str:
        """
        Classify the intent of the query.
        
        Args:
            query: User's query string
            
        Returns:
            Intent type: "direct_lookup", "summarization", "metrics_extraction", or "general"
        """
        query_lower = query.lower()
        
        # Direct lookup patterns
        lookup_patterns = [
            r"what is (?:the )?(?:conclusion|result|finding|outcome)",
            r"what (?:does|did) (?:the )?(?:paper|document|study)",
            r"what (?:is|are) (?:the )?(?:conclusion|result|finding)"
        ]
        for pattern in lookup_patterns:
            if re.search(pattern, query_lower):
                return "direct_lookup"
        
        # Summarization patterns
        summarization_patterns = [
            r"summarize",
            r"summary of",
            r"brief overview",
            r"key points",
            r"main (?:points|ideas|findings)"
        ]
        for pattern in summarization_patterns:
            if re.search(pattern, query_lower):
                return "summarization"
        
        # Metrics extraction patterns
        metrics_patterns = [
            r"(?:accuracy|f1[-\s]?score|precision|recall|f-measure)",
            r"what (?:are|is) (?:the )?(?:accuracy|f1|precision|recall)",
            r"(?:performance|evaluation) (?:metrics|results)"
        ]
        for pattern in metrics_patterns:
            if re.search(pattern, query_lower):
                return "metrics_extraction"
        
        return "general"
    
    def _retrieve_context(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant context from vector store.
        
        Args:
            query: Search query
            document_id: Optional document filter
            top_k: Number of results to retrieve
            
        Returns:
            List of context documents
        """
        if not self.vector_store:
            logger.warning("Vector store not available for context retrieval")
            return []
        
        try:
            filter_dict = {"document_id": document_id} if document_id else None
            results = self.vector_store.search(query, top_k=top_k, filter=filter_dict)
            return results
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def _handle_direct_lookup(
        self,
        question: str,
        context: List[Dict]
    ) -> Dict:
        """Handle direct content lookup queries."""
        if not context:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Build context string
        context_text = self._build_context_string(context, max_chunks=3)
        
        # Generate prompt
        prompt = f"""Based on the following document content, provide a direct and concise answer to the question.

Question: {question}

Document Content:
{context_text}

Instructions:
- Provide a direct, factual answer
- Be concise (2-3 sentences maximum)
- Only use information from the provided content
- If the answer is not in the content, say "The information is not available in the provided content."

Answer:"""
        
        # Generate response
        answer = self._generate_with_gemini(prompt)
        
        # Format response
        return self._format_response(answer, context, "direct_lookup")
    
    def _handle_summarization(
        self,
        question: str,
        context: List[Dict]
    ) -> Dict:
        """Handle summarization queries."""
        if not context:
            return {
                "answer": "I couldn't find relevant content to summarize.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Extract topic from question
        topic = self._extract_topic_from_query(question)
        
        # Build context string (use more chunks for summarization)
        context_text = self._build_context_string(context, max_chunks=10)
        
        # Generate prompt
        prompt = f"""Summarize the following content about {topic}.

Content:
{context_text}

Instructions:
- Provide a comprehensive summary (3-5 paragraphs)
- Include main points and key findings
- Organize information logically
- Highlight important conclusions

Summary:"""
        
        # Generate response
        answer = self._generate_with_gemini(prompt)
        
        # Format response
        return self._format_response(answer, context, "summarization")
    
    def _handle_metrics_extraction(
        self,
        question: str,
        context: List[Dict]
    ) -> Dict:
        """Handle metrics extraction queries."""
        if not context:
            return {
                "answer": "I couldn't find relevant content with metrics.",
                "sources": [],
                "confidence": 0.0,
                "metrics": {}
            }
        
        # Build context string
        context_text = self._build_context_string(context, max_chunks=5)
        
        # Extract metric names from question
        metrics_to_find = self._extract_metric_names(question)
        
        # Generate prompt
        prompt = f"""Extract evaluation metrics from the following content.

Question: {question}
Metrics to find: {', '.join(metrics_to_find) if metrics_to_find else 'all evaluation metrics'}

Content:
{context_text}

Instructions:
- Extract all evaluation metrics (accuracy, F1-score, precision, recall, etc.)
- Return results in JSON format
- Include metric name, value, and unit if applicable
- If a metric is not found, set its value to null

Format your response as JSON:
{{
  "accuracy": <value or null>,
  "f1_score": <value or null>,
  "precision": <value or null>,
  "recall": <value or null>,
  "other_metrics": {{}}
}}

JSON Response:"""
        
        # Generate response
        answer = self._generate_with_gemini(prompt)
        
        # Parse JSON from response
        metrics = self._parse_metrics_json(answer)
        
        # Format response
        response = self._format_response(answer, context, "metrics_extraction")
        response["metrics"] = metrics
        
        return response
    
    def _handle_general_query(
        self,
        question: str,
        context: List[Dict]
    ) -> Dict:
        """Handle general queries."""
        if not context:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Build context string
        context_text = self._build_context_string(context, max_chunks=5)
        
        # Generate prompt
        prompt = f"""Answer the following question based on the provided document content.

Question: {question}

Document Content:
{context_text}

Instructions:
- Provide a clear and informative answer
- Use only information from the provided content
- If the answer is not in the content, say so
- Include relevant details and context

Answer:"""
        
        # Generate response
        answer = self._generate_with_gemini(prompt)
        
        # Format response
        return self._format_response(answer, context, "general")
    
    def _build_context_string(
        self,
        context: List[Dict],
        max_chunks: int = 5
    ) -> str:
        """Build a formatted context string from retrieved chunks."""
        if not context:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(context[:max_chunks], 1):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            section = metadata.get("section", "Unknown Section")
            doc_id = metadata.get("document_id", "Unknown Document")
            
            context_parts.append(
                f"[Chunk {i} from {doc_id}, Section: {section}]\n{content}\n"
            )
        
        return "\n".join(context_parts)
    
    def _generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Gemini API with runtime fallback."""
        if not self.model and not (hasattr(self, 'fallback_models') and self.fallback_models):
            return "Error: Gemini API is not available. Please configure your API key."
        
        # Try primary model first
        models_to_try = []
        if self.model:
            models_to_try.append(self.model)
        if hasattr(self, 'fallback_models') and self.fallback_models:
            models_to_try.extend(self.fallback_models)
        
        last_error = None
        for model_instance in models_to_try:
            try:
                response = model_instance.generate_content(prompt)
                answer = response.text if hasattr(response, 'text') else str(response)
                # If successful, update primary model if we used a fallback
                if model_instance != self.model:
                    logger.info(f"Switched to fallback model that works")
                    self.model = model_instance
                return answer
            except Exception as e:
                last_error = e
                logger.debug(f"Model failed: {str(e)}")
                continue
        
        # All stored models failed, try creating new ones without function calling
        if GEMINI_AVAILABLE and self.api_key:
            logger.warning("All stored models failed, trying to create new model instances...")
            emergency_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
            for model_name in emergency_models:
                try:
                    emergency_model = genai.GenerativeModel(model_name)
                    response = emergency_model.generate_content(prompt)
                    answer = response.text if hasattr(response, 'text') else str(response)
                    logger.info(f"Emergency fallback model {model_name} worked!")
                    self.model = emergency_model
                    self.model_name = model_name
                    return answer
                except Exception as e:
                    logger.debug(f"Emergency model {model_name} failed: {str(e)}")
                    continue
        
        # All models failed
        error_msg = str(last_error) if last_error else "Unknown error"
        logger.error(f"Error generating response with Gemini: {error_msg}")
        return f"Error generating response: {error_msg}"
    
    def _format_response(
        self,
        answer: str,
        context: List[Dict],
        intent: str
    ) -> Dict:
        """Format response with citations and metadata."""
        # Extract sources
        sources = []
        seen_docs = set()
        
        for chunk in context:
            metadata = chunk.get("metadata", {})
            doc_id = metadata.get("document_id", "Unknown")
            section = metadata.get("section", "Unknown Section")
            
            source_key = f"{doc_id}_{section}"
            if source_key not in seen_docs:
                sources.append({
                    "document": doc_id,
                    "section": section,
                    "relevance_score": chunk.get("score", 0.0)
                })
                seen_docs.add(source_key)
        
        # Calculate confidence based on context quality
        confidence = self._calculate_confidence(context, intent)
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "intent": intent,
            "context_chunks_used": len(context)
        }
    
    def _calculate_confidence(
        self,
        context: List[Dict],
        intent: str
    ) -> float:
        """Calculate confidence score based on context quality."""
        if not context:
            return 0.0
        
        # Average relevance scores
        scores = [chunk.get("score", 0.0) for chunk in context]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Adjust based on number of chunks
        chunk_bonus = min(len(context) / 5.0, 0.2)  # Up to 0.2 bonus
        
        confidence = min(avg_score + chunk_bonus, 1.0)
        return round(confidence, 2)
    
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract topic from summarization query."""
        # Remove common summarization words
        query = re.sub(r"(?:summarize|summary of|brief overview of)\s*", "", query, flags=re.IGNORECASE)
        query = re.sub(r"(?:the|a|an)\s+", "", query, flags=re.IGNORECASE)
        return query.strip() or "the content"
    
    def _extract_metric_names(self, query: str) -> List[str]:
        """Extract metric names from query."""
        metrics = []
        metric_patterns = {
            "accuracy": r"accuracy",
            "f1_score": r"f1[-\s]?score|f-?1",
            "precision": r"precision",
            "recall": r"recall",
            "f_measure": r"f-measure|f measure"
        }
        
        query_lower = query.lower()
        for metric_name, pattern in metric_patterns.items():
            if re.search(pattern, query_lower):
                metrics.append(metric_name)
        
        return metrics if metrics else ["all"]
    
    def _parse_metrics_json(self, text: str) -> Dict:
        """Parse JSON metrics from response text."""
        try:
            # Try to extract JSON from text
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                metrics = json.loads(json_str)
                return metrics
        except json.JSONDecodeError:
            pass
        
        # Fallback: try to extract metrics manually
        metrics = {}
        patterns = {
            "accuracy": r"accuracy[:\s]+([\d.]+)",
            "f1_score": r"f1[-\s]?score[:\s]+([\d.]+)",
            "precision": r"precision[:\s]+([\d.]+)",
            "recall": r"recall[:\s]+([\d.]+)"
        }
        
        text_lower = text.lower()
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                try:
                    metrics[metric_name] = float(match.group(1))
                except ValueError:
                    pass
        
        return metrics if metrics else {}
    
    def direct_lookup(self, question: str, document_id: str) -> Dict:
        """
        Perform direct content lookup for a specific document.
        
        Args:
            question: User's question
            document_id: Identifier of the target document
            
        Returns:
            Dictionary containing answer and source information
        """
        logger.info(f"Direct lookup: {question} in {document_id}")
        
        # Extract section if mentioned
        section = self._extract_section_from_query(question)
        
        # Retrieve context with document filter
        context = self._retrieve_context(question, document_id=document_id, top_k=3)
        
        # Filter by section if specified
        if section and context:
            context = [c for c in context if section.lower() in c.get("metadata", {}).get("section", "").lower()]
        
        return self._handle_direct_lookup(question, context)
    
    def summarize(self, question: str, document_id: str) -> Dict:
        """
        Generate a summary based on the query and document.
        
        Args:
            question: User's summarization request
            document_id: Identifier of the target document
            
        Returns:
            Dictionary containing summary
        """
        logger.info(f"Summarization: {question} for {document_id}")
        
        # Extract topic/section from question
        topic = self._extract_topic_from_query(question)
        
        # Retrieve context
        context = self._retrieve_context(question, document_id=document_id, top_k=10)
        
        return self._handle_summarization(question, context)
    
    def extract_metrics(self, question: str, document_id: str) -> Dict:
        """
        Extract evaluation metrics (accuracy, F1-score, etc.) from a document.
        
        Args:
            question: User's query about metrics
            document_id: Identifier of the target document
            
        Returns:
            Dictionary containing extracted metrics
        """
        logger.info(f"Metrics extraction: {question} from {document_id}")
        
        # Retrieve context
        context = self._retrieve_context(question, document_id=document_id, top_k=5)
        
        return self._handle_metrics_extraction(question, context)
    
    def _extract_section_from_query(self, query: str) -> Optional[str]:
        """Extract section name from query."""
        # Pattern: "conclusion of", "methodology in", etc.
        section_patterns = [
            r"(?:conclusion|methodology|introduction|results|discussion|abstract)",
            r"section\s+([\w\s]+)",
            r"chapter\s+([\w\s]+)"
        ]
        
        query_lower = query.lower()
        for pattern in section_patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1) if match.groups() else match.group(0)
        
        return None
    
    def query_with_functions(
        self,
        question: str,
        context: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Process a query with function calling support.
        Automatically detects if Arxiv search is needed and calls the function.
        
        Args:
            question: User's question
            context: Optional context documents
            
        Returns:
            Dictionary containing answer, function results, and metadata
        """
        if not question:
            return {
                "answer": "Please provide a question.",
                "sources": [],
                "confidence": 0.0
            }
        
        if not self.model or not self.enable_function_calling:
            # Fallback to regular query
            return self.query(question, context)
        
        logger.info(f"Processing query with function calling: {question}")
        
        # Check if query needs Arxiv search
        arxiv_keywords = [
            "find paper", "search paper", "look up paper", "arxiv",
            "find research", "search for paper", "paper about", "papers on"
        ]
        needs_arxiv = any(keyword in question.lower() for keyword in arxiv_keywords)
        
        if needs_arxiv and self.arxiv_func:
            # Use function calling
            try:
                # Generate prompt that may trigger function call
                prompt = f"""User question: {question}

If the user is asking to find, search, or look up academic papers, use the search_arxiv_paper function.
Otherwise, provide a helpful response.

Please help the user with their request."""
                
                response = self.model.generate_content(prompt)
                
                # Check if function was called
                function_calls = []
                arxiv_results = []
                
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content:
                        parts = candidate.content.parts
                        for part in parts:
                            if hasattr(part, 'function_call'):
                                func_call = part.function_call
                                function_calls.append(func_call)
                                
                                # Execute function
                                if func_call.name == "search_arxiv_paper":
                                    args = dict(func_call.args)
                                    func_result = self.arxiv_func.execute_function(
                                        func_call.name,
                                        args
                                    )
                                    arxiv_results = func_result.get("result", [])
                
                # Generate final response with function results
                if arxiv_results:
                    papers_text = self.arxiv_func.format_papers_for_response(arxiv_results)
                    final_prompt = f"""Based on the Arxiv search results, provide a helpful response to the user's question.

User question: {question}

Arxiv Search Results:
{papers_text}

Provide a natural language response summarizing the papers found and their relevance to the query."""
                    
                    final_response = self.model.generate_content(final_prompt)
                    answer = final_response.text if hasattr(final_response, 'text') else str(final_response)
                else:
                    # No function call, use regular response
                    answer = response.text if hasattr(response, 'text') else str(response)
                
                return {
                    "answer": answer,
                    "arxiv_results": arxiv_results,
                    "function_calls": len(function_calls),
                    "sources": [],
                    "confidence": 0.8 if arxiv_results else 0.5
                }
                
            except Exception as e:
                logger.error(f"Error in function calling: {str(e)}")
                # Fallback to regular query
                return self.query(question, context)
        else:
            # Regular query without function calling
            return self.query(question, context)
