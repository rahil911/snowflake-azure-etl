"""
Context Extractor for Data Intelligence Agent

This module handles context extraction and management for business intelligence queries,
maintaining conversation history, business context, and data context to enhance
query understanding and response generation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, validator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from shared.schemas.data_models import BusinessEntity, DataQualityMetric
from shared.schemas.agent_communication import Intent, EntityExtraction
from shared.utils.caching import get_cache_manager
from shared.utils.validation import ValidationHelper
from shared.utils.retry import RetryStrategy, retry_on_exception
from shared.config.settings import Settings


logger = logging.getLogger(__name__)


class ContextType(Enum):
    """Types of context that can be extracted."""
    CONVERSATION = "conversation"
    BUSINESS = "business"
    DATA = "data"
    TEMPORAL = "temporal"
    DOMAIN = "domain"


class ContextScope(Enum):
    """Scope of context relevance."""
    SESSION = "session"
    USER = "user"
    DOMAIN = "domain"
    GLOBAL = "global"


@dataclass
class ContextItem:
    """Individual context item with metadata."""
    key: str
    value: Any
    context_type: ContextType
    scope: ContextScope
    timestamp: datetime
    confidence: float = 1.0
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, ttl_seconds: int = 3600) -> bool:
        """Check if context item has expired."""
        return (datetime.now() - self.timestamp).total_seconds() > ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "context_type": self.context_type.value,
            "scope": self.scope.value,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata
        }


class ConversationContext(BaseModel):
    """Conversation context tracking."""
    session_id: str
    user_id: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    entities_mentioned: Set[str] = Field(default_factory=set)
    topics_discussed: List[str] = Field(default_factory=list)
    current_intent: Optional[str] = None
    last_query_type: Optional[str] = None
    active_filters: Dict[str, Any] = Field(default_factory=dict)
    data_sources_used: Set[str] = Field(default_factory=set)
    
    class Config:
        validate_assignment = True


class BusinessContext(BaseModel):
    """Business domain context."""
    industry: Optional[str] = None
    department: Optional[str] = None
    role: Optional[str] = None
    business_units: List[str] = Field(default_factory=list)
    kpis_of_interest: List[str] = Field(default_factory=list)
    reporting_periods: List[str] = Field(default_factory=list)
    business_rules: Dict[str, Any] = Field(default_factory=dict)
    domain_vocabulary: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True


class DataContext(BaseModel):
    """Data context and schema information."""
    available_tables: List[str] = Field(default_factory=list)
    table_relationships: Dict[str, List[str]] = Field(default_factory=dict)
    column_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    data_quality_status: Dict[str, str] = Field(default_factory=dict)
    recent_data_updates: List[str] = Field(default_factory=list)
    data_freshness: Dict[str, datetime] = Field(default_factory=dict)
    user_permissions: Set[str] = Field(default_factory=set)
    
    class Config:
        validate_assignment = True


class ContextExtractionRequest(BaseModel):
    """Request for context extraction."""
    message: str
    session_id: str
    user_id: Optional[str] = None
    message_history: List[Dict[str, Any]] = Field(default_factory=list)
    extract_types: List[ContextType] = Field(default_factory=lambda: list(ContextType))
    include_metadata: bool = True
    
    class Config:
        validate_assignment = True


class ContextExtractionResponse(BaseModel):
    """Response from context extraction."""
    extracted_context: Dict[str, Any]
    confidence_scores: Dict[str, float]
    context_summary: str
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True


class ContextExtractor:
    """
    Advanced context extraction and management for business intelligence queries.
    
    This class handles:
    - Conversation context tracking
    - Business domain context
    - Data schema and availability context
    - Temporal context extraction
    - Context persistence and retrieval
    """
    
    def __init__(self, settings: Settings):
        """Initialize the context extractor."""
        self.settings = settings
        self.cache = get_cache_manager()
        self.validator = ValidationHelper()
        self.retry_strategy = RetryStrategy.EXPONENTIAL
        
        # Context storage
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.business_contexts: Dict[str, BusinessContext] = {}
        self.data_context = DataContext()
        self.context_items: Dict[str, List[ContextItem]] = {}
        
        # Context extraction patterns
        self._setup_extraction_patterns()
        
        # Business vocabulary and mappings
        self._load_business_vocabulary()
        
        logger.info("Context extractor initialized")
    
    def _setup_extraction_patterns(self):
        """Setup regex patterns for context extraction."""
        self.patterns = {
            'temporal': {
                'date_ranges': re.compile(r'(last|past|previous|next)\s+(\d+)\s+(day|week|month|quarter|year)s?', re.IGNORECASE),
                'specific_dates': re.compile(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', re.IGNORECASE),
                'relative_time': re.compile(r'(yesterday|today|tomorrow|this\s+week|last\s+week|this\s+month|last\s+month)', re.IGNORECASE),
                'quarters': re.compile(r'Q[1-4]\s+\d{4}', re.IGNORECASE),
                'fiscal_years': re.compile(r'FY\s*\d{4}', re.IGNORECASE)
            },
            'business_entities': {
                'departments': re.compile(r'(sales|marketing|finance|operations|hr|it|customer\s+service)', re.IGNORECASE),
                'metrics': re.compile(r'(revenue|profit|sales|cost|margin|roi|customer\s+satisfaction|churn)', re.IGNORECASE),
                'regions': re.compile(r'(north|south|east|west|central|northeast|northwest|southeast|southwest)', re.IGNORECASE),
                'products': re.compile(r'(product|service|offering|solution)', re.IGNORECASE)
            },
            'data_operations': {
                'aggregations': re.compile(r'(sum|count|average|total|maximum|minimum|median)', re.IGNORECASE),
                'comparisons': re.compile(r'(compare|versus|vs|against|between)', re.IGNORECASE),
                'trends': re.compile(r'(trend|trending|growth|decline|increase|decrease)', re.IGNORECASE),
                'filters': re.compile(r'(where|filter|only|exclude|include)', re.IGNORECASE)
            },
            'intent_indicators': {
                'question_words': re.compile(r'^(what|how|when|where|why|which|who)', re.IGNORECASE),
                'action_words': re.compile(r'(show|display|find|get|calculate|analyze)', re.IGNORECASE),
                'conditional': re.compile(r'(if|when|unless|provided)', re.IGNORECASE)
            }
        }
    
    def _load_business_vocabulary(self):
        """Load business vocabulary and domain mappings."""
        self.business_vocabulary = {
            'synonyms': {
                'revenue': ['sales', 'income', 'earnings', 'turnover'],
                'profit': ['margin', 'earnings', 'net_income'],
                'customer': ['client', 'account', 'buyer'],
                'product': ['item', 'service', 'offering'],
                'region': ['territory', 'area', 'zone', 'market']
            },
            'abbreviations': {
                'roi': 'return_on_investment',
                'ltv': 'lifetime_value',
                'cac': 'customer_acquisition_cost',
                'mrr': 'monthly_recurring_revenue',
                'arr': 'annual_recurring_revenue'
            },
            'business_terms': {
                'kpis': ['revenue', 'profit_margin', 'customer_satisfaction', 'churn_rate', 'conversion_rate'],
                'periods': ['daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
                'operations': ['sales', 'marketing', 'support', 'operations', 'finance']
            }
        }
    
    async def extract_context(self, request: ContextExtractionRequest) -> ContextExtractionResponse:
        """
        Extract comprehensive context from a message and conversation history.
        
        Args:
            request: Context extraction request
            
        Returns:
            Extracted context with confidence scores and recommendations
        """
        try:
            start_time = datetime.now()
            logger.info(f"Extracting context for session: {request.session_id}")
            
            # Extract different types of context
            extracted_context = {}
            confidence_scores = {}
            recommendations = []
            warnings = []
            
            for context_type in request.extract_types:
                if context_type == ContextType.CONVERSATION:
                    conv_context, confidence = await self._extract_conversation_context(request)
                    extracted_context['conversation'] = conv_context
                    confidence_scores['conversation'] = confidence
                
                elif context_type == ContextType.BUSINESS:
                    biz_context, confidence = await self._extract_business_context(request)
                    extracted_context['business'] = biz_context
                    confidence_scores['business'] = confidence
                
                elif context_type == ContextType.DATA:
                    data_context, confidence = await self._extract_data_context(request)
                    extracted_context['data'] = data_context
                    confidence_scores['data'] = confidence
                
                elif context_type == ContextType.TEMPORAL:
                    temp_context, confidence = await self._extract_temporal_context(request)
                    extracted_context['temporal'] = temp_context
                    confidence_scores['temporal'] = confidence
                
                elif context_type == ContextType.DOMAIN:
                    domain_context, confidence = await self._extract_domain_context(request)
                    extracted_context['domain'] = domain_context
                    confidence_scores['domain'] = confidence
            
            # Generate context summary
            context_summary = self._generate_context_summary(extracted_context)
            
            # Generate recommendations based on extracted context
            recommendations = self._generate_recommendations(extracted_context, confidence_scores)
            
            # Check for potential issues or warnings
            warnings = self._check_context_warnings(extracted_context, confidence_scores)
            
            # Update context storage
            await self._update_context_storage(request.session_id, extracted_context)
            
            response = ContextExtractionResponse(
                extracted_context=extracted_context,
                confidence_scores=confidence_scores,
                context_summary=context_summary,
                recommendations=recommendations,
                warnings=warnings,
                metadata={
                    'extraction_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'session_id': request.session_id,
                    'context_types_extracted': [ct.value for ct in request.extract_types]
                }
            )
            
            logger.info(f"Context extraction completed for session {request.session_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error in context extraction: {str(e)}")
            raise
    
    async def _extract_conversation_context(self, request: ContextExtractionRequest) -> Tuple[Dict[str, Any], float]:
        """Extract conversation-specific context."""
        try:
            context = {}
            confidence = 0.8
            
            # Get or create conversation context
            conv_context = self.conversation_contexts.get(
                request.session_id, 
                ConversationContext(session_id=request.session_id, user_id=request.user_id)
            )
            
            # Update with current message
            conv_context.messages.append({
                'content': request.message,
                'timestamp': datetime.now().isoformat(),
                'type': 'human'
            })
            
            # Extract entities mentioned in current message
            entities = self._extract_entities_from_text(request.message)
            conv_context.entities_mentioned.update(entities)
            
            # Identify topics being discussed
            topics = self._extract_topics_from_text(request.message)
            conv_context.topics_discussed.extend(topics)
            
            # Analyze message history for patterns
            if request.message_history:
                historical_patterns = self._analyze_message_history(request.message_history)
                context.update(historical_patterns)
            
            # Extract current intent indicators
            intent_indicators = self._extract_intent_indicators(request.message)
            context['intent_indicators'] = intent_indicators
            
            # Context from conversation flow
            context.update({
                'session_length': len(conv_context.messages),
                'entities_mentioned': list(conv_context.entities_mentioned),
                'topics_discussed': conv_context.topics_discussed[-5:],  # Last 5 topics
                'message_type': self._classify_message_type(request.message),
                'conversation_stage': self._determine_conversation_stage(conv_context)
            })
            
            self.conversation_contexts[request.session_id] = conv_context
            
            return context, confidence
            
        except Exception as e:
            logger.error(f"Error extracting conversation context: {str(e)}")
            return {}, 0.0
    
    async def _extract_business_context(self, request: ContextExtractionRequest) -> Tuple[Dict[str, Any], float]:
        """Extract business domain context."""
        try:
            context = {}
            confidence = 0.7
            
            # Extract business entities
            business_entities = self._extract_business_entities(request.message)
            context['business_entities'] = business_entities
            
            # Extract KPI mentions
            kpis = self._extract_kpi_mentions(request.message)
            context['kpis_mentioned'] = kpis
            
            # Extract department/function context
            departments = self._extract_department_context(request.message)
            context['departments'] = departments
            
            # Extract business operations context
            operations = self._extract_operations_context(request.message)
            context['operations'] = operations
            
            # Business rules and constraints
            rules = self._extract_business_rules(request.message)
            context['business_rules'] = rules
            
            # Industry-specific context
            industry_context = self._extract_industry_context(request.message)
            context['industry_context'] = industry_context
            
            if any(context.values()):
                confidence = 0.85
            
            return context, confidence
            
        except Exception as e:
            logger.error(f"Error extracting business context: {str(e)}")
            return {}, 0.0
    
    async def _extract_data_context(self, request: ContextExtractionRequest) -> Tuple[Dict[str, Any], float]:
        """Extract data-related context."""
        try:
            context = {}
            confidence = 0.8
            
            # Extract table/data source mentions
            data_sources = self._extract_data_source_mentions(request.message)
            context['data_sources'] = data_sources
            
            # Extract column/field mentions
            columns = self._extract_column_mentions(request.message)
            context['columns_mentioned'] = columns
            
            # Extract aggregation requirements
            aggregations = self._extract_aggregation_context(request.message)
            context['aggregations'] = aggregations
            
            # Extract filter context
            filters = self._extract_filter_context(request.message)
            context['filters'] = filters
            
            # Extract data quality concerns
            quality_concerns = self._extract_quality_concerns(request.message)
            context['quality_concerns'] = quality_concerns
            
            # Data availability context
            availability = await self._check_data_availability(data_sources)
            context['data_availability'] = availability
            
            return context, confidence
            
        except Exception as e:
            logger.error(f"Error extracting data context: {str(e)}")
            return {}, 0.0
    
    async def _extract_temporal_context(self, request: ContextExtractionRequest) -> Tuple[Dict[str, Any], float]:
        """Extract temporal/time-related context."""
        try:
            context = {}
            confidence = 0.9
            
            # Extract date ranges
            date_ranges = self._extract_date_ranges(request.message)
            context['date_ranges'] = date_ranges
            
            # Extract relative time expressions
            relative_time = self._extract_relative_time(request.message)
            context['relative_time'] = relative_time
            
            # Extract fiscal periods
            fiscal_periods = self._extract_fiscal_periods(request.message)
            context['fiscal_periods'] = fiscal_periods
            
            # Extract time granularity
            granularity = self._extract_time_granularity(request.message)
            context['time_granularity'] = granularity
            
            # Extract temporal comparisons
            comparisons = self._extract_temporal_comparisons(request.message)
            context['temporal_comparisons'] = comparisons
            
            return context, confidence
            
        except Exception as e:
            logger.error(f"Error extracting temporal context: {str(e)}")
            return {}, 0.0
    
    async def _extract_domain_context(self, request: ContextExtractionRequest) -> Tuple[Dict[str, Any], float]:
        """Extract domain-specific context."""
        try:
            context = {}
            confidence = 0.7
            
            # Extract domain vocabulary
            domain_terms = self._extract_domain_terms(request.message)
            context['domain_terms'] = domain_terms
            
            # Extract industry-specific patterns
            industry_patterns = self._extract_industry_patterns(request.message)
            context['industry_patterns'] = industry_patterns
            
            # Extract regulatory context
            regulatory_context = self._extract_regulatory_context(request.message)
            context['regulatory_context'] = regulatory_context
            
            # Extract compliance requirements
            compliance = self._extract_compliance_requirements(request.message)
            context['compliance_requirements'] = compliance
            
            return context, confidence
            
        except Exception as e:
            logger.error(f"Error extracting domain context: {str(e)}")
            return {}, 0.0
    
    def _extract_entities_from_text(self, text: str) -> Set[str]:
        """Extract business entities from text."""
        entities = set()
        
        # Use patterns to find entities
        for category, pattern in self.patterns['business_entities'].items():
            matches = pattern.findall(text)
            entities.update(matches)
        
        return entities
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from text."""
        topics = []
        
        # Simple topic extraction based on keywords
        text_lower = text.lower()
        
        business_topics = {
            'sales': ['sales', 'revenue', 'selling'],
            'finance': ['finance', 'budget', 'cost', 'profit'],
            'marketing': ['marketing', 'campaign', 'promotion'],
            'operations': ['operations', 'process', 'efficiency'],
            'customer': ['customer', 'client', 'satisfaction']
        }
        
        for topic, keywords in business_topics.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_intent_indicators(self, text: str) -> Dict[str, List[str]]:
        """Extract intent indicators from text."""
        indicators = {}
        
        for category, pattern in self.patterns['intent_indicators'].items():
            matches = pattern.findall(text)
            if matches:
                indicators[category] = matches
        
        return indicators
    
    def _classify_message_type(self, message: str) -> str:
        """Classify the type of message."""
        message_lower = message.lower().strip()
        
        if message_lower.endswith('?'):
            return 'question'
        elif any(word in message_lower for word in ['show', 'display', 'give', 'provide']):
            return 'request'
        elif any(word in message_lower for word in ['analyze', 'calculate', 'compute']):
            return 'analysis'
        elif any(word in message_lower for word in ['compare', 'contrast', 'versus']):
            return 'comparison'
        else:
            return 'statement'
    
    def _determine_conversation_stage(self, conv_context: ConversationContext) -> str:
        """Determine the current stage of conversation."""
        message_count = len(conv_context.messages)
        
        if message_count <= 1:
            return 'initial'
        elif message_count <= 3:
            return 'clarification'
        elif message_count <= 10:
            return 'exploration'
        else:
            return 'deep_analysis'
    
    def _analyze_message_history(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in message history."""
        patterns = {
            'common_topics': [],
            'recurring_entities': [],
            'preferred_time_periods': [],
            'typical_query_types': []
        }
        
        # Simple analysis of historical patterns
        all_text = ' '.join([msg.get('content', '') for msg in history])
        
        # Extract common entities
        entities = self._extract_entities_from_text(all_text)
        patterns['recurring_entities'] = list(entities)[:5]  # Top 5
        
        # Extract common topics
        topics = self._extract_topics_from_text(all_text)
        patterns['common_topics'] = list(set(topics))
        
        return patterns
    
    def _extract_business_entities(self, text: str) -> List[str]:
        """Extract business entities from text."""
        entities = []
        
        for category, pattern in self.patterns['business_entities'].items():
            matches = pattern.findall(text)
            entities.extend(matches)
        
        return entities
    
    def _extract_kpi_mentions(self, text: str) -> List[str]:
        """Extract KPI mentions from text."""
        kpis = []
        text_lower = text.lower()
        
        for kpi in self.business_vocabulary['business_terms']['kpis']:
            if kpi.replace('_', ' ') in text_lower:
                kpis.append(kpi)
        
        return kpis
    
    def _extract_department_context(self, text: str) -> List[str]:
        """Extract department context from text."""
        departments = []
        matches = self.patterns['business_entities']['departments'].findall(text)
        departments.extend(matches)
        return departments
    
    def _extract_operations_context(self, text: str) -> List[str]:
        """Extract business operations context."""
        operations = []
        matches = self.patterns['data_operations']['aggregations'].findall(text)
        operations.extend(matches)
        return operations
    
    def _extract_business_rules(self, text: str) -> List[str]:
        """Extract business rules and constraints."""
        rules = []
        
        # Simple rule extraction based on conditional patterns
        conditional_matches = self.patterns['intent_indicators']['conditional'].findall(text)
        if conditional_matches:
            rules.extend(conditional_matches)
        
        return rules
    
    def _extract_industry_context(self, text: str) -> Dict[str, Any]:
        """Extract industry-specific context."""
        context = {
            'industry_terms': [],
            'sector_specific': [],
            'regulatory_mentions': []
        }
        
        # This would be expanded based on specific industry knowledge
        return context
    
    def _extract_data_source_mentions(self, text: str) -> List[str]:
        """Extract data source mentions."""
        sources = []
        text_lower = text.lower()
        
        # Common data source patterns
        data_sources = ['sales', 'customers', 'products', 'orders', 'transactions', 'users']
        
        for source in data_sources:
            if source in text_lower:
                sources.append(source)
        
        return sources
    
    def _extract_column_mentions(self, text: str) -> List[str]:
        """Extract column/field mentions."""
        columns = []
        
        # Common column patterns
        column_patterns = ['id', 'name', 'date', 'amount', 'quantity', 'price', 'status']
        text_lower = text.lower()
        
        for pattern in column_patterns:
            if pattern in text_lower:
                columns.append(pattern)
        
        return columns
    
    def _extract_aggregation_context(self, text: str) -> List[str]:
        """Extract aggregation requirements."""
        aggregations = []
        matches = self.patterns['data_operations']['aggregations'].findall(text)
        aggregations.extend(matches)
        return aggregations
    
    def _extract_filter_context(self, text: str) -> List[str]:
        """Extract filter context."""
        filters = []
        matches = self.patterns['data_operations']['filters'].findall(text)
        filters.extend(matches)
        return filters
    
    def _extract_quality_concerns(self, text: str) -> List[str]:
        """Extract data quality concerns."""
        concerns = []
        quality_keywords = ['missing', 'null', 'incomplete', 'invalid', 'duplicate', 'accuracy']
        text_lower = text.lower()
        
        for keyword in quality_keywords:
            if keyword in text_lower:
                concerns.append(keyword)
        
        return concerns
    
    async def _check_data_availability(self, data_sources: List[str]) -> Dict[str, str]:
        """Check availability of requested data sources."""
        availability = {}
        
        for source in data_sources:
            # This would integrate with actual data availability checks
            availability[source] = 'available'  # Simplified
        
        return availability
    
    def _extract_date_ranges(self, text: str) -> List[str]:
        """Extract date ranges from text."""
        ranges = []
        
        # Use temporal patterns
        matches = self.patterns['temporal']['date_ranges'].findall(text)
        ranges.extend([' '.join(match) for match in matches])
        
        specific_dates = self.patterns['temporal']['specific_dates'].findall(text)
        ranges.extend(specific_dates)
        
        return ranges
    
    def _extract_relative_time(self, text: str) -> List[str]:
        """Extract relative time expressions."""
        relative_times = []
        matches = self.patterns['temporal']['relative_time'].findall(text)
        relative_times.extend(matches)
        return relative_times
    
    def _extract_fiscal_periods(self, text: str) -> List[str]:
        """Extract fiscal periods."""
        periods = []
        
        quarter_matches = self.patterns['temporal']['quarters'].findall(text)
        periods.extend(quarter_matches)
        
        fy_matches = self.patterns['temporal']['fiscal_years'].findall(text)
        periods.extend(fy_matches)
        
        return periods
    
    def _extract_time_granularity(self, text: str) -> Optional[str]:
        """Extract time granularity requirements."""
        granularities = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
        text_lower = text.lower()
        
        for granularity in granularities:
            if granularity in text_lower:
                return granularity
        
        return None
    
    def _extract_temporal_comparisons(self, text: str) -> List[str]:
        """Extract temporal comparison requirements."""
        comparisons = []
        
        comparison_patterns = [
            'year over year', 'month over month', 'quarter over quarter',
            'vs last year', 'compared to', 'versus previous'
        ]
        
        text_lower = text.lower()
        for pattern in comparison_patterns:
            if pattern in text_lower:
                comparisons.append(pattern)
        
        return comparisons
    
    def _extract_domain_terms(self, text: str) -> List[str]:
        """Extract domain-specific terms."""
        terms = []
        
        # Extract terms from business vocabulary
        for category, term_list in self.business_vocabulary['business_terms'].items():
            for term in term_list:
                if term.replace('_', ' ') in text.lower():
                    terms.append(term)
        
        return terms
    
    def _extract_industry_patterns(self, text: str) -> List[str]:
        """Extract industry-specific patterns."""
        patterns = []
        # This would be expanded based on specific industry knowledge
        return patterns
    
    def _extract_regulatory_context(self, text: str) -> List[str]:
        """Extract regulatory context."""
        regulatory = []
        
        regulatory_terms = ['compliance', 'regulation', 'audit', 'gdpr', 'hipaa', 'sox']
        text_lower = text.lower()
        
        for term in regulatory_terms:
            if term in text_lower:
                regulatory.append(term)
        
        return regulatory
    
    def _extract_compliance_requirements(self, text: str) -> List[str]:
        """Extract compliance requirements."""
        requirements = []
        
        compliance_patterns = ['pii', 'sensitive', 'confidential', 'restricted', 'authorized']
        text_lower = text.lower()
        
        for pattern in compliance_patterns:
            if pattern in text_lower:
                requirements.append(pattern)
        
        return requirements
    
    def _generate_context_summary(self, extracted_context: Dict[str, Any]) -> str:
        """Generate a summary of extracted context."""
        summary_parts = []
        
        if 'conversation' in extracted_context:
            conv = extracted_context['conversation']
            summary_parts.append(f"Conversation stage: {conv.get('conversation_stage', 'unknown')}")
        
        if 'business' in extracted_context:
            biz = extracted_context['business']
            kpis = biz.get('kpis_mentioned', [])
            if kpis:
                summary_parts.append(f"KPIs mentioned: {', '.join(kpis)}")
        
        if 'temporal' in extracted_context:
            temp = extracted_context['temporal']
            ranges = temp.get('date_ranges', [])
            if ranges:
                summary_parts.append(f"Time periods: {', '.join(ranges)}")
        
        if 'data' in extracted_context:
            data = extracted_context['data']
            sources = data.get('data_sources', [])
            if sources:
                summary_parts.append(f"Data sources: {', '.join(sources)}")
        
        if not summary_parts:
            return "General business intelligence query with minimal specific context."
        
        return "; ".join(summary_parts)
    
    def _generate_recommendations(self, extracted_context: Dict[str, Any], confidence_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on extracted context."""
        recommendations = []
        
        # Low confidence recommendations
        for context_type, confidence in confidence_scores.items():
            if confidence < 0.6:
                recommendations.append(f"Consider clarifying {context_type} context for better results")
        
        # Context-specific recommendations
        if 'temporal' in extracted_context:
            temp_context = extracted_context['temporal']
            if not temp_context.get('date_ranges') and not temp_context.get('relative_time'):
                recommendations.append("Consider specifying a time period for more focused analysis")
        
        if 'data' in extracted_context:
            data_context = extracted_context['data']
            if not data_context.get('data_sources'):
                recommendations.append("Specify which data sources or tables to analyze")
        
        return recommendations
    
    def _check_context_warnings(self, extracted_context: Dict[str, Any], confidence_scores: Dict[str, float]) -> List[str]:
        """Check for potential issues or warnings."""
        warnings = []
        
        # Low confidence warnings
        low_confidence_contexts = [ctx for ctx, conf in confidence_scores.items() if conf < 0.5]
        if low_confidence_contexts:
            warnings.append(f"Low confidence in extracting: {', '.join(low_confidence_contexts)}")
        
        # Data availability warnings
        if 'data' in extracted_context:
            data_context = extracted_context['data']
            quality_concerns = data_context.get('quality_concerns', [])
            if quality_concerns:
                warnings.append(f"Potential data quality issues: {', '.join(quality_concerns)}")
        
        # Temporal warnings
        if 'temporal' in extracted_context:
            temp_context = extracted_context['temporal']
            if len(temp_context.get('date_ranges', [])) > 3:
                warnings.append("Multiple time periods specified - results may be complex")
        
        return warnings
    
    async def _update_context_storage(self, session_id: str, extracted_context: Dict[str, Any]):
        """Update persistent context storage."""
        try:
            # Create context items for storage
            timestamp = datetime.now()
            
            for context_type, context_data in extracted_context.items():
                if session_id not in self.context_items:
                    self.context_items[session_id] = []
                
                context_item = ContextItem(
                    key=f"{context_type}_context",
                    value=context_data,
                    context_type=ContextType(context_type),
                    scope=ContextScope.SESSION,
                    timestamp=timestamp,
                    source="context_extractor"
                )
                
                self.context_items[session_id].append(context_item)
            
            # Clean up expired context items
            await self._cleanup_expired_context(session_id)
            
        except Exception as e:
            logger.error(f"Error updating context storage: {str(e)}")
    
    async def _cleanup_expired_context(self, session_id: str, ttl_seconds: int = 3600):
        """Clean up expired context items."""
        if session_id in self.context_items:
            self.context_items[session_id] = [
                item for item in self.context_items[session_id]
                if not item.is_expired(ttl_seconds)
            ]
    
    async def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive context for a session."""
        try:
            context = {}
            
            # Get conversation context
            if session_id in self.conversation_contexts:
                context['conversation'] = self.conversation_contexts[session_id].dict()
            
            # Get business context
            if session_id in self.business_contexts:
                context['business'] = self.business_contexts[session_id].dict()
            
            # Get stored context items
            if session_id in self.context_items:
                context['stored_items'] = [item.to_dict() for item in self.context_items[session_id]]
            
            return context if context else None
            
        except Exception as e:
            logger.error(f"Error getting session context: {str(e)}")
            return None
    
    async def clear_session_context(self, session_id: str):
        """Clear all context for a session."""
        try:
            if session_id in self.conversation_contexts:
                del self.conversation_contexts[session_id]
            
            if session_id in self.business_contexts:
                del self.business_contexts[session_id]
            
            if session_id in self.context_items:
                del self.context_items[session_id]
            
            logger.info(f"Cleared context for session: {session_id}")
            
        except Exception as e:
            logger.error(f"Error clearing session context: {str(e)}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the context extractor."""
        return {
            'status': 'healthy',
            'active_sessions': len(self.conversation_contexts),
            'stored_context_items': sum(len(items) for items in self.context_items.values()),
            'business_vocabulary_loaded': bool(self.business_vocabulary),
            'extraction_patterns_loaded': bool(self.patterns)
        } 