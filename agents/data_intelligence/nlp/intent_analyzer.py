"""
Intent Analysis for Business Intelligence Queries

This module analyzes natural language input to understand business intent and extract
relevant entities for data intelligence operations.

Features:
- Intent classification for different types of business questions
- Entity extraction (customers, products, time periods, metrics)
- Context awareness and conversation state management
- Confidence scoring for intent classification
- Business domain understanding

Intent Types:
- data_query: Direct data questions
- data_quality: Data quality analysis requests
- business_insight: Business intelligence insights
- pattern_analysis: Trend and pattern detection
- recommendation: Business recommendations
- comparison: Comparative analysis
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Session A Foundation imports
from shared.config.logging_config import setup_logging
from shared.utils.metrics import track_performance, get_metrics_collector
from shared.utils.caching import cache_result


class IntentType(Enum):
    """Types of business intelligence intents."""
    DATA_QUERY = "data_query"
    DATA_QUALITY = "data_quality"
    BUSINESS_INSIGHT = "business_insight"
    PATTERN_ANALYSIS = "pattern_analysis"
    RECOMMENDATION = "recommendation"
    COMPARISON = "comparison"
    GENERAL = "general"


class EntityType(Enum):
    """Types of entities that can be extracted."""
    METRIC = "metric"
    TIME_PERIOD = "time_period"
    CUSTOMER = "customer"
    PRODUCT = "product"
    STORE = "store"
    CHANNEL = "channel"
    CATEGORY = "category"
    BRAND = "brand"
    LOCATION = "location"
    NUMBER = "number"
    COMPARISON_OPERATOR = "comparison_operator"


@dataclass
class Entity:
    """Extracted entity with metadata."""
    entity_type: EntityType
    value: str
    normalized_value: str
    confidence: float
    span: Tuple[int, int]  # Start and end positions in text
    metadata: Dict[str, Any]


@dataclass
class Intent:
    """Analyzed intent with entities and confidence."""
    intent_type: IntentType
    confidence: float
    entities: List[Entity]
    subintent: Optional[str] = None
    complexity: str = "medium"
    metadata: Dict[str, Any] = None


class IntentAnalyzer:
    """
    Intent Analyzer for Business Intelligence Queries.
    
    Analyzes natural language business questions to understand intent
    and extract relevant entities for data processing.
    """
    
    def __init__(self):
        """Initialize the Intent Analyzer with patterns and mappings."""
        self.logger = setup_logging("nlp.intent_analyzer")
        self.metrics = get_metrics_collector()
        
        # Performance tracking
        self.analysis_counter = self.metrics.counter("intent_analysis_total")
        self.success_counter = self.metrics.counter("intent_analysis_successful")
        self.error_counter = self.metrics.counter("intent_analysis_failed")
        self.analysis_timer = self.metrics.timer("intent_analysis_time")
        
        # Load patterns and mappings
        self._load_intent_patterns()
        self._load_entity_patterns()
        self._load_business_vocabulary()
        
        self.logger.info("Intent Analyzer initialized successfully")
    
    def _load_intent_patterns(self) -> None:
        """Load patterns for intent classification."""
        
        self.intent_patterns = {
            IntentType.DATA_QUERY: [
                # Direct data questions
                r'\b(?:what|show|give|get|find|tell)\s+(?:me\s+)?(?:the\s+)?(?:total|sum|amount|count|number)',
                r'\bhow\s+(?:much|many|long)',
                r'\bwhat\s+(?:is|are|was|were)\s+(?:the|our)',
                r'\bshow\s+(?:me\s+)?(?:all|the|our)',
                r'\blist\s+(?:all|the|our)',
                r'\bdisplay\s+(?:the|our)',
                
                # Sales-specific queries
                r'\b(?:sales|revenue|profit|income|earnings)\b',
                r'\b(?:sold|transactions|orders|purchases)\b',
                r'\btop\s+\d+\s+(?:customers|products|stores)',
                r'\bperformance\s+(?:of|for|by)',
                
                # Time-based queries
                r'\b(?:this|last|previous)\s+(?:year|month|quarter|week|day)',
                r'\b(?:ytd|year.to.date|monthly|quarterly|annually)',
                r'\bfrom\s+\d{4}\s+to\s+\d{4}',
                r'\bin\s+\d{4}',
                r'\bbetween\s+.*\s+and\s+'
            ],
            
            IntentType.DATA_QUALITY: [
                r'\bdata\s+quality\b',
                r'\bquality\s+(?:check|analysis|assessment|issues)',
                r'\bdata\s+(?:validation|integrity|completeness|accuracy)',
                r'\b(?:missing|null|empty|blank)\s+(?:data|values|records)',
                r'\b(?:duplicates|duplicate\s+records)',
                r'\bdata\s+(?:errors|issues|problems)',
                r'\bvalidate\s+(?:the\s+)?data',
                r'\bcheck\s+(?:for\s+)?(?:errors|issues|problems)',
                r'\b(?:clean|cleanse)\s+(?:the\s+)?data'
            ],
            
            IntentType.BUSINESS_INSIGHT: [
                r'\b(?:insights?|analysis|analytics|intelligence)\b',
                r'\bwhy\s+(?:is|are|did|do)',
                r'\bwhat\s+(?:does|do)\s+(?:this|these)\s+(?:mean|indicate|suggest)',
                r'\b(?:understand|explain|interpret)\b',
                r'\b(?:business\s+)?(?:insights?|intelligence|analysis)',
                r'\bkey\s+(?:metrics|indicators|findings)',
                r'\bdrill\s+down',
                r'\broot\s+cause',
                r'\b(?:correlations?|relationships?)\b',
                r'\bimpact\s+(?:of|on)',
                r'\bwhat\s+(?:caused|drives|influences)'
            ],
            
            IntentType.PATTERN_ANALYSIS: [
                r'\b(?:trends?|patterns?|seasonality)\b',
                r'\b(?:trending|growing|declining|increasing|decreasing)\b',
                r'\bover\s+time',
                r'\b(?:forecast|predict|projection)\b',
                r'\b(?:anomalies|outliers|unusual)\b',
                r'\bcycles?\b',
                r'\btime\s+series',
                r'\b(?:growth|decline)\s+(?:rate|pattern)',
                r'\byear\s+over\s+year',
                r'\bmonth\s+over\s+month',
                r'\bcompare\s+.*\s+over\s+time'
            ],
            
            IntentType.RECOMMENDATION: [
                r'\b(?:recommend|suggest|advise|propose)\b',
                r'\bwhat\s+should\s+(?:we|i)',
                r'\bhow\s+(?:can|should)\s+(?:we|i)\s+(?:improve|optimize|increase)',
                r'\b(?:opportunities|improvements?)\b',
                r'\bnext\s+steps?',
                r'\baction\s+(?:plan|items?)',
                r'\b(?:strategy|strategies)\b',
                r'\boptimize',
                r'\bmaximize',
                r'\bminimize',
                r'\bbest\s+(?:practices?|approach)'
            ],
            
            IntentType.COMPARISON: [
                r'\bcompare\s+.*\s+(?:vs|versus|against|with|to)',
                r'\b(?:vs|versus)\b',
                r'\bdifference\s+between',
                r'\bbetter\s+(?:than|performing)',
                r'\bworse\s+(?:than|performing)',
                r'\bhigher\s+(?:than|compared)',
                r'\blower\s+(?:than|compared)',
                r'\brank(?:ing)?\b',
                r'\btop\s+vs\s+bottom',
                r'\bbenchmark'
            ]
        }
    
    def _load_entity_patterns(self) -> None:
        """Load patterns for entity extraction."""
        
        self.entity_patterns = {
            EntityType.METRIC: [
                (r'\b(?:sales|revenue|profit|income|earnings)\b', 'sales_metrics'),
                (r'\b(?:quantity|volume|units|count|number)\b', 'volume_metrics'),
                (r'\b(?:customers?|clients?)\b', 'customer_metrics'),
                (r'\b(?:products?|items?)\b', 'product_metrics'),
                (r'\b(?:transactions?|orders?|purchases?)\b', 'transaction_metrics'),
                (r'\b(?:margin|markup|cost)\b', 'financial_metrics'),
                (r'\b(?:growth|increase|decrease|change)\b', 'change_metrics')
            ],
            
            EntityType.TIME_PERIOD: [
                (r'\b(?:this|current)\s+(?:year|month|quarter|week|day)\b', 'current_period'),
                (r'\b(?:last|previous)\s+(?:year|month|quarter|week|day)\b', 'previous_period'),
                (r'\b(?:next)\s+(?:year|month|quarter|week|day)\b', 'future_period'),
                (r'\b(?:ytd|year.to.date)\b', 'ytd'),
                (r'\b(?:q[1-4]|quarter\s+[1-4])\b', 'quarter'),
                (r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b', 'month'),
                (r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', 'month_abbrev'),
                (r'\b\d{4}\b', 'year'),
                (r'\b(?:20\d{2})\b', 'year_full'),
                (r'\b(?:daily|weekly|monthly|quarterly|annually)\b', 'frequency')
            ],
            
            EntityType.CUSTOMER: [
                (r'\bcustomer\s+(?:id\s+)?["\']?([A-Z0-9]+)["\']?\b', 'customer_id'),
                (r'\bcustomer\s+["\']([^"\']+)["\']', 'customer_name'),
                (r'\bclient\s+["\']([^"\']+)["\']', 'client_name'),
                (r'\btop\s+(\d+)\s+customers?\b', 'top_customers'),
                (r'\bbottom\s+(\d+)\s+customers?\b', 'bottom_customers')
            ],
            
            EntityType.PRODUCT: [
                (r'\bproduct\s+(?:id\s+)?["\']?([A-Z0-9]+)["\']?\b', 'product_id'),
                (r'\bproduct\s+["\']([^"\']+)["\']', 'product_name'),
                (r'\bitem\s+["\']([^"\']+)["\']', 'item_name'),
                (r'\btop\s+(\d+)\s+products?\b', 'top_products'),
                (r'\bbottom\s+(\d+)\s+products?\b', 'bottom_products')
            ],
            
            EntityType.CATEGORY: [
                (r'\bcategory\s+["\']([^"\']+)["\']', 'product_category'),
                (r'\bsubcategory\s+["\']([^"\']+)["\']', 'product_subcategory'),
                (r'\bin\s+(?:the\s+)?([A-Za-z\s]+)\s+category\b', 'category_mention')
            ],
            
            EntityType.BRAND: [
                (r'\bbrand\s+["\']([^"\']+)["\']', 'brand_name'),
                (r'\b([A-Z][a-z]+)\s+brand\b', 'brand_mention')
            ],
            
            EntityType.STORE: [
                (r'\bstore\s+(?:id\s+)?["\']?([A-Z0-9]+)["\']?\b', 'store_id'),
                (r'\bstore\s+["\']([^"\']+)["\']', 'store_name'),
                (r'\blocation\s+["\']([^"\']+)["\']', 'location_name'),
                (r'\btop\s+(\d+)\s+stores?\b', 'top_stores'),
                (r'\bbottom\s+(\d+)\s+stores?\b', 'bottom_stores')
            ],
            
            EntityType.LOCATION: [
                (r'\b(?:in|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:city|state|region)\b', 'location_name'),
                (r'\b([A-Z]{2})\s+state\b', 'state_code'),
                (r'\bcity\s+["\']([^"\']+)["\']', 'city_name'),
                (r'\bstate\s+["\']([^"\']+)["\']', 'state_name'),
                (r'\bregion\s+["\']([^"\']+)["\']', 'region_name')
            ],
            
            EntityType.NUMBER: [
                (r'\btop\s+(\d+)\b', 'top_n'),
                (r'\bbottom\s+(\d+)\b', 'bottom_n'),
                (r'\blast\s+(\d+)\s+(?:days?|weeks?|months?|years?)\b', 'last_n_periods'),
                (r'\bnext\s+(\d+)\s+(?:days?|weeks?|months?|years?)\b', 'next_n_periods'),
                (r'\b(\d+(?:\.\d+)?)\s*%\b', 'percentage'),
                (r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)\b', 'dollar_amount'),
                (r'\b(\d+(?:,\d{3})*)\b', 'number')
            ],
            
            EntityType.COMPARISON_OPERATOR: [
                (r'\b(?:greater|more|higher|above|over)\s+than\b', 'greater_than'),
                (r'\b(?:less|fewer|lower|below|under)\s+than\b', 'less_than'),
                (r'\b(?:equal|equals?|same\s+as)\b', 'equal_to'),
                (r'\bbetween\b', 'between'),
                (r'\b(?:vs|versus)\b', 'versus')
            ]
        }
    
    def _load_business_vocabulary(self) -> None:
        """Load business domain vocabulary for better understanding."""
        
        self.business_vocabulary = {
            # Sales terminology
            'sales_terms': [
                'sales', 'revenue', 'income', 'earnings', 'proceeds', 'turnover',
                'gross sales', 'net sales', 'total sales'
            ],
            
            'profit_terms': [
                'profit', 'margin', 'markup', 'earnings', 'income', 'return',
                'gross profit', 'net profit', 'operating profit'
            ],
            
            'volume_terms': [
                'quantity', 'volume', 'units', 'count', 'number', 'amount',
                'units sold', 'quantity sold', 'volume sold'
            ],
            
            # Time terminology
            'time_terms': [
                'year', 'month', 'quarter', 'week', 'day', 'period',
                'annually', 'monthly', 'quarterly', 'weekly', 'daily',
                'ytd', 'year-to-date', 'mtd', 'month-to-date'
            ],
            
            # Business entities
            'customer_terms': [
                'customer', 'client', 'buyer', 'purchaser', 'consumer', 'account'
            ],
            
            'product_terms': [
                'product', 'item', 'sku', 'merchandise', 'goods', 'article'
            ],
            
            # Business analysis terms
            'analysis_terms': [
                'analysis', 'insight', 'intelligence', 'analytics', 'report',
                'trend', 'pattern', 'forecast', 'prediction', 'projection'
            ],
            
            # Performance terms
            'performance_terms': [
                'performance', 'kpi', 'metric', 'indicator', 'measure',
                'benchmark', 'target', 'goal', 'objective'
            ]
        }
    
    @track_performance(tags={"operation": "analyze_intent"})
    async def analyze_intent(
        self, 
        text: str, 
        context: Dict[str, Any] = None
    ) -> Intent:
        """
        Analyze natural language text to determine intent and extract entities.
        
        Args:
            text: Natural language input
            context: Conversation context and metadata
            
        Returns:
            Intent object with classification and extracted entities
        """
        self.analysis_counter.increment()
        
        try:
            # Normalize text
            normalized_text = self._normalize_text(text)
            
            # Classify intent
            intent_type, intent_confidence = await self._classify_intent(normalized_text, context)
            
            # Extract entities
            entities = await self._extract_entities(text, normalized_text, intent_type)
            
            # Determine complexity
            complexity = self._determine_complexity(normalized_text, entities, intent_type)
            
            # Extract subintent for more specific processing
            subintent = await self._extract_subintent(normalized_text, intent_type, entities)
            
            # Create intent object
            intent = Intent(
                intent_type=intent_type,
                confidence=intent_confidence,
                entities=entities,
                subintent=subintent,
                complexity=complexity,
                metadata={
                    'original_text': text,
                    'normalized_text': normalized_text,
                    'entity_count': len(entities),
                    'context_used': bool(context)
                }
            )
            
            self.success_counter.increment()
            
            self.logger.info(
                f"Intent analyzed successfully",
                extra={
                    "intent_type": intent_type.value,
                    "confidence": intent_confidence,
                    "entity_count": len(entities),
                    "complexity": complexity
                }
            )
            
            return intent
            
        except Exception as e:
            self.error_counter.increment()
            self.logger.error(f"Intent analysis failed: {str(e)}")
            raise
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for better processing."""
        
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove punctuation at the end
        normalized = re.sub(r'[?!.]+$', '', normalized)
        
        # Normalize common contractions
        contractions = {
            "what's": "what is",
            "how's": "how is",
            "where's": "where is",
            "when's": "when is",
            "who's": "who is",
            "won't": "will not",
            "can't": "cannot",
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not"
        }
        
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        return normalized.strip()
    
    async def _classify_intent(
        self, 
        text: str, 
        context: Dict[str, Any] = None
    ) -> Tuple[IntentType, float]:
        """Classify the intent of the input text."""
        
        # Score each intent type
        intent_scores = {}
        
        for intent_type, patterns in self.intent_patterns.items():
            score = 0.0
            pattern_matches = 0
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    pattern_matches += 1
                    # Weight patterns differently based on specificity
                    if len(pattern) > 50:  # More specific patterns
                        score += 0.3
                    elif len(pattern) > 30:  # Medium specificity
                        score += 0.2
                    else:  # General patterns
                        score += 0.1
            
            # Bonus for multiple pattern matches
            if pattern_matches > 1:
                score += 0.1 * (pattern_matches - 1)
            
            # Context-based adjustments
            if context:
                score = self._adjust_score_with_context(score, intent_type, context)
            
            intent_scores[intent_type] = min(score, 1.0)
        
        # Find the best intent
        if not intent_scores or max(intent_scores.values()) < 0.1:
            return IntentType.GENERAL, 0.5
        
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]
        
        return best_intent, confidence
    
    def _adjust_score_with_context(
        self, 
        base_score: float, 
        intent_type: IntentType, 
        context: Dict[str, Any]
    ) -> float:
        """Adjust intent score based on conversation context."""
        
        adjusted_score = base_score
        
        # Previous intent influence
        if 'previous_intent' in context:
            prev_intent = context['previous_intent']
            if prev_intent == intent_type.value:
                adjusted_score += 0.1  # Continuity bonus
        
        # Recent topics influence
        if 'recent_topics' in context:
            topics = context['recent_topics']
            
            if intent_type == IntentType.DATA_QUALITY and 'quality' in topics:
                adjusted_score += 0.15
            elif intent_type == IntentType.PATTERN_ANALYSIS and any(t in topics for t in ['trend', 'pattern', 'time']):
                adjusted_score += 0.15
            elif intent_type == IntentType.RECOMMENDATION and 'recommendation' in topics:
                adjusted_score += 0.15
        
        # User role influence
        if 'user_role' in context:
            role = context['user_role']
            
            if role == 'analyst' and intent_type in [IntentType.PATTERN_ANALYSIS, IntentType.DATA_QUALITY]:
                adjusted_score += 0.1
            elif role == 'executive' and intent_type in [IntentType.BUSINESS_INSIGHT, IntentType.RECOMMENDATION]:
                adjusted_score += 0.1
            elif role == 'manager' and intent_type == IntentType.COMPARISON:
                adjusted_score += 0.1
        
        return min(adjusted_score, 1.0)
    
    async def _extract_entities(
        self, 
        original_text: str, 
        normalized_text: str, 
        intent_type: IntentType
    ) -> List[Entity]:
        """Extract entities from the text."""
        
        entities = []
        
        # Extract entities using patterns
        for entity_type, patterns in self.entity_patterns.items():
            for pattern, subtype in patterns:
                matches = re.finditer(pattern, original_text, re.IGNORECASE)
                
                for match in matches:
                    # Extract the matched value
                    if match.groups():
                        value = match.group(1)  # Use first capture group
                    else:
                        value = match.group(0)  # Use entire match
                    
                    # Normalize the value
                    normalized_value = self._normalize_entity_value(value, entity_type, subtype)
                    
                    # Calculate confidence based on pattern specificity and context
                    confidence = self._calculate_entity_confidence(
                        value, entity_type, subtype, intent_type, normalized_text
                    )
                    
                    # Skip low-confidence entities
                    if confidence < 0.3:
                        continue
                    
                    entity = Entity(
                        entity_type=entity_type,
                        value=value,
                        normalized_value=normalized_value,
                        confidence=confidence,
                        span=(match.start(), match.end()),
                        metadata={
                            'subtype': subtype,
                            'pattern': pattern,
                            'original_match': match.group(0)
                        }
                    )
                    
                    entities.append(entity)
        
        # Remove overlapping entities (keep higher confidence ones)
        entities = self._remove_overlapping_entities(entities)
        
        # Sort by position in text
        entities.sort(key=lambda e: e.span[0])
        
        return entities
    
    def _normalize_entity_value(
        self, 
        value: str, 
        entity_type: EntityType, 
        subtype: str
    ) -> str:
        """Normalize entity values for consistent processing."""
        
        if entity_type == EntityType.TIME_PERIOD:
            # Normalize time periods
            value_lower = value.lower()
            
            # Current time mappings
            if subtype == 'current_period':
                if 'year' in value_lower:
                    return f"current_year_{datetime.now().year}"
                elif 'month' in value_lower:
                    return f"current_month_{datetime.now().strftime('%Y_%m')}"
                elif 'quarter' in value_lower:
                    quarter = (datetime.now().month - 1) // 3 + 1
                    return f"current_quarter_{datetime.now().year}_Q{quarter}"
            
            # Previous time mappings
            elif subtype == 'previous_period':
                if 'year' in value_lower:
                    return f"previous_year_{datetime.now().year - 1}"
                elif 'month' in value_lower:
                    prev_month = datetime.now().replace(day=1) - timedelta(days=1)
                    return f"previous_month_{prev_month.strftime('%Y_%m')}"
            
            # Month names
            elif subtype in ['month', 'month_abbrev']:
                month_map = {
                    'jan': 'january', 'feb': 'february', 'mar': 'march',
                    'apr': 'april', 'may': 'may', 'jun': 'june',
                    'jul': 'july', 'aug': 'august', 'sep': 'september',
                    'oct': 'october', 'nov': 'november', 'dec': 'december'
                }
                return month_map.get(value_lower[:3], value_lower)
        
        elif entity_type == EntityType.NUMBER:
            # Normalize numbers
            if subtype == 'percentage':
                return f"percentage_{value}"
            elif subtype == 'dollar_amount':
                # Remove commas and extract numeric value
                numeric_value = re.sub(r'[,$]', '', value)
                return f"dollar_{numeric_value}"
            elif subtype == 'number':
                # Remove commas
                return re.sub(r',', '', value)
        
        elif entity_type in [EntityType.CUSTOMER, EntityType.PRODUCT, EntityType.STORE]:
            # Normalize names (title case, remove extra spaces)
            return ' '.join(word.capitalize() for word in value.split())
        
        # Default: clean up the value
        return value.strip()
    
    def _calculate_entity_confidence(
        self, 
        value: str, 
        entity_type: EntityType, 
        subtype: str, 
        intent_type: IntentType,
        text: str
    ) -> float:
        """Calculate confidence score for extracted entity."""
        
        base_confidence = 0.7
        
        # Adjust based on entity type specificity
        if entity_type == EntityType.NUMBER and re.match(r'^\d+$', value):
            base_confidence = 0.9  # High confidence for pure numbers
        elif entity_type == EntityType.TIME_PERIOD:
            if subtype in ['year', 'year_full'] and len(value) == 4:
                base_confidence = 0.95  # Very high confidence for 4-digit years
            elif subtype in ['month', 'month_abbrev']:
                base_confidence = 0.9
        
        # Adjust based on intent-entity relevance
        relevant_combinations = {
            IntentType.DATA_QUERY: [EntityType.METRIC, EntityType.TIME_PERIOD, EntityType.NUMBER],
            IntentType.PATTERN_ANALYSIS: [EntityType.TIME_PERIOD, EntityType.METRIC],
            IntentType.COMPARISON: [EntityType.COMPARISON_OPERATOR, EntityType.METRIC],
            IntentType.DATA_QUALITY: [EntityType.METRIC],
            IntentType.BUSINESS_INSIGHT: [EntityType.METRIC, EntityType.TIME_PERIOD]
        }
        
        if intent_type in relevant_combinations:
            if entity_type in relevant_combinations[intent_type]:
                base_confidence += 0.1
        
        # Adjust based on value length and format
        if len(value) < 2:
            base_confidence -= 0.2  # Very short values are less reliable
        elif len(value) > 50:
            base_confidence -= 0.1  # Very long values might be spurious
        
        # Check for proper formatting
        if entity_type == EntityType.TIME_PERIOD:
            if re.match(r'\b(19|20)\d{2}\b', value):  # Valid year format
                base_confidence += 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _remove_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping higher confidence ones."""
        
        if not entities:
            return entities
        
        # Sort by confidence (descending)
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
        
        filtered_entities = []
        used_spans = []
        
        for entity in sorted_entities:
            # Check if this entity overlaps with any already selected entity
            overlaps = False
            for start, end in used_spans:
                if not (entity.span[1] <= start or entity.span[0] >= end):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_entities.append(entity)
                used_spans.append(entity.span)
        
        return filtered_entities
    
    def _determine_complexity(
        self, 
        text: str, 
        entities: List[Entity], 
        intent_type: IntentType
    ) -> str:
        """Determine the complexity of the query."""
        
        complexity_score = 0
        
        # Base complexity by intent type
        intent_complexity = {
            IntentType.GENERAL: 1,
            IntentType.DATA_QUERY: 2,
            IntentType.DATA_QUALITY: 3,
            IntentType.COMPARISON: 4,
            IntentType.BUSINESS_INSIGHT: 5,
            IntentType.PATTERN_ANALYSIS: 6,
            IntentType.RECOMMENDATION: 7
        }
        
        complexity_score += intent_complexity.get(intent_type, 3)
        
        # Adjust based on number of entities
        complexity_score += len(entities)
        
        # Adjust based on specific patterns
        if 'compare' in text and 'with' in text:
            complexity_score += 2
        
        if any(word in text for word in ['trend', 'pattern', 'forecast', 'predict']):
            complexity_score += 2
        
        if any(word in text for word in ['why', 'how', 'insight', 'analysis']):
            complexity_score += 1
        
        # Check for multiple time periods
        time_entities = [e for e in entities if e.entity_type == EntityType.TIME_PERIOD]
        if len(time_entities) > 1:
            complexity_score += 2
        
        # Determine final complexity
        if complexity_score <= 3:
            return "low"
        elif complexity_score <= 6:
            return "medium"
        elif complexity_score <= 10:
            return "high"
        else:
            return "very_high"
    
    async def _extract_subintent(
        self, 
        text: str, 
        intent_type: IntentType, 
        entities: List[Entity]
    ) -> Optional[str]:
        """Extract more specific subintent for detailed processing."""
        
        if intent_type == IntentType.DATA_QUERY:
            # Determine type of data query
            if any(word in text for word in ['top', 'highest', 'best', 'maximum']):
                return "top_performers"
            elif any(word in text for word in ['bottom', 'lowest', 'worst', 'minimum']):
                return "bottom_performers"
            elif any(word in text for word in ['total', 'sum', 'aggregate']):
                return "aggregation"
            elif any(word in text for word in ['average', 'mean', 'avg']):
                return "average"
            elif any(word in text for word in ['count', 'number', 'how many']):
                return "count"
            else:
                return "general_data"
        
        elif intent_type == IntentType.PATTERN_ANALYSIS:
            if any(word in text for word in ['trend', 'trending', 'direction']):
                return "trend_analysis"
            elif any(word in text for word in ['seasonal', 'seasonality', 'cycle']):
                return "seasonality_analysis"
            elif any(word in text for word in ['forecast', 'predict', 'projection']):
                return "forecasting"
            elif any(word in text for word in ['anomaly', 'outlier', 'unusual']):
                return "anomaly_detection"
            else:
                return "general_pattern"
        
        elif intent_type == IntentType.COMPARISON:
            if 'year over year' in text or 'yoy' in text:
                return "year_over_year"
            elif 'month over month' in text or 'mom' in text:
                return "month_over_month"
            elif any(word in text for word in ['vs', 'versus', 'against']):
                return "direct_comparison"
            elif 'benchmark' in text:
                return "benchmarking"
            else:
                return "general_comparison"
        
        elif intent_type == IntentType.BUSINESS_INSIGHT:
            if any(word in text for word in ['why', 'reason', 'cause']):
                return "causal_analysis"
            elif any(word in text for word in ['correlation', 'relationship']):
                return "correlation_analysis"
            elif any(word in text for word in ['impact', 'effect', 'influence']):
                return "impact_analysis"
            else:
                return "general_insight"
        
        elif intent_type == IntentType.RECOMMENDATION:
            if any(word in text for word in ['improve', 'optimize', 'enhance']):
                return "improvement_recommendations"
            elif any(word in text for word in ['strategy', 'plan', 'approach']):
                return "strategic_recommendations"
            elif any(word in text for word in ['action', 'next steps']):
                return "action_recommendations"
            else:
                return "general_recommendations"
        
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the Intent Analyzer."""
        
        try:
            # Test basic functionality
            test_text = "What were our top 5 products by sales last month?"
            intent = await self.analyze_intent(test_text)
            
            return {
                "status": "healthy",
                "details": "Intent analysis operational",
                "test_intent_classified": intent.intent_type.value,
                "test_entities_extracted": len(intent.entities),
                "patterns_loaded": sum(len(patterns) for patterns in self.intent_patterns.values()),
                "entity_patterns_loaded": sum(len(patterns) for patterns in self.entity_patterns.values())
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Intent analysis failed health check"
            } 