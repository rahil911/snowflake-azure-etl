"""
Response Formatter for Data Intelligence Agent

This module formats responses for business users, transforming technical
analysis results into business-friendly presentations and narratives.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from pydantic import BaseModel, Field
from shared.config.settings import Settings
from shared.utils.caching import get_cache_manager
from shared.utils.metrics import get_metrics_collector, track_performance

logger = logging.getLogger(__name__)

class ResponseFormat(Enum):
    """Response format types for different audiences."""
    EXECUTIVE_SUMMARY = "executive_summary"
    BUSINESS_NARRATIVE = "business_narrative"
    DETAILED_ANALYSIS = "detailed_analysis"
    DASHBOARD_SUMMARY = "dashboard_summary"
    CONVERSATIONAL = "conversational"
    TECHNICAL_REPORT = "technical_report"
    BULLET_POINTS = "bullet_points"
    INFOGRAPHIC = "infographic"

class AudienceType(Enum):
    """Target audience types for response customization."""
    EXECUTIVE = "executive"
    BUSINESS_ANALYST = "business_analyst"
    DEPARTMENT_MANAGER = "department_manager"
    GENERAL_USER = "general_user"
    TECHNICAL_USER = "technical_user"
    EXTERNAL_STAKEHOLDER = "external_stakeholder"

class ContentPriority(Enum):
    """Priority levels for content elements."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SUPPLEMENTARY = "supplementary"

@dataclass
class ContentElement:
    """Individual content element in a response."""
    content_type: str
    priority: ContentPriority
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    audience_relevance: Dict[AudienceType, float] = field(default_factory=dict)

@dataclass
class FormattedResponse:
    """Formatted response for business users."""
    format_type: ResponseFormat
    audience: AudienceType
    title: str
    summary: str
    content_elements: List[ContentElement]
    recommendations: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    formatted_at: datetime = field(default_factory=datetime.now)

class ResponseFormattingRequest(BaseModel):
    """Request for response formatting."""
    data: Dict[str, Any]
    format_type: ResponseFormat = ResponseFormat.BUSINESS_NARRATIVE
    audience: AudienceType = AudienceType.GENERAL_USER
    context: Dict[str, Any] = Field(default_factory=dict)
    include_technical_details: bool = False
    include_recommendations: bool = True
    include_charts: bool = True
    max_length: Optional[int] = None
    tone: str = "professional"
    
    class Config:
        validate_assignment = True

class ResponseFormatter:
    """
    Response Formatter for creating business-friendly responses.
    
    Transforms technical analysis results into various presentation
    formats tailored for different business audiences.
    """
    
    def __init__(self, settings: Settings):
        """Initialize response formatter."""
        self.settings = settings
        self.cache = get_cache_manager()
        self.metrics = get_metrics_collector()
        
        # Configuration
        self.config = {
            'default_format': settings.RESPONSE_FORMATTER.get('default_format', 'business_narrative'),
            'cache_ttl': settings.RESPONSE_FORMATTER.get('cache_ttl', 1800),
            'max_summary_length': settings.RESPONSE_FORMATTER.get('max_summary_length', 500),
            'max_detail_length': settings.RESPONSE_FORMATTER.get('max_detail_length', 2000),
            'include_metadata': settings.RESPONSE_FORMATTER.get('include_metadata', True)
        }
        
        # Formatting templates and rules
        self._setup_formatting_templates()
        self._setup_audience_preferences()
        self._setup_content_prioritization()
        
        logger.info("Response Formatter initialized")

    def _setup_formatting_templates(self):
        """Setup formatting templates for different response types."""
        self.templates = {
            ResponseFormat.EXECUTIVE_SUMMARY: {
                'structure': ['headline', 'key_findings', 'impact', 'next_steps'],
                'max_length': 300,
                'tone': 'authoritative',
                'focus': 'outcomes',
                'detail_level': 'high_level'
            },
            ResponseFormat.BUSINESS_NARRATIVE: {
                'structure': ['context', 'analysis', 'insights', 'implications', 'recommendations'],
                'max_length': 800,
                'tone': 'professional',
                'focus': 'story',
                'detail_level': 'medium'
            },
            ResponseFormat.DETAILED_ANALYSIS: {
                'structure': ['overview', 'methodology', 'findings', 'detailed_analysis', 'conclusions'],
                'max_length': 2000,
                'tone': 'analytical',
                'focus': 'thoroughness',
                'detail_level': 'comprehensive'
            },
            ResponseFormat.DASHBOARD_SUMMARY: {
                'structure': ['kpis', 'trends', 'alerts', 'quick_actions'],
                'max_length': 200,
                'tone': 'concise',
                'focus': 'metrics',
                'detail_level': 'summary'
            },
            ResponseFormat.CONVERSATIONAL: {
                'structure': ['acknowledgment', 'explanation', 'insights', 'suggestions'],
                'max_length': 600,
                'tone': 'friendly',
                'focus': 'dialogue',
                'detail_level': 'accessible'
            }
        }

    def _setup_audience_preferences(self):
        """Setup preferences for different audience types."""
        self.audience_preferences = {
            AudienceType.EXECUTIVE: {
                'preferred_formats': [ResponseFormat.EXECUTIVE_SUMMARY, ResponseFormat.DASHBOARD_SUMMARY],
                'focus_areas': ['business_impact', 'strategic_implications', 'roi'],
                'avoid_technical': True,
                'include_recommendations': True,
                'max_reading_time': 2  # minutes
            },
            AudienceType.BUSINESS_ANALYST: {
                'preferred_formats': [ResponseFormat.DETAILED_ANALYSIS, ResponseFormat.BUSINESS_NARRATIVE],
                'focus_areas': ['data_insights', 'trends', 'methodology', 'recommendations'],
                'avoid_technical': False,
                'include_recommendations': True,
                'max_reading_time': 10
            },
            AudienceType.DEPARTMENT_MANAGER: {
                'preferred_formats': [ResponseFormat.BUSINESS_NARRATIVE, ResponseFormat.DASHBOARD_SUMMARY],
                'focus_areas': ['operational_impact', 'team_performance', 'action_items'],
                'avoid_technical': True,
                'include_recommendations': True,
                'max_reading_time': 5
            },
            AudienceType.GENERAL_USER: {
                'preferred_formats': [ResponseFormat.CONVERSATIONAL, ResponseFormat.BUSINESS_NARRATIVE],
                'focus_areas': ['key_insights', 'practical_implications', 'easy_actions'],
                'avoid_technical': True,
                'include_recommendations': True,
                'max_reading_time': 3
            }
        }

    def _setup_content_prioritization(self):
        """Setup content prioritization rules."""
        self.priority_rules = {
            'data_quality_issues': ContentPriority.CRITICAL,
            'significant_trends': ContentPriority.HIGH,
            'business_insights': ContentPriority.HIGH,
            'actionable_recommendations': ContentPriority.HIGH,
            'statistical_details': ContentPriority.MEDIUM,
            'methodology_notes': ContentPriority.LOW,
            'technical_specifications': ContentPriority.SUPPLEMENTARY
        }

    @track_performance(tags={"operation": "format_response"})
    async def format_response(self, request: ResponseFormattingRequest) -> FormattedResponse:
        """
        Format response based on request specifications.
        
        Args:
            request: Formatting request with data and preferences
            
        Returns:
            Formatted response ready for presentation
        """
        try:
            # Determine optimal format if not specified
            if not request.format_type:
                request.format_type = self._determine_optimal_format(request.audience, request.data)
            
            # Extract content elements from data
            content_elements = await self._extract_content_elements(request.data, request)
            
            # Prioritize content based on audience
            prioritized_content = self._prioritize_content(content_elements, request.audience)
            
            # Generate response components
            title = self._generate_title(request.data, request.format_type)
            summary = await self._generate_summary(prioritized_content, request)
            recommendations = await self._extract_recommendations(request.data, request)
            
            # Apply format-specific structuring
            structured_content = await self._apply_format_structure(
                prioritized_content, request.format_type, request
            )
            
            # Create formatted response
            formatted_response = FormattedResponse(
                format_type=request.format_type,
                audience=request.audience,
                title=title,
                summary=summary,
                content_elements=structured_content,
                recommendations=recommendations,
                supporting_data=self._extract_supporting_data(request.data),
                metadata={
                    'original_data_size': len(str(request.data)),
                    'processing_time': datetime.now().isoformat(),
                    'format_applied': request.format_type.value,
                    'audience_targeted': request.audience.value,
                    'content_elements_count': len(structured_content)
                }
            )
            
            # Record metrics
            await self._record_formatting_metrics(request, formatted_response)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return self._create_error_response(str(e), request)

    def _determine_optimal_format(self, audience: AudienceType, data: Dict[str, Any]) -> ResponseFormat:
        """Determine optimal response format based on audience and data."""
        preferences = self.audience_preferences.get(audience, {})
        preferred_formats = preferences.get('preferred_formats', [ResponseFormat.BUSINESS_NARRATIVE])
        
        # Consider data complexity
        data_complexity = self._assess_data_complexity(data)
        
        if data_complexity > 0.8 and ResponseFormat.DETAILED_ANALYSIS in preferred_formats:
            return ResponseFormat.DETAILED_ANALYSIS
        elif audience == AudienceType.EXECUTIVE:
            return ResponseFormat.EXECUTIVE_SUMMARY
        elif 'insights' in data and len(data.get('insights', [])) > 5:
            return ResponseFormat.BUSINESS_NARRATIVE
        else:
            return preferred_formats[0] if preferred_formats else ResponseFormat.BUSINESS_NARRATIVE

    def _assess_data_complexity(self, data: Dict[str, Any]) -> float:
        """Assess complexity of data to determine appropriate format."""
        complexity_score = 0.0
        
        # Factor in data volume
        if 'row_count' in data:
            row_count = data['row_count']
            if row_count > 10000:
                complexity_score += 0.3
            elif row_count > 1000:
                complexity_score += 0.1
        
        # Factor in number of insights
        insights_count = len(data.get('insights', []))
        if insights_count > 10:
            complexity_score += 0.3
        elif insights_count > 5:
            complexity_score += 0.2
        
        # Factor in number of recommendations
        recommendations_count = len(data.get('recommendations', []))
        if recommendations_count > 5:
            complexity_score += 0.2
        
        # Factor in data quality issues
        issues_count = len(data.get('quality_issues', []))
        if issues_count > 0:
            complexity_score += 0.2
        
        return min(complexity_score, 1.0)

    async def _extract_content_elements(self, data: Dict[str, Any], request: ResponseFormattingRequest) -> List[ContentElement]:
        """Extract content elements from analysis data."""
        elements = []
        
        # Extract key findings
        if 'insights' in data:
            for insight in data['insights']:
                element = ContentElement(
                    content_type='insight',
                    priority=ContentPriority.HIGH,
                    content=insight,
                    metadata={'source': 'analysis'}
                )
                elements.append(element)
        
        # Extract recommendations
        if 'recommendations' in data:
            for recommendation in data['recommendations']:
                element = ContentElement(
                    content_type='recommendation',
                    priority=ContentPriority.HIGH,
                    content=recommendation,
                    metadata={'source': 'analysis', 'actionable': True}
                )
                elements.append(element)
        
        # Extract summary information
        if 'summary' in data:
            element = ContentElement(
                content_type='summary',
                priority=ContentPriority.CRITICAL,
                content=data['summary'],
                metadata={'source': 'analysis'}
            )
            elements.append(element)
        
        # Extract data quality information
        if 'quality_issues' in data and data['quality_issues']:
            quality_summary = f"Data quality analysis identified {len(data['quality_issues'])} issues requiring attention."
            element = ContentElement(
                content_type='quality_alert',
                priority=ContentPriority.HIGH,
                content=quality_summary,
                metadata={'source': 'quality_analysis', 'alert': True}
            )
            elements.append(element)
        
        # Extract statistical findings
        if 'statistics' in data:
            stats = data['statistics']
            if isinstance(stats, dict):
                for stat_name, stat_value in stats.items():
                    if isinstance(stat_value, (int, float)):
                        content = f"{stat_name.replace('_', ' ').title()}: {stat_value:,.2f}"
                        element = ContentElement(
                            content_type='statistic',
                            priority=ContentPriority.MEDIUM,
                            content=content,
                            metadata={'source': 'statistics', 'metric': stat_name}
                        )
                        elements.append(element)
        
        return elements

    def _prioritize_content(self, elements: List[ContentElement], audience: AudienceType) -> List[ContentElement]:
        """Prioritize content elements based on audience preferences."""
        preferences = self.audience_preferences.get(audience, {})
        avoid_technical = preferences.get('avoid_technical', False)
        
        # Filter out technical content if needed
        if avoid_technical:
            elements = [e for e in elements if e.content_type not in ['technical_detail', 'methodology']]
        
        # Sort by priority and audience relevance
        def priority_score(element):
            base_priority = {
                ContentPriority.CRITICAL: 5,
                ContentPriority.HIGH: 4,
                ContentPriority.MEDIUM: 3,
                ContentPriority.LOW: 2,
                ContentPriority.SUPPLEMENTARY: 1
            }.get(element.priority, 0)
            
            # Boost score based on audience relevance
            relevance = element.audience_relevance.get(audience, 0.5)
            return base_priority + relevance
        
        return sorted(elements, key=priority_score, reverse=True)

    def _generate_title(self, data: Dict[str, Any], format_type: ResponseFormat) -> str:
        """Generate appropriate title based on data and format."""
        if format_type == ResponseFormat.EXECUTIVE_SUMMARY:
            return "Executive Summary: Data Analysis Results"
        elif format_type == ResponseFormat.DASHBOARD_SUMMARY:
            return "Dashboard Overview"
        elif format_type == ResponseFormat.DETAILED_ANALYSIS:
            return "Comprehensive Data Analysis Report"
        elif 'analysis_type' in data:
            analysis_type = data['analysis_type'].replace('_', ' ').title()
            return f"{analysis_type} Analysis Results"
        else:
            return "Data Analysis Insights"

    async def _generate_summary(self, content_elements: List[ContentElement], request: ResponseFormattingRequest) -> str:
        """Generate executive summary from content elements."""
        try:
            template = self.templates.get(request.format_type, {})
            max_length = request.max_length or template.get('max_length', 500)
            
            # Extract critical and high priority content
            key_elements = [e for e in content_elements if e.priority in [ContentPriority.CRITICAL, ContentPriority.HIGH]]
            
            if not key_elements:
                return "Analysis completed successfully with insights available in detailed results."
            
            # Build summary based on format type
            if request.format_type == ResponseFormat.EXECUTIVE_SUMMARY:
                summary = self._build_executive_summary(key_elements)
            elif request.format_type == ResponseFormat.CONVERSATIONAL:
                summary = self._build_conversational_summary(key_elements)
            else:
                summary = self._build_business_summary(key_elements)
            
            # Trim to max length if needed
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Analysis completed with insights available in the detailed results."

    def _build_executive_summary(self, elements: List[ContentElement]) -> str:
        """Build executive-style summary."""
        summary_parts = []
        
        # Find key insights
        insights = [e.content for e in elements if e.content_type == 'insight']
        if insights:
            summary_parts.append(f"Key Finding: {insights[0]}")
        
        # Find critical recommendations
        recommendations = [e.content for e in elements if e.content_type == 'recommendation']
        if recommendations:
            summary_parts.append(f"Priority Action: {recommendations[0]}")
        
        # Add data overview if available
        summaries = [e.content for e in elements if e.content_type == 'summary']
        if summaries:
            summary_parts.append(summaries[0])
        
        return " ".join(summary_parts)

    def _build_conversational_summary(self, elements: List[ContentElement]) -> str:
        """Build conversational-style summary."""
        summary_parts = ["Here's what I found from your data:"]
        
        # Add insights in conversational tone
        insights = [e.content for e in elements if e.content_type == 'insight']
        for insight in insights[:2]:  # Top 2 insights
            summary_parts.append(f"• {insight}")
        
        # Add actionable recommendations
        recommendations = [e.content for e in elements if e.content_type == 'recommendation']
        if recommendations:
            summary_parts.append(f"I recommend: {recommendations[0]}")
        
        return " ".join(summary_parts)

    def _build_business_summary(self, elements: List[ContentElement]) -> str:
        """Build business-focused summary."""
        summary_parts = []
        
        # Start with data overview
        summaries = [e.content for e in elements if e.content_type == 'summary']
        if summaries:
            summary_parts.append(summaries[0])
        
        # Add key business insights
        insights = [e.content for e in elements if e.content_type == 'insight']
        if insights:
            insight_text = "Key insights include: " + "; ".join(insights[:3])
            summary_parts.append(insight_text)
        
        return " ".join(summary_parts)

    async def _extract_recommendations(self, data: Dict[str, Any], request: ResponseFormattingRequest) -> List[str]:
        """Extract and format recommendations."""
        recommendations = []
        
        if 'recommendations' in data:
            raw_recommendations = data['recommendations']
            
            # Format based on audience
            if request.audience == AudienceType.EXECUTIVE:
                # Focus on strategic recommendations
                recommendations = [rec for rec in raw_recommendations if 'strategic' in rec.lower() or 'business' in rec.lower()]
                if not recommendations:
                    recommendations = raw_recommendations[:3]  # Top 3 for executives
            else:
                recommendations = raw_recommendations
        
        # Add default recommendation if none found
        if not recommendations and request.include_recommendations:
            recommendations = ["Continue monitoring these metrics for ongoing business optimization"]
        
        return recommendations

    async def _apply_format_structure(self, content_elements: List[ContentElement], format_type: ResponseFormat, request: ResponseFormattingRequest) -> List[ContentElement]:
        """Apply format-specific structure to content elements."""
        template = self.templates.get(format_type, {})
        structure = template.get('structure', ['content'])
        
        # Reorganize content based on structure
        structured_elements = []
        
        for section in structure:
            section_elements = [e for e in content_elements if self._element_fits_section(e, section)]
            
            if section_elements:
                # Add section header if needed
                if len(structure) > 1 and format_type == ResponseFormat.DETAILED_ANALYSIS:
                    header_element = ContentElement(
                        content_type='section_header',
                        priority=ContentPriority.MEDIUM,
                        content=section.replace('_', ' ').title(),
                        metadata={'section': section}
                    )
                    structured_elements.append(header_element)
                
                structured_elements.extend(section_elements)
        
        return structured_elements

    def _element_fits_section(self, element: ContentElement, section: str) -> bool:
        """Determine if content element fits in a specific section."""
        content_type = element.content_type
        
        section_mappings = {
            'headline': ['summary'],
            'key_findings': ['insight', 'summary'],
            'analysis': ['insight', 'statistic'],
            'insights': ['insight'],
            'recommendations': ['recommendation'],
            'next_steps': ['recommendation'],
            'methodology': ['technical_detail'],
            'overview': ['summary', 'insight'],
            'kpis': ['statistic'],
            'alerts': ['quality_alert']
        }
        
        return content_type in section_mappings.get(section, [content_type])

    def _extract_supporting_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract supporting data for detailed view."""
        supporting_data = {}
        
        # Extract charts data
        if 'charts' in data:
            supporting_data['charts'] = data['charts']
        
        # Extract metadata
        if 'metadata' in data:
            supporting_data['metadata'] = data['metadata']
        
        # Extract raw statistics
        if 'statistics' in data:
            supporting_data['statistics'] = data['statistics']
        
        return supporting_data

    def _create_error_response(self, error_message: str, request: ResponseFormattingRequest) -> FormattedResponse:
        """Create error response for formatting failures."""
        error_element = ContentElement(
            content_type='error',
            priority=ContentPriority.CRITICAL,
            content=f"Error formatting response: {error_message}",
            metadata={'error': True}
        )
        
        return FormattedResponse(
            format_type=request.format_type,
            audience=request.audience,
            title="Response Formatting Error",
            summary=f"Unable to format response: {error_message}",
            content_elements=[error_element],
            metadata={'error': True, 'error_message': error_message}
        )

    async def _record_formatting_metrics(self, request: ResponseFormattingRequest, response: FormattedResponse):
        """Record formatting metrics."""
        try:
            metrics_data = {
                'format_type': request.format_type.value,
                'audience': request.audience.value,
                'content_elements_count': len(response.content_elements),
                'response_length': len(response.summary),
                'include_recommendations': request.include_recommendations,
                'include_charts': request.include_charts
            }
            
            await self.metrics.record_event('response_formatting', metrics_data)
        except Exception as e:
            logger.warning(f"Error recording metrics: {str(e)}")

    def get_available_formats(self) -> List[Dict[str, Any]]:
        """Get list of available response formats."""
        return [
            {
                'format': format_type.value,
                'name': format_type.value.replace('_', ' ').title(),
                'description': template.get('focus', 'Standard format'),
                'max_length': template.get('max_length', 500),
                'detail_level': template.get('detail_level', 'medium')
            }
            for format_type, template in self.templates.items()
        ]

    def get_audience_preferences(self, audience: AudienceType) -> Dict[str, Any]:
        """Get preferences for specific audience type."""
        return self.audience_preferences.get(audience, {})

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of response formatter."""
        return {
            'service': 'response_formatter',
            'status': 'healthy',
            'cache_enabled': self.cache is not None,
            'metrics_enabled': self.metrics is not None,
            'available_formats': len(self.templates),
            'supported_audiences': len(self.audience_preferences),
            'config': {
                'default_format': self.config['default_format'],
                'max_summary_length': self.config['max_summary_length'],
                'include_metadata': self.config['include_metadata']
            },
            'timestamp': datetime.now().isoformat()
        } 