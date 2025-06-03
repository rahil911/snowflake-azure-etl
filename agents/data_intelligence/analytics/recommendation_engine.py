"""
Recommendation Engine for Data Intelligence Agent

This module generates actionable business recommendations based on:
- Data analysis results and insights
- Detected patterns and trends
- Business context and rules
- Performance metrics and KPIs
- Best practices and optimization opportunities
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
import numpy as np
from pathlib import Path
from collections import defaultdict

from pydantic import BaseModel, Field, validator
import pandas as pd

from shared.schemas.data_models import QueryResult, BusinessEntity, DataQualityMetric
from shared.schemas.data_models import AnalysisResult
from shared.utils.caching import get_cache_manager
from shared.utils.validation import ValidationHelper
from shared.utils.retry import RetryStrategy, retry_on_exception
from shared.config.settings import Settings


logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Types of business recommendations."""
    OPTIMIZATION = "optimization"
    ALERT = "alert"
    OPPORTUNITY = "opportunity"
    RISK_MITIGATION = "risk_mitigation"
    PROCESS_IMPROVEMENT = "process_improvement"
    COST_REDUCTION = "cost_reduction"
    REVENUE_ENHANCEMENT = "revenue_enhancement"
    QUALITY_IMPROVEMENT = "quality_improvement"
    STRATEGIC = "strategic"
    TACTICAL = "tactical"


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationCategory(Enum):
    """Categories of business recommendations."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    CUSTOMER = "customer"
    PRODUCT = "product"
    MARKETING = "marketing"
    TECHNOLOGY = "technology"
    COMPLIANCE = "compliance"
    STRATEGY = "strategy"


class ActionType(Enum):
    """Types of recommended actions."""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    MONITORING = "monitoring"
    INVESTIGATION = "investigation"


@dataclass
class BusinessRecommendation:
    """Individual business recommendation with metadata."""
    recommendation_type: RecommendationType
    category: RecommendationCategory
    priority: RecommendationPriority
    action_type: ActionType
    title: str
    description: str
    rationale: str
    expected_impact: str
    confidence: float
    implementation_effort: str  # low, medium, high
    timeline: str  # immediate, days, weeks, months
    success_metrics: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "recommendation_type": self.recommendation_type.value,
            "category": self.category.value,
            "priority": self.priority.value,
            "action_type": self.action_type.value,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
            "implementation_effort": self.implementation_effort,
            "timeline": self.timeline,
            "success_metrics": self.success_metrics,
            "prerequisites": self.prerequisites,
            "risks": self.risks,
            "supporting_data": self.supporting_data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class RecommendationContext(BaseModel):
    """Context for generating recommendations."""
    business_domain: Optional[str] = None
    industry: Optional[str] = None
    company_size: Optional[str] = None
    current_challenges: List[str] = Field(default_factory=list)
    strategic_objectives: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    available_resources: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True


class RecommendationRequest(BaseModel):
    """Request for recommendation generation."""
    analysis_results: Dict[str, Any]
    insights: List[Dict[str, Any]] = Field(default_factory=list)
    patterns: List[Dict[str, Any]] = Field(default_factory=list)
    business_context: Optional[RecommendationContext] = None
    focus_areas: List[RecommendationCategory] = Field(default_factory=list)
    priority_filter: Optional[RecommendationPriority] = None
    max_recommendations: int = 10
    include_implementation_plan: bool = True
    
    class Config:
        validate_assignment = True


class RecommendationResponse(BaseModel):
    """Response from recommendation generation."""
    recommendations: List[Dict[str, Any]]
    summary: str
    priority_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    quick_wins: List[Dict[str, Any]] = Field(default_factory=list)
    strategic_initiatives: List[Dict[str, Any]] = Field(default_factory=list)
    implementation_roadmap: Optional[Dict[str, Any]] = None
    success_framework: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        validate_assignment = True


class RecommendationEngine:
    """
    Advanced recommendation engine for business intelligence insights.
    
    This class handles:
    - Business recommendation generation
    - Actionable insights derivation
    - Implementation planning
    - Impact assessment
    - Risk evaluation
    - Success metrics definition
    """
    
    def __init__(self, settings: Settings):
        """Initialize the recommendation engine."""
        self.settings = settings
        self.cache = get_cache_manager()
        self.validator = ValidationHelper()
        self.retry_strategy = RetryStrategy.EXPONENTIAL
        
        # Recommendation configuration
        self.config = {
            'min_confidence_threshold': 0.6,
            'max_recommendations_per_category': 3,
            'impact_weight': 0.4,
            'effort_weight': 0.3,
            'confidence_weight': 0.3
        }
        
        # Business rules and frameworks
        self._setup_business_frameworks()
        
        # Recommendation templates
        self._setup_recommendation_templates()
        
        logger.info("Recommendation engine initialized")
    
    def _setup_business_frameworks(self):
        """Setup business frameworks and best practices."""
        self.frameworks = {
            'financial': {
                'kpis': ['revenue_growth', 'profit_margin', 'cost_reduction', 'roi'],
                'best_practices': [
                    'Monitor cash flow regularly',
                    'Optimize operational efficiency',
                    'Diversify revenue streams',
                    'Control costs without sacrificing quality'
                ],
                'thresholds': {
                    'revenue_decline': -0.05,  # 5% decline
                    'margin_concern': 0.10,    # 10% margin
                    'cost_increase': 0.15      # 15% cost increase
                }
            },
            'operational': {
                'kpis': ['efficiency', 'quality', 'cycle_time', 'error_rate'],
                'best_practices': [
                    'Implement continuous improvement',
                    'Automate repetitive processes',
                    'Monitor quality metrics',
                    'Optimize resource utilization'
                ],
                'thresholds': {
                    'efficiency_target': 0.85,   # 85% efficiency
                    'error_rate_limit': 0.02,    # 2% error rate
                    'quality_standard': 0.95     # 95% quality
                }
            },
            'customer': {
                'kpis': ['satisfaction', 'retention', 'churn_rate', 'lifetime_value'],
                'best_practices': [
                    'Personalize customer experiences',
                    'Respond quickly to customer issues',
                    'Collect and act on feedback',
                    'Build long-term relationships'
                ],
                'thresholds': {
                    'satisfaction_target': 4.0,   # 4.0/5.0 satisfaction
                    'churn_limit': 0.05,          # 5% churn rate
                    'response_time': 24           # 24 hours response
                }
            }
        }
    
    def _setup_recommendation_templates(self):
        """Setup recommendation templates for different scenarios."""
        self.templates = {
            'declining_metric': {
                'type': RecommendationType.ALERT,
                'priority': RecommendationPriority.HIGH,
                'action_type': ActionType.IMMEDIATE,
                'template': "Address declining {metric} through {action}",
                'rationale_template': "{metric} has declined by {percentage}, requiring immediate attention"
            },
            'optimization_opportunity': {
                'type': RecommendationType.OPTIMIZATION,
                'priority': RecommendationPriority.MEDIUM,
                'action_type': ActionType.SHORT_TERM,
                'template': "Optimize {area} to improve {outcome}",
                'rationale_template': "Analysis shows {opportunity} in {area} with potential {impact}"
            },
            'seasonal_pattern': {
                'type': RecommendationType.STRATEGIC,
                'priority': RecommendationPriority.MEDIUM,
                'action_type': ActionType.LONG_TERM,
                'template': "Leverage {pattern} seasonal pattern for {benefit}",
                'rationale_template': "Strong {pattern} seasonality detected with {strength} strength"
            },
            'anomaly_investigation': {
                'type': RecommendationType.ALERT,
                'priority': RecommendationPriority.HIGH,
                'action_type': ActionType.INVESTIGATION,
                'template': "Investigate anomalous {metric} values",
                'rationale_template': "Detected {count} anomalous values in {metric} requiring investigation"
            },
            'correlation_leverage': {
                'type': RecommendationType.OPPORTUNITY,
                'priority': RecommendationPriority.MEDIUM,
                'action_type': ActionType.TACTICAL,
                'template': "Leverage correlation between {var1} and {var2}",
                'rationale_template': "Strong {direction} correlation ({value}) found between {var1} and {var2}"
            }
        }
    
    async def generate_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        Generate comprehensive business recommendations.
        
        Args:
            request: Recommendation request with analysis results and context
            
        Returns:
            Generated recommendations with implementation guidance
        """
        try:
            start_time = datetime.now()
            logger.info("Starting recommendation generation")
            
            # Generate recommendations from different sources
            recommendations = []
            
            # Recommendations from insights
            if request.insights:
                insight_recommendations = await self._generate_from_insights(request.insights, request)
                recommendations.extend(insight_recommendations)
            
            # Recommendations from patterns
            if request.patterns:
                pattern_recommendations = await self._generate_from_patterns(request.patterns, request)
                recommendations.extend(pattern_recommendations)
            
            # Recommendations from analysis results
            if request.analysis_results:
                analysis_recommendations = await self._generate_from_analysis(request.analysis_results, request)
                recommendations.extend(analysis_recommendations)
            
            # Filter and prioritize recommendations
            filtered_recommendations = self._filter_recommendations(recommendations, request)
            prioritized_recommendations = self._prioritize_recommendations(filtered_recommendations)
            
            # Limit to max recommendations
            final_recommendations = prioritized_recommendations[:request.max_recommendations]
            
            # Categorize recommendations
            priority_recs = [r for r in final_recommendations if r.priority in [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH]]
            quick_wins = [r for r in final_recommendations if r.implementation_effort == "low" and r.timeline in ["immediate", "days"]]
            strategic_recs = [r for r in final_recommendations if r.action_type == ActionType.LONG_TERM]
            
            # Generate implementation roadmap
            implementation_roadmap = None
            if request.include_implementation_plan:
                implementation_roadmap = self._generate_implementation_roadmap(final_recommendations)
            
            # Generate success framework
            success_framework = self._generate_success_framework(final_recommendations)
            
            # Generate summary
            summary = self._generate_summary(final_recommendations)
            
            response = RecommendationResponse(
                recommendations=[rec.to_dict() for rec in final_recommendations],
                summary=summary,
                priority_recommendations=[rec.to_dict() for rec in priority_recs],
                quick_wins=[rec.to_dict() for rec in quick_wins],
                strategic_initiatives=[rec.to_dict() for rec in strategic_recs],
                implementation_roadmap=implementation_roadmap,
                success_framework=success_framework,
                metadata={
                    'total_recommendations': len(final_recommendations),
                    'generation_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'sources_analyzed': {
                        'insights': len(request.insights),
                        'patterns': len(request.patterns),
                        'analysis_results': bool(request.analysis_results)
                    }
                }
            )
            
            logger.info(f"Recommendation generation completed with {len(final_recommendations)} recommendations")
            return response
            
        except Exception as e:
            logger.error(f"Error in recommendation generation: {str(e)}")
            raise
    
    async def _generate_from_insights(self, insights: List[Dict[str, Any]], request: RecommendationRequest) -> List[BusinessRecommendation]:
        """Generate recommendations from business insights."""
        recommendations = []
        
        try:
            for insight in insights:
                insight_type = insight.get('insight_type', '')
                category = insight.get('category', '')
                severity = insight.get('severity', '')
                
                if insight_type == 'trend':
                    trend_rec = await self._generate_trend_recommendation(insight, request)
                    if trend_rec:
                        recommendations.append(trend_rec)
                
                elif insight_type == 'anomaly':
                    anomaly_rec = await self._generate_anomaly_recommendation(insight, request)
                    if anomaly_rec:
                        recommendations.append(anomaly_rec)
                
                elif insight_type == 'performance':
                    performance_rec = await self._generate_performance_recommendation(insight, request)
                    if performance_rec:
                        recommendations.append(performance_rec)
                
                elif insight_type == 'correlation':
                    correlation_rec = await self._generate_correlation_recommendation(insight, request)
                    if correlation_rec:
                        recommendations.append(correlation_rec)
        
        except Exception as e:
            logger.error(f"Error generating recommendations from insights: {str(e)}")
        
        return recommendations
    
    async def _generate_trend_recommendation(self, insight: Dict[str, Any], request: RecommendationRequest) -> Optional[BusinessRecommendation]:
        """Generate recommendation for trend insights."""
        try:
            value = insight.get('value', {})
            direction = value.get('direction', '')
            growth_rate = value.get('growth_rate', 0)
            
            category = self._map_category(insight.get('category', ''))
            
            if direction == 'decreasing' and growth_rate < -10:
                # Declining trend - high priority
                recommendation = BusinessRecommendation(
                    recommendation_type=RecommendationType.ALERT,
                    category=category,
                    priority=RecommendationPriority.HIGH,
                    action_type=ActionType.IMMEDIATE,
                    title=f"Address Declining {insight.get('title', 'Metric')}",
                    description=f"Implement corrective measures to reverse negative trend in {insight.get('title', 'metric')}",
                    rationale=f"Metric shows {growth_rate:.1f}% decline requiring immediate intervention",
                    expected_impact="Prevent further deterioration and stabilize performance",
                    confidence=insight.get('confidence', 0.8),
                    implementation_effort="medium",
                    timeline="immediate",
                    success_metrics=[f"Reverse declining trend", f"Achieve positive growth"],
                    prerequisites=["Root cause analysis", "Resource allocation"],
                    risks=["Continued decline if not addressed", "Resource investment required"],
                    supporting_data=value
                )
                return recommendation
            
            elif direction == 'increasing' and growth_rate > 20:
                # Strong growth - opportunity
                recommendation = BusinessRecommendation(
                    recommendation_type=RecommendationType.OPPORTUNITY,
                    category=category,
                    priority=RecommendationPriority.MEDIUM,
                    action_type=ActionType.SHORT_TERM,
                    title=f"Leverage Growing {insight.get('title', 'Metric')}",
                    description=f"Scale resources and processes to support continued growth in {insight.get('title', 'metric')}",
                    rationale=f"Strong {growth_rate:.1f}% growth presents scaling opportunity",
                    expected_impact="Maximize growth potential and market position",
                    confidence=insight.get('confidence', 0.8),
                    implementation_effort="medium",
                    timeline="weeks",
                    success_metrics=[f"Sustain growth rate", f"Scale operations efficiently"],
                    prerequisites=["Capacity planning", "Resource scaling"],
                    risks=["Growth may plateau", "Scaling challenges"],
                    supporting_data=value
                )
                return recommendation
        
        except Exception as e:
            logger.error(f"Error generating trend recommendation: {str(e)}")
        
        return None
    
    async def _generate_anomaly_recommendation(self, insight: Dict[str, Any], request: RecommendationRequest) -> Optional[BusinessRecommendation]:
        """Generate recommendation for anomaly insights."""
        try:
            value = insight.get('value', {})
            anomaly_count = value.get('anomaly_count', 0)
            max_z_score = value.get('max_z_score', 0)
            
            category = self._map_category(insight.get('category', ''))
            
            if max_z_score > 3:
                priority = RecommendationPriority.CRITICAL
                timeline = "immediate"
            elif max_z_score > 2.5:
                priority = RecommendationPriority.HIGH
                timeline = "days"
            else:
                priority = RecommendationPriority.MEDIUM
                timeline = "weeks"
            
            recommendation = BusinessRecommendation(
                recommendation_type=RecommendationType.ALERT,
                category=category,
                priority=priority,
                action_type=ActionType.INVESTIGATION,
                title=f"Investigate Anomalies in {insight.get('title', 'Data')}",
                description=f"Investigate and address {anomaly_count} anomalous values detected",
                rationale=f"Detected {anomaly_count} anomalies with max z-score of {max_z_score:.2f}",
                expected_impact="Identify and resolve data quality or process issues",
                confidence=insight.get('confidence', 0.9),
                implementation_effort="low",
                timeline=timeline,
                success_metrics=["Root cause identified", "Anomalies resolved", "Prevention measures implemented"],
                prerequisites=["Data access", "Domain expertise"],
                risks=["Underlying issues may persist", "Data quality concerns"],
                supporting_data=value
            )
            return recommendation
        
        except Exception as e:
            logger.error(f"Error generating anomaly recommendation: {str(e)}")
        
        return None
    
    async def _generate_performance_recommendation(self, insight: Dict[str, Any], request: RecommendationRequest) -> Optional[BusinessRecommendation]:
        """Generate recommendation for performance insights."""
        try:
            value = insight.get('value', {})
            meets_threshold = value.get('meets_threshold', True)
            performance_gap = value.get('performance_gap', 0)
            
            category = self._map_category(insight.get('category', ''))
            
            if not meets_threshold:
                if performance_gap > 0.2:
                    priority = RecommendationPriority.HIGH
                    effort = "high"
                elif performance_gap > 0.1:
                    priority = RecommendationPriority.MEDIUM
                    effort = "medium"
                else:
                    priority = RecommendationPriority.LOW
                    effort = "low"
                
                recommendation = BusinessRecommendation(
                    recommendation_type=RecommendationType.PROCESS_IMPROVEMENT,
                    category=category,
                    priority=priority,
                    action_type=ActionType.SHORT_TERM,
                    title=f"Improve {insight.get('title', 'Performance')}",
                    description=f"Implement improvements to meet performance thresholds",
                    rationale=f"Performance is {performance_gap:.1%} below target threshold",
                    expected_impact=f"Close {performance_gap:.1%} performance gap",
                    confidence=insight.get('confidence', 0.85),
                    implementation_effort=effort,
                    timeline="weeks",
                    success_metrics=["Meet performance threshold", "Sustain improvements"],
                    prerequisites=["Performance analysis", "Improvement planning"],
                    risks=["Implementation challenges", "Resource constraints"],
                    supporting_data=value
                )
                return recommendation
        
        except Exception as e:
            logger.error(f"Error generating performance recommendation: {str(e)}")
        
        return None
    
    async def _generate_correlation_recommendation(self, insight: Dict[str, Any], request: RecommendationRequest) -> Optional[BusinessRecommendation]:
        """Generate recommendation for correlation insights."""
        try:
            value = insight.get('value', {})
            correlation = value.get('correlation_coefficient', 0)
            strength = value.get('strength', '')
            direction = value.get('direction', '')
            var1 = value.get('variable1', '')
            var2 = value.get('variable2', '')
            
            category = self._map_category(insight.get('category', ''))
            
            if strength in ['strong', 'moderate'] and abs(correlation) > 0.6:
                recommendation = BusinessRecommendation(
                    recommendation_type=RecommendationType.OPPORTUNITY,
                    category=category,
                    priority=RecommendationPriority.MEDIUM,
                    action_type=ActionType.TACTICAL,
                    title=f"Leverage {strength.title()} Correlation",
                    description=f"Use {direction} correlation between {var1} and {var2} for business advantage",
                    rationale=f"{strength.title()} {direction} correlation ({correlation:.3f}) found",
                    expected_impact="Improve predictability and decision making",
                    confidence=abs(correlation),
                    implementation_effort="low",
                    timeline="weeks",
                    success_metrics=["Correlation utilized in planning", "Improved forecasting accuracy"],
                    prerequisites=["Data validation", "Process integration"],
                    risks=["Correlation may change", "Spurious correlation risk"],
                    supporting_data=value
                )
                return recommendation
        
        except Exception as e:
            logger.error(f"Error generating correlation recommendation: {str(e)}")
        
        return None
    
    async def _generate_from_patterns(self, patterns: List[Dict[str, Any]], request: RecommendationRequest) -> List[BusinessRecommendation]:
        """Generate recommendations from detected patterns."""
        recommendations = []
        
        try:
            for pattern in patterns:
                pattern_type = pattern.get('pattern_type', '')
                
                if pattern_type == 'seasonal':
                    seasonal_rec = await self._generate_seasonal_recommendation(pattern, request)
                    if seasonal_rec:
                        recommendations.append(seasonal_rec)
                
                elif pattern_type == 'cyclical':
                    cyclical_rec = await self._generate_cyclical_recommendation(pattern, request)
                    if cyclical_rec:
                        recommendations.append(cyclical_rec)
                
                elif pattern_type == 'distribution':
                    distribution_rec = await self._generate_distribution_recommendation(pattern, request)
                    if distribution_rec:
                        recommendations.append(distribution_rec)
        
        except Exception as e:
            logger.error(f"Error generating recommendations from patterns: {str(e)}")
        
        return recommendations
    
    async def _generate_seasonal_recommendation(self, pattern: Dict[str, Any], request: RecommendationRequest) -> Optional[BusinessRecommendation]:
        """Generate recommendation for seasonal patterns."""
        try:
            pattern_data = pattern.get('pattern_data', {})
            seasonal_type = pattern_data.get('seasonal_type', '')
            strength = pattern_data.get('strength', 0)
            
            if strength > 0.7:  # Strong seasonality
                recommendation = BusinessRecommendation(
                    recommendation_type=RecommendationType.STRATEGIC,
                    category=RecommendationCategory.OPERATIONAL,
                    priority=RecommendationPriority.MEDIUM,
                    action_type=ActionType.LONG_TERM,
                    title=f"Implement {seasonal_type.title()} Seasonal Planning",
                    description=f"Develop {seasonal_type} planning processes to leverage strong seasonal patterns",
                    rationale=f"Strong {seasonal_type} seasonality ({strength:.1%}) provides planning opportunity",
                    expected_impact="Improve resource allocation and forecasting accuracy",
                    confidence=pattern.get('confidence', 0.8),
                    implementation_effort="medium",
                    timeline="months",
                    success_metrics=["Seasonal planning implemented", "Forecast accuracy improved"],
                    prerequisites=["Historical data analysis", "Planning process design"],
                    risks=["Seasonal patterns may change", "Implementation complexity"],
                    supporting_data=pattern_data
                )
                return recommendation
        
        except Exception as e:
            logger.error(f"Error generating seasonal recommendation: {str(e)}")
        
        return None
    
    async def _generate_cyclical_recommendation(self, pattern: Dict[str, Any], request: RecommendationRequest) -> Optional[BusinessRecommendation]:
        """Generate recommendation for cyclical patterns."""
        try:
            pattern_data = pattern.get('pattern_data', {})
            cycle_length = pattern_data.get('cycle_length', 0)
            regularity_score = pattern_data.get('regularity_score', 0)
            
            if regularity_score > 0.6:  # Regular cycles
                recommendation = BusinessRecommendation(
                    recommendation_type=RecommendationType.OPTIMIZATION,
                    category=RecommendationCategory.OPERATIONAL,
                    priority=RecommendationPriority.MEDIUM,
                    action_type=ActionType.SHORT_TERM,
                    title=f"Optimize for {cycle_length:.1f}-Period Cycles",
                    description=f"Align operations with detected {cycle_length:.1f}-period cyclical patterns",
                    rationale=f"Regular cycles detected with {regularity_score:.1%} consistency",
                    expected_impact="Improve operational efficiency and resource utilization",
                    confidence=pattern.get('confidence', 0.7),
                    implementation_effort="medium",
                    timeline="weeks",
                    success_metrics=["Cycle-aligned operations", "Improved efficiency"],
                    prerequisites=["Process analysis", "Resource planning"],
                    risks=["Cycle disruption", "Over-optimization"],
                    supporting_data=pattern_data
                )
                return recommendation
        
        except Exception as e:
            logger.error(f"Error generating cyclical recommendation: {str(e)}")
        
        return None
    
    async def _generate_distribution_recommendation(self, pattern: Dict[str, Any], request: RecommendationRequest) -> Optional[BusinessRecommendation]:
        """Generate recommendation for distribution patterns."""
        try:
            pattern_data = pattern.get('pattern_data', {})
            distribution_type = pattern_data.get('distribution_type', '')
            outlier_percentage = pattern_data.get('outlier_percentage', 0)
            
            if 'skewed' in distribution_type or outlier_percentage > 0.05:
                recommendation = BusinessRecommendation(
                    recommendation_type=RecommendationType.QUALITY_IMPROVEMENT,
                    category=RecommendationCategory.OPERATIONAL,
                    priority=RecommendationPriority.MEDIUM,
                    action_type=ActionType.SHORT_TERM,
                    title="Address Data Distribution Issues",
                    description=f"Address {distribution_type} distribution and {outlier_percentage:.1%} outliers",
                    rationale=f"Data shows {distribution_type} distribution with quality concerns",
                    expected_impact="Improve data quality and analysis reliability",
                    confidence=pattern.get('confidence', 0.7),
                    implementation_effort="low",
                    timeline="weeks",
                    success_metrics=["Outliers reduced", "Distribution normalized"],
                    prerequisites=["Data quality analysis", "Outlier investigation"],
                    risks=["Data transformation challenges", "Information loss"],
                    supporting_data=pattern_data
                )
                return recommendation
        
        except Exception as e:
            logger.error(f"Error generating distribution recommendation: {str(e)}")
        
        return None
    
    async def _generate_from_analysis(self, analysis_results: Dict[str, Any], request: RecommendationRequest) -> List[BusinessRecommendation]:
        """Generate recommendations from general analysis results."""
        recommendations = []
        
        try:
            # Analyze key metrics and thresholds
            for framework_name, framework in self.frameworks.items():
                thresholds = framework.get('thresholds', {})
                
                for metric, threshold in thresholds.items():
                    if metric in analysis_results:
                        value = analysis_results[metric]
                        
                        if isinstance(value, (int, float)):
                            if self._threshold_violated(metric, value, threshold):
                                threshold_rec = self._generate_threshold_recommendation(
                                    metric, value, threshold, framework_name
                                )
                                if threshold_rec:
                                    recommendations.append(threshold_rec)
        
        except Exception as e:
            logger.error(f"Error generating recommendations from analysis: {str(e)}")
        
        return recommendations
    
    def _threshold_violated(self, metric: str, value: float, threshold: float) -> bool:
        """Check if a threshold is violated."""
        if 'decline' in metric or 'decrease' in metric:
            return value < threshold
        elif 'increase' in metric or 'growth' in metric:
            return value > threshold
        elif 'rate' in metric or 'percentage' in metric:
            return value > threshold
        else:
            return value < threshold
    
    def _generate_threshold_recommendation(self, metric: str, value: float, threshold: float, framework: str) -> BusinessRecommendation:
        """Generate recommendation for threshold violations."""
        category = self._map_framework_to_category(framework)
        
        if abs(value - threshold) / threshold > 0.2:
            priority = RecommendationPriority.HIGH
        else:
            priority = RecommendationPriority.MEDIUM
        
        recommendation = BusinessRecommendation(
            recommendation_type=RecommendationType.ALERT,
            category=category,
            priority=priority,
            action_type=ActionType.IMMEDIATE,
            title=f"Address {metric.replace('_', ' ').title()} Threshold Violation",
            description=f"Take corrective action for {metric} threshold violation",
            rationale=f"{metric} value ({value:.3f}) violates threshold ({threshold:.3f})",
            expected_impact="Return metric to acceptable range",
            confidence=0.9,
            implementation_effort="medium",
            timeline="immediate",
            success_metrics=[f"Meet {metric} threshold"],
            prerequisites=["Root cause analysis"],
            risks=["Continued violation", "Performance impact"]
        )
        
        return recommendation
    
    def _map_category(self, category_str: str) -> RecommendationCategory:
        """Map category string to RecommendationCategory enum."""
        category_mapping = {
            'financial': RecommendationCategory.FINANCIAL,
            'operational': RecommendationCategory.OPERATIONAL,
            'customer': RecommendationCategory.CUSTOMER,
            'product': RecommendationCategory.PRODUCT,
            'quality': RecommendationCategory.OPERATIONAL,
            'efficiency': RecommendationCategory.OPERATIONAL
        }
        return category_mapping.get(category_str.lower(), RecommendationCategory.OPERATIONAL)
    
    def _map_framework_to_category(self, framework: str) -> RecommendationCategory:
        """Map framework to recommendation category."""
        framework_mapping = {
            'financial': RecommendationCategory.FINANCIAL,
            'operational': RecommendationCategory.OPERATIONAL,
            'customer': RecommendationCategory.CUSTOMER
        }
        return framework_mapping.get(framework, RecommendationCategory.OPERATIONAL)
    
    def _filter_recommendations(self, recommendations: List[BusinessRecommendation], request: RecommendationRequest) -> List[BusinessRecommendation]:
        """Filter recommendations based on request criteria."""
        filtered = recommendations
        
        # Filter by confidence threshold
        filtered = [r for r in filtered if r.confidence >= self.config['min_confidence_threshold']]
        
        # Filter by focus areas
        if request.focus_areas:
            filtered = [r for r in filtered if r.category in request.focus_areas]
        
        # Filter by priority
        if request.priority_filter:
            priority_order = [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH, RecommendationPriority.MEDIUM, RecommendationPriority.LOW]
            min_priority_index = priority_order.index(request.priority_filter)
            filtered = [r for r in filtered if priority_order.index(r.priority) <= min_priority_index]
        
        return filtered
    
    def _prioritize_recommendations(self, recommendations: List[BusinessRecommendation]) -> List[BusinessRecommendation]:
        """Prioritize recommendations using weighted scoring."""
        def calculate_priority_score(rec: BusinessRecommendation) -> float:
            # Priority weight
            priority_weights = {
                RecommendationPriority.CRITICAL: 1.0,
                RecommendationPriority.HIGH: 0.8,
                RecommendationPriority.MEDIUM: 0.6,
                RecommendationPriority.LOW: 0.4
            }
            
            # Effort weight (lower effort = higher score)
            effort_weights = {
                'low': 1.0,
                'medium': 0.7,
                'high': 0.4
            }
            
            priority_score = priority_weights.get(rec.priority, 0.5)
            effort_score = effort_weights.get(rec.implementation_effort, 0.5)
            confidence_score = rec.confidence
            
            # Weighted total score
            total_score = (
                priority_score * self.config['impact_weight'] +
                effort_score * self.config['effort_weight'] +
                confidence_score * self.config['confidence_weight']
            )
            
            return total_score
        
        # Sort by priority score (descending)
        return sorted(recommendations, key=calculate_priority_score, reverse=True)
    
    def _generate_implementation_roadmap(self, recommendations: List[BusinessRecommendation]) -> Dict[str, Any]:
        """Generate implementation roadmap for recommendations."""
        roadmap = {
            'immediate': [],
            'short_term': [],
            'medium_term': [],
            'long_term': []
        }
        
        timeline_mapping = {
            'immediate': 'immediate',
            'days': 'short_term',
            'weeks': 'medium_term',
            'months': 'long_term'
        }
        
        for rec in recommendations:
            timeline_category = timeline_mapping.get(rec.timeline, 'medium_term')
            roadmap[timeline_category].append({
                'title': rec.title,
                'effort': rec.implementation_effort,
                'priority': rec.priority.value,
                'success_metrics': rec.success_metrics
            })
        
        return roadmap
    
    def _generate_success_framework(self, recommendations: List[BusinessRecommendation]) -> Dict[str, Any]:
        """Generate success measurement framework."""
        all_metrics = []
        categories = defaultdict(list)
        
        for rec in recommendations:
            all_metrics.extend(rec.success_metrics)
            categories[rec.category.value].extend(rec.success_metrics)
        
        # Remove duplicates
        unique_metrics = list(dict.fromkeys(all_metrics))
        
        return {
            'overall_metrics': unique_metrics[:10],  # Top 10
            'category_metrics': {k: list(dict.fromkeys(v))[:5] for k, v in categories.items()},
            'measurement_frequency': 'weekly',
            'review_cycle': 'monthly'
        }
    
    def _generate_summary(self, recommendations: List[BusinessRecommendation]) -> str:
        """Generate summary of recommendations."""
        if not recommendations:
            return "No actionable recommendations generated."
        
        priority_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for rec in recommendations:
            priority_counts[rec.priority.value] += 1
            category_counts[rec.category.value] += 1
        
        summary_parts = [f"Generated {len(recommendations)} actionable recommendations"]
        
        if priority_counts:
            priority_summary = ", ".join([f"{count} {priority}" for priority, count in priority_counts.items()])
            summary_parts.append(f"Priority distribution: {priority_summary}")
        
        return "; ".join(summary_parts)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the recommendation engine."""
        return {
            'status': 'healthy',
            'frameworks_loaded': len(self.frameworks),
            'templates_loaded': len(self.templates),
            'config_loaded': bool(self.config)
        } 