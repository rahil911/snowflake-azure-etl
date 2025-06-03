"""
Result processing system for MCP tool outputs.

This module provides the ResultProcessor class that handles formatting,
visualization, transformation, and export of results from MCP tool executions.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
import base64
import io

from pydantic import BaseModel, Field, ConfigDict
import google.genai as genai

from shared.schemas.agent_communication import (
    AgentMessage, AgentRole, MessageType, ConversationContext
)
from shared.config.logging_config import setup_logging
from shared.utils.metrics import get_metrics_collector


class CoordinatorError(Exception):
    """Base exception for coordinator errors."""
    pass
from .tool_executor import ExecutionResult, ToolDefinition


class ResultFormat(str, Enum):
    """Output format types."""
    JSON = "json"
    TABLE = "table"
    CHART = "chart"
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"
    IMAGE = "image"


class ChartType(str, Enum):
    """Chart visualization types."""
    LINE = "line"
    BAR = "bar"
    COLUMN = "column"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    TREEMAP = "treemap"
    GAUGE = "gauge"


class ProcessingStatus(str, Enum):
    """Result processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VisualizationConfig:
    """Configuration for result visualization."""
    chart_type: ChartType
    title: str = ""
    x_axis_label: str = ""
    y_axis_label: str = ""
    
    # Data mapping
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    group_by_column: Optional[str] = None
    value_column: Optional[str] = None
    
    # Styling
    width: int = 800
    height: int = 600
    color_scheme: str = "default"
    theme: str = "light"
    
    # Interactive features
    interactive: bool = True
    show_legend: bool = True
    show_grid: bool = True
    show_values: bool = False
    
    # Aggregation
    aggregation_method: Optional[str] = None  # sum, avg, count, etc.
    
    # Custom options
    custom_options: Dict[str, Any] = field(default_factory=dict)


class ProcessingRule(BaseModel):
    """Rule for automatic result processing."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    description: str = Field(..., description="Rule description")
    
    # Matching criteria
    tool_patterns: List[str] = Field(default_factory=list, description="Tool name patterns")
    result_patterns: List[str] = Field(default_factory=list, description="Result content patterns")
    data_types: List[str] = Field(default_factory=list, description="Expected data types")
    
    # Processing actions
    auto_visualize: bool = Field(default=False, description="Create automatic visualizations")
    default_format: ResultFormat = Field(default=ResultFormat.JSON, description="Default output format")
    chart_config: Optional[VisualizationConfig] = None
    
    # Transformations
    transformations: List[str] = Field(default_factory=list, description="Data transformations to apply")
    
    # Metadata
    priority: int = Field(default=0, description="Rule priority")
    enabled: bool = Field(default=True, description="Whether rule is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ProcessedResult(BaseModel):
    """Processed and formatted result."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str = Field(..., description="Source execution ID")
    tool_id: str = Field(..., description="Source tool ID")
    
    # Original result
    raw_result: Any = Field(..., description="Original tool result")
    
    # Processed formats
    formats: Dict[ResultFormat, Any] = Field(default_factory=dict, description="Processed formats")
    
    # Visualizations
    visualizations: Dict[str, Any] = Field(default_factory=dict, description="Generated visualizations")
    
    # Metadata
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    applied_rules: List[str] = Field(default_factory=list, description="Applied processing rules")
    
    # Statistics
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    data_size_bytes: Optional[int] = None
    
    # Processing metrics
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None


class DataTransformer:
    """Transforms data into different formats and structures."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    async def transform_to_table(self, data: Any) -> Dict[str, Any]:
        """Transform data to table format."""
        if isinstance(data, list):
            if not data:
                return {"headers": [], "rows": [], "total_rows": 0}
            
            # Check if it's a list of dictionaries
            if isinstance(data[0], dict):
                headers = list(data[0].keys())
                rows = [[item.get(header, "") for header in headers] for item in data]
                
                return {
                    "headers": headers,
                    "rows": rows,
                    "total_rows": len(rows)
                }
            
            # List of scalars
            else:
                return {
                    "headers": ["Value"],
                    "rows": [[item] for item in data],
                    "total_rows": len(data)
                }
        
        elif isinstance(data, dict):
            # Check if it has tabular structure
            if "rows" in data and "columns" in data:
                return {
                    "headers": data.get("columns", []),
                    "rows": data.get("rows", []),
                    "total_rows": len(data.get("rows", []))
                }
            
            # Convert dict to key-value table
            else:
                return {
                    "headers": ["Key", "Value"],
                    "rows": [[str(k), str(v)] for k, v in data.items()],
                    "total_rows": len(data)
                }
        
        else:
            # Single value
            return {
                "headers": ["Value"],
                "rows": [[str(data)]],
                "total_rows": 1
            }
    
    async def transform_to_csv(self, data: Any) -> str:
        """Transform data to CSV format."""
        table_data = await self.transform_to_table(data)
        
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(table_data["headers"])
        
        # Write rows
        for row in table_data["rows"]:
            writer.writerow(row)
        
        return output.getvalue()
    
    async def transform_to_markdown(self, data: Any) -> str:
        """Transform data to Markdown format."""
        if isinstance(data, dict) and "rows" in data and "columns" in data:
            # Tabular data
            table_data = await self.transform_to_table(data)
            
            md_lines = []
            
            # Headers
            headers = table_data["headers"]
            md_lines.append("| " + " | ".join(headers) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            
            # Rows
            for row in table_data["rows"]:
                md_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
            
            return "\n".join(md_lines)
        
        elif isinstance(data, list):
            # List format
            md_lines = []
            for i, item in enumerate(data, 1):
                md_lines.append(f"{i}. {item}")
            
            return "\n".join(md_lines)
        
        elif isinstance(data, dict):
            # Dictionary format
            md_lines = []
            for key, value in data.items():
                md_lines.append(f"**{key}**: {value}")
            
            return "\n".join(md_lines)
        
        else:
            return str(data)
    
    async def transform_to_html(self, data: Any) -> str:
        """Transform data to HTML format."""
        if isinstance(data, dict) and "rows" in data and "columns" in data:
            # Tabular data
            table_data = await self.transform_to_table(data)
            
            html_lines = ["<table border='1' cellpadding='5' cellspacing='0'>"]
            
            # Headers
            headers = table_data["headers"]
            html_lines.append("<thead><tr>")
            for header in headers:
                html_lines.append(f"<th>{header}</th>")
            html_lines.append("</tr></thead>")
            
            # Rows
            html_lines.append("<tbody>")
            for row in table_data["rows"]:
                html_lines.append("<tr>")
                for cell in row:
                    html_lines.append(f"<td>{cell}</td>")
                html_lines.append("</tr>")
            html_lines.append("</tbody>")
            
            html_lines.append("</table>")
            
            return "\n".join(html_lines)
        
        else:
            return f"<pre>{json.dumps(data, indent=2)}</pre>"
    
    def detect_chart_opportunities(self, data: Any) -> List[VisualizationConfig]:
        """Detect potential chart configurations for data."""
        configs = []
        
        if not isinstance(data, dict) or "rows" not in data or "columns" not in data:
            return configs
        
        columns = data.get("columns", [])
        rows = data.get("rows", [])
        
        if len(columns) < 2 or len(rows) == 0:
            return configs
        
        # Analyze column types
        numeric_columns = []
        categorical_columns = []
        date_columns = []
        
        for i, column in enumerate(columns):
            column_data = [row[i] for row in rows if i < len(row)]
            
            # Check if numeric
            try:
                [float(val) for val in column_data[:10] if val is not None]
                numeric_columns.append(column)
                continue
            except (ValueError, TypeError):
                pass
            
            # Check if date
            try:
                from dateutil import parser
                [parser.parse(str(val)) for val in column_data[:5] if val is not None]
                date_columns.append(column)
                continue
            except:
                pass
            
            # Assume categorical
            categorical_columns.append(column)
        
        # Generate chart configurations
        if len(numeric_columns) >= 2:
            # Scatter plot
            configs.append(VisualizationConfig(
                chart_type=ChartType.SCATTER,
                title="Scatter Plot",
                x_column=numeric_columns[0],
                y_column=numeric_columns[1],
                x_axis_label=numeric_columns[0],
                y_axis_label=numeric_columns[1]
            ))
        
        if categorical_columns and numeric_columns:
            # Bar chart
            configs.append(VisualizationConfig(
                chart_type=ChartType.BAR,
                title="Bar Chart",
                x_column=categorical_columns[0],
                y_column=numeric_columns[0],
                x_axis_label=categorical_columns[0],
                y_axis_label=numeric_columns[0]
            ))
            
            # Pie chart for single category
            if len(set(row[columns.index(categorical_columns[0])] for row in rows)) <= 10:
                configs.append(VisualizationConfig(
                    chart_type=ChartType.PIE,
                    title="Distribution",
                    group_by_column=categorical_columns[0],
                    value_column=numeric_columns[0]
                ))
        
        if date_columns and numeric_columns:
            # Line chart
            configs.append(VisualizationConfig(
                chart_type=ChartType.LINE,
                title="Time Series",
                x_column=date_columns[0],
                y_column=numeric_columns[0],
                x_axis_label=date_columns[0],
                y_axis_label=numeric_columns[0]
            ))
        
        return configs


class VisualizationGenerator:
    """Generates visualizations from processed data."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    async def create_chart(
        self,
        data: Any,
        config: VisualizationConfig
    ) -> Dict[str, Any]:
        """Create a chart from data using the provided configuration."""
        try:
            # For now, return a mock chart configuration
            # In production, this would generate actual charts using libraries like
            # matplotlib, plotly, or return configuration for frontend chart libraries
            
            chart_spec = {
                "type": config.chart_type.value,
                "title": config.title,
                "data": self._prepare_chart_data(data, config),
                "layout": {
                    "width": config.width,
                    "height": config.height,
                    "theme": config.theme,
                    "xAxis": {
                        "title": config.x_axis_label,
                        "show": True
                    },
                    "yAxis": {
                        "title": config.y_axis_label,
                        "show": True
                    },
                    "legend": {"show": config.show_legend},
                    "grid": {"show": config.show_grid}
                },
                "options": {
                    "interactive": config.interactive,
                    "showValues": config.show_values,
                    "colorScheme": config.color_scheme,
                    **config.custom_options
                }
            }
            
            return {
                "success": True,
                "chart_spec": chart_spec,
                "format": "plotly",  # or matplotlib, chartjs, etc.
                "data_points": len(chart_spec["data"]) if isinstance(chart_spec["data"], list) else 0
            }
            
        except Exception as e:
            self.logger.error(f"Chart creation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "chart_spec": None
            }
    
    def _prepare_chart_data(self, data: Any, config: VisualizationConfig) -> List[Dict[str, Any]]:
        """Prepare data for chart rendering."""
        if not isinstance(data, dict) or "rows" not in data or "columns" not in data:
            return []
        
        columns = data.get("columns", [])
        rows = data.get("rows", [])
        
        chart_data = []
        
        try:
            if config.chart_type in [ChartType.BAR, ChartType.COLUMN]:
                # Bar/Column chart data
                x_idx = columns.index(config.x_column) if config.x_column in columns else 0
                y_idx = columns.index(config.y_column) if config.y_column in columns else 1
                
                for row in rows:
                    if len(row) > max(x_idx, y_idx):
                        chart_data.append({
                            "x": row[x_idx],
                            "y": float(row[y_idx]) if row[y_idx] is not None else 0
                        })
            
            elif config.chart_type == ChartType.PIE:
                # Pie chart data - aggregate by group
                group_idx = columns.index(config.group_by_column) if config.group_by_column in columns else 0
                value_idx = columns.index(config.value_column) if config.value_column in columns else 1
                
                aggregated = {}
                for row in rows:
                    if len(row) > max(group_idx, value_idx):
                        group = row[group_idx]
                        value = float(row[value_idx]) if row[value_idx] is not None else 0
                        aggregated[group] = aggregated.get(group, 0) + value
                
                for label, value in aggregated.items():
                    chart_data.append({"label": label, "value": value})
            
            elif config.chart_type in [ChartType.LINE, ChartType.AREA]:
                # Line/Area chart data
                x_idx = columns.index(config.x_column) if config.x_column in columns else 0
                y_idx = columns.index(config.y_column) if config.y_column in columns else 1
                
                for row in rows:
                    if len(row) > max(x_idx, y_idx):
                        chart_data.append({
                            "x": row[x_idx],
                            "y": float(row[y_idx]) if row[y_idx] is not None else 0
                        })
                
                # Sort by x value for line charts
                try:
                    chart_data.sort(key=lambda item: item["x"])
                except:
                    pass
            
            elif config.chart_type == ChartType.SCATTER:
                # Scatter plot data
                x_idx = columns.index(config.x_column) if config.x_column in columns else 0
                y_idx = columns.index(config.y_column) if config.y_column in columns else 1
                
                for row in rows:
                    if len(row) > max(x_idx, y_idx):
                        chart_data.append({
                            "x": float(row[x_idx]) if row[x_idx] is not None else 0,
                            "y": float(row[y_idx]) if row[y_idx] is not None else 0
                        })
            
        except Exception as e:
            self.logger.error(f"Chart data preparation failed: {e}")
            return []
        
        return chart_data


class ResultProcessor:
    """
    Processes and formats results from MCP tool executions.
    
    Features:
    - Multiple output formats
    - Automatic visualization generation
    - Data transformation
    - Export capabilities
    - Processing rules
    """
    
    def __init__(self, settings = None):
        self.logger = setup_logging(self.__class__.__name__)
        self.metrics = get_metrics_collector()
        
        # Components
        self.data_transformer = DataTransformer()
        self.visualization_generator = VisualizationGenerator()
        
        # Processing rules
        self.processing_rules: List[ProcessingRule] = []
        
        # Processed results cache
        self.processed_results: Dict[str, ProcessedResult] = {}
        
        # Background processing queue
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.processing_task: Optional[asyncio.Task] = None
        
        self._initialize_default_rules()
        
        self.logger.info("ResultProcessor initialized")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default processing rules."""
        # Database query results
        db_rule = ProcessingRule(
            rule_id="database_results",
            name="Database Query Results",
            description="Process database query results with table formatting",
            tool_patterns=["*database*", "*sql*", "*query*"],
            data_types=["tabular"],
            auto_visualize=True,
            default_format=ResultFormat.TABLE,
            transformations=["table", "csv", "markdown"]
        )
        self.processing_rules.append(db_rule)
        
        # Analytics results
        analytics_rule = ProcessingRule(
            rule_id="analytics_results",
            name="Analytics Results",
            description="Process analytics results with automatic charts",
            tool_patterns=["*analyt*", "*insight*", "*metric*"],
            auto_visualize=True,
            default_format=ResultFormat.CHART,
            chart_config=VisualizationConfig(
                chart_type=ChartType.BAR,
                title="Analytics Results",
                interactive=True,
                show_legend=True
            ),
            transformations=["table", "chart", "markdown"]
        )
        self.processing_rules.append(analytics_rule)
        
        # ETL results
        etl_rule = ProcessingRule(
            rule_id="etl_results",
            name="ETL Processing Results",
            description="Process ETL operation results",
            tool_patterns=["*etl*", "*transform*", "*load*"],
            default_format=ResultFormat.JSON,
            transformations=["table", "markdown"]
        )
        self.processing_rules.append(etl_rule)
    
    async def process_result(
        self,
        execution_result: ExecutionResult,
        tool_definition: Optional[ToolDefinition] = None,
        requested_formats: Optional[List[ResultFormat]] = None,
        auto_visualize: bool = True
    ) -> ProcessedResult:
        """
        Process an execution result into multiple formats.
        
        Args:
            execution_result: Result from tool execution
            tool_definition: Tool definition for context
            requested_formats: Specific formats to generate
            auto_visualize: Whether to generate automatic visualizations
            
        Returns:
            Processed result with multiple formats
        """
        start_time = time.time()
        
        try:
            # Create processed result
            processed = ProcessedResult(
                execution_id=execution_result.execution_id,
                tool_id=execution_result.tool_id,
                raw_result=execution_result.result,
                processing_status=ProcessingStatus.PROCESSING
            )
            
            # Store in cache
            self.processed_results[processed.result_id] = processed
            
            # Find matching processing rules
            matching_rules = self._find_matching_rules(execution_result, tool_definition)
            processed.applied_rules = [rule.rule_id for rule in matching_rules]
            
            # Determine formats to generate
            formats_to_generate = set()
            
            if requested_formats:
                formats_to_generate.update(requested_formats)
            
            # Add formats from rules
            for rule in matching_rules:
                formats_to_generate.add(rule.default_format)
                if "table" in rule.transformations:
                    formats_to_generate.add(ResultFormat.TABLE)
                if "csv" in rule.transformations:
                    formats_to_generate.add(ResultFormat.CSV)
                if "markdown" in rule.transformations:
                    formats_to_generate.add(ResultFormat.MARKDOWN)
                if "chart" in rule.transformations:
                    formats_to_generate.add(ResultFormat.CHART)
            
            # Always include JSON as base format
            formats_to_generate.add(ResultFormat.JSON)
            
            # Generate formats
            for format_type in formats_to_generate:
                try:
                    formatted_result = await self._format_result(
                        execution_result.result, format_type
                    )
                    processed.formats[format_type] = formatted_result
                except Exception as e:
                    self.logger.error(f"Format generation failed for {format_type}: {e}")
            
            # Generate visualizations
            if auto_visualize:
                # Check rules for auto-visualization
                should_visualize = any(rule.auto_visualize for rule in matching_rules)
                
                if should_visualize:
                    await self._generate_visualizations(processed, matching_rules)
            
            # Calculate statistics
            await self._calculate_statistics(processed)
            
            # Update status
            processed.processing_status = ProcessingStatus.COMPLETED
            processed.processed_at = datetime.utcnow()
            processed.processing_time_ms = (time.time() - start_time) * 1000
            
            self.logger.info(
                f"Processed result {processed.result_id} with {len(processed.formats)} formats"
            )
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Result processing failed: {e}")
            
            # Create error result
            error_processed = ProcessedResult(
                execution_id=execution_result.execution_id,
                tool_id=execution_result.tool_id,
                raw_result=execution_result.result,
                processing_status=ProcessingStatus.FAILED,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            return error_processed
    
    def _find_matching_rules(
        self,
        execution_result: ExecutionResult,
        tool_definition: Optional[ToolDefinition]
    ) -> List[ProcessingRule]:
        """Find processing rules that match the execution result."""
        matching_rules = []
        
        for rule in self.processing_rules:
            if not rule.enabled:
                continue
            
            # Check tool patterns
            if rule.tool_patterns and tool_definition:
                tool_name = tool_definition.name.lower()
                if not any(self._pattern_match(pattern, tool_name) for pattern in rule.tool_patterns):
                    continue
            
            # Check result patterns (simplified)
            if rule.result_patterns:
                result_str = str(execution_result.result).lower()
                if not any(pattern.lower() in result_str for pattern in rule.result_patterns):
                    continue
            
            matching_rules.append(rule)
        
        # Sort by priority
        matching_rules.sort(key=lambda r: r.priority, reverse=True)
        
        return matching_rules
    
    def _pattern_match(self, pattern: str, text: str) -> bool:
        """Simple pattern matching with wildcards."""
        if "*" not in pattern:
            return pattern.lower() in text.lower()
        
        # Simple wildcard matching
        pattern_parts = pattern.lower().split("*")
        text_lower = text.lower()
        
        current_pos = 0
        for part in pattern_parts:
            if not part:
                continue
            
            pos = text_lower.find(part, current_pos)
            if pos == -1:
                return False
            current_pos = pos + len(part)
        
        return True
    
    async def _format_result(self, result: Any, format_type: ResultFormat) -> Any:
        """Format result into specific format type."""
        if format_type == ResultFormat.JSON:
            return json.loads(json.dumps(result, default=str))
        
        elif format_type == ResultFormat.TABLE:
            return await self.data_transformer.transform_to_table(result)
        
        elif format_type == ResultFormat.CSV:
            return await self.data_transformer.transform_to_csv(result)
        
        elif format_type == ResultFormat.MARKDOWN:
            return await self.data_transformer.transform_to_markdown(result)
        
        elif format_type == ResultFormat.HTML:
            return await self.data_transformer.transform_to_html(result)
        
        elif format_type == ResultFormat.TEXT:
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result, indent=2)
        
        else:
            # Default to JSON for unsupported formats
            return json.loads(json.dumps(result, default=str))
    
    async def _generate_visualizations(
        self,
        processed: ProcessedResult,
        rules: List[ProcessingRule]
    ) -> None:
        """Generate visualizations for the processed result."""
        try:
            # Detect chart opportunities
            chart_configs = self.data_transformer.detect_chart_opportunities(processed.raw_result)
            
            # Add rule-specified charts
            for rule in rules:
                if rule.chart_config:
                    chart_configs.append(rule.chart_config)
            
            # Generate charts
            for i, config in enumerate(chart_configs):
                chart_id = f"chart_{i}"
                
                chart_result = await self.visualization_generator.create_chart(
                    processed.raw_result, config
                )
                
                if chart_result["success"]:
                    processed.visualizations[chart_id] = {
                        "config": config,
                        "chart_spec": chart_result["chart_spec"],
                        "format": chart_result["format"],
                        "data_points": chart_result["data_points"]
                    }
                
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
    
    async def _calculate_statistics(self, processed: ProcessedResult) -> None:
        """Calculate statistics about the processed result."""
        try:
            # Check if we have table format
            if ResultFormat.TABLE in processed.formats:
                table_data = processed.formats[ResultFormat.TABLE]
                processed.row_count = table_data.get("total_rows", 0)
                processed.column_count = len(table_data.get("headers", []))
            
            # Calculate data size
            result_str = json.dumps(processed.raw_result, default=str)
            processed.data_size_bytes = len(result_str.encode('utf-8'))
            
        except Exception as e:
            self.logger.error(f"Statistics calculation failed: {e}")
    
    async def export_result(
        self,
        result_id: str,
        format_type: ResultFormat,
        export_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Export a processed result in specific format.
        
        Args:
            result_id: Processed result ID
            format_type: Export format
            export_options: Format-specific options
            
        Returns:
            Export result with data and metadata
        """
        if result_id not in self.processed_results:
            raise CoordinatorError(f"Processed result {result_id} not found")
        
        processed = self.processed_results[result_id]
        
        try:
            if format_type in processed.formats:
                # Use existing format
                export_data = processed.formats[format_type]
            else:
                # Generate format on demand
                export_data = await self._format_result(processed.raw_result, format_type)
            
            # Apply export options
            if export_options and format_type == ResultFormat.CSV:
                # CSV-specific options
                delimiter = export_options.get("delimiter", ",")
                if delimiter != ",":
                    # Re-generate with custom delimiter
                    export_data = await self._generate_csv_with_delimiter(
                        processed.raw_result, delimiter
                    )
            
            return {
                "success": True,
                "format": format_type.value,
                "data": export_data,
                "metadata": {
                    "result_id": result_id,
                    "execution_id": processed.execution_id,
                    "tool_id": processed.tool_id,
                    "row_count": processed.row_count,
                    "column_count": processed.column_count,
                    "data_size_bytes": processed.data_size_bytes,
                    "exported_at": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "format": format_type.value
            }
    
    async def _generate_csv_with_delimiter(self, data: Any, delimiter: str) -> str:
        """Generate CSV with custom delimiter."""
        table_data = await self.data_transformer.transform_to_table(data)
        
        import csv
        output = io.StringIO()
        writer = csv.writer(output, delimiter=delimiter)
        
        # Write headers
        writer.writerow(table_data["headers"])
        
        # Write rows
        for row in table_data["rows"]:
            writer.writerow(row)
        
        return output.getvalue()
    
    def add_processing_rule(self, rule: ProcessingRule) -> bool:
        """Add a new processing rule."""
        try:
            # Check for duplicate rule ID
            if any(r.rule_id == rule.rule_id for r in self.processing_rules):
                self.logger.warning(f"Processing rule {rule.rule_id} already exists")
                return False
            
            self.processing_rules.append(rule)
            self.processing_rules.sort(key=lambda r: r.priority, reverse=True)
            
            self.logger.info(f"Added processing rule: {rule.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add processing rule: {e}")
            return False
    
    def remove_processing_rule(self, rule_id: str) -> bool:
        """Remove a processing rule."""
        for i, rule in enumerate(self.processing_rules):
            if rule.rule_id == rule_id:
                del self.processing_rules[i]
                self.logger.info(f"Removed processing rule: {rule_id}")
                return True
        
        return False
    
    def get_processed_result(self, result_id: str) -> Optional[ProcessedResult]:
        """Get a processed result by ID."""
        return self.processed_results.get(result_id)
    
    def list_processed_results(
        self,
        tool_id: Optional[str] = None,
        status: Optional[ProcessingStatus] = None,
        limit: int = 100
    ) -> List[ProcessedResult]:
        """List processed results with optional filtering."""
        results = list(self.processed_results.values())
        
        # Filter by tool ID
        if tool_id:
            results = [r for r in results if r.tool_id == tool_id]
        
        # Filter by status
        if status:
            results = [r for r in results if r.processing_status == status]
        
        # Sort by creation time (newest first)
        results.sort(key=lambda r: r.created_at, reverse=True)
        
        return results[:limit]
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive processing metrics."""
        results = list(self.processed_results.values())
        
        if not results:
            return {"total_processed": 0}
        
        # Calculate metrics
        total_processed = len(results)
        completed = len([r for r in results if r.processing_status == ProcessingStatus.COMPLETED])
        failed = len([r for r in results if r.processing_status == ProcessingStatus.FAILED])
        
        avg_processing_time = sum(
            r.processing_time_ms for r in results 
            if r.processing_status == ProcessingStatus.COMPLETED
        ) / max(completed, 1)
        
        # Format usage stats
        format_usage = {}
        for result in results:
            for format_type in result.formats.keys():
                format_usage[format_type.value] = format_usage.get(format_type.value, 0) + 1
        
        # Visualization stats
        total_visualizations = sum(len(r.visualizations) for r in results)
        
        return {
            "total_processed": total_processed,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total_processed if total_processed > 0 else 0,
            "average_processing_time_ms": avg_processing_time,
            "format_usage": format_usage,
            "total_visualizations": total_visualizations,
            "processing_rules": len(self.processing_rules),
            "cache_size": len(self.processed_results)
        }
    
    def cleanup_old_results(self, max_age_hours: int = 24) -> int:
        """Clean up old processed results."""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        old_result_ids = [
            result_id for result_id, result in self.processed_results.items()
            if result.created_at < cutoff_time
        ]
        
        for result_id in old_result_ids:
            del self.processed_results[result_id]
        
        self.logger.info(f"Cleaned up {len(old_result_ids)} old processed results")
        
        return len(old_result_ids) 