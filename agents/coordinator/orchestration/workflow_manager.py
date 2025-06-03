"""
Multi-agent workflow management for the coordinator agent.

This module provides the WorkflowManager class that orchestrates complex 
multi-agent workflows, manages workflow execution, and handles inter-agent
communication patterns.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, ConfigDict
import google.genai as genai

from shared.schemas.agent_communication import (
    AgentMessage, AgentRole, MessageType, ConversationContext
)
from shared.config.logging_config import setup_logging
from shared.utils.metrics import get_metrics_collector


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowPriority(str, Enum):
    """Workflow priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class StepType(str, Enum):
    """Types of workflow steps."""
    AGENT_CALL = "agent_call"
    PARALLEL_CALLS = "parallel_calls"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    WAIT = "wait"
    MERGE = "merge"
    TRANSFORM = "transform"


@dataclass
class WorkflowStep:
    """Individual workflow step definition."""
    step_id: str
    step_type: StepType
    agent_role: Optional[AgentRole] = None
    input_template: Optional[str] = None
    output_mapping: Optional[Dict[str, str]] = None
    condition: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    parallel_steps: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout_seconds: int = 300
    
    # Execution state
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: int = 0


class WorkflowTemplate(BaseModel):
    """Workflow template definition."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Human-readable template name")
    description: str = Field(..., description="Template description")
    version: str = Field(default="1.0", description="Template version")
    
    # Workflow structure
    steps: Dict[str, WorkflowStep] = Field(..., description="Workflow steps")
    start_step: str = Field(..., description="Initial step ID")
    end_steps: Set[str] = Field(..., description="Terminal step IDs")
    
    # Configuration
    max_execution_time: int = Field(default=3600, description="Max execution time in seconds")
    max_retries: int = Field(default=3, description="Max workflow retries")
    allow_parallel: bool = Field(default=True, description="Allow parallel execution")
    
    # Metadata
    tags: Set[str] = Field(default_factory=set, description="Template tags")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowInstance(BaseModel):
    """Running workflow instance."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    instance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    template_id: str = Field(..., description="Source template ID")
    conversation_id: str = Field(..., description="Associated conversation")
    
    # Execution state
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    priority: WorkflowPriority = Field(default=WorkflowPriority.NORMAL)
    current_step: Optional[str] = Field(default=None)
    completed_steps: Set[str] = Field(default_factory=set)
    failed_steps: Set[str] = Field(default_factory=set)
    
    # Context and data
    input_context: Dict[str, Any] = Field(default_factory=dict)
    step_outputs: Dict[str, Any] = Field(default_factory=dict)
    global_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metrics
    total_steps: int = Field(default=0)
    execution_time: Optional[float] = None
    retry_count: int = Field(default=0)


class WorkflowManager:
    """
    Manages multi-agent workflow orchestration.
    
    Features:
    - Workflow template management
    - Instance execution and tracking
    - Parallel and sequential execution
    - Error handling and retries
    - Performance monitoring
    """
    
    def __init__(
        self,
        agent_router: Optional[object] = None,
        context_aggregator: Optional[object] = None,
        settings = None
    ):
        self.logger = setup_logging(self.__class__.__name__)
        self.agent_router = agent_router
        self.context_aggregator = context_aggregator
        self.metrics = get_metrics_collector()
        
        # Templates and instances
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.instances: Dict[str, WorkflowInstance] = {}
        self.running_instances: Set[str] = set()
        
        # Execution management
        self.execution_tasks: Dict[str, asyncio.Task] = {}
        self.step_handlers: Dict[StepType, Callable] = {
            StepType.AGENT_CALL: self._execute_agent_call,
            StepType.PARALLEL_CALLS: self._execute_parallel_calls,
            StepType.CONDITIONAL: self._execute_conditional,
            StepType.LOOP: self._execute_loop,
            StepType.WAIT: self._execute_wait,
            StepType.MERGE: self._execute_merge,
            StepType.TRANSFORM: self._execute_transform,
        }
        
        # Performance tracking
        self.performance_metrics = {
            'workflows_executed': 0,
            'workflows_failed': 0,
            'avg_execution_time': 0.0,
            'total_steps_executed': 0,
            'parallel_executions': 0
        }
        
        self.logger.info("WorkflowManager initialized")
    
    async def register_template(self, template: WorkflowTemplate) -> bool:
        """
        Register a new workflow template.
        
        Args:
            template: Workflow template to register
            
        Returns:
            True if registration successful
        """
        try:
            # Validate template
            self._validate_template(template)
            
            # Store template
            self.templates[template.template_id] = template
            
            self.logger.info(f"Workflow template registered: {template.template_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register template {template.template_id}: {e}")
            return False
    
    def _validate_template(self, template: WorkflowTemplate) -> None:
        """Validate workflow template structure."""
        # Check start step exists
        if template.start_step not in template.steps:
            raise ValueError(f"Start step '{template.start_step}' not found in steps")
        
        # Check end steps exist
        for end_step in template.end_steps:
            if end_step not in template.steps:
                raise ValueError(f"End step '{end_step}' not found in steps")
        
        # Validate dependencies
        for step_id, step in template.steps.items():
            for dep in step.dependencies:
                if dep not in template.steps:
                    raise ValueError(f"Dependency '{dep}' not found for step '{step_id}'")
            
            # Check parallel steps
            for parallel_step in step.parallel_steps:
                if parallel_step not in template.steps:
                    raise ValueError(f"Parallel step '{parallel_step}' not found for step '{step_id}'")
    
    async def create_instance(
        self,
        template_id: str,
        conversation_id: str,
        input_context: Optional[Dict[str, Any]] = None,
        priority: WorkflowPriority = WorkflowPriority.NORMAL
    ) -> Optional[str]:
        """
        Create a new workflow instance.
        
        Args:
            template_id: Template to use
            conversation_id: Associated conversation
            input_context: Initial context data
            priority: Execution priority
            
        Returns:
            Instance ID if created successfully
        """
        try:
            if template_id not in self.templates:
                raise ValueError(f"Template '{template_id}' not found")
            
            template = self.templates[template_id]
            
            # Create instance
            instance = WorkflowInstance(
                template_id=template_id,
                conversation_id=conversation_id,
                input_context=input_context or {},
                priority=priority,
                total_steps=len(template.steps)
            )
            
            # Store instance
            self.instances[instance.instance_id] = instance
            
            self.logger.info(f"Workflow instance created: {instance.instance_id}")
            return instance.instance_id
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow instance: {e}")
            return None
    
    async def execute_workflow(self, instance_id: str) -> bool:
        """
        Execute a workflow instance.
        
        Args:
            instance_id: Instance to execute
            
        Returns:
            True if execution started successfully
        """
        try:
            if instance_id not in self.instances:
                raise ValueError(f"Instance '{instance_id}' not found")
            
            instance = self.instances[instance_id]
            template = self.templates[instance.template_id]
            
            # Check if already running
            if instance_id in self.running_instances:
                self.logger.warning(f"Instance {instance_id} already running")
                return False
            
            # Start execution
            instance.status = WorkflowStatus.RUNNING
            instance.started_at = datetime.utcnow()
            self.running_instances.add(instance_id)
            
            # Create execution task
            task = asyncio.create_task(
                self._execute_workflow_steps(instance_id)
            )
            self.execution_tasks[instance_id] = task
            
            self.logger.info(f"Workflow execution started: {instance_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start workflow execution: {e}")
            return False
    
    async def _execute_workflow_steps(self, instance_id: str) -> None:
        """Execute workflow steps for an instance."""
        try:
            instance = self.instances[instance_id]
            template = self.templates[instance.template_id]
            
            # Start from initial step
            current_steps = {template.start_step}
            
            while current_steps and instance.status == WorkflowStatus.RUNNING:
                next_steps = set()
                
                # Execute current steps
                for step_id in current_steps:
                    if step_id in instance.completed_steps:
                        continue
                    
                    step = template.steps[step_id]
                    
                    # Check dependencies
                    if not self._check_dependencies(step, instance):
                        continue
                    
                    # Execute step
                    success = await self._execute_step(instance_id, step_id)
                    
                    if success:
                        instance.completed_steps.add(step_id)
                        
                        # Add next steps
                        for next_step_id, next_step in template.steps.items():
                            if step_id in next_step.dependencies:
                                next_steps.add(next_step_id)
                    else:
                        instance.failed_steps.add(step_id)
                        
                        # Handle retry logic
                        if step.attempts < step.retry_count:
                            next_steps.add(step_id)  # Retry
                        else:
                            # Workflow failed
                            instance.status = WorkflowStatus.FAILED
                            break
                
                # Move to next steps
                current_steps = next_steps
                
                # Check completion
                if any(step_id in instance.completed_steps for step_id in template.end_steps):
                    instance.status = WorkflowStatus.COMPLETED
                    break
            
            # Finalize execution
            await self._finalize_execution(instance_id)
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            instance = self.instances[instance_id]
            instance.status = WorkflowStatus.FAILED
            await self._finalize_execution(instance_id)
    
    def _check_dependencies(self, step: WorkflowStep, instance: WorkflowInstance) -> bool:
        """Check if step dependencies are satisfied."""
        for dep in step.dependencies:
            if dep not in instance.completed_steps:
                return False
        return True
    
    async def _execute_step(self, instance_id: str, step_id: str) -> bool:
        """Execute a single workflow step."""
        try:
            instance = self.instances[instance_id]
            template = self.templates[instance.template_id]
            step = template.steps[step_id]
            
            # Update step state
            step.status = WorkflowStatus.RUNNING
            step.started_at = datetime.utcnow()
            step.attempts += 1
            instance.current_step = step_id
            
            self.logger.debug(f"Executing step {step_id} (attempt {step.attempts})")
            
            # Execute based on step type
            handler = self.step_handlers.get(step.step_type)
            if not handler:
                raise ValueError(f"Unknown step type: {step.step_type}")
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    handler(instance_id, step_id),
                    timeout=step.timeout_seconds
                )
                
                # Store result
                step.result = result
                step.status = WorkflowStatus.COMPLETED
                step.completed_at = datetime.utcnow()
                instance.step_outputs[step_id] = result
                
                self.logger.debug(f"Step {step_id} completed successfully")
                return True
                
            except asyncio.TimeoutError:
                step.error = f"Step timed out after {step.timeout_seconds} seconds"
                step.status = WorkflowStatus.FAILED
                self.logger.error(f"Step {step_id} timed out")
                return False
            
        except Exception as e:
            step.error = str(e)
            step.status = WorkflowStatus.FAILED
            self.logger.error(f"Step {step_id} failed: {e}")
            return False
    
    async def _execute_agent_call(self, instance_id: str, step_id: str) -> Any:
        """Execute agent call step."""
        instance = self.instances[instance_id]
        template = self.templates[instance.template_id]
        step = template.steps[step_id]
        
        if not step.agent_role:
            raise ValueError(f"Agent role not specified for step {step_id}")
        
        # Prepare input
        input_data = self._prepare_step_input(instance, step)
        
        # Create agent message
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            conversation_id=instance.conversation_id,
            sender_role=AgentRole.COORDINATOR,
            target_role=step.agent_role,
            message_type=MessageType.QUERY,
            content=input_data,
            context=instance.global_context
        )
        
        # Route to agent
        if self.agent_router:
            response = await self.agent_router.route_message(message)
            return response.content if response else None
        
        return None
    
    async def _execute_parallel_calls(self, instance_id: str, step_id: str) -> Dict[str, Any]:
        """Execute parallel agent calls."""
        instance = self.instances[instance_id]
        template = self.templates[instance.template_id]
        step = template.steps[step_id]
        
        # Execute parallel steps
        tasks = []
        for parallel_step_id in step.parallel_steps:
            if parallel_step_id in template.steps:
                task = asyncio.create_task(
                    self._execute_step(instance_id, parallel_step_id)
                )
                tasks.append((parallel_step_id, task))
        
        # Wait for all tasks
        results = {}
        for step_id, task in tasks:
            try:
                result = await task
                results[step_id] = result
            except Exception as e:
                results[step_id] = {"error": str(e)}
        
        self.performance_metrics['parallel_executions'] += 1
        return results
    
    async def _execute_conditional(self, instance_id: str, step_id: str) -> Any:
        """Execute conditional step."""
        instance = self.instances[instance_id]
        template = self.templates[instance.template_id]
        step = template.steps[step_id]
        
        if not step.condition:
            raise ValueError(f"Condition not specified for step {step_id}")
        
        # Evaluate condition (simplified)
        condition_result = self._evaluate_condition(step.condition, instance)
        
        return {"condition_met": condition_result}
    
    async def _execute_loop(self, instance_id: str, step_id: str) -> Any:
        """Execute loop step."""
        # Simplified loop implementation
        return {"loop_completed": True}
    
    async def _execute_wait(self, instance_id: str, step_id: str) -> Any:
        """Execute wait step."""
        instance = self.instances[instance_id]
        template = self.templates[instance.template_id]
        step = template.steps[step_id]
        
        # Extract wait time from input
        wait_time = step.input_template or "1"
        try:
            await asyncio.sleep(float(wait_time))
        except ValueError:
            await asyncio.sleep(1)
        
        return {"waited": True}
    
    async def _execute_merge(self, instance_id: str, step_id: str) -> Any:
        """Execute merge step to combine outputs."""
        instance = self.instances[instance_id]
        template = self.templates[instance.template_id]
        step = template.steps[step_id]
        
        # Merge outputs from dependencies
        merged_data = {}
        for dep_step_id in step.dependencies:
            if dep_step_id in instance.step_outputs:
                merged_data[dep_step_id] = instance.step_outputs[dep_step_id]
        
        if self.context_aggregator:
            aggregated = await self.context_aggregator.aggregate_responses(
                list(merged_data.values())
            )
            return aggregated
        
        return merged_data
    
    async def _execute_transform(self, instance_id: str, step_id: str) -> Any:
        """Execute data transformation step."""
        instance = self.instances[instance_id]
        template = self.templates[instance.template_id]
        step = template.steps[step_id]
        
        # Apply output mapping transformation
        input_data = self._prepare_step_input(instance, step)
        
        if step.output_mapping:
            transformed = {}
            for key, mapping in step.output_mapping.items():
                # Simple mapping implementation
                if mapping in input_data:
                    transformed[key] = input_data[mapping]
            return transformed
        
        return input_data
    
    def _prepare_step_input(self, instance: WorkflowInstance, step: WorkflowStep) -> Dict[str, Any]:
        """Prepare input data for step execution."""
        input_data = instance.input_context.copy()
        
        # Add dependency outputs
        for dep_step_id in step.dependencies:
            if dep_step_id in instance.step_outputs:
                input_data[f"step_{dep_step_id}"] = instance.step_outputs[dep_step_id]
        
        # Add global context
        input_data.update(instance.global_context)
        
        return input_data
    
    def _evaluate_condition(self, condition: str, instance: WorkflowInstance) -> bool:
        """Evaluate a condition string (simplified)."""
        # Simple condition evaluation
        # In production, this would use a proper expression evaluator
        try:
            # Basic conditions like "step_X.success == true"
            if "success" in condition:
                return True
            return False
        except Exception:
            return False
    
    async def _finalize_execution(self, instance_id: str) -> None:
        """Finalize workflow execution."""
        try:
            instance = self.instances[instance_id]
            
            # Update timing
            instance.completed_at = datetime.utcnow()
            if instance.started_at:
                instance.execution_time = (
                    instance.completed_at - instance.started_at
                ).total_seconds()
            
            # Update metrics
            self.performance_metrics['workflows_executed'] += 1
            if instance.status == WorkflowStatus.FAILED:
                self.performance_metrics['workflows_failed'] += 1
            
            self.performance_metrics['total_steps_executed'] += len(instance.completed_steps)
            
            if instance.execution_time:
                # Update average execution time
                total_workflows = self.performance_metrics['workflows_executed']
                current_avg = self.performance_metrics['avg_execution_time']
                self.performance_metrics['avg_execution_time'] = (
                    (current_avg * (total_workflows - 1) + instance.execution_time) / total_workflows
                )
            
            # Cleanup
            self.running_instances.discard(instance_id)
            self.execution_tasks.pop(instance_id, None)
            
            self.logger.info(
                f"Workflow {instance_id} finalized: {instance.status.value} "
                f"({len(instance.completed_steps)}/{instance.total_steps} steps)"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to finalize workflow {instance_id}: {e}")
    
    async def cancel_workflow(self, instance_id: str) -> bool:
        """Cancel a running workflow."""
        try:
            if instance_id not in self.instances:
                return False
            
            instance = self.instances[instance_id]
            
            # Cancel if running
            if instance_id in self.running_instances:
                instance.status = WorkflowStatus.CANCELLED
                
                # Cancel execution task
                if instance_id in self.execution_tasks:
                    task = self.execution_tasks[instance_id]
                    task.cancel()
                
                await self._finalize_execution(instance_id)
            
            self.logger.info(f"Workflow {instance_id} cancelled")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel workflow {instance_id}: {e}")
            return False
    
    def get_instance_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow instance status."""
        if instance_id not in self.instances:
            return None
        
        instance = self.instances[instance_id]
        
        return {
            "instance_id": instance.instance_id,
            "template_id": instance.template_id,
            "status": instance.status.value,
            "priority": instance.priority.value,
            "current_step": instance.current_step,
            "progress": {
                "completed_steps": len(instance.completed_steps),
                "total_steps": instance.total_steps,
                "failed_steps": len(instance.failed_steps)
            },
            "timing": {
                "created_at": instance.created_at.isoformat(),
                "started_at": instance.started_at.isoformat() if instance.started_at else None,
                "completed_at": instance.completed_at.isoformat() if instance.completed_at else None,
                "execution_time": instance.execution_time
            },
            "retry_count": instance.retry_count
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get workflow performance metrics."""
        return {
            "templates_registered": len(self.templates),
            "instances_created": len(self.instances),
            "currently_running": len(self.running_instances),
            "performance": self.performance_metrics.copy()
        }
    
    async def cleanup_completed_instances(self, older_than_hours: int = 24) -> int:
        """Clean up old completed instances."""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        cleaned_count = 0
        
        instances_to_remove = []
        for instance_id, instance in self.instances.items():
            if (instance.status in {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED} and
                instance.completed_at and instance.completed_at < cutoff_time):
                instances_to_remove.append(instance_id)
        
        for instance_id in instances_to_remove:
            del self.instances[instance_id]
            cleaned_count += 1
        
        self.logger.info(f"Cleaned up {cleaned_count} old workflow instances")
        return cleaned_count 