"""V2 Workflow execution endpoints.

This module implements the V2 Workflow API endpoints for executing flows with
enhanced error handling, timeout protection, and structured responses.

Endpoints:
    POST /workflow: Execute a workflow (sync, stream, or background modes)
    GET /workflow: Get workflow job status by job_id
    POST /workflow/stop: Stop a running workflow execution

Features:
    - Developer API protection (requires developer_api_enabled setting)
    - Comprehensive error handling with structured error responses
    - Timeout protection for long-running executions
    - Support for multiple execution modes (sync, stream, background)
    - API key authentication required for all endpoints

Configuration:
    EXECUTION_TIMEOUT: Maximum execution time for synchronous workflows (300 seconds)
"""

from __future__ import annotations

import asyncio
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from lfx.graph.graph.base import Graph
from lfx.schema.workflow import (
    WORKFLOW_EXECUTION_RESPONSES,
    WORKFLOW_STATUS_RESPONSES,
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    WorkflowJobResponse,
    WorkflowStopRequest,
    WorkflowStopResponse,
    JobStatus,
)
from lfx.services.deps import get_settings_service

from langflow.api.v1.schemas import RunResponse
from langflow.api.v2.converters import (
    create_error_response,
    parse_flat_inputs,
    run_response_to_workflow_response,
)
from lfx.log.logger import logger
from langflow.services.deps import get_task_service, get_queue_service
from langflow.helpers.flow import get_flow_by_id_or_endpoint_name
from langflow.processing.process import process_tweaks, run_graph_internal
from langflow.services.auth.utils import api_key_security
from langflow.services.database.models.flow.model import FlowRead
from langflow.services.database.models.user.model import UserRead
from lfx.schema.workflow import (
    WORKFLOW_EXECUTION_RESPONSES,
    WORKFLOW_STATUS_RESPONSES,
    WorkflowExecutionRequest,
    WorkflowExecutionResponse,
    WorkflowJobResponse,
    WorkflowStopRequest,
    WorkflowStopResponse,
    JobStatus,
    ErrorDetail
)

# Configuration constants
EXECUTION_TIMEOUT = 300  # 5 minutes default timeout for sync execution


def check_developer_api_enabled() -> None:
    """Check if developer API is enabled.

    This dependency function protects all workflow endpoints by verifying that
    the developer API feature is enabled in the application settings.

    Raises:
        HTTPException: 403 Forbidden if developer_api_enabled setting is False

    Note:
        This is used as a router-level dependency to protect all workflow endpoints.
    """
    settings = get_settings_service().settings
    if not settings.developer_api_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Developer API disabled",
                "code": "DEVELOPER_API_DISABLED",
                "message": "Developer API is not enabled. Contact administrator to enable this feature.",
            },
        )

router = APIRouter(prefix="/workflows", tags=["Workflow"], dependencies=[Depends(check_developer_api_enabled)])

async def execute_sync_workflow_with_timeout(
    workflow_request: WorkflowExecutionRequest,
    flow: FlowRead,
    job_id: str,
    api_key_user: UserRead,
    background_tasks: BackgroundTasks,
) -> WorkflowExecutionResponse:
    """Execute workflow with timeout protection.

    Args:
        workflow_request: The workflow execution request
        flow: The flow to execute
        job_id: Generated job ID for tracking
        api_key_user: Authenticated user
        background_tasks: FastAPI background tasks

    Returns:
        WorkflowExecutionResponse with complete results

    Raises:
        WorkflowTimeoutError: If execution exceeds timeout
        WorkflowValidationError: If flow validation fails
    """
    try:
        return await asyncio.wait_for(
            execute_sync_workflow(
                workflow_request=workflow_request,
                flow=flow,
                job_id=job_id,
                api_key_user=api_key_user,
                background_tasks=background_tasks,
            ),
            timeout=EXECUTION_TIMEOUT,
        )
    except asyncio.TimeoutError as e:
        msg = f"Execution exceeded {EXECUTION_TIMEOUT} seconds"
        raise WorkflowTimeoutError(msg) from e


async def execute_sync_workflow(
    workflow_request: WorkflowExecutionRequest,
    flow: FlowRead,
    job_id: str,
    api_key_user: UserRead,
    background_tasks: BackgroundTasks,  # noqa: ARG001
) -> WorkflowExecutionResponse:
    """Execute workflow synchronously and return complete results.

    This function implements a two-tier error handling strategy:
        1. System-level errors (validation, graph build): Raised as exceptions
        2. Component execution errors: Returned in response body with HTTP 200

    This approach allows clients to receive partial results even when some
    components fail, which is useful for debugging and incremental processing.

    Execution Flow:
        1. Parse flat inputs into tweaks and session_id
        2. Validate flow data exists
        3. Build graph from flow data with tweaks applied
        4. Identify terminal nodes for execution
        5. Execute graph and collect results
        6. Convert V1 RunResponse to V2 WorkflowExecutionResponse

    Args:
        workflow_request: The workflow execution request with inputs and configuration
        flow: The flow model from database
        job_id: Generated job ID for tracking this execution
        api_key_user: Authenticated user for permission checks
        background_tasks: FastAPI background tasks (unused in sync mode)

    Returns:
        WorkflowExecutionResponse: Complete execution results with outputs and metadata

    Raises:
        WorkflowValidationError: If flow data is None or graph build fails
    """
    # Parse flat inputs structure
    tweaks, session_id = parse_flat_inputs(workflow_request.inputs or {})

    # Validate flow data - this is a system error, not execution error
    if flow.data is None:
        msg = f"Flow {flow.id} has no data. The flow may be corrupted."
        raise WorkflowValidationError(msg)

    # Build graph - system error if this fails
    try:
        flow_id_str = str(flow.id)
        user_id = str(api_key_user.id)
        graph_data = flow.data.copy()
        graph_data = process_tweaks(graph_data, tweaks, stream=False)
        graph = Graph.from_payload(graph_data, flow_id=flow_id_str, user_id=user_id, flow_name=flow.name)
    except Exception as e:
        msg = f"Failed to build graph from flow data: {e!s}"
        raise WorkflowValidationError(msg) from e

    # Get terminal nodes - these are the outputs we want
    terminal_node_ids = graph.get_terminal_nodes()

    # Execute graph - component errors are caught and returned in response body
    try:
        task_result, execution_session_id = await run_graph_internal(
            graph=graph,
            flow_id=flow_id_str,
            session_id=session_id,
            inputs=None,
            outputs=terminal_node_ids,
            stream=False,
        )

        # Build RunResponse
        run_response = RunResponse(outputs=task_result, session_id=execution_session_id)

        # Convert to WorkflowExecutionResponse
        return run_response_to_workflow_response(
            run_response=run_response,
            flow_id=workflow_request.flow_id,
            job_id=job_id,
            workflow_request=workflow_request,
            graph=graph,
        )

    except asyncio.CancelledError:
        # Re-raise CancelledError to allow timeout mechanism to work properly
        # This ensures asyncio.wait_for() can properly cancel and raise TimeoutError
        raise
    except Exception as exc:  # noqa: BLE001
        # Component execution errors - return in response body with HTTP 200
        # This allows partial results and detailed error information per component
        return create_error_response(
            flow_id=workflow_request.flow_id,
            job_id=job_id,
            workflow_request=workflow_request,
            error=exc,
        )


async def execute_workflow_background(
    workflow_request: WorkflowExecutionRequest,
    flow: Flow,
    api_key_user: UserRead
) -> WorkflowJobResponse:
    """Execute workflow in the background and return job ID for the user to track the execution status."""
    try:
        # Parse flat inputs structure
        tweaks, session_id = parse_flat_inputs(workflow_request.inputs or {})

        # Validate flow data
        if flow.data is None:
            msg = f"Flow {flow.id} has no data"
            raise ValueError(msg)

        # Build the graph once
        flow_id_str = str(flow.id)
        user_id = str(api_key_user.id)
        graph_data = flow.data.copy()
        graph_data = process_tweaks(graph_data, tweaks, stream=False)
        graph = Graph.from_payload(graph_data, flow_id=flow_id_str, user_id=user_id, flow_name=flow.name)
        
        # Get terminal nodes
        terminal_node_ids = graph.get_terminal_nodes()

        # Launch background task
        task_service = get_task_service()
        job_id = await task_service.fire_and_forget_task(
            run_graph_internal,
            graph=graph,
            flow_id=flow_id_str,
            session_id=session_id,
            inputs=None,
            outputs=terminal_node_ids,
            stream=False,
        )
        status = JobStatus.QUEUED
        return WorkflowJobResponse(
            job_id=job_id,
            status=status
        )
    
    except Exception as exc:  # noqa: BLE001
        return create_error_response(
            flow_id=workflow_request.flow_id,
            job_id=None,
            workflow_request=workflow_request,
            error=exc,
        )


@router.post(
    "",
    response_model=None,
    response_model_exclude_none=True,
    responses=WORKFLOW_EXECUTION_RESPONSES,
    summary="Execute Workflow",
    description="Execute a workflow with support for sync, stream, and background modes",
)
async def execute_workflow(
    workflow_request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    api_key_user: Annotated[UserRead, Depends(api_key_security)],
) -> WorkflowExecutionResponse | WorkflowJobResponse | StreamingResponse:
    """Execute a workflow with multiple execution modes.

    - **sync**: Returns complete results immediately (background=False, stream=False)
    - **stream**: Returns server-sent events in real-time (stream=True)
    - **background**: Starts job and returns job ID immediately (background=True)
    """
    # Validate flow exists and user has permission
    flow = await get_flow_by_id_or_endpoint_name(workflow_request.flow_id, api_key_user.id)
    if not flow:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Flow identifier {workflow_request.flow_id} not found"
        )

    # Phase 1: Sync mode only (stream=false, background=false)
    if not workflow_request.stream and not workflow_request.background:
        # Generate job_id for tracking
        job_id = str(uuid4())
        return await execute_sync_workflow(
            workflow_request=workflow_request,
            flow=flow,
            job_id=job_id,
            api_key_user=api_key_user,
            background_tasks=background_tasks,
        )

    # Phase 2: Background mode (to be implemented)
    if workflow_request.background:
        try:
            return await execute_workflow_background(
                workflow_request=workflow_request,
                flow=flow,
                api_key_user=api_key_user
            )
        except Exception as e:
            print(e)
            # logger.aerror("Failed to queue workflow task in background mode", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while starting the background task.",
            )

    # Phase 3: Streaming mode (to be implemented)
    # This should never be reached due to the conditions above, but included for completeness
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Streaming execution not yet available.")


@router.get(
    "",
    response_model=None,
    response_model_exclude_none=True,
    responses=WORKFLOW_STATUS_RESPONSES,
    summary="Get Workflow Status",
    description="Get status of workflow job by job ID",
)
async def get_workflow_status(
    api_key_user: Annotated[UserRead, Depends(api_key_security)],  # noqa: ARG001
    job_id: Annotated[str, Query(description="Job ID to query")],
) -> WorkflowJobResponse:
    """Get workflow job status and results by job ID."""
    task_service = get_task_service()
    
    status_val = await task_service.get_task_status(job_id)
    # Check if we have error info to return
    errors = []
    if status_val in [JobStatus.FAILED, JobStatus.ERROR]:
        result = await task_service.get_task_result(job_id)
        if isinstance(result, str):
            errors.append(ErrorDetail(error=result, code=status_val.upper()))

    return WorkflowJobResponse(
        job_id=job_id,
        status=status_val,
        errors=errors
    )


@router.post(
    "/stop",
    response_model=WorkflowStopResponse,
    summary="Stop Workflow",
    description="Stop a running workflow execution",
)
async def stop_workflow(
    request: WorkflowStopRequest,
    api_key_user: Annotated[UserRead, Depends(api_key_security)],  # noqa: ARG001
) -> WorkflowStopResponse:
    """Stop a running workflow execution by job_id."""
    task_service = get_task_service()
    
    try:
        # Check current status
        current_status = await task_service.get_task_status(request.job_id)
        
        if current_status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            return WorkflowStopResponse(
                job_id=request.job_id,
                status="error",
                message=f"Job {request.job_id} already finished with status {current_status}"
            )

        # Trigger cleanup/cancel
        if not task_service.use_celery:
            job_queue_service = get_queue_service()
            await job_queue_service.cleanup_job(request.job_id)
        else:
            # Celery revoke logic
            from langflow.worker import celery_app
            celery_app.control.revoke(request.job_id, terminate=True)

        return WorkflowStopResponse(
            job_id=request.job_id,
            status="stopped",
            message=f"Stop request sent for job {request.job_id}"
        )
    except Exception as e:
        return WorkflowStopResponse(
            job_id=request.job_id,
            status="error",
            message=str(e)
        )
