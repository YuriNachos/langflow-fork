from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

from langflow.services.base import Service
from langflow.services.task.backends.anyio import AnyIOBackend
from langflow.services.task.backends.celery import CeleryBackend
from uuid import uuid4

if TYPE_CHECKING:
    from fastapi import BackgroundTasks
    from lfx.services.settings.service import SettingsService

    from langflow.services.task.backends.base import TaskBackend


class TaskService(Service):
    name = "task_service"

    def __init__(self, settings_service: SettingsService):
        self.settings_service = settings_service
        self.use_celery = self.settings_service.settings.celery_enabled
        self.backend = self.get_backend()

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def get_backend(self) -> TaskBackend:
        if self.use_celery:
            return CeleryBackend()
        return AnyIOBackend()

    async def fire_and_forget_task(
        self, task_func: Callable[..., Any], background_tasks: BackgroundTasks, *args: Any, **kwargs: Any
    ) -> str:
        """Launch a task in the background and forget about it. (Edge case function for background tasks)

        Note: This is required since AnyIOBackend does not support background tasks and is blocking.

        This method abstracts the background execution. If Celery is enabled,
        it uses the distributed queue. Otherwise, it offloads to FastAPI's
        BackgroundTasks for immediate response.

        Args:
            task_func: The task function to launch.
            background_tasks: FastAPI background tasks to handle local async execution.
            *args: Positional arguments (typically 'run_graph_internal')
            **kwargs: Keyword arguments for the task function.

        Returns:
            str: A task_id that can be used to track the task if needed.
        """
        if self.use_celery:
            task_id, _ = self.backend.launch_task(task_func, *args, **kwargs)
            return task_id

        task_id = str(uuid4())
        background_tasks.add_task(task_func, *args, **kwargs)
        return task_id
    

    # In your TaskService class
    async def launch_and_await_task(
        self,
        task_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return await task_func(*args, **kwargs)

    async def launch_task(self, task_func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        task = self.backend.launch_task(task_func, *args, **kwargs)
        return await task if isinstance(task, Coroutine) else task
