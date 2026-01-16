from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from celery.result import AsyncResult

from langflow.services.task.backends.base import TaskBackend

if TYPE_CHECKING:
    from langflow.worker import celery_app
    from celery import Task


class CeleryBackend(TaskBackend):
    name = "celery"

    def __init__(self) -> None:
        from langflow.worker import celery_app

        self.celery_app = celery_app

    def launch_task(self, task_func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[str, Any]:
        from celery.result import AsyncResult

        # I need to type the delay method to make it easier
        if not hasattr(task_func, "delay"):
            msg = f"Task function {task_func} does not have a delay method"
            raise ValueError(msg)
        task = task_func.delay(*args, **kwargs)
        return task.id, AsyncResult(task.id, app=self.celery_app)

    def get_task(self, task_id: str) -> Any:
        from celery.result import AsyncResult

        return AsyncResult(task_id, app=self.celery_app)

    def get_task_status(self, task_id: str) -> str:
        task = self.get_task(task_id)
        state_map = {
            "PENDING": JobStatus.QUEUED,
            "STARTED": JobStatus.IN_PROGRESS,
            "RETRY": JobStatus.IN_PROGRESS,
            "SUCCESS": JobStatus.COMPLETED,
            "FAILURE": JobStatus.FAILED,
            "REVOKED": JobStatus.ERROR,
        }
        return state_map.get(task.state, JobStatus.QUEUED)
