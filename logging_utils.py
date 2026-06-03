import logging
from contextvars import ContextVar

# Context variables to store request info for the current request
user_id_ctx = ContextVar("user_id", default="system")
request_id_ctx = ContextVar("request_id", default="none")

class RequestContextFilter(logging.Filter):
    """
    Logging filter that injects the current user_id and request_id 
    from contextvars into the log record.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.user_id = user_id_ctx.get()
        record.request_id = request_id_ctx.get()
        return True
