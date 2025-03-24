# llm_docs_logger.py
import logging
import os
import warnings
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Set

from rich.console import Console
from rich.theme import Theme

# Ignore all deprecation warnings everywhere
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Track initialized loggers to prevent duplicate handlers
_initialized_loggers: Set[str] = set()
# Flag for multiprocessing context
_is_worker_process = False

# Create rich console with custom theme
console_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
    "package": "magenta",
    "url": "blue underline",
    "path": "cyan",
    "tokens": "yellow",
    "time": "white dim"
})
console = Console(theme=console_theme)

class LLMDocsLogger:
    def __init__(self, name: str = "llm_docs"):
        self.logger = logging.getLogger(name)
        
        # Only configure the logger once per name to prevent duplicate handlers
        if name not in _initialized_loggers:
            # Remove any existing handlers first
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
                
            # Add our custom handler
            handler = logging.StreamHandler()
            handler.setFormatter(LLMDocsFormatter())
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)  # Default to INFO level
            
            # If a log file is specified, add a file handler too
            log_file = get_log_path(f"{name}.log")
                            
            if log_file:
                try:
                    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
                    file_formatter = logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                    )
                    file_handler.setFormatter(file_formatter)
                    self.logger.addHandler(file_handler)
                except Exception as e:
                    console.print(f"Warning: Could not create log file {log_file}: {e}", style="warning")
            
            # Mark this logger as initialized
            _initialized_loggers.add(name)

    def info(self, msg: str, **extra):
        if _is_worker_process:
            # Simplified output for worker processes to avoid duplicates
            console.print(f"INFO: {msg}")
        else:
            self.logger.info(msg, extra=extra)
            if not any(handler.formatter and isinstance(handler.formatter, LLMDocsFormatter) for handler in self.logger.handlers):
                # If no fancy formatter is attached, also print with rich console
                console.print(msg)

    def debug(self, msg: str, **extra):
        if _is_worker_process:
            # Usually we don't need debug messages from workers
            pass
        else:
            self.logger.debug(msg, extra=extra)
            
    def warning(self, msg: str, **extra):
        if _is_worker_process:
            console.print(f"WARNING: {msg}", style="warning")
        else:
            self.logger.warning(msg, extra=extra)

    def error(self, msg: str, exc_info: Optional[bool] = False, **extra):
        if _is_worker_process:
            console.print(f"ERROR: {msg}", style="error")
            if exc_info:
                import traceback
                traceback.print_exc()
        else:
            self.logger.error(msg, exc_info=exc_info, extra=extra)

    def exception(self, msg, *args, **kwargs):
        """Log an exception with traceback information."""
        if _is_worker_process:
            console.print(f"EXCEPTION: {msg}", style="error")
            import traceback
            traceback.print_exc()
        else:
            self.error(msg, exc_info=True, **kwargs)
            
    # LLM docs specific logging methods
    def package_discovery(self, package_name: str, downloads: Optional[int] = None, **extra):
        """Log package discovery information"""
        msg = f"🔍 Found package: [package]{package_name}[/package]"
        if downloads:
            msg += f" ([tokens]{downloads:,}[/tokens] monthly downloads)"
        console.print(msg)
        self.info(msg, **extra)
        
    def doc_url_found(self, package_name: str, url: str, **extra):
        """Log when documentation URL is found"""
        msg = f"🌐 Documentation URL for [package]{package_name}[/package]: [url]{url}[/url]"
        console.print(msg)
        self.info(msg, **extra)
        
    def doc_extraction_start(self, package_name: str, **extra):
        """Log start of documentation extraction"""
        msg = f"📚 Extracting documentation for [package]{package_name}[/package]"
        console.print(msg)
        self.info(msg, **extra)
        
    def doc_extraction_complete(self, package_name: str, path: str, pages_count: int, **extra):
        """Log completion of documentation extraction"""
        msg = f"✅ Extracted documentation for [package]{package_name}[/package]: [path]{path}[/path] ({pages_count} pages)"
        console.print(msg)
        self.info(msg, **extra)
        
    def distillation_start(self, package_name: str, chunk_num: int, total_chunks: int, **extra):
        """Log start of distillation for a chunk"""
        msg = f"🧠 Distilling chunk {chunk_num}/{total_chunks} for [package]{package_name}[/package]"
        console.print(msg)
        self.info(msg, **extra)
        
    def distillation_complete(self, package_name: str, path: str, tokens_used: int, **extra):
        """Log completion of distillation"""
        msg = f"✨ Completed distillation for [package]{package_name}[/package]: [path]{path}[/path] ([tokens]{tokens_used:,}[/tokens] tokens used)"
        console.print(msg)
        self.info(msg, **extra)
        
    def token_usage(self, prompt_tokens: int, completion_tokens: int, model: str, **extra):
        """Log token usage information"""
        total = prompt_tokens + completion_tokens
        msg = f"💰 Token usage: [tokens]{prompt_tokens:,}[/tokens] prompt + [tokens]{completion_tokens:,}[/tokens] completion = [tokens]{total:,}[/tokens] total ({model})"
        console.print(msg)
        self.info(msg, **extra)
        
    def processing_error(self, package_name: str, error_msg: str, **extra):
        """Log processing error"""
        msg = f"❌ Error processing [package]{package_name}[/package]: {error_msg}"
        console.print(msg, style="error")
        self.error(msg, **extra)
        
    def browser_action(self, action: str, url: str = None, **extra):
        """Log browser automation actions"""
        msg = f"🌐 Browser: {action}"
        if url:
            msg += f" [url]{url}[/url]"
        console.print(msg)
        self.debug(msg, **extra)
        
    def api_call(self, provider: str, endpoint: str, **extra):
        """Log API calls to LLM providers"""
        msg = f"🔄 API call to {provider}: {endpoint}"
        console.print(msg, style="info dim")
        self.debug(msg, **extra)
        
    def database_operation(self, operation: str, details: str, **extra):
        """Log database operations"""
        msg = f"💾 Database {operation}: {details}"
        console.print(msg)
        self.debug(msg, **extra)
        
    def rate_limit(self, delay: float, **extra):
        """Log rate limiting events"""
        msg = f"⏱️ Rate limit pause: {delay:.2f}s"
        console.print(msg, style="time")
        self.debug(msg, **extra)


class LLMDocsFormatter(logging.Formatter):
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        message = record.msg
        extra = ""
        if hasattr(record, 'extra') and record.extra:
            extra = ' '.join(f"{k}={v}" for k, v in record.extra.items())
            extra = f" [{extra}]"
        return f"{timestamp} {message}{extra}"


# For multiprocessing worker processes
def set_worker_process_mode():
    """
    Call this function at the start of any worker process to avoid duplicate logging.
    This causes the worker's logger to use simplified logging that won't duplicate.
    """
    global _is_worker_process
    _is_worker_process = True


def get_log_path(filename="llm_docs.log"):
    """Create logs directory in the project root and return the full path"""
    # Get the current working directory (project root)
    project_root = os.getcwd()
    # Create logs directory path
    logs_dir = os.path.join(project_root, "logs")
    # Create the directory if it doesn't exist
    os.makedirs(logs_dir, exist_ok=True)
    # Return full path to the log file
    return os.path.join(logs_dir, filename)


# Create a singleton instance
logger = LLMDocsLogger()