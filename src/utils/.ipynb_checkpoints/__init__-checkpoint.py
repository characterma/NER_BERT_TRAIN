from functools import wraps
from datetime import datetime
from loguru import logger

def log_step(num_equals=10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            step_name = func.__name__
            message = f"{'=' * num_equals} {step_name} {'=' * num_equals}"
            
            # 函数开始时的日志
            logger.info(f"Starting: {message}")
            
            # 记录开始时间
            start_time = datetime.now()
            
            try:
                # 执行原函数
                result = func(*args, **kwargs)
                return result
            finally:
                # 计算执行时间
                end_time = datetime.now()
                execution_time = end_time - start_time
                
                # 函数结束时的日志
                logger.info(f"Completed: {message} (Execution time: {execution_time})")
        
        return wrapper
    return decorator