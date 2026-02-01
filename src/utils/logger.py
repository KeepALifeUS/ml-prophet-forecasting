"""
Structured logging utilities for Prophet Forecasting System

Enterprise-grade logging with Context7 patterns, structured output,
and comprehensive observability features for production environments.
"""

import logging
import sys
from typing import Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
import json
from enum import Enum

import structlog
from structlog.types import Processor


class LogLevel(str, Enum):
    """Уровни логирования"""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Форматы логирования"""
    JSON = "json"
    TEXT = "text"
    COLORED = "colored"


# Глобальная конфигурация логирования
_logging_configured = False
_log_level = LogLevel.INFO
_log_format = LogFormat.JSON


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    format_type: LogFormat = LogFormat.JSON,
    log_file: Optional[Union[str, Path]] = None,
    service_name: str = "prophet-forecasting",
    service_version: str = "5.0.0",
    environment: str = "development"
) -> None:
    """
    Конфигурация структурированного логирования для всего приложения
    
    Args:
        level: Уровень логирования
        format_type: Формат вывода логов
        log_file: Путь к файлу логов (опционально)
        service_name: Имя сервиса
        service_version: Версия сервиса
        environment: Среда выполнения
    """
    global _logging_configured, _log_level, _log_format
    
    if _logging_configured:
        return
    
    _log_level = level
    _log_format = format_type
    
    # Процессоры для структурирования логов
    processors = [
        # Добавление timestamp
        structlog.processors.TimeStamper(fmt="ISO"),
        
        # Добавление уровня лога
        structlog.stdlib.add_log_level,
        
        # Добавление информации о логгере
        structlog.stdlib.add_logger_name,
        
        # Добавление контекста сервиса
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        
        # Обработка исключений
        structlog.processors.format_exc_info,
        
        # Кастомный процессор для добавления метаданных сервиса
        _add_service_context(service_name, service_version, environment),
    ]
    
    # Выбор финального процессора на основе формата
    if format_type == LogFormat.JSON:
        processors.append(structlog.processors.JSONRenderer())
    elif format_type == LogFormat.COLORED:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    else:  # TEXT
        processors.append(
            structlog.processors.KeyValueRenderer(
                key_order=['timestamp', 'level', 'logger', 'event']
            )
        )
    
    # Конфигурация structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Конфигурация стандартного logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.value)
    )
    
    # Настройка файлового логирования
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.value))
        
        # Для файла всегда используем JSON формат
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
        file_handler.setFormatter(file_formatter)
        
        # Добавление к root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    # Подавление избыточных логов от сторонних библиотек
    _suppress_noisy_loggers()
    
    _logging_configured = True


def _add_service_context(
    service_name: str, 
    service_version: str, 
    environment: str
) -> Processor:
    """
    Создание процессора для добавления контекста сервиса
    
    Args:
        service_name: Имя сервиса
        service_version: Версия сервиса
        environment: Среда выполнения
        
    Returns:
        Процессор structlog
    """
    def processor(logger, method_name, event_dict):
        event_dict.update({
            'service': service_name,
            'version': service_version,
            'environment': environment,
            'pid': structlog.processors._get_process_id(),
        })
        return event_dict
    
    return processor


def _suppress_noisy_loggers():
    """Подавление избыточного логирования от сторонних библиотек"""
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3.connectionpool',
        'matplotlib',
        'PIL',
        'asyncio',
        'concurrent.futures',
        'cmdstanpy'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Получение настроенного структурированного логгера
    
    Args:
        name: Имя логгера (опционально, по умолчанию __name__ caller'а)
        
    Returns:
        Настроенный структурированный логгер
    """
    # Автоконфигурация при первом использовании
    if not _logging_configured:
        configure_logging()
    
    # Получение имени из caller'а если не указано
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return structlog.get_logger(name)


def get_request_logger(
    request_id: str,
    user_id: Optional[str] = None,
    endpoint: Optional[str] = None
) -> structlog.BoundLogger:
    """
    Получение логгера с контекстом HTTP запроса
    
    Args:
        request_id: ID запроса для трейсинга
        user_id: ID пользователя (если есть)
        endpoint: Эндпоинт API
        
    Returns:
        Логгер с привязанным контекстом запроса
    """
    logger = get_logger("api")
    
    context = {
        'request_id': request_id,
        'endpoint': endpoint
    }
    
    if user_id:
        context['user_id'] = user_id
    
    return logger.bind(**context)


def get_model_logger(
    symbol: str,
    timeframe: str,
    model_version: Optional[str] = None,
    operation: Optional[str] = None
) -> structlog.BoundLogger:
    """
    Получение логгера с контекстом модели
    
    Args:
        symbol: Символ криптовалюты
        timeframe: Таймфрейм
        model_version: Версия модели
        operation: Текущая операция (train, predict, validate)
        
    Returns:
        Логгер с привязанным контекстом модели
    """
    logger = get_logger("model")
    
    context = {
        'symbol': symbol,
        'timeframe': timeframe
    }
    
    if model_version:
        context['model_version'] = model_version
    if operation:
        context['operation'] = operation
    
    return logger.bind(**context)


def log_performance_metrics(
    logger: structlog.BoundLogger,
    operation: str,
    duration_seconds: float,
    success: bool = True,
    additional_metrics: Optional[Dict[str, Any]] = None
):
    """
    Логирование метрик производительности
    
    Args:
        logger: Логгер для записи
        operation: Название операции
        duration_seconds: Длительность в секундах
        success: Успешность операции
        additional_metrics: Дополнительные метрики
    """
    metrics = {
        'operation': operation,
        'duration_seconds': round(duration_seconds, 4),
        'success': success,
        'performance_log': True
    }
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    if success:
        logger.info(f"Performance: {operation} completed", **metrics)
    else:
        logger.error(f"Performance: {operation} failed", **metrics)


def log_forecast_metrics(
    logger: structlog.BoundLogger,
    symbol: str,
    timeframe: str,
    forecast_points: int,
    training_samples: int,
    metrics: Dict[str, float]
):
    """
    Логирование метрик прогнозирования
    
    Args:
        logger: Логгер для записи
        symbol: Символ криптовалюты
        timeframe: Таймфрейм
        forecast_points: Количество точек прогноза
        training_samples: Количество образцов для обучения
        metrics: Метрики качества модели
    """
    log_data = {
        'symbol': symbol,
        'timeframe': timeframe,
        'forecast_points': forecast_points,
        'training_samples': training_samples,
        'forecast_metrics': metrics,
        'metrics_log': True
    }
    
    logger.info("Forecast metrics computed", **log_data)


def log_model_training(
    logger: structlog.BoundLogger,
    symbol: str,
    timeframe: str,
    training_duration: float,
    samples_count: int,
    model_params: Dict[str, Any],
    validation_metrics: Optional[Dict[str, float]] = None
):
    """
    Логирование процесса обучения модели
    
    Args:
        logger: Логгер для записи
        symbol: Символ криптовалюты
        timeframe: Таймфрейм
        training_duration: Длительность обучения
        samples_count: Количество образцов
        model_params: Параметры модели
        validation_metrics: Метрики валидации
    """
    log_data = {
        'symbol': symbol,
        'timeframe': timeframe,
        'training_duration_seconds': round(training_duration, 4),
        'training_samples': samples_count,
        'model_parameters': model_params,
        'training_log': True
    }
    
    if validation_metrics:
        log_data['validation_metrics'] = validation_metrics
    
    logger.info("Model training completed", **log_data)


class LoggerMixin:
    """
    Mixin класс для добавления логирования в другие классы
    
    Предоставляет удобный интерфейс для логирования в методах класса
    с автоматическим контекстом класса.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None
        self._log_context = {}
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Получение логгера для класса"""
        if self._logger is None:
            class_name = self.__class__.__name__
            module_name = self.__class__.__module__
            logger_name = f"{module_name}.{class_name}"
            
            base_logger = get_logger(logger_name)
            
            # Добавление контекста класса
            context = {
                'class': class_name,
                **self._log_context
            }
            
            self._logger = base_logger.bind(**context)
        
        return self._logger
    
    def set_log_context(self, **kwargs):
        """
        Установка дополнительного контекста для логирования
        
        Args:
            **kwargs: Контекстные переменные
        """
        self._log_context.update(kwargs)
        # Сброс логгера для пересоздания с новым контекстом
        self._logger = None
    
    def log_operation_start(self, operation: str, **kwargs):
        """Логирование начала операции"""
        self.logger.info(f"Starting {operation}", operation=operation, **kwargs)
    
    def log_operation_end(self, operation: str, success: bool = True, **kwargs):
        """Логирование завершения операции"""
        if success:
            self.logger.info(f"Completed {operation}", operation=operation, success=success, **kwargs)
        else:
            self.logger.error(f"Failed {operation}", operation=operation, success=success, **kwargs)


# Декораторы для автоматического логирования

def log_function_calls(
    include_args: bool = False,
    include_result: bool = False,
    log_level: LogLevel = LogLevel.DEBUG
):
    """
    Декоратор для автоматического логирования вызовов функций
    
    Args:
        include_args: Включать аргументы в лог
        include_result: Включать результат в лог
        log_level: Уровень логирования
        
    Returns:
        Декоратор функции
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            log_data = {
                'function': func.__name__,
                'function_call': True
            }
            
            if include_args:
                log_data['args'] = args
                log_data['kwargs'] = kwargs
            
            # Логирование начала
            getattr(logger, log_level.value.lower())(f"Calling {func.__name__}", **log_data)
            
            try:
                result = func(*args, **kwargs)
                
                # Логирование успеха
                success_data = log_data.copy()
                if include_result:
                    success_data['result'] = result
                
                getattr(logger, log_level.value.lower())(f"Completed {func.__name__}", **success_data)
                
                return result
                
            except Exception as e:
                # Логирование ошибки
                error_data = log_data.copy()
                error_data.update({
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                
                logger.error(f"Failed {func.__name__}", **error_data)
                raise
        
        return wrapper
    return decorator


def timed_operation(operation_name: Optional[str] = None):
    """
    Декоратор для измерения времени выполнения операций
    
    Args:
        operation_name: Имя операции (по умолчанию имя функции)
        
    Returns:
        Декоратор функции
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            logger = get_logger(func.__module__)
            op_name = operation_name or func.__name__
            
            start_time = time.time()
            logger.debug(f"Starting timed operation: {op_name}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                log_performance_metrics(
                    logger=logger,
                    operation=op_name,
                    duration_seconds=duration,
                    success=True
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                log_performance_metrics(
                    logger=logger,
                    operation=op_name,
                    duration_seconds=duration,
                    success=False,
                    additional_metrics={'error': str(e)}
                )
                
                raise
        
        return wrapper
    return decorator