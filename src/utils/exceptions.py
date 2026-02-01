"""
Custom exceptions for Prophet Forecasting System

Comprehensive exception hierarchy for proper error handling and Context7 
enterprise patterns with detailed logging and error context.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class ProphetForecastingException(Exception):
    """
    Базовое исключение для системы прогнозирования Prophet
    
    Все специфические исключения должны наследоваться от этого класса
    для обеспечения единообразной обработки ошибок.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Инициализация базового исключения
        
        Args:
            message: Сообщение об ошибке
            error_code: Код ошибки для программной обработки
            details: Дополнительные детали ошибки
            original_exception: Исходное исключение (если есть)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now()
        
        # Добавляем исходное исключение в детали
        if original_exception:
            self.details['original_error'] = str(original_exception)
            self.details['original_type'] = type(original_exception).__name__
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Конвертация исключения в словарь для JSON сериализации
        
        Returns:
            Словарь с информацией об ошибке
        """
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'original_exception': str(self.original_exception) if self.original_exception else None
        }
    
    def __str__(self) -> str:
        """Строковое представление ошибки"""
        base_msg = f"[{self.error_code}] {self.message}"
        if self.details:
            base_msg += f" | Details: {self.details}"
        return base_msg


class ModelNotTrainedException(ProphetForecastingException):
    """
    Исключение для случаев использования необученной модели
    
    Вызывается когда пытаются выполнить прогнозирование или другие операции
    с моделью, которая еще не была обучена.
    """
    
    def __init__(
        self, 
        message: str = "Model must be trained before use",
        model_info: Optional[Dict[str, Any]] = None
    ):
        details = {"model_info": model_info} if model_info else {}
        super().__init__(
            message=message,
            error_code="MODEL_NOT_TRAINED",
            details=details
        )


class InsufficientDataException(ProphetForecastingException):
    """
    Исключение для случаев недостатка данных
    
    Вызывается когда предоставленных данных недостаточно для обучения модели
    или выполнения других операций.
    """
    
    def __init__(
        self, 
        message: str,
        required_samples: Optional[int] = None,
        provided_samples: Optional[int] = None,
        min_period_days: Optional[int] = None
    ):
        details = {}
        if required_samples is not None:
            details['required_samples'] = required_samples
        if provided_samples is not None:
            details['provided_samples'] = provided_samples
        if min_period_days is not None:
            details['min_period_days'] = min_period_days
            
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_DATA",
            details=details
        )


class InvalidDataException(ProphetForecastingException):
    """
    Исключение для некорректных входных данных
    
    Вызывается при обнаружении некорректного формата данных, 
    отсутствии обязательных колонок, некорректных типов данных и т.д.
    """
    
    def __init__(
        self, 
        message: str,
        validation_errors: Optional[Dict[str, Any]] = None,
        data_info: Optional[Dict[str, Any]] = None
    ):
        details = {}
        if validation_errors:
            details['validation_errors'] = validation_errors
        if data_info:
            details['data_info'] = data_info
            
        super().__init__(
            message=message,
            error_code="INVALID_DATA", 
            details=details
        )


class ModelTrainingException(ProphetForecastingException):
    """
    Исключение для ошибок обучения модели
    
    Вызывается при ошибках в процессе обучения модели Prophet,
    включая проблемы с параметрами, конвергенцией и т.д.
    """
    
    def __init__(
        self, 
        message: str,
        model_params: Optional[Dict[str, Any]] = None,
        training_stage: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if model_params:
            details['model_params'] = model_params
        if training_stage:
            details['training_stage'] = training_stage
            
        super().__init__(
            message=message,
            error_code="MODEL_TRAINING_ERROR",
            details=details,
            original_exception=original_exception
        )


class PredictionException(ProphetForecastingException):
    """
    Исключение для ошибок прогнозирования
    
    Вызывается при ошибках в процессе создания прогнозов,
    включая проблемы с входными данными, параметрами прогноза и т.д.
    """
    
    def __init__(
        self, 
        message: str,
        prediction_params: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if prediction_params:
            details['prediction_params'] = prediction_params
        if model_info:
            details['model_info'] = model_info
            
        super().__init__(
            message=message,
            error_code="PREDICTION_ERROR",
            details=details,
            original_exception=original_exception
        )


class ConfigurationException(ProphetForecastingException):
    """
    Исключение для ошибок конфигурации
    
    Вызывается при некорректных параметрах конфигурации,
    отсутствии необходимых настроек и т.д.
    """
    
    def __init__(
        self, 
        message: str,
        config_section: Optional[str] = None,
        invalid_params: Optional[Dict[str, Any]] = None
    ):
        details = {}
        if config_section:
            details['config_section'] = config_section
        if invalid_params:
            details['invalid_params'] = invalid_params
            
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details
        )


class DataProcessingException(ProphetForecastingException):
    """
    Исключение для ошибок обработки данных
    
    Вызывается при ошибках в preprocessing, feature engineering,
    загрузке данных с бирж и других операциях с данными.
    """
    
    def __init__(
        self, 
        message: str,
        processing_stage: Optional[str] = None,
        data_source: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if processing_stage:
            details['processing_stage'] = processing_stage
        if data_source:
            details['data_source'] = data_source
            
        super().__init__(
            message=message,
            error_code="DATA_PROCESSING_ERROR",
            details=details,
            original_exception=original_exception
        )


class APIException(ProphetForecastingException):
    """
    Исключение для ошибок API
    
    Вызывается при ошибках в REST API endpoints,
    WebSocket соединениях и других API операциях.
    """
    
    def __init__(
        self, 
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None
    ):
        details = {}
        if status_code:
            details['status_code'] = status_code
        if endpoint:
            details['endpoint'] = endpoint
        if request_data:
            details['request_data'] = request_data
            
        super().__init__(
            message=message,
            error_code="API_ERROR",
            details=details
        )


class ValidationException(ProphetForecastingException):
    """
    Исключение для ошибок валидации прогнозов
    
    Вызывается при ошибках в кросс-валидации, метриках качества,
    backtesting и других процедурах валидации.
    """
    
    def __init__(
        self, 
        message: str,
        validation_type: Optional[str] = None,
        failed_metrics: Optional[Dict[str, Any]] = None,
        threshold_violations: Optional[Dict[str, Any]] = None
    ):
        details = {}
        if validation_type:
            details['validation_type'] = validation_type
        if failed_metrics:
            details['failed_metrics'] = failed_metrics
        if threshold_violations:
            details['threshold_violations'] = threshold_violations
            
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )


class OptimizationException(ProphetForecastingException):
    """
    Исключение для ошибок оптимизации гиперпараметров
    
    Вызывается при ошибках в Bayesian optimization, Grid Search
    и других методах оптимизации параметров модели.
    """
    
    def __init__(
        self, 
        message: str,
        optimization_method: Optional[str] = None,
        trial_info: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if optimization_method:
            details['optimization_method'] = optimization_method
        if trial_info:
            details['trial_info'] = trial_info
            
        super().__init__(
            message=message,
            error_code="OPTIMIZATION_ERROR",
            details=details,
            original_exception=original_exception
        )


# Вспомогательные функции для работы с исключениями

def handle_prophet_exception(func):
    """
    Декоратор для обработки исключений в функциях Prophet
    
    Автоматически оборачивает стандартные исключения в специализированные
    исключения системы прогнозирования.
    
    Args:
        func: Функция для декорирования
        
    Returns:
        Обернутая функция
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ProphetForecastingException:
            # Уже наше исключение - просто пропускаем
            raise
        except ValueError as e:
            raise InvalidDataException(
                f"Invalid data in {func.__name__}: {e}",
                original_exception=e
            )
        except KeyError as e:
            raise InvalidDataException(
                f"Missing required field in {func.__name__}: {e}",
                original_exception=e
            )
        except FileNotFoundError as e:
            raise DataProcessingException(
                f"File not found in {func.__name__}: {e}",
                original_exception=e
            )
        except Exception as e:
            # Общее исключение для непредвиденных случаев
            raise ProphetForecastingException(
                f"Unexpected error in {func.__name__}: {e}",
                original_exception=e
            )
    
    return wrapper


def create_error_response(exception: ProphetForecastingException) -> Dict[str, Any]:
    """
    Создание стандартизированного ответа об ошибке для API
    
    Args:
        exception: Исключение системы прогнозирования
        
    Returns:
        Словарь с информацией об ошибке для API ответа
    """
    return {
        "success": False,
        "error": {
            "type": exception.__class__.__name__,
            "code": exception.error_code,
            "message": exception.message,
            "details": exception.details,
            "timestamp": exception.timestamp.isoformat()
        }
    }


def log_exception(logger, exception: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Логирование исключения с контекстом
    
    Args:
        logger: Объект логгера
        exception: Исключение для логирования
        context: Дополнительный контекст
    """
    context = context or {}
    
    if isinstance(exception, ProphetForecastingException):
        logger.error(
            f"Prophet Exception: {exception.message}",
            extra={
                "error_code": exception.error_code,
                "error_type": exception.__class__.__name__,
                "details": exception.details,
                "timestamp": exception.timestamp.isoformat(),
                **context
            },
            exc_info=exception.original_exception is not None
        )
    else:
        logger.error(
            f"Unexpected exception: {exception}",
            extra={
                "error_type": type(exception).__name__,
                **context
            },
            exc_info=True
        )