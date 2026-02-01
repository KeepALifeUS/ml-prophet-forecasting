"""
Configuration management for Prophet forecasting system.

Provides comprehensive configuration management using Pydantic with Context7 patterns
for enterprise-grade deployment, monitoring, and performance optimization.
"""

from typing import Dict, List, Optional, Union, Any, Literal
from datetime import datetime, timedelta
from pathlib import Path
import os
from enum import Enum

from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.types import PositiveInt, PositiveFloat, confloat


class LogLevel(str, Enum):
    """Уровни логирования"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SeasonalityMode(str, Enum):
    """Режимы сезонности Prophet"""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class GrowthMode(str, Enum):
    """Режимы роста Prophet"""
    LINEAR = "linear"
    LOGISTIC = "logistic"
    FLAT = "flat"


class OptimizationMethod(str, Enum):
    """Методы оптимизации гиперпараметров"""
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    OPTUNA = "optuna"


class ModelConfig(BaseSettings):
    """
    Конфигурация модели Prophet с enterprise настройками
    """
    
    # === Основные параметры Prophet ===
    growth: GrowthMode = Field(
        default=GrowthMode.LINEAR,
        description="Тип роста тренда"
    )
    
    seasonality_mode: SeasonalityMode = Field(
        default=SeasonalityMode.ADDITIVE,
        description="Режим сезонности"
    )
    
    changepoint_prior_scale: PositiveFloat = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Гибкость изменения трендов"
    )
    
    seasonality_prior_scale: PositiveFloat = Field(
        default=10.0,
        ge=0.01,
        le=100.0,
        description="Сила сезонности"
    )
    
    holidays_prior_scale: PositiveFloat = Field(
        default=10.0,
        ge=0.01,
        le=100.0,
        description="Влияние праздников"
    )
    
    daily_seasonality: Union[bool, str, int] = Field(
        default="auto",
        description="Дневная сезонность"
    )
    
    weekly_seasonality: Union[bool, str, int] = Field(
        default="auto", 
        description="Недельная сезонность"
    )
    
    yearly_seasonality: Union[bool, str, int] = Field(
        default="auto",
        description="Годовая сезонность"
    )
    
    # === Точки изменения ===
    n_changepoints: PositiveInt = Field(
        default=25,
        ge=1,
        le=100,
        description="Количество точек изменения"
    )
    
    changepoint_range: confloat(gt=0, le=1) = Field(
        default=0.8,
        description="Доля истории для поиска changepoints"
    )
    
    # === Интервалы неопределенности ===
    interval_width: confloat(gt=0, lt=1) = Field(
        default=0.8,
        description="Ширина доверительного интервала"
    )
    
    uncertainty_samples: PositiveInt = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Количество сэмплов для неопределенности"
    )
    
    # === Кастомная сезонность ===
    custom_seasonalities: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "name": "crypto_hourly",
                "period": 24,
                "fourier_order": 8,
                "mode": "additive"
            },
            {
                "name": "crypto_weekly", 
                "period": 168,  # 24*7 часов
                "fourier_order": 10,
                "mode": "additive"
            },
            {
                "name": "crypto_monthly",
                "period": 720,  # 24*30 часов
                "fourier_order": 5,
                "mode": "additive"
            }
        ],
        description="Кастомная сезонность для крипто рынков"
    )
    
    # === Регрессоры ===
    additional_regressors: List[str] = Field(
        default_factory=lambda: [
            "volume_ma",
            "volatility", 
            "rsi",
            "macd",
            "sentiment_score",
            "btc_dominance"
        ],
        description="Дополнительные регрессоры"
    )
    
    # === Валидация ===
    
    @validator('daily_seasonality', 'weekly_seasonality', 'yearly_seasonality')
    def validate_seasonality(cls, v):
        """Валидация параметров сезонности"""
        if isinstance(v, str) and v not in ["auto"]:
            raise ValueError("String seasonality must be 'auto'")
        if isinstance(v, int) and v < 0:
            raise ValueError("Integer seasonality must be >= 0")
        return v
    
    class Config:
        env_prefix = "PROPHET_MODEL_"
        case_sensitive = False


class DataConfig(BaseSettings):
    """
    Конфигурация обработки данных
    """
    
    # === Источники данных ===
    default_exchange: str = Field(
        default="binance",
        description="Биржа по умолчанию"
    )
    
    supported_exchanges: List[str] = Field(
        default_factory=lambda: ["binance", "coinbase", "kraken", "okx"],
        description="Поддерживаемые биржи"
    )
    
    # === Параметры данных ===
    min_history_days: PositiveInt = Field(
        default=365,
        ge=30,
        le=3650,
        description="Минимум дней истории"
    )
    
    max_history_days: PositiveInt = Field(
        default=1095,  # 3 года
        ge=365,
        le=3650,
        description="Максимум дней истории"
    )
    
    forecast_horizon_days: PositiveInt = Field(
        default=30,
        ge=1,
        le=365,
        description="Горизонт прогноза в днях"
    )
    
    # === Обработка данных ===
    outlier_detection: bool = Field(
        default=True,
        description="Включить детекцию выбросов"
    )
    
    outlier_threshold: PositiveFloat = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="Порог для выбросов (в std)"
    )
    
    missing_data_strategy: Literal["interpolate", "forward_fill", "drop", "median"] = Field(
        default="interpolate",
        description="Стратегия для пропущенных данных"
    )
    
    data_validation: bool = Field(
        default=True,
        description="Включить валидацию данных"
    )
    
    # === Кэширование ===
    cache_enabled: bool = Field(
        default=True,
        description="Включить кэширование данных"
    )
    
    cache_ttl_hours: PositiveInt = Field(
        default=6,
        ge=1,
        le=168,
        description="TTL кэша в часах"
    )
    
    class Config:
        env_prefix = "PROPHET_DATA_"
        case_sensitive = False


class APIConfig(BaseSettings):
    """
    Конфигурация API сервера
    """
    
    # === Сервер ===
    host: str = Field(default="0.0.0.0", description="Хост API")
    port: PositiveInt = Field(default=8000, ge=1000, le=65535, description="Порт API")
    debug: bool = Field(default=False, description="Режим отладки")
    
    # === CORS ===
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"],
        description="Разрешенные CORS origins"
    )
    
    # === Rate Limiting ===
    rate_limit_enabled: bool = Field(default=True, description="Включить rate limiting")
    rate_limit_requests: PositiveInt = Field(default=100, description="Запросов в минуту")
    rate_limit_window: PositiveInt = Field(default=60, description="Окно в секундах")
    
    # === WebSocket ===
    websocket_enabled: bool = Field(default=True, description="Включить WebSocket")
    websocket_max_connections: PositiveInt = Field(default=100, description="Макс подключений")
    
    # === Аутентификация ===
    auth_enabled: bool = Field(default=False, description="Включить аутентификацию")
    auth_secret_key: Optional[str] = Field(default=None, description="Секретный ключ")
    auth_algorithm: str = Field(default="HS256", description="Алгоритм JWT")
    
    class Config:
        env_prefix = "PROPHET_API_"
        case_sensitive = False


class OptimizationConfig(BaseSettings):
    """
    Конфигурация оптимизации гиперпараметров
    """
    
    method: OptimizationMethod = Field(
        default=OptimizationMethod.BAYESIAN,
        description="Метод оптимизации"
    )
    
    n_trials: PositiveInt = Field(
        default=100,
        ge=10,
        le=1000,
        description="Количество испытаний"
    )
    
    timeout_hours: PositiveFloat = Field(
        default=2.0,
        ge=0.1,
        le=24.0,
        description="Таймаут в часах"
    )
    
    cv_folds: PositiveInt = Field(
        default=5,
        ge=2,
        le=10,
        description="Количество фолдов для кросс-валидации"
    )
    
    # === Параметры для оптимизации ===
    param_space: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "changepoint_prior_scale": {"low": 0.001, "high": 0.5, "type": "float"},
            "seasonality_prior_scale": {"low": 0.01, "high": 100.0, "type": "float"},
            "holidays_prior_scale": {"low": 0.01, "high": 100.0, "type": "float"},
            "n_changepoints": {"low": 5, "high": 50, "type": "int"},
            "changepoint_range": {"low": 0.6, "high": 0.95, "type": "float"}
        },
        description="Пространство параметров для оптимизации"
    )
    
    # === Метрики для оптимизации ===
    optimization_metric: Literal["mape", "mae", "rmse", "smape"] = Field(
        default="mape",
        description="Метрика для оптимизации"
    )
    
    class Config:
        env_prefix = "PROPHET_OPTIMIZATION_"
        case_sensitive = False


class MonitoringConfig(BaseSettings):
    """
    Конфигурация мониторинга и наблюдаемости
    """
    
    # === Логирование ===
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Уровень логирования")
    log_format: str = Field(default="json", description="Формат логов")
    log_file: Optional[str] = Field(default=None, description="Файл логов")
    
    # === Метрики ===
    metrics_enabled: bool = Field(default=True, description="Включить метрики")
    metrics_port: PositiveInt = Field(default=9090, description="Порт метрик")
    
    # === Health checks ===
    health_check_enabled: bool = Field(default=True, description="Включить health checks")
    health_check_interval: PositiveInt = Field(default=30, description="Интервал проверки")
    
    # === Tracing ===
    tracing_enabled: bool = Field(default=False, description="Включить tracing")
    tracing_endpoint: Optional[str] = Field(default=None, description="Endpoint трейсинга")
    
    class Config:
        env_prefix = "PROPHET_MONITORING_"
        case_sensitive = False


class ProphetConfig(BaseSettings):
    """
    Главная конфигурация Prophet forecasting системы
    
    Объединяет все аспекты конфигурации с Context7 enterprise patterns
    """
    
    # === Общие настройки ===
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Среда выполнения"
    )
    
    service_name: str = Field(
        default="prophet-forecasting",
        description="Имя сервиса"
    )
    
    version: str = Field(
        default="5.0.0",
        description="Версия сервиса"
    )
    
    # === Компоненты конфигурации ===
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # === База данных ===
    database_url: str = Field(
        default="postgresql://user:pass@localhost:5432/prophet_db",
        description="URL базы данных"
    )
    
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="URL Redis"
    )
    
    # === Paths ===
    models_dir: Path = Field(
        default=Path("./models"),
        description="Директория моделей"
    )
    
    data_dir: Path = Field(
        default=Path("./data"),
        description="Директория данных"
    )
    
    logs_dir: Path = Field(
        default=Path("./logs"),
        description="Директория логов"
    )
    
    @root_validator
    def validate_paths(cls, values):
        """Создать директории если не существуют"""
        for path_key in ["models_dir", "data_dir", "logs_dir"]:
            if path_key in values:
                path = values[path_key]
                if isinstance(path, str):
                    path = Path(path)
                    values[path_key] = path
                path.mkdir(parents=True, exist_ok=True)
        return values
    
    @validator("database_url", "redis_url")
    def validate_urls(cls, v):
        """Базовая валидация URL"""
        if not v or not isinstance(v, str):
            raise ValueError("URL must be a non-empty string")
        return v
    
    def get_model_config_for_crypto(self, symbol: str) -> ModelConfig:
        """
        Получить конфигурацию модели для конкретной криптовалюты
        
        Args:
            symbol: Символ криптовалюты
            
        Returns:
            Специализированная конфигурация модели
        """
        config = self.model.copy()
        
        # Адаптация параметров для разных криптовалют
        if symbol.upper() in ["BTC", "ETH"]:
            # Для крупных криптовалют - больше гибкости
            config.changepoint_prior_scale = 0.1
            config.n_changepoints = 35
        elif symbol.upper() in ["DOGE", "SHIB"]:
            # Для мем-коинов - больше сезонности
            config.seasonality_prior_scale = 15.0
            config.daily_seasonality = True
        
        return config
    
    def is_production(self) -> bool:
        """Проверить production среду"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Проверить development среду"""
        return self.environment == "development"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_all = True
        extra = "forbid"  # Запретить лишние поля


# Глобальная конфигурация
_config: Optional[ProphetConfig] = None


def get_config() -> ProphetConfig:
    """
    Получить глобальную конфигурацию (singleton pattern)
    
    Returns:
        Экземпляр ProphetConfig
    """
    global _config
    if _config is None:
        _config = ProphetConfig()
    return _config


def reload_config() -> ProphetConfig:
    """
    Перезагрузить конфигурацию
    
    Returns:
        Новый экземпляр ProphetConfig
    """
    global _config
    _config = ProphetConfig()
    return _config


# Вспомогательные функции
def load_config_from_file(config_path: Union[str, Path]) -> ProphetConfig:
    """
    Загрузить конфигурацию из файла
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Экземпляр ProphetConfig
    """
    import yaml
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    return ProphetConfig(**config_data)


def save_config_to_file(config: ProphetConfig, config_path: Union[str, Path]) -> None:
    """
    Сохранить конфигурацию в файл
    
    Args:
        config: Экземпляр конфигурации
        config_path: Путь для сохранения
    """
    import yaml
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.dict(), f, default_flow_style=False, allow_unicode=True)