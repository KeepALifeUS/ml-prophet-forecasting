"""
Comprehensive forecast validation and backtesting framework.

Enterprise-grade validation system for Prophet forecasting models with Context7 patterns,
including time series cross-validation, backtesting, performance analysis, and model comparison.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..models.prophet_model import ProphetForecaster, ForecastResult
from ..models.advanced_prophet import AdvancedProphetModel, AdvancedForecastResult
from ..preprocessing.data_processor import CryptoDataProcessor, ProcessedData
from ..config.prophet_config import get_config, ProphetConfig
from ..utils.logger import get_logger, LoggerMixin, timed_operation
from ..utils.exceptions import ValidationException, ModelNotTrainedException
from ..utils.metrics import ForecastMetrics, MetricResult, calculate_metrics

logger = get_logger(__name__)


class ValidationStrategy(str, Enum):
    """Стратегии валидации"""
    TIME_SERIES_SPLIT = "time_series_split"
    ROLLING_WINDOW = "rolling_window"
    EXPANDING_WINDOW = "expanding_window"
    WALK_FORWARD = "walk_forward"
    HOLDOUT = "holdout"


@dataclass
class ValidationConfig:
    """
    Конфигурация для валидации
    """
    strategy: ValidationStrategy
    n_splits: int = 5
    test_size_ratio: float = 0.2
    min_train_size: int = 100
    step_size: int = 1
    gap_size: int = 0  # Разрыв между train и test
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class BacktestResult:
    """
    Результат бэктестинга
    """
    symbol: str
    timeframe: str
    validation_strategy: ValidationStrategy
    splits_count: int
    total_test_samples: int
    overall_metrics: Dict[str, MetricResult]
    split_metrics: List[Dict[str, MetricResult]]
    predictions: pd.DataFrame
    actuals: pd.DataFrame
    split_details: List[Dict[str, Any]]
    execution_time: float
    validation_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "validation_strategy": self.validation_strategy.value,
            "splits_count": self.splits_count,
            "total_test_samples": self.total_test_samples,
            "overall_metrics": {k: v.to_dict() for k, v in self.overall_metrics.items()},
            "split_metrics": [
                {k: v.to_dict() for k, v in split_metrics.items()}
                for split_metrics in self.split_metrics
            ],
            "predictions": self.predictions.to_dict('records'),
            "actuals": self.actuals.to_dict('records'),
            "split_details": self.split_details,
            "execution_time": self.execution_time,
            "validation_timestamp": self.validation_timestamp.isoformat()
        }


@dataclass
class ModelComparison:
    """
    Результат сравнения моделей
    """
    models: List[str]
    comparison_metrics: Dict[str, Dict[str, float]]
    ranking: List[Tuple[str, float]]
    best_model: str
    statistical_tests: Dict[str, Dict[str, Any]]
    comparison_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "models": self.models,
            "comparison_metrics": self.comparison_metrics,
            "ranking": self.ranking,
            "best_model": self.best_model,
            "statistical_tests": self.statistical_tests,
            "comparison_timestamp": self.comparison_timestamp.isoformat()
        }


class ForecastValidator(LoggerMixin):
    """
    Комплексная система валидации прогнозов
    
    Основные возможности:
    - Временные разбиения данных с различными стратегиями
    - Бэктестинг с реалистичными условиями
    - Кросс-валидация специально для временных рядов
    - Статистическое сравнение моделей
    - Анализ стабильности прогнозов
    - Детекция переобучения и drift
    - Визуализация результатов валидации
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str = "1h",
        config: Optional[ProphetConfig] = None
    ):
        """
        Инициализация валидатора
        
        Args:
            symbol: Символ криптовалюты
            timeframe: Таймфрейм данных
            config: Конфигурация системы
        """
        super().__init__()
        
        self.symbol = symbol.upper()
        self.timeframe = timeframe.lower()
        self.config = config or get_config()
        
        # Инструменты для анализа
        self.metrics_calculator = ForecastMetrics(symbol=symbol, timeframe=timeframe)
        
        # История валидации
        self.validation_history: List[BacktestResult] = []
        self.comparison_history: List[ModelComparison] = []
        
        # Контекст логирования
        self.set_log_context(
            symbol=self.symbol,
            timeframe=self.timeframe,
            component="validator"
        )
        
        self.logger.info(f"Инициализирован ForecastValidator для {self.symbol} ({self.timeframe})")
    
    @timed_operation("backtest_model")
    def backtest_model(
        self,
        model: Union[ProphetForecaster, AdvancedProphetModel],
        data: Union[pd.DataFrame, ProcessedData],
        validation_config: Optional[ValidationConfig] = None,
        target_column: str = "close",
        metrics_list: Optional[List[str]] = None
    ) -> BacktestResult:
        """
        Выполнение бэктестинга модели
        
        Args:
            model: Модель для тестирования
            data: Данные для тестирования
            validation_config: Конфигурация валидации
            target_column: Целевая колонка
            metrics_list: Список метрик для вычисления
            
        Returns:
            Результат бэктестинга
            
        Raises:
            ValidationException: При ошибке валидации
        """
        try:
            self.log_operation_start("backtest_model", 
                                   model_type=type(model).__name__,
                                   target_column=target_column)
            
            start_time = datetime.now()
            
            # Настройки по умолчанию
            if validation_config is None:
                validation_config = ValidationConfig(
                    strategy=ValidationStrategy.TIME_SERIES_SPLIT,
                    n_splits=5,
                    test_size_ratio=0.2
                )
            
            if metrics_list is None:
                metrics_list = ['mae', 'rmse', 'mape', 'directional_accuracy']
            
            # Подготовка данных
            if isinstance(data, pd.DataFrame):
                # Проверка обученности модели или создание временной
                if not hasattr(model, 'is_trained') or not model.is_trained:
                    self.logger.warning("Model is not trained, training on validation data")
                    # Используем первые 80% для обучения
                    train_size = int(len(data) * 0.8)
                    train_data = data.iloc[:train_size]
                    model.train(train_data)
                
                processed_data = self._prepare_data_for_validation(data, target_column)
            else:
                processed_data = data
            
            # Создание разбиений
            splits = self._create_validation_splits(
                processed_data.prophet_df, 
                validation_config
            )
            
            # Выполнение валидации по каждому разбиению
            split_results = []
            all_predictions = []
            all_actuals = []
            
            for i, (train_indices, test_indices) in enumerate(splits):
                self.logger.debug(f"Выполнение разбиения {i+1}/{len(splits)}")
                
                # Данные для разбиения
                train_data = processed_data.prophet_df.iloc[train_indices]
                test_data = processed_data.prophet_df.iloc[test_indices]
                
                # Обучение модели на train данных
                temp_model = self._create_temp_model(model)
                temp_model.train(train_data)
                
                # Прогноз на test данных
                forecast = temp_model.predict(periods=len(test_data), include_history=False)
                
                # Выравнивание данных для сравнения
                actual_values = test_data['y'].values
                predicted_values = forecast['yhat'].values[:len(actual_values)]
                
                # Вычисление метрик для разбиения
                split_metrics = self.metrics_calculator.calculate_all_metrics(
                    actual_values, predicted_values
                )
                
                # Сохранение результатов разбиения
                split_result = {
                    'split_id': i,
                    'train_size': len(train_indices),
                    'test_size': len(test_indices),
                    'train_period': {
                        'start': train_data['ds'].min().isoformat(),
                        'end': train_data['ds'].max().isoformat()
                    },
                    'test_period': {
                        'start': test_data['ds'].min().isoformat(),
                        'end': test_data['ds'].max().isoformat()
                    },
                    'metrics': split_metrics
                }
                
                split_results.append(split_result)
                all_predictions.extend(predicted_values)
                all_actuals.extend(actual_values)
            
            # Общие метрики по всем разбиениям
            overall_metrics = self.metrics_calculator.calculate_all_metrics(
                all_actuals, all_predictions
            )
            
            # Создание DataFrame с результатами
            predictions_df = pd.DataFrame({
                'timestamp': range(len(all_predictions)),
                'predicted': all_predictions,
                'actual': all_actuals,
                'error': np.array(all_predictions) - np.array(all_actuals)
            })
            
            actuals_df = pd.DataFrame({
                'timestamp': range(len(all_actuals)),
                'actual': all_actuals
            })
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Создание результата
            result = BacktestResult(
                symbol=self.symbol,
                timeframe=self.timeframe,
                validation_strategy=validation_config.strategy,
                splits_count=len(splits),
                total_test_samples=len(all_actuals),
                overall_metrics=overall_metrics,
                split_metrics=[split['metrics'] for split in split_results],
                predictions=predictions_df,
                actuals=actuals_df,
                split_details=[{k: v for k, v in split.items() if k != 'metrics'} for split in split_results],
                execution_time=execution_time,
                validation_timestamp=datetime.now()
            )
            
            # Сохранение в историю
            self.validation_history.append(result)
            
            self.log_operation_end("backtest_model", success=True,
                                 splits_count=len(splits),
                                 test_samples=len(all_actuals),
                                 execution_time=execution_time)
            
            return result
            
        except Exception as e:
            self.log_operation_end("backtest_model", success=False, error=str(e))
            raise ValidationException(
                f"Backtesting failed: {e}",
                validation_type="backtest",
                original_exception=e
            )
    
    def _prepare_data_for_validation(
        self, 
        data: pd.DataFrame, 
        target_column: str
    ) -> ProcessedData:
        """Подготовка данных для валидации"""
        processor = CryptoDataProcessor(
            symbol=self.symbol,
            timeframe=self.timeframe,
            config=self.config.data
        )
        
        return processor.process_ohlcv_data(
            data, 
            target_column=target_column,
            include_features=False  # Для простоты валидации
        )
    
    def _create_validation_splits(
        self, 
        data: pd.DataFrame, 
        config: ValidationConfig
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Создание разбиений для валидации"""
        n_samples = len(data)
        splits = []
        
        if config.strategy == ValidationStrategy.TIME_SERIES_SPLIT:
            # Стандартное временное разбиение
            test_size = max(int(n_samples * config.test_size_ratio), config.min_train_size)
            train_size = n_samples - test_size
            
            # Создание нескольких разбиений со сдвигом
            for i in range(config.n_splits):
                start_idx = i * (train_size // config.n_splits)
                end_train = start_idx + train_size
                start_test = end_train + config.gap_size
                end_test = min(start_test + test_size, n_samples)
                
                if end_test > start_test and end_train > start_idx:
                    train_indices = np.arange(start_idx, end_train)
                    test_indices = np.arange(start_test, end_test)
                    splits.append((train_indices, test_indices))
        
        elif config.strategy == ValidationStrategy.ROLLING_WINDOW:
            # Скользящее окно
            train_size = max(int(n_samples * (1 - config.test_size_ratio)), config.min_train_size)
            test_size = int(n_samples * config.test_size_ratio)
            
            for i in range(0, n_samples - train_size - test_size, config.step_size):
                train_start = i
                train_end = i + train_size
                test_start = train_end + config.gap_size
                test_end = test_start + test_size
                
                if test_end <= n_samples:
                    train_indices = np.arange(train_start, train_end)
                    test_indices = np.arange(test_start, test_end)
                    splits.append((train_indices, test_indices))
        
        elif config.strategy == ValidationStrategy.EXPANDING_WINDOW:
            # Расширяющееся окно
            test_size = int(n_samples * config.test_size_ratio)
            
            for i in range(config.n_splits):
                train_end = config.min_train_size + i * (n_samples - config.min_train_size - test_size) // (config.n_splits - 1)
                test_start = train_end + config.gap_size
                test_end = test_start + test_size
                
                if test_end <= n_samples and train_end > 0:
                    train_indices = np.arange(0, train_end)
                    test_indices = np.arange(test_start, test_end)
                    splits.append((train_indices, test_indices))
        
        elif config.strategy == ValidationStrategy.WALK_FORWARD:
            # Пошаговая валидация
            step_size = max(1, (n_samples - config.min_train_size) // config.n_splits)
            test_size = max(1, int(n_samples * config.test_size_ratio))
            
            for i in range(config.n_splits):
                train_end = config.min_train_size + i * step_size
                test_start = train_end + config.gap_size
                test_end = min(test_start + test_size, n_samples)
                
                if test_end > test_start and train_end > 0:
                    train_indices = np.arange(0, train_end)
                    test_indices = np.arange(test_start, test_end)
                    splits.append((train_indices, test_indices))
        
        elif config.strategy == ValidationStrategy.HOLDOUT:
            # Простое разделение train/test
            split_point = int(n_samples * (1 - config.test_size_ratio))
            train_indices = np.arange(0, split_point)
            test_indices = np.arange(split_point + config.gap_size, n_samples)
            splits.append((train_indices, test_indices))
        
        self.logger.debug(f"Создано {len(splits)} разбиений для валидации")
        return splits
    
    def _create_temp_model(self, original_model: Union[ProphetForecaster, AdvancedProphetModel]):
        """Создание временной копии модели"""
        if isinstance(original_model, ProphetForecaster):
            return ProphetForecaster(
                symbol=original_model.symbol,
                timeframe=original_model.timeframe,
                config=original_model.config,
                model_config=original_model.model_config
            )
        elif isinstance(original_model, AdvancedProphetModel):
            return AdvancedProphetModel(
                symbol=original_model.symbol,
                timeframe=original_model.timeframe,
                config=original_model.config
            )
        else:
            raise ValidationException(f"Unsupported model type: {type(original_model)}")
    
    @timed_operation("cross_validate_prophet")
    def cross_validate_prophet(
        self,
        model: Union[ProphetForecaster, AdvancedProphetModel],
        data: Union[pd.DataFrame, ProcessedData],
        initial: str = "365 days",
        period: str = "30 days", 
        horizon: str = "90 days",
        cutoffs: Optional[List[datetime]] = None
    ) -> Dict[str, Any]:
        """
        Кросс-валидация специально для Prophet
        
        Args:
            model: Prophet модель
            data: Данные для валидации
            initial: Начальный период обучения
            period: Период между cutoffs
            horizon: Горизонт прогнозирования
            cutoffs: Конкретные точки разрезов
            
        Returns:
            Результаты кросс-валидации
        """
        try:
            self.log_operation_start("cross_validate_prophet",
                                   initial=initial, period=period, horizon=horizon)
            
            # Подготовка данных
            if isinstance(data, pd.DataFrame):
                processed_data = self._prepare_data_for_validation(data, "close")
            else:
                processed_data = data
            
            # Обучение модели если не обучена
            if not hasattr(model, 'is_trained') or not model.is_trained:
                model.train(processed_data.prophet_df)
            
            # Получение внутренней Prophet модели
            if isinstance(model, ProphetForecaster):
                prophet_model = model.model
            elif isinstance(model, AdvancedProphetModel):
                prophet_model = model.base_model
            else:
                raise ValidationException("Invalid model type for Prophet cross-validation")
            
            if prophet_model is None:
                raise ModelNotTrainedException("Prophet model not trained")
            
            # Выполнение кросс-валидации
            cv_results = cross_validation(
                prophet_model,
                initial=initial,
                period=period,
                horizon=horizon,
                cutoffs=cutoffs,
                parallel="processes"
            )
            
            # Вычисление метрик производительности
            performance_metrics_df = performance_metrics(cv_results)
            
            # Агрегирование результатов
            results = {
                'cv_results': cv_results,
                'performance_metrics': performance_metrics_df,
                'summary_metrics': {
                    'mae': performance_metrics_df['mae'].mean(),
                    'mape': performance_metrics_df['mape'].mean(),
                    'rmse': performance_metrics_df['rmse'].mean(),
                    'coverage': performance_metrics_df.get('coverage', pd.Series([0])).mean()
                },
                'cutoffs_count': len(cv_results['cutoff'].unique()),
                'total_forecasts': len(cv_results)
            }
            
            self.log_operation_end("cross_validate_prophet", success=True,
                                 cutoffs_count=results['cutoffs_count'],
                                 total_forecasts=results['total_forecasts'])
            
            return results
            
        except Exception as e:
            self.log_operation_end("cross_validate_prophet", success=False, error=str(e))
            raise ValidationException(f"Prophet cross-validation failed: {e}")
    
    def compare_models(
        self,
        models: Dict[str, Union[ProphetForecaster, AdvancedProphetModel]],
        data: Union[pd.DataFrame, ProcessedData],
        validation_config: Optional[ValidationConfig] = None
    ) -> ModelComparison:
        """
        Сравнение нескольких моделей
        
        Args:
            models: Словарь моделей {name: model}
            data: Данные для сравнения
            validation_config: Конфигурация валидации
            
        Returns:
            Результат сравнения моделей
        """
        try:
            self.logger.info(f"Сравнение {len(models)} моделей")
            
            # Результаты бэктестинга для каждой модели
            model_results = {}
            for name, model in models.items():
                self.logger.debug(f"Бэктестинг модели: {name}")
                result = self.backtest_model(model, data, validation_config)
                model_results[name] = result
            
            # Сравнение метрик
            comparison_metrics = {}
            for name, result in model_results.items():
                comparison_metrics[name] = {
                    metric_name: metric.value 
                    for metric_name, metric in result.overall_metrics.items()
                }
            
            # Ранжирование моделей (по MAPE, чем меньше - тем лучше)
            ranking = []
            for name in models.keys():
                mape = comparison_metrics[name].get('mape', float('inf'))
                ranking.append((name, mape))
            
            ranking.sort(key=lambda x: x[1])
            best_model = ranking[0][0] if ranking else None
            
            # Статистические тесты (простая реализация)
            statistical_tests = self._perform_statistical_tests(model_results)
            
            comparison = ModelComparison(
                models=list(models.keys()),
                comparison_metrics=comparison_metrics,
                ranking=ranking,
                best_model=best_model,
                statistical_tests=statistical_tests,
                comparison_timestamp=datetime.now()
            )
            
            self.comparison_history.append(comparison)
            
            self.logger.info(f"Лучшая модель: {best_model} (MAPE: {ranking[0][1]:.4f})")
            return comparison
            
        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            raise ValidationException(f"Model comparison failed: {e}")
    
    def _perform_statistical_tests(
        self, 
        model_results: Dict[str, BacktestResult]
    ) -> Dict[str, Dict[str, Any]]:
        """Выполнение статистических тестов"""
        tests = {}
        
        try:
            from scipy.stats import ttest_rel
            
            # Парные t-тесты между моделями
            model_names = list(model_results.keys())
            for i, model1 in enumerate(model_names):
                for model2 in model_names[i+1:]:
                    test_name = f"{model1}_vs_{model2}"
                    
                    # Получение ошибок
                    errors1 = model_results[model1].predictions['error'].values
                    errors2 = model_results[model2].predictions['error'].values
                    
                    # Выравнивание по длине
                    min_len = min(len(errors1), len(errors2))
                    errors1 = errors1[:min_len]
                    errors2 = errors2[:min_len]
                    
                    if len(errors1) > 10:  # Минимум для t-теста
                        statistic, p_value = ttest_rel(np.abs(errors1), np.abs(errors2))
                        tests[test_name] = {
                            'test_type': 'paired_t_test',
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'sample_size': len(errors1)
                        }
                        
        except Exception as e:
            self.logger.warning(f"Statistical tests failed: {e}")
        
        return tests
    
    def analyze_forecast_stability(
        self,
        model: Union[ProphetForecaster, AdvancedProphetModel],
        data: Union[pd.DataFrame, ProcessedData],
        n_runs: int = 10,
        noise_level: float = 0.01
    ) -> Dict[str, Any]:
        """
        Анализ стабильности прогнозов модели
        
        Args:
            model: Модель для анализа
            data: Данные для анализа
            n_runs: Количество прогонов
            noise_level: Уровень шума для тестирования
            
        Returns:
            Результаты анализа стабильности
        """
        try:
            self.logger.info(f"Анализ стабильности прогнозов ({n_runs} прогонов)")
            
            # Подготовка данных
            if isinstance(data, pd.DataFrame):
                processed_data = self._prepare_data_for_validation(data, "close")
            else:
                processed_data = data
            
            forecasts = []
            
            for run in range(n_runs):
                # Добавление небольшого шума к данным
                noisy_data = processed_data.prophet_df.copy()
                noise = np.random.normal(0, noise_level * noisy_data['y'].std(), len(noisy_data))
                noisy_data['y'] += noise
                
                # Создание временной модели
                temp_model = self._create_temp_model(model)
                temp_model.train(noisy_data)
                
                # Прогноз
                forecast = temp_model.predict(periods=30, include_history=False)
                forecasts.append(forecast['yhat'].values)
            
            # Анализ стабильности
            forecasts_array = np.array(forecasts)
            
            stability_metrics = {
                'runs_count': n_runs,
                'forecast_length': forecasts_array.shape[1],
                'mean_forecast': forecasts_array.mean(axis=0).tolist(),
                'std_forecast': forecasts_array.std(axis=0).tolist(),
                'coefficient_of_variation': (forecasts_array.std(axis=0) / np.abs(forecasts_array.mean(axis=0))).tolist(),
                'min_forecast': forecasts_array.min(axis=0).tolist(),
                'max_forecast': forecasts_array.max(axis=0).tolist(),
                'overall_stability': {
                    'mean_cv': float((forecasts_array.std(axis=0) / np.abs(forecasts_array.mean(axis=0))).mean()),
                    'max_cv': float((forecasts_array.std(axis=0) / np.abs(forecasts_array.mean(axis=0))).max()),
                    'stable_points_ratio': float(np.mean((forecasts_array.std(axis=0) / np.abs(forecasts_array.mean(axis=0))) < 0.1))
                }
            }
            
            self.logger.info(f"Анализ стабильности завершен. Средний CV: {stability_metrics['overall_stability']['mean_cv']:.4f}")
            return stability_metrics
            
        except Exception as e:
            self.logger.error(f"Stability analysis failed: {e}")
            raise ValidationException(f"Stability analysis failed: {e}")
    
    def create_validation_report(
        self,
        backtest_result: BacktestResult,
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Создание детального отчета по валидации
        
        Args:
            backtest_result: Результат бэктестинга
            include_plots: Включить графики
            
        Returns:
            Детальный отчет
        """
        try:
            report = {
                'summary': {
                    'symbol': backtest_result.symbol,
                    'timeframe': backtest_result.timeframe,
                    'validation_strategy': backtest_result.validation_strategy.value,
                    'validation_date': backtest_result.validation_timestamp.isoformat(),
                    'execution_time': backtest_result.execution_time,
                    'splits_count': backtest_result.splits_count,
                    'total_test_samples': backtest_result.total_test_samples
                },
                
                'overall_performance': {
                    metric_name: metric.to_dict() 
                    for metric_name, metric in backtest_result.overall_metrics.items()
                },
                
                'split_analysis': {
                    'individual_splits': [
                        {f'split_{i}': {k: v.to_dict() for k, v in split_metrics.items()}}
                        for i, split_metrics in enumerate(backtest_result.split_metrics)
                    ],
                    'consistency_analysis': self._analyze_split_consistency(backtest_result)
                },
                
                'error_analysis': self._analyze_errors(backtest_result),
                
                'recommendations': self._generate_recommendations(backtest_result)
            }
            
            if include_plots:
                report['visualizations'] = self._create_validation_plots(backtest_result)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report creation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_split_consistency(self, result: BacktestResult) -> Dict[str, Any]:
        """Анализ консистентности между разбиениями"""
        if len(result.split_metrics) < 2:
            return {'message': 'Insufficient splits for consistency analysis'}
        
        # Извлечение метрик по разбиениям
        metric_names = list(result.split_metrics[0].keys())
        consistency = {}
        
        for metric_name in metric_names:
            values = [split[metric_name].value for split in result.split_metrics]
            consistency[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'coefficient_of_variation': float(np.std(values) / np.abs(np.mean(values))) if np.mean(values) != 0 else 0,
                'range': float(np.max(values) - np.min(values))
            }
        
        return consistency
    
    def _analyze_errors(self, result: BacktestResult) -> Dict[str, Any]:
        """Анализ ошибок прогнозирования"""
        errors = result.predictions['error'].values
        
        return {
            'error_distribution': {
                'mean': float(np.mean(errors)),
                'median': float(np.median(errors)),
                'std': float(np.std(errors)),
                'skewness': float(self._calculate_skewness(errors)),
                'kurtosis': float(self._calculate_kurtosis(errors))
            },
            'outlier_analysis': {
                'outliers_count': int(np.sum(np.abs(errors) > 3 * np.std(errors))),
                'outliers_percentage': float(np.mean(np.abs(errors) > 3 * np.std(errors)) * 100)
            },
            'trend_analysis': self._analyze_error_trends(errors)
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Вычисление коэффициента асимметрии"""
        try:
            from scipy.stats import skew
            return skew(data)
        except:
            # Простая реализация
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Вычисление коэффициента эксцесса"""
        try:
            from scipy.stats import kurtosis
            return kurtosis(data)
        except:
            # Простая реализация
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
    
    def _analyze_error_trends(self, errors: np.ndarray) -> Dict[str, Any]:
        """Анализ трендов в ошибках"""
        if len(errors) < 10:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Корреляция ошибок с временем
        time_indices = np.arange(len(errors))
        correlation = np.corrcoef(time_indices, errors)[0, 1]
        
        # Автокорреляция
        autocorr = np.corrcoef(errors[:-1], errors[1:])[0, 1] if len(errors) > 1 else 0
        
        return {
            'temporal_correlation': float(correlation) if not np.isnan(correlation) else 0,
            'autocorrelation_lag1': float(autocorr) if not np.isnan(autocorr) else 0,
            'trend_detected': abs(correlation) > 0.3 if not np.isnan(correlation) else False
        }
    
    def _generate_recommendations(self, result: BacktestResult) -> List[str]:
        """Генерация рекомендаций на основе результатов"""
        recommendations = []
        
        # Анализ общих метрик
        if 'mape' in result.overall_metrics:
            mape = result.overall_metrics['mape'].value
            if mape > 20:
                recommendations.append("High MAPE detected. Consider feature engineering or hyperparameter tuning.")
            elif mape < 5:
                recommendations.append("Excellent accuracy achieved. Monitor for potential overfitting.")
        
        if 'directional_accuracy' in result.overall_metrics:
            da = result.overall_metrics['directional_accuracy'].value
            if da < 55:
                recommendations.append("Poor directional accuracy. Consider trend-focused features.")
            elif da > 70:
                recommendations.append("Good trend prediction capability detected.")
        
        # Анализ консистентности
        if len(result.split_metrics) > 1:
            mape_values = [split.get('mape', MetricResult('mape', 0, 'accuracy', '', False, False)).value 
                          for split in result.split_metrics]
            cv = np.std(mape_values) / np.mean(mape_values) if np.mean(mape_values) > 0 else 0
            
            if cv > 0.3:
                recommendations.append("High variability across validation splits. Model may be unstable.")
        
        # Рекомендации по времени выполнения
        if result.execution_time > 300:  # 5 минут
            recommendations.append("Long validation time detected. Consider model simplification.")
        
        if not recommendations:
            recommendations.append("Model performance appears satisfactory across all metrics.")
        
        return recommendations
    
    def _create_validation_plots(self, result: BacktestResult) -> Dict[str, str]:
        """Создание графиков валидации"""
        plots = {}
        
        try:
            # График предсказаний vs фактических значений
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=result.actuals['actual'],
                y=result.predictions['predicted'],
                mode='markers',
                name='Predictions vs Actual',
                opacity=0.6
            ))
            
            # Линия идеального предсказания
            min_val = min(result.actuals['actual'].min(), result.predictions['predicted'].min())
            max_val = max(result.actuals['actual'].max(), result.predictions['predicted'].max())
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
            fig_scatter.update_layout(
                title='Predictions vs Actual Values',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values'
            )
            
            plots['predictions_scatter'] = fig_scatter.to_html()
            
            # График ошибок во времени
            fig_errors = go.Figure()
            fig_errors.add_trace(go.Scatter(
                x=result.predictions['timestamp'],
                y=result.predictions['error'],
                mode='lines+markers',
                name='Prediction Errors'
            ))
            
            fig_errors.update_layout(
                title='Prediction Errors Over Time',
                xaxis_title='Time',
                yaxis_title='Error'
            )
            
            plots['errors_timeline'] = fig_errors.to_html()
            
        except Exception as e:
            self.logger.warning(f"Plot creation failed: {e}")
            plots['error'] = str(e)
        
        return plots
    
    async def backtest_model_async(self, *args, **kwargs) -> BacktestResult:
        """Асинхронный бэктестинг"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.backtest_model, *args, **kwargs)
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Сводка по истории валидации"""
        return {
            'total_validations': len(self.validation_history),
            'total_comparisons': len(self.comparison_history),
            'recent_validations': [
                {
                    'symbol': v.symbol,
                    'timeframe': v.timeframe,
                    'strategy': v.validation_strategy.value,
                    'timestamp': v.validation_timestamp.isoformat(),
                    'overall_mape': v.overall_metrics.get('mape', MetricResult('mape', 0, 'accuracy', '', False, False)).value
                }
                for v in self.validation_history[-5:]
            ]
        }