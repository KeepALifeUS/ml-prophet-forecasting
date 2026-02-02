"""
Core Prophet Model Wrapper for Cryptocurrency Forecasting

Enterprise-grade Prophet implementation with enterprise patterns for production-ready
cryptocurrency price predictions with advanced features and comprehensive error handling.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import warnings
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config.prophet_config import get_config, ModelConfig, ProphetConfig
from ..utils.logger import get_logger
from ..utils.exceptions import (
    ModelNotTrainedException, 
    InsufficientDataException,
    InvalidDataException,
    ModelTrainingException,
    PredictionException
)

# Подавить предупреждения Prophet
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*cmdstanpy.*")

logger = get_logger(__name__)


@dataclass
class ForecastResult:
    """
    Результат прогнозирования с полной информацией
    """
    symbol: str
    timeframe: str
    forecast_df: pd.DataFrame
    metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    changepoints: List[datetime]
    trend_components: Dict[str, pd.Series]
    seasonality_components: Dict[str, pd.Series]
    forecast_timestamp: datetime
    model_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для JSON serialization"""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "forecast": self.forecast_df.to_dict('records'),
            "metrics": self.metrics,
            "confidence_intervals": self.confidence_intervals,
            "changepoints": [cp.isoformat() for cp in self.changepoints],
            "trend_components": {k: v.to_dict() for k, v in self.trend_components.items()},
            "seasonality_components": {k: v.to_dict() for k, v in self.seasonality_components.items()},
            "forecast_timestamp": self.forecast_timestamp.isoformat(),
            "model_version": self.model_version
        }


class ProphetForecaster:
    """
    Enterprise Prophet модель для прогнозирования цен криптовалют
    
    Основные возможности:
    - Поддержка множественных криптовалют и таймфреймов
    - Автоматическая детекция сезонности
    - Учет праздников и событий
    - Детекция точек изменения тренда
    - Кросс-валидация и метрики качества
    - Интервалы неопределенности
    - Сохранение и загрузка моделей
    - Асинхронные операции
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str = "1h",
        config: Optional[ProphetConfig] = None,
        model_config: Optional[ModelConfig] = None
    ):
        """
        Инициализация Prophet модели
        
        Args:
            symbol: Символ криптовалюты (например, "BTC")
            timeframe: Таймфрейм данных (например, "1h", "4h", "1d")
            config: Общая конфигурация системы
            model_config: Специфическая конфигурация модели
        """
        self.symbol = symbol.upper()
        self.timeframe = timeframe.lower()
        self.config = config or get_config()
        self.model_config = model_config or self.config.get_model_config_for_crypto(symbol)
        
        # Состояние модели
        self.model: Optional[Prophet] = None
        self.is_trained = False
        self.training_data: Optional[pd.DataFrame] = None
        self.last_training_time: Optional[datetime] = None
        self.model_version = "5.0.0"
        
        # Метрики и результаты
        self.training_metrics: Dict[str, float] = {}
        self.cv_results: Optional[pd.DataFrame] = None
        self.feature_importance: Dict[str, float] = {}
        
        # Логгер
        self.logger = get_logger(f"{__name__}.{self.symbol}.{self.timeframe}")
        
        self.logger.info(
            f"Инициализирован ProphetForecaster для {self.symbol} ({self.timeframe})",
            extra={
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "model_config": asdict(self.model_config)
            }
        )
    
    def _create_prophet_model(self) -> Prophet:
        """
        Создать экземпляр Prophet с настройками из конфигурации
        
        Returns:
            Настроенный экземпляр Prophet
        """
        try:
            # Основные параметры
            prophet_params = {
                'growth': self.model_config.growth.value,
                'seasonality_mode': self.model_config.seasonality_mode.value,
                'changepoint_prior_scale': self.model_config.changepoint_prior_scale,
                'seasonality_prior_scale': self.model_config.seasonality_prior_scale,
                'holidays_prior_scale': self.model_config.holidays_prior_scale,
                'daily_seasonality': self._convert_seasonality(self.model_config.daily_seasonality),
                'weekly_seasonality': self._convert_seasonality(self.model_config.weekly_seasonality),
                'yearly_seasonality': self._convert_seasonality(self.model_config.yearly_seasonality),
                'n_changepoints': self.model_config.n_changepoints,
                'changepoint_range': self.model_config.changepoint_range,
                'interval_width': self.model_config.interval_width,
                'uncertainty_samples': self.model_config.uncertainty_samples
            }
            
            # Создание модели
            model = Prophet(**prophet_params)
            
            # Добавление кастомной сезонности
            for seasonality in self.model_config.custom_seasonalities:
                model.add_seasonality(
                    name=seasonality['name'],
                    period=seasonality['period'],
                    fourier_order=seasonality['fourier_order'],
                    mode=seasonality.get('mode', 'additive')
                )
            
            # Добавление регрессоров (будут добавлены при наличии в данных)
            for regressor in self.model_config.additional_regressors:
                try:
                    model.add_regressor(regressor)
                except Exception as e:
                    self.logger.warning(f"Не удалось добавить регрессор {regressor}: {e}")
            
            self.logger.info(f"Создана Prophet модель с параметрами: {prophet_params}")
            return model
            
        except Exception as e:
            self.logger.error(f"Ошибка создания Prophet модели: {e}")
            raise ModelTrainingException(f"Не удалось создать Prophet модель: {e}")
    
    def _convert_seasonality(self, value: Union[bool, str, int]) -> Union[bool, int]:
        """Конвертация параметров сезонности"""
        if isinstance(value, str) and value == "auto":
            return "auto"
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value >= 0:
            return value
        return False
    
    def _validate_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Валидация и подготовка данных для обучения
        
        Args:
            df: DataFrame с данными (должен содержать 'ds' и 'y')
            
        Returns:
            Валидированный DataFrame
            
        Raises:
            InvalidDataException: При некорректных данных
        """
        try:
            # Проверка наличия обязательных колонок
            required_cols = ['ds', 'y']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise InvalidDataException(f"Отсутствуют обязательные колонки: {missing_cols}")
            
            # Проверка количества записей
            min_records = max(2 * self.model_config.n_changepoints, 100)
            if len(df) < min_records:
                raise InsufficientDataException(
                    f"Недостаточно данных для обучения: {len(df)} < {min_records}"
                )
            
            # Конвертация типов
            df = df.copy()
            df['ds'] = pd.to_datetime(df['ds'])
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            
            # Удаление NaN значений
            initial_rows = len(df)
            df = df.dropna(subset=['ds', 'y'])
            if len(df) < initial_rows * 0.8:
                self.logger.warning(f"Удалено {initial_rows - len(df)} строк с NaN значениями")
            
            # Проверка на дубликаты по времени
            if df['ds'].duplicated().any():
                self.logger.warning("Найдены дубликаты по времени, группируем по среднему")
                df = df.groupby('ds').agg({
                    'y': 'mean',
                    **{col: 'mean' for col in df.columns if col not in ['ds', 'y']}
                }).reset_index()
            
            # Сортировка по времени
            df = df.sort_values('ds').reset_index(drop=True)
            
            # Проверка на положительные значения для logistic growth
            if self.model_config.growth.value == 'logistic':
                if (df['y'] <= 0).any():
                    self.logger.warning("Найдены неположительные значения для logistic growth")
                    df = df[df['y'] > 0]
                
                # Добавление cap для logistic growth
                df['cap'] = df['y'].max() * 1.2
                df['floor'] = df['y'].min() * 0.8
            
            self.logger.info(
                f"Данные валидированы: {len(df)} записей, "
                f"период с {df['ds'].min()} по {df['ds'].max()}"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка валидации данных: {e}")
            raise InvalidDataException(f"Некорректные данные для обучения: {e}")
    
    def train(
        self, 
        data: pd.DataFrame,
        holidays: Optional[pd.DataFrame] = None,
        regressors_data: Optional[Dict[str, pd.Series]] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Обучение Prophet модели
        
        Args:
            data: Данные для обучения с колонками 'ds' (время) и 'y' (цена)
            holidays: DataFrame с праздниками (опционально)
            regressors_data: Дополнительные регрессоры
            validate: Выполнить кросс-валидацию после обучения
            
        Returns:
            Словарь с метриками обучения
            
        Raises:
            ModelTrainingException: При ошибке обучения
        """
        try:
            self.logger.info(f"Начало обучения модели для {self.symbol} ({self.timeframe})")
            
            # Валидация данных
            validated_data = self._validate_training_data(data)
            
            # Добавление регрессоров к данным
            if regressors_data:
                for regressor_name, regressor_values in regressors_data.items():
                    if regressor_name in self.model_config.additional_regressors:
                        validated_data[regressor_name] = regressor_values
            
            # Создание модели
            self.model = self._create_prophet_model()
            
            # Добавление праздников
            if holidays is not None:
                self.model.holidays = holidays
            
            # Обучение
            training_start = datetime.now()
            self.model.fit(validated_data)
            training_time = (datetime.now() - training_start).total_seconds()
            
            # Сохранение состояния
            self.is_trained = True
            self.training_data = validated_data
            self.last_training_time = datetime.now()
            
            # Базовые метрики
            self.training_metrics = {
                'training_time_seconds': training_time,
                'training_samples': len(validated_data),
                'training_period_days': (validated_data['ds'].max() - validated_data['ds'].min()).days,
                'mean_absolute_error_training': 0.0,  # Будет вычислено при валидации
                'mean_squared_error_training': 0.0,
                'r2_score_training': 0.0
            }
            
            # Кросс-валидация
            if validate and len(validated_data) > 200:  # Только для достаточного объема данных
                try:
                    cv_results = self._perform_cross_validation(validated_data)
                    self.training_metrics.update(cv_results)
                except Exception as e:
                    self.logger.warning(f"Кросс-валидация не выполнена: {e}")
            
            self.logger.info(
                f"Обучение завершено за {training_time:.2f}с, "
                f"размер данных: {len(validated_data)} записей",
                extra={"metrics": self.training_metrics}
            )
            
            return self.training_metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {e}")
            raise ModelTrainingException(f"Не удалось обучить модель: {e}")
    
    def _perform_cross_validation(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Выполнение кросс-валидации модели
        
        Args:
            data: Данные для валидации
            
        Returns:
            Метрики кросс-валидации
        """
        try:
            # Параметры кросс-валидации
            initial_days = len(data) * 0.7  # 70% для initial
            period_days = max(1, len(data) * 0.1)  # 10% для period
            horizon_days = max(1, len(data) * 0.2)  # 20% для horizon
            
            initial = f"{int(initial_days)} days"
            period = f"{int(period_days)} days"
            horizon = f"{int(horizon_days)} days"
            
            self.logger.info(f"Кросс-валидация: initial={initial}, period={period}, horizon={horizon}")
            
            # Выполнение кросс-валидации
            cv_results = cross_validation(
                self.model,
                initial=initial,
                period=period,
                horizon=horizon,
                parallel="processes"
            )
            
            # Вычисление метрик
            metrics = performance_metrics(cv_results)
            
            # Сохранение результатов
            self.cv_results = cv_results
            
            # Агрегированные метрики
            cv_metrics = {
                'cv_mae': metrics['mae'].mean(),
                'cv_mape': metrics['mape'].mean(),
                'cv_rmse': metrics['rmse'].mean(),
                'cv_coverage': metrics['coverage'].mean() if 'coverage' in metrics else 0.0,
                'cv_folds': len(cv_results['cutoff'].unique())
            }
            
            self.logger.info(f"Кросс-валидация завершена: {cv_metrics}")
            return cv_metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка кросс-валидации: {e}")
            return {}
    
    def predict(
        self,
        periods: Optional[int] = None,
        future_data: Optional[pd.DataFrame] = None,
        include_history: bool = False
    ) -> ForecastResult:
        """
        Прогнозирование с Prophet моделью
        
        Args:
            periods: Количество периодов для прогноза
            future_data: Предопределенные будущие даты с регрессорами
            include_history: Включить исторические данные в результат
            
        Returns:
            Результат прогнозирования
            
        Raises:
            ModelNotTrainedException: Если модель не обучена
            PredictionException: При ошибке прогнозирования
        """
        if not self.is_trained or self.model is None:
            raise ModelNotTrainedException("Модель должна быть обучена перед прогнозированием")
        
        try:
            self.logger.info(f"Начало прогнозирования для {self.symbol} ({self.timeframe})")
            
            # Создание future dataframe
            if future_data is not None:
                future = future_data.copy()
            else:
                periods = periods or self.config.data.forecast_horizon_days
                future = self.model.make_future_dataframe(periods=periods, include_history=include_history)
                
                # Добавление cap/floor для logistic growth
                if self.model_config.growth.value == 'logistic':
                    future['cap'] = self.training_data['y'].max() * 1.2
                    future['floor'] = self.training_data['y'].min() * 0.8
            
            # Прогнозирование
            forecast = self.model.predict(future)
            
            # Если нужна только будущая часть
            if not include_history and future_data is None:
                last_training_date = self.training_data['ds'].max()
                forecast = forecast[forecast['ds'] > last_training_date]
            
            # Извлечение компонентов
            changepoints = [pd.to_datetime(cp) for cp in self.model.changepoints]
            
            # Компоненты тренда и сезонности
            components = self.model.predict(future)
            trend_components = {
                'trend': components['trend'],
                'trend_lower': components.get('trend_lower', components['trend']),
                'trend_upper': components.get('trend_upper', components['trend'])
            }
            
            seasonality_components = {}
            for component in ['daily', 'weekly', 'yearly']:
                if f'{component}' in components.columns:
                    seasonality_components[component] = components[component]
            
            # Кастомная сезонность
            for seasonality in self.model_config.custom_seasonalities:
                name = seasonality['name']
                if name in components.columns:
                    seasonality_components[name] = components[name]
            
            # Доверительные интервалы
            confidence_intervals = {}
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                for idx, row in forecast.iterrows():
                    confidence_intervals[row['ds'].isoformat()] = (
                        float(row['yhat_lower']),
                        float(row['yhat_upper'])
                    )
            
            # Вычисление метрик прогноза
            prediction_metrics = self._calculate_prediction_metrics(forecast)
            
            # Создание результата
            result = ForecastResult(
                symbol=self.symbol,
                timeframe=self.timeframe,
                forecast_df=forecast,
                metrics=prediction_metrics,
                confidence_intervals=confidence_intervals,
                changepoints=changepoints,
                trend_components=trend_components,
                seasonality_components=seasonality_components,
                forecast_timestamp=datetime.now(),
                model_version=self.model_version
            )
            
            self.logger.info(
                f"Прогноз завершен: {len(forecast)} точек, "
                f"период с {forecast['ds'].min()} по {forecast['ds'].max()}",
                extra={"prediction_metrics": prediction_metrics}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка прогнозирования: {e}")
            raise PredictionException(f"Не удалось выполнить прогноз: {e}")
    
    def _calculate_prediction_metrics(self, forecast: pd.DataFrame) -> Dict[str, float]:
        """Вычисление метрик качества прогноза"""
        try:
            metrics = {}
            
            # Базовые статистики прогноза
            metrics['forecast_mean'] = float(forecast['yhat'].mean())
            metrics['forecast_std'] = float(forecast['yhat'].std())
            metrics['forecast_min'] = float(forecast['yhat'].min())
            metrics['forecast_max'] = float(forecast['yhat'].max())
            
            # Ширина доверительного интервала
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                interval_width = forecast['yhat_upper'] - forecast['yhat_lower']
                metrics['mean_interval_width'] = float(interval_width.mean())
                metrics['interval_width_std'] = float(interval_width.std())
            
            # Тренд направления
            if len(forecast) > 1:
                trend_direction = (forecast['yhat'].iloc[-1] - forecast['yhat'].iloc[0]) / len(forecast)
                metrics['trend_direction'] = float(trend_direction)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Не удалось вычислить метрики прогноза: {e}")
            return {}
    
    async def predict_async(
        self,
        periods: Optional[int] = None,
        future_data: Optional[pd.DataFrame] = None,
        include_history: bool = False
    ) -> ForecastResult:
        """
        Асинхронное прогнозирование
        
        Args:
            periods: Количество периодов для прогноза
            future_data: Предопределенные будущие даты
            include_history: Включить исторические данные
            
        Returns:
            Результат прогнозирования
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.predict, periods, future_data, include_history
        )
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Сохранение обученной модели
        
        Args:
            filepath: Путь для сохранения модели
            
        Raises:
            ModelNotTrainedException: Если модель не обучена
        """
        if not self.is_trained or self.model is None:
            raise ModelNotTrainedException("Нет обученной модели для сохранения")
        
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Данные для сохранения
            model_data = {
                'model': self.model,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'model_config': asdict(self.model_config),
                'training_metrics': self.training_metrics,
                'last_training_time': self.last_training_time,
                'model_version': self.model_version,
                'cv_results': self.cv_results
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Модель сохранена: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения модели: {e}")
            raise ModelTrainingException(f"Не удалось сохранить модель: {e}")
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Загрузка обученной модели
        
        Args:
            filepath: Путь к файлу модели
            
        Raises:
            FileNotFoundError: Если файл не найден
            ModelTrainingException: При ошибке загрузки
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Файл модели не найден: {filepath}")
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Восстановление состояния
            self.model = model_data['model']
            self.symbol = model_data['symbol']
            self.timeframe = model_data['timeframe']
            self.training_metrics = model_data.get('training_metrics', {})
            self.last_training_time = model_data.get('last_training_time')
            self.model_version = model_data.get('model_version', '5.0.0')
            self.cv_results = model_data.get('cv_results')
            self.is_trained = True
            
            self.logger.info(f"Модель загружена: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            raise ModelTrainingException(f"Не удалось загрузить модель: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели
        
        Returns:
            Словарь с информацией о модели
        """
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'is_trained': self.is_trained,
            'model_version': self.model_version,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_metrics': self.training_metrics,
            'model_config': asdict(self.model_config),
            'has_cv_results': self.cv_results is not None,
            'training_data_size': len(self.training_data) if self.training_data is not None else 0
        }
    
    def plot_forecast(
        self, 
        forecast_result: ForecastResult,
        show_components: bool = True,
        save_path: Optional[Union[str, Path]] = None
    ) -> go.Figure:
        """
        Создание интерактивного графика прогноза
        
        Args:
            forecast_result: Результат прогнозирования
            show_components: Показать компоненты (тренд, сезонность)
            save_path: Путь для сохранения графика
            
        Returns:
            Plotly Figure объект
        """
        try:
            # Создание subplots
            rows = 3 if show_components else 1
            fig = make_subplots(
                rows=rows,
                cols=1,
                subplot_titles=['Price Forecast', 'Trend Component', 'Seasonality Components'],
                shared_xaxes=True,
                vertical_spacing=0.08
            )
            
            forecast = forecast_result.forecast_df
            
            # Основной прогноз
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Доверительный интервал
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(0,100,200,0.2)',
                        name='Confidence Interval',
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )
            
            # Исторические данные
            if self.training_data is not None:
                fig.add_trace(
                    go.Scatter(
                        x=self.training_data['ds'],
                        y=self.training_data['y'],
                        mode='lines',
                        name='Historical',
                        line=dict(color='black', width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
            
            # Компоненты
            if show_components and rows > 1:
                # Тренд
                if 'trend' in forecast_result.trend_components:
                    fig.add_trace(
                        go.Scatter(
                            x=forecast['ds'],
                            y=forecast_result.trend_components['trend'],
                            mode='lines',
                            name='Trend',
                            line=dict(color='green', width=2)
                        ),
                        row=2, col=1
                    )
                
                # Сезонность
                seasonality_colors = ['red', 'orange', 'purple', 'brown']
                for i, (name, component) in enumerate(forecast_result.seasonality_components.items()):
                    if i < len(seasonality_colors):
                        fig.add_trace(
                            go.Scatter(
                                x=forecast['ds'],
                                y=component,
                                mode='lines',
                                name=f'{name.title()} Seasonality',
                                line=dict(color=seasonality_colors[i], width=1)
                            ),
                            row=3, col=1
                        )
            
            # Точки изменения
            for changepoint in forecast_result.changepoints:
                fig.add_vline(
                    x=changepoint,
                    line_dash="dash",
                    line_color="red",
                    opacity=0.5,
                    annotation_text="Changepoint"
                )
            
            # Обновление layout
            fig.update_layout(
                title=f'{self.symbol} Price Forecast ({self.timeframe})',
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified',
                height=800 if show_components else 400,
                showlegend=True
            )
            
            # Сохранение
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(save_path)
                self.logger.info(f"График сохранен: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Ошибка создания графика: {e}")
            return go.Figure()  # Пустой график при ошибке