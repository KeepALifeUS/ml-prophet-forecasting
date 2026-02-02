"""
Advanced Prophet Model with Multivariate Features

Enterprise-grade Prophet implementation with advanced features for cryptocurrency
forecasting, including external regressors, custom seasonality, Bayesian optimization,
and enterprise patterns for production deployment.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle
import warnings
from pathlib import Path

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import optuna
from optuna.samplers import TPESampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

from .prophet_model import ProphetForecaster, ForecastResult
from ..config.prophet_config import get_config, ProphetConfig, ModelConfig, OptimizationMethod
from ..preprocessing.data_processor import CryptoDataProcessor, ProcessedData
from ..utils.logger import get_logger, LoggerMixin, timed_operation
from ..utils.exceptions import (
    ModelTrainingException,
    PredictionException,
    OptimizationException,
    ValidationException
)
from ..utils.metrics import ForecastMetrics

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """
    Результат оптимизации гиперпараметров
    """
    best_params: Dict[str, Any]
    best_score: float
    optimization_method: str
    trials_count: int
    optimization_duration: float
    cv_results: Optional[pd.DataFrame]
    study_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "optimization_method": self.optimization_method,
            "trials_count": self.trials_count,
            "optimization_duration": self.optimization_duration,
            "cv_results": self.cv_results.to_dict('records') if self.cv_results is not None else None,
            "study_stats": self.study_stats
        }


@dataclass
class AdvancedForecastResult(ForecastResult):
    """
    Расширенный результат прогнозирования с дополнительной информацией
    """
    regressor_contributions: Dict[str, pd.Series]
    uncertainty_decomposition: Dict[str, pd.Series]
    model_diagnostics: Dict[str, Any]
    feature_importance: Dict[str, float]
    optimization_history: Optional[OptimizationResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Расширенная конвертация в словарь"""
        base_dict = super().to_dict()
        base_dict.update({
            "regressor_contributions": {
                k: v.to_dict() for k, v in self.regressor_contributions.items()
            },
            "uncertainty_decomposition": {
                k: v.to_dict() for k, v in self.uncertainty_decomposition.items()
            },
            "model_diagnostics": self.model_diagnostics,
            "feature_importance": self.feature_importance,
            "optimization_history": self.optimization_history.to_dict() if self.optimization_history else None
        })
        return base_dict


class AdvancedProphetModel(LoggerMixin):
    """
    Продвинутая Prophet модель с многомерными признаками
    
    Расширенные возможности:
    - Автоматический отбор и обработка внешних регрессоров
    - Bayesian оптимизация гиперпараметров
    - Кастомные компоненты сезонности для крипто рынков
    - Анализ важности признаков
    - Декомпозиция неопределенности
    - Ensemble методы с несколькими моделями
    - Автоматическое детектирование аномалий
    - Advanced cross-validation strategies
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str = "1h",
        config: Optional[ProphetConfig] = None
    ):
        """
        Инициализация продвинутой Prophet модели
        
        Args:
            symbol: Символ криптовалюты
            timeframe: Таймфрейм данных
            config: Конфигурация модели
        """
        super().__init__()
        
        self.symbol = symbol.upper()
        self.timeframe = timeframe.lower()
        self.config = config or get_config()
        self.model_config = self.config.get_model_config_for_crypto(symbol)
        
        # Состояние модели
        self.base_model: Optional[Prophet] = None
        self.ensemble_models: Dict[str, Prophet] = {}
        self.is_trained = False
        self.training_data: Optional[ProcessedData] = None
        self.last_training_time: Optional[datetime] = None
        
        # Обработчик данных
        self.data_processor = CryptoDataProcessor(
            symbol=symbol,
            timeframe=timeframe,
            config=self.config.data
        )
        
        # Инструменты для анализа
        self.scaler = StandardScaler()
        self.feature_selector: Optional[Any] = None
        self.optimization_result: Optional[OptimizationResult] = None
        
        # Метрики и диагностика
        self.feature_importance: Dict[str, float] = {}
        self.model_diagnostics: Dict[str, Any] = {}
        self.training_metrics: Dict[str, float] = {}
        
        # Контекст логирования
        self.set_log_context(
            symbol=self.symbol,
            timeframe=self.timeframe,
            model_type="AdvancedProphet"
        )
        
        self.logger.info(f"Инициализирована AdvancedProphetModel для {self.symbol} ({self.timeframe})")
    
    @timed_operation("optimize_hyperparameters")
    def optimize_hyperparameters(
        self,
        data: Union[pd.DataFrame, ProcessedData],
        method: Optional[OptimizationMethod] = None,
        n_trials: Optional[int] = None,
        timeout_hours: Optional[float] = None,
        cv_strategy: str = "time_series"
    ) -> OptimizationResult:
        """
        Оптимизация гиперпараметров модели
        
        Args:
            data: Данные для оптимизации
            method: Метод оптимизации
            n_trials: Количество попыток
            timeout_hours: Таймаут в часах
            cv_strategy: Стратегия кросс-валидации
            
        Returns:
            Результат оптимизации
            
        Raises:
            OptimizationException: При ошибке оптимизации
        """
        try:
            method = method or self.config.optimization.method
            n_trials = n_trials or self.config.optimization.n_trials
            timeout_hours = timeout_hours or self.config.optimization.timeout_hours
            
            self.log_operation_start("optimize_hyperparameters", 
                                   method=method.value, n_trials=n_trials)
            
            # Подготовка данных
            if isinstance(data, pd.DataFrame):
                processed_data = self.data_processor.process_ohlcv_data(data, include_features=True)
            else:
                processed_data = data
            
            start_time = datetime.now()
            
            # Выбор метода оптимизации
            if method == OptimizationMethod.BAYESIAN or method == OptimizationMethod.OPTUNA:
                result = self._optimize_with_optuna(
                    processed_data, n_trials, timeout_hours, cv_strategy
                )
            elif method == OptimizationMethod.GRID_SEARCH:
                result = self._optimize_with_grid_search(processed_data, cv_strategy)
            elif method == OptimizationMethod.RANDOM_SEARCH:
                result = self._optimize_with_random_search(processed_data, n_trials, cv_strategy)
            else:
                raise OptimizationException(f"Unsupported optimization method: {method}")
            
            # Сохранение результата
            result.optimization_duration = (datetime.now() - start_time).total_seconds()
            self.optimization_result = result
            
            # Обновление конфигурации модели с лучшими параметрами
            self._update_model_config_with_best_params(result.best_params)
            
            self.log_operation_end("optimize_hyperparameters", success=True,
                                 best_score=result.best_score,
                                 trials_count=result.trials_count)
            
            return result
            
        except Exception as e:
            self.log_operation_end("optimize_hyperparameters", success=False, error=str(e))
            raise OptimizationException(
                f"Failed to optimize hyperparameters: {e}",
                optimization_method=method.value if method else "unknown",
                original_exception=e
            )
    
    def _optimize_with_optuna(
        self,
        data: ProcessedData,
        n_trials: int,
        timeout_hours: float,
        cv_strategy: str
    ) -> OptimizationResult:
        """Оптимизация с использованием Optuna"""
        
        def objective(trial):
            """Целевая функция для Optuna"""
            # Предлагаемые параметры
            params = {}
            param_space = self.config.optimization.param_space
            
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            # Создание временной модели с предлагаемыми параметрами
            temp_model = self._create_prophet_with_params(params)
            
            # Кросс-валидация
            try:
                cv_score = self._evaluate_model_cv(temp_model, data, cv_strategy)
                return cv_score
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return float('inf')  # Плохой результат для минимизации
        
        # Создание исследования
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )
        
        # Оптимизация
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_hours * 3600,
            show_progress_bar=True
        )
        
        # Создание результата
        study_stats = {
            'n_trials': len(study.trials),
            'best_trial_number': study.best_trial.number,
            'best_value': study.best_value,
            'datetime_start': study.trials[0].datetime_start.isoformat() if study.trials else None,
            'datetime_complete': study.trials[-1].datetime_complete.isoformat() if study.trials else None
        }
        
        result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            optimization_method="optuna",
            trials_count=len(study.trials),
            optimization_duration=0,  # Будет установлено в вызывающем коде
            cv_results=None,
            study_stats=study_stats
        )
        
        return result
    
    def _optimize_with_grid_search(self, data: ProcessedData, cv_strategy: str) -> OptimizationResult:
        """Оптимизация методом Grid Search"""
        from itertools import product
        
        param_space = self.config.optimization.param_space
        
        # Создание сетки параметров
        param_names = []
        param_values = []
        
        for name, config in param_space.items():
            param_names.append(name)
            if config['type'] == 'float':
                values = np.linspace(config['low'], config['high'], 5)
            elif config['type'] == 'int':
                values = range(config['low'], config['high'] + 1, max(1, (config['high'] - config['low']) // 4))
            else:
                values = config.get('choices', [config.get('default')])
            param_values.append(values)
        
        # Перебор всех комбинаций
        best_score = float('inf')
        best_params = {}
        results = []
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            try:
                temp_model = self._create_prophet_with_params(params)
                score = self._evaluate_model_cv(temp_model, data, cv_strategy)
                
                results.append({'params': params.copy(), 'score': score})
                
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                self.logger.warning(f"Grid search trial failed: {e}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_method="grid_search",
            trials_count=len(results),
            optimization_duration=0,
            cv_results=None,
            study_stats={'grid_results': results}
        )
    
    def _optimize_with_random_search(
        self, 
        data: ProcessedData, 
        n_trials: int, 
        cv_strategy: str
    ) -> OptimizationResult:
        """Оптимизация методом Random Search"""
        import random
        
        param_space = self.config.optimization.param_space
        
        best_score = float('inf')
        best_params = {}
        results = []
        
        for trial in range(n_trials):
            params = {}
            
            for name, config in param_space.items():
                if config['type'] == 'float':
                    params[name] = random.uniform(config['low'], config['high'])
                elif config['type'] == 'int':
                    params[name] = random.randint(config['low'], config['high'])
                elif config['type'] == 'categorical':
                    params[name] = random.choice(config['choices'])
            
            try:
                temp_model = self._create_prophet_with_params(params)
                score = self._evaluate_model_cv(temp_model, data, cv_strategy)
                
                results.append({'params': params.copy(), 'score': score})
                
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                self.logger.warning(f"Random search trial failed: {e}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_method="random_search", 
            trials_count=len(results),
            optimization_duration=0,
            cv_results=None,
            study_stats={'random_results': results}
        )
    
    def _create_prophet_with_params(self, params: Dict[str, Any]) -> Prophet:
        """Создание Prophet модели с заданными параметрами"""
        prophet_params = {
            'growth': params.get('growth', self.model_config.growth.value),
            'seasonality_mode': self.model_config.seasonality_mode.value,
            'changepoint_prior_scale': params.get('changepoint_prior_scale', self.model_config.changepoint_prior_scale),
            'seasonality_prior_scale': params.get('seasonality_prior_scale', self.model_config.seasonality_prior_scale),
            'holidays_prior_scale': params.get('holidays_prior_scale', self.model_config.holidays_prior_scale),
            'daily_seasonality': self.model_config.daily_seasonality,
            'weekly_seasonality': self.model_config.weekly_seasonality,
            'yearly_seasonality': self.model_config.yearly_seasonality,
            'n_changepoints': params.get('n_changepoints', self.model_config.n_changepoints),
            'changepoint_range': params.get('changepoint_range', self.model_config.changepoint_range),
            'interval_width': self.model_config.interval_width,
            'uncertainty_samples': self.model_config.uncertainty_samples
        }
        
        return Prophet(**prophet_params)
    
    def _evaluate_model_cv(self, model: Prophet, data: ProcessedData, cv_strategy: str) -> float:
        """Оценка модели через кросс-валидацию"""
        try:
            # Подготовка данных для кросс-валидации
            df = data.prophet_df.copy()
            
            # Добавление регрессоров
            for regressor in self.model_config.additional_regressors:
                if regressor in df.columns:
                    model.add_regressor(regressor)
            
            # Кастомная сезонность
            for seasonality in self.model_config.custom_seasonalities:
                model.add_seasonality(
                    name=seasonality['name'],
                    period=seasonality['period'],
                    fourier_order=seasonality['fourier_order'],
                    mode=seasonality.get('mode', 'additive')
                )
            
            # Обучение
            model.fit(df)
            
            # Параметры кросс-валидации
            total_days = (df['ds'].max() - df['ds'].min()).days
            initial_days = int(total_days * 0.7)
            period_days = max(1, int(total_days * 0.1))
            horizon_days = max(1, int(total_days * 0.2))
            
            initial = f"{initial_days} days"
            period = f"{period_days} days"
            horizon = f"{horizon_days} days"
            
            # Выполнение кросс-валидации
            cv_results = cross_validation(
                model, initial=initial, period=period, horizon=horizon
            )
            
            # Вычисление метрик
            metrics = performance_metrics(cv_results)
            
            # Возвращаем метрику для оптимизации
            metric_name = self.config.optimization.optimization_metric
            if metric_name in metrics.columns:
                return metrics[metric_name].mean()
            else:
                return metrics['mape'].mean()  # Fallback
                
        except Exception as e:
            self.logger.warning(f"CV evaluation failed: {e}")
            return float('inf')
    
    def _update_model_config_with_best_params(self, best_params: Dict[str, Any]):
        """Обновление конфигурации модели лучшими параметрами"""
        for param_name, param_value in best_params.items():
            if hasattr(self.model_config, param_name):
                setattr(self.model_config, param_name, param_value)
                self.logger.debug(f"Updated {param_name} = {param_value}")
    
    @timed_operation("train_advanced_model")
    def train(
        self,
        data: Union[pd.DataFrame, ProcessedData],
        auto_optimize: bool = False,
        feature_selection: bool = True,
        ensemble: bool = False
    ) -> Dict[str, Any]:
        """
        Обучение продвинутой Prophet модели
        
        Args:
            data: Данные для обучения
            auto_optimize: Автоматическая оптимизация гиперпараметров
            feature_selection: Автоматический отбор признаков
            ensemble: Создание ensemble модели
            
        Returns:
            Метрики обучения
        """
        try:
            self.log_operation_start("train_advanced_model",
                                   auto_optimize=auto_optimize,
                                   feature_selection=feature_selection,
                                   ensemble=ensemble)
            
            # Подготовка данных
            if isinstance(data, pd.DataFrame):
                processed_data = self.data_processor.process_ohlcv_data(data, include_features=True)
            else:
                processed_data = data
            
            self.training_data = processed_data
            
            # Автоматическая оптимизация (если требуется)
            if auto_optimize:
                self.logger.info("Запуск автоматической оптимизации гиперпараметров")
                self.optimize_hyperparameters(processed_data)
            
            # Отбор признаков
            if feature_selection:
                processed_data = self._select_features(processed_data)
            
            # Обучение основной модели
            training_metrics = self._train_base_model(processed_data)
            
            # Создание ensemble (если требуется)
            if ensemble:
                ensemble_metrics = self._train_ensemble_models(processed_data)
                training_metrics.update(ensemble_metrics)
            
            # Анализ важности признаков
            self._analyze_feature_importance(processed_data)
            
            # Диагностика модели
            self._run_model_diagnostics(processed_data)
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            self.log_operation_end("train_advanced_model", success=True,
                                 training_samples=len(processed_data.prophet_df))
            
            return training_metrics
            
        except Exception as e:
            self.log_operation_end("train_advanced_model", success=False, error=str(e))
            raise ModelTrainingException(f"Failed to train advanced model: {e}", original_exception=e)
    
    def _select_features(self, data: ProcessedData) -> ProcessedData:
        """Автоматический отбор признаков"""
        try:
            df = data.prophet_df.copy()
            features_cols = [col for col in df.columns if col not in ['ds', 'y']]
            
            if not features_cols:
                return data
            
            # Корреляционный анализ
            correlations = {}
            for col in features_cols:
                try:
                    corr, p_value = pearsonr(df['y'], df[col].fillna(0))
                    if not np.isnan(corr):
                        correlations[col] = abs(corr)
                except:
                    continue
            
            # Отбор топ признаков по корреляции
            top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
            selected_features = [feat for feat, corr in top_features if corr > 0.1]
            
            self.logger.info(f"Отобрано {len(selected_features)} признаков из {len(features_cols)}")
            
            # Создание нового ProcessedData с отобранными признаками
            selected_cols = ['ds', 'y'] + selected_features
            filtered_df = df[selected_cols].copy()
            
            return ProcessedData(
                prophet_df=filtered_df,
                original_df=data.original_df,
                features_df=data.features_df[selected_features] if not data.features_df.empty else pd.DataFrame(),
                metadata={**data.metadata, 'selected_features': selected_features}
            )
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            return data
    
    def _train_base_model(self, data: ProcessedData) -> Dict[str, float]:
        """Обучение основной модели"""
        df = data.prophet_df.copy()
        
        # Создание модели
        self.base_model = Prophet(
            growth=self.model_config.growth.value,
            seasonality_mode=self.model_config.seasonality_mode.value,
            changepoint_prior_scale=self.model_config.changepoint_prior_scale,
            seasonality_prior_scale=self.model_config.seasonality_prior_scale,
            holidays_prior_scale=self.model_config.holidays_prior_scale,
            daily_seasonality=self.model_config.daily_seasonality,
            weekly_seasonality=self.model_config.weekly_seasonality,
            yearly_seasonality=self.model_config.yearly_seasonality,
            n_changepoints=self.model_config.n_changepoints,
            changepoint_range=self.model_config.changepoint_range,
            interval_width=self.model_config.interval_width,
            uncertainty_samples=self.model_config.uncertainty_samples
        )
        
        # Добавление регрессоров
        regressor_cols = [col for col in df.columns if col not in ['ds', 'y']]
        for col in regressor_cols:
            if col in self.model_config.additional_regressors or len(regressor_cols) <= 10:
                try:
                    self.base_model.add_regressor(col)
                except:
                    continue
        
        # Кастомная сезонность
        for seasonality in self.model_config.custom_seasonalities:
            try:
                self.base_model.add_seasonality(
                    name=seasonality['name'],
                    period=seasonality['period'],
                    fourier_order=seasonality['fourier_order'],
                    mode=seasonality.get('mode', 'additive')
                )
            except:
                continue
        
        # Обучение
        start_time = datetime.now()
        self.base_model.fit(df)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Метрики
        metrics = {
            'training_time_seconds': training_time,
            'training_samples': len(df),
            'regressors_count': len(regressor_cols),
            'model_type': 'base_prophet'
        }
        
        self.training_metrics.update(metrics)
        return metrics
    
    def _train_ensemble_models(self, data: ProcessedData) -> Dict[str, float]:
        """Обучение ensemble моделей"""
        # Пока простая реализация - можно расширить
        ensemble_metrics = {
            'ensemble_models': 0,
            'ensemble_training_time': 0
        }
        
        try:
            # Модель с разными параметрами changepoint_prior_scale
            for i, scale in enumerate([0.01, 0.05, 0.1]):
                model_name = f"ensemble_{i}"
                model = Prophet(
                    changepoint_prior_scale=scale,
                    seasonality_mode=self.model_config.seasonality_mode.value
                )
                
                # Добавление основных регрессоров
                regressor_cols = [col for col in data.prophet_df.columns if col not in ['ds', 'y']][:5]
                for col in regressor_cols:
                    try:
                        model.add_regressor(col)
                    except:
                        continue
                
                model.fit(data.prophet_df)
                self.ensemble_models[model_name] = model
                
            ensemble_metrics['ensemble_models'] = len(self.ensemble_models)
            self.logger.info(f"Обучено {len(self.ensemble_models)} ensemble моделей")
            
        except Exception as e:
            self.logger.warning(f"Ensemble training failed: {e}")
        
        return ensemble_metrics
    
    def _analyze_feature_importance(self, data: ProcessedData):
        """Анализ важности признаков"""
        if self.base_model is None:
            return
        
        try:
            # Простой анализ на основе корреляции с остатками
            df = data.prophet_df.copy()
            
            # Получение прогнозов на обучающих данных
            forecast = self.base_model.predict(df[['ds'] + [col for col in df.columns if col not in ['ds', 'y']]])
            residuals = df['y'] - forecast['yhat']
            
            # Корреляция признаков с остатками
            feature_cols = [col for col in df.columns if col not in ['ds', 'y']]
            for col in feature_cols:
                try:
                    corr = np.corrcoef(residuals.fillna(0), df[col].fillna(0))[0, 1]
                    if not np.isnan(corr):
                        self.feature_importance[col] = abs(corr)
                except:
                    continue
                    
            self.logger.debug(f"Вычислена важность для {len(self.feature_importance)} признаков")
            
        except Exception as e:
            self.logger.warning(f"Feature importance analysis failed: {e}")
    
    def _run_model_diagnostics(self, data: ProcessedData):
        """Запуск диагностики модели"""
        try:
            df = data.prophet_df.copy()
            
            # Базовая диагностика
            self.model_diagnostics = {
                'training_data_shape': df.shape,
                'training_period_days': (df['ds'].max() - df['ds'].min()).days,
                'missing_values_percent': (df.isnull().sum() / len(df) * 100).to_dict(),
                'model_components': {
                    'changepoints_count': len(self.base_model.changepoints) if self.base_model else 0,
                    'seasonalities': list(self.base_model.seasonalities.keys()) if self.base_model else [],
                    'extra_regressors': list(self.base_model.extra_regressors.keys()) if self.base_model else []
                }
            }
            
            # Добавление метрик оптимизации (если есть)
            if self.optimization_result:
                self.model_diagnostics['optimization'] = {
                    'best_score': self.optimization_result.best_score,
                    'trials_count': self.optimization_result.trials_count,
                    'optimization_method': self.optimization_result.optimization_method
                }
            
        except Exception as e:
            self.logger.warning(f"Model diagnostics failed: {e}")
            self.model_diagnostics = {}
    
    def predict(
        self,
        periods: Optional[int] = None,
        future_data: Optional[pd.DataFrame] = None,
        include_history: bool = False,
        uncertainty_analysis: bool = True
    ) -> AdvancedForecastResult:
        """
        Продвинутое прогнозирование с дополнительным анализом
        
        Args:
            periods: Количество периодов для прогноза
            future_data: Будущие данные с регрессорами
            include_history: Включить историю
            uncertainty_analysis: Анализ неопределенности
            
        Returns:
            Расширенный результат прогнозирования
        """
        if not self.is_trained or self.base_model is None:
            raise PredictionException("Model must be trained before prediction")
        
        try:
            self.logger.info("Начало продвинутого прогнозирования")
            
            # Базовый прогноз
            base_forecast = self._predict_base_model(periods, future_data, include_history)
            
            # Ensemble прогноз (если есть)
            ensemble_forecasts = self._predict_ensemble_models(periods, future_data, include_history)
            
            # Анализ вклада регрессоров
            regressor_contributions = self._analyze_regressor_contributions(base_forecast)
            
            # Декомпозиция неопределенности
            uncertainty_decomposition = {}
            if uncertainty_analysis:
                uncertainty_decomposition = self._decompose_uncertainty(base_forecast)
            
            # Создание расширенного результата
            result = AdvancedForecastResult(
                symbol=self.symbol,
                timeframe=self.timeframe,
                forecast_df=base_forecast,
                metrics=self._calculate_forecast_metrics(base_forecast),
                confidence_intervals=self._extract_confidence_intervals(base_forecast),
                changepoints=[pd.to_datetime(cp) for cp in self.base_model.changepoints],
                trend_components=self._extract_trend_components(base_forecast),
                seasonality_components=self._extract_seasonality_components(base_forecast),
                forecast_timestamp=datetime.now(),
                model_version="5.0.0-advanced",
                regressor_contributions=regressor_contributions,
                uncertainty_decomposition=uncertainty_decomposition,
                model_diagnostics=self.model_diagnostics,
                feature_importance=self.feature_importance,
                optimization_history=self.optimization_result
            )
            
            self.logger.info(f"Прогноз завершен: {len(base_forecast)} точек")
            return result
            
        except Exception as e:
            self.logger.error(f"Advanced prediction failed: {e}")
            raise PredictionException(f"Advanced prediction failed: {e}", original_exception=e)
    
    def _predict_base_model(
        self, 
        periods: Optional[int], 
        future_data: Optional[pd.DataFrame], 
        include_history: bool
    ) -> pd.DataFrame:
        """Прогноз базовой моделью"""
        if future_data is not None:
            future = future_data.copy()
        else:
            periods = periods or self.config.data.forecast_horizon_days
            future = self.base_model.make_future_dataframe(periods=periods, include_history=include_history)
            
            # Добавление регрессоров (нулевые значения для простоты)
            for regressor in self.base_model.extra_regressors.keys():
                if regressor not in future.columns:
                    future[regressor] = 0
        
        return self.base_model.predict(future)
    
    def _predict_ensemble_models(
        self, 
        periods: Optional[int], 
        future_data: Optional[pd.DataFrame], 
        include_history: bool
    ) -> Dict[str, pd.DataFrame]:
        """Прогноз ensemble моделей"""
        ensemble_forecasts = {}
        
        for name, model in self.ensemble_models.items():
            try:
                if future_data is not None:
                    future = future_data.copy()
                else:
                    periods = periods or self.config.data.forecast_horizon_days
                    future = model.make_future_dataframe(periods=periods, include_history=include_history)
                    
                    # Добавление регрессоров
                    for regressor in model.extra_regressors.keys():
                        if regressor not in future.columns:
                            future[regressor] = 0
                
                forecast = model.predict(future)
                ensemble_forecasts[name] = forecast
                
            except Exception as e:
                self.logger.warning(f"Ensemble model {name} prediction failed: {e}")
        
        return ensemble_forecasts
    
    def _analyze_regressor_contributions(self, forecast: pd.DataFrame) -> Dict[str, pd.Series]:
        """Анализ вклада регрессоров в прогноз"""
        contributions = {}
        
        try:
            # Поиск колонок регрессоров в прогнозе
            regressor_cols = [col for col in forecast.columns 
                            if col not in ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
            
            for col in regressor_cols:
                if col in forecast.columns:
                    contributions[col] = forecast[col]
                    
        except Exception as e:
            self.logger.warning(f"Regressor contribution analysis failed: {e}")
        
        return contributions
    
    def _decompose_uncertainty(self, forecast: pd.DataFrame) -> Dict[str, pd.Series]:
        """Декомпозиция неопределенности"""
        decomposition = {}
        
        try:
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                # Общая неопределенность
                total_uncertainty = forecast['yhat_upper'] - forecast['yhat_lower']
                decomposition['total_uncertainty'] = total_uncertainty
                
                # Процентная неопределенность
                relative_uncertainty = total_uncertainty / forecast['yhat'].abs() * 100
                decomposition['relative_uncertainty'] = relative_uncertainty
                
        except Exception as e:
            self.logger.warning(f"Uncertainty decomposition failed: {e}")
        
        return decomposition
    
    def _calculate_forecast_metrics(self, forecast: pd.DataFrame) -> Dict[str, float]:
        """Вычисление метрик прогноза"""
        metrics = {}
        
        try:
            metrics['forecast_points'] = len(forecast)
            metrics['forecast_mean'] = float(forecast['yhat'].mean())
            metrics['forecast_std'] = float(forecast['yhat'].std())
            
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                interval_width = forecast['yhat_upper'] - forecast['yhat_lower']
                metrics['mean_interval_width'] = float(interval_width.mean())
                
        except Exception as e:
            self.logger.warning(f"Forecast metrics calculation failed: {e}")
        
        return metrics
    
    def _extract_confidence_intervals(self, forecast: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Извлечение доверительных интервалов"""
        intervals = {}
        
        try:
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                for idx, row in forecast.iterrows():
                    intervals[row['ds'].isoformat()] = (
                        float(row['yhat_lower']),
                        float(row['yhat_upper'])
                    )
        except Exception as e:
            self.logger.warning(f"Confidence intervals extraction failed: {e}")
        
        return intervals
    
    def _extract_trend_components(self, forecast: pd.DataFrame) -> Dict[str, pd.Series]:
        """Извлечение компонентов тренда"""
        components = {}
        
        try:
            if 'trend' in forecast.columns:
                components['trend'] = forecast['trend']
                
        except Exception as e:
            self.logger.warning(f"Trend components extraction failed: {e}")
        
        return components
    
    def _extract_seasonality_components(self, forecast: pd.DataFrame) -> Dict[str, pd.Series]:
        """Извлечение компонентов сезонности"""
        components = {}
        
        try:
            seasonality_cols = ['daily', 'weekly', 'yearly']
            for col in seasonality_cols:
                if col in forecast.columns:
                    components[col] = forecast[col]
                    
            # Кастомная сезонность
            for seasonality in self.model_config.custom_seasonalities:
                name = seasonality['name']
                if name in forecast.columns:
                    components[name] = forecast[name]
                    
        except Exception as e:
            self.logger.warning(f"Seasonality components extraction failed: {e}")
        
        return components
    
    async def train_async(self, *args, **kwargs) -> Dict[str, Any]:
        """Асинхронное обучение"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.train, *args, **kwargs)
    
    async def predict_async(self, *args, **kwargs) -> AdvancedForecastResult:
        """Асинхронное прогнозирование"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, *args, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Расширенная информация о модели"""
        base_info = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'is_trained': self.is_trained,
            'model_version': '5.0.0-advanced',
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'model_diagnostics': self.model_diagnostics,
            'optimization_result': self.optimization_result.to_dict() if self.optimization_result else None,
            'ensemble_models_count': len(self.ensemble_models),
            'base_model_trained': self.base_model is not None,
            'training_data_available': self.training_data is not None
        }
        
        return base_info