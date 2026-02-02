"""
Data preprocessing module for Prophet forecasting system.

Enterprise-grade data processing for OHLCV cryptocurrency data with enterprise patterns,
including feature engineering, data validation, outlier detection, and Prophet format preparation.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

from ..config.prophet_config import get_config, DataConfig
from ..utils.logger import get_logger, LoggerMixin, timed_operation
from ..utils.exceptions import (
    DataProcessingException,
    InvalidDataException,
    InsufficientDataException
)
from ..utils.helpers import (
    validate_ohlcv_data,
    clean_numeric_data,
    detect_outliers_iqr,
    create_time_features,
    optimize_dataframe_memory,
    ensure_datetime
)

logger = get_logger(__name__)


@dataclass
class ProcessedData:
    """
    Результат обработки данных для Prophet
    """
    prophet_df: pd.DataFrame  # Данные в формате Prophet (ds, y, ...)
    original_df: pd.DataFrame  # Исходные данные
    features_df: pd.DataFrame  # Дополнительные признаки
    metadata: Dict[str, Any]  # Метаданные обработки
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для сериализации"""
        return {
            "prophet_data": self.prophet_df.to_dict('records'),
            "original_data": self.original_df.to_dict('records'),
            "features": self.features_df.to_dict('records') if not self.features_df.empty else {},
            "metadata": self.metadata
        }


class CryptoDataProcessor(LoggerMixin):
    """
    Комплексный процессор данных для криптовалютного прогнозирования
    
    Основные функции:
    - Валидация и очистка OHLCV данных
    - Детекция и обработка выбросов  
    - Feature engineering для crypto рынков
    - Подготовка данных в формате Prophet
    - Создание дополнительных регрессоров
    - Оптимизация производительности
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str = "1h",
        config: Optional[DataConfig] = None
    ):
        """
        Инициализация процессора данных
        
        Args:
            symbol: Символ криптовалюты
            timeframe: Таймфрейм данных
            config: Конфигурация обработки данных
        """
        super().__init__()
        
        self.symbol = symbol.upper()
        self.timeframe = timeframe.lower()
        self.config = config or get_config().data
        
        # Установка контекста для логирования
        self.set_log_context(
            symbol=self.symbol,
            timeframe=self.timeframe
        )
        
        # Кэш для обработанных данных
        self._cache: Dict[str, ProcessedData] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        self.logger.info(f"Инициализирован CryptoDataProcessor для {self.symbol} ({self.timeframe})")
    
    @timed_operation("process_ohlcv_data")
    def process_ohlcv_data(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        target_column: str = "close",
        include_features: bool = True,
        cache_key: Optional[str] = None
    ) -> ProcessedData:
        """
        Основная функция обработки OHLCV данных
        
        Args:
            data: Исходные OHLCV данные
            target_column: Целевая колонка для прогнозирования
            include_features: Создавать дополнительные признаки
            cache_key: Ключ для кэширования результата
            
        Returns:
            ProcessedData с подготовленными данными
            
        Raises:
            DataProcessingException: При ошибке обработки
            InvalidDataException: При невалидных данных
        """
        try:
            self.log_operation_start("process_ohlcv_data", 
                                   target_column=target_column, 
                                   include_features=include_features)
            
            # Проверка кэша
            if cache_key and self._is_cache_valid(cache_key):
                self.logger.debug(f"Возвращены данные из кэша: {cache_key}")
                return self._cache[cache_key]
            
            # Конвертация данных в DataFrame
            df = self._to_dataframe(data)
            
            # Валидация исходных данных
            self._validate_input_data(df, target_column)
            
            # Сохранение копии исходных данных
            original_df = df.copy()
            
            # Этапы обработки
            metadata = {
                'processing_steps': [],
                'original_shape': df.shape,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'target_column': target_column,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # 1. Нормализация временных меток
            df = self._normalize_timestamps(df)
            metadata['processing_steps'].append('normalize_timestamps')
            
            # 2. Очистка данных
            df = self._clean_data(df, target_column)
            metadata['processing_steps'].append('clean_data')
            
            # 3. Обработка выбросов
            df, outliers_info = self._handle_outliers(df, target_column)
            metadata['outliers_info'] = outliers_info
            metadata['processing_steps'].append('handle_outliers')
            
            # 4. Создание базового Prophet DataFrame
            prophet_df = self._create_prophet_dataframe(df, target_column)
            metadata['processing_steps'].append('create_prophet_dataframe')
            
            # 5. Feature Engineering
            features_df = pd.DataFrame()
            if include_features:
                features_df = self._create_features(df, prophet_df)
                
                # Добавление признаков к Prophet данным
                prophet_df = self._merge_features(prophet_df, features_df)
                metadata['processing_steps'].append('create_features')
            
            # 6. Финальная валидация
            self._validate_processed_data(prophet_df, features_df)
            metadata['processing_steps'].append('validate_processed_data')
            
            # 7. Оптимизация памяти
            prophet_df, memory_stats = optimize_dataframe_memory(prophet_df)
            metadata['memory_optimization'] = memory_stats
            metadata['processing_steps'].append('optimize_memory')
            
            # Финальные метаданные
            metadata.update({
                'final_shape': prophet_df.shape,
                'features_count': len(features_df.columns) if not features_df.empty else 0,
                'time_range': {
                    'start': prophet_df['ds'].min().isoformat(),
                    'end': prophet_df['ds'].max().isoformat(),
                    'periods': len(prophet_df)
                }
            })
            
            # Создание результата
            result = ProcessedData(
                prophet_df=prophet_df,
                original_df=original_df,
                features_df=features_df,
                metadata=metadata
            )
            
            # Кэширование
            if cache_key:
                self._cache_result(cache_key, result)
            
            self.log_operation_end("process_ohlcv_data", success=True,
                                 processed_records=len(prophet_df),
                                 features_created=len(features_df.columns) if not features_df.empty else 0)
            
            return result
            
        except Exception as e:
            self.log_operation_end("process_ohlcv_data", success=False, error=str(e))
            raise DataProcessingException(
                f"Failed to process OHLCV data: {e}",
                processing_stage="process_ohlcv_data",
                original_exception=e
            )
    
    def _to_dataframe(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """Конвертация различных форматов данных в DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise InvalidDataException(f"Unsupported data type: {type(data)}")
    
    def _validate_input_data(self, df: pd.DataFrame, target_column: str):
        """Валидация входных данных"""
        # Базовая валидация OHLCV
        required_columns = ['timestamp', target_column]
        
        # Проверка наличия временной колонки
        time_cols = ['timestamp', 'time', 'datetime', 'date', 'ds']
        time_col = None
        for col in time_cols:
            if col in df.columns:
                time_col = col
                break
        
        if time_col is None:
            raise InvalidDataException("No timestamp column found. Expected one of: " + ", ".join(time_cols))
        
        # Проверка целевой колонки
        if target_column not in df.columns:
            available_cols = ", ".join(df.columns.tolist())
            raise InvalidDataException(
                f"Target column '{target_column}' not found. Available columns: {available_cols}"
            )
        
        # Проверка минимального количества записей
        min_records = self.config.min_history_days * (1440 // self._timeframe_to_minutes())
        if len(df) < min_records:
            raise InsufficientDataException(
                f"Insufficient data: {len(df)} records, minimum required: {min_records}",
                required_samples=min_records,
                provided_samples=len(df)
            )
        
        self.logger.info(f"Input data validated: {len(df)} records, target='{target_column}'")
    
    def _timeframe_to_minutes(self) -> int:
        """Конвертация таймфрейма в минуты"""
        timeframe_map = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60,
            '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080
        }
        return timeframe_map.get(self.timeframe, 60)
    
    def _normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Нормализация временных меток"""
        df = df.copy()
        
        # Поиск колонки с временными метками
        time_cols = ['timestamp', 'time', 'datetime', 'date', 'ds']
        time_col = None
        for col in time_cols:
            if col in df.columns:
                time_col = col
                break
        
        if time_col:
            # Конвертация в datetime
            df[time_col] = pd.to_datetime(df[time_col])
            
            # Переименование в 'ds' для Prophet
            if time_col != 'ds':
                df['ds'] = df[time_col]
                if time_col != 'timestamp':  # Сохраняем исходную колонку
                    df = df.drop(columns=[time_col])
        
        # Сортировка по времени
        df = df.sort_values('ds').reset_index(drop=True)
        
        # Проверка на дубликаты по времени
        if df['ds'].duplicated().any():
            self.logger.warning("Найдены дубликаты временных меток, группируем по среднему")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            agg_dict = {col: 'mean' for col in numeric_cols}
            df = df.groupby('ds').agg(agg_dict).reset_index()
        
        self.logger.debug(f"Временные метки нормализованы: {len(df)} записей")
        return df
    
    def _clean_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Очистка данных"""
        df = df.copy()
        initial_count = len(df)
        
        # Удаление строк с пустой целевой колонкой
        df = df.dropna(subset=[target_column])
        
        # Удаление нулевых и отрицательных цен
        if target_column in ['open', 'high', 'low', 'close']:
            df = df[df[target_column] > 0]
        
        # Очистка числовых колонок
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df = clean_numeric_data(
                df, 
                columns=numeric_cols,
                fill_method=self.config.missing_data_strategy,
                remove_outliers=False  # Обработаем отдельно
            )
        
        # Удаление дубликатов
        df = df.drop_duplicates(subset=['ds'], keep='first')
        
        cleaned_count = len(df)
        removed_count = initial_count - cleaned_count
        
        if removed_count > 0:
            self.logger.info(f"Удалено {removed_count} записей при очистке данных")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, Dict]:
        """Обработка выбросов"""
        if not self.config.outlier_detection:
            return df, {}
        
        df = df.copy()
        outliers_info = {}
        
        # Детекция выбросов в целевой колонке
        if len(df) > 10:
            outliers_mask, stats = detect_outliers_iqr(
                df[target_column], 
                multiplier=self.config.outlier_threshold
            )
            
            outliers_info[target_column] = stats
            
            if outliers_mask.any():
                outlier_count = outliers_mask.sum()
                self.logger.warning(
                    f"Обнаружено {outlier_count} выбросов в колонке {target_column} "
                    f"({stats['outliers_percentage']:.2f}%)"
                )
                
                # Стратегии обработки выбросов
                if stats['outliers_percentage'] < 5.0:  # Менее 5% - удаляем
                    df = df[~outliers_mask]
                    outliers_info[target_column]['action'] = 'removed'
                else:  # Слишком много - заменяем на границы
                    df.loc[outliers_mask & (df[target_column] < stats['lower_bound']), target_column] = stats['lower_bound']
                    df.loc[outliers_mask & (df[target_column] > stats['upper_bound']), target_column] = stats['upper_bound']
                    outliers_info[target_column]['action'] = 'capped'
        
        return df, outliers_info
    
    def _create_prophet_dataframe(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Создание DataFrame в формате Prophet"""
        prophet_df = pd.DataFrame()
        
        # Обязательные колонки для Prophet
        prophet_df['ds'] = df['ds']
        prophet_df['y'] = df[target_column]
        
        # Добавление cap и floor для logistic growth (если нужно)
        # Будет определяться в модели на основе конфигурации
        
        return prophet_df
    
    @timed_operation("create_features")
    def _create_features(self, original_df: pd.DataFrame, prophet_df: pd.DataFrame) -> pd.DataFrame:
        """Создание дополнительных признаков"""
        features_df = pd.DataFrame(index=prophet_df.index)
        
        try:
            # 1. Временные признаки
            time_features = create_time_features(
                prophet_df['ds'], 
                features=['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend']
            )
            features_df = pd.concat([features_df, time_features], axis=1)
            
            # 2. Технические индикаторы (если есть OHLCV данные)
            if all(col in original_df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                tech_features = self._create_technical_indicators(original_df)
                features_df = pd.concat([features_df, tech_features], axis=1)
            
            # 3. Ценовые признаки
            price_features = self._create_price_features(prophet_df['y'])
            features_df = pd.concat([features_df, price_features], axis=1)
            
            # 4. Волатильность
            volatility_features = self._create_volatility_features(prophet_df['y'])
            features_df = pd.concat([features_df, volatility_features], axis=1)
            
            # 5. Лаговые признаки
            lag_features = self._create_lag_features(prophet_df['y'])
            features_df = pd.concat([features_df, lag_features], axis=1)
            
            # Удаление NaN значений
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info(f"Создано {len(features_df.columns)} признаков")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Ошибка создания признаков: {e}")
            return pd.DataFrame(index=prophet_df.index)  # Пустой DataFrame при ошибке
    
    def _create_technical_indicators(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """Создание технических индикаторов"""
        if periods is None:
            periods = [14, 21, 50]
        
        indicators = pd.DataFrame(index=df.index)
        
        try:
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            
            # RSI
            for period in periods:
                if len(close) > period:
                    delta = close.diff()
                    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    indicators[f'rsi_{period}'] = rsi
            
            # Moving Averages
            for period in periods:
                if len(close) > period:
                    indicators[f'sma_{period}'] = close.rolling(window=period).mean()
                    indicators[f'ema_{period}'] = close.ewm(span=period).mean()
            
            # MACD
            if len(close) > 26:
                exp1 = close.ewm(span=12).mean()
                exp2 = close.ewm(span=26).mean()
                macd = exp1 - exp2
                macd_signal = macd.ewm(span=9).mean()
                indicators['macd'] = macd
                indicators['macd_signal'] = macd_signal
                indicators['macd_histogram'] = macd - macd_signal
            
            # Bollinger Bands
            for period in [20]:
                if len(close) > period:
                    sma = close.rolling(window=period).mean()
                    std = close.rolling(window=period).std()
                    indicators[f'bb_upper_{period}'] = sma + (std * 2)
                    indicators[f'bb_lower_{period}'] = sma - (std * 2)
                    indicators[f'bb_width_{period}'] = (indicators[f'bb_upper_{period}'] - indicators[f'bb_lower_{period}']) / sma
            
            # Volume indicators
            if len(volume) > 0:
                for period in [14, 21]:
                    if len(volume) > period:
                        indicators[f'volume_ma_{period}'] = volume.rolling(window=period).mean()
                
                # Volume-Price Trend
                if len(close) > 1:
                    vpt = volume * (close.pct_change())
                    indicators['vpt'] = vpt.cumsum()
            
        except Exception as e:
            self.logger.warning(f"Ошибка создания технических индикаторов: {e}")
        
        return indicators
    
    def _create_price_features(self, prices: pd.Series) -> pd.DataFrame:
        """Создание ценовых признаков"""
        features = pd.DataFrame(index=prices.index)
        
        try:
            # Returns
            features['returns_1'] = prices.pct_change()
            features['returns_5'] = prices.pct_change(5)
            features['returns_10'] = prices.pct_change(10)
            
            # Log returns
            features['log_returns_1'] = np.log(prices / prices.shift(1))
            
            # Price ratios
            for period in [5, 10, 20]:
                if len(prices) > period:
                    features[f'price_ratio_{period}'] = prices / prices.shift(period)
            
            # Z-scores (price normalization)
            for window in [20, 50]:
                if len(prices) > window:
                    rolling_mean = prices.rolling(window=window).mean()
                    rolling_std = prices.rolling(window=window).std()
                    features[f'z_score_{window}'] = (prices - rolling_mean) / rolling_std
                    
        except Exception as e:
            self.logger.warning(f"Ошибка создания ценовых признаков: {e}")
        
        return features
    
    def _create_volatility_features(self, prices: pd.Series) -> pd.DataFrame:
        """Создание признаков волатильности"""
        features = pd.DataFrame(index=prices.index)
        
        try:
            returns = prices.pct_change()
            
            # Rolling volatility
            for window in [10, 20, 50]:
                if len(returns) > window:
                    features[f'volatility_{window}'] = returns.rolling(window=window).std()
            
            # EWMA volatility
            for span in [10, 20]:
                features[f'ewma_volatility_{span}'] = returns.ewm(span=span).std()
            
            # Realized volatility (if high frequency data)
            if len(returns) > 0:
                features['realized_volatility'] = returns.abs()
            
        except Exception as e:
            self.logger.warning(f"Ошибка создания признаков волатильности: {e}")
        
        return features
    
    def _create_lag_features(self, values: pd.Series, lags: List[int] = None) -> pd.DataFrame:
        """Создание лаговых признаков"""
        if lags is None:
            lags = [1, 2, 3, 7, 14]  # Адаптируется к таймфрейму
        
        features = pd.DataFrame(index=values.index)
        
        try:
            for lag in lags:
                if len(values) > lag:
                    features[f'lag_{lag}'] = values.shift(lag)
                    features[f'lag_diff_{lag}'] = values - values.shift(lag)
                    features[f'lag_ratio_{lag}'] = values / values.shift(lag)
            
        except Exception as e:
            self.logger.warning(f"Ошибка создания лаговых признаков: {e}")
        
        return features
    
    def _merge_features(self, prophet_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """Объединение признаков с Prophet данными"""
        if features_df.empty:
            return prophet_df
        
        # Убеждаемся, что индексы совпадают
        if len(prophet_df) != len(features_df):
            self.logger.warning(f"Несоответствие размеров: prophet_df={len(prophet_df)}, features_df={len(features_df)}")
            # Обрезаем до минимального размера
            min_len = min(len(prophet_df), len(features_df))
            prophet_df = prophet_df.iloc[:min_len]
            features_df = features_df.iloc[:min_len]
        
        # Объединяем по индексу
        result = pd.concat([prophet_df, features_df], axis=1)
        
        # Удаляем колонки с слишком многими NaN
        threshold = len(result) * 0.1  # Более 90% NaN - удаляем
        result = result.dropna(axis=1, thresh=int(threshold))
        
        return result
    
    def _validate_processed_data(self, prophet_df: pd.DataFrame, features_df: pd.DataFrame):
        """Валидация обработанных данных"""
        # Проверка обязательных колонок Prophet
        required_cols = ['ds', 'y']
        missing_cols = [col for col in required_cols if col not in prophet_df.columns]
        if missing_cols:
            raise DataProcessingException(f"Missing required Prophet columns: {missing_cols}")
        
        # Проверка типов данных
        if not pd.api.types.is_datetime64_any_dtype(prophet_df['ds']):
            raise DataProcessingException("Column 'ds' must be datetime type")
        
        if not pd.api.types.is_numeric_dtype(prophet_df['y']):
            raise DataProcessingException("Column 'y' must be numeric type")
        
        # Проверка на NaN в ключевых колонках
        if prophet_df[['ds', 'y']].isna().any().any():
            raise DataProcessingException("NaN values found in Prophet key columns")
        
        # Проверка сортировки по времени
        if not prophet_df['ds'].is_monotonic_increasing:
            self.logger.warning("Data is not sorted by timestamp, sorting...")
            prophet_df.sort_values('ds', inplace=True)
            prophet_df.reset_index(drop=True, inplace=True)
        
        self.logger.debug("Processed data validation successful")
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Проверка валидности кэша"""
        if cache_key not in self._cache or cache_key not in self._cache_timestamps:
            return False
        
        # Проверка TTL
        cache_age = datetime.now() - self._cache_timestamps[cache_key]
        max_age = timedelta(hours=self.config.cache_ttl_hours)
        
        return cache_age < max_age
    
    def _cache_result(self, cache_key: str, result: ProcessedData):
        """Кэширование результата"""
        if self.config.cache_enabled:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = datetime.now()
            
            # Ограничение размера кэша
            max_cache_size = 10
            if len(self._cache) > max_cache_size:
                oldest_key = min(self._cache_timestamps.keys(), 
                               key=self._cache_timestamps.get)
                del self._cache[oldest_key]
                del self._cache_timestamps[oldest_key]
    
    async def process_ohlcv_data_async(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        target_column: str = "close",
        include_features: bool = True,
        cache_key: Optional[str] = None
    ) -> ProcessedData:
        """
        Асинхронная обработка OHLCV данных
        
        Args:
            data: Исходные данные
            target_column: Целевая колонка
            include_features: Создавать признаки
            cache_key: Ключ кэша
            
        Returns:
            ProcessedData результат
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.process_ohlcv_data,
            data, target_column, include_features, cache_key
        )
    
    def clear_cache(self):
        """Очистка кэша"""
        self._cache.clear()
        self._cache_timestamps.clear()
        self.logger.debug("Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Статистика кэша"""
        return {
            'cache_size': len(self._cache),
            'cache_keys': list(self._cache.keys()),
            'cache_timestamps': {
                k: v.isoformat() for k, v in self._cache_timestamps.items()
            }
        }