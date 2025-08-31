import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import multiprocessing, gc, psutil, time, warnings

warnings.filterwarnings('ignore')


class OptimizedTradingSignalAnalyzer:
    """
    Performance-optimized trading signal analyzer with concurrent processing
    and memory-efficient handling for large datasets
    """

    def __init__(self, data_file='cleaned_stock_market_tweets.csv',
                 max_workers=None, chunk_size=1000, memory_limit_gb=4):
        """
        Initialize with performance optimization parameters
        """
        self.data_file = data_file
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
        self.chunk_size = chunk_size
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024

        self.df = None
        self.tfidf_vectorizer = None
        self.svd = None
        self.scaler = StandardScaler()

        print(f"Initialized with {self.max_workers} workers, chunk size: {chunk_size}")

    def monitor_memory_usage(self):
        """Monitor current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return memory_mb

    def load_data_optimized(self):
        """
        Memory-efficient data loading with chunked processing
        """
        print("Loading data with memory optimization...")
        start_time = time.time()

        # Get file size for memory planning
        try:
            estimated_rows = sum(1 for _ in open(self.data_file, 'r', encoding='utf-8', errors='ignore')) - 1
            print(f"Estimated {estimated_rows} rows to process")
        except Exception:
            print("Could not estimate file size, proceeding with default chunk size")

        # Determine optimal chunk size based on memory
        optimal_chunk_size = max(100, self.chunk_size)
        print(f"Using chunk size: {optimal_chunk_size}")

        # Load data in chunks
        chunks = []
        chunk_iter = pd.read_csv(self.data_file, chunksize=optimal_chunk_size)

        for i, chunk in enumerate(chunk_iter):
            # Convert timestamps efficiently
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], cache=True, errors='coerce')

            # Ensure required columns exist
            for col in ['content', 'username', 'retweets', 'likes', 'replies']:
                if col not in chunk.columns:
                    if col in ['retweets', 'likes', 'replies']:
                        chunk[col] = 0
                    else:
                        chunk[col] = ''

            # Clean text columns
            chunk['content'] = chunk['content'].fillna('').astype(str)
            chunk['username'] = chunk['username'].fillna('').astype('category')

            # Ensure trading_signal exists and is categorical
            if 'trading_signal' in chunk.columns:
                chunk['trading_signal'] = chunk['trading_signal'].astype('category')
            else:
                chunk['trading_signal'] = pd.Categorical(['HOLD'] * len(chunk))

            chunks.append(chunk)

            # Monitor memory usage each loop (avoid uninitialized variable)
            memory_mb = self.monitor_memory_usage()
            if i % 10 == 0:
                print(f"Processed chunk {i}, Memory usage: {memory_mb:.1f} MB")

            # Force garbage collection if memory usage is high
            if memory_mb > (self.memory_limit_bytes / 1024 / 1024 * 0.8):
                gc.collect()

        self.df = pd.concat(chunks, ignore_index=True)

        load_time = time.time() - start_time
        print(f"Data loaded in {load_time:.2f} seconds")
        print(f"Final dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.monitor_memory_usage():.1f} MB")

        return self

    def process_text_chunk(self, chunk_data):
        """
        Process a chunk of text data for TF-IDF vectorization
        """
        chunk_idx, text_chunk = chunk_data
        try:
            tfidf_chunk = self.tfidf_vectorizer.transform(text_chunk)
            svd_chunk = self.svd.transform(tfidf_chunk)
            return chunk_idx, svd_chunk
        except Exception as e:
            print(f"Error processing chunk {chunk_idx}: {e}")
            return chunk_idx, None

    def text_to_signal_conversion_parallel(self):
        """
        Parallel text-to-signal conversion using concurrent processing
        """
        print("Starting parallel text-to-signal conversion...")
        start_time = time.time()

        # Make sure content is clean strings
        self.df['content'] = self.df['content'].fillna('').astype(str)

        # Fit TF-IDF on sample for memory efficiency
        sample_size = min(10000, len(self.df))
        sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
        sample_text = self.df.iloc[sample_indices]['content']

        print(f"Fitting TF-IDF on sample of {sample_size} tweets...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            max_df=0.85,
            min_df=3,
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='unicode'
        )

        # Fit on sample and transform sample for SVD fitting
        sample_tfidf = self.tfidf_vectorizer.fit_transform(sample_text)

        print("Fitting SVD on sample...")
        n_features = sample_tfidf.shape[1]
        if n_features == 0:
            raise ValueError(
                "TF-IDF produced 0 features. Relax vectorizer params "
                "(e.g., lower min_df, raise max_features, adjust ngram_range)."
            )
        # TruncatedSVD allows n_components <= n_features
        n_components = min(20, max(1, n_features))
        print(f"Using {n_components} SVD components (from {n_features} TF-IDF features)")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(sample_tfidf)

        # Process full dataset in parallel chunks
        print("Processing full dataset in parallel...")
        content_chunks = []
        chunk_indices = []

        for i in range(0, len(self.df), self.chunk_size):
            end_idx = min(i + self.chunk_size, len(self.df))
            chunk = self.df.iloc[i:end_idx]['content']
            content_chunks.append((i // self.chunk_size, chunk))
            chunk_indices.append((i, end_idx))

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.process_text_chunk, content_chunks))

        # Combine results with correct shape
        print("Combining parallel results...")
        text_features = np.zeros((len(self.df), self.svd.n_components))

        for chunk_idx, svd_result in results:
            if svd_result is not None:
                start_idx, end_idx = chunk_indices[chunk_idx]
                text_features[start_idx:end_idx, :] = svd_result

        # Add features to dataframe
        for i in range(text_features.shape[1]):
            self.df[f'text_feature_{i + 1}'] = text_features[:, i]

        conversion_time = time.time() - start_time
        print(f"Text conversion completed in {conversion_time:.2f} seconds")

        return self

    def create_engagement_signals_parallel(self):
        """
        Create engagement signals using vectorized operations
        """
        print("Creating engagement signals...")
        start_time = time.time()

        # Ensure numeric types
        for col in ['retweets', 'likes', 'replies']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        engagement_cols = ['retweets', 'likes', 'replies']
        engagement_data = self.df[engagement_cols].values

        # Fit and transform in one step
        engagement_normalized = self.scaler.fit_transform(engagement_data)
        self.df['engagement_signal'] = np.mean(engagement_normalized, axis=1)

        # Calculate viral score
        self.df['viral_score'] = (
            self.df['retweets'] * 0.4 +
            self.df['likes'] * 0.3 +
            self.df['replies'] * 0.3
        )

        # Normalize viral score
        viral_min = self.df['viral_score'].min()
        viral_max = self.df['viral_score'].max()
        if viral_max != viral_min:
            self.df['viral_score_norm'] = (self.df['viral_score'] - viral_min) / (viral_max - viral_min)
        else:
            self.df['viral_score_norm'] = 0.5  # Default middle value if no variation

        engagement_time = time.time() - start_time
        print(f"Engagement signals created in {engagement_time:.2f} seconds")

        return self

    def aggregate_trading_signals_parallel(self):
        """
        Aggregate trading signals using vectorized operations
        """
        print("Aggregating trading signals...")
        start_time = time.time()

        # Select top text features (handle small counts gracefully)
        text_feature_cols = [col for col in self.df.columns if col.startswith('text_feature_')]
        if not text_feature_cols:
            # No text features available; fall back to zero text component
            text_component = pd.Series(np.zeros(len(self.df)), index=self.df.index)
        else:
            feature_vars = self.df[text_feature_cols].var().sort_values(ascending=False)
            top_features = feature_vars.head(min(5, len(feature_vars))).index.tolist()
            text_component = self.df[top_features].mean(axis=1)

        # Define weights
        weights = {
            'text_component': 0.5,
            'engagement_component': 0.3,
            'viral_component': 0.2
        }

        # Vectorized signal aggregation with safe normalization
        def zscore_safe(s: pd.Series):
            std = s.std()
            return (s - s.mean()) / std if std > 0 else pd.Series(np.zeros(len(s)), index=s.index)

        text_norm = zscore_safe(text_component)
        engagement_norm = zscore_safe(self.df['engagement_signal'])
        viral_norm = zscore_safe(self.df['viral_score_norm'])

        # Composite signal
        self.df['composite_signal'] = (
            weights['text_component'] * text_norm +
            weights['engagement_component'] * engagement_norm +
            weights['viral_component'] * viral_norm
        )

        # Calculate confidence intervals
        signal_std = self.df['composite_signal'].std()
        signal_mean = self.df['composite_signal'].mean()
        if signal_std > 0:
            confidence = 1 - np.abs(self.df['composite_signal'] - signal_mean) / (3 * signal_std)
        else:
            confidence = pd.Series(np.ones(len(self.df)), index=self.df.index)

        self.df['signal_confidence'] = np.clip(confidence, 0, 1)

        # Generate trading signals
        self.df['trading_signal'] = self._generate_trading_signals_vectorized()

        aggregation_time = time.time() - start_time
        print(f"Signal aggregation completed in {aggregation_time:.2f} seconds")

        return self

    def _generate_trading_signals_vectorized(self):
        """
        Vectorized trading signal generation for performance
        """
        signal = self.df['composite_signal'].values
        confidence = self.df['signal_confidence'].values

        # Vectorized conditions
        strong_buy = (signal > 0.5) & (confidence > 0.7)
        buy = (signal > 0.2) & (confidence > 0.6) & ~strong_buy
        strong_sell = (signal < -0.5) & (confidence > 0.7)
        sell = (signal < -0.2) & (confidence > 0.6) & ~strong_sell

        # Create result array
        result = np.full(len(signal), 'HOLD', dtype=object)
        result[strong_buy] = 'STRONG_BUY'
        result[buy] = 'BUY'
        result[strong_sell] = 'STRONG_SELL'
        result[sell] = 'SELL'

        return result

    def save_results_optimized(self, output_file='trading_signals_optimized.csv'):
        """
        Memory-efficient saving
        """
        print(f"Saving results to {output_file}...")
        start_time = time.time()

        # Select essential columns to reduce file size
        essential_cols = [
            'username', 'timestamp', 'content',
            'retweets', 'likes', 'replies',
            'composite_signal', 'signal_confidence', 'trading_signal'
        ]

        # Make sure timestamp is stringifiable
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')

        self.df[essential_cols].to_csv(output_file, index=False)

        save_time = time.time() - start_time
        print(f"Results saved in {save_time:.2f} seconds")

        return self

    def performance_benchmark(self):
        """
        Generate performance benchmarks and scalability metrics
        """
        print("\n=== PERFORMANCE BENCHMARKS ===")

        # Memory usage
        memory_mb = self.monitor_memory_usage()
        print(f"Peak memory usage: {memory_mb:.1f} MB")

        # Data processing rate
        total_rows = len(self.df)
        print(f"Processed {total_rows:,} rows")

        # Estimated scalability (simple heuristic)
        memory_per_row = (memory_mb / max(1, total_rows))
        max_rows_4gb = int(4000 / max(1e-6, memory_per_row))
        max_rows_16gb = int(16000 / max(1e-6, memory_per_row))

        print(f"Memory per row: {memory_per_row:.4f} MB")
        print(f"Estimated capacity (4GB RAM): {max_rows_4gb:,} rows")
        print(f"Estimated capacity (16GB RAM): {max_rows_16gb:,} rows")

        # CPU utilization estimate
        print(f"Worker threads: {self.max_workers}")
        print(f"Chunk size: {self.chunk_size}")

        return self

    def run_optimized_analysis(self):
        """
        Run the complete optimized analysis pipeline
        """
        print("Starting optimized trading signal analysis...")
        total_start_time = time.time()

        pipeline_result = (
            self.load_data_optimized()
            .text_to_signal_conversion_parallel()
            .create_engagement_signals_parallel()
            .aggregate_trading_signals_parallel()
            .save_results_optimized()
            .performance_benchmark()
        )

        total_time = time.time() - total_start_time
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Processing rate: {len(self.df) / max(1e-6, total_time):.0f} rows/second")

        return pipeline_result


# Main execution
if __name__ == "__main__":
    analyzer = OptimizedTradingSignalAnalyzer(
        data_file='cleaned_stock_market_tweets.csv',
        max_workers=4,
        chunk_size=500,
        memory_limit_gb=4
    )

    analyzer.run_optimized_analysis()

    print("\n" + "=" * 50)
    print("OPTIMIZATION FEATURES IMPLEMENTED:")
    print("✓ Concurrent processing with ThreadPoolExecutor")
    print("✓ Memory-efficient chunked data loading")
    print("✓ Vectorized operations for signal generation")
    print("✓ Memory monitoring and garbage collection")
    print("✓ Adaptive SVD components based on TF-IDF features")
    print("✓ Performance benchmarking")
    print("✓ Error handling and edge case management")
    print("=" * 50)
