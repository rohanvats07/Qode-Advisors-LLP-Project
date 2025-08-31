import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradingSignalAnalyzer:
    """
        Convert textual Twitter data into quantitative trading signals
    """
    
    def __init__(self, data_file='cleaned_stock_market_tweets.csv'):
        """Initialize with cleaned data file"""
        self.data_file = data_file
        self.df = None
        self.tfidf_vectorizer = None
        self.svd = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
            Load and prepare the cleaned data
        """
        self.df = pd.read_csv(self.data_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        print(f"Loaded {len(self.df)} tweets")
        return self
        
    def text_to_signal_conversion(self):
        """
            Convert tweet content to numerical vectors using TF-IDF
        """

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,        # Limit features for memory efficiency
            stop_words='english',     # Remove common English stopwords
            max_df=0.85,             # Ignore terms appearing in >85% of documents
            min_df=3,                # Ignore terms appearing in <3 documents
            ngram_range=(1, 2),      # Include unigrams and bigrams
            lowercase=True,          # Convert to lowercase
            strip_accents='unicode'  # Handle Unicode characters
        )
        
        # Fit and transform tweet content
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['content'])
        
        # Dimensionality reduction using TruncatedSVD
        n_features = tfidf_matrix.shape[1]
        n_components = min(20, n_features - 1)  # must be less than n_features
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        text_features = self.svd.fit_transform(tfidf_matrix)
        
        # Add text features to dataframe
        for i in range(text_features.shape[1]):
            self.df[f'text_feature_{i+1}'] = text_features[:, i]
                
        print(f"Generated {text_features.shape[1]} text features")
        return self
    
    def create_engagement_signals(self):
        """
            Create engagement-based trading signals
        """

        # Normalize engagement metrics
        engagement_cols = ['retweets', 'likes', 'replies']
        engagement_data = self.df[engagement_cols].values
        engagement_normalized = self.scaler.fit_transform(engagement_data)
        
        # Create engagement signals
        self.df['engagement_signal'] = np.mean(engagement_normalized, axis=1)
        self.df['viral_score'] = (
            self.df['retweets'] * 0.4 + 
            self.df['likes'] * 0.3 + 
            self.df['replies'] * 0.3
        )
        
        # Normalize viral score
        self.df['viral_score_norm'] = (
            self.df['viral_score'] - self.df['viral_score'].min()
        ) / (self.df['viral_score'].max() - self.df['viral_score'].min())
        
        return self
        
    def aggregate_trading_signals(self):
        """
            Combine multiple features into composite trading signals with confidence intervals
        """
        
        # Select top text features based on variance
        text_feature_cols = [col for col in self.df.columns if col.startswith('text_feature_')]
        feature_vars = self.df[text_feature_cols].var().sort_values(ascending=False)
        top_features = feature_vars.head(5).index.tolist()
        
        # Create composite signal using weighted combination
        weights = {
            'text_component': 0.5,
            'engagement_component': 0.3,
            'viral_component': 0.2
        }
        
        # Text component (average of top 5 text features)
        text_component = self.df[top_features].mean(axis=1)
        
        # Normalize components
        text_norm = (text_component - text_component.mean()) / text_component.std()
        engagement_norm = (self.df['engagement_signal'] - self.df['engagement_signal'].mean()) / self.df['engagement_signal'].std()
        viral_norm = (self.df['viral_score_norm'] - self.df['viral_score_norm'].mean()) / self.df['viral_score_norm'].std()
        
        # Composite signal
        self.df['composite_signal'] = (
            weights['text_component'] * text_norm +
            weights['engagement_component'] * engagement_norm +
            weights['viral_component'] * viral_norm
        )
        
        # Calculate confidence intervals
        self.df['signal_confidence'] = self._calculate_confidence_intervals()
        
        # Create trading recommendations
        self.df['trading_signal'] = self._generate_trading_signals()
        
        return self
        
    def _calculate_confidence_intervals(self):
        """
            Calculate confidence intervals for signals
        """
        signal_std = self.df['composite_signal'].std()
        n = len(self.df)
        margin_error = 1.96 * signal_std / np.sqrt(n)
        
        # Confidence score based on distance from mean
        signal_mean = self.df['composite_signal'].mean()
        confidence = 1 - np.abs(self.df['composite_signal'] - signal_mean) / (3 * signal_std)
        return np.clip(confidence, 0, 1)  # Clip to [0, 1] range
        
    def _generate_trading_signals(self):
        """
            Generate discrete trading signals
        """
        signals = []
        for idx, row in self.df.iterrows():
            signal = row['composite_signal']
            confidence = row['signal_confidence']
            
            if signal > 0.5 and confidence > 0.7:
                signals.append('STRONG_BUY')
            elif signal > 0.2 and confidence > 0.6:
                signals.append('BUY')
            elif signal < -0.5 and confidence > 0.7:
                signals.append('STRONG_SELL')
            elif signal < -0.2 and confidence > 0.6:
                signals.append('SELL')
            else:
                signals.append('HOLD')
                
        return signals
        
    def memory_efficient_visualization(self, sample_size=1000):
        """
            Create memory-efficient visualizations using data sampling
        """
        
        # Sample data for visualization
        sample_df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)
        sample_df = sample_df.sort_values('timestamp')
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Composite Signal Over Time
        axes[0, 0].plot(sample_df['timestamp'], sample_df['composite_signal'], 
                       alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Composite Trading Signal Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Signal Strength')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Signal Distribution
        axes[0, 1].hist(sample_df['composite_signal'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Signal Distribution')
        axes[0, 1].set_xlabel('Signal Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confidence vs Signal Scatter
        scatter = axes[1, 0].scatter(sample_df['composite_signal'], sample_df['signal_confidence'], 
                                   alpha=0.6, c=sample_df['viral_score_norm'], cmap='viridis')
        axes[1, 0].set_title('Signal Confidence vs Signal Strength')
        axes[1, 0].set_xlabel('Signal Strength')
        axes[1, 0].set_ylabel('Confidence')
        plt.colorbar(scatter, ax=axes[1, 0], label='Viral Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Trading Signal Distribution
        signal_counts = sample_df['trading_signal'].value_counts()
        axes[1, 1].pie(signal_counts.values, labels=signal_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Trading Signal Distribution')
        
        plt.tight_layout()
        plt.savefig('trading_signals_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
        
    def generate_summary_report(self):
        """Generate comprehensive analysis summary"""
        print("\n=== TRADING SIGNAL ANALYSIS SUMMARY ===")
        
        # Basic statistics
        print(f"Total tweets analyzed: {len(self.df)}")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"Unique users: {self.df['username'].nunique()}")
        
        # Signal statistics
        print(f"\nComposite Signal Statistics:")
        print(f"Mean: {self.df['composite_signal'].mean():.4f}")
        print(f"Std: {self.df['composite_signal'].std():.4f}")
        print(f"Min: {self.df['composite_signal'].min():.4f}")
        print(f"Max: {self.df['composite_signal'].max():.4f}")
        
        # Trading signal distribution
        print(f"\nTrading Signal Distribution:")
        signal_dist = self.df['trading_signal'].value_counts()
        for signal, count in signal_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"{signal}: {count} ({percentage:.1f}%)")
            
        # Confidence statistics
        print(f"\nConfidence Statistics:")
        print(f"Average confidence: {self.df['signal_confidence'].mean():.4f}")
        high_conf = (self.df['signal_confidence'] > 0.7).sum()
        print(f"High confidence signals (>0.7): {high_conf} ({high_conf/len(self.df)*100:.1f}%)")
        
        return self
        
    def save_results(self, output_file='trading_signals_analysis.csv'):
        """Save analysis results"""
        self.df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        return self
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        return (self.load_data()
                   .text_to_signal_conversion()
                   .create_engagement_signals()
                   .aggregate_trading_signals()
                   .memory_efficient_visualization()
                   .generate_summary_report()
                   .save_results())

if __name__ == "__main__":
    analyzer = TradingSignalAnalyzer('cleaned_stock_market_tweets.csv')
    analyzer.run_complete_analysis()
    
    print("\n<------------------ Analysis Complete ------------------------->")
    print("Files generated:")
    print("- trading_signals_analysis.csv (detailed results)")
    print("- trading_signals_analysis.png (visualizations)")
