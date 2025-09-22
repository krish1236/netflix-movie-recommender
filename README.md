# Netflix Movie Recommender

A neural network-based movie recommendation system that analyzes Netflix engagement data to provide personalized content suggestions.

## 🎯 What It Does

This system creates intelligent movie and TV show recommendations by:

- **Analyzing 16,000+ Netflix titles** from real engagement data (Jan-June 2025)
- **Learning from multiple data sources**: Title text, content type, language, release era, and popularity metrics
- **Using neural embeddings** to capture semantic similarity between titles
- **Weighing recommendations** by popularity and global availability

## 🧠 How It Works

### 1. **Multi-Modal Learning**
- **Text Analysis**: Tokenizes movie titles and learns word embeddings
- **Metadata Processing**: Encodes content type, language, and release era
- **Combined Features**: Merges text and metadata through neural networks

### 2. **Neural Architecture**
```
Title Input → Embedding Layer → Global Average Pooling
                                                    ↓
                                              Concatenate → Dense → Final Embedding
                                                    ↑
Metadata Input → Dense Layer ────────────────────────┘
```

### 3. **Recommendation Engine**
- Computes cosine similarity between content embeddings  
- Weights results by popularity (hours viewed) and availability
- Returns top-K most relevant recommendations

## 📊 Data Sources

- **Netflix Engagement Report** (Jan-June 2025)
- **7,508 TV Shows** and **8,674 Movies**
- **Metrics**: Hours viewed, views, release dates, runtime, global availability

## 🚀 Quick Start

### Prerequisites
```bash
python 3.8+
pandas
tensorflow
numpy
scikit-learn
openpyxl
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/netflix-movie-recommender.git
cd netflix-movie-recommender

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas tensorflow numpy scikit-learn openpyxl
```

### Usage
```bash
python src/movie_suggestion.py
```

### Example Output
```
Testing recommendations for: 'Adolescence: Limited Series'

                                   Title  Hours Viewed     score
1    Squid Game: Season 2 // 오징어 게임: 시즌 2     840300000  0.990516
18   When Life Gives You Tangerines...     577000000  0.682484  
9    Ginny & Georgia: Season 3             508200000  0.604422
8    The Night Agent: Season 2             457800000  0.543185
2    Squid Game: Season 3 // 오징어 게임: 시즌 3     438600000  0.519024
```

## 🛠 Project Structure

```
netflix-movie-recommender/
├── README.md
├── src/
│   └── movie_suggestion.py    # Main recommendation engine
├── data/
│   └── sample_netflix_data.csv # Sample dataset
└── venv/                      # Virtual environment
```

## 🎯 Features

- ✅ **Real Netflix Data**: Works with actual engagement reports
- ✅ **Multi-Language Support**: Detects Korean, English content
- ✅ **Content Type Inference**: Automatically categorizes movies vs TV shows  
- ✅ **Era-Based Analysis**: Considers release time periods
- ✅ **Popularity Weighting**: Factors in viewership data
- ✅ **Neural Embeddings**: 32-dimensional semantic representations

## 🔬 Technical Details

### Neural Network Architecture
- **Embedding Layer**: Converts title tokens to dense vectors
- **Global Average Pooling**: Aggregates word embeddings
- **Dense Layers**: Processes metadata and combines features
- **Output**: 32-dimensional content embeddings

### Similarity Computation
- **Cosine Similarity**: Measures content similarity
- **Popularity Score**: Hours viewed normalization
- **Availability Filter**: Global availability weighting

## 📈 Performance

- **Dataset Size**: 16,182 titles
- **Vocabulary Size**: Dynamic based on title corpus
- **Embedding Dimensions**: 32
- **Training**: 10 epochs with dummy data (proof of concept)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- Netflix for providing engagement data
- TensorFlow team for the machine learning framework
- The open-source Python community

---

**Built with ❤️ using Python, TensorFlow, and real Netflix data**
