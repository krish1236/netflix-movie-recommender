import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Input, Concatenate
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load data from real Netflix Excel file
try:
    excel_file = '/Users/Downloads/What_We_Watched_A_Netflix_Engagement_Report_2025Jan-Jun.xlsx'
    
    shows_df = pd.read_excel(excel_file, sheet_name='Shows', skiprows=5)
    shows_df = shows_df.dropna(subset=['Title'])
    shows_df = shows_df.drop(columns=['Unnamed: 0'], errors='ignore')
    shows_df['Content_Type'] = 'TV Show'
    
    movies_df = pd.read_excel(excel_file, sheet_name='Movies', skiprows=5)
    movies_df = movies_df.dropna(subset=['Title'])
    movies_df = movies_df.drop(columns=['Unnamed: 0'], errors='ignore')
    movies_df['Content_Type'] = 'Movie'
    
    df = pd.concat([shows_df, movies_df], ignore_index=True)
    df = df.dropna(subset=['Title'])
    
    print(f"Successfully loaded {len(shows_df)} shows and {len(movies_df)} movies")
    
except Exception as e:
    print(f"Error reading Excel file: {e}")
    print("Falling back to sample CSV data...")
    df = pd.read_csv('../data/sample_netflix_data.csv')
    df = df.dropna()

print("Available columns:", df.columns.tolist())
print("Data shape:", df.shape)
print("Sample data:")
print(df.head())

# Create categorical features from available data
df['Content Type'] = df['Title'].apply(lambda x: 'Series' if ('Season' in str(x) or 'Limited Series' in str(x)) else 'Movie')
df['Language'] = df['Title'].apply(lambda x: 'Korean' if any(char in str(x) for char in ['오', '게', '임']) else 'English')

# Encode categoricals
le_type = LabelEncoder()
le_lang = LabelEncoder()
le_global = LabelEncoder()
df['content_type_encoded'] = le_type.fit_transform(df['Content Type'])
df['language_encoded'] = le_lang.fit_transform(df['Language'])
df['available_globally_encoded'] = le_global.fit_transform(df['Available Globally?'])

# Bin release date into eras
df['release_date'] = pd.to_datetime(df['Release Date'])
df['era_bin'] = pd.cut(df['release_date'].dt.year, bins=[0, 2010, 2020, 2025, float('inf')], labels=[0,1,2,3])

# Normalize hours viewed for popularity weight
df['hours_normalized'] = (df['Hours Viewed'] - df['Hours Viewed'].min()) / (df['Hours Viewed'].max() - df['Hours Viewed'].min())

# Title tokenization for text embedding
def tokenize_title(title):
    return re.findall(r'\b\w+\b', title.lower())[:10]

df['title_tokens'] = df['Title'].apply(tokenize_title)
all_words = [word for tokens in df['title_tokens'] for word in tokens]
vocab = list(set(all_words))
word_to_id = {w: i for i, w in enumerate(vocab)}
df['title_token_ids'] = df['title_tokens'].apply(lambda tokens: [word_to_id.get(w, 0) for w in tokens])

# Neural network parameters
max_tokens = 10
vocab_size = len(vocab)
embedding_dim = 32

# Build neural network model
title_input = Input(shape=(max_tokens,), name='title_tokens')
meta_input = Input(shape=(4,), name='metadata')

title_embed = Embedding(vocab_size, embedding_dim)(title_input)
title_pooled = tf.keras.layers.GlobalAveragePooling1D()(title_embed)

meta_embed = Dense(embedding_dim, activation='relu')(meta_input)

combined = Concatenate()([title_pooled, meta_embed])
output = Dense(embedding_dim, activation='relu')(combined)
model = Model(inputs=[title_input, meta_input], outputs=output)

# Train model with dummy data
model.compile(optimizer='adam', loss='mse')
X_title_dummy = np.random.randint(0, vocab_size, (1000, max_tokens))
X_meta_dummy = np.random.randint(0, 4, (1000, 4))
y_dummy = np.random.rand(1000, embedding_dim)
model.fit([X_title_dummy, X_meta_dummy], y_dummy, epochs=10, verbose=0)

# Generate embeddings for all titles
df['title_token_ids_padded'] = df['title_token_ids'].apply(lambda x: x + [0]*(max_tokens - len(x)))
title_tokens_padded = np.array(list(df['title_token_ids_padded']))
meta_features = df[['era_bin', 'content_type_encoded', 'language_encoded', 'available_globally_encoded']].values
embeddings = model.predict([title_tokens_padded, meta_features])
df['embedding'] = list(embeddings)


def recommend(title, df, k=10):
    if title not in df['Title'].values:
        return "Title not found."
    
    idx = df[df['Title'] == title].index[0]
    query_embed = df.iloc[idx]['embedding']
    
    sim_scores = cosine_similarity([query_embed], df['embedding'].tolist())[0]
    df['similarity'] = sim_scores
    
    df['score'] = df['similarity'] * df['hours_normalized'] * (df['Available Globally?'] == 'Yes').astype(int)
    
    recs = df[df['Title'] != title].nlargest(k, 'score')[['Title', 'Hours Viewed', 'score']]
    return recs

# Test recommendations
if len(df) > 0:
    test_title = df['Title'].iloc[0]
    print(f"\nTesting recommendations for: '{test_title}'")
    print(recommend(test_title, df))
else:
    print("No data found in the dataset")
