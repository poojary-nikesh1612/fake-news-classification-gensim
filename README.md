# Fake News Classification using Gensim

This project performs fake news classification using pretrained word embeddings from Gensim and Logistic Regression.

## Technologies Used

- Python
- Pandas
- NumPy
- spaCy
- Gensim
- Scikit-learn

## Project Workflow

1. Load fake and real news dataset
2. Preprocess text using spaCy
   - Stopword removal
   - Punctuation removal
   - Lemmatization
3. Load pretrained word embeddings using Gensim
4. Generate sentence vectors using mean word embeddings
5. Train Logistic Regression classifier
6. Evaluate model performance

## Pretrained Embedding Model

The project uses pretrained embeddings loaded from Gensim Downloader.

```python
import gensim.downloader as api

nlp = api.load("glove-wiki-gigaword-100")
```

## Sentence Vector Creation

Word vectors are converted into sentence vectors using average embeddings.

```python
nlp.get_mean_vector(filtered_tokens)
```

## Model Training

Classifier used:

```python
LogisticRegression()
```

## Dataset

Dataset contains:
- Fake news articles
- Real news articles

## Evaluation

Model performance evaluated using:

```python
classification_report()
```
