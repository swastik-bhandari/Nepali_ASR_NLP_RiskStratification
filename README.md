## Project Overview

This project implements a comprehensive text preprocessing and classification pipeline for Nepali language text, specifically designed for maternal and neonatal health risk assessment. The system processes Nepali text (representing voices of Female Community Health Volunteers and Community Health Workers) through multiple stages including normalization, tokenization, stopword removal, and suffix stripping before feeding it into machine learning models for risk classification.

**Note**: This project uses synthetic data representing FCHV and CHW communications. Implementation details will be updated as the project progresses.


## Features

- **Comprehensive Nepali Text Normalization**: Handles various Unicode variations, diacritics, and character mappings specific to Nepali script
- **Stopword Removal**: Removes common Nepali words that don't contribute to meaning
- **Suffix Stripping**: Handles Nepali grammatical suffixes and postpositions
- **Multi-Model Comparison**: Evaluates SVM, Logistic Regression, and Graph Convolutional Networks (GCN)
- **Systematic Experimentation**: Tests different preprocessing combinations to identify optimal configurations

## Pipeline Architecture

```
Text Input (FCHV/CHW Communications)
          ↓
    Normalization
          ↓
    Tokenization
          ↓
  Stopwords Removal
          ↓
   Suffix Stripping
          ↓
    Vectorization
          ↓
      ML Model
          ↓
Risk Classification (High/Medium/Low)
```

## Preprocessing Components

### 1. Normalization

The normalization component handles various Unicode variations and standardizes Nepali text:

- **Vowel Normalization**: Maps long vowels to short vowels (ई→इ, ऊ→उ)
- **Character Equivalence**: Handles phonetically similar characters (व→ब, श→स, ष→स)
- **Diacritic Removal**: Removes various diacritical marks (ः, ँ, ं, ़, ्)
- **Compound Character Simplification**: Simplifies conjunct characters (ज्ञ→ग्य, क्ष→छ्य, त्र→तिर)
- **Punctuation Removal**: Removes all punctuation marks and special characters

**Example:**
```
Before: महिलालाई छाती दुखिरहेको छ र सास फेर्न गाह्रो छ।
After:  महिलालाइ छाति दुखिरहेको छ र सास फेर्न गाह्रो छ
```

### 2. Stopwords

The system includes a comprehensive list of 200+ Nepali stopwords including:
- Pronouns (म, तिमी, उ, हामी, etc.)
- Common verbs (छ, थियो, हुनेछ, etc.)
- Conjunctions (र, वा, तर, etc.)
- Prepositions (मा, बाट, लागि, etc.)
- Common adverbs and particles

### 3. Suffix Stripping

Handles 100+ Nepali grammatical suffixes and postpositions:
- Case markers (का, की, को, ले, लाई, etc.)
- Plural markers (हरू, हरु)
- Locative markers (मा, भित्र, माथि, तिर, etc.)
- Associative markers (संग, सँग, सित, etc.)
- Honorifics (जी, ज्यू)
- Complex compound suffixes (हरूसँग, हरूसमेत, हरूमध्ये, etc.)

## Experimental Setup

The project conducts systematic experiments to evaluate the impact of different preprocessing stages on model performance.

### Experiment Matrix

| Experiment | Normalization | Stopwords | Suffix Stripping | Model |
|------------|---------------|-----------|------------------|-------|
| **SVM Series** |
| E1 | ❌ | ❌ | ❌ | SVM |
| E2 | ✅ | ❌ | ❌ | SVM |
| E3 | ✅ | ✅ | ❌ | SVM |
| E4 | ✅ | ✅ | ✅ | SVM |
| **Logistic Regression Series** |
| E1 | ❌ | ❌ | ❌ | Logistic Regression |
| E2 | ✅ | ❌ | ❌ | Logistic Regression |
| E3 | ✅ | ✅ | ❌ | Logistic Regression |
| E4 | ✅ | ✅ | ✅ | Logistic Regression |
| **GCN Series** |
| E1 | ❌ | ❌ | ❌ | GCN (Transductive) |
| E2 | ✅ | ❌ | ❌ | GCN (Transductive) |
| E3 | ✅ | ✅ | ❌ | GCN (Transductive) |
| E4 | ✅ | ✅ | ✅ | GCN (Transductive) |

### Models

1. **Support Vector Machine (SVM)**: Traditional machine learning approach with vectorized text features
2. **Logistic Regression**: Baseline statistical model for multi-class classification
3. **Graph Convolutional Network (GCN)**: Deep learning approach using transductive learning for graph-structured data
