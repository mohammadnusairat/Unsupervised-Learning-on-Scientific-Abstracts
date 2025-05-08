# Unsupervised Learning on Scientific Abstracts    

## 📌 Project Overview

This project applies unsupervised machine learning techniques to a dataset of **5,000 scientific abstracts**, with the goal of extracting meaningful patterns and topics from unstructured text data. The process includes **natural language processing**, **feature engineering**, **dimensionality reduction**, and **clustering**.

We ultimately cluster the abstracts using **K-Means** and visualize the clusters using **word clouds** and **3D scatter plots**.

---

## ⚙️ Technologies & Libraries Used

- **Python 3.8+**
- **pandas, numpy** — Data manipulation and numerical processing  
- **NLTK** — Tokenization, POS tagging, and lemmatization  
- **scikit-learn**
  - `TfidfVectorizer` for text vectorization
  - `TruncatedSVD` for dimensionality reduction
  - `KMeans` for clustering
  - `PCA` for 3D visualization
- **matplotlib** — Plotting and visualization
- **wordcloud** — Word cloud generation (optional; used in plotting functions)

---

## 🧠 Main Components

### 1. Text Preprocessing
- Lowercasing, removing punctuation, URLs, LaTeX artifacts
- POS tagging and lemmatization using WordNet
- Filtering top 25 high-frequency words

### 2. Feature Engineering
- Abstracts are transformed into **TF-IDF vectors**
- The sparsity of the matrix is computed to understand term distribution

### 3. Dimensionality Reduction
- Applied **TruncatedSVD** to reduce TF-IDF feature space to 50 dimensions
- PCA (3 components) is used for 3D scatterplot visualization

### 4. Clustering
- Abstracts are clustered using **KMeans**
- The best cluster configuration is selected using **BIC (Bayesian Information Criterion)** approximation
- Abstracts are segmented into `K` clusters for further analysis

### 5. Cluster Interpretation
- For each cluster:
  - Extract the **top 50 most frequent words**
  - Generate **word clouds**
  - Visualize clusters in 3D with **PCA-based scatterplots**

---

## 🧪 How to Run

1. Install required packages:
    ```bash
    pip install pandas numpy scikit-learn nltk matplotlib wordcloud
    ```

2. Download NLTK resources:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    ```

3. Run the notebook:
    - Open `main.ipynb` in Jupyter or Colab
    - Follow the instructions to execute all cells and visualize results
    - Uncomment any needed library/framework installations

4. For testing:
    - Open `main.py` which contains all required function definitions

---

## 📁 Files

- `main.ipynb` — Interactive notebook with visualizations and written responses  
- `main.py` — Core functions for processing, clustering, and evaluation  
- `abstracts_5k.csv` — CSV file containing 5,000 scientific abstracts  
- `README.md` — Project documentation

---

## ✍️ Notes

- You may experiment with different values of `K` or dimensionality reduction techniques to explore alternate clustering outcomes.
- This project demonstrates how unsupervised learning can uncover thematic structures in large-scale textual datasets.
