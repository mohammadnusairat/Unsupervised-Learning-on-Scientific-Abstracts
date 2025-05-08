import nltk
import pandas as pd
import numpy as np
import sklearn.cluster
import re
import string
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import PCA
from wordcloud import WordCloud
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt


"""
Instructions:

    Copy over bodies of functions listed below from your
    main.ipynb for testing.

"""


posMapping = {
# "First_Letter by nltk.pos_tag":"POS_for_lemmatizer"
    "N":'n',
    "V":'v',
    "J":'a',
    "R":'r'
}

def process(text, lemmatizer=nltk.WordNetLemmatizer()):

    """ 
    Normalizes case and handles punctuation

    Parameters
    ----------
    text: str: 
        raw text
    lemmatizer: nltk.WordNetLemmatizer() 
        an instance of a class implementing the lemmatize() method
        (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    
    Returns
    -------
    list(str)
        tokenized text
    """

    # Step 1: Convert to lower case
    text = text.lower()
    
    # Step 2: Remove URLs
    text = re.sub(r'http:/\S+|www\S+|https:/\S+', '', text, flags=re.MULTILINE)
    
    # Step 3: Remove inline LaTeX equations
    text = re.sub(r'\$.*?\$', '', text)  # Strict match for $...$

    # Step 4: Remove LaTeX text decorations (e.g., \textit{}, \underline{})
    text = re.sub(r'\\text\w*{(.*?)}', r'\1', text)

    # Step 5: Replace hyphens with spaces for compound words
    text = re.sub(r'[-]', ' ', text)

    # Step 6: Remove or handle punctuation
    # Handle specific punctuation cases and remove others
    text = re.sub(r"'s", '', text)  # Remove 's
    text = re.sub(r"'", '', text)  # Replace other apostrophes with ''
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) # Replace any punctuation with a space
    
    # Step 7: Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Step 8: Lemmatize tokens based on POS
    lemmatized_tokens = []
    pos_tags = nltk.pos_tag(tokens)  # Get POS tags
    
    for word, tag in pos_tags:
        # Get the first letter of the POS tag
        first_letter = tag[0]
        pos = posMapping.get(first_letter, 'n')  # Default to noun if not found
        try:
            lemmatized_word = lemmatizer.lemmatize(word, pos=pos)
            lemmatized_tokens.append(lemmatized_word)
        except Exception:
            continue  # Ignore words that cannot be lemmatized

    return lemmatized_tokens

def process_abstracts(df, lemmatizer=nltk.WordNetLemmatizer()):
    """
    Process all abstracts in the dataframe using calls to the process() function

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe object containing a column 'abstract'
    lemmatizer: nltk.WordNetLemmatizer
        An instance of the WordNetLemmatizer class implementing the lemmatize() method
    
    Returns
    -------
    pd.DataFrame
        Same as the dataframe 'df' except for the 'abstracts' column transformed
        from str to list(str) using function `process()`
    """

    # Apply the process() function to the 'abstract' column
    df['abstract'] = df['abstract'].apply(lambda x: process(x, lemmatizer))
    
    return df

def filter_words(tokenized_text:list, words_to_filter:set):
    """
    Returns a tokens list with the words in `words_to_filter`
    filtered out

    Parameters
    ----------
    tokenized_text : list(str)
        List of text tokens
    words_to_filter : set(str)
        Set of words to filter out

    Returns
    -------
    list(str)
        List of text tokens with words in
        `words_to_filter` filtered out
    """

    # Use filter() - https://docs.python.org/3/library/functions.html#filter
    
    # Use filter() to exclude tokens present in words_to_filter
    filtered_tokens = list(filter(lambda token: token not in words_to_filter, tokenized_text))
    
    return filtered_tokens

def filter_words_in_series(tokenized_text_ser:pd.Series, words_to_filter:set):
    """
    Returns a `pd.Series` object containing a list of tokenized
    text with words in `words_to_filter` removed

    Parameters
    ----------
    tokenized_text_ser : pd.Series
        Series object containing a list of tokenized texts
    words_to_filter : set(str)
        Set of words to filter out

    Returns
    -------
    pd.Series
        Series object containing the list of tokenized texts
        with words in `words_to_filter` removed
    """

    # Apply the filter_words function to each element of the Series
    filtered_series = tokenized_text_ser.apply(lambda tokens: filter_words(tokens, words_to_filter))
    
    return filtered_series

def top_25_hf_words(abstracts_ser:pd.Series):
    """
    Returns the top 25 most commonly occurring words
    across all abstracts in a series object containing
    a list of abstract texts

    Parameters
    ----------
    abstracts_ser : pd.Series
        Series objects containing a list of abstracts

    Returns
    -------
    set(str)
        Set of top 25 high frequency words
    """

    # Flatten the tokenized lists into a single list
    all_tokens = [token for tokens in abstracts_ser for token in tokens]
    
    # Dictionary to store word frequencies
    word_counts = {}
    for token in all_tokens:
        if token in word_counts:
            word_counts[token] += 1
        else:
            word_counts[token] = 1
    
    # Sort the dictionary by frequency in descending order and get the top 25 keys
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_25_words = {word for word, count in sorted_words[:25]}
    
    return top_25_words

def create_features_tfidf(abstracts:pd.DataFrame):
    """
    Compute TF-IDF features for the abstracts dataset

    Parameters
    ----------
    abstracts : pd.DataFrame
        Dataframe with a column named 'abstract'
        containing list of abstracts
    
    Returns
    -------
    TfidfVectorizer()
        Instance of the class TfidfVectorizer
    scipy.sparse._csr_matrix
        TF-IDF feature matrix
    """

    # Combine tokenized abstracts into single strings for TF-IDF processing
    text_data = abstracts['abstract'].apply(lambda tokens: ' '.join(tokens))
    
    # Create an instance of TfidfVectorizer
    vectorizer = TfidfVectorizer(
        min_df=2,        # Ignore terms that appear in fewer than 2 documents
        tokenizer=lambda x: x.split(),  # Use pre-tokenized data
        lowercase=False  # Data is already in lowercase
    )
    
    # Fit and transform the data
    tfidf_matrix = vectorizer.fit_transform(text_data)
    
    return tfidf_matrix, vectorizer

def sparsity(A:np.ndarray):
    """
    Determine the sparsity of a matrix
    by dividing the number of non-zero
    elements with the total number of
    elements of matrix `matrix`
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix
    
    Returns
    -------
    float
        A measure of the sparsity of A
    """

    # Total number of elements in the matrix
    total_elements = A.size
    
    # Count the number of zero elements
    zero_elements = total_elements - np.count_nonzero(A)
    
    # Calculate sparsity
    sparsity_measure = zero_elements / total_elements
    
    return sparsity_measure

def reduce_tfidf_dimensions(feat_mat:np.ndarray, dim:int=50):
    """
    Reduce dimensionality of a sparse feature matrix as input
    
    Parameters
    ----------
    feat_mat:np.ndarray
        Sparse feature matrix
    dim:int
        Dimensionality of output feature matrix (i.e. number of
        columns)

    Returns
    -------
    np.ndarray
        Dense feature matrix
    """

    # Instantiate TruncatedSVD with the specified number of components
    svd = TruncatedSVD(n_components=dim)
    
    # Perform dimensionality reduction
    reduced_matrix = svd.fit_transform(feat_mat)
    
    return reduced_matrix

def fit_k_means(feat_mat:np.ndarray, cluster_count=5, iterations=5):
    """
    Fit the abstracts data feature vectors into `cluster_count`
    clusters using the K Means cluster algorithm & select
    the best clustering out of `iterations` number
    of clusterings

    Parameters
    ----------
    feat_mat : np.ndarray
        Feature matrix encoding feature vectors
        for all 5k abstracts
    cluster_count : int
        Number of distinct clusters
    iterations : int
        Number of training runs
    
    Returns
    -------
    sklearn.cluster.KMeans
        Instance of the sklearn.cluster.KMeans class
        representing the best clustering
    """

    best_kmeans = None
    best_wcss = np.inf  # Start with infinity for comparison
    
    for _ in range(iterations):
        # Create and fit KMeans
        kmeans = KMeans(n_clusters=cluster_count, init='k-means++', random_state=None, n_init=1)
        kmeans.fit(feat_mat)
        
        # Check WCSS (inertia_)
        if kmeans.inertia_ < best_wcss:
            best_wcss = kmeans.inertia_
            best_kmeans = kmeans
    
    return best_kmeans

def best_bic_clustering(abstracts_feats:np.ndarray, iterations:int=5, K_range:list=[1,20]):
    """
    Fit K-Means model for a range of K values
    and determine the best model that has
    the least BIC score

    Parameters
    ----------
    abstracts_feats : np.ndarray
        Feature vector matrix storing vectors representing
        each abstract
    iterations : int
        Number of training runs
    K_range : list(int)
        List of two integer values specifying the range of
        K values to test K-Means models across

    Returns
    -------
    sklearn.cluster.KMeans
        Instance of the KMeans class representing the
        best K-Means model across different K values
    """

    best_kmeans = None
    best_bic_score = np.inf
    feature_dimensionality = abstracts_feats.shape[1]

    for k in range(K_range[0], K_range[1] + 1):
        # Perform K-Means clustering
        kmeans = fit_k_means(abstracts_feats, cluster_count=k, iterations=iterations)

        # Calculate the WCSS (first term of BIC score)
        wcss = kmeans.inertia_

        # Calculate BIC score: WCSS + log(D) * K
        bic_score = wcss + np.log(feature_dimensionality) * k

        # Update the best model if the current BIC score is lower
        if bic_score < best_bic_score:
            best_bic_score = bic_score
            best_kmeans = kmeans

    return best_kmeans

def filter_abstracts(best_clustering:sklearn.cluster.KMeans, abstracts_feats:np.ndarray, abstracts:pd.DataFrame):
    """
    Filter abstracts based on their assigned cluster
    
    
    Parameters
    ----------
    best_clustering:sklearn.cluster.KMeans
        Instance of the `KMeans` class representing the
        best clustering in terms of the number of clusters
        K
    abstracts_feats:np.ndarray
        Feature matrix containing vectors representing
        each abstract in the dataset
    abstracts : pd.DataFrame
        Dataframe with a column 'abstract' containing abstracts
        as tokenized texts

    Returns
    -------
    tuple(pd.Series)
        Python tuple object containing K pd.Series objects,
        each containing abstracts belonging to one of the
        K clustered labels
    """

    # Predict cluster labels for the abstracts
    cluster_labels = best_clustering.predict(abstracts_feats)
    
    # Number of clusters
    num_clusters = best_clustering.n_clusters

    # Create a tuple of pd.Series for each cluster
    clusters = tuple(
        abstracts['abstract'][cluster_labels == cluster].reset_index(drop=True)
        for cluster in range(num_clusters)
    )
    
    return clusters
    
def top_50_freq_dict(clustered_abstracts:tuple):
    """
    Compute K dictionaries mapping most
    frequent words in each of the K clusters
    to their frequencies

    Parameters
    ----------
    clustered_abstracts:tuple
        Tuple of K `pd.Series` objects each
        containing abstract texts belonging
        to one of the K clusters
    
    Returns
    -------
    tuple(dict)
        Tuple of K dictionaries each mapping
        top-50 most frequent words from each
        of the K clusters to their frequencies
        of occurrence
    """

    # Initialize an empty list to store frequency dictionaries
    freq_dicts = []

    for cluster in clustered_abstracts:
        # Flatten all abstracts in the cluster into a single list of words
        all_words = [word for abstract in cluster for word in abstract]
        
        # Manually calculate word frequencies using a dictionary
        word_counts = {}
        for word in all_words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Sort the dictionary by frequency in descending order and select the top 50 words
        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_50_words = dict(sorted_word_counts[:50])
        
        # Append the frequency dictionary to the list
        freq_dicts.append(top_50_words)
    
    # Convert the list of dictionaries into a tuple
    return tuple(freq_dicts)

def generate_word_cloud( word_dictionary:dict):
    """
    Generate a word cloud based on provided dictionary of
    words mapped to their frequencies

    Parameters
    ----------
    word_dictionary:dict
        Dictionary mapping words to their frequency of
        occurrence in some text

    Returns
    -------
    WordCloud
        Instance of the class WordCloud
    """

    # Create an instance of WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='black')
    
    # Generate the word cloud from the frequency dictionary
    wordcloud.generate_from_frequencies(word_dictionary)
    
    return wordcloud

def plot_word_clouds(clustered_abstracts_top50:tuple):
    """
    Plot word clouds of all K abstracts clusters

    Parameters
    ----------
    clustered_abstracts_top50:tuple
        Tuple of dictionaries of top 50 most ocurring
        words in each of the K clusters
    """

    # Number of clusters (K)
    K = len(clustered_abstracts_top50)
    
    # Set up a grid of subplots (5 rows x 4 cols grid)
    rows, cols = 5, 4
    fig = plt.figure(figsize=(12, 12))  # Adjust size for better visualization
    
    # Loop through each cluster and plot its word cloud
    for i, word_freqs in enumerate(clustered_abstracts_top50):
        # Generate the word cloud
        wc = generate_word_cloud(word_freqs)
        
        # Add a subplot
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_facecolor('black') # Set black bg for subplots
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')  # Turn off axis
        
        # Add title to indicate cluster
        ax.set_title(f'Cluster {i + 1}', fontsize=14)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def clusters_3d_scatterplot(abstracts_feats:np.ndarray, clustering:KMeans, abstracts:pd.DataFrame):
    """
    Generate a 3D Scatterplot of the best K-Means clustering
    model using PCA

    Parameters
    ----------
    abstracts_feats : np.ndarray
        Feature matrix for abstracts data
    clustering:KMeans
        Instance of KMeans class
    abstracts:pd.DataFrame
        Dataframe containing a column
        'abstract'
    """

    # Step 1: Reduce dimensions with PCA
    pca = PCA(n_components=3)
    reduced_feats = pca.fit_transform(abstracts_feats)
    
    # Step 2: Get cluster labels
    labels = clustering.labels_
    
    # Step 3: Get top 5 words for each cluster for the legend
    top_words_per_cluster = []
    for cluster in range(clustering.n_clusters):
        cluster_abstracts = abstracts['abstract'][labels == cluster]
        words = [word for abstract in cluster_abstracts for word in abstract]
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = [word[0] for word in sorted_words[:5]]
        top_words_per_cluster.append(", ".join(top_words))
    
    # Step 4: Create a 3D scatter plot
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot with cluster-specific colors
    scatter = ax.scatter(
        reduced_feats[:, 0], 
        reduced_feats[:, 1], 
        reduced_feats[:, 2], 
        c=labels, 
        cmap='turbo', 
        s=15
    )
    
    # Set plot limits
    ax.set_xlim([-0.18, 0.18])
    ax.set_ylim([-0.18, 0.18])
    ax.set_zlim([-0.18, 0.18])
    
    # Label axes
    ax.set_xlabel("PCA Feature 1")
    ax.set_ylabel("PCA Feature 2")
    ax.set_zlabel("PCA Feature 3")
    
    # Add legend with cluster topics
    legend_labels = [f"Cluster {i + 1} with topics: {words}" for i, words in enumerate(top_words_per_cluster)]
    ax.legend(
        handles=scatter.legend_elements()[0], 
        labels=legend_labels, 
        loc="center right", 
        bbox_to_anchor=(1.05, 1), 
        title="Clusters"
    )
    
    # Set title
    plt.title("KMeans Clustering 3D Plot")
    plt.tight_layout()
    plt.show()
