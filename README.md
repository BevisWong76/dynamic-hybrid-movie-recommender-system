# Dynamic Hybrid Movie Recommender System

## Overview
This project is an interactive **Dynamic Hybrid Movie Recommender System** built with **Streamlit**, **Surprise**, and **Scikit-learn**. It combines three distinct recommendation approaches into a single weighted pipeline:

* **Demographic Filtering:** Recommends popular, high-rated movies based on the IMDb weighted rating formula.

* **Content-Based Filtering:** Calculates multi-feature cosine similarity across film metadata (genres, keywords, collection, director, screenplay, and cast).

* **Collaborative Filtering (SVD):** Predicts personalized user ratings using Matrix Factorization via Singular Value Decomposition.


> For in-depth details for Exploratory Data Analysis (EDA), Model Selection and Tuning, and Results, please refer to the [Full Technical Report](movie_recommendation_report.md).

---

## Demo

![Streamlit App Demo](./assets/Demo.gif)

**Try the Live Interactive Web App:** [Movie Hybrid Recommendation System on Streamlit Cloud](https://dynamic-hybrid-movie-recommender-system-a9f86gwfzfhyf4jfbtnlzq.streamlit.app)

---

## Key Features & Highlights

* **Dynamic & Custom Weighting:** Dynamically balances hybrid model weights using a Sigmoid curve based on user activity, or allows manual slider control down to pure single-model filtering.

* **Cold-Start / Guest Handling:** Automatic guest mode (User ID = 0) that locks collaborative filtering to zero and relies on demographic and content signals.

* **Sub-Module Fine-Tuning:** Fine-grained weight controls for individual metadata features within the content-based engine.

* **Real-Time Interactive UI:** Single-page Streamlit interface optimized with @st.fragment state management for zero UI jittering and persistent result rendering.


---
## Key Results  

### 1. Demographic Filtering: Weighted Score vs. Raw Vote Average
To prevent obscure movies with a single 10/10 rating from dominating the top recommendations, a **Bayesian weighted rating formula** (the weighted rating formula publicized by IMDb) was implemented. The density comparison below illustrates how the weighted demographic score normalizes high-variance raw vote averages into a reliable Gaussian-like distribution centered around ~6.3 to 6.5.

<p align="center">
  <img src="plots/Demographics/03_distribution_vote_average_vs_weighted_score.png" alt="Distribution Comparison: Raw Vote Average vs. Weighted Demographic Score" width="90%">
</p>

### 2. Content-Based Sub-Module Similarity Breakdown
The content-based filtering engine breaks down movie features into separate TF-IDF and Count Vectorizer matrix similarities. The heatmap below displays feature-level similarity scores when generating recommendations based on *"Interstellar"*. Films like *"Inception"* score higher on director similarity (Christopher Nolan), while hard sci-fi films like *"The Martian"* and *"A.I. Artificial Intelligence"* match strongly on genres and keywords.

<p align="center">
  <img src="plots/Content_Based/03_feature_similarity_breakdown_Interstellar.png" alt="Feature Similarity Breakdown for Interstellar" width="90%">
</p>

### 3. SVD Latent Feature Space (t-SNE Visualization)
To evaluate the collaborative filtering model, the learned item latent factor embeddings (from the tuned SVD model) were reduced into 2D space using **t-SNE**. The visualization demonstrates how implicit user rating patterns cluster semantically related movies together in the lower-dimensional vector space.

<p align="center">
  <img src="plots/Collaborative/05_t-SNE_svd_embeddings.png" alt="t-SNE Visualization of SVD Movie Latent Embeddings" width="90%">
</p>

---

### 4. Collaborative Filtering: Top-K Recommendation Metrics
Evaluating the SVD model across varying $K$ cutoffs ($K \in \{3, 5, 10, 15, 20\}$) on the held-out test set highlights strong top-of-funnel accuracy:
* **Hit Rate@K** reaches **0.94** at $K=3$ and saturates at **1.00** by $K=10$.
* **Precision@K** starts high at **0.69** for $K=3$.
* **Recall@K** steadily increases to **0.87** at $K=20$.

<p align="center">
  <img src="plots/Collaborative/03_tuned_top_k_metrics_test.png" alt="Top-K Recommendation Metrics for Tuned SVD Model" width="90%">
</p>

---

## Tech Stack

* **Language:** Python `3.12.11`
* **Data Processing & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Scikit-Surprise
* **Visualization:** Matplotlib, Seaborn
* **Model Persistence:** Joblib
* **Web Framework:** Streamlit

---

## Project Structure

```text
dynamic-hybrid-movie-recommender-system/
├── assets/
│   └── Demo.gif                    # Demonstration GIF for README
├── Movie_data/                     # Data set
├── movie_recommendation.ipynb      # Complete Machine Learning pipeline
├── movie_recommendation_report.md  # Comprehensive technical report
├── app.py                          # Interactive Streamlit web application
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

The execution pipeline automatically generates and manages the following runtime directories:

```text
├── models/                     # Stores trained model files
│   ├── content_based/          # Matrices and vectorizer
│   └── collaborative/          # SVD model and evaluation summery
└── plots/                      # Generated visualizations
    ├── Rating_Distribution/    # Rating Distribution per user and year
    ├── Demographics/           # Analysis of the demogephic scroes
    ├── Content_Based/          # Analysis of the content-based model
    └── Collaborative/          # Analysis of the SVD model performance and model insight
```

---

## How to Run

First clone the repository:
```bash
git clone https://github.com/BevisWong76/dynamic-hybrid-movie-recommender-system.git
cd dynamic-hybrid-movie-recommender-system
```

You can then set up the project locally using either the standard Python `venv` or the ultra-fast `uv` package manager.

### Option 1: Using Standard Python `venv` (Traditional)

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# macOS / Linux:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Option 2: Using `uv` (Recommended for Speed)

`uv` is an extremely fast Python package installer and resolver written in Rust.

1. Install `uv` (if you haven't already):
```bash
pip install uv
```

2.  Create a virtual environment:

```bash
uv venv
```

3. Activate the virtual environment:
```bash
# Windows (Command Prompt):
.venv\Scripts\activate.bat

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# macOS / Linux:
source .venv/bin/activate
```

 4. Install dependencies:
```bash
uv pip install --upgrade pip
uv pip install -r requirements.txt
```

### Run the Streamlit App

Once the dependencies are installed and the model artifacts are generated, launch the interactive web application:

```bash
streamlit run app.py
```
---

## Acknowledgements

* **Dataset:** ["The Movie Dataset" from Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (provided by Rounak Banik).
* **Inspiration & Resources:** 
  * [Surprise (Simple Python RecommendatIon System Engine)](https://surpriselib.com/) for collaborative filtering algorithms and Matrix Factorization tools.
  * [Streamlit Documentation](https://docs.streamlit.io/) for high-performance interactive web application components and state management paradigms.

