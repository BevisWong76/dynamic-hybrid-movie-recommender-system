import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Hybrid Recommender System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Hybrid Recommendation System")
st.markdown("An interactive dynamic hybrid recommender using **Demographic**, **Content-Based**, and **SVD Collaborative Filtering**.")
st.divider()

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------
def normalize_weights(weights_dict_or_list):
    """ Make sure the weights sum to 1.0 """
    if isinstance(weights_dict_or_list, dict):
        total = sum(weights_dict_or_list.values())
        if total == 0:
            n = len(weights_dict_or_list)
            return {k: 1.0 / n for k in weights_dict_or_list}
        return {k: v / total for k, v in weights_dict_or_list.items()}
    else:
        total = sum(weights_dict_or_list)
        if total == 0:
            n = len(weights_dict_or_list)
            return [1.0 / n] * n
        return [v / total for v in weights_dict_or_list]

def get_dynamic_weights_sigmoid(n_ratings, midpoint=25, k=0.1):
    """Calculate Sigmoid Dynamic Weights"""
    if n_ratings == 0:
        return [0.35, 0.65, 0.0]
    
    svd_w = 0.85 / (1 + np.exp(-k * (n_ratings - midpoint)))
    remaining = 1.0 - svd_w
    content_w = remaining * 0.65
    demo_w = remaining * 0.35
    return [demo_w, content_w, svd_w]

# -----------------------------------------------------------------------------
# 3. Load Models and Datasets
# -----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    models_dir  = Path("models")
    content_dir = Path("models/content_based")
    collab_dir  = Path("models/collaborative")
    
    movie_score = pd.read_csv(models_dir / "movie_score.csv")
    user_counts = pd.read_csv(models_dir / "user_rating_counts.csv")
    svd_model = joblib.load(collab_dir / "tuned_SVD_model.pkl")
    
    matrices = {
        'soup'       : joblib.load(content_dir / "matrices/soup_matrix.pkl"),
        'keywords'   : joblib.load(content_dir / "matrices/keywords_matrix.pkl"),
        'genres'     : joblib.load(content_dir / "matrices/genres_matrix.pkl"),
        'collection' : joblib.load(content_dir / "matrices/collection_matrix.pkl"),
        'director'   : joblib.load(content_dir / "matrices/director_matrix.pkl"),
        'screenplay' : joblib.load(content_dir / "matrices/screenplay_matrix.pkl"),
        'cast'       : joblib.load(content_dir / "matrices/cast_matrix.pkl")
    }
    
    return movie_score, user_counts, svd_model, matrices

movie_score, user_counts, svd_model, matrices = load_assets()

def compute_content_similarity(selected_titles, cb_weights_dict):
    indices = movie_score[movie_score['title'].isin(selected_titles)].index
    if len(indices) == 0:
        return pd.Series(0.0, index=movie_score.index)
    
    norm_cb_w = normalize_weights(cb_weights_dict)
    total_sim = np.zeros(len(movie_score))
    
    for feature_name, weight in norm_cb_w.items():
        if weight > 0 and feature_name in matrices:
            matrix = matrices[feature_name]
            sim_matrix = cosine_similarity(matrix[indices], matrix)
            mean_sim = sim_matrix.mean(axis=0)
            total_sim += weight * mean_sim
            
    return pd.Series(total_sim, index=movie_score.index)

# -----------------------------------------------------------------------------
# 4. Global State and User Identity Selection
# -----------------------------------------------------------------------------
if 'recs_result' not in st.session_state:
    st.session_state['recs_result'] = None

st.header("1. User Profile Setup")
user_list = [0] + sorted(user_counts['userId'].unique().tolist())
selected_user_id = st.selectbox(
    "Select User ID (0 = Guest / Non-registered User):",
    options=user_list,
    index=0
)

is_guest = (selected_user_id == 0)

if is_guest:
    n_ratings = 0
    st.info("👤 **Guest Mode Active**: SVD Collaborative Filtering is **locked to 0**.")
else:
    user_row = user_counts[user_counts['userId'] == selected_user_id]
    n_ratings = int(user_row['num_ratings'].values[0]) if not user_row.empty else 0
    st.success(f"👤 **User {selected_user_id} Loaded**: Total ratings in database = **{n_ratings}**.")

st.divider()

# -----------------------------------------------------------------------------
# 5. Isolated Component using @st.fragment
# -----------------------------------------------------------------------------
@st.fragment
def render_recommender_section(user_id, is_guest_user, user_ratings_count):
    st.header("2. Hybrid Recommender Weights Configuration")
    
    weight_mode = st.radio(
        "Choose Weight Allocation Mode:",
        ["Dynamic Weights (Sigmoid Curve)", "Custom Manual Weights"],
        horizontal=True,
        key="isolated_weight_mode"
    )

    with st.container(height=170, border=False):
        if weight_mode == "Dynamic Weights (Sigmoid Curve)":
            calc_weights = get_dynamic_weights_sigmoid(user_ratings_count)
            if is_guest_user:
                calc_weights[2] = 0.0
                calc_weights = normalize_weights(calc_weights)
                
            demo_w, content_w, svd_w = calc_weights
            st.info(f"💡 **Sigmoid Dynamic Weights**: Demographic = **{demo_w:.1%}** | Content-Based = **{content_w:.1%}** | SVD = **{svd_w:.1%}**")
        else:
            st.caption("⚙️ **Adjust Custom Weights** (Will automatically normalize to 100%):")
            col_w1, col_w2, col_w3 = st.columns(3)
            with col_w1:
                raw_demo = st.slider("Demographic Weight", 0.0, 1.0, 0.2, step=0.05, key="slider_demo")
            with col_w2:
                raw_content = st.slider("Content-Based Weight", 0.0, 1.0, 0.4, step=0.05, key="slider_content")
            with col_w3:
                raw_svd = 0.0 if is_guest_user else st.slider("SVD Weight", 0.0, 1.0, 0.4, step=0.05, key="slider_svd")
                    
            norm_w = normalize_weights([raw_demo, raw_content, raw_svd])
            demo_w, content_w, svd_w = norm_w[0], norm_w[1], norm_w[2]
            st.caption(f"Normalized Final Weights -> Demo: **{demo_w:.1%}** | Content: **{content_w:.1%}** | SVD: **{svd_w:.1%}**")

    st.divider()

    # ---  3: Content-Based sub-module weights ---
    st.header("3. Content-Based Sub-Module Weights")
    DEFAULT_CB_WEIGHTS = {
        'keywords': 0.25, 'genres': 0.25, 'collection': 0.10,
        'director': 0.15, 'screenplay': 0.10, 'cast': 0.15
    }

    cb_cols = st.columns(6)
    raw_cb_weights = {}
    for idx, (feature, default_val) in enumerate(DEFAULT_CB_WEIGHTS.items()):
        with cb_cols[idx]:
            raw_cb_weights[feature] = st.slider(
                f"{feature.capitalize()}", 0.0, 1.0, default_val, step=0.05, key=f"cb_slider_{feature}"
            )

    st.divider()

    # ---  4: Movie Selection and Recommendation Generation ---
    st.header("4. Select Favorite Movie & Generate Recommendations")
    col_input1, col_input2 = st.columns([3, 1])

    with col_input1:
        all_titles = sorted(movie_score['title'].dropna().unique().tolist())
        selected_movies = st.multiselect(
            "Choose one or more movies you like as input:",
            options=all_titles,
            default=["Spirited Away"] if "Spirited Away" in all_titles else [all_titles[0]],
            key="selected_movies_select"
        )

    with col_input2:
        top_n = st.number_input("Number of Recs:", min_value=1, max_value=50, value=10, key="top_n_input")

    # Button to trigger recommendation generation
    if st.button("🚀 Generate Recommendations", type="primary", width='stretch', key="gen_btn"):
        if not selected_movies:
            st.warning("Please select at least one movie first!")
        else:
            with st.spinner("Computing real similarity vectors and hybrid scores..."):
                df_hybrid = movie_score.copy()
                
                # 1. Real Content-Based Score
                df_hybrid['sim_score'] = compute_content_similarity(selected_movies, raw_cb_weights)
                
                # 2. SVD Score
                all_movie_ids = df_hybrid['id'].values
                df_hybrid['svd_score'] = [
                    svd_model.predict(user_id, iid).est for iid in all_movie_ids
                ]
                
                # 3. MinMaxScaler normalization
                scaler = MinMaxScaler()
                df_hybrid['demo_norm'] = scaler.fit_transform(df_hybrid[['score']])
                df_hybrid['content_norm'] = scaler.fit_transform(df_hybrid[['sim_score']])
                df_hybrid['svd_norm'] = scaler.fit_transform(df_hybrid[['svd_score']])
                
                # 4. Final Weighted Score
                df_hybrid['final_score'] = (
                    (demo_w * df_hybrid['demo_norm']) +
                    (content_w * df_hybrid['content_norm']) +
                    (svd_w * df_hybrid['svd_norm'])
                )
                
                # exclude already selected movies from recommendations
                df_filtered = df_hybrid[~df_hybrid['title'].isin(selected_movies)]
                results = df_filtered.sort_values(by='final_score', ascending=False).head(top_n)
                
                # save results to session_state
                st.session_state['recs_result'] = {
                    'user_id': user_id,
                    'top_n': top_n,
                    'data': results[['title', 'score', 'sim_score', 'svd_score', 'final_score']]
                }
                
                st.rerun()

render_recommender_section(selected_user_id, is_guest, n_ratings)

st.divider()

# -----------------------------------------------------------------------------
# 6. Display Recommendations
# -----------------------------------------------------------------------------
if st.session_state['recs_result'] is not None:
    res_info = st.session_state['recs_result']
    st.subheader(f"Top {res_info['top_n']} Recommendations for User {res_info['user_id']}")
    
    display_df = res_info['data'].copy()
    display_df.columns = ['Movie Title', 'Demo Rating', 'Content Sim', 'SVD Pred', 'Final Hybrid Score']
    
    st.dataframe(
        display_df.style.format({
            'Demo Rating': '{:.2f}',
            'Content Sim': '{:.4f}',
            'SVD Pred': '{:.2f}',
            'Final Hybrid Score': '{:.4f}'
        }),
        width='stretch'
    )