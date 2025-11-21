import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG
# ============================================================

MOVIES_METADATA_PATH = "movies_metadata.csv"
RATINGS_PATH = "ratings_small.csv"
LINKS_SMALL_PATH = "links_small.csv"

N_FACTORS = 40
N_EPOCHS = 15
LR = 0.01
REG = 0.05

RANDOM_SEED = 42
RATING_THRESHOLD_RELEVANT = 4.0
K_EVAL = 10

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Try to use Sentence-BERT, fall back to TF-IDF on ANY error (DLL issues, etc.)
try:
    from sentence_transformers import SentenceTransformer
    USE_SBERT = True
except Exception:
    SentenceTransformer = None
    USE_SBERT = False


# ============================================================
# DATA LOADING & PREP
# ============================================================

def load_and_merge_data():
    movies_meta = pd.read_csv(MOVIES_METADATA_PATH, low_memory=False)
    ratings = pd.read_csv(RATINGS_PATH)
    links_small = pd.read_csv(LINKS_SMALL_PATH)

    # Clean/align IDs
    movies_meta = movies_meta[movies_meta['id'].apply(lambda x: str(x).isdigit())]
    movies_meta['id'] = movies_meta['id'].astype(int)

    links_small = links_small[links_small['tmdbId'].notna()]
    links_small['tmdbId'] = links_small['tmdbId'].astype(int)

    # Join MovieLens movieId with TMDb metadata
    movies = links_small.merge(
        movies_meta,
        left_on="tmdbId",
        right_on="id",
        how="inner",
        suffixes=("_link", "_meta")
    )

    movies = movies[['movieId', 'title', 'overview', 'genres']].drop_duplicates('movieId')

    # Keep ratings only for movies we have metadata for
    ratings = ratings[ratings['movieId'].isin(movies['movieId'])]

    movies['overview'] = movies['overview'].fillna('')
    movies['genres'] = movies['genres'].fillna('')

    movies['text'] = (
        movies['title'].fillna('') + ' ' +
        movies['genres'] + ' ' +
        movies['overview']
    )

    return movies.reset_index(drop=True), ratings.reset_index(drop=True)


def train_test_split_by_user(ratings, test_frac=0.2, min_ratings_for_split=2):
    train_rows, test_rows = [], []

    for user_id, user_ratings in ratings.groupby('userId'):
        if len(user_ratings) >= min_ratings_for_split:
            test_size = max(1, int(round(test_frac * len(user_ratings))))
            user_ratings = user_ratings.sample(frac=1.0, random_state=RANDOM_SEED)
            test_user = user_ratings.iloc[:test_size]
            train_user = user_ratings.iloc[test_size:]
            train_rows.append(train_user)
            test_rows.append(test_user)
        else:
            train_rows.append(user_ratings)

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


def build_index_mappings(movies, train_ratings):
    movie_ids_in_train = train_ratings['movieId'].unique()
    movies = movies[movies['movieId'].isin(movie_ids_in_train)].reset_index(drop=True)

    movie_id_to_idx = {m: i for i, m in enumerate(movies['movieId'].values)}
    idx_to_movie_id = {i: m for m, i in movie_id_to_idx.items()}

    user_ids = train_ratings['userId'].unique()
    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
    idx_to_user_id = {i: u for u, i in user_id_to_idx.items()}

    movies['movie_idx'] = movies['movieId'].map(movie_id_to_idx)

    train_ratings = train_ratings.copy()
    train_ratings['user_idx'] = train_ratings['userId'].map(user_id_to_idx)
    train_ratings['movie_idx'] = train_ratings['movieId'].map(movie_id_to_idx)

    return movies, train_ratings, user_id_to_idx, idx_to_user_id, movie_id_to_idx, idx_to_movie_id


def add_indices_to_test(test_ratings, user_id_to_idx, movie_id_to_idx):
    test_ratings = test_ratings[
        test_ratings['userId'].isin(user_id_to_idx.keys())
        & test_ratings['movieId'].isin(movie_id_to_idx.keys())
    ].copy()
    test_ratings['user_idx'] = test_ratings['userId'].map(user_id_to_idx)
    test_ratings['movie_idx'] = test_ratings['movieId'].map(movie_id_to_idx)
    return test_ratings.reset_index(drop=True)


# ============================================================
# NLP CONTENT EMBEDDINGS
# ============================================================

def build_content_embeddings(movies):
    texts = movies['text'].fillna('').tolist()

    if USE_SBERT and SentenceTransformer is not None:
        st.write("Using Sentence-BERT embeddings (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)
    else:
        st.write("Using TF-IDF embeddings (sentence-transformers unavailable).")
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )
        embeddings = vectorizer.fit_transform(texts)
        embeddings = embeddings.toarray().astype(np.float32)

    embeddings = normalize(embeddings, norm='l2')
    return embeddings


def build_user_content_profile(user_idx, train_ratings, movie_embeddings):
    user_r = train_ratings[train_ratings['user_idx'] == user_idx]
    if user_r.empty:
        return np.zeros(movie_embeddings.shape[1], dtype=np.float32)

    profile = np.zeros(movie_embeddings.shape[1], dtype=np.float32)
    total_weight = 0.0
    for _, row in user_r.iterrows():
        midx = int(row['movie_idx'])
        rating = float(row['rating'])
        profile += rating * movie_embeddings[midx]
        total_weight += rating

    if total_weight > 0:
        profile /= total_weight

    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm
    return profile


def content_scores_for_user_profile(user_profile, movie_embeddings):
    if np.all(user_profile == 0):
        return np.zeros(movie_embeddings.shape[0], dtype=np.float32)
    scores = movie_embeddings @ user_profile
    return scores


# ============================================================
# MATRIX FACTORIZATION (CF)
# ============================================================

class BiasedMF:
    def __init__(self, n_users, n_items, n_factors=40, lr=0.01, reg=0.05, n_epochs=15, random_state=42):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, train_ratings):
        rng = np.random.default_rng(self.random_state)

        self.global_mean = train_ratings['rating'].mean()
        self.bu = np.zeros(self.n_users, dtype=np.float32)
        self.bi = np.zeros(self.n_items, dtype=np.float32)
        self.P = 0.1 * rng.standard_normal((self.n_users, self.n_factors)).astype(np.float32)
        self.Q = 0.1 * rng.standard_normal((self.n_items, self.n_factors)).astype(np.float32)

        idxs = np.arange(len(train_ratings))

        for epoch in range(self.n_epochs):
            np.random.shuffle(idxs)
            se = 0.0
            for idx in idxs:
                row = train_ratings.iloc[idx]
                u = int(row['user_idx'])
                i = int(row['movie_idx'])
                r_ui = float(row['rating'])

                pred = self.predict_single_idx(u, i)
                err = r_ui - pred
                se += err * err

                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                pu = self.P[u, :].copy()
                qi = self.Q[i, :].copy()

                self.P[u, :] += self.lr * (err * qi - self.reg * pu)
                self.Q[i, :] += self.lr * (err * pu - self.reg * qi)

            rmse_epoch = math.sqrt(se / len(train_ratings))
            st.write(f"[MF] Epoch {epoch+1}/{self.n_epochs} - train RMSE: {rmse_epoch:.4f}")

    def predict_single_idx(self, u_idx, i_idx):
        pred = (
            self.global_mean +
            self.bu[u_idx] +
            self.bi[i_idx] +
            self.P[u_idx, :].dot(self.Q[i_idx, :])
        )
        return pred

    def predict_matrix(self):
        bu_col = self.bu.reshape(-1, 1)
        bi_row = self.bi.reshape(1, -1)
        baseline = self.global_mean + bu_col + bi_row
        interaction = self.P @ self.Q.T
        return baseline + interaction


# ============================================================
# HYBRID SCORING & RECOMMENDER
# ============================================================

def normalize_scores(x):
    x = np.array(x, dtype=np.float32)
    if np.all(x == 0):
        return x
    min_x = x.min()
    max_x = x.max()
    if max_x == min_x:
        return np.zeros_like(x)
    return (x - min_x) / (max_x - min_x)


def build_popularity_scores(train_ratings, n_movies):
    movie_stats = train_ratings.groupby('movie_idx')['rating'].agg(['count', 'mean'])
    pop = np.zeros(n_movies, dtype=np.float32)
    for midx, row in movie_stats.iterrows():
        pop[midx] = float(row['mean']) * math.log(1.0 + float(row['count']))
    return normalize_scores(pop)


def hybrid_scores_for_user(
    user_idx,
    mf_pred_matrix,
    train_ratings,
    movie_embeddings,
    alpha=0.6,
    beta=0.4,
    exclude_train=True,
    popularity_scores=None
):
    cf_scores = mf_pred_matrix[user_idx]
    user_profile = build_user_content_profile(user_idx, train_ratings, movie_embeddings)
    content_scores = content_scores_for_user_profile(user_profile, movie_embeddings)

    cf_norm = normalize_scores(cf_scores)
    content_norm = normalize_scores(content_scores)

    hybrid = alpha * cf_norm + beta * content_norm

    if exclude_train:
        watched = train_ratings[train_ratings['user_idx'] == user_idx]['movie_idx'].values
        hybrid[watched] = -np.inf

    if np.all(np.isneginf(hybrid)) and popularity_scores is not None:
        hybrid = popularity_scores.copy()

    return hybrid


def hybrid_scores_for_cold_start(
    liked_movie_idxs,
    movie_embeddings,
    popularity_scores=None,
    beta=1.0
):
    if liked_movie_idxs is None or len(liked_movie_idxs) == 0:
        if popularity_scores is not None:
            return popularity_scores.copy()
        else:
            return np.zeros(movie_embeddings.shape[0], dtype=np.float32)

    profile = np.zeros(movie_embeddings.shape[1], dtype=np.float32)
    for midx in liked_movie_idxs:
        profile += movie_embeddings[midx]
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm

    content_scores = content_scores_for_user_profile(profile, movie_embeddings)
    content_norm = normalize_scores(content_scores)
    hybrid = beta * content_norm
    return hybrid


def recommend_movies(
    user_id,
    user_id_to_idx,
    idx_to_movie_id,
    movies_df,
    mf_pred_matrix,
    train_ratings,
    movie_embeddings,
    n_recs=10,
    alpha=0.6,
    beta=0.4,
    popularity_scores=None,
    cold_start_liked_movie_ids=None,
    movie_id_to_idx=None
):
    n_movies = len(idx_to_movie_id)

    if user_id in user_id_to_idx:
        user_idx = user_id_to_idx[user_id]
        scores = hybrid_scores_for_user(
            user_idx,
            mf_pred_matrix,
            train_ratings,
            movie_embeddings,
            alpha=alpha,
            beta=beta,
            exclude_train=True,
            popularity_scores=popularity_scores
        )
    else:
        if cold_start_liked_movie_ids is not None and movie_id_to_idx is not None:
            liked_idxs = [
                movie_id_to_idx[mid] for mid in cold_start_liked_movie_ids
                if mid in movie_id_to_idx
            ]
        else:
            liked_idxs = []
        scores = hybrid_scores_for_cold_start(
            liked_idxs,
            movie_embeddings,
            popularity_scores=popularity_scores,
            beta=beta
        )

    top_indices = np.argsort(scores)[::-1][:n_recs]
    recs = movies_df.iloc[top_indices][['movieId', 'title']]
    recs = recs.copy()
    recs['score'] = scores[top_indices]
    return recs.reset_index(drop=True)


# ============================================================
# EVALUATION
# ============================================================

def rmse(model, test_ratings):
    se = 0.0
    n = 0
    for _, row in test_ratings.iterrows():
        u = int(row['user_idx'])
        i = int(row['movie_idx'])
        r = float(row['rating'])
        r_hat = model.predict_single_idx(u, i)
        se += (r - r_hat) ** 2
        n += 1
    return math.sqrt(se / n) if n > 0 else float('nan')


def ranking_metrics(
    test_ratings,
    train_ratings,
    mf_pred_matrix,
    movie_embeddings,
    k=10,
    rating_threshold=4.0,
    alpha=0.6,
    beta=0.4,
    popularity_scores=None
):
    test_by_user = defaultdict(list)
    for _, row in test_ratings.iterrows():
        if row['rating'] >= rating_threshold:
            test_by_user[int(row['user_idx'])].append(int(row['movie_idx']))

    precs, recs, ndcgs = [], [], []

    for user_idx, relevant_items in test_by_user.items():
        if len(relevant_items) == 0:
            continue

        scores = hybrid_scores_for_user(
            user_idx,
            mf_pred_matrix,
            train_ratings,
            movie_embeddings,
            alpha=alpha,
            beta=beta,
            exclude_train=True,
            popularity_scores=popularity_scores
        )

        top_indices = np.argsort(scores)[::-1][:k]
        recommended = list(top_indices)
        rel_set = set(relevant_items)

        hits = [1 if m in rel_set else 0 for m in recommended]
        n_hits = sum(hits)

        prec = n_hits / k
        rec = n_hits / len(rel_set)

        dcg = 0.0
        for rank, m in enumerate(recommended, start=1):
            if m in rel_set:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(rel_set), k)
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        precs.append(prec)
        recs.append(rec)
        ndcgs.append(ndcg)

    mean_prec = float(np.mean(precs)) if precs else 0.0
    mean_rec = float(np.mean(recs)) if recs else 0.0
    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

    return mean_prec, mean_rec, mean_ndcg


# ============================================================
# CACHED PIPELINE
# ============================================================

@st.cache_resource(show_spinner=True)
def build_recommender_pipeline():
    st.write("📥 Loading and preparing data...")
    movies, ratings = load_and_merge_data()

    st.write("✂️ Splitting train/test...")
    train_ratings, test_ratings = train_test_split_by_user(ratings, test_frac=0.2)

    st.write("🔢 Building index mappings...")
    movies, train_ratings, user_id_to_idx, idx_to_user_id, movie_id_to_idx, idx_to_movie_id = \
        build_index_mappings(movies, train_ratings)
    test_ratings = add_indices_to_test(test_ratings, user_id_to_idx, movie_id_to_idx)

    n_users = len(user_id_to_idx)
    n_movies = len(movie_id_to_idx)

    st.write("🧠 Building content embeddings...")
    movie_embeddings = build_content_embeddings(movies)

    st.write("🔥 Training matrix factorization model...")
    mf_model = BiasedMF(
        n_users=n_users,
        n_items=n_movies,
        n_factors=N_FACTORS,
        lr=LR,
        reg=REG,
        n_epochs=N_EPOCHS,
        random_state=RANDOM_SEED
    )
    mf_model.fit(train_ratings)

    st.write("📈 Precomputing prediction matrix...")
    mf_pred_matrix = mf_model.predict_matrix()

    st.write("⭐ Computing popularity scores...")
    popularity_scores = build_popularity_scores(train_ratings, n_movies)

    st.write("📏 Evaluating model...")
    test_rmse = rmse(mf_model, test_ratings)
    prec_k, rec_k, ndcg_k = ranking_metrics(
        test_ratings,
        train_ratings,
        mf_pred_matrix,
        movie_embeddings,
        k=K_EVAL,
        rating_threshold=RATING_THRESHOLD_RELEVANT,
        alpha=0.6,
        beta=0.4,
        popularity_scores=popularity_scores
    )

    metrics = {
        "rmse": test_rmse,
        "precision_k": prec_k,
        "recall_k": rec_k,
        "ndcg_k": ndcg_k
    }

    artifacts = {
        "movies": movies,
        "ratings": ratings,
        "train_ratings": train_ratings,
        "test_ratings": test_ratings,
        "user_id_to_idx": user_id_to_idx,
        "idx_to_user_id": idx_to_user_id,
        "movie_id_to_idx": movie_id_to_idx,
        "idx_to_movie_id": idx_to_movie_id,
        "movie_embeddings": movie_embeddings,
        "mf_model": mf_model,
        "mf_pred_matrix": mf_pred_matrix,
        "popularity_scores": popularity_scores,
        "metrics": metrics
    }

    return artifacts


# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    st.title("🎬 Personalized Movie Recommendation System")
    st.write("Hybrid of **Collaborative Filtering** + **Content-Based (NLP)** recommendations.")

    artifacts = build_recommender_pipeline()

    movies = artifacts["movies"]
    train_ratings = artifacts["train_ratings"]
    user_id_to_idx = artifacts["user_id_to_idx"]
    idx_to_movie_id = artifacts["idx_to_movie_id"]
    movie_id_to_idx = artifacts["movie_id_to_idx"]
    movie_embeddings = artifacts["movie_embeddings"]
    mf_pred_matrix = artifacts["mf_pred_matrix"]
    popularity_scores = artifacts["popularity_scores"]
    metrics = artifacts["metrics"]

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")

    mode = st.sidebar.radio("Mode", ["Known user", "Cold-start user"])
    alpha = st.sidebar.slider("Weight for CF (alpha)", 0.0, 1.0, 0.6, 0.05)
    beta = 1.0 - alpha
    st.sidebar.write(f"Weight for Content (beta): **{beta:.2f}**")
    n_recs = st.sidebar.slider("Number of recommendations", 5, 20, 10)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Metrics (on test set)")
    st.sidebar.write(f"RMSE: **{metrics['rmse']:.4f}**")
    st.sidebar.write(f"Precision@{K_EVAL}: **{metrics['precision_k']:.4f}**")
    st.sidebar.write(f"Recall@{K_EVAL}: **{metrics['recall_k']:.4f}**")
    st.sidebar.write(f"NDCG@{K_EVAL}: **{metrics['ndcg_k']:.4f}**")

    col1, col2 = st.columns([2, 3])

    if mode == "Known user":
        with col1:
            st.subheader("👤 Choose an existing user")
            all_users = sorted(user_id_to_idx.keys())
            selected_user = st.selectbox("User ID", all_users)

            st.markdown("#### This user's top-rated movies (train set)")
            uidx = user_id_to_idx[selected_user]
            user_r = train_ratings[train_ratings['user_idx'] == uidx].copy()
            user_r = user_r.sort_values('rating', ascending=False).head(10)
            user_r = user_r.merge(movies[['movieId', 'title']], on='movieId', how='left')
            st.dataframe(user_r[['movieId', 'title', 'rating']])

        with col2:
            st.subheader("🎯 Recommendations")
            if st.button("Generate recommendations"):
                recs = recommend_movies(
                    user_id=selected_user,
                    user_id_to_idx=user_id_to_idx,
                    idx_to_movie_id=idx_to_movie_id,
                    movies_df=movies,
                    mf_pred_matrix=mf_pred_matrix,
                    train_ratings=train_ratings,
                    movie_embeddings=movie_embeddings,
                    n_recs=n_recs,
                    alpha=alpha,
                    beta=beta,
                    popularity_scores=popularity_scores,
                    cold_start_liked_movie_ids=None,
                    movie_id_to_idx=movie_id_to_idx
                )
                st.dataframe(recs)

    else:  # Cold-start user
        with col1:
            st.subheader("🆕 Cold-start: new user preferences")
            st.write("Select some movies this new user likes:")

            all_titles = movies['title'].tolist()
            liked_titles = st.multiselect(
                "Liked movies",
                options=all_titles,
                default=all_titles[:3]  # arbitrary default
            )

            liked_ids = movies[movies['title'].isin(liked_titles)]['movieId'].tolist()

        with col2:
            st.subheader("🎯 Recommendations for new user")
            if st.button("Generate cold-start recommendations"):
                recs = recommend_movies(
                    user_id=-1,  # unknown
                    user_id_to_idx=user_id_to_idx,
                    idx_to_movie_id=idx_to_movie_id,
                    movies_df=movies,
                    mf_pred_matrix=mf_pred_matrix,
                    train_ratings=train_ratings,
                    movie_embeddings=movie_embeddings,
                    n_recs=n_recs,
                    alpha=0.0,          # ignore CF for unknown user
                    beta=1.0,           # content + popularity
                    popularity_scores=popularity_scores,
                    cold_start_liked_movie_ids=liked_ids,
                    movie_id_to_idx=movie_id_to_idx
                )
                st.dataframe(recs)


if __name__ == "__main__":
    main()
