import os
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Try to use Sentence-BERT for better NLP; fall back to TF-IDF if unavailable
try:
    from sentence_transformers import SentenceTransformer
    USE_SBERT = True
except Exception:
    from sklearn.feature_extraction.text import TfidfVectorizer
    USE_SBERT = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ============================================================
# 1. CONFIG
# ============================================================

MOVIES_METADATA_PATH = "movies_metadata.csv"
RATINGS_PATH = "ratings_small.csv"
LINKS_SMALL_PATH = "links_small.csv"

# Collaborative filtering hyperparameters
N_FACTORS = 50
N_EPOCHS = 20
LR = 0.01
REG = 0.05

# Evaluation
RANDOM_SEED = 42
RATING_THRESHOLD_RELEVANT = 4.0  # for ranking metrics
K_EVAL = 10                      # for Precision@K, Recall@K, NDCG@K

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 2. LOAD AND MERGE DATA
# ============================================================

def load_and_merge_data():
    # Load raw files
    movies_meta = pd.read_csv(MOVIES_METADATA_PATH, low_memory=False)
    ratings = pd.read_csv(RATINGS_PATH)
    links_small = pd.read_csv(LINKS_SMALL_PATH)

    # Clean IDs in movies_metadata: keep only valid numeric IDs
    # movies_metadata.id is TMDb id as string; we need it as numeric
    movies_meta = movies_meta.copy()
    movies_meta = movies_meta[movies_meta['id'].apply(lambda x: str(x).isdigit())]
    movies_meta['id'] = movies_meta['id'].astype(int)

    # links_small.tmdbId is TMDb id; may have NaNs or non-numeric
    links_small = links_small.copy()
    links_small = links_small[links_small['tmdbId'].notna()]
    links_small['tmdbId'] = links_small['tmdbId'].astype(int)

    # Merge links_small with movies_meta to attach metadata to MovieLens movieId
    movies = links_small.merge(
        movies_meta,
        left_on="tmdbId",
        right_on="id",
        how="inner",
        suffixes=("_link", "_meta")
    )

    # Keep only needed columns
    movies = movies[['movieId', 'title', 'overview', 'genres']].drop_duplicates('movieId')

    # Filter ratings to only movies we have metadata for
    ratings = ratings[ratings['movieId'].isin(movies['movieId'])]

    # Basic cleaning of text columns
    movies['overview'] = movies['overview'].fillna('')
    movies['genres'] = movies['genres'].fillna('')

    # Build a simple text field (can be enriched with more columns if desired)
    movies['text'] = (
        movies['title'].fillna('') + ' ' +
        movies['genres'] + ' ' +
        movies['overview']
    )

    return movies.reset_index(drop=True), ratings.reset_index(drop=True)


# ============================================================
# 3. TRAIN/TEST SPLIT BY USER
# ============================================================

def train_test_split_by_user(ratings, test_frac=0.2, min_ratings_for_split=2):
    """
    For each user:
      - if they have >= min_ratings_for_split ratings: sample test_frac as test, rest as train
      - else: all in train
    Guarantees that any user with test data also has at least one train rating.
    """
    train_rows = []
    test_rows = []

    for user_id, user_ratings in ratings.groupby('userId'):
        if len(user_ratings) >= min_ratings_for_split:
            test_size = max(1, int(round(test_frac * len(user_ratings))))
            user_ratings = user_ratings.sample(frac=1.0, random_state=RANDOM_SEED)  # shuffle
            test_user = user_ratings.iloc[:test_size]
            train_user = user_ratings.iloc[test_size:]
            train_rows.append(train_user)
            test_rows.append(test_user)
        else:
            train_rows.append(user_ratings)

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df


# ============================================================
# 4. INDEX MAPPING & SPARSE MATRIX
# ============================================================

def build_index_mappings(movies, train_ratings):
    """
    Create integer indices for users and movies.
    Only movies that exist in 'movies' and in train_ratings are kept.
    """
    # Intersect movies present in train_ratings
    movie_ids_in_train = train_ratings['movieId'].unique()
    movies = movies[movies['movieId'].isin(movie_ids_in_train)].reset_index(drop=True)

    movie_id_to_idx = {m: i for i, m in enumerate(movies['movieId'].values)}
    idx_to_movie_id = {i: m for m, i in movie_id_to_idx.items()}

    user_ids = train_ratings['userId'].unique()
    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
    idx_to_user_id = {i: u for u, i in user_id_to_idx.items()}

    movies['movie_idx'] = movies['movieId'].map(movie_id_to_idx)

    # Map indices into train and test ratings
    train_ratings = train_ratings.copy()
    train_ratings['user_idx'] = train_ratings['userId'].map(user_id_to_idx)
    train_ratings['movie_idx'] = train_ratings['movieId'].map(movie_id_to_idx)

    return movies, train_ratings, user_id_to_idx, idx_to_user_id, movie_id_to_idx, idx_to_movie_id


def add_indices_to_test(test_ratings, user_id_to_idx, movie_id_to_idx):
    test_ratings = test_ratings.copy()
    test_ratings = test_ratings[
        test_ratings['userId'].isin(user_id_to_idx.keys())
        & test_ratings['movieId'].isin(movie_id_to_idx.keys())
    ]
    test_ratings['user_idx'] = test_ratings['userId'].map(user_id_to_idx)
    test_ratings['movie_idx'] = test_ratings['movieId'].map(movie_id_to_idx)
    return test_ratings.reset_index(drop=True)


def build_sparse_matrix(train_ratings, n_users, n_movies):
    R = csr_matrix(
        (train_ratings['rating'], (train_ratings['user_idx'], train_ratings['movie_idx'])),
        shape=(n_users, n_movies)
    )
    return R


# ============================================================
# 5. BETTER NLP: SBERT (or TF-IDF fallback)
# ============================================================

def build_content_embeddings(movies):
    texts = movies['text'].fillna('').tolist()

    if USE_SBERT:
        print("Using Sentence-BERT embeddings (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)
    else:
        print("sentence-transformers not installed, falling back to TF-IDF...")
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )
        embeddings = vectorizer.fit_transform(texts)
        # For simplicity, convert to dense (OK for ~9k movies)
        embeddings = embeddings.toarray().astype(np.float32)

    # Normalize for cosine similarity via dot-product
    embeddings = normalize(embeddings, norm='l2')
    return embeddings  # shape: (n_movies, dim)


def build_user_content_profile(user_idx, train_ratings, movie_embeddings):
    user_r = train_ratings[train_ratings['user_idx'] == user_idx]
    if user_r.empty:
        return np.zeros(movie_embeddings.shape[1], dtype=np.float32)

    # Weight by rating (could also center ratings by user mean)
    profile = np.zeros(movie_embeddings.shape[1], dtype=np.float32)
    total_weight = 0.0
    for _, row in user_r.iterrows():
        midx = int(row['movie_idx'])
        rating = float(row['rating'])
        profile += rating * movie_embeddings[midx]
        total_weight += rating

    if total_weight > 0:
        profile /= total_weight

    # Normalize profile
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm
    return profile


def content_scores_for_user_profile(user_profile, movie_embeddings):
    """
    Cosine similarity since everything is L2-normalized.
    """
    if np.all(user_profile == 0):
        return np.zeros(movie_embeddings.shape[0], dtype=np.float32)
    scores = movie_embeddings @ user_profile  # (n_movies,)
    return scores


# ============================================================
# 6. BETTER CF: BIASED MATRIX FACTORIZATION
# ============================================================

class BiasedMF:
    """
    Biased Matrix Factorization with SGD:
      r_hat_ui = mu + b_u + b_i + p_u^T q_i
    """
    def __init__(self, n_users, n_items, n_factors=50, lr=0.01, reg=0.05, n_epochs=10, random_state=42):
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

        # Shuffle training data each epoch
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

                # SGD updates
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                pu = self.P[u, :].copy()
                qi = self.Q[i, :].copy()

                self.P[u, :] += self.lr * (err * qi - self.reg * pu)
                self.Q[i, :] += self.lr * (err * pu - self.reg * qi)

            rmse_epoch = math.sqrt(se / len(train_ratings))
            print(f"[MF] Epoch {epoch+1}/{self.n_epochs} - training RMSE (on train): {rmse_epoch:.4f}")

    def predict_single_idx(self, u_idx, i_idx):
        pred = (self.global_mean +
                self.bu[u_idx] +
                self.bi[i_idx] +
                self.P[u_idx, :].dot(self.Q[i_idx, :]))
        return pred

    def predict_matrix(self):
        """
        Return full user-item prediction matrix.
        """
        bu_col = self.bu.reshape(-1, 1)
        bi_row = self.bi.reshape(1, -1)
        baseline = self.global_mean + bu_col + bi_row  # (n_users x n_items)
        interaction = self.P @ self.Q.T
        return baseline + interaction


# ============================================================
# 7. HYBRID RECOMMENDER
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
    # popularity = avg_rating * log(1 + count)
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
    """
    Compute hybrid (CF + content) scores for a known user.
    """
    cf_scores = mf_pred_matrix[user_idx]  # (n_movies,)

    # Content-based scores from user's content profile (based on TRAIN history)
    user_profile = build_user_content_profile(user_idx, train_ratings, movie_embeddings)
    content_scores = content_scores_for_user_profile(user_profile, movie_embeddings)

    cf_norm = normalize_scores(cf_scores)
    content_norm = normalize_scores(content_scores)

    hybrid = alpha * cf_norm + beta * content_norm

    # Exclude already-watched movies in train set
    if exclude_train:
        watched = train_ratings[train_ratings['user_idx'] == user_idx]['movie_idx'].values
        hybrid[watched] = -np.inf

    # If all -inf (user has no train ratings somehow), fall back to popularity
    if np.all(np.isneginf(hybrid)) and popularity_scores is not None:
        hybrid = popularity_scores.copy()

    return hybrid


def hybrid_scores_for_cold_start(
    liked_movie_idxs,
    movie_embeddings,
    popularity_scores=None,
    beta=1.0
):
    """
    Cold-start user: we only have a set of liked movies (movie_idxs).
    We build a content profile from those and score purely by content.
    """
    if liked_movie_idxs is None or len(liked_movie_idxs) == 0:
        # No info at all: fall back to popularity
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
    # pure content; beta is just scaling
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
    """
    If user_id is known -> hybrid CF + content.
    If user_id is unknown:
        - if cold_start_liked_movie_ids provided -> content-only using those.
        - else -> popularity-based.
    """
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

    # Top-N indices
    top_indices = np.argsort(scores)[::-1][:n_recs]
    recs = movies_df.iloc[top_indices][['movieId', 'title']]
    recs = recs.copy()
    recs['score'] = scores[top_indices]
    return recs.reset_index(drop=True)


# ============================================================
# 8. EVALUATION: RMSE, PREC@K, RECALL@K, NDCG@K
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
    """
    Compute mean Precision@K, Recall@K, NDCG@K over users with at least 1 relevant item in test.
    """
    # Build mapping of user -> test positive items
    test_by_user = defaultdict(list)
    for _, row in test_ratings.iterrows():
        if row['rating'] >= rating_threshold:
            test_by_user[int(row['user_idx'])].append(int(row['movie_idx']))

    precs = []
    recs = []
    ndcgs = []

    for user_idx, relevant_items in test_by_user.items():
        if len(relevant_items) == 0:
            continue

        # Compute hybrid scores for that user
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

        # Get top-k recommendations (indices of movies)
        top_indices = np.argsort(scores)[::-1][:k]

        recommended = list(top_indices)
        rel_set = set(relevant_items)

        # Precision & Recall
        hits = [1 if m in rel_set else 0 for m in recommended]
        n_hits = sum(hits)

        prec = n_hits / k
        rec = n_hits / len(rel_set)

        # NDCG
        dcg = 0.0
        for rank, m in enumerate(recommended, start=1):
            if m in rel_set:
                dcg += 1.0 / math.log2(rank + 1)
        # ideal DCG: assume all relevant items are ranked at top
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
# 9. MAIN PIPELINE
# ============================================================

def main():
    print("Loading and merging data...")
    movies, ratings = load_and_merge_data()
    print(f"Movies: {len(movies)}, Ratings: {len(ratings)}")

    print("Splitting train/test by user...")
    train_ratings, test_ratings = train_test_split_by_user(ratings, test_frac=0.2)
    print(f"Train ratings: {len(train_ratings)}, Test ratings: {len(test_ratings)}")

    print("Building index mappings...")
    movies, train_ratings, user_id_to_idx, idx_to_user_id, movie_id_to_idx, idx_to_movie_id = \
        build_index_mappings(movies, train_ratings)

    test_ratings = add_indices_to_test(test_ratings, user_id_to_idx, movie_id_to_idx)

    n_users = len(user_id_to_idx)
    n_movies = len(movie_id_to_idx)
    print(f"n_users = {n_users}, n_movies = {n_movies}")

    print("Building content embeddings...")
    movie_embeddings = build_content_embeddings(movies)

    print("Building popularity scores...")
    popularity_scores = build_popularity_scores(train_ratings, n_movies)

    print("Training biased MF model...")
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

    print("Precomputing full prediction matrix...")
    mf_pred_matrix = mf_model.predict_matrix()

    print("Evaluating RMSE on test set...")
    test_ratings_idx = test_ratings.copy()
    test_ratings_idx = test_ratings_idx[
        test_ratings_idx['userId'].isin(user_id_to_idx.keys())
        & test_ratings_idx['movieId'].isin(movie_id_to_idx.keys())
    ]
    test_rmse = rmse(mf_model, test_ratings_idx)
    print(f"Test RMSE: {test_rmse:.4f}")

    print(f"Evaluating ranking metrics @K={K_EVAL}...")
    prec_k, rec_k, ndcg_k = ranking_metrics(
        test_ratings_idx,
        train_ratings,
        mf_pred_matrix,
        movie_embeddings,
        k=K_EVAL,
        rating_threshold=RATING_THRESHOLD_RELEVANT,
        alpha=0.6,
        beta=0.4,
        popularity_scores=popularity_scores
    )
    print(f"Precision@{K_EVAL}: {prec_k:.4f}")
    print(f"Recall@{K_EVAL}:    {rec_k:.4f}")
    print(f"NDCG@{K_EVAL}:      {ndcg_k:.4f}")

    # --------------------------------------------------------
    # Example: Personalized recommendations for an existing user
    # --------------------------------------------------------
    example_user_id = list(user_id_to_idx.keys())[0]
    print("\nExample recommendations for existing user:", example_user_id)
    recs = recommend_movies(
        user_id=example_user_id,
        user_id_to_idx=user_id_to_idx,
        idx_to_movie_id=idx_to_movie_id,
        movies_df=movies,
        mf_pred_matrix=mf_pred_matrix,
        train_ratings=train_ratings,
        movie_embeddings=movie_embeddings,
        n_recs=10,
        alpha=0.6,
        beta=0.4,
        popularity_scores=popularity_scores,
        cold_start_liked_movie_ids=None,
        movie_id_to_idx=movie_id_to_idx
    )
    print(recs)

    # --------------------------------------------------------
    # Example: Cold-start user (unknown user_id) with liked movies
    # --------------------------------------------------------
    # Take some movies from existing dataset as "liked"
    example_liked_movies = movies['movieId'].sample(5, random_state=RANDOM_SEED).tolist()
    print("\nCold-start example user with liked_movie_ids:", example_liked_movies)

    recs_cold = recommend_movies(
        user_id=999999,  # unknown user
        user_id_to_idx=user_id_to_idx,
        idx_to_movie_id=idx_to_movie_id,
        movies_df=movies,
        mf_pred_matrix=mf_pred_matrix,
        train_ratings=train_ratings,
        movie_embeddings=movie_embeddings,
        n_recs=10,
        alpha=0.0,  # no CF for truly cold-start
        beta=1.0,
        popularity_scores=popularity_scores,
        cold_start_liked_movie_ids=example_liked_movies,
        movie_id_to_idx=movie_id_to_idx
    )
    print(recs_cold)


if __name__ == "__main__":
    main()