from typing import List, Dict, Tuple, Any
import pandas as pd
import numpy as np

"""Method to load the track data from the tracks.csv file, converting types of features as desired."""
def load_tracks(path):
    # path: path to tracks.csv file
    dataframe = pd.read_csv(path, index_col=0) 
        
    return dataframe


"""Method to merge tracks and ratings into one common dataframe, keeping song_id/track_id common."""
def merge_ratings(tracks: pd.DataFrame, ratings: pd.DataFrame):
    # tracks: dataframe of track features, ratings: dataframe of user ratings
    # Drop overlapping track metadata from ratings to avoid suffixing
    overlap = set(tracks.columns) & set(ratings.columns)
    overlap -= {"track_id", "song_id"}  # keep join keys
    ratings_clean = ratings.drop(columns=list(overlap), errors="ignore")
    merged = ratings_clean.merge(
        tracks,
        left_on="song_id",
        right_on="track_id",
        how="left"
    )
    return merged

"""Method to return a sorted series of probabilities of songs getting 5 stars given a certain feature"""
def compute_p5_given_feature(dataframe: pd.DataFrame, features: list, laplace: float = 1.0):
    # dataframe: merged dataframe of tracks and ratings, features: list of feature column names to condition on
    """GROUP BY FEATURE VALUES TO GET COUNTS OF RATINGS AND 5 STAR RATINGS"""
    groups = dataframe.groupby(features)["rating"] # group the rating count per feature value
    total = groups.count().rename("total_count") # total count of ratings
    total5 = groups.apply(lambda x : (x==5).sum()).rename("total_5count") # total count of 5star ratings
    probs = pd.concat([total, total5], axis = 1) # organize counts of ratings and 5 stars per feature value

    """LAPLACE SMOOTHING AND CONDITIONAL PROBABILITY CALCULATION
       P(5 | FEATURE = VAL) = P(5 AND FEATURE = VAL) + a / P(FEATURE = VAL) + 2a
       which is here exactly total_5count / total_count"""
    probs["prob5 | f"] = (probs["total_5count"] + laplace) / (probs["total_count"] + laplace * 2)

    return probs["prob5 | f"].sort_values(ascending = False) # sort from highest likely to lowest


def compute_pk_given_feature_for_user(k, user_id, dataframe: pd.DataFrame, features: list, laplace: float = 1.0):
    df_user = dataframe.loc[dataframe["user_id"] == user_id]
    groups = df_user.groupby(features)["rating"] # group the rating count per feature value
    total = groups.count().rename("total_count") # total count of ratings
    total5 = groups.apply(lambda x : (x==k).sum()).rename(f"total_{k}count") # total count of 5star ratings
    probs = pd.concat([total, total5], axis = 1) # organize counts of ratings and 5 stars per feature value

    """LAPLACE SMOOTHING AND CONDITIONAL PROBABILITY CALCULATION
       P(5 | FEATURE = VAL) = P(5 AND FEATURE = VAL) + a / P(FEATURE = VAL) + 2a
       which is here exactly total_5count / total_count"""
    probs["prob5 | f"] = (probs[f"total_{k}count"] + laplace) / (probs["total_count"] + laplace * 2)

    return probs["prob5 | f"].sort_values(ascending = False) # sort from highest likely to lowest

def get_time_to_favorite(ratings_df):
    """
    Calculates Tu (rounds until first 5*) for each user.
    """
    # Sort by user and the order they listened (round_idx)
    if ratings_df is None or ratings_df.empty:
        return pd.DataFrame(columns=['user_id', 'Tu'])

    df = ratings_df.copy()

    if 'user_id' not in df.columns or 'rating' not in df.columns:
        return pd.DataFrame(columns=['user_id', 'Tu'])

    if 'round_idx' not in df.columns:
        df['round_idx'] = df.groupby('user_id').cumcount() + 1
    else:
        df['round_idx'] = pd.to_numeric(df['round_idx'], errors='coerce')
        if df['round_idx'].isna().any():
            mask = df['round_idx'].isna()
            df.loc[mask, 'round_idx'] = df[mask].groupby('user_id').cumcount() + 1

    df = df.sort_values(by=['user_id', 'round_idx'])

    user_times = []

    # Group by user
    for user_id, group in df.groupby('user_id'):
        # Find rows where rating is 5
        fives = group[group['rating'] == 5]

        if not fives.empty:
            # The first 5* rating determines Tu
            # We assume round_idx is 0-indexed or 1-indexed, but we need the count.
            # If round_idx starts at arbitrary numbers, we use rank.

            first_five_idx = fives.iloc[0]['round_idx']
            # IMPORTANT if path is user_ratings.csv add +1 to this function

            # Calculate position in the session (1st song, 2nd song, etc.)
            # If round_idx is sequential 1..N:
            tu = first_five_idx

            # Alternative: if round_idx is not clean, use array position + 1
            # tu = np.where(group['rating'].values == 5)[0][0] + 1

            user_times.append({'user_id': user_id, 'Tu': tu})
        else:
            # "Discard users who never give a 5 star rating"
            user_times.append({'user_id': user_id, 'Tu': np.nan})

    return pd.DataFrame(user_times, columns=['user_id', 'Tu'])

"""Method to return Tu patience score for a given user from the ratings dataframe"""


def get_user_patience(user_id, ratings_df):
    tu_data = get_time_to_favorite(ratings_df)  # get Tu data for all users

    user_row = tu_data[tu_data['user_id'] == user_id]  # filter for the given user_id

    # RETURN USER'S Tu IF EXISTS, ELSE RETURN AVERAGE Tu, ELSE RETURN DEFAULT 10
    if not user_row.empty:
        return user_row.iloc[0]['Tu']
    else:
        if not tu_data.empty:
            return tu_data['Tu'].mean()
        else:
            return 10

def compute_pfeature(feature, dataframe: pd.DataFrame):
    groups = dataframe.groupby(feature)
    counts = groups.size()
    return counts / counts.sum()

def compute_global_pk(k, dataframe: pd.DataFrame):
    return (dataframe["rating"] == k).mean()

"""Method to calculate importance score for a feature based on its impact on 5-star ratings"""

# calculate binomial importance
def calculate_importance(feature, dataframe: pd.DataFrame):
    pf = compute_pfeature(feature, dataframe)  # P(f=v)
    p5_f = compute_p5_given_feature(dataframe, feature)  # P(5 | f=v)
    p5 = compute_global_pk(5, dataframe)  # P(5)

    common_index = pf.index.intersection(p5_f.index)  # index alignment

    importance = (pf.loc[common_index] * (p5_f.loc[common_index] - p5).abs()).sum()  # weighted importance calculation

    return importance

def compute_global_expected_utility(df,utility):
    counts = df["rating"].value_counts(normalize=True)
    return sum(counts.get(r, 0) * utility[r] for r in utility) # r stands for rating


"""Method to return a sorted dataframe of recommended tracks based on utility scores"""


# 1. OPTIMIZED RECOMMENDER (Vectorized)
def utility_based_recommender(tracks: pd.DataFrame, 
                              user_ratings_df: pd.DataFrame, 
                              user_id, 
                              n_recommendations=10, 
                              random_state=None):
    
    random = np.random.RandomState(random_state)
    
    # --- LOGIC FIX: Do not recalculate weights based on just the current user's 3 ratings.
    # Use fixed weights for cold-start, or pass in a pre-computed global_ratings_df if available.
    FEATURE_WEIGHTS = {
        "primary_artist_name": 0.9,

       "ab_genre_dortmund_value": 0.20,

       "ab_genre_rosamerica_value": 0.20,

       "ab_mood_happy_value": 0.10,

       "ab_mood_sad_value": 0.10,

       "ab_mood_party_value": 0.10,

       "ab_mood_relaxed_value": 0.10,

       "ab_mood_aggressive_value": 0.10,

       "ab_timbre_value": 0.04,

       "ab_voice_instrumental_value": 0.04,

       "track_popularity": 0.20,

       "album_release_year": 0.01,
    }
    
    # Normalize weights
    total_w = sum(FEATURE_WEIGHTS.values())
    weights = {k: v / total_w for k, v in FEATURE_WEIGHTS.items()}

    # --- VECTORIZED CALCULATION ---
    # Instead of iterating rows, we initialize an array of zeros
    scores = np.zeros(len(tracks))
    
    # Pre-calculate user preferences (P(f=v | 5))
    # We mix Global Priors with User Priors to handle sparse data (Bayesian Average approach)
    # For this snippet, I will assume we stick to your logic but vectorized.
    
    # Note: In a real scenario, you need to handle columns that aren't categorical (like popularity) differently.
    
    global_p5 = 0.5 # Default fallback probability
    
    for feature, weight in weights.items():
        if feature not in tracks.columns: continue

        # 1. Get the user's probability map for this feature
        # (This function needs to handle the case where the user hasn't seen a specific feature value)
        probs_series = compute_pk_given_feature_for_user(5, user_id, user_ratings_df, [feature], 1.0)
        
        # 2. Map these probabilities to the tracks dataframe
        # tracks[feature] contains values like 'Rock', 'Pop'. 
        # probs_series contains {'Rock': 0.8, 'Pop': 0.2}
        feature_scores = tracks[feature].map(probs_series)
        
        # 3. Fill NaNs (where user hasn't rated this feature value yet) with global default
        feature_scores = feature_scores.fillna(global_p5)
        
        # 4. Add to total score
        scores += (feature_scores.values * weight)

    # Apply Gamma
    gamma = 1.8
    utilities = np.maximum(scores, 0.0) ** gamma
    
    # ... Rest of your normalization and exploration logic ...
    
    # Normalize
    probabilities = utilities / (np.sum(utilities) + 1e-9) # Avoid div/0
    
    # (Patience logic omitted for brevity, but should be calculated ONCE outside)
    
    # Sample
    n_tracks = len(tracks)
    take = min(n_recommendations, n_tracks)
    
    # Use argpartition for efficiency if N is large, but random.choice is fine for <100k
    recommended_indices = random.choice(n_tracks, size=take, replace=False, p=probabilities)
    
    results = tracks.iloc[recommended_indices].copy()
    results['score'] = probabilities[recommended_indices]
    
    return results.sort_values('score', ascending=False)

# 2. FIXING THE QUERY FLOW
# Load tracks ONCE globally
GLOBAL_TRACKS = load_tracks("tracks.csv") # Ensure path is correct relative to script

def query(song_ratings: List[Dict[str, Any]], topk: int = 5) -> List[Tuple[str, str]]:
    
    dummy_user_id = "DUMMY_USER"
    
    # 1. Prepare User Data
    rows = []
    for index, rating in enumerate(song_ratings):
        song_id = rating.get("spotify_id")
        r_val = int(rating.get("rating", 3))
        rows.append({"user_id": dummy_user_id, "song_id": song_id, "rating": r_val, "round_idx": index})
    
    dummy_ratings_df = pd.DataFrame(rows)

    # 2. Merge User Ratings with Track Data 
    # (We need this to know the Genre/Mood of the songs the user just rated)
    # Check if 'track_id' is index or column in GLOBAL_TRACKS
    if 'track_id' not in GLOBAL_TRACKS.columns and GLOBAL_TRACKS.index.name == 'track_id':
         left_on_key = "song_id"
         right_index_bool = True
         right_on_key = None
    else:
         left_on_key = "song_id"
         right_index_bool = False
         right_on_key = "track_id"

    # We merge to get a dataframe representing the USER'S HISTORY enriched with features
    user_history_enriched = dummy_ratings_df.merge(
        GLOBAL_TRACKS,
        left_on=left_on_key,
        right_index=right_index_bool,
        right_on=right_on_key,
        how="left"
    )

    # 3. Pass GLOBAL tracks (candidates) and ENRICHED history to recommender
    recs = utility_based_recommender(
        tracks=GLOBAL_TRACKS, 
        user_ratings_df=user_history_enriched, # Pass the enriched history
        user_id=dummy_user_id, 
        n_recommendations=topk
    )

    return list(zip(recs.index if 'track_id' not in recs else recs['track_id'], recs['track_name']))

def test_recommender():
    """Test the recommender function."""
    example_ratings = [
        {"song": "Shape of You", "rating": 5, "spotify_id": "7qiZfU4dY1lWllzX7mPBI3", "artist": "Ed Sheeran"},
        {"song": "Blinding Lights", "rating": 5, "spotify_id": "0VjIjW4GlUZAMYd2vXMi3b", "artist": "The Weeknd"},
        {"song": "Paris", "rating": 5, "spotify_id": "72jbDTw1piOOj770jWNeaG", "artist": "The Chainsmokers"}
    ]
    
    recommendations = query(example_ratings, topk=10)
    print(f"Generated {len(recommendations)} recommendations:")
    for i, (track_id, track_name) in enumerate(recommendations, 1):
        print(f"{i}. {track_name} (ID: {track_id})")

if __name__ == "__main__":
    test_recommender()