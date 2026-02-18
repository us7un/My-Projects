# Part 3: Recommender Design

"""
Part 3: Recommender Design

Design and implement two different recommendation algorithms for the music system.
"""

import pandas as pd
import part1 as p1
import part2 as p2
import numpy as np

"""Method to return Tu patience score for a given user from the ratings dataframe"""
def get_user_patience(user_id, ratings_df):
    tu_data = p2.get_time_to_favorite(ratings_df)  # get Tu data for all users

    user_row = tu_data[tu_data['user_id'] == user_id]  # filter for the given user_id

    # RETURN USER'S Tu IF EXISTS, ELSE RETURN AVERAGE Tu, ELSE RETURN DEFAULT 10
    if not user_row.empty:
        return user_row.iloc[0]['Tu']
    else:
        if not tu_data.empty:
            return tu_data['Tu'].mean()
        else:
            return 10


"""Method to calculate importance score for a feature based on its impact on 5-star ratings"""
# calculate binomial importance
def calculate_importance(feature, dataframe: pd.DataFrame):
    pf = p1.compute_pfeature(feature, dataframe)  # P(f=v)
    p5_f = p1.compute_p5_given_feature(dataframe, feature)  # P(5 | f=v)
    p5 = p1.compute_global_pk(5, dataframe)  # P(5)

    common_index = pf.index.intersection(p5_f.index)  # index alignment

    importance = (pf.loc[common_index] * (p5_f.loc[common_index] - p5).abs()).sum()  # weighted importance calculation

    return importance
"""Helper method"""
def compute_global_expected_utility(df,utility):
    counts = df["rating"].value_counts(normalize=True)
    return sum(counts.get(r, 0) * utility[r] for r in utility) # r stands for rating

"""Method to calculate multinomial importance of a feature"""
def calculate_importance_multinomial(feature, dataframe):
    utility_weights = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}

    # P(f=v)
    pf = p1.compute_pfeature(feature, dataframe)

    # P(r | f=v)
    groups = dataframe.groupby([feature, "rating"]).size().unstack(fill_value=0)

    # Laplace smoothing
    alpha = 1.0
    K = 5
    probs = (groups + alpha).div(groups.sum(axis=1) + alpha * K, axis=0)

    # global expected utility
    global_u = compute_global_expected_utility(dataframe,utility_weights)

    importance = 0.0
    for v in probs.index:
        local_u = sum(probs.loc[v, r] * utility_weights[r] for r in utility_weights)
        importance += pf.loc[v] * abs(local_u - global_u) # weighted importance calculation

    return importance

"""Method to compute net utility of a feature"""
def compute_net_utility_for_feature(user_id, merged, feature):
    df_user = merged.loc[merged["user_id"] == user_id]
    if df_user.empty: # if the user does not exist
        return pd.Series(dtype=float)
    counts = df_user.groupby([feature, "rating"]).size().unstack(fill_value=0) # table of track ratings
    for k in range(1, 6):
        if k not in counts.columns:
            counts[k] = 0
    # Laplace smoothing
    alpha = 1.0
    K = 5

    row_totals = counts.sum(axis=1)
    probs = (counts + alpha).div(row_totals + alpha * K, axis=0)
    utility_weights = {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}

    net_utility = pd.Series(0.0, index=probs.index) # initialize net_utility series

    for rating, weight in utility_weights.items():
        net_utility += probs[rating] * weight

    return net_utility


"""Method to return a sorted series of probabilities of features given it got 5 stars"""
# This function applies conditional filtering based on the user preferences and recommends different songs
def conditional_filtering(merged, tracks,ratings, user_id, n_recommendations=10):
    patience_score = get_user_patience(user_id, ratings) # get the user patience level
    feature_weights = { # assign default weights for each related feature (biased weights)
        "primary_artist_name": 0.25,
        "ab_genre_dortmund_value": 0.1,
        "ab_genre_rosamerica_value": 0.1,
        "ab_mood_happy_value": 0.04,
        "ab_mood_sad_value": 0.04,
        "ab_mood_party_value": 0.04,
        "ab_mood_relaxed_value": 0.04,
        "ab_mood_aggressive_value": 0.04,
        "ab_timbre_value": 0.05,
        "ab_voice_instrumental_value": 0.05,
        "track_popularity": 0.13,
        "album_release_year": 0.12,
    }

    for key in feature_weights.keys():
        feature_weights[key] = calculate_importance_multinomial(key, merged) # calculate importance of each feature

    avg_patience = p2.get_time_to_favorite(ratings)["Tu"].mean() # mean patience value based on the given dataset

    if patience_score < avg_patience: # if the user is impatient
        factor = (avg_patience - patience_score) / avg_patience  #
        # if the user is impatient we increase the score of the popular and guarantee tracks
        if "track_popularity" in feature_weights:
            feature_weights["track_popularity"] *= (2 - factor) # the multiplier is in the [1,2] interval
    else:
        if "track_popularity" in feature_weights:
            feature_weights["track_popularity"] *= 0.5 # more niche and outlier tracks are recommended

    # scaling the scores in order to make their sum 1.0
    current_total_weight = sum(feature_weights.values())
    if current_total_weight > 0:
        for key in feature_weights:
            feature_weights[key] /= current_total_weight

    feature_tables = {}
    for feature in feature_weights:
        feature_tables[feature] = compute_net_utility_for_feature(user_id, merged, feature)

    scores = []
    rated_tracks = set(
        merged.loc[merged["user_id"] == user_id, "track_id"]
    )
    candidate_tracks = tracks[~tracks["track_id"].isin(rated_tracks)] # do not recommend the songs that the user already listened and rated
    # calculate the scores for each track
    for track in candidate_tracks.itertuples(index=False):
        score = 0.0
        for feature, weight in feature_weights.items():
            value = getattr(track, feature)

            pk_feature = feature_tables[feature] # get the utility value of the feature

            if value in pk_feature.index:
                score += weight * pk_feature[value] # calculate the score
            else:
                score += weight * 0  # fallback
        scores.append(score)

    score_dataframe = candidate_tracks[['track_name','track_id']].copy()
    score_dataframe['score'] = scores

    return score_dataframe.sort_values('score', ascending=False).head(n_recommendations)

"""MonteCarlo version of get_user_patience"""
def get_user_patience_for_monte_carlo(tu_data, user_id, ratings_df):
    user_row = tu_data[tu_data['user_id'] == user_id]  # filter for the given user_id

    # RETURN USER'S Tu IF EXISTS, ELSE RETURN AVERAGE Tu, ELSE RETURN DEFAULT 10
    if not user_row.empty:
        return user_row.iloc[0]['Tu']
    else:
        if not tu_data.empty:
            return tu_data['Tu'].mean()
        else:
            return 10

"""Method to get feature likings of a certain user"""
def feature_weights(tu_data, merged,ratings, user_id):
    patience_score = get_user_patience_for_monte_carlo(tu_data,user_id, ratings)  # get the user patience level
    feature_weights = {  # assign default weights for each related feature (biased weights)
        "primary_artist_name": 0.25,
        "ab_genre_dortmund_value": 0.1,
        "ab_genre_rosamerica_value": 0.1,
        "ab_mood_happy_value": 0.04,
        "ab_mood_sad_value": 0.04,
        "ab_mood_party_value": 0.04,
        "ab_mood_relaxed_value": 0.04,
        "ab_mood_aggressive_value": 0.04,
        "ab_timbre_value": 0.05,
        "ab_voice_instrumental_value": 0.05,
        "track_popularity": 0.13,
        "album_release_year": 0.12,
    }

    for key in feature_weights.keys():
        feature_weights[key] = calculate_importance_multinomial(key, merged)  # calculate importance of each feature

    avg_patience = p2.get_time_to_favorite(ratings)["Tu"].mean()  # mean patience value based on the given dataset

    if patience_score < avg_patience:  # if the user is impatient
        factor = (avg_patience - patience_score) / avg_patience  #
        # if the user is impatient we increase the score of the popular and guarantee tracks
        if "track_popularity" in feature_weights:
            feature_weights["track_popularity"] *= (2 - factor)  # the multiplier is in the [1,2] interval
    else:
        if "track_popularity" in feature_weights:
            feature_weights["track_popularity"] *= 0.5  # more niche and outlier tracks are recommended

    # scaling the scores in order to make their sum 1.0
    current_total_weight = sum(feature_weights.values())
    if current_total_weight > 0:
        for key in feature_weights:
            feature_weights[key] /= current_total_weight

    return feature_weights

"""Conditional filtering recommender"""
# This function applies conditional filtering based on the user preferences and recommends different songs
def conditional_filtering_for_monte_carlo(merged, tracks, user_id, feature_weights, n_recommendations=10):
    feature_tables = {}
    for feature in feature_weights:
        feature_tables[feature] = compute_net_utility_for_feature(user_id, merged, feature)

    scores = []
    rated_tracks = set(
        merged.loc[merged["user_id"] == user_id, "track_id"]
    )
    candidate_tracks = tracks[~tracks["track_id"].isin(rated_tracks)] # do not recommend the songs that the user already listened and rated
    # calculate the scores for each track
    for track in candidate_tracks.itertuples(index=False):
        score = 0.0
        for feature, weight in feature_weights.items():
            value = getattr(track, feature)

            pk_feature = feature_tables[feature] # get the utility value of the feature

            if value in pk_feature.index:
                score += weight * pk_feature[value] # calculate the score
            else:
                score += weight * 0  # fallback
        scores.append(score)

    score_dataframe = candidate_tracks[['track_name','track_id']].copy()
    score_dataframe['score'] = scores

    return score_dataframe.sort_values('score', ascending=False).head(n_recommendations)


"""Method to return a sorted dataframe of recommended tracks based on utility scores, i.e. utility based probabilistic recommender"""
def utility_based_recommender(merged, tracks, ratings, user_id, n_recommendations=10, random_state=None):
    random = np.random.RandomState(random_state)

    # COMPUTE TRACK UTILITY ESTIMATE BASED ON PREVIOUS WEIGHTS
    FEATURE_WEIGHTS = {
        "primary_artist_name": 0.35,
        "ab_genre_dortmund_value": 0.15,
        "ab_genre_rosamerica_value": 0.15,
        "ab_mood_happy_value": 0.04,
        "ab_mood_sad_value": 0.04,
        "ab_mood_party_value": 0.04,
        "ab_mood_relaxed_value": 0.04,
        "ab_mood_aggressive_value": 0.04,
        "ab_timbre_value": 0.05,
        "ab_voice_instrumental_value": 0.05,
        "track_popularity": 0.03,
        "album_release_year": 0.02,
    }

    for key in list(FEATURE_WEIGHTS.keys()):
        FEATURE_WEIGHTS[key] = calculate_importance(key, merged)  # calculate importance

    current_total_weight = sum(FEATURE_WEIGHTS.values())  # normalize weights
    if current_total_weight > 0:
        for key in FEATURE_WEIGHTS:
            FEATURE_WEIGHTS[key] /= current_total_weight  # normalize
    else:
        n = len(FEATURE_WEIGHTS)
        for key in FEATURE_WEIGHTS:
            FEATURE_WEIGHTS[key] = 1.0 / n  # equal weights if all zero (not likely for large data)

    # CALCULATE UTILITY SCORES
    feature_tables = {}
    for feature in FEATURE_WEIGHTS:
        feature_tables[feature] = p1.compute_pk_given_feature_for_user(5, user_id, merged, [feature], 1)  # P(f=v | 5)

    global_p5 = p1.compute_global_pk(5, merged)  # P(5)

    utilities = []
    for track in tracks.itertuples(index=False):
        utility = 0.0
        for feature, weight in FEATURE_WEIGHTS.items():
            value = getattr(track, feature)  # get feature value
            pk_feature = feature_tables[feature]  # get P(f=v | 5)
            if value in pk_feature.index:  # if value exists
                utility += weight * pk_feature[value]  # weighted contribution
            else:
                utility += weight * global_p5  # fallback to global P(5)
        utilities.append(utility)  # store utility
    utilities = np.array(utilities)  # convert to numpy array

    gamma = 1.8  # >1 favors top utilities (try 1.4-2.5).
    utilities = np.maximum(utilities, 0.0) ** gamma

    # NORMALIZE UTILITIES TO PROBABILITIES
    sum_utilities = np.sum(utilities)
    if sum_utilities > 0:
        probabilities = utilities / sum_utilities
    else:
        # Fallback: Uniform distribution if all utilities are 0
        probabilities = np.ones(len(utilities)) / len(utilities)
    
    # MIX ACCORDING TO PATIENCE SCORE
    patience_score = get_user_patience(user_id, ratings)
    tu_dataframe = p2.get_time_to_favorite(ratings)
    avg_patience = tu_dataframe["Tu"].mean()

    relative_patience = patience_score / avg_patience if avg_patience > 0 else 1.0
    exploration_rate = float(
        np.clip(0.03 + 0.15 * (relative_patience - 1), 0.02, 0.12))  # between 0.02 and 0.12, patience-based clamp

    # focus exploration on the worst X% of items
    tail_frac = 0.2  # fraction of items considered bottom
    n = len(probabilities)
    tail_k = max(1, int(np.ceil(tail_frac * n)))
    tail_idx = np.argsort(probabilities)[:tail_k]  # indices of bottom items (lowest prob)
    exploration_distance = np.zeros_like(probabilities)
    inv_scores = 1.0 - probabilities[tail_idx]
    exploration_distance[tail_idx] = inv_scores / inv_scores.sum()
    final_probabilities = (1.0 - exploration_rate) * probabilities + exploration_rate * exploration_distance  # mix

    # RENORMALIZE FINAL PROBABILITIES
    sum_final = np.sum(final_probabilities)
    if sum_final > 0:
        final_probabilities /= sum_final
    else:
        # Fallback: Uniform distribution
        final_probabilities = np.ones(len(final_probabilities)) / len(final_probabilities)

    # SAMPLE RECOMMENDATIONS BASED ON FINAL PROBABILITIES
    n_tracks = len(tracks)  # number of tracks
    take = min(n_recommendations, n_tracks)  # number to take
    recommended_indices = random.choice(n_tracks, size=take, replace=False,
                                        p=final_probabilities)  # sample without replacement

    results = tracks[['track_name', 'track_id']].copy().reset_index(drop=True)  # prepare results dataframe
    results['score'] = final_probabilities  # add scores

    return results.iloc[recommended_indices].sort_values('score',
                                                         ascending=False)  # return recommended tracks sorted by score

"""Method to compute feature weights for utility based recommender"""
def feature_for_utility(merged):
    # COMPUTE TRACK UTILITY ESTIMATE BASED ON PREVIOUS WEIGHTS
    FEATURE_WEIGHTS = {
        "primary_artist_name": 0.35,
        "ab_genre_dortmund_value": 0.15,
        "ab_genre_rosamerica_value": 0.15,
        "ab_mood_happy_value": 0.04,
        "ab_mood_sad_value": 0.04,
        "ab_mood_party_value": 0.04,
        "ab_mood_relaxed_value": 0.04,
        "ab_mood_aggressive_value": 0.04,
        "ab_timbre_value": 0.05,
        "ab_voice_instrumental_value": 0.05,
        "track_popularity": 0.03,
        "album_release_year": 0.02,
    }

    for key in list(FEATURE_WEIGHTS.keys()):
        FEATURE_WEIGHTS[key] = calculate_importance(key, merged)  # calculate importance

    current_total_weight = sum(FEATURE_WEIGHTS.values())  # normalize weights
    if current_total_weight > 0:
        for key in FEATURE_WEIGHTS:
            FEATURE_WEIGHTS[key] /= current_total_weight  # normalize
    else:
        n = len(FEATURE_WEIGHTS)
        for key in FEATURE_WEIGHTS:
            FEATURE_WEIGHTS[key] = 1.0 / n  # equal weights if all zero (not likely for large data)
    return FEATURE_WEIGHTS

"""MonteCarlo version of probabilistic recommender"""
def utility_based_recommender_for_monte_carlo(merged, tracks, ratings, user_id, FEATURE_WEIGHTS, n_recommendations=10, random_state=None):
    random = np.random.RandomState(random_state)
    # CALCULATE UTILITY SCORES
    feature_tables = {}
    for feature in FEATURE_WEIGHTS:
        feature_tables[feature] = p1.compute_pk_given_feature_for_user(5, user_id, merged, [feature], 1)  # P(f=v | 5)

    global_p5 = p1.compute_global_pk(5, merged)  # P(5)

    utilities = []
    for track in tracks.itertuples(index=False):
        utility = 0.0
        for feature, weight in FEATURE_WEIGHTS.items():
            value = getattr(track, feature)  # get feature value
            pk_feature = feature_tables[feature]  # get P(f=v | 5)
            if value in pk_feature.index:  # if value exists
                utility += weight * pk_feature[value]  # weighted contribution
            else:
                utility += weight * global_p5  # fallback to global P(5)
        utilities.append(utility)  # store utility
    utilities = np.array(utilities)  # convert to numpy array

    gamma = 1.8  # >1 favors top utilities (try 1.4-2.5).
    utilities = np.maximum(utilities, 0.0) ** gamma

    # NORMALIZE UTILITIES TO PROBABILITIES
    probabilities = utilities / np.sum(utilities)

    # MIX ACCORDING TO PATIENCE SCORE
    patience_score = get_user_patience(user_id, ratings)
    tu_dataframe = p2.get_time_to_favorite(ratings)
    avg_patience = tu_dataframe["Tu"].mean()

    relative_patience = patience_score / avg_patience if avg_patience > 0 else 1.0
    exploration_rate = float(
        np.clip(0.03 + 0.15 * (relative_patience - 1), 0.02, 0.12))  # between 0.02 and 0.12, patience-based clamp

    # focus exploration on the worst X% of items
    tail_frac = 0.2  # fraction of items considered bottom
    n = len(probabilities)
    tail_k = max(1, int(np.ceil(tail_frac * n)))
    tail_idx = np.argsort(probabilities)[:tail_k]  # indices of bottom items (lowest prob)
    exploration_distance = np.zeros_like(probabilities)
    inv_scores = 1.0 - probabilities[tail_idx]
    exploration_distance[tail_idx] = inv_scores / inv_scores.sum()
    final_probabilities = (1.0 - exploration_rate) * probabilities + exploration_rate * exploration_distance  # mix
    probs = np.sum(final_probabilities)  # renormalize


    # SAMPLE RECOMMENDATIONS BASED ON FINAL PROBABILITIES
    n_tracks = len(tracks)  # number of tracks
    take = min(n_recommendations, n_tracks)  # number to take
    if probs > 0:
        final_probabilities /= probs
    else:
        final_probabilities = np.ones(n_tracks) / n_tracks
    recommended_indices = random.choice(n_tracks, size=take, replace=False,
                                        p=final_probabilities)  # sample without replacement

    results = tracks[['track_name', 'track_id']].copy().reset_index(drop=True)  # prepare results dataframe
    results['score'] = final_probabilities  # add scores

    return results.iloc[recommended_indices].sort_values('score',ascending=False)  # return recommended tracks sorted by score

def main():
    """Load the tracks and ratings to recommend songs"""

    # CHANGE THE FILENAMES ACCORDING TO THE CSV FILES IN DATA
    tracks = p1.load_tracks("../data/tracks.csv")
    ratings = p1.load_ratings("../data/ratings.csv")
    merged = p1.merge_ratings(tracks, ratings)
    top5 = conditional_filtering(merged, tracks, ratings, "U0002",  n_recommendations=10)
    # CONDITIONAL FILTERING RECOMMENDER
    print("1")
    print(top5.to_string(index=False))
    print()
    # PROBABILISTIC RECOMMENDER
    print("2")
    top5 = utility_based_recommender(merged, tracks, ratings, "U0002", n_recommendations=10)
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
