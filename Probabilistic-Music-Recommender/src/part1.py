# Part 1: Conditional Probability Modeling

"""
Part 1: Conditional Probability Modeling

Estimate how likely a song is to receive a 5★ rating using conditional probabilities.
"""

import pandas as pd
import numpy as np

"""Method to load the track data from the tracks.csv file, converting types of features as desired."""
def load_tracks(path):
    # path: path to tracks.csv file
    dataframe = pd.read_csv(path, index_col = 0, dtype={"track_id": str}) # read the csv into a pandas dataframe
    dataframe["explicit"] = dataframe["explicit"].astype(str).map({"True": True, "False": False}) # convert explicit to bool
    return dataframe

"""Method to load the rating data from the ratings.csv file, converting types of features as desired."""
def load_ratings(path):
    # path: path to ratings.csv file
    dataframe = pd.read_csv(path, dtype={"user_id": str,"track_id": str, "rating": int}) # read the csv into a pandas dataframe
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

"""Method to return a sorted series of probabilities of songs having a certain feature given it got 5 stars"""
def compute_feature_given_p5(dataframe: pd.DataFrame, features: list, laplace: float = 1.0):
    # dataframe: merged dataframe of tracks and ratings, features: list of feature column names to condition on
    """COMPUTE P(5 | FEATURE = VAL), SAME AS ABOVE"""
    groups = dataframe.groupby(features)["rating"]
    total = groups.count().rename("total_count")
    total5 = groups.apply(lambda x: (x == 5).sum()).rename("total_5count")
    probs = pd.concat([total, total5], axis = 1)
    probs["prob5 | f"] = (probs["total_5count"] + laplace) / (probs["total_count"] + laplace * 2)

    """COMPUTE P(FEATURES)"""
    total_ratings = len(dataframe) # get the total number of entries in the dataframe
    K = probs.shape[0] # how many entries we smooth out (CARDINALITY OF OUR RATINGS SET)
    # LAPLACE SMOOTHING FOR P(FEATURES)
    probs["probfeature"] = (probs["total_count"] + laplace) / (total_ratings + laplace * K)

    """COMPUTE P(5)"""
    total5_all = int((dataframe["rating"] == 5).sum()) # count of all 5 star ratings
    # LAPLACE SMOOTHING FOR P(5)
    prob5_all = (total5_all + laplace) / (total_ratings + laplace * 2)
    probs["prob5"] = prob5_all

    """APPLY BAYES RULE TO SMOOTHED DATA"""
    probs["f | prob5"] = probs["prob5 | f"] * probs["probfeature"] / probs["prob5"]
    # NORMALIZE TO 1 FOR PROBABILITY DISTRIBUTION TO BE CORRECT
    sums = probs["f | prob5"].sum()
    if sums > 0:
        probs["f | prob5"] /= sums # so they sum up to 1

    return probs["f | prob5"].sort_values(ascending = False) # sort from highest likely to lowest

"""Method to compute analogous probabilities for personal session data and compares results with rest of the group's dataset"""
def personal_and_group_analysis(personal_probs_list):
    # personal_probs_list is a list of pandas Series, each containing P(5★ | feature) for a member
    '''BUILD DATAFRAME: COLUMNS ARE MEMBERS' PROBABILITIES'''
    combined_df = pd.concat(personal_probs_list, axis=1)
    combined_df.columns = [f'Pm(prob5 | f)M_{i+1}' for i in range(len(personal_probs_list))]

    '''AVERAGE PROBABILITIES ACROSS MEMBERS FOR GROUP INSIGHT'''
    combined_df['Pgroup(prob5 | f)'] = combined_df.mean(axis=1)

    return combined_df.sort_values(by='Pgroup(prob5 | f)', ascending=False)

def compute_global_pk(k, dataframe: pd.DataFrame):
    return (dataframe["rating"] == k).mean()

def compute_pfeature(feature, dataframe: pd.DataFrame):
    groups = dataframe.groupby(feature)
    counts = groups.size()
    return counts / counts.sum()


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


def main():



    # First member data
    tracks = load_tracks("../data/tracks.csv") # EDIT HERE IF USING DIFFERENT DATA
    ratings_global = load_ratings("../data/ratings.csv") # EDIT HERE IF USING DIFFERENT DATA
    merged_global = merge_ratings(tracks, ratings_global)
    ratings_ustun = load_ratings("../data/ratings_ustun.csv") # EDIT HERE IF USING DIFFERENT DATA
    merged_ustun = merge_ratings(tracks, ratings_ustun)

    # Second member data
    ratings_enes = load_ratings("../data/ratings_enes.csv") # EDIT HERE
    merged_enes = merge_ratings(tracks, ratings_enes)

    member_dataframes = [merged_ustun, merged_enes] # ADD MORE DATAFRAMES IF MORE MEMBERS
    
    # ACCUMULATE PERSONAL PROBABILITIES SERIES
    personal_probs = [compute_p5_given_feature(dataframe, ["explicit"]) for dataframe in member_dataframes]

    # SERIES OF ANALYSIS RESULTS
    print("Global Analysis of P(5★ | explicit):")
    print(compute_p5_given_feature(merged_global, ["explicit"]))

    print("\nAnalysis of P(explicit | 5★):")
    print(compute_feature_given_p5(merged_global, ["explicit"]))
    
    print("\nAnalysis of P(5★ | primary_artist_name, explicit):")
    print(compute_p5_given_feature(merged_global, ["primary_artist_name", "explicit"]).head(10))

    print("\nAnalysis of P(primary_artist_name, explicit | 5★):")
    print(compute_feature_given_p5(merged_global, ["primary_artist_name", "explicit"]).head(10))

    print("\nPersonal and Group Analysis of P(5★ | explicit):")
    print(personal_and_group_analysis(personal_probs))




if __name__ == "__main__":
    main()
