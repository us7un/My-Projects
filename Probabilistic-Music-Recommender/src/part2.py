# Part 2: User Variability Modeling

"""
Part 2: User Variability Modeling

Model how many recommendations it takes for users to rate a song 5★ using
geometric and Beta-geometric distributions.
"""

import pandas as pd
import numpy as np
import part1 as p1

import matplotlib.pyplot as plt
import math


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


def fit_geometric(tu_series):
    # MLE for Geometric: p = 1 / mean(Tu)
    p_hat = 1 / tu_series.mean()
    return p_hat


def geometricDist(ratings_df):
    # Calculate Tu for all users
    tu_data = get_time_to_favorite(ratings_df)
    

    p_geo = fit_geometric(tu_data['Tu'])
    

    return tu_data, p_geo


def geometricPMF(x, p_geo):

    return pow((1-p_geo), x-1)*p_geo


def lbeta_manual(a, b):
    """Computes log(Beta(a, b)) using log-gamma function."""
    # log(B(a, b)) = lgamma(a) + lgamma(b) - lgamma(a+b)
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_geometric_neg_log_lik(params, tu_data):
    """
    Negative Log Likelihood for Beta-Geometric Distribution.
    P(T=t) = B(alpha+1, beta+t-1) / B(alpha, beta)
    """
    alpha, beta = params
    if alpha <= 0.001 or beta <= 0.001:
        return 1e15  # Return huge error for invalid params

    # We maximize sum of log probabilities, so we minimize negative sum
    # log P(t) = lbeta(alpha+1, beta+t-1) - lbeta(alpha, beta)

    n = len(tu_data)

    # Constant term: - sum( log B(alpha, beta) ) = - N * lbeta(alpha, beta)
    term_const = n * lbeta_manual(alpha, beta)

    # Variable term: sum( lbeta(alpha+1, beta+t-1) )
    term_var = 0.0
    for t in tu_data:
        term_var += lbeta_manual(alpha + 1, beta + t - 1)

    # NLL = - (term_var - term_const)
    return -(term_var - term_const)


def coordinate_descent(func, data, start=(1.0, 1.0), step=1.0, tol=1e-4, max_iter=200):

    a, b = start
    best = func([a, b], data)

    for _ in range(max_iter):
        improved = False

        # --- optimize a ---
        for direction in (+1, -1):
            a_try = a + direction * step
            if a_try <= 0:
                continue
            s = func([a_try, b], data)
            if s < best:
                a, best = a_try, s
                improved = True

        # --- optimize b ---
        for direction in (+1, -1):
            b_try = b + direction * step
            if b_try <= 0:
                continue
            s = func([a, b_try], data)
            if s < best:
                b, best = b_try, s
                improved = True

        # If no improvement, reduce step size
        if not improved:
            step *= 0.5
            if step < tol:
                break

    return [a, b]


def bgDist(func, tu_values):
    # Initial guess for alpha, beta
    alpha_hat, beta_hat = func(beta_geometric_neg_log_lik, tu_values)
    print(f"[Beta-Geometric] Estimated alpha: {alpha_hat:.4f}, beta: {beta_hat:.4f}")
    
    #cost = beta_geometric_neg_log_lik((alpha_hat,beta_hat), tu_values)
    #print("neg_log_lik: {cost}")


    return alpha_hat, beta_hat


def model(ratings_df):
    
    tu_data, p_geo = geometricDist(ratings_df)
    
    tu_values = tu_data['Tu'].dropna().values # Exclude users without 5-star ratings
    
    print(f"Number of valid users found: {len(tu_values)}")
    print(f"Average rounds to find a favorite: {tu_data['Tu'].mean():.2f}")
    print(f"\n[Geometric Model] Estimated p: {p_geo:.4f}")

    alpha_hat, beta_hat = bgDist(coordinate_descent ,tu_values)
    #alpha_hat, beta_hat = bgDist(custom_optimize ,tu_values)

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    plt.figure(figsize=(12, 6))

    # Observed Data
    max_t = tu_values.max()
    x = np.arange(1, max_t + 1)
    counts = tu_data['Tu'].value_counts(normalize=True).sort_index()
    plt.bar(counts.index, counts.values, alpha=0.5,
            label='Observed Data', color='gray')

    # Geometric Fit
    geo_probs = geometricPMF(x, p_geo)
    plt.plot(x, geo_probs, 'r--',
             label=f'Geometric (p={p_geo:.2f})', linewidth=2)

    # Beta-Geometric Fit
    # P(T=t) = B(alpha+1, beta+t-1) / B(alpha, beta)
    bg_probs = []
    const_part = lbeta_manual(alpha_hat, beta_hat)
    for t in x:
        log_p = lbeta_manual(alpha_hat + 1, beta_hat + t - 1) - const_part
        bg_probs.append(math.exp(log_p))
    plt.plot(x, bg_probs, 'b-', label=f'Beta-Geo (a={alpha_hat:.1f}, b={beta_hat:.1f})')
    
    print("\nplotting graph (it should appear in screen)")
    
    plt.xlabel('Rounds until 5-star rating (Tu)')
    plt.ylabel('Probability')
    plt.title('Fit Comparison: Geometric vs. Beta-Geometric')
    plt.legend()
    plt.show()


def mann_whitney_manual(x, y):
    """
    Manual Mann-Whitney U test (two-sided) using normal approximation.
    Returns: U statistic, p-value
    """
    # Convert to numpy arrays and drop NaNs
    x = np.asarray(x.dropna(), dtype=float)
    y = np.asarray(y.dropna(), dtype=float)

    n1 = len(x)
    n2 = len(y)

    if n1 == 0 or n2 == 0:
        return np.nan, np.nan

    # Combine samples
    combined = np.concatenate([x, y])

    # Rank the combined data (average ranks for ties)
    order = np.argsort(combined)
    ranks = np.empty(len(combined), dtype=float)

    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) - 1 and combined[order[j]] == combined[order[j + 1]]:
            j += 1

        # Average rank for ties
        avg_rank = (i + j + 2) / 2.0  # ranks start at 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank

        i = j + 1

    # Sum of ranks for group x
    R1 = np.sum(ranks[:n1])

    # Mann–Whitney U statistics
    U1 = R1 - n1 * (n1 + 1) / 2
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    # Normal approximation
    mu_U = n1 * n2 / 2
    sigma_U = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    if sigma_U == 0:
        return U, 1.0

    z = (U - mu_U) / sigma_U

    # Two-sided p-value using error function
    p_value = 2 * 0.5 * (1 + math.erf(z / math.sqrt(2)))

    return U, p_value


def hypothesisTesting(ratings, tracks):
    
    # ==========================================
    # 5. HYPOTHESIS TESTING
    # ==========================================
    # Define Groups: e.g., Users who like "Popular" vs "Niche" tracks
    # Logic: If a user's *first* 5-star song has popularity > 50, they are Group A.
    
    # Merge Tu data with song info to check the popularity of the favorite song
    # We need to find which song corresponded to the Tu event
    first_faves = ratings[ratings['rating'] == 5].sort_values('round_idx').groupby('user_id').first().reset_index()
    first_faves = first_faves.merge(tracks[['track_id', 'track_popularity']], left_on='song_id', right_on='track_id')
    
    # Split groups
    median_pop = first_faves['track_popularity'].median()
    group_a_users = first_faves[first_faves['track_popularity'] >= median_pop]['user_id']
    group_b_users = first_faves[first_faves['track_popularity'] < median_pop]['user_id']
    
    tu_data =  get_time_to_favorite(ratings)
    
    # Get Tu values for each group
    tu_a = tu_data[tu_data['user_id'].isin(group_a_users)]['Tu']
    tu_b = tu_data[tu_data['user_id'].isin(group_b_users)]['Tu']
    
    print("\n[Hypothesis Test]")
    print(f"Group A (Popular Tastes): n={len(tu_a)}, mean Tu={tu_a.mean():.2f}")
    print(f"Group B (Niche Tastes): n={len(tu_b)}, mean Tu={tu_b.mean():.2f}")
    
    # Check Normality (likely not normal, so Mann-Whitney is safer)
    # But description says "t-test if normal, Mann-Whitney if skewed" [cite: 95, 96]
   
    
    u_stat, p_val = mann_whitney_manual(tu_a, tu_b)
    print(f"\n[Hypothesis Test] Mann-Whitney U: {u_stat}, p-value: {p_val:.5f}")
    if p_val < 0.05:
        print("Significant difference found.")
    else:
        print("No significant difference found.")
        

def main():

    ratings_df = p1.load_ratings("../data/ratings.csv")
    tracks_df = p1.load_tracks("../data/tracks.csv")

    model(ratings_df)
    
    hypothesisTesting(ratings_df, tracks_df)
    
    print("\nFinished executing")


if __name__ == "__main__":
    main()
