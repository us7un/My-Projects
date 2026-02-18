# Part 4: Monte Carlo Evaluation

"""
Part 4: Monte Carlo Evaluation

Evaluate recommendation models using statistical simulation and Monte Carlo methods.
"""
import numpy as np
import part1 as p1
import part2 as p2
import part3 as p3
import matplotlib.pyplot as plt
import time


def train_and_test_split(merged, test_ratio=0.2):
    """
    Split user data into train and test sets.
    Randomly selects a user and splits their ratings.

    """
    unique_users = merged["user_id"].unique()
    num_users = len(unique_users)

    if num_users == 0:
        return None, None, None

    # Randomly select a user
    random_user_idx = np.random.randint(0, num_users)
    user_id = unique_users[random_user_idx]

    # Get all songs rated by this user
    user_ratings = merged[merged["user_id"] == user_id]
    num_songs = len(user_ratings)

    if num_songs < 2:
        return None, None, None

    # Split into test set
    test_size = max(1, int(num_songs * test_ratio))
    test_songs = user_ratings.sample(n=test_size, random_state=None)

    # Create training set (all data except test songs)
    merged_train = merged.drop(test_songs.index)

    return merged_train, test_songs, user_id


def compute_hit_at_k(recommendations_df, test_songs_df, k):
    """
    Compute Hit@k: whether at least one test song appears in top-k recommendations.
    """
    top_k_recs = recommendations_df.head(k)
    test_track_ids = set(test_songs_df["track_id"].values)
    rec_track_ids = set(top_k_recs["track_id"].values)

    return 1 if len(test_track_ids & rec_track_ids) > 0 else 0


def compute_average_rating(recommendations_df, test_songs_df, k):
    """
    Compute average rating of recommended songs that appear in test set.
    """
    top_k_recs = recommendations_df.head(k)
    test_track_ids = test_songs_df["track_id"].values

    # Find which recommended songs are in test set
    hits = top_k_recs[top_k_recs["track_id"].isin(test_track_ids)]

    if len(hits) == 0:
        return None

    # Get ratings for these songs
    hit_ids = hits["track_id"].values
    ratings = test_songs_df[test_songs_df["track_id"].isin(hit_ids)]["rating"]

    return ratings.mean()


def compute_time_to_five_star(recommendations_df, test_songs_df, max_search=100):
    """
    Find the position (1-indexed) of the first 5-star song in recommendations.
    """
    five_star_test = test_songs_df[test_songs_df["rating"] == 5]

    if five_star_test.empty:
        return None

    target_ids = set(five_star_test["track_id"].values)

    # Check each recommendation
    for idx in range(min(len(recommendations_df), max_search)):
        if recommendations_df.iloc[idx]["track_id"] in target_ids:
            return idx + 1  # 1-indexed position

    return max_search  # Not found within max_search


def monte_carlo_simulation(merged, tracks, ratings, n_simulations=1000, k=5, max_search=100):
    """
    Run Monte Carlo simulation to evaluate both recommendation models.
    """

    # Precompute model parameters once
    tu_data = p2.get_time_to_favorite(ratings)
    FEATURE_WEIGHTS_UTILITY = p3.feature_for_utility(merged)

    results = {
        'conditional': {
            'hit_at_k': [],
            'avg_ratings': [],
            'time_to_5star': []
        },
        'utility': {
            'hit_at_k': [],
            'avg_ratings': [],
            'time_to_5star': []
        }
    }

    successful_rounds = 0

    for round_idx in range(n_simulations):
        if (round_idx + 1) % 100 == 0:
            print(f"Progress: {round_idx + 1}/{n_simulations} rounds completed...") # for interactivity

        # Split data into train/test
        merged_train, test_songs, user_id = train_and_test_split(merged)

        if merged_train is None or test_songs is None:
            continue

        try:
            # Get user-specific feature weights for conditional model
            feature_weights_cond = p3.feature_weights(tu_data, merged, ratings, user_id)

            # Generate recommendations from both models
            conditional_recs = p3.conditional_filtering_for_monte_carlo(
                merged_train, tracks, user_id, feature_weights_cond,
                n_recommendations=max_search
            ).reset_index(drop=True)

            utility_recs = p3.utility_based_recommender_for_monte_carlo(
                merged_train, tracks, ratings, user_id, FEATURE_WEIGHTS_UTILITY,
                n_recommendations=max_search
            ).reset_index(drop=True)

        except Exception as e:
            # Skip this round if there's an error
            continue

        # Compute metrics for both models
        for model_name, recs in [('conditional', conditional_recs), ('utility', utility_recs)]:
            # 1. Hit@k
            hit = compute_hit_at_k(recs, test_songs, k)
            results[model_name]['hit_at_k'].append(hit)

            # 2. Average rating (only if there's a hit)
            avg_rating = compute_average_rating(recs, test_songs, k)
            if avg_rating is not None:
                results[model_name]['avg_ratings'].append(avg_rating)

            # 3. Time to 5-star (only if test set contains 5-star songs)
            time_to_5 = compute_time_to_five_star(recs, test_songs, max_search)
            if time_to_5 is not None:
                results[model_name]['time_to_5star'].append(time_to_5)

        successful_rounds += 1
    return results


def compute_confidence_interval(data, confidence=0.95):
    """
    Compute mean and 95% confidence interval for a list of values.
    Uses manual t-distribution approximation.
    """
    if len(data) == 0:
        return 0, (0, 0)

    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)  # Sample standard deviation
    sem = std / np.sqrt(n)  # Standard error of the mean

    # Critical value for 95% CI (t-distribution approximation)
    if n > 30:
        t_critical = 1.96
    elif n > 20:
        t_critical = 2.086
    elif n > 10:
        t_critical = 2.228
    else:
        t_critical = 2.571

    margin_of_error = t_critical * sem
    ci = (mean - margin_of_error, mean + margin_of_error)

    return mean, ci


def compare_models(results_model_a, results_model_b, metric_name):
    """
    Compare two models on a specific metric by computing means and differences.
    """
    data_a = np.array(results_model_a)
    data_b = np.array(results_model_b)

    if len(data_a) == 0 or len(data_b) == 0:
        return "Insufficient data for comparison (one or both lists are empty)."
    mean_a = np.mean(data_a)
    mean_b = np.mean(data_b)
    mean_diff = mean_a - mean_b
    ci_info = ""
    lower = 0
    upper = 0
    if len(data_a) != len(data_b): # for average time ratings use welch's t-test
        sem_a = np.std(data_a, ddof=1) / np.sqrt(len(data_a)) if len(data_a) > 1 else 0
        sem_b = np.std(data_b, ddof=1) / np.sqrt(len(data_b)) if len(data_b) > 1 else 0

        se_diff = np.sqrt(sem_a ** 2 + sem_b ** 2)

        margin_of_error = 1.96 * se_diff
        lower = mean_diff - margin_of_error
        upper = mean_diff + margin_of_error

        significant = not (lower <= 0 <= upper)

    elif "time" in metric_name.lower() or "time-to-5" in metric_name.lower():
        u , p = p2.mann_whitney_manual(data_a, data_b)
        if p < 0.05:
            significant = True
        else:
            significant = False
        ci_info = f"p-value: {p:.4f}"

    else:
        diff = data_a - data_b
        mean_diff, ci = compute_confidence_interval(diff)
        lower, upper = ci
        significant = not (lower <= 0 <= upper)

    if not significant:
        return "Statistically insignificant"

    if "time" in metric_name.lower() or "time-to-5" in metric_name.lower():
        if mean_diff < 0:
            return f"Conditional is better: finds 5-star songs {abs(mean_diff):.2f} recommendations faster ({ci_info})."
        else:
            return f"Utility is better: finds 5-star songs {mean_diff:.2f} recommendations faster ({ci_info})."

    elif "hit" in metric_name.lower():
        if mean_diff > 0:
            return f"Conditional is better: outperforms by {mean_diff * 100:.2f}% (95% CI of diff: [{lower * 100:.2f}%, {upper * 100:.2f}%])."
        else:
            return f"Utility is better: outperforms by {abs(mean_diff) * 100:.2f}% (95% CI of diff: [{abs(upper) * 100:.2f}%, {abs(lower * 100):.2f}%])."

    else:
        if mean_diff > 0:
            return f"Conditional is better: outperforms by {mean_diff:.2f} points (95% CI of diff: [{lower:.2f}, {upper:.2f}])."
        else:
            return f"Utility is better: outperforms by {abs(mean_diff):.2f} points (95% CI of diff: [{abs(upper):.2f}, {abs(lower):.2f}])."


def plot_results(results, k=5):
    """
    Create comprehensive visualization of Monte Carlo results.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Monte Carlo Evaluation: Conditional vs Utility-Based Recommenders',
                 fontsize=16, fontweight='bold', y=0.995)

    models = ['Conditional', 'Utility']
    colors = ['#2E86AB', '#A23B72']  # Blue and Purple


    # 1. Hit@k
    hit_means = []
    hit_cis = []
    for model_key in ['conditional', 'utility']:
        mean, ci = compute_confidence_interval(results[model_key]['hit_at_k'])
        hit_means.append(mean)
        hit_cis.append(ci)

    axes[0, 0].bar(models, hit_means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 0].errorbar(models, hit_means,
                        yerr=[[hit_means[i] - hit_cis[i][0] for i in range(2)],
                              [hit_cis[i][1] - hit_means[i] for i in range(2)]],
                        fmt='none', color='black', capsize=8, linewidth=2)
    axes[0, 0].set_title(f'Hit@{k} Rate (Higher is Better)', fontsize=13, fontweight='bold', pad=10)
    axes[0, 0].set_ylabel('Hit Rate', fontsize=11)
    axes[0, 0].set_ylim(0, max(hit_means) * 1.3 if max(hit_means) > 0 else 1)
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(hit_means):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. Average Rating
    rating_means = []
    rating_cis = []
    for model_key in ['conditional', 'utility']:
        mean, ci = compute_confidence_interval(results[model_key]['avg_ratings'])
        rating_means.append(mean)
        rating_cis.append(ci)

    axes[0, 1].bar(models, rating_means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 1].errorbar(models, rating_means,
                        yerr=[[rating_means[i] - rating_cis[i][0] for i in range(2)],
                              [rating_cis[i][1] - rating_means[i] for i in range(2)]],
                        fmt='none', color='black', capsize=8, linewidth=2)
    axes[0, 1].set_title('Average Rating of Hits (Higher is Better)', fontsize=13, fontweight='bold', pad=10)
    axes[0, 1].set_ylabel('Rating (1-5)', fontsize=11)
    axes[0, 1].set_ylim(0, 5.5)
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(rating_means):
        axes[0, 1].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. Time-to-5 star
    time_means = []
    time_cis = []
    for model_key in ['conditional', 'utility']:
        mean, ci = compute_confidence_interval(results[model_key]['time_to_5star'])
        time_means.append(mean)
        time_cis.append(ci)

    axes[0, 2].bar(models, time_means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0, 2].errorbar(models, time_means,
                        yerr=[[time_means[i] - time_cis[i][0] for i in range(2)],
                              [time_cis[i][1] - time_means[i] for i in range(2)]],
                        fmt='none', color='black', capsize=8, linewidth=2)
    axes[0, 2].set_title('Time to Find 5 star Song (Lower is Better)', fontsize=13, fontweight='bold', pad=10)
    axes[0, 2].set_ylabel('# Recommendations', fontsize=11)
    axes[0, 2].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(time_means):
        axes[0, 2].text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 4. Hit@k Distribution
    axes[1, 0].hist([results['conditional']['hit_at_k'],
                     results['utility']['hit_at_k']],
                    label=models, color=colors, alpha=0.6, bins=[0, 0.5, 1],
                    edgecolor='black', linewidth=1.5)
    axes[1, 0].set_title(f'Distribution of Hit@{k}', fontsize=13, fontweight='bold', pad=10)
    axes[1, 0].set_xlabel('Hit (0 = No Hit, 1 = Hit)', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3, linestyle='--')

    # 5. Average Rating Distribution
    if results['conditional']['avg_ratings'] and results['utility']['avg_ratings']:
        axes[1, 1].hist([results['conditional']['avg_ratings'],
                         results['utility']['avg_ratings']],
                        label=models, color=colors, alpha=0.6, bins=20,
                        edgecolor='black', linewidth=1)
        axes[1, 1].set_title('Distribution of Average Ratings', fontsize=13, fontweight='bold', pad=10)
        axes[1, 1].set_xlabel('Rating', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(alpha=0.3, linestyle='--')

    # 6. Time-to-5 star Distribution
    if results['conditional']['time_to_5star'] and results['utility']['time_to_5star']:
        axes[1, 2].hist([results['conditional']['time_to_5star'],
                         results['utility']['time_to_5star']],
                        label=models, color=colors, alpha=0.6, bins=30,
                        edgecolor='black', linewidth=1)
        # Add mean lines
        axes[1, 2].axvline(np.mean(results['conditional']['time_to_5star']),
                           color=colors[0], linestyle='--', linewidth=2.5, alpha=0.9)
        axes[1, 2].axvline(np.mean(results['utility']['time_to_5star']),
                           color=colors[1], linestyle='--', linewidth=2.5, alpha=0.9)
        axes[1, 2].set_title('Distribution of Time to 5star (Lower is Better)', fontsize=13, fontweight='bold', pad=10)
        axes[1, 2].set_xlabel('# Recommendations', fontsize=11)
        axes[1, 2].set_ylabel('Frequency', fontsize=11)
        axes[1, 2].legend(fontsize=10)
        axes[1, 2].grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig('monte_carlo_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_report(results, k=5):
    """
    Print detailed evaluation report with confidence intervals.
    """
    metrics_info = [
        ('Hit@k', 'hit_at_k'),
        ('Average Rating', 'avg_ratings'),
        ('Time-to-5 star', 'time_to_5star')
    ]

    for metric_name, metric_key in metrics_info:
        print(f"\n{metric_name}:")

        # Print stats for each model
        for model_name, model_key in [('Conditional Filtering', 'conditional'),
                                      ('Utility-Based Sampling', 'utility')]:
            data = results[model_key][metric_key]
            mean, ci = compute_confidence_interval(data)

            print(f"\n  {model_name}:")
            print(f"    Sample size: {len(data)}")
            print(f"    Mean: {mean:.4f}")
            print(f"    95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")


        # Compare models
        comparison = compare_models(
            results['conditional'][metric_key],
            results['utility'][metric_key],
            metric_name
        )

        print(f"\n Model Comparison:")
        print(f"    {comparison}")




def main():
    """
    Main function to run Monte Carlo evaluation.
    """

    # Load data
    tracks = p1.load_tracks("../data/tracks.csv")
    ratings = p1.load_ratings("../data/ratings.csv")
    merged = p1.merge_ratings(tracks, ratings)

    # Run Monte Carlo simulation
    start_time = time.time()

    results = monte_carlo_simulation(
        merged, tracks, ratings,
        n_simulations=3000,  # Use 1000-5000 for final submission
        k=5,
        max_search=100
    )

    elapsed = time.time() - start_time
    print(f"\n✓ Simulation completed in {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")

    # Print detailed report
    print_report(results, k=5)

    # Create visualizations
    plot_results(results, k=5)


if __name__ == "__main__":
    main()
