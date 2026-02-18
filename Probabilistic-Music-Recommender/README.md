# 🎵 Tunes Duel: Probabilistic Music Recommendation System

**Course:** CmpE 343: Introduction to Probability and Statistics (Fall 2025)  
**Team:** Team Rocket  

---

## 📌 Project Overview

This project simulates and analyzes a music recommendation system using a dataset of songs (`tracks.csv`) and user rating sessions (`ratings.csv`). The goal is to apply probability and statistics concepts—such as **conditional probability**, **Bayesian inference**, and **distribution modeling**—to predict user preferences and design an effective recommender system.

Project is divided into four main parts:

1. **Conditional Probability Modeling**  
   Estimating the likelihood of a 5-star rating based on track features.
2. **User Variability Modeling**  
   Analyzing user patience ($T_u$, time-to-favorite) using Geometric and Beta-Geometric distributions.
3. **Recommender Design**  
   Creating a utility-based recommendation engine.
4. **Monte Carlo Evaluation**  
   Simulating user sessions to evaluate model performance.

---

## Project Structure

```
├── README.md                    # This file
├── src/                         # Source code
│   ├── part1.py                 # Conditional Probability Modeling
│   ├── part2.py                 # User Variability Modeling  
│   ├── part3.py                 # Recommender Design
│   ├── part4.py                 # Monte Carlo Evaluation
│   └── recommender.py           # Main recommender for Tune Duel
└── data/                        # Dataset files (add your data here)
    ├── tracks.csv              # Song metadata (tracks)
    ├── ratings.csv             # Syntetic user interactions
    ├── user_ratings.csv        # User interactions
    ├── ratings_enes.csv             
    ├── ratings_ustun.csv             
    ├── ratings_yusuf.csv             
                
```

## 🚀 Features & Methods

### Part 1: Conditional Probability Modeling

We calculate smoothed conditional probabilities to determine what makes a *"favorite"* song.

- **Forward Probability** $P(5^* \mid \text{feature})$  
  Uses **Laplace Smoothing** to estimate the probability of a user rating a song 5 stars given a specific feature (e.g., explicit content, artist).

- **Inverse Probability** $P(\text{feature} \mid 5^*)$  
  Applies **Bayes' Rule** to understand the characteristics of top-rated songs.

- **Group Analysis**  
  Aggregates individual team member preferences to create a group profile.

---
### Part 2: User Variability Modeling

We model user patience ($T_u$ = number of recommendations until the first 5-star rating).

**Key Features**
1.  **Data Processing**: Calculates $T_u$ for every user session, filtering out users who never found a favorite song.

2.  **Probabilistic Modeling**:
    * **Geometric Model**: Assumes all users share a constant probability $p$ of finding a favorite.
    * **Beta-Geometric Model**: Captures user heterogeneity by assuming success probabilities follow a $\text{Beta}(\alpha, \beta)$ distribution. Parameters are estimated using **Maximum Likelihood Estimation (MLE)** via coordinate descent.

3.  **Visualization**: Generates a plot comparing the observed data histogram against the fitted Geometric and Beta-Geometric probability mass functions.

4.  **Hypothesis Testing**: Performs a **Mann-Whitney U test** to determine if there is a significant difference in patience between users with "Popular" music tastes versus "Niche" music tastes.

**Expected Output:**

```text
Number of valid users found: 291
Average rounds to find a favorite: 3.69

[Geometric Model] Estimated p: 0.2707
[Beta-Geometric] Estimated alpha: 3.0530, beta: 5.6068

plotting graph (it should appear in screen)

[Hypothesis Test]
Group A (Popular Tastes): n=156, mean Tu=3.69
Group B (Niche Tastes): n=135, mean Tu=3.70

[Hypothesis Test] Mann-Whitney U: 10015.0, p-value: 0.47189
No significant difference found.

Finished executing
```
---

### Part 3: Recommender Design

**Conditional Filtered Recommender** 
  A probabilistic approach that ranks tracks by calculating the conditional likelihood of a user enjoying a song based on their specific preference history.
  
- **Mechanism:** 
  Instead of a global search, this model filters the dataset to identify tracks that match the user's high-probability attributes (e.g., specific artists or genres that have historically yielded 5-star ratings).
  
- **Weights:** 
  Weights are derived directly from user interaction data. These parameters are correlated with the user's past rating decisions, ensuring the model adapts to the user's unique taste profile.

**Utility-Based Recommender**  
  A vectorized implementation that calculates a utility score for tracks based on weighted feature importance.

- **Weights**  
  Features such as:
  - `primary_artist_name`
  - `ab_genre`
  - `track_popularity`  
  are weighted to predict user enjoyment.

- **Mechanism**  
  Combines global priors with user-specific history to handle:
  - cold-start problems  
  - sparse data scenarios


---

### Part 4: Monte Carlo Evaluation

We implement a stochastic simulation engine to robustly evaluate and compare the performance of the **Conditional Filtering** and **Utility-Based** recommendation models. Instead of a static train-test split, this module performs repeated random sampling to measure how models perform across a diverse range of user behaviors.

**Methodology**
- **Monte Carlo Simulation:** The system executes a loop of $N$ iterations (default: 3000 rounds). In each round:
  1. A random user is selected from the dataset.
  2. Their data is dynamically split into a training set (80%) and a hidden test set (20%).
  3. Both models generate recommendations based on the training data.
  4. Performance is measured against the hidden test set.

**Evaluation Metrics**
1. **Hit@k:** The probability of a relevant song appearing in the top-$k$ recommendations (Accuracy).
2. **Average Rating:** The mean rating of successfully recommended songs (Quality).
3. **Time-to-5$\star$:** The number of recommendations required to find the first 5-star song (Efficiency/Patience).

**Statistical Analysis**
- **Confidence Intervals (CI):** We compute 95% Confidence Intervals for all metrics to visualize performance stability.
- **Hypothesis Testing:**
  - **Paired Differences:** Used for Hit Rate (same user, two models).
  - **Independent Differences:** Used for Average Rating (unequal sample sizes due to missed hits).
  - **Mann-Whitney U Test:** Used for *Time-to-5$\star$* to handle non-normal distributions (e.g., penalties at max search depth) and validate significant differences in search efficiency.

**Key Insight:** The simulation reveals the trade-off between **Global Popularity** (Utility Model) and **Personalized Filtering** (Conditional Model). Results show that while utility-based approaches are risky for niche users (often hitting the search limit), conditional filtering offers significantly more robust and consistent discovery.


---

## 💻 Installation & Usage

### Prerequisites

- Python 3.x  
- pandas  
- numpy  
- matplotlib (for plotting)
- time

```bash
pip install pandas numpy matplotlib
```

Usage for part 1,2,3,4:
```bash
python src/part1.py
python src/part2.py
python src/part3.py
python src/part4.py
```

Usage for recommender:
```bash
python src/recommender.py
```

## Authors
- Yusuf Tamer Akyol
- Üstün Yılmaz 
- Enes Hamza Üstün
