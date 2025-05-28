import pandas as pd
from cvcpr_function import (
    cluster_variable, fuzzy_label_clusters, calculate_co_occurrence,
    find_scenarios, recommend_value,
    compare_with_lightfm, compare_with_surprise_svd, compare_with_case_recommender
)

# Load sample dataset
df = pd.read_csv("demo_contextual_data.csv", index_col="Entity")

# Prepare training/test split
train_df = df.iloc[:4].copy()
test_df = df.iloc[4:].copy()

# --- CVCPR Pipeline ---
cluster_dicts = {}
for col in ['c1', 'c2', 'c3']:
    cluster_dicts[col] = cluster_variable(train_df.reset_index(), 'Entity', col, n_clusters=2)

co_matrix = calculate_co_occurrence(cluster_dicts, m=len(train_df))
scenarios = find_scenarios(cluster_dicts, co_matrix, tau=0.5)

# Create dummy scenario vectors and user_vector (mean encoding)
scenario_vectors = [train_df[['c1', 'c2', 'c3', 'c4']].mean().values for _ in scenarios]
user_vector = test_df[['c1', 'c2', 'c3', 'c4']].iloc[0].values
Y_values = [train_df['c4'].mean()] * len(scenario_vectors)

# Recommend
recommended_val, predicted_y = recommend_value(scenarios, scenario_vectors, user_vector, target_index=3, Y_values=Y_values)
print(f"\n🧠 CVCPR recommended value for c4: {recommended_val:.2f} | Estimated Y: {predicted_y:.2f}")

# --- LightFM ---
results_lf, metrics_lf = compare_with_lightfm(train_df, test_df, controllable_col='c4')
print("\n📊 LightFM:", metrics_lf)

# --- Surprise SVD ---
results_svd, metrics_svd = compare_with_surprise_svd(train_df, test_df, controllable_col='c4')
print("\n📊 Surprise SVD:", metrics_svd)

# --- Case Recommender UserKNN ---
results_knn, metrics_knn = compare_with_case_recommender(train_df, test_df, controllable_col='c4', model_type='UserKNN')
print("\n📊 UserKNN:", metrics_knn)

# --- Case Recommender MF ---
results_mf, metrics_mf = compare_with_case_recommender(train_df, test_df, controllable_col='c4', model_type='MF')
print("\n📊 Matrix Factorization:", metrics_mf)
