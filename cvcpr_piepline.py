import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import pairwise_distances
from pyclustering.cluster.kmedoids import kmedoids

# ==========================================================
# CVCPR - FULL DATA PREPROCESSING AND CLUSTERING PIPELINE
# ==========================================================

# ---------------------------
# CONFIGURATIONS
# ---------------------------

# Data paths (relative for reproducibility)
DATA_FOLDER = 'data'
TRAIN_FILENAME = 'training_data.csv'
LABEL_FILENAME = 'cvcpr_labels.json'

# Load flattened dataset (training file)
train_path = os.path.join(DATA_FOLDER, TRAIN_FILENAME)
df = pd.read_csv(train_path)

# Load label configuration (clustering setup)
label_path = os.path.join(DATA_FOLDER, LABEL_FILENAME)
with open(label_path, "r") as f:
    label_config = json.load(f)

# ---------------------------
# DATA PREPARATION
# ---------------------------

# Extract contextual variable names
contextual_vars = [col for col in df.columns if col not in ['Province', 'Farm_ID', 'Y_GrainYield']]

# Build data dictionary: {variable: Province x Farms matrix}
data_dict = {}
for var in contextual_vars + ['Y_GrainYield']:
    pivot = df.pivot_table(index='Province', columns='Farm_ID', values=var, aggfunc='first')
    data_dict[var] = pivot

# Retrieve province names
province_names = data_dict[list(data_dict.keys())[0]].index.tolist()

# ---------------------------
# HELPER: Cluster Dict Builder
# ---------------------------
def build_cluster_dict_for_variable(flat_data, variable_colname, province_names, clusters, var_name):
    cluster_dict = {}
    for i, cluster_indices in enumerate(clusters):
        cluster_id = f'{var_name}_cluster_{i+1}'
        provinces_in_cluster = [province_names[idx] for idx in cluster_indices]
        cluster_data = flat_data[flat_data['Province'].isin(provinces_in_cluster)]
        stats = cluster_data[variable_colname].describe()
        cluster_dict[cluster_id] = {
            'Provinces': provinces_in_cluster,
            'Stats': stats.to_dict()
        }
    return cluster_dict

# ---------------------------
# CLUSTERING FUNCTION
# ---------------------------
def cluster_variables(data_dict, flat_data, label_config, random_state=42):
    cluster_results = {}
    cluster_dicts = {}

    for config_key in label_config.keys():
        var_index = int(config_key[1:])  # extract variable index: e.g. c1 -> 1
        var_key = list(data_dict.keys())[var_index - 1]
        matrix = data_dict[var_key]

        print(f"\n📊 Clustering for variable: {var_key}")

        n_clusters = label_config[config_key]["n_clusters"]
        print(f"  Number of clusters: {n_clusters}")

        # Compute province-level average
        province_averages = matrix.mean(axis=1, skipna=True).values.reshape(-1, 1)
        dist_matrix = pairwise_distances(province_averages, metric='euclidean')

        # Initialize K-Medoids
        np.random.seed(random_state)
        init_medoids = np.random.choice(range(len(dist_matrix)), n_clusters, replace=False).tolist()

        kmedoids_instance = kmedoids(dist_matrix, init_medoids, data_type='distance_matrix')
        kmedoids_instance.process()
        clusters = kmedoids_instance.get_clusters()
        medoids = kmedoids_instance.get_medoids()

        # Build assignments
        cluster_assignment = np.zeros(len(dist_matrix), dtype=int)
        for cluster_id, cluster_indices in enumerate(clusters):
            for idx in cluster_indices:
                cluster_assignment[idx] = cluster_id

        # Save results
        cluster_results[config_key] = {
            "assignments": cluster_assignment.tolist(),
            "centroids": medoids
        }

        # Build cluster dictionary for inspection
        cluster_dict = build_cluster_dict_for_variable(
            flat_data=flat_data,
            variable_colname=var_key,
            province_names=province_names,
            clusters=clusters,
            var_name=config_key
        )
        cluster_dicts[config_key] = cluster_dict

        # Print summary per cluster
        for i, cluster_indices in enumerate(clusters):
            provinces_in_cluster = [province_names[idx] for idx in cluster_indices]
            centroid_idx = medoids[i]
            centroid_value = province_averages[centroid_idx][0]
            print(f"  Cluster {i+1}: centroid={centroid_value:.3f} — Provinces: {provinces_in_cluster}")

    return cluster_results, cluster_dicts

# ---------------------------
# 🔥 RUN CLUSTERING PIPELINE
# ---------------------------
cluster_results, cluster_dicts = cluster_variables(data_dict, df, label_config)

# ==========================================================
# SCENARIO MAPPING + PERFORMANCE ESTIMATION MODULE
# ==========================================================

from sklearn.metrics.pairwise import euclidean_distances

# ----------------------------------
# 1️⃣ Co-occurrence matrix builder
# ----------------------------------
def calculate_co_occurrence_matrix(cluster_dicts, m):
    co_occurrence = {}
    variables = list(cluster_dicts.keys())

    for i_idx, var_i in enumerate(variables):
        i = int(var_i.replace('c', ''))
        for j_idx, var_j in enumerate(variables):
            if i_idx >= j_idx:
                continue
            j = int(var_j.replace('c', ''))

            for cluster_i_id, cluster_i_info in cluster_dicts[var_i].items():
                p = int(cluster_i_id.split('_')[-1])
                provinces_i = set(cluster_i_info['Provinces'])

                for cluster_j_id, cluster_j_info in cluster_dicts[var_j].items():
                    q = int(cluster_j_id.split('_')[-1])
                    provinces_j = set(cluster_j_info['Provinces'])

                    key = (i, p, j, q)
                    co_occurrence[key] = len(provinces_i & provinces_j) / m
    return co_occurrence

# ----------------------------------
# 2️⃣ DFS-based scenario builder
# ----------------------------------
def find_complete_scenarios(cluster_dicts, co_occurrence, tau):
    scenarios = []
    variables = sorted(cluster_dicts.keys(), key=lambda x: int(x.replace('c', '')))
    n = len(variables)
    var_indices = {var: int(var.replace('c', '')) for var in variables}

    def is_correlated(path, path_vars, next_var, next_cluster):
        next_var_idx = var_indices[next_var]
        next_cluster_idx = int(next_cluster.split('_')[-1])
        for prev_var, prev_cluster in zip(path_vars, path):
            prev_var_idx = var_indices[prev_var]
            prev_cluster_idx = int(prev_cluster.split('_')[-1])

            if prev_var_idx < next_var_idx:
                i, p, j, q = prev_var_idx, prev_cluster_idx, next_var_idx, next_cluster_idx
            else:
                i, p, j, q = next_var_idx, next_cluster_idx, prev_var_idx, prev_cluster_idx

            if co_occurrence.get((i, p, j, q), 0) < tau:
                return False
        return True

    def dfs(partial_path, path_vars, var_idx):
        if var_idx >= n:
            scenarios.append(tuple(partial_path))
            return
        current_var = variables[var_idx]
        for cluster_id in cluster_dicts[current_var]:
            if not partial_path or is_correlated(partial_path, path_vars, current_var, cluster_id):
                partial_path.append(cluster_id)
                path_vars.append(current_var)
                dfs(partial_path, path_vars, var_idx + 1)
                partial_path.pop()
                path_vars.pop()

    dfs([], [], 0)
    return scenarios

# ----------------------------------
# 3️⃣ Full scenario mapping pipeline
# ----------------------------------
def run_scenario_mapping(cluster_dicts, tau_values, m):
    results = {}
    co_occurrence = calculate_co_occurrence_matrix(cluster_dicts, m)

    for tau in tau_values:
        print(f"\n🧪 Tau = {tau:.2f}")
        scenarios = find_complete_scenarios(cluster_dicts, co_occurrence, tau)
        print(f"✅ {len(scenarios)} complete scenarios found")
        results[tau] = scenarios

    return results

# ----------------------------------
# 4️⃣ Build scenario vectors (centroids)
# ----------------------------------
def build_scenario_vectors(scenarios, cluster_dicts):
    scenario_vectors = []
    for scenario in scenarios:
        vector = []
        for var_index, cluster_id in enumerate(scenario, start=1):
            var_key = f'c{var_index}'
            cluster_stats = cluster_dicts[var_key][cluster_id]['Stats']
            cluster_mean = cluster_stats['mean']
            vector.append(cluster_mean)
        scenario_vectors.append(tuple(vector))
    return scenario_vectors

# ----------------------------------
# 5️⃣ Estimate Y for each scenario
# ----------------------------------
def estimate_scenario_performance(data_dict, scenario_vectors):
    feature_vectors = []
    provinces = []

    for province in data_dict['c1_Soil_pH'].index:
        row = []
        for var in list(data_dict.keys())[:-1]:  # Exclude Y_GrainYield
            province_vector = data_dict[var].loc[province].values
            province_mean = np.nanmean(province_vector)
            row.append(province_mean)
        feature_vectors.append(row)
        provinces.append(province)

    X_real = np.array(feature_vectors)
    Y_real = np.array([np.nanmean(data_dict['Y_GrainYield'].loc[province].values) for province in provinces])

    scenario_estimates = []
    for scenario_vec in scenario_vectors:
        scenario_array = np.array(scenario_vec).reshape(1, -1)
        dists = euclidean_distances(scenario_array, X_real)
        nearest_index = np.argmin(dists)
        estimated_y = Y_real[nearest_index]
        scenario_estimates.append(estimated_y)

    return scenario_estimates

# ==========================================================
# 🔥 EXECUTION FOR YOUR NEW DATA 🔥
# ==========================================================

# You can adjust tau list
tau_values = [0.1]
m = 50  # update m depending on your dataset

results = run_scenario_mapping(cluster_dicts, tau_values, m)
scenarios = results[0.1]

# Convert scenarios into vectors
scenario_vectors = build_scenario_vectors(scenarios, cluster_dicts)

# Estimate Y for each scenario
scenario_estimates = estimate_scenario_performance(data_dict, scenario_vectors)

# Export scenario table
scenario_df = pd.DataFrame(scenario_vectors, columns=[f'c{i}' for i in range(1, len(scenario_vectors[0]) + 1)])
scenario_df['Estimated_Y'] = scenario_estimates
scenario_df.to_excel(os.path.join(DATA_FOLDER, "estimated_scenarios_with_Y.xlsx"), index=False)
print("✅ Scenario estimations saved to 'estimated_scenarios_with_Y.xlsx'")

# Print few examples
for i, (scenario, y) in enumerate(zip(scenario_vectors, scenario_estimates), 1):
    print(f"Scenario {i}:")
    print(f"  Cluster Means → {scenario}")
    print(f"  Estimated Y   → {y:.2f}")
    print("-" * 60)

# Save scenario vectors & estimates for later evaluation
np.save(os.path.join(DATA_FOLDER, 'scenario_vectors.npy'), scenario_vectors)
np.save(os.path.join(DATA_FOLDER, 'scenario_estimates.npy'), scenario_estimates)
