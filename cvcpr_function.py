import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances, mean_absolute_error, mean_squared_error
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils.metric import distance_metric, type_metric
from caserec.recommenders.rating_prediction.userknn import UserKNN
from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
import os

# ========================
# STEP 1: CLUSTERING
# ========================

def cluster_variable(data, group_col, value_col, n_clusters=2, dist_metric='euclidean'):
    grouped = data.groupby(group_col)[value_col].mean().reset_index()
    X = grouped[[value_col]].values
    distance_matrix = pairwise_distances(X, metric=dist_metric)
    init_medoids = list(np.linspace(0, len(grouped)-1, n_clusters, dtype=int))

    metric = distance_metric(type_metric.USER_DEFINED, func=lambda a, b: distance_matrix[a][b])
    kmed = kmedoids(distance_matrix, init_medoids, metric=metric, data_type='distance_matrix')
    kmed.process()

    clusters = kmed.get_clusters()
    cluster_dict = {}
    for i, cluster in enumerate(clusters):
        entities = grouped.loc[cluster, group_col].tolist()
        stats = data[data[group_col].isin(entities)][value_col].describe()
        cluster_dict[f'cluster_{i+1}'] = {'Entities': entities, 'Stats': stats}
    return cluster_dict

# ========================
# STEP 2: FUZZY LABELING
# ========================

def fuzzy_label_clusters(cluster_dict, n_clusters, delta=0.6, alpha=1.0):
    centroids = [info['Stats']['mean'] for info in cluster_dict.values()]
    mu, sigma = np.mean(centroids), np.std(centroids)
    label_map = {}

    def sigmoid(x): return 1 / (1 + np.exp(-alpha * (x - mu)))

    for cluster_id, info in cluster_dict.items():
        x = info['Stats']['mean']
        if n_clusters == 2:
            score = sigmoid(x)
            label = 'HIGH' if score >= 0.5 else 'LOW'
            conf = abs(score - 0.5) * 2
        else:
            if x < mu - sigma / 2:
                label, conf = 'LOW', 1
            elif x > mu + sigma / 2:
                label, conf = 'HIGH', 1
            else:
                label, conf = 'MODERATE', 1
        if conf < delta:
            label += "-UNCERTAIN"
        label_map[cluster_id] = label
    return label_map

# ========================
# STEP 3: SCENARIO MAPPING
# ========================

def calculate_co_occurrence(cluster_dicts, m):
    co_occurrence = {}
    variables = sorted(cluster_dicts.keys())
    for i, vi in enumerate(variables):
        for j in range(i + 1, len(variables)):
            vj = variables[j]
            for ci, info_i in cluster_dicts[vi].items():
                for cj, info_j in cluster_dicts[vj].items():
                    overlap = len(set(info_i['Entities']) & set(info_j['Entities']))
                    co_occurrence[(vi, ci, vj, cj)] = overlap / m
    return co_occurrence

def find_scenarios(cluster_dicts, co_occurrence, tau):
    variables = sorted(cluster_dicts.keys())
    scenarios = []

    def dfs(path, var_idx):
        if var_idx == len(variables):
            scenarios.append(list(path))
            return
        var = variables[var_idx]
        for cid in cluster_dicts[var]:
            if all(co_occurrence.get((variables[i], path[i], var, cid), 0) >= tau for i in range(len(path))):
                dfs(path + [cid], var_idx + 1)

    dfs([], 0)
    return scenarios

# ========================
# STEP 4: CVCPR RECOMMENDATION
# ========================

def recommend_value(scenarios, scenario_vectors, user_vector, target_index, Y_values):
    X = np.delete(scenario_vectors, target_index, axis=1)
    user_vec = np.delete(user_vector, target_index)
    dists = pairwise_distances([user_vec], X)[0]
    closest_idxs = np.where(dists == dists.min())[0]
    best_idx = max(closest_idxs, key=lambda i: Y_values[i])
    return scenario_vectors[best_idx][target_index], Y_values[best_idx]

# ========================
# BASELINE: LightFM
# ========================

from lightfm import LightFM
from scipy.sparse import csr_matrix

def compare_with_lightfm(train_df, test_df, controllable_col, loss_fn='warp', epochs=30):
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df['user'] = train_df.index
    test_df['user'] = test_df.index

    user_ids = train_df['user'].unique()
    item_ids = [controllable_col]

    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {controllable_col: 0}

    n_users, n_items = len(user_map), 1
    train_matrix = csr_matrix((n_users, n_items), dtype=np.float32)

    for _, row in train_df.iterrows():
        u = user_map[row['user']]
        train_matrix[u, 0] = row[controllable_col]

    test_matrix = csr_matrix((n_users, n_items), dtype=np.float32)
    for _, row in test_df.iterrows():
        if row['user'] in user_map:
            u = user_map[row['user']]
            test_matrix[u, 0] = row[controllable_col]

    model = LightFM(loss=loss_fn)
    model.fit(train_matrix, epochs=epochs, verbose=False)

    predictions, actuals, rows = [], [], []
    for _, row in test_df.iterrows():
        if row['user'] in user_map:
            u = user_map[row['user']]
            pred = model.predict(np.array([u]), np.array([0]))[0]

            min_val = train_df[controllable_col].min()
            max_val = train_df[controllable_col].max()
            scaled = np.clip(pred, 0, 1)
            scaled = min_val + scaled * (max_val - min_val)

            predictions.append(scaled)
            actuals.append(row[controllable_col])
            rows.append({'Entity': row['user'], 'True': row[controllable_col], 'Predicted': round(scaled, 2)})

    results_df = pd.DataFrame(rows)
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    mape = np.mean(np.abs(np.array(actuals) - np.array(predictions)) / np.maximum(np.array(actuals), 1e-6))
    acc = 100 * (1 - mape)

    return results_df, {'MAE': mae, 'MSE': mse, 'MAPE': mape, 'Accuracy (%)': acc}

# ========================
# BASELINE: Surprise SVD
# ========================

from surprise import Dataset, Reader, SVD

def compare_with_surprise_svd(train_df, test_df, controllable_col):
    train = train_df.copy()
    test = test_df.copy()

    train['user'] = train.index
    test['user'] = test.index
    train['item'] = controllable_col
    test['item'] = controllable_col

    train = train.rename(columns={controllable_col: 'rating'})[['user', 'item', 'rating']]
    test = test.rename(columns={controllable_col: 'rating'})[['user', 'item', 'rating']]

    reader = Reader(rating_scale=(train['rating'].min(), train['rating'].max()))
    data = Dataset.load_from_df(train, reader)
    trainset = data.build_full_trainset()

    model = SVD()
    model.fit(trainset)

    rows, y_true, y_pred = [], [], []
    for uid, iid, true_r in test.itertuples(index=False):
        pred = model.predict(uid, iid)
        rows.append({'Entity': uid, 'True': true_r, 'Predicted': round(pred.est, 2)})
        y_true.append(true_r)
        y_pred.append(pred.est)

    results_df = pd.DataFrame(rows)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs(np.array(y_true) - np.array(y_pred)) / np.maximum(np.array(y_true), 1e-6))
    acc = 100 * (1 - mape)

    return results_df, {'MAE': mae, 'MSE': mse, 'MAPE': mape, 'Accuracy (%)': acc}


# ============================
# BASELINE: Case Recommender
# ============================



def compare_with_case_recommender(train_df, test_df, controllable_col, model_type='UserKNN'):
    """
    Evaluate Case Recommender models: UserKNN or MatrixFactorization.

    Parameters:
    - train_df: DataFrame for training
    - test_df: DataFrame for testing
    - controllable_col: the controllable variable (e.g., 'c4')
    - model_type: 'UserKNN' or 'MF'

    Returns:
    - results_df: prediction DataFrame
    - metrics: dict of MAE, MSE, MAPE, Accuracy
    """
    train = train_df.copy()
    test = test_df.copy()

    # Map to numeric user IDs
    if not pd.api.types.is_numeric_dtype(train.index):
        user_map = {u: i for i, u in enumerate(train.index.unique())}
        rev_map = {v: k for k, v in user_map.items()}
        train['user_id'] = train.index.map(user_map)
        test['user_id'] = test.index.map(user_map)
    else:
        train['user_id'] = train.index
        test['user_id'] = test.index
        user_map = {u: u for u in train.index.unique()}
        rev_map = user_map

    train = train.dropna(subset=[controllable_col])
    train['item_id'] = 1
    train = train.rename(columns={controllable_col: 'rating'})[['user_id', 'item_id', 'rating']]
    test = test.dropna(subset=[controllable_col])
    test['item_id'] = 1
    test = test.rename(columns={controllable_col: 'rating'})[['user_id', 'item_id', 'rating']]
    test = test[test['user_id'].isin(train['user_id'])]

    train_file = 'train_temp_case.txt'
    test_file = 'test_temp_case.txt'
    train.to_csv(train_file, sep='\t', header=False, index=False)
    test.to_csv(test_file, sep='\t', header=False, index=False)

    predictions = []
    try:
        if model_type.lower() == 'userknn':
            model = UserKNN(train_file, test_file, similarity_metric='correlation', k_neighbors=min(5, len(train['user_id'].unique())-1))
        else:
            n_factors = min(5, len(train['user_id'].unique())-1)
            model = MatrixFactorization(train_file, test_file, n_factors=n_factors, learn_rate=0.01, epochs=30)
        model.compute()
        predictions = model.predictions
    except Exception as e:
        print(f"⚠️ Error in {model_type}: {e}")
        predictions = []

    os.remove(train_file)
    os.remove(test_file)

    if not predictions:
        return None, {}

    if len(predictions[0]) == 4:
        pred_df = pd.DataFrame(predictions, columns=['user_id', 'item_id', 'true', 'pred'])
    elif len(predictions[0]) == 3:
        pred_df = pd.DataFrame(predictions, columns=['user_id', 'item_id', 'pred'])
        pred_df = pd.merge(pred_df, test, on=['user_id', 'item_id'], how='inner')
        pred_df = pred_df.rename(columns={'rating': 'true'})
    else:
        print(f"⚠️ Unexpected prediction format: {predictions[0]}")
        return None, {}

    pred_df['user'] = pred_df['user_id'].map(rev_map)
    pred_df['true'] = pred_df['true'].astype(float)
    pred_df['pred'] = pred_df['pred'].astype(float)

    mae = mean_absolute_error(pred_df['true'], pred_df['pred'])
    mse = mean_squared_error(pred_df['true'], pred_df['pred'])
    mape = (np.abs(pred_df['true'] - pred_df['pred']) / np.maximum(np.abs(pred_df['true']), 1e-6)).mean() * 100
    accuracy = 100 - mape

    results_df = pred_df[['user', 'true', 'pred']]
    results_df.columns = ['Entity', 'True', 'Predicted']
    results_df['Predicted'] = results_df['Predicted'].round(2)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'MAPE': mape,
        'Accuracy (%)': accuracy
    }

    return results_df, metrics