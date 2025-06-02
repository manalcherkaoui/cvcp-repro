import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Optional imports - will gracefully handle if not available
try:
    from surprise import Dataset, Reader, SVD
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print(" Surprise library not available. SVD baseline will be skipped.")

try:
    from lightfm import LightFM
    LIGHTFM_AVAILABLE = True
except ImportError:
    LIGHTFM_AVAILABLE = False
    print(" LightFM library not available. LightFM baseline will be skipped.")

try:
    from caserec.recommenders.rating_prediction.userknn import UserKNN
    from caserec.recommenders.rating_prediction.matrixfactorization import MatrixFactorization
    CASEREC_AVAILABLE = True
except ImportError:
    CASEREC_AVAILABLE = False
    print(" CaseRecommender library not available. UserKNN and UserFM baselines will be skipped.")

try:
    import xlearn as xl
    XLEARN_AVAILABLE = True
except ImportError:
    XLEARN_AVAILABLE = False
    print(" xLearn library not available. CARSFM baseline will be skipped.")

# ==========================================================
# COMPLETE CVCPR BASELINE EVALUATION PIPELINE
# ==========================================================

class CVCPRBaselineEvaluator:
    """
    Complete evaluation pipeline respecting all your exact method implementations
    """
    
    def __init__(self, data_folder='data'):
        self.data_folder = data_folder
        self.R = 10  # Controllable variable is c10
        self.tau_example = 0.1
        
        # File paths - using your exact structure
        self.train_file_path = os.path.join(data_folder, 'training_data.csv')
        self.eval_file_path = os.path.join(data_folder, 'testing_data.csv')
        
        # Column renaming dictionary
        self.rename_dict = {
            'c1_Soil_pH': 'c1',
            'c2_CaCO3': 'c2', 
            'c3_OM': 'c3',
            'c4_K2O': 'c4',
            'c5_Sand': 'c5',
            'c6_Clay': 'c6',
            'c7_Nitrogen': 'c7',
            'c8_Phosphorous': 'c8',
            'c9_Potassium': 'c9',
            'c10_SowingRate': 'c10',
            'Y_GrainYield': 'Y'
        }
        
        self.all_results = {}
        self.all_metrics = []
        
    def load_and_prepare_common_data(self):
        """Load data once for all methods"""
        print(" Loading and preparing data...")
        
        # Load evaluation data
        evaluation_df = pd.read_csv(self.eval_file_path)
        evaluation_df = evaluation_df.rename(columns=self.rename_dict)
        self.evaluation_df_grouped = evaluation_df.groupby('Province').mean(numeric_only=True).reset_index()
        
        print(f" Data loaded: {len(self.evaluation_df_grouped)} evaluation provinces")
        return True
    
    def load_cvcpr_scenarios(self):
        """Load CVCPR scenario results"""
        scenario_vectors_path = os.path.join(self.data_folder, 'scenario_vectors.npy')
        scenario_estimates_path = os.path.join(self.data_folder, 'scenario_estimates.npy')
        
        if not (os.path.exists(scenario_vectors_path) and os.path.exists(scenario_estimates_path)):
            print(" CVCPR scenario files not found. Run main pipeline first.")
            return False
        
        self.scenario_vectors = np.load(scenario_vectors_path, allow_pickle=True)
        self.scenario_estimates = np.load(scenario_estimates_path, allow_pickle=True)
        
        print(f" Loaded {len(self.scenario_vectors)} CVCPR scenarios")
        return True
    
    def calculate_metrics(self, true_values, predicted_values, method_name):
        """Calculate standard evaluation metrics"""
        mae = mean_absolute_error(true_values, predicted_values)
        mse = mean_squared_error(true_values, predicted_values)
        mape = np.mean(np.abs((np.array(true_values) - np.array(predicted_values)) / np.maximum(np.abs(np.array(true_values)), 1e-10))) * 100
        accuracy = 100 - mape
        
        return {
            'method': method_name,
            'mae': mae,
            'mse': mse,
            'mape': mape,
            'accuracy': accuracy
        }
    
    def evaluate_cvcpr_approach(self):
        """YOUR EXACT CVCPR IMPLEMENTATION"""
        print("\n🎯 Evaluating CVCPR Approach...")
        
        if not hasattr(self, 'scenario_vectors'):
            if not self.load_cvcpr_scenarios():
                return None, None
        
        evaluation_results = []
        feature_cols = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10']
        
        for idx, row in self.evaluation_df_grouped.iterrows():
            # Select all features directly
            features = row[feature_cols]
            # Remove controllable variable (c10) manually
            user_context = features.drop(f'c{self.R}')
            user_vector = user_context.values.reshape(1, -1)
            
            # Build partial scenario vectors by removing controllable variable
            partial_scenario_vectors = [
                np.array(scenario)[:self.R - 1].tolist() + np.array(scenario)[self.R:].tolist()
                for scenario in self.scenario_vectors
            ]
            partial_scenario_vectors = np.array(partial_scenario_vectors)
            
            # Compute distances
            distances = pairwise_distances(user_vector, partial_scenario_vectors, metric='euclidean')[0]
            min_distance = distances.min()
            best_indices = np.where(distances == min_distance)[0]
            best_index = max(best_indices, key=lambda i: self.scenario_estimates[i])
            
            best_scenario_vector = self.scenario_vectors[best_index]
            recommended_c10 = best_scenario_vector[self.R - 1]
            
            evaluation_results.append({
                'Province': row['Province'],
                'True_c10': row[f'c{self.R}'],
                'Recommended_c10': recommended_c10,
                'True_Y': row['Y'],
                'Estimated_Y': self.scenario_estimates[best_index],
                'Distance': min_distance
            })
        
        results_df = pd.DataFrame(evaluation_results)
        
        # Error Metrics
        mae = (results_df['Recommended_c10'] - results_df['True_c10']).abs().mean()
        mse = ((results_df['Recommended_c10'] - results_df['True_c10']) ** 2).mean()
        mape = ((results_df['Recommended_c10'] - results_df['True_c10']).abs() / results_df['True_c10']).mean()
        accuracy = 100 - (mape * 100)
        
        metrics = {
            'method': 'CVCPR',
            'mae': mae,
            'mse': mse,
            'mape': mape * 100,
            'accuracy': accuracy
        }
        
        print(f" CVCPR Results: MAE={mae:.2f}, Accuracy={accuracy:.2f}%")
        return results_df, metrics
    
    def evaluate_svd_surprise(self):
        """YOUR EXACT SVD/SURPRISE IMPLEMENTATION"""
        if not SURPRISE_AVAILABLE:
            print("\n Skipping SVD - Surprise library not available")
            return None, None
            
        print("\n🔄 Evaluating SVD (Surprise) Baseline...")
        
        # Load and prepare TRAINING DATASET
        train_df = pd.read_csv(self.train_file_path)
        
        # Aggregate training data at province level (mean of farms)
        contextual_vars = [col for col in train_df.columns if col not in ['Province', 'Farm_ID', 'Y_GrainYield']]
        train_data_dict = {}
        
        for var in contextual_vars + ['Y_GrainYield']:
            pivot = train_df.pivot_table(index='Province', columns='Farm_ID', values=var, aggfunc='first')
            train_data_dict[var] = pivot
        
        # Build province-level training dataframe
        province_names = train_data_dict['c1_Soil_pH'].index.tolist()
        province_data = []
        
        for province in province_names:
            row = []
            for var in list(train_data_dict.keys())[:-1]:  # exclude Y
                province_vector = train_data_dict[var].loc[province].values
                province_mean = np.nanmean(province_vector)
                row.append(province_mean)
            province_data.append(row)
        
        contextual_df = pd.DataFrame(province_data, columns=contextual_vars)
        contextual_df['Province'] = province_names
        contextual_df = contextual_df.reset_index(drop=True)
        contextual_df['user'] = contextual_df.index
        
        # For training: we use c10 (your new controllable variable)
        train_svd_df = contextual_df[['user', 'c10_SowingRate']].copy()
        train_svd_df['item'] = 'sowing_rate'
        train_svd_df = train_svd_df.rename(columns={'c10_SowingRate': 'rating'})
        train_svd_df = train_svd_df[['user', 'item', 'rating']]
        
        # Prepare TEST DATASET
        test_df_grouped = self.evaluation_df_grouped.copy()
        test_df_grouped['user'] = test_df_grouped.index
        
        test_svd_df = test_df_grouped[['user', 'c10']].copy()
        test_svd_df['item'] = 'sowing_rate'
        test_svd_df = test_svd_df.rename(columns={'c10': 'rating'})
        test_svd_df = test_svd_df[['user', 'item', 'rating']]
        
        # Train Surprise SVD model
        reader = Reader(rating_scale=(train_svd_df['rating'].min(), train_svd_df['rating'].max()))
        train_data = Dataset.load_from_df(train_svd_df, reader)
        trainset = train_data.build_full_trainset()
        model = SVD()
        model.fit(trainset)
        
        # Predict and Evaluate
        prediction_rows = []
        true_c10 = []
        estimated_c10 = []
        
        for uid, iid, true_r in test_svd_df.itertuples(index=False):
            pred = model.predict(uid, iid)
            prediction_rows.append({
                'Province': uid,
                'True_c10': true_r,
                'Recommended_c10': round(pred.est, 2)
            })
            true_c10.append(true_r)
            estimated_c10.append(pred.est)
        
        results_df = pd.DataFrame(prediction_rows)
        
        # Error Metrics
        mae_svd = mean_absolute_error(true_c10, estimated_c10)
        mse_svd = mean_squared_error(true_c10, estimated_c10)
        mape_svd = (np.abs(np.array(true_c10) - np.array(estimated_c10)) / np.array(true_c10)).mean()
        accuracy_svd = 100 - (mape_svd * 100)
        
        metrics = {
            'method': 'SVD_Surprise',
            'mae': mae_svd,
            'mse': mse_svd,
            'mape': mape_svd * 100,
            'accuracy': accuracy_svd
        }
        
        print(f" SVD Results: MAE={mae_svd:.2f}, Accuracy={accuracy_svd:.2f}%")
        return results_df, metrics
    
    def evaluate_lightfm(self):
        """YOUR EXACT LIGHTFM IMPLEMENTATION"""
        if not LIGHTFM_AVAILABLE:
            print("\n Skipping LightFM - LightFM library not available")
            return None, None
            
        print("\n🔄 Evaluating LightFM Baseline...")
        
        # Load and prepare TRAINING DATASET
        train_df = pd.read_csv(self.train_file_path)
        
        # Aggregate training data at province level (mean of farms)
        contextual_vars = [col for col in train_df.columns if col not in ['Province', 'Farm_ID', 'Y_GrainYield']]
        train_data_dict = {}
        
        for var in contextual_vars + ['Y_GrainYield']:
            pivot = train_df.pivot_table(index='Province', columns='Farm_ID', values=var, aggfunc='first')
            train_data_dict[var] = pivot
        
        province_names = train_data_dict['c1_Soil_pH'].index.tolist()
        province_data = []
        
        for province in province_names:
            row = []
            for var in list(train_data_dict.keys())[:-1]:
                province_vector = train_data_dict[var].loc[province].values
                province_mean = np.nanmean(province_vector)
                row.append(province_mean)
            province_data.append(row)
        
        contextual_df = pd.DataFrame(province_data, columns=contextual_vars)
        contextual_df['Province'] = province_names
        contextual_df = contextual_df.reset_index(drop=True)
        contextual_df['user'] = contextual_df.index
        
        # Build interaction matrix for training
        user_ids = contextual_df['user'].unique()
        item_ids = ['sowing_rate']
        
        user_mapping = {u: i for i, u in enumerate(user_ids)}
        item_mapping = {i: idx for idx, i in enumerate(item_ids)}
        
        n_users = len(user_mapping)
        n_items = len(item_mapping)
        train_interactions = csr_matrix((n_users, n_items), dtype=np.float32)
        
        for _, row in contextual_df.iterrows():
            user_idx = user_mapping[row['user']]
            item_idx = item_mapping['sowing_rate']
            train_interactions[user_idx, item_idx] = row['c10_SowingRate']
        
        # Store original min/max for later scaling
        min_rating = contextual_df['c10_SowingRate'].min()
        max_rating = contextual_df['c10_SowingRate'].max()
        
        # Prepare TEST DATASET
        test_df_grouped = self.evaluation_df_grouped.copy()
        test_df_grouped['user'] = test_df_grouped.index
        
        # Train LightFM model
        model = LightFM(loss='warp')
        model.fit(train_interactions, epochs=30, verbose=False)
        
        # Predict and Evaluate
        prediction_rows = []
        true_c10 = []
        estimated_c10 = []
        
        # Get LightFM raw predictions
        raw_preds = []
        
        for _, row in test_df_grouped.iterrows():
            user = row['user']
            
            if user in user_mapping:
                user_idx = user_mapping[user]
                item_idx = item_mapping['sowing_rate']
                pred = model.predict(np.array([user_idx]), np.array([item_idx]))[0]
                raw_preds.append(pred)
        
        # Rescale LightFM predictions back to the original range
        pred_min = min(raw_preds)
        pred_max = max(raw_preds)
        
        for idx, row in enumerate(test_df_grouped.itertuples(index=False)):
            user = row.user
            true_val = row.c10
            pred = raw_preds[idx]
            
            # Rescale to original rating scale:
            scaled_pred = min_rating + (max_rating - min_rating) * (pred - pred_min) / (pred_max - pred_min)
            
            prediction_rows.append({
                'Province': user,
                'True_c10': true_val,
                'Recommended_c10': round(scaled_pred, 2)
            })
            true_c10.append(true_val)
            estimated_c10.append(scaled_pred)
        
        results_df = pd.DataFrame(prediction_rows)
        
        # Error Metrics
        mae_lightfm = mean_absolute_error(true_c10, estimated_c10)
        mse_lightfm = mean_squared_error(true_c10, estimated_c10)
        mape_lightfm = (np.abs(np.array(true_c10) - np.array(estimated_c10)) / np.array(true_c10)).mean()
        accuracy_lightfm = 100 - (mape_lightfm * 100)
        
        metrics = {
            'method': 'LightFM',
            'mae': mae_lightfm,
            'mse': mse_lightfm,
            'mape': mape_lightfm * 100,
            'accuracy': accuracy_lightfm
        }
        
        print(f" LightFM Results: MAE={mae_lightfm:.2f}, Accuracy={accuracy_lightfm:.2f}%")
        return results_df, metrics
    
    def evaluate_caserec_methods(self):
        """YOUR EXACT USERKNN AND MATRIX FACTORIZATION IMPLEMENTATION"""
        if not CASEREC_AVAILABLE:
            print("\n Skipping CaseRecommender methods - library not available")
            return None, None, None, None
            
        print("\n Evaluating CaseRecommender Methods...")
        
        # Load and prepare TRAINING DATASET
        train_df = pd.read_csv(self.train_file_path)
        
        # Province-level aggregation
        contextual_vars = [col for col in train_df.columns if col not in ['Province', 'Farm_ID', 'Y_GrainYield']]
        train_data_dict = {}
        
        for var in contextual_vars + ['Y_GrainYield']:
            pivot = train_df.pivot_table(index='Province', columns='Farm_ID', values=var, aggfunc='first')
            train_data_dict[var] = pivot
        
        province_names = train_data_dict['c1_Soil_pH'].index.tolist()
        province_data = []
        
        for province in province_names:
            row = []
            for var in list(train_data_dict.keys())[:-1]:
                province_vector = train_data_dict[var].loc[province].values
                province_mean = np.nanmean(province_vector)
                row.append(province_mean)
            province_data.append(row)
        
        contextual_df = pd.DataFrame(province_data, columns=contextual_vars)
        contextual_df['Province'] = province_names
        contextual_df = contextual_df.reset_index(drop=True)
        contextual_df['user'] = contextual_df.index
        
        # Prepare training data for Case Recommender
        user_mapping = {user: idx for idx, user in enumerate(contextual_df['user'])}
        reverse_user_mapping = {idx: user for user, idx in user_mapping.items()}
        
        contextual_df['user_id'] = contextual_df['user'].map(user_mapping)
        
        train_df_cr = contextual_df[['user_id', 'c10_SowingRate']].copy()
        train_df_cr = train_df_cr.rename(columns={'c10_SowingRate': 'rating'})
        train_df_cr['item_id'] = 1
        train_df_cr = train_df_cr[['user_id', 'item_id', 'rating']]
        
        train_file = os.path.join(self.data_folder, 'temp_train_data.csv')
        train_df_cr.to_csv(train_file, sep='\t', header=False, index=False)
        
        # Prepare TEST DATASET
        test_df_grouped = self.evaluation_df_grouped.copy()
        test_df_grouped['user'] = test_df_grouped.index
        test_df_grouped['user_id'] = test_df_grouped['user'].map(user_mapping)
        
        test_df_cr = test_df_grouped[['user_id', 'c10']].copy()
        test_df_cr = test_df_cr.rename(columns={'c10': 'rating'})
        test_df_cr['item_id'] = 1
        test_df_cr = test_df_cr[['user_id', 'item_id', 'rating']]
        test_df_cr = test_df_cr.dropna()
        
        test_file = os.path.join(self.data_folder, 'temp_test_data.csv')
        test_df_cr.to_csv(test_file, sep='\t', header=False, index=False)
        
        def evaluate_predictions(predictions, model_name):
            if not predictions:
                print(f"\n No predictions available for {model_name}")
                return None, None
            
            # Auto detect prediction format length
            num_cols = len(predictions[0])
            
            if num_cols == 4:
                # Format: [user_id, item_id, true, pred]
                pred_df = pd.DataFrame(predictions, columns=['user_id', 'item_id', 'true', 'pred'])
            elif num_cols == 3:
                # Format: [user_id, item_id, pred]  (no true values included)
                pred_df = pd.DataFrame(predictions, columns=['user_id', 'item_id', 'pred'])
                
                # Merge with test_df to retrieve true values
                pred_df = pd.merge(pred_df, test_df_cr, on=['user_id', 'item_id'], how='inner')
                pred_df = pred_df.rename(columns={'rating': 'true'})
            else:
                print(f" Unexpected prediction format with {num_cols} columns!")
                return None, None
            
            pred_df['true'] = pred_df['true'].astype(float)
            pred_df['pred'] = pred_df['pred'].astype(float)
            pred_df['Province'] = pred_df['user_id'].map(reverse_user_mapping)
            
            true_values = pred_df['true'].values
            pred_values = pred_df['pred'].values
            
            mae = mean_absolute_error(true_values, pred_values)
            mse = mean_squared_error(true_values, pred_values)
            mape = (np.abs(true_values - pred_values) / np.maximum(np.abs(true_values), 1e-10)).mean() * 100
            accuracy = 100 - mape
            
            results_df = pred_df[['Province', 'true', 'pred']].copy()
            results_df.columns = ['Province', 'True_c10', 'Recommended_c10']
            results_df['Recommended_c10'] = results_df['Recommended_c10'].round(2)
            
            metrics = {
                'method': model_name,
                'mae': mae,
                'mse': mse,
                'mape': mape,
                'accuracy': accuracy
            }
            
            print(f"{model_name} Results: MAE={mae:.2f}, Accuracy={accuracy:.2f}%")
            return results_df, metrics
        
        try:
            # UserKNN
            print("Training UserKNN model...")
            user_knn = UserKNN(train_file, test_file, similarity_metric='correlation', 
                             k_neighbors=min(5, len(train_df_cr['user_id'].unique())-1))
            user_knn.compute()
            predictions_user_knn = user_knn.predictions
            userknn_results, userknn_metrics = evaluate_predictions(predictions_user_knn, "UserKNN")
            
            # Matrix Factorization
            print("Training Matrix Factorization model...")
            num_factors = min(5, len(train_df_cr['user_id'].unique())-1)
            mf = MatrixFactorization(train_file, test_file, factors=num_factors, learn_rate=0.01, epochs=30)
            mf.compute()
            predictions_mf = mf.predictions
            userfm_results, userfm_metrics = evaluate_predictions(predictions_mf, "UserFM")
            
        except Exception as e:
            print(f"CaseRecommender evaluation failed: {e}")
            userknn_results, userknn_metrics = None, None
            userfm_results, userfm_metrics = None, None
        
        finally:
            # Clean up
            try:
                os.remove(train_file)
                os.remove(test_file)
            except:
                pass
        
        return userknn_results, userknn_metrics, userfm_results, userfm_metrics
    
    def evaluate_carsfm_xlearn(self):
        """YOUR EXACT FACTORIZATION MACHINES IMPLEMENTATION"""
        if not XLEARN_AVAILABLE:
            print("\n Skipping CARSFM - xLearn library not available")
            return None, None
            
        print("\nEvaluating CARSFM (Factorization Machines) Baseline...")
        
        # Load and prepare TRAINING DATASET
        train_df = pd.read_csv(self.train_file_path)
        
        # Province-level aggregation (as always)
        contextual_vars = [col for col in train_df.columns if col not in ['Province', 'Farm_ID', 'Y_GrainYield']]
        train_data_dict = {}
        
        for var in contextual_vars + ['Y_GrainYield']:
            pivot = train_df.pivot_table(index='Province', columns='Farm_ID', values=var, aggfunc='first')
            train_data_dict[var] = pivot
        
        province_names = train_data_dict['c1_Soil_pH'].index.tolist()
        province_data = []
        
        for province in province_names:
            row = []
            for var in list(train_data_dict.keys())[:-1]:  # exclude Y
                province_vector = train_data_dict[var].loc[province].values
                province_mean = np.nanmean(province_vector)
                row.append(province_mean)
            province_data.append(row)
        
        contextual_df = pd.DataFrame(province_data, columns=contextual_vars)
        contextual_df['Province'] = province_names
        contextual_df = contextual_df.reset_index(drop=True)
        contextual_df['user'] = contextual_df.index
        
        # Prepare TEST DATASET
        test_df_grouped = self.evaluation_df_grouped.copy()
        test_df_grouped['user'] = test_df_grouped.index
        
        # Normalize features and target together
        # Use same features as usual (excluding target c10)
        context_features = [col for col in contextual_df.columns if col not in ['Province', 'user', 'c10_SowingRate']]
        
        # Combine train and test to apply same scaler
        all_data = pd.concat([contextual_df, test_df_grouped], ignore_index=True)
        all_data[context_features] = all_data[context_features].fillna(0)
        
        # Feature scaler
        feature_scaler = StandardScaler()
        all_data[context_features] = feature_scaler.fit_transform(all_data[context_features])
        
        # Target scaler (only for c10)
        target_scaler = StandardScaler()
        train_targets = contextual_df['c10_SowingRate'].values.reshape(-1, 1)
        target_scaler.fit(train_targets)
        
        # Apply back normalized target
        all_data['c10_normalized'] = target_scaler.transform(pd.concat([
            contextual_df['c10_SowingRate'], 
            test_df_grouped['c10']
        ], ignore_index=True).values.reshape(-1, 1))
        
        # Separate back train and test
        train_data = all_data.iloc[:len(contextual_df)].reset_index(drop=True)
        test_data = all_data.iloc[len(contextual_df):].reset_index(drop=True)
        
        # Write files for xLearn
        train_file = os.path.join(self.data_folder, 'temp_fm_train.txt')
        test_file = os.path.join(self.data_folder, 'temp_fm_test.txt')
        
        # Write train file
        with open(train_file, 'w') as f:
            for i, row in train_data.iterrows():
                label = row['c10_normalized']
                features = ' '.join([f"{idx}:{row[feat]}" for idx, feat in enumerate(context_features)])
                f.write(f"{label} {features}\n")
        
        # Write test file
        with open(test_file, 'w') as f:
            for i, row in test_data.iterrows():
                label = row['c10_normalized']
                features = ' '.join([f"{idx}:{row[feat]}" for idx, feat in enumerate(context_features)])
                f.write(f"{label} {features}\n")
        
        # True target for evaluation (non-normalized)
        true_c10_test = test_df_grouped['c10'].values
        
        try:
            fm_model = xl.create_fm()
            fm_model.setTrain(train_file)
            fm_model.setValidate(test_file)
            
            param = {
                'task': 'reg',
                'lr': 0.1,
                'lambda': 0.001,
                'metric': 'mae',
                'k': 10
            }
            
            model_file = os.path.join(self.data_folder, 'temp_fm_model.out')
            output_file = os.path.join(self.data_folder, 'temp_output.txt')
            
            fm_model.fit(param, model_file)
            fm_model.setTest(test_file)
            fm_model.predict(model_file, output_file)
            
            # Read predictions (normalized)
            predictions_normalized = []
            with open(output_file, 'r') as f:
                for line in f:
                    predictions_normalized.append(float(line.strip()))
            predictions_normalized = np.array(predictions_normalized)
            
            # Inverse transform back to original scale
            predictions_original = target_scaler.inverse_transform(predictions_normalized.reshape(-1, 1)).flatten()
            
            # Evaluate
            mae = mean_absolute_error(true_c10_test, predictions_original)
            mse = mean_squared_error(true_c10_test, predictions_original)
            mape = np.mean(np.abs((true_c10_test - predictions_original) / true_c10_test)) * 100
            accuracy = 100 - mape
            
            results_df = pd.DataFrame({
                'Province': test_df_grouped['Province'],
                'True_c10': true_c10_test,
                'Recommended_c10': np.round(predictions_original, 2)
            })
            
            metrics = {
                'method': 'CARSFM',
                'mae': mae,
                'mse': mse,
                'mape': mape,
                'accuracy': accuracy
            }
            
            print(f"CARSFM Results: MAE={mae:.2f}, Accuracy={accuracy:.2f}%")
            
        except Exception as e:
            print(f"CARSFM failed: {e}")
            results_df, metrics = None, None
        
        finally:
            # Clean up
            for file_path in [train_file, test_file, model_file, output_file]:
                try:
                    os.remove(file_path)
                except:
                    pass
        
        return results_df, metrics
    
    def evaluate_tensor_cars(self):
        """YOUR EXACT TENSOR CARS IMPLEMENTATION"""
        print("\n🔄 Evaluating Tensor CARS Baseline...")
        
        # Load and prepare TRAINING DATASET
        train_df = pd.read_csv(self.train_file_path)
        
        # Apply consistent renaming directly to train dataset
        train_df = train_df.rename(columns=self.rename_dict)
        
        # Province-level aggregation
        contextual_vars = [col for col in train_df.columns if col not in ['Province', 'Farm_ID', 'Y']]
        train_data_dict = {}
        
        for var in contextual_vars + ['Y']:
            pivot = train_df.pivot_table(index='Province', columns='Farm_ID', values=var, aggfunc='first')
            train_data_dict[var] = pivot
        
        province_names = train_data_dict['c1'].index.tolist()
        province_data = []
        
        for province in province_names:
            row = []
            for var in list(train_data_dict.keys())[:-1]:  # exclude Y
                province_vector = train_data_dict[var].loc[province].values
                province_mean = np.nanmean(province_vector)
                row.append(province_mean)
            province_data.append(row)
        
        contextual_df = pd.DataFrame(province_data, columns=contextual_vars)
        contextual_df['Province'] = province_names
        contextual_df = contextual_df.reset_index(drop=True)
        contextual_df['user_id'] = contextual_df.index
        
        # Prepare EVALUATION DATASET
        test_df_grouped = self.evaluation_df_grouped.copy()
        test_df_grouped['user_id'] = test_df_grouped.index
        
        # Select Contextual Feature for Tensor (excluding target c10)
        all_context_features = [col for col in contextual_df.columns 
                                if col not in ['Province', 'user_id', 'c10']]
        
        context_feature = all_context_features[0]  # take first feature as context (ex: 'c1')
        print(f"Using context feature: {context_feature}")
        
        # Discretize Context into Bins
        n_bins = 3
        feature_min = contextual_df[context_feature].min()
        feature_max = contextual_df[context_feature].max()
        bin_edges = np.linspace(feature_min, feature_max, n_bins + 1)
        
        contextual_df['context_bin'] = np.digitize(contextual_df[context_feature], bin_edges[1:-1])
        
        # Compute Tensor Averages
        tensor_averages = contextual_df.groupby(['user_id', 'context_bin'])['c10'].mean().reset_index()
        global_avg = contextual_df['c10'].mean()
        province_avgs = contextual_df.groupby('user_id')['c10'].mean()
        
        # Apply Tensor Prediction on Evaluation Dataset
        test_df_grouped['context_bin'] = np.digitize(test_df_grouped[context_feature], bin_edges[1:-1])
        y_test = test_df_grouped['c10'].values
        
        predictions = []
        methods_used = []
        
        for idx, row in test_df_grouped.iterrows():
            user_id = row['user_id']
            context_bin = row['context_bin']
            
            match = tensor_averages[(tensor_averages['user_id'] == user_id) & 
                                   (tensor_averages['context_bin'] == context_bin)]
            
            if not match.empty:
                predictions.append(match.iloc[0]['c10'])
                methods_used.append('tensor')
            elif user_id in province_avgs:
                predictions.append(province_avgs[user_id])
                methods_used.append('province_avg')
            else:
                predictions.append(global_avg)
                methods_used.append('global_avg')
        
        predictions = np.array(predictions)
        
        # Evaluation Metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        accuracy = 100 - mape
        
        results_df = pd.DataFrame({
            'Province': test_df_grouped['Province'],
            'True_c10': y_test,
            'Context_Bin': test_df_grouped['context_bin'],
            'Recommended_c10': np.round(predictions, 2),
            'Method': methods_used
        })
        
        metrics = {
            'method': 'Tensor_CARS',
            'mae': mae,
            'mse': mse,
            'mape': mape,
            'accuracy': accuracy
        }
        
        method_counts = pd.Series(methods_used).value_counts()
        print(f"Prediction methods used:")
        for method, count in method_counts.items():
            print(f"- {method}: {count} ({count/len(methods_used)*100:.1f}%)")
        
        print(f"Tensor CARS Results: MAE={mae:.2f}, Accuracy={accuracy:.2f}%")
        return results_df, metrics
    
    def run_complete_evaluation(self):
        """Run all your exact methods and compare results"""
        print("Starting Complete CVCPR Baseline Evaluation...")
        
        if not self.load_and_prepare_common_data():
            return None
        
        # 1. CVCPR Approach
        try:
            cvcpr_results, cvcpr_metrics = self.evaluate_cvcpr_approach()
            if cvcpr_results is not None:
                self.all_results['CVCPR'] = cvcpr_results
                self.all_metrics.append(cvcpr_metrics)
        except Exception as e:
            print(f"CVCPR evaluation failed: {e}")
        
        # 2. SVD/Surprise
        try:
            svd_results, svd_metrics = self.evaluate_svd_surprise()
            if svd_results is not None:
                self.all_results['SVD_Surprise'] = svd_results
                self.all_metrics.append(svd_metrics)
        except Exception as e:
            print(f"SVD evaluation failed: {e}")
        
        # 3. LightFM
        try:
            lightfm_results, lightfm_metrics = self.evaluate_lightfm()
            if lightfm_results is not None:
                self.all_results['LightFM'] = lightfm_results
                self.all_metrics.append(lightfm_metrics)
        except Exception as e:
            print(f"LightFM evaluation failed: {e}")
        
        # 4. UserKNN and UserFM (CaseRecommender)
        try:
            userknn_results, userknn_metrics, userfm_results, userfm_metrics = self.evaluate_caserec_methods()
            if userknn_results is not None:
                self.all_results['UserKNN'] = userknn_results
                self.all_metrics.append(userknn_metrics)
            if userfm_results is not None:
                self.all_results['UserFM'] = userfm_results
                self.all_metrics.append(userfm_metrics)
        except Exception as e:
            print(f"CaseRecommender evaluation failed: {e}")
        
        # 5. CARSFM (Factorization Machines - xLearn)
        try:
            carsfm_results, carsfm_metrics = self.evaluate_carsfm_xlearn()
            if carsfm_results is not None:
                self.all_results['CARSFM'] = carsfm_results
                self.all_metrics.append(carsfm_metrics)
        except Exception as e:
            print(f"CARSFM evaluation failed: {e}")
        
        # 6. Tensor CARS
        try:
            tensor_results, tensor_metrics = self.evaluate_tensor_cars()
            if tensor_results is not None:
                self.all_results['Tensor_CARS'] = tensor_results
                self.all_metrics.append(tensor_metrics)
        except Exception as e:
            print(f"Tensor CARS evaluation failed: {e}")
        
        # Compile comparison results
        comparison_df = pd.DataFrame(self.all_metrics)
        comparison_df = comparison_df.sort_values('mae')
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Save detailed results
        results_path = os.path.join(self.data_folder, 'complete_baseline_evaluation_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        save_results = {}
        for method, df in self.all_results.items():
            save_results[method] = df.to_dict('records')
        
        save_data = {
            'detailed_results': save_results,
            'summary_metrics': comparison_df.to_dict('records'),
            'evaluation_params': {
                'controllable_variable': f'c{self.R}',
                'n_evaluation_provinces': len(self.evaluation_df_grouped),
                'tau_example': self.tau_example
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"\n Detailed results saved to: {results_path}")
        
        # Find best performing method
        best_method = comparison_df.iloc[0]
        print(f"\n BEST PERFORMING METHOD: {best_method['method']}")
        print(f"   MAE: {best_method['mae']:.3f}")
        print(f"   Accuracy: {best_method['accuracy']:.2f}%")
        
        if 'CVCPR' in comparison_df['method'].values:
            cvcpr_row = comparison_df[comparison_df['method'] == 'CVCPR'].iloc[0]
            cvcpr_rank = comparison_df[comparison_df['method'] == 'CVCPR'].index[0] + 1
            print(f"\n CVCPR PERFORMANCE:")
            print(f"   Rank: {cvcpr_rank}/{len(comparison_df)}")
            print(f"   MAE: {cvcpr_row['mae']:.3f}")
            print(f"   Accuracy: {cvcpr_row['accuracy']:.2f}%")
            
            if cvcpr_rank == 1:
                print("   CVCPR is the BEST performing method!")
            else:
                improvement_needed = cvcpr_row['mae'] - best_method['mae']
                print(f"   Improvement needed: {improvement_needed:.3f} MAE points")
        
        return comparison_df, self.all_results

# ==========================================================
# 🔥 MAIN EXECUTION FUNCTION
# ==========================================================

def run_complete_cvcpr_evaluation(data_folder='data'):
    """Run the complete evaluation pipeline with all your exact methods"""
    evaluator = CVCPRBaselineEvaluator(data_folder=data_folder)
    comparison_results, detailed_results = evaluator.run_complete_evaluation()
    return evaluator, comparison_results, detailed_results

# ==========================================================
# RUN EVALUATION
# ==========================================================

if __name__ == "__main__":
    evaluator, comparison_df, all_results = run_complete_cvcpr_evaluation()