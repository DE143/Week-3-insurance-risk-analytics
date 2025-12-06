"""
Machine learning modeling module for insurance risk analytics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsuranceModeler:
    """Machine learning modeler for insurance risk assessment"""
    
    def __init__(self, data, features, preprocessor):
        """
        Initialize InsuranceModeler
        
        Args:
            data: pandas.DataFrame containing the data
            features: List of feature names
            preprocessor: sklearn preprocessor pipeline
        """
        self.data = data.copy()
        self.features = features
        self.preprocessor = preprocessor
        self.models = {}
        self.results = {}
        
    def prepare_claim_severity_data(self):
        """
        Prepare data for claim severity prediction
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) or None if insufficient data
        """
        if 'TotalClaims' not in self.data.columns:
            logger.error("TotalClaims column not found")
            return None
        
        # Filter only policies with claims
        claim_data = self.data[self.data['TotalClaims'] > 0].copy()
        
        if len(claim_data) < 20:
            logger.warning(f"Insufficient claim data: {len(claim_data)} samples (need at least 20)")
            return None
        
        X = claim_data[self.features]
        y = claim_data['TotalClaims']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        logger.info(f"Claim severity data prepared: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def train_claim_severity_models(self):
        """
        Train regression models for claim severity prediction
        
        Returns:
            dict: Trained models and results
        """
        logger.info("Training claim severity models...")
        
        # Prepare data
        data_split = self.prepare_claim_severity_data()
        if data_split is None:
            logger.error("Could not prepare claim severity data")
            return {}
        
        X_train, X_test, y_train, y_test = data_split
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0, max_depth=5),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, max_depth=5)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"  Training {name}...")
            
            try:
                # Create pipeline
                pipeline = self._create_model_pipeline(model, model_type='regressor')
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Predict
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train, 
                    cv=min(5, len(X_train)), 
                    scoring='r2', 
                    n_jobs=-1
                )
                
                results[name] = {
                    'model': pipeline,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"    R²: {r2:.4f}, RMSE: ZAR {rmse:,.2f}")
                
            except Exception as e:
                logger.error(f"    Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.results['claim_severity'] = results
        
        # Select best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model_name = max(valid_results.items(), key=lambda x: x[1]['r2'])[0]
            self.models['claim_severity'] = valid_results[best_model_name]['model']
            logger.info(f"Best claim severity model: {best_model_name} (R²: {valid_results[best_model_name]['r2']:.4f})")
        
        return results
    
    def prepare_claim_probability_data(self):
        """
        Prepare data for claim probability prediction
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) or None if insufficient data
        """
        if 'HasClaim' not in self.data.columns:
            logger.error("HasClaim column not found")
            return None
        
        X = self.data[self.features]
        y = self.data['HasClaim']
        
        # Check class balance
        claim_count = y.sum()
        no_claim_count = len(y) - claim_count
        
        if claim_count < 10:
            logger.warning(f"Insufficient claim data: {claim_count} positive samples (need at least 10)")
            return None
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
        )
        
        logger.info(f"Claim probability data prepared: {len(X_train)} train, {len(X_test)} test samples")
        logger.info(f"  Positive samples: {claim_count}/{len(y)} ({claim_count/len(y):.1%})")
        
        return X_train, X_test, y_train, y_test
    
    def train_claim_probability_models(self):
        """
        Train classification models for claim probability prediction
        
        Returns:
            dict: Trained models and results
        """
        logger.info("Training claim probability models...")
        
        # Prepare data
        data_split = self.prepare_claim_probability_data()
        if data_split is None:
            logger.error("Could not prepare claim probability data")
            return {}
        
        X_train, X_test, y_train, y_test = data_split
        
        # Calculate class weights for imbalanced data
        scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0, max_depth=5, scale_pos_weight=scale_pos_weight),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, max_depth=5, class_weight='balanced')
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"  Training {name}...")
            
            try:
                # Create pipeline
                pipeline = self._create_model_pipeline(model, model_type='classifier')
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Predict
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    pipeline, X_train, y_train, 
                    cv=min(5, y_train.sum()), 
                    scoring='roc_auc', 
                    n_jobs=-1
                )
                
                results[name] = {
                    'model': pipeline,
                    'roc_auc': roc_auc,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"    ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.error(f"    Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.results['claim_probability'] = results
        
        # Select best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model_name = max(valid_results.items(), key=lambda x: x[1]['roc_auc'])[0]
            self.models['claim_probability'] = valid_results[best_model_name]['model']
            logger.info(f"Best claim probability model: {best_model_name} (ROC AUC: {valid_results[best_model_name]['roc_auc']:.4f})")
        
        return results
    
    def prepare_premium_optimization_data(self):
        """
        Prepare data for premium optimization
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) or None if insufficient data
        """
        # Calculate optimal premium based on risk
        if 'RiskFactor' in self.data.columns and 'TotalPremium' in self.data.columns:
            self.data['OptimalPremium'] = self.data['TotalPremium'] * self.data['RiskFactor']
            target_col = 'OptimalPremium'
        elif 'TotalPremium' in self.data.columns:
            target_col = 'TotalPremium'
        else:
            logger.error("No premium data available")
            return None
        
        X = self.data[self.features]
        y = self.data[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        logger.info(f"Premium optimization data prepared: {len(X_train)} train, {len(X_test)} test samples")
        logger.info(f"  Target: {target_col}, Mean: ZAR {y.mean():,.2f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_premium_optimization_models(self):
        """
        Train models for premium optimization
        
        Returns:
            dict: Trained models and results
        """
        logger.info("Training premium optimization models...")
        
        # Prepare data
        data_split = self.prepare_premium_optimization_data()
        if data_split is None:
            logger.error("Could not prepare premium optimization data")
            return {}
        
        X_train, X_test, y_train, y_test = data_split
        
        # Define models
        models = {
            'XGBoost': XGBRegressor(n_estimators=150, random_state=42, verbosity=0, max_depth=5),
            'LightGBM': lgb.LGBMRegressor(n_estimators=150, random_state=42, verbose=-1, max_depth=5),
            'Random Forest': RandomForestRegressor(n_estimators=150, random_state=42, max_depth=10)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"  Training {name}...")
            
            try:
                # Create pipeline
                pipeline = self._create_model_pipeline(model, model_type='regressor')
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Predict
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'model': pipeline,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                logger.info(f"    R²: {r2:.4f}, RMSE: ZAR {rmse:,.2f}")
                
            except Exception as e:
                logger.error(f"    Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.results['premium_optimization'] = results
        
        # Select best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model_name = max(valid_results.items(), key=lambda x: x[1]['r2'])[0]
            self.models['premium_optimization'] = valid_results[best_model_name]['model']
            logger.info(f"Best premium optimization model: {best_model_name} (R²: {valid_results[best_model_name]['r2']:.4f})")
        
        return results
    
    def _create_model_pipeline(self, model, model_type='regressor'):
        """
        Create a pipeline with preprocessing and model
        
        Args:
            model: The model to include in pipeline
            model_type: Type of model ('regressor' or 'classifier')
        
        Returns:
            Pipeline: sklearn pipeline
        """
        from sklearn.pipeline import Pipeline
        
        if model_type == 'regressor':
            return Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
        else:  # classifier
            return Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
    
    def perform_shap_analysis(self, model_name='claim_severity', sample_size=100):
        """
        Perform SHAP analysis for model interpretation
        
        Args:
            model_name: Name of model to analyze
            sample_size: Number of samples to use for SHAP analysis
        
        Returns:
            dict: SHAP analysis results
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        # Check if model is tree-based (required for TreeExplainer)
        model_type = model.named_steps['regressor'] if 'regressor' in model.named_steps else model.named_steps['classifier']
        
        tree_based_models = ['RandomForest', 'XGB', 'LGBM']
        is_tree_based = any(name in str(type(model_type)) for name in tree_based_models)
        
        if not is_tree_based:
            logger.warning("SHAP analysis only works with tree-based models")
            return None
        
        try:
            # Get training data
            if model_name == 'claim_severity':
                data_split = self.prepare_claim_severity_data()
            elif model_name == 'claim_probability':
                data_split = self.prepare_claim_probability_data()
            elif model_name == 'premium_optimization':
                data_split = self.prepare_premium_optimization_data()
            else:
                logger.error(f"Unknown model name: {model_name}")
                return None
            
            if data_split is None:
                return None
            
            X_train, _, _, _ = data_split
            
            # Sample data for faster computation
            if len(X_train) > sample_size:
                X_sample = X_train.sample(n=min(sample_size, len(X_train)), random_state=42)
            else:
                X_sample = X_train
            
            # Transform data
            X_processed = model.named_steps['preprocessor'].transform(X_sample)
            
            # Get feature names
            preprocessor = model.named_steps['preprocessor']
            feature_names = self._get_feature_names(preprocessor)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model_type)
            shap_values = explainer.shap_values(X_processed)
            
            # If shap_values is a list (for classification), take the second element (positive class)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Calculate feature importance
            importance_df = pd.DataFrame({
                'feature': feature_names[:shap_values.shape[1]],
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False).head(10)
            
            # Create visualizations
            self._create_shap_visualizations(shap_values, X_processed, feature_names, model_name)
            
            shap_results = {
                'feature_importance': importance_df.to_dict('records'),
                'shap_values_sample': shap_values.tolist()[:10],  # First 10 samples
                'expected_value': explainer.expected_value
            }
            
            logger.info("SHAP analysis completed")
            return shap_results
            
        except Exception as e:
            logger.error(f"Error in SHAP analysis: {e}")
            return None
    
    def _get_feature_names(self, preprocessor):
        """Get feature names after preprocessing"""
        # Get numerical feature names
        num_features = preprocessor.transformers_[0][2]  # Numerical columns
        
        # Get categorical feature names after one-hot encoding
        if len(preprocessor.transformers_) > 1:
            cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
            cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
            all_features = list(num_features) + list(cat_features)
        else:
            all_features = list(num_features)
        
        return all_features
    
    def _create_shap_visualizations(self, shap_values, X_processed, feature_names, model_name):
        """Create SHAP visualizations"""
        # Create directory for figures
        os.makedirs('../reports/figures', exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_processed, feature_names=feature_names, 
                         max_display=15, show=False)
        plt.title(f'SHAP Summary Plot - {model_name.replace("_", " ").title()}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'../reports/figures/shap_summary_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance bar plot
        importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(importance)[-10:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance[top_indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_importance[::-1])
        plt.yticks(range(len(top_features)), top_features[::-1])
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Top 10 Feature Importance - {model_name.replace("_", " ").title()}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'../reports/figures/shap_importance_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP visualizations saved for {model_name}")
    
    def create_risk_based_pricing(self, sample_size=5):
        """
        Create risk-based pricing recommendations
        
        Args:
            sample_size: Number of sample calculations to show
        
        Returns:
            dict: Risk-based pricing results
        """
        if 'claim_probability' not in self.models or 'claim_severity' not in self.models:
            logger.error("Required models not available")
            return None
        
        clf_model = self.models['claim_probability']
        severity_model = self.models['claim_severity']
        
        # Get sample policies
        sample_indices = np.random.choice(len(self.data), min(sample_size, len(self.data)), replace=False)
        
        results = []
        
        for idx in sample_indices:
            policy = self.data.iloc[idx]
            
            try:
                # Prepare input
                input_data = pd.DataFrame([policy[self.features]])
                
                # Predict claim probability
                claim_prob = clf_model.predict_proba(input_data)[0, 1]
                
                # Predict claim severity
                predicted_severity = severity_model.predict(input_data)[0]
                
                # Calculate risk-based premium
                expense_loading = 0.15  # 15% for expenses
                profit_margin = 0.10    # 10% profit margin
                
                pure_premium = claim_prob * predicted_severity
                risk_based_premium = pure_premium * (1 + expense_loading) * (1 + profit_margin)
                
                current_premium = policy.get('TotalPremium', 0)
                
                results.append({
                    'policy_index': int(idx),
                    'current_premium': float(current_premium),
                    'risk_based_premium': float(risk_based_premium),
                    'adjustment_percent': float((risk_based_premium / current_premium - 1) * 100) if current_premium > 0 else 0,
                    'claim_probability': float(claim_prob),
                    'predicted_severity': float(predicted_severity),
                    'pure_premium': float(pure_premium)
                })
                
            except Exception as e:
                logger.warning(f"Error calculating premium for policy {idx}: {e}")
        
        # Create summary
        if results:
            summary = {
                'average_current_premium': np.mean([r['current_premium'] for r in results]),
                'average_risk_based_premium': np.mean([r['risk_based_premium'] for r in results]),
                'average_adjustment': np.mean([r['adjustment_percent'] for r in results]),
                'sample_calculations': results
            }
            
            # Print sample calculations
            print("\n" + "="*80)
            print("RISK-BASED PRICING SAMPLE CALCULATIONS")
            print("="*80)
            
            for r in results:
                print(f"\nPolicy {r['policy_index']}:")
                print(f"  Current Premium: ZAR {r['current_premium']:,.2f}")
                print(f"  Risk-Based Premium: ZAR {r['risk_based_premium']:,.2f}")
                print(f"  Adjustment: {r['adjustment_percent']:+.1f}%")
                print(f"  Claim Probability: {r['claim_probability']:.2%}")
                print(f"  Predicted Severity: ZAR {r['predicted_severity']:,.2f}")
            
            print(f"\nAverage Adjustment: {summary['average_adjustment']:+.1f}%")
            
            return summary
        else:
            return None
    
    def save_models(self, output_dir='../models'):
        """
        Save trained models
        
        Args:
            output_dir: Directory to save models
        
        Returns:
            dict: Paths to saved models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = {}
        
        # Save individual models
        for model_name, model in self.models.items():
            try:
                path = f'{output_dir}/{model_name}_model.pkl'
                joblib.dump(model, path)
                saved_paths[model_name] = path
                logger.info(f"Saved {model_name} model to {path}")
            except Exception as e:
                logger.error(f"Error saving {model_name} model: {e}")
        
        # Save all results
        try:
            results_path = f'{output_dir}/model_results.json'
            with open(results_path, 'w') as f:
                # Convert models to strings to avoid serialization issues
                serializable_results = {}
                for category, category_results in self.results.items():
                    serializable_results[category] = {}
                    for model_name, model_result in category_results.items():
                        if 'error' in model_result:
                            serializable_results[category][model_name] = model_result
                        else:
                            serializable_results[category][model_name] = {
                                k: (v if k != 'model' else 'model_object') 
                                for k, v in model_result.items()
                            }
                
                json.dump(serializable_results, f, indent=2, default=str)
            
            saved_paths['results'] = results_path
            logger.info(f"Saved model results to {results_path}")
        except Exception as e:
            logger.error(f"Error saving model results: {e}")
        
        # Save metadata
        try:
            metadata = {
                'created_date': datetime.now().isoformat(),
                'features': self.features,
                'models_trained': list(self.models.keys()),
                'dataset_size': len(self.data),
                'claim_frequency': self.data.get('HasClaim', pd.Series([0])).mean()
            }
            
            metadata_path = f'{output_dir}/model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            saved_paths['metadata'] = metadata_path
            logger.info(f"Saved model metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
        
        return saved_paths
    
    def generate_report(self):
        """
        Generate model performance report
        
        Returns:
            pandas.DataFrame: Model comparison report
        """
        report_data = []
        
        for category, category_results in self.results.items():
            for model_name, model_result in category_results.items():
                if 'error' not in model_result:
                    if category == 'claim_severity':
                        report_data.append({
                            'Category': 'Claim Severity',
                            'Model': model_name,
                            'R²': model_result.get('r2', np.nan),
                            'RMSE': model_result.get('rmse', np.nan),
                            'MAE': model_result.get('mae', np.nan),
                            'CV R²': model_result.get('cv_mean', np.nan)
                        })
                    elif category == 'claim_probability':
                        report_data.append({
                            'Category': 'Claim Probability',
                            'Model': model_name,
                            'ROC AUC': model_result.get('roc_auc', np.nan),
                            'Accuracy': model_result.get('accuracy', np.nan),
                            'F1 Score': model_result.get('f1', np.nan),
                            'CV AUC': model_result.get('cv_mean', np.nan)
                        })
                    elif category == 'premium_optimization':
                        report_data.append({
                            'Category': 'Premium Optimization',
                            'Model': model_name,
                            'R²': model_result.get('r2', np.nan),
                            'RMSE': model_result.get('rmse', np.nan),
                            'MAE': model_result.get('mae', np.nan)
                        })
        
        if report_data:
            report_df = pd.DataFrame(report_data)
            
            # Format numerical columns
            for col in ['R²', 'ROC AUC', 'Accuracy', 'F1 Score', 'CV R²', 'CV AUC']:
                if col in report_df.columns:
                    report_df[col] = report_df[col].apply(lambda x: f'{x:.4f}' if not pd.isna(x) else 'N/A')
            
            for col in ['RMSE', 'MAE']:
                if col in report_df.columns:
                    report_df[col] = report_df[col].apply(lambda x: f'ZAR {x:,.2f}' if not pd.isna(x) else 'N/A')
            
            return report_df
        else:
            return pd.DataFrame()


def run_modeling_pipeline(data_path, features=None, output_dir='../models'):
    """
    Run complete modeling pipeline
    
    Args:
        data_path: Path to processed data
        features: List of features to use (optional)
        output_dir: Output directory for models
    
    Returns:
        InsuranceModeler: Modeler object with trained models
    """
    # Load data
    data = pd.read_csv(data_path)
    logger.info(f"Data loaded: {data.shape}")
    
    # Import DataProcessor to get features and preprocessor
    # from src.data_processing import DataProcessor
    from data_processing import DataProcessor
    
    # Process data
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.engineer_features()
    
    # Select features
    if features is None:
        features = processor.select_features()
    
    # Create preprocessor
    preprocessor = processor.create_preprocessor()
    
    # Initialize modeler
    modeler = InsuranceModeler(processor.df, features, preprocessor)
    
    # Train models
    logger.info("Starting modeling pipeline...")
    
    # 1. Claim severity models
    severity_results = modeler.train_claim_severity_models()
    
    # 2. Claim probability models
    probability_results = modeler.train_claim_probability_models()
    
    # 3. Premium optimization models
    premium_results = modeler.train_premium_optimization_models()
    
    # Generate report
    report_df = modeler.generate_report()
    if not report_df.empty:
        print("\n" + "="*80)
        print("MODEL PERFORMANCE REPORT")
        print("="*80)
        print(report_df.to_string(index=False))
    
    # SHAP analysis for best severity model
    if 'claim_severity' in modeler.models:
        shap_results = modeler.perform_shap_analysis('claim_severity')
        if shap_results:
            print("\n" + "-"*80)
            print("TOP 10 FEATURE IMPORTANCE (from SHAP analysis)")
            print("-"*80)
            importance_df = pd.DataFrame(shap_results['feature_importance'])
            print(importance_df.to_string(index=False))
    
    # Risk-based pricing
    pricing_results = modeler.create_risk_based_pricing(sample_size=5)
    
    # Save models
    saved_paths = modeler.save_models(output_dir)
    
    # Business recommendations
    print("\n" + "="*80)
    print("BUSINESS RECOMMENDATIONS")
    print("="*80)
    
    models_available = []
    if 'claim_severity' in modeler.models:
        models_available.append("Claim Severity Prediction")
    if 'claim_probability' in modeler.models:
        models_available.append("Claim Probability Prediction")
    if 'premium_optimization' in modeler.models:
        models_available.append("Premium Optimization")
    
    if models_available:
        print(f"\n✓ Models Available: {', '.join(models_available)}")
        print("\nRecommended Actions:")
        print("  1. Implement risk-based pricing using predicted probability × severity")
        print("  2. Adjust premiums by ±30% based on risk score")
        print("  3. Create personalized marketing campaigns")
        print("  4. Monitor model performance monthly")
        print("  5. Retrain models quarterly with new data")
    else:
        print("\n✗ No reliable models trained")
        print("  → Consider collecting more data or different features")
    
    logger.info("Modeling pipeline completed")
    return modeler


if __name__ == "__main__":
    # Example usage
    data_path = "../data/processed/insurance_processed.csv"
    
    print("Running modeling pipeline...")
    modeler = run_modeling_pipeline(data_path)
    
    print(f"\nModeling completed!")
    print(f"Models saved to: ../models/")