"""
Data processing module for insurance risk analytics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing and feature engineering class"""
    
    def __init__(self, data_path=None):
        """
        Initialize DataProcessor
        
        Args:
            data_path: Path to the data file
        """
        self.data_path = data_path
        self.df = None
        self.features = None
        self.preprocessor = None
        self.categorical_cols = None
        self.numerical_cols = None
    
    def load_data(self, data_path=None):
        """
        Load data from CSV file
        
        Args:
            data_path: Path to data file (optional)
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        if data_path:
            self.data_path = data_path
        
        if not self.data_path:
            raise ValueError("No data path provided")
        
        try:
            # Convert relative path to absolute if needed
            if not os.path.isabs(self.data_path):
                # Get the directory of the current script (data_processing.py)
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # Go up one level to project root
                project_root = os.path.dirname(current_dir)
                # Construct absolute path
                abs_data_path = os.path.join(project_root, self.data_path.lstrip("./"))
                
                # Try the constructed path
                if os.path.exists(abs_data_path):
                    self.data_path = abs_data_path
                else:
                    # Try relative to current directory
                    self.data_path = os.path.abspath(self.data_path)
            
            # Debug: print the path being used
            logger.info(f"Looking for data at: {self.data_path}")
            logger.info(f"File exists: {os.path.exists(self.data_path)}")
            
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully from {self.data_path}")
            logger.info(f"Dataset shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            
            # Try alternative paths
            alternative_paths = [
                self.data_path,
                os.path.join("data", "raw", "insurance.csv"),
                os.path.join("..", "data", "raw", "insurance.csv"),
                os.path.abspath(os.path.join("data", "raw", "insurance.csv")),
                os.path.abspath(os.path.join("..", "data", "raw", "insurance.csv"))
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    logger.info(f"Found file at alternative path: {alt_path}")
                    self.df = pd.read_csv(alt_path)
                    self.data_path = alt_path
                    return self.df
            
            raise
    
    def explore_data(self):
        """Explore and display data information"""
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return
        
        print("="*80)
        print("DATA EXPLORATION")
        print("="*80)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"\nColumns ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns.tolist(), 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\nData Types:")
        print(self.df.dtypes)
        
        print(f"\nMissing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            for col, count in missing[missing > 0].items():
                print(f"  - {col}: {count} ({count/len(self.df):.1%})")
        else:
            print("  No missing values")
        
        print(f"\nDescriptive Statistics:")
        print(self.df.describe())
        
        return self.df
    
    def engineer_features(self):
        """
        Create new features from existing data
        
        Returns:
            pandas.DataFrame: Data with engineered features
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return
        
        logger.info("Engineering features...")
        
        # Create basic insurance metrics
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium'].replace(0, 1)
            self.df['HasClaim'] = (self.df['TotalClaims'] > 0).astype(int)
            logger.info("Created LossRatio and HasClaim")
        
        # Create VehicleAge if RegistrationYear exists
        if 'RegistrationYear' in self.df.columns:
            self.df['VehicleAge'] = 2023 - self.df['RegistrationYear']
            logger.info("Created VehicleAge")
        
        # Create risk score
        risk_factors = []
        
        # Add risk factors based on available columns
        if 'smoker' in self.df.columns:
            self.df['SmokerRisk'] = (self.df['smoker'] == 'yes') * 0.3
            risk_factors.append('SmokerRisk')
        
        if 'DriverAge' in self.df.columns:
            self.df['AgeRisk'] = (self.df['DriverAge'] > 60) * 0.2
            risk_factors.append('AgeRisk')
        
        if 'VehicleAge' in self.df.columns:
            self.df['VehicleAgeRisk'] = (self.df['VehicleAge'] > 10) * 0.2
            risk_factors.append('VehicleAgeRisk')
        
        if 'RiskScore' in self.df.columns:
            self.df['BMIRisk'] = (self.df['RiskScore'] > 30) * 0.1
            risk_factors.append('BMIRisk')
        
        if 'NumberOfDependents' in self.df.columns:
            self.df['DependentsRisk'] = (self.df['NumberOfDependents'] > 2) * 0.1
            risk_factors.append('DependentsRisk')
        
        if 'TotalPremium' in self.df.columns:
            premium_75 = self.df['TotalPremium'].quantile(0.75)
            self.df['PremiumRisk'] = (self.df['TotalPremium'] > premium_75) * 0.1
            risk_factors.append('PremiumRisk')
        
        # Create overall risk score
        if risk_factors:
            self.df['RiskScore_Overall'] = self.df[risk_factors].sum(axis=1)
            if self.df['RiskScore_Overall'].mean() > 0:
                self.df['RiskFactor'] = self.df['RiskScore_Overall'] / self.df['RiskScore_Overall'].mean()
            else:
                self.df['RiskFactor'] = 1.0
            logger.info(f"Created RiskFactor from {len(risk_factors)} factors")
        
        # Create interaction features
        if 'DriverAge' in self.df.columns and 'VehicleAge' in self.df.columns:
            self.df['Age_VehicleAge_Interaction'] = self.df['DriverAge'] * self.df['VehicleAge']
            logger.info("Created Age_VehicleAge_Interaction")
        
        # Calculate profit margin
        if 'TotalPremium' in self.df.columns and 'TotalClaims' in self.df.columns:
            self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
            logger.info("Created Margin")
        
        logger.info(f"Total features after engineering: {len(self.df.columns)}")
        return self.df
    
    def select_features(self, custom_features=None):
        """
        Select features for modeling
        
        Args:
            custom_features: List of custom features to use (optional)
        
        Returns:
            list: Selected features
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return
        
        # Define default feature categories
        default_feature_categories = {
            'Client': ['Gender', 'DriverAge', 'RiskScore', 'NumberOfDependents', 'smoker'],
            'Geographic': ['Province', 'PostalCode'],
            'Vehicle': ['VehicleType', 'Make', 'Model', 'VehicleAge'],
            'Insurance': ['SumInsured', 'TotalPremium', 'PremiumPerInsured'],
            'Engineered': ['RiskFactor', 'Age_VehicleAge_Interaction', 'ClaimRatio', 'Margin']
        }
        
        if custom_features:
            self.features = [col for col in custom_features if col in self.df.columns]
        else:
            # Select features that exist in the dataframe
            self.features = []
            for category, feature_list in default_feature_categories.items():
                available = [col for col in feature_list if col in self.df.columns]
                if available:
                    self.features.extend(available)
                    logger.info(f"Selected {len(available)} {category} features")
        
        # Add any missing essential features
        essential_features = ['HasClaim', 'LossRatio']
        for feature in essential_features:
            if feature in self.df.columns and feature not in self.features:
                self.features.append(feature)
        
        logger.info(f"Selected {len(self.features)} features for modeling")
        return self.features
    
    def create_preprocessor(self, features=None):
        """
        Create preprocessing pipeline
        
        Args:
            features: List of features to preprocess (optional)
        
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        if features:
            self.features = features
        
        if self.features is None:
            logger.error("No features selected. Call select_features() first.")
            return
        
        # Get the feature data
        X = self.df[self.features].copy()
        
        # Identify categorical and numerical columns
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        logger.info(f"Categorical features: {len(self.categorical_cols)}")
        logger.info(f"Numerical features: {len(self.numerical_cols)}")
        
        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])
        
        logger.info("Preprocessing pipeline created successfully")
        return self.preprocessor
    
    def prepare_targets(self):
        """
        Prepare target variables for modeling
        
        Returns:
            dict: Dictionary of target variables
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return
        
        targets = {
            'TotalClaims': 'TotalClaims' if 'TotalClaims' in self.df.columns else None,
            'TotalPremium': 'TotalPremium' if 'TotalPremium' in self.df.columns else None,
            'HasClaim': 'HasClaim' if 'HasClaim' in self.df.columns else None,
            'LossRatio': 'LossRatio' if 'LossRatio' in self.df.columns else None,
            'Margin': 'Margin' if 'Margin' in self.df.columns else None
        }
        
        # Remove None values
        targets = {k: v for k, v in targets.items() if v is not None}
        
        logger.info(f"Prepared {len(targets)} target variables")
        return targets
    
    def save_processed_data(self, output_path):
        """
        Save processed data to CSV
        
        Args:
            output_path: Path to save processed data
        
        Returns:
            str: Path where data was saved
        """
        if self.df is None:
            logger.error("No data to save. Process data first.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            self.df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def get_feature_matrix(self, features=None):
        """
        Get feature matrix X
        
        Args:
            features: List of features to use (optional)
        
        Returns:
            pandas.DataFrame: Feature matrix
        """
        if features:
            self.features = features
        
        if self.features is None:
            logger.error("No features selected. Call select_features() first.")
            return
        
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return
        
        return self.df[self.features].copy()
    
    def get_target_vector(self, target_name):
        """
        Get target vector y
        
        Args:
            target_name: Name of target variable
        
        Returns:
            pandas.Series: Target vector
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return
        
        if target_name not in self.df.columns:
            logger.error(f"Target {target_name} not found in data")
            return
        
        return self.df[target_name].copy()


def process_insurance_data(input_path, output_path, custom_features=None):
    """
    Process insurance data pipeline
    
    Args:
        input_path: Path to input data
        output_path: Path to save processed data
        custom_features: Custom feature list (optional)
    
    Returns:
        tuple: (DataProcessor, features, preprocessor)
    """
    # Initialize processor
    processor = DataProcessor(input_path)
    
    # Load data
    processor.load_data()
    
    # Explore data
    processor.explore_data()
    
    # Engineer features
    processor.engineer_features()
    
    # Select features
    features = processor.select_features(custom_features)
    
    # Create preprocessor
    preprocessor = processor.create_preprocessor()
    
    # Save processed data
    processor.save_processed_data(output_path)
    
    return processor, features, preprocessor


if __name__ == "__main__":
    # Example usage
    input_path = "../data/raw/insurance.csv"
    output_path = "../data/processed/insurance_processed.csv"
    
    print("Running data processing pipeline...")
    processor, features, preprocessor = process_insurance_data(input_path, output_path)
    
    print(f"\nProcessing completed!")
    print(f"Features selected: {len(features)}")
    print(f"Data saved to: {output_path}")