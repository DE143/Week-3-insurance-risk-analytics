"""
Hypothesis testing module for insurance risk analytics
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, chi2_contingency, mannwhitneyu, levene
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HypothesisTester:
    """Class for conducting statistical hypothesis tests"""
    
    def __init__(self, data):
        """
        Initialize HypothesisTester
        
        Args:
            data: pandas.DataFrame containing the data
        """
        self.data = data.copy()
        self.results = {}
        
    def test_province_risk_differences(self, alpha=0.05):
        """
        Test H0: There are no risk differences across provinces
        
        Args:
            alpha: Significance level
        
        Returns:
            dict: Test results
        """
        logger.info("Testing province risk differences...")
        
        if 'Province' not in self.data.columns or 'LossRatio' not in self.data.columns:
            logger.error("Required columns (Province, LossRatio) not found")
            return None
        
        results = {
            'hypothesis': 'Risk differences across provinces',
            'null_hypothesis': 'There are no risk differences across provinces',
            'alpha': alpha,
            'tests': {}
        }
        
        # Test 1: ANOVA for loss ratio
        provinces = self.data['Province'].unique()
        province_groups = [self.data[self.data['Province'] == prov]['LossRatio'].dropna() 
                          for prov in provinces]
        
        if len(province_groups) >= 2 and all(len(g) > 1 for g in province_groups):
            f_stat, p_value = f_oneway(*province_groups)
            results['tests']['anova_loss_ratio'] = {
                'test': 'ANOVA',
                'metric': 'Loss Ratio',
                'statistic': f_stat,
                'p_value': p_value,
                'reject_null': p_value < alpha,
                'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0: {'Significant' if p_value < alpha else 'No significant'} differences in loss ratio across provinces"
            }
            
            # Post-hoc Tukey's HSD if ANOVA is significant
            if p_value < alpha:
                try:
                    tukey = pairwise_tukeyhsd(
                        endog=self.data['LossRatio'].dropna(),
                        groups=self.data['Province'].dropna(),
                        alpha=alpha
                    )
                    results['tests']['tukey_hsd'] = {
                        'test': "Tukey's HSD",
                        'summary': str(tukey.summary())
                    }
                except Exception as e:
                    logger.warning(f"Tukey's HSD failed: {e}")
        
        # Test 2: Chi-square for claim frequency
        if 'HasClaim' in self.data.columns:
            contingency = pd.crosstab(self.data['Province'], self.data['HasClaim'])
            if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                results['tests']['chi_square_claim_frequency'] = {
                    'test': 'Chi-square',
                    'metric': 'Claim Frequency',
                    'statistic': chi2,
                    'p_value': p_value,
                    'reject_null': p_value < alpha,
                    'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0: {'Significant' if p_value < alpha else 'No significant'} differences in claim frequency across provinces"
                }
        
        self.results['province_risk'] = results
        return results
    
    def test_zipcode_risk_differences(self, alpha=0.05, min_samples=5):
        """
        Test H0: There are no risk differences between zip codes
        
        Args:
            alpha: Significance level
            min_samples: Minimum samples per zip code
        
        Returns:
            dict: Test results
        """
        logger.info("Testing zip code risk differences...")
        
        if 'PostalCode' not in self.data.columns or 'LossRatio' not in self.data.columns:
            logger.error("Required columns (PostalCode, LossRatio) not found")
            return None
        
        results = {
            'hypothesis': 'Risk differences between zip codes',
            'null_hypothesis': 'There are no risk differences between zip codes',
            'alpha': alpha,
            'tests': {}
        }
        
        # Get zip codes with sufficient data
        zip_counts = self.data['PostalCode'].value_counts()
        valid_zips = zip_counts[zip_counts >= min_samples].index.tolist()
        
        if len(valid_zips) >= 4:
            # Take top 10 zip codes
            top_zips = valid_zips[:10]
            zip_subset = self.data[self.data['PostalCode'].isin(top_zips)].copy()
            
            # Create high and low risk groups
            zip_risk = zip_subset.groupby('PostalCode')['LossRatio'].mean().sort_values()
            n_comparison = min(3, len(zip_risk) // 2)
            
            if n_comparison >= 1:
                high_risk_zips = zip_risk.tail(n_comparison).index.tolist()
                low_risk_zips = zip_risk.head(n_comparison).index.tolist()
                
                high_risk_data = zip_subset[zip_subset['PostalCode'].isin(high_risk_zips)]['LossRatio'].dropna()
                low_risk_data = zip_subset[zip_subset['PostalCode'].isin(low_risk_zips)]['LossRatio'].dropna()
                
                if len(high_risk_data) > 1 and len(low_risk_data) > 1:
                    # Welch's t-test (doesn't assume equal variances)
                    t_stat, p_value = ttest_ind(high_risk_data, low_risk_data, equal_var=False)
                    
                    results['tests']['t_test_high_vs_low'] = {
                        'test': "Welch's t-test",
                        'metric': 'Loss Ratio',
                        'high_risk_zips': high_risk_zips,
                        'low_risk_zips': low_risk_zips,
                        'high_risk_mean': high_risk_data.mean(),
                        'low_risk_mean': low_risk_data.mean(),
                        'difference': high_risk_data.mean() - low_risk_data.mean(),
                        'statistic': t_stat,
                        'p_value': p_value,
                        'reject_null': p_value < alpha,
                        'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0: {'Significant' if p_value < alpha else 'No significant'} risk difference between high and low risk zip codes"
                    }
        
        self.results['zipcode_risk'] = results
        return results
    
    def test_zipcode_margin_differences(self, alpha=0.05, min_samples=5):
        """
        Test H0: There is no significant margin difference between zip codes
        
        Args:
            alpha: Significance level
            min_samples: Minimum samples per zip code
        
        Returns:
            dict: Test results
        """
        logger.info("Testing zip code margin differences...")
        
        if 'PostalCode' not in self.data.columns or 'Margin' not in self.data.columns:
            logger.error("Required columns (PostalCode, Margin) not found")
            return None
        
        results = {
            'hypothesis': 'Margin differences between zip codes',
            'null_hypothesis': 'There is no significant margin difference between zip codes',
            'alpha': alpha,
            'tests': {}
        }
        
        # Get zip codes with sufficient data
        zip_counts = self.data['PostalCode'].value_counts()
        valid_zips = zip_counts[zip_counts >= min_samples].index.tolist()
        
        if len(valid_zips) >= 4:
            # Take top 10 zip codes
            top_zips = valid_zips[:10]
            zip_subset = self.data[self.data['PostalCode'].isin(top_zips)].copy()
            
            # Calculate margin statistics
            zip_margin = zip_subset.groupby('PostalCode')['Margin'].agg(['mean', 'std', 'count'])
            zip_margin = zip_margin.sort_values('mean', ascending=False)
            
            n_comparison = min(3, len(zip_margin) // 2)
            
            if n_comparison >= 1:
                high_margin_zips = zip_margin.head(n_comparison).index.tolist()
                low_margin_zips = zip_margin.tail(n_comparison).index.tolist()
                
                high_margin_data = zip_subset[zip_subset['PostalCode'].isin(high_margin_zips)]['Margin'].dropna()
                low_margin_data = zip_subset[zip_subset['PostalCode'].isin(low_margin_zips)]['Margin'].dropna()
                
                if len(high_margin_data) > 1 and len(low_margin_data) > 1:
                    # Mann-Whitney U test (non-parametric)
                    u_stat, p_value = mannwhitneyu(high_margin_data, low_margin_data, alternative='two-sided')
                    
                    results['tests']['mann_whitney_margin'] = {
                        'test': 'Mann-Whitney U',
                        'metric': 'Margin',
                        'high_margin_zips': high_margin_zips,
                        'low_margin_zips': low_margin_zips,
                        'high_margin_mean': high_margin_data.mean(),
                        'low_margin_mean': low_margin_data.mean(),
                        'difference': high_margin_data.mean() - low_margin_data.mean(),
                        'statistic': u_stat,
                        'p_value': p_value,
                        'reject_null': p_value < alpha,
                        'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0: {'Significant' if p_value < alpha else 'No significant'} margin difference between high and low margin zip codes"
                    }
        
        self.results['zipcode_margin'] = results
        return results
    
    def test_gender_risk_differences(self, alpha=0.05):
        """
        Test H0: There is no significant risk difference between Women and Men
        
        Args:
            alpha: Significance level
        
        Returns:
            dict: Test results
        """
        logger.info("Testing gender risk differences...")
        
        # Standardize gender column names
        gender_col = None
        for col in ['Gender', 'sex', 'gender']:
            if col in self.data.columns:
                gender_col = col
                break
        
        if not gender_col:
            logger.error("Gender column not found")
            return None
        
        if 'LossRatio' not in self.data.columns or 'HasClaim' not in self.data.columns:
            logger.error("Required columns (LossRatio, HasClaim) not found")
            return None
        
        results = {
            'hypothesis': 'Risk differences between Women and Men',
            'null_hypothesis': 'There is no significant risk difference between Women and Men',
            'alpha': alpha,
            'tests': {}
        }
        
        # Filter and standardize gender data
        gender_data = self.data.copy()
        gender_data[gender_col] = gender_data[gender_col].str.lower()
        gender_data = gender_data[gender_data[gender_col].isin(['male', 'female', 'm', 'f'])]
        
        if len(gender_data) == 0:
            logger.error("No valid gender data found")
            return results
        
        # Test 1: Claim frequency (Chi-square)
        contingency = pd.crosstab(gender_data[gender_col], gender_data['HasClaim'])
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            results['tests']['chi_square_claim_frequency'] = {
                'test': 'Chi-square',
                'metric': 'Claim Frequency',
                'statistic': chi2,
                'p_value': p_value,
                'reject_null': p_value < alpha,
                'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0: {'Significant' if p_value < alpha else 'No significant'} differences in claim frequency by gender"
            }
        
        # Test 2: Loss ratio (Mann-Whitney U)
        male_loss = gender_data[gender_data[gender_col].str.contains('male')]['LossRatio'].dropna()
        female_loss = gender_data[gender_data[gender_col].str.contains('female')]['LossRatio'].dropna()
        
        if len(male_loss) > 1 and len(female_loss) > 1:
            u_stat, p_value = mannwhitneyu(male_loss, female_loss, alternative='two-sided')
            results['tests']['mann_whitney_loss_ratio'] = {
                'test': 'Mann-Whitney U',
                'metric': 'Loss Ratio',
                'male_mean': male_loss.mean(),
                'female_mean': female_loss.mean(),
                'difference': male_loss.mean() - female_loss.mean(),
                'statistic': u_stat,
                'p_value': p_value,
                'reject_null': p_value < alpha,
                'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0: {'Significant' if p_value < alpha else 'No significant'} differences in loss ratio by gender"
            }
        
        # Test 3: Claim severity (only for policies with claims)
        if 'TotalClaims' in self.data.columns:
            male_claims = gender_data[
                (gender_data[gender_col].str.contains('male')) & 
                (gender_data['TotalClaims'] > 0)
            ]['TotalClaims'].dropna()
            
            female_claims = gender_data[
                (gender_data[gender_col].str.contains('female')) & 
                (gender_data['TotalClaims'] > 0)
            ]['TotalClaims'].dropna()
            
            if len(male_claims) > 1 and len(female_claims) > 1:
                t_stat, p_value = ttest_ind(male_claims, female_claims, equal_var=False)
                results['tests']['t_test_claim_severity'] = {
                    'test': "Welch's t-test",
                    'metric': 'Claim Severity',
                    'male_mean': male_claims.mean(),
                    'female_mean': female_claims.mean(),
                    'difference': male_claims.mean() - female_claims.mean(),
                    'statistic': t_stat,
                    'p_value': p_value,
                    'reject_null': p_value < alpha,
                    'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} H0: {'Significant' if p_value < alpha else 'No significant'} differences in claim severity by gender"
                }
        
        self.results['gender_risk'] = results
        return results
    
    def create_summary_report(self):
        """
        Create summary report of all hypothesis tests
        
        Returns:
            pandas.DataFrame: Summary report
        """
        summary_data = []
        
        for test_name, result in self.results.items():
            for test_key, test_result in result.get('tests', {}).items():
                summary_data.append({
                    'Hypothesis': result['hypothesis'],
                    'Test': test_result.get('test', 'N/A'),
                    'Metric': test_result.get('metric', 'N/A'),
                    'P-value': test_result.get('p_value', np.nan),
                    'Result': 'REJECT NULL' if test_result.get('reject_null', False) else 'FAIL TO REJECT',
                    'Interpretation': test_result.get('interpretation', 'N/A')
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            return summary_df
        else:
            return pd.DataFrame()
    
    def visualize_results(self, output_dir='../reports/figures'):
        """
        Create visualizations for hypothesis test results
        
        Args:
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Province Risk Differences
        if 'Province' in self.data.columns and 'LossRatio' in self.data.columns:
            plt.figure(figsize=(12, 6))
            province_avg = self.data.groupby('Province')['LossRatio'].mean().sort_values(ascending=False)
            bars = plt.bar(range(len(province_avg)), province_avg.values)
            plt.title('Average Loss Ratio by Province', fontsize=14)
            plt.xlabel('Province')
            plt.ylabel('Average Loss Ratio')
            plt.xticks(range(len(province_avg)), province_avg.index, rotation=45)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, province_avg.values)):
                plt.text(i, value + 0.001, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/h1_province_risk.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Gender Risk Differences
        gender_col = None
        for col in ['Gender', 'sex', 'gender']:
            if col in self.data.columns:
                gender_col = col
                break
        
        if gender_col and 'HasClaim' in self.data.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Claim frequency by gender
            gender_data = self.data.copy()
            gender_data[gender_col] = gender_data[gender_col].str.lower()
            gender_data = gender_data[gender_data[gender_col].isin(['male', 'female', 'm', 'f'])]
            
            if len(gender_data) > 0:
                contingency = pd.crosstab(gender_data[gender_col], gender_data['HasClaim'])
                contingency.plot(kind='bar', ax=axes[0], color=['lightblue', 'lightcoral'])
                axes[0].set_title('Claim Frequency by Gender', fontsize=14)
                axes[0].set_xlabel('Gender')
                axes[0].set_ylabel('Number of Policies')
                axes[0].legend(['No Claim', 'Claim'])
                axes[0].tick_params(axis='x', rotation=0)
            
            # Loss ratio by gender
            if 'LossRatio' in self.data.columns:
                sns.boxplot(x=gender_col, y='LossRatio', 
                           data=self.data[self.data['LossRatio'] < 2],  # Clip extreme values
                           ax=axes[1])
                axes[1].set_title('Loss Ratio Distribution by Gender', fontsize=14)
                axes[1].set_xlabel('Gender')
                axes[1].set_ylabel('Loss Ratio')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/h4_gender_risk.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Zip Code Analysis
        if 'PostalCode' in self.data.columns and 'LossRatio' in self.data.columns:
            # Get top 10 zip codes by sample size
            top_zips = self.data['PostalCode'].value_counts().head(10).index.tolist()
            if len(top_zips) >= 2:
                zip_subset = self.data[self.data['PostalCode'].isin(top_zips)].copy()
                
                plt.figure(figsize=(12, 6))
                zip_risk = zip_subset.groupby('PostalCode')['LossRatio'].mean().sort_values(ascending=False)
                plt.bar(range(len(zip_risk)), zip_risk.values)
                plt.title('Loss Ratio by Top 10 Zip Codes', fontsize=14)
                plt.xlabel('Zip Code')
                plt.ylabel('Average Loss Ratio')
                plt.xticks(range(len(zip_risk)), zip_risk.index, rotation=45)
                plt.tight_layout()
                plt.savefig(f'{output_dir}/h2_zipcode_risk.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def save_results(self, output_path='../reports/hypothesis_testing_results.csv'):
        """
        Save hypothesis test results to CSV
        
        Args:
            output_path: Path to save results
        
        Returns:
            str: Path where results were saved
        """
        summary_df = self.create_summary_report()
        
        if not summary_df.empty:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            summary_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            return output_path
        else:
            logger.warning("No results to save")
            return None


def run_hypothesis_tests(data_path, output_dir='../reports'):
    """
    Run complete hypothesis testing pipeline
    
    Args:
        data_path: Path to processed data
        output_dir: Output directory for results
    
    Returns:
        HypothesisTester: Tester object with results
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Initialize tester
    tester = HypothesisTester(data)
    
    # Run all tests
    logger.info("Starting hypothesis testing pipeline...")
    
    # Test 1: Province risk differences
    tester.test_province_risk_differences(alpha=0.05)
    
    # Test 2: Zip code risk differences
    tester.test_zipcode_risk_differences(alpha=0.05, min_samples=5)
    
    # Test 3: Zip code margin differences
    tester.test_zipcode_margin_differences(alpha=0.05, min_samples=5)
    
    # Test 4: Gender risk differences
    tester.test_gender_risk_differences(alpha=0.05)
    
    # Create summary report
    summary_df = tester.create_summary_report()
    if not summary_df.empty:
        print("\n" + "="*80)
        print("HYPOTHESIS TESTING SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
    
    # Create visualizations
    tester.visualize_results(f'{output_dir}/figures')
    
    # Save results
    tester.save_results(f'{output_dir}/hypothesis_testing_results.csv')
    
    # Business recommendations
    print("\n" + "="*80)
    print("BUSINESS RECOMMENDATIONS")
    print("="*80)
    
    # Check each hypothesis and provide recommendations
    for test_name, result in tester.results.items():
        print(f"\n{result['hypothesis']}:")
        
        # Check if any test rejected the null
        rejected = any(test.get('reject_null', False) for test in result.get('tests', {}).values())
        
        if rejected:
            print(f"  ✓ Evidence found to reject null hypothesis")
            print(f"  → Action: Consider {result['hypothesis'].lower()} in pricing and marketing")
        else:
            print(f"  ✗ No strong evidence to reject null hypothesis")
            print(f"  → Action: {result['hypothesis'].split(':')[0]} may not be a key differentiator")
    
    logger.info("Hypothesis testing pipeline completed")
    return tester


if __name__ == "__main__":
    # Example usage
    data_path = "../data/processed/insurance_processed.csv"
    
    print("Running hypothesis testing pipeline...")
    tester = run_hypothesis_tests(data_path)
    
    print(f"\nTesting completed!")
    print(f"Results saved to: ../reports/")