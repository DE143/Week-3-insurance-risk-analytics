"""
Visualization module for insurance risk analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import os

class InsuranceVisualizer:
    """Visualization class for insurance analytics"""
    
    def __init__(self, data):
        """
        Initialize InsuranceVisualizer
        
        Args:
            data: pandas.DataFrame containing the data
        """
        self.data = data.copy()
        self.set_style()
    
    def set_style(self, style='seaborn-v0_8-darkgrid'):
        """Set matplotlib style"""
        plt.style.use(style)
        sns.set_palette("husl")
    
    def create_distribution_plots(self, output_dir='../reports/figures'):
        """
        Create distribution plots for key variables
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        plots_created = 0
        
        # 1. Total Premium Distribution
        if 'TotalPremium' in self.data.columns:
            axes[0].hist(self.data['TotalPremium'], bins=50, color='blue', alpha=0.7, edgecolor='black')
            axes[0].set_title('Total Premium Distribution', fontsize=12)
            axes[0].set_xlabel('Premium (ZAR)')
            axes[0].set_ylabel('Frequency')
            axes[0].axvline(x=self.data['TotalPremium'].mean(), color='red', linestyle='--', 
                           label=f'Mean: ZAR {self.data["TotalPremium"].mean():,.0f}')
            axes[0].legend(fontsize=9)
            plots_created += 1
        
        # 2. Total Claims Distribution (claimants only)
        if 'TotalClaims' in self.data.columns:
            claimants = self.data[self.data['TotalClaims'] > 0]
            if len(claimants) > 0:
                axes[1].hist(claimants['TotalClaims'], bins=50, color='red', alpha=0.7, edgecolor='black')
                axes[1].set_title('Claim Amount Distribution (Claimants Only)', fontsize=12)
                axes[1].set_xlabel('Claim Amount (ZAR)')
                axes[1].set_ylabel('Frequency')
                axes[1].axvline(x=claimants['TotalClaims'].mean(), color='red', linestyle='--',
                               label=f'Mean: ZAR {claimants["TotalClaims"].mean():,.0f}')
                axes[1].legend(fontsize=9)
                plots_created += 1
        
        # 3. Loss Ratio Distribution
        if 'LossRatio' in self.data.columns:
            loss_ratio_clipped = self.data['LossRatio'].clip(0, 2)
            axes[2].hist(loss_ratio_clipped, bins=50, color='green', alpha=0.7, edgecolor='black')
            axes[2].set_title('Loss Ratio Distribution (Clipped 0-2)', fontsize=12)
            axes[2].set_xlabel('Loss Ratio')
            axes[2].set_ylabel('Frequency')
            axes[2].axvline(x=self.data['LossRatio'].mean(), color='red', linestyle='--',
                           label=f'Mean: {self.data["LossRatio"].mean():.2%}')
            axes[2].legend(fontsize=9)
            plots_created += 1
        
        # 4. Driver Age Distribution
        if 'DriverAge' in self.data.columns:
            axes[3].hist(self.data['DriverAge'], bins=30, color='purple', alpha=0.7, edgecolor='black')
            axes[3].set_title('Driver Age Distribution', fontsize=12)
            axes[3].set_xlabel('Driver Age (Years)')
            axes[3].set_ylabel('Frequency')
            axes[3].axvline(x=self.data['DriverAge'].mean(), color='red', linestyle='--',
                           label=f'Mean: {self.data["DriverAge"].mean():.1f} years')
            axes[3].legend(fontsize=9)
            plots_created += 1
        
        # 5. Vehicle Age Distribution
        if 'VehicleAge' in self.data.columns:
            axes[4].hist(self.data['VehicleAge'], bins=20, color='orange', alpha=0.7, edgecolor='black')
            axes[4].set_title('Vehicle Age Distribution', fontsize=12)
            axes[4].set_xlabel('Vehicle Age (Years)')
            axes[4].set_ylabel('Frequency')
            axes[4].axvline(x=self.data['VehicleAge'].mean(), color='red', linestyle='--',
                           label=f'Mean: {self.data["VehicleAge"].mean():.1f} years')
            axes[4].legend(fontsize=9)
            plots_created += 1
        
        # 6. Claim Frequency Distribution
        if 'HasClaim' in self.data.columns:
            claim_counts = self.data['HasClaim'].value_counts()
            axes[5].bar(['No Claim', 'Claim'], claim_counts.values, color=['lightblue', 'lightcoral'])
            axes[5].set_title('Claim Frequency Distribution', fontsize=12)
            axes[5].set_xlabel('Claim Status')
            axes[5].set_ylabel('Number of Policies')
            
            # Add percentage labels
            total = len(self.data)
            for i, count in enumerate(claim_counts.values):
                percentage = (count / total) * 100
                axes[5].text(i, count, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
            
            plots_created += 1
        
        # Remove empty subplots
        for i in range(plots_created, 6):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Distribution plots saved to {output_dir}/distribution_analysis.png")
    
    def create_correlation_matrix(self, output_dir='../reports/figures'):
        """
        Create correlation matrix heatmap
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Select numerical columns for correlation
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Keep only columns with sufficient variation
        numerical_cols = [col for col in numerical_cols if self.data[col].nunique() > 1]
        
        if len(numerical_cols) < 2:
            print("✗ Not enough numerical columns for correlation matrix")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.data[numerical_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": .8, 'label': 'Correlation'})
        plt.title('Correlation Matrix of Numerical Variables', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print top correlations
        print("\nTop Correlations (> 0.3 or < -0.3):")
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.3:
                    correlations.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))
        
        # Sort by absolute correlation value
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        for var1, var2, corr in correlations[:10]:  # Show top 10
            print(f"  {var1} vs {var2}: {corr:.3f}")
        
        print(f"✓ Correlation matrix saved to {output_dir}/correlation_matrix.png")
    
    def create_risk_heatmap(self, output_dir='../reports/figures'):
        """
        Create risk heatmap by province and vehicle type
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Check required columns
        required_cols = ['Province', 'VehicleType', 'LossRatio']
        if not all(col in self.data.columns for col in required_cols):
            print(f"✗ Missing columns for heatmap: {[c for c in required_cols if c not in self.data.columns]}")
            return
        
        # Create pivot table
        pivot_data = self.data.pivot_table(
            values='LossRatio', 
            index='Province', 
            columns='VehicleType', 
            aggfunc='mean',
            fill_value=0
        )
        
        # Create heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_data, annot=True, fmt='.2%', cmap='RdYlGn_r', center=0.1, 
                   linewidths=1, cbar_kws={'label': 'Loss Ratio'})
        plt.title('Risk Heatmap: Loss Ratio by Province and Vehicle Type', fontsize=16)
        plt.xlabel('Vehicle Type')
        plt.ylabel('Province')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/risk_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Risk heatmap saved to {output_dir}/risk_heatmap.png")
    
    def create_temporal_trends(self, output_dir='../reports/figures'):
        """
        Create temporal trend visualizations
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Check for date column
        date_col = None
        for col in ['TransactionMonth', 'TransactionDate', 'Date']:
            if col in self.data.columns:
                date_col = col
                break
        
        if date_col is None:
            print("✗ No date column found for temporal analysis")
            return
        
        # Convert to datetime
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        
        # Group by month
        self.data['MonthYear'] = self.data[date_col].dt.to_period('M')
        monthly_trends = self.data.groupby('MonthYear').agg({
            'TotalPremium': 'sum',
            'TotalClaims': 'sum',
            'HasClaim': 'mean' if 'HasClaim' in self.data.columns else None
        }).dropna(axis=1, how='all')
        
        monthly_trends['LossRatio'] = monthly_trends['TotalClaims'] / monthly_trends['TotalPremium']
        monthly_trends['MonthStr'] = monthly_trends.index.astype(str)
        
        # Create subplots
        n_plots = len(monthly_trends.columns) - 1  # Exclude MonthStr
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Premium trend
        if 'TotalPremium' in monthly_trends.columns:
            axes[plot_idx].plot(monthly_trends['MonthStr'], monthly_trends['TotalPremium'], 
                              marker='o', color='blue', linewidth=2)
            axes[plot_idx].set_title('Monthly Premium Collection', fontsize=14)
            axes[plot_idx].set_xlabel('Month')
            axes[plot_idx].set_ylabel('Total Premium (ZAR)')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].fill_between(monthly_trends['MonthStr'], monthly_trends['TotalPremium'], 
                                      alpha=0.3, color='blue')
            plot_idx += 1
        
        # Claims trend
        if 'TotalClaims' in monthly_trends.columns:
            axes[plot_idx].plot(monthly_trends['MonthStr'], monthly_trends['TotalClaims'], 
                              marker='o', color='red', linewidth=2)
            axes[plot_idx].set_title('Monthly Claims Payout', fontsize=14)
            axes[plot_idx].set_xlabel('Month')
            axes[plot_idx].set_ylabel('Total Claims (ZAR)')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].fill_between(monthly_trends['MonthStr'], monthly_trends['TotalClaims'], 
                                      alpha=0.3, color='red')
            plot_idx += 1
        
        # Loss ratio trend
        if 'LossRatio' in monthly_trends.columns:
            axes[plot_idx].plot(monthly_trends['MonthStr'], monthly_trends['LossRatio'], 
                              marker='o', color='orange', linewidth=2)
            axes[plot_idx].set_title('Monthly Loss Ratio Trend', fontsize=14)
            axes[plot_idx].set_xlabel('Month')
            axes[plot_idx].set_ylabel('Loss Ratio')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            axes[plot_idx].grid(True, alpha=0.3)
            if 'LossRatio' in self.data.columns:
                axes[plot_idx].axhline(y=self.data['LossRatio'].mean(), color='red', 
                                     linestyle='--', alpha=0.7, label='Overall Average')
                axes[plot_idx].legend()
            axes[plot_idx].fill_between(monthly_trends['MonthStr'], monthly_trends['LossRatio'], 
                                      alpha=0.3, color='orange')
            plot_idx += 1
        
        # Claim frequency trend
        if 'HasClaim' in monthly_trends.columns:
            axes[plot_idx].plot(monthly_trends['MonthStr'], monthly_trends['HasClaim'], 
                              marker='o', color='green', linewidth=2)
            axes[plot_idx].set_title('Monthly Claim Frequency', fontsize=14)
            axes[plot_idx].set_xlabel('Month')
            axes[plot_idx].set_ylabel('Claim Frequency')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            axes[plot_idx].grid(True, alpha=0.3)
            if 'HasClaim' in self.data.columns:
                axes[plot_idx].axhline(y=self.data['HasClaim'].mean(), color='red', 
                                     linestyle='--', alpha=0.7, label='Overall Average')
                axes[plot_idx].legend()
            axes[plot_idx].fill_between(monthly_trends['MonthStr'], monthly_trends['HasClaim'], 
                                      alpha=0.3, color='green')
            plot_idx += 1
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/temporal_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Temporal trends saved to {output_dir}/temporal_trends.png")
    
    def create_interactive_dashboard(self, output_dir='../reports'):
        """
        Create interactive Plotly dashboard
        
        Args:
            output_dir: Directory to save dashboard
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Premium Distribution', 'Claim Distribution', 'Loss Ratio by Province',
                           'Risk Factors', 'Temporal Trends', 'Correlation Heatmap'),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}, {'type': 'bar'}],
                   [{'type': 'box'}, {'type': 'scatter'}, {'type': 'heatmap'}]]
        )
        
        # 1. Premium Distribution
        if 'TotalPremium' in self.data.columns:
            fig.add_trace(
                go.Histogram(x=self.data['TotalPremium'], name='Premium', nbinsx=50,
                            marker_color='blue', opacity=0.7),
                row=1, col=1
            )
        
        # 2. Claim Distribution (claimants only)
        if 'TotalClaims' in self.data.columns:
            claimants = self.data[self.data['TotalClaims'] > 0]
            if len(claimants) > 0:
                fig.add_trace(
                    go.Histogram(x=claimants['TotalClaims'], name='Claims', nbinsx=50,
                                marker_color='red', opacity=0.7),
                    row=1, col=2
                )
        
        # 3. Loss Ratio by Province
        if 'Province' in self.data.columns and 'LossRatio' in self.data.columns:
            province_loss = self.data.groupby('Province')['LossRatio'].mean().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(x=province_loss.index, y=province_loss.values,
                      name='Loss Ratio', marker_color='green'),
                row=1, col=3
            )
        
        # 4. Risk Factors Box Plot
        if 'RiskFactor' in self.data.columns:
            # Create risk categories
            # Check if we have enough unique values
            if self.data['RiskFactor'].nunique() >= 4:
                self.data['RiskCategory'] = pd.qcut(self.data['RiskFactor'], q=4, 
                                                   labels=['Low', 'Medium', 'High', 'Very High'],
                                                   duplicates='drop')
            else:
                # Use equal spacing if not enough unique values
                min_val = self.data['RiskFactor'].min()
                max_val = self.data['RiskFactor'].max()
                bins = np.linspace(min_val, max_val, 5)
                self.data['RiskCategory'] = pd.cut(self.data['RiskFactor'], bins=bins, 
                                                  labels=['Low', 'Medium', 'High', 'Very High'],
                                                  include_lowest=True)

            if 'TotalPremium' in self.data.columns:
                fig.add_trace(
                    go.Box(x=self.data['RiskCategory'], y=self.data['TotalPremium'],
                          name='Premium by Risk', marker_color='purple'),
                    row=2, col=1
                )
        
        # 5. Temporal Trends
        date_col = None
        for col in ['TransactionMonth', 'TransactionDate', 'Date']:
            if col in self.data.columns:
                date_col = col
                break
        
        if date_col and 'TotalPremium' in self.data.columns:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            monthly_data = self.data.groupby(self.data[date_col].dt.to_period('M')).agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum' if 'TotalClaims' in self.data.columns else None
            }).dropna(axis=1, how='all')
            
            monthly_data.index = monthly_data.index.astype(str)
            
            fig.add_trace(
                go.Scatter(x=monthly_data.index, y=monthly_data['TotalPremium'],
                          name='Monthly Premium', mode='lines+markers',
                          line=dict(color='blue', width=2)),
                row=2, col=2
            )
            
            if 'TotalClaims' in monthly_data.columns:
                fig.add_trace(
                    go.Scatter(x=monthly_data.index, y=monthly_data['TotalClaims'],
                              name='Monthly Claims', mode='lines+markers',
                              line=dict(color='red', width=2)),
                    row=2, col=2
                )
        
        # 6. Correlation Heatmap
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numerical_cols = [col for col in numerical_cols if self.data[col].nunique() > 1]
        
        if len(numerical_cols) >= 2:
            corr_matrix = self.data[numerical_cols].corr()
            
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values,
                          x=corr_matrix.columns,
                          y=corr_matrix.index,
                          colorscale='RdBu',
                          zmin=-1, zmax=1,
                          colorbar=dict(title="Correlation")),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title_text="Insurance Risk Analytics Dashboard",
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        # Save as HTML
        dashboard_path = f'{output_dir}/interactive_dashboard.html'
        fig.write_html(dashboard_path)
        
        print(f"✓ Interactive dashboard saved to {dashboard_path}")
        return dashboard_path
    
    def create_model_performance_charts(self, model_results, output_dir='../reports/figures'):
        """
        Create model performance visualizations
        
        Args:
            model_results: Dictionary containing model results
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create model comparison plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Regression Models Comparison
        if 'claim_severity' in model_results:
            severity_data = []
            for model_name, result in model_results['claim_severity'].items():
                if 'error' not in result:
                    severity_data.append({
                        'Model': model_name,
                        'R²': result.get('r2', 0),
                        'RMSE': result.get('rmse', 0)
                    })
            
            if severity_data:
                severity_df = pd.DataFrame(severity_data)
                axes[0].bar(range(len(severity_df)), severity_df['R²'].values)
                axes[0].set_title('Claim Severity Models (R² Score)', fontsize=14)
                axes[0].set_xlabel('Model')
                axes[0].set_ylabel('R² Score')
                axes[0].set_xticks(range(len(severity_df)))
                axes[0].set_xticklabels(severity_df['Model'], rotation=45)
                axes[0].set_ylim([0, 1])
                
                # Add value labels
                for i, v in enumerate(severity_df['R²'].values):
                    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. Classification Models Comparison
        if 'claim_probability' in model_results:
            probability_data = []
            for model_name, result in model_results['claim_probability'].items():
                if 'error' not in result:
                    probability_data.append({
                        'Model': model_name,
                        'ROC AUC': result.get('roc_auc', 0),
                        'F1 Score': result.get('f1', 0)
                    })
            
            if probability_data:
                probability_df = pd.DataFrame(probability_data)
                x = np.arange(len(probability_df))
                width = 0.35
                
                axes[1].bar(x - width/2, probability_df['ROC AUC'].values, width, label='ROC AUC')
                axes[1].bar(x + width/2, probability_df['F1 Score'].values, width, label='F1 Score')
                axes[1].set_title('Claim Probability Models', fontsize=14)
                axes[1].set_xlabel('Model')
                axes[1].set_ylabel('Score')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(probability_df['Model'], rotation=45)
                axes[1].set_ylim([0, 1])
                axes[1].legend()
                
                # Add value labels
                for i, (roc_auc, f1) in enumerate(zip(probability_df['ROC AUC'].values, probability_df['F1 Score'].values)):
                    axes[1].text(i - width/2, roc_auc + 0.01, f'{roc_auc:.3f}', ha='center', va='bottom', fontsize=9)
                    axes[1].text(i + width/2, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Model performance charts saved to {output_dir}/model_performance.png")
    
    def create_outlier_detection_plots(self, output_dir='../reports/figures'):
        """
        Create outlier detection box plots
        
        Args:
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Select key numerical columns
        key_columns = []
        for col in ['TotalPremium', 'TotalClaims', 'LossRatio', 'DriverAge', 'VehicleAge']:
            if col in self.data.columns:
                key_columns.append(col)
        
        if len(key_columns) == 0:
            print("✗ No numerical columns for outlier detection")
            return
        
        # Create subplots
        n_cols = min(3, len(key_columns))
        n_rows = (len(key_columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes = axes.ravel()
        
        for i, col in enumerate(key_columns):
            if i < len(axes):
                # Prepare data (clip extreme values for better visualization)
                if col == 'LossRatio':
                    data = self.data[col].clip(0, 2)
                elif col == 'TotalClaims':
                    data = self.data[self.data[col] > 0][col] if (self.data[col] > 0).any() else self.data[col]
                else:
                    data = self.data[col]
                
                # Create boxplot
                bp = axes[i].boxplot(data.dropna(), patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                axes[i].set_title(f'{col} Outliers', fontsize=12)
                axes[i].set_ylabel(col)
                
                # Calculate and display outlier statistics
                if len(data.dropna()) > 0:
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                    outlier_pct = len(outliers) / len(data) * 100
                    
                    axes[i].text(0.95, 0.95, f'{outlier_pct:.1f}% outliers', 
                               transform=axes[i].transAxes, ha='right', va='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Remove empty subplots
        for i in range(len(key_columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/outlier_detection.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Outlier detection plots saved to {output_dir}/outlier_detection.png")


def create_all_visualizations(data_path, output_dir='../reports'):
    """
    Create all visualizations from data
    
    Args:
        data_path: Path to processed data
        output_dir: Output directory for visualizations
    
    Returns:
        dict: Paths to created visualizations
    """
    # Load data
    data = pd.read_csv(data_path)
    
    # Initialize visualizer
    visualizer = InsuranceVisualizer(data)
    
    print("Creating visualizations...")
    
    # Create all visualizations
    visualizer.create_distribution_plots(f'{output_dir}/figures')
    visualizer.create_correlation_matrix(f'{output_dir}/figures')
    visualizer.create_risk_heatmap(f'{output_dir}/figures')
    visualizer.create_temporal_trends(f'{output_dir}/figures')
    visualizer.create_outlier_detection_plots(f'{output_dir}/figures')
    
    # Create interactive dashboard
    dashboard_path = visualizer.create_interactive_dashboard(output_dir)
    
    print("\n✓ All visualizations created successfully!")
    print(f"  Static plots saved to: {output_dir}/figures/")
    print(f"  Interactive dashboard: {dashboard_path}")
    
    return {
        'static_plots': f'{output_dir}/figures',
        'interactive_dashboard': dashboard_path
    }


if __name__ == "__main__":
    # Example usage
    data_path = "../data/processed/insurance_processed.csv"
    
    print("Creating insurance analytics visualizations...")
    paths = create_all_visualizations(data_path)
    
    print(f"\nVisualization pipeline completed!")