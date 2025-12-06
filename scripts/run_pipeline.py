#!/usr/bin/env python3
"""
Main pipeline script for insurance risk analytics
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import process_insurance_data
from hypothesis_testing import run_hypothesis_tests
from modeling import run_modeling_pipeline
from visualization import create_all_visualizations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_absolute_path(relative_path):
    """
    Convert relative path to absolute path based on project root
    
    Args:
        relative_path: Relative path from project root
    
    Returns:
        str: Absolute path
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to get project root (since script is in scripts/)
    project_root = os.path.dirname(script_dir)
    
    # Clean the relative path (remove leading ./ or ../)
    if relative_path.startswith('./'):
        relative_path = relative_path[2:]
    elif relative_path.startswith('../'):
        # Handle going up from project root
        parts = relative_path.split('/')
        up_levels = 0
        while parts and parts[0] == '..':
            up_levels += 1
            parts.pop(0)
        
        if up_levels > 0:
            # Go up specified levels from project root
            for _ in range(up_levels - 1):  # -1 because we already started from scripts/
                project_root = os.path.dirname(project_root)
        
        relative_path = '/'.join(parts)
    
    # Join with project root
    absolute_path = os.path.join(project_root, relative_path)
    
    # Normalize path
    absolute_path = os.path.normpath(absolute_path)
    
    logger.debug(f"Converted '{relative_path}' to '{absolute_path}'")
    return absolute_path

def run_complete_pipeline(input_path, output_base='../'):
    """
    Run complete insurance analytics pipeline
    
    Args:
        input_path: Path to raw data (can be relative or absolute)
        output_base: Base output directory (can be relative or absolute)
    """
    start_time = datetime.now()
    logger.info(f"Starting insurance analytics pipeline at {start_time}")
    
    # Convert paths to absolute
    input_path = get_absolute_path(input_path)
    output_base = get_absolute_path(output_base)
    
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output base: {output_base}")
    
    # Verify input file exists
    if not os.path.exists(input_path):
        logger.error(f"Input file does not exist: {input_path}")
        print(f"\n❌ ERROR: Input file not found: {input_path}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Please check the file exists and the path is correct.")
        return False
    
    # Create output directories
    data_dir = os.path.join(output_base, 'data', 'processed')
    reports_dir = os.path.join(output_base, 'reports')
    models_dir = os.path.join(output_base, 'models')
    figures_dir = os.path.join(reports_dir, 'figures')
    
    for directory in [data_dir, reports_dir, models_dir, figures_dir]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    output_path = os.path.join(data_dir, 'insurance_processed.csv')
    
    try:
        # Step 1: Data Processing
        logger.info("="*80)
        logger.info("STEP 1: DATA PROCESSING")
        logger.info("="*80)
        
        print("\n" + "="*80)
        print("STEP 1: DATA PROCESSING")
        print("="*80)
        print(f"Loading data from: {input_path}")
        
        processor, features, preprocessor = process_insurance_data(
            input_path, 
            output_path
        )
        
        print("✓ Data processing completed successfully")
        
        # Step 2: Hypothesis Testing
        logger.info("\n" + "="*80)
        logger.info("STEP 2: HYPOTHESIS TESTING")
        logger.info("="*80)
        
        print("\n" + "="*80)
        print("STEP 2: HYPOTHESIS TESTING")
        print("="*80)
        
        tester = run_hypothesis_tests(output_path, reports_dir)
        
        print("✓ Hypothesis testing completed successfully")
        
        # Step 3: Predictive Modeling
        logger.info("\n" + "="*80)
        logger.info("STEP 3: PREDICTIVE MODELING")
        logger.info("="*80)
        
        print("\n" + "="*80)
        print("STEP 3: PREDICTIVE MODELING")
        print("="*80)
        
        modeler = run_modeling_pipeline(output_path, features, models_dir)
        
        print("✓ Predictive modeling completed successfully")
        
        # Step 4: Visualization
        logger.info("\n" + "="*80)
        logger.info("STEP 4: VISUALIZATION")
        logger.info("="*80)
        
        print("\n" + "="*80)
        print("STEP 4: VISUALIZATION")
        print("="*80)
        
        create_all_visualizations(output_path, reports_dir)
        
        print("✓ Visualizations created successfully")
        
        # Calculate execution time
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        # Generate final report
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        print("\n" + "="*80)
        print("INSURANCE ANALYTICS PIPELINE - FINAL REPORT")
        print("="*80)
        
        print(f"\n📊 EXECUTION SUMMARY:")
        print(f"   Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Total Time: {execution_time}")
        
        print(f"\n📁 OUTPUTS GENERATED:")
        print(f"   1. Processed Data: {output_path}")
        print(f"   2. Hypothesis Test Results: {os.path.join(reports_dir, 'hypothesis_testing_results.csv')}")
        print(f"   3. Trained Models: {models_dir}/")
        print(f"   4. Visualizations: {figures_dir}/")
        print(f"   5. Interactive Dashboard: {os.path.join(reports_dir, 'interactive_dashboard.html')}")
        print(f"   6. Log File: pipeline.log")
        
        print(f"\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ PIPELINE FAILED: {e}")
        print(f"   Check pipeline.log for detailed error information.")
        return False

def run_single_step(step, input_path, output_base):
    """
    Run a single pipeline step
    
    Args:
        step: Which step to run ('process', 'test', 'model', 'visualize')
        input_path: Path to input data
        output_base: Base output directory
    """
    # Convert paths to absolute
    input_path = get_absolute_path(input_path)
    output_base = get_absolute_path(output_base)
    
    # Create output directories
    data_dir = os.path.join(output_base, 'data', 'processed')
    reports_dir = os.path.join(output_base, 'reports')
    models_dir = os.path.join(output_base, 'models')
    
    output_path = os.path.join(data_dir, 'insurance_processed.csv')
    
    if step == 'process':
        print("\n" + "="*80)
        print("RUNNING DATA PROCESSING STEP ONLY")
        print("="*80)
        
        # Make sure processed data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        processor, features, preprocessor = process_insurance_data(
            input_path, 
            output_path
        )
        
        print(f"✓ Data processing completed. Output saved to: {output_path}")
        
    elif step == 'test':
        print("\n" + "="*80)
        print("RUNNING HYPOTHESIS TESTING STEP ONLY")
        print("="*80)
        
        if not os.path.exists(output_path):
            print(f"❌ Processed data not found at: {output_path}")
            print("   Run data processing step first or provide correct path.")
            return False
        
        os.makedirs(reports_dir, exist_ok=True)
        tester = run_hypothesis_tests(output_path, reports_dir)
        print(f"✓ Hypothesis testing completed. Results saved to: {reports_dir}")
        
    elif step == 'model':
        print("\n" + "="*80)
        print("RUNNING PREDICTIVE MODELING STEP ONLY")
        print("="*80)
        
        if not os.path.exists(output_path):
            print(f"❌ Processed data not found at: {output_path}")
            print("   Run data processing step first or provide correct path.")
            return False
        
        # For modeling, we need features - we'll load the processor to get them
        import pandas as pd
        from data_processing import DataProcessor
        
        processor = DataProcessor(output_path)
        processor.load_data()
        processor.engineer_features()
        features = processor.select_features()
        
        os.makedirs(models_dir, exist_ok=True)
        modeler = run_modeling_pipeline(output_path, features, models_dir)
        print(f"✓ Predictive modeling completed. Models saved to: {models_dir}")
        
    elif step == 'visualize':
        print("\n" + "="*80)
        print("RUNNING VISUALIZATION STEP ONLY")
        print("="*80)
        
        if not os.path.exists(output_path):
            print(f"❌ Processed data not found at: {output_path}")
            print("   Run data processing step first or provide correct path.")
            return False
        
        os.makedirs(reports_dir, exist_ok=True)
        create_all_visualizations(output_path, reports_dir)
        print(f"✓ Visualizations created. Saved to: {reports_dir}")
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Insurance Risk Analytics Pipeline')
    parser.add_argument('--input', type=str, default='../data/raw/insurance.csv',
                       help='Path to input data file (relative to project root)')
    parser.add_argument('--output', type=str, default='../',
                       help='Base output directory (relative to project root)')
    parser.add_argument('--step', type=str, 
                       choices=['all', 'process', 'test', 'model', 'visualize'],
                       default='all', 
                       help='Which step to run: all, process, test, model, or visualize')
    
    args = parser.parse_args()
    
    # Get project root for display
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    print(f"Project root: {project_root}")
    print(f"Current directory: {os.getcwd()}")
    
    if args.step == 'all':
        print("\n" + "="*80)
        print("RUNNING COMPLETE INSURANCE ANALYTICS PIPELINE")
        print("="*80)
        success = run_complete_pipeline(args.input, args.output)
    else:
        success = run_single_step(args.step, args.input, args.output)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()