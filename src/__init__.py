"""
Insurance Risk Analytics Package
"""

__version__ = '1.0.0'
__author__ = 'Insurance Analytics Team'

from .data_processing import DataProcessor, process_insurance_data
from .hypothesis_testing import HypothesisTester, run_hypothesis_tests
from .modeling import InsuranceModeler, run_modeling_pipeline
from .visualization import InsuranceVisualizer, create_all_visualizations

__all__ = [
    'DataProcessor',
    'process_insurance_data',
    'HypothesisTester',
    'run_hypothesis_tests',
    'InsuranceModeler',
    'run_modeling_pipeline',
    'InsuranceVisualizer',
    'create_all_visualizations'
]