#!/usr/bin/env python3
"""
Data Version Control Setup Script
"""

import os
import subprocess
import sys

def setup_dvc():
    """Initialize DVC for the project"""
    
    print("Setting up Data Version Control...")
    
    # 1. Initialize DVC
    print("\n1. Initializing DVC...")
    subprocess.run(["dvc", "init"], check=True)
    
    # 2. Create local remote storage
    print("\n2. Setting up local remote storage...")
    storage_path = os.path.join(os.getcwd(), "dvc_storage")
    os.makedirs(storage_path, exist_ok=True)
    
    subprocess.run(["dvc", "remote", "add", "-d", "localstorage", storage_path], check=True)
    
    # 3. Configure DVC
    print("\n3. Configuring DVC...")
    subprocess.run(["dvc", "config", "core.autostage", "true"], check=True)
    
    # 4. Create .dvcignore file
    print("\n4. Creating .dvcignore...")
    dvcignore_content = """# DVC ignore patterns
*.tmp
*.temp
*.log
*.pyc
__pycache__/
.ipynb_checkpoints/
.DS_Store
.env
venv/
env/
.venv/
"""
    
    with open(".dvcignore", "w") as f:
        f.write(dvcignore_content)
    
    # 5. Add data directory to DVC
    print("\n5. Adding data to DVC...")
    data_dir = "data/raw"
    if os.path.exists(data_dir):
        subprocess.run(["dvc", "add", data_dir], check=True)
    else:
        print(f"Warning: {data_dir} not found. Creating sample structure...")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create sample data file
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'id': range(100),
            'value': np.random.randn(100)
        })
        sample_data.to_csv(os.path.join(data_dir, "sample_data.csv"), index=False)
        
        subprocess.run(["dvc", "add", data_dir], check=True)
    
    # 6. Create DVC pipeline
    print("\n6. Creating DVC pipeline...")
    dvc_yaml = """stages:
  process_data:
    cmd: python src/data_processing.py
    deps:
      - src/data_processing.py
      - data/raw
    params:
      - config/params.yaml
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
  
  train_model:
    cmd: python src/modeling.py
    deps:
      - src/modeling.py
      - data/processed/train.csv
    params:
      - config/params.yaml
    outs:
      - models/model.pkl
      - models/metrics.json
  
  evaluate:
    cmd: python src/evaluation.py
    deps:
      - src/evaluation.py
      - models/model.pkl
      - data/processed/test.csv
    metrics:
      - models/metrics.json:
          cache: false
"""
    
    with open("dvc.yaml", "w") as f:
        f.write(dvc_yaml)
    
    # Create params file
    params_content = """data:
  raw_path: data/raw
  processed_path: data/processed
  test_size: 0.2
  random_state: 42

model:
  n_estimators: 100
  max_depth: 10
  learning_rate: 0.1
  random_state: 42

features:
  categorical: ["Province", "Gender", "VehicleType"]
  numerical: ["VehicleAge", "SumInsured"]
  target: "TotalClaims"
"""
    
    os.makedirs("config", exist_ok=True)
    with open("config/params.yaml", "w") as f:
        f.write(params_content)
    
    print("\n7. Committing DVC files to Git...")
    subprocess.run(["git", "add", ".dvc", ".dvcignore", "dvc.yaml", "config/params.yaml", "data/.gitignore"], check=True)
    subprocess.run(["git", "commit", "-m", "Initialize DVC setup"], check=True)
    
    print("\n8. Pushing to DVC remote...")
    subprocess.run(["dvc", "push"], check=True)
    
    print("\n✅ DVC setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: dvc repro  # to execute the pipeline")
    print("2. Run: dvc metrics show  # to view metrics")
    print("3. Run: dvc dag  # to view pipeline graph")

if __name__ == "__main__":
    setup_dvc()