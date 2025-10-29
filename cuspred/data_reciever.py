import pandas as pd
from pathlib import Path

def load_datasets():
    # Use relative path to go up one level
    project_root = Path(__file__).parent.parent
    ds_path = project_root / 'ds'
    
    print(f"Looking for files in: {ds_path.absolute()}")
    
    if not ds_path.exists():
        raise FileNotFoundError(f"Directory '{ds_path}' does not exist!")
    
    datasets = {}
    
    file_configs = [
        ('menu_ingredient.JSON', pd.read_json),
        ('date_seasons.csv', pd.read_csv),
        ('sales_history.csv', pd.read_csv),
        ('date_event.csv', pd.read_csv),
        ('date_weekend.csv', pd.read_csv)
    ]
    
    for filename, reader_func in file_configs:
        file_path = ds_path / filename
        if file_path.exists():
            try:
                datasets[filename.split('.')[0]] = reader_func(file_path)
                print(f"✓ Loaded: {filename}")
            except Exception as e:
                print(f"✗ Error reading {filename}: {e}")
        else:
            print(f"✗ File not found: {filename}")
    
    return datasets

datasets = load_datasets()