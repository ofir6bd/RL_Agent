#!/usr/bin/env python3
"""
Test script to verify RL Agent UI setup
Checks all components and provides diagnostics
"""
import sys
from pathlib import Path

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ✗ WARNING: Python 3.8+ required")
        return False
    return True

def check_dependencies():
    """Check required Python packages."""
    packages = {
        'flask': 'Flask web framework',
        'flask_cors': 'CORS support',
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'yaml': 'PyYAML',
    }
    
    all_ok = True
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✓ {package:15} ({description})")
        except ImportError:
            print(f"✗ {package:15} ({description}) - MISSING")
            all_ok = False
    
    return all_ok

def check_files():
    """Check required files and folders."""
    files_to_check = {
        'app.py': 'Flask backend',
        'ui.html': 'Frontend HTML',
        'src/env/dose_env.py': 'Environment code',
        'src/agents/ppo.py': 'PPO agent code',
        'configs/default.yaml': 'Configuration',
    }
    
    all_ok = True
    for filepath, description in files_to_check.items():
        if Path(filepath).exists():
            print(f"✓ {filepath:30} ({description})")
        else:
            print(f"✗ {filepath:30} ({description}) - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_data():
    """Check data directories."""
    data_dirs = {
        'data/processed': 'Processed data',
        'data/processed/validation': 'Validation set',
        'runs': 'Model checkpoints',
    }
    
    all_ok = True
    print()
    print("Data directories:")
    for dirname, description in data_dirs.items():
        dirpath = Path(dirname)
        if dirpath.exists():
            items = list(dirpath.iterdir())
            print(f"✓ {dirname:30} ({description}) - {len(items)} items")
        else:
            print(f"✗ {dirname:30} ({description}) - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_models():
    """Check if trained models exist."""
    models = {
        'runs/best.pt': 'Best model',
        'runs/last.pt': 'Last checkpoint',
    }
    
    all_ok = True
    print()
    print("Model files:")
    for filepath, description in models.items():
        if Path(filepath).exists():
            size_mb = Path(filepath).stat().st_size / (1024*1024)
            print(f"✓ {filepath:30} ({description}) - {size_mb:.1f} MB")
        else:
            print(f"✗ {filepath:30} ({description}) - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_patients():
    """Check available patients."""
    validation_dir = Path('data/processed/validation')
    if not validation_dir.exists():
        print("\n✗ Validation data directory not found")
        return False
    
    patients = sorted([p.name for p in validation_dir.iterdir() if p.is_dir()])
    print()
    print(f"✓ Found {len(patients)} patients in validation set")
    if patients:
        print(f"  Examples: {', '.join(patients[:5])}")
        if len(patients) > 5:
            print(f"  ... and {len(patients) - 5} more")
    
    return len(patients) > 0

def main():
    print()
    print("=" * 60)
    print("  RL Agent UI - System Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("Python Dependencies", check_dependencies),
        ("Project Files", check_files),
        ("Data Directories", check_data),
        ("Model Files", check_models),
        ("Patient Data", check_patients),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\nChecking {name}...")
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Error during check: {e}")
            results.append((name, False))
    
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print()
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
        if not result:
            all_passed = False
    
    print()
    
    if all_passed:
        print("✓ All checks passed! System is ready to use.")
        print()
        print("Next steps:")
        print("  1. Windows:  double-click run_ui.bat")
        print("  2. Mac/Linux: ./run_ui.sh")
        print("  3. Open ui.html in your browser")
        print()
        return 0
    else:
        print("✗ Some checks failed. See details above.")
        print()
        print("Common fixes:")
        print("  - Install dependencies: pip install -r requirements_ui.txt")
        print("  - Preprocess data: python scripts/preprocess.py")
        print("  - Check configs: python app.py --debug")
        print()
        return 1

if __name__ == '__main__':
    sys.exit(main())
