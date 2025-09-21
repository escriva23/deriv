# install_enhancements.py - Installation script for pattern-aware enhancements
import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🚀 Installing Pattern-Aware AI Trading Enhancements")
    print("=" * 60)
    
    # Required packages
    packages = [
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0", 
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "lightgbm>=4.0.0",
        "xgboost>=1.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib"
    ]
    
    print("Installing required packages...")
    
    failed_packages = []
    
    for package in packages:
        print(f"Installing {package}...")
        if install_package(package):
            print(f"✅ {package} installed successfully")
        else:
            print(f"❌ Failed to install {package}")
            failed_packages.append(package)
    
    print("\n" + "=" * 60)
    
    if not failed_packages:
        print("🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Run test suite: python test_enhancements.py")
        print("2. Run enhanced backtester: python enhanced_backtester.py")
        print("3. Deploy enhanced system: python enhanced_coordinator.py")
    else:
        print(f"⚠️ {len(failed_packages)} packages failed to install:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        print("\nTry installing them manually:")
        for pkg in failed_packages:
            print(f"pip install {pkg}")
    
    print("\n🧠 Your pattern-aware AI trading system is almost ready!")

if __name__ == "__main__":
    main()

