# system_status_report.py - Comprehensive status report for enhanced AI trading system
import sys
import os
import sqlite3
import importlib
import logging
from datetime import datetime
from config import config

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'scipy', 
        'lightgbm', 'xgboost', 'matplotlib', 'seaborn', 'joblib'
    ]
    
    missing = []
    installed = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    return installed, missing

def check_database_tables():
    """Check if all required database tables exist"""
    required_tables = [
        'pattern_features', 'calibration_data', 'online_samples', 
        'model_updates', 'martingale_sequences', 'martingale_trades', 'trades'
    ]
    
    try:
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        present = [t for t in required_tables if t in existing_tables]
        missing = [t for t in required_tables if t not in existing_tables]
        
        conn.close()
        return present, missing, existing_tables
    except Exception as e:
        return [], required_tables, []

def check_enhanced_modules():
    """Check if all enhanced modules can be imported"""
    modules = {
        'pattern_calibration': 'ProbabilityCalibrator',
        'pattern_detectors': 'AdvancedPatternEngine', 
        'meta_controller': 'MetaController',
        'martingale_system': 'MartingaleRecoverySystem',
        'online_learning': 'AdaptiveLearningSystem',
        'enhanced_backtester': 'EnhancedBacktester',
        'enhanced_ai_predictor': 'EnhancedAIPredictor'
    }
    
    working = []
    broken = []
    
    for module_name, class_name in modules.items():
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)  # Check if class exists
            working.append(f"{module_name}.{class_name}")
        except Exception as e:
            broken.append(f"{module_name}.{class_name}: {str(e)}")
    
    return working, broken

def check_original_system():
    """Check if original system components are still working"""
    original_modules = [
        'ai_predictor', 'coordinator', 'signal_manager', 
        'risk_manager', 'data_collector', 'trading_executor', 'backtester'
    ]
    
    working = []
    broken = []
    
    for module_name in original_modules:
        try:
            importlib.import_module(module_name)
            working.append(module_name)
        except Exception as e:
            broken.append(f"{module_name}: {str(e)}")
    
    return working, broken

def main():
    print("🧠 Pattern-Aware AI Trading System - Status Report")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version}")
    print(f"Database Path: {config.DB_PATH}")
    print()
    
    # Check dependencies
    print("📦 DEPENDENCIES")
    print("-" * 30)
    installed_deps, missing_deps = check_dependencies()
    
    print(f"✅ Installed ({len(installed_deps)}): {', '.join(installed_deps)}")
    if missing_deps:
        print(f"❌ Missing ({len(missing_deps)}): {', '.join(missing_deps)}")
        print("   Run: pip install -r enhanced_requirements.txt")
    else:
        print("🎉 All dependencies installed!")
    print()
    
    # Check database
    print("🗄️ DATABASE")
    print("-" * 30)
    present_tables, missing_tables, all_tables = check_database_tables()
    
    print(f"✅ Required tables present ({len(present_tables)}): {', '.join(present_tables)}")
    if missing_tables:
        print(f"❌ Missing tables ({len(missing_tables)}): {', '.join(missing_tables)}")
        print("   Run: python create_database_tables.py")
    else:
        print("🎉 All required tables present!")
    
    print(f"📋 Total tables in database: {len(all_tables)}")
    print()
    
    # Check enhanced modules
    print("🚀 ENHANCED MODULES")
    print("-" * 30)
    working_enhanced, broken_enhanced = check_enhanced_modules()
    
    for module in working_enhanced:
        print(f"✅ {module}")
    
    if broken_enhanced:
        print("\nBroken modules:")
        for module in broken_enhanced:
            print(f"❌ {module}")
    else:
        print("🎉 All enhanced modules working!")
    print()
    
    # Check original system
    print("🔧 ORIGINAL SYSTEM")
    print("-" * 30)
    working_orig, broken_orig = check_original_system()
    
    for module in working_orig:
        print(f"✅ {module}")
    
    if broken_orig:
        print("\nBroken modules:")
        for module in broken_orig:
            print(f"⚠️ {module}")
    print()
    
    # Summary
    print("📊 SUMMARY")
    print("-" * 30)
    total_score = 0
    max_score = 4
    
    if not missing_deps:
        total_score += 1
        print("✅ Dependencies: GOOD")
    else:
        print("❌ Dependencies: MISSING")
    
    if not missing_tables:
        total_score += 1
        print("✅ Database: GOOD")
    else:
        print("❌ Database: MISSING TABLES")
    
    if not broken_enhanced:
        total_score += 1
        print("✅ Enhanced modules: WORKING")
    else:
        print("❌ Enhanced modules: ERRORS")
    
    if len(broken_orig) <= 1:  # Allow 1 broken original module
        total_score += 1
        print("✅ Original system: MOSTLY WORKING")
    else:
        print("❌ Original system: MULTIPLE ERRORS")
    
    print()
    print(f"🎯 SYSTEM HEALTH: {total_score}/{max_score} ({total_score/max_score*100:.0f}%)")
    
    if total_score == max_score:
        print("🎉 SYSTEM FULLY OPERATIONAL!")
        print("\nReady for:")
        print("• Enhanced backtesting: python enhanced_backtester.py")
        print("• Live trading: python enhanced_coordinator.py")
        print("• Pattern analysis: python pattern_detectors.py")
    elif total_score >= 3:
        print("✅ SYSTEM MOSTLY OPERATIONAL!")
        print("Minor issues detected - check details above")
    else:
        print("⚠️ SYSTEM NEEDS ATTENTION!")
        print("Multiple issues detected - fix critical errors first")

if __name__ == "__main__":
    main()
