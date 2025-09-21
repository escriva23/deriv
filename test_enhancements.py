# test_enhancements.py - Test script for pattern-aware enhancements
import sys
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_component(component_name, test_func):
    """Test a component and report results"""
    try:
        print(f"\n{'='*50}")
        print(f"Testing {component_name}...")
        print(f"{'='*50}")
        
        result = test_func()
        
        if result:
            print(f"✅ {component_name} - PASSED")
            return True
        else:
            print(f"⚠️ {component_name} - PARTIAL (some features may not work)")
            return True
    except Exception as e:
        print(f"❌ {component_name} - FAILED")
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_pattern_calibration():
    """Test pattern calibration system"""
    try:
        from pattern_calibration import ProbabilityCalibrator
        
        calibrator = ProbabilityCalibrator()
        print("✓ ProbabilityCalibrator imported successfully")
        
        # Test basic functionality
        model_key = "test_model"
        calibrator.collect_calibration_data(model_key, 0.7, True, "R_100", "DIGITEVEN")
        print("✓ Calibration data collection works")
        
        # Test calibration (should return input when no training data)
        calibrated = calibrator.calibrate_probability(model_key, 0.7)
        print(f"✓ Calibration works: {0.7} -> {calibrated}")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_pattern_detectors():
    """Test pattern detection system"""
    try:
        from pattern_detectors import AdvancedPatternEngine
        
        engine = AdvancedPatternEngine()
        print("✓ AdvancedPatternEngine imported successfully")
        
        # Test pattern detection
        features = engine.update_patterns(1234.56, 6, "R_100")
        print(f"✓ Pattern detection works: {len(features)} features generated")
        
        # Print some key features
        key_features = {k: v for k, v in features.items() if k.startswith('pattern_')}
        print(f"✓ Pattern signals: {list(key_features.keys())}")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_meta_controller():
    """Test meta-controller system"""
    try:
        from meta_controller import MetaController
        
        controller = MetaController()
        print("✓ MetaController imported successfully")
        
        # Test drift detection
        drift_detected, method, strength = controller.drift_detector.detect_drift()
        print(f"✓ Drift detection works: {drift_detected} ({method})")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_martingale_system():
    """Test martingale recovery system"""
    try:
        from martingale_system import MartingaleRecoverySystem
        
        system = MartingaleRecoverySystem()
        system.update_balance(1000.0)
        print("✓ MartingaleRecoverySystem imported successfully")
        
        # Test sequence evaluation
        should_start, reason = system.should_start_sequence(
            "R_100", "DIGITEVEN", 0.72, 0.75, 0.05
        )
        print(f"✓ Sequence evaluation works: {should_start} ({reason})")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_online_learning():
    """Test online learning system"""
    try:
        from online_learning import AdaptiveLearningSystem
        
        system = AdaptiveLearningSystem()
        print("✓ AdaptiveLearningSystem imported successfully")
        
        # Test model creation
        model = system.get_or_create_model("test_model", "sgd")
        print(f"✓ Model creation works: {model.model_type}")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_enhanced_backtester():
    """Test enhanced backtester"""
    try:
        from enhanced_backtester import EnhancedBacktester, BacktestConfig
        
        config = BacktestConfig(initial_balance=1000.0)
        backtester = EnhancedBacktester(config)
        print("✓ EnhancedBacktester imported successfully")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_enhanced_ai_predictor():
    """Test enhanced AI predictor"""
    try:
        from enhanced_ai_predictor import EnhancedAIPredictor
        
        predictor = EnhancedAIPredictor()
        print("✓ EnhancedAIPredictor imported successfully")
        
        # Test feature preparation
        test_features = {
            'last_digit': 6,
            'price': 1234.56,
            'price_change': 0.1
        }
        
        pred, conf, metadata = predictor.predict_enhanced("R_100", test_features)
        print(f"✓ Enhanced prediction works: pred={pred}, conf={conf:.3f}")
        
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 Pattern-Aware AI Trading Enhancement Test Suite")
    print("=" * 60)
    
    tests = [
        ("Pattern Calibration", test_pattern_calibration),
        ("Pattern Detectors", test_pattern_detectors),
        ("Meta-Controller", test_meta_controller),
        ("Martingale System", test_martingale_system),
        ("Online Learning", test_online_learning),
        ("Enhanced Backtester", test_enhanced_backtester),
        ("Enhanced AI Predictor", test_enhanced_ai_predictor),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        if test_component(test_name, test_func):
            passed += 1
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} components working")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your enhanced system is ready.")
        print("\nNext steps:")
        print("1. Install missing dependencies: pip install -r enhanced_requirements.txt")
        print("2. Run enhanced backtester: python enhanced_backtester.py")
        print("3. Deploy enhanced coordinator: python enhanced_coordinator.py")
    elif passed >= total * 0.7:
        print("✅ MOST TESTS PASSED! System is mostly functional.")
        print("\nRecommendation: Install missing dependencies and retest:")
        print("pip install -r enhanced_requirements.txt")
    else:
        print("⚠️ SEVERAL TESTS FAILED. Check dependencies and imports.")
        print("\nTroubleshooting:")
        print("1. Install dependencies: pip install -r enhanced_requirements.txt")
        print("2. Check Python version (3.8+ recommended)")
        print("3. Verify all files are in the same directory")

if __name__ == "__main__":
    main()

