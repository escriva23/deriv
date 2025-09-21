# multi_bot_launcher.py - Launch and manage all bots in the multi-bot system
import os
import sys
import time
import logging
import subprocess
import threading
import signal
from typing import List, Dict
from shared_config import config

# Load .env file if it exists
def load_env_file():
    """Load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✅ Loaded environment variables from .env file")
    else:
        print(f"⚠️  No .env file found at {env_path}")

# Load .env file at startup
load_env_file()

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class MultiBotLauncher:
    def __init__(self):
        self.processes = {}
        self.running = False
        
        # Bot configurations
        self.bot_configs = {
            'probe_a': {
                'script': 'probe_a.py',
                'name': 'Probe A (Parity)',
                'required_env': 'PROBE_A_TOKEN'
            },
            'probe_b': {
                'script': 'probe_b.py', 
                'name': 'Probe B (Over/Under)',
                'required_env': 'PROBE_B_TOKEN'
            },
            'probe_c': {
                'script': 'probe_c.py',
                'name': 'Probe C (Momentum)', 
                'required_env': 'PROBE_C_TOKEN'
            },
            'coordinator': {
                'script': 'coordinator.py',
                'name': 'Coordinator (Real)',
                'required_env': 'COORDINATOR_TOKEN'
            }
        }
        
        logger.info("Multi-Bot Launcher initialized")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        logger.info("Checking prerequisites...")
        
        # Check if Redis is running
        try:
            import redis
            r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB)
            r.ping()
            logger.info("✅ Redis connection successful")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.error("Please start Redis server before running the multi-bot system")
            return False
        
        # Check environment variables
        missing_tokens = []
        for bot_id, bot_config in self.bot_configs.items():
            token_env = bot_config['required_env']
            if not os.getenv(token_env):
                missing_tokens.append(token_env)
        
        if missing_tokens:
            logger.error("❌ Missing required environment variables:")
            for token in missing_tokens:
                logger.error(f"   - {token}")
            logger.error("Please set these environment variables with your Deriv API tokens")
            return False
        
        logger.info("✅ All environment variables are set")
        
        # Check if bot scripts exist
        missing_scripts = []
        for bot_id, bot_config in self.bot_configs.items():
            script_path = bot_config['script']
            if not os.path.exists(script_path):
                missing_scripts.append(script_path)
        
        if missing_scripts:
            logger.error("❌ Missing bot scripts:")
            for script in missing_scripts:
                logger.error(f"   - {script}")
            return False
        
        logger.info("✅ All bot scripts are available")
        
        # Check database
        if os.path.exists(config.DB_PATH):
            logger.info("✅ Database file exists")
        else:
            logger.warning("⚠️ Database file not found - will be created automatically")
        
        return True
    
    def start_bot(self, bot_id: str) -> bool:
        """Start a specific bot"""
        if bot_id in self.processes:
            logger.warning(f"Bot {bot_id} is already running")
            return False
        
        bot_config = self.bot_configs[bot_id]
        script_path = bot_config['script']
        bot_name = bot_config['name']
        
        try:
            logger.info(f"Starting {bot_name}...")
            
            # Start the bot process
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.processes[bot_id] = {
                'process': process,
                'name': bot_name,
                'start_time': time.time()
            }
            
            # Start output monitoring thread
            monitor_thread = threading.Thread(
                target=self.monitor_bot_output,
                args=(bot_id, process),
                daemon=True
            )
            monitor_thread.start()
            
            logger.info(f"✅ {bot_name} started successfully (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start {bot_name}: {e}")
            return False
    
    def monitor_bot_output(self, bot_id: str, process: subprocess.Popen):
        """Monitor bot output and log it"""
        bot_name = self.bot_configs[bot_id]['name']
        
        try:
            while process.poll() is None:
                # Read stdout
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        logger.info(f"[{bot_id}] {line.strip()}")
                
                # Read stderr
                if process.stderr:
                    line = process.stderr.readline()
                    if line:
                        logger.error(f"[{bot_id}] ERROR: {line.strip()}")
                
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error monitoring {bot_name}: {e}")
        
        # Process has ended
        if bot_id in self.processes:
            exit_code = process.poll()
            if exit_code == 0:
                logger.info(f"{bot_name} exited normally")
            else:
                logger.error(f"{bot_name} exited with code {exit_code}")
            
            del self.processes[bot_id]
    
    def stop_bot(self, bot_id: str) -> bool:
        """Stop a specific bot"""
        if bot_id not in self.processes:
            logger.warning(f"Bot {bot_id} is not running")
            return False
        
        bot_info = self.processes[bot_id]
        process = bot_info['process']
        bot_name = bot_info['name']
        
        try:
            logger.info(f"Stopping {bot_name}...")
            
            # Send SIGTERM first
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
                logger.info(f"✅ {bot_name} stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                logger.warning(f"Force killing {bot_name}...")
                process.kill()
                process.wait()
                logger.info(f"✅ {bot_name} force stopped")
            
            del self.processes[bot_id]
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to stop {bot_name}: {e}")
            return False
    
    def start_all_bots(self, exclude_coordinator: bool = False) -> bool:
        """Start all bots in the correct order"""
        logger.info("🚀 Starting multi-bot trading system...")
        
        # Start probe bots first
        probe_bots = ['probe_a', 'probe_b', 'probe_c']
        
        for bot_id in probe_bots:
            if not self.start_bot(bot_id):
                logger.error(f"Failed to start {bot_id}, aborting startup")
                self.stop_all_bots()
                return False
            
            # Small delay between bot starts
            time.sleep(2)
        
        # Wait for probe bots to initialize
        logger.info("Waiting for probe bots to initialize...")
        time.sleep(10)
        
        # Start coordinator if not excluded
        if not exclude_coordinator:
            if not self.start_bot('coordinator'):
                logger.error("Failed to start coordinator")
                self.stop_all_bots()
                return False
        
        self.running = True
        logger.info("🎉 Multi-bot system started successfully!")
        return True
    
    def stop_all_bots(self):
        """Stop all running bots"""
        logger.info("🛑 Stopping all bots...")
        
        # Stop coordinator first
        if 'coordinator' in self.processes:
            self.stop_bot('coordinator')
        
        # Stop probe bots
        for bot_id in ['probe_a', 'probe_b', 'probe_c']:
            if bot_id in self.processes:
                self.stop_bot(bot_id)
        
        self.running = False
        logger.info("✅ All bots stopped")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        status = {
            'running_bots': len(self.processes),
            'bots': {}
        }
        
        for bot_id, bot_info in self.processes.items():
            process = bot_info['process']
            status['bots'][bot_id] = {
                'name': bot_info['name'],
                'pid': process.pid,
                'running_time': time.time() - bot_info['start_time'],
                'status': 'running' if process.poll() is None else 'stopped'
            }
        
        return status
    
    def print_status(self):
        """Print current system status"""
        status = self.get_system_status()
        
        print("\n" + "=" * 60)
        print("MULTI-BOT SYSTEM STATUS")
        print("=" * 60)
        print(f"Running Bots: {status['running_bots']}/4")
        print()
        
        for bot_id, bot_info in status['bots'].items():
            running_time = int(bot_info['running_time'])
            hours = running_time // 3600
            minutes = (running_time % 3600) // 60
            seconds = running_time % 60
            
            status_icon = "🟢" if bot_info['status'] == 'running' else "🔴"
            print(f"{status_icon} {bot_info['name']}")
            print(f"   PID: {bot_info['pid']}")
            print(f"   Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
            print()
    
    def run_interactive_mode(self):
        """Run in interactive mode with commands"""
        logger.info("Starting interactive mode...")
        logger.info("Available commands: status, start <bot>, stop <bot>, restart <bot>, quit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "status":
                    self.print_status()
                elif command.startswith("start "):
                    bot_id = command.split(" ", 1)[1]
                    if bot_id in self.bot_configs:
                        self.start_bot(bot_id)
                    else:
                        print(f"Unknown bot: {bot_id}")
                elif command.startswith("stop "):
                    bot_id = command.split(" ", 1)[1]
                    if bot_id in self.bot_configs:
                        self.stop_bot(bot_id)
                    else:
                        print(f"Unknown bot: {bot_id}")
                elif command.startswith("restart "):
                    bot_id = command.split(" ", 1)[1]
                    if bot_id in self.bot_configs:
                        self.stop_bot(bot_id)
                        time.sleep(2)
                        self.start_bot(bot_id)
                    else:
                        print(f"Unknown bot: {bot_id}")
                elif command == "help":
                    print("Available commands:")
                    print("  status          - Show system status")
                    print("  start <bot>     - Start specific bot")
                    print("  stop <bot>      - Stop specific bot") 
                    print("  restart <bot>   - Restart specific bot")
                    print("  quit/exit       - Exit launcher")
                    print("\nAvailable bots: probe_a, probe_b, probe_c, coordinator")
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        logger.info("Exiting interactive mode...")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_all_bots()
        sys.exit(0)
    
    def run(self, mode: str = "full", interactive: bool = False):
        """Main run method"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites not met, exiting...")
            return False
        
        try:
            if mode == "probes_only":
                # Start only probe bots (for testing)
                success = self.start_all_bots(exclude_coordinator=True)
            else:
                # Start all bots
                success = self.start_all_bots()
            
            if not success:
                return False
            
            if interactive:
                self.run_interactive_mode()
            else:
                # Monitor mode - just keep running and print status periodically
                logger.info("Running in monitor mode. Press Ctrl+C to stop.")
                try:
                    while self.running:
                        time.sleep(60)  # Print status every minute
                        self.print_status()
                except KeyboardInterrupt:
                    pass
            
            return True
            
        finally:
            self.stop_all_bots()

def main():
    """Main launcher function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Bot Trading System Launcher')
    parser.add_argument('--mode', choices=['full', 'probes_only'], default='full',
                       help='Launch mode: full (all bots) or probes_only (demo bots only)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode with command prompt')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check prerequisites and exit')
    
    args = parser.parse_args()
    
    launcher = MultiBotLauncher()
    
    if args.check_only:
        success = launcher.check_prerequisites()
        if success:
            print("✅ All prerequisites are met!")
            return 0
        else:
            print("❌ Prerequisites check failed!")
            return 1
    
    success = launcher.run(mode=args.mode, interactive=args.interactive)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
