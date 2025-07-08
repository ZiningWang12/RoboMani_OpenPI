"""
WebSocket Client for Pooltool Environment
Communicates with OpenPI Policy Server following LIBERO pattern
"""

import asyncio
import json
import websockets
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

from env import PooltoolEnvironment, make_example


@dataclass
class ClientConfig:
    """Client configuration."""
    server_url: str = "ws://localhost:8000"
    timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0


class PooltoolWebSocketClient:
    """WebSocket client for Pooltool environment."""
    
    def __init__(self, config: ClientConfig = None):
        self.config = config or ClientConfig()
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.logger = logging.getLogger(__name__)
        
    async def connect(self) -> bool:
        """Connect to the policy server."""
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"Connecting to {self.config.server_url} (attempt {attempt + 1})")
                self.websocket = await websockets.connect(
                    self.config.server_url,
                    timeout=self.config.timeout
                )
                self.logger.info("✅ Connected to policy server")
                return True
                
            except Exception as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    
        self.logger.error("❌ Failed to connect to policy server")
        return False
    
    async def disconnect(self):
        """Disconnect from the server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            self.logger.info("Disconnected from policy server")
    
    async def get_action(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get action from policy server given observation."""
        if not self.websocket:
            self.logger.error("Not connected to server")
            return None
            
        try:
            # Convert numpy arrays to lists for JSON serialization
            obs_serializable = self._make_serializable(observation)
            
            # Send observation
            await self.websocket.send(json.dumps({
                "type": "observation",
                "data": obs_serializable
            }))
            
            # Receive action
            response = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=self.config.timeout
            )
            
            action_data = json.loads(response)
            if action_data.get("type") == "action":
                return action_data.get("data")
            else:
                self.logger.warning(f"Unexpected response type: {action_data.get('type')}")
                return None
                
        except asyncio.TimeoutError:
            self.logger.error("Timeout waiting for action from server")
            return None
        except Exception as e:
            self.logger.error(f"Error communicating with server: {e}")
            return None
    
    def _make_serializable(self, data: Any) -> Any:
        """Convert numpy arrays and other non-serializable types to serializable format."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_serializable(x) for x in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        else:
            return data


class PooltoolRunner:
    """Main runner for Pooltool environment with WebSocket communication."""
    
    def __init__(self, 
                 use_server: bool = False,
                 server_url: str = "ws://localhost:8000",
                 num_trials: int = 5,
                 render_mode: str = "rgb_array"):
        self.use_server = use_server
        self.num_trials = num_trials
        self.render_mode = render_mode
        
        # Initialize environment
        self.env = PooltoolEnvironment(
            table_size=(2.54, 1.27),
            num_balls=16,
            render_mode=render_mode,
            seed=42
        )
        
        # Initialize client if using server
        self.client = None
        if use_server:
            config = ClientConfig(server_url=server_url)
            self.client = PooltoolWebSocketClient(config)
            
        self.logger = logging.getLogger(__name__)
    
    async def run_single_trial(self, trial_id: int) -> Dict[str, Any]:
        """Run a single trial."""
        self.logger.info(f"🎱 Starting trial {trial_id + 1}/{self.num_trials}")
        
        # Reset environment
        self.env.reset()
        
        # Trial statistics
        steps = 0
        total_reward = 0.0
        actions_taken = []
        
        while not self.env.is_episode_complete() and steps < 1000:  # Max 1000 steps
            # Get observation
            obs = self.env.get_observation()
            
            # Get action from server or use random action
            if self.client:
                action = await self.client.get_action(obs)
                if action is None:
                    self.logger.warning("Failed to get action from server, using random action")
                    action = self._generate_random_action()
            else:
                action = self._generate_random_action()
            
            # Apply action
            self.env.apply_action(action)
            actions_taken.append(action)
            
            steps += 1
            
            # Log progress
            if steps % 100 == 0:
                cue_ball = self.env.balls[0]
                speed = np.linalg.norm(cue_ball.velocity)
                self.logger.info(f"  Step {steps}: Cue ball speed = {speed:.2f} m/s")
        
        # Trial results
        result = {
            "trial_id": trial_id,
            "steps": steps,
            "completed": self.env.is_episode_complete(),
            "total_reward": total_reward,
            "final_ball_positions": [ball.position.tolist() for ball in self.env.balls],
            "actions_count": len(actions_taken)
        }
        
        self.logger.info(f"✅ Trial {trial_id + 1} completed: {steps} steps, "
                        f"{'success' if result['completed'] else 'timeout'}")
        
        return result
    
    def _generate_random_action(self) -> Dict[str, Any]:
        """Generate a random action for testing."""
        return {
            "arm_action": np.random.uniform(-1, 1, 6).tolist(),
            "cue_action": {
                "power": np.random.uniform(0.1, 2.0),
                "angle": np.random.uniform(0, 2 * np.pi),
                "english": [0.0, 0.0]  # No side spin for now
            }
        }
    
    async def run_all_trials(self) -> Dict[str, Any]:
        """Run all trials and return results."""
        self.logger.info(f"🚀 Starting {self.num_trials} trials")
        
        # Connect to server if needed
        if self.client:
            if not await self.client.connect():
                self.logger.error("Failed to connect to server, running without server")
                self.client = None
        
        # Run trials
        results = []
        for trial_id in range(self.num_trials):
            try:
                result = await self.run_single_trial(trial_id)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Trial {trial_id + 1} failed: {e}")
                results.append({
                    "trial_id": trial_id,
                    "error": str(e),
                    "completed": False
                })
        
        # Disconnect from server
        if self.client:
            await self.client.disconnect()
        
        # Clean up environment
        self.env.close()
        
        # Summary statistics
        completed_trials = sum(1 for r in results if r.get("completed", False))
        avg_steps = np.mean([r.get("steps", 0) for r in results if "steps" in r])
        
        summary = {
            "total_trials": self.num_trials,
            "completed_trials": completed_trials,
            "success_rate": completed_trials / self.num_trials,
            "average_steps": avg_steps,
            "results": results
        }
        
        self.logger.info(f"🎉 Trials completed: {completed_trials}/{self.num_trials} "
                        f"({100 * summary['success_rate']:.1f}% success rate)")
        
        return summary


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pooltool Environment Runner")
    parser.add_argument("--server-url", default="ws://localhost:8000", 
                       help="Policy server WebSocket URL")
    parser.add_argument("--use-server", action="store_true",
                       help="Connect to policy server")
    parser.add_argument("--num-trials", type=int, default=5,
                       help="Number of trials to run")
    parser.add_argument("--render-mode", choices=["human", "rgb_array"], 
                       default="rgb_array", help="Rendering mode")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run trials
    runner = PooltoolRunner(
        use_server=args.use_server,
        server_url=args.server_url,
        num_trials=args.num_trials,
        render_mode=args.render_mode
    )
    
    results = await runner.run_all_trials()
    
    # Save results
    import json
    with open("pooltool_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to pooltool_results.json")
    print(f"Success rate: {100 * results['success_rate']:.1f}%")


if __name__ == "__main__":
    asyncio.run(main()) 