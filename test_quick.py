"""
Simple test to verify environment setup
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.env.core import VehicularEdgeEnvironment, EnvironmentConfig


def test_environment():
    """Test basic environment functionality"""
    print("Testing VehicularEdgeEnvironment...")
    
    # Create simple config
    config = EnvironmentConfig(
        num_users=2,
        num_mec_servers=1,
        num_auxiliary_vehicles=1,
        episode_length=10
    )
    
    # Create environment
    env = VehicularEdgeEnvironment(config)
    print(f"✅ Environment created")
    print(f"   State space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Test reset
    state = env.reset()
    print(f"✅ Environment reset, state shape: {state.shape}")
    
    # Test step
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    print(f"✅ Step executed")
    print(f"   Reward: {reward:.4f}")
    print(f"   Done: {done}")
    print(f"   Info: {info}")
    
    # Test get_metrics
    metrics = env.get_metrics()
    print(f"✅ Metrics retrieved:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_environment()
