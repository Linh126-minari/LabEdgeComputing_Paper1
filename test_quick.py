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
    
    # Test reset (Gymnasium API — T1.4)
    state, info = env.reset(seed=42)
    print(f"✅ Environment reset, state shape: {state.shape}")
    print(f"   Action mask: {info['action_mask'].sum()}/{info['action_mask'].size} khả thi")

    # Test step
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    print(f"✅ Step executed")
    print(f"   Reward: {reward:.4f} (cents)")
    print(f"   Terminated/Truncated: {terminated}/{truncated}")
    print(f"   Elapsed time: {info['elapsed_time']:.2f}s (semi-MDP)")
    print(f"   tau_eff: {info['tau_eff']:.3f}s")
    
    # Test get_metrics
    metrics = env.get_metrics()
    print(f"✅ Metrics retrieved:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_environment()
