#!/usr/bin/env python3
"""
Run policy inference on the real robot.

Usage:
    uv run python run_policy.py --num_loops 5 --actions_per_loop 10
"""

import argparse
import numpy as np
from real_world_env import RealWorldEnv, DEFAULT_TASK_INSTRUCTION


def main():
    parser = argparse.ArgumentParser(description="Run policy inference on real robot")
    parser.add_argument("--num_loops", type=int, default=1,
                        help="Number of inference loops to run (default: 1)")
    parser.add_argument("--actions_per_loop", type=int, default=10,
                        help="Number of actions to execute per loop (default: 10)")
    parser.add_argument("--speed_factor", type=float, default=0.3,
                        help="Speed factor for robot motion (default: 0.3)")
    parser.add_argument("--no_reset", action="store_true",
                        help="Skip resetting robot to home position")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Running Policy Inference on Real Robot")
    print("=" * 60)
    print(f"  Loops: {args.num_loops}")
    print(f"  Actions per loop: {args.actions_per_loop}")
    print(f"  Speed factor: {args.speed_factor}")
    
    # Create environment with robot and policy
    print("\nInitializing environment...")
    env = RealWorldEnv(
        use_robot=True,
        use_policy=True,
    )
    
    try:
        # Reset robot to home position
        if not args.no_reset:
            print("\nResetting robot to home position...")
            env.reset(go_home=True)
        
        # Main loop
        for loop_idx in range(args.num_loops):
            print("\n" + "=" * 60)
            print(f"Loop {loop_idx + 1}/{args.num_loops}")
            print("=" * 60)
            
            # Get observation
            print("\nGetting observation...")
            observation = env.get_observation()
            print(f"  Images: {len(observation['images'])} images")
            print(f"  State shape: {observation['state'].shape}")
            print(f"  Joint positions: {observation['state'][:7]}")
            print(f"  Gripper width: {observation['state'][7]:.4f}m")
            
            # Run inference with default task instruction
            print(f"\nRunning inference...")
            print(f"  Task: {DEFAULT_TASK_INSTRUCTION}")
            actions = env.get_action(observation=observation, denormalize=True)
            print(f"  Output actions shape: {actions.shape}")
            print(f"  First action - joint deltas: {actions[0, :7]}")
            print(f"  First action - gripper: {actions[0, 7]:.4f}m")
            
            # Execute actions
            num_actions = min(args.actions_per_loop, len(actions))
            print(f"\nExecuting {num_actions} actions...")
            
            for i in range(num_actions):
                print(f"\n  Step {i+1}/{num_actions}")
                print(f"    Joint deltas: {actions[i, :7]}")
                print(f"    Gripper: {actions[i, 7]:.4f}m")
                
                env.step(actions[i], speed_factor=args.speed_factor)
                
                # Get new state after action
                new_state = env.get_robot_state()
                if new_state is not None:
                    print(f"    New joints: {new_state}")
        
        print("\n" + "=" * 60)
        print("Execution complete!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        print("\nClosing environment...")
        env.close()


if __name__ == "__main__":
    main()
