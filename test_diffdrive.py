#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import time

# This script is meant to be run from inside the kibub-diff-drive directory
# (e.g. `cd kibub-diff-drive && python test_diffdrive.py`), so it imports its
# sibling modules directly rather than via a `robot.diffdrive` package path.
from diffdrive_client import DiffDriveClient
from config_diffdrive import DiffDriveClientConfig

HOST_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffdrive_host.py")


def is_host_running():
    """Check if DiffDriveHost is currently running."""
    result = subprocess.run(
        ["pgrep", "-f", "diffdrive_host.py"],
        capture_output=True,
    )
    return result.returncode == 0


def start_host_bg():
    """Start DiffDriveHost in the background, capturing stderr."""
    print("Starting DiffDriveHost in background...")
    # Start the host in background, capturing stderr to check for errors
    proc = subprocess.Popen(
        [sys.executable, HOST_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd(),
    )
    # Wait for host to initialize
    time.sleep(3.0)

    # Check if process is still running (didn't crash)
    if proc.poll() is not None:
        _, stderr = proc.communicate()
        stderr_text = stderr.decode("utf-8", errors="replace")
        print(f"ERROR: DiffDriveHost crashed immediately.")
        print(f"stderr output:\n{stderr_text}")
        return None

    print("DiffDriveHost started.")
    return proc


def ensure_host_running():
    """Ensure DiffDriveHost is running, starting it if needed."""
    if is_host_running():
        print("DiffDriveHost is already running.")
        return True

    start_host_bg()

    # Verify it started
    for _ in range(10):
        if is_host_running():
            print("DiffDriveHost is now running.")
            return True
        time.sleep(0.5)

    print("ERROR: Failed to start DiffDriveHost.")
    return False


def move_wheel_speed(client, left_rpm, right_rpm, duration_s, print_every=0.1):
    end_time = time.monotonic() + duration_s
    while time.monotonic() < end_time:
        client.send_action({"left_rpm": left_rpm, "right_rpm": right_rpm})
        obs = client.get_observation()
        print(
            f"  left_rpm={obs['left_rpm']:.1f}, right_rpm={obs['right_rpm']:.1f}, "
            f"x.vel={obs['x.vel']:.3f}, theta.vel={obs['theta.vel']:.3f}"
        )
        time.sleep(print_every)


def move_body_speed(client, x_vel, theta_vel, duration_s, print_every=0.1):
    end_time = time.monotonic() + duration_s
    while time.monotonic() < end_time:
        client.send_action({"x.vel": x_vel, "theta.vel": theta_vel})
        obs = client.get_observation()
        print(
            f"  left_rpm={obs['left_rpm']:.1f}, right_rpm={obs['right_rpm']:.1f}, "
            f"x.vel={obs['x.vel']:.3f}, theta.vel={obs['theta.vel']:.3f}"
        )
        time.sleep(print_every)


def stop(client, duration_s=1.0, print_every=0.1):
    end_time = time.monotonic() + duration_s
    while time.monotonic() < end_time:
        client.send_action({"left_rpm": 0.0, "right_rpm": 0.0})
        obs = client.get_observation()
        print(
            f"  left_rpm={obs['left_rpm']:.1f}, right_rpm={obs['right_rpm']:.1f}, "
            f"x.vel={obs['x.vel']:.3f}, theta.vel={obs['theta.vel']:.3f}"
        )
        time.sleep(print_every)


def test_left_wheel_rpm(client, speed_rpm):
    print(f"\n=== Phase 1: Left wheel test (RPM mode) at {speed_rpm} RPM ===")
    print("Forward:")
    move_wheel_speed(client, speed_rpm, 0, 3.0)
    print("Stop:")
    stop(client, 1.0)
    print("Backward:")
    move_wheel_speed(client, -speed_rpm, 0, 3.0)
    print("Stop:")
    stop(client, 1.0)


def test_right_wheel_rpm(client, speed_rpm):
    print(f"\n=== Phase 1: Right wheel test (RPM mode) at {speed_rpm} RPM ===")
    print("Forward:")
    move_wheel_speed(client, 0, speed_rpm, 3.0)
    print("Stop:")
    stop(client, 1.0)
    print("Backward:")
    move_wheel_speed(client, 0, -speed_rpm, 3.0)
    print("Stop:")
    stop(client, 1.0)


def test_both_wheels_rpm(client, speed_rpm):
    print(f"\n=== Phase 1: Both wheels test (RPM mode) at {speed_rpm} RPM ===")
    print("Forward:")
    move_wheel_speed(client, speed_rpm, speed_rpm, 3.0)
    print("Stop:")
    stop(client, 1.0)
    print("Backward:")
    move_wheel_speed(client, -speed_rpm, -speed_rpm, 3.0)
    print("Stop:")
    stop(client, 1.0)


def test_left_wheel_body(client, wheel_radius, base_width, speed_rpm):
    print(f"\n=== Phase 2: Left wheel test (body velocity mode) at {speed_rpm} RPM ===")
    left_angular = speed_rpm * 2 * 3.14159265359 / 60
    x_vel = left_angular * wheel_radius
    theta_vel = -x_vel * 2 / base_width
    print(f"Computed: x.vel={x_vel:.3f}, theta.vel={theta_vel:.3f}")
    print("Forward:")
    move_body_speed(client, x_vel, theta_vel, 3.0)
    print("Stop:")
    stop(client, 1.0)
    print("Backward:")
    move_body_speed(client, -x_vel, -theta_vel, 3.0)
    print("Stop:")
    stop(client, 1.0)


def test_right_wheel_body(client, wheel_radius, base_width, speed_rpm):
    print(
        f"\n=== Phase 2: Right wheel test (body velocity mode) at {speed_rpm} RPM ==="
    )
    right_angular = speed_rpm * 2 * 3.14159265359 / 60
    x_vel = right_angular * wheel_radius
    theta_vel = x_vel * 2 / base_width
    print(f"Computed: x.vel={x_vel:.3f}, theta.vel={theta_vel:.3f}")
    print("Forward:")
    move_body_speed(client, x_vel, theta_vel, 3.0)
    print("Stop:")
    stop(client, 1.0)
    print("Backward:")
    move_body_speed(client, -x_vel, -theta_vel, 3.0)
    print("Stop:")
    stop(client, 1.0)


def test_both_wheels_body(client, wheel_radius, speed_rpm):
    print(
        f"\n=== Phase 2: Both wheels test (body velocity mode) at {speed_rpm} RPM ==="
    )
    left_angular = speed_rpm * 2 * 3.14159265359 / 60
    x_vel = left_angular * wheel_radius
    theta_vel = 0.0
    print(f"Computed: x.vel={x_vel:.3f}, theta.vel={theta_vel:.3f}")
    print("Forward:")
    move_body_speed(client, x_vel, theta_vel, 3.0)
    print("Stop:")
    stop(client, 1.0)
    print("Backward:")
    move_body_speed(client, -x_vel, theta_vel, 3.0)
    print("Stop:")
    stop(client, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Test DiffDrive client")
    parser.add_argument(
        "--speed-rpm", type=float, default=60.0, help="Test speed in RPM"
    )
    args = parser.parse_args()

    # Ensure DiffDriveHost is running before connecting
    if not ensure_host_running():
        print("ERROR: Could not start DiffDriveHost. Exiting.")
        sys.exit(1)

    config = DiffDriveClientConfig(
        remote_ip="localhost",
        port_zmq_cmd=5555,
        port_zmq_observations=5556,
    )
    client = DiffDriveClient(config)

    print("Connecting to localhost:5555...")
    client.connect()

    print("Checking host connectivity...")
    client.send_action({"left_rpm": 0.0, "right_rpm": 0.0})
    time.sleep(0.5)
    obs = client.get_observation()
    if not obs or obs.get("left_rpm") is None:
        print("ERROR: Host not responding. Exiting.")
        client.disconnect()
        return
    print("Host is reachable.\n")

    wheel_radius = client.wheel_radius
    base_width = client.base_width
    print(f"Using wheel_radius={wheel_radius}, base_width={base_width}")

    test_both_wheels_rpm(client, args.speed_rpm)
    test_left_wheel_rpm(client, args.speed_rpm)
    test_right_wheel_rpm(client, args.speed_rpm)

    test_both_wheels_body(client, wheel_radius, args.speed_rpm)
    test_left_wheel_body(client, wheel_radius, base_width, args.speed_rpm)
    test_right_wheel_body(client, wheel_radius, base_width, args.speed_rpm)

    print("\n=== All tests complete ===")
    client.send_action({"left_rpm": 0.0, "right_rpm": 0.0})
    client.disconnect()


if __name__ == "__main__":
    main()
