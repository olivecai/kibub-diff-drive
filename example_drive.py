import sys
import time

# Point this at wherever you copied kibub-diff-drive on this machine
sys.path.insert(0, "/home/ocai/openclaw-embodied/skills/diff-drive/")

from easy_diffdrive import DiffDriveRemote

# remote_ip is the IP address of the ROBOT machine (the one running diffdrive_host.py).
# If this script is running on the SAME machine as the host, use "localhost".
with DiffDriveRemote(remote_ip="10.145.8.176") as wheels:
    print("Waiting for host to respond...")
    if not wheels.wait_until_reachable(timeout_s=5.0):
        print("Could not reach the host. Is diffdrive_host.py running on the robot?")
        sys.exit(1)

    print("Driving forward...")
    wheels.drive(x_vel=0.1, theta_vel=0.0)  # 0.1 m/s forward, no turning
    time.sleep(2.0)

    print("Stopping...")
    wheels.stop()

    state = wheels.get_state()
    print("Final state:", state)
