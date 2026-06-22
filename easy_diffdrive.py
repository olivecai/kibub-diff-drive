#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple, high-level interface for driving a DiffDrive robot over the network.

This wraps DiffDriveClient (ZMQ) so callers never touch sockets, JSON, or
the lerobot Robot interface directly. Use this from any machine on the same
network as the robot (e.g. a laptop talking to a Pi/ESP32 wired to the motors).

    from robot.diffdrive.easy_diffdrive import DiffDriveRemote

    with DiffDriveRemote(remote_ip="10.145.4.97") as wheels:
        wheels.drive(x_vel=0.1, theta_vel=0.0)   # forward at 0.1 m/s
        state = wheels.get_state()
        print(state.left_rpm, state.right_rpm, state.x_vel, state.theta_vel)
        wheels.stop()

If you're not sure the host process is running on the robot, see
`ensure_host_running()` in this module, or just run
`python -m robot.diffdrive.diffdrive_host` on the robot machine first.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from config_diffdrive import DiffDriveClientConfig
from diffdrive_client import DiffDriveClient

logger = logging.getLogger(__name__)


@dataclass
class WheelState:
    """A snapshot of the robot's wheel/body state. Plain dataclass, no dict digging."""

    left_rpm: float
    right_rpm: float
    x_vel: float
    theta_vel: float

    def __repr__(self) -> str:
        return (
            f"WheelState(left_rpm={self.left_rpm:.1f}, right_rpm={self.right_rpm:.1f}, "
            f"x_vel={self.x_vel:.3f} m/s, theta_vel={self.theta_vel:.3f} rad/s)"
        )


class DiffDriveNotConnectedError(RuntimeError):
    """Raised when an operation is attempted before connect() or after disconnect()."""


class DiffDriveRemote:
    """A simple, friendly wrapper around DiffDriveClient.

    Handles connect/disconnect, validates inputs, and exposes a small set of
    obvious methods instead of raw action/observation dicts.

    Supports use as a context manager:

        with DiffDriveRemote(remote_ip="10.145.4.97") as wheels:
            wheels.drive(0.1, 0.0)
    """

    def __init__(
        self,
        remote_ip: str = "localhost",
        port_zmq_cmd: int = 5555,
        port_zmq_observations: int = 5556,
        max_wheel_rpm: float | None = None,
        connect_timeout_s: float = 5.0,
    ):
        config_kwargs = dict(
            remote_ip=remote_ip,
            port_zmq_cmd=port_zmq_cmd,
            port_zmq_observations=port_zmq_observations,
            connect_timeout_s=connect_timeout_s,
        )
        if max_wheel_rpm is not None:
            config_kwargs["max_wheel_rpm"] = max_wheel_rpm

        self._config = DiffDriveClientConfig(**config_kwargs)
        self._client = DiffDriveClient(self._config)
        self._connected = False

    # -- connection management -------------------------------------------------

    def connect(self) -> None:
        """Open the ZMQ connection to the DiffDriveHost. Call once before use."""
        if self._connected:
            logger.debug("DiffDriveRemote already connected; ignoring connect()")
            return
        self._client.connect()
        self._connected = True
        logger.info(
            f"Connected to DiffDriveHost at {self._config.remote_ip}:"
            f"{self._config.port_zmq_cmd}"
        )

    def disconnect(self) -> None:
        """Stop the wheels and close the ZMQ connection."""
        if not self._connected:
            return
        try:
            self.stop()
        except Exception as e:
            logger.warning(f"Error sending stop on disconnect: {e}")
        self._client.disconnect()
        self._connected = False
        logger.info("Disconnected from DiffDriveHost")

    def __enter__(self) -> "DiffDriveRemote":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    def _require_connected(self) -> None:
        if not self._connected:
            raise DiffDriveNotConnectedError(
                "DiffDriveRemote is not connected. Call .connect() first "
                "(or use it as a context manager: `with DiffDriveRemote(...) as wheels:`)."
            )

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- driving ------------------------------------------------------------

    def drive(self, x_vel: float, theta_vel: float) -> WheelState:
        """Command body-frame velocity.

        Args:
            x_vel: forward velocity in m/s (positive = forward).
            theta_vel: angular velocity in rad/s (positive = counterclockwise).

        Returns:
            An echo of the command just sent: x_vel/theta_vel are what you
            asked for, left_rpm/right_rpm will read 0.0 here because wheel
            RPM is computed on the robot, not locally. Call get_state()
            shortly after to see the motors' actual reported RPM.
        """
        self._require_connected()
        x_vel = float(x_vel)
        theta_vel = float(theta_vel)
        action_sent = self._client.send_action({"x.vel": x_vel, "theta.vel": theta_vel})
        print("Action sent:", x_vel, theta_vel)
        return WheelState(
            left_rpm=float(action_sent["left_rpm"]),
            right_rpm=float(action_sent["right_rpm"]),
            x_vel=float(action_sent["x.vel"]),
            theta_vel=float(action_sent["theta.vel"]),
        )

    def set_wheel_rpm(self, left_rpm: float, right_rpm: float) -> WheelState:
        """Command each wheel's RPM directly, bypassing body-velocity kinematics.

        Useful for low-level testing (e.g. confirming wiring/direction per wheel).
        """
        self._require_connected()
        action_sent = self._client.send_action(
            {"left_rpm": float(left_rpm), "right_rpm": float(right_rpm)}
        )
        return WheelState(
            left_rpm=float(action_sent["left_rpm"]),
            right_rpm=float(action_sent["right_rpm"]),
            x_vel=float(action_sent["x.vel"]),
            theta_vel=float(action_sent["theta.vel"]),
        )

    def stop(self) -> WheelState:
        """Stop both wheels immediately."""
        return self.set_wheel_rpm(0.0, 0.0)

    # -- reading state --------------------------------------------------------

    def get_state(self) -> WheelState:
        """Fetch the latest reported wheel/body state from the robot."""
        self._require_connected()
        obs = self._client.get_observation()
        return WheelState(
            left_rpm=float(obs["left_rpm"]),
            right_rpm=float(obs["right_rpm"]),
            x_vel=float(obs["x.vel"]),
            theta_vel=float(obs["theta.vel"]),
        )

    def wait_until_reachable(self, timeout_s: float = 5.0, poll_interval_s: float = 0.2) -> bool:
        """Block until the host responds to a zero-velocity ping, or timeout.

        Returns True if the host became reachable, False on timeout.
        Useful right after connect(), since the host may not have sent any
        observations yet.
        """
        self._require_connected()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._client.send_action({"left_rpm": 0.0, "right_rpm": 0.0})
            time.sleep(poll_interval_s)
            obs = self._client.get_observation()
            if obs and obs.get("left_rpm") is not None:
                return True
        return False
