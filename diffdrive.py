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

import json
import logging
import queue
import threading
import time
from functools import cached_property

import numpy as np
import serial

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot.robots.robot import Robot
from config_diffdrive import DiffDriveConfig

logger = logging.getLogger(__name__)


class DiffDrive(Robot):
    config_class = DiffDriveConfig
    name = "diffdrive"

    def __init__(self, config: DiffDriveConfig):
        super().__init__(config)
        self.config = config
        self.id = config.id

        self.serial_port = config.port
        self.baud_rate = config.baud_rate
        self.left_motor_id = config.left_motor_id
        self.right_motor_id = config.right_motor_id
        self.left_direction = config.left_direction
        self.right_direction = config.right_direction
        self.wheel_diameter = config.wheel_diameter
        self.wheel_radius = self.wheel_diameter / 2
        self.base_width = config.base_width
        self.max_rpm = config.max_rpm
        self.max_wheel_rpm = config.max_wheel_rpm
        self.accel_time = config.accel_time
        self.heartbeat_ms = config.heartbeat_ms

        self.serial = None
        self._is_connected = False

        self.last_left_rpm = 0.0
        self.last_right_rpm = 0.0

        self._reader_thread: threading.Thread | None = None
        self._reader_stop = threading.Event()
        self._response_queue: queue.Queue[dict] = queue.Queue()
        self._write_lock = threading.Lock()

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        return {
            "left_rpm": float,
            "right_rpm": float,
            "x.vel": float,
            "theta.vel": float,
        }

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            name: (cfg.height, cfg.width, 3)
            for name, cfg in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self) -> None:
        self.serial = serial.Serial(
            self.serial_port,
            self.baud_rate,
            timeout=0.01,
            dsrdtr=False,
        )
        self.serial.setRTS(False)
        self.serial.setDTR(False)
        time.sleep(2)
        self._send_heartbeat()
        self._is_connected = True

        # Start background reader thread to drain the serial RX buffer
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name="diffdrive-serial-reader"
        )
        self._reader_thread.start()

        logger.info(f"{self} connected on {self.serial_port}")

    @staticmethod
    def _sanitize_cmd(cmd_dict: dict) -> dict:
        """Sanitize command values for the DDSM115 ESP32 protocol.

        Removes IEEE 754 negative zero (which the ESP32 may misinterpret as a
        non-zero speed) and rounds float RPM values to integers (the DDSM115
        speed-loop protocol expects integer RPM).
        """
        sanitized = {}
        for key, val in cmd_dict.items():
            if isinstance(val, float):
                # Round RPM-like values and eliminate -0.0
                val = round(val)
                val = int(val)  # convert to int for clean JSON encoding
                if key == "cmd":
                    val = int(val)
            sanitized[key] = val
        return sanitized

    def _send_command(self, cmd_dict: dict) -> None:
        cmd_dict = self._sanitize_cmd(cmd_dict)
        cmd_str = json.dumps(cmd_dict) + "\n"
        with self._write_lock:
            self.serial.write(cmd_str.encode("utf-8"))

    def _send_heartbeat(self) -> None:
        self._send_command({"T": 11001, "time": self.heartbeat_ms})

    def _reader_loop(self) -> None:
        """Background thread that drains the serial RX buffer into _response_queue.

        Without this, incoming motor feedback accumulates in the OS serial buffer
        and can cause the ESP32 to stall or drop commands.
        """
        buf = b""
        while not self._reader_stop.is_set():
            try:
                if self.serial is None or not self.serial.is_open:
                    break
                chunk = self.serial.read(self.serial.in_waiting or 1)
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line.decode())
                        self._response_queue.put(obj)
                    except json.JSONDecodeError:
                        logger.debug(f"Non-JSON from serial: {line!r}")
            except Exception as e:
                if not self._reader_stop.is_set():
                    logger.warning(f"Serial reader error: {e}")
                break

    def calibrate(self) -> None:
        pass

    def _body_to_wheel_rpm(self, x_vel: float, theta_vel: float) -> tuple[float, float]:
        # Differential drive kinematics: convert body velocity to wheel linear speeds.
        # These formulas assume standard sign convention: positive theta_vel is CCW rotation.
        left_linear = x_vel - theta_vel * self.base_width / 2
        right_linear = x_vel + theta_vel * self.base_width / 2

        left_angular = left_linear / self.wheel_radius
        right_angular = right_linear / self.wheel_radius

        # Apply motor direction multipliers to convert from wheel speed to motor command.
        left_rpm = left_angular * self.left_direction * 60 / (2 * np.pi)
        right_rpm = right_angular * self.right_direction * 60 / (2 * np.pi)

        left_rpm = float(np.clip(left_rpm, -self.max_wheel_rpm, self.max_wheel_rpm))
        right_rpm = float(np.clip(right_rpm, -self.max_wheel_rpm, self.max_wheel_rpm))

        # Round to integer RPM to avoid floating-point noise (e.g. 59.99999999)
        # and eliminate IEEE 754 negative zero which the ESP32 firmware may
        # misinterpret as a non-zero speed command.
        left_rpm = float(round(left_rpm))
        right_rpm = float(round(right_rpm))

        return left_rpm, right_rpm

    def _wheel_rpm_to_body(
        self, left_rpm: float, right_rpm: float
    ) -> tuple[float, float]:
        left_angular = left_rpm * self.left_direction * 2 * np.pi / 60
        right_angular = right_rpm * self.right_direction * 2 * np.pi / 60

        left_linear = left_angular * self.wheel_radius
        right_linear = right_angular * self.wheel_radius

        x_vel = (left_linear + right_linear) / 2
        theta_vel = (right_linear - left_linear) / self.base_width

        # Eliminate IEEE 754 negative zero from direction multiplication
        x_vel = 0.0 if x_vel == 0.0 else x_vel
        theta_vel = 0.0 if theta_vel == 0.0 else theta_vel

        return float(x_vel), float(theta_vel)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        self._send_heartbeat()

        obs_dict: RobotObservation = {
            "left_rpm": self.last_left_rpm,
            "right_rpm": self.last_right_rpm,
            "x.vel": 0.0,
            "theta.vel": 0.0,
            OBS_STATE: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }

        # Drain all available responses from the background reader queue
        try:
            while True:
                data = self._response_queue.get_nowait()
                if "rpm" in data:
                    self.last_left_rpm = data["rpm"].get(str(self.left_motor_id), 0.0)
                    self.last_right_rpm = data["rpm"].get(str(self.right_motor_id), 0.0)
                    x_vel, theta_vel = self._wheel_rpm_to_body(
                        self.last_left_rpm, self.last_right_rpm
                    )
                    obs_dict["left_rpm"] = self.last_left_rpm
                    obs_dict["right_rpm"] = self.last_right_rpm
                    obs_dict["x.vel"] = x_vel
                    obs_dict["theta.vel"] = theta_vel
                    obs_dict[OBS_STATE] = np.array(
                        [self.last_left_rpm, self.last_right_rpm, x_vel, theta_vel],
                        dtype=np.float32,
                    )
        except queue.Empty:
            pass

        for cam_name in self.config.cameras:
            obs_dict[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        x_vel = action.get("x.vel", None)
        theta_vel = action.get("theta.vel", None)
        left_rpm_input = action.get("left_rpm", None)
        right_rpm_input = action.get("right_rpm", None)
        act = action.get("act", self.accel_time)

        # Support direct wheel RPM commands
        if left_rpm_input is not None or right_rpm_input is not None:
            # Use provided wheel RPMs directly (with defaults)
            left_rpm = left_rpm_input if left_rpm_input is not None else 0.0
            right_rpm = right_rpm_input if right_rpm_input is not None else 0.0

            # Apply motor direction multipliers and clip for motor command
            left_rpm_out = float(
                np.clip(
                    round(left_rpm * self.left_direction),
                    -self.max_wheel_rpm,
                    self.max_wheel_rpm,
                )
            )
            right_rpm_out = float(
                np.clip(
                    round(right_rpm * self.right_direction),
                    -self.max_wheel_rpm,
                    self.max_wheel_rpm,
                )
            )

            # Also compute the corresponding body velocities for reporting
            x_vel, theta_vel = self._wheel_rpm_to_body(left_rpm_out, right_rpm_out)
        else:
            # Standard body velocity control
            x_vel = x_vel if x_vel is not None else 0.0
            theta_vel = theta_vel if theta_vel is not None else 0.0
            left_rpm_out, right_rpm_out = self._body_to_wheel_rpm(x_vel, theta_vel)

        self._send_command(
            {
                "T": 10010,
                "id": self.left_motor_id,
                "cmd": left_rpm_out,
                "act": act,
            }
        )
        self.serial.flush()
        time.sleep(0.005)

        self._send_command(
            {
                "T": 10010,
                "id": self.right_motor_id,
                "cmd": right_rpm_out,
                "act": act,
            }
        )
        self.serial.flush()

        self.last_left_rpm = left_rpm_out
        self.last_right_rpm = right_rpm_out

        action_sent = {
            "left_rpm": left_rpm_out,
            "right_rpm": right_rpm_out,
            "x.vel": x_vel,
            "theta.vel": theta_vel,
            "act": act,
        }
        action_sent[ACTION] = np.array(
            [left_rpm_out, right_rpm_out, x_vel, theta_vel], dtype=np.float32
        )
        return action_sent

    def stop_base(self) -> None:
        if self.serial:
            self._send_command(
                {"T": 10010, "id": self.left_motor_id, "cmd": 0, "act": self.accel_time}
            )
            self.serial.flush()
            time.sleep(0.005)
            self._send_command(
                {
                    "T": 10010,
                    "id": self.right_motor_id,
                    "cmd": 0,
                    "act": self.accel_time,
                }
            )
            self.serial.flush()
            self.last_left_rpm = 0.0
            self.last_right_rpm = 0.0
        logger.info("Base motors stopped")

    @check_if_not_connected
    def disconnect(self) -> None:
        self.stop_base()
        self._reader_stop.set()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if self.serial:
            self.serial.close()
        self._is_connected = False
        logger.info(f"{self} disconnected")

    def configure(self) -> None:
        pass
