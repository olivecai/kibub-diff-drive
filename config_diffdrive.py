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

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig


def diffdrive_cameras_config() -> dict[str, CameraConfig]:
    return {}


@RobotConfig.register_subclass("diffdrive")
@dataclass
class DiffDriveConfig(RobotConfig):
    port: str = "/dev/diff_drive"
    baud_rate: int = 115200

    left_motor_id: int = 1
    right_motor_id: int = 2

    left_direction: int = 1
    right_direction: int = -1

    wheel_diameter: float = 0.1
    base_width: float = 0.4

    max_rpm: float = 200.0
    max_wheel_rpm: float = 200.0
    accel_time: int = 50
    heartbeat_ms: int = 1000

    cameras: dict[str, CameraConfig] = field(default_factory=diffdrive_cameras_config)


@dataclass
class DiffDriveHostConfig:
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    connection_time_s: int = 86400

    watchdog_timeout_ms: int = 500

    max_loop_freq_hz: int = 100


@RobotConfig.register_subclass("diffdrive_client")
@dataclass
class DiffDriveClientConfig(RobotConfig):
    remote_ip: str = "localhost"
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    wheel_diameter: float = 0.1
    base_width: float = 0.4

    max_wheel_rpm: float = 200.0

    left_direction: int = 1
    right_direction: int = 1

    polling_timeout_ms: int = 5
    connect_timeout_s: int = 5

    cameras: dict[str, CameraConfig] = field(default_factory=diffdrive_cameras_config)
