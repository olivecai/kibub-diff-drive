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
from functools import cached_property

import numpy as np

from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from lerobot.robots.robot import Robot
from config_diffdrive import DiffDriveClientConfig

logger = logging.getLogger(__name__)


class DiffDriveClient(Robot):
    config_class = DiffDriveClientConfig
    name = "diffdrive_client"

    def __init__(self, config: DiffDriveClientConfig):
        import zmq

        self._zmq = zmq
        super().__init__(config)
        self.config = config
        self.id = config.id

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations

        self.wheel_diameter = config.wheel_diameter
        self.wheel_radius = self.wheel_diameter / 2
        self.base_width = config.base_width
        self.left_direction = config.left_direction
        self.right_direction = config.right_direction
        self.max_wheel_rpm = config.max_wheel_rpm

        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s

        self.zmq_context = None
        self.zmq_cmd_socket = None
        self.zmq_observation_socket = None

        self.last_observation = {}

        self._is_connected = False
        self.logs = {}

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
        zmq = self._zmq
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PUSH)
        zmq_cmd_locator = f"tcp://{self.remote_ip}:{self.port_zmq_cmd}"
        self.zmq_cmd_socket.connect(zmq_cmd_locator)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PULL)
        zmq_observations_locator = (
            f"tcp://{self.remote_ip}:{self.port_zmq_observations}"
        )
        self.zmq_observation_socket.connect(zmq_observations_locator)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)

        # Don't wait for observations at connect time — the host may not have
        # sent any yet.  The first send_action will unblock the host's loop.
        self._is_connected = True

    def calibrate(self) -> None:
        pass

    def _poll_and_get_latest_message(self) -> str | None:
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self.zmq_observation_socket, zmq.POLLIN)

        try:
            socks = dict(poller.poll(self.polling_timeout_ms))
        except zmq.ZMQError as e:
            logging.error(f"ZMQ polling error: {e}")
            return None

        if self.zmq_observation_socket not in socks:
            return None

        last_msg = None
        while True:
            try:
                msg = self.zmq_observation_socket.recv_string(zmq.NOBLOCK)
                last_msg = msg
            except zmq.Again:
                break

        return last_msg

    def _parse_observation_json(self, obs_string: str) -> dict | None:
        try:
            return json.loads(obs_string)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON observation: {e}")
            return None

    def _build_observation(
        self, left_rpm: float, right_rpm: float
    ) -> tuple[dict[str, np.ndarray], RobotObservation]:
        left_angular = left_rpm * self.left_direction * 2 * np.pi / 60
        right_angular = right_rpm * self.right_direction * 2 * np.pi / 60

        left_linear = left_angular * self.wheel_radius
        right_linear = right_angular * self.wheel_radius

        x_vel = (left_linear + right_linear) / 2
        theta_vel = (right_linear - left_linear) / self.base_width

        flat_state = {
            "left_rpm": left_rpm,
            "right_rpm": right_rpm,
            "x.vel": x_vel,
            "theta.vel": theta_vel,
        }

        state_vec = np.array([left_rpm, right_rpm, x_vel, theta_vel], dtype=np.float32)
        obs_dict: RobotObservation = {**flat_state, OBS_STATE: state_vec}

        return {}, obs_dict

    def _get_data(self) -> tuple[dict[str, np.ndarray], RobotObservation]:
        latest_message_str = self._poll_and_get_latest_message()

        if latest_message_str is None:
            return {}, self.last_observation

        observation = self._parse_observation_json(latest_message_str)

        if observation is None:
            return {}, self.last_observation

        try:
            left_rpm = observation.get("left_rpm", 0.0)
            right_rpm = observation.get("right_rpm", 0.0)
            frames, obs_dict = self._build_observation(left_rpm, right_rpm)
            self.last_observation = obs_dict
            return frames, obs_dict
        except Exception as e:
            logging.error(
                f"Error processing observation data, serving last observation: {e}"
            )
            return {}, self.last_observation

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        frames, obs_dict = self._get_data()

        for cam_name, frame in frames.items():
            if frame is None:
                logging.warning("Frame is None")
                frame = np.zeros((640, 480, 3), dtype=np.uint8)
            obs_dict[cam_name] = frame

        for cam_name in self.config.cameras:
            if cam_name not in obs_dict:
                obs_dict[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)

        return obs_dict

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        try:
            self.zmq_cmd_socket.send_string(json.dumps(action), flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            logging.debug("Dropped action: DiffDrive Host not ready")

        actions = np.array(
            [action.get(k, 0.0) for k in self._state_ft.keys()], dtype=np.float32
        )

        action_sent = {key: actions[i] for i, key in enumerate(self._state_ft.keys())}
        action_sent[ACTION] = actions
        return action_sent

    @check_if_not_connected
    def disconnect(self) -> None:
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()
        self._is_connected = False
