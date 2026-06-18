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
import time
from dataclasses import dataclass, field

import draccus
import zmq

from config_diffdrive import DiffDriveConfig, DiffDriveHostConfig
from diffdrive import DiffDrive

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DiffDriveServerConfig:
    robot: DiffDriveConfig = field(default_factory=DiffDriveConfig)
    host: DiffDriveHostConfig = field(default_factory=DiffDriveHostConfig)


class DiffDriveHost:
    def __init__(self, config: DiffDriveHostConfig, robot: DiffDrive):
        self.zmq_context = zmq.Context()
        self.zmq_cmd_socket = self.zmq_context.socket(zmq.PULL)
        self.zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_cmd_socket.bind(f"tcp://*:{config.port_zmq_cmd}")

        self.zmq_observation_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)
        self.zmq_observation_socket.bind(f"tcp://*:{config.port_zmq_observations}")

        self.connection_time_s = config.connection_time_s
        self.watchdog_timeout_ms = config.watchdog_timeout_ms
        self.max_loop_freq_hz = config.max_loop_freq_hz

        self.robot = robot

    def disconnect(self) -> None:
        self.zmq_observation_socket.close()
        self.zmq_cmd_socket.close()
        self.zmq_context.term()


@draccus.wrap()
def main(cfg: DiffDriveServerConfig):
    logging.info("Configuring DiffDrive")
    robot = DiffDrive(cfg.robot)

    logging.info("Connecting DiffDrive")
    robot.connect()

    logging.info("Starting DiffDriveHost")
    host = DiffDriveHost(cfg.host, robot)

    last_cmd_time = time.time()
    watchdog_active = False
    logger.info("Waiting for commands...")

    try:
        start = time.perf_counter()
        duration = 0
        while duration < host.connection_time_s:
            loop_start_time = time.time()

            try:
                msg = host.zmq_cmd_socket.recv_string(zmq.NOBLOCK)
                data = dict(json.loads(msg))
                _action_sent = robot.send_action(data)
                last_cmd_time = time.time()
                watchdog_active = False
            except zmq.Again:
                if not watchdog_active:
                    pass
            except Exception as e:
                logger.error("Message fetching failed: %s", e)

            now = time.time()
            if (
                now - last_cmd_time > host.watchdog_timeout_ms / 1000
            ) and not watchdog_active:
                logger.warning(
                    f"Command not received for more than {host.watchdog_timeout_ms} ms. Stopping the base."
                )
                watchdog_active = True
                robot.stop_base()

            observation = robot.get_observation()

            obs_to_send = {
                "left_rpm": float(observation["left_rpm"]),
                "right_rpm": float(observation["right_rpm"]),
                "timestamp": time.time(),
            }

            try:
                host.zmq_observation_socket.send_string(
                    json.dumps(obs_to_send), flags=zmq.NOBLOCK
                )
            except zmq.Again:
                logger.debug("Dropping observation, no client connected")

            elapsed = time.time() - loop_start_time
            time.sleep(max(1 / host.max_loop_freq_hz - elapsed, 0))
            duration = time.perf_counter() - start

        print("Cycle time reached.")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting...")
    finally:
        print("Shutting down DiffDrive Host.")
        robot.disconnect()
        host.disconnect()

    logger.info("Finished DiffDrive cleanly")


if __name__ == "__main__":
    main()
