#!/usr/bin/env python3
"""
Record robot trajectory from odometry and overlay commanded waypoints (converted to odom frame).
Only the plot image is saved; no CSV output.

Default:
    - Odometry topic: /odometry/filtered   (nav_msgs/Odometry)
    - Waypoint topic: /waypoint            (std_msgs/Float32MultiArray, robot-relative)
    - Output dir:    ~/fk/ME5413_Final_Project/visualnav-transformer/deployment/trajectory
Run:
    python trajectory_with_waypoints.py
    # 可选参数：
    # python trajectory_with_waypoints.py --odom-topic /odom --waypoint-topic /my_waypoint --log-dir ~/fk/logs
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion


def normalize_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (theta + math.pi) % (2 * math.pi) - math.pi


@dataclass
class OdometrySample:
    stamp: float
    x: float
    y: float
    yaw: Optional[float] = None
    linear_speed: Optional[float] = None


@dataclass
class WaypointSample:
    stamp: float
    rel_dx: float
    rel_dy: float
    x: float  # converted to odom/world coordinates
    y: float
    yaw: Optional[float] = None  # global yaw if hx/hy provided


class TrajectoryWithWaypointsLogger:
    """Stores odometry poses and waypoints, exports only PNG."""

    def __init__(
        self,
        log_dir: str | Path,
        plot_name: str = "trajectory_with_waypoints.png",
    ) -> None:
        self._odom_samples: List[OdometrySample] = []
        self._waypoint_samples: List[WaypointSample] = []

        self._log_dir = Path(log_dir).expanduser()
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._plot_path = self._log_dir / plot_name

    @property
    def odom_samples(self) -> Sequence[OdometrySample]:
        return tuple(self._odom_samples)

    @property
    def waypoint_samples(self) -> Sequence[WaypointSample]:
        return tuple(self._waypoint_samples)

    def record_odom(
        self,
        x: float,
        y: float,
        yaw: Optional[float],
        linear_speed: Optional[float],
        stamp: float,
    ) -> None:
        self._odom_samples.append(
            OdometrySample(
                stamp=stamp,
                x=float(x),
                y=float(y),
                yaw=None if yaw is None else float(yaw),
                linear_speed=None if linear_speed is None else float(linear_speed),
            )
        )

    def record_waypoint(
        self,
        rel_dx: float,
        rel_dy: float,
        world_x: float,
        world_y: float,
        yaw: Optional[float],
        stamp: float,
    ) -> None:
        self._waypoint_samples.append(
            WaypointSample(
                stamp=stamp,
                rel_dx=float(rel_dx),
                rel_dy=float(rel_dy),
                x=float(world_x),
                y=float(world_y),
                yaw=None if yaw is None else float(yaw),
            )
        )

    def plot(
        self,
        title: str = "Trajectory & Waypoints",
        dpi: int = 150,
        annotate_waypoints: bool = False,
    ) -> Path:
        if not self._odom_samples:
            raise RuntimeError("No odometry samples captured yet.")

        xs = [s.x for s in self._odom_samples]
        ys = [s.y for s in self._odom_samples]

        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.plot(xs, ys, linewidth=2, color="#1f77b4", label="Odometry path")
        ax.scatter(xs[0], ys[0], color="green", marker="o", s=80, label="Start")
        ax.scatter(xs[-1], ys[-1], color="red", marker="x", s=80, label="End")

        if self._waypoint_samples:
            wx = [w.x for w in self._waypoint_samples]
            wy = [w.y for w in self._waypoint_samples]
            ax.scatter(wx, wy, color="#ff7f0e", marker=".", s=10, label="Waypoints")
            if annotate_waypoints:
                for idx, (x_pt, y_pt) in enumerate(zip(wx, wy)):
                    ax.text(
                        x_pt,
                        y_pt,
                        f"{idx}",
                        fontsize=8,
                        color="#ff7f0e",
                        ha="center",
                        va="bottom",
                    )

        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self._plot_path)
        plt.close(fig)

        return self._plot_path


class TrajectoryRecorderNode:
    """ROS node subscribing to odometry and waypoint topics, saving only PNG."""

    def __init__(
        self,
        odom_topic: str,
        waypoint_topic: str,
        log_dir: str,
        annotate_waypoints: bool,
    ) -> None:
        self._logger = TrajectoryWithWaypointsLogger(log_dir=log_dir)
        self._odom_topic = odom_topic
        self._waypoint_topic = waypoint_topic
        self._annotate_waypoints = annotate_waypoints
        self._last_odom: Optional[OdometrySample] = None

        rospy.Subscriber(self._odom_topic, Odometry, self._odom_cb, queue_size=50)
        rospy.Subscriber(self._waypoint_topic, Float32MultiArray, self._waypoint_cb, queue_size=50)
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "Recording odometry from %s and waypoints from %s",
            self._odom_topic,
            self._waypoint_topic,
        )

    def _odom_cb(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        twist = msg.twist.twist

        x = pose.position.x
        y = pose.position.y
        quat = pose.orientation
        yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]

        linear_speed = math.sqrt(
            twist.linear.x ** 2 + twist.linear.y ** 2 + twist.linear.z ** 2
        )

        stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0 else time.time()

        sample = OdometrySample(
            stamp=stamp,
            x=float(x),
            y=float(y),
            yaw=float(yaw),
            linear_speed=float(linear_speed),
        )
        self._logger.record_odom(
            x=sample.x,
            y=sample.y,
            yaw=sample.yaw,
            linear_speed=sample.linear_speed,
            stamp=sample.stamp,
        )
        self._last_odom = sample

    def _waypoint_cb(self, msg: Float32MultiArray) -> None:
        if self._last_odom is None:
            rospy.logwarn_throttle(5.0, "Waypoint received before any odometry; skipping.")
            return

        data = list(msg.data)
        if len(data) < 2:
            rospy.logwarn_throttle(5.0, "Waypoint message shorter than 2 elements; skipping.")
            return

        rel_dx, rel_dy = data[0], data[1]

        # Convert waypoint from robot frame to odom/world frame using latest pose.
        robot_yaw = self._last_odom.yaw if self._last_odom.yaw is not None else 0.0
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        world_x = self._last_odom.x + rel_dx * cos_yaw - rel_dy * sin_yaw
        world_y = self._last_odom.y + rel_dx * sin_yaw + rel_dy * cos_yaw

        way_yaw = None
        if len(data) >= 4:
            hx, hy = data[2], data[3]
            if not math.isclose(hx, 0.0, abs_tol=1e-6) or not math.isclose(hy, 0.0, abs_tol=1e-6):
                rel_yaw = math.atan2(hy, hx)
                way_yaw = normalize_angle(robot_yaw + rel_yaw)

        stamp = rospy.get_time()
        if stamp == 0.0:
            stamp = time.time()

        self._logger.record_waypoint(
            rel_dx=rel_dx,
            rel_dy=rel_dy,
            world_x=world_x,
            world_y=world_y,
            yaw=way_yaw,
            stamp=stamp,
        )

    def _on_shutdown(self) -> None:
        if not self._logger.odom_samples:
            rospy.logwarn("No odometry samples received; nothing to save.")
            return
        try:
            plot_path = self._logger.plot(annotate_waypoints=self._annotate_waypoints)
            rospy.loginfo("Saved combined plot to %s", plot_path)
        except Exception as exc:  # pylint: disable=broad-except
            rospy.logerr("Failed to export trajectory/waypoints: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record odometry trajectory with waypoint overlays.")
    parser.add_argument(
        "--odom-topic",
        default="/odometry/filtered",
        help="Odometry topic (nav_msgs/Odometry) (default: /odometry/filtered)",
    )
    parser.add_argument(
        "--waypoint-topic",
        default="/waypoint",
        help="Waypoint topic (std_msgs/Float32MultiArray, robot-relative) (default: /waypoint)",
    )
    parser.add_argument(
        "--log-dir",
        default="~/fk/ME5413_Final_Project/visualnav-transformer/deployment/trajectory",
        help="Directory for PNG output (default: ~/fk/ME5413_Final_Project/visualnav-transformer/deployment/trajectory)",
    )
    parser.add_argument(
        "--annotate-waypoints",
        action="store_true",
        help="Annotate waypoint indices on the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rospy.init_node("trajectory_with_waypoints_logger", anonymous=True)
    TrajectoryRecorderNode(
        odom_topic=args.odom_topic,
        waypoint_topic=args.waypoint_topic,
        log_dir=args.log_dir,
        annotate_waypoints=args.annotate_waypoints,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
