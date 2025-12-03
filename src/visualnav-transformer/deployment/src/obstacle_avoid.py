#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray  
from geometry_msgs.msg import Point  

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "../config/robot.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)

MAX_V = float(CONFIG.get("max_v", 1.0))
MAX_W = float(CONFIG.get("max_w", 0.6))
VEL_TOPIC = str(CONFIG.get("vel_teleop_topic", "/cmd_vel"))
SCAN_TOPIC = str(CONFIG.get("base_scan_topic", "/front/scan"))
ODOM_TOPIC = str(CONFIG.get("odom_topic", "/odometry/filtered"))
FRAME_RATE = float(CONFIG.get("frame_rate", 10.0))
DT = 1 / 15 #离散时间步长

WAYPOINT_TOPIC = "/waypoint"
REACHED_GOAL_TOPIC = "/topoplan/reached_goal"

# 超时处理
WAYPOINT_TIMEOUT = 1.5
LIDAR_TIMEOUT = 0.4
ODOM_TIMEOUT = 0.4

ACC_V = 10
ACC_W = 10

TRAJ_HORIZON = 25
SAMPLES_V = 18
SAMPLES_W = 20

CLEARANCE_HARD = 0.18
ROBOT_RADIUS = 0.20

WEIGHT_GOAL = 8 #8
WEIGHT_TIME = 5 #5
WEIGHT_OBS = 1.8 #1.8
WEIGHT_SMOOTH = 2 #2
WEIGHT_ORIENT = 1.0 #1.0

MIN_LINEAR_CMD = 0


@dataclass
class RobotState:
    x: float
    y: float
    yaw: float
    v: float
    w: float


@dataclass
class VelocityCommand:
    linear: float
    angular: float
    cost: float


class TimeStampedData:
    def __init__(self) -> None:
        self.stamp: float = float("-inf")
        self.data = None

    def update(self, data) -> None:
        self.data = data
        self.stamp = rospy.get_time()

    def valid(self, timeout: float) -> bool:
        return (rospy.get_time() - self.stamp) <= timeout


class TEBLocalPlanner:
    def __init__(self) -> None:
        rospy.init_node("TEB_CONTROLLER", anonymous=False)

        self._waypoint = TimeStampedData()
        self._scan = TimeStampedData()
        self._odom = TimeStampedData()
        self._goal_reached = False
        
        # 路径长度记录相关变量
        self._total_path_length = 0.0  # 总路径长度
        self._last_position = None  # 上一次位置记录
        self._odom_received = False  # 里程计数据接收标志
        
        # 导航状态检测变量
        self._navigating = False  # 导航状态标志
        self._is_moving = False  # 当前是否在运动
        self._previous_moving_state = False  # 上一次运动状态
        self._stop_threshold = 0.01  # 停止阈值（线速度和角速度）
        
        # 导航时间记录变量
        self._navigation_start_time = None  # 导航开始时间
        
        self._last_cmd = Twist()

        rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, self._waypoint_cb, queue_size=1)
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self._scan_cb, queue_size=1)
        rospy.Subscriber(ODOM_TOPIC, Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, self._goal_cb, queue_size=1)

        self._cmd_pub = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)

        # 最优轨迹
        self._path_pub = rospy.Publisher("~planned_path", Path, queue_size=1)
        # 所有“可行采样轨迹”的集合
        self._samples_pub = rospy.Publisher("~sampled_paths", MarkerArray, queue_size=1)

        rate_hz = max(int(math.ceil(1.0 / DT)), 10)
        self._rate = rospy.Rate(rate_hz)

        self._front_thresh = float(rospy.get_param("~obs_front_threshold", 0.60))
        self._side_margin = float(rospy.get_param("~obs_side_margin", 0.10))
        self._last_obstacle_side = "clear"

        rospy.loginfo("TEB-inspired controller ready. Publishing to %s", VEL_TOPIC)

    # ----------------------------- callbacks -----------------------------

    def _waypoint_cb(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            rospy.logwarn_throttle(2.0, "TEB: waypoint has fewer than 2 entries, ignoring.")
            return
        self._waypoint.update(np.array(msg.data[:4], dtype=np.float32))

    def _scan_cb(self, msg: LaserScan) -> None:
        self._scan.update(msg)

    def _odom_cb(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        twist = msg.twist.twist
        yaw = self._quat_to_yaw(pose.orientation)
        state = RobotState(
            x=pose.position.x,
            y=pose.position.y,
            yaw=yaw,
            v=twist.linear.x,
            w=twist.angular.z,
        )
        self._odom.update(state)
        
        # 记录当前位置用于路径长度计算
        current_position = (state.x, state.y)
        
        # 检测机器人是否在运动（基于里程计的线速度和角速度）
        self._is_moving = abs(state.v) > self._stop_threshold or abs(state.w) > self._stop_threshold
        
        # 里程计数据首次接收时记录初始位置
        if not self._odom_received:
            self._last_position = current_position
            self._odom_received = True
            # rospy.loginfo("里程计数据已接收，准备记录路径长度和导航时间")
        
        # 更新运动状态并处理路径长度和时间记录
        if self._odom_received and self._last_position is not None:
            # 如果机器人从静止变为运动，开始记录新路径和导航时间
            if self._is_moving and not self._previous_moving_state:
                self._navigating = True
                # 记录导航开始时间
                if self._navigation_start_time is None:
                    self._navigation_start_time = rospy.Time.now()
                    rospy.loginfo(f"开始记录（起始时间: {self._navigation_start_time}）")
            
            # 在导航状态下使用里程计数据累加路径长度
            if self._navigating and self._is_moving:
                # 计算与上一次位置的距离增量
                dx = current_position[0] - self._last_position[0]
                dy = current_position[1] - self._last_position[1]
                distance_increment = math.hypot(dx, dy)
                # 只有当距离增量大于阈值时才累加，避免噪声干扰
                if distance_increment > 0.001:
                    self._total_path_length += distance_increment
            
            # 记录本次位置用于下次计算
            self._last_position = current_position
            
        # 更新运动状态记录
        self._previous_moving_state = self._is_moving

    def _goal_cb(self, msg: Bool) -> None:
        previous_state = self._goal_reached
        self._goal_reached = msg.data
        
        # 如果从未到达变为已到达且在导航状态，打印信息并退出程序
        if not previous_state and self._goal_reached and self._navigating:
            rospy.loginfo("目标点已到达！")
            rospy.loginfo(f"总路径长度(基于里程计) = {self._total_path_length:.4f} 米")
            
            # 计算并打印导航时间
            if self._navigation_start_time is not None:
                navigation_duration = (rospy.Time.now() - self._navigation_start_time).to_sec()
                rospy.loginfo(f"总导航时间 = {navigation_duration:.2f} 秒")
            
            # rospy.loginfo("导航完成，正在退出程序...")
            # 优雅地关闭ROS节点
            rospy.signal_shutdown("导航目标已到达，程序退出")
            # 确保进程完全退出
            sys.exit(0)

    # ----------------------------- utils -----------------------------

    @staticmethod
    def _quat_to_yaw(orientation) -> float:
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _wrap_angle(theta: float) -> float:
        return (theta + math.pi) % (2.0 * math.pi) - math.pi

    # ----------------------------- main loop -----------------------------

    def spin(self) -> None:
        # 用于检测机器人停止的计数器
        stop_counter = 0
        stop_threshold_count = 15  # 连续检测到停止多少次才认为真正停止
        
        while not rospy.is_shutdown():
            cmd = Twist()

            if self._goal_reached:
                self._publish_stop(cmd)
                rospy.loginfo("目标点已到达，准备停止机器人...")
                # 确保机器人完全停止
                self._rate.sleep()
                # 当_goal_reached为True时，_goal_cb中已设置退出逻辑，这里无需额外处理
                continue

            if not self._inputs_ready():
                self._publish_stop(cmd)
                rospy.logwarn_throttle(2.0, "inputs not ready (waypoint/scan/odom).")
                self._rate.sleep()
                continue
            
            # 检查机器人是否已经停止运动，如果是且处于导航状态
            if self._navigating:
                if not self._is_moving:
                    stop_counter += 1
                else:
                    stop_counter = 0
                
                # 如果连续检测到机器人停止，打印路径长度和导航时间并重置
                if stop_counter >= stop_threshold_count:                    
                    rospy.loginfo(f"总路径长度 = {self._total_path_length:.4f} 米")
                    
                    # 计算并打印导航时间
                    if self._navigation_start_time is not None:
                        navigation_duration = (rospy.Time.now() - self._navigation_start_time).to_sec()
                        rospy.loginfo(f"总导航时间 = {navigation_duration:.2f} 秒")
                    
                    # 重置所有状态
                    self._navigating = False
                    self._total_path_length = 0.0
                    self._navigation_start_time = None
                    stop_counter = 0

            waypoint = self._waypoint.data
            rel_goal = np.array(waypoint[:2], dtype=np.float32)

            desired_heading: Optional[float] = None
            if waypoint is not None and len(waypoint) >= 4:
                heading_vec = waypoint[2:4]
                if np.linalg.norm(heading_vec) > 1e-4:
                    desired_heading = math.atan2(heading_vec[1], heading_vec[0])

            best_cmd = self._plan(self._odom.data, rel_goal, self._scan.data, desired_heading)
            cmd.linear.x = best_cmd.linear
            cmd.angular.z = best_cmd.angular

            self._cmd_pub.publish(cmd)
            rospy.loginfo_throttle(1.0, "v=%.2f m/s, w=%.2f rad/s", cmd.linear.x, cmd.angular.z)
            self._last_cmd = cmd
            self._rate.sleep()

    def _inputs_ready(self) -> bool:
        return (
            self._waypoint.valid(WAYPOINT_TIMEOUT)
            and self._scan.valid(LIDAR_TIMEOUT)
            and self._odom.valid(ODOM_TIMEOUT)
        )

    def _publish_stop(self, cmd: Twist) -> None:
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self._cmd_pub.publish(cmd)
        self._last_cmd = cmd

    # ----------------------------- planner -----------------------------

    def _plan(self, odom_state: RobotState, rel_goal: np.ndarray, scan: LaserScan, desired_heading: Optional[float]) -> VelocityCommand:
        dyn_window = self._compute_dynamic_window(odom_state.v, odom_state.w)
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        ranges[~np.isfinite(ranges)] = np.inf
        angles = scan.angle_min + np.arange(ranges.size, dtype=np.float32) * scan.angle_increment

        v_samples = np.linspace(dyn_window[0], dyn_window[1], SAMPLES_V)
        w_samples = np.linspace(dyn_window[2], dyn_window[3], SAMPLES_W)

        best = VelocityCommand(0.0, 0.0, float("inf"))
        best_traj = None
        feasible_trajs = []   # 存 (traj, cost)

        for v in v_samples:
            for w in w_samples:
                ...
                traj = self._simulate_band(v, w)
                feasible, min_clear = self._check_clearance(
                    traj, ranges, angles, scan.range_min, scan.range_max
                )
                if not feasible:
                    continue

                # 先算代价
                cost = self._evaluate_cost(
                    traj=traj,
                    rel_goal=rel_goal,
                    desired_heading=desired_heading,
                    clearance=min_clear,
                    v=v,
                    w=w,
                    prev_cmd=self._last_cmd,
                )

                # 存起来，后面排序
                feasible_trajs.append((traj, cost))

                # 更新最优
                if cost < best.cost:
                    best = VelocityCommand(v, w, cost)
                    best_traj = traj


        # 无解，修正
        if math.isinf(best.cost) or best_traj is None:
            # 清掉上一次的采样显示
            self._publish_sampled_trajs([], odom_state)
            return self._fallback_spin(dyn_window, ranges, angles)
        
        # 所有轨迹合集
        self._publish_sampled_trajs(feasible_trajs, odom_state)
        # 最佳轨迹
        self._publish_path(best_traj, odom_state)

        return best

    # ----------------------------- helpers -----------------------------

    def _compute_dynamic_window(self, current_v: float, current_w: float) -> Tuple[float, float, float, float]:
        v_min = max(-MAX_V, current_v - ACC_V * DT)
        v_max = min(MAX_V, current_v + ACC_V * DT)
        w_min = max(-MAX_W, current_w - ACC_W * DT)
        w_max = min(MAX_W, current_w + ACC_W * DT)
        return (v_min, v_max, w_min, w_max)

    def _simulate_band(self, v: float, w: float) -> np.ndarray:
        poses = np.zeros((TRAJ_HORIZON, 3), dtype=np.float32)
        x = 0.0
        y = 0.0
        yaw = 0.0
        for i in range(TRAJ_HORIZON):
            x += v * math.cos(yaw) * DT
            y += v * math.sin(yaw) * DT
            yaw += w * DT
            yaw = self._wrap_angle(yaw)
            poses[i, 0] = x
            poses[i, 1] = y
            poses[i, 2] = yaw
        return poses

    def _check_clearance(self, traj: np.ndarray, ranges: np.ndarray, angles: np.ndarray, rmin: float, rmax: float) -> Tuple[bool, float]:
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        pts = np.stack([xs, ys], axis=1)

        min_clearance = float("inf")
        feasible = True

        valid = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)
        if not np.any(valid):
            return True, float("inf")

        for i in range(traj.shape[0]):
            px, py, _ = traj[i]
            dists = np.linalg.norm(pts[valid] - np.array([px, py]), axis=1)
            local_min = float(np.min(dists)) - ROBOT_RADIUS
            min_clearance = min(min_clearance, local_min)
            if local_min < CLEARANCE_HARD:
                feasible = False
                break

        if not np.isfinite(min_clearance):
            min_clearance = float("inf")

        return feasible, min_clearance

    def _evaluate_cost(self, traj: np.ndarray, rel_goal: np.ndarray, desired_heading: Optional[float], clearance: float, v: float, w: float, prev_cmd: Twist) -> float:
        dx = rel_goal[0] - traj[-1, 0]
        dy = rel_goal[1] - traj[-1, 1]
        goal_cost = WEIGHT_GOAL * math.hypot(dx, dy)

        time_cost = WEIGHT_TIME * (1.0 - min(abs(v) / MAX_V, 1.0)) * 0.5

        if clearance == float("inf"):
            obs_cost = 0.0
        else:
            obs_cost = WEIGHT_OBS * math.exp(-3.0 * clearance / CLEARANCE_HARD)

        dv = v - prev_cmd.linear.x
        dw = w - prev_cmd.angular.z
        smooth_cost = WEIGHT_SMOOTH * 0.3 * (abs(dv) + 0.5 * abs(dw))

        orient_cost = 0.0
        if desired_heading is not None:
            yaw_end = traj[-1, 2]
            d_yaw = self._wrap_angle(desired_heading - yaw_end)
            orient_cost = WEIGHT_ORIENT * abs(d_yaw)

        return goal_cost + time_cost + obs_cost + smooth_cost + orient_cost

    def _fallback_spin(self, dyn_window: Tuple[float, float, float, float], ranges: np.ndarray, angles: np.ndarray) -> VelocityCommand:
        finite = np.isfinite(ranges)
        free = np.where(finite & (ranges > (ROBOT_RADIUS + CLEARANCE_HARD + 0.05)))[0]
        direction = 1.0
        if free.size > 0:
            left = np.sum((angles[free] > 0.0) & (angles[free] <= math.radians(90.0)))
            right = np.sum((angles[free] < 0.0) & (angles[free] >= -math.radians(90.0)))
            direction = 1.0 if left >= right else -1.0

        v = 0.0
        w = direction * min(max(abs(dyn_window[3]), abs(dyn_window[2])), 0.4)
        w = max(min(w, dyn_window[3]), dyn_window[2])
        if abs(w) < 0.08:
            w = 0.08 * direction

        return VelocityCommand(v, w, 0.0)

    # ----------------------------- path publish -----------------------------

    def _publish_path(self, band: np.ndarray, odom_state: RobotState) -> None:
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "odom"

        base_x = odom_state.x
        base_y = odom_state.y
        base_yaw = odom_state.yaw
        cos_y = math.cos(base_yaw)
        sin_y = math.sin(base_yaw)

        for i in range(band.shape[0]):
            bx, by, byaw = band[i]

            gx = base_x + bx * cos_y - by * sin_y
            gy = base_y + bx * sin_y + by * cos_y
            gyaw = base_yaw + byaw

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = gx
            pose.pose.position.y = gy
            pose.pose.position.z = 0.0

            qz = math.sin(gyaw * 0.5)
            qw = math.cos(gyaw * 0.5)
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw

            path_msg.poses.append(pose)

        self._path_pub.publish(path_msg)

    from geometry_msgs.msg import Point  # 顶上要有这个

    def _publish_sampled_trajs(self, traj_cost_list, odom_state):        
        ma = MarkerArray()
        now = rospy.Time.now()
        frame_id = "base_link"

        # 没轨迹就清掉
        if not traj_cost_list:
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = frame_id
            m.action = Marker.DELETEALL
            ma.markers.append(m)
            self._samples_pub.publish(ma)
            return

        # 按 cost 从小到大排
        traj_cost_list = sorted(traj_cost_list, key=lambda x: x[1])

        MAX_DRAW = 8  # 最多画 8 条，突出最优
        for idx, (band, cost) in enumerate(traj_cost_list[:MAX_DRAW]):
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = frame_id
            m.ns = "sampled_trajs"
            m.id = idx
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.01  # 线宽
            # 浅一点的蓝色
            m.color.r = 0.2
            m.color.g = 0.6
            m.color.b = 1.0
            m.color.a = 0.5

            for i in range(band.shape[0]):
                bx, by, _ = band[i]
                pt = Point()
                pt.x = float(bx)   # 直接用积分出来的base_link坐标
                pt.y = float(by)
                pt.z = 0.0
                m.points.append(pt)

            ma.markers.append(m)

        self._samples_pub.publish(ma)



def main() -> None:
    planner = TEBLocalPlanner()
    planner.spin()


if __name__ == "__main__":
    main()
