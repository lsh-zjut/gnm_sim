#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, ast
import argparse
import rospy
from geometry_msgs.msg import Pose, Twist, Quaternion
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, SetModelStateRequest

def parse_vec2(s, default):
    if s is None: return list(default)
    if isinstance(s, str): return [float(x) for x in ast.literal_eval(s)]
    return [float(s[0]), float(s[1])]

def parse_waypoints(s, default):
    if s is None: return [list(p) for p in default]
    return [[float(p[0]), float(p[1])] for p in ast.literal_eval(s)]

def yaw_to_quat(yaw):
    half = 0.5 * yaw
    return Quaternion(0.0, 0.0, math.sin(half), math.cos(half))

class ObstacleMover:
    def __init__(self, args):
        self.model_name = args.model_name
        self.pattern    = args.pattern          # line | circle | waypoints
        self.z          = float(args.z)
        self.follow_tangent = bool(args.follow_tangent)
        self.speed      = float(args.speed)     # m/s
        self.hz         = float(args.hz)

        # geometry
        self.p0     = parse_vec2(args.p0, [0.0, 0.0])
        self.p1     = parse_vec2(args.p1, [2.0, 0.0])
        self.center = parse_vec2(args.center, [0.0, 0.0])
        self.radius = float(args.radius)
        self.waypoints = parse_waypoints(args.waypoints, [[0,0],[2,0],[2,1],[0,1]])

        # state
        self.dir_sign = 1.0     # for ping-pong line
        self.line_s   = 0.0
        self.wp_idx   = 0
        self.seg_s    = 0.0
        self.t_prev   = rospy.Time.now()

        # service
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    def spin(self):
        rate = rospy.Rate(self.hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - self.t_prev).to_sec()
            # 防止某些暂停/卡顿产生的异常大 dt
            if dt <= 0.0 or dt > 0.2:
                dt = 1.0 / self.hz
            x, y, yaw = self._next_pose(dt)
            self._send(x, y, self.z, yaw)
            self.t_prev = now
            rate.sleep()

    # --- kinematics ---
    def _next_pose(self, dt):
        if self.pattern == "line":
            return self._line_pingpong(dt)
        elif self.pattern == "circle":
            return self._circle(dt)
        elif self.pattern == "waypoints":
            return self._waypoints(dt)
        else:
            rospy.logwarn_throttle(2.0, "Unknown pattern '%s', fallback to line", self.pattern)
            return self._line_pingpong(dt)

    def _line_pingpong(self, dt):
        p0, p1 = self.p0, self.p1
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
        L = max(1e-9, math.hypot(dx, dy))

        s = self.line_s + self.dir_sign * self.speed * dt
        if s > L:
            s = L - (s - L)
            self.dir_sign = -1.0
        elif s < 0.0:
            s = -s
            self.dir_sign = +1.0
        self.line_s = s

        r = s / L
        x = p0[0] + r * dx
        y = p0[1] + r * dy
        yaw = math.atan2(dy, dx) if self.follow_tangent else 0.0
        return x, y, yaw

    def _circle(self, dt):
        # 用累计角度而不是离散步长，避免“齿轮感”
        # 角速度 = v / R
        R = max(1e-6, self.radius)
        omega = self.speed / R
        # 用仿真时间积分角度
        theta = getattr(self, "_theta", 0.0) + omega * dt
        self._theta = theta

        cx, cy = self.center
        x = cx + R * math.cos(theta)
        y = cy + R * math.sin(theta)
        yaw = (theta + math.pi/2.0) if self.follow_tangent else 0.0
        return x, y, yaw

    def _waypoints(self, dt):
        if len(self.waypoints) < 2:
            return self._line_pingpong(dt)

        i = self.wp_idx
        j = (i + 1) % len(self.waypoints)
        p0 = self.waypoints[i]
        p1 = self.waypoints[j]
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
        L = max(1e-9, math.hypot(dx, dy))

        s = self.seg_s + self.speed * dt
        if s >= L:
            self.wp_idx = j
            self.seg_s  = 0.0
            s = 0.0
            # 进入下一段后，立即刷新段向量
            j = (self.wp_idx + 1) % len(self.waypoints)
            p0 = self.waypoints[self.wp_idx]
            p1 = self.waypoints[j]
            dx, dy = p1[0]-p0[0], p1[1]-p0[1]
            L = max(1e-9, math.hypot(dx, dy))

        self.seg_s = s
        r = s / L
        x = p0[0] + r * dx
        y = p0[1] + r * dy
        yaw = math.atan2(dy, dx) if self.follow_tangent else 0.0
        return x, y, yaw

    # --- send to Gazebo ---
    def _send(self, x, y, z, yaw):
        req = SetModelStateRequest()
        st = ModelState()
        st.model_name = self.model_name
        st.reference_frame = "world"
        st.pose.position.x = x
        st.pose.position.y = y
        st.pose.position.z = z
        st.pose.orientation = yaw_to_quat(yaw)
        st.twist = Twist()  # 不注入速度，避免奇异动力学
        req.model_state = st
        try:
            resp = self.set_state(req)
            if not resp.success:
                rospy.logwarn_throttle(1.0, "SetModelState failed: %s", resp.status_message)
        except rospy.ServiceException as e:
            rospy.logerr_throttle(1.0, "Service error: %s", str(e))

def build_argparser():
    ap = argparse.ArgumentParser(description="Move a (static) Gazebo model smoothly via /gazebo/set_model_state")
    ap.add_argument("--model-name", required=True, help="Gazebo <model name='...'>")
    ap.add_argument("--pattern", choices=["line", "circle", "waypoints"], default="line")
    ap.add_argument("--speed", type=float, default=0.3, help="Linear speed (m/s)")
    ap.add_argument("--hz", type=float, default=120.0, help="Update frequency")
    ap.add_argument("--z", type=float, default=0.0, help="Fixed Z height")
    ap.add_argument("--follow-tangent", action="store_true", help="Yaw follows path tangent")

    # line
    ap.add_argument("--p0", type=str, default=None, help='e.g. "[0.0, 0.0]"')
    ap.add_argument("--p1", type=str, default=None, help='e.g. "[2.0, 0.0]"')
    # circle
    ap.add_argument("--center", type=str, default=None, help='e.g. "[1.0, 1.0]"')
    ap.add_argument("--radius", type=float, default=1.0)
    # waypoints
    ap.add_argument("--waypoints", type=str, default=None, help='e.g. "[[0,0],[2,0],[2,1],[0,1]]"')
    return ap

def main():
    args = build_argparser().parse_args()
    rospy.init_node("move_obstacle_py", anonymous=True)
    mover = ObstacleMover(args)
    mover.spin()

if __name__ == "__main__":
    main()
