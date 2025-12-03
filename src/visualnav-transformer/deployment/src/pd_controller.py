import numpy as np
import yaml
import math
import sys
from typing import Tuple

# ROS
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool

from topic_names import (WAYPOINT_TOPIC, 
				 REACHED_GOAL_TOPIC)
from ros_data import ROSData
from utils import clip_angle
from nav_msgs.msg import Odometry

# CONSTS
CONFIG_PATH = "../config/robot.yaml"
with open(CONFIG_PATH, "r") as f:
	robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
VEL_TOPIC = robot_config["vel_teleop_topic"]
# DT = 1/robot_config["frame_rate"]
DT = 1/15
RATE = 9
EPS = 1e-8
WAYPOINT_TIMEOUT = 1 # seconds # TODO: tune this
FLIP_ANG_VEL = np.pi/4

# GLOBALS
vel_msg = Twist()
waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
reached_goal = False
reverse_mode = False
current_yaw = None

# 路径长度和时间记录相关变量
total_path_length = 0.0  # 总路径长度
last_position = None  # 上一次位置记录
navigation_start_time = None  # 导航开始时间
odom_received = False  # 里程计数据接收标志
is_navigating = False  # 是否处于导航状态
stop_threshold = 0.01  # 停止阈值（线速度）

def clip_angle(theta) -> float:
	"""Clip angle to [-pi, pi]"""
	theta %= 2 * np.pi
	if -np.pi < theta < np.pi:
		return theta
	return theta - 2 * np.pi
      

def pd_controller(waypoint: np.ndarray) -> Tuple[float]:
	"""PD controller for the robot"""
	assert len(waypoint) == 2 or len(waypoint) == 4, "waypoint must be a 2D or 4D vector"
	if len(waypoint) == 2:
		dx, dy = waypoint
	else:
		dx, dy, hx, hy = waypoint
	# this controller only uses the predicted heading if dx and dy near zero
	if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
		v = 0
		w = clip_angle(np.arctan2(hy, hx))/DT		
	elif np.abs(dx) < EPS:
		v =  0
		w = np.sign(dy) * np.pi/(2*DT)
	else:
		v = dx / DT
		w = np.arctan(dy/dx) / DT
	v = np.clip(v, 0, MAX_V)
	w = np.clip(w, -MAX_W, MAX_W)
	return v, w


def callback_drive(waypoint_msg: Float32MultiArray):
	"""Callback function for the waypoint subscriber"""
	global vel_msg
	print("seting waypoint")
	waypoint.set(waypoint_msg.data)
	
	
def callback_reached_goal(reached_goal_msg: Bool):
	"""Callback function for the reached goal subscriber"""
	global reached_goal
	previous_state = reached_goal
	reached_goal = reached_goal_msg.data
	
	# 如果从未到达变为已到达且在导航状态，打印路径长度和导航时间并退出
	if not previous_state and reached_goal and is_navigating:
		rospy.loginfo("PD: 目标点已到达！")
		rospy.loginfo(f"PD: 总路径长度 = {total_path_length:.4f} 米")
		
		# 计算并打印导航时间
		if navigation_start_time is not None:
			navigation_duration = (rospy.Time.now() - navigation_start_time).to_sec()
			rospy.loginfo(f"PD: 总导航时间 = {navigation_duration:.2f} 秒")
		
		rospy.loginfo("PD: 导航完成，正在退出程序...")


def odom_callback(odom_msg):
	"""里程计数据回调函数，用于计算路径长度"""
	global total_path_length, last_position, odom_received, is_navigating, navigation_start_time
	
	# 获取当前位置
	current_position = (odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y)
	current_velocity = odom_msg.twist.twist.linear.x
	
	# 里程计数据首次接收时记录初始位置
	if not odom_received:
		last_position = current_position
		odom_received = True
		rospy.loginfo("PD: 里程计数据已接收，准备记录路径长度和导航时间")
	
	# 更新导航状态
	is_moving = abs(current_velocity) > stop_threshold
	
	# 如果机器人正在运动且不在导航状态，开始记录新路径和导航时间
	if is_moving and not is_navigating and odom_received:
		is_navigating = True
		navigation_start_time = rospy.Time.now()
		rospy.loginfo(f"PD: 检测到机器人开始运动，开始记录路径长度和导航时间（起始时间: {navigation_start_time}）")
	# 如果机器人停止运动，结束导航状态
	elif not is_moving and is_navigating and total_path_length > 0.1:  # 确保已经移动了一定距离
		is_navigating = False
		rospy.loginfo("PD: 检测到机器人停止运动")
		rospy.loginfo(f"PD: 总路径长度 = {total_path_length:.4f} 米")
		if navigation_start_time is not None:
			navigation_duration = (rospy.Time.now() - navigation_start_time).to_sec()
			rospy.loginfo(f"PD: 总导航时间 = {navigation_duration:.2f} 秒")
		total_path_length = 0.0  # 重置路径长度
		navigation_start_time = None  # 重置导航时间
	
	# 在导航状态下使用里程计数据累加路径长度
	if is_navigating and odom_received and last_position is not None:
		# 计算与上一次位置的距离增量
		dx = current_position[0] - last_position[0]
		dy = current_position[1] - last_position[1]
		distance_increment = math.hypot(dx, dy)
		# 只有当距离增量大于阈值时才累加，避免噪声干扰
		if distance_increment > 0.001:
			total_path_length += distance_increment
	
	# 记录本次位置用于下次计算
	last_position = current_position

def main():
	global vel_msg, reverse_mode
	rospy.init_node("PD_CONTROLLER", anonymous=False)
	waypoint_sub = rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, callback_drive, queue_size=1)
	reached_goal_sub = rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, callback_reached_goal, queue_size=1)
	# 从配置中获取正确的里程计话题
	odom_topic = robot_config["odom_topic"]
	odom_sub = rospy.Subscriber(odom_topic, Odometry, odom_callback, queue_size=1)
	vel_out = rospy.Publisher(VEL_TOPIC, Twist, queue_size=1)
	rate = rospy.Rate(RATE)
	print("Registered with master node. Waiting for waypoints...")
	print(f"路径长度和时间记录功能已启用，订阅主题: {odom_topic}")
	while not rospy.is_shutdown():
		vel_msg = Twist()
		if reached_goal:
			vel_out.publish(vel_msg)
			print("Reached goal! Stopping...")
			# 打印最终路径长度和时间
			rospy.loginfo(f"PD: 最终总路径长度 = {total_path_length:.4f} 米")
			if navigation_start_time is not None:
				navigation_duration = (rospy.Time.now() - navigation_start_time).to_sec()
				rospy.loginfo(f"PD: 最终总导航时间 = {navigation_duration:.2f} 秒")
			# 退出程序
			rospy.signal_shutdown("导航目标已到达，程序退出")
			sys.exit(0)
			return
		elif waypoint.is_valid(verbose=True):
			v, w = pd_controller(waypoint.get())
			if reverse_mode:
				v *= -1
			vel_msg.linear.x = v
			vel_msg.angular.z = w
			print(f"publishing new vel: {v}, {w}")
		vel_out.publish(vel_msg)
		rate.sleep()
	

if __name__ == '__main__':
	main()
