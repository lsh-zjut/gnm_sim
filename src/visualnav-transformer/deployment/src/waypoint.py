#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
waypoint_analyzer.py

解析waypoint话题中的数据，计算并显示距离和角度信息。
"""

import math
import rospy
from std_msgs.msg import Float32MultiArray

class WaypointAnalyzer:
    def __init__(self):
        rospy.init_node('waypoint_analyzer', anonymous=False)
        
        # 从topic_names导入或直接定义waypoint话题名称
        self.waypoint_topic = rospy.get_param('~waypoint_topic', '/waypoint')
        
        # 订阅waypoint话题
        self.sub = rospy.Subscriber(self.waypoint_topic, Float32MultiArray, self.waypoint_callback)
        
        rospy.loginfo(f"Waypoint Analyzer已启动，正在监听话题: {self.waypoint_topic}")
        rospy.loginfo("格式说明: 距离(m), 角度(度)")
        rospy.loginfo("-" * 50)
    
    def waypoint_callback(self, msg):
        """处理接收到的waypoint消息"""
        data = msg.data
        
        if len(data) >= 2:
            # 提取x和y坐标
            x = data[0]
            y = data[1]
            
            # 计算距离
            distance = math.hypot(x, y)
            
            # 计算角度（弧度转度）
            angle_rad = math.atan2(y, x)
            angle_deg = math.degrees(angle_rad)
            
            # 打印结果
            rospy.loginfo(f"距离: {distance:.3f} m, 角度: {angle_deg:.2f}°")
            
            # 如果有多个waypoint（如GNM模型预测的轨迹点），分别计算每个点
            if len(data) > 4 and len(data) % 2 == 0:
                rospy.loginfo("轨迹点分析:")
                for i in range(0, min(len(data), 10), 2):  # 最多显示5个点
                    wx = data[i]
                    wy = data[i+1]
                    w_distance = math.hypot(wx, wy)
                    w_angle_rad = math.atan2(wy, wx)
                    w_angle_deg = math.degrees(w_angle_rad)
                    rospy.loginfo(f"  点{i//2}: 距离={w_distance:.3f}m, 角度={w_angle_deg:.2f}°")
                rospy.loginfo("-" * 50)
        else:
            rospy.logwarn(f"接收到的waypoint数据不完整，数据长度: {len(data)}, 内容: {data}")
    
    def run(self):
        """保持节点运行"""
        rospy.spin()

if __name__ == '__main__':
    try:
        analyzer = WaypointAnalyzer()
        analyzer.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Waypoint Analyzer节点已停止")