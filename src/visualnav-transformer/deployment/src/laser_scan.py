#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
激光雷达障碍物分析器：解析并显示激光雷达数据中的障碍物信息
"""
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan

class ObstacleAnalyzer:
    def __init__(self):
        rospy.init_node('obstacle_analyzer', anonymous=True)
        # 订阅激光雷达话题
        self.scan_sub = rospy.Subscriber('/front/scan', LaserScan, self.scan_callback, queue_size=1)
        rospy.loginfo("障碍物分析器已启动，正在监听激光雷达数据...")
        rospy.spin()
    
    def scan_callback(self, scan_msg):
        """处理激光雷达数据回调"""
        # 解析激光雷达基本参数
        angle_min = scan_msg.angle_min
        angle_max = scan_msg.angle_max
        angle_increment = scan_msg.angle_increment
        range_min = scan_msg.range_min
        range_max = scan_msg.range_max
        ranges = np.array(scan_msg.ranges)
        
        print("="*60)
        print("激光雷达数据解析结果")
        print("="*60)
        
        # 计算障碍物信息
        valid_ranges = ranges[(ranges > range_min) & (ranges < range_max)]
        valid_count = len(valid_ranges)
        
        print(f"有效障碍物点数: {valid_count}")
        
        if valid_count > 0:
            # 找出最近和最远的障碍物
            min_dist = np.min(valid_ranges)
            max_dist = np.max(valid_ranges)
            
            print(f"\n障碍物距离统计:")
            print(f"最近障碍物距离: {min_dist:.2f}m")
            print(f"最远障碍物距离: {max_dist:.2f}m")
            
            # 找出最近障碍物的角度
            min_idx = np.argmin(ranges)
            min_angle = angle_min + min_idx * angle_increment
            min_angle_deg = np.degrees(min_angle)
            
            print(f"\n最近障碍物信息:")
            print(f"距离: {ranges[min_idx]:.2f}m")
            print(f"角度: {min_angle_deg:.1f}°")
            print(f"相对位置: X={ranges[min_idx]*np.cos(min_angle):.2f}m, Y={ranges[min_idx]*np.sin(min_angle):.2f}m")

if __name__ == '__main__':
    try:
        ObstacleAnalyzer()
    except rospy.ROSInterruptException:
        pass