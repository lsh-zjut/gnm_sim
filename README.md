# gnm_sim 工作空间

基于 ROS Noetic 的仿真工作空间，包含 `sim_world`（世界/launch）和 `jackal_description`（Jackal 机器人描述）等包，用于在 Gazebo 中加载世界并生成 Jackal 机器人。

## 环境准备
- Ubuntu 20.04 + ROS Noetic（已安装 `ros-desktop-full` 或包含 `gazebo_ros` 的安装）。
- Python3 工具：`sudo apt-get install python3-rosdep python3-empy python3-vcstool build-essential`
- 初始化 rosdep（只需一次）：  
  ```bash
  sudo rosdep init        # 若已执行过可跳过
  rosdep update
  ```

## 克隆与构建
1) 克隆仓库：  
   ```bash
   git clone https://github.com/lsh-zjut/gnm_sim.git                 
   ```
2) 设置环境并安装依赖：  
   ```bash
   source /opt/ros/noetic/setup.bash
   rosdep install --from-paths src --ignore-src -r -y
   ```
3) 编译：  
   ```bash
   catkin_make
   ```
4) 加载工作空间环境：  
   ```bash
   source devel/setup.bash
   ```

## 运行示例
- 启动前准备 Gazebo 模型：将 `models` 放入 `~/.gazebo/`，否则启动gazebo为黑屏。模型可在此处下载：<https://drive.google.com/drive/folders/15ZlNQRygDhuBKT8wAKXB_oRsIn1Kpv4V>
- 启动世界并生成 Jackal，`world0.launch`为室内环境，`world1.launch`为室外环境：  
  ```bash
  roslaunch sim_world world0.launch
  ```

## 常用话题
- 速度指令：`/cmd_vel`（经 `twist_mux` 汇总，内部发往 `/jackal_velocity_controller/cmd_vel`）
- 里程计：`/jackal_velocity_controller/odom`，EKF 融合后 `odometry/filtered`
- 关节状态：`/joint_states`
- 激光：`/front/scan`
- 相机：`/front/rgb/image_raw`、`/front/depth/image_raw`、`/front/depth/points` 

## 调整小车相机视野/距离
- 默认深度相机参数位置：`src/jackal_description/urdf/accessories/kinect.urdf.xacro`
  - `<horizontal_fov>` 控制视场角（弧度）。
  - `<clip><near>` / `<clip><far>` 控制最近/最远距离。
  - 修改后重新 `roslaunch sim_world world.launch` 即会生效（xacro 会重新生成 URDF）。
- 如果用 pointgrey/flea3 等相机，参数在 `src/jackal_description/urdf/accessories.urdf.xacro` 调用的相机宏里，可调整 `hfov`、`width`、`height` 等。

## 常用操作
- 清理后重编译：  
  ```bash
  rm -rf build devel
  catkin_make
  ```
- 每次新开终端运行前，记得 `source devel/setup.bash`（或追加到 `~/.bashrc`）。
