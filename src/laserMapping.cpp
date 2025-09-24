/**
 * @file laserMapping.cpp
 * @brief FAST-LIO2 激光建图主程序 - 基于LOAM算法的实时激光雷达里程计与建图实现
 * 
 * 这是一个基于以下论文描述的算法的高级实现：
 *   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
 *     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
 * 
 * 主要功能：
 * - 实时激光雷达点云处理和特征提取
 * - 基于扩展卡尔曼滤波(EKF)的状态估计
 * - 增量式KD树地图构建和维护
 * - IMU与激光雷达数据融合
 * - 实时里程计和轨迹发布
 * 
 * 修改者: Livox               dev@livoxtech.com
 * 原作者: Ji Zhang, Carnegie Mellon University
 */

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above cop，yright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.，
// ================= 系统头文件 =================
#include <omp.h>        // OpenMP并行计算库
#include <mutex>        // 互斥锁，用于线程同步
#include <math.h>       // 数学函数库
#include <thread>       // C++线程库
#include <fstream>      // 文件流操作
#include <csignal>      // 信号处理
#include <chrono>       // 时间处理
#include <unistd.h>     // Unix标准函数
#include <Python.h>     // Python接口（可能用于调试或扩展功能）

// ================= 数学和几何库 =================
#include <so3_math.h>   // SO3群数学运算（旋转矩阵相关）
#include <Eigen/Core>   // Eigen线性代数库核心模块

// ================= ROS2相关头文件 =================
#include <rclcpp/rclcpp.hpp>                          // ROS2 C++客户端库
#include <nav_msgs/msg/odometry.hpp>                  // 里程计消息类型
#include <nav_msgs/msg/path.hpp>                      // 路径消息类型
#include <visualization_msgs/msg/marker.hpp>          // 可视化标记消息
#include <sensor_msgs/msg/point_cloud2.hpp>          // 点云消息类型
#include <sensor_msgs/msg/imu.hpp>                    // IMU消息类型
#include <std_srvs/srv/trigger.hpp>                   // 标准触发服务
#include <tf2_ros/transform_broadcaster.h>            // TF2坐标变换广播器
#include <geometry_msgs/msg/transform_stamped.hpp>    // 带时间戳的坐标变换消息
#include <geometry_msgs/msg/vector3.hpp>              // 三维向量消息

// ================= 点云处理库 =================
#include <pcl_conversions/pcl_conversions.h>  // PCL与ROS消息转换
#include <pcl/point_cloud.h>                  // PCL点云类
#include <pcl/point_types.h>                  // PCL点类型定义
#include <pcl/filters/voxel_grid.h>           // PCL体素网格滤波器
#include <pcl/io/pcd_io.h>                    // PCL PCD文件读写

// ================= 自定义头文件 =================
#include "IMU_Processing.hpp"                 // IMU数据处理类
#include "preprocess.h"                       // 点云预处理类
#include <ikd-Tree/ikd_Tree.h>                // 增量式KD树实现
#include <livox_ros_driver2/msg/custom_msg.hpp>  // Livox激光雷达驱动消息

// ================= 系统参数宏定义 =================
#define INIT_TIME           (0.1)      // EKF初始化时间阈值(秒)，系统启动后需要等待的时间
#define LASER_POINT_COV     (0.001)    // 激光点测量噪声协方差，用于EKF观测模型
#define MAXN                (720000)   // 最大点数限制，用于数组大小分配
#define PUBFRAME_PERIOD     (20)       // 发布帧周期，控制数据发布频率

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

// ================= 算法核心参数 =================
float res_last[100000] = {0.0};    // 上一次迭代的残差数组，存储每个点到平面的距离
float DET_RANGE = 300.0f;           // 激光雷达有效检测范围(米)
const float MOV_THRESHOLD = 1.5f;   // 地图移动阈值，当载体移动超过该值时更新本地地图
double time_diff_lidar_to_imu = 0.0; // LiDAR和IMU之间的时间差(秒)

// ================= 线程同步原语 =================
mutex mtx_buffer;              // 缓冲区互斥锁，保护数据缓冲区的线程安全
condition_variable sig_buffer; // 条件变量，用于线程间通知和等待

// ================= 路径和配置参数 =================
string root_dir = ROOT_DIR;                    // 根目录路径
string map_file_path, lid_topic, imu_topic;    // 地图文件路径、LiDAR话题名、IMU话题名

// ================= 算法状态变量 =================
double res_mean_last = 0.05;        // 上一次迭代的平均残差
double total_residual = 0.0;         // 当前迭代的总残差
double last_timestamp_lidar = 0;     // 最后一次LiDAR数据的时间戳
double last_timestamp_imu = -1.0;    // 最后一次IMU数据的时间戳

// ================= IMU噪声模型参数 =================
double gyr_cov = 0.1;       // 陀螺仪测量噪声协方差
double acc_cov = 0.1;       // 加速度计测量噪声协方差
double b_gyr_cov = 0.0001;  // 陀螺仪偏置噪声协方差
double b_acc_cov = 0.0001;  // 加速度计偏置噪声协方差

// ================= 点云处理参数 =================
double filter_size_corner_min = 0;  // 角点下采样滤波器的叶子大小
double filter_size_surf_min = 0;    // 平面点下采样滤波器的叶子大小
double filter_size_map_min = 0;     // 地图点下采样滤波器的叶子大小
double fov_deg = 0;                 // 激光雷达视场角度(度)

// ================= 地图和时间相关参数 =================
double cube_len = 0;           // 本地地图立方体的边长(米)
double HALF_FOV_COS = 0;       // 半视场角的余弦值
double FOV_DEG = 0;            // 实际使用的视场角度
double total_distance = 0;     // 累计行驶距离
double lidar_end_time = 0;     // 当前扫描结束时间
double first_lidar_time = 0.0; // 第一帧LiDAR数据的时间戳

// ================= 计数器和状态变量 =================
int effct_feat_num = 0;      // 有效特征点数量
int time_log_counter = 0;    // 时间日志计数器
int scan_count = 0;          // 扫描帧计数器
int publish_count = 0;       // 发布数据计数器
int iterCount = 0;           // 迭代计数器
int feats_down_size = 0;     // 下采样后的特征点数量，
int NUM_MAX_ITERATIONS = 0;  // EKF最大迭代次数
int laserCloudValidNum = 0;  // 有效激光点云数量
int pcd_save_interval = -1;  // PCD文件保存间隔
int pcd_index = 0;           // PCD文件索引

// ================= 状态标志数组和开关 =================
bool point_selected_surf[100000] = {0};  // 标记每个点是否被选为有效的平面点
bool lidar_pushed;           // 标记LiDAR数据是否已推入处理队列
bool flg_first_scan = true;  // 标记是否为第一次扫描
bool flg_exit = false;       // 程序退出标志
bool flg_EKF_inited;         // EKF是否已初始化
bool scan_pub_en = false;    // 是否启用扫描数据发布
bool dense_pub_en = false;   // 是否启用密集点云发布
bool scan_body_pub_en = false; // 是否启用载体系点云发布
bool is_first_lidar = true;    // 是否为第一帧LiDAR数据

// ================= 数据容器和缓冲区 =================
vector<vector<int>>  pointSearchInd_surf;  // 平面点的搜索索引容器
vector<BoxPointType> cub_needrm;           // 需要从地图中删除的盒子区域
vector<PointVector>  Nearest_Points;       // 每个点的最近邻点集合
vector<double>       extrinT(3, 0.0);      // LiDAR到IMU的外参平移向量
vector<double>       extrinR(9, 0.0);      // LiDAR到IMU的外参旋转矩阵(行优先存储)
deque<double>                     time_buffer;  // 时间戳缓冲队列
deque<PointCloudXYZI::Ptr>        lidar_buffer; // LiDAR点云数据缓冲队列
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer; // IMU数据缓冲队列

// ================= 点云数据指针 =================
PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());      // 从地图中提取的特征点
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());   // 去畔变后的特征点云
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());   // 载体系下采样后的特征点
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());  // 世界系下采样后的特征点
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));  // 每个点对应平面的法向量
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1)); // 原始激光点云(用于EKF观测)
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); // 对应的法向量(用于EKF观测)
PointCloudXYZI::Ptr _featsArray;                             // 特征点数组指针

// ================= PCL滤波器 =================
pcl::VoxelGrid<PointType> downSizeFilterSurf;  // 平面点体素网格下采样滤波器
pcl::VoxelGrid<PointType> downSizeFilterMap;   // 地图点体素网格下采样滤波器

// ================= 增量式KD树 =================
KD_TREE<PointType> ikdtree;  // 增量式KD树，用于快速最近邻搜索和动态地图维护

// ================= 坐标变换和空间参考点 =================
V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);   // 载体系X轴方向参考点，用于视场角计算
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);  // 世界系X轴方向参考点，由载体系变换而来
V3D euler_cur;                                 // 当前姿态的欧拉角表示(滚转、俯仰、偏航)
V3D position_last(Zero3d);                     // 上一时刻的位置
V3D Lidar_T_wrt_IMU(Zero3d);                   // LiDAR相对于IMU的平移向量(外参)
M3D Lidar_R_wrt_IMU(Eye3d);                    // LiDAR相对于IMU的旋转矩阵(外参)

// ================= 扩展卡尔曼滤波(EKF)输入输出 =================
/*** EKF数据结构和状态估计器 ***/
MeasureGroup Measures;                           // 测量数据组，包含LiDAR和IMU数据
esekfom::esekf<state_ikfom, 12, input_ikfom> kf; // 扩展卡尔曼滤波器，12维状态空间
state_ikfom state_point;                         // 当前状态点，包含位置、姿态、速度等
vect3 pos_lid;                                   // LiDAR在世界系下的位置

// ================= ROS2消息类型 =================
nav_msgs::msg::Path path;                    // ROS路径消息，存储载体的运动轨迹
nav_msgs::msg::Odometry odomAftMapped;       // 建图后的里程计消息
geometry_msgs::msg::Quaternion geoQuat;      // 几何四元数，表示姿态
geometry_msgs::msg::PoseStamped msg_body_pose; // 带时间戳的载体姿态消息

// ================= 数据处理器 =================
shared_ptr<Preprocess> p_pre(new Preprocess());  // 点云预处理器，负责特征提取和滤波
shared_ptr<ImuProcess> p_imu(new ImuProcess());   // IMU数据处理器，负责IMU预积分和去畔变

/**
 * @brief 信号处理函数 - 处理程序中断信号(Ctrl+C)
 * 
 * 当接收到SIGINT信号时，设置退出标志并清理资源
 * 保证程序能够优雅地关闭，保存必要的数据
 * 
 * @param sig 信号编号
 */
void SigHandle(int sig)
{
    flg_exit = true;                             // 设置全局退出标志
    std::cout << "catch sig %d" << sig << std::endl;  // 输出捕获到的信号
    sig_buffer.notify_all();                     // 通知所有等待的线程
    rclcpp::shutdown();                          // 关闭ROS2节点
}

/**
 * @brief 将LIO系统状态输出到日志文件
 * 
 * 将当前的系统状态(位置、姿态、速度、偏置等)写入文件
 * 用于后续的数据分析和轨迹重现
 * 
 * @param fp 文件指针，指向要写入的日志文件
 */
inline void dump_lio_state_to_log(FILE *fp)  
{
    // 将旋转矩阵转换为轴角表示
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);  // 相对时间戳
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));   // 旋转角度(rad)
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // 位置(m)
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                        // 角速度(当前未使用)
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // 线速度(m/s)
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                        // 加速度(当前未使用)
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // 陀螺仪偏置
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // 加速度计偏置
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // 重力向量
    fprintf(fp, "\r\n");   // 换行
    fflush(fp);           // 刷新文件缓冲区，确保数据写入
}

/**
 * @brief 将点从载体系变换到世界系(使用IKFOM状态)
 * 
 * 这是主要的坐标变换函数，将LiDAR载体系下的点变换到世界坐标系
 * 变换过程: 载体系 -> IMU系 -> 世界系
 * 公式: P_world = R_world_imu * (R_imu_lidar * P_lidar + T_imu_lidar) + T_world_imu
 * 
 * @param pi 输入点(载体系)
 * @param po 输出点(世界系)
 * @param s IKFOM状态，包含位置、姿态和外参信息
 */
void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    // 将点坐标转换为Eigen向量
    V3D p_body(pi->x, pi->y, pi->z);
    
    // 坐标变换: 载体系 -> IMU系 -> 世界系
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    // 将结果赋值给输出点
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;  // 保持强度信息不变
}


/**
 * @brief 将点从载体系变换到世界系(使用全局状态)
 * 
 * 使用全局状态变量 state_point 进行坐标变换
 * 这是上一个函数的简化版本，直接使用全局状态
 * 
 * @param pi 输入点(载体系)
 * @param po 输出点(世界系)
 */
void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    // 使用全局状态进行坐标变换
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

/**
 * @brief 模板函数 - 将矩阵形式的点从载体系变换到世界系
 * 
 * 这是一个模板函数，可以处理不同数值类型的矩阵
 * 主要用于雅可比矩阵计算和数值优化
 * 
 * @tparam T 数值类型(如double、float等)
 * @param pi 输入点矩阵(载体系)
 * @param po 输出点矩阵(世界系)
 */
template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    // 使用全局状态进行坐标变换
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);
    
    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

/**
 * @brief RGB点从载体系变换到世界系
 * 
 * 与PointBodyToWorld函数功能相同，但专门用于处理RGB点云
 * 保持了颜色信息和强度参数的一致性
 * 
 * @param pi 输入RGB点(载体系)
 * @param po 输出RGB点(世界系)
 */
void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;  // 保持强度/颜色信息
}

/**
 * @brief RGB点从LiDAR载体系变换到IMU载体系
 * 
 * 将点从LiDAR坐标系变换到IMU坐标系，不涉及世界坐标系
 * 主要用于发布载体系下的点云数据
 * 变换公式: P_imu = R_imu_lidar * P_lidar + T_imu_lidar
 * 
 * @param pi 输入RGB点(LiDAR系)
 * @param po 输出RGB点(IMU系)
 */
void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    // 仅做外参变换，LiDAR系 -> IMU系
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

/**
 * @brief 收集KD树中被删除的点的缓存
 * 
 * 从iKD树中获取最近被删除的点，用于内存管理和数据回收
 * 这些点可能是由于地图更新或者距离太远而被删除的
 * 
 * 注意：当前实现中只获取但未使用这些被删除的点
 */
void points_cache_collect()
{
    PointVector points_history;  // 存储被删除点的容器
    ikdtree.acquire_removed_points(points_history);  // 从iKD树中获取被删除的点
    
    // 可选: 将被删除的点添加到特征数组中(当前被注释)
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

// ================= 本地地图管理变量 =================
BoxPointType LocalMap_Points;        // 本地地图的边界盒

bool Localmap_Initialized = false;   // 本地地图是否已初始化

/**
 * @brief 基于LiDAR视场角(FOV)的地图分割和维护
 * 
 * 这是一个关键函数，负责维护本地地图的有效性和实时性
 * 主要功能：
 * 1. 初始化本地地图区域
 * 2. 检测载体是否移动到地图边缘
 * 3. 动态更新地图范围，删除远离区域的点
 * 4. 保持地图在可接受的大小范围内
 */
void lasermap_fov_segment()
{
    cub_needrm.clear();          // 清空需要删除的区域列表
    kdtree_delete_counter = 0;   // 重置删除计数器
    kdtree_delete_time = 0.0;    // 重置删除时间
    
    // 将载体系X轴参考点变换到世界系(用于FOV计算)
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;  // 获取当前LiDAR位置
    
    // 第一次初始化本地地图
    if (!Localmap_Initialized){
        // 以当前位置为中心创建立方体地图区域
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;  // 最小边界
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;  // 最大边界
        }
        Localmap_Initialized = true;
        return;
    }
    
    // 计算当前位置到地图边缘的距离
    float dist_to_map_edge[3][2];  // [xyz][min/max]
    bool need_move = false;
    
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]); // 到最小边界距离
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]); // 到最大边界距离
        
        // 如果任意方向上距离边界太近，则需要移动地图
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || 
            dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) 
            need_move = true;
    }
    
    if (!need_move) return;  // 如果不需要移动，直接返回
    
    // 计算地图移动距离和新的地图边界
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    
    // 计算每次移动的距离(保证地图大小稳定)
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, 
                         double(DET_RANGE * (MOV_THRESHOLD - 1)));
    
    // 对每个维度检查是否需要移动
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        
        // 如果当前位置接近最小边界，向负方向移动地图
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);  // 标记旧区域需要删除
        } 
        // 如果当前位置接近最大边界，向正方向移动地图
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);  // 标记旧区域需要删除
        }
    }
    
    LocalMap_Points = New_LocalMap_Points;  // 更新地图边界

    points_cache_collect();  // 收集被删除点的缓存
    
    // 实际执行点删除操作
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) 
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

/**
 * @brief 标准ROS2点云消息的回调函数
 * 
 * 处理从非-Livox激光雷达(如Velodyne, Ouster等)接收到的点云数据
 * 主要功能：
 * 1. 时间戳同步检查和异常处理
 * 2. 点云数据预处理(去噪、特征提取等)
 * 3. 数据缓冲区管理
 * 4. 性能统计和日志记录
 * 
 * @param msg 接收到的ROS2点云消息
 */
void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) 
{
    mtx_buffer.lock();  // 加锁保护缓冲区
    scan_count++;       // 扫描帧计数器递增
    
    double cur_time = get_time_sec(msg->header.stamp);  // 获取当前消息时间戳
    double preprocess_start_time = omp_get_wtime();     // 记录预处理开始时间
    
    // 检查时间戳是否向后回退(时间乱序检测)
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();  // 清空缓冲区以避免数据混乱
    }
    
    // 第一帧数据特殊处理
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    // 创建新的点云指针并进行预处理
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);  // 点云预处理(去噪、特征提取等)
    
    // 将处理后的数据存入缓冲区
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    
    // 记录预处理耗时用于性能分析
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    
    mtx_buffer.unlock();    // 解锁
    sig_buffer.notify_all(); // 通知处理线程有新数据到达
}

// ================= 时间同步相关变量 =================
double timediff_lidar_wrt_imu = 0.0;  // LiDAR相对于IMU的时间差(自动计算)
bool   timediff_set_flg = false;      // 时间差是否已设置的标志

/**
 * @brief Livox激光雷达自定义消息的回调函数
 * 
 * 处理Livox激光雷达的专用数据格式，相比标准点云消息：
 * 1. 支持更高精度的时间戳管理
 * 2. 具备自动时间同步功能
 * 3. 更好的数据一致性检查
 * 4. 专针Livox雷达特性的优化处理
 * 
 * @param msg Livox自定义消息指针
 */
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg) 
{
    mtx_buffer.lock();  // 加锁保护缓冲区
    
    double cur_time = get_time_sec(msg->header.stamp);  // 提取当前时间戳
    double preprocess_start_time = omp_get_wtime();     // 记录预处理开始时间
    scan_count++;  // 扫描帧计数器递增
    
    // 时间序列一致性检查
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();  // 清空缓冲区防止数据混乱
    }
    
    if(is_first_lidar)
    {
        is_first_lidar = false;  // 标记第一帧已处理
    }
    
    last_timestamp_lidar = cur_time;  // 更新最后一次LiDAR时间戳
    
    // === 时间同步检查和警告 ===
    // 在非同步模式下，检查IMU和LiDAR时间差是否过大
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && 
        !imu_buffer.empty() && !lidar_buffer.empty())
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",
               last_timestamp_imu, last_timestamp_lidar);
    }

    // === 自动时间同步功能 ===
    // 在启用时间同步且未设置时间差时，自动计算时间差
    if (time_sync_en && !timediff_set_flg && 
        abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        // 计算LiDAR相对于IMU的时间偏移（+0.1是经验值）
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    // === 点云数据处理和存储 ===
    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);  // Livox数据专用预处理
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    // 记录预处理性能数据
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    
    mtx_buffer.unlock();     // 解锁
    sig_buffer.notify_all(); // 通知处理线程
}

/**
 * @brief IMU数据回调函数
 * 
 * 处理来自IMU传感器的数据，主要功能包括：
 * 1. 时间戳同步和校正
 * 2. 数据缓冲区管理
 * 3. 异常数据检测和处理
 * 4. 与L主算法的数据同步
 * 
 * @param msg_in 接收到的IMU消息
 */
void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    publish_count++;  // 发布计数器递增
    
    // 创建消息副本用于时间戳修正
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
    
    // === 时间戳同步处理 ===
    // 第一种方式：使用预设的时间偏移
    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    
    // 第二种方式：使用自动计算的时间差(优先级更高)
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
    }

    double timestamp = get_time_sec(msg->header.stamp);  // 获取校正后的时间戳

    mtx_buffer.lock();  // 加锁保护缓冲区

    // === 数据一致性检查 ===
    // 检查时间戳是否向后回退(可能由于系统重启或数据丢失)
    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        imu_buffer.clear();  // 清空缓冲区以保证数据一致性
    }

    last_timestamp_imu = timestamp;  // 更新最后一次IMU时间戳

    // === 数据存储和通知 ===
    imu_buffer.push_back(msg);  // 将处理后的IMU数据放入缓冲区
    mtx_buffer.unlock();        // 解锁
    sig_buffer.notify_all();    // 通知处理线程有新数据
}

// ================= 数据同步相关变量 =================
double lidar_mean_scantime = 0.0;  // LiDAR平均扫描时间，用于自适应计算
int    scan_num = 0;               // 扫描次数计数器，用于平均值计算

/**
 * @brief 同步LiDAR和IMU数据包
 * 
 * 这是数据融合的核心函数，负责将异步到达的LiDAR和IMU数据
 * 组织成时间同步的测量组合。主要功能：
 * 1. 确保每个LiDAR扫描都有对应的IMU数据
 * 2. 处理不同传感器的采样率差异
 * 3. 动态计算扫描时间窗口
 * 4. 维护数据的时间顺序性
 * 
 * @param meas 输出的同步测量组，包含一帧LiDAR和多帧IMU数据
 * @return true 成功同步了一组数据
 * @return false 数据不足或时间不同步，需要等待更多数据
 */
bool sync_packages(MeasureGroup &meas)
{
    // === 数据可用性检查 ===
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;  // 任意一个缓冲区为空则无法同步
    }

    // === 处理LiDAR数据 ===
    if(!lidar_pushed)  // 如果还没有推入当前LiDAR数据
    {
        meas.lidar = lidar_buffer.front();          // 获取最早的LiDAR数据
        meas.lidar_beg_time = time_buffer.front();  // 设置扫描开始时间
        
        // === 动态计算扫描结束时间 ===
        if (meas.lidar->points.size() <= 1)  // 点数太少，使用平均扫描时间
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        // 如果最后一个点的时间戳太小，使用平均时间
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else  // 正常情况：使用最后一个点的时间戳
        {
            scan_num++;  // 更新扫描计数
            // 从最后一个点的curvature字段中提取时间戳(微秒 -> 秒)
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            
            // 更新平均扫描时间(递推平均)
            double current_scan_time = meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (current_scan_time - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;  // 设置扫描结束时间
        lidar_pushed = true;  // 标记LiDAR数据已推入
    }

    // === 检查IMU数据是否足够 ===
    if (last_timestamp_imu < lidar_end_time)
    {
        return false;  // IMU数据不足，需要等待更多IMU数据
    }

    // === 提取对应时间范围内的IMU数据 ===
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();  // 清空上一次的IMU数据
    
    // === 提取时间同步的IMU数据 ===
    // 循环提取所有在LiDAR扫描时间范围内的IMU数据
    // 这确保了IMU数据与LiDAR数据的时间一致性
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);  // 获取IMU时间戳
        if(imu_time > lidar_end_time) break;  // 超出时间范围则退出循环
        
        meas.imu.push_back(imu_buffer.front());  // 将数据添加到测量组
        imu_buffer.pop_front();  // 从缓冲队列中移除已处理的数据
        
        // 更新下一个IMU数据的时间（为下次循环准备）
        if (!imu_buffer.empty())
            imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    }

    // === 清理已处理的LiDAR数据 ===
    lidar_buffer.pop_front();  // 移除已处理的LiDAR数据
    time_buffer.pop_front();   // 移除对应的时间戳
    lidar_pushed = false;      // 重置推入标志以处理下一帧
    
    return true;  // 成功同步了一组数据
}

// ================= 地图增量更新相关变量 =================
int process_increments = 0;  // 处理增量计数器(当前未使用)

/**
 * @brief 增量式地图更新函数
 * 
 * 这是建图系统的核心函数之一，负责将新的特征点添加到全局地图中。
 * 主要功能：
 * 1. 坐标变换：将载体系点变换到世界系
 * 2. 重复检测：避免在同一区域添加过多相似点
 * 3. 智能下采样：根据点密度决定是否需要下采样
 * 4. 动态维护：保持地图的实时性和精度
 */
void map_incremental()
{
    // === 初始化存储容器 ===
    PointVector PointToAdd;            // 需要添加到地图且需要下采样的点
    PointVector PointNoNeedDownsample; // 需要添加到地图但不需下采样的点
    
    // 预分配内存提高性能
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    
    // === 核心处理循环：遍历所有下采样后的特征点 ===
    for (int i = 0; i < feats_down_size; i++)
    {
        // 步骤1：坐标变换 - 将点从载体系(LiDAR坐标系)变换到世界系
        // 使用当前的位姿估计进行变换
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        
        // 步骤2：智能地图管理 - 决定是否需要将该点添加到全局地图中
        // 条件检查：1) 存在最近邻点（表明在现有地图附近）
        //           2) EKF已初始化（位姿估计可靠）
        if (!Nearest_Points[i].empty() && flg_EKF_inited)  // 有最近邻点且EKF已初始化
        {
            const PointVector &points_near = Nearest_Points[i];  // 获取最近邻点集合
            bool need_add = true;  // 默认需要添加
            
            // === 计算网格中心点(用于下采样决策) ===
            PointType mid_point;
            // 计算当前点所在体素网格的中心点
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            
            float dist = calc_dist(feats_down_world->points[i], mid_point);  // 当前点到中心的距离
            
            // === 检查是否需要下采样 ===
            // 如果最近邻点距离中心太远，则不需要下采样直接添加
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && 
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && 
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;  // 跳过后续处理
            }
            // === 重复点检测 ===
            // 检查最近邻点中是否已有点更接近网格中心
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                
                // 如果已有点更靠近中心，则不需要添加当前点
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            
            // 根据决策结果添加点
            if (need_add) 
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else  // 没有最近邻点或EKF未初始化，直接添加
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    // === 实际执行点添加操作 ===
    double st_time = omp_get_wtime();  // 记录开始时间
    
    // 添加需要下采样的点(第二个参数true表示启用下采样)
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    
    // 添加不需要下采样的点(第二个参数false表示禁用下采样)
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    
    // 统计总添加点数
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    
    // 记录增量更新耗时
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

// ================= 发布相关的点云数据 =================
PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());   // 等待发布的点云数据(用于地图发布)
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());  // 等待保存的点云数据(用于PCD文件)

/**
 * @brief 发布世界坐标系下的点云数据
 * 
 * 将当前帧的点云数据从载体系变换到世界系并发布。
 * 这是主要的点云可视化输出，用于RViz等工具显示。
 * 支持两种模式：密集点云或下采样点云
 * 
 * @param pubLaserCloudFull ROS2点云发布器指针
 */
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    // 检查是否启用扫描数据发布
    if(scan_pub_en)
    {
        // === 选择发布的点云数据类型 ===
        // dense_pub_en=true: 发布密集点云(feats_undistort)
        // dense_pub_en=false: 发布下采样点云(feats_down_body)
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        
        // 创建世界系点云容器
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        // === 坐标变换: 载体系 -> 世界系 ===
        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], 
                                &laserCloudWorld->points[i]);
        }

        // === 转换为ROS2消息并发布 ===
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);  // PCL -> ROS2消息转换
        
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);  // 设置时间戳
        laserCloudmsg.header.frame_id = "camera_init";              // 设置坐标系 ID
        
        pubLaserCloudFull->publish(laserCloudmsg);  // 发布点云数据
        publish_count -= PUBFRAME_PERIOD;           // 更新发布计数器
    }

    // ================= PCD文件保存功能(当前被禁用) =================
    /*
     * PCD文件保存功能说明：
     * 1. 需要确保系统有足够的内存空间
     * 2. PCD保存会显著影响实时性能，请谨慎使用
     * 3. 建议在离线处理或数据采集时启用
     * 4. 可通过修改pcd_save_en参数来启用此功能
     */
    /*
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        // 将所有点从载体系变换到世界系
        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;  // 累加点云数据

        static int scan_wait_num = 0;  // 等待扫描数计数器
        scan_wait_num++;
        
        // 按间隔保存PCD文件
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval)
        {
            pcd_index++;  // PCD文件索引递增
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);  // 保存为二进制PCD文件
            pcl_wait_save->clear();  // 清空缓存，
            scan_wait_num = 0;       // 重置计数器
        }
    }
    */
}

/**
 * @brief 发布载体坐标系(IMU系)下的点云数据
 * 
 * 将点云数据从激光雷达坐标系变换到IMU坐标系并发布。
 * 这对于需要在载体坐标系下分析数据的应用非常有用，
 * 比如路径规划、障碍物检测等。
 * 
 * @param pubLaserCloudFull_body 载体系点云发布器指针
 */
void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    // 创建IMU载体系点云容器
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    // === 坐标变换: LiDAR系 -> IMU系 ===
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], 
                            &laserCloudIMUBody->points[i]);
    }

    // === 转换为ROS2消息并发布 ===
    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);    // PCL -> ROS2消息转换
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);  // 设置时间戳
    laserCloudmsg.header.frame_id = "body";                     // 设置坐标系为载体系
    pubLaserCloudFull_body->publish(laserCloudmsg);             // 发布数据
    publish_count -= PUBFRAME_PERIOD;                          // 更新发布计数器
}

/**
 * @brief 发布有效特征点云(用于EKF观测的点)
 * 
 * 发布在EKF更新过程中实际使用的有效特征点。
 * 这些点通过了平面拟合检验且残差在可接受范围内。
 * 主要用于算法调试和性能分析。
 * 
 * @param pubLaserCloudEffect 有效点云发布器指针
 */
void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    // 创建世界系有效点云容器
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effct_feat_num, 1));
    
    // === 坐标变换: 载体系 -> 世界系 ===
    // 只变换在EKF中被使用的有效点
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], 
                            &laserCloudWorld->points[i]);
    }
    
    // === 转换为ROS2消息并发布 ===
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);          // PCL -> ROS2消息转换
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);   // 设置时间戳
    laserCloudFullRes3.header.frame_id = "camera_init";              // 设置坐标系 ID
    pubLaserCloudEffect->publish(laserCloudFullRes3);                 // 发布数据
}

/**
 * @brief 发布累积地图点云
 * 
 * 将当前帧的点云数据累加到全局地图中并发布。
 * 这个函数与 publish_frame_world 不同，是累积性的，
 * 会保存所有历史点云数据形成完整地图。
 * 注意：这会消耗大量内存，仅在必要时启用。
 * 
 * @param pubLaserCloudMap 地图点云发布器指针
 */
void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap)
{
    // === 选择发布的点云数据类型 ===
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    // === 坐标变换: 载体系 -> 世界系 ===
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i], 
                            &laserCloudWorld->points[i]);
    }
    
    // === 累积地图数据 ===
    // 将当前帧数据添加到全局地图中
    *pcl_wait_pub += *laserCloudWorld;

    // === 转换为ROS2消息并发布 ===
    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*pcl_wait_pub, laserCloudmsg);              // 发布累积地图
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time); // 设置时间戳
    laserCloudmsg.header.frame_id = "camera_init";            // 设置坐标系 ID
    pubLaserCloudMap->publish(laserCloudmsg);                 // 发布地图数据

    // === 可选: 发布KD树中的地图点(当前被注释) ===
    // 这里可以选择发布从 KD树中提取的地图点而不是累积点云
    // sensor_msgs::msg::PointCloud2 laserCloudMap;
    // pcl::toROSMsg(*featsFromMap, laserCloudMap);
    // laserCloudMap.header.stamp = get_ros_time(lidar_end_time);
    // laserCloudMap.header.frame_id = "camera_init";
    // pubLaserCloudMap->publish(laserCloudMap);
}

/**
 * @brief 将累积地图保存为PCD文件
 * 
 * 将在 pcl_wait_pub 中累积的所有点云数据保存为二进制PCD文件。
 * 这个函数通常在程序结束或接收到保存命令时调用。
 * PCD文件可以用于后续的地图分析、可视化或其他应用。
 */
void save_to_pcd()
{
    pcl::PCDWriter pcd_writer;  // 创建PCD数据写入器
    // 以二进制格式写入PCD文件(相比ASCII格式更紧凑、读写更快)
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
}

/**
 * @brief 模板函数 - 设置姿态消息的位置和方向
 * 
 * 通用的姿态设置函数，可以用于不同类型的ROS消息。
 * 从全局状态变量中提取位置和姿态信息，
 * 赋值给输出消息的对应字段。
 * 
 * @tparam T ROS消息类型(如Odometry、PoseStamped等)
 * @param out 要设置的ROS消息引用
 */
template<typename T>
void set_posestamp(T & out)
{
    // === 设置位置信息 ===
    out.pose.position.x = state_point.pos(0);  // X坐标
    out.pose.position.y = state_point.pos(1);  // Y坐标
    out.pose.position.z = state_point.pos(2);  // Z坐标
    
    // === 设置方向信息(四元数) ===
    out.pose.orientation.x = geoQuat.x;  // 四元数X分量
    out.pose.orientation.y = geoQuat.y;  // 四元数Y分量
    out.pose.orientation.z = geoQuat.z;  // 四元数Z分量
    out.pose.orientation.w = geoQuat.w;  // 四元数W分量(实部)
}

/**
 * @brief 发布里程计信息和坐标变换
 * 
 * 这是系统的主要输出函数之一，发布以下信息：
 * 1. 里程计消息：包含位置、姿态和协方差矩阵
 * 2. TF变换：从世界坐标系到载体坐标系的变换
 * 3. 为后续的导航和控制提供必要信息
 * 
 * @param pubOdomAftMapped 里程计消息发布器
 * @param tf_br TF2坐标变换广播器
 */
void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, 
                     std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    // === 设置里程计消息基本信息 ===
    odomAftMapped.header.frame_id = "camera_init";  // 世界坐标系
    odomAftMapped.child_frame_id = "body";          // 载体坐标系
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);  // 时间戳
    
    // === 设置位置和姿态 ===
    set_posestamp(odomAftMapped.pose);  // 使用模板函数设置姿态
    pubOdomAftMapped->publish(odomAftMapped);  // 发布里程计消息
    
    // === 设置协方差矩阵 ===
    // 从EKF中获取状态协方差矩阵
    auto P = kf.get_P();
    
    // 6x6协方差矩阵映射: [x,y,z,roll,pitch,yaw]
    // 注意：EKF状态空间中位置和姿态的顺序与ROS不同
    for (int i = 0; i < 6; i++)
    {
        int k = i < 3 ? i + 3 : i - 3;  // 坐标映射转换
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);  // 与x的协方差
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);  // 与y的协方差
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);  // 与z的协方差
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);  // 与roll的协方差
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);  // 与pitch的协方差
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);  // 与yaw的协方差
    }

    // === 发布TF变换 ===
    // 创建从世界坐标系到载体坐标系的变换
    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "camera_init";           // 父坐标系
    trans.header.stamp = odomAftMapped.header.stamp;  // 时间戳
    trans.child_frame_id = "body";                   // 子坐标系
    
    // 设置平移量
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    
    // 设置旋转量(四元数)
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    
    tf_br->sendTransform(trans);  // 发送TF变换
}

/**
 * @brief 发布载体运动轨迹路径
 * 
 * 维护和发布载体的历史运动轨迹，用于RViz中的路径显示。
 * 为了避免路径点过多导致RViz崩溃，采用了采样策略。
 * 只在每10帧中的第1帧时添加新的轨迹点。
 * 
 * @param pubPath 路径消息发布器
 */
void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    // === 设置当前姿态 ===
    set_posestamp(msg_body_pose);  // 设置位置和方向
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time);  // 设置时间戳
    msg_body_pose.header.frame_id = "camera_init";             // 设置坐标系

    // === 路径点采样策略 ===
    // 注意：如果路径点过多，RViz可能会崩溃或变得非常卡顿
    static int jjj = 0;  // 静态计数器，保持在函数调用间的状态
    jjj++;
    
    // 只在每10帧中添加一个轨迹点，减少数据量
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);  // 将当前姿态添加到轨迹中
        pubPath->publish(path);               // 发布更新后的轨迹
    }
}

/**
 * @brief EKF观测模型的共享函数 - FAST-LIO算法的核心
 * 
 * 这是整个系统最关键的函数，实现了以下核心功能：
 * 1. 点云与地图的关联：将当前帧点云与全局地图进行匹配
 * 2. 平面特征提取：从k个最近邻点中拟合平面方程
 * 3. 残差计算：计算点到平面的距离作为观测残差
 * 4. 雅可比矩阵构造：为EKF更新构造观测雅可比矩阵H
 * 
 * @param s 当前IKFOM状态，包含位置、姿态、速度等
 * @param ekfom_data EKF数据结构，包含观测矩阵H和残差向量h
 */
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();  // 记录匹配开始时间
    
    // === 初始化数据存储容器 ===
    laserCloudOri->clear();   // 清空原始激光点云(用于EKF观测)
    corr_normvect->clear();   // 清空对应的法向量
    total_residual = 0.0;     // 重置总残差

    // === 最近邻搜索和残差计算 ===
    #ifdef MP_EN  // 如果启用多线程处理
        omp_set_num_threads(MP_PROC_NUM);  // 设置线程数
        #pragma omp parallel for           // 并行处理每个点
    #endif
    
    // === 核心处理循环：逐点构建EKF观测模型 ===
    // 这个循环是 FAST-LIO2 算法的核心，实现了点到面的 ICP 匹配
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];   // 当前点在LiDAR载体系下的坐标
        PointType &point_world = feats_down_world->points[i];  // 当前点在世界系下的坐标

        // === 步骤1：高精度坐标变换 ===
        // 使用当前状态估计进行实时坐标变换
        V3D p_body(point_body.x, point_body.y, point_body.z);  // 载体系下的点坐标
        
        // 完整的坐标变换链：LiDAR系 -> IMU系 -> 世界系
        // p_global = R_wi * (R_il * p_body + t_il) + t_wi
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        // === 步骤2：最近邻搜索数据结构初始化 ===
        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);  // 存储最近邻点距离平方的容器
        auto &points_near = Nearest_Points[i];              // 获取当前点的最近邻点集合引用

        // === 步骤3：智能的最近邻搜索策略 ===
        if (ekfom_data.converge)  // 只在EKF收敛时执行搜索（提高计算效率）
        {
            // 使用 iKD-Tree 高效搜索最近的 NUM_MATCH_POINTS(5) 个点
            // 这些点将用于后续的平面拟合
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            
            // 双重质量检查：
            // 1. 数量检查：确保找到足够的最近邻点
            // 2. 距离检查：限制最远点的距离(5m内)
            point_selected_surf[i] = (points_near.size() >= NUM_MATCH_POINTS) && 
                                   (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= 5.0);
        }

        if (!point_selected_surf[i]) continue;  // 如果点不符合条件，跳过

        // === 步骤4：高精度平面拟合与健壮性检验 ===
        VF(4) pabcd;  // 平面方程系数向量: ax + by + cz + d = 0
        point_selected_surf[i] = false;  // 先重置选择标志，待验证通过后再设置true
        
        // 使用最小二乘法拟合平面，阈值0.1m用于过滤大的拟合误差
        if (esti_plane(pabcd, points_near, 0.1f))  // 平面拟合成功
        {
            // 计算当前点到拟合平面的有符号距离(点到面距离)
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + 
                       pabcd(2) * point_world.z + pabcd(3);
            
            // === 点云质量评估算法 ===
            // 基于点到面距离和点的深度信息计算质量分数
            // 远距离点和大距离点的质量分数会降低
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            // === 严格的质量门槛检验 ===
            // 只有高质量的观测才能被用于EKF更新
            if (s > 0.9)  // 质量分数阈值(非常严格)
            {
                point_selected_surf[i] = true;         // 正式标记为有效观测点
                normvec->points[i].x = pabcd(0);        // 保存平面法向量 x 分量
                normvec->points[i].y = pabcd(1);        // 保存平面法向量 y 分量  
                normvec->points[i].z = pabcd(2);        // 保存平面法向量 z 分量
                normvec->points[i].intensity = pd2;     // 将点到面距离存在intensity字段
                res_last[i] = abs(pd2);                 // 记录残差绝对值用于统计
            }
        }
    }
    
    // === 步骤5：整理有效观测数据 ===
    effct_feat_num = 0;  // 重置有效特征点计数器

    // 将所有通过验证的点整理到连续的存储空间中
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])  // 如果点被选中为有效观测
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];  // 存储原始点云
            corr_normvect->points[effct_feat_num] = normvec->points[i];          // 存储对应法向量
            total_residual += res_last[i];  // 累积残差
            effct_feat_num++;               // 增加有效点计数
        }
    }

    // === 检查有效观测数量 ===
    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;  // 标记数据无效
        std::cerr << "No Effective Points!" << std::endl;
        return;  // 早期返回，终止EKF更新
    }

    // === 计算性能统计 ===
    res_mean_last = total_residual / effct_feat_num;  // 计算平均残差
    match_time += omp_get_wtime() - match_start;      // 更新匹配时间
    double solve_start_ = omp_get_wtime();            // 记录雅可比计算开始时间
    
    // === 步骤6：构造EKF观测模型 ===
    /*** 计算观测雅可比矩阵H和观测向量h ***/
    
    // 初始化雅可比矩阵H: effct_feat_num x 12 (或23维状态)
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);
    ekfom_data.h.resize(effct_feat_num);  // 观测向量

    // 对每个有效观测点计算雅可比
    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p = laserCloudOri->points[i];  // 获取激光点
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);  // 载体系下的点
        
        // === 构造反对称矩阵(用于旋转导数) ===
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);  // [p]x 反对称矩阵
        
        // 变换到IMU系下的点
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);  // IMU系下点的反对称矩阵

        // === 获取最近表面/角点的法向量 ===
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);  // 平面法向量

        // === 计算观测雅可比矩阵H的各个分块 ===
        V3D C(s.rot.conjugate() * norm_vec);  // 法向量在载体系下的表示
        V3D A(point_crossmat * C);            // 对姿态的偏导数
        
        if (extrinsic_est_en)  // 如果启用外参估计
        {
            // 计算对外参的偏导数
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);
            
            // 填充完整的雅可比行: [位置, 姿态, 外参旋转, 外参平移]
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z,  // 位置偏导数
                                                VEC_FROM_ARRAY(A),              // 姿态偏导数  
                                                VEC_FROM_ARRAY(B),              // 外参旋转偏导数
                                                VEC_FROM_ARRAY(C);              // 外参平移偏导数
        }
        else  // 如果不估计外参
        {
            // 只填充位置和姿态的偏导数，外参部分置零
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z,  // 位置偏导数
                                                VEC_FROM_ARRAY(A),              // 姿态偏导数
                                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0;  // 外参偏导数置零
        }

        // === 设置观测值：点到最近表面/角点的距离 ===
        ekfom_data.h(i) = -norm_p.intensity;  // 负号是因为残差定义
    }
    
    solve_time += omp_get_wtime() - solve_start_;  // 更新求解时间统计
}

// ================= FAST-LIO2 主节点类 =================
/**
 * @class LaserMappingNode
 * @brief FAST-LIO2 激光平满惯性里程计与建图系统的主节点
 * 
 * 这是整个系统的核心类，集成了以下主要模块：
 * 1. 数据输入: LiDAR和IMU数据接收和预处理
 * 2. 状态估计: 基于IKFOM的扩展卡尔曼滤波
 * 3. 地图构建: 增量式KD树维护的实时地图
 * 4. 数据发布: 里程计、轨迹、点云等信息发布
 * 5. 系统服务: 地图保存等附加功能
 * 
 * 系统特点：
 * - 高频率实时处理(10-20Hz)
 * - 支持多种激光雷达类型(Livox, Velodyne, Ouster等)
 * - 自适应外参校准和偏置估计
 * - 内存高效的增量式地图维护
 */
class LaserMappingNode : public rclcpp::Node
{
public:
    /**
     * @brief 构造函数 - 初始化FAST-LIO2节点
     * 
     * 执行以下初始化任务：
     * 1. 参数声明和加载
     * 2. 数据结构初始化
     * 3. ROS2订阅者和发布者设置
     * 4. 定时器和服务配置
     * 5. EKF和地图系统初始化
     * 
     * @param options ROS2节点选项，默认使用标准配置
     */
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        // === 步骤1：声明所有ROS参数 ===
        // 发布相关参数
        this->declare_parameter<bool>("publish.path_en", true);        // 是否发布路径
        this->declare_parameter<bool>("publish.effect_map_en", false); // 是否发布有效点地图
        this->declare_parameter<bool>("publish.map_en", false);        // 是否发布累积地图
        this->declare_parameter<bool>("publish.scan_publish_en", true);     // 是否发布扫描点云
        this->declare_parameter<bool>("publish.dense_publish_en", true);    // 是否发布稠密点云
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true); // 是否发布载体系扫描
        this->declare_parameter<int>("max_iteration", 4);                   // EKF最大迭代次数
        this->declare_parameter<string>("map_file_path", "");              // 地图文件保存路径
        
        // 数据输入相关参数
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar"); // LiDAR话题名
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");   // IMU话题名
        this->declare_parameter<bool>("common.time_sync_en", false);         // 是否启用时间同步
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0); // LiDAR到IMU的时间偏移
        
        // 点云滤波参数
        this->declare_parameter<double>("filter_size_corner", 0.5);  // 角点滤波器尺寸
        this->declare_parameter<double>("filter_size_surf", 0.5);    // 面点滤波器尺寸
        this->declare_parameter<double>("filter_size_map", 0.5);     // 地图点滤波器尺寸
        this->declare_parameter<double>("cube_side_length", 200.);   // 地图立方体边长
        
        // 建图相关参数  
        this->declare_parameter<float>("mapping.det_range", 300.);    // 检测范围(米)
        this->declare_parameter<double>("mapping.fov_degree", 180.);  // 视场角度
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);      // 陀螺仪方差
        this->declare_parameter<double>("mapping.acc_cov", 0.1);      // 加速度计方差
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001); // 陀螺仪偏置方差
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001); // 加速度计偏置方差
        
        // 点云预处理参数
        this->declare_parameter<double>("preprocess.blind", 0.01);       // 盲区距离(米)
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);     // LiDAR类型
        this->declare_parameter<int>("preprocess.scan_line", 16);        // 扫描线数
        this->declare_parameter<int>("preprocess.timestamp_unit", US);   // 时间戳单位
        this->declare_parameter<int>("preprocess.scan_rate", 10);        // 扫描频率
        this->declare_parameter<int>("point_filter_num", 2);             // 点滤波数量
        this->declare_parameter<bool>("feature_extract_enable", false);  // 是否启用特征提取
        
        // 系统日志和调试参数
        this->declare_parameter<bool>("runtime_pos_log_enable", false);  // 是否启用运行时位置日志
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true); // 是否启用外参估计
        
        // PCD文件保存参数
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);    // 是否保存PCD文件
        this->declare_parameter<int>("pcd_save.interval", -1);           // PCD保存间隔
        
        // 外参矩阵参数
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>()); // 外参平移矩阵
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>()); // 外参旋转矩阵

        // === 步骤2：获取参数值并赋给全局变量 ===
        // 发布相关参数获取
        this->get_parameter_or<bool>("publish.path_en", path_en, true);                    // 路径发布开关
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);       // 有效地图发布开关
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);                // 地图发布开关
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);       // 扫描发布开关
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);     // 稠密点云发布开关
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true); // 载体系扫描发布开关
        
        // 算法控制参数
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);              // EKF最大迭代次数
        this->get_parameter_or<string>("map_file_path", map_file_path, "");              // 地图文件路径
        
        // 传感器话题参数
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");   // LiDAR话题
        this->get_parameter_or<string>("common.imu_topic", imu_topic, "/livox/imu");     // IMU话题
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);        // 时间同步开关
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0); // 时间偏移
        
        // 点云滤波参数获取
        this->get_parameter_or<double>("filter_size_corner", filter_size_corner_min, 0.5); // 角点滤波尺寸
        this->get_parameter_or<double>("filter_size_surf", filter_size_surf_min, 0.5);     // 面点滤波尺寸
        this->get_parameter_or<double>("filter_size_map", filter_size_map_min, 0.5);       // 地图滤波尺寸
        this->get_parameter_or<double>("cube_side_length", cube_len, 200.f);               // 地图立方体边长
        
        // 建图算法参数
        this->get_parameter_or<float>("mapping.det_range", DET_RANGE, 300.f);             // 检测范围
        this->get_parameter_or<double>("mapping.fov_degree", fov_deg, 180.f);             // 视场角度
        this->get_parameter_or<double>("mapping.gyr_cov", gyr_cov, 0.1);                 // 陀螺仪方差
        this->get_parameter_or<double>("mapping.acc_cov", acc_cov, 0.1);                 // 加速度计方差
        this->get_parameter_or<double>("mapping.b_gyr_cov", b_gyr_cov, 0.0001);          // 陀螺仪偏置方差
        this->get_parameter_or<double>("mapping.b_acc_cov", b_acc_cov, 0.0001);          // 加速度计偏置方差
        
        // 点云预处理参数获取
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);          // 盲区距离
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);   // LiDAR类型
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);         // 扫描线数
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);  // 时间戳单位
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);       // 扫描频率
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);     // 点滤波数量
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false); // 特征提取开关
        
        // 系统控制参数
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);      // 运行时位置日志
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true); // 外参估计开关
        
        // PCD保存参数
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);        // PCD保存开关
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);         // PCD保存间隔
        
        // 外参矩阵参数
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>()); // 外参平移矩阵
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>()); // 外参旋转矩阵

        // 输出初始化信息
        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);

        // === 步骤3：初始化数据结构 ===
        // 初始化路径消息头
        path.header.stamp = this->get_clock()->now();
        path.header.frame_id = "camera_init";  // 基础坐标系

        // 计算视场角相关参数
        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);  // 限制视场角在179.9度以内
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);             // 计算半视场角的余弦值

        // 初始化特征点云数组
        _featsArray.reset(new PointCloudXYZI());

        // 初始化点选择和残差数组
        memset(point_selected_surf, true, sizeof(point_selected_surf));   // 所有点初始都被选中
        memset(res_last, -1000.0f, sizeof(res_last));                    // 初始化残差为很小的负值
        
        // 配置点云滤波器
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min); // 表面点滤波器
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);      // 地图点滤波器

        // === 步骤4：配置IMU处理模块 ===
        // 设置外参矩阵（LiDAR到IMU的变换）
        Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);  // 平移向量
        Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);  // 旋转矩阵
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        
        // 设置IMU噪声协方差
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));           // 陀螺仪噪声方差
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));           // 加速度计噪声方差
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov)); // 陀螺仪偏置方差
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov)); // 加速度计偏置方差

        // === 步骤5：初始化EKF系统 ===
        double epsi[23] = {0.001};  // EKF收敛阈值数组
        fill(epsi, epsi + 23, 0.001);  // 填充所有元素为0.001
        
        // 初始化动态共享EKF，注册状态转移、雅可比和观测模型函数
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        // === 步骤6：设置调试日志文件 ===
        /*** 调试记录文件初始化 ***/
        string pos_log_dir = root_dir + "/Log/pos_log.txt";  // 位置日志文件路径
        fp = fopen(pos_log_dir.c_str(), "w");               // 打开位置日志文件

        // 打开调试输出文件
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), ios::out);  // 预处理矩阵调试文件
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);  // 输出矩阵调试文件
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), ios::out);      // 通用调试文件
        
        // 检查文件是否成功打开
        if (fout_pre && fout_out)
            cout << "~~~~" << ROOT_DIR << " file opened" << endl;
        else
            cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

        // === 步骤7：初始化ROS2订阅者和发布者 ===
        /*** ROS订阅者初始化 ***/
        // 根据LiDAR类型选择不同的订阅者
        if (p_pre->lidar_type == AVIA)  // 如果是Livox Avia类型
        {
            // 订阅Livox自定义消息格式
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                lid_topic, 20, livox_pcl_cbk);
        }
        else  // 其他类型LiDAR（Velodyne, Ouster等）
        {
            // 订阅标准PointCloud2消息格式
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
        }
        
        // 订阅IMU数据
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 10, imu_cbk);
        
        // 初始化各类发布者
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);      // 配准后点云
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20); // 载体系点云
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);     // 有效点云
        pubLaserCloudMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 20);             // 地图点云
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);                    // 里程计信息
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);                                     // 路径信息
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);                               // TF变换广播器

        // === 步骤8：初始化定时器和服务 ===
        // 主处理循环定时器（100Hz）
        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, 
                                     std::bind(&LaserMappingNode::timer_callback, this));

        // 地图发布定时器（1Hz）
        auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
        map_pub_timer_ = rclcpp::create_timer(this, this->get_clock(), map_period_ms, 
                                             std::bind(&LaserMappingNode::map_publish_callback, this));

        // 地图保存服务
        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "map_save",
            std::bind(&LaserMappingNode::map_save_callback, this,
                    std::placeholders::_1, std::placeholders::_2));
        
        RCLCPP_INFO(this->get_logger(), "Node init finished.");  // 输出初始化完成信息
    }

    /**
     * @brief 析构函数 - 清理资源和关闭文件
     * 
     * 在节点销毁时自动调用，负责：
     * 1. 关闭所有打开的日志文件
     * 2. 释放内存资源
     * 3. 确保数据安全保存
     */
    ~LaserMappingNode()
    {
        fout_out.close();  // 关闭输出调试文件
        fout_pre.close();  // 关闭预处理调试文件
        fclose(fp);        // 关闭位置日志文件
    }

private:
    /**
     * @brief 主处理循环定时器回调函数
     * 
     * 这是FAST-LIO2系统的核心处理循环，以100Hz频率运行。
     * 主要执行以下步骤：
     * 1. 数据同步: 同步LiDAR和IMU数据
     * 2. IMU预积分: 估计初始状态
     * 3. 点云处理: 去畔化和变换
     * 4. 特征匹配: 建立点到面对应关系
     * 5. 状态更新: EKF估计位姿
     * 6. 地图更新: 增量式更新局部地图
     * 7. 结果发布: 发布里程计和点云信息
     */
    void timer_callback()
    {
        // === 步骤1：数据同步检查 ===
        if(sync_packages(Measures))  // 尝试同步LiDAR和IMU数据包
        {
            // === 初始化检查 ===
            if (flg_first_scan)  // 如果是第一帧扫描
            {
                first_lidar_time = Measures.lidar_beg_time;  // 记录第一帧LiDAR时间
                p_imu->first_lidar_time = first_lidar_time;   // 设置IMU处理器的初始时间
                flg_first_scan = false;  // 标记初始化完成
                return;  // 退出，等待下一帧
            }

            // === 步骤2：初始化时间计数器 ===
            double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

            // 重置各个时间统计变量
            match_time = 0;              // 匹配时间
            kdtree_search_time = 0.0;    // KD树搜索时间
            solve_time = 0;              // 求解时间
            solve_const_H_time = 0;      // 雅可比矩阵构造时间
            svd_time = 0;                // SVD分解时间
            t0 = omp_get_wtime();        // 记录开始时间

            // === 步骤3：IMU数据处理和状态预测 ===
            p_imu->Process(Measures, kf, feats_undistort);  // IMU预积分和点云去畔化
            state_point = kf.get_x();   // 获取当前状态估计
            
            // 计算LiDAR在世界坐标系下的位置
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            // === 检查点云数据有效性 ===
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;  // 无效数据，跳过当前帧
            }

            // === 检查EKF是否初始化完成 ===
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;  // 需要等待一段时间才能初始化
            // === 步骤4：地图管理和视场分割 ===

            // 重置各个时间统计变量
            match_time = 0;              // 匹配时间
            kdtree_search_time = 0.0;    // KD树搜索时间
            solve_time = 0;              // 求解时间
            solve_const_H_time = 0;      // 雅可比矩阵构造时间
            svd_time = 0;                // SVD分解时间
            t0 = omp_get_wtime();        // 记录开始时间

            // === 步骤3：IMU数据处理和状态预测 ===
            p_imu->Process(Measures, kf, feats_undistort);  // IMU预积分和点云去畔化
            state_point = kf.get_x();   // 获取当前状态估计
            
            // 计算LiDAR在世界坐标系下的位置
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            // === 检查点云数据有效性 ===
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;  // 无效数据，跳过当前帧
            }

            // === 检查EKF是否初始化完成 ===
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;  // 需要等待一段时间才能初始化
            /*** 在LiDAR视场内分割地图 ***/
            lasermap_fov_segment();  // 根据当前位置和视场角度分割地图区域

            // === 步骤5：点云下采样 ===
            /*** 对扫描中的特征点进行下采样 ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);  // 设置输入点云
            downSizeFilterSurf.filter(*feats_down_body);        // 执行体素滤波
            t1 = omp_get_wtime();                               // 记录下采样结束时间
            feats_down_size = feats_down_body->points.size();   // 获取下采样后的点数
            
            // === 步骤6：初始化地图KD树 ===
            /*** 初始化地图KD树 ***/
            if(ikdtree.Root_Node == nullptr)  // 如果KD树还未初始化
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)  // 确保有足够的点数进行初始化
                {
                    // 设置KD树的下采样参数
                    ikdtree.set_downsample_param(filter_size_map_min);
                    
                    // 将载体系点云转换到世界坐标系
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    
                    // 构建KD树
                    ikdtree.Build(feats_down_world->points);
                }
                return;  // 初始化后返回，等待下一帧数据
            }
            
            // 获取地图统计信息
            int featsFromMapNum = ikdtree.validnum();  // 地图中有效点数
            kdtree_size_st = ikdtree.size();           // KD树大小
            
            // 调试信息输出（已注释）:
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            // === 步骤7：ICP和迭代卡尔曼滤波更新 ===
            /*** ICP和迭代卡尔曼滤波更新 ***/
            if (feats_down_size < 5)  // 检查点数是否足够
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;  // 点数不足，跳过当前帧
            }
            
            // 初始化存储空间
            normvec->resize(feats_down_size);       // 法向量数组，存储对应点的平面法向量
            feats_down_world->resize(feats_down_size); // 世界坐标系下的点云

            // === 调试信息输出 ===
            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);  // 将外参旋转矩阵转为欧拉角
            
            // 输出当前状态信息到调试文件
            fout_pre << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " 
                     << euler_cur.transpose() << " " << state_point.pos.transpose() << " " 
                     << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " 
                     << state_point.vel.transpose() << " " << state_point.bg.transpose() << " " 
                     << state_point.ba.transpose() << " " << state_point.grav << endl;

            // === 地图点可视化（可选） ===
            if(0) // 如果需要查看地图点，将此处改为 "if(1)"
            {
                PointVector().swap(ikdtree.PCL_Storage);  // 清空存储
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD); // 展开KD树
                featsFromMap->clear();                    // 清空地图点云
                featsFromMap->points = ikdtree.PCL_Storage; // 复制所有地图点
            }

            // === 初始化搜索结构 ===
            pointSearchInd_surf.resize(feats_down_size);  // 表面点搜索索引
            Nearest_Points.resize(feats_down_size);       // 最近邻点集合
            int rematch_num = 0;                          // 重新匹配计数器
            bool nearest_search_en = true;  // 启用最近邻搜索

            t2 = omp_get_wtime();  // 记录预处理结束时间
            
            // === 步骤8：迭代状态估计（FAST-LIO2核心算法） ===
            /*** 迭代状态估计 ***/
            double t_update_start = omp_get_wtime();  // 记录EKF更新开始时间
            double solve_H_time = 0;                  // 雅可比矩阵求解时间
            
            /**
             * 执行迭代动态共享EKF更新 - FAST-LIO2的核心算法
             * 这个函数内部会：
             * 1. 调用h_share_model函数计算观测模型和雅可比矩阵
             * 2. 迭代求解最优状态估计
             * 3. 更新协方差矩阵
             * 4. 实现点到面的ICP匹配和状态优化
             */
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            
            // === 获取更新后的状态估计 ===
            state_point = kf.get_x();  // 从 EKF 获取最新的 23 维状态向量
                                       // 包括: [p, v, q, ba, bg, g, p_L, q_L] 
                                       // p: 位置, v: 速度, q: 姿态, ba/bg: 偏置, g: 重力, p_L/q_L: 外参
            
            euler_cur = SO3ToEuler(state_point.rot);  // 将姿态四元数转为欧拉角（用于输出显示）
            
            // 计算 LiDAR 在世界坐标系下的位置
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            
            // === 准备发布数据：将 SO3 旋转转为四元数格式 ===
            geoQuat.x = state_point.rot.coeffs()[0];  // 四元数 x 分量
            geoQuat.y = state_point.rot.coeffs()[1];  // 四元数 y 分量
            geoQuat.z = state_point.rot.coeffs()[2];  // 四元数 z 分量
            geoQuat.w = state_point.rot.coeffs()[3];  // 四元数 w 分量

            double t_update_end = omp_get_wtime();  // 记录EKF更新结束时间

            // === 步骤9：发布里程计信息 ===
            /******* 发布里程计 *******/
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

            // === 步骤10：更新地图 ===
            /*** 将特征点添加到地图KD树 ***/
            t3 = omp_get_wtime();  // 记录地图更新开始时间
            map_incremental();     // 执行增量式地图更新
            t5 = omp_get_wtime();  // 记录地图更新结束时间
            
            // === 步骤11：发布各类信息 ===
            /******* 发布点云和路径信息 *******/
            if (path_en)                         publish_path(pubPath_);                 // 发布轨迹路径
            if (scan_pub_en)                     publish_frame_world(pubLaserCloudFull_); // 发布世界系点云
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_); // 发布载体系点云
            if (effect_pub_en)                   publish_effect_world(pubLaserCloudEffect_); // 发布有效点云
            // 地图发布由单独的定时器处理：
            // if (map_pub_en) publish_map(pubLaserCloudMap_);

            // === 步骤12：调试和性能统计 ===
            /*** 调试变量和性能统计 ***/
            if (runtime_pos_log)  // 如果启用运行时日志
            {
                frame_num++;  // 增加帧计数器
                kdtree_size_end = ikdtree.size();  // 记录KD树最终大小
                
                // 计算各个阶段的平均时间消耗（滑动平均）
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;  // 总耗时
                aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;  // ICP时间
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;  // 匹配时间
                aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num + (kdtree_incremental_time) / frame_num;  // 地图增量时间
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;  // 求解时间
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;  // 构造H矩阵时间
                
                // 记录详细的性能数据
                T1[time_log_counter] = Measures.lidar_beg_time;        // 时间戳
                s_plot[time_log_counter] = t5 - t0;                   // 总处理时间
                s_plot2[time_log_counter] = feats_undistort->points.size(); // 原始点数
                s_plot3[time_log_counter] = kdtree_incremental_time;   // KD树增量时间
                s_plot4[time_log_counter] = kdtree_search_time;        // KD树搜索时间
                s_plot5[time_log_counter] = kdtree_delete_counter;     // KD树删除计数
                s_plot6[time_log_counter] = kdtree_delete_time;        // KD树删除时间
                s_plot7[time_log_counter] = kdtree_size_st;            // KD树初始大小
                s_plot8[time_log_counter] = kdtree_size_end;           // KD树最终大小
                s_plot9[time_log_counter] = aver_time_consu;           // 平均总耗时
                s_plot10[time_log_counter] = add_point_size;           // 新增点数
                time_log_counter++;  // 增加日志计数器
                
                // 打印性能统计信息
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",
                       t1-t0, aver_time_match, aver_time_solve, t3-t1, t5-t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
                
                // 输出详细状态信息到文件
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " 
                         << euler_cur.transpose() << " " << state_point.pos.transpose() << " " 
                         << ext_euler.transpose() << " " << state_point.offset_T_L_I.transpose() << " " 
                         << state_point.vel.transpose() << " " << state_point.bg.transpose() << " " 
                         << state_point.ba.transpose() << " " << state_point.grav << " " 
                         << feats_undistort->points.size() << endl;
                
                // 将LIO状态输出到日志文件
                dump_lio_state_to_log(fp);
            }
        }
    }

    /**
     * @brief 地图发布定时器回调函数
     * 
     * 每秒调用一次，负责发布完整的地图点云。
     * 由于地图数据量很大，采用低频率发布以避免占用过多带宽。
     */
    void map_publish_callback()
    {
        if (map_pub_en) publish_map(pubLaserCloudMap_);  // 如果启用地图发布，则发布地图
    }

    /**
     * @brief 地图保存服务回调函数
     * 
     * 处理来自ROS2客户端的地图保存请求。
     * 将当前的地图点云保存为PCD文件格式。
     * 
     * @param req 服务请求（空请求）
     * @param res 服务响应，包含成功标志和消息
     */
    void map_save_callback(std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to %s...", map_file_path.c_str());
        
        if (pcd_save_en)  // 如果启用了PCD保存功能
        {
            save_to_pcd();      // 执行保存操作
            res->success = true;
            res->message = "Map saved.";  // 返回成功信息
        }
        else  // 如果禁用了PCD保存功能
        {
            res->success = false;
            res->message = "Map save disabled.";  // 返回失败信息
        }
    }

private:
    // === ROS2发布者和订阅者 ===
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;      // 配准后世界系点云发布者
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_; // 配准后载体系点云发布者
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;    // 有效点云发布者
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;       // 地图点云发布者
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;             // 里程计信息发布者
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;                          // 轨迹路径发布者
    
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;                     // IMU数据订阅者
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;          // 标准点云订阅者
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;   // Livox点云订阅者

    // === ROS2服务和定时器 ===
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;  // TF变换广播器
    rclcpp::TimerBase::SharedPtr timer_;                            // 主处理循环定时器(100Hz)
    rclcpp::TimerBase::SharedPtr map_pub_timer_;                    // 地图发布定时器(1Hz)
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_; // 地图保存服务

    // === 系统控制标志 ===
    bool effect_pub_en = false;    // 有效点云发布开关
    bool map_pub_en = false;       // 地图发布开关
    
    // === 算法状态变量 ===
    int effect_feat_num = 0;       // 有效特征点数量
    int frame_num = 0;             // 处理的帧数统计
    
    // === 性能统计变量 ===
    double deltaT, deltaR;                     // 时间和旋转增量
    double aver_time_consu = 0;                // 平均总耗时
    double aver_time_icp = 0;                  // 平均ICP时间
    double aver_time_match = 0;                // 平均匹配时间
    double aver_time_incre = 0;                // 平均地图增量时间
    double aver_time_solve = 0;                // 平均求解时间
    double aver_time_const_H_time = 0;         // 平均构造H矩阵时间
    
    // === EKF状态标志 ===
    bool flg_EKF_converged;        // EKF收敛标志
    bool EKF_stop_flg = 0;         // EKF停止标志
    double epsi[23] = {0.001};     // EKF收敛阈值数组

    // === 调试和日志文件 ===
    FILE *fp;                      // 位置日志文件指针
    ofstream fout_pre;             // 预处理调试文件流
    ofstream fout_out;             // 输出调试文件流
    ofstream fout_dbg;             // 通用调试文件流
};

// ================= 主程序入口 =================
/**
 * @brief FAST-LIO2系统主函数
 * 
 * 这是整个FAST-LIO2系统的程序入口点，主要负责：
 * 1. 初始化ROS2系统和节点
 * 2. 设置信号处理器（用于优雅关闭）
 * 3. 启动多线程执行器进行事件循环
 * 4. 处理程序退出和资源清理
 * 
 * 命令行参数：
 * @param argc 命令行参数个数
 * @param argv 命令行参数数组
 * @return int 程序退出码（0表示正常退出）
 */
int main(int argc, char** argv)
{
    // === 步骤1：初始化ROS2系统 ===
    rclcpp::init(argc, argv);  // 初始化ROS2通信系统和命令行参数解析

    // === 步骤2：设置信号处理器 ===
    signal(SIGINT, SigHandle);  // 注册CTRL+C信号处理器，实现优雅关闭

    // === 步骤3：创建并运行节点 ===
    rclcpp::spin(std::make_shared<LaserMappingNode>());  // 创建LaserMappingNode实例并开始事件循环

    // === 步骤4：清理ROS2资源 ===
    if (rclcpp::ok())  // 检查ROS2系统是否正常
        rclcpp::shutdown();  // 关闭ROS2系统
    
    // === 步骤5：保存地图数据 ===
    /**************** 保存地图 ****************/
    /* 注意事项：
    /* 1. 确保有足够的内存空间
    /* 2. PCD保存会显著影响实时性能 **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)  // 如果有待保存数据且启用了PCD保存
    {
        string file_name = string("scans.pcd");  // 设置文件名
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);  // 构造完整路径
        pcl::PCDWriter pcd_writer;  // 创建PCD写入器
        cout << "current scan saved to /PCD/" << file_name << endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);  // 以二进制格式保存点云
    }

    // === 步骤6：保存性能统计日志 ===
    if (runtime_pos_log)  // 如果启用了运行时位置日志
    {
        // 初始化日志数据存储容器
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;
        
        // 打开CSV日志文件
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(), "w");
        
        // 写入CSV文件头
        fprintf(fp2, "time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        
        // 逐行写入性能数据
        for (int i = 0; i < time_log_counter; i++) {
            // 将每一帧的性能数据写入CSV文件
            fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",
                    T1[i],               // 时间戳
                    s_plot[i],           // 总处理时间
                    int(s_plot2[i]),     // 扫描点数量
                    s_plot3[i],          // 增量时间
                    s_plot4[i],          // 搜索时间
                    int(s_plot5[i]),     // 删除点数
                    s_plot6[i],          // 删除时间
                    int(s_plot7[i]),     // KD树初始大小
                    int(s_plot8[i]),     // KD树最终大小
                    int(s_plot10[i]),    // 新增点数
                    s_plot11[i]);        // 预处理时间
            
            // 为图表分析准备数据（备用）
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);                    // 平均总耗时
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);      // 增量+删除时间
            s_vec3.push_back(s_plot4[i]);                   // 搜索时间
            s_vec5.push_back(s_plot[i]);                    // 总处理时间
        }
        
        fclose(fp2);  // 关闭CSV文件
        cout << "Performance log saved to: " << log_dir << endl;
    }

    // === 程序正常退出 ===
    cout << "FAST-LIO2 system shutdown completed." << endl;
    return 0;  // 返回成功退出码
}
