import os.path

# 从 ament_index_python 包中导入函数，用于获取 ROS 2 包的共享目录路径
from ament_index_python.packages import get_package_share_directory

# 导入 ROS 2 的 Launch API，用于定义启动文件
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument  # 声明启动参数
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution  # 启动参数的替换和路径拼接
from launch.conditions import IfCondition  # 条件判断，用于控制节点是否启动

# 导入 ROS 2 的 Launch API，用于启动 ROS 2 节点
from launch_ros.actions import Node


def generate_launch_description():
    # 获取 'fast_lio' 包的共享目录路径
    package_path = get_package_share_directory('fast_lio')
    # 默认配置文件路径
    default_config_path = os.path.join(package_path, 'config')
    # 默认 RViz 配置文件路径
    default_rviz_config_path = os.path.join(
        package_path, 'rviz', 'fastlio.rviz')

    # 定义启动参数，允许用户通过命令行或其他方式传递参数
    use_sim_time = LaunchConfiguration('use_sim_time')  # 是否使用仿真时间
    config_path = LaunchConfiguration('config_path')  # 配置文件路径
    config_file = LaunchConfiguration('config_file')  # 配置文件名
    rviz_use = LaunchConfiguration('rviz')  # 是否启用 RViz
    rviz_cfg = LaunchConfiguration('rviz_cfg')  # RViz 配置文件路径

    # 声明启动参数及其默认值
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true'  # 如果为 true，则使用仿真时间
    )
    declare_config_path_cmd = DeclareLaunchArgument(
        'config_path', default_value=default_config_path,
        description='Yaml config file path'  # 配置文件路径
    )
    decalre_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='mid360.yaml',
        description='Config file'  # 配置文件名
    )
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Use RViz to monitor results'  # 是否使用 RViz
    )
    declare_rviz_config_path_cmd = DeclareLaunchArgument(
        'rviz_cfg', default_value=default_rviz_config_path,
        description='RViz config file path'  # RViz 配置文件路径
    )

    # 定义 fast_lio 节点
    fast_lio_node = Node(
        package='fast_lio',  # 节点所属的 ROS 2 包
        executable='fastlio_mapping',  # 可执行文件名称
        parameters=[PathJoinSubstitution([config_path, config_file]),  # 配置文件路径和文件名
                    {'use_sim_time': use_sim_time}],  # 是否使用仿真时间
        output='screen'  # 输出到终端
    )
    # 定义 RViz 节点
    rviz_node = Node(
        package='rviz2',  # 节点所属的 ROS 2 包
        executable='rviz2',  # 可执行文件名称
        arguments=['-d', rviz_cfg],  # 启动参数，指定 RViz 配置文件
        condition=IfCondition(rviz_use)  # 条件判断，只有在 rviz_use 为 true 时才启动
    )

    # 创建 LaunchDescription 对象，用于存储所有的启动项
    ld = LaunchDescription()
    # 添加声明的启动参数
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(decalre_config_file_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)

    # 添加定义的节点
    ld.add_action(fast_lio_node)
    ld.add_action(rviz_node)
    # 启动了 fast_lio_mapping 节点和 RViz 节点
    # 返回 LaunchDescription 对象
    return ld
