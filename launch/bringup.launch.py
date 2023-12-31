# MIT License
#
# Copyright (c) 2023 Stanford Autonomous Systems Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    TURTLEBOT3_MODEL = os.environ["TURTLEBOT3_MODEL"]

    usb_port = LaunchConfiguration("usb_port", default="/dev/ttyACM0")
    camera = LaunchConfiguration("camera", default="false")

    return LaunchDescription([
        DeclareLaunchArgument(
            "usb_port",
            default_value=usb_port,
            description="Connected USB port with OpenCR"
        ),
        DeclareLaunchArgument(
            "camera",
            default_value=camera,
            choices=["true", "false"],
            description="set to true to bring up USB camera and detector"
        ),

        ########################################################
        # -------------------- TurtleBot3 -------------------- #
        ########################################################

        IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare("turtlebot3_bringup"),
                "launch",
                "turtlebot3_state_publisher.launch.py",
            ]),
        ),
        Node(
            package="turtlebot3_node",
            executable="turtlebot3_ros",
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare("turtlebot3_bringup"),
                    "param",
                    TURTLEBOT3_MODEL + ".yaml",
                ]),
            ],
            arguments=["-i", usb_port],
        ),

        ########################################################
        # --------------------- Velodyne --------------------- #
        ########################################################

        IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare("asl_tb3_driver"),
                "launch",
                "velodyne.launch.py",
            ]),
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=["0.0175", "0", "0.215", "0", "0", "0", "1", "base_footprint", "velodyne"],
        ),

        ########################################################
        # ----------------------- SLAM ----------------------- #
        ########################################################

        Node(
            parameters=[
                PathJoinSubstitution([FindPackageShare("asl_tb3_driver"), "configs", "slam_params.yaml"]),
                {"use_sim_time": False},
            ],
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
        ),

        ########################################################
        # -------------------- USB Camera -------------------- #
        ########################################################

        IncludeLaunchDescription(
            PathJoinSubstitution([
                FindPackageShare("asl_tb3_driver"),
                "launch",
                "camera.launch.py",
            ]),
            condition=IfCondition(camera),
        )
    ])
