#!/usr/bin/env python

import rospy
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped

def get_plan(start_x, start_y, goal_x, goal_y):
    rospy.wait_for_service('/move_base/make_plan')
    try:
        make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)

        start = PoseStamped()
        start.header.frame_id = "map"
        start.pose.position.x = start_x
        start.pose.position.y = start_y

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y

        tolerance = 0.1  # 10cm tolerance

        plan = make_plan(start, goal, tolerance)
        return plan.plan
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def time_parameterize_path(path, velocity=0.5):
    time_parameterized_path = []
    time = 0.0

    for i in range(len(path.poses) - 1):
        current_pose = path.poses[i]
        next_pose = path.poses[i+1]
        
        dx = next_pose.pose.position.x - current_pose.pose.position.x
        dy = next_pose.pose.position.y - current_pose.pose.position.y
        distance = (dx**2 + dy**2)**0.5
        
        time_parameterized_path.append((time, current_pose))
        
        time += distance / velocity

    # Add the final pose
    time_parameterized_path.append((time, path.poses[-1]))

    return time_parameterized_path

if __name__ == '__main__':
    rospy.init_node('path_retriever')

    # Define start and goal positions
    start_x, start_y = 0.0, 0.0  # Assuming the robot starts at (0,0)
    goal_x, goal_y = 3.35, 1.46    # Goal position

    # Get the plan
    path = get_plan(start_x, start_y, goal_x, goal_y)

    if path and path.poses:
        # Time parameterize the path
        time_parameterized_path = time_parameterize_path(path)

        # Print the time-parameterized path
        for time, pose in time_parameterized_path:
            print(f"Time: {time:.2f}, Position: ({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})")
    else:
        print("Failed to retrieve path or path is empty")