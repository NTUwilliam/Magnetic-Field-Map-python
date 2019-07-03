#! /usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import pickle
from skimage.measure import LineModelND, ransac
from geometry_msgs.msg import Twist

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import linear_model
from sklearn import linear_model, datasets
import math
import threading


rotate_alpha=1
velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

def callback(msg):
    # print len(msg.ranges)
    # print msg.ranges[0]
    # print msg.ranges[360]
    # print msg.ranges[719]
    scan_points = laser_to_pointcloud_transform(msg)
    # print(scan_points)
    print(len(scan_points))
    # input("Press Enter to continue...")

    # plt.ion()
    # plt.cla()
    
    # plt.plot(scan_points[:,0],scan_points[:,1], '*')
    
    # plt.ylabel('some numbers')
    # plt.show()
    # plt.ioff()
    # # plt.draw()


    # print(scan_points)
    print(msg.header)

    # np.save("points.txt",scan_points)

    
    # with open("points.txt", 'w') as f:
    #     pickle.dump(scan_points, f)

    # f= open("points.txt","w+")
    # f.write(scan_points)

    #! /usr/bin/env python

    

    MIN_SAMPLES = 3

    # x = np.linspace(0, 2, 100)

    # xs, ys = [], []

    # # generate points for thee lines described by a and b,
    # # we also add some noise:
    # for a, b in [(1.0, 2), (0.5, 1), (1.2, -1)]:
    #     xs.extend(x)
    #     ys.extend(a * x + b + .1 * np.random.randn(len(x)))


    # ,
    # GIVE the input as xs, ys
    xs = np.array(scan_points[:,0])
    ys = np.array(scan_points[:,1])

    # print( np.isnan(xs) )
    # print( np.isnan(ys) )

    # print( np.isfinite(ys) )
   
    # first get the indices where the values are finite
    ys_finite_idx = np.isfinite(ys)

    # print("ys_finite_idx: ",ys_finite_idx)

    # second get the values
    xs_finite = xs[ys_finite_idx]
    ys_finite = ys[ys_finite_idx]


    # plt.plot(xs_finite, ys_finite, "r.")
    # plt.show()
    # plt.draw()

    colors = "rgbky"
    idx = 0

    my_coef =[]

    # while len(xs_finite) > 100:
    if len(xs_finite) > 100:
        # MIN_SAMPLES

        # build design matrix for linear regressor
        X = np.ones((len(xs_finite), 2))
        X[:, 1] = xs_finite

        ransac = linear_model.RANSACRegressor(
            residual_threshold=.05, min_samples=MIN_SAMPLES
        )

        res = ransac.fit(X, ys_finite)
        # coef = ransac.get_params()
        print("intercept: ",ransac.estimator_.intercept_)
        print("coef: ",ransac.estimator_.coef_)
        print("coef(1): ",ransac.estimator_.coef_[1])

        my_coef.append(ransac.estimator_.coef_)

        # angle
        angle = math.atan( ransac.estimator_.coef_[1] )
        print("angle", angle)
        


        # self, deep=True

        # vector of boolean values, describes which points belong
        # to the fitted line:
        inlier_mask = ransac.inlier_mask_

        # plot point cloud:
        xinlier = xs_finite[inlier_mask]
        yinlier = ys_finite[inlier_mask]


        


        # n_samples = 1000
        # n_outliers = 50


        # X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
        #                                     n_informative=1, noise=10,
        #                                     coef=True, random_state=0)

        # circle through colors:
        color = colors[idx % len(colors)]
        idx += 1
        # plt.plot(xinlier, yinlier, color + "*")

        # only keep the outliers:
        xs_finite = xs_finite[~inlier_mask]
        ys_finite = ys_finite[~inlier_mask]


    # plt.show()
    # # plt.draw()
    # plt.pause(0.001)

    if (len(my_coef)>0):
        print(my_coef[0])
        
        print("my_coef[0][1]:", my_coef[0][1])
        k_wall = my_coef[0][1]
        rotate_alpha = math.atan(k_wall)
        print("rotate_alpha = ",rotate_alpha)
        print("rotate_alpha (degree) = ",rotate_alpha*180/math.pi)
        # publish_rotate(rotate_alpha*180/math.pi)

        # if k_wall > 0:
        #     print("k_wall > 0")
        #     rotate_alpha = math.atan(k_wall)
        # else:
        #     print("k_wall < 0")
        #     rotate_alpha = math.pi - math.atan(k_wall)
        # print("rotate_alpha = ",rotate_alpha)
    

        # # angle
        # angle = math.atan( ransac.estimator_.coef_[1] )
        # print("angle", angle)

        
        command=Twist()
        # r = rospy.Rate(20)
        
        print("In publisher!")
        print("rotate_alpha",rotate_alpha)
        command.angular.z= (rotate_alpha - 0) / 7.5
        velocity_publisher.publish(command)
        

        # while not rospy.is_shutdown():
        #     print("In publisher!")
        #     print("rotate_alpha",rotate_alpha)
        #     command.angular.z= (rotate_alpha - 0) / 20.0
        #     velocity_publisher.publish(command)
        #     r.sleep()




    # my_RANSAC_transform_for_lines(scan_points)
    # rotate(b)


##################### Part I: Laser to pointcloud transform ####################
def laser_to_pointcloud_transform(msg):
    # i, x, y, current_angle

    # def __init__(self,values):
    # self.i = 0Gmapping
    # self.x = float
    # self.y = float
    # self.current_angle = float

    array_p = np.zeros((len(msg.ranges), 2))
    # for i in range (0,720):
    for i in range (len(msg.ranges)):

        current_angle = (msg.angle_min) + (msg.angle_increment)*i
        x = msg.ranges[i] * np.cos(current_angle)
        y = msg.ranges[i] * np.sin(current_angle)
        
        array_p[i][0] = x
        array_p[i][1] = y
        
    return array_p

def publish_rotate(angle):
    vel_msg = Twist()

    #We wont use linear components
    vel_msg.linear.x=0
    vel_msg.linear.y=0
    vel_msg.linear.z=0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    # Checking if our movement is CW or CCW
    # if clockwise:
    #     vel_msg.angular.z = -abs(angular_speed)
    # else:
    #     vel_msg.angular.z = abs(angular_speed)

    #Converting from angles to radians
    speed = 5
    angular_speed = speed*2*math.pi/360
    relative_angle = angle*2*math.pi/360
    vel_msg.angular.z = abs(angular_speed)

    # Setting the current time for distance calculus
    t0 = rospy.Time.now().to_sec()
    print("t0=",t0)
    current_angle = 0
    print("current_angle = ",current_angle)
    print("relative_angle = ",relative_angle)

    # time to rotate
    t_rotate = abs(relative_angle / angular_speed)
    print("t_rotate=",t_rotate)

    # velocity_publisher.publish(vel_msg)
    # print("published!!!!!!!!!!!!!!!!!!!!!!!")
    # Checking if our movement is CW or CCW
    if relative_angle < 0:
        vel_msg.angular.z = -abs(angular_speed)
    else:
        vel_msg.angular.z = abs(angular_speed)

    time_i = 0
    while(time_i < t_rotate):
        print("time_i=",time_i)
        velocity_publisher.publish(vel_msg)
        
        time_i = time_i+1

    # rospy.signal_shutdown("once")

    # while (current_angle < abs(relative_angle)):
    #     velocity_publisher.publish(vel_msg)
    #     # print("published!!!!!!!!!!!!!!!!!!!!!!!")
    #     t1 = rospy.Time.now().to_sec()
    #     current_angle = angular_speed*(t1-t0)
##################################################################################

################## Part II: RANSAC transform to find fitted line ##################
# def my_RANSAC_transform_for_lines(scan_points):


# # fit line using all data
#     model = LineModelND()
#     model.estimate(scan_points)

# # robustly fit line only using inlier data with RANSAC algorithm
#     model_robust, inliers = ransac(scan_points, LineModelND, min_samples=2,
#                                residual_threshold=1, max_trials=1000)
#     outliers = inliers == False

# # generate coordinates of estimated models
#     line_x = np.arange(0, 5)
#     line_y = model.predict_y(line_x)
#     line_y_robust = model_robust.predict_y(line_x)

#     fig, ax = plt.subplots()
#     ax.plot(scan_points[inliers, 0], scan_points[inliers, 1], '.b', alpha=0.8,
#         label='Inlier data')
#     ax.plot(scan_points[outliers, 0], scan_points[outliers, 1], '.r', alpha=0.8,
#         label='Outlier data')
#     ax.plot(line_x, line_y, '-k', label='Line model from all data')
#     ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
#     ax.legend(loc='lower left')
#     plt.show()

#     a = np.arctan([line_x, line_y_robust])
#     b = np.arctan([line_x, line_y])

#     print(a, b)
# ###############################################################################

# ################# Part III: Send the rotation command to robot ################
# def rotate():
#     #Starts a new node
#     rospy.init_node('robot_cleaner', anonymous=True)
#     velocity_publisher = rospy.Publisher('/husky/cmd_vel', Twist, queue_size=10)
#     vel_msg = Twist()

#     # Receiveing the user's input
#     print("Let's rotate your robot")
#     speed = input("Input your speed (degrees/sec):") ## check husky angular speed
#     angle = input(b)
#     clockwise = input("Clockwise?: ") #True or false

#     #Converting from angles to radians
#     angular_speed = speed*2*PI/360
#     relative_angle = angle*2*PI/360

#     #We wont use linear components
#     vel_msg.linear.x=0
#     vel_msg.linear.y=0
#     vel_msg.linear.z=0
#     vel_msg.angular.x = 0
#     vel_msg.angular.y = 0

#     # Checking if our movement is CW or CCW
#     if clockwise:
#         vel_msg.angular.z = -abs(angular_speed)
#     else:
#         vel_msg.angular.z = abs(angular_speed)
#     # Setting the current time for distance calculus
#     t0 = rospy.Time.now().to_sec()
#     current_angle = 0

#     while(current_angle < relative_angle):
#         velocity_publisher.publish(vel_msg)
#         t1 = rospy.Time.now().to_sec()
#         current_angle = angular_speed*(t1-t0)


#     #Forcing our robot to stop
#     vel_msg.angular.z = 0
#     velocity_publisher.publish(vel_msg)
#     rospy.spin()

# if __name__ == '__main__':
#     try:
#         # Testing our function
#         rotate()
#     except rospy.ROSInterruptException:
#         pass
#########################################################################################

def publish_rotate_function():
    # x = threading.Thread(target=publish_rotate_function)
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    vel_msg = Twist()

    #We wont use linear components
    vel_msg.linear.x=0
    vel_msg.linear.y=0
    vel_msg.linear.z=0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0

    # Checking if our movement is CW or CCW
    # if clockwise:
    #     vel_msg.angular.z = -abs(angular_speed)
    # else:
    #     vel_msg.angular.z = abs(angular_speed)

    #Converting from angles to radians
    speed = 5
    angle = 20

    angular_speed = speed*2*math.pi/360
    relative_angle = angle*2*math.pi/360
    vel_msg.angular.z = abs(angular_speed)

    # Setting the current time for distance calculus
    # t0 = rospy.Time.now().to_sec()
    # print("t0=",t0)
    # current_angle = 0
    # print("current_angle = ",current_angle)
    print("relative_angle = ",relative_angle)

    # time to rotate
    t_rotate = abs(relative_angle / angular_speed)
    print("t_rotate=",t_rotate)

    # velocity_publisher.publish(vel_msg)
    # print("published!!!!!!!!!!!!!!!!!!!!!!!")
    # Checking if our movement is CW or CCW
    if relative_angle < 0:
        vel_msg.angular.z = -abs(angular_speed)
    else:
        vel_msg.angular.z = abs(angular_speed)

    r = rospy.Rate(100)
    velocity_publisher.publish(vel_msg)
    print("published")

    # time_i = 0
    # while(time_i < t_rotate):
    #     print("time_i=",time_i)
    #     velocity_publisher.publish(vel_msg)
        
    #     time_i = time_i+1




rospy.init_node('scan_values')
# ,disable_signals=True
sub = rospy.Subscriber('/scan', LaserScan, callback)
# velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
# command=Twist()
# r = rospy.Rate(20)

# while not rospy.is_shutdown():
#     print("In publisher!")
#     print("rotate_alpha",rotate_alpha)
#     command.angular.z= (rotate_alpha - 0) / 20.0
#     velocity_publisher.publish(command)
#     r.sleep()

# x = threading.Thread(target=publish_rotate_function)
# print("before start!")
# x.start()


rospy.spin()



