#include "ros/ros.h"
#include "geometry_msgs/Twist.h" 
#include "turtlesim/Pose.h"
#include <cmath>

class TurtleLawnmower{
    bool rightRotation = false; 

    bool leftRotation = false;

    ros::NodeHandle nh_;

    ros::Subscriber sub_;

    ros::Publisher pub_;

    public: 
        TurtleLawnmower();
        ~TurtleLawnmower();
        
        void turtleCallback
        (const turtlesim::Pose::ConstPtr& msg);
};

TurtleLawnmower::TurtleLawnmower() {
    ROS_INFO("Init");
    sub_ = nh_.subscribe("turtle1/pose", 1, &TurtleLawnmower::turtleCallback, this);
    pub_ = nh_.advertise<geometry_msgs::Twist>("turtle1/cmd_vel", 1);
}

TurtleLawnmower::~TurtleLawnmower() {
    ROS_INFO("Destructor");
}

void TurtleLawnmower::turtleCallback(const turtlesim::Pose::ConstPtr& msg) {
    geometry_msgs::Twist move; 
    move.linear.x = 1.0;

    if (msg->x > 10 && msg->theta!=M_PI){
        move.angular.z = 3.0;
    }
    else if (msg->x < 1 && msg->theta!=0)
    {
        move.angular.z = -3.0;
    } 
    else {
        move.angular.z = 0;
    }

    //Curve is not perfect because turtle goes out border where it turns around 
    //I tried using semaphore logic but then the problem lies in my subscriber on turtle1/pose which is not that frequent and rarely
    //comes to precisely 3.14 value 
    //Maybe using while loop that iterates through time calculated by formula --> time_for_turn = target_angle / angular_speed
    //would result in correct trajectory

    pub_.publish(move);
    //ROS_INFO("%f", move.linear.x);
}


int main(int argc, char** argv) {

    ros::init(argc, argv, "turtle_lawnmower_node");

    TurtleLawnmower TtMower;

    ros::spin();

    return 0;
}

