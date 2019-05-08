#ifndef PIECE_LOCATOR_HPP
#define PIECE_LOCATOR_HPP

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/sfm.hpp>

#include <ros/ros.h>
#include <tf/transform_listener.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include <geometry_msgs/Point.h>

#include <algorithm>
#include <functional>

#include <exception>

#include "jps_feature_matching/ImageTransform.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

class PieceLocator
{
  private:
    ros::NodeHandle nh;
    ros::Timer timer;
    tf::StampedTransform base_camera_tf;
    tf::TransformBroadcaster tf_broadcaster;
    tf::TransformListener tf_listener;
    vector< vector< vector<Point2f> > > piece_central_points;
    vector< vector<Mat> > projection_matrices;
    vector< vector<Point3f> > piece_poses_3d;
    Mat camera_matrix;
    Mat e_cw;
    /*
     * \brief Image subscriber object
     */
    ros::Subscriber image_sub;
    ros::Subscriber camera_info_sub;

  public:
    PieceLocator(ros::NodeHandle& nh);
    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);
    void imageSubscriberCallback(
        const jps_feature_matching::ImageTransformConstPtr& msg);
    void timerCallback(const ros::TimerEvent& event);
};

void PoseToTransformation(tf::Point p, tf::Quaternion q, Mat &e);

void MatrixMultiplication(Mat a, Mat b, Mat& c);

#endif /* PIECE_LOCATOR_HPP */
