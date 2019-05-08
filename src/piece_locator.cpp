#include "piece_locator.h"

PieceLocator::PieceLocator(ros::NodeHandle& nh)
{
  int i;
  this->nh = nh;
  this->timer = this->nh.createTimer(
      ros::Duration(0.1),
      &PieceLocator::timerCallback,
      this);
  this->image_sub = this->nh.subscribe(
      "input_image",
      1,
      &PieceLocator::imageSubscriberCallback,
      this);
  this->camera_info_sub = this->nh.subscribe(
      "camera_info",
      1,
      &PieceLocator::cameraInfoCallback,
      this);

  for(i = 0; i < 4; ++i)
  {
    this->piece_central_points.push_back(vector< vector<Point2f> >());
    this->projection_matrices.push_back(vector<Mat>());
    this->piece_poses_3d.push_back(vector<Point3f>());
  }
}

void PieceLocator::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
{
  int i;

  if(this->camera_matrix.rows < 3)
  {
    this->camera_matrix = Mat::zeros(3, 4, CV_64F);

    for(i = 0; i < msg->K.size(); ++i)
    {
      this->camera_matrix.at<double>(i / 3, i % 3) = msg->K[i];
    }
  }
  else
  {
    // No operation
  }
}

void PieceLocator::imageSubscriberCallback(
    const jps_feature_matching::ImageTransformConstPtr& msg)
{
  int i;
  int j = msg->piece_index;
  int k;
  Mat proj_mat;
  tf::Transform piece_transform;
  std::ostringstream piece_name_stream;


  this->piece_central_points[j].push_back(vector<Point2f>());
  k = this->piece_central_points[j].size();

  for(i = 0; i < msg->transformed_points.size(); ++i)
  {
    piece_central_points[j][k].push_back(Point2f(
          msg->transformed_points[i].x,
          msg->transformed_points[i].y));
  }

  // Generate the projection matrix as [K 0] * E_cw and store both,
  // where p_c = E_cw * p_w
  MatrixMultiplication(this->camera_matrix, this->e_cw, proj_mat);
  this->projection_matrices[j].push_back(proj_mat);

  if(this->projection_matrices[j].size() >= 4)
  {
    // Generate the transformation for the current piece from the base frame.
    sfm::triangulatePoints(
        this->piece_central_points[j],
        this->projection_matrices[j],
        this->piece_poses_3d[j]);

    // Compute the x-axis.
    Point3f x_axis = Point3f(
        piece_poses_3d[j][1].x - piece_poses_3d[j][0].x,
        piece_poses_3d[j][1].y - piece_poses_3d[j][0].y,
        piece_poses_3d[j][1].z - piece_poses_3d[j][0].z);
    double x_axis_norm = sqrt(
        (x_axis.x * x_axis.x) + (x_axis.y * x_axis.y) + (x_axis.z * x_axis.z));
    x_axis.x /= x_axis_norm;
    x_axis.y /= x_axis_norm;
    x_axis.z /= x_axis_norm;

    // Compute the temporary y-axis.
    Point3f y_axis_temp = Point3f(
        piece_poses_3d[j][1].x - piece_poses_3d[j][0].x,
        piece_poses_3d[j][1].y - piece_poses_3d[j][0].y,
        piece_poses_3d[j][1].z - piece_poses_3d[j][0].z, );
    // Compute the z-axis.
    Point3f z_axis = x_axis.cross(y_axis_temp);
    double z_axis_norm = sqrt(
        (z_axis.x * z_axis.x) + (z_axis.y * z_axis.y) + (z_axis.z * z_axis.z));
    z_axis.x /= z_axis_norm;
    z_axis.y /= z_axis_norm;
    z_axis.z /= z_axis_norm;

    // Compute the actual y-axis, with corrections for any distortion.
    Point3f y_axis = z_axis.cross(x_axis);

    // Obtain the quaternion from the rotation matrix.
    piece_transform.setQuaternion(AxesToQuaternion(x_axis, y_axis, z_axis));

    // Set and publish the transformation.
    piece_transform.setOrigin(
        piece_poses_3d[j][0].x,
        piece_poses_3d[j][0].y,
        piece_poses_3d[j][0].z);
    piece_name_stream << "/piece_" << j;
    this->tf_broadcaster.sendTransform(tf::StampedTransform(
          piece_transform,
          ros::time::now(),
          "/base",
          piece_name_stream.str()));
  }
  else
  {
    // No operation
  }
}

void PieceLocator::timerCallback(const ros::TimerEvent& event)
{
  try
  {
    this->tf_listener.lookupTransform(
        "/camera_link", "/base", ros::Time(0), this->base_camera_tf);
    PoseToTransformation(
        this->base_camera_tf.getOrigin(),
        this->base_camera_tf.getRotation(),
        this->e_cw);
  }
  catch(tf::TransformException ex)
  {
    ROS_ERROR("%s", ex.what());
  }
}

/*
 */
void PoseToTransformation(tf::Point p, tf::Quaternion q, Mat &e)
{
  double qx2 = q.x() * q.x();
  double qy2 = q.y() * q.y();
  double qz2 = q.z() * q.z();
  double qxqy = q.x() * q.y();
  double qxqz = q.x() * q.z();
  double qxqw = q.x() * q.w();
  double qyqz = q.y() * q.z();
  double qyqw = q.y() * q.w();
  double qzqw = q.z() * q.w();

  e = Mat::zeros(4, 4, CV_64F);
  e.at<double>(0, 0) = 1 - (2.0 * (qy2 + qz2));
  e.at<double>(0, 1) = 2.0 * (qxqy + qzqw);
  e.at<double>(0, 2) = 2.0 * (qxqz - qyqw);
  e.at<double>(1, 0) = 2.0 * (qxqy - qzqw);
  e.at<double>(1, 1) = 1 - (2.0 * (qx2 + qz2));
  e.at<double>(1, 2) = 2.0 * (qyqz + qxqw);
  e.at<double>(2, 0) = 2.0 * (qxqz + qyqw);
  e.at<double>(2, 1) = 2.0 * (qyqz - qxqw);
  e.at<double>(2, 2) = 1 - (2.0 * (qx2 + qy2));
  e.at<double>(0, 3) = p.x();
  e.at<double>(1, 3) = p.y();
  e.at<double>(2, 3) = p.z();
  e.at<double>(3, 3) = 1.0;
}

/*
 * http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
 */
tf::Quaternion AxesToQuaternion(Point3f x_axis, Point3f, y_axis, Point3f z_axis)
{
  double rot_trace = x_axis.x + y_axis.y + z_axis.z;
  double q_norm_factor;
  double qx, qy, qz, qw;

  if(rot_trace > 0.0)
  {
    q_norm_factor = sqrt(tr+1.0) * 2; // q_norm_factor=4*qw
    qw = 0.25 * q_norm_factor;
    qx = (y_axis.z - z_axis.y) / q_norm_factor;
    qy = (z_axis.x - x_axis.z) / q_norm_factor;
    qz = (x_axis.y - y_axis.x) / q_norm_factor;
  }
  else if((x_axis.x > y_axis.y)&(x_axis.x > z_axis.z))
  {
    q_norm_factor = sqrt(1.0 + x_axis.x - y_axis.y - z_axis.z) * 2; // q_norm_factor=4*qx
    qw = (y_axis.z - z_axis.y) / q_norm_factor;
    qx = 0.25 * q_norm_factor;
    qy = (y_axis.x + x_axis.y) / q_norm_factor;
    qz = (z_axis.x + x_axis.z) / q_norm_factor;
  }
  else if(y_axis.y > z_axis.z)
  {
    q_norm_factor = sqrt(1.0 + y_axis.y - x_axis.x - z_axis.z) * 2; // q_norm_factor=4*qy
    qw = (z_axis.x - x_axis.z) / q_norm_factor;
    qx = (y_axis.x + x_axis.y) / q_norm_factor;
    qy = 0.25 * q_norm_factor;
    qz = (z_axis.y + y_axis.z) / q_norm_factor;
  }
  else
  {
    q_norm_factor = sqrt(1.0 + z_axis.z - x_axis.x - y_axis.y) * 2; // q_norm_factor=4*qz
    qw = (x_axis.y - y_axis.x) / q_norm_factor;
    qx = (z_axis.x + x_axis.z) / q_norm_factor;
    qy = (z_axis.y + y_axis.z) / q_norm_factor;
    qz = 0.25 * q_norm_factor;
  }

  return tf::Quaternion(qx, qy, qz, qw);
}

void MatrixMultiplication(Mat a, Mat b, Mat& c)
{
  int i;
  int j;
  int k;

  c = Mat::zeros(a.rows, b.cols, a.type());

  for(i = 0; i < c.rows; ++i)
  {
    for(j = 0; j < c.cols; ++j)
    {
      for(k = 0; k < a.cols; ++k)
      {
        c.at<double>(i, j) += a.at<double>(i, k) * b.at<double>(k, j);
      }
    }
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "piece_locator_node");
  ros::NodeHandle nh;
  PieceLocator p(nh);
  ros::spin();

  return 0;
}

