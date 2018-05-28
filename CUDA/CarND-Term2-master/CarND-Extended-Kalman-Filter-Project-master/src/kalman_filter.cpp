#include "kalman_filter.h"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
    x_ = F_ * x_;
    MatrixXd Ft_ = F_.transpose();
    P_ = F_* P_* Ft_ + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
    VectorXd z_pred = H_ * x_;
    VectorXd y = z - z_pred;
    
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;
    
    x_ = x_ + (K*y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size,x_size);
    P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
    
    float px = x_[0];
    float py = x_[1];
    float vx = x_[2];
    float vy = x_[3];
    
    float rho;
    float phi;
    float rhodot;
    
    if(fabs(px)<0.0001 || fabs(py)<0.0001){
        if(fabs(px)<0.0001){
            px = 0.0001;
            
        }
        if(fabs(py)<0.0001){
            py = 0.0001;
        }
        
        rho = sqrt(px*px + py*py);
        phi = 0;
        rhodot = 0;
    }
    
    else{
        rho = sqrt(px*px + py*py);
        phi = atan2(py,px);
        rhodot = (px*vx + py*vy)/rho;
    }
    
    //if(phi > 3.14159){
    //    phi -= 6.2831;
    //}
    //if(phi < -3.14159){
    //    phi += 6.2831;
    //}
    
    VectorXd z_pred(3);
    z_pred << rho, phi, rhodot;
    VectorXd y = z - z_pred;
    
    if(y(1) > 3.14159){
        y(1) -= 6.2831;
    }
    if(y(1) < -3.14159){
        y(1) += 6.2831;
    }
    
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    x_ = x_ + (K*y);
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size,x_size);
    P_ = (I - K * H_) * P_;
}
