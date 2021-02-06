#include "PID.h"

/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  p_error_ = 0;
  i_error_ = 0;
  d_error_ = 0;
  total_error_ = 0;

  Kp_ = Kp;
  Ki_ = Ki;
  Kd_ = Kd;
}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  total_error_ = cte * cte;

  d_error_ = cte - p_error_;
  i_error_ += cte;
  p_error_ = cte;
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  return total_error_;  // TODO: Add your total error calc here!
}

double PID::Output(){
  double output = Kp_ * p_error_ + Ki_ * i_error_ + Kd_ * d_error_;
  if(output > 1){
    output = 1;
  }
  if(output < -1){
    output = -1;
  }
  return output;
}