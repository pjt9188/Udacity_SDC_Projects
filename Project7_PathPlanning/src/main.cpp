#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int     lane = 2;
double  ref_vel = 0; // mph 

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side 
          //   of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          json msgJson;

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          /**
           * TODO: define a path made up of (x,y) points that the car will visit
           *   sequentially every .02 seconds
           */        
        
          /*******************************   Congfiguration   **************************************/
          // int     lane = 2;
          // double  ref_vel = 0; // mph
          int     prev_size = previous_path_x.size(); // Save previous remained path list in the next path list
          double  safety_distance = 30; // m

          /********************************   Sensor Fusion   **************************************/
          
          bool too_close = false;

          for(int i = 0; i < sensor_fusion.size(); i++){
            float d = sensor_fusion[i][6];
            
            // When the adjacent car is in the same lane
            if(d > (4*lane - 4) && d < (4*lane) ){
              double  vx           = sensor_fusion[i][3];
              double  vy           = sensor_fusion[i][4];
              double  check_speed  = sqrt(vx*vx + vy*vy);
              double  check_car_s  = sensor_fusion[i][5];

              check_car_s += ((double)prev_size * 0.02 * check_speed);

              // check s value whether it is greater than mine + safety_distance
              if( (check_car_s > end_path_s) && ((check_car_s - end_path_s) < safety_distance) ){
                // ref_vel = check_speed * MPS2MPH;
                too_close = true;
              }
            }
          }

          if(too_close){
            ref_vel -= 0.224;
          }
          else if(ref_vel < (MAX_VEL - 0.5)){
            // ref_vel = ((ref_vel + 0.224) > (MAX_VEL - 0.5)) ? (MAX_VEL - 0.5) : (ref_vel + 0.224);
            ref_vel += 0.224;
          }

          std::cout << ref_vel << std::endl;

          /********************************   Path Planning   **************************************/          
          for (int i = 0; i < prev_size; ++i) {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          // Make spline using previous path
          //  Add two previous points which is in the previous path to the point lists 
          double ref_x, ref_y, ref_yaw;
          double ref_x_prev, ref_y_prev;
          vector<double> ptsx, ptsy;

          if (prev_size < 2) {
            ref_x = car_x;
            ref_y = car_y;
            ref_yaw = deg2rad(car_yaw);
            ref_x_prev = car_x - cos(ref_yaw);
            ref_y_prev = car_y - sin(ref_yaw);
          } 
          else {
            ref_x = previous_path_x[prev_size-1];
            ref_y = previous_path_y[prev_size-1];
            ref_x_prev = previous_path_x[prev_size-2];
            ref_y_prev = previous_path_y[prev_size-2];
            ref_yaw = atan2(ref_y-ref_y_prev,ref_x-ref_x_prev);
          }

          ptsx.push_back(ref_x_prev);
          ptsx.push_back(ref_x);
          
          ptsy.push_back(ref_y_prev);
          ptsy.push_back(ref_y);

          //  Add three points which is evenly spaced by 30 meters ahead from the starting reference
          vector<double> next_wpt;
          for (int i = 1; i <= 3; i++){
            next_wpt = getXY(car_s + 30*i, (4*lane - 2), map_waypoints_s, map_waypoints_x, map_waypoints_y);
            ptsx.push_back(next_wpt[0]);
            ptsy.push_back(next_wpt[1]);
          }

          //  Convert points into local coordinates
          for(int i = 0; i<ptsx.size(); i++){
            double shift_x = ptsx[i] - ref_x;
            double shift_y = ptsy[i] - ref_y;

            ptsx[i] =  cos(ref_yaw) * shift_x + sin(ref_yaw) * shift_y;
            ptsy[i] = -sin(ref_yaw) * shift_x + cos(ref_yaw) * shift_y;
          }

          //  Create a spline in local coordinates
          tk::spline s_local;
          s_local.set_points(ptsx, ptsy);

          double target_x = 30.0;
          double target_y = s_local(target_x);
          double target_dist = sqrt(target_x * target_x + target_y * target_y);
          
          for(int i = 1; i <= 50 - previous_path_x.size(); i++){
            // if(too_close){
            //   double delta_acc = MAX_JERK * 0.02 * 0.9;
            //   ref_vel -= delta_acc*0.02;
            // }
            // else if(ref_vel < (MAX_VEL - 0.5)){
            //   ref_vel = ( (ref_vel + delta_acc*0.02) > (MAX_VEL - 0.5) ) ? (MAX_VEL - 0.5) : (ref_vel + delta_acc*0.02);
            // }

            double N = target_dist / ((ref_vel / MPS2MPH) * 0.02);
            double target_dx = target_x / N;

            double x_local = target_dx * i;
            double y_local = s_local(x_local);
            double x_glob = cos(ref_yaw) * x_local - sin(ref_yaw) * y_local + ref_x;
            double y_glob = sin(ref_yaw) * x_local + cos(ref_yaw) * y_local + ref_y;

            next_x_vals.push_back(x_glob);
            next_y_vals.push_back(y_glob);
          }
      
          /**
           * END
           */

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}