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

/*******************************   States   *************************************/
int     car_state = KL;
int     car_lane = 1;
double  ref_vel = 0; // mps

bool    front_car_is_too_close         = false;
bool    can_change_to_the_left_lane    = false;
bool    can_change_to_the_right_lane   = false;

/********************************************************************************/

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
        
          /*******************************   Congfiguration   *************************************/
          car_speed /= MPS2MPH;                     // convert mile/h to m/s
          const double  dt = 0.02;                  // time interval of the path
          const int     num_of_path_points = 50;    // total path points
          const double  safety_distance = 50;       // m

          const int     prev_size = previous_path_x.size(); // Save previous remained path list in the next path list

          ref_vel   = car_speed;                    // m/s
          car_lane  = (int)car_d / 4;               // lane 0 to 2
          
          // Flags
          front_car_is_too_close         = false;
          can_change_to_the_left_lane    = (car_lane == 1) || (car_lane == 2);
          can_change_to_the_right_lane   = (car_lane == 0) || (car_lane == 1);

          /*******************************   Prediction   *****************************************/
          vector<vector<double>> prediction;

          for(int i = 0; i < sensor_fusion.size(); i++){
            double  vx          = sensor_fusion[i][3];  // m/s
            double  vy          = sensor_fusion[i][4];  // m/s
            double  s           = sensor_fusion[i][5];  // m
            double  d           = sensor_fusion[i][6];  // m
            int     lane        = (int)d % 4;

            // When sensed car is near my car
            if( (s - car_s <= safety_distance) && (s - car_s >= -safety_distance / 3) 
            && (d >= 0.0 and d <= 12.0) && (fabs(d - car_d) <= 6) ){
              double  speed = sqrt(vx*vx + vy*vy);  // m/s
              vector<double> pred = {s, d, speed};
              prediction.push_back(pred);
            }
          }

          /*******************************   Finite State Transition   ******************************/
          vector<int> possible_next_states;
          switch(car_state){
            case CL:
              possible_next_states.push_back(CL);
              possible_next_states.push_back(KL);
              break;

            case KL:
              possible_next_states.push_back(KL);
              if(can_change_to_the_right_lane){
                possible_next_states.push_back(CR);
              }
              if(can_change_to_the_left_lane){
                possible_next_states.push_back(CL);
              }
              break;

            case CR:
              possible_next_states.push_back(CR);
              possible_next_states.push_back(KL);
              break;
          }

          /*******************************   Behavior Planning   *************************************/
          // initiate the target speeds of each lane(left lane, current lane, right lane) by 90% of the maximum speed in m/s
          double target_speed[3] = {MAX_VEL / MPS2MPH * 0.9, MAX_VEL / MPS2MPH * 0.9, MAX_VEL / MPS2MPH * 0.9}; // m/s
          double near_car_s[3] = {999, 999, 999};
          
          // check whether car can change the lane or not
          for(int i = 0; i < prediction.size(); i++){
            double  s     = prediction[i][0];
            double  d     = prediction[i][1];
            double  speed = prediction[i][2]; // m/s
            int     lane  = (int)d / 4;

            switch(lane - car_lane){
              case -1 :   // when the sensed car is left of the my car
                can_change_to_the_left_lane = can_change_to_the_left_lane && (fabs(s - car_s) >= safety_distance);
                target_speed[0] = (target_speed[0] > speed) ? speed : target_speed[0];
                near_car_s[0] = (near_car_s[0] > fabs(s - car_s))? fabs(s - car_s) : near_car_s[0];
                break;
              
              case 0:     // when the sensed car is on the same lane of the my car
                if(s > car_s){
                  target_speed[1] = (target_speed[1] > speed) ? speed : target_speed[1];
                }
                near_car_s[1] = (near_car_s[1] > fabs(s - car_s))? fabs(s - car_s) : near_car_s[1];
                break;

              case 1 :   // when the sensed car is right of the my car
                can_change_to_the_right_lane = can_change_to_the_right_lane && (fabs(s - car_s) >= safety_distance);
                target_speed[2] = (target_speed[2] > speed) ? speed : target_speed[2];
                near_car_s[2] = (near_car_s[2] > fabs(s - car_s))? fabs(s - car_s) : near_car_s[2];
                break;
            }

            if(fabs(d - car_d) < 2){
              front_car_is_too_close = (s - car_s >= 0) && (s - car_s <= safety_distance * 0.6);
            }
          }
          

          /*******************************   Cost Calculation   ***************************************/
          double  cost[3]  = {999., 999., 999.};   // CL, KL, CR
          double  min_cost  = 999.;
          int     min_state = 0;
          
          for(int i = 0; i < possible_next_states.size(); i++){
            int next_state = possible_next_states[i];
            
            switch(next_state){
              case CL:
                if(can_change_to_the_left_lane){
                  // Delta s is calculated in the assumption that the car moves in the constant acceleration
                  cost[0] = calculate_cost(target_speed[0], (target_speed[0] + car_speed)/2 * dt * num_of_path_points, fmod(car_d, 4.0) + 2);
                  if(min_cost > cost[0]){
                    min_cost = cost[0];
                    min_state = CL;
                  }
                }
                break;
              
              case KL:
                cost[1] = calculate_cost(target_speed[1], (target_speed[1] + car_speed)/2 * dt * num_of_path_points, fmod(car_d, 4.0) - 2);
                if(min_cost > cost[1]){
                    min_cost = cost[1];
                    min_state = KL;
                  }
                break;

              case CR:
                if(can_change_to_the_right_lane){
                  cost[2] = calculate_cost(target_speed[2], (target_speed[2] + car_speed)/2 * dt * num_of_path_points, 6 - fmod(car_d, 4.0));
                  if(min_cost > cost[2]){
                    min_cost = cost[2];
                    min_state = CR;
                  }
                }
                break;
            }  
          }

          /********************************   Path Generation   **************************************/
          // Decelerate when front car is too close
          if(front_car_is_too_close){
            ref_vel -= 1;
          }
          else{
            switch(min_state){
              case CL:
                ref_vel = target_speed[0];
                car_lane -= 1;
                car_state = CL;
                break;
              
              case KL:
                ref_vel = target_speed[1];
                car_state = KL;
                break;
              
              case CR:
                ref_vel = target_speed[2];
                car_lane += 1;
                car_state = CR;
                break;
            }

            // Acclerate the car speed within 0.75 m/s
            if(ref_vel >= car_speed){
              ref_vel = (ref_vel - car_speed > 0.75) ? car_speed + 0.75 : ref_vel;
            }
            else{
              ref_vel = (ref_vel - car_speed < -0.75) ? car_speed - 0.75 : ref_vel;
            } 
          }
          
          std::cout<< "ref_vel : " << ref_vel * MPS2MPH << std::endl;

          for (int i = 0; i < prev_size; ++i) {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          // Make spline using previous path
          double ref_x, ref_y, ref_yaw;
          double ref_x_prev, ref_y_prev;
          vector<double> ptsx, ptsy;

          //  Add previous points which is in the previous path to the point lists 
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

            for(int i = prev_size; i > 2; i--){
              ptsx.push_back(previous_path_x[prev_size-i]);
              ptsy.push_back(previous_path_y[prev_size-i]);
            }
          }

          ptsx.push_back(ref_x_prev);
          ptsy.push_back(ref_y_prev);
          
          ptsx.push_back(ref_x);
          ptsy.push_back(ref_y);

          //  Add three points which is evenly spaced by 30 meters ahead from the starting reference
          vector<double> next_wpt;
          for (int i = 1; i <= 3; i++){
            next_wpt = getXY(car_s + 15 + 30*i, (4*car_lane + 2), map_waypoints_s, map_waypoints_x, map_waypoints_y);
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
            double N = target_dist / (ref_vel * 0.02);
            double target_dx = target_x / N;

            double x_local = target_dx * i;
            double y_local = s_local(x_local);
            double x_glob = cos(ref_yaw) * x_local - sin(ref_yaw) * y_local + ref_x;
            double y_glob = sin(ref_yaw) * x_local + cos(ref_yaw) * y_local + ref_y;

            next_x_vals.push_back(x_glob);
            next_y_vals.push_back(y_glob);
          }


          // Monitoring
          std::cout <<"lane : " << car_lane << "    " << "car_state : ";
          switch(car_state){
            case CL:
              std::cout << "CL" << std::endl;
              break;
            
            case KL:
              std::cout << "KL" << std::endl;
              break;

            case CR:
              std::cout << "CR" << std::endl;
              break;
          }
          
          std::cout <<"Flags : " << can_change_to_the_left_lane << "    "
                    << front_car_is_too_close << "    " 
                    << can_change_to_the_right_lane << std::endl;
          
          std::cout <<"speed : ";
          for(int i = 0; i < 3; i++){
            std::cout << target_speed[i] * MPS2MPH << "   ";
          }
          std::cout << std::endl;
          
          std::cout <<"distance : ";
          for(int i = 0; i < 3; i++){
            std::cout << near_car_s[i]<< "   ";
          }
          std::cout<< std::endl << std::endl;
      
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