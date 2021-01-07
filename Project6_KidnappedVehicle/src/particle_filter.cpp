/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include <cmath>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  // Create normal distributions for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Gaussian sampling of particles
  for (int i = 0; i < num_particles; i++){
    struct Particle sample_particle;
    
    sample_particle.id      = i;
    sample_particle.x       = dist_x(gen);
    sample_particle.y       = dist_y(gen);
    sample_particle.theta   = dist_theta(gen);
    sample_particle.weight  = 1.0;

    particles.push_back(sample_particle);
  }
  
  // set initialized flag true
  is_initialized = true;
  
  // initialize weights of all particles
  weights.assign(num_particles, 1.0);
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  // Create normal distributions for noises of x, y and theta
  for (std::vector<Particle>::size_type i = 0; i < particles.size(); ++i){
    double x_old     = particles[i].x;
    double y_old     = particles[i].y;
    double theta_old = particles[i].theta;

    double x_pred, y_pred, theta_pred;
    
    theta_pred      = std::fmod((theta_old + yaw_rate * delta_t) , (2 * M_PI));
    x_pred          = x_old + velocity / yaw_rate * (sin(theta_pred) - sin(theta_old));
    y_pred          = y_old + velocity / yaw_rate * (cos(theta_old) - cos(theta_pred));

    normal_distribution<double> dist_x(x_pred, std_pos[0]);
    normal_distribution<double> dist_y(y_pred, std_pos[1]);
    normal_distribution<double> dist_theta(theta_pred, std_pos[2]);

    particles[i].x      = dist_x(gen);
    particles[i].y      = dist_y(gen);
    particles[i].theta  = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(std::vector<LandmarkObs>::size_type i = 0; i < observations.size(); ++i){
    double dist_min = std::numeric_limits<double>::max();

    for(std::vector<LandmarkObs>::size_type j = 0; j < predicted.size(); ++j){
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if(distance < dist_min){
          observations[i].id  = predicted[j].id;
          dist_min = distance;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  for(std::vector<Particle>::size_type i = 0; i < particles.size(); i++){
    /** 
     * Transform the local observations to from global coordinate system with respect to the each particle
     * |  cos(theta) -sin(theta)   particle.x |   | observation.x |
     * |  sin(theta)  cos(theta)   particle.y | * | observation.y |
     * |     0            0             1     |   |       1       |
     */
    std::vector<LandmarkObs> observations_global;    
    for(std::vector<LandmarkObs>::size_type j = 0; j < observations.size(); j++){
      struct LandmarkObs  observation_transformed;

      observation_transformed.id  = observations[j].id;
      observation_transformed.x   = cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y + particles[i].x;
      observation_transformed.y   = sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y + particles[i].y;

      observations_global.push_back(observation_transformed);
    }

    /**
     * Find the landmarks in the sensor range
     */

    std::vector<LandmarkObs> landmarks_nearby;
    for(size_t j = 0; j < map_landmarks.landmark_list.size(); j++){
      struct LandmarkObs  landmark;
      landmark.id     = map_landmarks.landmark_list[j].id_i;
      landmark.x      = map_landmarks.landmark_list[j].x_f;
      landmark.y      = map_landmarks.landmark_list[j].y_f;
      double distance = dist(landmark.x, landmark.y, particles[i].x, particles[i].y);
      if(distance <= sensor_range){
        landmarks_nearby.push_back(landmark);
      }
    }
    
    /**
     * Associate observation and landmark
     */
    dataAssociation(landmarks_nearby, observations_global);
    
    particles[i].associations.clear();
    for(std::vector<LandmarkObs>::size_type j = 0; j < observations_global.size(); ++j){
      particles[i].associations.push_back(observations_global[j].id);
    }
    /**
     * Calculate weight of the particle
     */
    double weight = 1.0;

    for(std::vector<LandmarkObs>::size_type j = 0; j < observations_global.size(); ++j){
      double mu_x, mu_y;
      double std_x  = std_landmark[0];
      double std_y  = std_landmark[1];
      
      for(std::vector<LandmarkObs>::size_type k = 0; k < landmarks_nearby.size(); ++k){
        if(observations_global[j].id == landmarks_nearby[k].id){
          mu_x = landmarks_nearby[k].x;
          mu_y = landmarks_nearby[k].y;
          break;
        }
      }
      
      double den = 2 * M_PI * std_x * std_y;
      double num = exp( -pow(observations_global[j].x - mu_x, 2) / (2 * std_x * std_x) - pow(observations_global[j].y - mu_y, 2) / (2 * std_y * std_y) );

      weight *= (num / den);
    }

    particles[i].weight = weight;
  }

  // normalize weights of particles
  double total_weight = 0.0;
  for(std::vector<Particle>::size_type i = 0; i < particles.size(); i++){
    total_weight += particles[i].weight;
  }

  for(std::vector<Particle>::size_type i = 0; i < particles.size(); i++){
    particles[i].weight /= (total_weight +  std::numeric_limits<double>::epsilon()); // add epsilon to avoid zero division
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  weights.clear();
  for(std::vector<Particle>::size_type i = 0; i< particles.size(); i++){
    weights.push_back(particles[i].weight);
  }

  std::discrete_distribution<int> weighted_distribution(weights.begin(), weights.end());

  std::vector<Particle> resampled_particles;
  for(size_t n = 0; n < num_particles; ++n){
    int sampled_index = weighted_distribution(gen);
    resampled_particles.push_back(particles[sampled_index]);
  }
  
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}