/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 10;
    default_random_engine gen;
    
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    for(int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;
        particles.push_back(p);
        
    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    
    double new_theta;
    double new_x_mean;
    double new_y_mean;
    
    for(int i = 0; i < num_particles; i++) {
        if(fabs(yaw_rate) < 0.001) {
            new_theta = particles[i].theta;
            new_x_mean = particles[i].x + (velocity * delta_t * cos(new_theta));
            new_y_mean = particles[i].y + (velocity * delta_t * sin(new_theta));
        }
        else {
            new_theta = particles[i].theta + yaw_rate * delta_t;
            new_x_mean = particles[i].x + (velocity / yaw_rate) * (sin(new_theta) - sin(particles[i].theta));
            new_y_mean = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(new_theta));
        }
        
        normal_distribution<double> dist_x(new_x_mean, std_pos[0]);
        normal_distribution<double> dist_y(new_y_mean, std_pos[1]);
        normal_distribution<double> dist_theta(new_theta, std_pos[2]);
        
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    LandmarkObs converted_obs;
    LandmarkObs best_landmark;
    
    weights.clear();
    
    for(int i = 0; i < int(particles.size()); i++) {
        double prob = 1;
        for(int k = 0; k < int(observations.size()); k++) {
            converted_obs = transformCoords(particles[i], observations[k]);
            best_landmark = associateLandmark(converted_obs, map_landmarks, std_landmark);
            double e = calculateWeights(converted_obs, best_landmark, std_landmark);
            prob *= e;
        }
        
        particles[i].weight = prob;
        weights.push_back(prob);
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<Particle> new_particles;
    int index;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> weight_distribution(weights.begin(), weights.end());
    
    for(int i = 0; i < num_particles; i++) {
        index = weight_distribution(gen);
        new_particles.push_back(particles[index]);
    }
    particles = new_particles;

}

LandmarkObs ParticleFilter::transformCoords(Particle part, LandmarkObs obs) {
    LandmarkObs transfromed_coords;
    
    transfromed_coords.id = obs.id;
    transfromed_coords.x = obs.x * cos(part.theta) - obs.y * sin(part.theta) + part.x;
    transfromed_coords.y = obs.x * sin(part.theta) + obs.y * cos(part.theta) + part.y;
    
    return transfromed_coords;
}

LandmarkObs ParticleFilter::associateLandmark(LandmarkObs converted_obs, Map map_landmarks, double std_landmark[]) {
    LandmarkObs best_landmark;
    
    for(int m = 0; m < int(map_landmarks.landmark_list.size()); m++) {
        double min_dist;
        double distance = dist(converted_obs.x, converted_obs.y, map_landmarks.landmark_list[m].x_f, map_landmarks.landmark_list[m].y_f);
        
        if(m == 0) {
            min_dist = distance;
            best_landmark.id = map_landmarks.landmark_list[m].id_i;
            best_landmark.x = map_landmarks.landmark_list[m].x_f;
            best_landmark.y = map_landmarks.landmark_list[m].y_f;
        }
        else if(distance < min_dist) {
            min_dist = distance;
            best_landmark.id = map_landmarks.landmark_list[m].id_i;
            best_landmark.x = map_landmarks.landmark_list[m].x_f;
            best_landmark.y = map_landmarks.landmark_list[m].y_f;
        }
    }
    
    return best_landmark;
}

double ParticleFilter::calculateWeights(LandmarkObs obs, LandmarkObs best_landmark, double std_landmark[]) {
    const double sigma_x = std_landmark[0];
    const double sigma_y = std_landmark[1];
    const double d_x = obs.x - best_landmark.x;
    const double d_y = obs.y - best_landmark.y;
    
    double e = (1/(2. * M_PI * sigma_x * sigma_y)) * exp(-(d_x * d_x / (2 * sigma_x * sigma_x)) + (d_y * d_y / (2 * sigma_y * sigma_y)));
    
    return e;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
