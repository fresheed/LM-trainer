/*
 * fann_cpp_test.cpp
 *
 *  Created on: 29 мая 2016 г.
 *      Author: fresheed
 */
#include <iostream>

#include <doublefann.h>
#include <fann_cpp.h>
#include <fann.h>
#include "ann_wrappers/AnnWrapper.hpp"
#include "ann_wrappers/FannAnnWrapper.hpp"


using namespace std;

int main2(){  
  const float learning_rate = 0.7f;
  const unsigned int num_layers = 3;
  const unsigned int num_input = 2;
  const unsigned int num_hidden = 3;
  const unsigned int num_output = 1;
  const float desired_error = 0.001f;
  const unsigned int max_iterations = 300000;
  const unsigned int iterations_between_reports = 1000;
  FANN::neural_net net;
  net.create_standard(num_layers, num_input, num_hidden, num_output);

  net.set_learning_rate(learning_rate);
  
  net.set_activation_steepness_hidden(1.0);
  net.set_activation_steepness_output(1.0);

  net.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
  net.set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);

  net.print_connections();

}


int main(){
	//main2();
	FannAnnWrapper *fw=new FannAnnWrapper();

	delete fw;
}


