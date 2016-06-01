#include <stdio.h>

#include <doublefann.h>
#include <fann_cpp.h>

#include "FannTester.h"
#include "../trainer/LMTrainer.h"
using namespace FANN;

void FannTester::testWithFannNet(){
	neural_net* net=getInitializedFannNet();
	training_data* data=getFannData();

	LMTrainer trainer;
	trainer.trainFann(net, data);

	delete net;
	delete data;
}

neural_net* FannTester::getInitializedFannNet(){
	const float learning_rate = 0.7f;
	const int num_layers = 3;
	const int num_input = 2;
	const int num_hidden = 3;
	const int num_output = 1;
	printf("Creating net: Nlayers=%d, Ninputs=%d, Noutputs=%d\n", num_layers, num_input, num_output);

	neural_net* net=new FANN::neural_net();

	net->create_standard(num_layers, num_input, num_hidden, num_output);

	net->set_learning_rate(learning_rate);

	net->set_activation_steepness_hidden(1.0);
	net->set_activation_steepness_output(1.0);

	net->set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
	net->set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);
	return net;
}

FANN::training_data* FannTester::getFannData(){
	training_data* data=new training_data();
	printf("Creating EMPTY train data \n");
	return data;
}
