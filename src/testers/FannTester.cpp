#include <stdio.h>

#include <doublefann.h>
#include <fann_cpp.h>

#include "FannTester.h"
#include "../trainer/LMTrainer.h"
#include <iostream>
using namespace FANN;
using namespace std;

void FannTester::testWithFannNet(){
	FannHacker* net=getInitializedFannNet();
	training_data* data=getFannData();

	LMTrainer trainer;
	trainer.trainFann(net, data);

	delete net;
	delete data;
}

FannHacker* FannTester::getInitializedFannNet(){
	const float learning_rate = 0.7f;
	const int num_layers = 3;
	const int num_input = 2;
	const int num_hidden = 2;
	const int num_output = 2;
	printf("Creating net: Nlayers=%d, Ninputs=%d, Noutputs=%d\n", num_layers, num_input, num_output);

	FannHacker* net=new FannHacker();
	net->create_standard(num_layers, num_input, num_hidden, num_output);

	net->set_activation_steepness_hidden(1.0);
	net->set_activation_steepness_output(1.0);

	net->set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC);
	net->set_activation_function_output(FANN::SIGMOID_SYMMETRIC);

	const int Nweights=12;
	double weights_values[Nweights]={-0.585146d,  -0.845445d,  -0.496637d,  0.218401d,  -0.014935d,  -0.510967d,
									-0.698637d,  -0.100947d,  -0.042599d ,  0.759842d ,  0.156978d ,  0.750875d};

	net->set_weight(0, 3, weights_values[0]);
	net->set_weight(1, 3, weights_values[1]);
	net->set_weight(2, 3, weights_values[2]);
	net->set_weight(0, 4, weights_values[3]);
	net->set_weight(1, 4, weights_values[4]);
	net->set_weight(2, 4, weights_values[5]);

	net->set_weight(3, 6, weights_values[6]);
	net->set_weight(4, 6, weights_values[7]);
	net->set_weight(5, 6, weights_values[8]);
	net->set_weight(3, 7, weights_values[9]);
	net->set_weight(4, 7, weights_values[10]);
	net->set_weight(5, 7, weights_values[11]);

	return net;
}

FANN::training_data* FannTester::getFannData(){
	training_data* data=new training_data();
	bool success=data->read_train_from_file("train_data/xor.data");
	cout << "loaded: "<< success;
	return data;
}
