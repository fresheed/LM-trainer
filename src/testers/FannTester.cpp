#include <stdio.h>

#include <doublefann.h>

#include "FannTester.h"
#include "../trainer/LMTrainer.h"
#include <iostream>

using namespace std;

void FannTester::testWithFannNet(){
	struct fann* net=getInitializedFannNet();
	struct fann_train_data* data=getFannData();

	LMTrainer trainer;
	trainer.trainFann(net, data);

	fann_destroy(net);
	fann_destroy_train(data);
}

struct fann* FannTester::getInitializedFannNet(){
	const int num_layers = 3;
	const int num_input = 2;
	const int num_hidden = 2;
	const int num_output = 2;
	printf("Creating net: Nlayers=%d, Ninputs=%d, Noutputs=%d\n", num_layers, num_input, num_output);

	struct fann *ann = fann_create_standard(num_layers, num_input,
			num_hidden, num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_steepness_hidden(ann, 1.0);
	fann_set_activation_steepness_output(ann, 1.0);

	const int Nweights=12;
	double weights_values[Nweights]={-0.585146d,  -0.845445d,  -0.496637d,  0.218401d,  -0.014935d,  -0.510967d,
									-0.698637d,  -0.100947d,  -0.042599d ,  0.759842d ,  0.156978d ,  0.750875d};

	fann_set_weight(ann, 0, 3, weights_values[0]);
	fann_set_weight(ann, 1, 3, weights_values[1]);
	fann_set_weight(ann, 2, 3, weights_values[2]);
	fann_set_weight(ann, 0, 4, weights_values[3]);
	fann_set_weight(ann, 1, 4, weights_values[4]);
	fann_set_weight(ann, 2, 4, weights_values[5]);

	fann_set_weight(ann, 3, 6, weights_values[6]);
	fann_set_weight(ann, 4, 6, weights_values[7]);
	fann_set_weight(ann, 5, 6, weights_values[8]);
	fann_set_weight(ann, 3, 7, weights_values[9]);
	fann_set_weight(ann, 4, 7, weights_values[10]);
	fann_set_weight(ann, 5, 7, weights_values[11]);

	return ann;
}

struct fann_train_data* FannTester::getFannData(){
	struct fann_train_data* data=fann_read_train_from_file("train_data/xor.data");
	return data;
}
