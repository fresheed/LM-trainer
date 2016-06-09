#include <stdio.h>

#include <doublefann.h>
#include "FannTester.h"
#include "../trainer/LMTrainer.h"
#include <iostream>
#include <typeinfo>

using namespace std;

void FannTester::testWithFannNet(){

//	struct fann* net=getInitializedFannNet();
//	struct fann_train_data* data=getFannData();

//	struct fann* net=get4ClassesFannNet();
//	struct fann_train_data* data=get4ClassesFannData();

//	struct fann* net=get5ClassesFannNet();
//	struct fann_train_data* data=get5ClassesFannData();

	struct fann* net=get3ClassesFannNet();
	struct fann_train_data* data=get3ClassesFannData();

	// struct fann* net=get7ClassesFannNet();
	// struct fann_train_data* data=get7ClassesFannData();


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

struct fann* FannTester::get4ClassesFannNet(){
	const int num_layers = 4;
	const int num_input = 2;
	const int num_hidden1 = 4;
	const int num_hidden2 = 3;
	const int num_output = 4;

	struct fann *ann = fann_create_standard(num_layers, num_input,
			num_hidden1, num_hidden2, num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_steepness_hidden(ann, 1.0);
	fann_set_activation_steepness_output(ann, 1.0);

	// explain ?
	fann_randomize_weights(ann, -0.5, 0.5);

	return ann;
}

struct fann_train_data* FannTester::get4ClassesFannData(){
	struct fann_train_data* data=fann_read_train_from_file("train_data/four_classes.set");
	fann_shuffle_train_data(data);
	return data;
}
//

struct fann* FannTester::get5ClassesFannNet(){
	const int num_layers = 4;
	const int num_input = 5;
	const int num_hidden1 = 7;
	const int num_hidden2 = 5;
	const int num_output = 5;

	struct fann *ann = fann_create_standard(num_layers, num_input,
			num_hidden1, num_hidden2, num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_steepness_hidden(ann, 1.0);
	fann_set_activation_steepness_output(ann, 1.0);

	// explain ?
	fann_randomize_weights(ann, -0.5, 0.5);

	return ann;
}

struct fann_train_data* FannTester::get5ClassesFannData(){
	struct fann_train_data* data=fann_read_train_from_file("train_data/5f_5c_5175e.set");
	//struct fann_train_data* data=fann_read_train_from_file("train_data/5f_5c_5175e_1000.set");
	return data;
}

struct fann* FannTester::get3ClassesFannNet(){
	const int num_layers = 4;
	const int num_input = 24;
	const int num_hidden1 = 7;
	const int num_hidden2 = 5;
	const int num_output = 3;

	struct fann *ann = fann_create_standard(num_layers, num_input,
			num_hidden1, num_hidden2, num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_steepness_hidden(ann, 1.0);
	fann_set_activation_steepness_output(ann, 1.0);

	// explain ?
	fann_randomize_weights(ann, -0.5, 0.5);

	return ann;
}

struct fann_train_data* FannTester::get3ClassesFannData(){
	struct fann_train_data* data=fann_read_train_from_file("train_data/24f_3c_487e.set");
	return data;
}

struct fann* FannTester::get7ClassesFannNet(){
	const int num_layers = 4;
	const int num_input = 33;
	const int num_hidden1 = 7;
	const int num_hidden2 = 5;
	const int num_output = 7;

	struct fann *ann = fann_create_standard(num_layers, num_input,
			num_hidden1, num_hidden2, num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_steepness_hidden(ann, 1.0);
	fann_set_activation_steepness_output(ann, 1.0);

	// explain ?
	fann_randomize_weights(ann, -0.5, 0.5);

	return ann;
}

struct fann_train_data* FannTester::get7ClassesFannData(){
	struct fann_train_data* data=fann_read_train_from_file("train_data/33f_7c_6684e.set");
	return data;
}
