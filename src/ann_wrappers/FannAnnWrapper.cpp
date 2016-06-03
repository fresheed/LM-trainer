#include "FannAnnWrapper.hpp"

using namespace std;
#include <iostream>
#include <exception>
#include "../data_wrappers/FannDataWrapper.h"

FannAnnWrapper::FannAnnWrapper(struct fann* net){
	fann_net=net;
	connections=new fann_connection[getWeightAmount()];
	connections_actual=false;
	layer_sizes=new unsigned int[getLayersAmount()];
	fann_get_layer_array(fann_net, layer_sizes);

	layer_first_neurons=new unsigned int[getLayersAmount()];
	int cumsum=0;
	for (int i=0; i<getLayersAmount(); i++){
		layer_first_neurons[i]=cumsum;
		cumsum+=getNeuronsInLayer(i);
	}
}

int FannAnnWrapper::getInputsAmount(){
	return fann_get_num_input(fann_net);
}

int FannAnnWrapper::getOutputsAmount(){
	return fann_get_num_output(fann_net);
}

int FannAnnWrapper::getLayersAmount(){
	return fann_get_num_layers(fann_net);
}

int FannAnnWrapper::getNeuronsInLayer(int layer){
	int neurons=layer_sizes[layer];
	if (layer != getLayersAmount()-1)
		neurons++; //layer sizes does not include bias
	return neurons;

}

int FannAnnWrapper::getWeightAmount(){
	int num_conns=fann_get_total_connections(fann_net);
	return num_conns;
}

void FannAnnWrapper::updateConnections(){
	if (!connections_actual){
		fann_get_connection_array(fann_net, connections);
		connections_actual=true;
	}
}

double FannAnnWrapper::getWeightByIndex(int index){
	updateConnections();
	return connections[index].weight;
}

double FannAnnWrapper::getWeightInLayer(int layer, int index_in_layer){
	//unsigned int neuron_index=layer_first_neurons[layer];
	//return getWeightByIndex(neuron_index+index_in_layer);
	return 1e10d;
}

void FannAnnWrapper::printWeights(){
	cout << endl;
	cout << "Weights: "<< endl;
	for (int n=0; n<getWeightAmount(); n++){
		cout << getWeightByIndex(n) << " ";
	}
	cout << endl;
}

void FannAnnWrapper::fillErrorMatrix(DataWrapper* train_data, Mtx* error_matrix){
	int num_data=train_data->getExamplesAmount();
	int outs=getOutputsAmount();

	for (int ex_i=0; ex_i<num_data; ex_i++){
		double* cur_input=train_data->getInputByIndex(ex_i);
		double* desired_out=train_data->getDesiredOutputByIndex(ex_i);
		double* net_out=fann_run(fann_net, cur_input);
		for (int cur_out=0; cur_out<outs; cur_out++){
			error_matrix->set( outs*ex_i + cur_out, 0, net_out[cur_out]-desired_out[cur_out] );
		}
	}
}


void FannAnnWrapper::fillJacobianMatrix(DataWrapper* train_data, Mtx* jacobian_matrix){
//	if ( FannDataWrapper *fann_train_data = dynamic_cast< FannDataWrapper* >( train_data ) ) {
//		cout << "Found FANN data"<< endl;
//	}
//	else {
//		cout << "Some other train data"<< endl;
//	}

#define tmp_hide
#ifndef tmp_hide
		struct fann *test_ann=data->test_ann;


		struct fann_neuron *neuron_it, *first_neuron=test_ann->first_layer->first_neuron, *prev_neurons;
		struct fann_layer *current_layer, *first_layer=test_ann->first_layer, *after_last_layer=test_ann->last_layer;
		fann_type *error_begin=test_ann->train_errors,
				*last_layer_errors=error_begin+(((after_last_layer-1)->first_neuron)-first_neuron);

		//add_to_weights(work_ann, test_ann, cur_dws);

		size_t inp_iter;
		for (inp_iter = 0; inp_iter < train_data->num_data; inp_iter++) {
			fann_reset_MSE(test_ann);
			memset(test_ann->train_errors, 0, (test_ann->total_neurons) * sizeof(fann_type));

			fann_run(test_ann, train_data->input[inp_iter]);
			//setup error for last layer
			fann_type *error_it=last_layer_errors;
			for(neuron_it = (after_last_layer-1)->first_neuron; neuron_it != (after_last_layer-1)->last_neuron; neuron_it++){
				// do not iterate connections
				if (neuron_it->last_con == neuron_it->first_con) continue; //skip bias neuron
				*error_it=(-1)*fann_activation_derived(neuron_it->activation_function, neuron_it->activation_steepness,
						neuron_it->value, neuron_it->sum);
				error_it++;
			}

			//backprop gradient
			fann_backpropagate_MSE(test_ann);

			int i;

			//extract weights deltas
			const fann_type learning_rate=1.0;
			for(current_layer = (first_layer + 1); current_layer != after_last_layer; current_layer++) {
				struct fann_neuron *last_neuron = current_layer->last_neuron;
				prev_neurons = (current_layer - 1)->first_neuron;

				for(neuron_it = current_layer->first_neuron; neuron_it != last_neuron; neuron_it++)	{
					fann_type base_error = error_begin[neuron_it - first_neuron] * learning_rate;
					int num_connections = neuron_it->last_con - neuron_it->first_con;
					int i;
					for(i = 0; i != num_connections; i++){
						fann_type dwi = base_error * prev_neurons[i].value;
						gsl_matrix_float_set (J, inp_iter, neuron_it->first_con+i, dwi);
					}

				}
			}
		}
#endif
}

double FannAnnWrapper::getErrorOnSet(DataWrapper* train_data){
	return -1;
}

FannAnnWrapper::~FannAnnWrapper(){
	// coed
	delete connections;
	delete layer_sizes;
	delete layer_first_neurons;
}
