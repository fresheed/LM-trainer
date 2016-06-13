#include "FannAnnWrapper.h"

using namespace std;
#include <iostream>
#include <exception>
#include <string.h>
#include "../data_wrappers/FannDataWrapper.h"
#include <typeinfo>

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

void FannAnnWrapper::addToWeights(Mtx* delta_weights){
	connections_actual=false;
	updateConnections(); // should remove it
	for (int i=0; i<getWeightAmount(); i++){
		connections[i].weight+=delta_weights->get(i, 0);
	}
	fann_set_weight_array(fann_net, connections, getWeightAmount());
	updateConnections(); // should remove it
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

void FannAnnWrapper::explore_ann(struct fann * fann_net){
	struct fann_layer* first_layer=fann_net->first_layer;
	cout << fann_net->connection_rate<<endl;
	cout << "af: "<<(fann_net->first_layer +1)->first_neuron ->activation_function << endl;
	double test_inp[2]={4.5, -4.5};
	fann_run(fann_net, test_inp);
	for (unsigned int l=0; l<fann_get_num_layers(fann_net); l++){
		cout << "layer "<<l<<endl;
		struct fann_neuron* first_neuron_in_layer=(first_layer+l)->first_neuron;
		struct fann_neuron* neuron_it;
		for (int n=0; n<getNeuronsInLayer(l); n++){
			cout <<"  neuron "<<n<<endl;
			neuron_it=first_neuron_in_layer+n;
			cout <<"    "<<neuron_it<<endl;
			cout <<"    "<<neuron_it->activation_function<<endl;
			cout <<"    "<<neuron_it->sum<<endl;
			cout <<"    "<<neuron_it->value<<endl;
			cout <<" df "<<fann_activation_derived(neuron_it->activation_function, neuron_it->activation_steepness,
							neuron_it->value, neuron_it->sum) <<endl;
		}
	}

}


void FannAnnWrapper::fillJacobianMatrix(DataWrapper* train_data, Mtx* jacobian_matrix){
	check_errors_allocated();

	struct fann_neuron *neuron_it, *first_neuron=fann_net->first_layer->first_neuron, *prev_neurons;
	struct fann_layer *current_layer, *first_layer=fann_net->first_layer, *after_last_layer=fann_net->last_layer;
	fann_type *error_begin=fann_net->train_errors,
			*last_layer_errors=error_begin+(((after_last_layer-1)->first_neuron)-first_neuron);

	int num_data=train_data->getExamplesAmount();
	int num_out=getOutputsAmount();
	for (int inp_iter = 0; inp_iter < num_data; inp_iter++) {

		// iterate by amount of outputs,
		// setting non-zero only current output
		for (int cur_out=0; cur_out<num_out; cur_out++){
			fann_reset_MSE(fann_net);
			memset(fann_net->train_errors, 0, (fann_net->total_neurons) * sizeof(fann_type));

			fann_run(fann_net, train_data->getInputByIndex(inp_iter));
			//setup error for last layer
			fann_type *error_it=last_layer_errors;

			neuron_it=((after_last_layer-1)->first_neuron)+cur_out;

			error_it[cur_out]=(-1)*fann_activation_derived(neuron_it->activation_function, neuron_it->activation_steepness,
											neuron_it->value, neuron_it->sum);

			//backprop gradient
			fann_backpropagate_MSE(fann_net);

			//extract weights deltas
			const fann_type learning_rate=1.0;
			for(current_layer = (first_layer + 1); current_layer != after_last_layer; current_layer++) {
				struct fann_neuron *last_neuron = current_layer->last_neuron;
				prev_neurons = (current_layer - 1)->first_neuron;

				for(neuron_it = current_layer->first_neuron; neuron_it != last_neuron; neuron_it++)	{
					fann_type base_error = error_begin[neuron_it - first_neuron] * learning_rate;
					int num_connections = neuron_it->last_con - neuron_it->first_con;
					for(int i = 0; i != num_connections; i++){
						fann_type dwi = base_error * prev_neurons[i].value;
						//gsl_matrix_float_set (J, inp_iter, neuron_it->first_con+i, dwi);
						jacobian_matrix->set(inp_iter*num_out + cur_out, neuron_it->first_con+i, dwi);
					}

				}
			}
		}

	}
}

void FannAnnWrapper::check_errors_allocated(){
	if(fann_net->train_errors == NULL){
		fann_net->train_errors = (fann_type *) calloc(fann_net->total_neurons, sizeof(fann_type));
		if(fann_net->train_errors == NULL){
			fann_error((struct fann_error *) fann_net, FANN_E_CANT_ALLOCATE_MEM);
			return;
		}
	} else {
		memset(fann_net->train_errors, 0, (fann_net->total_neurons) * sizeof(fann_type));
	}
}

double FannAnnWrapper::getErrorOnSet(DataWrapper* train_data){
	if (FannDataWrapper* fann_data_ptr=dynamic_cast<FannDataWrapper*>(train_data)){
		return fann_test_data(fann_net, fann_data_ptr->getInternalFannTrainData());
	} else {
		cout << "Non-FANN data not supported in FannAnnWrapper!" << endl;
		return -1e10;
	}
}

double FannAnnWrapper::getClassificationPrecisionOnSet(DataWrapper* train_data){
	int examples_total=train_data->getExamplesAmount();
	int Nclasses=getOutputsAmount();
	cout << "Outputs: "<<Nclasses<<endl;
	int examples_correct=0;
	for (int i=0; i<examples_total; i++){
		double* cur_input=train_data->getInputByIndex(i);
		double* desired_out=train_data->getDesiredOutputByIndex(i);
		double* net_out=fann_run(fann_net, cur_input);
		int desired_class= getClassFromVector(desired_out, Nclasses);
		int net_class=getClassFromVector(net_out, Nclasses);
		examples_correct += ( desired_class == net_class );
	}
	return (double)examples_correct/examples_total;
}


int FannAnnWrapper::getClassFromVector(double* outs, int Nclasses){
	double max=-1e10;
	int max_ind=-1;
	for (int i=0; i<Nclasses; i++){
		if (outs[i]>max){
			max_ind=i;
			max=outs[i];
		}
	}
	return max_ind;
}

FannAnnWrapper::~FannAnnWrapper(){
	delete[] connections;
	delete[] layer_sizes;
	delete[] layer_first_neurons;
}
