#include "FannAnnWrapper.hpp"

using namespace FANN;

using namespace std;
#include <iostream>

FannAnnWrapper::FannAnnWrapper(neural_net* net){
	fann_net=net;
	connections=new connection[getWeightAmount()];
	connections_actual=false;
	layer_sizes=new unsigned int[getLayersAmount()];
	fann_net->get_layer_array(layer_sizes);

	layer_first_neurons=new unsigned int[getLayersAmount()];
	int cumsum=0;
	for (int i=0; i<getLayersAmount(); i++){
		layer_first_neurons[i]=cumsum;
		cumsum+=getNeuronsInLayer(i);
	}
}

int FannAnnWrapper::getInputsAmount(){
	return fann_net->get_num_input();
}

int FannAnnWrapper::getOutputsAmount(){
	return fann_net->get_num_output();
}

int FannAnnWrapper::getLayersAmount(){
	return fann_net->get_num_layers();
}

int FannAnnWrapper::getNeuronsInLayer(int layer){
	int neurons=layer_sizes[layer];
	if (layer != getLayersAmount()-1)
		neurons++; //layer sizes does not include bias
	return neurons;

}

int FannAnnWrapper::getWeightAmount(){
	int num_conns=fann_net->get_total_connections();

	return num_conns;
}

void FannAnnWrapper::updateConnections(){
	if (!connections_actual){
		fann_net->get_connection_array(connections);
		connections_actual=true;
	}
}

double FannAnnWrapper::getWeightByIndex(int index){
	updateConnections();
	return connections[index].weight;
}

double FannAnnWrapper::getWeightInLayer(int layer, int index_in_layer){
	unsigned int neuron_index=layer_first_neurons[layer];
	return getWeightByIndex(neuron_index+index_in_layer);
}

FannAnnWrapper::~FannAnnWrapper(){
	// coed
	delete connections;
	delete layer_sizes;
	delete layer_first_neurons;
}
