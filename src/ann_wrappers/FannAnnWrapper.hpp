/*
 * FannAnnWrapper.hpp
 *
 *  Created on: 30 мая 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_ANN_WRAPPERS_FANNANNWRAPPER_HPP_
#define SRC_ANN_WRAPPERS_FANNANNWRAPPER_HPP_
#include "AnnWrapper.hpp"

#include <doublefann.h>
#include <fann_cpp.h>
using namespace FANN;

class FannHacker : public neural_net {
public:
	struct fann* getInternalAnnPointer(){ return this->ann; };
};

class FannAnnWrapper : public AnnWrapper {
public:
	FannAnnWrapper(FannHacker* net);

	int getInputsAmount();
	int getOutputsAmount();
	int getLayersAmount();
	int getWeightAmount();
	int getNeuronsInLayer(int layer);

	double getWeightByIndex(int index);
	double getWeightInLayer(int layer, int index);
	virtual void printWeights();

	double getErrorOnSet(DataWrapper* train_data);

	void fillErrorMatrix(DataWrapper* train_data, Mtx* error_matrix);
	void fillJacobianMatrix(DataWrapper* train_data, Mtx* jacobian_matrix);

	~FannAnnWrapper();

private:
	FannHacker* fann_net;
	connection* connections;
	unsigned int *layer_sizes, *layer_first_neurons;


	bool connections_actual;
	void updateConnections();
};





#endif /* SRC_ANN_WRAPPERS_FANNANNWRAPPER_HPP_ */
