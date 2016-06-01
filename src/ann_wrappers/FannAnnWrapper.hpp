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

class FannAnnWrapper : public AnnWrapper {
public:
	FannAnnWrapper(neural_net* net);
	double testDataSet();

	int getInputsAmount();
	int getOutputsAmount();
	int getLayersAmount();
	int getWeightAmount();
	int getNeuronsInLayer(int layer);

	double getWeightByIndex(int index);
	double getWeightInLayer(int layer, int index);

	~FannAnnWrapper();
private:
	neural_net* fann_net;
	connection* connections;
	unsigned int *layer_sizes, *layer_first_neurons;

	bool connections_actual;
	void updateConnections();
};


#endif /* SRC_ANN_WRAPPERS_FANNANNWRAPPER_HPP_ */
