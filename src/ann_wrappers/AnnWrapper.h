/*
 * AnnWrapper.h
 *
 *  Created on: 30 мая 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_ANN_WRAPPERS_ANNWRAPPER_H_
#define SRC_ANN_WRAPPERS_ANNWRAPPER_H_

#include "../matrix_wrappers/Mtx.h"
#include "../data_wrappers/DataWrapper.h"

class AnnWrapper {
public:
	virtual int getInputsAmount() =0;
	virtual int getOutputsAmount() =0;
	virtual int getLayersAmount() =0;
	virtual int getWeightAmount() =0;
	virtual int getNeuronsInLayer(int layer) =0;

//	virtual double getWeightByIndex(int index) =0;
//	virtual double getWeightInLayer(int layer, int index) =0;
	virtual void addToWeights(Mtx* delta_weights) =0;
	virtual void printWeights(){};

	virtual void fillErrorMatrix(DataWrapper* train_data, Mtx* error_matrix) =0;
	virtual void fillJacobianMatrix(DataWrapper* train_data, Mtx* jacobian_matrix) =0;

	virtual double getErrorOnSet(DataWrapper* train_data) =0;
	virtual double getClassificationPrecisionOnSet(DataWrapper* train_data) =0;

	virtual ~AnnWrapper(){};
protected:
	AnnWrapper(){};

};



#endif /* SRC_ANN_WRAPPERS_ANNWRAPPER_H_ */
