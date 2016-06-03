/*
 * FannDataWrapper.h
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_DATA_WRAPPERS_FANNDATAWRAPPER_H_
#define SRC_DATA_WRAPPERS_FANNDATAWRAPPER_H_

#include <doublefann.h>

#include "DataWrapper.h"

class FannDataWrapper: public DataWrapper {
public:
	FannDataWrapper(struct fann_train_data* td);
	int getExamplesAmount();
	double* getInputByIndex(int index);
	double* getDesiredOutputByIndex(int index);
	struct fann_train_data* getInternalFannTrainData();
	~FannDataWrapper();
private:
	struct fann_train_data* train_data;
};

#endif /* SRC_DATA_WRAPPERS_FANNDATAWRAPPER_H_ */
