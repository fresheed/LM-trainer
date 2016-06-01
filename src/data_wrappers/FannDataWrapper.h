/*
 * FannDataWrapper.h
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_DATA_WRAPPERS_FANNDATAWRAPPER_H_
#define SRC_DATA_WRAPPERS_FANNDATAWRAPPER_H_

#include <doublefann.h>
#include <fann_cpp.h>
using namespace FANN;

#include "DataWrapper.h"

class FannDataWrapper: public DataWrapper {
public:
	FannDataWrapper(training_data* td);
	int getExamplesAmount();
	double* getInputByIndex(int index);
	double* getDesiredOutputByIndex(int index);
	~FannDataWrapper();
private:
	training_data* train_data;
};

#endif /* SRC_DATA_WRAPPERS_FANNDATAWRAPPER_H_ */
