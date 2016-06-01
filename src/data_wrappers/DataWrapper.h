/*
 * DataWrapper.h
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_DATA_WRAPPERS_DATAWRAPPER_H_
#define SRC_DATA_WRAPPERS_DATAWRAPPER_H_

class DataWrapper {
public:
	DataWrapper(){};
	virtual int getExamplesAmount() =0;
	virtual double* getInputByIndex(int index) =0;
	virtual double* getDesiredOutputByIndex(int index) =0;
	virtual ~DataWrapper(){};
};

#endif /* SRC_DATA_WRAPPERS_DATAWRAPPER_H_ */
