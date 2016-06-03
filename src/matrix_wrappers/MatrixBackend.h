/*
 * MatrixBackend.h
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_MATRIX_WRAPPERS_MATRIXBACKEND_H_
#define SRC_MATRIX_WRAPPERS_MATRIXBACKEND_H_

#include "Mtx.hpp"
#include "../ann_wrappers/AnnWrapper.hpp"
#include "../data_wrappers/DataWrapper.h"

class MatrixBackend {
public:
	MatrixBackend(AnnWrapper* ann, DataWrapper* data){};
	virtual Mtx* getErrorMatrix() =0;
	virtual Mtx* getJacobianMatrix() =0;
	virtual Mtx* computeDWForLambda(double lambda) =0;
	virtual ~MatrixBackend(){};
};

#endif /* SRC_MATRIX_WRAPPERS_MATRIXBACKEND_H_ */
