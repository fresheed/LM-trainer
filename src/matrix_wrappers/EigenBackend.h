/*
 * EigenBackend.h
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_MATRIX_WRAPPERS_EIGENBACKEND_H_
#define SRC_MATRIX_WRAPPERS_EIGENBACKEND_H_

#include "MatrixBackend.h"
#include "../eigen/Eigen/Dense"
#include "Mtx.hpp"
using namespace Eigen;


class EigenBackend: public MatrixBackend {
public:
	EigenBackend(AnnWrapper* ann, DataWrapper* data);
	Mtx* getErrorMatrix();
	Mtx* getJacobianMatrix();
	~EigenBackend();
private:
	class EigenMatrix : public Mtx {
		public:
			EigenMatrix(int rows, int columns);
			void describe();
			~EigenMatrix();
			MatrixXd *mtx;
	};

	EigenMatrix* error_matrix;
	EigenMatrix* jacobian_matrix;
};

#endif /* SRC_MATRIX_WRAPPERS_EIGENBACKEND_H_ */
