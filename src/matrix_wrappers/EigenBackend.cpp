/*
 * EigenBackend.cpp
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#include "EigenBackend.h"
#include "MatrixBackend.h"

#include <iostream>
using namespace std;

EigenBackend::EigenBackend(AnnWrapper* ann, DataWrapper *data) : MatrixBackend(ann, data) {
	error_matrix=new EigenMatrix( ann->getOutputsAmount() * data->getExamplesAmount() , 1);
	jacobian_matrix=new EigenMatrix(ann->getOutputsAmount() * data->getExamplesAmount(), ann->getWeightAmount());
}

Mtx* EigenBackend::getErrorMatrix(){
	return error_matrix;
}

Mtx* EigenBackend::getJacobianMatrix(){
	return error_matrix;
}


EigenBackend::~EigenBackend() {
	delete error_matrix;
	delete jacobian_matrix;
}


//
// Eigen matrix functions
//
EigenBackend::EigenMatrix::EigenMatrix(int rows, int columns) : Mtx(rows, columns){

}

EigenBackend::EigenMatrix::~EigenMatrix(){

}

void EigenBackend::EigenMatrix::describe(){
	cout << "Eigen-based matrix" << endl;
}

