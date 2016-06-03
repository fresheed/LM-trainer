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
	weights=new EigenMatrix(ann->getWeightAmount(), 1);
}

Mtx* EigenBackend::getErrorMatrix(){
	return error_matrix;
}

Mtx* EigenBackend::getJacobianMatrix(){
	return jacobian_matrix;
}


Mtx* EigenBackend::computeDWForLambda(double lambda){
	// (JtJ + lI)dw=-Jt*err
	MatrixXd jac_trans=jacobian_matrix->mtx->transpose();
	MatrixXd jTj= jac_trans * (*(jacobian_matrix->mtx));
	MatrixXd eye=MatrixXd::Identity(jTj.rows(), jTj.cols());
	MatrixXd left_part= (jTj + lambda*eye);

	const double mult=1;
	MatrixXd right_part= mult*jac_trans*(  *(error_matrix->mtx));

	//VectorXd ans = left_part.colPivHouseholderQr().solve(right_part);
	VectorXd ans = left_part.fullPivLu().solve(right_part);
	weights->mtx->topRows(ans.rows())=ans.head(ans.rows());

	return weights;
}

double EigenBackend::computeMseForErrors(){
	MatrixXd errors=(*(error_matrix->mtx));
	MatrixXd sum_squares=errors.transpose() * errors; // actually 1x1
	// to comply with octave
	double factor=2.0;
	return sum_squares(0,0)/(factor * errors.rows() );
}

EigenBackend::~EigenBackend() {
	delete error_matrix;
	delete jacobian_matrix;
}


//
// Eigen matrix functions
//
EigenBackend::EigenMatrix::EigenMatrix(int rows, int columns) : Mtx(rows, columns){
	//mtx=new MatrixXd(3, 3);
	mtx=new MatrixXd(rows, columns);
}

void EigenBackend::EigenMatrix::set(int row, int column, double value){
	(*mtx)(row, column)=value;
}

double EigenBackend::EigenMatrix::get(int row, int column){
	return (*mtx)(row, column);
}

void EigenBackend::EigenMatrix::scale(double factor){
	(*mtx)*=factor;
}

EigenBackend::EigenMatrix::~EigenMatrix(){
	delete mtx;
}

void EigenBackend::EigenMatrix::print(){
	cout << *mtx << endl;
}

void EigenBackend::EigenMatrix::describe(){
	cout << "Eigen-based matrix" << endl;
}


