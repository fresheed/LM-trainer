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
	int outputs=ann->getOutputsAmount(), examples=data->getExamplesAmount(), wgts=ann->getWeightAmount();

	error_matrix=new EigenMatrix( outputs * examples, 1);
	jacobian_matrix=new EigenMatrix(outputs * examples, wgts);
	weights=new EigenMatrix(wgts, 1);

	jT=new MatrixXd(wgts, outputs*examples);
	jTj=new MatrixXd(wgts, wgts);
	jT_by_err=new MatrixXd(wgts, 1);
}

Mtx* EigenBackend::getErrorMatrix(){
	return error_matrix;
}

Mtx* EigenBackend::getJacobianMatrix(){
	return jacobian_matrix;
}

void EigenBackend::initForEpoch(){
	*jT = jacobian_matrix->mtx->transpose();
	*jTj = (*jT) * (*(jacobian_matrix->mtx));
	*jT_by_err = (*jT) * (*(error_matrix->mtx));
//	*jT = *jTj = *jT_by_err = jacobian_matrix->mtx->transpose();
//	*jTj *= (*(jacobian_matrix->mtx));
//	*jT_by_err *= (*(error_matrix->mtx));
}

Mtx* EigenBackend::computeDWForLambda(double lambda){
//	// (JtJ + lI)dw=Jt*err
//	MatrixXd jac_trans=jacobian_matrix->mtx->transpose();
//	MatrixXd jTj= jac_trans * (*(jacobian_matrix->mtx));
//	MatrixXd eye=MatrixXd::Identity(jTj.rows(), jTj.cols());
//	MatrixXd left_part= (jTj + lambda*eye);
//
//	const double mult=1;
//	MatrixXd right_part= mult*jac_trans*(  *(error_matrix->mtx));
//
//	VectorXd ans = left_part.colPivHouseholderQr().solve(right_part);
//	//VectorXd ans = left_part.fullPivLu().solve(right_part);
//	weights->mtx->topRows(ans.rows())=ans.head(ans.rows());
//
//	return weights;

	MatrixXd left_part=(*jTj) + lambda*(MatrixXd::Identity(jTj->rows(), jTj->cols()));
	VectorXd ans = left_part.colPivHouseholderQr().solve(*jT_by_err);
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
	delete weights;

	delete jT;
	delete jTj;
	delete jT_by_err;
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


