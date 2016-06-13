/*
 * FannDataWrapper.cpp
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#include "FannDataWrapper.h"
#include <iostream>
using namespace std;

FannDataWrapper::FannDataWrapper(struct fann_train_data *td) : DataWrapper() {
	train_data=td;
}

int FannDataWrapper::getExamplesAmount(){
	return train_data->num_data;
}

double* FannDataWrapper::getInputByIndex(int index){
	return train_data->input[index];
}

double* FannDataWrapper::getDesiredOutputByIndex(int index){
	return train_data->output[index];
}

struct fann_train_data* FannDataWrapper::getInternalFannTrainData(){
	return train_data;
}

FannDataWrapper::~FannDataWrapper() {
}

