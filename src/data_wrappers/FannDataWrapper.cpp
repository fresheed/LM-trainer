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
	//cout << "train:" << train_data->get_train_input(index)[0] << train_data->get_train_input(index)[1]<<endl;
	return fann_get_train_input(train_data, index);
}

double* FannDataWrapper::getDesiredOutputByIndex(int index){
	return fann_get_train_output(train_data, index);
}

struct fann_train_data* FannDataWrapper::getInternalFannTrainData(){
	return train_data;
}

FannDataWrapper::~FannDataWrapper() {
	// TODO Auto-generated destructor stub
}

