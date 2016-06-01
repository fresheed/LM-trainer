/*
 * FannDataWrapper.cpp
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#include "FannDataWrapper.h"

FannDataWrapper::FannDataWrapper(training_data *td) : DataWrapper() {
	train_data=td;
}

int FannDataWrapper::getExamplesAmount(){
	return train_data->length_train_data();
}

double* FannDataWrapper::getInputByIndex(int index){
	return train_data->get_train_input(index);
}

double* FannDataWrapper::getDesiredOutputByIndex(int index){
	return train_data->get_train_output(index);
}

FannDataWrapper::~FannDataWrapper() {
	// TODO Auto-generated destructor stub
}

