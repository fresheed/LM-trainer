/*
 * LMTrainer.cpp
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#include "LMTrainer.h"
#include "../ann_wrappers/FannAnnWrapper.hpp"
#include "../data_wrappers/FannDataWrapper.h"
#include "../matrix_wrappers/MatrixBackend.h"
#include "../matrix_wrappers/EigenBackend.h"
#include "../matrix_wrappers/Mtx.hpp"


using namespace std;
#include <iostream>
#include <string.h>

LMTrainer::LMTrainer() {

}

void LMTrainer::trainFann(neural_net* fann_net, training_data* fann_data){
	net=new FannAnnWrapper(fann_net);
	train_data=new FannDataWrapper(fann_data);

	// remove
	cout << "weights: "<<net->getWeightAmount()<<endl;
	cout << net->getNeuronsInLayer(0)<<" "<<net->getNeuronsInLayer(1)<<" "<<net->getNeuronsInLayer(2)<<endl;
	// remove

	trainNetOnData();


	delete net;
	delete train_data;
}

void LMTrainer::trainNetOnData(){
	log("Training started");
	initTrainParams();
	initBackend();
	while (! (isTrainTargetReached() || isForceStopNeeded()) ) {
		trainEpoch();
	}
	log("Training ended");
}

bool LMTrainer::isTrainTargetReached(){
	log("target not reached");
	return false;
}

bool LMTrainer::isForceStopNeeded(){
	log("stopping - DEBUG purposes");
	return true;
}

void LMTrainer::trainEpoch(){

}

void LMTrainer::initTrainParams(){
	log("SKIPPING initialization");
}

void LMTrainer::initBackend(){
	backend=new EigenBackend(net, train_data);
}



void LMTrainer::log(const char* msg){
	cout << "  "<< msg << endl;
}

LMTrainer::~LMTrainer() {

}

