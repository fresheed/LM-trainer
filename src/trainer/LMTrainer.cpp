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

void LMTrainer::trainFann(struct fann* fann_net, struct fann_train_data* fann_data){
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
	while (! (isTrainCompleted()) ) {
		trainEpoch();
		cur_train_epoch++;
	}
	log("Training ended");
	delete backend;
}

void LMTrainer::trainEpoch(){
	log("    next epoch");

	Mtx* error_mtx=backend->getErrorMatrix();
	net->fillErrorMatrix(train_data, error_mtx);
	cout << "Errors: " << endl;
	error_mtx->print();

	Mtx* jacobian_mtx=backend->getJacobianMatrix();
	net->fillJacobianMatrix(train_data, jacobian_mtx);
	cout << "Jacobian: " << endl;

	adjustWeightsUntilSuccess();
}

void LMTrainer::adjustWeightsUntilSuccess(){
	log(" !!! weight adjustment not implemented yet !!!");

//	double error_before_adjust=net->getErrorOnSet(train_data);
//	double current_error=1e10; // just for first iteration
//	while (!isTrainCompleted()){
//		Mtx* delta_weights_to_test=backend->computeWeightsWithLambda(lambda);
//		net->addToWeights(delta_weights_to_test);
//		current_error=net->getErrorOnSet(train_data);
//		if (current_error > error_before_adjust){
//			lambda*=mu;
//			rollbackWeights(delta_weights_to_test);
//		} else {
//			lambda/=mu;
//			break;
//		}
//	}
}

void LMTrainer::rollbackWeights(Mtx* delta_weights){
//	backend->scaleMatrix(delta_weights, -1d);
//	net->addToWeights(delta_weights);
}


bool LMTrainer::isTrainCompleted(){
	if (cur_train_epoch>=max_train_epochs) {
		log("    stopping because max epochs reached");
		return true;
	}
	if (lambda >= lambda_max) {
		log("    stopping because lambda increased too much");
		return true;
	}
	return false;
}


void LMTrainer::initTrainParams(){
	log("Initializating LM and train params...");
	max_train_epochs=1;
	cur_train_epoch=0;
	lambda=1e-3;
	mu=10;
	lambda_max=1e6;
}

void LMTrainer::initBackend(){
	backend=new EigenBackend(net, train_data);
}



void LMTrainer::log(const char* msg){
	cout << "  "<< msg << endl;
}

LMTrainer::~LMTrainer() {

}

