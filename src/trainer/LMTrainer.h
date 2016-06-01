/*
 * LMTrainer.h
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_TRAINER_LMTRAINER_H_
#define SRC_TRAINER_LMTRAINER_H_

#include <doublefann.h>
#include <fann_cpp.h>
#include "../ann_wrappers/AnnWrapper.hpp"
#include "../data_wrappers/DataWrapper.h"
#include "../matrix_wrappers/MatrixBackend.h"

using namespace FANN;

class LMTrainer {
public:
	LMTrainer();
	void trainFann(neural_net* net, training_data* data);
	~LMTrainer();
private:
	void trainNetOnData();
	AnnWrapper* net;
	DataWrapper* train_data;
	MatrixBackend* backend;

	//train params
	double lambda, lambda_max, lambda_min, mu;
	double max_train_epochs, cur_train_epoch;

	void initTrainParams();
	void initBackend();

	bool isTrainTargetReached();
	bool isForceStopNeeded();

	void trainEpoch();

	void log(const char* msg); // to preserve indentation
};

#endif /* SRC_TRAINER_LMTRAINER_H_ */
