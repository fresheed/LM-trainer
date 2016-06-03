/*
 * LMTrainer.h
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_TRAINER_LMTRAINER_H_
#define SRC_TRAINER_LMTRAINER_H_

#include <doublefann.h>
#include "../ann_wrappers/AnnWrapper.hpp"
#include "../data_wrappers/DataWrapper.h"
#include "../matrix_wrappers/MatrixBackend.h"
#include "../ann_wrappers/FannAnnWrapper.hpp"

class LMTrainer {
public:
	LMTrainer();
	void trainFann(struct fann* net, struct fann_train_data* data);
	~LMTrainer();
private:
	void trainNetOnData();
	AnnWrapper* net;
	DataWrapper* train_data;
	MatrixBackend* backend;

	//train params
	double lambda, lambda_max, mu;
	double max_train_epochs, cur_train_epoch;

	void initTrainParams();
	void initBackend();

	void trainEpoch();
	void adjustWeightsUntilSuccess();
	void rollbackWeights(Mtx* delta_weights);

	bool isTrainCompleted();
	void log(const char* msg); // to preserve indentation
};

#endif /* SRC_TRAINER_LMTRAINER_H_ */
