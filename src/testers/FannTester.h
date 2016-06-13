/*
 * FannTester.h
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_TESTERS_FANNTESTER_H_
#define SRC_TESTERS_FANNTESTER_H_

#include "../ann_wrappers/FannAnnWrapper.h"

class FannTester {
public:
	void testWithFannNet();
private:
	struct fann* getInitializedFannNet();
	struct fann_train_data* getFannData();

	struct fann* get4ClassesFannNet();
	struct fann_train_data* get4ClassesFannData();

	struct fann* get5ClassesFannNet();
	struct fann_train_data* get5ClassesFannData();

	struct fann* get3ClassesFannNet();
	struct fann_train_data* get3ClassesFannData();

	struct fann* get7ClassesFannNet();
	struct fann_train_data* get7ClassesFannData();

};



#endif /* SRC_TESTERS_FANNTESTER_H_ */
