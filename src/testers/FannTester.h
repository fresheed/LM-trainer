/*
 * FannTester.h
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_TESTERS_FANNTESTER_H_
#define SRC_TESTERS_FANNTESTER_H_

#include "../ann_wrappers/FannAnnWrapper.hpp"

class FannTester {
public:
	//FannTester();
	//~FannTester();
	void testWithFannNet();
private:
	struct fann* getInitializedFannNet();
	struct fann_train_data* getFannData();

};



#endif /* SRC_TESTERS_FANNTESTER_H_ */
