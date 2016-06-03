/*
 * FannTester.h
 *
 *  Created on: 1 июня 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_TESTERS_FANNTESTER_H_
#define SRC_TESTERS_FANNTESTER_H_

using namespace FANN;
#include "../ann_wrappers/FannAnnWrapper.hpp"

class FannTester {
public:
	//FannTester();
	//~FannTester();
	void testWithFannNet();
private:
	FannHacker* getInitializedFannNet();
	FANN::training_data* getFannData();

};



#endif /* SRC_TESTERS_FANNTESTER_H_ */
