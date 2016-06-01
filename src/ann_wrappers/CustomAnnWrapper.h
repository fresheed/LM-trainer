/*
 * CustomAnnWrapper.h
 *
 *  Created on: 31 мая 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_ANN_WRAPPERS_CUSTOMANNWRAPPER_H_
#define SRC_ANN_WRAPPERS_CUSTOMANNWRAPPER_H_

#include "AnnWrapper.hpp"

class CustomAnnWrapper: public AnnWrapper {
public:
	CustomAnnWrapper();
	double testDataSet();
	~CustomAnnWrapper();
};

#endif /* SRC_ANN_WRAPPERS_CUSTOMANNWRAPPER_H_ */
