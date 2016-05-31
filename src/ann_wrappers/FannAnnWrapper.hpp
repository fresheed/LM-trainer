/*
 * FannAnnWrapper.hpp
 *
 *  Created on: 30 мая 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_ANN_WRAPPERS_FANNANNWRAPPER_HPP_
#define SRC_ANN_WRAPPERS_FANNANNWRAPPER_HPP_
#include "AnnWrapper.hpp"

class FannAnnWrapper : public AnnWrapper {
public:
	FannAnnWrapper();
	double testDataSet();
	~FannAnnWrapper();
};


#endif /* SRC_ANN_WRAPPERS_FANNANNWRAPPER_HPP_ */
