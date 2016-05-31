/*
 * AnnWrapper.hpp
 *
 *  Created on: 30 мая 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_ANN_WRAPPERS_ANNWRAPPER_HPP_
#define SRC_ANN_WRAPPERS_ANNWRAPPER_HPP_

class AnnWrapper {
public:
	virtual double testDataSet() =0;
	virtual ~AnnWrapper(){};
protected:
	AnnWrapper(){};

};



#endif /* SRC_ANN_WRAPPERS_ANNWRAPPER_HPP_ */
