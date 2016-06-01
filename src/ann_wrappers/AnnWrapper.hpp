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
	virtual int getInputsAmount() =0;
	virtual int getOutputsAmount() =0;
	virtual int getLayersAmount() =0;
	virtual int getWeightAmount() =0;
	virtual int getNeuronsInLayer(int layer) =0;

	virtual double getWeightByIndex(int index) =0;
	virtual double getWeightInLayer(int layer, int index) =0;


	virtual ~AnnWrapper(){};
protected:
	AnnWrapper(){};

};



#endif /* SRC_ANN_WRAPPERS_ANNWRAPPER_HPP_ */
