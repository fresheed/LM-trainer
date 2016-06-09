/*
 * Matrix.h
 *
 *  Created on: 31 мая 2016 г.
 *      Author: fresheed
 */

#ifndef SRC_MATRIX_WRAPPERS_MTX_H_
#define SRC_MATRIX_WRAPPERS_MTX_H_

class Mtx {
public:
//	virtual double getElement(int row, int column) =0;
//	virtual double setElement(int row, int column, double value) =0;
//	virtual void matrixSum(Mtx &);
//	virtual void matrixMult(Mtx &mult);
//	virtual void scalarMult(double factor);
	virtual void set(int row, int column, double value) =0;
	virtual double get(int row, int column) =0;
	virtual void scale(double factor) =0;
	virtual void print() =0;
protected:
	Mtx(int rows, int columns){};
	virtual void describe() =0;
	virtual ~Mtx(){};
};



#endif /* SRC_MATRIX_WRAPPERS_MTX_H_ */
