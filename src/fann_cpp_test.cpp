#include <doublefann.h>

#include <iostream>
#include "ann_wrappers/CustomAnnWrapper.h"
#include "ann_wrappers/FannAnnWrapper.h"
#include "testers/FannTester.h"

using namespace std;


int main(){
	//main2();
	FannTester fann_tester;
	fann_tester.testWithFannNet();
	return 0;
}


