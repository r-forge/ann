////////////////////////////////////////////////////////////////////
// predict.cpp: Artificial Neural Network optimized by Genetic Algorithm 
// Copyright (C) 2011-2012 Francis Roy-Desrosiers
//
// This file is part of ANN.
//
// ANN is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3 of the License.
//
// ANN is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ANN.  If not, see <http://www.gnu.org/licenses/>.
////////////////////////////////////////////////////////////////////

#ifndef ANN_PREDICT_CPP
#define ANN_PREDICT_CPP

#include "fstream"
#include "iostream"

#include <R.h>
#include <Rmath.h>
#include <vector>



#include "predict.h"

//using namespace std;



//Constructor for ANNGA.default
//PREDICT::PREDICT(){}

PREDICT::PREDICT(int nbLayers_,int * neuronPerLayer_,int nbData_,double **dataIn_)
	:GA(){


	nbLayers		= nbLayers_;
	
	//double dminW = 0.0, dmaxW = 0.0;
	double wRange[2]={0.0,0.0};

	ann= new ANN(nbLayers, neuronPerLayer_,wRange);

	nbOfData		= nbData_;
	nbOfInput		= neuronPerLayer_[0];
	nbOfOutput		= neuronPerLayer_[nbLayers_-1];
	dataIn  		= dataIn_;

	outputANN = new double*[nbOfData];
	for(int i = 0 ; i < nbOfData ; i++)
		outputANN[i] = new double[nbOfOutput];
}



PREDICT::~PREDICT(){
	delete [] dataIn;
	delete [] dataOut;

	for(int i = 0 ; i < nbOfData ; i++)
		delete [] outputANN[i];
	delete [] outputANN;
	delete ann;
}


void	PREDICT::generation(bool print = false){
	

}//end for

void	PREDICT::load(double *chromosome){
	ann->loadW(chromosome);

}//end for

void PREDICT::getPredictResult(){

	for(int i = 0 ; i < nbOfData ; i++){  
		ann->feedForward (dataIn[i]);
		ann->getOutput(i);
		for(int j = 0 ; j < nbOfOutput ; j++){
			outputANN[i][j] = ann->getOutput(j);
		//Rprintf("[%d] %4.2f \n",i,outputANN[i][j]);
		}
	}
}







#endif
