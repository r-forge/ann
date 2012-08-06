////////////////////////////////////////////////////////////////////
// ann.cpp: Artificial Neural Network optimized by Genetic Algorithm 
//
// Copyright (C) 2009, 2010 Emre Caglar
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
#ifndef ANN_CPP
#define ANN_CPP
#include "ANN.h"
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <R.h>

using namespace std;
ANN::ANN(){
//Rprintf("ANN()\n");
}

ANN::ANN(int numOfLay, int *neuronPerLayer_, double *wRange_){
//Rprintf("ANN(.....)\n");
init(numOfLay,neuronPerLayer_, wRange_);
}


ANN::ANN(const ANN& a){
//Rprintf("ANN(ANN& a)\n");
	nbLayers	= a.nbLayers;
	wRange[0]=a.wRange[0];
	wRange[1]=a.wRange[1];

	neuronPerLayer=a.neuronPerLayer;
	mse=a.mse;


	X 		= new double*[nbLayers];
	for(int i = 0 ; i < nbLayers ; i++){
		X[i] = new double[neuronPerLayer[i]];
	}

	//allocate memory for weights
	W 	= new double **[nbLayers];
	for(int i = 1 ; i < nbLayers ; i++){  
		W[i] = new double*[neuronPerLayer[i]];
		for(int j = 0 ; j < neuronPerLayer[i] ; j++){
			W[i][j] = new double[neuronPerLayer[i-1]+1];
			for(int k = 0 ; k < neuronPerLayer[i-1]+1 ; k++){
				W[i][j][k] = a.W[i][j][k];  
			}
		}
	}
}

void ANN::init(int numOfLay, int *neuronPerLayer_, double *wRange_){

//Rprintf("ANN.init()\n");
	nbLayers	= numOfLay;
	wRange[0]=wRange_[0];
	wRange[1]=wRange_[1];
	neuronPerLayer=neuronPerLayer_;
	mse=0;

	X 		= new double*[nbLayers];
	for(int i = 0 ; i < nbLayers ; i++){
		//allocate memory for neuron values - represented as 2D matrix -
		X[i] = new double[neuronPerLayer[i]];
	}

	//allocate memory for weights
	W 	= new double **[nbLayers];
	for(int i = 1 ; i < nbLayers ; i++){  
		W[i] = new double*[neuronPerLayer[i]];
		for(int j = 0 ; j < neuronPerLayer[i] ; j++){
			W[i][j] = new double[neuronPerLayer[i-1]+1];
			for(int k = 0 ; k < neuronPerLayer[i-1]+1 ; k++){
				W[i][j][k] = unifRand(wRange[0], wRange[1]);   
			}
		}
	}
}
////////////////////////////////////////////
ANN::~ANN(){

	for (int i = 1; i < nbLayers; ++i) {
		for (int j = 0; j < neuronPerLayer[i]; ++j){
			delete [] W[i][j];
		}
		delete [] W[i];
	}
	delete [] W;


	for(int i  = 0 ; i < nbLayers ; ++i)
		delete[] X[i];
	delete[] X;

	delete[] neuronPerLayer;

}
////////////////////////////////////////////
void ANN::release (){

	for (int i = 1; i < nbLayers; ++i) {
		for (int j = 0; j < neuronPerLayer[i]; ++j){
			delete [] W[i][j];
		}
		delete [] W[i];
	}
	delete [] W;


	for(int i  = 0 ; i < nbLayers ; ++i)
		delete[] X[i];
	delete[] X;

	delete[] neuronPerLayer;

}

void ANN::copyANN(ANN*& a){

	for(int i = 1 ; i < nbLayers ; i++){  
		for(int j = 0 ; j < neuronPerLayer[i] ; j++){
			for(int k = 0 ; k < neuronPerLayer[i-1]+1 ; k++){
				W[i][j][k] =  a->W[i][j][k];  
			}
		}
	}
	mse=a->mse;
}

ANN& ANN::operator = (const ANN& a)
{
      if (this == &a)
            return * this;

	for(int i = 1 ; i < nbLayers ; i++){  
		for(int j = 0 ; j < neuronPerLayer[i] ; j++){
			for(int k = 0 ; k < neuronPerLayer[i-1]+1 ; k++){
				W[i][j][k] =  a.W[i][j][k];  
			}
		}
	}
	mse=a.mse;

      return * this;
}

////////////////////////////////////////////
void ANN::feedForward(double *input){

	double sum;
	//first set input layer with input pattern
	for(int i = 0 ; i < neuronPerLayer[0] ; i++){
		X[0][i] = input[i];
	}

	//then feed forward it
	//for each layer in neural network
	for(int i = 1 ; i < nbLayers ; i++){

		//for each neuron on a particular layer
		for(int j = 0 ; j < neuronPerLayer[i] ; j++){

			sum = 0.0; //reset sum
			for(int k = 0 ; k < neuronPerLayer[i-1] ; k++){
				sum += X[i-1][k] * W[i][j][k];
			}
			//add bias
			sum += W[i][j][neuronPerLayer[i-1]];

			X[i][j] = sigmoid(sum);
			
		}
	}

}




////////////////////////////////////////////
double ANN::getOutput(int index){
	
	return X[nbLayers-1][index];
}
////////////////////////////////////////////
double ANN::getMeanSquareError(double* target){

		double mse=0;
		for( int i = 0 ; i < neuronPerLayer[nbLayers-1] ; i++){
			mse = mse + (target[i]-X[nbLayers-1][i])*(target[i]-X[nbLayers-1][i]);
		}
		
		return mse;
	
}
////////////////////////////////////////////

void ANN::loadW(double *weightVector){

	for(int layer = 1; layer < nbLayers ; ++layer){ 
		for(int neuron = 0 ; neuron < neuronPerLayer[layer] ; neuron++){
			for(int preNeuron = 0 ; preNeuron < neuronPerLayer[layer-1] ; preNeuron++){
				W[layer][neuron][preNeuron] = *(weightVector++); 
			}
			//load bias
			W[layer][neuron][neuronPerLayer[layer-1]] = *(weightVector++);
		}
	}
}
////////////////////////////////////////////

void ANN::print(){
Rprintf("Best chromosome->");
	for(int i = 1 ; i < nbLayers ; i++){  
		for(int j = 0 ; j < neuronPerLayer[i] ; j++){
			for(int k = 0 ; k < neuronPerLayer[i-1]+1 ; k++){
			Rprintf("%4.2f/",W[i][j][k]);
			}
		}	
	}
	Rprintf("\n");


}











#endif

