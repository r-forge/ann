////////////////////////////////////////////////////////////////////
// CUDAANNMP.h: Artificial Neural Network optimized by Genetic Algorithm 
// Based on CUDAANNMP r6 project
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

#ifndef CUDAANNMP_H
#define CUDAANNMP_H


#include "ANN.cpp"
#include "population.cpp"
#include <vector>




class CUDAANNMP : public GA{



public:	// to train the network
	CUDAANNMP();
	CUDAANNMP(int nbLayers_ , int * neuronPerLayer_ ,int nbData_, double **dataIn_, 
	double **dataOut_,  int maxPop_, double mutRate_, double crossRate_,
	double *wRange_,bool printBestChromosome_, int nbThreads_, 
	unsigned int seed);
	
	//virtual 
	virtual ~CUDAANNMP();
	
	void	init(	int nbLayers_, int * neuronPerLayer_, int maxPop_, 
		double mutRate_, double crossRate_,int nbData_, double **dataIn_,
		double **dataOut_,double *wRange_ );

	void	generation(bool);


private:



};




#endif
