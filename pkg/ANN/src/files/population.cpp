////////////////////////////////////////////////////////////////////
// population.cpp: Artificial Neural Network optimized by Genetic Algorithm 
//
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

#ifndef POPULATION_CPP
#define POPULATION_CPP

#include <cmath>
#include <algorithm>

//#include "stat.cpp"
#include "population.h"
#include "ANN.cpp"


POPULATION::POPULATION(){}

POPULATION::POPULATION(	int nbLayers, int * neuronPerLayer, 
			int nbData, double **dataIn,
			double **dataOut, int maxPop,
			double *wRange, double **tmatIn,double **tmatOut){

	init(nbLayers, neuronPerLayer, nbData, dataIn,
		dataOut, maxPop, wRange, tmatIn,tmatOut);

}


POPULATION::~POPULATION(){}


void POPULATION::copy(POPULATION& pP){

int i;
	#pragma omp parallel for private(i)
	for(i = 0 ; i < pP.getSize() ; i++){
		pop[i]->copyANN(pP.pop[i]);
	}
}


void POPULATION::init(	int nbLayers, int * neuronPerLayer, 
			int nbData, double **dataIn,
			double **dataOut, int maxPop,
			double *wRange, double **tmatIn,double **tmatOut){



	for(int i = 0 ; i < maxPop ; i++){
		obj = new ANN(nbLayers, neuronPerLayer,wRange);
		pop.push_back(obj);
	}

}


void	POPULATION::print() const{
	for(int i = 1 ; i < nbLayers  ; i++){

		for(int j = 0 ; j < neuronPerLayer[i] ; j++){

			for(int k = 0 ; k < neuronPerLayer[i-1]+1 ; k++){
				Rprintf ("%2.2f]",pop[0]->W[i][j][k]);
			}	
		}
	}
}


/*int	POPULATION::select(){
	return pop.size();
}*/
/*int POPULATION::select(){ //must change name
	//sumMSE;  must change name
	int N	=pop.size();
	double sumMSE=0;
	double threshold = unifRand(0.0, 1.0)*sumMSE;
	double sum = 0;
	double higherMSE = pop[N]->MSE();
	int i;
	for(i = 0 ; sum < threshold && i < N ; i++){
		sum += higherMSE-pop[i]->MSE();
	}
	Rprintf ("chosen %d", i-1);
	return (i - 1);
}*/

/*Individual* Population::select(double sumFitness)
{
	float rndm_fitness = 0;
	double partsum = 0;
	int i = 0;

	rndm_fitness = RNG::getInstance()->random() * sumFitness;

	do {
		partsum += genome[i]->fitness;
		i++;
	}
	while(partsum < rndm_fitness && i < maxSize);

	return &genome[i - 1];
}*/




void	POPULATION::reorder(){
	//sort(pop.begin(), pop.end(), sortPop);
}








#endif
