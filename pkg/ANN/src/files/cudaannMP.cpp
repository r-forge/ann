////////////////////////////////////////////////////////////////////
// CUDAANNMP.cpp: Artificial Neural Network optimized by Genetic Algorithm 
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

#ifndef CUDAANNMP_CPP
#define CUDAANNMP_CPP

#include "fstream"
#include "iostream"

#include <R.h>
#include <Rmath.h>
#include <vector>



#include "cudaannMP.h"

//using namespace std;



//Constructor for ANNGA.default
CUDAANNMP::CUDAANNMP(){}

CUDAANNMP::CUDAANNMP(int nbLayers_ , int * neuronPerLayer_ ,int nbData_, double **dataIn_, 
	double **dataOut_,  int maxPop_, double mutRate_, double crossRate_,
	double *wRange_,bool printBestChromosome_, int nbThreads_, 
	unsigned int seed)
	:GA(){

initGA(nbLayers_, neuronPerLayer_, nbData_, dataIn_, 
	 dataOut_, maxPop_, mutRate_, crossRate_, 
	 wRange_, printBestChromosome_, nbThreads_, 	
	 seed);

init(   nbLayers_,neuronPerLayer_,maxPop_, 
	mutRate_, crossRate_, nbData_, dataIn_,
	dataOut_, wRange_ );


}
void	CUDAANNMP::init(	int nbLayers_, int * neuronPerLayer_, int maxPop_, 
		double mutRate_, double crossRate_,int nbData_, double **dataIn_,
		double **dataOut_,double *wRange_ ){
	

		//vectorFitness.clear();			
}



CUDAANNMP::~CUDAANNMP(){}


void	CUDAANNMP::generation(bool print = false){
	
	p2.copy(p); 

	int r1,r2,r3;
		double ran;
double nnMSE;
int i,j,k;
int v;
#pragma omp parallel for private(v,r1,r2,r3,ran,nnMSE,i,j,k)
for( v = 0 ; v < maxPopulation ; v++){

	do{
		r1 = unifRandInt (0,maxPopulation-1);
		r2 = unifRandInt (0,maxPopulation-1);
		r3 = unifRandInt (0,maxPopulation-1);
	}while(r1 == r2 || r1 == r3 || r2 == r3 || r1 == v || r2 == v || r3 == v);



	for( i = 1 ; i < nbLayers ; i++){  
	  for( j = 0 ; j < neuronPerLayer[i] ; j++){
	    for( k = 0 ; k < neuronPerLayer[i-1]+1 ; k++){
		ran =	unifRand();
		if(ran < crossRate){
			p.pop[v]->W[i][j][k] = p2.pop[r1]->W[i][j][k] + (mutRate * (p2.pop[r3]->W[i][j][k] - p2.pop[r2]->W[i][j][k]));
		}else{
			p.pop[v]->W[i][j][k] = p2.pop[v]->W[i][j][k];				
		}
	    }
	  }	
	}



	p.pop[v]->MSE( getFitness (p.pop[v]) );
	if(p.pop[v]->MSE() >  p2.pop[v]->MSE() ){
	for( i = 1 ; i < nbLayers ; i++){  
	  for( j = 0 ; j < neuronPerLayer[i] ; j++){
	    for( k = 0 ; k < neuronPerLayer[i-1]+1 ; k++){
		p.pop[v]->W[i][j][k] = p2.pop[v]->W[i][j][k];
	    }
	  }	
	}		
	p.pop[v]->MSE(p2.pop[v]->MSE());

	
	}//if(fitnessTrialIndividual <  p.pop[v]->MSE() )



	}//End for(int i = 0 ; i < maxPopulation ; i++)

	s.stat(p);
	if(print==true){
		if(printBestChromosome==true){ 
			p.pop[0]->print();
			//nn[0].print();	
		}
	}

			

	vectorFitness.push_back (p.pop[0]->MSE());
}//end for









#endif
