////////////////////////////////////////////////////////////////////
// ga.cpp: Artificial Neural Network optimized by Genetic Algorithm 
// Based on CUDAANN r6 project
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

#ifndef GA_CPP
#define GA_CPP

#include "fstream"
#include "iostream"

#include <R.h>
#include <Rmath.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "globalFunction.cpp"
#include "ANN.cpp"
#include "population.cpp"
#include "stat.cpp"
#include "ga.h"



using namespace std;




GA::GA(){}
//Constructor for ANNGA.default
GA::GA(int nbLayers,int * neuronPerLayer,int lengthData,double **tmatIn,double **tmatOut,int iMaxPopulation,double dmutRate,double dcrossRate,double *wRange,  bool rprintBestChromosome, int passThreads, unsigned int seed){

initGA(nbLayers, neuronPerLayer, lengthData, tmatIn, tmatOut, iMaxPopulation, dmutRate, dcrossRate, wRange, rprintBestChromosome, passThreads, seed);

}


void GA::initGA(int nbLayers_,int * neuronPerLayer_,int lengthData,double **tmatIn,double **tmatOut,int iMaxPopulation,double dmutRate,double dcrossRate,double *_wRange,  bool rprintBestChromosome, int passThreads, unsigned int seed){
			srand (seed); //SET SEED
			maxPopulation		= iMaxPopulation;
			nbLayers		= nbLayers_;  //// a changer


			mutRate			=dmutRate;
			crossRate		=dcrossRate;	
			wRange			=_wRange;

			printBestChromosome 	=rprintBestChromosome;
			neuronPerLayer=neuronPerLayer_; // pas besoin ...

			nbOfData		= lengthData;
			nbOfInput		= neuronPerLayer[0];
			nbOfOutput		= neuronPerLayer[nbLayers-1];

			dataIn 			= tmatIn;
			dataOut 		= tmatOut;
			outputANN 		= new double*[nbOfData];

			for(int i = 0 ; i < nbOfData ; i++)
				outputANN[i] = new double[nbOfOutput];
			
			num_of_threads = passThreads;
			//num_of_threads = 2;
			#ifdef _OPENMP
			omp_set_num_threads(num_of_threads);
			Rprintf("* Number of thread [%d] \n",num_of_threads );
			#endif
			


			 mWeightConNum = 0;			
			 for (int i = 1 ; i < nbLayers ; i++)
				 mWeightConNum +=neuronPerLayer[i] * neuronPerLayer[i-1] + neuronPerLayer[i];
			 


			nn1.init(nbLayers, neuronPerLayer,wRange);
			nn2.init(nbLayers, neuronPerLayer,wRange);

	 
			p.init(nbLayers, neuronPerLayer, 
			lengthData, tmatIn, tmatOut, maxPopulation, wRange,tmatIn, tmatOut);
			p2.init(nbLayers, neuronPerLayer, 
			lengthData, tmatIn, tmatOut, maxPopulation, wRange,tmatIn, tmatOut);

			for(int i = 0 ; i < maxPopulation ; i++){
			p.pop[i]->MSE(getFitness (p.pop[i]));		
			}
			s.stat(p);

			vectorFitness.clear();	
}


////////////////////////////////////////////
GA::~GA(){}
////////////////////////////////////////////
////////////////////////////////////////////


				    
double		GA::getFitness(ANN& individual){

	double sumError = 0;
	for(int i = 0 ; i < nbOfData ; i++){ //for each training sample
		individual.feedForward (dataIn[i]);
		sumError += individual.getMeanSquareError (dataOut[i]);
	}	
	return sumError/(double)nbOfData;	
}

double		GA::getFitness(ANN* individual){

	double sumError = 0;
	for(int i = 0 ; i < nbOfData ; i++){ //for each training sample
		individual->feedForward (dataIn[i]);
		sumError += individual->getMeanSquareError (dataOut[i]);
	}	
	return sumError/(double)nbOfData;	
}
////////////////////////////////////////////
void GA::getANNresult(bool loadBest=true){


	for(int i = 0 ; i < nbOfData ; i++){  
		p.pop[0]->feedForward (dataIn[i]);
		p.pop[0]->getOutput(i);
		for(int j = 0 ; j < nbOfOutput ; j++){
			outputANN[i][j] = p.pop[0]->getOutput(j);
		}
	}
}




////////////////////////////////////////////

////////////////////////////////////////////
void		GA::release (){	

//	for (int i = 0; i < maxPopulation; ++i)
//		delete [] nn[i];
//	delete [] nn;

	//delete [] mFitnessValues;
	//delete [] mNeuronNum; 
	//delete [] diff;
	//delete [] crossedTrialIndividual;
	delete [] dataIn;
	delete [] dataOut;

	for(int i = 0 ; i < nbOfData ; i++)
		delete [] outputANN[i];
	delete [] outputANN;

	//ann[]->release ();
}
void		GA::releasePredict (){	


/*	delete [] dataIn;
	delete [] dataOut;

	for(int i = 0 ; i < nbOfData ; i++)
		delete [] outputANN[i];
	delete [] outputANN;*/

	//delete ann;
}
////////////////////////////////////////////




#endif
