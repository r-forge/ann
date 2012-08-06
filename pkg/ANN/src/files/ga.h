////////////////////////////////////////////////////////////////////
// ga.h: Artificial Neural Network optimized by Genetic Algorithm 
// Based on CUDAANN project
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

#ifndef GA_H
#define GA_H



#include <vector>
#include <algorithm>

/*inline bool	sortPop(const ANN* ann1,const ANN* ann2){
	return ( (ann1->MSE())  <  (ann2->MSE()) );
}*/
//sort(nn.begin(), nn.end(), sortPop);






class GA{

public:	
	GA();

	GA(int nbLayers , int * neuronPerLayer_ ,int lengthData, double **tmatIn, 
	double **tmatOut,  int iMaxPopulation, double dmutRate, double dcrossRate, 
	double *wRange,bool rprintBestChromosome, int passThreads, unsigned int seed); 	
	
	// to predict from a trained the network
	void initPredictGA(int nbLayers , int * neuronPerLayer_ ,int lengthData, double **tmatIn); 
	
	void	initGA(int nbLayers_ , int *neuronPerLayer_ ,int lengthData, double **tmatIn,
	double **tmatOut,  int iMaxPopulation, double dmutRate, double dcrossRate,
	double *wRange,bool rprintBestChromosome, int passThreads, unsigned int seed); 

	virtual ~GA();

	bool		printBestChromosome;

	int 		num_of_threads;	
	int 		maxPopulation;
	int		nbOfData;
	int		nbOfInput;
	int		nbOfOutput;
	int*		neuronPerLayer;

	double 		mutRate;
	double 		crossRate;
	double*		wRange;	
	double**	dataIn;
	double**	dataOut;
	double**	outputANN;
	
	virtual void	generation(bool)=0;	
	void		release();
	void		releasePredict ();
	void		getANNresult(bool);
	void		getPredictResult();

	double		getFitness(ANN& individual);
	double		getFitness(ANN* individual);


	ANN nn1;
	ANN nn2; 

	vector<double> vectorFitness;

	STAT s;
	POPULATION p;				//p is the population
	POPULATION p2;				//p2 is a copy for multi thread purpose

	int		nbLayers;
	int		mWeightConNum;    	//number of weight connections between neurons


	inline double bestMSE(){
	return p.pop[0]->MSE();
	}

};




#endif
