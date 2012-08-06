////////////////////////////////////////////////////////////////////
// population.h: Artificial Neural Network optimized by Genetic Algorithm 
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
#ifndef POPULATION_H
#define POPULATION_H
#include <vector>
//#include "stat.cpp"


using namespace std;




inline bool	sortPop(ANN* ann1,ANN* ann2){
	return ( (ann1->MSE())  <  (ann2->MSE()) );
}


class POPULATION{



public:

	POPULATION();
	//POPULATION(const POPULATION* p);
	POPULATION(int nbLayers, int * neuronPerLayer, 
			int nbData, double **dataIn,
			double **dataOut, int maxPop,
			double *wRange,double **tmatIn,double **tmatOut);
	virtual ~POPULATION();
	//POPULATION& operator = (const POPULATION& p);
	//POPULATION& operator = (POPULATION* p);
	void init(int nbLayers, int * neuronPerLayer, 
			int nbData, double **dataIn,
			double **dataOut, int maxPop,
			double *wRange,double **tmatIn,double **tmatOut);
	void print() const;
	void copy(POPULATION& p); //should add const

	vector<ANN*> pop;
	ANN* 	obj;

	void	reorder();

	void	newChro();
	int	select();

	int nbLayers;
	int * neuronPerLayer;
	int nbData; 
	double **dataIn; 
	double *wRange;

	
	//int	getSize() const;
inline int	getSize() const{
	return (int)pop.size();
}


};
#endif
