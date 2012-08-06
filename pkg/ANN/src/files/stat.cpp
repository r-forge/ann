////////////////////////////////////////////////////////////////////
// stat.cpp: Artificial Neural Network optimized by Genetic Algorithm 
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

#ifndef STAT_CPP
#define STAT_CPP
#include <algorithm>
#include "stat.h"



STAT::STAT(){
	sumMSE=0.0;
	avgMSE=0.0;
	generation=-1;



}

STAT::~STAT(){}

void STAT::stat(POPULATION& p){
	
	//p.reorder();
	double best = 99999999999.9;

	sort(p.pop.begin(), p.pop.end(), sortPop); /// SHOULD BE THE LINE 41

	sumMSE=0;
	int n = p.pop.size();	
	for(int i = 0 ; i < n ; i++){ 
		sumMSE+= p.pop[i]->MSE();
		if(p.pop[i]->MSE() < best){
			best = p.pop[i]->MSE();
			bestI = i;
		}


	}
	avgMSE = sumMSE/(double)n;
	++generation;


	Rprintf("\n*Generation: %d   best fitness : %4.8f Mean of population:%4.8f\n",generation, p.pop[bestI]->MSE(),avgMSE);
}







#endif
