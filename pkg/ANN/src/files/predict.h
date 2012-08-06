////////////////////////////////////////////////////////////////////
// predict.h: Artificial Neural Network optimized by Genetic Algorithm 
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

#ifndef ANN_PREDICT_H
#define ANN_PREDICT_H


#include "ANN.cpp"
#include "population.cpp"
#include <vector>




class PREDICT : public GA{



public:	// to train the network
	//PREDICT();
	PREDICT(int nbLayers_,int * neuronPerLayer_,int nbData_,double **dataIn_);
	virtual ~PREDICT();

	void	generation(bool);
	void getPredictResult();
	void load(double *chromosome);
	ANN* ann;
private:

};




#endif
