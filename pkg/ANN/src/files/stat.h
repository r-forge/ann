////////////////////////////////////////////////////////////////////
// stat.h: Artificial Neural Network optimized by Genetic Algorithm 
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

#ifndef STAT_H
#define STAT_H




class STAT{
public:	
	STAT(); 	
	virtual ~STAT();

	void stat(POPULATION& p);
	void stat(vector<ANN>& nn);

	inline double	SumMSE() const{ return sumMSE; }
	inline double	AvgMSE() const{ return avgMSE; }
	inline int	nbGen()  const{ return generation; }
	inline int bestIndividual() const{ return bestI; }

private:
	double	sumMSE;
	double	avgMSE;
	int	generation;
	int	bestI;
	
};


#endif
