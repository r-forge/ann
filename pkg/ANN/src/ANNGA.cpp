////////////////////////////////////////////////////////////////////
// RcppANNGA.cpp: Artificial Neural Network optimized by Genetic Algorithm 
//
// Copyright (C) 2011-2012 Francis Roy-Desrosiers
//
// This file is part of ANN.
//
// ANN is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ANN is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ANN.  If not, see <http://www.gnu.org/licenses/>.
////////////////////////////////////////////////////////////////////


#include <cmath>
#include "files/globalFunction.cpp"
#include "files/ga.cpp"
#include "files/cudaannMP.cpp"
#include "files/predict.cpp"
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <Rcpp.h>

using namespace Rcpp ;
extern "C"{

////////////////////////////////////////////////////////
//RcppExport 
SEXP ANNGA(SEXP matrixInput,
	SEXP matrixOutput,
	SEXP design, 
	SEXP maxPop, 
	SEXP mutation, 
	SEXP crossover, 
	SEXP maxW, 
	SEXP minW, 
	SEXP maxGen, 
	SEXP error,
	SEXP rprintBestChromosome,
	SEXP rthreads,
	SEXP rCppSeed) {

 try {
////////////////////////////////////////////////////////
///////////////////INITIALIZATION///////////////////////
////////////////////////////////////////////////////////
	
	
	int *Rdim 		= INTEGER(getAttrib(matrixOutput, R_DimSymbol));
	int kOut 		= Rdim[1];
	Rdim 			= INTEGER(getAttrib(matrixInput, R_DimSymbol));
	int kIn 		= Rdim[1];	  //cols
	int lengthData 		= Rdim[0]; //rows
		
	double** matIn 		= new double*[lengthData];
	double** matOut 	= new double*[lengthData];


	/* EXAMPLE R matrix to CPP array 
	for( i=0; i<nrow; i++){ 
                for( j=0; j<ncol; j++){ 
                        Rprintf( " %f ", p[i+nrow*j] ) ; 
                } 
                Rprintf( "\\n" ) ; 
        };*/ 

	for (int i=0; i<lengthData; i++) {
	matIn[i]=new double[kIn];
	    	for (int j=0; j<kIn; j++) {
			matIn[i][j] = REAL(matrixInput)[i+lengthData*j];
	    	}
	}
	for (int i=0; i<lengthData; i++) {
		matOut[i]=new double[kOut];
	    	for (int j=0; j<kOut; j++) {
			matOut[i][j] = REAL(matrixOutput)[i+lengthData*j];
	    	}
	}

	int *nbNeuronPerLayer	=INTEGER(design);
	int 	nbOfLayer 	=length(design);
	int 	iMaxPop		=INTEGER(maxPop)[0];
	double 	dmutation 	=REAL(mutation)[0];
	double 	dcrossover	=REAL(crossover)[0];
	double	weightRange[2]	={REAL(minW)[0] , REAL(maxW)[0]};
	int 	imaxGen		=INTEGER(maxGen)[0];
	double 	derror		=REAL(error)[0];
	bool printBestChromosome= (bool)INTEGER(rprintBestChromosome)[0];
	int	threads 	= INTEGER(rthreads)[0];
	int	cppSeed		= INTEGER(rCppSeed)[0];


////////////////////////////////////////////////////////
////////////////////NETWORK CREATION////////////////////
////////////////////////////////////////////////////////


	CUDAANNMP *ANNT	= new CUDAANNMP(nbOfLayer , nbNeuronPerLayer ,lengthData,
				matIn, matOut, iMaxPop, dmutation,  
				dcrossover, weightRange, printBestChromosome,
				threads, cppSeed);


////////////////////////////////////////////////////////
///////////////////NETWORK OPTIMIZATION/////////////////
////////////////////////////////////////////////////////



	while(ANNT->s.nbGen()<= imaxGen && ANNT->bestMSE()>derror){
		ANNT->generation (true); 
	}

	ANNT->getANNresult(true);


////////////////////////////////////////////////////////
//////////////////OUTPUT FOR R//////////////////////////
////////////////////////////////////////////////////////


	Rcpp::NumericMatrix ANNOutput(lengthData,kOut);
	for (int i=0; i<lengthData; i++) {
	    for (int j=0; j<kOut; j++) {
		ANNOutput(i,j) = ANNT->outputANN[i][j];
	    }
	}
	
	Rcpp::NumericVector chromosome(ANNT->mWeightConNum);
	for (int i=0; i<ANNT->mWeightConNum; i++) {
		chromosome[i] = 0.0; //ANNT->mChromosomes[ANNT->bestIndividual][i];
	}
	
	Rcpp::NumericVector RvectorFitness(ANNT->vectorFitness.size());	
	for (int i=0; i< (int)ANNT->vectorFitness.size(); i++) {
		RvectorFitness[i] = ANNT->vectorFitness[i];	
	}

	double finalMSE = ANNT->bestMSE();
	double nbGen	= ANNT->s.nbGen();
	//ANNT->releasePredict (); //SHOULD CALL DESTRUCTOR
	return Rcpp::List::create(Rcpp::Named("input") 		= matrixInput,
				  Rcpp::Named("desiredOutput")	= matrixOutput,
				  Rcpp::Named("output") 	= ANNOutput, 
				  Rcpp::Named("nbNeuronPerLayer") = design,
				  Rcpp::Named("population") 	= maxPop,
				  Rcpp::Named("mutation") 	= mutation,
				  Rcpp::Named("crossover") 	= crossover,
				  Rcpp::Named("maxW") 		= maxW,
				  Rcpp::Named("minW") 		= minW,
				  Rcpp::Named("maxGen") 	= maxGen, 
				  Rcpp::Named("mse") 		= finalMSE,
				  Rcpp::Named("bestChromosome") = chromosome,
				  Rcpp::Named("desiredEror") 	= error,
				  Rcpp::Named("nbOfGen")        = nbGen,
				  Rcpp::Named("vectorFitness")  = RvectorFitness
				  );
	
    } catch( std::exception &ex ) {
	forward_exception_to_r( ex );
    } catch(...) { 
	::Rf_error( "c++ exception (unknown reason)" ); 
    }
 return R_NilValue;
}



////////////////////////////////////////////////////////
SEXP predictANNGA(SEXP matrixInput,SEXP design, SEXP chromosome) {

try {

////////////////////////////////////////////////////////
///////////////////INITIALIZATION///////////////////////
////////////////////////////////////////////////////////
 

	int *Rdim = INTEGER(getAttrib(matrixInput, R_DimSymbol));
	int lengthData = Rdim[0]; //rows
	int kIn = Rdim[1];	  //cols
	double** matIn = new double*[lengthData];

	for (int i=0; i<lengthData; i++) {
	matIn[i]=new double[kIn];
	    	for (int j=0; j<kIn; j++) {
			matIn[i][j] = REAL(matrixInput)[i+lengthData*j];
	    	}
	}

	int *nbNeuronPerLayer	=INTEGER(design);	
	int 	nbOfLayer	= length(design);
	int 	kOut 		= nbNeuronPerLayer[nbOfLayer-1];
	double *cppChromosome 	= REAL(chromosome);


////////////////////////////////////////////////////////
////////////////////NETWORK CREATION////////////////////
////////////////////////////////////////////////////////


	PREDICT *ANNT= new PREDICT(nbOfLayer , nbNeuronPerLayer ,lengthData, matIn);
	ANNT->load(cppChromosome);
	ANNT->getPredictResult();



////////////////////////////////////////////////////////
//////////////////OUTPUT FOR R//////////////////////////
////////////////////////////////////////////////////////


	Rcpp::NumericMatrix output(lengthData,kOut);
	for (int i=0; i<lengthData; i++) {
	    for (int j=0; j<kOut; j++) {
		output(i,j) = ANNT->outputANN[i][j];
	    }
	}
	
	//ANNT->release (); //release the memory, TO DO NOT WORKING
	return Rcpp::List::create(Rcpp::Named("predict") = output);

    } catch( std::exception &ex ) {
	forward_exception_to_r( ex );
    } catch(...) { 
	::Rf_error( "c++ exception (unknown reason)" ); 
    }
 return R_NilValue;

}


}//end extern "C"

