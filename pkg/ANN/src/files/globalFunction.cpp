#ifndef GLOBAL_FUNCTION_ANN_CPP
#define GLOBAL_FUNCTION_ANN_CPP


#include <cstdlib>

inline  double unifRand()
{   
return rand()/ (double)RAND_MAX ;
}

inline  double unifRand(double min, double max)
{   
return rand() * (max - min) / (double)RAND_MAX + min;
}

inline int unifRandInt(int min,int max)
{
return (int)(rand() % (max - min + 1) + min);
}

//The following code (gaussRand) is a modification of the (box_muller) found on http://www.taygeta.com/random/gaussian.html
void gaussRand(double &y1,double &y2){

        double x1, x2, w; //y1, y2;
	

         do {
                 x1 = 2.0 * (rand() /(double)RAND_MAX) - 1.0;
                 x2 = 2.0 * (rand() /(double)RAND_MAX) - 1.0;
                 w = x1 * x1 + x2 * x2;

         } while ( w >= 1.0 );

         w = sqrt( (-2.0 * log( w ) ) / w );
         y1 = x1*w;
         y2 = x2*w;

}


/* boxmuller.c           Implements the Polar form of the Box-Muller
                         Transformation

                      (c) Copyright 1994, Everett F. Carter Jr.
                          Permission is granted by the author to use
			  this software for any application provided this
			  copyright notice is preserved.

*/
//The polar form of the Box-Muller transformation is both faster and more robust numerically. The algorithmic description of it is:
/* normal random variate generator *//* mean m, standard deviation s */
/*float box_muller(float m, float s)	
{				        
	float x1, x2, w, y1;
	static float y2;
	static int use_last = 0;

	if (use_last)		        // use value from previous call 
	{
		y1 = y2;
		use_last = 0;
	}
	else
	{
		do {
			x1 = 2.0 * ranf() - 1.0;
			x2 = 2.0 * ranf() - 1.0;
			w = x1 * x1 + x2 * x2;
		} while ( w >= 1.0 );

		w = sqrt( (-2.0 * log( w ) ) / w );
		y1 = x1 * w;
		y2 = x2 * w;
		use_last = 1;
	}

	return( m + y1 * s );
}*/














#endif
