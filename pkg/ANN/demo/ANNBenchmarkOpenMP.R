#ANN Multi-threading benchmark
#Francis Roy-Desrosiers 2011-2012


require(ANN)
install.packages("rbenchmark")
require(rbenchmark)

ANNBenchmarkOpenMP.ANN <- function() {
set.seed(1)
t               <-seq(0, 2 * pi, 0.01)
#sin(t) + noise
noisy_sin       <-sin(t) + 0.5 * rnorm( length(t) ) 

fLog            <-  function(x)	1/(1 + exp( -x ))


output          <-matrix( fLog(noisy_sin) )
input           <-matrix( seq( 0, 1, length.out=length(t) ) )


res <- benchmark(ANNGA(x=input, y=output, threads=1,printBestChromosone=FALSE),
                 ANNGA(x=input, y=output, threads=2,printBestChromosone=FALSE),
                 ANNGA(x=input, y=output, threads=3,printBestChromosone=FALSE),
                 ANNGA(x=input, y=output, threads=4,printBestChromosone=FALSE),
                 ANNGA(x=input, y=output, threads=5,printBestChromosone=FALSE),
                 ANNGA(x=input, y=output, threads=6,printBestChromosone=FALSE),
                 ANNGA(x=input, y=output, threads=7,printBestChromosone=FALSE),
                 ANNGA(x=input, y=output, threads=8,printBestChromosone=FALSE),
                 columns=c("test", "replications", "elapsed",
                           "relative", "user.self", "sys.self"),
                 order="relative",
                 replications=3)


print(res) 


}

ANNBenchmarkOpenMP.ANN()
