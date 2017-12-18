MKL_COMPILER_LIB=/opt/intel/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64/
MKL_MKL_LIB=/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/
MKL_INC=/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/include/
#gemm_test: cblas_sgemmx.c
#	gcc -o gemm_test cblas_sgemmx.c -I$(MKL_ROOT)/include -L$(MKL_ROOT)/lib/intel64  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -ldl


CC=gcc
CFLAGS=-I$(MKL_INC)  -L$(MKL_MKL_LIB) -L$(MKL_COMPILER_LIB)
LDFLAGS=-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

%.o: %.c 
	$(CC) -c -o $@ $< $(CFLAGS) $(LDFLAGS)

gemm_test: common_func.o cblas_sgemmx.o 
	gcc -o gemm_test common_func.o cblas_sgemmx.o $(CFLAGS) $(LDFLAGS)

clean:
	rm ./gemm_test
	rm ./*.o
