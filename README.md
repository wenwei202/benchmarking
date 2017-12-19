# Dependency
Intel MKL: install and add mkl to the `LD_LIBRARY_PATH`:
```
# ~/.bashrc
MKL_COMPILER_LIB=/opt/intel/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64/
MKL_MKL_LIB=/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64/
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${MKL_COMPILER_LIB}:${MKL_MKL_LIB}"
```

# Build
```
make all
```

# Run
GEMM benchmarking:
```
./gemm_test 6000 3000 20
```

SparseBlas benchmarking:
```
./spblas_test 4 8 6 0.8
```
