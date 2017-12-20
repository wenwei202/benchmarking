/*******************************************************************************
* Copyright 1999-2017 Intel Corporation All Rights Reserved.
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*******************************************************************************/

/*
!  Content:
!      C B L A S _ S G E M M  Example Program Text ( C Interface )
!******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"

#include "mkl_example.h"

int main(int argc, char *argv[])
{

      srand (time(NULL));

      MKL_INT         m, n, k;
      float sparsity = 0.0;
      MKL_INT         lda, ldb, ldc;
      MKL_INT         rmaxa, cmaxa, rmaxb, cmaxb, rmaxc, cmaxc;
      float           alpha=1.0, beta=0.0;
      float          *a, *spa_vals, *b, *c;
      MKL_INT *nz_idx, *nz_ptr;
      CBLAS_LAYOUT    layout=CblasRowMajor;
      CBLAS_TRANSPOSE transA=CblasNoTrans, transB=CblasNoTrans;
      MKL_INT         ma, na, mb, nb;

/*       Get input parameters                                  */

      if( argc != 5 ) {
          printf("\n please specify M K N sparsity");
          return 1;
      }
      m = atoi(argv[1]);
      k = atoi(argv[2]);
      n = atoi(argv[3]);
      sparsity = atof(argv[4]);
      printf("m=%d, k=%d, n=%d, sparsity=%f\n", m, k, n, sparsity);

      MKL_INT num = m*k;
      MKL_INT nonzero_num = (int)(num*(1-sparsity));
      if(nonzero_num<=0) nonzero_num = 1;
      else if(nonzero_num>num) nonzero_num=num;
      const MKL_INT job[] = {0,0,0,2,m*k,1};

/*       Get input data                                        */
      if( transA == CblasNoTrans ) {
         rmaxa = m;
         cmaxa = k;
         ma    = m;
         na    = k;
      } else {
         rmaxa = k;
         cmaxa = m;
         ma    = k;
         na    = m;
      }
      if( transB == CblasNoTrans ) {
         rmaxb = k;
         cmaxb = n;
         mb    = k;
         nb    = n;
      } else {
         rmaxb = n;
         cmaxb = k;
         mb    = n;
         nb    = k;
      }
      rmaxc = m;
      cmaxc = n;
      a = (float *)mkl_calloc(rmaxa*cmaxa, sizeof( float ), 64);
      spa_vals = (float *)mkl_calloc(nonzero_num, sizeof( float ), 64);
      nz_idx = (MKL_INT *)mkl_calloc(nonzero_num, sizeof( MKL_INT ), 64);
      nz_ptr = (MKL_INT *)mkl_calloc(m+1, sizeof( MKL_INT ), 64);
      b = (float *)mkl_calloc(rmaxb*cmaxb, sizeof( float ), 64);
      c = (float *)mkl_calloc(rmaxc*cmaxc, sizeof( float ), 64);
      if( a == NULL || b == NULL || c == NULL || spa_vals == NULL || nz_idx==NULL || nz_ptr==NULL) {
          printf( "\n Can't allocate memory for arrays\n");
          return 1;
      }
      if (layout == CblasRowMajor) {
          lda=cmaxa;
          ldb=cmaxb;
          ldc=cmaxc;
      } else {
          lda=rmaxa;
          ldb=rmaxb;
          ldc=rmaxc;
      }
/*
      if( GetArrayS(in_file, &layout, GENERAL_MATRIX, &ma, &na, a, &lda) != 0 ) {
        printf("\n ERROR of array A reading\n");
        return 1;
      }
      if( GetArrayS(in_file, &layout, GENERAL_MATRIX, &mb, &nb, b, &ldb) != 0 ) {
        printf("\n ERROR of array B reading\n");
        return 1;
      }
      if( GetArrayS(in_file, &layout, GENERAL_MATRIX, &m,  &n,  c, &ldc) != 0 ) {
        printf("\n ERROR of array C reading\n");
        return 1;
      }
*/
      //initial matrix
      MKL_INT info;
      FillSparseMatrixS( a, rmaxa*cmaxa, nonzero_num);
      mkl_sdnscsr(job, &m, &k, a, &lda, spa_vals, nz_idx, nz_ptr, &info);
      if(info) {
         printf("the mkl_sdnscsr routine is interrupted processing the %d-th row because there is no space in the arrays acsr and ja according to the value nzmax.", info);
         return -1;
      }
      FillMatrixS('r', b, rmaxb*cmaxb);
      FillMatrixS('r', c, rmaxc*cmaxc);

/*       Print input data                                      */

      printf("\n     INPUT DATA");
      printf("\n       M=%d  N=%d  K=%d", m, n, k);
      printf("\n       ALPHA=%5.1f  BETA=%5.1f", alpha, beta);
      PrintParameters("TRANSA, TRANSB", transA, transB);
      PrintParameters("LAYOUT", layout);
      //PrintArrayS(&layout, FULLPRINT, GENERAL_MATRIX, &ma, &na, a, &lda, "A");
      //PrintArrayS(&layout, FULLPRINT, GENERAL_MATRIX, &mb, &nb, b, &ldb, "B");
      //PrintArrayS(&layout, FULLPRINT, GENERAL_MATRIX, &m, &n, c, &ldc, "C");
      
      //first run
      const char *matdescra = "GXXCX";//6 bytes
      const char transa = 'N';
      mkl_scsrmm(&transa, &m , &n, &k, &alpha , matdescra, 
                     spa_vals, nz_idx, nz_ptr, nz_ptr+1, b, &ldb, &beta, c, &ldc);

      clock_t start = clock(), diff;
      float total=0.0;
      int test_cnt=5000;
      for(int cnt=0; cnt<test_cnt; cnt++){
            FillMatrixS('r', b, rmaxb*cmaxb); //change b
            start = clock();
            mkl_scsrmm(&transa, &m , &n, &k, &alpha , matdescra, 
                     spa_vals, nz_idx, nz_ptr, nz_ptr+1, b, &ldb, &beta, c, &ldc);
            diff = clock() - start;
            float msec = diff*1000.0 / CLOCKS_PER_SEC;
            total += msec;
            printf("Test %d: %f ms\n", cnt, msec);
      }
      printf("Average: %f ms\n", total/test_cnt);
      

/*       Print output data                                     */

      //printf("\n\n     OUTPUT DATA");
      //PrintArrayS(&layout, FULLPRINT, GENERAL_MATRIX, &m, &n, c, &ldc, "C");

      mkl_free(a);
      mkl_free(b);
      mkl_free(c);
      mkl_free(spa_vals);
      mkl_free(nz_idx);
      mkl_free(nz_ptr);

      printf("\nTest done!\n");
      return 0;
}

