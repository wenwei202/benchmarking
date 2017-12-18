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
#include "mkl.h"

#include "mkl_example.h"

int main(int argc, char *argv[])
{
      FILE *in_file;
      char *in_file_name;

      MKL_INT         m, n, k;
      MKL_INT         lda, ldb, ldc;
      MKL_INT         rmaxa, cmaxa, rmaxb, cmaxb, rmaxc, cmaxc;
      float           alpha, beta;
      float          *a, *b, *c;
      CBLAS_LAYOUT    layout;
      CBLAS_TRANSPOSE transA, transB;
      MKL_INT         ma, na, mb, nb;

      printf("\n     C B L A S _ S G E M M  EXAMPLE PROGRAM\n");

/*       Get input parameters                                  */

      if( argc == 1 ) {
          printf("\n You must specify in_file data file as 1-st parameter");
          return 1;
      }
      in_file_name = argv[1];

/*       Get input data                                        */

      if( (in_file = fopen( in_file_name, "r" )) == NULL ) {
         printf("\n ERROR on OPEN '%s' with mode=\"r\"\n", in_file_name);
         return 1;
      }
      if( GetIntegerParameters(in_file, &m, &n, &k) != 3 ) {
          printf("\n ERROR of m, n, k reading\n");
          return 1;
      }
      if( GetScalarsS(in_file, &alpha, &beta) != 2 ) {
          printf("\n ERROR of alpha, beta reading\n");
          return 1;
      }
      if( GetCblasCharParameters(in_file, &transA, &transB, &layout) != 3 ) {
          printf("\n ERROR of transA, transB, layout reading\n");
          return 1;
      }

      if( transA == CblasNoTrans ) {
         rmaxa = m + 1;
         cmaxa = k;
         ma    = m;
         na    = k;
      } else {
         rmaxa = k + 1;
         cmaxa = m;
         ma    = k;
         na    = m;
      }
      if( transB == CblasNoTrans ) {
         rmaxb = k + 1;
         cmaxb = n;
         mb    = k;
         nb    = n;
      } else {
         rmaxb = n + 1;
         cmaxb = k;
         mb    = n;
         nb    = k;
      }
      rmaxc = m + 1;
      cmaxc = n;
      a = (float *)mkl_calloc(rmaxa*cmaxa, sizeof( float ), 64);
      b = (float *)mkl_calloc(rmaxb*cmaxb, sizeof( float ), 64);
      c = (float *)mkl_calloc(rmaxc*cmaxc, sizeof( float ), 64);
      if( a == NULL || b == NULL || c == NULL ) {
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
      fclose(in_file);

/*       Print input data                                      */

      printf("\n     INPUT DATA");
      printf("\n       M=%d  N=%d  K=%d", m, n, k);
      printf("\n       ALPHA=%5.1f  BETA=%5.1f", alpha, beta);
      PrintParameters("TRANSA, TRANSB", transA, transB);
      PrintParameters("LAYOUT", layout);
      PrintArrayS(&layout, FULLPRINT, GENERAL_MATRIX, &ma, &na, a, &lda, "A");
      PrintArrayS(&layout, FULLPRINT, GENERAL_MATRIX, &mb, &nb, b, &ldb, "B");
      PrintArrayS(&layout, FULLPRINT, GENERAL_MATRIX, &m, &n, c, &ldc, "C");

/*      Call SGEMM subroutine ( C Interface )                  */

      cblas_sgemm(layout, transA, transB, m, n, k, alpha,
                  a, lda, b, ldb, beta, c, ldc);

/*       Print output data                                     */

      printf("\n\n     OUTPUT DATA");
      PrintArrayS(&layout, FULLPRINT, GENERAL_MATRIX, &m, &n, c, &ldc, "C");

      mkl_free(a);
      mkl_free(b);
      mkl_free(c);

      return 0;
}

