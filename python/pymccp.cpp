#include "pymccp.h"
#include <iostream>
#include <boost/cstdint.hpp>

using namespace std;

int * pymcc_classify(double *x, double *y, double *z, int32_t n, double scaleDomain2Spacing, double curvatureThreshold) {
  uint8_t *classification = mcc_classify(x, y, z, n, scaleDomain2Spacing, curvatureThreshold);
  int *intClassification = new int[n];
  for (int i = 0; i < n; i++) {
    intClassification[i] = static_cast<int>(classification[i]);
  }
  delete[] classification;
  return intClassification;
}

void pymcc_free_int(int *p) {
  delete[] p;
}

void pymcc_free_double(double *p) {
  delete[] p;
}

double * pymcc_pass(double *x, double *y, double *z, int32_t n, double scaleDomainSpacing) {
  return mcc_pass(x, y, z, n, scaleDomainSpacing);
}
