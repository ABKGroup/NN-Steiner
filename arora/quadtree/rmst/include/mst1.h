#ifndef _MST1_H_
#define _MST1_H_

#include "global.h"

void  mst1_package_init( long  n );
void  mst1_package_done();
void  mst1( long n, Point* pt, long* parent );

#endif 

