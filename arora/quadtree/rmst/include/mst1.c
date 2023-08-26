#include <assert.h>
#include <stdlib.h>
#include "global.h"
#include "dist.h"
#include "err.h"

/****************************************************************************/
/*
*/

long  mst1_max_points = 0; 
long* dist_to_tree = (long*)NULL;

/****************************************************************************/
/*
*/

void  mst1_package_init( long  n )
{
  if( mst1_max_points < n )
  {
    dist_to_tree = (long*)realloc( (void*)dist_to_tree, 
                                   (size_t)n*sizeof(long) );
    if( !dist_to_tree )
    {
      err_exit( "Cannot allocate memory in mst1_package_init!" );
    }
    mst1_max_points = n;
  }
}

/****************************************************************************/
/*
*/

void  mst1_package_done()
{
  mst1_max_points = 0;
  if( dist_to_tree ) 
  {
    free( (void*)dist_to_tree );
    dist_to_tree = (long*)NULL;
  }
}

/****************************************************************************/
/*
   O(n^2) implementation of Prim's algorithm for computing the MST.
*/

void  mst1
(
  long    n,
  Point*  pt,
  long*   parent
)
{
  long  i, j, k;
  long  d1, max_d, best_d;
  long  best_i;
  long  root = 0;     

  mst1_package_init( n );

  max_d  = 0;
  for( i = 0;  i < n;  i++ )
  {
    parent[i]        = root;
    dist_to_tree[i]  = dist( pt[i], pt[root] );
    if( dist_to_tree[i] > max_d )
    {
      max_d = dist_to_tree[i];
    }
  }
 
  parent[root]        = root;
  dist_to_tree[root]  = 0;

  for( k = 1;  k < n;  k++ )   /* n-1 points to add */
  {

    best_d = max_d + 1;
    best_i = -1;

    for( i = 0;  i < n;  i++ )
    {
      if( (dist_to_tree[i] != 0) && (dist_to_tree[i] < best_d) )   
      {
        best_d  = dist_to_tree[i];
        best_i  = i;
      }
    }

#ifdef DEBUG
    assert( best_i != -1 );
#endif
 
    /*
      pt[best_i] enters the tree
    */
    dist_to_tree[best_i] = 0;

    /*
      update dist_to_tree for remaining nodes 
    */
    for( j = 0;  j < n;  j++ )
    {
      if( dist_to_tree[j] != 0 )
      {
        d1 = dist( pt[j], pt[best_i] );
        if( d1 < dist_to_tree[j] )
        {
          dist_to_tree[j] = d1;
          parent[j] = best_i; 
        }
      }
    }
  }
}
/****************************************************************************/
/****************************************************************************/

