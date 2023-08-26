#include  <stdio.h>
#include  <stdlib.h>
#include  <string.h>
#include  "global.h"
#include  "mst1.h"
#include  "mst2.h"
#include  "mst8.h"
#include  "dist.h"
#include  "err.h"
#include  "unixtimer.h"

/****************************************************************************/

void   print_usage( char*  cmd )
{
  fprintf( stderr, "Usage: %s ", cmd );
  fprintf( stderr, "[-prim|" ); 
  fprintf( stderr, "-guibas|" ); 
  fprintf( stderr, "-scheffer] " ); 
  fprintf( stderr, "[-length] " );
  fprintf( stderr, "[-tree] " );
  fprintf( stderr, "[-iterations N] \n" );
}

/****************************************************************************/

int  main
(
  int      argc,
  char**   argv
)
{
  int     print_tree   = FALSE;
  int     print_length = FALSE;
  long    num_iter = 1;
  long    i, n;
  Point*  pt;
  long*   parent;
  double  len;  

  void (*mst)( long, Point*, long* ) = mst1;
  void (*mst_package_init)( long )   = mst1_package_init;
  void (*mst_package_done)()         = mst1_package_done;

  /* 
    read command line parameters
  */
  for( i = 1; i < argc; i++) 
  {
    if(!strncmp(argv[i], "-p", strlen("-p"))) 
    {
      mst              = mst1;
      mst_package_init = mst1_package_init;
      mst_package_done = mst1_package_done;
    }
    else if(!strncmp(argv[i], "-g", strlen("-g"))) 
    {
      mst              = mst2;
      mst_package_init = mst2_package_init;
      mst_package_done = mst2_package_done;
    }
    else if(!strncmp(argv[i], "-s", strlen("-s"))) 
    {
      mst      = mst8;
      mst_package_init = mst8_package_init;
      mst_package_done = mst8_package_done;
    }
    else if(!strncmp(argv[i], "-t", strlen("-t"))) 
    {
      print_tree = TRUE;
    }
    else if(!strncmp(argv[i], "-l", strlen("-l"))) 
    {
      print_length = TRUE;
    }
    else if(!strncmp(argv[i], "-i", strlen("-i"))) 
    {
      num_iter = atol(argv[++i]);
    }

    else /* unrecognized parameter */
    {
      print_usage( argv[0] );
      return  -1;
    }
  }

  if( num_iter < 1 )
  {
    num_iter = 1;
  }

  /* 
    allocate memory and read terminals
  */
  if( scanf( "%ld\n", &n ) != 1 )
  {
    err_exit( "Missing number of points" );
  }

  if( n <= 0 ) return  -1;

  pt     = (Point*)calloc( n, sizeof(Point) );
  parent = (long*)calloc( n, sizeof(long) );

  if( !pt || !parent )
  {
    err_exit( "Cannot allocate memory in main()!" );
  }

  for( i = 0;  i < n;  i++ )
  {
    if( scanf( "%ld %ld\n", &(pt[i].x), &(pt[i].y) ) != 2 )
    {
      err_exit( "Missing point coordinates!" );
    } 
  }

  /********************************** 
    time 'num_iter' MST computations 
  */
  mst_package_init( n );
  start_timer();
  for( i = 0;  i < num_iter;  i++ )
  { 
    (*mst)( n, pt, parent );
  }

  /********************************** 
    print results 
  */

  printf( "NUM_POINTS: %ld  CPU_SEC: %f ", n, cpu_seconds()/num_iter );

  if( print_length )
  {
    len = 0.0;
    for( i = 0;  i < n;  i++ )
    {
      len += (double)dist( pt[i], pt[parent[i]] );
    }
    printf( "TREE_LEN: %f \n", len );    
  }

  if( print_tree )
  {
    for( i = 0;  i < n;  i++ )
    {
      printf( "%ld ( %ld , %ld ) -> %ld ( %ld , %ld ) %f\n", 
              i, pt[i].x, pt[i].y,
              parent[i], pt[parent[i]].x, pt[parent[i]].y, 
              (double)dist( pt[i], pt[parent[i]] ) );    
    }
  }

  /********************************** 
    free allocated memory 
  */
 
  free( pt );
  free( parent );
  mst_package_done();

  return  0;
}

/****************************************************************************/
/****************************************************************************/
