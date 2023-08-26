/*
Generate a minimum spanning tree from geometrical points.  The general MST
algorithms are N^2.  For the Euclidean case, it is possible at least in theory
to do better (taking all the edges from the Voronoi dual, then using Kruskal's
algorithm or priority first on those O(N) edges.)
We want something similar in speed, but simpler.  So we do this:
 
The basic algorithm is priority first.  We start with a point, add its
nearest neighbor, add the point that is closest to either of these, etc.
We keep the distances to all points not yet included in a priority queue 
(implemented as a heap) so we can easily find the smallest of all the known
edges.
 
As we add each point, we need to compute all the neighbors of that point.
This is where the efficiency gain comes in, since we know from theory that
all edges in the final graph come from the Voronoi dual of the original
points.  Therefore we only need to look around a point until all possible
points of this type are included.  Points further away do not need to
be considered since these edges will never end up in the final graph.
 
We do this search in each of four quadrants, oriented at a 45 degree angle
to the main axes.  We search out until we find some nearby points.  With
these points, we can calculate some limits on how far away other points in
the quadrant could be and still be Voronoi neighbors.  We then search out this
far.  If we find any points not yet in our priority queue, or points that
are in the queue but at a longer distance, we add/change the queue to
reflect the new distances.
 
Assuming the search is done with a quad tree, and we start with roughly the
right search size, this takes N*logN time.  The queue can never contain more
than N entries, where N is the number of points.  Hence the memory
requirements are O(N).
 
Regions of constant distance under the Manhatten metric are diamonds.  Since
we want to do searches on these efficiently, we take advantage of a coordinate
transformation.  When we store the points, and when we set search regions,
we transform the original coordinates by:
   x' = x + y;
   y' = x - y;
Then we can use the existing fast and efficient rectangular area searches.
 
 */

#include "global.h" 
#include "mst8.h"
#include <stdio.h>
#include "sd2.h"
//#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MININT 0x80000000
#define TRUE 1
#define FALSE 0
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define NERR(n) printf("Error %d\n", n)
static char debug_flags[256];
 
static int *heap = NULL;        /* array of indexes that form a heap */
static int N;                   /* number of edges in the heap */
static int HeapSize;            /* size of the heap array */
 
typedef struct point8 POINT;
struct point8 {
   int x,y;     /* These first fields are defined by the user */
   void *handle;/* user defined handle for this point */
   int ring;    /* index of next equivalent pin; forms a ring */
 
                /* These fields are computed during the solution.  The heap */
                /* just says that point 'i' is in the heap.  For information */
                /* about that point, see these fields. */
   int w;       /* weight - distance from the solution thus far. -1 for points*/
                /* that are declared 'equivalent' tho they are at different */
                /* locations.  */
   int other;   /* other end of arc with this weight.  -1 for first point, */
                /* since this point has no 'other' vertex to connect to    */
   int where;   /* where is this vertex stored in the heap  - needed for */
                /* updating distances. -1 if not in heap (because it has */
                /* already been used.) */
   short ext_edges;     /* some users of this package care how many times */
                        /* a given point was used in the solution, so we  */
                        /* keep track.  This counts 'external' edges.  */
   short  int_edges;    /* likewise for internal edges */
   };
 
static POINT *pts = NULL;
static int NPts;
static int PtsAllocated;
 
/* The following dynamically allocated array will be filled during the course */
/* of compution, and will describe all the arcs needed to form a solution.    */
/* Since the number of arcs in the solution is bounded by the number of input */
/* points, we allocate it in parallel with the point array. */
typedef struct soln SOLN;
struct soln {
    int p1, p2;         /* the indices into the point array */
                        /* if both >= 0, a regular edge */
                        /* if both <= 0, a internally connected edge */
    };
static SOLN *soln_arcs;
int _MST_Nsoln;
 
static XYBASE *h_layer_obstacles;  /* obstacles on all horizontal layers */
static XYBASE *v_layer_obstacles;  /* and vertical layers */
static XYBASE *h_obstacles;        /* Resulting composite blockages */
static XYBASE *v_obstacles;        /*  computed from the above data */
 
/* The following heap routines are based on the discussions in "Algorithms"
by Sedgewick. */
 
/* the k'th entry might violate the heap condition.  Move up if needed */
void up_heap(int k)
{
    int v;
    pts[heap[0]].w = MININT;
    v = heap[k];
    for(;pts[heap[k>>1]].w >= pts[v].w;){
        heap[k] = heap[k>>1];
        pts[heap[k]].where = k;
        k >>= 1;
        }
    heap[k] = v;
    pts[v].where = k;
}
 
/* insert the pts[k] into as a new entry into the heap. */
void insert_heap(int k)
{
    N++;
    if (N >= HeapSize) {
        HeapSize <<= 1;
        heap = (int *)realloc((char *)heap, HeapSize*sizeof(int));
        }
    heap[N] = k;
    pts[k].where = N;
    up_heap(N);
}
 
/* Move an item down in the heap until it finds its appropriate spot */
void down_heap(int k)
{
    int v;
    int j;
    v = heap[k];
    for(;k <= (N>>1); ) {
        j = k<<1;       /* make j point to the smaller of the two children */
        if (j < N) {
            if (pts[heap[j]].w > pts[heap[j+1]].w)
                j++;
            }
        if (pts[v].w <= pts[heap[j]].w) {  /* heap condition is satisifed */
            heap[k] = v;
            pts[v].where = k;
            return;
            }
        heap[k] = heap[j];              /* move child up */
        pts[heap[k]].where = k;
        k = j;
        }
    /* if we get here, K has no children, so heap is satisfied */
    heap[k] = v;
    pts[v].where = k;
}
 
/* remove the first item from the heap.  Return its index into pts[]  */
int remove_heap()
{
    int save = heap[1];
    heap[1] = heap[N];
    pts[heap[1]].where = 1;
    N--;
    down_heap(1);
    return save;
}
 
/* Initialize the heap */
void init_heap()
{
    N = 0;
}
 
/* The value of pts[j] has changed.  Move it to the right place in the heap. */
void modify_heap(int j, int dnew)
{
    int old = pts[j].w;
    pts[j].w = dnew;
    if (dnew < old)
        up_heap(pts[j].where);
    else
        down_heap(pts[j].where);
}
 
/* build a union-find data structure to detect connectivity as defined by
pairs of connected points.  We use the 'where' field of the data structure
since it won't be needed until we create the heap. Once again the basic
algorithms can be found in Sedgewick.  */
 
#define dad(i) (pts[i].where)
int fast_find(int x,int y,int combine)
{
    int i,j,t;
    x++; y++;   /* convert to 1 based indexing since 0 is special */
    for(i=x; dad(i) > 0; )
        i = dad(i);
    for(j=y; dad(j) > 0; )
        j = dad(j);
    for (; dad(x) > 0; ) 
        { t = x; x = dad(x); dad(t) = i;}
    for (; dad(y) > 0; )
        { t = y; y = dad(y); dad(t) = j;}
    if (combine && i != j) {
        if (dad(j) < dad(i))
            {dad(j) = dad(j) + dad(i) - 1; dad(i) = j;}
        else
            {dad(i) = dad(i) + dad(j) - 1; dad(j) = i;}
        }
    return (i != j);
}
 
 
/* we use smaller than normal MAXINT and MININT here. because we are doing
a coordinate transformation that can double the abs() of the numbers. */
#undef MAXINT
#undef MININT
#define MAXINT 0x3FFFFFFF
#define MININT ((int)0xC0000000)
 
/* transform a search box by 45 degrees, then set it. */
void warped_DB_set_xytree_search_box(XYBASE *t,int x0,int y0,int x2,int y2)
{
    int wx0, wy0, wx2, wy2;
    wx0 = x0 + y0;
    wy0 = x0 - y0;
    wx2 = x2 + y2;
    wy2 = x2 - y2;
    DB_set_xytree_search_box_32(t,wx0,wy0,wx2,wy2);
}
 
/* two variables to keep track of the average distance to the nearest point,
kept over all the quadrants where we found anything at all. */
static int NFound;
static double total_dist;
 
/* a list of all items found in a square.  Dynamic to allow for worst-case,
which is thousands of objects all equidistant from the first point */
static XYITEM **list = NULL;
static int NList;
 
/* look up under a diamond shape.  Start at (x0,y0) and expand the opposite
corner of the diamond by (dx,dy).  We find all points that could possibly
belong to the minimum spanning tree graph.  */
int find_nearest_quad(XYBASE *t, int id, int dx, int dy)
{
    int x2, y2;
    double delta_x = dx, delta_y = dy;
    XYITEM *item;
    int x0 = pts[id].x, y0 = pts[id].y;
    int d, d_best, d_best1, d_best2;
    int i,nl;
    int found;

/* now set the starting box size to the average of the minimum distances
found already.  The 2.0 is an empirical constant that affects performance
but not correctness.  If it's too big, we'll find points we won't use; if
it's too small, we'll need to repeat searches with bigger boxes. */
    if (NFound > 0 && total_dist > 0.0) {
         delta_x *= total_dist*2.0 / NFound;
         delta_y *= total_dist*2.0 / NFound;
         }
/* next, expand the box till we find something not at the search origin */
    /* printf("Starting at %d %d\n",x0,y0); */

    for(;;) {
        x2 = (int)(x0 + delta_x);
        y2 = (int)(y0 + delta_y);
        found = FALSE;
        /* printf("Searching (%d %d) (%d %d)\n", x0, y0, x2, y2); */
        warped_DB_set_xytree_search_box(t,x0,y0,x2,y2);
        for(nl=0; DB_get_next_xyitem_32(&item); nl++) {
            if (nl >= NList) {
                NList <<= 1;
                list = 
                  (XYITEM **)realloc((char *)list, NList * sizeof(XYITEM *));
                }
            list[nl] = item;
            found |= (x0 != ((item->x1+item->y1)>>1)) || (y0 != ((item->x1-item->y1)>>1));
            }
        if (found)
            break;  /* stop if we find anything */
        if (x2==MAXINT  || x2 == MININT || y2 == MAXINT || y2 == MININT)
            return 0;
        /* Nothing, double the search box size */
        delta_x *= 2.0;
        delta_y *= 2.0;
        if (x0 + delta_x > MAXINT)
            delta_x = MAXINT - (double)x0;
        if (x0 + delta_x < MININT)
            delta_x = MININT - (double)x0;
        if (y0 + delta_y > MAXINT)
            delta_y = MAXINT - (double)y0;
        if (y0 + delta_y < MININT)
            delta_y = MININT - (double)y0;
        }
    /* we got a point, we measure the distance, and then set the bounding box
    so that we get all points within that distance.  We calculate the bounds
    for each octant separately since we can set tighter bounds within the
    octant than in the quadrant as a whole.  So if we get lucky and have a
    point in each octant, we can get a tighter bound overall. */
    d_best = d_best1 = d_best2 = MAXINT;
    for (i=0; i<nl; i++) {
        int node = (int)list[i]->ud;
        int diffx = x0 - pts[node].x;
        int diffy = y0 - pts[node].y;
        /* printf("Found %d %d\n", pts[node].x, pts[node].y); */
        d = abs(diffx) + abs(diffy);
        if (d == 0)     /* points exactly on top don't count */
            continue;
        if (d < d_best)
            d_best = d;
        /* convert to a case looking straight up */
        if (dx != 0 && dy == 0)
            {int tmp = diffx; diffx = diffy; diffy = tmp;}
        /* for points that are not exactly straight up, we can set tighter
        bounds, but they only apply to the octant the point is in */
        if (diffx != 0)
            {if (abs(diffx)*3 > abs(diffy))
                d = (d+1)>>1;  /* round up */
            else
                d -= 2*abs(diffx);
            if (diffx > 0)
                {if (d < d_best1) d_best1 = d;}
            if (diffx < 0)
                {if (d < d_best2) d_best2 = d;}
            }
        }
    /* If both octant bounds exist, and both are less than the overall bound,
    then we can use the greater of the two as the overall bound */
    if (d_best1 < d_best && d_best2 < d_best) {
        /* printf("%d %d %d\n", d_best, d_best1, d_best2); */
        d_best = d_best1 > d_best2 ? d_best1 : d_best2;
        }
 
    /* update average... */
    NFound++;
    total_dist += d_best;
 
    /* Any other points that share a Veronoi edge cannot be further than twice
    d_best far away.  Find all points that meet this condition and enter their
    edges. */
    d = d_best * 2;
    /* first check to see if we already have the points we need. */
    if (d < (abs((int)delta_x) + abs((int)delta_y))) {
        /*printf("Opp..%d %d\n",d, abs((int)delta_x) + abs((int)delta_y) );*/
        for(i=0; i<nl; i++) {
            int node = (int)list[i]->ud;
            int dnew = abs(x0-pts[node].x)+abs(y0-pts[node].y);
            if (dnew <= d && pts[node].where == 0) {
                pts[node].w = dnew;
                pts[node].other = id;
                insert_heap(node);
                }
            if (dnew <= d && pts[node].where > 0 && dnew < pts[node].w) {
                pts[node].other = id;
                modify_heap(node, dnew);
                }
            }
        return 1;
        }
    /* if we don't have the points we need, set the search box and find them.*/
    warped_DB_set_xytree_search_box(t,x0, y0, x0+d*dx, y0+d*dy);
    for( ; DB_get_next_xyitem_32(&item); ) {
        int node = (int)item->ud;
        int dnew = abs(x0-pts[node].x)+abs(y0-pts[node].y);
        if (pts[node].where == 0) {
            pts[node].w = dnew;
            pts[node].other = id;
            insert_heap(node);
            }
        if (pts[node].where > 0 && dnew < pts[node].w) {
            pts[node].other = id;
            modify_heap(node, dnew);
            }
        }
    return 1;  /* found something */
}
 
extern int *spice_node_numbers;
 
XYBASE *make_xytree_32();
 
/* Compute a minimum Manhatten Spanning Tree.  Return status */
double mst()
{
    XYBASE *t;
    int i,k;
    double total = 0.0; /* we will accumulate total tree length here */
 
    NFound = 0;         /* set two globals for keeping track of averages */
    total_dist = 0.0;
    if (NPts < 2)
        return 0.0;
 
    init_heap();
 
    /* first, enter all the points into a (rotated by 45 degrees) xy tree */
    t = make_xytree_32(0);
    for(i=0; i<NPts; i++) {
        int xp = pts[i].x + pts[i].y;
        int yp = pts[i].x - pts[i].y;
        register_area_32(t, xp, yp,  xp, yp, (void *)i);
        pts[i].where = 0;
        }
    rebalance_xytree_32(t);
 
    heap[0] = NPts;
    pts[0].w = 0;
    pts[0].other = -1;  /* nonsense value to detect first point */
    insert_heap(0);
    for(i=0; i<NPts && N>0; i++) {
        int j = remove_heap();
        int other = pts[j].other;
        /* total += pts[j].w; */  /* if tree can be routed without obstruction*/
        if (other != -1) {
            /* UI_line(pts[j].x, pts[j].y, pts[other].x,pts[other].y, WHITE); */
            /*total += (*fn)( pts[j].x, pts[j].y, pts[j].handle,
                   pts[other].x,pts[other].y, pts[other].handle); */
            if (_MST_Nsoln >= PtsAllocated)
                NERR(549);
            soln_arcs[_MST_Nsoln].p1 = pts[j].w >= 0 ? j : -j;
            soln_arcs[_MST_Nsoln].p2 = pts[j].w >= 0 ? other : -other;
            _MST_Nsoln++;
            if (pts[j].w >= 0) {
                pts[j].ext_edges++;
                pts[other].ext_edges++;
                }
            else {
                pts[j].int_edges++;
                pts[other].int_edges++;
                }
            }
        /* total += abs(pts[j].x-pts[other].x) + abs(pts[j].y-pts[other].y); */
        /* printf("%d other %d total=%10g\n",j,other, total); */
        pts[j].where = -1;
        find_nearest_quad(t, j, 0, 1);
        find_nearest_quad(t, j, -1, 0);
        find_nearest_quad(t, j, 0, -1);
        find_nearest_quad(t, j, 1, 0);
        for(k = pts[j].ring; k != j; k = pts[k].ring) {
            if (pts[k].where > 0) {  /* we've seen this point before */
                pts[k].other = j;
                modify_heap(k,-1);
                }
            if (pts[k].where == 0) {    /* it's a new point */
                pts[k].w = -1;
                pts[k].other = j;
                insert_heap(k);
                }
            }
        }
 
//  printf("B:found %d MST edges; %d left, %10g total length\n", i, N, total);
    free_xytree_32(t);
    return total;
}
 
/* set up the package to handle mst problems of n points.  If you guess n
too small, it will still work but may be more inefficient due to the need
for reallocs().  */
void mst8_package_init(long n)
{
    if (pts != NULL || list != NULL || heap != NULL)
        return; /* don't reallocate if already allocated */
    PtsAllocated = n;
    pts = (POINT *)malloc(PtsAllocated * sizeof(POINT));
    soln_arcs = (SOLN *)malloc(PtsAllocated * sizeof(SOLN));
    NList = 100;
    list = (XYITEM **)malloc(NList * sizeof(XYITEM *));
    HeapSize = n+1;
    heap = (int *)malloc(HeapSize * sizeof(int));
    h_layer_obstacles = (XYBASE *)make_xytree_32(0);
    v_layer_obstacles = (XYBASE *)make_xytree_32(0);
    h_obstacles = (XYBASE *)make_xytree_32(0);
    v_obstacles = (XYBASE *)make_xytree_32(0);
}
 
/* called after all mst() calls are made */
void mst8_package_done()
{
    free((char *)pts);  pts = NULL;
    free((char *)list); list = NULL;
    free((char *)heap); heap = NULL;
    free_xytree_32(h_layer_obstacles);
    free_xytree_32(v_layer_obstacles);
    free_xytree_32(h_obstacles);
    free_xytree_32(v_obstacles);
}
 
/* add a new point to the list.  If the list is not big enough, realloc */
void add_new_point(int x, int y, void *handle)
{
    if (NPts >= PtsAllocated) {
        PtsAllocated <<= 1;
        pts = (POINT *)realloc((char *)pts, PtsAllocated * sizeof(POINT));
        soln_arcs = 
         (SOLN *)realloc((char *)soln_arcs, PtsAllocated * sizeof(SOLN));
        }
    if(debug_flags[18])
        printf("Point: %d %d\n", x, y);
    pts[NPts].x = x;
    pts[NPts].y = y;
    pts[NPts].handle = handle;
    pts[NPts].where = 0;        /* for fast_find routine */
    pts[NPts].ring = NPts;      /* Ring with 1 element; points to itself */
    pts[NPts].int_edges = 0;    /* not used in solution, yet */
    pts[NPts].ext_edges = 0;    /* not used in solution, yet */
    NPts++;
}
 
/* starts a new net, expecting new points */
void start_net()
{
    NPts = 0;           /* No points yet */
    _MST_Nsoln = 0;          /* No arcs in the solution */
}
 
/* a local function used for sorting handles. */
#define P(a) ((int)(((POINT *)a)->handle))

extern "C"
{
  int handle_fn(const void *a,const void *b)
  { return P(a) > P(b) ? 1 : ( P(a) < P(b) ? -1 : 0); }
}
 
/* since the order of the points does not matter, we sort by the handles for
easier access later */
void end_net()
{
/* the heap and fast_find algorithms need an extra element.  Add it */
add_new_point(0,0,(void *)0);
NPts--;
qsort((char *)pts, NPts, sizeof(POINT), handle_fn);
}
 
/* returns the index corresponding to a given handle.  Returns -1 if not
in array.  Assumes array is sorted by handle so it can use binary search */
int handle_to_index(void *h)
{
    int low = 0;        /* must be this one or higher */
    int high = NPts-1;  /* must be this one or lower */
    int ih = (int)h;
    if (NPts == 0 || ih < (int)pts[0].handle || ih > (int)pts[NPts-1].handle)
        return -1;
    for(; low <= high; ) {
        int test = (low + high)>>1;
        int it = (int)pts[test].handle;
        if (it == ih)
            return test;
        else if (it < ih)
            low = test+1;
        else
            high = test -1;
        }
    return -1;
}
 
/* make the two points with handles h1 and h2 equivalent (meaning they
are already connected before we start working on the tree).  We do this
by doing a union-find to see if they are already connected.  If not,
swapping their next pointers will result in a larger ring.  This is why the
union-find is necessary - if they are in the same ring, and you swap their
next pointers, then you break the one ring into 2. */
void make_equiv(void *h1, void *h2)
{
    int i1 = handle_to_index(h1), i2 = handle_to_index(h2);
    if (i1 >= 0 && i2 >= 0 && fast_find(i1,i2, 1)) {
        int t = pts[i1].ring;           /* swap 'next in ring' pointers */
        pts[i1].ring = pts[i2].ring;
        pts[i2].ring = t;
        }
}
 
/* makes the last two points equivalent.  Don't use if sorting will change
the order */
void make_last_two_equiv()
{
    int temp;

    if(debug_flags[18])
        printf("Last two equiv\n");
    temp = pts[NPts-1].ring;
    pts[NPts-1].ring = pts[NPts-2].ring;
    pts[NPts-2].ring = temp;
    /*UI_line(pts[NPts-1].x, pts[NPts-1].y, pts[NPts-2].x,pts[NPts-2].y, RED);*/
}
 
/* Tells how many times the ith point was used in the MST solution. */
int mst_used(int i, int *int_edges, int *ext_edges)
{
    if (i < 0 || i >= NPts)
        return FALSE;
    *int_edges = pts[i].int_edges;
    *ext_edges = pts[i].ext_edges;
    return TRUE;
}
 
/* gives the ith arc back from the MST stuff.  Returns TRUE for an arc that
the MST added, FALSE for an internal arc */
int mst_arc(int i, int *x1, int *y1, void **p1, int *x2, int *y2, void **p2)
{
    int i1 = soln_arcs[i].p1, i2 = soln_arcs[i].p2;
    int status = i1 >= 0 && i2 >= 0;
    i1 = abs(i1);
    i2 = abs(i2);
    *x1 = pts[i1].x;
    *y1 = pts[i1].y;
    *p1 = pts[i1].handle;
    *x2 = pts[i2].x;
    *y2 = pts[i2].y;
    *p2 = pts[i2].handle;
    return status;
}
 
double mst_cost(const int& numPoints,const double* xArray,const double*
yArray)
{
    int i,j;
//  char what;
    mst8_package_init( (long)(numPoints + 5) );
    start_net();
    for(i=0; i<numPoints; i++) 
        add_new_point((int)xArray[i],(int)yArray[i],(void *)i);
    end_net();
    mst();
    j = 0;
    for(i=0; i<_MST_Nsoln; i++) 
    {
        int x1, y1, x2, y2;
        void *p1, *p2;
        int st = mst_arc(i, &x1, &y1, &p1, &x2, &y2, &p2);
        if (st) j += abs(x1-x2) + abs(y1-y2);
    }
    //printf("total = %d\n", j);
    mst8_package_done();
    return (double)j;
} 


void mst8
(
  long   n, 
  Point* pt,
  long* parent
)
{
    long  i;
    int x1, y1, x2, y2;
    void *p1, *p2;
    int st;
         
    mst8_package_init( n );
    start_net();

    for( i = 0;  i < n;  i++ )
    {
       add_new_point( (int)pt[i].x, (int)pt[i].y, (void *)i );
    }
    end_net();

    mst();

    for(i=0; i<_MST_Nsoln; i++) {
        st = mst_arc(i, &x1, &y1, &p1, &x2, &y2, &p2);
        if (st)
        {
          parent[(long)p1] = (long)p2;
        } 
    }
     
    mst8_package_done();
}

/*main( int argc, char **argv)
{
    int x,y;
    int i,j;
    char what;
    mst8_package_init( (long)1000 );
    for(start_net(), i=j=0;;) {
        if (scanf("%1s", &what) != 1) break;
        if (what == 'P') {
            scanf("%d %d", &x, &y); 
            add_new_point(x,y,(void *)(i++));
            }
        if (what == 'E') {
            if(j++ == 0) end_net();
            scanf("%d %d", &x, &y); 
            make_equiv((void *)x,(void *)y);
            }
        }
//  printf("%d points, %d equivs\n", i,j);
    mst();
    for(i=0; i<NPts; i++) {
        int ie, ee;
        mst_used(i, &ie, &ee);
        printf("Point %d used %d times by internal, %d times by external\n", 
        i, ie, ee);
        }
    j = 0;
    for(i=0; i<_MST_Nsoln; i++) {
        int x1, y1, x2, y2;
        void *p1, *p2;
        int st = mst_arc(i, &x1, &y1, &p1, &x2, &y2, &p2);
        if (st)
           j += abs(x1-x2) + abs(y1-y2);
        printf("%d to %d (%s)\n", (int)p1, (int)p2, st?"Normal":"Internal");
        }
    printf("total = %d\n", j);
    mst8_package_done();
} */ 
/* double mst(double (*fn)(int,int,void*,int,int,void*) ) */

