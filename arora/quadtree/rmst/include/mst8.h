#ifndef _MST8_H_
#define _MST8_H_

#include "global.h"

void  mst8_package_init(long n);
void  mst8_package_done();
void  mst8( long n, Point* pt, long* parent );


void add_new_point(int x, int y, void *handle);
void make_equiv(void *h1, void *h2);
void end_net();
void start_net();
void make_last_two_equiv();
double mst();

double mst_cost(const int&,const double*,const double*);

int mst_arc(int i, int *x1, int *y1, void **p1, int *x2, int *y2, void **p2); 
int mst_used(int i, int *int_edges, int *ext_edges);
 
typedef enum obstacle_type {HORIZONTAL_OBS, VERTICAL_OBS, NONE} obstacle_type;
 
void add_an_obstacle(int x1,int y1,int x2,int y2, obstacle_type obs, int li );
void and_all_obstacles(int n_h_layers, int n_v_layers);
int wire_est_set_fn(const char *pstr, double d, char *arg);
double internal_to_mm(int n);
double internal_area_to_mm2(int n);
double internal_to_microns(int n);
void set_mpy_term(double d);

#endif 

