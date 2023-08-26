#ifndef _SD2_
#define _SD2_
 
 
typedef struct xy_base XYBASE;
typedef struct area_list_item XYITEM;           /* 16 bytes */
typedef struct xytree XYTREE;                   /* 26 bytes */
 
struct area_list_item {
   struct area_list_item *next;
   int x1,y1,x2,y2;
   void *ud;            /* user data */
   };
 
int DB_get_next_xyitem_32(XYITEM **);
void DB_set_xytree_search_box_32(XYBASE *, int, int, int, int);
XYBASE *make_xytree_32(int);
void register_area_32(XYBASE *,  int, int, int, int, void*);
int unregister_area_32(XYBASE *, int, int, int, int, void*);
void rebalance_xytree_32(XYBASE *);
void free_xytree_32(XYBASE *);
int xy_apply_32( XYBASE *, int,int,int,int, int(*fn)(XYITEM*, void*), void *);
 
#endif 
 

