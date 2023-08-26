/*==================================  sd2.cpp =======================*/

/* $Log: sd2.cpp,v $
// Revision 1.3  1995/08/18  22:41:34  billsir
// made FEF_area and sd2 able to be included by the same file
//
// Revision 1.2  1995/04/03  23:09:09  doig
// PC fixes
//
// Revision 1.1  1995/03/29  07:55:54  lou
// Initial revision
//
 * Revision 1.17  89/12/06  18:50:13  ehud
 * reduced limit of 97 to 96, etc
 *  */
/* 
 *  23-Nov-88  djt  added  unregister_area_check.
 * 
 * This set of routines is designed for area searching in an editor
 * environment.  Optimized for mostly small objects.
 * This whole package nternally assumes x2 >= x1 and y2 >= y1.
 * This is fixed first thing in each user-visible routine.
 
   XYTREE *
   make_xytree();
      makes up an area data structure, and returns a pointer to it.
 
   register_area(xy,x1,y1,x2,y2,u);
   XYTREE *xy; int x1,y1,x2,y2,u;
      This registers an area for later use.  u is a user field;
      ud1 is usually a type, and ud2 a pointer, but both are officially
      integers.  Since things are added to the tree without rebalancing,
      tree might be (very) lopsided if you do enough things without re-
      balancing.  However, insertion is always fast.
 
   unregister_area(tl, x1,y1,x2,y2, u)
   XYTREE *xy; int x1,y1,x2,y2,u;
      removes an area from further condsideration and frees the area memory.
 
   rebalance_xytree(xy)
   XYTREE **xy;
      Rebalances the tree so that searches proceed efficiently.
 
   area_apply(xy, x1,y1,x2,y2, fn)
   XYTREE *xy;
      Calls fn(ud1,ud2) on all intersecting areas.
 
   xy_apply(xy, x1, y1, x2, y2, fn, arg)
   XY_TREE *xy;
      calls fn(x, arg) on all interesting areas, where x is a (XYITEM *).
      See SDarea.h for more details.
 
   free_xytree(xy)
   XYTREE *xy;
      Calls free on all stuff that has been allocated.
 
   print_xytree(xy,n)
   XYTREE *xy;
      Prints and XYtree indented 2N spaces from left margin.
 
*/

#include <stdio.h>
#include <stdlib.h>
//#include <malloc.h>
#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
 
#include "sd2.h"
// internals from sd2.h, moved here to avoid collisions with
// FEF_area.h
//****************************************************************
//****************************************************************
//****************************************************************
 
struct xy_base {
   XYBASE *next;                        /* must be first field */
   int     xyitem_free_list_count;
   XYITEM *xyitem_free_list;
   int     xytree_free_list_count;
   XYTREE *xytree_free_list;
   void *start;                         /* first free byte */
   int bytes_left;                      /* first free byte */
   XYTREE *root;                        /* root of the tree */
   };
 
#ifndef MAXINT
#define MAXINT 0x7FFFFFFF
#endif
#ifndef MININT
#define MININT 0x80000000
#endif
 
#define ARRAY_SIZE 3
 
struct xytree {
   union a {
      struct xytree *xy;
      struct area_list_item *al;
      } ptr[ARRAY_SIZE];
   int coord;           /* coordinate where this node is split */
   int x1,y1,x2,y2;     /* bounds for all included items */
   char is_list[ARRAY_SIZE];
   char xsplit;         /* coord represents an X to split at (else a Y) */
   };
 
struct search_item {
    int addr;
    int is_list;
    } ;
 
/* size of allocated search stacks */
#define SEARCH_STACK_SIZE 100
 
typedef struct save_search SAVE_SEARCH_BUF;
struct save_search {
    int tos;
    int x1, y1, x2, y2;
    struct search_item stack[SEARCH_STACK_SIZE];
    };
 
typedef struct search_cntxt SEARCH_CONTEXT;
struct search_cntxt
{
    struct search_item search_stack[SEARCH_STACK_SIZE];
    struct search_item *tos;
    int SearchX1, SearchY1, SearchX2, SearchY2;
};
 
/* XY_THRESHOLD is the list size at which we try a split.  Empirically,
 * it matters very little between roughly 4-25.  Larger values use less
 * memory for tree nodes */
#define XY_THRESHOLD 32
#define BLOCK_SIZE 16360
 
#define LEFT       0
#define MIDDLE     1
#define RIGHT      2
 
#define FIXORDER(c1,c2) {if (c1 > c2){int t = c1; c1 = c2; c2 = t;}}
 
#define DISJOINT(ax1,ay1,ax2,ay2,a)  ((ax1 > (a)->x2 || ax2 < (a)->x1 || \
   ay1 > (a)->y2 || ay2 < (a)->y1) )
 
#define OVERLAPS(ax1,ay1,ax2,ay2,a)  ( ax1 <=(a)->x2 && ax2 >=(a)->x1 && \
   ay1 <=(a)->y2 && ay2 >=(a)->y1  )
   
#define FINDSIDE(tl,x1,y1,x2,y2,num) { \
        if (tl->xsplit) \
        { \
            if (x2 < tl->coord) num = LEFT; \
            else if(x1 > tl->coord) num = RIGHT; \
            else num = MIDDLE; \
        } \
        else /* it is split by Y value */\
        { \
            if (y2 < tl->coord) num = LEFT; \
            else if(y1 > tl->coord) num = RIGHT; \
            else num = MIDDLE; \
        } }
 
#define ADD_ITEM(t,a,b) {(++t)->addr = (int)(a); t->is_list = (b);}
 
//****************************************************************
//****************************************************************
//****************************************************************
// end of internals from sd2.h
 
void TEXT_out(char *s)
{
    printf("%s\n",s);
}
 
/* give me a pointer to n bytes of free memory from a pool. */
static void *gimme_mem(XYBASE *b,int n)
{
    void *addr;
    if (n > b->bytes_left) {    /* need to allocate new block */
        XYBASE *nb;
        nb = (XYBASE *)malloc(BLOCK_SIZE);
        nb->next = b->next;
        b->next = nb;
        b->start = (void *)((char *)nb + sizeof(XYBASE *));
        b->bytes_left = BLOCK_SIZE - sizeof(XYBASE *);
        }
    addr = b->start;
    b->start = (void *)((char *)b->start + n);
    b->bytes_left -= n;
    return addr;
}
 
static XYITEM*  gimme_new_xyitem(XYBASE *b)
{
    XYITEM *p;
    if (b->xyitem_free_list == NULL)
        return (XYITEM *)gimme_mem(b,sizeof(XYITEM));
    p = b->xyitem_free_list;
    b->xyitem_free_list = b->xyitem_free_list->next;
    b->xyitem_free_list_count--;
    return p;
}
 
static XYTREE*  gimme_new_tree_node(XYBASE *b)
{
    XYTREE *p;
    if (b->xytree_free_list == NULL)
        return (XYTREE *)gimme_mem(b,sizeof(XYTREE));
    p = b->xytree_free_list;
    b->xytree_free_list = b->xytree_free_list->ptr[0].xy;
    b->xytree_free_list_count--;
    return p;
}
 
static void free_xytree_node(XYBASE *b,XYTREE *p)
{
    p->ptr[0].xy = b->xytree_free_list;
    b->xytree_free_list = p;
    b->xytree_free_list_count++;
}
 
static void free_xyitem(XYBASE *b,XYITEM *p)
{
    p->next = b->xyitem_free_list;
    b->xyitem_free_list = p;
    b->xyitem_free_list_count++;
}
 
/* Checks the tree free lists to be sure the numbers are correct. */
int check_tree_free_list_32(XYBASE *b)
{
    XYITEM *p;
    XYTREE *q;
    int   cnt;
 
    for( p = b->xyitem_free_list, cnt = b->xyitem_free_list_count;
         p != NULL;
         p = p->next, cnt--);
    if( cnt != 0 ) printf("***-error-*** in tree_free_list cnt=%d\n", cnt);
    for( q = b->xytree_free_list, cnt = b->xytree_free_list_count;
         q != NULL;
         q = q->ptr[0].xy, cnt--);
    if( cnt != 0 ) printf("***-error-*** in tree_free_node_list cnt=%d\n", cnt);
    return cnt;
}
 
static XYTREE *
new_tree_root(XYBASE *b,int x)
{
    XYTREE *tl;
    tl = gimme_new_tree_node(b);
    /* printf("%8d new tree\n",tl); fflush(stdout); */
    tl->xsplit = TRUE;
    tl->coord = x;
    tl->ptr[0].al = NULL; tl->ptr[1].al = NULL; tl->ptr[RIGHT].al = NULL;
    tl->is_list[0] = TRUE; tl->is_list[1] = TRUE; tl->is_list[RIGHT] = TRUE;
    tl->x1 = MAXINT; tl->y1 = MAXINT;
    tl->x2 = MININT; tl->y2 = MININT;
    return tl;
}
 
 /* starts a new tree.  X should be a guess about the middle of the design,
but it is not important provided you do a balance later. */
XYBASE *
make_xytree_32(int x)
{
    XYBASE *b;
    
    b = (XYBASE *)malloc(BLOCK_SIZE);
    b->next = NULL;                     /* only one block */
 
    /* Free lists are empty to start with */
    b->xyitem_free_list_count = 0;
    b->xyitem_free_list = NULL;
    b->xytree_free_list_count = 0;
    b->xytree_free_list = NULL;
 
    /* and the rest of the initial block can be allocated. */
    b->bytes_left = BLOCK_SIZE - sizeof(XYBASE);
    b->start = (void *)((char *)b + sizeof(XYBASE));
 
    b->root = new_tree_root(b,x);
    return b;
}
 
void
register_area_32(XYBASE *b, int x1,int y1,int x2,int y2, void *u)
{
    XYTREE *tl = b->root;
    XYITEM *a, **base;
    int num;    /* should be 0, 1, or 2 */
    FIXORDER(x1,x2);
    FIXORDER(y1,y2);
    a = gimme_new_xyitem(b);
    /* printf("%8d new item\n",a); fflush(stdout); */
    a->x1 = x1; a->x2 = x2; a->y1 = y1; a->y2 = y2;
    a->ud = u;
    /* now find the right list to add this to. */
    for(;;tl = tl->ptr[num].xy) /* this for loop searches down the tree */
    {
        if (x1 < tl->x1) tl->x1 = x1; /* expand BBOX of tree node */
        if (x2 > tl->x2) tl->x2 = x2;
        if (y1 < tl->y1) tl->y1 = y1;
        if (y2 > tl->y2) tl->y2 = y2;
        FINDSIDE(tl,x1,y1,x2,y2,num);
        if (tl->is_list[num])
            break;
    }
    base = &(tl->ptr[num].al);
    a -> next = *base;
    *base = a;
}
 
/* recompute the bounding box fields of an XYTREE node.  Return TRUE if
 * they needed to be adjusted */
static int adjust_bb(XYTREE *tr)
{
    register int x1,y1,x2,y2;
    register int i;
    x1 = y1 = MAXINT;
    x2 = y2 = MININT;
    for(i=LEFT; i<= RIGHT; i++)
        if (tr->is_list[i]) {
            register XYITEM *p = tr->ptr[i].al;
            for(; p!=NULL; p = p->next) {
                if (p->x1 < x1) x1 = p->x1;
                if (p->x2 > x2) x2 = p->x2;
                if (p->y1 < y1) y1 = p->y1;
                if (p->y2 > y2) y2 = p->y2;
                }
            }
        else {
            register XYTREE *q = tr->ptr[i].xy;
            if (q->x1 < x1) x1 = q->x1;
            if (q->x2 > x2) x2 = q->x2;
            if (q->y1 < y1) y1 = q->y1;
            if (q->y2 > y2) y2 = q->y2;
            }
    if (x1 != tr->x1 || x2 != tr->x2 || y1 != tr->y1 || y2 != tr->y2) {
        tr->x1 = x1; tr->y1 = y1;
        tr->x2 = x2; tr->y2 = y2;
        return TRUE;
        }
    else
        return FALSE;   /* no changes */
}
 
#define XY_STACK_SIZE    100
#define XY_STACK_SIZE_M1  99
static XYTREE *xy_stack[XY_STACK_SIZE];
#define unsigned(a) (unsigned)(a)
 
/* delete an item from the tree.  It must be specified exactly, and it is
 * considered an error for it not to be present.  The only tricky part here
 * is recomputing the bounding boxes.  We keep an explicit stack instead of
 * doing this by recursion. */
int unregister_area_32(XYBASE *b, int x1,int y1,int x2,int y2, void *u)
{
    XYTREE *tl = b->root;
    XYITEM *l, **tr;
    XYTREE **xs;
    int num;    /* should be 0, 1, or 2 */
    char msg[256];
    FIXORDER(x1,x2);
    FIXORDER(y1,y2);
    /* printf("unregister %d %d %d %d %d\n",x1,y1,x2,y2,u);
    ** fflush(stdout); */
    xs = &xy_stack[0];
 
    /* now find the appropriate list */
    for(;;tl = tl->ptr[num].xy) /* this for loop searches down the tree */
    {
        *xs++ = tl;     /* make a stack of xytree nodes as we go.. */
        FINDSIDE(tl,x1,y1,x2,y2,num);
        if (tl->is_list[num])
            break;
    }
    /* now do a linked list delete by keeping a trailing pointer, 'tr' */
    tr = &(tl->ptr[num].al);
    for(l = *tr; l != NULL; tr = &(l->next), l = l->next)
        if (l->x1 == x1 && l->x2 == x2 && l->y1 == y1 && l->y2 == y2 &&
        l->ud == u)
        {
            *tr = l->next;
            free_xyitem(b,l);
            break;
        }
    if (l == NULL) {
        (void)sprintf(msg,
        "Unregister_area - %d %d %d %d %d",x1,y1,x2,y2,unsigned(u));
        TEXT_out(msg);
        return FALSE;
        }
    if (xs > &xy_stack[XY_STACK_SIZE_M1]) {
        (void)printf("Stack too big in unregister_area\n");
        exit(2);
        }
    for(xs--; xs >= &xy_stack[0]; xs--)  /* work way back up the stack */
        if (!adjust_bb(*xs))        /* and quit if we don't need to adjust */
            return TRUE;
    return TRUE;
}
/*********************************************************/
/* check only (no deletion) */
/* FALSE if area is unregistered */
int unregister_area_check_32(XYBASE *b, int x1,int y1,int x2,int y2, void *u)
{
    XYTREE *tl = b->root;
    XYITEM *l, **tr;
    int num;    /* should be 0, 1, or 2 */
 
    FIXORDER(x1,x2);
    FIXORDER(y1,y2);
    /* now find the right list */
    for(;;tl = tl->ptr[num].xy) /* this for loop searches down the tree */
    {
        FINDSIDE(tl,x1,y1,x2,y2,num);
        if (tl->is_list[num])
            break;
    }
    /* now do a linked list search by keeping a trailing pointer, 'tr' */
    tr = &(tl->ptr[num].al);
    for(l = *tr; l != NULL; tr = &(l->next), l = l->next)
        if (l->x1 == x1 && l->x2 == x2 && l->y1 == y1 && l->y2 == y2 &&
        l->ud == u)
            return TRUE;
 
    return FALSE;
}
/*********************************************************/
 
#define SEARCHCALL_U(a,start,fn) { \
            for(a = start; a != NULL; a = a->next) \
                if (!DISJOINT(x1,y1,x2,y2,a)) \
                    (*fn)(a->ud); }
 
/* requires x1,x2 and y1, y2 be in the correct order. */
static void sorted_area_apply(XYTREE *tl,int x1,int y1,int x2,int y2,int (*fn)(void *))
{
    XYITEM *a;
    int num;
    if (tl == NULL || DISJOINT(x1,y1,x2,y2, tl))
        return;
    /* first search the list of areas */
    FINDSIDE(tl,x1,y1,x2,y2,num);
    if (num == LEFT || num == MIDDLE) /* search left side */
    {
        if (tl->is_list[LEFT])
            SEARCHCALL_U(a, tl->ptr[LEFT].al,fn)
        else
            sorted_area_apply(tl->ptr[LEFT].xy,x1,y1,x2,y2,fn);
    }
    /* note: you must always search the middle list */
    if (tl->is_list[MIDDLE])
        SEARCHCALL_U(a, tl->ptr[MIDDLE].al,fn)
    else
        sorted_area_apply(tl->ptr[MIDDLE].xy,x1,y1,x2,y2,fn);
    if (num == MIDDLE || num == RIGHT)
    {
        if (tl->is_list[RIGHT])
            SEARCHCALL_U(a, tl->ptr[RIGHT].al,fn)
        else
            sorted_area_apply(tl->ptr[RIGHT].xy,x1,y1,x2,y2,fn);
    }
}
 
/* generic routine - x1,x2 and y1,y2 in any order */
void area_apply_32(XYBASE *b, int x1,int y1,int x2,int y2, int (*fn)(void *))
{
    XYTREE *tl = b->root;
    FIXORDER(x1,x2);
    FIXORDER(y1,y2);
    sorted_area_apply(tl, x1,y1,x2,y2, fn);
}
 
#define SEARCHCALL_XY(a,start,fn) { \
            for(a = start; a != NULL; a = a->next) \
                if (!DISJOINT(x1,y1,x2,y2,a)) \
                    {if ((*fn)(a, arg)) return TRUE; } }
 
/* applies the function fn() to all the XYITEMS in the area. */
/* requires x1,x2 and y1, y2 be in the correct order. */
/* if fn() returns non-zero, the search is aborted.  */
/* in turn, xy_apply() returns non-zero if the search is aborted */
static int
sorted_xy_apply(
 XYTREE *tl,                    /* Tree to be searched */
 int x1,int y1,int x2,int y2,   /* bounds to search */
 int (*fn)(XYITEM *, void *),   /* function to call on all objects found */
 void *arg)                     /* another arg to pass to function */
{
    XYITEM *a;
    int num;
    if (tl == NULL || DISJOINT(x1,y1,x2,y2, tl))
         return FALSE;
    /* first search the list of areas */
    FINDSIDE(tl,x1,y1,x2,y2,num);
    if (num == LEFT || num == MIDDLE) /* search left side */
    {
        if (tl->is_list[LEFT])
            SEARCHCALL_XY(a, tl->ptr[LEFT].al,fn)
        else
            { if(sorted_xy_apply(tl->ptr[LEFT].xy,x1,y1,x2,y2,fn,arg)) return TRUE;}
    }
    /* note: you must always search the middle list */
    if (tl->is_list[MIDDLE])
        SEARCHCALL_XY(a, tl->ptr[MIDDLE].al,fn)
    else
        {if (sorted_xy_apply(tl->ptr[MIDDLE].xy,x1,y1,x2,y2,fn,arg)) return TRUE;}
    if (num == MIDDLE || num == RIGHT)
    {
        if (tl->is_list[RIGHT])
            SEARCHCALL_XY(a, tl->ptr[RIGHT].al,fn)
        else
            {if(sorted_xy_apply(tl->ptr[RIGHT].xy,x1,y1,x2,y2,fn,arg)) return TRUE;}
    }
    return FALSE;       /* search OK so far */
}
 
int xy_apply_32(
 XYBASE *b,
 int x1,int y1,int x2,int y2,
 int (*fn)(XYITEM *, void *),
 void *arg)
{
    XYTREE *tl = b->root;
    FIXORDER(x1,x2);
    FIXORDER(y1,y2);
    return sorted_xy_apply(tl, x1, y1, x2, y2, fn, arg);
}
 
/* frees the whole tree */
void free_xytree_32(XYBASE *b)
{
    XYBASE *next;
    for(; b != NULL; b = next) {
        next = b->next;
        free((char *)b);
        }
}
 
#define RETURN_LL {*what = TRUE; return (int)ll;}
/* this routine does the dirty work of rebalancing the tree.  It is
 not directly callable by the user.  It is given a linked list of items
 starting with TL.
 If the result should be a linked list, what = TRUE and it returns a
 pointer to the first item in the list.  If the result should be a tree,
 it mallocs the node, and returns a pointer to it with what = FALSE;
 */
static int rebalance_xytree_guts(XYBASE *tb, XYITEM *tl, char *what)
{
    int x1=MAXINT, y1 = MAXINT, x2 = MININT, y2 = MININT;
    int n = 0;          /* number of items in linked list */
    XYTREE *b;  /* pointer to new tree node, if needed */
    int xmid, ymid;     /* midpoints for potential dividing lines */
    XYITEM *ll = tl, *l, *next;
    int i;
    int fom_x, fom_y;   /* figures of merit for splitting x or y */
    /* next, count the items in the linked list, and compute bounds */
    for(l=ll; l != NULL; l = l->next)
    {
        n++;
        if (l->x1 < x1) x1 = l->x1;
        if (l->x2 > x2) x2 = l->x2;
        if (l->y1 < y1) y1 = l->y1;
        if (l->y2 > y2) y2 = l->y2;
    }
    /* printf("got %d items, (%d %d) (%d %d)\n",n,x1,y1,x2,y2);
      fflush(stdout); */
    /* if too few items, return a simple linked list */
    if (n < XY_THRESHOLD)
        RETURN_LL;
    /* compute potential x and y splits */
    xmid = (x1+x2)/2;
    ymid = (y1+y2)/2;
    x1 = x2 = y1 = y2 = 0;
    for(l=ll; l != NULL; l = l->next)
    {
        if (l->x2 < xmid) x1++;
        if (l->x1 > xmid) x2++;
        if (l->y2 < ymid) y1++;
        if (l->y1 > ymid) y2++;
    }
    /* printf("splits %d %d %d %d\n",x1,x2,y1,y2);fflush(stdout); */
    /* pick the better split.  A split is good if the two sides are balanced
     * and there are few things in the middle.  Since all things are either
     * on one side or in the middle, N-X1-X2 counts the number in the middle.
     * the expected search time is x1*log(x1)+x2*log(x2)+xm*log(xm).  We use
     * pseudo logs = number of shifts till we get 0 .
     * fom = figure_of_merit (we want the lowest)
     * ttl = time to lookup.  roughly log(x), base 2, but ttl(0)==1.
     * note - the middle should count more, since we have to search
     * it more often (in fact on every search) hence we use n as the multiplier,
     * not (n-x1-x2) as you might expect */
     /*
    fom_x = x1 * ttl(x1) + n*ttl(n-x1-x2) + x2*ttl(x2);
    fom_y = y1 * ttl(y1) + n*ttl(n-y1-y2) + y2*ttl(y2);
     */
    fom_x = n-x1-x2;
    fom_y = n-y1-y2;
    if (fom_x <= fom_y)
    {
        if (x1 + x2 < XY_THRESHOLD)
            RETURN_LL;
        b = new_tree_root(tb, 0);
        b->xsplit = TRUE;
        b->coord = xmid;
        for(l=ll; l != NULL; l = next)
        {
            next = l->next;
            if (l->x2 < xmid){ l->next = b->ptr[0].al; b->ptr[0].al = l;}
            else if (l->x1 > xmid){ l->next = b->ptr[2].al; b->ptr[2].al = l;}
            else { l->next = b->ptr[1].al; b->ptr[1].al = l;}
        }
    }
    else /* split in the Y direction */
    {
        if (y1 + y2 < XY_THRESHOLD)
            RETURN_LL;
        b = new_tree_root(tb, 0);
        b->xsplit = FALSE;
        b->coord = ymid;
        for(l=ll; l != NULL; l = next)
        {
            next = l->next;
            if (l->y2 < ymid){ l->next = b->ptr[0].al; b->ptr[0].al = l;}
            else if (l->y1 > ymid){ l->next = b->ptr[2].al; b->ptr[2].al = l;}
            else { l->next = b->ptr[1].al; b->ptr[1].al = l;}
        }
    }
    /* now rebalance the subtrees */
    for(i=LEFT; i<= RIGHT; i++)
        b->ptr[i].xy = (XYTREE *)rebalance_xytree_guts(tb, 
        b->ptr[i].al, &(b->is_list[i]));
    *what = FALSE;
    return (int)b;
}
 
/* take the tree pointed to by T, and add it to the linked list at ll */
static void Tree_to_linked_list(XYTREE *t, XYITEM **ll)
{
    XYITEM *l, *next;
    int i;
    for(i=LEFT; i<=RIGHT; i++)
        if (t->is_list[i])
        {
            for(l = t->ptr[i].al; l != NULL; l = next) 
            {
                next = l->next;
                l->next = *ll;
                *ll = l;
            }
        }
        else
            Tree_to_linked_list(t->ptr[i].xy, ll);
}
 
/* free the XY nodes of the tree.  if FREE_ROOT, free the root node */
static void free_xynodes(XYBASE *tb, XYTREE *b, int free_root)
{
    int i;
    for(i=LEFT; i<=RIGHT; i++)
        if (!b->is_list[i])
            free_xynodes(tb, b->ptr[i].xy, TRUE);
    /* printf("%8d free tree\n",b); fflush(stdout); */
    if (free_root)
        free_xytree_node(tb, b);
}
 
static void recompute_all_bbs(XYTREE *p)
{
    int i;
    /* first, recompute all subtrees */
    for(i=LEFT; i<=RIGHT; i++)
        if (!p->is_list[i])
            recompute_all_bbs(p->ptr[i].xy);
    /* and then fix yourself, accounting for any lists */
    (void)adjust_bb(p);
}
 
void rebalance_xytree_32(XYBASE *b)
{
    XYITEM *ll = NULL;
    char what;
    int temp;
    /* first, turn the tree into a linked list */
    Tree_to_linked_list(b->root, &ll);
    /* then free the tree structures */
    free_xynodes(b, b->root, TRUE /* TRUE means to free the root node */);
    /* then rebalance the tree */
    temp = rebalance_xytree_guts(b, ll,&what);
    if (what == FALSE) /* a tree was returned */
        b->root = (XYTREE *)temp;
    else        /* a linked list was returned.  make a fake node */
    {
        b->root = new_tree_root(b, MININT);
        b->root->ptr[RIGHT].al = (XYITEM *)temp;
    }
    recompute_all_bbs(b->root);
}

/*
static void pr(int u)
{ (void)printf("called with %d\n",u); }
*/
 
#define PBL(n) {int j; for(j=0; j<n; j++) (void)printf(" "); }
static void print_xytree_internal(XYTREE *b,int n)
{
     int i;
     XYITEM *l;
     PBL(n); (void)printf("split in %c at %d  bbox=(%d %d) (%d %d)\n",
     b->xsplit?'X':'Y', b->coord, b->x1, b->y1, b->x2, b->y2);
     for(i=0; i<3; i++)
     {
        PBL(n+1);
        (void)printf("branch %d is a %s\n",i,b->is_list[i]?"list":"tree");
        if (b->is_list[i])
        {
            for(l=b->ptr[i].al; l != NULL; l = l->next)
            {
                PBL(n+1);
                (void)printf( "(%d %d) (%d %d) %d\n",
                l->x1, l->y1, l->x2, l->y2, unsigned(l->ud));
            }
        }
        else
            print_xytree_internal(b->ptr[i].xy,n+2);
    }
}
 
void print_xytree_32(XYBASE *b, int n)
{
    print_xytree_internal(b->root, n);
}
 
#define SEARCH_STACK_SIZE 100
#define SEARCH_STACK_LIMIT 96
static struct search_item *tos;
static struct search_item search_stack[SEARCH_STACK_SIZE];
static int sx1,sy1,sx2,sy2; /* globals for searching */
 
/* returns TRUE if it found something else, FALSE if not */
int
DB_get_next_xyitem_32(XYITEM **ptr)
{
    for(; tos >= search_stack; )
    {
        if (!tos->is_list)      /* it is a tree */
        {
            int s;
            XYTREE *t = (XYTREE *)((tos--)->addr);
            if (DISJOINT(sx1,sy1,sx2,sy2,t)) /* check search area against */
                continue;                    /* bbox of stuff in tree */
            if (tos > &search_stack[SEARCH_STACK_LIMIT])
            {
                (void)printf("search stack too deep - some ignored");
                tos = &search_stack[SEARCH_STACK_LIMIT]; /*why?*/
            }
            FINDSIDE(t,sx1,sy1,sx2,sy2, s);
            if (s != RIGHT)
                ADD_ITEM(tos,t->ptr[LEFT].al, t->is_list[LEFT]);
            ADD_ITEM(tos,t->ptr[MIDDLE].al, t->is_list[MIDDLE]);
            if (s != LEFT)
                ADD_ITEM(tos,t->ptr[RIGHT].al, t->is_list[RIGHT]);
         }
        else /* item is a list.  search it for OK values */
        {
            register XYITEM *l;
            for(l=(XYITEM *)(tos->addr); l != NULL; l = l->next)
                if (!DISJOINT(sx1,sy1,sx2,sy2,l) )
                {
                    *ptr = l;
                    tos->addr = (int)(l->next);
                    return TRUE;
                }
         /* nothing in this list. decrement stack and try again */
            tos--;
         }
    }
    return FALSE;
}
 
void DB_set_xytree_search_box_32(XYBASE *b ,int x1,int y1,int x2,int y2)
{
    sx1 = x1; sy1 = y1;
    sx2 = x2; sy2 = y2;
    FIXORDER(sx1,sx2);
    FIXORDER(sy1,sy2);
    tos = search_stack;
    tos->addr = (int)(b->root);
    tos->is_list = FALSE;
}
 
#ifdef STANDALONE
TEXT_out(a)
char *a;
{
    (void)printf("%s\n",a);
}
int local_fn(a,b)
 int a,b;
{
}
 
#define MAXi  3   
#define MAXj 1000
main()
{
    int i;
    for (i = 0; i < MAXi; i++)
    {
        XYBASE *t;
        /* printf("start...\n"); fflush(stdout); */
        t = make_xytree(4);
        register_area(t,2,2,6,6,0);
        register_area(t,5,5,10,10,1);
        register_area(t,3,10,5,15,2);
        print_xytree(t,0);
        unregister_area(t,5,5,10,10,1);
        check_tree_free_list(t);
        print_xytree(t,0);
        register_area(t,7,8,8,8,3);
        register_area(t,1,8,3,8,4);
        register_area(t,0,8,10,8,5);
        register_area(t,4,0,4,12,6);
        register_area(t,1,4,3,4,7);
        check_tree_free_list(t);
        /* now add MAXj things to the tree */
        {
            int x1,y1,x2,y2;
            int j;
            for(j=0; j<MAXj; j++)
            {
                int temp;
                x1 = random() & 0xFFFF;
                y1 = random() & 0xFFFF;
                x2 = random() & 0xFFFF;
                y2 = random() & 0xFFFF;
                if (x1 > x2) { temp = x1; x1 = x2; x2 = temp;}
                if (y1 > y2) { temp = y1; y1 = y2; y2 = temp;}
                temp = random();
                if ((temp & 0xFF) == 0)
                    register_area(t, x1, y1, x2, y2, j, j);
                else if (temp & 4)
                    register_area(t, x1, y1, x1, y2,j,j);
                else
                    register_area(t, x1, y1, x2, y1,j,j);
            }
        }
        (void)printf("rebalance\n"); fflush(stdout);
        rebalance_xytree(t);
        /* we do 10 lookups for every add... */
        {
            int x1,y1,x2,y2;
            int j;
            for(j=0; j<10*MAXj; j++)
            {
                int temp;
                x1 = random() & 0xFFFF;
                y1 = random() & 0xFFFF;
                x2 = random() & 0xFFFF;
                y2 = random() & 0xFFFF;
                if (x1 > x2) { temp = x1; x1 = x2; x2 = temp;}
                if (y1 > y2) { temp = y1; y1 = y2; y2 = temp;}
                temp = random();
                if ((temp & 0xFF) == 0)
                    /* do nothing */;
                else if (temp & 4)
                    x2 = x1;
                else
                    y2 = y1;
                area_apply(t, x1, y1, x2, y2, local_fn);
            }
        }
        unregister_area(t,2,2,6,6,0,0);
        unregister_area(t,5,5,10,10,1,1);
        unregister_area(t,3,10,5,15,2,2);
        unregister_area(t,7,8,8,8,3,3);
        unregister_area(t,1,8,3,8,4,4);
        unregister_area(t,0,8,10,8,5,5);
        unregister_area(t,4,0,4,12,6,6);
        unregister_area(t,1,4,3,4,7,12);        /* should fail */
        /* print_xytree(t, 0); */
        /* for(;;)
        {
            int x1,y1,x2,y2;
            (void)printf("Enter coords:");
            scanf("%d %d %d %d",&x1,&y1,&x2,&y2);
            area_apply(t,x1,y1,x2,y2,pr);
            if (x1 < 0) break;
        } */
        /* printf("free:\n"); fflush(stdout); */
        free_xytree(t);
    }
}
#endif

