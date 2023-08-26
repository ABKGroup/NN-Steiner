#CC	= CC
CC	= g++
#CC	= purify g++

#DBG_FLAGS  = -g
#DBG_FLAGS  = -DDEBUG -g
#DBG_FLAGS  = -DDEBUG
OPT_FLAGS  = -O3
#OPT_FLAGS  = 
CFLAGS	= -Wall $(DBG_FLAGS) $(OPT_FLAGS)
LIBS	= -lm

SRCS0	= main.c unixtimer.c err.c dist.c mst1.c\
          neighbors.c heap.c mst2.c\
          sd2.c mst8.c
OBJS0	= $(SRCS0:.c=.o)
MAIN0	= rmst

SRCS1	= gen.c
OBJS1	= $(SRCS1:.c=.o)
MAIN1	= gen


rmst: 	${OBJS0}
	$(CC) -o ${MAIN0} ${OBJS0} $(LIBS)

gen: 	${OBJS1}
	$(CC) -o ${MAIN1} ${OBJS1} $(LIBS)

new:    clean rmst gen

test:
	@echo "Testing Prim's algorithm..."
	@rmst -l -p < sample.in 
	@echo "Testing Scheffer's algorithm..."
	@rmst -l -s < sample.in
	@echo "Testing Guibas and Stolfi's algorithm..."
	@rmst -l -g < sample.in
	@echo "Expected RMST length: 57303366.000000"

clean:
	rm -f *.o
	rm -f ${MAIN0}
	rm -f ${MAIN1}
	rm -f core

