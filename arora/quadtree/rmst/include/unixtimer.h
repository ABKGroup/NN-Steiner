#ifndef _UNIXTIMER_H_
#define _UNIXTIMER_H_

/**********************************************************************
 * Starts Unix timer 
 */
void start_timer();

/**********************************************************************
 * Returns number of process seconds since last call to start_timer()
 */
double cpu_seconds();

#endif
