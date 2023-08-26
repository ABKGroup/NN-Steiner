#include <sys/time.h>

void start_timer()
{
  struct itimerval pval, poval;

  /*Now set the process timer*/
  pval.it_value.tv_sec = 1000000;
  pval.it_value.tv_usec = 0;
  pval.it_interval.tv_sec = 0;
  pval.it_interval.tv_usec = 0;
  setitimer(ITIMER_PROF, &pval, &poval);
}

/* Returns the number of process seconds since last call to start_timer() */
double cpu_seconds() 
{
  struct itimerval val;
  long secs, usecs;
  double temp;

  getitimer(ITIMER_PROF, &val);
  secs = 1000000 - val.it_value.tv_sec - 1;
  usecs = 1000000 - val.it_value.tv_usec;

  temp = (double) secs + ((double) usecs/1000000.0);

  return(temp);
}

