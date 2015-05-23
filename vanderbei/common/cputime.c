/*********************************************************************/
/***    Copyright (c) Robert J. Vanderbei, 1994                    ***/
/***    All Rights Reserved                                        ***/
/*********************************************************************/

/* This is a function to return the elasped CPU time since this process
 * started running.  
 */

#include <time.h>

double cputimer()
{
  double elaptime;

  elaptime = ((double)clock()) / CLOCKS_PER_SEC;

  return elaptime;
}

