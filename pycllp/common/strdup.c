/*********************************************************************/
/***    Copyright (c) Robert J. Vanderbei, 1994                    ***/
/***    All Rights Reserved                                        ***/
/*********************************************************************/

/*	@(#)my_strdup.c	1.2	*/
/*LINTLIBRARY*/
/* string duplication
   returns pointer to a	new string which is the	duplicate of string
   pointed to by s1
   NULL	is returned if new string can't	be created
*/

#include <stdlib.h>
#include <string.h>
#include "myalloc.h"
#ifndef	NULL
#define	NULL	0
#endif

char *my_strdup(
	char	*s1
) 
{  
	char *s2;

	MALLOC(s2, strlen(s1)+1, char);

	if ( s2	== NULL	) {
		return NULL;
	} else {
		return (char *)strcpy(s2, s1);
	}
}
