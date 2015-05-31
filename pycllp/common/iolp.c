/*********************************************************************/
/***    Copyright (c) Robert J. Vanderbei, 1994                    ***/
/***    All Rights Reserved                                        ***/
/*********************************************************************/

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef QuadPrec
#include "Quad.h"
#define double Quad
#else
#define high(x) (x)
#endif

#include "lp.h"
#include "hash.h"
#include "myalloc.h"

/* Prototype static functions */

static void error(
	int	num,
	char	*text
);

static void warn(
	int	num,
	char	*text
);

static int newstate(
	char	*line
);

static char *my_strstr(
	char	*s,
	char	*t
);

static void qksort(
	int	*v,
	double	*data,
	int	left,
	int	right
);

static void swap(
	int	*v, 
	double	*data,
	int	i, 
	int	j
);

/* Define functions */

LP	*openlp(void)
{
	LP *lp;

	MALLOC(lp, 1, LP);

	if (lp != NULL) {
		lp->m = 0;
		lp->n = 0;
		lp->nz = 0;
		lp->qnz = 0;
		lp->name[0]   ='\0';
		lp->obj[0]    ='\0';
		lp->rhs[0]    ='\0';
		lp->bounds[0] ='\0';
		lp->ranges[0] ='\0';
		lp->f = 0.0;
		lp->r =	NULL;
		lp->l =	NULL;
		lp->u =	NULL;
		lp->iQ = NULL;
		lp->kQ = NULL;
		lp->Q =	NULL;
		lp->w = NULL;
		lp->x = NULL;
		lp->y = NULL;
		lp->z = NULL;
		lp->kAt = NULL;
		lp->iAt = NULL;
		lp->At  = NULL;
		lp->rowlab = NULL;
		lp->collab = NULL;
		lp->varsgn = NULL;
		lp->tier = NULL;
		lp->max	    = 1;     /*	max = -1,   min	= 1 */
		lp->sf_req  = 8;     /*	significant figures requested */
		lp->itnlim  = 200;   /* iteration limit */
		lp->timlim  = HUGE_VAL; /* time limit */
		lp->verbose = 2;     /*	verbosity level	*/
		lp->inftol  = 1.0e-5;/*	infeasibility requested	*/
		lp->init_vars	  = deflt_hook;
		lp->h_init        = deflt_hook;
		lp->h_update 	  = deflt_hook;
		lp->h_step   	  = deflt_hook;
	}

	return lp;
}

void	closelp(
	LP	*lp
)
{
	int i, j, m, n,	np;

	n = lp->n; m= lp->m; np	= lp->np;

	FREE(lp->A);
	FREE(lp->iA);
	FREE(lp->kA);
	FREE(lp->At);
	FREE(lp->iAt);
	FREE(lp->kAt);
	FREE(lp->b);
	FREE(lp->c);
	FREE(lp->r);
	FREE(lp->l);
	FREE(lp->u);
	FREE(lp->varsgn);
	FREE(lp->Q);
	FREE(lp->iQ);
	FREE(lp->kQ);
	FREE(lp->w);
	FREE(lp->x);
	FREE(lp->y);
	FREE(lp->z);
	if (lp->rowlab != NULL) for (i=0; i<m; i++)  FREE(lp->rowlab[i]);
	if (lp->collab != NULL) for (j=0; j<n; j++)  FREE(lp->collab[j]);
	if (lp->param  != NULL) for (j=0; j<np;	j++) FREE(lp->param[j]);
	FREE(lp->rowlab);
	FREE(lp->collab);
	FREE(lp->param);

	FREE(lp);
}

void	readlp(
	int	argc,
	char	*argv[],
	LP	*lp
)
{
	int		np, m, n, nz, qnz, *iA,	*kA, *iQ, *kQ, maxflag,	sf_req,	
			verbose, itnlim, *varsgn;
	double		*A, *b,	*c, f=0.0, *r, *l, *u, *Q, *diagQ, 
			inftol, timlim;
	char		**rowlab, **collab, **param;
	char		name[256], obj[256], rhs[256], bounds[256],ranges[256];

	int	row, i,	j, k, j_previous = -1, argno, real_int_flg = 1;
	FILE	*fp;
	char	line[240], str[256], s[2][256];
	char	*type, *label0,	*label1, *label2, *valstr1, *valstr2;
	int	len=0, state=HEADER;
	int	np2, m2, n2, nz2, qnz2;
	int	*mark;
	double	value;

	int	nkeys =	11;
	static char    *key_word[] = { "MAX", "SIGFIG",	"INFTOL", "MIN",
					"OBJ", "RHS", "RANGES",	"BOUNDS",
					"VERBOSE", "ITNLIM", "TIMLIM"};
	static char	*label[] = {
			"Maximize (MAX)",
			"Significant digits (SIGFIG).................",
			"Feasibility tolerance (INFTOL)..............",
			"Minimize (MIN)",
			"Objective vector (OBJ)......................",
			"Right hand side (RHS).......................",
			"Ranges (RANGES).............................",
			"Bounds (BOUNDS).............................",
			"Verbosity level (VERBOSE)...................",
			"Iteration limit (ITNLIM)....................",
			"Time limit (in seconds) (TIMLIM)............"
			};

	char *fmt = "%s \n",
	     *fmts = "%s %s \n",
	     *fmte = "%s %e \n",
	     *fmtd = "%s %d \n";

	/* initialize parameters */

	m = 0; 
	n = 0; 
	nz = 0;	
	qnz = 0;
	np = 0;
	maxflag	 = lp->max;	   /* max = -1,	  min =	1 */
	sf_req	 = lp->sf_req;	   /* significant figures requested */
	verbose	 = lp->verbose;	   /* verbosity	level */
	itnlim   = lp->itnlim;     /* iteration limit */
	timlim   = lp->timlim;     /* time limit */
	inftol	 = lp->inftol;	   /* infeasibility requested */

	name[0]	  = '\0';
	obj[0]	  = '\0';
	rhs[0]	  = '\0';
	bounds[0] = '\0';
	ranges[0] = '\0';

	/*---------------------------------------------------------+
	|	compute	number of bytes	in input files		  */

	for (argno=0; argno<argc; argno++) {
		struct stat buf;
		if ( stat(argv[argno],&buf) != 0 ) buf.st_size = 400000;
		len = len + (int)buf.st_size;
	}


	hash_init( len/17 );
	m2 = len/100+1;	n2 = len/100+1;	nz2 = len/100+1; qnz2 =	len/1000+1;
	np2 = 10;

	MALLOC(	mark,	m2,   int );
	MALLOC(	iA,	nz2,  int );
	MALLOC(	kA,	n2,   int );
	MALLOC(	iQ,	qnz2, int );
	MALLOC(	 A,	nz2,  double );
	MALLOC(	 Q,	qnz2, double );
	MALLOC(	 r,	m2,   double );
	MALLOC(	 u,	n2,   double );
	MALLOC(	 rowlab, m2,  char * );
	MALLOC(	 collab, n2,  char * );
	MALLOC(	 varsgn, n2,  int );
	MALLOC(	 param,	np2,  char * );

	type	= line + 1;
	label0	= line + 4;
	label1	= line + 14;
	valstr1	= line + 24;
	label2	= line + 39;
	valstr2	= line + 49;
	s[0]  [255] = '\0';
	s[1]  [255] = '\0';
	line  [79] = '\0';

	for (argno=0; argno<argc; argno++) {
	  if ( strcmp(argv[argno],"-") == 0 ) fp = stdin;
	  else                                fp = fopen(argv[argno],"r");
	  if ( fp == NULL ) error(2,argv[argno]);

	  while	( fgets	( line,	240, fp	) != NULL ) {
	   
	      if (line[0] == '*') continue;

	      len = strlen(line);
	      for (j=len-1; j<79; j++) line[j]=' ';
	      if ( state != HEADER ) {
		line[3]='\0'; line[12]='\0'; line[22]='\0'; line[36]='\0'; 
		line[47]='\0'; line[61]='\0'; line  [79] = '\0';
	      }

	      switch(state) {
	      case HEADER:
		sscanf(line, "%s%s",s[0],s[1]);
		if ( strncmp ( s[0], "NAME", 4 ) == 0 )	{
		  strncpy (name, s[1], 256);
		  state	= NAME;
		} else {
		  strcpy(str,s[0]);
		  strcat(str,"P");
		  param[np] = my_strdup(s[1]);
		  install(str,np);
		  np++;
		  if (np >= np2) {
		    np2	= 2*np2;
		    REALLOC( param, np2, char * );
		  }
		  for (	i = 0;	i < nkeys;  ++i	) {
		    if ( strcmp ( s[0], key_word[i] ) == 0 )
		    {
			switch ( i )
			{
			    case 0:
			       maxflag = -1;
			       if (verbose>1) {
				 printf(fmt,label[0]); fflush(stdout);
			       }
			       break;
			    case 1:
			       sf_req =	atoi ( s[1] );
			       if (verbose>1) {
				 printf(fmtd,label[1],sf_req); fflush(stdout);
			       }
			       break;
			    case 2:
			       inftol =	atof ( s[1] );
			       if (verbose>1) {
				 printf(fmte,label[2],inftol); fflush(stdout);
			       }
			       break;
			    case 3:
			       maxflag = 1;
			       if (verbose>1) {
				 printf(fmt,label[3]); fflush(stdout);
			       }
			       break;
			    case 4:
			       strncpy ( obj, s[1], 256 );
			       if (verbose>1) {
				 printf(fmts,label[4],obj); fflush(stdout);
			       }
			       break;
			    case 5:
			       strncpy ( rhs, s[1], 256 );
			       if (verbose>1) {
				 printf(fmts,label[5],rhs); fflush(stdout);
			       }
			       break;
			    case 6:
			       strncpy ( ranges, s[1], 256 );
			       if (verbose>1) {
				 printf(fmts,label[6],ranges); fflush(stdout);
			       }
			       break;
			    case 7:
			       strncpy ( bounds, s[1], 256 );
			       if (verbose>1) {
				 printf(fmts,label[7],bounds); fflush(stdout);
			       }
			       break;
			    case 8:
			       verbose = atoi (	s[1] );
			       if (verbose>1) {
			       printf(fmtd,label[8],verbose); fflush(stdout);
			       }
			       break;
			    case 9:
			       itnlim = atoi (	s[1] );
			       if (verbose>1) {
			       printf(fmtd,label[9],itnlim); fflush(stdout);
			       }
			       break;
			    case 10:
			       timlim = atof (	s[1] );
			       if (verbose>1) {
			       printf(fmte,label[10],timlim); fflush(stdout);
			       }
			       break;
			}
		    }
		  }
		}
		break;
	      case NAME:
		if ( strcmp ( line, "ROW" ) == 0 ) state = ROWS;
		else warn(20,line);
		break;
	      case ROWS:
		if ( line[0] !=	' ') {
			if ( strcmp ( line, "COL" ) == 0 ) {
			    state = COLS;
			    CALLOC( b, m, double );
			} else warn(21,type);
		} else {
			switch(type[0] == ' ' ?	type[1]	: type[0]) {
				case 'L':
					r[m] = HUGE_VAL;
					mark[m]	= 1;
					break;
				case 'E':
					r[m] = 0.0;
					mark[m]	= 0;
					break;
				case 'G':
					r[m] = HUGE_VAL;
					mark[m]	= 0;
					break;
				case 'N':
					if (obj[0] == '\0')
						strncpy	(obj, label0, 256);
					if ( my_strstr(label0, obj) != NULL ) 
						strncpy	(obj, label0, 256);
					mark[m]	= 2;
			}

			rowlab[m] = my_strdup(label0);
			strcpy(str,label0);
			strcat(str,"R");
			install(str,m);
			m++;
			if (m >= m2) {
			    m2 = 2*m2;
			    REALLOC(  r,      m2, double );
			    REALLOC( mark,    m2, int );
			    REALLOC(  rowlab, m2, char * );
			}
		}
		break;

	      case COLS:

		if ( line[0] !=	' ' ) {
			CALLOC(	c,	n, double );
			CALLOC(	diagQ,	n, double );
			CALLOC(	l,	n, double );
			MALLOC(	kQ,   n+1, int );

			state =	newstate(line);
		} else {
		    strcpy(str,label0);
		    strcat(str,"C");
		    if (getindex(str) != -1) {	/* seen	this col before	*/
			  /* check to see if agrees with previous col */
			  if ( strcmp( collab[n-1], label0 ) != 0 )
			      error(35,label0);
		    } else {			/* new col label */
			if ( strcmp(label1,"'MARKER'") == 0 ) {
			  real_int_flg = 3 - real_int_flg; /* swap 1 and 2 */
			} else {
			  kA[n]	= nz;
			  collab[n] = my_strdup(label0);
			  varsgn[n] = real_int_flg;
			  u[n]	    = HUGE_VAL;
			  install(str,n);
			  n++;
			  if (n	>= n2) {
			    n2 = 2*n2;
			    REALLOC( kA,     n2, int );
			    REALLOC( collab, n2, char *	);
			    REALLOC( varsgn, n2, int );
			    REALLOC( u,	     n2, double	);
			  }
			}
		    }

		    if ( len >=	25 ) {
		       value = atof(valstr1);
		       if (value != 0.0) {
			  strcpy(str, label1);
			  strcat(str,"R");
			  i = getindex(str);
			  if (i	!= -1) {
			    iA[nz] = i;	A[nz] =	value;
			    nz++;
			    if (nz >= nz2) {
			      nz2 = 2*nz2;
			      REALLOC( iA, nz2,	int );
			      REALLOC(	A, nz2,	double );
			    }
			  } else warn (30,label1);
		       }
		    } 
		    if ( len >=	50 ) {
		       value = atof(valstr2);
		       if (value != 0.0) {
			  strcpy(str, label2);
			  strcat(str,"R");
			  i = getindex(str);
			  if (i	!= -1) {
			    iA[nz] = i;	A[nz] =	value;
			    nz++;
			    if (nz >= nz2) {
			      nz2 = 2*nz2;
			      REALLOC( iA, nz2,	int );
			      REALLOC(	A, nz2,	double );
			    }
			  } else warn (30,label2);
		       }
		    }
		}
		break;

	      case RHS:

		if ( line[0] !=	' ' ) {
			state =	newstate(line);
		} else {
		    if (rhs[0] == '\0' ) strncpy (rhs, label0, 256);
		    if ( my_strstr(label0, rhs)	!= NULL	) {
			if ( len >= 50 ) {
			  value	= atof(valstr2);
			  if (value != 0.0) {
			    strcpy(str,	label2);
			    strcat(str,"R");
			    i =	getindex(str);
			    if (i != -1) {
			      b[i] = value;
			    } else warn	( 31, label2 );
			  }
			}
			value =	atof(valstr1);
			if (value != 0.0) {
			    strcpy(str,	label1);
			    strcat(str,"R");
			    i =	getindex(str);
			    if (i != -1) {
			      b[i] = value;
			    } else warn	( 31, label1 );
			}
		    }
		}
		break;

	      case RNGS:

		if ( line[0] !=	' ' ) {
			state =	newstate(line);
		} else {
		    if (ranges[0]=='\0') strncpy(ranges,label0,256);
		    if ( my_strstr(label0, ranges) != NULL ) {
		       if ( len	>= 50 )	{
			 value = atof(valstr2);
			 if (value != 0.0) {
				strcpy(str, label2);
				strcat(str,"R");
				i = getindex(str);
				if (i != -1) {
					r[i] = value;
				} else warn (32,label2);
			 }
		       }
		       value = atof(valstr1);
		       if (value != 0.0) {
			  strcpy(str, label1);
			  strcat(str,"R");
			  i = getindex(str);
			  if (i	!= -1) {
			      r[i] = value;
			  } else warn (32,label1);
		       }
		    }
		}
		break;

	      case BNDS:

		if ( line[0] !=	' ' ) {
			state =	newstate(line);
		} else {
		  if (bounds[0]=='\0') strncpy(bounds,label0,256);
		  if ( my_strstr(label0, bounds) != NULL ) {
		    value = atof(valstr1);
		    strcpy(str,	label1);
		    strcat(str,"C");
		    j =	getindex(str);
		    if (j != -1) {
			if ( strcmp ( type, "LO" ) == 0	) {
			    l[j] = value;
			} else if ( strcmp ( type, "UP"	) == 0 ) {
			    u[j] = value;
			} else if ( strcmp ( type, "FX"	) == 0 ) {
			    l[j] = value;
			    u[j] = value;
			} else if ( strcmp ( type, "FR"	) == 0 ) {
			    l[j] = -HUGE_VAL;
			    u[j] =  HUGE_VAL;
			} else if ( strcmp ( type, "PL"	) == 0 ) {
			    u[j] = HUGE_VAL;
			} else if ( strcmp ( type, "MI"	) == 0 ) {
			    u[j] = l[j];
			    l[j] = -HUGE_VAL;
			} else if ( strcmp ( type, "BV"	) == 0 ) {
			    l[j] = 0.0;
			    u[j] = 1.0;
			    varsgn[j] =	2;   /*	indicates Integer Variable */
			} else if ( strcmp ( type, "LI"	) == 0 ) {
			    l[j] = value;
			    varsgn[j] =	2;   /*	indicates Integer Variable */
			} else if ( strcmp ( type, "UI"	) == 0 ) {
			    u[j] = value;
			    varsgn[j] =	2;   /*	indicates Integer Variable */
			} else if ( strcmp ( type, "SC"	) == 0 ) {
			    l[j] = 0.0;
			    u[j] = value;
			    varsgn[j] =	3;   /*	indicates Semi-Continuous Var */
			} else warn(27,type);
		    } else warn	(33,label1);
		  }
		}
		break;

	      case QUADS:

		if ( line[0] !=	' ' ) {
			state =	newstate(line);
		} else {
		    strcpy(str,label0);
		    strcat(str,"C");
		    j =	getindex(str);
		    if (j == -1) {		
			  warn (34,label0);
		    } else {
			if (j >	j_previous) {
			    int	jj;
			    for	(jj=j_previous+1; jj<=j; jj++) {
				kQ[jj] = qnz;
			    }
			    j_previous = j;
			} else if (j < j_previous) error (36,"");
			if ( len >= 25 ) {
			       value = atof(valstr1);
			       if (value != 0.0) {
				  strcpy(str, label1);
				  strcat(str,"C");
				  i = getindex(str);
				  if (i	!= -1) {
				    if (i>j) {
					iQ[qnz]	= i; Q[qnz] = value;
					qnz++;
					if (qnz	>= qnz2) {
					  qnz2 = 2*qnz2;
					  REALLOC( iQ, qnz2, int );
					  REALLOC(  Q, qnz2, double );
					}
				    } else if (i==j) {
					diagQ[j] = value;
				    } else warn	(35,"");
				  } else warn (34,label1);
			       }
			} 
			if ( len >= 50 ) {
			       value = atof(valstr2);
			       if (value != 0.0) {
				  strcpy(str, label2);
				  strcat(str,"C");
				  i = getindex(str);
				  if (i	!= -1) {
				    if (i>j) {
					iQ[qnz]	= i; Q[qnz] = value;
					qnz++;
					if (qnz	>= qnz2) {
					  qnz2 = 2*qnz2;
					  REALLOC( iQ, qnz2, int );
					  REALLOC(  Q, qnz2, double );
					}
				    } else if (i==j) {
					diagQ[j] = value;
				    } else warn	(35,"");
				  } else warn (34,label2);
			       }
			}
		    }
		}
		break;

	      }
	  }
	  fclose(fp);
	}
	if (name[0] == '\0') error(11,""); 
	if (state != END) warn(41,""); 

	kA[n] =	nz;
	for (j=j_previous+1; j<=n; j++)	{
		kQ[j] =	qnz;
	}

	REALLOC( A,	 nz,  double );
	REALLOC( iA,	 nz,  int );
	REALLOC( kA,	 n+1, int );
	REALLOC( kQ,	 n+1, int );
	REALLOC( c,	 n,   double );
	REALLOC( u,	 n,   double );
	REALLOC( l,	 n,   double );
	REALLOC( diagQ,	 n,   double );
	REALLOC( varsgn, n,   int );
	REALLOC( collab, n,   char * );

	{
	    int	ic, inew, knew,	*iwork;

	    strcpy(str,obj); 
	    strcat(str,"R"); 
	    ic = getindex(str);
	    if (ic == -1) warn(40,obj);
	    else 
	    if (mark[ic] != 2) warn(40,obj);

	    knew = 0;
	    for	(j=0; j<n; j++)	{
		for (k=kA[j], kA[j] = knew; k<kA[j+1]; k++) {
		    i =	iA[k];
		    if (i == ic) {		/* extract objective */
			c[j] = A[k];
		    } else
		    if (mark[i]	== 2) {		/* remove N rows */
		    } else
		    if (mark[i]	== 1) {		/* negate L rows */
			A[knew]	= -A[k];
			iA[knew] = iA[k];
			knew++;
		    } else {			/* just	copy over */
			A[knew]	= A[k];
			iA[knew] = iA[k];
			knew++;
		    }
		}
	    }
	    nz = knew;
	    kA[n] = knew;
	    MALLOC( iwork, m, int );
	    inew = 0;
	    for	(i=0; i<m; i++)	{
		if (i != ic && mark[i] != 2 ) {
			iwork[i] = inew;
			if (mark[i] == 1) {
				b[inew]	= -b[i];
			} else {
				b[inew]	=  b[i];
			}
			r[inew]	= r[i];
			rowlab[inew] = rowlab[i];
			inew++;
		}
	    }
	    m =	inew;
	    for	(k=0; k<nz; k++) {
		iA[k] =	iwork[iA[k]];
	    }
	    FREE( iwork	);
	}
	REALLOC( A,	 nz, double );
	REALLOC( iA,	 nz, int );
	REALLOC( b,	 m, double );
	REALLOC( r,	 m, double );
	REALLOC( rowlab, m, char * );

	/*----------------------------------------------------+
	| Symmetrize the Q matrix			     */

	{
		int cnt=0, *iwork, *iQ_new, *kQ_new;
		double *Q_new;

		for (j=0; j<n; j++) if (diagQ[j] != 0.0) cnt++;
		qnz = 2*qnz + cnt;

		MALLOC(	kQ_new,	n+1, int );
		MALLOC(	iQ_new,	qnz, int );
		MALLOC(	 Q_new,	qnz, double );

		/* make	iwork[]	contain	row degrees plus 1 if diag is nonzero */

		CALLOC(	iwork, n+1, int	);

		for (k=0; k<kQ[n]; k++)	{
			row = iQ[k];
			iwork[row]++;
		}
		for (j=0; j<n; j++) {
			if (diagQ[j] !=	0.0) iwork[j]++;
		}

		/* make	kQ_new[j], for j=0,1,...,n  */

		kQ_new[0] = 0;
		for (j=0; j<n; j++) {
			kQ_new[j+1] = kQ_new[j]	+ kQ[j+1]-kQ[j]	+ iwork[j];
		}

		/* spread out iQ[], Q[]	*/

		for (j=0; j<n; j++) {
			iwork[j] = kQ_new[j];
			for (k=kQ[j]; k<kQ[j+1]; k++) {
				Q_new[	iwork[j] ]  = Q[k];
				iQ_new[	iwork[j] ]  = iQ[k];
				iwork[j]++;
			}
		}

		/* fill	in with	the other half */

		for (j=0; j<n; j++) {
			if (diagQ[j] !=	0.0) {
				iQ_new[	iwork[j] ] = j;
				Q_new[	iwork[j] ] = diagQ[j];
				iwork[j]++;
			}
			for (k=kQ[j]; k<kQ[j+1]; k++) {
				row = iQ[k];
				iQ_new[	iwork[row] ] = j;
				Q_new[	iwork[row] ] = Q[k];
				iwork[row]++;
			}
		}

		for (j=0; j<n; j++) qksort(iQ_new, Q_new, 
					kQ_new[j], kQ_new[j+1]-1);
		FREE( iwork ); FREE( diagQ ); 
		FREE( kQ ); FREE( iQ );	FREE( Q	);
		kQ = kQ_new; iQ	= iQ_new; Q = Q_new;
	}

	nz = kA[n];

	REALLOC( rowlab, m, char * );
	REALLOC( collab, n, char * );

	FREE(mark); 

	/*------------------------------------------------------+
	| Finish up                                            */

	lp->m =	m;
	lp->n =	n;
	lp->nz = nz;
	lp->A =	A;
	lp->iA = iA;
	lp->kA = kA;
	lp->b =	b;
	lp->c =	c;
	lp->f =	f;
	lp->r =	r;
	lp->l =	l;
	lp->u =	u;
	lp->varsgn = varsgn;
	lp->rowlab = rowlab;
	lp->collab = collab;
	lp->qnz	= qnz;
	lp->Q	= Q;
	lp->iQ	= iQ;
	lp->kQ	= kQ;
	lp->max	    = maxflag;
	lp->inftol  = inftol;
	lp->sf_req  = sf_req;
	lp->verbose = verbose;
	lp->itnlim  = itnlim;
	lp->timlim  = timlim;
	lp->param = param;
	lp->np = np;
	strcpy(lp->name,name);
	strcpy(lp->obj,obj);
	strcpy(lp->rhs,rhs);
	strcpy(lp->ranges,ranges);
	strcpy(lp->bounds,bounds);
}

void	writelp(
	LP    *lp,
	char	*fname
)
{
	int i,j,k,ii,kstart;
	FILE *fp;
	int m, n, nz, *iA, *kA, *iQ, *kQ, max;
	double *A, *Q, *b, *c, *r, *l, *u;
	char **rowlab, **collab, *name, *obj, *rhs, *ranges, *bounds;

	m = lp->m;		/* number of rows */
	n = lp->n;		/* number of columns */
	nz = lp->nz;		/* number of nonzeros */
	A = lp->A;	/* pointer to array of nonzero values in A */
	iA = lp->iA;	/* pointer to array of corresponding row indices */
	kA = lp->kA;	/* pointer to array of indices into A (and iA)
				indicating where each new column of A begins */
	Q = lp->Q;	/* pointer to array of nonzero values in Q */
	iQ = lp->iQ;	/* pointer to array of corresponding row indices */
	kQ = lp->kQ;	/* pointer to array of indices into Q (and iQ)
				indicating where each new column of A begins */
	b = lp->b;	/* pointer to array containing right-hand side */
	c = lp->c;	/* pointer to array containing objective function */
	r = lp->r;	/* pointer to array containing range vector */
	l = lp->l;	/* pointer to array containing lower bounds */
	u = lp->u;	/* pointer to array containing upper bounds */

	rowlab = lp->rowlab;	/* array of strings containing row labels */
	collab = lp->collab;	/* array of strings containing column labels */

	name = lp->name;	/* string containing problem name */
	obj  = lp->obj;	/* string containing objective function	name */
	rhs  = lp->rhs;	/* string containing right-hand	side name */
	ranges = lp->ranges;/* string containing range set name */
	bounds = lp->bounds;/* string containing bound set name */
	max = lp->max;/* -1 means max */

	if ( ( fp = fopen(fname,"w") ) == NULL ) error(2,fname);
	if ( max == -1 ) fprintf(fp,"MAX\n");
	fprintf(fp,"NAME          %8s\n",name);
	fprintf(fp,"ROWS\n");
	fprintf(fp," N  %8s\n",obj);
	for (i=0; i<m; i++) {
	    if (r[i] ==	0.0) {
		fprintf(fp," E  %8s\n",rowlab[i]);
	    } else {
		fprintf(fp," G  %8s\n",rowlab[i]);
	    }
	}
	fprintf(fp,"COLUMNS\n");
	for (j=0; j<n; j++) {
	/*
	    for	(k=kA[j]; k<kA[j+1]-1; k+=2) {
		fprintf(fp,"    %8s  %8s  %12f   %8s  %12f\n",
			collab[j],rowlab[iA[k]],A[k],rowlab[iA[k+1]],A[k+1]);
		
	    }
	    if ( k<kA[j+1] ) 
		fprintf(fp,"    %8s  %8s  %12f\n",collab[j],rowlab[iA[k]],A[k]);
	    if ( c[j] != 0.0 ) 
		fprintf(fp,"    %8s  %8s  %12f\n",collab[j],obj,c[j]);
	*/
	    if ( c[j] != 0.0 ) {
		k = kA[j];
		if (k<kA[j+1]) { /* column not empty */
		    fprintf(fp,"    %8s  %8s  %12f   %8s  %12f\n",
			collab[j],obj,c[j],rowlab[iA[k]],A[k]);
		} else {
		    fprintf(fp,"    %8s  %8s  %12f\n",
			collab[j],obj,c[j]);
		}
		for	(k=kA[j]+1; k<kA[j+1]-1; k+=2) {
			fprintf(fp,"    %8s  %8s  %12f   %8s  %12f\n",
			collab[j],rowlab[iA[k]],A[k],rowlab[iA[k+1]],A[k+1]);
		}
	    } else {
	        for	(k=kA[j]; k<kA[j+1]-1; k+=2) {
			fprintf(fp,"    %8s  %8s  %12f   %8s  %12f\n",
			collab[j],rowlab[iA[k]],A[k],rowlab[iA[k+1]],A[k+1]);
		
	        }
	    }
	    if ( k<kA[j+1] ) 
		fprintf(fp,"    %8s  %8s  %12f\n",collab[j],rowlab[iA[k]],A[k]);
	}

	fprintf(fp,"RHS\n");
	ii = 0;
	for (i=0; i<m; i++) {
	    if ( b[i] != 0.0 ) {
		if (ii%2 == 0) {
		    fprintf(fp,"    %8s  %8s  %12f",rhs,rowlab[i],b[i]);
		} else {
		    fprintf(fp,"   %8s  %12f\n",rowlab[i],b[i]);
		}
		ii++;
	    }
	}
	if (ii%2 != 0) fprintf(fp,"\n");

	fprintf(fp,"BOUNDS\n");
	for (j=0; j<n; j++) {
	    if ( l[j] != 0.0 ) { 
		fprintf(fp," LO %8s  %8s  %12f \n",bounds,collab[j],l[j]); 
	    }
	    if ( u[j] != HUGE_VAL ) { 
		fprintf(fp," UP %8s  %8s  %12f \n",bounds,collab[j],u[j]); 
	    }
	}

	fprintf(fp,"RANGES\n");
	for (i=0; i<m; i++) {
	    if ( r[i] != 0.0 && r[i] != HUGE_VAL ) { 
		fprintf(fp,"    %8s  %8s  %12f \n",ranges,rowlab[i],r[i]); 
	    }
	}

        fprintf(fp,"QUADS\n"); 
        for (j=0; j<n; j++) { /* loop over columns */
            kstart = kQ[j];  /* first col entry */
            while ((kstart < kQ[j+1]) && (iQ[kstart] < j))
                kstart++;  /* move kstart to first lower triangular entry */
            for (k = kstart; k<kQ[j+1]-1; k+=2) {
                fprintf(fp,"  %8s %8s %12f %8s %12f\n",
                collab[j],collab[iQ[k]],Q[k],collab[iQ[k+1]],Q[k+1]); 
	    }
            if ( k<kQ[j+1] ) {
                fprintf(fp,"  %8s %8s %12f\n",collab[j],collab[iQ[k]],Q[k]); 
	    }
        }

	fprintf(fp,"ENDATA\n");
	fclose(fp);
}

void	writesol(
	LP	*lp,
	char	fname[]
)
{
	int	m, n, *varsgn;
	double	*rowact, *x, *y, *z, *l, *u, *b, *r, eps;
	char	**rowlab, **collab;

	int	i,j;
	FILE	*fp;

	m	= lp->m;
	n	= lp->n;
	varsgn	= lp->varsgn;
	rowlab	= lp->rowlab;
	collab	= lp->collab;
	x	= lp->x;
	y	= lp->y;
	z	= lp->z;
	l       = lp->l;
	u       = lp->u;
	b       = lp->b;
        r       = lp->r;
	eps     = lp->inftol*1.2;

	CALLOC (rowact, m, double);
	for (j=0; j<n; j++) for (i=lp->kA[j]; i<lp->kA[j+1]; i++) {
	   if (lp->iA[i] < m) {
	       rowact[lp->iA[i]] += lp->x[j]*lp->A[i];
	   }
	}

	if ( ( fp = fopen(fname, "w") )	== NULL	) error(2,fname);
	fprintf(fp,"COLUMNS SECTION\n");
	fprintf(fp,"   index       label  primal_val reduced_cst");
	fprintf(fp,"    lower_bd    upper_bd   OB_flag\n");
	for (j=0; j<n; j++) {
	   if (l[j]>-HUGE_VAL && u[j]<HUGE_VAL)
		fprintf(fp,"%8d  %10s %11.4e %11.4e %11.4e %11.4e",
		       j,collab[j],x[j],z[j],l[j],u[j]);
              else if (l[j]>-HUGE_VAL)
		fprintf(fp,"%8d  %10s %11.4e %11.4e %11.4e    Infinity",
		       j,collab[j],x[j],z[j],l[j]);
                else if (u[j]<HUGE_VAL) 
		   fprintf(fp,"%8d  %10s %11.4e %11.4e   -Infinity %11.4e",
		           j,collab[j],x[j],z[j],u[j]);
                   else
		   fprintf(fp,"%8d  %10s %11.4e %11.4e   -Infinity    Infinity",
		           j,collab[j],x[j],z[j]);
           if (x[j]<l[j]-eps || x[j]>u[j]+eps) fprintf (fp,"      OB\n");
	      else                             fprintf (fp,"\n");
           }
	fprintf(fp,"ROWS SECTION\n");
	fprintf(fp,"   index       label    dual_val  row_actvty");
	fprintf(fp," rght_hnd_sd       range   OB_flag\n");
	for (i=0; i<m; i++) {
	   if (r[i]<HUGE_VAL)
	        fprintf(fp,"%8d  %10s %11.4e %11.4e %11.4e %11.4e",
		       i,rowlab[i],y[i],rowact[i],b[i],r[i]);
              else
	        fprintf(fp,"%8d  %10s %11.4e %11.4e %11.4e    Infinity",
		       i,rowlab[i],y[i],rowact[i],b[i]);
           if (rowact[i]<b[i]-eps || rowact[i]>b[i]+r[i]+eps) 
					   fprintf (fp,"     OB\n");
	      else                         fprintf (fp,"\n");
           }
	fprintf(fp,"ENDOUT\n");
	fclose(fp);
}

/* Define static functions */

static int newstate(
	char	*line
)
{
	if ( strcmp ( line, "RHS" ) == 0 ) return RHS ;
	else
	if ( strcmp ( line, "RAN" ) == 0 ) return RNGS ;
	else
	if ( strcmp ( line, "BOU" ) == 0 ) return BNDS ;
	else
	if ( strcmp ( line, "QUA" ) == 0 ) return QUADS	;
	else
	if ( strcmp ( line, "END" ) == 0 ) return END ;
	else error(26,line);
	return -1;
}

static void error(
	int	num,
	char	*text
)
{
	char str[50];

	switch (num) {
	case   2: sprintf(str,"cannot open file %s\n",text);
		  break;
	case   3: sprintf(str,"cannot read file %s\n",text);
		  break;
	case   4: sprintf(str,"cannot create file %s\n",text);
		  break;
	case   5: sprintf(str,"cannot write file %s\n",text);
		  break;
	case   6: sprintf(str,"cannot allocate space\n");
		  break;
	case   9: sprintf(str,"cannot solve dual as primal when ranges are present\n");
		  break;
	case  10: sprintf(str,"dimension conflict in %s\n",text);
		  break;
	case  11: sprintf(str,"NAME not found\n");
		  break;
	case  26: sprintf(str,"unrecognized section label: \n   %s \n",text);
		  break;
	case  35: sprintf(str,"column %s out of order in COLUMNS section\n",text);
		  break;
	case  36: sprintf(str,"columns out of order in QUADS section\n");
		  break;
	}
	my_exit(num,str);
}

static void warn(
	int	num,
	char	*text
)
{
	switch (num) {
	case  20: printf("expected ROWS after NAME instead of %s\n",
			text);
		  break;
	case  21: printf("expected L, E, G, N, or COLUMNS instead of %s\n",
			text);
		  break;
	case  27: printf("unrecognized bound type %s \n",text);
		  break;
	case  30: printf("row label %s from COLUMNS section missing in ROWS section\n",text);
		  break;
	case  31: printf("row label %s from RHS section missing in ROWS section\n",text);
		  break;
	case  32: printf("row label %s from RANGES section missing in ROWS section\n",text);
		  break;
	case  33: printf("col label %s from BOUNDS section missing in COLUMNS section\n",text);
		  break;
	case  34: printf("col label %s from QUADS section missing in COLUMNS section\n",text);
		  break;
	case  35: printf("QUADS section contains entry from upper triangle\n");
		  break;
	case  40: printf("objective function %s not found \n",text);
		  break;
	case  41: printf("ENDATA not found \n");
		  break;
	}
}

static char *my_strstr(
	char	*s,
	char	*t
)
{
	register char *p, *q, *r;

	for ( p=s; *p!='\0'; p++ ) {
		for (q=p, r=t; *r!='\0'	&& *q==*r; q++,	r++ ) ;
		if ( r!=t && *r=='\0' )	return p;
	}
	return (char *)NULL;
}

/*  ***	qksort:	sort v[left]...v[right]	into increasing	order ***
	reference: The C Programming Language,
		   Kernighan and Ritchie
		   2nd.	edition, page 87.  */

static void qksort(
	int	*v,
	double	*data,
	int	left,
	int	right
)
{
	int i, last;

	if (left >= right) return;   /*	do nothing if array contains
					fewer than two elements	*/

	swap(v,	data, left, (left + right)/2);	/* move	partition elem */
	last = left;				/* to v[left] */
	for (i = left+1; i <= right; i++)	/* partition */
	    if (v[i] < v[left])
		swap(v,	data, ++last, i);
	swap(v,	data, left, last);		/* restore partition elem */
	qksort(v, data,	left, last-1);
	qksort(v, data,	last+1,	right);
}

/*  ***	swap: interchange v[i] and v[j]	*/

static void swap(
	int	*v, 
	double	*data,
	int	i, 
	int	j
)
{
	int temp;
	double tmp_data;

	temp = v[i];
	v[i] = v[j];
	v[j] = temp;

	tmp_data = data[i];
	data[i]	= data[j];
	data[j]	= tmp_data;
}

int	getparam(
	char *label0
)
{
	char str[10];

	strcpy(str,label0);
	strcat(str,"P");
	return getindex(str);
}

void	my_exit(
	int num,
	char *str
)
{
	printf("ERROR(%d): %s\n", num, str); exit(1);
}
