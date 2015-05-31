#include <stdlib.h>
#include <string.h>

#ifdef QuadPrec
#include "Quad.h"
#define double Quad
#else
#define high(x) (x)
#endif

#include "lp.h"
#include "myalloc.h"

void amplinterface(int argc, char **argv);

main(int argc, char **argv)
{
	int	status;
	char	fname[128];	/* solution file name */
	LP	*kp;
	static char *statmsg[] = {
		"optimal solution",	/* 0 */
		"primal unbounded",	/* 1 */
		"primal infeasible",	/* 2 */
		"dual unbounded",	/* 3 */
		"dual infeasible",	/* 4 */
		"iteration limit",	/* 5 */
		"infinite lower bounds - not implemented",	/* 6 */
		"suboptimal solution"   /* 7 */
	};

	if (argc >= 3 && !strcmp(argv[2], "-AMPL")) {
		amplinterface(argc, argv);
		return 0;
	}

	kp = openlp();
	if (kp == NULL) { fprintf(stderr,"Bug: openlp failure!\n"); exit(1); }

	printf("%s\n%s\n%s%5s%s\n%s\n%s\n",
		"\t+-------------------------------------------------+",
		"\t                                                   ",
		"\t   ",argv[0],":   Version 1.00 : (Copyright) 1995        ",
		"\t                                                   ",
		"\t+-------------------------------------------------+");
	fflush(stdout);

	argc--; argv++;
	readlp(argc,argv,kp);

	status = solvelp(kp);
	printf("%s \n", statmsg[status]);

	strncpy(fname, kp->name, sizeof(fname)-5);
	strcat(fname, ".out");
	writesol(kp,fname);
	return 0;
}
