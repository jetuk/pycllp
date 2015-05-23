/****************************************************************
Copyright (C) AT&T 1992
All Rights Reserved

Permission to use, copy, modify, and distribute this software and
its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both that the copyright notice and this
permission notice and warranty disclaimer appear in supporting
documentation, and that the name of AT&T or any of its entities
not be used in advertising or publicity pertaining to
distribution of the software without specific, written prior
permission.

AT&T DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL AT&T OR ANY OF ITS ENTITIES BE LIABLE FOR ANY
SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
THIS SOFTWARE.
****************************************************************/

#include "nlp.h"

/* Examine $mip_priorities.
 * return number of valid names in $mip_priorities and, if positive, set
 *		*startp = array of starting subscripts (offset 0)
 *		*prip = array of priorities
 *		*nump = number of variables with this priority.
 * pmax = max allowed priority.  Thus
 *
 *	if ( (n = mip_pri(&start, &num, &pri, pmax)) > 0 )
 *
 * then for i = 0, 1, ..., n-1, there are num[i] consecutive variables
 * with priority pri[i], starting with variable start[i].
 */

#ifdef KR_headers
 extern char *getenv();
#endif

 int
#ifdef KR_headers
mip_pri(startp, nump, prip, pmax)
	int **startp, **nump, **prip; fint pmax;
#else
mip_pri(int **startp, int **nump, int **prip, fint pmax)
#endif
{
	static FILE *f;
	static char *s0;
	int i, j, k, len, listsize, nn, nn1, nnames, nntk, nreal;
	int *ntk, *num, *num0, *num1, *p, *perm,
		*pv, *pv0, *start, *start0, *start1;
	char **pn0, **pn, *ps, *s, *s1, *s2, *se;
	fint L;
	char buf[512], buf0[512], namechar[256];
	static int bad, echo = 1, quit = 1;

	if (!(s0 = getenv("mip_priorities")))
		return 0;
	nn1 = 0;
	while(*s0 && *s0 <= ' ')
		s0++;
	while(*s0 == '!') {
		for(s1 = ++s0; *++s1 > ' '; );
		k = s1 - s0;
		if (k == 4) {
			if (!strncmp(s0, "quit", k))
				quit = bad = 0;
			else if (!strncmp(s0, "echo", k))
				echo = 0;
			else
				goto badc;
			}
		else {
 badc:
			printf(
			 "Bad control word \"!%.*s\" in $mip_priorities\n",
				k, s0);
			bad += quit;
			}
		for(s0 = s1; *s0 && *s0 <= ' '; s0++);
		}
	if (!*s0)
		goto done1;
	s = quit ? "can't process" : "ignoring";
	if (!maxcolnamelen) {
		printf("No .col file; %s $mip_priorities.\n%s", s,
			"To get a .col file, in your AMPl session say\n");
		if (s = getenv("solver"))
			printf("\toption %s_auxfiles c;\n%s", s,
				"before saying\n\tsolve;\nor\n");
		printf("\toption auxfiles c;\nbefore saying\n\twrite ...\n");
		goto bailout;
		}
	strcpy(stub_end, ".col");
	if (!(f = fopen(filename, "r"))) {
		printf("Can't open %s; %s $mip_priorities.\n", filename, s);
 bailout:
		bad += quit;
		need_nl = 0;
		goto done1;
		}
	/* Guess there will only be a small number of names mentioned	*/
	/* in $mip_priorities; first we'll find the names, then we'll	*/
	/* match each new name in the .col file against all the		*/
	/* $mip_priorities names.	*/

	len = nnames = 0;
	need_nl = 0;
	s = se = s0;
	for(;;) {
		se = s;
		while(*s > ' ')
			s++;
		s1 = s;
		while(*s <= ' ')
			if (!*s++) {
				printf("Missing priority for %.*s.\n",
					s1-se, se);
				bad += quit;
				goto counted;
				}
		L = strtol(s2 = s, &s, 10);
		if (s == s2 || *s > ' ' || L <= 0 || L > pmax) {
			bad += quit;
			printf("\"%.*s\" is not a positive integer <= %ld:\n",
				s-s2, s2, (long)pmax);
			printf("\tbad $mip_priorities entry for %.*s.\n",
				s1-se, se);
			while(*s > ' ')
				s++;
			while(*s <= ' ' && *s)
				s++;
			if (*s)
				printf("Ignoring further entries in $mip_priorities.\n");
			goto counted;
			}
		nnames++;
		len += s1 - se;
		se = s;
		while(*s <= ' ')
			if (!*s++)
				goto counted;
		}
 counted:
	if (!nnames)
		goto done1;
	start = start0 = start1 = (int *)Malloc(nnames*(3*sizeof(int)));
	num = num0 = num1 = start0 + nnames;
	pv = pv0 = num0 + nnames;

	pn = pn0 = (char **)Malloc(nnames*(sizeof(char*)+sizeof(int)+1) + len);
	p = perm = (int *)(pn0 + nnames);
	ps = (char *)(perm + nnames);
	i = nn1 = 0;
	for(s = s0; s < se; ) {
		*start1++ = *num1++ = -1;
		*p++ = i++;
		*pn++ = ps;
		while((*ps = *s++) > ' ')
			ps++;
		*ps++ = 0;
		while(*s <= ' ')
			s++;
		*pv++ = (int)strtol(s, &s, 10);
		while(*s <= ' ')
			if (s++ >= se)
				goto scanned;
		}
 scanned:
	*buf0 = 0;
	ntk = 0;
	memset(namechar, 1, sizeof(namechar));
	namechar['\n'] = namechar['['/*]*/] = namechar[0] = 0;
	nn = nntk = nnames;
	listsize = 0;
	nreal = n_var - (nlogv + niv);
	for(i = 0; fgets(buf, sizeof(buf), f); i++) {
		for(s = buf; namechar[*(unsigned char *)s]; s++);
		if (s == buf) {
			printf("Bad .row file %s\n", filename);
			bad += quit;
			goto done;
			}
		*s = 0;
		if (strcmp(buf0, buf)) {
			strcpy(buf0, buf);
			if (ntk) {
				listsize += *ntk = i - j;
				ntk = 0;
				--nntk;
				}
			for(j = 0; j < nn; j++) {
				k = perm[j];
				if (!strcmp(pn0[k],buf)) {
					if (j < --nn) {
						perm[j] = perm[nn];
						perm[nn] = k;
						}
					start0[k] = i;
					ntk = num0 + k;
					/* no break: in case of duplicates, */
					/* let last one win. */
					}
				}
			if (ntk && i < nreal) {
				printf(
		"%s$mip_priority entry for noninteger variable %s.\n",
					quit ? "" : "Ignoring ", buf0);
				*ntk = -2;
				ntk = 0;
				}
			j = i;
			}
		}
	if (ntk) {
		listsize += *ntk = i - j;
		--nntk;
		}
	if (nn) {
		/* report bad names in $mip_priorities */
		for(j = 0; j < nn; j++) {
			k = perm[j];
			printf(
			"Ignoring $mip_priorities pair \"%s %d\" -- bad name.\n",
				pn0[k], pv0[k]);
			bad += quit;
			}
		}
	if (nntk > nn) { /* Is this necessary? */
		/* report duplicates */
		for(j = nn; j < nnames; j++) {
			k = perm[j];
			if (num0[k] == -1) {
				bad += quit;
				printf(
				"%suplicate $mip_priorities pair \"%s %d\".\n",
					quit? "D" : "Ignoring d",
					pn0[k], pv0[k]);
				}
			}
		}
	if (!listsize)
		goto done;
	if (echo)
		printf(bad ? "Valid $mip_priorities seen:\n"
				: "$mip_priorities used:\n");
	nn = 0;
	do if (num0[nn] > 0) {
		if (echo)
			printf("\t%s\t%d\n", pn0[nn], pv0[nn1] = pv0[nn]);
		pv0[nn1] = pv0[nn];
		start0[nn1] = start0[nn];
		num0[nn1++] = num0[nn];
		}
		while(++nn < nnames);
 done:
	fclose(f);
	free((char *)pn0);
	if (nn1 && !bad) {
		*startp = start0;
		*nump = num0;
		*prip = pv0;
		}
	else
		free((char *)start0);
 done1:
	if (bad)
		exit(2);
	return nn1;
	}
