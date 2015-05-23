/****************************************************************
Copyright (C) AT&T 1992, 1993, 1994
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

#include "jacdim.h"
#ifdef SYMANTEC
#define SVS
#define MCW_EM _MCW_EM
#define PC_53 _PC_53
#endif

#ifdef SVS
#include "float.h"
#else
#define _control87(a,b) /*nothing*/
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef KR_headers
 extern void badread(), what_prog();
 extern int Sscanf();
#else
 extern void badread(void), what_prog(void);
 extern int Sscanf(char*, const char*, ...);
#endif

 static void
#ifdef KR_headers
badints(got, wanted)
#else
badints(int got, int wanted)
#endif
{
	badread();
	fprintf(stderr, "got only %d integers; wanted %d\n", got, wanted);
	exit(1);
	}

 static void
#ifdef KR_headers
read2(nl, x, y) FILE *nl; int *x, *y;
#else
read2(FILE *nl, int *x, int *y)
#endif
{
	char *s;
	int k;

	s = read_line(nl);
	k = Sscanf(s, " %d %d", x, y);
	if (k != 2)
		badints(k, 2);
	}

 FILE *
#ifdef KR_headers
jacdim0(stub, stub_len) char *stub; fint stub_len;
#else
jacdim0(char *stub, fint stub_len)
#endif
{
	FILE *nl;
	int i, k;
	char *s, *se;

	_control87(MCW_EM | PC_53, MCW_EM | MCW_PC);

	for(i = 0; i < stub_len; i++)
		if (stub[i] == ' ')
			break;
	filename = (char *)Malloc(i + 5);
	s = stub_end = filename + i;
	strncpy(filename, stub, i);
	strcpy(s, ".nl");
	nl = fopen(filename, "rb");
	if (!nl) {
		if (return_nofile)
			return 0;
		fflush(stdout);
		what_prog();
		fprintf(stderr, "can't open %s\n", filename);
		exit(1);
		}
	s = read_line(nl);
	binary_nl = 0;
	switch(*s) {
#ifdef DEPRECATED
		case 'E':	/* deprecated "-oe" format */
			{int ncsi = 0;
			k = Sscanf(s, "E%d %d %d %d %d %d", &n_var, &n_con,
				&n_obj, &maxrownamelen, &maxcolnamelen, &ncsi);
			if (k < 5)
				badints(k, 5);
			if (ncsi) {
				if (ncsi != 6) {
					badread();
					fprintf(stderr,
					 "expected 6th integer to be 0 or 6, not %d\n",
						ncsi);
					exit(1);
					}
				s = read_line(nl);
				k = Sscanf(s, " %d %d %d %d %d %d",
					&comb, &comc, &como, &comc1, &como1, &nfunc);
				if (k != 6)
					badints(k, 6);
				}
			}
			break;
#endif
		case 'b':
			binary_nl = 1;
		case 'g':
			if (k = ampl_options[0] = strtol(++s, &se, 10)) {
				if (k > 9) {
					fprintf(stderr,
					"ampl_options = %d is too large\n", k);
					exit(1);
					}
				for(i = 1; i <= k && se > s; i++)
					ampl_options[i] = strtol(s = se,&se,10);
				if (ampl_options[2] == 3)
					ampl_vbtol = strtod(s = se, &se);
				}
			s = read_line(nl);
			k = Sscanf(s, " %d %d %d %d", &n_var, &n_con,
				&n_obj, &nranges);
			if (k < 3)
				badints(k,3);
			read2(nl, &nlc, &nlo);
			read2(nl, &nlnc, &lnc);
			nlvb = -1;
			s = read_line(nl);
			k = Sscanf(s, " %d %d %d", &nlvc, &nlvo, &nlvb);
			if (k < 2)
				badints(k,2);
			read2(nl, &nwv, &nfunc);
			if (nlvb < 0)	/* ampl versions < 19930630 */
				read2(nl, &nbv, &niv);
			else {
				s = read_line(nl);
				k = Sscanf(s, " %d %d %d %d %d", &nbv, &niv,
					&nlvbi, &nlvci, &nlvoi);
				if (k != 5)
					badints(k,5);
				}
			read2(nl, &nzc, &nzo);
			read2(nl, &maxrownamelen, &maxcolnamelen);
			s = read_line(nl);
			k = Sscanf(s, " %d %d %d %d %d", &comb, &comc, &como,
					&comc1, &como1);
			if (k != 5)
				badints(k,5);
		}
#ifdef Student_Edition
	if (n_con > 300 || n_var > 300) {
		fflush(stdout);
		fprintf(stderr,
 "\nSorry, the student edition is limited to 300 variables and\n\
300 constraints.  You have %d variables and %d constraints.\n",
			n_var, n_con);
		exit(1);
		}
#endif
	if (n_con < 0 || n_var <= 0 || n_obj < 0) {
		what_prog();
		fprintf(stderr,
		"jacdim: got M = %d, N = %d, NO = %d\n", n_con, n_var, n_obj);
		exit(1);
		}
	x0len = n_var * sizeof(real);
	x0kind = 4;
	c_vars = o_vars = n_var;	/* confusion arises otherwise */
	return nl;
	}

 FILE *
#ifdef KR_headers
jacdim1(stub, M, N, NO, NZ, MXROW, MXCOL, stub_len)
 char *stub;
 fint *M, *N, *NO, *NZ, *MXROW, *MXCOL;
 fint stub_len;
#else
jacdim1(char *stub, fint *M, fint *N, fint *NO, fint *NZ,
		fint *MXROW, fint *MXCOL, fint stub_len)
#endif
{
	FILE *nl;

	if (nl = jacdim0(stub, stub_len)) {
		*M = n_con;
		*N = n_var;
		*NO = n_obj;
		*NZ = nzc;
		*MXROW = maxrownamelen;
		*MXCOL = maxcolnamelen;
		}
	return nl;
	}

 int
#ifdef KR_headers
jacdim_(stub, M, N, NO, NZ, MXROW, MXCOL, stub_len)
 char *stub;
 fint *M, *N, *NO, *NZ, *MXROW, *MXCOL;
 fint stub_len;
#else
jacdim_(char *stub, fint *M, fint *N, fint *NO, fint *NZ,
		fint *MXROW, fint *MXCOL, fint stub_len)
#endif
{
	FILE *nl;

	nl = jacdim1(stub, M, N, NO, NZ, MXROW, MXCOL, stub_len);
	if (!nl)
		return 1;
	X0 = (real *)Malloc(n_var*sizeof(real));
	edagread(nl);
	*NZ = nzjac;	/* For -oe; same as nzc under -ob and -og. */
	return 0;
	}

#ifdef __cplusplus
	}
#endif
