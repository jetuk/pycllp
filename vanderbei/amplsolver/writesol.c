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

#include "nlp.h"

#ifdef KR_headers

 void
write_soln(msg, x, y)
 char *msg;
 double *x, *y;
#else
#include "stdlib.h"
#ifdef __cplusplus
extern "C" {
#endif

 void
write_soln(char *msg, double *x, double *y)
#endif
{
	FILE *f;
	int i, k;
	char buf[80];
	static char *wkind[] = {"w", "wb"};
	typedef fint uiolen;
	uiolen L[6];
	fint n, m, z[4];
	size_t nn;

	strcpy(stub_end, ".sol");
	f = fopen(filename, wkind[binary_nl]);
	if (!f) {
		fprintf(stderr, "can't open %s\n", filename);
		exit(2);
		}
	z[0] = m = n_con;
	if (!y)
		m = 0;
	z[1] = m;
	z[2] = n = n_var;
	if (!x)
		n = 0;
	z[3] = n;
	k = (int)ampl_options[0];
	if (binary_nl) {
		L[0] = 6;
		L[1] = strlen(msg);
		L[2] = 0;
		L[3] = (ampl_options[0] + 5)*sizeof(fint) + 7;
		L[4] = m*sizeof(double);
		L[5] = n*sizeof(double);
		fwrite(L, sizeof(uiolen), 1, f);
		fwrite("binary", 6, 1, f);
		fwrite(L, sizeof(uiolen), 2, f);
		if (L[1]) {
			fwrite(msg, L[1], 1, f);
			fwrite(L+1, sizeof(uiolen), 2, f);
			}
		if (k) {
			fwrite(L+2, sizeof(uiolen), 2, f);
			fwrite("Options",7,1,f);
			nn = (size_t)ampl_options[0]+1;
			if (ampl_options[2] == 3)
				ampl_options[0] += 2;
			fwrite(ampl_options, sizeof(fint), nn, f);
			fwrite(z, sizeof(fint), 4, f);
			if (ampl_options[2] == 3)
				fwrite(&ampl_vbtol, sizeof(real), 1, f);
			fwrite(L+3, sizeof(uiolen), 2, f);
			}
		else {
			fwrite(L+2, sizeof(uiolen), 1, f);
			fwrite(L+4, sizeof(uiolen), 1, f);
			}
		if (y)
			fwrite(y, sizeof(double), m, f);
		fwrite(L+4, sizeof(uiolen), 2, f);
		if (x)
			fwrite(x, sizeof(double), n, f);
		fwrite(L+5, sizeof(uiolen), 1, f);
		if (obj_no) {
			L[0] = L[2] = sizeof(fint);
			L[1] = obj_no;
			fwrite(L, sizeof(fint), 3, f);
			}
		}
	else {
		fprintf(f, *msg ? "%s\n\n" : "\n", msg);
		if (k = (int)ampl_options[0]) {
			if (ampl_options[2] == 3)
				ampl_options[0] += 2;
			fprintf(f, "Options\n");
			for(i = 0; i <= k; i++)
				fprintf(f,"%ld\n",(long)ampl_options[i]);
			fprintf(f,"%ld\n%ld\n%ld\n%ld\n",
				(long)z[0],(long)z[1],(long)z[2],(long)z[3]);
			if (ampl_options[2] == 3) {
				g_fmtp(buf, ampl_vbtol, 0);
				fprintf(f, "%s\n", buf);
				}
			}
		while(--m >= 0) {
			g_fmtp(buf, *y++, 0);
			fprintf(f,"%s\n", buf);
			}
		while(--n >= 0) {
			g_fmtp(buf, *x++, 0);
			fprintf(f, "%s\n", buf);
			}
		if (obj_no)
			fprintf(f, "objno %d\n", obj_no);
		}
	fclose(f);
	if (i = need_nl)
		if (i > sizeof(buf)-1 || i < 0)
			printf("\n");
		else {
			buf[i] = 0;
			do buf[--i] = '\b';
				while(i > 0);
			printf(buf);
			}
	}
#ifdef __cplusplus
	}
#endif
