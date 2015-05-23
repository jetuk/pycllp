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

#define MSGGULP 1024

 static char *msg, *msg0, *msgend;
 fint msglen;

 static void
#ifdef KR_headers
msgput(b, n) char *b; int n;
#else
msgput(char *b, int n)
#endif
{
	char *be;
	fint msglen0;

	be = b + n;
	while(b < be) {
		if (msg >= msgend) {
			msglen0 = msglen;
			msg0 = Realloc(msg0, msglen += MSGGULP);
			msg = msg0 + msglen0;
			msgend = msg0 + msglen;
			}
		*msg++ = *b++;
		}
	}

 static void
#ifdef KR_headers
badnumber(a, b, what) fint a, b; char *what;
#else
badnumber(fint a, fint b, char *what)
#endif
{
	fprintf(stderr, "%s indicates %ld rather than %ld %s\n",
		filename, (long)a, (long)b, what);
	}

 static int
#ifdef KR_headers
decstring(buf, val) char *buf; real *val;
#else
decstring(char *buf, real *val)
#endif
{
	char *be;
	register int c;

	*val = strtod(buf, &be);
	return be <= buf || ((c = be[-1]) < '0' || c > '9') && c != '.';
	}

 char *
#ifdef KR_headers
read_soln(xp, yp) real **xp, **yp;
#else
read_soln(real **xp, real **yp)
#endif
{
	int binary, i, j, je, n, need_vbtol;
	FILE *f;
	char buf[512], *se;
	real t, vbtol, *y;
	typedef fint uiolen;
	uiolen L, L1, L2;
	fint Objno, nOpts, *z;

	strcpy(stub_end, ".sol");
	f = fopen(filename, "rb");
	if (!f) {
		fprintf(stderr, "Can't open %s\n", filename);
		return 0;
		}
	if (fread((char *)&L, sizeof(uiolen), 1, f) && L == 6) {
		/* binary files may be written by Fortran unformatted writes */
		binary = 1;
		if (!fread(buf, 6, 1, f)
		 || strncmp(buf,"binary",6)
		 || !fread((char *)&L, sizeof(uiolen), 1, f)
		 || L != 6) {
 badbinary:
			fprintf(stderr, "bad binary file %s\n", filename);
			goto done;
			}
		}
	else {
		binary = 0;
		rewind(f);
		}

	/* Read termination msg */
	nOpts = i = need_vbtol = 0;
	msg = msg0 = (char *)Malloc(msglen = MSGGULP);
	msgend = msg0 + MSGGULP;
	if (binary) {
		for(;; i++) {
			if (!fread((char *)&L,sizeof(uiolen),1,f))
				goto early_eof;
			if (L1 = L) {
				do {
					n = L < sizeof(buf) ? (int)L : (int)sizeof(buf);
					L -= n;
					if (!fread(buf, n, 1, f))
						goto early_eof;
					if (!L) {
						while(--n >= 0 && buf[n] == ' ');
						n++;
						}
					msgput(buf, n);
					}
					while(L);
				msgput("\n", 1);
				}
			if (!fread((char *)&L, sizeof(uiolen), 1, f))
				goto early_eof;
			if (L != L1)
				goto badbinary;
			if (!L)
				break;
			}
		L1 = n_con * sizeof(real);
		if (!fread((char *)&L, sizeof(uiolen), 1, f))
			goto badbinary;
		if (L == 8*sizeof(fint) + 7
		 || L == 9*sizeof(fint) + 7
		 || L == 8*sizeof(fint) + 7 + sizeof(real)
		 || L == 9*sizeof(fint) + 7 + sizeof(real)) {
			/* check for Options */
			if (!fread(buf, 7, 1, f))
				goto badbinary;
			if (strncmp(buf, "Options", 7))
				goto badbinary;
			if (!fread((char *)&nOpts, sizeof(fint), 1, f))
				goto badbinary;
			if (nOpts < 3 || nOpts > 6) {
 bad_nOpts:
				fprintf(stderr,
				"expected nOpts between 3 and 6; got %ld: ",
					(long)nOpts);
				goto badbinary;
				}
			if (nOpts > 4) {
				nOpts -= 2;
				need_vbtol = 1;
				}
			if (!fread((char *)(ampl_options+1), sizeof(fint),
					(size_t)(nOpts+4), f))
				goto badbinary;
			if (need_vbtol
			 && !fread((char *)&vbtol, sizeof(real), 1, f))
				goto badbinary;
			if (!fread((char *)&L2, sizeof(uiolen), 1, f)
				|| L != L2)
				goto badbinary;
			}
		else if (L != L1)
			goto badbinary;
		}
	else {
		for(;; i++) {
			if (!fgets(buf, sizeof(buf), f)) {
 early_eof:
				fprintf(stderr,
					"early end of file reading %s\n",
					filename);
 done:
				fclose(f);
 done1:
				return 0;
				}
			if (*buf == '\n' || *buf == '\r' && buf[1] == '\n')
				break;
			msgput(buf, strlen(buf));
			}
		while((j = getc(f)) == '\n');
		if (j != 'O')
			ungetc(j,f);
		else {
			if (!fgets(buf, sizeof(buf), f))
				goto early_eof;
			if (!strncmp(buf, "ptions", 6)) {
				if (!fgets(buf, sizeof(buf), f))
					goto early_eof;
				nOpts = strtol(buf,&se,10);
				if (se == buf)
					goto badline;
				if (nOpts < 3 || nOpts > 6)
					goto bad_nOpts;
				if (nOpts > 4) {
					nOpts -= 2;
					need_vbtol = 1;
					}
				je = (int)nOpts + 4;
				for(j = 1; j <= je; j++) {
					if (!fgets(buf, sizeof(buf), f))
						goto early_eof;
					ampl_options[j] = strtol(buf,&se,10);
					if (se == buf)
						goto badline;
					}
				if (need_vbtol && !fgets(buf, sizeof(buf), f))
					goto early_eof;
				/* We don't do anything here with vbtol, */
				/* so we don't bother converting it. */
				}
			}
		}
	msgput("", 1);	/* add null to end */

	if (i)
		fflush(stdout);

	if (ampl_options[0] = nOpts) {
		z = ampl_options + nOpts + 1;
		j = (int)z[3];
		if (j > n_var || j < 0) {
			badnumber(j, n_var, "variables");
			goto done;
			}
		j = (int)z[1];
		if (j > n_con || j < 0) {
			badnumber(j, n_con, "constraints");
			goto done;
			}
		if (binary) {
			L1 = j * sizeof(real);
			if (!fread((char *)&L, sizeof(uiolen), 1, f))
				goto badbinary;
			if (L != L1)
				goto badbinary;
			}
		}
	else
		j = n_con;
	if (!j) {
		*yp = 0;
		goto get_x;
		}
	y = *yp = (real *)Malloc(n_con * sizeof(real));
	if (binary) {
		if (fread((char *)y, sizeof(real), j, f) != j)
			goto early_eof;
		if (!fread((char *)&L, sizeof(uiolen), 1, f) || L != L1)
			goto badbinary;
		y += j;
		}
	else  for(i = 0; i < j; i++) {
		if (!fgets(buf, sizeof(buf), f))
			goto early_eof;
		if (binary) {
			*y++ = t;
			continue;
			}
		if (!decstring(buf, y++))
			continue;
 badline:
		fprintf(stderr, "bad line in %s: %s", filename, buf);
		goto done;
		}
	while(j < n_con)
		*y++ = 0;
 get_x:
	obj_no = 0;
	if (!(j = nOpts ? (int)z[3] : n_var)) {
		*xp = 0;
		goto ret;
		}
	y = *xp = (real *)Malloc(n_var*sizeof(real));

	if (binary) {
		L1 = j * sizeof(real);
		if (!fread((char *)&L, sizeof(uiolen), 1, f) || L != L1)
			goto badbinary;
		if (fread((char *)y, sizeof(real), j, f) != j)
			goto early_eof;
		y += j;
		/* do we have an obj_no ? */
		if (!fread((char *)&L, sizeof(uiolen), 1, f) || L != L1)
			goto badbinary;
		if (fread((char *)&L, sizeof(uiolen), 1, f)) {
			if (L != sizeof(fint))
				goto badbinary;
			if (!fread((char *)&Objno, sizeof(fint), 1, f))
				goto badbinary;
			obj_no = (int)Objno;
			}
		}
	else {
		for(i = j; i > 0; i--) {
			if (!fgets(buf, sizeof(buf), f))
				goto early_eof;
			if (decstring(buf, y++))
				goto badline;
			}
		if (fgets(buf,sizeof(buf), f)) {
			if (strncmp(buf,"objno ",6) || decstring(buf+6, &t)) {
				fprintf(stderr, "Bug: extra line in %s:\n%s",
					filename, buf);
				fflush(stderr);
				}
			else
				obj_no = (int)t;
			}
		}
	while(j < n_var)
		*y++ = 0;
 ret:
	fclose(f);
	return Realloc(msg0, msglen);
	}
