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

#ifndef NLP_H_included
#include "nlp.h"
#endif

#ifdef KR_headers
 extern void densej_();
 extern int jacdim_();
 extern FILE* jacdim0();
 extern FILE* jacdim1();	/* deprecated */
 extern void jacinc();
 extern void jacinc_();
 extern void conval_();
 extern void jacval_();
 extern real objval_();
 extern void objgrd_();
 extern void wrtsol_();
#else
#ifdef __cplusplus
extern "C" {
#endif
 extern void densej_(void);
 extern int  jacdim_(char *stub, fint *M, fint *N, fint *NO, fint *NZ,
		fint *MXROW, fint *MXCOL, fint stub_len);
 extern FILE*jacdim0(char *stub, fint stub_len);
 extern FILE*jacdim1(char *stub, fint *M, fint *N, fint *NO, fint *NZ,
		fint *MXROW, fint *MXCOL, fint stub_len);	/* deprecated */
 extern void jacinc(fint *JP, short *JI, real **X, real **LUp,
		real **LUrhsp, real *Inf);
 extern void jacinc_(fint *M, fint *N, fint *NZ,
		fint *JP, short *JI, real *X, real *L, real *U,
		real *Lrhs, real *Urhs, real *Inf);
 extern void conval_(fint *M, fint *N, real *X, real *F, fint *nerror);
 extern void jacval_(fint *M, fint *N, fint *NZ, real *X,
			real *JAC, fint *nerror);
 extern real objval_(fint *N, real *X, fint *NPROB, fint *nerror);
 extern void objgrd_(fint *N, real *X, fint *NPROB, real *G, fint *nerror);
 extern void wrtsol_(char *msg, fint *nmsg, real *x, real *y, fint msg_len);
#ifdef __cplusplus
	}
#endif
#endif
