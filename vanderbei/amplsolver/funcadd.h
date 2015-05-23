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

#ifndef FUNCADD_H_included
#define FUNCADD_H_included
#include "math.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef real
#define real double
#endif

 struct function;
 struct TVA;

 struct
arglist {
	int n;
	int *at;
	real *ra;
	char **sa;
	void *derivs;
	struct function *f;
	struct TVA *tva;
	};

typedef struct arglist arglist;
typedef struct TVA TVA;
typedef real ufunc(
#ifndef KR_headers
		   arglist *
#endif
			    );

extern void addfunc(
#ifndef KR_headers
		const char *name,
		ufunc *f,
		int type,
			/* 0 ==> force all arguments to be numeric */
			/* 1 ==> pass both symbolic and numeric arguments. */
		int nargs
			/* >=  0 ==> exactly that many args
			 * <= -1 ==> at least -(nargs+1) args
			 */
#endif
		);

#ifdef __cplusplus
	}
#endif
#endif
