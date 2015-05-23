/****************************************************************
Copyright (C) AT&T 1992, 1993
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

#include "signal.h"
#include "setjmp.h"

/* Separately compiled because of stack check problems with SVS C */

#ifdef __cplusplus
extern "C" {
#endif

extern jmp_buf fpe_jmpbuf;

#ifndef Sig_ret_type
#define Sig_ret_type void
#endif
 Sig_ret_type
#ifdef KR_headers
fpecatch(n) int n;
#else
fpecatch(int n)
#endif
{ longjmp(fpe_jmpbuf, 1); }

#ifdef __cplusplus
}
#endif
