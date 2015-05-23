/* This is a C++ class definition that implements quad precision arithmetic.
   It works with C++ codes.  

   CONVERSION HINTS:
   
   If the code is in C, first convert it to C++.  Most C programs can be 
   converted to C++ by simply changing stdio.h to stream.h and deleting all 
   calls to fflush().

   To use Quad.h, include Quad.h at the top of the program file.  Then
   substitute "Quad" for "double" as desired.

   Quad precision variables are implemented as a pair of doubles, one
   representing the "high" part and another representing the "low" part.
   Two functions 

	double high( Quad x );
	double low(  Quad x );

   extract the high and low parts and return them as doubles.  The high()
   routine is useful for converting I/O statements.  For example,

	replace			with

	printf("%lf", x);       printf("%lf", high(x));

   The implementation of MAX and MIN has been accomplished using new operators
   called ?> and ?<, respectively.  Hence:

	replace			with

	x > y ? x : y           x >? y
	x > y ? y : x           x <? y
	x > 0 ? x : -x          x >? -x
   
   Functions that return doubles must be explicitly converted to Quad before
   storing answer in a Quad variable (this should really be fixed).  Hence,

	replace                 with

	x = atof(str)           x = Quad( atof( str ) )

   Beware: quad precision as implemented in Quad.h is about 50 times slower
   than the same code using doubles!!!
*/

// This may look like C code, but it is really -*- C++ -*-

#include <stream.h>
#include <math.h>

class Quad
{
#ifdef __ATT_complex__
public:
#else
protected:
#endif

  double           hi;
  double           lo;

public:

  double           high() const;
  double           low() const;

                   Quad();
                   Quad(const Quad& y);
                   Quad(double hi, double lo=0);  // hi and lo may be bad labels

                  ~Quad();

  Quad&            operator =  (const Quad& y);

  Quad&            operator += (const Quad& y);
  Quad&            operator += (double y);
  Quad&            operator -= (const Quad& y);
  Quad&            operator -= (double y);
  Quad&            operator *= (const Quad& y);
  Quad&            operator *= (double y);

  Quad&            operator /= (const Quad& y); 
  Quad&            operator /= (double y); 
        	   operator double();

  void             error(const char* msg) const;
};

// non-inline functions

Quad   operator /  (const Quad& x, const Quad& y);
Quad   operator /  (const Quad& x, double y);
Quad   operator /  (double   x, const Quad& y);

// these only have precision of double - this should be fixed

Quad   log10 (const Quad& x);
Quad   log (const Quad& x);
Quad   exp (const Quad& x);
Quad   fabs (const Quad& x);
Quad   sqrt (const Quad& x);

istream&  operator >> (istream& s, Quad& x);
ostream&  operator << (ostream& s, const Quad& x);

// other functions defined as inlines

int  operator == (const Quad& x, const Quad& y);
int  operator == (const Quad& x, double y);
int  operator != (const Quad& x, const Quad& y);
int  operator != (const Quad& x, double y);
int  operator >  (const Quad& x, const Quad& y);
int  operator >  (const Quad& x, double y);
int  operator >= (const Quad& x, const Quad& y);
int  operator >= (const Quad& x, double y);
int  operator <  (const Quad& x, const Quad& y);
int  operator <  (const Quad& x, double y);
int  operator <= (const Quad& x, const Quad& y);
int  operator <= (const Quad& x, double y);

Quad  operator - (const Quad& x);
Quad  operator + (const Quad& x, const Quad& y);
Quad  operator + (const Quad& x, double y);
Quad  operator + (double x, const Quad& y);
Quad  operator - (const Quad& x, const Quad& y);
Quad  operator - (const Quad& x, double y);
Quad  operator - (double x, const Quad& y);
Quad  operator * (const Quad& x, const Quad& y);
Quad  operator * (const Quad& x, double y);
Quad  operator * (double x, const Quad& y);
Quad  operator << (const Quad& x, const Quad& y); // min
Quad  operator >> (const Quad& x, const Quad& y); // max

double  high(const Quad& x);
double  low(const Quad& x);
Quad    abs(const Quad& x);
