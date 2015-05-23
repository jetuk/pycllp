#include "Quad.h"

// /* inline */ members

// double  high( double hi ) { return hi; }
/* inline */ double  Quad::high() const { return hi; }
/* inline */ double  Quad::low() const { return lo; }

/* inline */ Quad::Quad() {}
/* inline */ Quad::Quad(const Quad& y) :hi(y.high()), lo(y.low()) {}
/* inline */ Quad::Quad(double h, double l) :hi(h), lo(l) {}

/* inline */ Quad::~Quad() {}

/* inline */ Quad&  Quad::operator =  (const Quad& y) 
{ 
  hi = y.high(); lo = y.low(); return *this; 
} 

/* inline */ Quad&  Quad::operator += (const Quad& y)
{ 
  // re += y.real();  im += y.imag(); return *this; 
  *this = *this + y; return *this;
}

/* inline */ Quad&  Quad::operator += (double y)
{ 
  // re += y; return *this; 
  *this = *this + y; return *this;
}

/* inline */ Quad&  Quad::operator -= (const Quad& y)
{ 
  // re -= y.real();  im -= y.imag(); return *this; 
  *this = *this - y; return *this;
}

/* inline */ Quad&  Quad::operator -= (double y)
{ 
  // re -= y; return *this; 
  *this = *this - y; return *this;
}

/* inline */ Quad&  Quad::operator *= (const Quad& y)
{  
  // double r = re * y.real() - im * y.imag();
  // im = re * y.imag() + im * y.real(); 
  // re = r; 
  // return *this; 
  *this = *this * y; return *this;
}

/* inline */ Quad&  Quad::operator *= (double y)
{  
  // re *=  y; im *=  y; return *this; 
  *this = *this * y; return *this;
}

/* inline */ Quad&  Quad::operator /= (const Quad& y)
{
  *this = *this / y; return *this;
}

/* inline */ Quad&  Quad::operator /= (double y)
{
  *this = *this / y; return *this;
}

/* inline */        Quad::operator double()
{
  return hi;
}


//  functions

/* inline */ int  operator == (const Quad& x, const Quad& y)
{
  return x.high() == y.high() && x.low() == y.low();
}

/* inline */ int  operator == (const Quad& x, double y)
{
  return x.low() == 0.0 && x.high() == y;
}

/* inline */ int  operator != (const Quad& x, const Quad& y)
{
  return x.high() != y.high() || x.low() != y.low();
}

/* inline */ int  operator != (const Quad& x, double y)
{
  return x.low() != 0.0 || x.high() != y;
}

/* inline */ int  operator >  (const Quad& x, const Quad& y)
{
  if ( x.high() != y.high() )
    return ( x.high() > y.high() );
  else
    return ( x.low() > y.low() );
}

/* inline */ int  operator >  (const Quad& x, double y)
{
  if ( x.high() != y )
    return ( x.high() > y );
  else
    return ( x.low() > 0.0 );
}

/* inline */ int  operator >= (const Quad& x, const Quad& y)
{
  if ( x.high() != y.high() )
    return ( x.high() >= y.high() );
  else
    return ( x.low() >= y.low() );
}

/* inline */ int  operator >= (const Quad& x, double y)
{
  if ( x.high() != y )
    return ( x.high() >= y );
  else
    return ( x.low() >= 0.0 );
}

/* inline */ int  operator <  (const Quad& x, const Quad& y)
{
  if ( x.high() != y.high() )
    return ( x.high() < y.high() );
  else
    return ( x.low() < y.low() );
}

/* inline */ int  operator <  (const Quad& x, double y)
{
  if ( x.high() != y )
    return ( x.high() < y );
  else
    return ( x.low() < 0.0 );
}

/* inline */ int  operator <= (const Quad& x, const Quad& y)
{
  if ( x.high() != y.high() )
    return ( x.high() <= y.high() );
  else
    return ( x.low() <= y.low() );
}

/* inline */ int  operator <= (const Quad& x, double y)
{
  if ( x.high() != y )
    return ( x.high() <= y );
  else
    return ( x.low() <= 0.0 );
}

/* inline */ Quad  operator - (const Quad& x)
{
  return Quad(-x.high(), -x.low());
}

/* inline */ Quad  operator + (const Quad& x, const Quad& y)
{
  double r, s;

  r = x.high() + y.high();
  if (fabs(x.high()) > fabs(y.high()))
    s = x.high() - r + y.high() + y.low() + x.low();
  else
    s = y.high() - r + x.high() + x.low() + y.low();
  return Quad(r+s, r - (r+s) + s);
}

/* inline */ Quad  operator + (const Quad& x, double y)
{
  double r, s;

  r = x.high() + y;
  if (fabs(x.high()) > fabs(y))
    s = x.high() - r + y + x.low();
  else
    s = y - r + x.high() + x.low();
  return Quad(r+s, r - (r+s) + s);
}

/* inline */ Quad  operator + (double x, const Quad& y)
{
  double r, s;

  r = x + y.high();
  if (fabs(x) > fabs(y.high()))
    s = x - r + y.high() + y.low();
  else
    s = y.high() - r + x + y.low();
  return Quad(r+s, r - (r+s) + s);
}

/* inline */ Quad  operator - (const Quad& x, const Quad& y)
{
  double r, s;

  r = x.high() - y.high();
  if (fabs(x.high()) > fabs(y.high()))
    s = x.high() - r - y.high() - y.low() + x.low();
  else
    s = -y.high() - r + x.high() + x.low() - y.low();
  return Quad(r+s, r - (r+s) + s);
}

/* inline */ Quad  operator - (const Quad& x, double y)
{
  double r, s;

  r = x.high() - y;
  if (fabs(x.high()) > fabs(y))
    s = x.high() - r - y + x.low();
  else
    s = -y - r + x.high() + x.low();
  return Quad(r+s, r - (r+s) + s);
}

/* inline */ Quad  operator - (double x, const Quad& y)
{
  double r, s;

  r = x - y.high();
  if (fabs(x) > fabs(y.high()))
    s = x - r - y.high() - y.low();
  else
    s = -y.high() - r + x - y.low();
  return Quad(r+s, r - (r+s) + s);
}

#define MYMAXDOUBLE 170141183460469229370504062281061498880.0

Quad multstep(double Num1, double Num2)
{
        double h1, t1, h2, t2, p, q;
	const int t = 56; /* The number of bits in the mantissa on the machine
                     representation of the double */
	const double MultCon = pow(2.0,(double) (t-t/2)) + 1.0e0;
	const double MaxValue = MYMAXDOUBLE/MultCon;


        if (Num1 >= MaxValue) {
                h1=Num1;
                t1=0.0e0;
        }
        else {
                p = Num1*MultCon;
                h1 = (Num1 - p) + p;
                t1 = Num1-h1;
        }
        if (Num2 >= MaxValue) {
                h2=Num2;
                t2=0.0;
        }
        else {
                p = Num2*MultCon;
                h2 = Num2 - p + p;
                t2 = Num2 - h2;
        }
        p = h1*h2;
        q = h1*t2 + t1*h2;
        return Quad( p+q, p - (p+q) + q + t1*t2 );
}

/* inline */ Quad  operator * (const Quad& x, const Quad& y)
{
  Quad c;
  double l;

  c = multstep( x.high(), y.high() );
  l = x.high()*y.low() + x.low()*y.high() + c.low();
  return Quad( c.high()+l, c.high() - (c.high()+l) + l );
}

/* inline */ Quad  operator * (const Quad& x, double y)
{
  Quad c;
  double l;

  c = multstep( x.high(), y );
  l = x.low()*y + c.low();
  return Quad( c.high()+l, c.high() - (c.high()+l) + l );
}

/* inline */ Quad  operator * (double x, const Quad& y)
{
  Quad c;
  double l;

  c = multstep( x, y.high() );
  l = x*y.low() + c.low();
  return Quad( c.high()+l, c.high() - (c.high()+l) + l );
}

Quad   operator /  (const Quad& x, const Quad& y)
{
  Quad c, u, result;
  double l;

  c = Quad( x.high()/y.high() );
  u = multstep( c.high(), y.high() );
  l = (x.high() - u.high() - u.low() + x.low() - c.high()*y.low())/y.high();
  return Quad( c.high()+l, c.high() - (c.high()+l) + l );
}

Quad   operator /  (const Quad& x, double y)
{
  Quad c, u, result;
  double l;

  c = Quad( x.high()/y );
  u = multstep( c.high(), y );
  l = (x.high() - u.high() - u.low() + x.low() )/y;
  return Quad( c.high()+l, c.high() - (c.high()+l) + l );
}

Quad   operator /  (double   x, const Quad& y)
{
  Quad c, u, result;
  double l;

  c = Quad( x/y.high() );
  u = multstep( c.high(), y.high() );
  l = (x - u.high() - u.low() - c.high()*y.low())/y.high();
  return Quad( c.high()+l, c.high() - (c.high()+l) + l );
}

/* inline */ Quad  operator << (const Quad& x, const Quad& y)
{
  if (x < y)
    return x;
  else
    return y;
}

/* inline */ Quad  operator >> (const Quad& x, const Quad& y)
{
  if (x > y)
    return x;
  else
    return y;
}

/* inline */ double  high(const Quad& x)
{
  return x.high();
}

/* inline */ double  low(const Quad& x)
{
  return x.low();
}

/* inline */ Quad  abs(const Quad& x)
{
  return Quad( fabs(x.high()), fabs(x.low()));
}

Quad   log10 (const Quad& x)
{
  return Quad( log10( x.high() ) );
}

Quad   log (const Quad& x)
{
  return Quad( log( x.high() ) );
}

Quad   exp (const Quad& x)
{
  return Quad( exp( x.high() ) );
}

Quad   fabs (const Quad& x)
{
  return Quad( fabs( x.high() ) );
}

Quad   sqrt (const Quad& x)
{
  return Quad( sqrt( x.high() ) );
}
