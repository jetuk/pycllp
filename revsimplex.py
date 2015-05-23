"""
File       rsm.py
Desc       Revised Simplex Method for Linear Programming using matrix methods.
Author     Ernesto P. Adorio, Ph.D.
           UPDEPP, UP at Clark
email      ernesto . adorio @ gmail . com
Reference: Wayne Winston, Introduction to Mathematical Programming, 2e
           Duxbury Press, pp. 567-572. 1995, QA402.5W53, ISBN 0-5324-23046-6.

Revisions  2009.02.11  first educational version.
"""

import numpy as np


def RevSimplex(cz, A, b):
   """
    performs the revised simplex algorithm as described in
    Hillier & Lieberman, "Intro. to Operations Research", 6th ed.
    The linear programming problem is

    Maximize Z = cz x

    subject to
    AX <= b;   x >= 0

    Input:
     cz array of size n of variable costs
     A  m by n size coefficient matrix.
     b  column vector of size m

    Return values
     Index  indicator vector
     Xb basis vector.
     X  an n + m column vector.
        X[0] to X[n-1] are nonslack variables.
        X[n] to X[n+m -1] are slack variables.
     B  an m by m basis matrix
   """
   def rscol(A, j):
      print "rscol input j = ", j
      n,m = A.shape
      if j < m:
         return A[:,j]
      if j < m+n:
         E = [0] * n
         E[j - m] = 1.0
         return E
      print "Error in rscol, j= ", j, "out of bounds"
      return None

   n,m = A.shape
   costZ = cz[:]
   costZ.extend([0.0]* m)

   Index = range(m, m+n)

   Binv = np.eye(n)   # B^(-1)

   # starting basis vector cost.
   cbv = []
   for j in Index:
     if  j < n:
       cbv.append(cz[j])
     else:
       cbv.append(0.0)
   print "Starting Cost vector for basis:", cbv



   K = m + n + 1 # g = 0: no greater thans ">" constraints.
   maxiter = 2 * K
   for iter in range(maxiter):
       print "Iteration #", iter
       print m,n, Binv, cbv
       print "Index=", Index
       cbvBinv = np.dot(cbv, Binv)
       print "iter #", iter, " cbvBinv =", cbvBinv

       # Price out each nobasic variables.
       minpriceout = 0.0
       priceoutvariable = None
       for i in range(m+n):
          if i not in Index :
             print "rscol(A, ", i,"= ", rscol(A, i)
             priceout = np.dot(cbvBinv, rscol(A, i)) - costZ[i]
             print "pricing out variable", i, priceout
             if priceout < minpriceout:
                minpriceout = priceout
                entvar = i
       print "minpriceout=", minpriceout
       if minpriceout >= 0:
          print "Optimal value found!"
          break
       print "entering variable = ", entvar

       BinvAj = np.dot(Binv, rscol(A, entvar))
       Binvb  = np.dot(Binv, b)

       print "BinvAj=",BinvAj
       print "Binvb =",Binvb

       #Ratio Test:
       minratio = 1e30
       for i in range(n):
          r = Binvb[i]/BinvAj[i]
          print"i", i, "ratio= ", r
          if r > 0 and r < minratio:
             minrow = i
             minratio = r
       print "minratio = ", minratio, "at row", minrow
       Index[minrow] = entvar
       print "Updated Index = ", Index
       cbv[minrow] = costZ[entvar]
       print "Updated basis cost vector cbv =", cbv

       #Perform elementary row operations on B^(-1).
       print ("Binv:")
       print(Binv)

       # revised  row:
       k = float(BinvAj[minrow])
       print "k = ", k
       # other rows:
       for i in range(n):
          if i != minrow:
            mult = BinvAj[i] / k
            print i, mult
            for j in range(n):
                Binv[i][j] = Binv[i][j] - mult * Binv[minrow][j]
       for j in range(n):
          Binv[minrow][j] /= k

       print "new Basis matrix"
       print(Binv)
   print "cbv= ", cbv
   print "b = ", b
   print "Binv=", Binv
   Binvb  = np.dot(Binv, b)

   print "Right hand side:", Binvb
   zmax = np.dot(cbv, Binvb)
   print "Optimal value = ",
   print "Index=", Index
   return zmax, Index, Binv, Binvb

if __name__ == "__main__":

   """
   c = [3,5]
   A = [[10,1],
        [0,2,0],
        [3,2,0]]
   b     = [4, 12, 18]
   m,n   = 3, 2
   Index = range(2, 5)
   B = eye(m, m)
   c_B= [0,0,0]
   """
   # Example in Chapter 10 of Winston
   cz = [2,3,4]     # coefficients of variables in objective function.
                         # Program will automatically add subvector
   A = np.array(
        [[3,2,1],   # Coefficient matrix,
        [2,5,3]])   # Only for "<" constraint equations!

   b = [10, 15]       # Right hand side vector.

   zmax, Index, Binv, Binvb = RevSimplex(cz, A, b)
   print zmax
   print np.dot(A, Index)
