from . import BaseSolver

cdef extern from "glpk.h":
    ctypedef struct glp_prob:
        pass
    ctypedef struct glp_smcp:
        int msg_lev

    int GLP_MIN = 1 # minimization
    int GLP_MAX = 2 # maximization

    int GLP_FR = 1 # free (unbounded) variable
    int GLP_LO = 2 # variable with lower bound
    int GLP_UP = 3 # variable with upper bound
    int GLP_DB = 4 # double-bounded variable
    int GLP_FX = 5 # fixed variable

    int GLP_UNDEF = 1 # solution is undefined
    int GLP_FEAS = 2 # solution is feasible
    int GLP_INFEAS = 3 # solution is infeasible
    int GLP_NOFEAS = 4 # no feasible solution exists
    int GLP_OPT = 5 # solution is optimal
    int GLP_UNBND = 6 # solution is unbounded

    int GLP_MSG_OFF = 0 # no output
    int GLP_MSG_ERR = 1 # warning and error messages only
    int GLP_MSG_ON = 2 # normal output
    int GLP_MSG_ALL = 3 # all output
    int GLP_MSG_DBG = 4 # debug output

    glp_prob* glp_create_prob()
    void glp_init_smcp(glp_smcp *parm)
    void glp_erase_prob(glp_prob *P)
    void glp_delete_prob(glp_prob *P)
    void glp_free(void *ptr)

    int glp_add_rows(glp_prob *P, int nrs)
    int glp_add_cols(glp_prob *P, int ncs)

    void glp_set_mat_row(glp_prob *P, int i, int len, const int ind[], const double val[])
    void glp_set_mat_col(glp_prob *P, int j, int len, const int ind[], const double val[])
    void glp_load_matrix(glp_prob *P, int ne, const int ia[], const int ja[], const double ar[])

    void glp_set_row_bnds(glp_prob *P, int i, int type, double lb, double ub)
    void glp_set_col_bnds(glp_prob *P, int j, int type, double lb, double ub)
    void glp_set_obj_coef(glp_prob *P, int j, double coef)

    void glp_set_obj_dir(glp_prob *P, int dir);

    int glp_simplex(glp_prob *P, const glp_smcp *parm)

    int glp_get_status(glp_prob *P)

    double glp_get_row_prim(glp_prob *P, int i)
    double glp_get_col_prim(glp_prob *P, int j)



def glpk_solve(int[:] ia, int[:] ja, double[:] ar, double[:] b, double[:] c, double[:] x):
    cdef int i, ret
    cdef int m = b.shape[0]
    cdef int n = c.shape[0]
    cdef int ne = ar.shape[0]
    cdef glp_prob* prob
    cdef glp_smcp smcp


    prob = glp_create_prob()
    glp_init_smcp(&smcp)
    glp_set_obj_dir(prob, GLP_MAX)

    glp_add_rows(prob, m)
    glp_add_cols(prob, n)

    glp_load_matrix(prob, ne, &ia[0]-1, &ja[0]-1, &ar[0]-1)

    for i in range(m):
        glp_set_row_bnds(prob, i+1, GLP_FX, b[i], b[i])

    for i in range(n):
        glp_set_obj_coef(prob, i+1, c[i])
        glp_set_col_bnds(prob, i+1, GLP_LO, 0.0, 0.0)

    ret = glp_simplex(prob, &smcp)

    for i in range(n):
        x[i] = glp_get_col_prim(prob, i+1)

    return ret


