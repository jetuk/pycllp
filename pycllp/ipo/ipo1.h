#define DECLAREALL \
\
int m; \
int n; \
int nz; \
double *A; \
int *iA; \
int *kA; \
double *b; \
double *c; \
double f; \
double *r; \
double *l; \
double *u; \
double *At; \
int *iAt; \
int *kAt; \
double *w; \
double *x; \
double *y; \
double *z; \


#define	COPYBACK(lp) \
\
m = lp->m; \
n = lp->n; \
nz = lp->nz; \
A = lp->A; \
iA = lp->iA; \
kA = lp->kA; \
b = lp->b; \
c = lp->c; \
f = lp->f; \
r = lp->r; \
l = lp->l; \
u = lp->u; \
varsgn = lp->varsgn; \
rowlab = lp->rowlab; \
collab = lp->collab; \
qnz = lp->qnz; \
Q = lp->Q; \
iQ = lp->iQ; \
kQ = lp->kQ; \
At = lp->At; \
iAt = lp->iAt; \
kAt = lp->kAt; \
bndmark = lp->bndmark; \
rngmark = lp->rngmark; \
w = lp->w; \
x = lp->x; \
y = lp->y; \
z = lp->z; \
p = lp->p; \
q = lp->q; \
s = lp->s; \
t = lp->t; \
v = lp->v; \
ub = lp->ub; \
max = lp->max; \
inftol = lp->inftol; \
inftol2 = lp->inftol2; \
sf_req = lp->sf_req; \
itnlim = lp->itnlim; \
timlim = lp->timlim; \
verbose = lp->verbose; \
name = lp->name; \
obj = lp->obj; \
rhs = lp->rhs; \
ranges = lp->ranges; \
bounds = lp->bounds; \
param = lp->param; \
np = lp->np; \
stopping_rule = lp->stopping_rule; \
init_vars = lp->init_vars; \
h_init = lp->h_init; \
h_update = lp->h_update; \
h_step = lp->h_step; \
/*		Commented out since changing these in a hook can't help \
iter = lp->iter; \
elaptime = lp->elaptime; \
pres = lp->pres; \
dres = lp->dres; \
sigfig = lp->sigfig; \
primal_obj = lp->primal_obj; \
dual_obj = lp->dual_obj;
*/
