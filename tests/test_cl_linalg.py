import pyopencl as cl
import pycllp.cl
from pycllp.linalg import conjgrad
import numpy as np

if __name__ == '__main__':
    import os
    ctx = cl.create_some_context()
    directory = os.path.dirname(pycllp.cl.__file__)

    src = open(os.path.join(directory, 'conjugate_gradient.cl')).read()
    cl_prg = cl.Program(ctx, src).build(options=["-g"])

    m = 4
    nlp = 2
    np.random.seed(1234)
    A = np.empty((nlp, m, m))
    for ilp in range(nlp):
        AA = np.random.rand(m, m).astype(np.float32)
        A[ilp, ...] = np.dot(AA.T, AA)
    A = A.reshape(nlp*m*m)
    b = np.random.rand(m*nlp).astype(np.float32)
    x = np.zeros(m*nlp).astype(np.float32)
    x_np = x.copy()

    mf = cl.mem_flags
    g_A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    g_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    g_x = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=x)
    l_r = cl.LocalMemory(4*m)
    l_p = cl.LocalMemory(4*m)

    with cl.CommandQueue(ctx) as queue:
        cl_prg.conjgrad(queue, (nlp*m,), (m,),
                        np.int32(m), np.int32(100), g_A, g_b, g_x, l_r, l_p)
        cl.enqueue_copy(queue, x, g_x)

    print(x)
    for ilp in range(nlp):
        s = m*ilp
        e = m*(ilp+1)
        print(conjgrad(A[m*s:m*e].reshape(m, m), b[s:e], x_np[s:e]))

    print(x_np)

    x_np_la = np.linalg.solve(A[:m*m].reshape(m, m), b[:m])
    print(x_np_la)
