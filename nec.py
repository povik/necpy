import numpy as np
import scipy.integrate as integrate


def base_function(d_m, d, d_p, k=1.0):
    z = lambda d_: np.sin(k*d_/2)/np.cos(k*d_/2)

    if len(d_p) != 0 and len(d_m) != 0:
        z_m, z_p = np.sum(z(d_m)), np.sum(-z(d_p))
        s, c = np.sin(k*d/2), np.cos(k*d/2)
        det = -2*c*s - 2*z_m*z_p*c*s + z_p*(-s*s + c*c) + z_m*(-c*c + s*s)
        A = -1.0
        B = -A*(z_p+z_m)*s/det
        C = -A*(-2*s+(z_p-z_m)*c)/det
        
        s_m, c_m = np.sin(k*d_m/2), np.cos(k*d_m/2)
        s_p, c_p = np.sin(k*d_p/2), np.cos(k*d_p/2)
        
        return np.concatenate(
            (np.array([(s_m/c_m + c_m/s_m)/2, 1/(2*c_m), -1/(2*s_m)])*(B*c+C*s),
            [[A], [B], [C]],
            -np.array([(s_p/c_p + c_p/s_p)/2, -1/(2*c_p), -1/(2*s_p)])*(B*c-C*s)),
            axis=1
        )

    if len(d_p) == 0 and len(d_m) != 0:
        z_m = np.sum(z(d_m))
        s, c = np.sin(k*d/2), np.cos(k*d/2)
        det = -2*s*c + z_m*(-c*c + s*s)
        A = -1.0
        B = A*(-s*z_m)/det
        C = A*(c*z_m+2*s)/det

        s_m, c_m = np.sin(k*d_m/2), np.cos(k*d_m/2)
        
        return np.concatenate(
            (np.array([(s_m/c_m + c_m/s_m)/2, 1/(2*c_m), -1/(2*s_m)])*(B*c+C*s),
             [[A], [B], [C]]),
            axis=1
        )

    if len(d_p) != 0 and len(d_m) == 0:
        flipped = base_function(d_p, d, d_m, k=k)
        return np.concatenate(
            (flipped[:,-1].reshape((-1,1)), flipped[:,:-1]),
            axis=1
        )*np.array([[1],[-1],[1]])

    raise NotImplementedError()


def arange(lengths, coeffs, k=1.0):
    starts = [np.sum(lengths[:n]) for n in range(len(lengths))]
    offsets = [np.sum(lengths[:n])+lengths[n]/2 for n in range(len(lengths))]
    end = np.sum(lengths)
    
    g = list(reversed(list(zip(range(len(lengths)), starts, list(offsets)))))

    def f(x):
        if x > end:
             return 0
        for i, start, offset in g:
            if x > start:
                x_ = float(x-offset)
                coeff = coeffs[:,i]
                return coeff[0]+coeff[1]*np.sin(k*x_)+coeff[2]*np.cos(k*x_)
        return 0
    return f


def reverse(coeffs, revmap):
    return np.array([
        coeffs[0,:] * (1.0 - 2*revmap),
        coeffs[1,:],
        coeffs[2,:] * (1.0 - 2*revmap)
    ])


def centre(vertices, edges):
    max_ = np.max(vertices, axis=0)
    min_ = np.min(vertices, axis=0)
    
    return vertices - (max_ + min_)/2, edges


def grid(Nx, dx, Ny, dy):
    vertices = np.mgrid[0:Nx,0:Ny,0:1].T.reshape(-1,3)*np.array([dx, dy, 1.0]).reshape(1,3)
    edges = []
    for x in range(Nx):
        for y in range(Ny-1):
            edges += [(Nx*y+x, Nx*(y+1)+x)]
    for x in range(Nx-1):
        for y in range(Ny):
            edges += [(Nx*y+x+1, Nx*y+x)]
    vertices, edges = np.array(vertices), np.array(edges)
    return centre(vertices, edges)


def excitemat(vertices, edges):
    #  v------- sampling segment
    # (N, N)
    #     ^---- emitting segment

    v_a, v_b = vertices[edges[:,0]], vertices[edges[:,1]]

    v_d = np.sqrt(np.sum((v_b - v_a)**2, axis=1))
    v_n = (v_b - v_a) / v_d.reshape((-1, 1))

    m = (v_b + v_a) / 2

    z_a, z_b = (-v_d/2).reshape(1, -1), (v_d/2).reshape(1, -1)
    z_   = np.sum(m.reshape((-1, 1, 3))* v_n.reshape((1, -1, 3)), axis=2) \
           - np.sum(m*v_n, axis=1).reshape((1, -1))
    rho_ = np.sqrt(np.sum((m.reshape((-1, 1, 3)) - m.reshape((1, -1, 3)))**2, axis=2) - z_**2)

    bounds = lambda f, a, b: f(b) - f(a)

    eta = 1.0
    lambd = 1.0
    k = 1.0

    #G0 = np.exp(-1j*k*)

    # E_sin_rho = -1j*eta/(lambd*2*k**2*rho_)*G0()  
    # E_sin_z   =
    # E_cos_rho = 
    # E_cos_z   =
    # E_const_rho = 
    # E_const_z   = 

