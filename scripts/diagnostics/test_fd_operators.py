"""Checks for fd_two_forms.py.

1. Manufactured-solution test: the "div" stencil converges at second order to div(sigma) and the
   "local" stencil to C : grad grad u, for smooth u, lambda(y), mu(y).
2. One-dimensional transmission test: a plane P pulse crossing one smoothed interface at normal
   incidence. Sharp-interface theory gives a transmitted displacement amplitude of
   2 Z1 / (Z1 + Z2) for the divergence form and 2 c2 / (c1 + c2) for the local form.
"""
import numpy as np
import sympy as sp

import fd_two_forms as fd


def manufactured() -> None:
    x, y = sp.symbols("x y")
    lam = 2 + sp.sin(1.3 * y)
    mu = 1 + sp.cos(0.7 * y) / 2
    ux = sp.sin(1.1 * x + 0.6 * y)
    uy = sp.cos(0.8 * x - 1.2 * y)
    exx, eyy, exy = sp.diff(ux, x), sp.diff(uy, y), (sp.diff(ux, y) + sp.diff(uy, x)) / 2
    sxx, syy, sxy = lam * (exx + eyy) + 2 * mu * exx, lam * (exx + eyy) + 2 * mu * eyy, 2 * mu * exy
    div_x, div_y = sp.diff(sxx, x) + sp.diff(sxy, y), sp.diff(sxy, x) + sp.diff(syy, y)
    # local form: differentiate the stress with lambda, mu held constant
    L, M = sp.symbols("L M")
    sxx_c, syy_c, sxy_c = L * (exx + eyy) + 2 * M * exx, L * (exx + eyy) + 2 * M * eyy, 2 * M * exy
    loc_x = (sp.diff(sxx_c, x) + sp.diff(sxy_c, y)).subs({L: lam, M: mu})
    loc_y = (sp.diff(sxy_c, x) + sp.diff(syy_c, y)).subs({L: lam, M: mu})
    f = {k: sp.lambdify((x, y), v, "numpy") for k, v in
         dict(lam=lam, mu=mu, ux=ux, uy=uy, div_x=div_x, div_y=div_y, loc_x=loc_x, loc_y=loc_y).items()}
    print("manufactured solution: max error of the discrete operator")
    for form, kx, ky in (("div", "div_x", "div_y"), ("local", "loc_x", "loc_y")):
        errs = []
        for n in (41, 81, 161):
            g = np.linspace(0.0, 2.0, n)
            h = g[1] - g[0]
            X, Y = np.meshgrid(g, g, indexing="ij")
            rhs = fd.make_rhs(form, f["lam"](0 * g, g), f["mu"](0 * g, g), h)
            fx, fy = rhs(f["ux"](X, Y), f["uy"](X, Y))
            ex = np.abs(fx * fd.RHO - f[kx](X, Y)[1:-1, 1:-1]).max()
            ey = np.abs(fy * fd.RHO - f[ky](X, Y)[1:-1, 1:-1]).max()
            errs.append(max(ex, ey))
        orders = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
        print(f"  {form:5s}: errors {errs[0]:.2e} {errs[1]:.2e} {errs[2]:.2e}  observed order {orders[0]:.2f} {orders[1]:.2f}")


def transmission() -> None:
    lam1, mu1, lam2, mu2, rho = 40.0, 25.0, 50.0, 35.0, fd.RHO  # layers 3 and 4 of the benchmark
    c1, c2 = np.sqrt((lam1 + 2 * mu1) / rho), np.sqrt((lam2 + 2 * mu2) / rho)
    h, dt = 1e-3, 2.5e-4
    y = np.arange(-3.0, 3.0 + h / 2, h)
    step = 0.5 * (np.tanh(fd.SHARPNESS * y) + 1)
    a = (lam1 + 2 * mu1) * (1 - step) + (lam2 + 2 * mu2) * step
    a_up, a_dn = 0.5 * (a[1:-1] + a[2:]), 0.5 * (a[1:-1] + a[:-2])
    width = 0.1
    print("normal-incidence P pulse, layer 3 -> layer 4 (c1 = %.3f, c2 = %.3f)" % (c1, c2))
    theory = {"div": 2 * c1 / (c1 + c2), "local": 2 * c2 / (c1 + c2)}  # constant density: Z = rho c
    for form in ("div", "local"):
        pulse = lambda s: np.exp(-0.5 * (s / width) ** 2)
        u_old = pulse(y + 1.0 + c1 * dt)  # right-going pulse centered at y = -1
        u = pulse(y + 1.0)
        for _ in range(int(round(1.9 / dt))):
            if form == "div":
                f = a_up * (u[2:] - u[1:-1]) - a_dn * (u[1:-1] - u[:-2])
            else:
                f = a[1:-1] * (u[2:] - 2 * u[1:-1] + u[:-2])
            u_new = u.copy()
            u_new[1:-1] = 2 * u[1:-1] - u_old[1:-1] + dt**2 * f / (rho * h**2)
            u_old, u = u, u_new
        t_amp = u[y > 0].max()
        r_amp = u[y < 0][np.abs(u[y < 0]).argmax()]
        print(f"  {form:5s}: transmitted {t_amp:.4f} (sharp-interface theory {theory[form]:.4f}), "
              f"reflected {r_amp:+.4f} (theory {theory[form] - 1:+.4f})")


if __name__ == "__main__":
    manufactured()
    transmission()
