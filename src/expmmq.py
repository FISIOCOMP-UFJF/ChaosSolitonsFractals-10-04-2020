import numpy as np

# Ajuste de curva MMQ
def exp_mmq(dia, casos):
    n = len(dia)
    phi_0 = np.ones(n)
    phi_1 = dia
    F = np.log(casos)	
    A = np.array([[np.dot(phi_0,phi_0), np.dot(phi_0,phi_1)],
                  [np.dot(phi_1,phi_0),np.dot(phi_1,phi_1)]])
    b = np.array([np.dot(F,phi_0), np.dot(F,phi_1)])
    x = np.linalg.solve(A, b)	
    return np.exp(x[0]), x[1]
