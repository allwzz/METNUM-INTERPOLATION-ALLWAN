import numpy as np
import matplotlib.pyplot as plt

# Data yang diberikan
x_vals = np.array([5, 10, 15, 20, 25, 30, 35, 40])
y_vals = np.array([40, 30, 25, 40, 18, 20, 22, 15])

# Fungsi untuk interpolasi Lagrange
def lagrange_interp(x, x_pts, y_pts):
    def basis(k):
        return np.prod([(x - x_pts[j]) / (x_pts[k] - x_pts[j]) for j in range(len(x_pts)) if j != k])

    return sum(y_pts[k] * basis(k) for k in range(len(x_pts)))

# Fungsi untuk interpolasi Newton
def newton_interp(x, x_pts, y_pts):
    def div_diff(x_pts, y_pts):
        n = len(y_pts)
        table = np.zeros((n, n))
        table[:, 0] = y_pts

        for j in range(1, n):
            for i in range(n - j):
                table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x_pts[i + j] - x_pts[i])

        return table[0, :]

    coeffs = div_diff(x_pts, y_pts)
    n = len(coeffs)
    newton_poly = coeffs[0]

    for k in range(1, n):
        term = coeffs[k]
        for j in range(k):
            term *= (x - x_pts[j])
        newton_poly += term

    return newton_poly

# Plotting
x_plot = np.linspace(5, 40, 400)
y_lagrange_plot = [lagrange_interp(x, x_vals, y_vals) for x in x_plot]
y_newton_plot = [newton_interp(x, x_vals, y_vals) for x in x_plot]

plt.figure(figsize=(12, 6))
plt.plot(x_plot, y_lagrange_plot, label='Interpolasi Lagrange', color='orange') # Warna garis diubah menjadi oranye
plt.plot(x_plot, y_newton_plot, label='Interpolasi Newton', color='purple')      # Warna garis diubah menjadi ungu
plt.scatter(x_vals, y_vals, color='red', zorder=5)
plt.title('Interpolasi menggunakan Metode Lagrange dan Newton')
plt.xlabel('Tegangan, x (kg/mm²)')
plt.ylabel('Waktu patah, y (jam)')
plt.legend()
plt.grid(True)
plt.show()
