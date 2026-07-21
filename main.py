"""
Earth-Moon-Probe Orbital Dynamics Simulator
Author: Bo Bryan-Low
Description: Solves the restricted 3-body system equations of motion using 
scipy.integrate.solve_ivp to simulate lunar and spacecraft trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si

# Physical Constants (SI Units)
G = 6.67430e-11   # Universal Gravitational Constant (N m^2 / kg^2)
M_EARTH = 5.972e24 # Mass of Earth (kg)
M_MOON = 7.34767309e22 # Mass of Moon (kg)
R_MOON = 384400e3  # Earth-Moon distance (m)


def derivatives_moon(time: float, state: tuple, g_const: float, m_earth: float) -> tuple:
    """Computes derivatives for the 2-body Earth-Moon system."""
    xm, ym, vxm, vym = state
    rm = np.sqrt(xm**2 + ym**2)

    ax = -g_const * m_earth * xm / rm**3
    ay = -g_const * m_earth * ym / rm**3

    return (vxm, vym, ax, ay)


def derivatives_probe(time: float, state: tuple, g_const: float, m_earth: float, m_moon: float) -> tuple:
    """Computes derivatives for the 3-body Earth-Moon-Probe system."""
    xm, ym, vxm, vym, xp, yp, vxp, vyp = state

    rm = np.sqrt(xm**2 + ym**2)
    rp = np.sqrt(xp**2 + yp**2)
    rpm = np.sqrt((xp - xm)**2 + (yp - ym)**2)

    ax_m = -g_const * m_earth * xm / rm**3
    ay_m = -g_const * m_earth * ym / rm**3

    ax_p = -g_const * m_earth * xp / rp**3 - g_const * m_moon * (xp - xm) / rpm**3
    ay_p = -g_const * m_earth * yp / rp**3 - g_const * m_moon * (yp - ym) / rpm**3

    return (vxm, vym, ax_m, ay_m, vxp, vyp, ax_p, ay_p)


def simulate_lunar_orbit(t_max_days: float = 27.5, num_points: int = 2000):
    """Simulates and plots the Earth-Moon system."""
    t0 = 0.0
    t_max = t_max_days * 24 * 3600
    times = np.linspace(t0, t_max, num_points)

    initial_state = (R_MOON, 0.0, 0.0, np.sqrt(G * M_EARTH / R_MOON))

    results = si.solve_ivp(
        derivatives_moon,
        (t0, t_max),
        initial_state,
        t_eval=times,
        args=(G, M_EARTH),
        rtol=1e-9,
        atol=1e-12
    )

    plt.figure(figsize=(8, 8))
    plt.plot(0, 0, '.', markersize=20, label="Earth")
    plt.plot(results.y[0], results.y[1], label="Moon Orbit")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Orbit of the Moon around the Earth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def simulate_probe_system(t_max_days: float = 27.5, num_points: int = 2000):
    """Simulates and plots the Earth-Moon-Probe 3-body system."""
    t0 = 0.0
    t_max = t_max_days * 24 * 3600
    times = np.linspace(t0, t_max, num_points)

    r_probe_alt = 5995e3
    xp0 = R_MOON + r_probe_alt
    vym0 = np.sqrt(G * M_EARTH / R_MOON)
    vyp0 = vym0 + np.sqrt(G * M_MOON / r_probe_alt)

    initial_state = (R_MOON, 0.0, 0.0, vym0, xp0, 0.0, 0.0, vyp0)

    results = si.solve_ivp(
        derivatives_probe,
        (t0, t_max),
        initial_state,
        t_eval=times,
        args=(G, M_EARTH, M_MOON),
        rtol=1e-9,
        atol=1e-12
    )

    # Full System View
    plt.figure(figsize=(8, 8))
    plt.plot(0, 0, '.', markersize=20, label="Earth")
    plt.plot(results.y[0], results.y[1], label="Moon Orbit")
    plt.plot(results.y[4], results.y[5], label="Probe Orbit")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Earth-Moon-Probe System Trajectory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    print("Running Earth-Moon-Probe trajectory simulation...")
    simulate_probe_system()
