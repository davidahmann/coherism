import math

def simulate_wave_equation(steps=200, dt=0.05, kappa=0.5):
    # Simulate a single mode harmonic oscillator (0D QFT) for clarity
    # phi'' + phi + kappa * (phi)^2 * phi' = 0
    
    phi = 1.0
    phi_dot = 0.0
    
    # Baseline (kappa=0)
    phi_base = 1.0
    phi_dot_base = 0.0
    
    data = []
    
    t = 0.0
    for i in range(steps):
        # Coherist Friction: F = -kappa * Theta_00 * v
        # Assume Theta_00 ~ phi^2 (energy density fluctuation)
        friction = -kappa * (phi**2) * phi_dot
        
        # Update Coherist
        phi_ddot = -phi + friction
        phi_dot += phi_ddot * dt
        phi += phi_dot * dt
        
        # Update Baseline
        phi_ddot_base = -phi_base
        phi_dot_base += phi_ddot_base * dt
        phi_base += phi_dot_base * dt
        
        # Calculate Energy (Amplitude proxy)
        E_coherist = 0.5 * (phi_dot**2 + phi**2)
        E_base = 0.5 * (phi_dot_base**2 + phi_base**2)
        
        data.append((t, E_base, E_coherist))
        
        t += dt
        
    return data

if __name__ == "__main__":
    data = simulate_wave_equation()

    with open("simulation_data.dat", "w") as f:
        f.write("t E_base E_coherist\n")
        for (t, e_base, e_coh) in data:
            f.write(f"{t:.4f} {e_base:.8f} {e_coh:.8f}\n")
    print("Simulation data generated: simulation_data.dat")

    try:
        import matplotlib.pyplot as plt

        t = [row[0] for row in data]
        e_base = [row[1] for row in data]
        e_coh = [row[2] for row in data]

        plt.figure(figsize=(7, 4))
        plt.plot(t, e_base, label=r'Baseline ($\kappa=0$)', linewidth=2)
        plt.plot(t, e_coh, label=r'Coherist friction ($\kappa>0$)', linewidth=2)
        plt.xlabel('t')
        plt.ylabel(r'$E = \frac{1}{2}(\dot{\phi}^2 + \phi^2)$')
        plt.title('Toy Coherist Friction: Energy Saturation')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("toy_coherist_friction.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Figure generated: toy_coherist_friction.png")
    except Exception as exc:
        print(f"Skipping plot generation (matplotlib unavailable?): {exc}")
