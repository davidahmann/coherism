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
        
        data.append(f"{t:.2f} {E_base:.4f} {E_coherist:.4f}")
        
        t += dt
        
    return "\n".join(data)

if __name__ == "__main__":
    csv_content = simulate_wave_equation()
    with open("simulation_data.dat", "w") as f:
        f.write("t E_base E_coherist\n")
        f.write(csv_content)
    print("Simulation data generated: simulation_data.dat")
