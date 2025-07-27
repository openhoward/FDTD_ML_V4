

import meep as mp
import numpy as np

def run_meep_simulation(w1, w2, Lw):
    resolution = 25  # pixels/μm
    Lts = [2**m for m in range(4)]  # taper lengths
    dair, dpml_x, dpml_y = 3.0, 6.0, 2.0
    sy = dpml_y + dair + w2 + dair + dpml_y
    Si = mp.Medium(epsilon=12.0)
    boundary_layers = [mp.PML(dpml_x, direction=mp.X), mp.PML(dpml_y, direction=mp.Y)]
    lcen = 6.67
    fcen = 1 / lcen
    symmetries = [mp.Mirror(mp.Y)]

    R_flux_list = []
    for Lt in Lts:
        sx = dpml_x + Lw + Lt + Lw + dpml_x
        cell_size = mp.Vector3(sx, sy, 0)

        src_pt = mp.Vector3(-0.5 * sx + dpml_x + 0.2 * Lw)
        sources = [
            mp.EigenModeSource(
                src=mp.GaussianSource(fcen, fwidth=0.2 * fcen),
                center=src_pt,
                size=mp.Vector3(y=sy - 2 * dpml_y),
                eig_match_freq=True,
                eig_parity=mp.ODD_Z + mp.EVEN_Y,
            )
        ]
        vertices = [
            mp.Vector3(-0.5 * sx - 1, 0.5 * w1),
            mp.Vector3(0.5 * sx + 1, 0.5 * w1),
            mp.Vector3(0.5 * sx + 1, -0.5 * w1),
            mp.Vector3(-0.5 * sx - 1, -0.5 * w1),
        ]

        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=boundary_layers,
            geometry=[mp.Prism(vertices, height=mp.inf, material=Si)],
            sources=sources,
            symmetries=symmetries,
        )
        mon_pt = mp.Vector3(-0.5 * sx + dpml_x + 0.7 * Lw)
        flux = sim.add_flux(
            fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy - 2 * dpml_y))
        )
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))

        res = sim.get_eigenmode_coefficients(flux, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y)
        incident_coeffs = res.alpha
        incident_flux = mp.get_fluxes(flux)
        incident_flux_data = sim.get_flux_data(flux)
        sim.reset_meep()

        # Linear taper
        vertices = [
            mp.Vector3(-0.5 * sx - 1, 0.5 * w1),
            mp.Vector3(-0.5 * Lt, 0.5 * w1),
            mp.Vector3(0.5 * Lt, 0.5 * w2),
            mp.Vector3(0.5 * sx + 1, 0.5 * w2),
            mp.Vector3(0.5 * sx + 1, -0.5 * w2),
            mp.Vector3(0.5 * Lt, -0.5 * w2),
            mp.Vector3(-0.5 * Lt, -0.5 * w1),
            mp.Vector3(-0.5 * sx - 1, -0.5 * w1),
        ]
        sim = mp.Simulation(
            resolution=resolution,
            cell_size=cell_size,
            boundary_layers=boundary_layers,
            geometry=[mp.Prism(vertices, height=mp.inf, material=Si)],
            sources=sources,
            symmetries=symmetries,
        )
        flux = sim.add_flux(
            fcen, 0, 1, mp.FluxRegion(center=mon_pt, size=mp.Vector3(y=sy - 2 * dpml_y))
        )
        sim.load_minus_flux_data(flux, incident_flux_data)
        sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt, 1e-9))
        res2 = sim.get_eigenmode_coefficients(flux, [1], eig_parity=mp.ODD_Z + mp.EVEN_Y)
        taper_coeffs = res2.alpha
        taper_flux = mp.get_fluxes(flux)

        R_val = -taper_flux[0] / incident_flux[0]
        R_flux_list.append(R_val)
    return np.array(R_flux_list)

# Example dataset generation
X_data, Y_data = [], []
for w1 in [1.0, 1.2, 1.5]:
    for w2 in [2.0, 2.5]:
        for Lw in [8.0, 10.0]:
            R = run_meep_simulation(w1, w2, Lw)
            X_data.append(R)            # Input: reflection R (vector of 4 values)
            Y_data.append([w1, w2, Lw]) # Target parameters

X_data = np.array(X_data, dtype=np.float32)
Y_data = np.array(Y_data, dtype=np.float32)
np.save("R_inputs.npy", X_data)
np.save("params_targets.npy", Y_data)




import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load data
X_data = np.load("R_inputs.npy")
Y_data = np.load("params_targets.npy")

X_tensor = torch.from_numpy(X_data)
Y_tensor = torch.from_numpy(Y_data)

# Define model
model = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 3)  # Predict w1, w2, Lw
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# Test with one sample
with torch.no_grad():
    sample_R = X_tensor[0].unsqueeze(0)
    predicted_params = model(sample_R)
    print("Predicted [w1, w2, Lw]:", predicted_params.numpy())
    print("True [w1, w2, Lw]:", Y_tensor[0].numpy())
