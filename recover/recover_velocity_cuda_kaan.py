import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d


def rotation123(phi, theta, psi):
    cph, sph = torch.cos(phi), torch.sin(phi)
    cth, sth = torch.cos(theta), torch.sin(theta)
    cps, sps = torch.cos(psi), torch.sin(psi)

    R_x = torch.tensor([[1, 0, 0],
                        [0, cph, -sph],
                        [0, sph, cph]], device=phi.device)
    R_y = torch.tensor([[cth, 0, sth],
                        [0, 1, 0],
                        [-sth, 0, cth]], device=phi.device)
    R_z = torch.tensor([[cps, -sps, 0],
                        [sps, cps, 0],
                        [0, 0, 1]], device=phi.device)

    return R_z @ R_y @ R_x


def load_frames(flight, file_type, experiment):
    file_path = f"E:\\final_test_database\\{experiment}\\test\\{flight}\\{file_type}.npy"
    frames = np.load(file_path)
    return [torch.tensor(frame, device="cuda") for frame in frames]


def calculate_velocity(flow, omega, z, phi, theta, psi, f):
    channels, height, width = flow.shape
    R = rotation123(phi, theta, psi)
    alpha, beta, gamma = R[2, :]

    # Create coordinate grid
    x_coords = torch.arange(width, device=flow.device) - width // 2
    y_coords = torch.arange(height, device=flow.device) - height // 2
    x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='xy')

    h = - (alpha * x_grid / f + beta * y_grid / f + gamma) / z
    u_0, u_1 = flow[0] / f, flow[1] / f
    x_0_bar, x_1_bar = x_grid / f, y_grid / f

    # Derotate flow and compute A and b in a vectorized way
    A_u = (-h * torch.stack([-torch.ones_like(h), torch.zeros_like(h), x_0_bar], dim=2))
    A_v = (-h * torch.stack([torch.zeros_like(h), -torch.ones_like(h), x_1_bar], dim=2))
    A = torch.cat([A_u.view(-1, 3), A_v.view(-1, 3)], dim=0)

    b_u = u_0 - (x_0_bar * x_1_bar * omega[0] - (1 + x_0_bar ** 2) * omega[1] + x_1_bar * omega[2])
    b_v = u_1 - ((1 + x_1_bar ** 2) * omega[0] - x_1_bar * x_0_bar * omega[1] - x_0_bar * omega[2])
    b = torch.cat([b_u.view(-1, 1), b_v.view(-1, 1)], dim=0)

    # Solve for V_c using least squares
    V_c, _ = torch.lstsq(b, A)
    V_c = V_c[:3].squeeze()

    # Transform to global frame
    V = torch.linalg.inv(R.T) @ V_c
    return V


def flow_to_velocity(flow_list, side_length, flight):
    trajectory = np.load(f"E:\\ALED_v30\\test\\{flight}\\traj.npy", allow_pickle=True)
    t = trajectory.item()['t']
    x = trajectory.item()['x']
    interpolate = interp1d(t, x, kind='cubic', axis=0)

    frequency = 10  # Hz
    n = int(t[-1] * frequency)
    timestamps = torch.tensor(np.arange(0, n) * 1 / frequency, device="cuda")

    fov = torch.deg2rad(torch.tensor(128 / 200 * 45, device="cuda"))
    f = side_length / (2 * torch.tan(fov / 2))

    score_list = []
    z_list, omega_list = [], []

    for idx, timestamp in enumerate(timestamps[3:-2]):
        state = interpolate(float(timestamp.cpu()))
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r, m = state
        z_list.append(-z)
        omega_scalar = np.sqrt(p ** 2 + q ** 2 + r ** 2)
        omega_list.append(omega_scalar)

        omega = torch.tensor([p, q, r], device="cuda")
        flow = flow_list[idx]

        v = calculate_velocity(flow, omega, z, phi, theta, psi, f)
        score_list.append(torch.sqrt(((v[0] - vx) ** 2 + (v[1] - vy) ** 2 + (v[2] - vz) ** 2)) / (-z))

    return score_list, omega_list


if __name__ == "__main__":
    a = 128
    flight = "N12"
    flow_list = load_frames(flight, "prediction", "4")
    score, omega_list = flow_to_velocity(flow_list, a, flight)
