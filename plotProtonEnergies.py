"""Range steering by the initial energy: E[D] maps and laterally integrated depth dose (IDD)
for several initial energies of the ProtonUQ test case (output of mainProtonUQ.jl).

usage: python plotProtonEnergies.py results_ProtonUQ/data_E70_....npz results_ProtonUQ/data_E80_....npz ...
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = sys.argv[1:]
data = [np.load(f) for f in files]
outdir = os.path.dirname(files[0])

fig, axs = plt.subplots(1, len(data), figsize=(4.2 * len(data), 3.9), dpi=150)
axs = np.atleast_1d(axs)
for ax, d in zip(axs, data):
    n = int(d["NCells"][0]); x = d["xMid"]; inn = slice(1, n - 1)
    ED = d["ED"].reshape(n, n)            # [i (x), j (y)]
    im = ax.pcolormesh(x[inn], x[inn], 100 * ED[inn, inn].T / ED.max(), shading="auto", cmap="turbo", vmin=0, vmax=100)
    mat = d["matIndex"].reshape(n, n)
    if mat.max() > mat.min():  # material boundaries
        ax.contour(x[inn], x[inn], mat[inn, inn].T, levels=np.arange(mat.min() + 0.5, mat.max()), colors="w",
                   linewidths=1.2, linestyles="--")
    ax.set_title(r"$\mathbb{E}[D]$, $E_0$ = %d MeV" % round(d["E0"][0]))
    ax.set_xlabel("depth x [cm]"); ax.set_aspect("equal")
axs[0].set_ylabel("y [cm]")
fig.colorbar(im, ax=axs.tolist(), label="% of max", shrink=0.9)
fig.savefig(os.path.join(outdir, "protonUQ_energies_maps.png"), bbox_inches="tight")
fig.savefig(os.path.join(outdir, "protonUQ_energies_maps.pdf"), bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
for k, d in enumerate(data):
    n = int(d["NCells"][0]); x = d["xMid"][1:n - 1]
    m = d["EIDD"][1:n - 1]; sd = d["StdIDD"][1:n - 1]; nrm = m.max()
    ax.fill_between(x, 100 * (m - sd) / nrm, 100 * (m + sd) / nrm, color="C%d" % k, alpha=0.25)
    ax.plot(x, 100 * m / nrm, "C%d-" % k, lw=2, label=r"$E_0$ = %d MeV" % round(d["E0"][0]))
# shade depths where the beam band |y - y0| < 2w crosses a non-background material
d = data[0]; n = int(d["NCells"][0]); x = d["xMid"]; mat = d["matIndex"].reshape(n, n)
bg = np.bincount(mat.ravel()).argmax()
band = np.abs(x - d["y0"][0]) < 2 * d["w"][0]
hit = np.any(mat[:, band] != bg, axis=1); hit[[0, -1]] = False
edges = np.flatnonzero(np.diff(hit.astype(int)))
for a, b in zip(edges[::2], edges[1::2]):
    ax.axvspan(x[a + 1] - 0.5 * (x[1] - x[0]), x[b] + 0.5 * (x[1] - x[0]), color="0.88", zorder=0,
               label="other material in beam path" if a == edges[0] else None)
ax.set_xlabel("depth x [cm]"); ax.set_ylabel("laterally integrated dose [% of peak]")
ax.set_title(r"integrated depth dose, $\mathbb{E}\pm$Std", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(outdir, "protonUQ_energies_idd.png"))
fig.savefig(os.path.join(outdir, "protonUQ_energies_idd.pdf"))
print("written to", outdir)
