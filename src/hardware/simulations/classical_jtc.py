import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ──────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────
def get_image(filename):
    """Return (vector, label) from a single-row CSV file."""
    with open(filename, "r") as f:
        data = np.asarray(list(csv.reader(f)), dtype=np.int32)[0]
    return data[1:], data[0]  # pixels, label


def join_images(img1_vec, img2_vec, shape=(28, 28)):
    """Horizontal concatenation of two flat images (H×W → H×2W)."""
    img1 = img1_vec.reshape(shape)
    img2 = img2_vec.reshape(shape)
    return np.hstack((img1, img2))


# ──────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────
def plot_image(image, shape, cmap=cm.gray, title=None):
    plt.imshow(np.asarray(image).reshape(shape), cmap=cmap)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side-effect import)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_3d(image, shape, *, elev=35, azim=-45):
    """
    Pretty 3-D surface plot for the correlation plane.

    Parameters
    ----------
    image : array-like, flat
        Data (length = shape[0] * shape[1]) to be reshaped.
    shape : (int, int)
        (height, width) of the correlation plane.
    elev, azim : int or float
        Elevation and azimuth angles for the initial viewpoint.
    """
    Z = np.asarray(image, dtype=float).reshape(shape)

    y = np.arange(shape[0])
    x = np.arange(shape[1])
    Y, X = np.meshgrid(y, x, indexing="ij")

    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")

    # — surface —
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cm.gray,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=True,
    )

    # — nicest viewpoint & axes —
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X", labelpad=10)
    ax.set_ylabel("Y", labelpad=10)
    ax.set_zlabel("Correlation", labelpad=10)

    # put the highest peak on top of the colour-bar scale
    mappable = cm.ScalarMappable(cmap=cm.viridis)
    mappable.set_array(Z)
    fig.colorbar(mappable, ax=ax, shrink=0.60, aspect=15, pad=0.10)

    # nicer background grid
    ax.xaxis.pane.set_edgecolor("w")
    ax.yaxis.pane.set_edgecolor("w")
    ax.zaxis.pane.set_edgecolor("w")

    # reduce the default grid density
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))

    plt.tight_layout()
    plt.show()


img1_vec, lbl1 = get_image("img1.csv")
img2_vec, lbl2 = get_image("img2.csv")
img12_vec, lbl12 = get_image("img1(2).csv")

ref_image = img1_vec
input_image = img2_vec

# Build joint input plane (H×2W)
shape = (28, 28)
joint = join_images(ref_image, input_image, shape)
plot_image(joint, shape=(28, 56), title="Joint input plane")

# Joint power spectrum  |FFT(joint)|²
joint_fft = np.fft.fft2(joint)
power_spectrum = np.abs(joint_fft) ** 2
plot_image(
    power_spectrum,
    shape=(28, 56),
    cmap=cm.gray,
    title="Log power spectrum",
)

# Correlation plane  ℱ⁻¹{|FFT|²}  → then FFT-shift to centre peaks
corr = np.fft.ifft2(power_spectrum)
corr = np.fft.fftshift(np.real(corr))  # signed plane, centre is (0,0)
plot_image(corr, shape=(28, 56), cmap=cm.gray, title="Correlation plane (raw)")


H, W = corr.shape
dc_half = 3  # half-width of square mask
corr_masked = corr.copy()
corr_masked[
    H // 2 - dc_half : H // 2 + dc_half + 1, W // 2 - dc_half : W // 2 + dc_half + 1
] = 0.0
plot_image(
    corr_masked, shape=(28, 56), cmap=cm.viridis, title="Correlation plane (DC masked)"
)

# 6. Locate peak and compute similarity / distance exactly as in classical_jtc
peak_idx = np.unravel_index(np.argmax(corr_masked), corr_masked.shape)
peak_val = corr_masked[peak_idx]
dy = peak_idx[0] - H // 2  # shift from centre
dx = peak_idx[1] - W // 2

norm = np.linalg.norm(img1_vec) * np.linalg.norm(img2_vec)
similarity = peak_val / (2 * norm) if norm else 0.0
distance = 1.0 / similarity if similarity else np.inf

print(f"Label pair: ({lbl1}, {lbl2})")
print(f"Peak value     : {peak_val:.3f}")
print(f"Shift (dy, dx) : ({dy}, {dx})")
print(f"Similarity     : {similarity:.5f}")
print(f"Distance       : {distance:.5f}")

plot_3d(corr_masked, shape=(28, 56))
