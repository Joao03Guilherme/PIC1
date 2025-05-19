from ...hardware.devices.Camera import ThorlabsCamera

with ThorlabsCamera() as cam:
    cam.set_exposure(24.65 * 1e-3)  # 24.65 ms
    frame = cam.snap()  # numpy.ndarray, shape (H, W)
    print(f"Captured frame of shape: {frame.shape}")

    """Plot the captured frame."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plot the captured frame
    plt.figure(figsize=(6, 6))
    sns.heatmap(frame, cmap="viridis")
    plt.title("Captured Frame Heatmap")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.show()
