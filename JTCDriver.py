import numpy as np
from data import get_test_data  # your helper
from RBF.JTCorrelator import (
    jtc_classical,
    jtc_binary,
)

# 1) grab MNIST samples
vectors, labels = get_test_data(flatten=True, return_tuple=True)

ref_vec = vectors[labels == 3][0]  # reference digit “2”
comp_vec = vectors[labels == 2][0]  # comparison digit “1”

# 2) reshape and normalise
ref_img = ref_vec.reshape(28, 28).astype(np.float32) / 255.0
comp_img = comp_vec.reshape(28, 28).astype(np.float32) / 255.0

# 3) run the correlator
peak_classical = jtc_classical(ref_img, comp_img)
peak_binary = jtc_binary(ref_img, comp_img)  # Fourier-plane binarisation
peak_bin_inputs = jtc_binary(
    ref_img, comp_img, binarize_inputs=True
)  # also binarise the inputs

print("Classical peak :", peak_classical)
print("Binary peak    :", peak_binary)
print("Binary-inputs  :", peak_bin_inputs)
