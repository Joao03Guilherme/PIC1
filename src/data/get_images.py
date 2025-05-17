import os
import matplotlib.pyplot as plt
import numpy as np
from data import get_test_data
def save_digit_images(output_dir):
    """
    Loads the MNIST training data, takes one image from each class (0-9),
    and saves it as digits_x.png in the specified output directory.
    """
    X_train, y_train = get_test_data()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    for digit in range(10):
        # Find the first occurrence of the current digit in the labels
        try:
            idx = np.where(y_train == digit)[0][0]
        except IndexError:
            print(f"Could not find an image for digit {digit} in the training set.")
            continue

        image = X_train[idx].reshape(28, 28) # Reshape to 28x28

        plt.imshow(image, cmap='gray')
        plt.axis('off')
        output_path = os.path.join(output_dir, f"digits_{digit}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0) # Added bbox_inches and pad_inches
        print(f"Saved: {output_path}")
        plt.close() # Close the plot to free memory

if __name__ == '__main__':
    # Determine the absolute path for the output directory
    # Assuming this script is in src/data/ and figures are in latex/Quantum inspired classification/figures/digits/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir)) # Goes up two levels from src/data to PIC1
    output_directory = os.path.join(project_root, 'latex', 'Quantum inspired classification', 'figures', 'digits')
    
    save_digit_images(output_directory)
