
import os
import cv2
import numpy as np

def generate_mock_images(output_dir, num_images=10, size=(256, 256), prefix='img'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in range(num_images):
        img_path = os.path.join(output_dir, f"{prefix}_{i}.jpg")
        
        # Random noise + simple shapes
        img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
        
        # Add a "lesion" (red circle for colposcopy, purple cluster for histo)
        if 'colpo' in prefix:
             cv2.circle(img, (np.random.randint(0, size[0]), np.random.randint(0, size[1])), 
                        np.random.randint(10, 50), (0, 0, 255), -1)
        else:
             for _ in range(5):
                 cv2.circle(img, (np.random.randint(0, size[0]), np.random.randint(0, size[1])), 
                            np.random.randint(5, 20), (255, 0, 255), -1)

        cv2.imwrite(img_path, img)
    print(f"Generated {num_images} images in {output_dir}")

if __name__ == "__main__":
    generate_mock_images("data/colposcopy", num_images=20, prefix='colpo')
    generate_mock_images("data/histopathology", num_images=20, prefix='histo')
