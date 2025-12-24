import dlib
import cv2
import numpy as np
import imageio
from typing import List, Tuple, Any

class DLibFaceDetector:
    """
    A wrapper class for DLib's face detection and landmark prediction.
    
    This class handles the initialization of the DLib models and provides
    a simplified interface for extracting 68 facial landmarks plus
    image corner points for background stabilization.
    """
    
    def __init__(self, predictor_path: str) -> None:
        """
        Initialize the DLib detector and shape predictor.

        Args:
            predictor_path (str): File path to the .dat shape predictor model.
        """
        # Load models once
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def get_landmarks(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detects a face in the image and returns its landmarks as a list of points.

        Args:
            img (np.ndarray): The input image (BGR format).

        Returns:
            List[Tuple[int, int]]: A list of (x, y) coordinates for facial features
                                   and image corners.

        Raises:
            ValueError: If no faces are detected in the provided image.
        """
        # Pure logic: Image In -> Points Out
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if not faces:
            raise ValueError("No faces detected in image.")

        # Get landmarks
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        points: List[Tuple[int, int]] = []
        for n in range(68):
            points.append((landmarks.part(n).x, landmarks.part(n).y))

        # Append corners for background warping
        h, w = img.shape[:2]
        points.extend([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])
        
        return points


class TriangleMorpher:
    """
    A stateless utility class for geometry and math operations.
    
    This class handles Delaunay triangulation logic and the affine transformations
    required to morph specific triangular regions between two images.
    """
    
    @staticmethod
    def get_delaunay_indices(img_shape: Tuple[int, ...], points: List[Tuple[int, int]]) -> List[List[int]]:
        """
        Computes the Delaunay triangulation for a set of points.

        Args:
            img_shape (Tuple[int, ...]): Shape of the image (height, width, ...).
            points (List[Tuple[int, int]]): List of point coordinates.

        Returns:
            List[List[int]]: A list of lists, where each inner list contains 
                             the indices of the 3 points forming a triangle.
        """
        # Standard Delaunay logic
        rect = (0, 0, img_shape[1], img_shape[0])
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(points)
        triangle_list = subdiv.getTriangleList()

        point_dict = {(p[0], p[1]): i for i, p in enumerate(points)}
        indices = []
        
        for t in triangle_list:
            pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            if all(p in point_dict for p in pts):
                indices.append([point_dict[p] for p in pts])
                
        return indices

    @staticmethod
    def compute_morphed_triangle(
        img1: np.ndarray, 
        img2: np.ndarray, 
        t1: np.ndarray, 
        t2: np.ndarray, 
        t_morphed: np.ndarray, 
        alpha: float
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """
        Warps and blends a single triangular region from two source images.

        Args:
            img1 (np.ndarray): The starting image.
            img2 (np.ndarray): The ending image.
            t1 (np.ndarray): Coordinates of the triangle in img1.
            t2 (np.ndarray): Coordinates of the triangle in img2.
            t_morphed (np.ndarray): Coordinates of the triangle in the current frame.
            alpha (float): The blending factor (0.0 to 1.0).

        Returns:
            Tuple containing:
                - patch (np.ndarray): The blended rectangular image patch.
                - mask (np.ndarray): The triangular mask for the patch.
                - rect (Tuple[int, int, int, int]): The bounding box (x, y, w, h).
        """
        # Calculate Bounding Box for the DESTINATION triangle
        r = cv2.boundingRect(np.float32(t_morphed))
        x, y, w, h = r

        # Offset points relative to the bounding box (Top-Left is 0,0)
        # This is critical: We want to map pixels to this small box, not the global image.
        t1_rect = t1 - [x, y]
        t2_rect = t2 - [x, y]
        t_morphed_rect = t_morphed - [x, y]

        # Create Mask
        mask = np.zeros((h, w, 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_morphed_rect), (1, 1, 1), 16, 0)

        # Affine Transforms
        M1 = cv2.getAffineTransform(np.float32(t1), np.float32(t_morphed_rect))
        M2 = cv2.getAffineTransform(np.float32(t2), np.float32(t_morphed_rect))

        # Warp directly into the small patch size (w, h)
        # No more slicing [y:y+h]! We ask for exactly (w, h) from the start.
        warp1 = cv2.warpAffine(img1, M1, (w, h))
        warp2 = cv2.warpAffine(img2, M2, (w, h))

        # Blend
        patch = (1.0 - alpha) * warp1 + alpha * warp2

        return patch, mask, (x, y, w, h)


class MorphSequence:
    """
    Manager class that orchestrates the entire face morphing process.
    
    It uses injected dependencies to detect landmarks and math utilities
    to generate a GIF sequence.
    """

    def __init__(self, detector: Any) -> None:
        """
        Initialize the morph sequencer.

        Args:
            detector: A detector instance (e.g., DLibFaceDetector) that has 
                      a get_landmarks method.
        """
        # Dependency Injection: We store the tool we were given
        self.detector = detector

    def generate_gif(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray, 
        filename: str, 
        steps: int = 60
    ) -> None:
        """
        Generates a morph GIF between two images.

        Args:
            img1 (np.ndarray): The source image.
            img2 (np.ndarray): The target image.
            filename (str): Output filename (e.g., 'output.gif').
            steps (int, optional): Number of frames in the animation. Defaults to 60.
        """
        # Ask the Detector for points
        print("Detecting landmarks...")
        points1 = self.detector.get_landmarks(img1)
        points2 = self.detector.get_landmarks(img2)

        # Ask the Morpher Utility for triangle indices
        tri_indices = TriangleMorpher.get_delaunay_indices(img1.shape, points1)
        
        frames = []
        print(f"Generating {steps} frames...")

        # 3. The Main Loop
        for alpha in np.linspace(0, 1, steps):
            # Create a blank canvas for this frame
            current_frame = np.zeros_like(img1)
            
            # Interpolate points (Math for dots)
            morphed_points = []
            for i in range(len(points1)):
                x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
                y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
                morphed_points.append((x, y))

            # Process every triangle
            for t_idx in tri_indices:
                # Gather the 3 coordinates for this specific triangle
                t1 = np.array([points1[i] for i in t_idx])
                t2 = np.array([points2[i] for i in t_idx])
                t_morphed = np.array([morphed_points[i] for i in t_idx])

                patch, mask, rect = TriangleMorpher.compute_morphed_triangle(
                    img1, img2, t1, t2, t_morphed, alpha
                )
                
                # Paste onto canvas
                x, y, w, h = rect
                if patch.shape[:2] == (h, w): # Basic safety check
                    # Standard alpha blending
                    bg = current_frame[y:y+h, x:x+w]
                    # Cut a hole in background -> Add the patch
                    current_frame[y:y+h, x:x+w] = bg * (1 - mask) + patch * mask

            frames.append(np.uint8(current_frame))

        # 4. Save
        imageio.mimsave(filename, frames, duration=0.05)
        print("Done.")


# Example usage
if __name__ == "__main__":
    # Load Data
    image_a = cv2.imread("Donald_Trump.png")
    image_b = cv2.imread("Joe_Biden.png")

    # Setup the Provider
    my_detector = DLibFaceDetector("shape_predictor_68_face_landmarks.dat")

    # Setup the Manager (Injecting the Provider)
    app = MorphSequence(my_detector)

    # Run
    app.generate_gif(image_a, image_b, "output.gif")
