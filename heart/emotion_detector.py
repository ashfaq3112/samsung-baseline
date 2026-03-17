import cv2
import numpy as np
from fer import FER
from PIL import Image
import warnings

# We ignore some specific warnings from the FER lpip install fer tensorflowibrary to keep the output clean
warnings.filterwarnings('ignore')

class EmotionDetector:
    """HEART: Detects face and scene emotions"""
    
    def __init__(self):
        print("🔧 Initializing HEART (Emotion Detector)...")
        # Load the Face Emotion Recognition model (using MTCNN for better face finding)
        self.face_detector = FER(mtcnn=True)
        print("✓ FER detector loaded with MTCNN")
    
    def detect_face_emotion(self, image_path):
        """Detect dominant emotion from faces in the image"""
        try:
            # OpenCV loads images in BGR format, so we convert to RGB
            img = cv2.imread(str(image_path))
            if img is None:
                return self._default_face_result()
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect emotions
            result = self.face_detector.detect_emotions(img_rgb)
            
            if result and len(result) > 0:
                # If faces are found, get the top emotion
                emotions = result[0]['emotions']
                dominant = max(emotions.items(), key=lambda x: x[1])
                
                return {
                    'has_face': True,
                    'emotion': dominant[0],
                    'confidence': round(dominant[1], 3),
                    'all_emotions': emotions
                }
            
            return self._default_face_result()
            
        except Exception as e:
            # If the image is corrupt or can't be read
            return self._default_face_result()
    
    def detect_scene_emotion(self, image_path):
        """Detect scene emotion based on brightness and color (heuristics)"""
        try:
            img = np.array(Image.open(image_path).convert('RGB'))
            
            # Calculate brightness (light vs dark)
            brightness = np.mean(img)
            # Calculate saturation (how colorful it is)
            saturation = np.std(img)
            
            # Simple Logic:
            # - Bright & Colorful = Positive
            # - Dark or Dull = Negative
            # - In between = Neutral
            if brightness > 150 and saturation > 50:
                return 'positive'
            elif brightness < 100 or saturation < 30:
                return 'negative'
            else:
                return 'neutral'
                
        except Exception as e:
            return 'neutral'
    
    def _default_face_result(self):
        """Helper to return a 'blank' result when no face is found"""
        return {
            'has_face': False,
            'emotion': 'neutral',
            'confidence': 0.0,
            'all_emotions': {}
        }