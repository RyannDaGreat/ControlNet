import cv2


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        img = rp.as_byte_image(rp.as_rgb_image(img))
        return cv2.Canny(img, low_threshold, high_threshold)
