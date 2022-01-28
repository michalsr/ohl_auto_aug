import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps







class ShearX(object):
    RANGES = np.linspace(0, 0.3, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.transform(
            img.size, Image.AFFINE, (
                1, self.magnitude * random.choice([-1, 1]), 0, 0, 1, 0
            ), Image.BICUBIC, fillcolor=(128, 128, 128)
        )



class ShearY(object):
    RANGES = np.linspace(0, 0.3, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.transform(
            img.size, Image.AFFINE, (
                1, 0, 0, self.magnitude * random.choice([-1, 1]), 1, 0
            ), Image.BICUBIC, fillcolor=(128, 128, 128)
        )


class TranslateX(object):
    RANGES = np.linspace(0, 150 / 331, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.transform(
            img.size, Image.AFFINE, (
                1, 0, self.magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0
            ), fillcolor=(128, 128, 128)
        )


class TranslateY(object):
    RANGES = np.linspace(0, 150 / 331, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.transform(
            img.size, Image.AFFINE, (
                1, 0, 0, 0, 1, self.magnitude * img.size[1] * random.choice([-1, 1])
            ), fillcolor=(128, 128, 128)
        )


class Rotate(object):
    RANGES = np.linspace(0, 30, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.rotate(self.magnitude * random.choice([-1, 1]))



class Color(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageEnhance.Color(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )



class Posterize(object):
    RANGES = np.round(np.linspace(8, 4, 10), 0).astype(np.int)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageOps.posterize(img, self.magnitude)



class Solarize(object):
    RANGES = np.linspace(256, 0, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageOps.solarize(img, self.magnitude)



class Contrast(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageEnhance.Contrast(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )



class Sharpness(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageEnhance.Sharpness(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )



class Brightness(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageEnhance.Brightness(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )



class AutoContrast(object):
    RANGES = None
    
    def __init__(self):
        pass
    
    def __call__(self, img):
        return ImageOps.autocontrast(img)



class Equalize(object):
    RANGES = None
    
    def __init__(self):
        pass
    
    def __call__(self, img):
        return ImageOps.equalize(img)



class Invert(object):
    RANGES = None
    
    def __init__(self):
        pass
    
    def __call__(self, img):
        return ImageOps.invert(img)