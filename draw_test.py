from draw import Draw
import numpy as np
import unittest
import torch
from tqdm import tqdm

class DrawTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng()
        self.img_sizes = [100, np.random.randint(101, 150), np.random.randint(150, 200), np.random.randint(250, 300), np.random.randint(300, 400), np.random.randint(400, 500), np.random.randint(500, 600), np.random.randint(600, 800), np.random.randint(800, 999), 1000]
        self.draw_classes = [Draw(img_size, self.rng) for img_size in self.img_sizes]
        self.default_num_try = 100
        self.num_try = {
            "line": self.default_num_try,
            "rectangle": self.default_num_try,
            "ellipse": self.default_num_try,
            "stickman": self.default_num_try,
            "dashed_arrow": self.default_num_try
        }
        print("-"*50)
        print(self._testMethodName, ", img_sizes: ", self.img_sizes)
        print("-"*50)

    def test_line(self):
        """
        Tests the line method.
        """
        try:
            for img_size, draw in zip(self.img_sizes, self.draw_classes):
                for _ in tqdm(range(self.num_try["line"]), desc="line, img_size: " + str(img_size)):
                    line = draw.line()
                    self.assertEqual(line.dtype, torch.uint8, "dtype: " + str(line.dtype) + ", img_size: " + str(img_size))
                    self.assertEqual(line.shape, (1, img_size, img_size), "shape: " + str(line.shape) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.max(line) > 0, "max value: " + str(torch.max(line)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.max(line) == 1, "max value: " + str(torch.max(line)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.min(line) == 0, "min value: " + str(torch.min(line)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.sum((line > 0) & (line < 1)) == 0, "found value that is not 0 or 1, img_size: " + str(img_size))
        except Exception as e:
            self.fail("test failed with exception: " + str(e))
   

    def test_rectangle(self):
        """
        Tests the rectangle_outline method.
        """
        try:
            for img_size, draw in zip(self.img_sizes, self.draw_classes):
                for _ in tqdm(range(self.num_try["rectangle"]), desc="rectangle, img_size: " + str(img_size)):
                    rectangle = draw.rectangle_outline()
                    self.assertEqual(rectangle.dtype, torch.uint8, "dtype: " + str(rectangle.dtype) + ", img_size: " + str(img_size))
                    self.assertEqual(rectangle.shape, (1, img_size, img_size), "shape: " + str(rectangle.shape) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.max(rectangle) > 0, "max value: " + str(torch.max(rectangle)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.max(rectangle) == 1, "max value: " + str(torch.max(rectangle)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.min(rectangle) == 0, "min value: " + str(torch.min(rectangle)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.sum((rectangle > 0) & (rectangle < 1)) == 0, "found value that is not 0 or 1, img_size: " + str(img_size))
        except Exception as e:
            self.fail("test failed with exception: " + str(e))
    

    def test_ellipse(self):
        """
        Tests the ellipse method.
        """
        try:
            for img_size, draw in zip(self.img_sizes, self.draw_classes):
                for _ in tqdm(range(self.num_try["ellipse"]), desc="ellipse, img_size: " + str(img_size)):
                    ellipse = draw.ellipse()
                    self.assertEqual(ellipse.dtype, torch.uint8, "dtype: " + str(ellipse.dtype) + ", img_size: " + str(img_size))
                    self.assertEqual(ellipse.shape, (1, img_size, img_size), "shape: " + str(ellipse.shape) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.max(ellipse) > 0, "max value: " + str(torch.max(ellipse)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.max(ellipse) == 1, "max value: " + str(torch.max(ellipse)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.min(ellipse) == 0, "min value: " + str(torch.min(ellipse)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.sum((ellipse > 0) & (ellipse < 1)) == 0, "found value that is not 0 or 1, img_size: " + str(img_size))
        except Exception as e:
            self.fail("test failed with exception: " + str(e))
    

    def test_stickman(self):
        """
        Tests the stickman method.
        """
        try:
            for img_size, draw in zip(self.img_sizes, self.draw_classes):
                for _ in tqdm(range(self.num_try["stickman"]), desc="stickman, img_size: " + str(img_size)):
                    stickman = draw.stickman()
                    self.assertEqual(stickman.dtype, torch.uint8, "dtype: " + str(stickman.dtype) + ", img_size: " + str(img_size))
                    self.assertEqual(stickman.shape, (1, img_size, img_size), "shape: " + str(stickman.shape) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.max(stickman) > 0, "max value: " + str(torch.max(stickman)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.max(stickman) == 1, "max value: " + str(torch.max(stickman)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.min(stickman) == 0, "min value: " + str(torch.min(stickman)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.sum((stickman > 0) & (stickman < 1)) == 0, "found value that is not 0 or 1, img_size: " + str(img_size))
        except Exception as e:
            self.fail("test failed with exception: " + str(e))

    def test_dashed_arrow(self):
        """
        Tests the dashed_arrow method.
        """
        try:
            for img_size, draw in zip(self.img_sizes, self.draw_classes):
                for _ in tqdm(range(self.num_try["dashed_arrow"]), desc="dashed_arrow, img_size: " + str(img_size)):
                    dashed_arrow = draw.dashed_arrow()
                    self.assertEqual(dashed_arrow.dtype, torch.uint8, "dtype: " + str(dashed_arrow.dtype) + ", img_size: " + str(img_size))
                    self.assertEqual(dashed_arrow.shape, (1, img_size, img_size), "shape: " + str(dashed_arrow.shape) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.max(dashed_arrow) > 0, "max value: " + str(torch.max(dashed_arrow)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.max(dashed_arrow) == 1, "max value: " + str(torch.max(dashed_arrow)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.min(dashed_arrow) == 0, "min value: " + str(torch.min(dashed_arrow)) + ", img_size: " + str(img_size))
                    self.assertTrue(torch.sum((dashed_arrow > 0) & (dashed_arrow < 1)) == 0, "found value that is not 0 or 1, img_size: " + str(img_size))
        except Exception as e:
            self.fail("test failed with exception: " + str(e))


if __name__=='__main__':
    unittest.main()