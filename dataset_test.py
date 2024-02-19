import unittest
from dataset_main import ObjectDetection_DS
import torch

class DatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = ObjectDetection_DS(
            ds_size=100,
            img_size=256,
            shapes_per_image=(1, 3),
            class_probs=(1, 1, 1, 1, 1),
            rand_seed=12345,
            target_masks=True,
        )
    
    def test_types(self):
        """
        Tests the types of the dataset.
        """
        self.assertIsInstance(self.dataset.imgs, torch.Tensor, "imgs is not a torch.Tensor")
        self.assertIsInstance(self.dataset.targets, list, "targets is not a list")
        self.assertIsInstance(self.dataset.imgs[0], torch.Tensor, "imgs[0] is not a torch.Tensor")
        self.assertIsInstance(self.dataset.targets[0], dict, "targets[0] is not a dict")
        self.assertIsInstance(self.dataset.targets[0]["boxes"], torch.Tensor, "targets[0]['boxes'] is not a torch.Tensor")
        self.assertIsInstance(self.dataset.targets[0]["labels"], torch.Tensor, "targets[0]['labels'] is not a torch.Tensor")
        self.assertIsInstance(self.dataset.targets[0]["masks"], torch.Tensor, "targets[0]['masks'] is not a torch.Tensor")
    
    def test_lengths(self):
        """
        Tests the lengths of the dataset.
        """
        self.assertEqual(len(self.dataset.targets[0]["boxes"]), len(self.dataset.targets[0]["labels"]), "length of boxes and labels are not equal")
        self.assertEqual(len(self.dataset.targets[0]["boxes"]), len(self.dataset.targets[0]["masks"]), "length of boxes and masks are not equal")
        self.assertEqual(len(self.dataset.imgs), len(self.dataset.targets), "length of imgs and targets are not equal")
        
    def test_dtypes(self):
        """
        Tests the dtypes of the dataset.
        """
        self.assertEqual(self.dataset.imgs.dtype, torch.uint8, f"imgs dtype is not torch.uint8, dtype: {self.dataset.imgs.dtype}")
        self.assertEqual(self.dataset.targets[0]["boxes"].dtype, torch.float32, f"targets[0]['boxes'] dtype is not torch.float32, dtype: {self.dataset.targets[0]['boxes'].dtype}")
        self.assertEqual(self.dataset.targets[0]["labels"].dtype, torch.int64, f"targets[0]['labels'] dtype is not torch.int64, dtype: {self.dataset.targets[0]['labels'].dtype}")
        self.assertEqual(self.dataset.targets[0]["masks"].dtype, torch.bool, f"targets[0]['masks'] dtype is not torch.bool, dtype: {self.dataset.targets[0]['masks'].dtype}")


if __name__=='__main__':
    unittest.main()