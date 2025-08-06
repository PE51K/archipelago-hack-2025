#!/usr/bin/env python3
"""
Test script to verify the hackathon solution.py fix works correctly.
"""

import numpy as np

print("Testing import of predict function only...")
from solution import predict
print("✓ Successfully imported predict function")

print("\nTesting prediction with dummy image...")
dummy_image = np.zeros((720, 1280, 3), dtype=np.uint8)
result = predict(dummy_image)
print(f"✓ Prediction successful: {len(result)} image(s), {len(result[0])} detections")

print("\nTesting multiple images...")
dummy_images = [dummy_image, dummy_image]
result = predict(dummy_images)
print(f"✓ Multiple images prediction successful: {len(result)} image(s)")

print("\n🎉 All tests passed! The solution should work with the hackathon evaluation system.")
