import astro_vink

# Path to weights and image
weights_path = "weights/astrovink_q1.pth"
image_path = "example_cutout.jpg"

# Run inference
result = astro_vink.predict(image_path, weights_path)

print(f"Lens probability: {result['Lens']:.4f}")
print(f"NoLens probability: {result['NoLens']:.4f}")

# (Optional) Example if you have ground-truth labels
# y_true = [0, 1, 0, 1]   # 0 = Lens, 1 = NoLens
# y_pred = [0.92, 0.15, 0.81, 0.20]
# metrics = astro_vink.compute_metrics(y_true, y_pred)
# print(metrics)
