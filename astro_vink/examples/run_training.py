from astro_vink.train import train_base_model

output_weights = "weights/astrovink_base_checkpoint.pth"
data_dir = ""
train_base_model(data_dir=data_dir, output_path=output_weights)
