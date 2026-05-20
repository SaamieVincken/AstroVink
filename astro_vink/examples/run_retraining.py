from astro_vink.retrain import retrain_astrovink_q1

base_weights = "weights/astrovink_base_checkpoint.pth"
output_weights = "weights/astrovink_output_checkpoint.pth"
data_dir = ""
retrain_astrovink_q1(base_weights=base_weights, output_path=output_weights, data_dir=data_dir)
