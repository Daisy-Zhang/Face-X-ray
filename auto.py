import os
import sys

background_path = sys.argv[1]
img_db_path = sys.argv[2]
output_fake_path = sys.argv[3]
x_ray_fake_path = sys.argv[4]
output_real_path = sys.argv[5]
x_ray_real_path = sys.argv[6]

for filename in os.listdir(background_path):
	if filename.endswith('.png'):
		#print(filename)
		cmd = 'python3 X_ray.py ' + background_path + filename + ' ' + img_db_path + ' ' + output_fake_path + 'output_' + filename + ' ' + x_ray_fake_path + 'x_ray_' + filename
		print(cmd)
		os.system(cmd)
		# cmd = 'python3 X_ray_real.py ' + background_path + filename + ' ' + output_real_path + 'output_' + filename + ' ' + x_ray_real_path + 'x_ray_' + filename
		# print(cmd)
		# os.system(cmd)