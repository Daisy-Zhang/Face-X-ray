import os
import sys

background_path = sys.argv[1]
output_path = sys.argv[2]
img_db_path = sys.argv[3]

for filename in os.listdir(background_path):
	if filename.endswith('.png'):
		#print(filename)
		cmd = 'python3 X_ray.py ' + background_path + filename + ' ' + img_db_path + ' ' + output_path + filename
		print(cmd)
		os.system(cmd)