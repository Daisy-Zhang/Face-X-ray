import argparse
from generator import X_ray_generator

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--background_img_path', '-b', required=True, type = str, default = './background_img/', help = 'background images floder path')
    parser.add_argument('--img_db_path', '-i', required=True, type = str, default = './img_db/', help = 'image database for key-points-nearest-search')
    parser.add_argument('--output_img_path', '-oi', required=True, type = str, default = './output_img/', help = 'output blended image folder path to store output')
    parser.add_argument('--output_xray_path', '-ox', required=True, type = str, default = './output_x-ray/', help = 'output x-ray folder path to store output')

    args = parser.parse_args()

    my_generator = X_ray_generator(args.background_img_path, args.img_db_path, args.output_img_path, args.output_xray_path)
    my_generator.run()