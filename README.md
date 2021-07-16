# Face_X-ray
This is a un official implementation of *Lingzhi Li, Jianmin Bao, Ting Zhang, Hao Yang, Dong Chen, Fang Wen, Baining Guo: Face X-Ray for More General Face Forgery Detection. CVPR 2020: 5000-5009*.

## Install Dependancies

Users can use following command to install Dependancies:

```python
pip install -r requirements.txt
```

## How to Use

Users can use the following command to get usage:

```python
>> python main.py -h
usage: main.py [-h] --background_img_path BACKGROUND_IMG_PATH --img_db_path IMG_DB_PATH
               --output_img_path OUTPUT_IMG_PATH --output_xray_path OUTPUT_XRAY_PATH

optional arguments:
  -h, --help            show this help message and exit
  --background_img_path BACKGROUND_IMG_PATH, -b BACKGROUND_IMG_PATH
                        background images floder path
  --img_db_path IMG_DB_PATH, -i IMG_DB_PATH
                        image database for key-points-nearest-search
  --output_img_path OUTPUT_IMG_PATH, -oi OUTPUT_IMG_PATH
                        output blended image folder path to store output
  --output_xray_path OUTPUT_XRAY_PATH, -ox OUTPUT_XRAY_PATH
                        output x-ray folder path to store output
```

- **--background_img_path**: the background images database searched by our algorithm.
- **--img_db_path:** target face images.
- **--output_img_path**: output path of the generated face image.
- **--output_xray_path**: output path of the generated face x-ray.

An example can be as follows:

```python
python main.py --background_img_path=./background_img --img_db_path=./img_db --output_img_path=./output --output_xray_path=./output_xray
```

## Structure

### /background_img

Background images to search the nearest target face to swap.

### /img_db

Images to be swapped.

### convex_hull.py

Caculate the convex hull of people face.

### Gaussian_blur.py

Add Gaussian blur to face X-ray image as the paper described.

### get_face_alignment.py

Get the face landmarks of people images.

### nearest_search.py

Use nearest search of face landmarks to get the target image face.

### generator.py

The main class to generate face X-ray.

### utils.py

Some function used by above.
