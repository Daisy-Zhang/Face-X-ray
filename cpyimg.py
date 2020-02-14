import os
import sys

source = sys.argv[1]
target = sys.argv[2]

for dirname in os.listdir(source):
    for filename in os.listdir(source + '/' + dirname):
        if filename.startswith('REAL') and filename.endswith('.png'):
            cmd = 'cp ' + source + '/' + dirname + '/' + filename + ' ' + target
            print(cmd)
            os.system(cmd)