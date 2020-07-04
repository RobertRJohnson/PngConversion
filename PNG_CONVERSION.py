#!/usr/bin/env python
# coding: utf-8

# In[4]:


from PIL import Image
import glob
import argparse


# In[5]:


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--format", required=True,
	help="file format")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
args = vars(ap.parse_args())


# In[6]:

total_made = 0

for file in glob.glob(args["format"]):
    im = Image.open(file)
    rgb_im = im.convert('RGB')
    rgb_im.save(file.replace("png", "jpg"), quality=95)
    print(str(rgb_im))
    total_made += 1

print("the total number of imgs converted equals %s" %total_made)



# In[ ]:




