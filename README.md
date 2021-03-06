# Aesthetic Analysis of Rule Breaks in High Quality Photographs
Veteran photographers will often purposefully break common photographic rules in their photographs. They often do so based on feel, having an intuitive sense of what particular aspects of their photograph allow them to break certain rules. In this project, we use a rigorous machine learning and image processing based approach on a large dataset of high quality photographs, in order to illuminate actual patterns of conditions under which certain rules can be broken. 

## Dataset 

We used the AVA dataset, originally used for aesthetics classification (classifying high-quality vs. low-quality images). The dataset consists of a descriptor file of 255,530 photographs from photography contest website DBChallenge.com. The descriptor file contains the image ID of each photograph and the distribution of ratings from a scale of 1-10. The actual photographs themselves are not part of the dataset, so we needed to scrape them directly from DBChallenge.com. We primarily examined the top 5% highest rated images in the dataset and considered them high-quality photographs. As a contrast, we also looked at the bottom 5% rated images in the dataset and considered them low-quality photographs.   

**Note about this repository.** Due to space considerations on GitHub, we only included three sample images from each of the high-quality and low-quality set.  

## Approach 

We first devised a set of high-level features that described certain attributes of a photograph. Examples of features include: percentage of area taken by the subject, lighting contrast, aspect ratio, and blur. We then ran a classification model of high-quality vs. low-quality photographs, in order to establish which features are strongly associated with high-quality images. High-quality photographs that broke the rules were more likely to be misclassified. This principle formed the backbone of identifying rule breaks high-quality photographs. 

We also used the features directly to identify particular rule breaks computationally. For example, we used the rule of thirds feature to measure if the subject is being placed at the center of a photograph (you are generally taught to avoid this in photography). In addition, we computationally detected and described several binary properties (e.g. whether an animal is in the photograph, whether the photograph is a silhouette, etc.) to subdivide the images as to more accurately describe them. After isolating photographs that 
break particular rules, we use our features and classification in combination with consultation with a photography professional to ascertain what patterns exist among photographs breaking a particular rule. 

## Description of files

1. *createfeatures.py*. Primary file for creating all of the features. All but three of the features are included in the file. The other three, described next, were quite involved and separated into their own file. 
2. *nncolorfeature.py*. A feature that determines the percentage of nearest neighbors color-wise of a photograph that belong to the high-quality set. 
3. *ruleofthirdsfeature.py*. A feature that determines to what degree a photograph follows the "rule of thirds", the most well known rule in photography. 
4. *symmetryfeature.py*. A feature that measures how symmetrical a photograph is.  
5. *colorhist.py*. Functions for creating and working with color histograms, off of which some features are based. 
6. *convnet.py*. Builds a convolutional neural network (CNN) using Keras. 
7. *cnnmain.py*. Prepares the data and uses the CNN to identify binary properties.  
8. *util.py*. Various utility functions for the Python scripts.
9. *setup.R*. Sets up the features and binary properties for usage in R.
10. *classify.R*. Sets up the classification models and analysis.
11. *featureanalysis.R*. Analyzes the features using standard data analysis techniques.
12. *util2.R*. Utility functions for the R scripts. 
13. *high.txt and low.txt*. The features data for the high-quality and low-quality sets, respectively. 

## Description of folders

1. *Features Data.* Contains the outputs of createfeatures.py - A numbered text file for each photograph containing the features vectors. The nbhd subfolders contain text files for each photograph listing its K nearest neighbors as well as whether it is a high quality photograph (1 = high quality, 0 = low quality).

2. *Histograms.* Contains the color histogram for each photograph (as .txt). 

3. *Outlines.* Contains subject detection outlines (as .jpg) for each photograph. 

4. *Photographs.* Contains the actual photographs themselves (as .jpg). 

5. *Property Text Lists.* Contains list of images for each binary property.
