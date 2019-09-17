Download the 11K hands datasets from here: https://github.com/mahmoudnafifi/11K-Hands and store it in a directory named Hands. I recommend using Pycharm but you may need to install the modules mentioned below manually after going to Files -> Settings.

In this implementation, I have used the radius of pixels as 3 and the number of neighborhood points to be considered as 8. Smaller radius can miss features that vary over a longer spatial spread, while larger radius amount to heavy computation, which would slow down the execution. Using less neighbors causes a higher approximation, while again, using more neighbors increases the execution time significantly. Because of 8 neighbors, each of our histograms will be a list of length 10. The array of 192 histograms of length 10 returned is converted into a numpy array of size (1920,) for each image. This implementation uses a file system to store the feature vectors, where each line represents a flattened vector of histogram blocks of a single image.

To run it, use the following:
Python 3.6
numpy 1.17.2
opencv-python 3.3.0.10

run the lbp_all.py with given file structure and optional command line argument as the image id (default is 0000000 which runs for entire dataset)
It will store features in Output directory in respective files.

run the compare_and_rank.py with test file id as argument and one optional argument for the number of matches to be displayed (default all matches displayed)
