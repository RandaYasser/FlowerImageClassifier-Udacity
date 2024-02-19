# AI Programming with Python Project

**Project Overview:**
The project "Image Classifier with Deep Learning" is a part of the Udacity Data Scientist Nanodegree program focusing on deep learning. Its objective is to develop an image classifier capable of identifying various species of flowers. This classifier could potentially be integrated into a mobile application that identifies flowers captured by a phone camera. The task involved training the classifier using a dataset comprising 102 different flower categories.
You can find the dataset at https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

**Approach:**
The project utilized the torchvision library to manage the dataset. Data preprocessing involved splitting the dataset into three subsets: training, validation, and testing. Transformations like random scaling, cropping, and flipping were applied during training to enhance the network's ability to generalize, thereby improving its performance. Additionally, a mapping from category labels to category names was incorporated. Post-training, the model underwent testing, and inference for classification was implemented. PIL images were processed for compatibility with the PyTorch model.

**Outcome:**
The project relied on various software and Python libraries including Torch, PIL, Matplotlib.pyplot, Numpy, Seaborn, and Torchvision. Through the outlined methodologies, the classifier achieved an accuracy of 89.5% on the test dataset. A sanity check was conducted to ensure the absence of apparent bugs despite the satisfactory test accuracy. The probabilities for the top 5 classes were visualized through a bar graph, along with the input image.

**Software and Libraries:**
The project utilized the following software and Python libraries: NumPy, pandas, Sklearn / scikit-learn, Matplotlib (for data visualization), and Seaborn (for data visualization).