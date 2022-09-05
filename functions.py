import os
import shutil
import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.segmentation import slic, mark_boundaries
from sklearn import preprocessing, svm, metrics
from sklearn.cluster import MiniBatchKMeans

class Functions:
    def __init__(self):
        # create root folder where all the results will be saved
        if os.path.exists('temp'):
            shutil.rmtree('temp')
        else:
            os.makedirs('temp')

        self.k_means = None
        self.quantized = None
        self.centroids_lab = []
        self.source_superpixels = []
        self.target_superpixels = []
        self.source_surf = []
        self.target_surf = []
        self.gabor_kernels = []
        self.source_gabor = []
        self.target_gabor = []
        self.ab_channels = []
        self.ab_channels_idx = {}
        self.source_x = []
        self.source_y = []
        self.target_x = []
        self.target_y = []
    
    '''
    (i) IMAGE REPRESENTATION IN LAB COLOR FORMAT
    '''
    def convert_colorspace(self):
        # create folder to save results
        if os.path.exists('temp/lab_colorspace'):
            shutil.rmtree('temp/lab_colorspace')
        else:
            os.makedirs('temp/lab_colorspace')
        print('\nConverting source image\'s colorspace to LAB format\n.\n.\n.')

        rgb_image = cv2.imread('images/source.jpg')                     # load source image
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)          # convert from rgb to lab
        cv2.imwrite('temp/lab_colorspace/source_lab.jpg', lab_image)    # save

        print('Done!\n')

    '''
    (ii) QUANTIZATION OF IMAGE IN LAB COLOR FORMAT USING A TRAINING DATASET
    '''
    def quantization(self):
        # create folder to save results
        if os.path.exists('temp/quantized'):
            shutil.rmtree('temp/quantized')
        else:
            os.makedirs('temp/quantized')

        print('Quantizing source image\n.\n.\n.')

        # convert source image's color format into lab and reshape it
        source_rgb = cv2.imread('images/source.jpg')
        source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_BGR2LAB)
        height, width, depth = source_lab.shape
        reshaped = source_lab.reshape((height * width, depth))

        # initialize K-Means algorithm and fit it using the reshaped source image
        self.k_means = MiniBatchKMeans(n_clusters = 16)
        self.k_means.fit(reshaped)

        # store centroid lab colors
        self.centroids_lab = self.k_means.cluster_centers_.astype('uint8')

        # reconstruct the quantized version of the source image
        pixel_labels = self.k_means.predict(reshaped)
        self.quantized = self.centroids_lab[pixel_labels]
        self.quantized = self.quantized.reshape((height, width, depth))

        # convert quantized image into rgb and save it
        quantized_rgb = cv2.cvtColor(self.quantized, cv2.COLOR_LAB2BGR)
        cv2.imwrite('temp/quantized/source_quantized.jpg', quantized_rgb)
        cv2.imwrite('temp/quantized/source_lab_quantized.jpg', self.quantized)

        print('Done!\n')

    '''
    (iii) IMAGE SEGMENTATION INTO SUPERPIXELS USING THE SLIC ALGORITHM
    '''
    def slic(self):
        # create folders to save results
        for folder_name in ['temp/superpixels', 'temp/superpixels/source_superpixels', 'temp/superpixels/target_superpixels']:
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)
            else:
                os.makedirs(folder_name)

        print('Applying SLIC algorithm for source and target images\n.\n.\n.')

        # SOURCE IMAGE
        # get the rgb version of the source image and apply the SLIC algorithm on it
        source_rgb = cv2.imread('Images/source.jpg')
        groups = slic(image = img_as_float(source_rgb), n_segments = 100, compactness = 10, sigma = 1)
        group_ids = np.unique(groups)

        # iterate through each SLIC group
        for group in group_ids:
            # create a mask to separate the superpixel
            mask = np.zeros(self.quantized.shape[:2], dtype = "uint8")
            mask[groups == group] = 255
            superpixel = cv2.bitwise_and(self.quantized, self.quantized, mask = mask)

            # save the superpixels
            self.source_superpixels.append(superpixel)
            cv2.imwrite('temp/superpixels/source_superpixels/source_superpixel_' + str(group) + '.jpg', superpixel)

        # get the quantized source image, mark all SLIC boundaries and save the final image
        rgb_image = cv2.cvtColor(self.quantized, cv2.COLOR_LAB2BGR)
        slic_image = img_as_ubyte(mark_boundaries(rgb_image, groups))
        cv2.imwrite('temp/superpixels/source_slic.jpg', slic_image)

        # TARGET IMAGE
        # get the target image, convert it to grayscale and apply the SLIC algorithm on it
        target_rgb = cv2.imread('Images/target.jpg')
        target_grayscale = cv2.cvtColor(target_rgb, cv2.COLOR_BGR2GRAY)
        groups = slic(image = img_as_float(target_grayscale), n_segments = 100, compactness = 0.1, sigma = 1)
        group_ids = np.unique(groups)

        # iterate through each SLIC group
        for group in group_ids:
            # create a mask to separate the superpixel
            mask = np.zeros(target_grayscale.shape[:2], dtype = "uint8")
            mask[groups == group] = 255
            superpixel = cv2.bitwise_and(target_grayscale, target_grayscale, mask = mask)

            # save the superpixels
            self.target_superpixels.append(superpixel)
            cv2.imwrite('temp/superpixels/target_superpixels/target_superpixel_' + str(group) + '.jpg', superpixel)

        # mark all SLIC boundaries and save the final image
        slic_img = img_as_ubyte(mark_boundaries(target_grayscale, groups))
        cv2.imwrite('temp/superpixels/target_slic.jpg', slic_img)

        print('Done!\n')

    '''
    (iv) EXTRACT SURF FEATURES FOR EACH SUPERPIXEL
    '''
    def surf(self):
        # create folders to save results
        for folder_name in ['temp/features/surf', 'temp/features/surf/source', 'temp/features/surf/target']:
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)
            else:
                os.makedirs(folder_name)

        print('Computing SURF features for source and target images\n.\n.\n.')

        # initialize SURF object 
        surf = cv2.xfeatures2d.SURF_create()
        surf.setExtended(True)

        # SOURCE IMAGE
        for i, superpixel in enumerate(self.source_superpixels):
            keypoints, descriptors = surf.detectAndCompute(superpixel, None)
            self.source_surf.append(descriptors)

            # mark the keypoints on the superpixel and save it
            surf_image = cv2.drawKeypoints(superpixel, keypoints, None, (255, 0, 0), 4)
            cv2.imwrite('temp/features/surf/source/surf_' + str(i) + '.jpg', surf_image)
        
        # TARGET IMAGE
        for i, superpixel in enumerate(self.target_superpixels):
            keypoints, descriptors = surf.detectAndCompute(superpixel, None)
            self.target_surf.append(descriptors)

            # mark the keypoints on the superpixel and save it
            surf_image = cv2.drawKeypoints(superpixel, keypoints, None, (255, 0, 0), 4)
            cv2.imwrite('temp/features/surf/target/surf_' + str(i) + '.jpg', surf_image)

        print('Done!\n')

    '''
    (iv) EXTRACT GABOR FEATURES FOR EACH SUPERPIXEL
    '''
    def apply_kernels(self, superpixel):
        # create a response image that will include all reactions to the kernel and store each reaction image
        response = np.zeros_like(superpixel)
        responses = []

        for kernel in self.gabor_kernels:
            filtered = cv2.filter2D(superpixel, cv2.CV_8UC3, kernel)    # create a filter and apply the kernel to the image
            responses.append(filtered)                                  # store the individual respons
            np.maximum(response, filtered, response)                    # keep adding into the response image as we go through the kernels

        return response, responses

    def gabor(self):
        # create folders to save results
        for folder_name in ['temp/features/gabor', 'temp/features/gabor/source', 'temp/features/gabor/target']:
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)
            else:
                os.makedirs(folder_name)

        print('Computing GABOR features for source and target images\n.\n.\n.')

        # build gabor kernels so that we can use them in the apply_kernels function
        for i, theta in enumerate(np.arange(0, np.pi, np.pi / 16)):
            kernel = cv2.getGaborKernel((31, 31), 4.0, theta, 10.0, 0.5, 0.0, cv2.CV_32F)   # create 16 gabor kernels 
            kernel /= 1.5 * kernel.sum()                                                    # normalize kernel and save 
            self.gabor_kernels.append(kernel)

        for i, superpixel in enumerate(self.source_superpixels):
            gabor_image, responses = self.apply_kernels(superpixel)                                 # apply all gabor kernels to the superpixel
            self.source_gabor.append(responses)                                                     # store all 16 responses for later
            cv2.imwrite('temp/features/gabor/source/source_gabor_' + str(i) + '.jpg', gabor_image)  # save the response image of the superpixel

        for i, superpixel in enumerate(self.target_superpixels):
            gabor_image, responses = self.apply_kernels(superpixel)                                 # apply all gabor kernels to the superpixel
            self.target_gabor.append(responses)                                                     # store all 16 responses for later
            cv2.imwrite('temp/features/gabor/target/source_gabor_' + str(i) + '.jpg', gabor_image)  # save the response image of the superpixel

        print('Done!\n')

    '''
    (v) BUILDING LOCAL PREDICTION MODELS USING SVM COLOR CLASSIFIERS
    (vi) ESTIMATION OF THE COLOR CONTENT OF A GRAYSCALE IMAGE USING CUTTING GRAPHS ALGORITHMS
    '''
    def create_datasets(self):
        print('Creating dataset for source and target images\n.\n.\n.')

        # SOURCE IMAGE DATASET
        self.ab_channels = self.centroids_lab[:, 1:]      # store a, b values of all quantized colors from k-means
        for idx, color in enumerate(self.ab_channels):    # keep a LAB to index dictionary for all quantized colors
            self.ab_channels_idx[color[0], color[1]] = idx

        centroid_colors = []                            # calculate the LAB color for each superpixel
        for superpixel in self.source_superpixels:
            x_s, y_s, _ = np.nonzero(superpixel)        # find all nonzero pixels within the superpixel
            items = [superpixel[i, j, :] for i, j in zip(x_s, y_s)]
            items = np.array(items)

            # calculate the mean of L, a, b values
            avg_L = np.mean(items[:, 0])
            avg_a = np.mean(items[:, 1])
            avg_b = np.mean(items[:, 2])

            label = self.k_means.predict([[avg_L, avg_a, avg_b]])   # quantize the mean color of the superpixel using k-means
            color = self.centroids_lab[label, 1:]                   # store a, b values of the superpixel
            centroid_colors.append(color)

        # calculate 128 surf and 16 gabor values
        surf_avg = []
        gabor_avg = []
        for surf in self.source_surf:
            average = np.mean(surf, axis = 0).tolist()
            surf_avg.append(average)
        for gabor_superpixel in self.source_gabor:
            local_avg = []
            for gabor in gabor_superpixel:
                average = np.mean(gabor[gabor != 0])
                local_avg.append(average)
            gabor_avg.append(local_avg)

        superpixel_count = len(self.source_superpixels)
        for i in range(superpixel_count):
            # for each superpixel get the surf, gabor values and the color
            surf_feature = surf_avg[i]
            gabor_feature = gabor_avg[i]
            color = centroid_colors[i]

            self.source_x.append(surf_feature + gabor_feature)
            self.source_y.append(self.ab_channels_idx[color[0, 0], color[0, 1]])

        # regularize the dataset
        self.source_x = preprocessing.scale(self.source_x)

        # TARGET IMAGE DATASET
        # calculate 128 surf and 16 gabor values
        surf_avg = []
        gabor_avg = []
        for surf in self.target_surf:
            average = np.mean(surf, axis=0).tolist()
            surf_avg.append(average)
        for gabor_superpixel in self.target_gabor:
            local_avg = []
            for gabor in gabor_superpixel:
                average = np.mean(gabor[gabor != 0])
                local_avg.append(average)
            gabor_avg.append(local_avg)

        superpixel_count = len(self.target_superpixels)
        for i in range(superpixel_count):
            # for each superpixel get the surf and gabor values, without the color value in this case
            surf_feature = surf_avg[i]
            gabor_feature = gabor_avg[i]

            sample = surf_feature + gabor_feature
            self.target_x.append(sample)

        # regularize the dataset
        self.target_x = preprocessing.scale(self.target_x)

        print('Done!\n')

    def colorize_target(self):
        print('Colorizing target image\n.\n.\n.')

        model = svm.SVC()                                  # initialize an SVM model and train it using source the dataset we created
        model.fit(self.source_x, self.source_y)
        predictions = model.predict(self.source_x)         # make predictions and compute accuracy
        print("Prediction accuracy: " + str(metrics.accuracy_score(self.source_y, predictions)))

        labels = model.predict(self.target_x)              # get predicted labels using the SVM for the target dataset
        color_labels = self.ab_channels[labels]            # get a, b values for color

        # get target image and convert it to grayscale
        target = cv2.imread('images/target.jpg')
        target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        # create a blank copy of the target image to colorize
        target_colored = np.zeros((target.shape[0], target.shape[1], 3), dtype = 'uint8')

        for idx, superpixel in enumerate(self.target_superpixels):
            x_s, y_s = np.nonzero(superpixel)                   # for each superpixel find nonzero pixels
            for i, j in zip(x_s, y_s):                          # colorize every pixel according to the predicted values
                target_colored[i, j, 0] = target[i, j]          # L
                target_colored[i, j, 1] = color_labels[idx, 0]  # a
                target_colored[i, j, 2] = color_labels[idx, 1]  # b

        # convert the colorized image from LAB to RGB and store it
        target_colored = cv2.cvtColor(target_colored, cv2.COLOR_LAB2BGR)
        cv2.imwrite('temp/target_colored.jpg', target_colored)

        print('Done!\n')

if __name__ == "__main__":
    func = Functions()
    func.convert_colorspace()
    func.quantization()
    func.slic()
    func.surf()
    func.gabor()
    func.create_datasets()
    func.colorize_target()
