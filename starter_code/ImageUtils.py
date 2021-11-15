import numpy as np

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    ### END CODE HERE

    image = preprocess_image(image, training) # If any.

    image = np.transpose(image, [2, 0, 1])
    
    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """
    ### YOUR CODE HERE
    if training:
        # Resize the image to add four extra pixels on each side.
        resized_img = np.pad(image,((4,4),(4,4),(0,0)),constant_values=(0))
        # print(resized_img.shape)
        
        # Randomly crop a [32, 32] section of the image.
        
        i=np.random.choice(9)
        j=np.random.choice(9)
        cropped_img = resized_img[i:i+32,j:j+32,]
        #print('cropped_img shape is : ',cropped_img.shape)
        
        # Randomly flip the image horizontally.
        image = np.flip(cropped_img,axis=1)

    # Subtract off the mean and divide by the standard deviation of the pixels.
    image = (image-np.mean(image))/np.std(image)

    return image


# Other functions
### YOUR CODE HERE

### END CODE HERE