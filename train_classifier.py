from mvnc import mvncapi as mvnc
import math
from sklearn import neighbors
import numpy
import os
import sys
import os.path
import pickle
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from video_face_matcher import preprocess_image, whiten_image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
GRAPH_FILENAME = "facenet_celeb_ncs.graph"
IMAGES_DIR = './'

KNOWN_IMAGES_DIR = IMAGES_DIR + 'known_faces/'


def main():
    # Get a list of ALL the sticks that are plugged in
    # we need at least one
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No NCS devices found')
        quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])

    print("Initializing NCS....")
    # Open the NCS
    device.OpenDevice()

    # The graph file that was created with the ncsdk compiler
    graph_file_name = GRAPH_FILENAME

    # read in the graph file to memory buffer
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    print("Loading graph file into NCS...")
    # create the NCAPI graph instance from the memory buffer containing the graph file.
    graph = device.AllocateGraph(graph_in_memory)

    print("Training KNN classifier...")
    classifier = train(KNOWN_IMAGES_DIR, graph=graph, model_save_path="./models/knn_model.clf", n_neighbors=2)
    print("Training complete!")


def run_inference(image_to_classify, facenet_graph):


    image_to_classify = preprocess_image(image_to_classify)
    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    facenet_graph.LoadTensor(image_to_classify.astype(numpy.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = facenet_graph.GetResult()

    return output

def train(train_dir, graph, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        print("Finding and encoding faces for " + str(class_dir))

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Extract the face from the face location
                face = extract_faces(image, face_bounding_boxes)[0]
                #Obtain face encoding for the face
                encoding = run_inference(face, graph)

                # Add face encoding for current image to the training set
                X.append(encoding)
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def new_coord(top, right, bottom, left):
    width = right - left
    height = bottom - top
    top = int(top - height / 8)
    bottom = int(bottom + height / 8)
    left = int(left - width / 8)
    right = int(right + width / 8)
    print(top, right, bottom, left)
    return (top, right, bottom, left)

## Extracts cropped face images from a video frame based on the face location coordinates given to it
def extract_faces(vid_frame, face_locations):
    face_img_list = []
    for face_location in face_locations:

       # Print the location of each face in this image
      top, right, bottom, left = face_location
      print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(
          top, left, bottom, right))

      top, right, bottom, left = new_coord(top, right, bottom, left)

      # You can access the actual face itself like this:
      face_image = vid_frame[top:bottom, left:right]
      face_img_list.append(face_image)
    return face_img_list

if __name__ == "__main__":
    sys.exit(main())
