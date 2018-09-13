#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.


from mvnc import mvncapi as mvnc
import numpy as np
import cv2
import picamera
import sys
import os
import face_recognition
import pickle
import argparse

# Run an inference on the passed image
# image_to_classify is the image on which an inference will be performed
#    upon successful return this image will be overlayed with boxes
#    and labels identifying the found objects within the image.
# ssd_mobilenet_graph is the Graph object from the NCAPI which will
#    be used to peform the inference.
def run_inference(image_to_classify, facenet_graph):

    # ***************************************************************
    # Send the image to the NCS
    # ***************************************************************
    facenet_graph.LoadTensor(image_to_classify.astype(np.float16), None)

    # ***************************************************************
    # Get the result from the NCS
    # ***************************************************************
    output, userobj = facenet_graph.GetResult()

    return output

## Returns any detected face locations for a video frame
def get_face_loc(vid_frame):
    face_locations = face_recognition.face_locations(vid_frame)
    print("I found {} face(s) in this photograph.".format(len(face_locations)))
    return face_locations

def new_coord(top, right, bottom, left):
    width = right - left
    height = bottom - top
    top = int(top - height/8)
    bottom = int(bottom + height/8)
    left = int(left - width/8)
    right = int(right + width/8)
    print(top, right, bottom, left)
    return (top, right, bottom, left)

## Extracts cropped face images from a video frame based on the face location coordinates given to it
def extract_faces(vid_frame, face_locations):
    face_img_list=[]
    for face_location in face_locations:

       # Print the location of each face in this image
      top, right, bottom, left = face_location
      print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

      top, right, bottom, left = new_coord(top, right, bottom, left)


      # You can access the actual face itself like this:
      face_image = vid_frame[top:bottom, left:right]
      face_img_list.append(face_image)
    return face_img_list

# whiten an image
def whiten_image(source_image):
    source_mean = np.mean(source_image)
    source_standard_deviation = np.std(source_image)
    std_adjusted = np.maximum(source_standard_deviation, 1.0 / np.sqrt(source_image.size))
    whitened_image = np.multiply(np.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

# create a preprocessed image from the source image that matches the
# network expectations and return it
def preprocess_image(src):
    # scale the image
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    #whiten
    preprocessed_image = whiten_image(preprocessed_image)

    # return the preprocessed image
    return preprocessed_image

# determine if two images are of matching faces based on the
# the network output for both images.
def face_match(face1_output, face2_output):
    if (len(face1_output) != len(face2_output)):
        print('length mismatch in face_match')
        return False
    total_diff = 0
    for output_index in range(0, len(face1_output)):
        this_diff = np.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
    print('Face threshold difference is: ' + str(total_diff))

    if (total_diff < ARGS.match_threshold):
        # the total difference between the two is under the threshold so
        # the faces match.
        return True

    # differences between faces was over the threshold above so
    # they didn't match.
    return False

# handles key presses
# raw_key is the return value from cv2.waitkey
# returns False if program should end, or True if should continue
def handle_keys(raw_key):
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True


def predict(face_encodings, distance_threshold):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """

    with open(ARGS.model_path, 'rb') as f:
        knn_clf = pickle.load(f)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <=
                   distance_threshold for i in range(len(face_encodings))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred) if rec else ("unknown") for pred, rec in zip(knn_clf.predict(face_encodings), are_matches)]

# start the opencv webcam streaming and pass each frame
# from the camera to the facenet network for an inference
# Continue looping until the result of the camera frame inference
# matches the valid face output and then return.
# valid_output is inference result for the valid image
# validated image filename is the name of the valid image file
# graph is the ncsdk Graph object initialized with the facenet graph file
#   which we will run the inference on.
# returns None
def run_face_rec(camera, graph):

          pic = np.empty(ARGS.camera_res[::-1] + (3,), dtype=np.uint8)
          print("Capturing image of size: ", pic.shape)
          # Grab a single frame of video from the RPi camera as a np array
          camera.capture(pic, format="rgb")

          #Extract faces found in the image
          face_locations = get_face_loc(pic)
          face_images = extract_faces(pic, face_locations)

          #Perform inference only when when you detect faces
          if(len(face_images)):

            face_enc_list = []
            for face_idx, face in enumerate(face_images):
              # get a resized version of the image that is the dimensions
              # Facenet expects
              resized_image = preprocess_image(face)
              # run a single inference on the image and overwrite the
              # boxes and labels
              face_enc = run_inference(resized_image, graph)
              face_enc_list.append(face_enc)

            print(predict(face_enc_list, ARGS.match_threshold))

          else:
            print("No faces detected!")


def initCamera():
    camera = picamera.PiCamera()
    camera.resolution = ARGS.camera_res
    print("Camera ready!")
    return camera

# This function is called from the entry point to do
# all the work of the program
def main(ARGS):

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
    graph_file_name = ARGS.graph_file

    # read in the graph file to memory buffer
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    print("Loading graph file into NCS...")
    # create the NCAPI graph instance from the memory buffer containing the graph file.
    graph = device.AllocateGraph(graph_in_memory)

    #Setting up camera and button trigger
    camera = initCamera()

    run_face_rec(camera, graph)

    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recognize faces")

    parser.add_argument('-g', '--graph_file', type=str,
                        default="facenet_celeb_ncs.graph",
                        help="Path to the neural network graph file.")

    parser.add_argument('-C', '--camera_res', type=int,
                        nargs='+',
                        default=(320, 240),
                        help="Camera resolution (width, height) . ex. -C 320 240")

    parser.add_argument('-m', '--model_path', type=str,
                        default='./models/knn_model.clf',
                        help="Path to a trained and pickled KNN classifier model")

    parser.add_argument('-t', '--match_threshold', type=int,
                        default=0.8,
                        help="Distance threshold until which two faces can be considered the same")

    ARGS = parser.parse_args()
    sys.exit(main(ARGS))
