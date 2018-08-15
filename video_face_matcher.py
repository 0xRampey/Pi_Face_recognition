#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation. 
# License: MIT See LICENSE file in root directory.


from mvnc import mvncapi as mvnc
import numpy
import cv2
import sys
import os
import face_recognition

EXAMPLES_BASE_DIR='../../'
IMAGES_DIR = './'

VALIDATED_IMAGES_DIR = IMAGES_DIR + 'validated_images/'

GRAPH_FILENAME = "facenet_celeb_ncs.graph"

# name of the opencv window
CV_WINDOW_NAME = "FaceNet"

CAMERA_INDEX = 0
REQUEST_CAMERA_WIDTH = 640
REQUEST_CAMERA_HEIGHT = 480

# the same face will return 0.0
# different faces return higher numbers
# this is NOT between 0.0 and 1.0
FACE_MATCH_THRESHOLD = 0.65


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
    facenet_graph.LoadTensor(image_to_classify.astype(numpy.float16), None)

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

# overlays the boxes and labels onto the display image.
# display_image is the image on which to overlay to
# image info is a text string to overlay onto the image.
# matching is a Boolean specifying if the image was a match.
# returns None
def overlay_on_image(display_image, face_locations, face_name):
    # rect_width = 10
    # offset = int(rect_width/2)
    # if (image_info != None):
    #     cv2.putText(display_image, image_info, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    for (top, right, bottom, left) in face_locations:

        # Draw a box around the face
        cv2.rectangle(display_image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(display_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(display_image, face_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    # else:
    #     # not a match, red rectangle
    #     cv2.rectangle(display_image, (0+offset, 0+offset),
    #                   (display_image.shape[1]-offset-1, display_image.shape[0]-offset-1),
    #                   (0, 0, 255), 10)


# whiten an image
def whiten_image(source_image):
    source_mean = numpy.mean(source_image)
    source_standard_deviation = numpy.std(source_image)
    std_adjusted = numpy.maximum(source_standard_deviation, 1.0 / numpy.sqrt(source_image.size))
    whitened_image = numpy.multiply(numpy.subtract(source_image, source_mean), 1 / std_adjusted)
    return whitened_image

# create a preprocessed image from the source image that matches the
# network expectations and return it
def preprocess_image(src):
    # scale the image
    NETWORK_WIDTH = 160
    NETWORK_HEIGHT = 160
    preprocessed_image = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))

    #convert to RGB
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)

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
        this_diff = numpy.square(face1_output[output_index] - face2_output[output_index])
        total_diff += this_diff
    print('Face threshold difference is: ' + str(total_diff))

    if (total_diff < FACE_MATCH_THRESHOLD):
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


# start the opencv webcam streaming and pass each frame
# from the camera to the facenet network for an inference
# Continue looping until the result of the camera frame inference
# matches the valid face output and then return.
# valid_output is inference result for the valid image
# validated image filename is the name of the valid image file
# graph is the ncsdk Graph object initialized with the facenet graph file
#   which we will run the inference on.
# returns None
def run_camera(known_face_encodings, graph):
    camera_device = cv2.VideoCapture(CAMERA_INDEX)
    camera_device.set(cv2.CAP_PROP_FRAME_WIDTH, REQUEST_CAMERA_WIDTH)
    camera_device.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUEST_CAMERA_HEIGHT)

    actual_camera_width = camera_device.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_camera_height = camera_device.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print ('actual camera resolution: ' + str(actual_camera_width) + ' x ' + str(actual_camera_height))

    if ((camera_device == None) or (not camera_device.isOpened())):
        print ('Could not open camera.  Make sure it is plugged in.')
        print ('Also, if you installed python opencv via pip or pip3 you')
        print ('need to uninstall it and install from source with -D WITH_V4L=ON')
        print ('Use the provided script: install-opencv-from_source.sh')
        return

    frame_count = 0

    found_match = False

    while True :
        # Read image from camera,
        ret_val, vid_image = camera_device.read()
        if (not ret_val):
            print("No image from camera, exiting")
            break

        frame_count += 1
        frame_name = 'camera frame ' + str(frame_count)

        #Extract faces found in the image
        face_locations = get_face_loc(vid_image)
        face_images = extract_faces(vid_image, face_locations)

        #Perform inference only when when you detect faces
        if(len(face_images)):

          for face_idx, face in enumerate(face_images):
            unknown = True
            # get a resized version of the image that is the dimensions
            # Facenet expects
            resized_image = preprocess_image(face)

            # run a single inference on the image and overwrite the
            # boxes and labels
            face_enc = run_inference(resized_image, graph)
            for name, known_enc in known_face_encodings.items():
                if (face_match(known_enc, face_enc)):
                    print('PASS!  Found ' + name + '!')
                    unknown = False
                    # Since we found a match for our face, lets move on to the next face found in the frame
                    break
            if (unknown):
              print("Don't know who the face is :(")

        else:
            print("No faces detected!")

        raw_key = cv2.waitKey(0)
        if (raw_key != -1):
            if (handle_keys(raw_key) == False):
                print('user pressed Q')
                break


def load_known_face_encodings(img_dir, graph):
    known_face_enc={}
    dir_listings = os.listdir(img_dir)
    face_image_listings = [i for i in dir_listings if i.endswith('.jpg')]
    if (len(face_image_listings) < 1):
            print('No image files found')
            return 1
    for img in face_image_listings:
        img_name = img.split(".")[0]
        img = cv2.imread(img_dir + img)
        print("Loading face encoding of " + img_name)
        img = preprocess_image(img)
        known_face_enc[img_name] =  run_inference(img, graph)
    return known_face_enc

# This function is called from the entry point to do
# all the work of the program
def main():

    # Get a list of ALL the sticks that are plugged in
    # we need at least one
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No NCS devices found')
        quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])

    # Open the NCS
    device.OpenDevice()

    # The graph file that was created with the ncsdk compiler
    graph_file_name = GRAPH_FILENAME

    # read in the graph file to memory buffer
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    # create the NCAPI graph instance from the memory buffer containing the graph file.
    graph = device.AllocateGraph(graph_in_memory)

    # validated_image = cv2.imread(validated_image_filename)
    # validated_image = preprocess_image(validated_image)
    # valid_output = run_inference(validated_image, graph)

    known_face_encodings = load_known_face_encodings(VALIDATED_IMAGES_DIR, graph)

    run_camera(known_face_encodings, graph)

    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()


# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
