
import face_recognition
from PIL import Image

image = face_recognition.load_image_file("validated_images/prudhvi.jpg")
face_locations = face_recognition.face_locations(image)
print("I found {} face(s) in this photograph.".format(len(face_locations)))
    # face_img_list=[]
for face_location in face_locations:

       # Print the location of each face in this image
      top, right, bottom, left = face_location
      print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

      # You can access the actual face itself like this:
      face_image = image[top-80:bottom+80, left-20:right+20]
      im = Image.fromarray(face_image)
      im.save("validated_images/pru_crop.jpg")
