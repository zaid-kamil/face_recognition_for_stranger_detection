import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

path = 'faces/'
encoding_db_path = 'appdb/'
save_path = 'recordings/'


ts = datetime.strftime(datetime.now(), "_%Y%m%d_%H%M")
recording_file = os.path.join(save_path,  f'recording{ts}.avi')

known_face_encodings = []
known_face_names = []

camera_id = 0

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

known_faces = os.path.join(encoding_db_path,'known_faces.npy')
known_names = os.path.join(encoding_db_path,'known_names.npy')

font = cv2.FONT_HERSHEY_DUPLEX

print('Loading known faces...')
# Load a sample picture and learn how to recognize it.

if os.path.exists(known_faces):
    known_face_encodings = np.load(known_faces)
    known_face_names = np.load(known_names)
    print('Loaded known faces from disk.')
else:
    for image in os.listdir(path):
        print(image)
        images = face_recognition.load_image_file(path + image)
        images_encoding = face_recognition.face_encodings(images)[0]
        known_face_encodings.append(images_encoding)
        known_face_names.append(image[:-4])
    np.save(known_faces, known_face_encodings)
    np.save(known_names, known_face_names)
    print('Saved known faces to disk.')


video_capture = cv2.VideoCapture(0)
encoder = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(recording_file, encoder, 20.0, (640, 480))


def display_boxes(frame, face_locations, face_names):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        if 'Stranger' in name:
            # draw red box
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        else:
            # draw green box
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    return frame


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    if not ret:
        print('Failed to capture frame from camera. Check camera index or path of video \n')
        break

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        print(f'Found {len(face_locations)} faces in frame.')

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Stranger"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            print(" ".join(face_names),'found in frame')

            if 'stranger' in face_names:
                # add a red text at the top of the frame
                cv2.putText(frame, 'Stranger Detected', (10, 50), font, 1.0, (0, 0, 255), 1)
    
    
    process_this_frame = not process_this_frame 
    frame = display_boxes(frame, face_locations, face_names)
    # add a recording text at the top of the frame with a date time stamp in small font
    cv2.rectangle(frame, (0, 0), (640, 30), (100, 255, 100), cv2.FILLED)
    cv2.putText(frame, f'Recording {datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")}', (10, 20), font, .5, (0, 0, 0), 1)
    # Display the resulting image
    cv2.imshow('Video', frame)
    
    # save the video
    out.write(frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
out.release()