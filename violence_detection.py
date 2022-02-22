import tensorflow as tf
import cv2
import numpy as np
import action

violence_model = tf.keras.models.load_model('./modelHMD_withNewDataset1.h5')

def main(video_input_file_path, height, width):
    video_frames_optical_flow = list()
    i = 0
    cap = cv2.VideoCapture(video_input_file_path)
    ret1, frame1 = cap.read()
    frame1 = cv2.resize(frame1, (width, height))
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    if not cap.isOpened():
        print("Error opening video stream or file")

    while cap.isOpened():

        ret2, frame2 = cap.read()

        if ret2:

            frame2 = cv2.resize(frame2, (width, height))
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            video_frames_optical_flow.append(bgr)
            if len(video_frames_optical_flow) == 50:
                x1 = np.array(video_frames_optical_flow)
                x1 = x1.astype('float32')
                x1 /= 255
                x1 = x1.reshape((1, 50, 100, 100, 3))
                prediction = violence_model.predict(x1)
                if np.argmax(prediction, axis=1) == 0:
                    print('violence detected')
                    action.send_notification()
                else:
                    print('non violence')
                video_frames_optical_flow = []
        else:
            break

        i += 1
        prvs = next
    cap.release()
    cv2.destroyAllWindows()

main('rtsp://admin:@192.168.86.200/H265?ch=1&subtype=0', 100, 100)
# def generatorPredict(path):
#     optical_flow = extract_videos3D_optical_flow(path, 100, 100)

#     if len(optical_flow) < 50:
#       while len(optical_flow) < 50:
#           optical_flow.append(optical_flow[-1])
#     else:
#       optical_flow = optical_flow[0:50]

#     x1 = np.array(optical_flow)
#     x1 = x1.astype('float32')
#     x1 /= 255
#     x1 = x1.reshape((1, 50, 100, 100, 3))
#     yield x1
  
# video_name = '1.MP4'

# prediction = violence_model.predict(generatorPredict(video_name), steps=1,
#                                                     callbacks=None,
#                                                     max_queue_size=10,
#                                                     workers=1,
#                                                     use_multiprocessing=False,
#                                                     verbose=2)
# pred = np.argmax(prediction, axis=1)
# if pred == 0:
#   print("fight")
# else:
#   print("peace")
