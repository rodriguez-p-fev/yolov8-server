import cv2
import numpy
import socket
import struct
import threading
from io import BytesIO
from ultralytics import YOLO
import supervision as sv

class Streamer(threading.Thread):

    def __init__(self, hostname, port):
        threading.Thread.__init__(self)

        self.hostname = hostname
        self.port = port
        self.running = False
        self.streaming = False
        self.jpeg = None
        self.yolo_model = YOLO('yolov8s.pt')
        self.box_annotator = sv.BoxAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=1
        )

    def run(self):

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('Socket created')

        s.bind((self.hostname, self.port))
        print('Socket bind complete')

        payload_size = struct.calcsize("L")

        s.listen(10)
        print('Socket now listening')

        self.running = True

        

        while self.running:

            print('Start listening for connections...')

            conn, addr = s.accept()
            print("New connection accepted.")

            while True:

                data = conn.recv(payload_size)

                if data:
                    # Read frame size
                    msg_size = struct.unpack("L", data)[0]

                    # Read the payload (the actual frame)
                    data = b''
                    while len(data) < msg_size:
                        missing_data = conn.recv(msg_size - len(data))
                        if missing_data:
                            data += missing_data
                        else:
                            # Connection interrupted
                            self.streaming = False
                            break

                    # Skip building frame since streaming ended
                    if self.jpeg is not None and not self.streaming:
                        continue

                    # Convert the byte array to a 'jpeg' format
                    memfile = BytesIO()
                    memfile.write(data)
                    memfile.seek(0)
                    frame = numpy.load(memfile)
                    #print(frame.dtype)
                    #print(frame.shape)
                    #'''
                    result = next(self.yolo_model(frame, agnostic_nms=True, verbose=False, stream=True))
                    #print(result)
                    detections = sv.Detections.from_yolov8(result)
                    labels = [
                        f'{self.yolo_model.model.names[class_id]} {confidence:0.2f}'
                        for _, confidence, class_id, _ in detections
                    ]
                    frame=self.box_annotator.annotate(scene=frame,detections=detections,labels=labels)
                    #'''

                    ret, jpeg = cv2.imencode('.jpg', frame)
                    self.jpeg = jpeg

                    self.streaming = True



                else:
                    conn.close()
                    print('Closing connection...')
                    self.streaming = False
                    self.running = False
                    self.jpeg = None
                    break

        print('Exit thread.')

    def stop(self):
        self.running = False

    def get_jpeg(self):
        return self.jpeg.tobytes()
