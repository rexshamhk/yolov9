import cv2

# Create a video reader object for RTSP H.265 stream
reader = cv2.cudacodec.createVideoReader('rtsp://admin:Insight108!@192.168.140.95:554/Streaming/Channels/201', cv2.cudacodec.VideoReader_NVCODEC_H265)

# Read frames from the stream
while True:
    success, frame = reader.nextFrame()
    if not success:
        break
    # Process the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break