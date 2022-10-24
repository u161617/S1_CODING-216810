from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import cv2


def cutVideo(video):
    # Function that trims a video
    #   video: (str) Name of the file that we want to trim
    start = 0
    end = int(input("Choose how many seconds should last the video: "))
    ffmpeg_extract_subclip(
        video, start, end, targetname='outputs/'+"video_cut.mp4")
    return "video_cut.mp4"


def displayVideo(video):
    # Function that trims a video file and display it with it's YUV histogram.
    #   video: (str) Name of the video file
    cut_video = cutVideo(video)
    cap = cv2.VideoCapture('outputs/'+cut_video)
    video_out = cv2.VideoWriter('outputs/'+'video_histogram.avi', 0, 30, (1920, 1080))
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            plt.close('all')
            figure = createHistogram(frame)
            x_offset = y_offset = 50
            frame[y_offset:y_offset + figure.shape[0], x_offset:x_offset + figure.shape[1]] = figure
            video_out.write(frame)
        else:
            break

    cap.release()
    video_out.release()
    cv2.destroyAllWindows()


def createHistogram(image_in):
    # Function that creates an histogram of a given image
    #   image_in: (image) Image
    img_out = cv2.cvtColor(image_in, cv2.COLOR_RGB2YUV)
    colors = ("black", "magenta", "cyan")
    channel_ids = (0, 1, 2)

    fig = plt.figure()
    plt.xlim([0, 256])
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(
            img_out[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=c)

    plt.title("Color Histogram")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")

    # convert figure to image type
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return img


def resizeVideo(video):
    cut_video = cutVideo(video)
    cap = cv2.VideoCapture('outputs/'+cut_video)
    res = input('Choose the resolution you want:'
                '\n# 1: 720p'
                '\n# 2: 480p'
                '\n# 3: 360x240'
                '\n# 4: 160x120'
                '\n')
    # en funcion del input, se escoge una resolución
    if res == '1':
        resolution = (1280, 720)
    elif res == '2':
        resolution = (852, 480)
    elif res == '3':
        resolution = (360, 240)
    elif res == '4':
        resolution = (160, 120)

    video_out = cv2.VideoWriter('outputs/'+'video_resize.avi', 0, 30, resolution)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            b = cv2.resize(frame, resolution)
            video_out.write(b)
        else:
            break

    cap.release()
    video_out.release()
    cv2.destroyAllWindows()


def stereo2mono(sound):
    sound_file = AudioSegment.from_wav(sound)

    if sound_file.channels == 2:
        #convert stereo to mono
        sound_mono = sound_file.set_channels(1)
        sound_mono.export('outputs/'+sound.split('.')[0]+"_mono.wav", format="wav")
    if sound_file.channels == 1:
        #convert mono to stereo
        # afegim 3 milisegons de silenci a un dels dos canals per donar la sensació de stereo
        silence = AudioSegment.silent(duration=3)
        left_channel = right_channel = sound_file
        stereo_sound = AudioSegment.from_mono_audiosegments(left_channel+silence, silence+right_channel)
        stereo_sound.export('outputs/'+sound.split('.')[0]+"_stereo.wav", format="wav")


if __name__ == '__main__':
    video = "bbb.mp4"
    res = input('Escull una acció a fer:'
                '\n#1 Tallar un video'
                '\n#2 Veure lhistograma YUV del video'
                '\n#3 Canviar la resolució del video'
                '\n#4 Convertir de stereo a mono un audio'
                '\n')
    if res == '1':
        cutVideo(video)
    if res == '2':
        displayVideo(video)
    if res == '3':
        resizeVideo(video)
    if res == '4':
        audio = input('Que vols convertir:'
                      '\n#1 Audio Stereo a Mono'
                      '\n#2 Audio Mono a Stereo'
                      '\n')
        if audio == '1':
            sound = "stereo_sound.wav"
            stereo2mono(sound)

        if audio == '2':
            sound = "mono_sound.wav"
            stereo2mono(sound)
print('Els resultats estan a la carpeta de Outputs')
