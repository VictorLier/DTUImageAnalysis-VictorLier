from skimage import color
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
from skimage.filters import prewitt
from skimage.filters import threshold_otsu
from skimage.filters import median
import time
import cv2
import numpy as np

def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)



def Prewitt(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image

	return (prewitt(Image))	

def Threshold_otsu(Image):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image
	return threshold_otsu(Image)
 
def threshold_image(img_in, thres):
    if img_in.dtype == 'uint8':
        max = 255
        min = 0

    elif img_in.dtype == 'uint16':
        max = 65535
        min = 0
    
    elif img_in.dtype == 'uint32':
        max = 2**32 - 1
        min = 0

    elif img_in.dtype == 'float':
        max = 1
        min = 0

    else:  
        print("Billede type eksistere ikke")

    img_in[img_in > thres] = max
    img_in[img_in < thres] = min
    img = img_in

    return img_as_ubyte(img)

def MedianFilter(Image, Size):
	if len(Image.shape) == 3:
		Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
	else:
		Image = Image
	footprint = np.ones([Size,Size])
	Image_out = median(Image,footprint)

	return Image_out


def process_gray_image(img):
    """
    Do a simple processing of an input gray scale image and return the processed image.
    # https://scikit-image.org/docs/stable/user_guide/data_types.html#image-processing-pipeline
    """
    # Do something here:
    img = MedianFilter(img, 10)
    im_prew = Prewitt(img)
    thres = Threshold_otsu(im_prew)
    proc_img = threshold_image(im_prew,thres)
    return img_as_ubyte(proc_img)


def process_rgb_image(img):
    """
    Simple processing of a color (RGB) image
    """
    # Copy the image information so we do not change the original image
    proc_img = img.copy()
    r_comp = proc_img[:, :, 0]
    proc_img[:, :, 0] = 1 - r_comp
    return proc_img


def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # To keep track of frames per second using a high-performance counter
    old_time = time.perf_counter()
    fps = 0
    stop = False
    process_rgb = False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Change from OpenCV BGR to scikit image RGB
        new_image = new_frame[:, :, ::-1]
        new_image_gray = color.rgb2gray(new_image)
        if process_rgb:
            proc_img = process_rgb_image(new_image)
            # convert back to OpenCV BGR to show it
            proc_img = proc_img[:, :, ::-1]
        else:
            proc_img = process_gray_image(new_image_gray)

        # update FPS - but do it slowly to avoid fast changing number
        new_time = time.perf_counter()
        time_dif = new_time - old_time
        old_time = new_time
        fps = fps * 0.95 + 0.05 * 1 / time_dif

        # Put the FPS on the new_frame
        str_out = f"fps: {int(fps)}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Input gray', new_image_gray, 600, 10)
        show_in_moved_window('Processed image', proc_img, 1200, 10)

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()