# Import packages
from Video_Functions import peri_stimulus_video_clip, register_arena, get_background
from termcolor import colored
import cv2

'''
--------------   SET PARAMETERS    --------------
'''

# file path of behaviour video to register
video_file_path = ('/Volumes/Extreme SSD/Branco Lab/cam13.avi')

# file path of behaviour clip to save
save_file_path = '/Users/chenx/Desktop/Branco Lab/Shelter Choice Project/Maxwell-common-coordinate-behaviour/corrected_videos/'

# file path of fisheye correction -- set to an invalid location such as '' to skip fisheye correction
# A corrective mapping for the Branco lab's typical camera is included in the repo!
fisheye_map_location = ('/Users/chenx/Desktop/Branco Lab/Shelter Choice Project/Common-Coordinate-Behaviour-master/fisheye calibration maps/fisheye_maps.npy')

# frame of stimulus onset
stim_frame = 135000

# seconds before and after stimulus to display
window_pre = 10
window_post = 20

# frames per second of video
fps = 40

# name of experiment
experiment = 'Haran_Maxwell_test_footage_1'

# name of animal
animal_id = 'test_1'

# stimulus type
stim_type = 'auditory'

# The fisheye correction works on the entire frame. If not recording full-frame, put the x and y offset here
#We recorded FULL-FRAME so no need for offsets!
# x_offset = 120
# y_offset = 300
x_offset = 0
y_offset = 0





'''
--------------   GET BACKGROUND IMAGE    --------------
'''
print(colored('\nFetching background', 'green'))
background_image = get_background(video_file_path, start_frame=10000, avg_over=10)
# cv2.imshow('Background Image', background_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


'''
--------------   REGISTER ARENA TO COMMON COORDINATE BEHAVIOUR    --------------
'''
print(colored('\nRegistering arena', 'green'))
# Important: you'll need to modify the model_arena function (1st function in Video_Functions) using opencv,
#            to reflect your own arena instead of ours
registration = register_arena(background_image, fisheye_map_location, x_offset, y_offset, show_arena = False)
# This outputs a variable called 'registration' which is used in the video clip function below to register each frame
# feel free to save it, using: np.save(experiment + animal +'_transform',registration)



'''
--------------   SAVE VIDEO CLIPS    --------------
'''
videoname = '{}_{}_{}-{}\''.format(experiment,animal_id,stim_type, round(stim_frame / fps / 60))
print(colored('Creating behavior clip for ' + videoname, 'green'))

peri_stimulus_video_clip(video_file_path, videoname, save_file_path, stim_frame, window_pre, window_post,
                         registration, x_offset, y_offset, fps=fps, save_clip = True, counter = True)

print(colored('done'))