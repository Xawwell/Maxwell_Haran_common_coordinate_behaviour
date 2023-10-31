# Import packages
from Video_Functions import peri_stimulus_video_clip, register_arena, get_background, entire_video_clip
from termcolor import colored

'''
--------------   SET PARAMETERS    --------------
'''

# file path of behaviour video to register

video_file_path =r'G:\Files_from_npx_bonzai_pc\raw_data\sub-011_mouseid-1120293_behaviour_2023_06_16T09_19_12\cam.avi'

#r'G:\Files_from_npx_bonzai_pc\bins_and_meta\sub-003_mouseid-1119984_behaviour_2023_06_13T15_02_09\cam.avi'
#r'C:\Users\Haran Shani\Downloads\Common-Coordinate-Behaviour-master\Common-Coordinate-Behaviour-master\behavior videos\CA3481_loom.mp4'

# file path of behaviour clip to save
save_file_path = r'G:\Maxwell\Corrected_Video'

# file path of fisheye correction -- set to an invalid location such as '' to skip fisheye correction
# A corrective mapping for the Branco lab's typical camera is included in the repo!
fisheye_map_location = r'G:\Maxwell\code\WORKING_Maxwell_Common-Coordinate-Behaviour-master\Common-Coordinate-Behaviour-master\fisheye calibration maps\fisheye_maps.npy'

# frame of stimulus onset
stim_frame = 300

# seconds before and after stimulus to display
window_pre = 3
window_post = 7

# frames per second of video
fps = 40

# name of experiment
experiment = 'Maxwell'

# name of animal
animal_id = 'mouseid-1120293'

# stimulus type
stim_type = 'audiotory'

# The fisheye correction works on the entire frame. If not recording full-frame, put the x and y offset here
x_offset: int= 128
y_offset: int= 0





'''
--------------   GET BACKGROUND IMAGE    --------------
'''
print(colored('\nFetching background', 'green'))
background_image = get_background(video_file_path, start_frame=1000, avg_over=10)



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
videoname = '{}_{}_{}-{}\''.format(experiment,animal_id,stim_type, round(stim_frame / fps / 40))
print(colored('Creating behavior clip for ' + videoname, 'green'))

# peri_stimulus_video_clip(video_file_path, videoname, save_file_path, stim_frame, window_pre, window_post,
#                          registration, x_offset, y_offset, fps=fps, save_clip = True, counter = True)

#we are going to process the entire video
entire_video_clip(video_file_path, videoname , save_file_path, registration, x_offset, y_offset, fps=fps ,save_clip=True, border_size=20)

print(colored('done'))