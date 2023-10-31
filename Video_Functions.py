import cv2
import numpy as np
import scipy.misc
from termcolor import colored
from tqdm import tqdm
import glob
import os
import subprocess



# =================================================================================
#              CREATE MODEL ARENA FOR COMMON COORDINATE BEHAVIOUR
# =================================================================================
def calculate_triangle_corners(center, radius):
    # Calculate the coordinates of the three corners of the equilateral triangle
    angle = 60  # 60 degrees between each corner of the equilateral triangle

    corner1 = (
        int(center[0] + radius * np.cos(np.radians(30))),
        int(center[1] + radius * np.sin(np.radians(30)))
    )

    corner2 = (
        int(center[0] + radius * np.cos(np.radians(150))),
        int(center[1] + radius * np.sin(np.radians(150)))
    )

    corner3 = (
        int(center[0] + radius * np.cos(np.radians(270))),
        int(center[1] + radius * np.sin(np.radians(270)))
    )

    return corner1, corner2, corner3

def model_arena(size, show_arena=True):
    # Initialize the model arena image
    arena = np.zeros((1000, 1000), dtype=np.uint8)

    # Add the circular arena
    center = (500, 500)
    radius = 460
    cv2.circle(arena, center, radius, 255, -1)

    # Calculate triangle corners
    triangle_corners = calculate_triangle_corners(center, radius)

    # Draw the flipped equilateral triangle on the arena image with black lines
    cv2.polylines(arena, [np.array(triangle_corners, np.int32)], isClosed=True, color=0, thickness=6)

    # For registration, convert points to the resized dimensions
    points = [(pt[0] * size[0] / 1000, pt[1] * size[1] / 1000) for pt in triangle_corners]
    points = np.array(points)
    # Resize the arena to the size of your image
    model_arena = cv2.resize(arena, size)

    '''
    Customize this section above for your own arena!
    '''

    # resize the arena to the size of your image
    model_arena = cv2.resize(model_arena,size)

    if show_arena:
        cv2.imshow('model arena looks like this',model_arena)

    return model_arena, points


# =================================================================================
#              REGISTER A FRAME TO THE COMMON COORDINATE FRAMEWORK
# =================================================================================
def register_frame(frame, registration, x_offset, y_offset, map1, map2):
    '''
    register a grayscale image
    '''
    if [registration]:
        # make sure it's 1-D
        frame_register = frame[:, :, 0]
       # print('this is the frame shape before processing:', frame.shape)

        # do the fisheye correction, if applicable
        if registration[3]:
            # Add borders to the frame so it matches the fisheye map size
            frame_register = cv2.copyMakeBorder(frame_register, y_offset, int((map1.shape[0] - frame.shape[0]) - y_offset),
                                                x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)
            
            # Apply fisheye correction
            frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # Adjust cropping boundaries based on the provided logic
            if frame.shape[0] == map1.shape[0]:
                start_y = 0
                end_y = map1.shape[0]
            else:
                start_y = y_offset
                end_y = -int((map1.shape[0] - frame.shape[0]) - y_offset)

            if frame.shape[1] == map1.shape[1]:
                start_x = 0
                end_x = map1.shape[1]
            else:
                start_x = x_offset
                end_x = -int((map1.shape[1] - frame.shape[1]) - x_offset)

            # Apply cropping
            frame_register = frame_register[start_y:end_y, start_x:end_x]
           # print('Shape of frame_register after processing:', frame_register.shape)

        # apply the affine transformation from the registration (different from the fisheye mapping)
        frame = cv2.warpAffine(frame_register, registration[0], frame.shape[0:2])

    return frame






# =================================================================================
#              GET BACKGROUND IMAGE FROM VIDEO
# =================================================================================
def get_background(vidpath, start_frame = 1000, avg_over = 100):
    """ extract background: average over frames of video """

    vid = cv2.VideoCapture(vidpath)

    # initialize the video
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    print('this is the width of the video,',width)
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('this is the height of the video,',height)
    background = np.zeros((height, width))
    print('this is the shape of the background,',background.shape)

    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # initialize the counters
    every_other = int(num_frames / avg_over)
    j = 0

    for i in tqdm(range(num_frames)):

        if i % every_other == 0:
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vid.read()  # get the frame

            if ret:
                # store the current frame in as a numpy array
                background += frame[:, :, 0]
                j+=1


    background = (background / (j)).astype(np.uint8)
    cv2.imshow('background', background)
    cv2.waitKey(10)
    vid.release()

    return background
    

# =================================================================================
#              GENERATE PERI-STIMULUS VIDEO CLIPS and FLIGHT IMAGE
# =================================================================================
def peri_stimulus_video_clip(vidpath='', videoname='', savepath='', stim_frame=0,
        window_pre=5, window_post=10, registration=0, x_offset=0, y_offset=0, fps=False,
        save_clip=False, counter=True, border_size=20):
    '''
    DISPLAY AND SAVE A VIDEO OF THE BEHAVIORAL EVENT
    '''

    '''
    SET UP PARAMETERS
    '''
    # get the behaviour video parameters
    vid = cv2.VideoCapture(vidpath)
    if not fps: fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH));
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = int(stim_frame - (window_pre * fps));
    stop_frame = int(stim_frame + (window_post * fps))
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # set up the saved video clip parameters
    if save_clip:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_clip = cv2.VideoWriter(os.path.join(savepath, videoname + '.avi'), fourcc, fps,
                                     (width + 2 * border_size * counter, height + 2 * border_size * counter), False)

    # set the colors of the background panel
    if counter: pre_stim_color = 0; post_stim_color = 255

    # setup fisheye correction
    if registration[3]: maps = np.load(registration[3]); map1 = maps[:, :, 0:2]; map2 = maps[:, :, 2] * 0; print(colored('\nFisheye correction running', 'yellow'))
    else: map1 = None; map2 = None; print(colored('Fisheye correction unavailable', 'green'))

    '''
    RUN SAVING AND ANALYSIS OVER EACH FRAME
    '''
    while True:  # and not file_already_exists:
        ret, frame = vid.read()  # get the frame
        if ret:
            # get the frame number
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)

            # register the frame and apply fisheye correction
            frame = register_frame(frame, registration, x_offset, y_offset, map1, map2)

            # Show the boundary and the counter
            if counter:
                # get the counter value (time relative to stimulus onset)
                if frame_num < stim_frame:
                    fraction_done = (frame_num - start_frame) / (stim_frame - start_frame)
                    sign = ''
                else:
                    fraction_done = (frame_num - stim_frame) / (stop_frame - stim_frame)
                    sign = '+'
                cur_color = pre_stim_color * fraction_done + post_stim_color * (1 - fraction_done)

                # border and colored rectangle around frame
                frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=cur_color)

                # report the counter value (time relative to stimulus onset)
                frame_time = (frame_num - stim_frame) / fps
                frame_time = str(round(.2 * round(frame_time / .2), 1)) + '0' * (abs(frame_time) < 10)
                cv2.putText(frame, sign + str(frame_time) + 's', (width - 110, height + 10), 0, 1, 120, thickness=2)

            # display and save the frame (could comment it out for better)
            cv2.imshow(videoname, frame)
            if save_clip: video_clip.write(frame)

            # press 'q' to stop video, or wait until it's over
            if (cv2.waitKey(1) & 0xFF == ord('q')) or frame_num >= stop_frame:
                break

        # in case of being unable to open the frame from the video
        else:
            print('Problem with movie playback');
            cv2.waitKey(1);
            break

    '''
    WRAP UP
    '''
    vid.release()
    if save_clip: video_clip.release()


# ==================================================================================
# Generate lossless video, by having the same bitrate across input and output videos
# ==================================================================================

def get_video_bitrate(input_path):
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=bit_rate', '-of', 'default=noprint_wrappers=1:nokey=1', input_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        bitrate = int(result.stdout.decode().strip())
        
        # Convert from bps to kbps, if the bitrate seems too large
        if bitrate > 10000: 
            bitrate = bitrate // 1000 
        
        return bitrate
    except Exception as e:
        print("Error while getting video bitrate:", e)
        return None

def save_with_ffmpeg(src_path, dest_path, bitrate_kbps):
    cmd = [
    'ffmpeg',
    '-analyzeduration', '2147483647',
    '-probesize', '2147483647',
    '-i', src_path,
    '-c:v', 'libx264',
    '-b:v', f"{bitrate_kbps}k",
    dest_path
    ]

    subprocess.run(cmd)

from tqdm import tqdm
from termcolor import colored

def entire_video_clip(vidpath='', videoname='', savepath='', registration=0, x_offset=0, y_offset=0, fps=False,
                      save_clip=False, border_size=20):

    # Open the video file
    vid = cv2.VideoCapture(vidpath)
    if not vid.isOpened():
        print("Error: Couldn't open the video file.")
        return

    # Generate unique temporary filenames based on the input video's name
    base_name = os.path.basename(vidpath).split('.')[0]  # Extracts name without extension
    temp_output_filename = base_name + '_output.mp4'
    temp_reencoded_filename = base_name + '_temp_reencoded.mp4'

    if not fps:
        fps = vid.get(cv2.CAP_PROP_FPS)
    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_bitrate = get_video_bitrate(vidpath)
    print('this is the input bitrate,', input_bitrate)

    # setup fisheye correction
    if registration[3]:
        maps = np.load(registration[3])
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2] * 0
        print(colored('\nFisheye correction running', 'yellow'))
    else:
        map1 = None
        map2 = None
        print(colored('Fisheye correction unavailable', 'green'))

    # Use the H264 codec for the temporary video
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(temp_output_filename, fourcc, 40.0, (1024, 1024))

    # Generate a unique filename for the temporary video
    temp_filename = os.path.basename(vidpath).replace('.mp4', '_temp_uncompressed.mp4')
    temp_path = os.path.join(savepath, temp_filename)

    video_clip = cv2.VideoWriter(temp_path, fourcc, fps, (width, height), isColor=True)

    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use tqdm for the progress bar
    for _ in tqdm(range(total_frames), desc="Processing", ncols=100):
        ret, frame = vid.read()
        if ret:
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
            # Apply registration
            frame = register_frame(frame, registration, x_offset, y_offset, map1, map2)
            # cv2.imshow(videoname, frame)
            if save_clip:
                video_clip.write(frame)
            if (cv2.waitKey(1) & 0xFF == ord('q')) or frame_num >= total_frames:
                break
        else:
            print('Problem with movie playback')
            cv2.waitKey(1)
            break

    vid.release()
    if save_clip:
        video_clip.release()

        # Re-encode the temporary video to ensure it's in a clean state
        reencode_cmd = [
            'ffmpeg',
            '-i', temp_path,
            '-c:v', 'libx264',
            '-crf', '51',  # Adjust the quality as needed, 0 is lossless, 23 is default, 51 is worst
            temp_reencoded_filename
        ]
        subprocess.run(reencode_cmd)
    
        if input_bitrate:
            save_with_ffmpeg(temp_path, os.path.join(savepath, videoname + '.mp4'), input_bitrate)
            os.remove(temp_path)
            os.remove(temp_reencoded_filename)

# Your helper functions remain unchanged.


# ==================================================================================
# IMAGE REGISTRATION GUI (the following 4 functions are a bit messy and complicated)
# ==================================================================================
def register_arena(background, fisheye_map_location, y_offset, x_offset, show_arena = False):
    #imma hard code this
    x_offset: int= 128
    y_offset: int= 0
    """ extract background: first frame of first video of a session
    Allow user to specify ROIs on the background image """
    print('background.shape[::-1]',background.shape[::-1])
    # create model arena and background
    arena, arena_points = model_arena(background.shape[::-1], show_arena)

    # load the fisheye correction
    try:
        maps = np.load(fisheye_map_location)
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2]*0
        print('we have loaded the map')

        print("Shape of the loaded 'maps':", maps.shape)
        print("Size of the loaded 'maps':", maps.size)

        print("Shape of 'map1':", map1.shape)
        print("Size of 'map1':", map1.size)
        print('map1.shape[0]',map1.shape[0])
        print('map1.shape[1]',map1.shape[1])

        print("Shape of 'map2':", map2.shape)
        print("Size of 'map2':", map2.size)

        print("y_offset:", y_offset)
        print("Calculated bottom:", int((map1.shape[0] - background.shape[0]) - y_offset))
        print("x_offset:", x_offset)
        print("Calculated right:", int((map1.shape[1] - background.shape[1]) - x_offset))
        print("Background shape:", background.shape)
        print("Background dimensions:", background.ndim)

        print('this is the background shape,' , background.shape)
        background_copy = cv2.copyMakeBorder(background, y_offset, int((map1.shape[0] - background.shape[0]) - y_offset),
                                            x_offset, int((map1.shape[1] - background.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)
        print('this is the background_copy shape,' , background_copy.shape)

        background_copy = cv2.remap(background_copy, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        print('this is the background_copy shape after cv2 remap,' , background_copy.shape)
        
        # Adjusted cropping logic
        if background.shape[0] == map1.shape[0]:
            start_y = 0
            end_y = map1.shape[0]
        else:
            start_y = y_offset
            end_y = -int((map1.shape[0] - background.shape[0]) - y_offset)

        if background.shape[1] == map1.shape[1]:
            start_x = 0
            end_x = map1.shape[1]
        else:
            start_x = x_offset
            end_x = -int((map1.shape[1] - background.shape[1]) - x_offset)

        background_copy = background_copy[start_y:end_y, start_x:end_x]
        print('this is the background_copy shape after adjustments,' , background_copy.shape)

    except Exception as e:
        print('Error:', str(e))
        print('IDK WHY WE ARE IN THE EXCEPT BLOCK!!')
        background_copy = background.copy()
    #     fisheye_map_location = fisheye_map_location
    #     print(colored('fisheye correction not available','red'))

    # initialize clicked points
    blank_arena = arena.copy()
    background_data = [background_copy, np.array(([], [])).T]
    arena_data = [[], np.array(([], [])).T]
    cv2.namedWindow('registered background')
    alpha = .7
    colors = [[150, 0, 150], [0, 255, 0]]
    color_array = make_color_array(colors, background.shape)
    use_loaded_transform = False
    make_new_transform_immediately = False
    use_loaded_points = False

    # LOOP OVER TRANSFORM FILES
    file_num = -1;
    transform_files = glob.glob('*transform.npy')
    for file_num, transform_file in enumerate(transform_files[::-1]):

        # USE LOADED TRANSFORM AND SEE IF IT'S GOOD
        loaded_transform = np.load(transform_file)
        M = loaded_transform[0]
        background_data[1] = loaded_transform[1]
        arena_data[1] = loaded_transform[2]

        # registered_background = cv2.warpPerspective(background_copy, M, background.shape)
        registered_background = cv2.warpAffine(background_copy, M, background.shape)
        registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                       * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
        arena_color = (cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB)
                       * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

        overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
        print('Does transform ' + str(file_num+1) + ' / ' + str(len(transform_files)) + ' match this session?')
        print('\'y\' - yes! \'n\' - no. \'q\' - skip examining loaded transforms. \'p\' - update current transform')
        while True:
            cv2.imshow('registered background', overlaid_arenas)
            k = cv2.waitKey(10)
            if  k == ord('n'):
                break
            elif k == ord('y'):
                use_loaded_transform = True
                break
            elif k == ord('q'):
                make_new_transform_immediately = True
                break
            elif k == ord('p'):
                use_loaded_points = True
                break
        if use_loaded_transform:
            update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M]
            break
        elif make_new_transform_immediately or use_loaded_points:
            file_num = len(glob.glob('*transform.npy'))-1
            break

    if not use_loaded_transform:
        if not use_loaded_points:
            print("\n  Select reference points on the 'background image' in the indicated order")

            # initialize clicked point arrays
            background_data = [background_copy, np.array(([], [])).T]
            arena_data = [[], np.array(([], [])).T]

            # add markers to model arena
            for i, point in enumerate(arena_points.astype(np.uint32)):
                arena = cv2.circle(arena, (point[0], point[1]), 3, 255, -1)
                arena = cv2.circle(arena, (point[0], point[1]), 4, 0, 1)
                cv2.putText(arena, str(i+1), tuple(point), 0, .55, 150, thickness=2)

                point = np.reshape(point, (1, 2))
                arena_data[1] = np.concatenate((arena_data[1], point))

            # initialize GUI
            cv2.startWindowThread()
            cv2.namedWindow('background')
          

            print("Type of background_copy:", type(background_copy))
            print("Shape of background_copy:", background_copy.shape)
            # print("Minimum pixel value in background_copy:", background_copy.min())
            print("Maximum pixel value in background_copy:", background_copy.max())

            # Optionally, you can also display the image using imshow to visually inspect it
            cv2.imshow('background_copy', background_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow('background', background_copy)
            cv2.namedWindow('model arena')
            cv2.imshow('model arena', arena)

            # create functions to react to clicked points
            cv2.setMouseCallback('background', select_transform_points, background_data)  # Mouse callback

            while True: # take in clicked points until four points are clicked
                cv2.imshow('background',background_copy)

                number_clicked_points = background_data[1].shape[0]
                if number_clicked_points == len(arena_data[1]):
                    break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        # perform projective transform
        M, inliers = cv2.estimateAffine2D(background_data[1], arena_data[1])
        if M is None:
            print("Transformation estimation failed.")


        # REGISTER BACKGROUND, BE IT WITH LOADED OR CREATED TRANSFORM
        registered_background = cv2.warpAffine(background_copy, M, background.shape[::-1])

        # --------------------------------------------------
        # overlay images
        # --------------------------------------------------
        alpha = .7
        colors = [[150, 0, 150], [0, 255, 0]]
        color_array = make_color_array(colors, background.shape)

        registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                 * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
        arena_color = (cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB)
                       * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

        overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
        cv2.imshow('registered background', overlaid_arenas)

        # --------------------------------------------------
        # initialize GUI for correcting transform
        # --------------------------------------------------
        print("\n  On the 'registered background' pane: Left click model arena --> Right click model background")
        print('  (Model arena is green at bright locations and purple at dark locations)')
        print('\n  Advanced users: use arrow keys and \'wasd\' to fine-tune translation and scale as a final step')
        print('  Crème de la crème: use \'tfgh\' to fine-tune shear\n')
        print('  y: save and use transform')
        print('  r: reset transform')
        update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M, background_copy]
        reset_registration = False

        # create functions to react to additional clicked points
        cv2.setMouseCallback('registered background', additional_transform_points, update_transform_data)

        # take in clicked points until 'q' is pressed
        initial_number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
        M_initial = M
        M_indices = [(0,2),(1,2),(0,0),(1,1),(0,1),(1,0),(2,0),(2,2)]
        # M_indices_meanings = ['x-translate','y-translate','x-scale','y-scale','x->y shear','y->x shear','x perspective','y perspective']
        M_mod_keys = [2424832, 2555904, 2490368, 2621440, ord('w'), ord('a'), ord('s'), ord('d'), ord('f'), ord('t'),
                      ord('g'), ord('h'), ord('j'), ord('i'), ord('k'), ord('l')]
        while True:
            cv2.imshow('registered background',overlaid_arenas)
            cv2.imshow('background', registered_background)
            number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
            update_transform = False
            k = cv2.waitKey(10)
            # If a left and right point are clicked:
            if number_clicked_points[0]>initial_number_clicked_points[0] and number_clicked_points[1]>initial_number_clicked_points[1]:
                initial_number_clicked_points = number_clicked_points
                # update transform and overlay images
                try:
                    M = cv2.estimateRigidTransform(update_transform_data[1], update_transform_data[2],False) #True ~ full transform
                    update_transform = True
                except:
                    continue
            elif k in M_mod_keys: # if an arrow key if pressed
                if k == 2424832: # left arrow - x translate
                    M[M_indices[0]] = M[M_indices[0]] - abs(M_initial[M_indices[0]]) * .005
                elif k == 2555904: # right arrow - x translate
                    M[M_indices[0]] = M[M_indices[0]] + abs(M_initial[M_indices[0]]) * .005
                elif k == 2490368: # up arrow - y translate
                    M[M_indices[1]] = M[M_indices[1]] - abs(M_initial[M_indices[1]]) * .005
                elif k == 2621440: # down arrow - y translate
                    M[M_indices[1]] = M[M_indices[1]] + abs(M_initial[M_indices[1]]) * .005
                elif k == ord('a'): # down arrow - x scale
                    M[M_indices[2]] = M[M_indices[2]] + abs(M_initial[M_indices[2]]) * .005
                elif k == ord('d'): # down arrow - x scale
                    M[M_indices[2]] = M[M_indices[2]] - abs(M_initial[M_indices[2]]) * .005
                elif k == ord('s'): # down arrow - y scale
                    M[M_indices[3]] = M[M_indices[3]] + abs(M_initial[M_indices[3]]) * .005
                elif k == ord('w'): # down arrow - y scale
                    M[M_indices[3]] = M[M_indices[3]] - abs(M_initial[M_indices[3]]) * .005
                elif k == ord('f'): # down arrow - x-y shear
                    M[M_indices[4]] = M[M_indices[4]] - abs(M_initial[M_indices[4]]) * .005
                elif k == ord('h'): # down arrow - x-y shear
                    M[M_indices[4]] = M[M_indices[4]] + abs(M_initial[M_indices[4]]) * .005
                elif k == ord('t'): # down arrow - y-x shear
                    M[M_indices[5]] = M[M_indices[5]] - abs(M_initial[M_indices[5]]) * .005
                elif k == ord('g'): # down arrow - y-x shear
                    M[M_indices[5]] = M[M_indices[5]] + abs(M_initial[M_indices[5]]) * .005

                update_transform = True

            elif  k == ord('r'):
                print(colored('Transformation erased', 'green'))
                update_transform_data[1] = np.array(([],[])).T
                update_transform_data[2] = np.array(([],[])).T
                initial_number_clicked_points = [3,3]
                reset_registration = True
                break
            elif k == ord('q') or k == ord('y'):
                print(colored('\nRegistration completed\n', 'green'))
                break

            if update_transform:
                update_transform_data[3] = M
                # registered_background = cv2.warpPerspective(background_copy, M, background.shape)
                registered_background = cv2.warpAffine(background_copy, M, background.shape[::-1])
                registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                               * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
                overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
                update_transform_data[0] = overlaid_arenas

    cv2.destroyAllWindows()

    if reset_registration: register_arena(background, fisheye_map_location, y_offset, x_offset, show_arena = show_arena)

    return [M, update_transform_data[1], update_transform_data[2], fisheye_map_location]

# mouse callback function I
def select_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 3, 255, -1)
        data[0] = cv2.circle(data[0], (x, y), 4, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[1] = np.concatenate((data[1], clicks))

# mouse callback function II
def additional_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_RBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (200,0,0), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        M_inverse = cv2.invertAffineTransform(data[3])
        transformed_clicks = np.matmul(np.append(M_inverse,np.zeros((1,3)),0), [x, y, 1])


        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 2, (0, 0, 200), -1)
        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 3, 0, 1)

        clicks = np.reshape(transformed_clicks[0:2],(1,2))
        data[1] = np.concatenate((data[1], clicks))
    elif event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (0,200,200), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[2] = np.concatenate((data[2], clicks))

def make_color_array(colors, image_size):
    color_array = np.zeros((image_size[0],image_size[1], 3, len(colors)))  # create coloring arrays
    for c in range(len(colors)):
        for i in range(3):  # B, G, R
            color_array[:, :, i, c] = np.ones((image_size[0],image_size[1])) * colors[c][i] / sum(
                colors[c])
    return color_array





