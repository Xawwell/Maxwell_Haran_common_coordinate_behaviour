import cv2
import numpy as np
import scipy.misc
from termcolor import colored
from tqdm import tqdm
import glob
import os
import traceback



# =================================================================================
#              CREATE MODEL ARENA FOR COMMON COORDINATE BEHAVIOUR
# =================================================================================
def model_arena(size, show_arena=True):
    # Initialize the model arena image
    arena = np.zeros((1000, 1000), dtype=np.uint8)

    # Add the circular arena
    center = (500, 500)
    radius = 460
    cv2.circle(arena, center, radius, 255, -1)

    # Shelter dimensions
    shelter_width = 240
    shelter_height = 70

    # List of top-left and bottom-right coordinates for each of the shelters
    # Adjust these coordinates as needed for the desired locations
    shelters = [
        [(400, 25), (600, 75)],     # Top
        [(25, 750), (225, 800)],   # Bottom-Left
        [(775, 750), (975, 800)]    # Bottom-Right
    ]

    for top_left, bottom_right in shelters:
        cv2.rectangle(arena, top_left, bottom_right, 128, -1)

    # For registration, convert points to the resized dimensions
    points = [((tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2) for tl, br in shelters]
    points_resized = [(pt[0] * size[0] / 1000, pt[1] * size[1] / 1000) for pt in points]

    

    # Resize the arena to the size of your image
    model_arena = cv2.resize(arena, size)

    if show_arena:
        print('we are in show_arena')
        cv2.imshow('model arena looks like this',model_arena)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return model_arena, points

# =================================================================================
#              REGISTER A FRAME TO THE COMMON COORDINATE FRAMEWORK
# =================================================================================
def register_frame(frame, registration, x_offset, y_offset, map1, map2):
    '''
    Register a grayscale image.
    '''
    
    # Check if frame is not empty
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        print("Error: Empty frame passed to the function!")
        return frame

    # Make sure the frame is 1-D (grayscale)
    frame_register = frame[:, :, 0]
    
    # print("Original frame shape:", frame_register.shape)
    
    # Do the fisheye correction if applicable
    if registration and registration[3]:
        try:
            # Add borders to the frame for the fisheye correction
            frame_register = cv2.copyMakeBorder(frame_register, 
                                                y_offset, 
                                                int((map1.shape[0] - frame.shape[0]) - y_offset),
                                                x_offset, 
                                                int((map1.shape[1] - frame.shape[1]) - x_offset), 
                                                cv2.BORDER_CONSTANT, value=0)
            
            # print("Frame shape after adding borders:", frame_register.shape)

            # Apply the fisheye correction
            frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR, 
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # print("Frame shape after fisheye correction:", frame_register.shape)

            # # Crop the frame to get back to its original size
            # frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
            #                                 x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]

            # print("NO MORE cropping:", frame_register.shape)

        except Exception as e:
            print("Error during fisheye correction:", e)
    
    # Apply the affine transformation from the registration
    try:
        frame = cv2.warpAffine(frame_register, registration[0], frame.shape[0:2])
    except Exception as e:
        print("Error during affine transformation:", e)

    return frame


# =================================================================================
#              GET BACKGROUND IMAGE FROM VIDEO
# =================================================================================
"""
 this function processes a video, skips frames to reduce processing load, and averages the frames to generate a background image
"""
def get_background(vidpath, start_frame = 1000, avg_over = 100):
    """ extract background: average over frames of video """

    vid = cv2.VideoCapture(vidpath)

    # initialize the video
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background = np.zeros((height, width))
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
    if registration[3]: maps = np.load(registration[3]); map1 = maps[:, :, 0:2]; map2 = maps[:, :, 2] * 0;print(colored('Fisheye correction is RUNNING!', 'yellow'))
    else: map1 = None; map2 = None; print("map1 is", map1, "and map2 is", map2);print(colored('Fisheye correction unavailable', 'red'))

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

            # display and save the frame
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
# IMAGE REGISTRATION GUI (the following 4 functions are a bit messy and complicated)
# ==================================================================================
def register_arena(background, fisheye_map_location, y_offset, x_offset, show_arena = False):

    # create model arena and background
    arena, arena_points = model_arena(background.shape[::-1], show_arena)

    # load the fisheye correction
    try:
        maps = np.load(fisheye_map_location)
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2] * 0

        # Printing initial details
        print("map1.shape[0]:", map1.shape[0])
        print("map2.shape[0]:", map2.shape[0])
        print("background.shape[0]:", background.shape[0])
        print("y_offset:", y_offset)
        print("top border:", y_offset)
        print("bottom border:", int((map1.shape[0] - background.shape[0]) - y_offset))
        print("x_offset:", x_offset)
        print("left border:", x_offset)
        print("right border:", int((map1.shape[1] - background.shape[1]) - x_offset))

        # Initial check before starting operations
        print("Starting image operations...")  # <-- Added print

        # Border addition
        print("Before adding borders:", background.shape)
        background_copy = cv2.copyMakeBorder(
            background, 
            y_offset, 
            int((map1.shape[0] - background.shape[0]) - y_offset),
            x_offset, 
            int((map1.shape[1] - background.shape[1]) - x_offset),
            cv2.BORDER_CONSTANT, 
            value=0
        )
        print(f'After adding borders: {background_copy.shape}')

        # Remapping
        background_copy = cv2.remap(
            background_copy, 
            map1, 
            map2, 
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=0
        )
        print(f'After remapping: {background_copy.shape}')

        # # Cropping
        # y_end_index = -int((map1.shape[0] - background.shape[0]) - y_offset) if y_offset != 0 or (map1.shape[0] - background.shape[0]) != 0 else None
        # x_end_index = -int((map1.shape[1] - background.shape[1]) - x_offset) if x_offset != 0 or (map1.shape[1] - background.shape[1]) != 0 else None
        # background_copy = background_copy[y_offset:y_end_index, x_offset:x_end_index]

        # print(f'After cropping: {background_copy.shape}')
            
        # Final check
        assert background_copy.shape[0] > 0 and background_copy.shape[1] > 0, "Invalid image dimensions"

    except Exception as e:
        print("Exception occurred!")
        traceback.print_exc()  # This will print the detailed exception information
        background_copy = background.copy()
        fisheye_map_location = ''
        print('fisheye correction not available')

    # Initialize clicked points
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

            # add 1-2-3 markers to model arena
            # for i, point in enumerate(arena_points.astype(np.uint32)):
            for i, point in enumerate(np.array(arena_points).astype(np.uint32)):
                arena = cv2.circle(arena, (point[0], point[1]), 3, 255, -1)
                arena = cv2.circle(arena, (point[0], point[1]), 4, 0, 1)
                cv2.putText(arena, str(i+1), tuple(point), 0, .55, 150, thickness=2)

                point = np.reshape(point, (1, 2))
                arena_data[1] = np.concatenate((arena_data[1], point))

            # initialize GUI
            cv2.startWindowThread()
            cv2.namedWindow('background')
            print("Original background shape:", background.shape)

            print("this is background copy SHAPE:", background_copy.shape)
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
            # Handle the failure accordingly




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
    #What it returns is essentially the regisration variable!

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





