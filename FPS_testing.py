import cv2

# Define the video sources
video_sources = {
    'RAW video': 'G:\\Files_from_npx_bonzai_pc\\bins_and_meta\\sub-003_mouseid-1119984_behaviour_2023_06_13T15_02_09\\cam.avi',
    'Fisheye Corrected video': "G:\\Maxwell\\code\\WORKING_Maxwell_Common-Coordinate-Behaviour-master\\Common-Coordinate-Behaviour-master\\clips\\lossless\\Maxwell_mouseid-1119984_audiotory-0'.avi",
    'DLC output video': 'G:\\Files_from_npx_bonzai_pc\\bins_and_meta\\sub-003_mouseid-1119984_behaviour_2023_06_13T15_02_09\\camDLC_resnet50_Maxwelll_HaranOct20shuffle1_100000_labeled.mp4'
}


# Function to get video properties
def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bitrate = int(cap.get(cv2.CAP_PROP_BITRATE))

    # Convert codec to four characters
    codec_fourcc = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

    cap.release()

    return {
        'frame_count': frame_count,
        'frame_rate': frame_rate,
        'codec_fourcc': codec_fourcc,
        'width': width,
        'height': height,
        'bitrate': bitrate,
    }

# Iterate through the video sources and get properties
for source_name, source_path in video_sources.items():
    video_properties = get_video_properties(source_path)
    print(f"{source_name} Properties:")
    print(f"  Frame Count: {video_properties['frame_count']}")
    print(f"  Frame Rate: {video_properties['frame_rate']} FPS")
    print(f"  Codec FourCC: {video_properties['codec_fourcc']}")
    print(f"  Resolution: {video_properties['width']}x{video_properties['height']}")
    print(f"  Bitrate: {video_properties['bitrate']} bps")
    # Add additional checks here based on the provided information.
    print()