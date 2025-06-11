import os

root_dir = './data'

def sample_transformation(root_vid_dir):
    original = "videos_160"
    augment_dir = os.path.join(root_vid_dir, "augmented_data")
    all_vid_dirs = []
    all_vid_dirs.append(os.path.join(root_vid_dir, original))

    for dirpath, dirnames, _ in os.walk(augment_dir):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            all_vid_dirs.append(full_path)
    return all_vid_dirs
folders = sample_transformation(root_dir)
print(folders)