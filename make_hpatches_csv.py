import os
import csv
import shutil

def read_homography(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        H = [float(num) for line in lines for num in line.strip().split()]
        assert len(H) == 9, f"Invalid homography in {file_path}"
        return H

def get_image_shape(image_path):
    from PIL import Image
    with Image.open(image_path) as img:
        return img.height, img.width  # H, W

def generate_hpatches_csv(root_dir, output_folder):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"Removed existing output folder: {output_folder}")
    os.makedirs(output_folder)
    
    output_files = {
        i: open(os.path.join(output_folder, f"hpatches_1_{i}.csv"), 'w', newline='') 
        for i in range(2, 7)
    }
    writers = {}
    
    for i in range(2, 7):
        writer = csv.writer(output_files[i])
        writer.writerow(["obj", "im1", "im2", "Him", "Wim",
                         "H11", "H12", "H13",
                         "H21", "H22", "H23",
                         "H31", "H32", "H33"])
        writers[i] = writer
    
    for seq_name in os.listdir(root_dir):
        seq_path = os.path.join(root_dir, seq_name)
        if not os.path.isdir(seq_path):
            continue
        
        # Get image dimensions from im1
        im1_path = os.path.join(seq_path, "1.ppm")
        try:
            Him, Wim = get_image_shape(im1_path)
        except Exception as e:
            print(f"Skipping {seq_name}: {e}")
            continue
        
        for i in range(2, 7):
            H_file = os.path.join(seq_path, f"H_1_{i}")
            if not os.path.isfile(H_file):
                print(f"Missing: {H_file}")
                continue
            H = read_homography(H_file)
            row = [seq_name, 1, i, float(Him), float(Wim)] + H
            writers[i].writerow(row)
    
    for f in output_files.values():
        f.close()

# Example usage:
# generate_hpatches_csv("/path/to/hpatches-sequences-release", "/path/to/output/csv")
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate HPatches CSV files")
    parser.add_argument("root_dir", type=str, help="Path to the HPatches sequences directory")
    parser.add_argument("output_folder", type=str, help="Path to the output folder for CSV files")
    
    args = parser.parse_args()
    
    generate_hpatches_csv(args.root_dir, args.output_folder)