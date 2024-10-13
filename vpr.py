import os
import cv2
import torch
import faiss
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re



def generate_mapping_data(image_width):
    in_size = [image_width, int(image_width * 3 / 4)]
    edge = int(in_size[0] / 4)

    out_pix = np.zeros((in_size[1], in_size[0], 2), dtype="f4")
    xyz = np.zeros((in_size[1] * in_size[0] // 2, 3), dtype="f4")
    vals = np.zeros((in_size[1] * in_size[0] // 2, 3), dtype="i4")

    start, end = 0, 0
    rng_1 = np.arange(0, edge * 3)
    rng_2 = np.arange(edge, edge * 2)
    for i in range(in_size[0]):
        face = i // edge
        rng = rng_1 if face == 2 else rng_2

        end += len(rng)
        vals[start:end, 0] = rng
        vals[start:end, 1] = i
        vals[start:end, 2] = face
        start = end

    j, i, face = vals.T
    face[j < edge] = 4
    face[j >= 2 * edge] = 5

    a = 2.0 * i / edge
    b = 2.0 * j / edge
    one_arr = np.ones(len(a))
    for k in range(6):
        face_idx = face == k
        one_arr_idx = one_arr[face_idx]
        a_idx = a[face_idx]
        b_idx = b[face_idx]

        if k == 0:
            vals_to_use = [-one_arr_idx, 1.0 - a_idx, 3.0 - b_idx]
        elif k == 1:
            vals_to_use = [a_idx - 3.0, -one_arr_idx, 3.0 - b_idx]
        elif k == 2:
            vals_to_use = [one_arr_idx, a_idx - 5.0, 3.0 - b_idx]
        elif k == 3:
            vals_to_use = [7.0 - a_idx, one_arr_idx, 3.0 - b_idx]
        elif k == 4:
            vals_to_use = [b_idx - 1.0, a_idx - 5.0, one_arr_idx]
        elif k == 5:
            vals_to_use = [5.0 - b_idx, a_idx - 5.0, -one_arr_idx]

        xyz[face_idx] = np.array(vals_to_use).T

    x, y, z = xyz.T
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(z, r)

    uf = (2.0 * edge * (theta + np.pi) / np.pi) % in_size[0]
    uf[uf == in_size[0]] = 0.0
    vf = (2.0 * edge * (np.pi / 2 - phi) / np.pi)

    out_pix[j, i, 0] = vf
    out_pix[j, i, 1] = uf

    map_x_32 = out_pix[:, :, 1]
    map_y_32 = out_pix[:, :, 0]
    return map_x_32, map_y_32


TOP_K = 10  
BATCH_SIZE = 3
FRAME_DIFF_THRESHOLD = 2
FRAME_CLOSENESS_THRESHOLD = 3
IMG_WIDTH = 1600


model = torch.hub.load("gmberton/eigenplaces", "get_trained_model", backbone="ResNet50", fc_output_dim=2048).to('cuda')
model = model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_frame_number(image_path):
    """Extracts the frame number from image filename assuming 'frameXXXX.jpg' format."""
    match = re.search(r'frame(\d+)\.jpg', os.path.basename(image_path))
    return int(match.group(1)) if match else None

class ImageFolderDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.jpg')]
        self.transform = transform
        self.map_x_32, self.map_y_32 = generate_mapping_data(7680)
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = cv2.imread(image_path)
        img = cv2.remap(img, self.map_x_32, self.map_y_32, cv2.INTER_LANCZOS4)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        img_tensor = self.transform(img_rgb) if self.transform else img_rgb
        return img_tensor, image_path

# Feature Extraction
def extract_features_batch(dataloader, model, device):
    """Extracts features in batches from images using the model."""
    all_features, all_image_paths = [], []
    for images, image_paths in tqdm(dataloader, desc="Extracting features in batches"):
        images = images.to(device)
        with torch.no_grad():
            batch_features = model(images).cpu().numpy()  # Extract features
        all_features.append(batch_features)
        all_image_paths.extend(image_paths)
    return np.vstack(all_features), all_image_paths

# FAISS Index Creation
def build_faiss_index(image_folder, model, device, fc_output_dim, batch_size):
    """Builds FAISS index from images in the folder."""
    dataset = ImageFolderDataset(image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_features, all_image_paths = extract_features_batch(dataloader, model, device)
    
    # Initialize and populate FAISS index
    faiss_index = faiss.IndexFlatL2(fc_output_dim)
    faiss_index.add(all_features)
    return faiss_index, all_image_paths

# Search & Filtering
def search_top_k_matches(query_images, query_image_paths, faiss_index, model, device, image_paths, top_k, frame_diff_threshold, frame_closeness_threshold):
    """Search for top K matches and filter results based on frame distance."""
    query_images = query_images.to(device)
    with torch.no_grad():
        query_features = model(query_images).cpu().numpy()

    all_filtered_matches = []
    for idx, query_feature in enumerate(query_features):
        query_frame_number = extract_frame_number(query_image_paths[idx])
        distances, indices = faiss_index.search(query_feature.reshape(1, -1), top_k * 5)
        filtered_matches, added_frame_numbers = [], set()

        for rank, i in enumerate(indices[0]):
            if rank >= len(distances[0]):
                break
            candidate_image_path = image_paths[i]
            candidate_frame_number = extract_frame_number(candidate_image_path)
            
            # Apply filtering based on frame number distance and closeness threshold
            if abs(candidate_frame_number - query_frame_number) > frame_diff_threshold and not any(
                abs(candidate_frame_number - added_frame) <= frame_closeness_threshold for added_frame in added_frame_numbers
            ):
                filtered_matches.append((candidate_image_path, float(distances[0][rank])))
                added_frame_numbers.add(candidate_frame_number)
            
            if len(filtered_matches) == top_k:
                break
        
        all_filtered_matches.append(filtered_matches)
    
    return all_filtered_matches

# Result Handling
def save_results_to_json(results, output_file):
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

def create_image_pairs_file(results, output_txt_file):
    """Create a text file with image pairs."""
    with open(output_txt_file, 'w') as f:
        for result in results:
            query_image = result['query_image']
            for match in result['matches']:
                matched_image = match['image_path']
                f.write(f"{os.path.basename(query_image)} {os.path.basename(matched_image)}\n")
    print(f"Image pairs written to {output_txt_file}")

# Main Processing
def process_images(image_folder, model, device, fc_output_dim, batch_size, output_json="results.json", output_txt="image_pairs.txt"):
    """Process all images in the folder and save the results."""
    faiss_index, image_paths = build_faiss_index(image_folder, model, device, fc_output_dim, batch_size)
    
    # Create query dataloader
    query_dataset = ImageFolderDataset(image_folder, transform=transform)
    query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    results = []
    for query_images, query_image_paths in tqdm(query_dataloader, desc="Processing queries"):
        top_matches = search_top_k_matches(query_images, query_image_paths, faiss_index, model, device, image_paths, TOP_K, FRAME_DIFF_THRESHOLD, FRAME_CLOSENESS_THRESHOLD)
        for query_image_path, matches in zip(query_image_paths, top_matches):
            results.append({
                'query_image': query_image_path,
                'matches': [{'image_path': match, 'distance': dist} for match, dist in matches]
            })
    
    # save_results_to_json(results, output_json)
    create_image_pairs_file(results, output_txt)
    return results

# Main Execution
if __name__ == "__main__":
    image_folder = "/home/luffy/data/VID_20240622_155518_00_007_processed_0_2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fc_output_dim = 2048  # Adjust based on model

    # Process images and store results
    process_images(image_folder, model, device, fc_output_dim, BATCH_SIZE, output_json="results_optimized.json", output_txt="image_pairs_1.txt")
