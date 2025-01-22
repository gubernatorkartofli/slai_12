import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from matplotlib.gridspec import GridSpec
import traceback
import glob
from scipy.spatial.distance import cityblock
import time

# Wybór urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ładowanie wcześniej wytrenowanego modelu YOLO
yolo_model = YOLO(r"slai/best.pt")

# Ładowanie modelu ResNet50 do generowania embeddingów
resnet_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
resnet_model.to(device)
resnet_model.eval()

# Przekształcenia obrazu
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Funkcja konwertująca obrazy do PNG
def convert_to_png(folder):
    for ext in ['jpg', 'jpeg']:
        for image_path in glob.glob(os.path.join(folder, f'*.{ext}')):
            img = Image.open(image_path)
            png_image_path = os.path.splitext(image_path)[0] + '.png'
            img.save(png_image_path, 'PNG')
            os.remove(image_path)
            print(f"Converted {image_path} to {png_image_path}")

def draw_segmentation_masks(image, results):
    if results[0].masks is None:
        print("No segmentation masks found in the detection results.")
        return image
    img_array = np.array(image)
    result_image = results[0].plot()  # YOLO rysuje maski
    return Image.fromarray(result_image)  # Konwersja na format PIL

# Funkcja generująca embedding z detekcji na podstawie maski
def generate_embeddings_for_all_detections(image, results, conf_threshold=0.4):
    embeddings = []
    for detection in results[0].boxes:
        confidence = detection.conf[0].item()
        if confidence >= conf_threshold:
            detection_box = detection.xyxy[0].tolist()
            class_index = int(detection.cls[0].item())
            embedding = get_embedding_from_detection(image, detection_box)
            if embedding is not None:
                embeddings.append({
                    'embedding': embedding,
                    'confidence': confidence,
                    'box': detection_box,
                    'class_index': class_index
                })
    return embeddings

def cut_out_objects(image, results, scale_factor=1.0):
    cropped_objects = []
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    if results[0].masks is not None:
        for mask in results[0].masks.data:
            mask_array = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_resized = cv2.resize(mask_array, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask_rgb = np.stack([mask_resized] * 3, axis=-1)
            img_masked = cv2.bitwise_and(img_array, mask_rgb)
            img_masked_rgba = np.dstack([img_masked, mask_resized])
            if scale_factor != 1.0:
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    roi = img_masked_rgba[y:y + h, x:x + w]
                    new_width = int(w * scale_factor)
                    new_height = int(h * scale_factor)
                    x_pad = max(0, (new_width - w) // 2)
                    y_pad = max(0, (new_height - h) // 2)

                    padded_roi = cv2.copyMakeBorder(
                        roi,
                        y_pad, y_pad, x_pad, x_pad,
                        borderType=cv2.BORDER_CONSTANT,
                        value=(0, 0, 0, 0)
                    )
                    cropped_objects.append(padded_roi)
            else:
                cropped_objects.append(img_masked_rgba)
    return cropped_objects

def resize_to_fit(image, axis, bbox=None):
    """
    Przeskalowuje obraz, aby idealnie pasował do przestrzeni subplotu, zachowując proporcje.
    Opcjonalnie używa bounding boxa do dokładniejszego skalowania.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Rozmiary osi w pikselach
    bbox_display = axis.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    display_width, display_height = bbox_display.width * plt.gcf().dpi, bbox_display.height * plt.gcf().dpi

    # Jeśli podano bounding box, wyliczamy wymiary docelowe na podstawie proporcji
    if bbox:
        bbox_width = bbox[2] - bbox[0]  # Szerokość
        bbox_height = bbox[3] - bbox[1]  # Wysokość
        scale_w = display_width / bbox_width
        scale_h = display_height / bbox_height
    else:
        # Rozmiary obrazu
        img_width, img_height = image.size
        scale_w = display_width / img_width
        scale_h = display_height / img_height

    # Skalowanie obrazu, aby pasował do subplotu
    scale = max(scale_w, scale_h)  # Dopasowanie do najmniejszego wymiaru

    new_width = int(image.size[0] * scale)
    new_height = int(image.size[1] * scale)
    resized_image = image.resize((new_width, new_height))

    # Dodanie marginesów, aby wypełnić cały subplot
    background = Image.new("RGBA", (int(display_width), int(display_height)), (255, 255, 255, 0))
    offset = ((background.size[0] - resized_image.size[0]) // 2,
              (background.size[1] - resized_image.size[1]) // 2)
    background.paste(resized_image, offset)

    return background

def get_embedding_from_detection(image, detection_box):
    x1, y1, x2, y2 = map(int, detection_box)
    cropped_image = image.crop((x1, y1, x2, y2))
    if cropped_image.mode != 'RGB':
        cropped_image = cropped_image.convert('RGB')
    input_tensor = preprocess(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet_model(input_tensor).squeeze().cpu().numpy()
    return embedding  # Only return the embedding


def generate_embeddings_for_image(image, results, conf_threshold=0.1):
    embeddings = []
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_indices = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        for mask, box, class_index, confidence in zip(masks, boxes, class_indices, confidences):
            if confidence >= conf_threshold:
                embedding = get_embedding_from_detection(image, box)  # Only one value returned
                embeddings.append({
                    'embedding': embedding,
                    'class_index': int(class_index),
                    'confidence': confidence
                })

    return embeddings


# Funkcja wczytująca embeddingi z pliku
def load_embeddings(embeddings_file):
    embeddings_data = np.load(embeddings_file, allow_pickle=True)
    embeddings = embeddings_data['embeddings']
    filenames = embeddings_data['filenames']
    class_indices = embeddings_data['class_indices']
    return [{'embedding': emb, 'filename': filename, 'class_index': class_index} 
            for emb, filename, class_index in zip(embeddings, filenames, class_indices)]

# Funkcja generująca i zapisująca embeddingi do pliku
import os
import numpy as np
from PIL import Image

def generate_and_save_embeddings(image_folder, output_file):
    embeddings = []
    filenames = []
    class_indices = []

    # Check if the embeddings file already exists
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping generation.")
        return

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path)
            results = yolo_model(image_path)

            image_embeddings = generate_embeddings_for_image(image, results)
            for emb in image_embeddings:
                embeddings.append(emb['embedding'])
                filenames.append(filename)
                class_indices.append(emb['class_index'])

    # Save embeddings, filenames, and class indices to the npz file
    np.savez(output_file, embeddings=embeddings, filenames=filenames, class_indices=class_indices)
    print(f"Embeddings saved to {output_file}")


def normalize_cityblock(embedding1, embedding2, max_distance):
    if embedding1.shape != embedding2.shape:
        raise ValueError(f"Embedding shapes do not match: {embedding1.shape} vs {embedding2.shape}")
    distance = cityblock(embedding1, embedding2)
    
    if max_distance == 0:
        return 0
    
    return distance / max_distance
def calculate_similarity(cropped_embedding, database_embeddings, class_index, similarity_threshold=0.4):
    all_embeddings = np.array([embedding['embedding'] for embedding in database_embeddings])
    all_class_indices = np.array([embedding['class_index'] for embedding in database_embeddings])

    max_distance = np.max([cityblock(emb1, emb2) for emb1 in all_embeddings for emb2 in all_embeddings])
    if max_distance == 0:
        raise ValueError("Max distance between embeddings is zero, cannot normalize.")

    # Calculate cityblock distances between the cropped embedding and the database embeddings
    distances = np.array([normalize_cityblock(cropped_embedding, emb, max_distance) for emb in all_embeddings])
    
    # Filtracja po klasach
    same_class_indices = np.where(all_class_indices == class_index)[0]
    same_class_distances = distances[same_class_indices]

    if same_class_distances.size > 0 and np.min(same_class_distances) < similarity_threshold:
        most_similar_index = same_class_indices[np.argmin(same_class_distances)]
        return distances[most_similar_index], most_similar_index

    # Najbardziej podobny ogólny
    closest_index = np.argmin(distances)
    return distances[closest_index], closest_index


# Funkcja porównująca wykryte obiekty z bazą ubrań
def display_comparison(test_image, results, database_embeddings, input_folder, similarity_threshold=0.5, output_folder="output_images"):
    # Generating embeddings for all detections in the test image
    embeddings = generate_embeddings_for_all_detections(test_image, results)
    
    masked_test = draw_segmentation_masks(test_image, results)
    
    num_rows = len(embeddings)
    fig_compare = plt.figure(figsize=(15, 5 * num_rows))
    gs = GridSpec(num_rows, 3, width_ratios=[2, 1, 1])

    ax_test_image = fig_compare.add_subplot(gs[:, 0])
    ax_test_image.imshow(masked_test)
    ax_test_image.axis('off')
    ax_test_image.set_title("Obraz testowy")

    most_similar_result = None
    highest_similarity = -1

    cropped_image_folder = os.path.join(output_folder, "cropped_images")
    if not os.path.exists(cropped_image_folder):
        os.makedirs(cropped_image_folder)

    # Looping over detected embeddings
    for i, detection in enumerate(embeddings):
        cropped_embedding = detection['embedding']
        class_index = int(detection['class_index'])
        confidence = detection['confidence']
        
        # Calculate similarity between the cropped object and database embeddings
        similarity, matched_index = calculate_similarity(cropped_embedding, database_embeddings, class_index, similarity_threshold)

        # Track most similar result
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_result = (i, detection, matched_index, similarity)

        # Create subplot for the detected object
        ax_detection = fig_compare.add_subplot(gs[i, 1])
        cropped_objects = cut_out_objects(test_image, results)
        if i < len(cropped_objects):
            resized_image = resize_to_fit(cropped_objects[i], ax_detection)
            ax_detection.imshow(resized_image)
        else:
            ax_detection.text(0.5, 0.5, "No objects", ha='center', va='center', fontsize=12)
        ax_detection.set_title(f"Wykryta\nklasa: {results[0].names[class_index]}\nPewność: {confidence:.2f}")
        ax_detection.axis('off')

        # Show the most similar match
        ax_match = fig_compare.add_subplot(gs[i, 2])
        matched_image_path = os.path.join(input_folder, database_embeddings[matched_index]['filename'])
        matched_image = Image.open(matched_image_path)
        results_matched = yolo_model(matched_image)
        cropped_matched_image = cut_out_objects(matched_image, results_matched)

        if cropped_matched_image:
            ax_match.imshow(resize_to_fit(cropped_matched_image[0], ax_match))
        else:
            ax_match.text(0.5, 0.5, "No objects", ha='center', va='center', fontsize=12)
        ax_match.set_title(f"Najbardziej podobne:\nPodobieństwo: {similarity:.2f}")
        ax_match.axis('off')

        # Save the cropped test image to a file with a timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cropped_test_image_path = os.path.join(cropped_image_folder, f"cropped_test_image_{timestamp}_{i}.png")
        Image.fromarray(cropped_objects[i]).save(cropped_test_image_path)
    # Tight layout and display the result
    plt.tight_layout(pad=3.0)
    plt.show()



def main():
    input_folder = r"reszta_ubran\Baza_ubrań"  # Możliwość zmiany na inny folder
    try:
        # Convert JPG and JPEG images to PNG
        convert_to_png(input_folder)
        # Testowy obraz
        test_image_path = r"slai\refs\magda.png"
        test_image = Image.open(test_image_path)
        results = yolo_model(test_image)
        embeddings_file = r"slai\embeddings\embeddings_bare.npz"
        output_folder = r"slai\reszta_ubran\embeddings"
        # Update embeddings
        try:
            database_embeddings = load_embeddings(embeddings_file)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Error loading embeddings: {e}. Regenerating embeddings...")
            generate_and_save_embeddings(input_folder, embeddings_file)
            database_embeddings = load_embeddings(embeddings_file)
        
        display_comparison(test_image, results, database_embeddings, input_folder)#Ten przykład bez wizualizacji w Gradio 

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
