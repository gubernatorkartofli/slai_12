import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from matplotlib.gridspec import GridSpec
import numpy as np
import cv2
import traceback
from sklearn.metrics.pairwise import cosine_similarity
from gradio_client import Client, handle_file
from datetime import datetime
import time
import glob
from scipy.spatial.distance import cityblock
# Ścieżki i urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

yolo_model_path = os.path.join("slai\best.pt")
custom_model_path = os.path.join("slai\best_model.pth")
input_folder = r"Baza_ubrań"
output_folder = r"embeddings"
embeddings_file = os.path.join(output_folder, 'embeddings.npz')
cropped_image_folder = os.path.join(os.getcwd(), "cropped_images")

# Gradio 
try:
    hf_token = os.getenv("HUGGINGFACE_TOKEN")#Wymaga dodania tokenu HuggingFace
    client = Client("zhengchong/CatVTON", hf_token=hf_token) if hf_token else None
except:
    client = None
    print("Gradio client not available - continuing without virtual try-on")

# Dodatkowe foldery
if not os.path.exists(cropped_image_folder):
    os.makedirs(cropped_image_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Ładowanie YOLO i ResNet
yolo_model = YOLO(yolo_model_path)

def load_custom_resnet(model_path):
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(in_features=2048, out_features=128)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

resnet_model = load_custom_resnet(custom_model_path)

# Przekształcenia obrazu
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def convert_to_png(folder):#Gradio wymaga obrazów w formacie PNG
    for ext in ['jpg', 'jpeg']:
        for image_path in glob.glob(os.path.join(folder, f'*.{ext}')):
            img = Image.open(image_path)
            png_image_path = os.path.splitext(image_path)[0] + '.png'
            img.save(png_image_path, 'PNG')
            os.remove(image_path)
            print(f"Converted {image_path} to {png_image_path}")
# Wycinanie obiektów na podstawie detekcji YOLO
def cut_out_objects(image, results, scale_factor=1.1):
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

def resize_to_fit(image, axis, bbox=None):#Kosmetyczne powiększenie obrazu
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

def draw_segmentation_masks(image, results):
    if results[0].masks is None:
        print("No segmentation masks found in the detection results.")
        return image
    img_array = np.array(image)
    result_image = results[0].plot()  # YOLO rysuje maski
    return Image.fromarray(result_image)  # Konwersja na format PIL

def get_embedding_from_detection(image, detection_box):
    x1, y1, x2, y2 = map(int, detection_box)
    cropped_image = image.crop((x1, y1, x2, y2))
    if cropped_image.mode != 'RGB':
        cropped_image = cropped_image.convert('RGB')
    input_tensor = preprocess(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet_model(input_tensor).squeeze().cpu().numpy()
    return embedding

# Generowanie embeddingów dla wszystkich detekcji
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

# Generowanie i zapisywanie embeddingów
def generate_and_save_embeddings(input_folder, output_folder):
    embeddings = []
    filenames = []
    class_indices = []

    os.makedirs(output_folder, exist_ok=True)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)

        if not os.path.isfile(img_path):
            continue

        try:
            image = Image.open(img_path)
            results = yolo_model(image)
            detections = generate_embeddings_for_all_detections(image, results)

            for detection in detections:
                embeddings.append(detection['embedding'])
                filenames.append(img_name)
                class_indices.append(detection['class_index'])

        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            continue

    output_path = os.path.join(output_folder, 'embeddings.npz')
    np.savez_compressed(
        output_path,
        embeddings=np.array(embeddings),
        filenames=np.array(filenames),
        class_indices=np.array(class_indices)
    )
    print(f"Embeddings saved to {output_path}")

# Ładowanie embeddingów
def load_embeddings(embeddings_file):
    embeddings_data = np.load(embeddings_file, allow_pickle=True)
    embeddings = embeddings_data['embeddings']
    filenames = embeddings_data['filenames']
    class_indices = embeddings_data['class_indices']
    return [{'embedding': emb, 'filename': filename, 'class_index': class_index} 
            for emb, filename, class_index in zip(embeddings, filenames, class_indices)]

def normalize_cityblock(embedding1, embedding2, max_distance):
    """
    Normalizacja dystansu Manhattan  
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError(f"Embedding shapes do not match: {embedding1.shape} vs {embedding2.shape}")
    distance = cityblock(embedding1, embedding2)
    
    # Nie dzielimy przez zero :3
    if max_distance == 0:
        return 0
    
    return distance / max_distance

def calculate_similarity(cropped_embedding, database_embeddings, class_index, similarity_threshold=0.4):
    all_embeddings = np.array([embedding['embedding'] for embedding in database_embeddings])
    all_class_indices = np.array([embedding['class_index'] for embedding in database_embeddings])

    #Maksymalna odległość między embeddingami  w bazie
    max_distance = np.max([cityblock(emb1, emb2) for emb1 in all_embeddings for emb2 in all_embeddings])
    # Check if max_distance is valid and avoid division by zero
    if max_distance == 0:
        raise ValueError("Max distance between embeddings is zero, cannot normalize.")
    # Obliczanie dystansu ,,podobieństwa"
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



def display_comparison(test_image, results, database_embeddings, input_folder, person_image_path, similarity_threshold=0.5, output_folder="output_images"):
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

    for i, detection in enumerate(embeddings):
        cropped_embedding = detection['embedding']
        class_index = int(detection['class_index'])
        confidence = detection['confidence']
        similarity, matched_index = calculate_similarity(cropped_embedding, database_embeddings, class_index, similarity_threshold)

        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_result = (i, detection, matched_index, similarity)

        ax_detection = fig_compare.add_subplot(gs[i, 1])
        cropped_objects = cut_out_objects(test_image, results)
        if i < len(cropped_objects):
            resized_image = resize_to_fit(cropped_objects[i], ax_detection)
            ax_detection.imshow(resized_image)
        else:
            ax_detection.text(0.5, 0.5, "Brak obiektów", ha='center', va='center', fontsize=12)
        ax_detection.set_title(f"Wykryta\nklasa: {results[0].names[class_index]}\nPewność: {confidence:.2f}")
        ax_detection.axis('off')

        ax_match = fig_compare.add_subplot(gs[i, 2])
        matched_image_path = os.path.join(input_folder, database_embeddings[matched_index]['filename'])
        matched_image = Image.open(matched_image_path)
        results_matched = yolo_model(matched_image)
        cropped_matched_image = cut_out_objects(matched_image, results_matched)

        if cropped_matched_image:
            ax_match.imshow(resize_to_fit(cropped_matched_image[0], ax_match))
        else:
            ax_match.text(0.5, 0.5, "No objects", ha='center', va='center', fontsize=12)
        ax_match.set_title(f"Najbardziej podobne\nPodobieństwo: {(1- similarity):.2f}")
        ax_match.axis('off')

        # Save the cropped test image to a file with a timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cropped_test_image_path = os.path.join(cropped_image_folder, f"cropped_test_image_{timestamp}_{i}.png")
        Image.fromarray(cropped_objects[i]).save(cropped_test_image_path)

        if client is not None:
            try:
                result = client.predict(
                    person_image={"background": handle_file(person_image_path), "layers": [], "composite": None},
                    cloth_image=handle_file(matched_image_path),
                    num_inference_steps=50,
                    guidance_scale=2.5,
                    seed=42,
                    api_name="/submit_function_p2p"
                )
                # Print the result link
                print(f"Gradio Result: {result}")

                # Update person_image with the result from Gradio
                updated_image_path = result['output_image']  # Assuming the result contains the path to the updated image
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                saved_image_path = os.path.join(output_folder, f"updated_image_{timestamp}_{i}.png")
                Image.open(updated_image_path).save(saved_image_path)
                person_image_path = saved_image_path
            except Exception as e:
                print(f"Gradio client error: {e}")
        else:
            print("Nie można połączyć się z Gradio, pomijamy przymierzenie.")

    plt.tight_layout(pad=3.0)
    plt.show()

def main():
    try:
        # Convert JPG and JPEG images to PNG
        convert_to_png(input_folder)

        # Ustawienia początkowe
        person_image_path = r"slai\refs\magda.png"  # Możliwość zmiany na inny obraz
        if not os.path.exists(person_image_path):
            raise FileNotFoundError(f"Reference image not found: {person_image_path}")

        # Testowy obraz
        test_image_path = r"slai\refs\magda.png"
        test_image = Image.open(test_image_path)
        results = yolo_model(test_image)

        # Update embeddings
        try:
            database_embeddings = load_embeddings(embeddings_file)
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"Error loading embeddings: {e}. Regenerating embeddings...")
            generate_and_save_embeddings(input_folder, output_folder)
            database_embeddings = load_embeddings(embeddings_file)
        
        display_comparison(test_image, results, database_embeddings, input_folder, person_image_path)

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()