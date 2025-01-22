import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.gridspec import GridSpec
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

# Wybór urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ładowanie wcześniej wytrenowanego modelu YOLO
yolo_model = YOLO(r"kod\slai_final\best.pt")

# Ładowanie modelu ResNet50 do generowania embeddingów
resnet_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1])) 
resnet_model.to(device)
resnet_model.eval()

# Ładowanie modelu dyfuzyjnego Stable Diffusion
def load_diffusion_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe = pipe.to(device)
    return pipe

# Przekształcenia obrazu
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Funkcja generująca embedding z detekcji
def get_embedding_from_detection(image, detection_box):
    x1, y1, x2, y2 = map(int, detection_box)
    cropped_image = image.crop((x1, y1, x2, y2))
    if cropped_image.mode != 'RGB':
        cropped_image = cropped_image.convert('RGB')
    input_tensor = preprocess(cropped_image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet_model(input_tensor).squeeze().cpu().numpy()
    return embedding

# Funkcja generująca embeddingi dla wszystkich detekcji w obrazie
def generate_embeddings_for_all_detections(image, results, conf_threshold=0.4):
    embeddings = []
    for detection, mask in zip(results[0].boxes, results[0].masks.data if results[0].masks else []):
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
                    'class_index': class_index,
                    'mask': mask.cpu().numpy()
                })
    return embeddings

# Funkcja rysująca maski segmentacji
def draw_segmentation_masks(image, results):
    if results[0].masks is None:
        print("No segmentation masks found in the detection results.")
        return image
    img_array = np.array(image)
    result_image = results[0].plot()  # YOLO rysuje maski
    return Image.fromarray(result_image)  # Konwersja na format PIL

# Funkcja licząca podobieństwo i zwracająca najbardziej podobny element
def calculate_similarity(cropped_embedding, database_embeddings, class_index, similarity_threshold=0.4):
    if not database_embeddings:
        raise ValueError("Baza danych z embeddingami jest pusta.")

    all_embeddings = np.array([embedding['embedding'] for embedding in database_embeddings])
    all_class_indices = np.array([embedding['class_index'] for embedding in database_embeddings])

    similarities = cosine_similarity([cropped_embedding], all_embeddings)[0]

    # Filtrujemy po klasach
    same_class_indices = np.where(all_class_indices == class_index)[0]
    same_class_similarities = similarities[same_class_indices]

    if same_class_similarities.size > 0 and np.max(same_class_similarities) > similarity_threshold:
        most_similar_index = same_class_indices[np.argmax(same_class_similarities)]
        return similarities[most_similar_index], most_similar_index

    # Jeśli brak odpowiednika w tej samej klasie, zwracamy najbliższy ogólny
    closest_index = np.argmax(similarities)
    return similarities[closest_index], closest_index

# Funkcja generująca i zapisywania embeddingów
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

    # Zapisujemy embeddingi do pliku .npz
    output_path = os.path.join(output_folder, 'embeddings.npz')
    np.savez_compressed(
        output_path,
        embeddings=np.array(embeddings),
        filenames=np.array(filenames),
        class_indices=np.array(class_indices)
    )
    print(f"Embeddings saved to {output_path}")

# Funkcja ładowania embeddingów
def load_embeddings(embeddings_file):
    embeddings_data = np.load(embeddings_file, allow_pickle=True)
    embeddings = embeddings_data['embeddings']
    filenames = embeddings_data['filenames']
    class_indices = embeddings_data['class_indices']
    return [{'embedding': emb, 'filename': filename, 'class_index': class_index} 
            for emb, filename, class_index in zip(embeddings, filenames, class_indices)]
# Aktualizacja lub generowanie embeddingów
def update_embeddings(input_folder, output_folder, embeddings_file):
    if os.path.exists(embeddings_file):
        print("Checking for updates in the database...")
        existing_data = np.load(embeddings_file, allow_pickle=True)
        existing_filenames = list(existing_data['filenames'])
        existing_embeddings = list(existing_data['embeddings'])
        existing_class_indices = list(existing_data['class_indices'])

        current_images = [img for img in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, img))]
        new_images = [img for img in current_images if img not in existing_filenames]
        removed_images = [img for img in existing_filenames if img not in current_images]

        if removed_images:
            remaining_indices = [i for i, fname in enumerate(existing_filenames) if fname not in removed_images]
            existing_filenames = [existing_filenames[i] for i in remaining_indices]
            existing_embeddings = [existing_embeddings[i] for i in remaining_indices]
            existing_class_indices = [existing_class_indices[i] for i in remaining_indices]

        if new_images:
            for img_name in new_images:
                img_path = os.path.join(input_folder, img_name)
                image = Image.open(img_path)
                results = yolo_model(image)
                detections = generate_embeddings_for_all_detections(image, results)
                for detection in detections:
                    existing_embeddings.append(detection['embedding'])
                    existing_filenames.append(img_name)
                    existing_class_indices.append(detection['class_index'])

        np.savez_compressed(
            embeddings_file,
            embeddings=np.array(existing_embeddings),
            filenames=np.array(existing_filenames),
            class_indices=np.array(existing_class_indices)
        )
        print("Database updated successfully.")
    else:
        print("Embeddings file not found. Generating embeddings for all images...")
        generate_and_save_embeddings(input_folder, output_folder)

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
def resize_to_fit(image, axis):
    """
    Przeskalowuje obraz, aby idealnie pasował do przestrzeni subplotu, zachowując proporcje.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Rozmiary osi w pikselach
    bbox = axis.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
    display_width, display_height = bbox.width * plt.gcf().dpi, bbox.height * plt.gcf().dpi

    # Rozmiary obrazu
    img_width, img_height = image.size

    # Skalowanie obrazu, aby pasował do subplotu
    scale_w = (display_width / img_width)
    scale_h = display_height 
    scale = min(scale_w, scale_h)  # Dopasowanie do najmniejszego wymiaru

    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    resized_image = image.resize((new_width, new_height))

    # Dodanie marginesów, aby wypełnić cały subplot
    background = Image.new("RGBA", (int(display_width), int(display_height)), (255, 255, 255, 0))
    offset = ((background.size[0] - resized_image.size[0]) // 2,
              (background.size[1] - resized_image.size[1]) // 2)
    background.paste(resized_image, offset)

    return background


# Funkcja generująca nowy obraz na podstawie opisu tekstowego
def generate_image_from_text(pipe, prompt, output_path, num_inference_steps=50):
    """
    Generuje obraz na podstawie podanego promptu i zapisuje go do pliku.
    """
    with torch.autocast(device.type):
        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        image.save(output_path)
    print(f"Wygenerowano nowy obraz i zapisano do: {output_path}")

# Funkcja do konwersji wykrytych ubrań na opis tekstowy
def generate_prompt_from_detections(results):
    descriptions = []
    for detection in results[0].boxes:
        class_index = int(detection.cls[0].item())
        class_name = results[0].names[class_index]
        descriptions.append(f"{class_name}")
    return ", ".join(descriptions)
def load_diffusion_model_with_controlnet():
    model_id = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id = "lllyasviel/control_v11p_sd15_seg"

    # Załaduj model ControlNet z typem float32 na CPU
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32
    ).to("cpu")
    
    # Załaduj pipeline z ControlNetem
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float32
    )
    pipe = pipe.to(device)
    return pipe

# Funkcja generująca obraz za pomocą ControlNet i maski segmentacji
def generate_image_with_controlnet(pipe, image, mask, prompt, output_path, num_inference_steps=50):
    """
    Generuje obraz na podstawie podanego promptu i maski segmentacji.
    """
    # Jeśli control_image jest typu PIL Image
    control_image = image.convert("RGB")

    # Konwersja na tensor float32
    control_image_tensor = torch.from_numpy(np.array(control_image)).float().div(255)
    control_image_tensor = control_image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # Dodaj wymiar batcha i przenieś na odpowiednie urządzenie

    # Generowanie obrazu
    with torch.autocast(device.type, dtype=torch.float32):
        result = pipe(prompt, image=control_image_tensor, num_inference_steps=num_inference_steps).images[0]
        result.save(output_path)

    print(f"Wygenerowano nowy obraz z ControlNet i zapisano do: {output_path}")

# Funkcja do porównania obrazów na podstawie wykrytych obiektów
def display_comparison(test_image, results, database_embeddings, input_folder, similarity_threshold=0.2):
    embeddings = generate_embeddings_for_all_detections(test_image, results)
    masked_test = draw_segmentation_masks(test_image, results)

    num_rows = len(embeddings)
    fig_compare = plt.figure(figsize=(15, 5 * num_rows))
    gs = GridSpec(num_rows, 3, width_ratios=[2, 1, 1])

    ax_test_image = fig_compare.add_subplot(gs[:, 0])
    ax_test_image.imshow(masked_test)
    ax_test_image.axis('off')
    ax_test_image.set_title("Test Image")

    for i, detection in enumerate(embeddings):
        cropped_embedding = detection['embedding']
        class_index = detection['class_index']
        confidence = detection['confidence']
        similarity, matched_index = calculate_similarity(cropped_embedding, database_embeddings, class_index, similarity_threshold)

        ax_detection = fig_compare.add_subplot(gs[i, 1])
        cropped_objects = cut_out_objects(test_image, results)
        if i < len(cropped_objects):
            resized_image = resize_to_fit(cropped_objects[i], ax_detection)
            ax_detection.imshow(resized_image)
        else:
            ax_detection.text(0.5, 0.5, "No objects", ha='center', va='center', fontsize=12)
        ax_detection.set_title(f"Detected\nClass: {results[0].names[class_index]}\nConfidence: {confidence:.2f}")
        ax_detection.axis('off')

        ax_match = fig_compare.add_subplot(gs[i, 2])
        matched_image = Image.open(os.path.join(input_folder, database_embeddings[matched_index]['filename']))
        results_matched = yolo_model(matched_image)
        cropped_matched_image = cut_out_objects(matched_image, results_matched)

        if cropped_matched_image:
            ax_match.imshow(resize_to_fit(cropped_matched_image[0], ax_match))
        else:
            ax_match.text(0.5, 0.5, "No objects", ha='center', va='center', fontsize=12)
        ax_match.set_title(f"Most Similar\nSimilarity: {similarity:.2f}")
        ax_match.axis('off')

    plt.tight_layout(pad=3.0)
    plt.show()

# Funkcja główna
def main():
    input_folder = r"reszta_ubran\Baza_ubrań"
    output_folder = r"reszta_ubran\embeddings"
    embeddings_file = os.path.join(output_folder, 'embeddings.npz')

    # Aktualizacja bazy embeddingów
    update_embeddings(input_folder, output_folder, embeddings_file)
    database_embeddings = load_embeddings(embeddings_file)

    # Wczytywanie testowego obrazu
    test_image_path = r"reszta_ubran/refs/magda.png"
    test_image = Image.open(test_image_path)
    results = yolo_model(test_image)

    # Wyświetlanie porównania wyników
   #display_comparison(test_image, results, database_embeddings, input_folder)

    # Generowanie nowego obrazu za pomocą ControlNet
    diffusion_model = load_diffusion_model_with_controlnet()
    prompt = generate_prompt_from_detections(results)
    print(f"Wygenerowany prompt: {prompt}")

    # Generowanie maski segmentacji
    if results[0].masks:
        mask = results[0].masks.data[0].cpu().numpy()
        generated_image_path = r"reszta_ubran/generated_image_controlnet.png"
        generate_image_with_controlnet(diffusion_model, test_image, mask, prompt, generated_image_path)

        # Wyświetlenie wygenerowanego obrazu
        img = cv2.imread(generated_image_path)
        cv2.imshow('Generated Image with ControlNet', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Brak masek segmentacji, nie można użyć ControlNet.")

if __name__ == "__main__":
    main()