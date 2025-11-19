import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import MeanIoU
from sklearn.metrics import jaccard_score
from PIL import Image
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from tensorflow.keras.utils import to_categorical
import matplotlib.colors as mcolors

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Define paths for 2020 and 2024 datasets
BASE_DIR = r'M:/PROJECTS/DL_PROJECT/LATEST_CONTENT/LAB_DATASET/temp/'
YEARS = ['2020', '2024']
SUBSETS = ['train', 'val', 'test']

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
N_CHANNELS = 3  # RGB images
N_CLASSES = 6   # Water, Trees, Crops, Built, Bare, Rangeland

# Define class names and colors for visualization
CLASS_NAMES = ['Water', 'Trees', 'Crops', 'Built', 'Bare', 'Rangeland']
CLASS_COLORS = [
    [0, 0, 255],    # Water - Blue
    [0, 128, 0],    # Trees - Green
    [0, 255, 0],    # Crops - Light Green
    [128, 0, 0],    # Built - Maroon
    [255, 255, 0],  # Bare - Yellow
    [255, 165, 0]   # Rangeland - Orange
]

def create_unet_model(input_size=(IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNELS), n_classes=N_CLASSES):
    """Create a U-Net model for image segmentation with multiple classes"""
    inputs = Input(input_size)
    
    # Encoder (Contracting Path)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    # Bridge
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # Decoder (Expanding Path)
    up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    # Output layer - for multi-class segmentation
    outputs = Conv2D(n_classes, 1, activation='softmax')(conv9)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4), 
        loss='categorical_crossentropy', 
        metrics=['accuracy', MeanIoU(num_classes=n_classes)]
    )
    
    return model

def rgb_to_onehot(mask, class_colors):
    """Convert RGB mask to one-hot encoded mask"""
    height, width, _ = mask.shape
    onehot_mask = np.zeros((height, width, len(class_colors)))
    
    # Create lookup array for colors
    color_array = np.array(class_colors)
    
    # For each pixel in the mask, find the closest color and set the corresponding class
    for i in range(height):
        for j in range(width):
            pixel = mask[i, j]
            distances = np.sum(np.abs(color_array - pixel), axis=1)
            class_idx = np.argmin(distances)
            onehot_mask[i, j, class_idx] = 1
    
    return onehot_mask

def onehot_to_rgb(onehot_mask, class_colors):
    """Convert one-hot encoded mask to RGB mask for visualization"""
    height, width, n_classes = onehot_mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get class indices for each pixel
    class_indices = np.argmax(onehot_mask, axis=2)
    
    for c in range(n_classes):
        mask = (class_indices == c)
        rgb_mask[mask] = class_colors[c]
    
    return rgb_mask

def analyze_mask_data(year, subset='train', num_samples=10):
    """Analyze mask data to understand the structure and colors"""
    masks_path = os.path.join(BASE_DIR, year, subset, 'segment')
    mask_files = sorted([f for f in os.listdir(masks_path) if f.endswith('.png')])
    
    if num_samples:
        mask_files = mask_files[:min(num_samples, len(mask_files))]
    
    unique_colors = set()
    
    print(f"Analyzing mask data for {year} {subset}...")
    
    for mask_file in mask_files:
        mask_path = os.path.join(masks_path, mask_file)
        mask = np.array(Image.open(mask_path))
        
        # Check if image has 3 channels (RGB)
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            # Collect unique colors
            mask_reshaped = mask.reshape(-1, mask.shape[2])
            for pixel in mask_reshaped:
                if np.any(pixel > 0):  # Only non-black pixels
                    unique_colors.add(tuple(pixel))
        else:
            print(f"Mask {mask_file} is not RGB (shape: {mask.shape})")
    
    print(f"Analysis of {len(mask_files)} masks from {year} {subset}:")
    print(f"Number of unique colors found: {len(unique_colors)}")
    print("Sample of unique colors:", list(unique_colors)[:10])
    
    return unique_colors

def load_data(year, subset, limit=None):
    """Load images and masks for a specific year and subset with one-hot encoding for masks"""
    images_path = os.path.join(BASE_DIR, year, subset, 'images')
    masks_path = os.path.join(BASE_DIR, year, subset, 'segment')
    
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.png')])
    
    if limit:
        image_files = image_files[:limit]
    
    X = []
    y = []
    
    for img_file in image_files:
        # Load image
        img_path = os.path.join(images_path, img_file)
        img = load_img(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        img_array = img_to_array(img) / 255.0
        X.append(img_array)
        
        # Load corresponding mask and convert to one-hot encoding
        mask_path = os.path.join(masks_path, img_file)
        mask_img = Image.open(mask_path).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        mask_array = np.array(mask_img)
        
        # Convert RGB mask to one-hot encoding
        onehot_mask = rgb_to_onehot(mask_array, CLASS_COLORS)
        y.append(onehot_mask)
    
    return np.array(X), np.array(y)

def display_sample_images(year, subset, num_samples=3):
    """Display sample images and their masks with colors"""
    images_path = os.path.join(BASE_DIR, year, subset, 'images')
    masks_path = os.path.join(BASE_DIR, year, subset, 'segment')
    
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.png')])
    sampled_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    plt.figure(figsize=(12, 4*num_samples))
    
    for i, img_file in enumerate(sampled_files):
        # Load and display image
        img_path = os.path.join(images_path, img_file)
        img = load_img(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Load and display mask
        mask_path = os.path.join(masks_path, img_file)
        mask = load_img(mask_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
        
        # Plot image and mask
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(img)
        plt.title(f'{year} {subset} - Image: {img_file}')
        plt.axis('off')
        
        plt.subplot(num_samples, 2, 2*i+2)
        plt.imshow(mask)
        plt.title(f'{year} {subset} - Mask: {img_file}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_class_legend():
    """Create a legend for the class colors"""
    plt.figure(figsize=(8, 2))
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        plt.subplot(1, len(CLASS_NAMES), i+1)
        plt.fill([0, 1, 1, 0], [0, 0, 1, 1], color=[c/255 for c in color])
        plt.title(name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def train_models():
    """Train U-Net models for both 2020 and 2024 data"""
    models = {}
    histories = {}
    
    for year in YEARS:
        print(f"\nTraining model for {year}...")
        
        # Load training and validation data
        X_train, y_train = load_data(year, 'train')
        X_val, y_val = load_data(year, 'val')
        
        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
        
        # Create and train model
        model = create_unet_model()
        
        checkpoint = ModelCheckpoint(f'unet_model_{year}.h5', 
                                    monitor='val_loss',
                                    verbose=1, 
                                    save_best_only=True)
        
        early_stopping = EarlyStopping(monitor='val_loss', 
                                      patience=10, 
                                      verbose=1, 
                                      restore_best_weights=True)
        
        history = model.fit(X_train, y_train,
                           batch_size=8,
                           epochs=50,
                           validation_data=(X_val, y_val),
                           callbacks=[checkpoint, early_stopping])
        
        models[year] = model
        histories[year] = history
    
    return models, histories

def plot_training_history(histories):
    """Plot training history for both models"""
    plt.figure(figsize=(12, 10))
    
    for i, year in enumerate(YEARS):
        history = histories[year]
        
        # Plot accuracy
        plt.subplot(2, 2, 2*i+1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'{year} Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        # Plot loss
        plt.subplot(2, 2, 2*i+2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'{year} Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

def evaluate_models(models):
    """Evaluate models on test data"""
    results = {}
    
    for year in YEARS:
        model = models[year]
        X_test, y_test = load_data(year, 'test')
        
        print(f"\nEvaluating model for {year}...")
        metrics = model.evaluate(X_test, y_test, verbose=1)
        
        results[year] = {
            'loss': metrics[0],
            'accuracy': metrics[1],
            'mean_iou': metrics[2]
        }
        
        print(f"{year} Test loss: {metrics[0]:.4f}")
        print(f"{year} Test accuracy: {metrics[1]:.4f}")
        print(f"{year} Test Mean IoU: {metrics[2]:.4f}")
        
        # Calculate per-class IoU
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=3)
        y_true_classes = np.argmax(y_test, axis=3)
        
        class_ious = []
        for c in range(N_CLASSES):
            y_pred_c = (y_pred_classes == c)
            y_true_c = (y_true_classes == c)
            intersection = np.logical_and(y_pred_c, y_true_c).sum()
            union = np.logical_or(y_pred_c, y_true_c).sum()
            iou = intersection / (union + 1e-10)
            class_ious.append(iou)
            print(f"{year} Class {CLASS_NAMES[c]} IoU: {iou:.4f}")
        
        results[year]['class_ious'] = class_ious
    
    return results

def predict_and_visualize(models, num_samples=3):
    """Make predictions and visualize the results for both years"""
    for year in YEARS:
        model = models[year]
        X_test, y_test = load_data(year, 'test', limit=num_samples)
        
        predictions = model.predict(X_test)
        
        plt.figure(figsize=(15, 5*num_samples))
        for i in range(num_samples):
            # Original Image
            plt.subplot(num_samples, 3, 3*i+1)
            plt.imshow(X_test[i])
            plt.title(f'{year} Original Image')
            plt.axis('off')
            
            # Ground Truth Mask (convert one-hot back to RGB for visualization)
            true_mask_rgb = onehot_to_rgb(y_test[i], CLASS_COLORS)
            plt.subplot(num_samples, 3, 3*i+2)
            plt.imshow(true_mask_rgb)
            plt.title(f'{year} True Mask')
            plt.axis('off')
            
            # Predicted Mask (convert one-hot back to RGB for visualization)
            pred_mask_rgb = onehot_to_rgb(predictions[i], CLASS_COLORS)
            plt.subplot(num_samples, 3, 3*i+3)
            plt.imshow(pred_mask_rgb)
            plt.title(f'{year} Predicted Mask')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def detect_binary_change(models, num_samples=3):
    """Perform binary change detection between 2020 and 2024"""
    # Load test images for both years
    X_test_2020, y_test_2020 = load_data('2020', 'test')
    X_test_2024, y_test_2024 = load_data('2024', 'test')
    
    # Ensure we have the same number of images for both years
    min_samples = min(len(X_test_2020), len(X_test_2024))
    X_test_2020 = X_test_2020[:min_samples]
    y_test_2020 = y_test_2020[:min_samples]
    X_test_2024 = X_test_2024[:min_samples]
    y_test_2024 = y_test_2024[:min_samples]
    
    # Make predictions for both years
    pred_2020 = models['2020'].predict(X_test_2020)
    pred_2024 = models['2024'].predict(X_test_2024)
    
    # Convert predictions to class indices
    pred_classes_2020 = np.argmax(pred_2020, axis=3)
    pred_classes_2024 = np.argmax(pred_2024, axis=3)
    
    # Compute binary change mask (1 where classes are different, 0 where same)
    binary_change_masks = (pred_classes_2020 != pred_classes_2024).astype(np.uint8)
    
    # Visualize changes for a few samples
    display_samples = min(num_samples, min_samples)
    plt.figure(figsize=(15, 6*display_samples))
    
    for i in range(display_samples):
        # 2020 Image
        plt.subplot(display_samples, 4, 4*i+1)
        plt.imshow(X_test_2020[i])
        plt.title('2020 Image')
        plt.axis('off')
        
        # 2024 Image
        plt.subplot(display_samples, 4, 4*i+2)
        plt.imshow(X_test_2024[i])
        plt.title('2024 Image')
        plt.axis('off')
        
        # 2020 Predicted Mask
        pred_mask_2020 = onehot_to_rgb(pred_2020[i], CLASS_COLORS)
        plt.subplot(display_samples, 4, 4*i+3)
        plt.imshow(pred_mask_2020)
        plt.title('2020 Pred Mask')
        plt.axis('off')
        
        # 2024 Predicted Mask
        pred_mask_2024 = onehot_to_rgb(pred_2024[i], CLASS_COLORS)
        plt.subplot(display_samples, 4, 4*i+4)
        plt.imshow(pred_mask_2024)
        plt.title('2024 Pred Mask')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize binary change detection
    plt.figure(figsize=(15, 5*display_samples))
    
    for i in range(display_samples):
        # 2020 vs 2024 Images
        plt.subplot(display_samples, 3, 3*i+1)
        plt.imshow(X_test_2020[i])
        plt.title('2020 Image')
        plt.axis('off')
        
        plt.subplot(display_samples, 3, 3*i+2)
        plt.imshow(X_test_2024[i])
        plt.title('2024 Image')
        plt.axis('off')
        
        # Binary Change Mask
        plt.subplot(display_samples, 3, 3*i+3)
        plt.imshow(binary_change_masks[i], cmap='hot')
        plt.title('Binary Change Mask')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate change statistics
    change_percentage = np.mean(binary_change_masks) * 100
    
    print(f"Binary Change Analysis:")
    print(f"Total images analyzed: {min_samples}")
    print(f"Average change percentage: {change_percentage:.2f}%")
    
    return binary_change_masks

def detect_classwise_change(models, num_samples=3):
    """Perform class-wise change detection between 2020 and 2024"""
    # Load test images for both years
    X_test_2020, y_test_2020 = load_data('2020', 'test')
    X_test_2024, y_test_2024 = load_data('2024', 'test')
    
    # Ensure we have the same number of images for both years
    min_samples = min(len(X_test_2020), len(X_test_2024))
    X_test_2020 = X_test_2020[:min_samples]
    y_test_2020 = y_test_2020[:min_samples]
    X_test_2024 = X_test_2024[:min_samples]
    y_test_2024 = y_test_2024[:min_samples]
    
    # Make predictions for both years
    pred_2020 = models['2020'].predict(X_test_2020)
    pred_2024 = models['2024'].predict(X_test_2024)
    
    # Convert predictions to class indices
    pred_classes_2020 = np.argmax(pred_2020, axis=3)
    pred_classes_2024 = np.argmax(pred_2024, axis=3)
    
    # Create class transition matrix
    transition_matrix = np.zeros((N_CLASSES, N_CLASSES))
    
    # Fill transition matrix (from 2020 class to 2024 class)
    for i in range(min_samples):
        for c1 in range(N_CLASSES):
            for c2 in range(N_CLASSES):
                # Count pixels that changed from class c1 in 2020 to class c2 in 2024
                mask_2020 = (pred_classes_2020[i] == c1)
                mask_2024 = (pred_classes_2024[i] == c2)
                transition = np.logical_and(mask_2020, mask_2024)
                transition_matrix[c1, c2] += transition.sum()
    
    # Normalize transition matrix by row (from class)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix_norm = np.zeros_like(transition_matrix)
    for i in range(N_CLASSES):
        if row_sums[i] > 0:
            transition_matrix_norm[i] = transition_matrix[i] / row_sums[i]
    
    # Visualize the transition matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(transition_matrix_norm, cmap='Blues')
    plt.colorbar(label='Transition Probability')
    plt.xlabel('2024 Class')
    plt.ylabel('2020 Class')
    plt.title('Class Transition Matrix (2020 → 2024)')
    
    # Set tick labels
    plt.xticks(np.arange(N_CLASSES), CLASS_NAMES, rotation=45)
    plt.yticks(np.arange(N_CLASSES), CLASS_NAMES)
    
    # Add text annotations
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            text_color = 'white' if transition_matrix_norm[i, j] > 0.5 else 'black'
            plt.text(j, i, f'{transition_matrix_norm[i, j]:.2f}', 
                    ha='center', va='center', color=text_color)
    
    plt.tight_layout()
    plt.show()
    
    # Display class-wise change maps for a few samples
    display_samples = min(num_samples, min_samples)
    
    # Create a colormap for changed classes
    cmap = plt.cm.get_cmap('tab10', N_CLASSES)
    
    for i in range(display_samples):
        plt.figure(figsize=(15, 8))
        
        # 2020 Image and Prediction
        plt.subplot(2, 3, 1)
        plt.imshow(X_test_2020[i])
        plt.title('2020 Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(onehot_to_rgb(pred_2020[i], CLASS_COLORS))
        plt.title('2020 Prediction')
        plt.axis('off')
        
        # 2024 Image and Prediction
        plt.subplot(2, 3, 4)
        plt.imshow(X_test_2024[i])
        plt.title('2024 Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(onehot_to_rgb(pred_2024[i], CLASS_COLORS))
        plt.title('2024 Prediction')
        plt.axis('off')
        
        # Change Map
        change_mask = (pred_classes_2020[i] != pred_classes_2024[i])
        changed_classes_2024 = np.where(change_mask, pred_classes_2024[i], -1)
        
        # Class-wise change visualization
        plt.subplot(2, 3, 3)
        change_map = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        for c in range(N_CLASSES):
            change_map[changed_classes_2024 == c] = CLASS_COLORS[c]
        
        plt.imshow(change_map)
        plt.title('Class Changes')
        plt.axis('off')
        
        # Binary change visualization
        plt.subplot(2, 3, 6)
        plt.imshow(change_mask, cmap='hot')
        plt.title('Binary Change Mask')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Show class change statistics for this image
        class_changes = {}
        for c1 in range(N_CLASSES):
            for c2 in range(N_CLASSES):
                if c1 != c2:
                    mask_2020_c1 = (pred_classes_2020[i] == c1)
                    mask_2024_c2 = (pred_classes_2024[i] == c2)
                    change_c1_to_c2 = np.logical_and(mask_2020_c1, mask_2024_c2).sum()
                    if change_c1_to_c2 > 0:
                        class_changes[f"{CLASS_NAMES[c1]} → {CLASS_NAMES[c2]}"] = change_c1_to_c2
        
        # Sort by number of pixels changed
        sorted_changes = sorted(class_changes.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Image {i+1} - Top class changes:")
        for change, pixels in sorted_changes[:5]:  # Show top 5 changes
            print(f"  {change}: {pixels} pixels")
    
    return transition_matrix_norm

def visualize_change_overlay(models, num_samples=3):
    """Visualize change detection with colored overlays"""
    # Load test images for both years
    X_test_2020, _ = load_data('2020', 'test')
    X_test_2024, _ = load_data('2024', 'test')
    
    # Ensure we have the same number of images for both years
    min_samples = min(len(X_test_2020), len(X_test_2024))
    X_test_2020 = X_test_2020[:min_samples]
    X_test_2024 = X_test_2024[:min_samples]
    
    # Make predictions for both years
    pred_2020 = models['2020'].predict(X_test_2020)
    pred_2024 = models['2024'].predict(X_test_2024)
    
    # Convert predictions to class indices
    pred_classes_2020 = np.argmax(pred_2020, axis=3)
    pred_classes_2024 = np.argmax(pred_2024, axis=3)
    
    # Compute binary change mask
    binary_change = (pred_classes_2020 != pred_classes_2024).astype(np.uint8)
    
    # Visualize with colored overlays
    display_samples = min(num_samples, min_samples)
    
    for i in range(display_samples):
        # Create overlays for visualization
        pred_mask_2020_rgb = onehot_to_rgb(pred_2020[i], CLASS_COLORS)
        pred_mask_2024_rgb = onehot_to_rgb(pred_2024[i], CLASS_COLORS)
        
        plt.figure(figsize=(15, 6))
        
        # Original 2020 Image with Overlay
        plt.subplot(2, 3, 1)
        plt.imshow(X_test_2020[i])
        plt.title('2020 Original Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(pred_mask_2020_rgb)
        plt.title('2020 Predicted Classes')
        plt.axis('off')
        
        # Original 2024 Image with Overlay
        plt.subplot(2, 3, 4)
        plt.imshow(X_test_2024[i])
        plt.title('2024 Original Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(pred_mask_2024_rgb)
        plt.title('2024 Predicted Classes')
        plt.axis('off')
        
        # Change visualization - simple binary
        plt.subplot(2, 3, 3)
        plt.imshow(X_test_2024[i])  # Use 2024 as background
        change_mask = binary_change[i]
        plt.imshow(change_mask, cmap='hot', alpha=0.5)
        plt.title('Binary Change Overlay')
        plt.axis('off')
        
        # Change visualization - class transitions
        plt.subplot(2, 3, 6)
        # Create a mask that shows only the changed pixels with the new class color
        change_viz = np.zeros_like(pred_mask_2024_rgb)
        for c in range(N_CLASSES):
            # For each class in 2024, highlight pixels that changed to this class
            change_to_c = np.logical_and(binary_change[i], pred_classes_2024[i] == c)
            change_viz[change_to_c] = CLASS_COLORS[c]
        
        plt.imshow(X_test_2024[i])  # Use 2024 as background
        plt.imshow(change_viz, alpha=0.7)
        plt.title('Class Change Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Detailed change statistics for this sample
        print(f"\nChange statistics for sample {i+1}:")
        total_pixels = IMAGE_HEIGHT * IMAGE_WIDTH
        changed_pixels = np.sum(binary_change[i])
        print(f"Changed pixels: {changed_pixels} ({changed_pixels/total_pixels*100:.2f}% of image)")
        
        # Class-wise changes
        print("\nClass-wise changes:")
        print("Original Class (2020) → New Class (2024): Pixels (% of change)")
        
        # Track the most significant changes
        class_changes = {}
        for c1 in range(N_CLASSES):
            for c2 in range(N_CLASSES):
                if c1 != c2:  # Only interested in changes between different classes
                    # Pixels that were class c1 in 2020 and changed to class c2 in 2024
                    c1_to_c2 = np.logical_and(
                        np.logical_and(pred_classes_2020[i] == c1, pred_classes_2024[i] == c2),
                        binary_change[i]
                    ).sum()
                    
                    if c1_to_c2 > 0:
                        change_pct = c1_to_c2 / changed_pixels * 100
                        class_changes[f"{CLASS_NAMES[c1]} → {CLASS_NAMES[c2]}"] = (c1_to_c2, change_pct)
        
        # Sort by number of pixels and display
        sorted_changes = sorted(class_changes.items(), key=lambda x: x[1][0], reverse=True)
        for change, (pixels, pct) in sorted_changes:
            print(f"  {change}: {pixels} pixels ({pct:.2f}% of changes)")

def create_change_summary(models):
    """Create an overall summary of changes between 2020 and 2024"""
    # Load test images for both years
    X_test_2020, _ = load_data('2020', 'test')
    X_test_2024, _ = load_data('2024', 'test')
    
    # Ensure we have the same number of images
    min_samples = min(len(X_test_2020), len(X_test_2024))
    X_test_2020 = X_test_2020[:min_samples]
    X_test_2024 = X_test_2024[:min_samples]
    
    # Make predictions
    pred_2020 = models['2020'].predict(X_test_2020)
    pred_2024 = models['2024'].predict(X_test_2024)
    
    # Convert to class indices
    pred_classes_2020 = np.argmax(pred_2020, axis=3)
    pred_classes_2024 = np.argmax(pred_2024, axis=3)
    
    # Binary change detection
    binary_change = (pred_classes_2020 != pred_classes_2024)
    
    # Calculate global change statistics
    total_pixels = pred_classes_2020.size
    changed_pixels = np.sum(binary_change)
    change_percentage = changed_pixels / total_pixels * 100
    
    print(f"\nGLOBAL CHANGE ANALYSIS:")
    print(f"Total images analyzed: {min_samples}")
    print(f"Total pixels analyzed: {total_pixels}")
    print(f"Changed pixels: {changed_pixels} ({change_percentage:.2f}%)")
    
    # Class distribution in 2020 and 2024
    class_distribution_2020 = np.zeros(N_CLASSES)
    class_distribution_2024 = np.zeros(N_CLASSES)
    
    for c in range(N_CLASSES):
        class_distribution_2020[c] = np.sum(pred_classes_2020 == c) / total_pixels * 100
        class_distribution_2024[c] = np.sum(pred_classes_2024 == c) / total_pixels * 100
    
    # Create class distribution plot
    plt.figure(figsize=(12, 6))
    
    bar_width = 0.35
    index = np.arange(N_CLASSES)
    
    plt.bar(index, class_distribution_2020, bar_width, label='2020')
    plt.bar(index + bar_width, class_distribution_2024, bar_width, label='2024')
    
    plt.xlabel('Land Cover Class')
    plt.ylabel('Percentage of Area (%)')
    plt.title('Land Cover Class Distribution: 2020 vs 2024')
    plt.xticks(index + bar_width/2, CLASS_NAMES, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Create transition matrix for the entire dataset
    transition_matrix = np.zeros((N_CLASSES, N_CLASSES))
    
    for i in range(min_samples):
        for c1 in range(N_CLASSES):
            for c2 in range(N_CLASSES):
                # Count pixels that changed from class c1 in 2020 to class c2 in 2024
                mask_2020 = (pred_classes_2020[i] == c1)
                mask_2024 = (pred_classes_2024[i] == c2)
                transition = np.logical_and(mask_2020, mask_2024)
                transition_matrix[c1, c2] += transition.sum()
    
    # Convert to percentages (by row - from class)
    transition_percentages = np.zeros_like(transition_matrix, dtype=float)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    for i in range(N_CLASSES):
        if row_sums[i] > 0:
            transition_percentages[i] = transition_matrix[i] / row_sums[i] * 100
    
    # Plot transition matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(transition_percentages, cmap='Blues')
    plt.colorbar(label='Transition Percentage (%)')
    plt.xlabel('2024 Class')
    plt.ylabel('2020 Class')
    plt.title('Land Cover Class Transitions (2020 → 2024)')
    
    # Set tick labels
    plt.xticks(np.arange(N_CLASSES), CLASS_NAMES, rotation=45)
    plt.yticks(np.arange(N_CLASSES), CLASS_NAMES)
    
    # Add percentage text
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            text_color = 'white' if transition_percentages[i, j] > 50 else 'black'
            plt.text(j, i, f'{transition_percentages[i, j]:.1f}%', 
                    ha='center', va='center', color=text_color)
    
    plt.tight_layout()
    plt.show()
    
    # Most significant class transitions
    transitions = []
    for c1 in range(N_CLASSES):
        for c2 in range(N_CLASSES):
            if c1 != c2:  # Only interested in changes between different classes
                pixels = transition_matrix[c1, c2]
                percentage = transition_percentages[c1, c2]
                transitions.append((
                    CLASS_NAMES[c1], 
                    CLASS_NAMES[c2], 
                    pixels,
                    percentage
                ))
    
    # Sort by number of pixels
    sorted_transitions = sorted(transitions, key=lambda x: x[2], reverse=True)
    
    print("\nMost significant land cover transitions:")
    print("From Class → To Class: Pixels (% of source class)")
    for from_class, to_class, pixels, percentage in sorted_transitions[:10]:  # Top 10
        print(f"{from_class} → {to_class}: {pixels:.0f} pixels ({percentage:.2f}% of {from_class})")
    
    return {
        'change_percentage': change_percentage,
        'class_distribution_2020': class_distribution_2020,
        'class_distribution_2024': class_distribution_2024,
        'transition_matrix': transition_matrix,
        'transition_percentages': transition_percentages
    }

def run_complete_pipeline():
    """Run the complete pipeline for segmentation and change detection"""
    # Check if data exists
    for year in YEARS:
        for subset in SUBSETS:
            img_path = os.path.join(BASE_DIR, year, subset, 'images')
            mask_path = os.path.join(BASE_DIR, year, subset, 'segment')
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"Error: Directory not found - {img_path} or {mask_path}")
                return
    
    # First, analyze the data to understand the mask colors
    for year in YEARS:
        unique_colors = analyze_mask_data(year, 'train')
        print(f"Found {len(unique_colors)} unique colors in {year} training masks")
    
    # Display a legend for the class colors
    create_class_legend()
    
    # Display some sample images to verify data loading
    print("Displaying sample images from 2020 training set:")
    display_sample_images('2020', 'train')
    
    print("Displaying sample images from 2024 training set:")
    display_sample_images('2024', 'train')
    
    # Train models for both years
    models, histories = train_models()
    
    # Plot training history
    plot_training_history(histories)
    
    # Evaluate models
    results = evaluate_models(models)
    
    # Predict and visualize segmentation results
    predict_and_visualize(models)
    
    # Perform and visualize binary change detection
    binary_change_masks = detect_binary_change(models)
    
    # Perform and visualize class-wise change detection
    transition_matrix = detect_classwise_change(models)
    
    # Visualize change detection with colored overlays
    visualize_change_overlay(models)
    
    # Create summary of changes
    change_summary = create_change_summary(models)
    
    return models, results, change_summary

def save_all_results(models, results, change_summary):
    """Save all results and models"""
    import json
    
    # Save models
    for year, model in models.items():
        model.save(f'unet_model_{year}_final.h5')
        print(f"Model for {year} saved to unet_model_{year}_final.h5")
    
    # Save evaluation results
    serializable_results = {}
    for year, vals in results.items():
        serializable_results[year] = {k: float(v) if not isinstance(v, list) else [float(x) for x in v] 
                                    for k, v in vals.items()}
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print("Evaluation results saved to evaluation_results.json")
    
    # Save change summary
    serializable_summary = {
        'change_percentage': float(change_summary['change_percentage']),
        'class_distribution_2020': [float(x) for x in change_summary['class_distribution_2020']],
        'class_distribution_2024': [float(x) for x in change_summary['class_distribution_2024']],
        'transition_matrix': [[float(cell) for cell in row] for row in change_summary['transition_matrix']],
        'transition_percentages': [[float(cell) for cell in row] for row in change_summary['transition_percentages']]
    }
    
    with open('change_summary.json', 'w') as f:
        json.dump(serializable_summary, f, indent=4)
    
    print("Change summary saved to change_summary.json")

# Run the complete pipeline
if __name__ == "__main__":
    models, results, change_summary = run_complete_pipeline()
    save_all_results(models, results, change_summary)
    print("Pipeline completed successfully!")