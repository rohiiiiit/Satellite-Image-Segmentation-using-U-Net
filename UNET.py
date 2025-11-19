import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import BinaryIoU
from sklearn.metrics import accuracy_score, jaccard_score
from tensorflow.keras import backend as K
import rasterio
from rasterio.plot import show
import albumentations as A
from PIL import Image
import cv2
from tqdm import tqdm
import random
import glob

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Define constants
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 2
BASE_PATH = "M:/PROJECTS/DL_PROJECT/LATEST_CONTENT/LAB_DATASET/temp/"  # Base path

# Define dice coefficient for model evaluation
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Define the U-Net model architecture
def build_unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(input_size)
    
    # Encoder (Contracting Path)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.1)(p1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.2)(p2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.3)(p3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(0.4)(p4)
    
    # Bridge
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = Dropout(0.5)(c5)
    
    # Decoder (Expanding Path)
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = Dropout(0.4)(c6)
    
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = Dropout(0.3)(c7)
    
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = Dropout(0.2)(c8)
    
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = Dropout(0.1)(c9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Data loading and preprocessing functions
def load_and_preprocess_image(image_path):
    try:
        img = np.array(Image.open(image_path))
        if len(img.shape) == 2:  # If grayscale, convert to RGB
            img = np.stack((img,)*3, axis=-1)
        elif img.shape[2] > 3:  # If has alpha channel or more bands
            img = img[:,:,:3]
        
        # Resize to target dimensions
        if img.shape[0] != IMG_HEIGHT or img.shape[1] != IMG_WIDTH:
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        
        # Normalize to [0,1]
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def load_and_preprocess_mask(mask_path):
    try:
        # Now we assume all masks are PNG format
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) > 2:  # If it has multiple channels, take the first one
            mask = mask[:,:,0]
            
        # Resize to target dimensions
        if mask.shape[0] != IMG_HEIGHT or mask.shape[1] != IMG_WIDTH:
            mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        
        # Ensure binary mask (0 or 1)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        return mask
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return None

def data_generator(image_paths, mask_paths, batch_size=BATCH_SIZE, augment=False):
    num_samples = len(image_paths)
    indices = np.arange(num_samples)
    
    # Data augmentation
    if augment:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
        ])
    
    while True:
        np.random.shuffle(indices)
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_x = []
            batch_y = []
            
            for idx in batch_indices:
                img = load_and_preprocess_image(image_paths[idx])
                mask = load_and_preprocess_mask(mask_paths[idx])
                
                if img is not None and mask is not None:
                    if augment:
                        augmented = transform(image=img, mask=mask[:,:,0])
                        img = augmented['image']
                        mask = np.expand_dims(augmented['mask'], axis=-1)
                    
                    batch_x.append(img)
                    batch_y.append(mask)
            
            if batch_x and batch_y:  # Check if lists are not empty
                yield np.array(batch_x), np.array(batch_y)
            else:
                continue

# Load dataset paths - Modified to match the updated directory structure
def get_dataset_paths(base_path):
    datasets = {
        'train': {'2020': {'images': [], 'masks': []}, '2024': {'images': [], 'masks': []}},
        'val': {'2020': {'images': [], 'masks': []}, '2024': {'images': [], 'masks': []}},
        'test': {'2020': {'images': [], 'masks': []}, '2024': {'images': [], 'masks': []}}
    }
    
    # Updated to match your directory structure with segment folder containing PNGs
    for year in ['2020', '2024']:
        for split in ['train', 'val', 'test']:
            img_path = os.path.join(base_path, year, split, 'images')
            mask_path = os.path.join(base_path, year, split, 'segment')  # Changed 'segmented' to 'segment'
            
            if os.path.exists(img_path) and os.path.exists(mask_path):
                # Get all image files
                image_files = sorted(glob.glob(os.path.join(img_path, '*.png')))
                
                # Get all mask files (now PNG format)
                mask_files = sorted(glob.glob(os.path.join(mask_path, '*.png')))
                
                datasets[split][year]['images'] = image_files
                datasets[split][year]['masks'] = mask_files
                
                # Ensure matching image-mask pairs
                if len(datasets[split][year]['images']) != len(datasets[split][year]['masks']):
                    print(f"Warning: Mismatch in number of images and masks for {split}/{year}")
                    print(f"Images: {len(datasets[split][year]['images'])}, Masks: {len(datasets[split][year]['masks'])}")
    
    return datasets

# Change detection function
def detect_changes(model, img_2020, img_2024, threshold=0.5):
    # Get predictions for both years
    pred_2020 = model.predict(np.expand_dims(img_2020, axis=0))[0]
    pred_2024 = model.predict(np.expand_dims(img_2024, axis=0))[0]
    
    # Binarize predictions
    pred_2020_bin = (pred_2020 > threshold).astype(np.uint8)
    pred_2024_bin = (pred_2024 > threshold).astype(np.uint8)
    
    # Calculate change mask
    change_mask = np.abs(pred_2024_bin - pred_2020_bin)
    
    return pred_2020, pred_2024, change_mask

# Evaluation metrics
def calculate_metrics(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    
    # Flatten arrays
    y_true_flat = y_true_bin.flatten()
    y_pred_flat = y_pred_bin.flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    iou = jaccard_score(y_true_flat, y_pred_flat, average='binary')
    
    # Calculate Dice coefficient manually
    intersection = np.sum(y_true_flat * y_pred_flat)
    dice = (2. * intersection) / (np.sum(y_true_flat) + np.sum(y_pred_flat))
    
    return {
        'accuracy': accuracy,
        'iou': iou,
        'dice': dice
    }

# Visualization functions
def visualize_results(original_2020, original_2024, mask_2020, mask_2024, 
                      pred_2020, pred_2024, change_mask, metrics):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    axes[0, 0].imshow(original_2020)
    axes[0, 0].set_title('2020 Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_2020[:,:,0], cmap='gray')
    axes[0, 1].set_title('2020 Ground Truth')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_2020[:,:,0], cmap='gray')
    axes[0, 2].set_title('2020 Prediction')
    axes[0, 2].axis('off')
    
    # Overlay prediction on original image
    overlay_2020 = original_2020.copy()
    overlay_2020[pred_2020[:,:,0] > 0.5] = [1, 0, 0]  # Red for predicted areas
    axes[0, 3].imshow(overlay_2020)
    axes[0, 3].set_title('2020 Overlay')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(original_2024)
    axes[1, 0].set_title('2024 Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_2024[:,:,0], cmap='gray')
    axes[1, 1].set_title('2024 Ground Truth')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(pred_2024[:,:,0], cmap='gray')
    axes[1, 2].set_title('2024 Prediction')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(change_mask[:,:,0], cmap='jet')
    axes[1, 3].set_title('Change Detection')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    # Add metrics as text
    plt.figtext(0.5, 0.01, 
                f"Accuracy: {metrics['accuracy']:.4f} | IoU: {metrics['iou']:.4f} | Dice: {metrics['dice']:.4f}", 
                ha='center', fontsize=12, bbox={'facecolor':'white', 'alpha':0.8, 'pad':5})
    
    plt.show()

# Function to match pairs of images between 2020 and 2024
def match_image_pairs(img_paths_2020, img_paths_2024):
    """Match images from 2020 and 2024 based on their filenames."""
    matched_pairs = []
    
    # Extract base filenames without year and extension
    base_names_2020 = [os.path.splitext(os.path.basename(p))[0] for p in img_paths_2020]
    base_names_2024 = [os.path.splitext(os.path.basename(p))[0] for p in img_paths_2024]
    
    # Find common identifiers
    for i, name_2020 in enumerate(base_names_2020):
        for j, name_2024 in enumerate(base_names_2024):
            # This is a simple matching strategy - adjust based on your actual naming convention
            if name_2020 == name_2024 or name_2020 in name_2024 or name_2024 in name_2020:
                matched_pairs.append((img_paths_2020[i], img_paths_2024[j]))
                break
    
    return matched_pairs

# Main execution
def main():
    # Load dataset paths
    print("Loading dataset paths...")
    datasets = get_dataset_paths(BASE_PATH)
    
    # Verify data availability
    for split in ['train', 'val', 'test']:
        for year in ['2020', '2024']:
            print(f"{split} {year}: {len(datasets[split][year]['images'])} images, {len(datasets[split][year]['masks'])} masks")
    
    # Create data generators for each year separately
    train_img_paths_2020 = datasets['train']['2020']['images']
    train_mask_paths_2020 = datasets['train']['2020']['masks']
    train_img_paths_2024 = datasets['train']['2024']['images']
    train_mask_paths_2024 = datasets['train']['2024']['masks']
    
    # Combine datasets for training
    train_img_paths = train_img_paths_2020 + train_img_paths_2024
    train_mask_paths = train_mask_paths_2020 + train_mask_paths_2024
    
    val_img_paths = datasets['val']['2020']['images'] + datasets['val']['2024']['images']
    val_mask_paths = datasets['val']['2020']['masks'] + datasets['val']['2024']['masks']
    
    train_gen = data_generator(train_img_paths, train_mask_paths, batch_size=BATCH_SIZE, augment=True)
    val_gen = data_generator(val_img_paths, val_mask_paths, batch_size=BATCH_SIZE, augment=False)
    
    # Build and compile model
    print("Building U-Net model...")
    model = build_unet_model()
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss=dice_coef_loss, 
                  metrics=['accuracy', BinaryIoU(threshold=0.5), dice_coef])
    
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint('unet_sentinel_model.h5', 
                                 monitor='val_binary_io_u', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='max')
    
    early_stopping = EarlyStopping(monitor='val_binary_io_u', 
                                  patience=10, 
                                  verbose=1, 
                                  mode='max')
    
    reduce_lr = ReduceLROnPlateau(monitor='val_binary_io_u', 
                                  factor=0.2, 
                                  patience=5, 
                                  verbose=1, 
                                  min_lr=1e-6, 
                                  mode='max')
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Train model
    print("Training model...")
    steps_per_epoch = len(train_img_paths) // BATCH_SIZE
    validation_steps = max(1, len(val_img_paths) // BATCH_SIZE)
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['binary_io_u'], label='Training IoU')
    plt.plot(history.history['val_binary_io_u'], label='Validation IoU')
    plt.title('Model IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Load best model for evaluation
    model.load_weights('unet_sentinel_model.h5')
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_img_2020_paths = datasets['test']['2020']['images']
    test_mask_2020_paths = datasets['test']['2020']['masks']
    test_img_2024_paths = datasets['test']['2024']['images']
    test_mask_2024_paths = datasets['test']['2024']['masks']
    
    # Match images from 2020 and 2024 for change detection
    # Assuming filenames are similar between years
    matched_test_pairs = match_image_pairs(test_img_2020_paths, test_img_2024_paths)
    print(f"Found {len(matched_test_pairs)} matched image pairs for testing")
    
    if len(matched_test_pairs) == 0:
        print("No matched image pairs found. Using random pairing for demonstration.")
        # If no matches found, just use some random pairs for demonstration
        min_test_samples = min(len(test_img_2020_paths), len(test_img_2024_paths))
        matched_test_pairs = [(test_img_2020_paths[i], test_img_2024_paths[i]) 
                            for i in range(min(5, min_test_samples))]
    
    all_metrics = []
    
    # Process test samples for visualization
    for i, (img_path_2020, img_path_2024) in enumerate(matched_test_pairs[:5]):  # Limit to 5 visualizations
        # Find corresponding mask paths
        mask_path_2020 = test_mask_2020_paths[test_img_2020_paths.index(img_path_2020)]
        mask_path_2024 = test_mask_2024_paths[test_img_2024_paths.index(img_path_2024)]
        
        # Load test images
        img_2020 = load_and_preprocess_image(img_path_2020)
        mask_2020 = load_and_preprocess_mask(mask_path_2020)
        img_2024 = load_and_preprocess_image(img_path_2024)
        mask_2024 = load_and_preprocess_mask(mask_path_2024)
        
        # Perform change detection
        pred_2020, pred_2024, change_mask = detect_changes(model, img_2020, img_2024)
        
        # Calculate metrics
        metrics_2020 = calculate_metrics(mask_2020, pred_2020)
        metrics_2024 = calculate_metrics(mask_2024, pred_2024)
        
        # Average metrics for both years
        metrics = {
            'accuracy': (metrics_2020['accuracy'] + metrics_2024['accuracy']) / 2,
            'iou': (metrics_2020['iou'] + metrics_2024['iou']) / 2,
            'dice': (metrics_2020['dice'] + metrics_2024['dice']) / 2
        }
        
        all_metrics.append(metrics)
        
        # Visualize results
        print(f"\nTest Sample {i+1}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"IoU: {metrics['iou']:.4f}")
        print(f"Dice: {metrics['dice']:.4f}")
        
        visualize_results(img_2020, img_2024, mask_2020, mask_2024, 
                        pred_2020, pred_2024, change_mask, metrics)
    
    # Calculate average metrics across all test samples
    if all_metrics:
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
            'iou': np.mean([m['iou'] for m in all_metrics]),
            'dice': np.mean([m['dice'] for m in all_metrics])
        }
        
        print("\nAverage Metrics on Test Set:")
        print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
        print(f"IoU: {avg_metrics['iou']:.4f}")
        print(f"Dice: {avg_metrics['dice']:.4f}")
    
    # Save some example change detection results
    output_dir = "change_detection_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (img_path_2020, img_path_2024) in enumerate(matched_test_pairs[:10]):  # Save 10 examples
        # Get file names for reference
        img_name_2020 = os.path.basename(img_path_2020)
        img_name_2024 = os.path.basename(img_path_2024)
        
        # Load images
        img_2020 = load_and_preprocess_image(img_path_2020)
        img_2024 = load_and_preprocess_image(img_path_2024)
        
        # Perform change detection
        pred_2020, pred_2024, change_mask = detect_changes(model, img_2020, img_2024)
        
        # Save change detection mask
        change_mask_img = (change_mask[:,:,0] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"change_{i}_{img_name_2020}_to_{img_name_2024}.png"), change_mask_img)

if __name__ == "__main__":
    main()
