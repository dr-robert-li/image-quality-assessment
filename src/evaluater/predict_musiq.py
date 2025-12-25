
import os
import glob
import json
import argparse
import tensorflow as tf


def load_image_bytes(image_path):
    """Load image file as raw bytes."""
    with open(image_path, 'rb') as f:
        return f.read()


def image_file_to_list(img_path):
    """Convert single image path to list format."""
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]
    ext = os.path.splitext(img_path)[1][1:]  # Get extension without dot
    return img_dir, [{'image_id': img_id}], ext


def image_dir_to_list(img_dir, img_type='jpg'):
    """Convert directory of images to list format."""
    # Search for both lowercase and uppercase extensions
    img_paths = glob.glob(os.path.join(img_dir, '*.'+img_type.lower()))
    img_paths += glob.glob(os.path.join(img_dir, '*.'+img_type.upper()))

    # Also try common extensions if none found
    if not img_paths and img_type.lower() == 'jpg':
        img_paths = glob.glob(os.path.join(img_dir, '*.jpeg'))
        img_paths += glob.glob(os.path.join(img_dir, '*.JPEG'))

    samples = []
    actual_ext = None
    for img_path in sorted(img_paths):
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})
        if actual_ext is None:
            actual_ext = os.path.splitext(img_path)[1][1:]  # Get extension without dot

    return samples, actual_ext if actual_ext else img_type


def load_musiq_model(model_path):
    """Load MUSIQ SavedModel from local path."""
    print(f"Loading MUSIQ model from: {model_path}")
    model = tf.saved_model.load(model_path)
    predict_fn = model.signatures['serving_default']
    return predict_fn


def predict_single(predict_fn, image_path):
    """Run prediction on a single image."""
    image_bytes = load_image_bytes(image_path)
    result = predict_fn(tf.constant(image_bytes))
    # The output key may vary, typically 'output_0' or similar
    output_key = list(result.keys())[0]
    score = result[output_key].numpy()
    # Handle both scalar and array outputs
    if hasattr(score, 'flatten'):
        score = float(score.flatten()[0])
    else:
        score = float(score)
    return score


def main(model_path, image_source, predictions_file=None, img_format='jpg'):
    """
    Run MUSIQ image quality assessment on images.

    Args:
        model_path: Path to the MUSIQ SavedModel directory
        image_source: Path to a single image or directory of images
        predictions_file: Optional path to save predictions JSON
        img_format: Image format extension (default: jpg)
    """
    # Load model
    predict_fn = load_musiq_model(model_path)

    # Load samples
    if os.path.isfile(image_source):
        image_dir, samples, img_format = image_file_to_list(image_source)
    else:
        image_dir = image_source
        samples, img_format = image_dir_to_list(image_dir, img_type=img_format)

    if not samples:
        print(f"No images found in {image_source}")
        return

    print(f"Found {len(samples)} images to process")

    # Run predictions
    for i, sample in enumerate(samples):
        image_path = os.path.join(image_dir, f"{sample['image_id']}.{img_format}")
        try:
            score = predict_single(predict_fn, image_path)
            sample['musiq_score'] = score
            print(f"[{i+1}/{len(samples)}] {sample['image_id']}: {score:.4f}")
        except Exception as e:
            print(f"[{i+1}/{len(samples)}] {sample['image_id']}: Error - {str(e)}")
            sample['musiq_score'] = None

    # Output results
    print("\n" + "="*50)
    print("Results:")
    print(json.dumps(samples, indent=2))

    # Save to file if specified
    if predictions_file is not None:
        with open(predictions_file, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"\nPredictions saved to: {predictions_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MUSIQ Image Quality Assessment')
    parser.add_argument('-m', '--model-path',
                        help='Path to MUSIQ SavedModel directory',
                        default='models/MUSIQ')
    parser.add_argument('-is', '--image-source',
                        help='Image file or directory of images',
                        required=True)
    parser.add_argument('-pf', '--predictions-file',
                        help='Output file for predictions JSON',
                        required=False,
                        default=None)
    parser.add_argument('-f', '--img-format',
                        help='Image format extension (default: jpg)',
                        default='jpg')

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        image_source=args.image_source,
        predictions_file=args.predictions_file,
        img_format=args.img_format
    )
