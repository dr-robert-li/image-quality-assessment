import argparse
import tensorflow as tf

from src.handlers.model_builder import Nima


def main(base_model_name, weights_file, export_path):
    # Load model and weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # Export using TF SavedModel format
    tf.saved_model.save(nima.nima_model, export_path)

    print(f'TF model exported to: {export_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    parser.add_argument('-ep', '--export-path', help='path to save the tfs model', required=True)

    args = parser.parse_args()

    main(**args.__dict__)
