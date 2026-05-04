import argparse
from time import time

from satellite_trail_segmentation.model.evaluate import recreate_full_field
from satellite_trail_segmentation.utils.visualizations import plot_full_field
from satellite_trail_segmentation.utils.load_model import load_model_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate satellite trail segmentation model")
    
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--plot-save-path", type=str, required=True)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    split_type = 'val'
    index = 4

    start_time = time()
    model = load_model_weights(args.model_path)
    prepocessing_time = time() - start_time
    print(f'Preprocessing completed in: {prepocessing_time} s')

    start_pred = time()
    images = recreate_full_field(model, args.data_path, split_type=split_type, source_index=index)
    pred_time = time() - start_pred
    print(f'Prediction completed in: {pred_time} s')

    total_time = time() - start_time
    print(f'Pipeline total time: {total_time} s')
    

    plot_full_field(*images, source_index=index, save_path=args.plot_save_path)
    print('Plotting complete.')