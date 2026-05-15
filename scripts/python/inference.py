import argparse
from time import time

from satellite_trail_segmentation.unet_model.evaluate import recreate_full_field
from satellite_trail_segmentation.utils.visualizations import plot_full_field
from satellite_trail_segmentation.utils.load_model import load_model_weights
from satellite_trail_segmentation.postprocess.hough import hough_tranform
from satellite_trail_segmentation.unet_model.evaluate import image_threshold



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate satellite trail segmentation model")
    
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--plot-save-path", type=str, required=True)
    parser.add_argument("--hough-save-path", type=str, required=True)
    
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

    start_hough = time()
    image_hough = hough_tranform(images[1], hough_threshold=50, min_length=100, max_gap=250)
    hough_time = time() - start_hough
    print(f'Hough transform completed in: {hough_time} s')

    total_time = time() - start_time
    print(f'Pipeline total time: {total_time} s')
    
    start_plot = time()
    plot_full_field(*images, save_path=args.plot_save_path)
    plot_full_field(image_threshold(images[1]), images[2], image_hough, save_path=args.hough_save_path, threshold=None, title1="Prediction", title2="Mask", title3="Hough Prediction")
    print(f'Plotting complete in {time()-start_plot} s.')
    