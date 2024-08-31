import argparse
from argparse import ArgumentParser
import os
import os.path as osp

def parse_arguments():
    parser = argparse.ArgumentParser(description="Dark-enhanced Net Data and Model Preparation")

    parser.add_argument("--dataset-path", type=str, default="./data/dataset", 
                        help="Path to the dataset")
    parser.add_argument("--output-path", type=str, default="./data/output", 
                        help="Path to save the processed data")
    parser.add_argument("--scene-name", type=str, required=True, 
                        help="Scene name to process")

    parser.add_argument("--low-light-enhance", action="store_true", 
                        help="Apply low-light enhancement to images")
    parser.add_argument("--enhancement-module", type=str, default="ResEM", 
                        help="Choose the enhancement module to use, e.g., 'ResEM'")
    parser.add_argument("--brightness", type=float, default=1.0, 
                        help="Adjust brightness for data augmentation")
    parser.add_argument("--contrast", type=float, default=1.0, 
                        help="Adjust contrast for data augmentation")
    
    parser.add_argument("--image-size", type=int, default=224, 
                        help="Image size to resize for processing")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="Batch size for training and data processing")
    parser.add_argument("--num-workers", type=int, default=4, 
                        help="Number of workers for data loading")

    parser.add_argument("--model", type=str, default="DSPFormer", 
                        help="Choose the model architecture, e.g., 'DSPFormer'")
    parser.add_argument("--pose-count", type=int, default=4, 
                        help="Number of poses to consider in VPR task")
    parser.add_argument("--pose-distance", type=int, default=30, 
                        help="Minimum distance between poses")

    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                        help="Learning rate for model training")
    parser.add_argument("--optimizer", type=str, default="adam", 
                        help="Optimizer to use, e.g., 'adam' or 'adamW'")
    
    parser.add_argument("--vpr-task", action="store_true", 
                        help="Enable Visual Place Recognition task")
    parser.add_argument("--describe-by", type=str, default="all", 
                        help="Describe method for cells, options: 'closest', 'class', 'direction', 'random', 'all'")

    parser.add_argument("--save-model", action="store_true", 
                        help="Whether to save the model after training")
    parser.add_argument("--save-path", type=str, default="./checkpoints", 
                        help="Path to save model checkpoints")

    args = parser.parse_args()

    assert osp.isdir(args.dataset_path), f"Dataset path does not exist: {args.dataset_path}"
    assert args.describe_by in ("closest", "class", "direction", "random", "all"), "Invalid describe method"
    
    attribs = [
        args.output_path,
        args.scene_name,
        f"bs{args.batch_size}",
        f"lr{args.learning_rate}",
        args.model,
        args.enhancement_module if args.low_light_enhance else "noEnhance",
    ]
    args.output_path = "_".join([a for a in attribs if a])

    print(f"Processing scene {args.scene_name} with dataset at {args.dataset_path}")
    print(f"Processed data will be saved to: {args.output_path}")

    os.makedirs(args.output_path, exist_ok=True)

    return args

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
