import json
import os

def merge_coco_like_jsons(input_dir, output_path):
    """
    Merges all COCO-like JSON files in `input_dir` into a single JSON file 
    and writes it to `output_path`.
    """
    
    merged_info = None
    merged_images = []
    merged_categories = []
    merged_annotations = []
    
    # Loop over all files in input_dir
    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue  # skip non-JSON files
        
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Take the first non-empty info as the primary info
            if merged_info is None:
                merged_info = data.get("info", {})
            
            # Extend the lists
            merged_images.extend(data.get("images", []))
            merged_categories.extend(data.get("categories", []))
            merged_annotations.extend(data.get("annotations", []))
    
    # Build the final merged structure
    merged_data = {
        "info": merged_info if merged_info else {},
        "images": merged_images,
        "categories": merged_categories,
        "annotations": merged_annotations
    }

    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(merged_data, out_f, indent=2)

# if __name__ == "__main__":
#     input_directory = "data/terra_incognita/eccv_18_annotation_files/"
#     output_file = "data/terra_incognita/merged_annotation.json"
    
#     merge_coco_like_jsons(input_directory, output_file)
#     print(f"Merged JSON written to: {output_file}")
