# import os
# import shutil
# import pandas as pd
# from pathlib import Path

# # List of valid categories (62 total)
# VALID_CATEGORIES = [
#     "airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", 
#     "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", 
#     "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", 
#     "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", 
#     "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", 
#     "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", 
#     "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", 
#     "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", 
#     "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", 
#     "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", 
#     "water_treatment_facility", "wind_farm", "zoo"
# ]

# def create_fmow_subset(original_dir, target_dir, samples_per_class=200):
#     """
#     Create a smaller FMoW dataset with up to `samples_per_class` images per valid category.
#     - Filters out rows whose category is not in VALID_CATEGORIES (e.g. 'seq', 'train', 'test', etc.).
#     - Assumes row positions (0..N-1) correspond to file names: rgb_img_0.png, rgb_img_1.png, etc.
#     - Writes a new rgb_metadata.csv in target_dir with the sampled subset.
#     """
#     original_dir = Path(original_dir)
#     target_dir = Path(target_dir)
#     (target_dir / 'images').mkdir(parents=True, exist_ok=True)

#     # Read metadata. By default, this uses a RangeIndex (0..N-1) for rows.
#     metadata_path = original_dir / 'rgb_metadata.csv'
#     df = pd.read_csv(metadata_path)

#     # 1) Filter out rows whose 'category' is not in the known list of 62 categories.
#     df = df[df['category'].isin(VALID_CATEGORIES)].copy()

#     # 2) Ensure the DataFrame's row index is numeric 0..M-1
#     #    Then store the old row positions in a new column 'img_id'
#     df.reset_index(inplace=True)
#     df.rename(columns={'index': 'img_id'}, inplace=True)

#     # 3) Sample up to `samples_per_class` from each category (with replacement if needed).
#     df_small = (
#         df.groupby('category', group_keys=False)
#           .apply(lambda x: x.sample(n=samples_per_class,
#                                     replace=(len(x) < samples_per_class),
#                                     random_state=42))
#           .reset_index(drop=True)
#     )

#     # 4) Copy each sampled image from the original directory to the new one.
#     #    The 'img_id' column now tells us which row it originally was, so the file is `rgb_img_{img_id}.png`.
#     missing_count = 0
#     for _, row in df_small.iterrows():
#         img_id = row['img_id']
#         src = original_dir / 'images' / f"rgb_img_{img_id}.png"
#         dst = target_dir / 'images' / f"rgb_img_{img_id}.png"
#         if src.exists():
#             shutil.copy(src, dst)
#         else:
#             missing_count += 1
#             # Print a warning if the file doesn't exist
#             print(f"Warning: {src} not found, skipping.")

#     # 5) Save the new metadata CSV in the target directory
#     df_small.to_csv(target_dir / 'rgb_metadata.csv', index=False)

#     print(f"\nDone! Created a smaller FMoW dataset in '{target_dir}'")
#     print(f"   - Up to {samples_per_class} samples per class.")
#     if missing_count > 0:
#         print(f"   - {missing_count} images were not found and skipped.")

# if __name__ == "__main__":
#     # Example usage:
#     create_fmow_subset(
#         original_dir="data/fmow_v1.1",
#         target_dir="data/fmow_small",
#         samples_per_class=200
#     )

# import os
# import shutil
# import pandas as pd
# from pathlib import Path

# # List of valid categories (the official 62 classes)
# VALID_CATEGORIES = [
#     "airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", "archaeological_site", 
#     "barn", "border_checkpoint", "burial_site", "car_dealership", "construction_site", "crop_field", "dam", 
#     "debris_or_rubble", "educational_institution", "electric_substation", "factory_or_powerplant", 
#     "fire_station", "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station", 
#     "helipad", "hospital", "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", 
#     "military_facility", "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", 
#     "park", "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", 
#     "railway_bridge", "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", 
#     "single-unit_residential", "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", 
#     "surface_mine", "swimming_pool", "toll_booth", "tower", "tunnel_opening", "waste_disposal", 
#     "water_treatment_facility", "wind_farm", "zoo"
# ]

# def create_fmow_subset(original_dir, target_dir, samples_per_class=200):
#     """
#     Create a smaller FMoW dataset with up to `samples_per_class` images per valid category.
    
#     The new subset will be saved to `target_dir` with:
#       - A subfolder 'images/' containing the sampled images renamed as rgb_img_0.png, rgb_img_1.png, etc.
#       - A new rgb_metadata.csv whose rows (with a RangeIndex) match these new image names.
#       - A copy of country_code_mapping.csv from the original folder.
    
#     Args:
#         original_dir (str or Path): Path to the original FMoW folder (e.g., data/fmow_v1.1)
#         target_dir (str or Path): Path to save the new subset (e.g., data/fmow_small)
#         samples_per_class (int): Number of images per class to sample (default is 200)
#     """
#     original_dir = Path(original_dir)
#     target_dir = Path(target_dir)
#     (target_dir / 'images').mkdir(parents=True, exist_ok=True)

#     # Read the original metadata without setting an index.
#     metadata_path = original_dir / 'rgb_metadata.csv'
#     df = pd.read_csv(metadata_path)
    
#     # Create a new column 'orig_id' that is simply the row number (assumed to correspond to the image file name).
#     df["orig_id"] = df.index

#     # Filter to keep only rows whose 'category' is in the valid list.
#     df = df[df["category"].isin(VALID_CATEGORIES)].copy()

#     # Sample up to `samples_per_class` rows per category (with replacement if needed).
#     df_sampled = df.groupby("category", group_keys=False).apply(
#          lambda x: x.sample(n=samples_per_class, replace=(len(x) < samples_per_class), random_state=42)
#     )
#     df_sampled = df_sampled.reset_index(drop=True)
    
#     # Create a new sequential ID for the subset (this will be the new file name index).
#     df_sampled["new_id"] = range(len(df_sampled))
    
#     # Copy each sampled image using the original ID, renaming it with the new ID.
#     missing_count = 0
#     for _, row in df_sampled.iterrows():
#         orig_id = row["orig_id"]
#         new_id = row["new_id"]
#         src = original_dir / 'images' / f"rgb_img_{orig_id}.png"
#         dst = target_dir / 'images' / f"rgb_img_{new_id}.png"
#         if src.exists():
#             shutil.copy(src, dst)
#         else:
#             missing_count += 1
#             print(f"Warning: {src} not found, skipping.")

#     # Drop temporary columns and reset the index so that the new CSV has a default RangeIndex.
#     df_sampled = df_sampled.drop(columns=["orig_id", "new_id"]).reset_index(drop=True)
    
#     # Save the new metadata CSV.
#     df_sampled.to_csv(target_dir / 'rgb_metadata.csv', index=False)

#     # Also copy the auxiliary file country_code_mapping.csv so that the dataset loader finds it.
#     cc_src = original_dir / "country_code_mapping.csv"
#     cc_dst = target_dir / "country_code_mapping.csv"
#     if cc_src.exists():
#         shutil.copy(cc_src, cc_dst)
#     else:
#         print(f"Warning: {cc_src} not found.")

#     print(f"\nDone! Created a smaller FMoW dataset in '{target_dir}' with up to {samples_per_class} samples per class.")
#     if missing_count > 0:
#         print(f"{missing_count} images were not found and skipped.")

# if __name__ == "__main__":
#     create_fmow_subset(
#         original_dir="data/fmow_v1.1",
#         target_dir="data/fmow_small_v2",
#         samples_per_class=200
#     )

# ----filter DomainNet----
import os
import shutil

# List of classes you want to keep
class_names = [
    "airplane", "clock", "axe", "ball", "bicycle", "bird", "strawberry",
    "flower", "pizza", "bracelet", "bus", "cello", "bucket", "butterfly", "cup"
]

# Paths to source and target directories
source_dir = "/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data/domain_net/"
target_dir = "/mnt/VOL6/fangzhou/local/trainningcode/zilin_dg/NoiseRobustDG-main/data/domain_net_subset/"

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Loop over each domain directory in the source
for domain in os.listdir(source_dir):
    domain_path = os.path.join(source_dir, domain)
    
    # Make sure we're dealing with a directory (skip files)
    if os.path.isdir(domain_path):
        # Create the corresponding domain folder in the target directory
        new_domain_path = os.path.join(target_dir, domain)
        os.makedirs(new_domain_path, exist_ok=True)
        
        # Loop over each subdirectory (class) in this domain
        for class_dir in os.listdir(domain_path):
            class_path = os.path.join(domain_path, class_dir)
            
            # Check if the subdirectory is in our list of classes to keep
            if os.path.isdir(class_path) and class_dir in class_names:
                target_class_path = os.path.join(new_domain_path, class_dir)
                
                # Copy the entire class subdirectory to the new domain folder
                # For Python 3.8+, you can use dirs_exist_ok=True:
                # shutil.copytree(class_path, target_class_path, dirs_exist_ok=True)
                
                # For compatibility with older Python versions, you can:
                if not os.path.exists(target_class_path):
                    shutil.copytree(class_path, target_class_path)
                else:
                    print(f"Directory already exists, skipping: {target_class_path}")
                    
print("Finished copying selected classes.")
