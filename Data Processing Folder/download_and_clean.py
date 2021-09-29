import process_data_high_level
import numpy as np
import h5py

def main():
    print("hi")


if __name__ == '__main__':
    #main()
    current_date = "9-15-21"
    #import data & .txt file from sd card
    #process_data_high_level.copy_dirty_data_from_sd()

    #clean new folder
    #process_data_high_level.clean_conflicting_folder_and_create_empty(current_date)

    #clean valid name and image pairs into new folder with date name and data.txt inside it
    #process_data_high_level.save_clean_images_and_angles(current_date)

    #save new folder to zip (so we can upload to drive)
    #print("zip the data")
    #process_data_high_level.save_clean_data_to_zip(current_date)

    #delete from SD Card
    #process_data_high_level.clean_sd_card()

# Add unique images and angles to the master data folder and master txt document

    #all unique images (i.e. not in master) in every date save folder go into master
    #print("ALL imags not in master get saved there.")
    #process_data_high_level.copy_unique_images_and_values_to_master(current_date)

    #all any duplicates are deleted from the personal save folder (maybe cuz we can delete the old folders so no duplicates anyway)

    #turn master data folder and master txt docs into hdf5 arrays (ask to be augmented here, so we don't need to save double the images if we can augment them quickly)
    #print("save as h5")
    #process_data_high_level.augment_data_and_save_as_hdf5()

    #start training and save model
    process_data_high_level.load_data_normalize_and_train(current_date)
    print("Saving model")

    #delete data and txt working folder here
    print("remove folder")
    process_data_high_level.remove_working_data_and_file()
