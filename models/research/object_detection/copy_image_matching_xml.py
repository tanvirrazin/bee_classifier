import os
import shutil

source_dir = '/data/bhuiyan/bee_vs_nobee/bee_data/'
destination_dir = '/data/bhuiyan/bee_vs_nobee/models/research/object_detection/images/'

def run_copy(src, dst):
    src_dir = os.path.join(source_dir, src)
    dest_dir = os.path.join(destination_dir, dst)
    print(src_dir)
    print(dest_dir)
    for xml_file_name in os.listdir(dest_dir):
        if xml_file_name.endswith('.xml'):
            image_file_basename = xml_file_name.split('.xml')[0]
    
            if os.path.exists("{}{}{}".format(dest_dir, image_file_basename, ".jpg")):
                print("{}{}{}".format(image_file_basename, ".jpg", " already exists"))    
            elif os.path.exists("{}{}{}".format(dest_dir, image_file_basename, ".JPG")):
                print("{}{}{}".format(image_file_basename, ".JPG", " already exists"))
            elif os.path.exists("{}{}{}".format(dest_dir, image_file_basename, ".jpeg")):
                print("{}{}{}".format(image_file_basename, ".jpeg", " already exists"))
            elif os.path.exists("{}{}{}".format(dest_dir, image_file_basename, ".JPEG")):
                print("{}{}{}".format(image_file_basename, ".JPEG", " already exists"))
            else:
                from_file = "{}{}".format(src_dir, image_file_basename)
                to_file = "{}{}".format(dest_dir, image_file_basename)
                try:
                    shutil.copyfile(from_file+".jpg", to_file+".jpg")
                    print("Copied " + image_file_basename+".jpg")
                except:
                    try:
                        shutil.copyfile(from_file+".JPG", to_file+".JPG")
                        print("Copied " + image_file_basename+".JPG")
                    except:
                        try:
                            shutil.copyfile(from_file+".jpeg", to_file+".jpeg")
                            print("Copied " + image_file_basename+".jpeg")
                        except:
                            try:
                                shutil.copyfile(from_file+".JPEG", to_file+".JPEG")
                                print("Copying " + image_file_basename+".jpg")
                            except:
                                print("Not found")

for src in ['bumblebee/', 'honeybee/', 'centris/', 'rustybee/', 'andrena/', 'anthidium/', 'mallophora/', 'megachile/', 'melissodes/']:
    # run_copy(src, 'train/')
    # run_copy(src, 'test/')
    run_copy(src, 'masked_inat/')

