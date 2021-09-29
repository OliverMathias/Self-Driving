import os.path

#every run, timestamps new download folder and transfers folder from chip


def validateChip():
    print("Checking for SD Card...")
    chipPath = "/media/oliver/rootfs/home/pi/Desktop/"
    isdir = os.path.isdir(chipPath+"data/")
    isfile = os.path.isfile(chipPath+"data.txt")
    print("Data Folder Exist: ", isdir)
    print("Data Folder Exist: ", isdir)
    return (isdir and isfile)


if __name__ == '__main__':
    print(validateChip())
    #if it exists start moving to it's own folder
