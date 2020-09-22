import os
from imghdr import what
from PIL import Image

def ListFilesToTxt(dir, file, wildcard, recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname = os.path.join(dir, name)
        if os.path.isdir(fullname) & recursion:
            ListFilesToTxt(fullname, file, wildcard, recursion)
        else:
            for ext in exts:
                if name.endswith(ext):
                    try:
                        img = Image.open(dir +'/' + name).convert('RGB')
                        file.write(dir +'/' + name + "\n")
                        break
                    except:
                        print(dir+'/'+name)

                    # file.write(dir +'/' + name + "\n")
                    # break


def Test(dir = 'None', outfile = 'None', wildcard = 'None'):

    file = open(outfile, "w")
    if not file:
        print("cannot open the file %s for writing" % outfile)

    ListFilesToTxt(dir, file, wildcard, 1)

    file.close()
#正常ct
Test(dir = 'data/Normal', outfile = 'Normal.txt', wildcard = '.JPG .png .jpg')
#新冠肺炎
Test(dir = 'data/NCP', outfile = 'COVID.txt', wildcard = '.JPG .png .jpg')
#普通肺炎
Test(dir = 'data/CP', outfile = 'CP.txt', wildcard = '.JPG .png .jpg')
