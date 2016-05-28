from bs4 import BeautifulSoup as bso
import os
from os import path
import threading
import time
import argparse

def convert_one(input_file):
    
    soup = bso(open(input_file), 'xml')

    newsoup = bso('<annotation></annotation>', 'xml')

    ann = newsoup.annotation
    
    filename = soup.annotation.filename.string
    folder = soup.annotation.folder.string

    folder_tag = newsoup.new_tag('folder')
    filename_tag = newsoup.new_tag('filename')

    folder_tag.string = "VOC10"
    filename_tag.string = filename

    ann.append(folder_tag)
    ann.append(filename_tag)


    source_tag = newsoup.new_tag('source')
    db_tag = newsoup.new_tag('database')
    db_tag.string = 'The VOC2007 Database'
    src_ann_tag = newsoup.new_tag('annotation')
    src_ann_tag.string = 'PASCAL VOC2007'
    img_tag = newsoup.new_tag('image')
    img_tag.string = 'flickr'

    source_tag.append(db_tag)
    source_tag.append(src_ann_tag)
    source_tag.append(img_tag)

    ann.append(source_tag)

    height = ann.find('nrows') #height
    width = ann.find('ncols') #width

    if height is not None and width is not None:
        size_tag = newsoup.new_tag('size')
        width_tag = newsoup.new_tag('width')
        width_tag.string = width.string
        height_tag = newsoup.new_tag('height')
        height_tag.string = height.string
        depth_tag = newsoup.new_tag('depth')
        depth_tag.string = '3'

        size_tag.append(width_tag)
        size_tag.append(height_tag)
        size_tag.append(depth_tag)
        ann.append(size_tag)

    segment_tag = newsoup.new_tag('segmented')
    segment_tag.string = '1'

    ann.append(segment_tag)

    objects = soup.find_all('object')

    for obj in objects:

        deleted = obj.find('deleted')

        if deleted.string == '1':
            continue

        object_tag = newsoup.new_tag('object')

        name = obj.find('name').string

        name_tag = newsoup.new_tag('name')
        name_tag.string = name

        pose_tag = newsoup.new_tag('pose')
        pose_tag.string = 'Unspecified'
        trunc_tag = newsoup.new_tag('truncated')
        trunc_tag.string = '0'
        difficult_tag = newsoup.new_tag('difficult')
        difficult_tag.string = '0'


        bndbox_tag = newsoup.new_tag('bndbox')

        x_list = []
        y_list = []
        for x in obj.find_all('x'):
            x_list.append(int(x.string))

        for y in obj.find_all('y'):
            y_list.append(int(y.string))

        x_max, x_min = max(x_list), min(x_list)
        y_max, y_min = max(y_list), min(y_list)
        
        xmin_tag = newsoup.new_tag('xmin')
        xmin_tag.string = str(x_min)

        ymin_tag = newsoup.new_tag('ymin')
        ymin_tag.string = str(y_min)

        xmax_tag = newsoup.new_tag('xmax')
        xmax_tag.string = str(x_max)

        ymax_tag = newsoup.new_tag('ymax')
        ymax_tag.string = str(y_max)

        bndbox_tag.append(xmin_tag)
        bndbox_tag.append(ymin_tag)
        bndbox_tag.append(xmax_tag)
        bndbox_tag.append(ymax_tag)

        object_tag.append(name_tag)
        object_tag.append(pose_tag)
        object_tag.append(trunc_tag)
        object_tag.append(difficult_tag)
        object_tag.append(bndbox_tag)

        ann.append(object_tag)

    return newsoup.encode_contents()


def convert_list(file_list, outdir):
    for file in file_list:
        file = file.replace('\n','')
        xml_filename = file.split('/')[-1]
        new_xml_str = convert_one(file)
        filewriter = open(path.join(outdir,xml_filename), 'w')
        filewriter.write(new_xml_str)
        filewriter.flush()
        filewriter.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Arguments...')
    parser.add_argument('input', type=str,
                    help='Annotations list file')
    parser.add_argument('outdir', type=str,
                        help='Annotations output directory')
    parser.add_argument('threads', type=int,
                        help='number of threads')

    args = parser.parse_args()

    return args


class ConvertThread(threading.Thread):
    def __init__(self, threadID, file_list, outdir):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.file_list = file_list
        self.outdir = outdir
    def run(self):
        convert_list(self.file_list, self.outdir)


def main():
    args = parse_args()
    input_file = open(args.input,'r')
    outdir = args.outdir
    num_t = args.threads

    print "Convert MIT-CSAIL to PASC-VOC"

    lines = input_file.readlines()
    sub_size = len(lines) / num_t

    lists = []

    for i in range(0,num_t):
        sub_list = lines[i*sub_size:(i+1)*sub_size]
        lists.append(sub_list)

    lists[num_t - 1] = lines[sub_size*(num_t -1) : ]


    for idx,l in enumerate(lists):
        try:
            thread = ConvertThread(str(idx),l,outdir)
            thread.start()
        except:
           print "Error: unable to start thread"





if __name__ == "__main__":
    main()


#convert("can.xml")
