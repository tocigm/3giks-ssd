#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Distributed under terms of the MIT license.

import sys, os
import re
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

if __name__ == "__main__":


    argv = sys.argv

    if len(argv) != 2:
        print "Usage: python plot_loss.py <log_file_path>"
        quit()

    
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    loss_bbox = []
    loss_cls = []
    rpn_cls_loss = []
    rpn_loss_bbox = []

    f = open(argv[1])
    outfile = "loss_plot_%s.png" % os.path.splitext(os.path.basename(argv[1]))[0]
    lines = f.readlines()

    for index in range(0,len(lines)):
        line = lines[index].strip()
        if "Iteration" in line and "loss =" in line:
            #print lines[index]        
            step = int(re.search('Iteration ([0-9]+)', line).groups()[0])
            tr_l = float(re.search('loss = ([0-9\.]+)', line).groups()[0])
            train_loss.append([step, tr_l])
            line = lines[index + 1].strip()
            tr_l = float(re.search('mbox_loss = ([0-9\.]+)', line).groups()[0])
            loss_bbox.append([step, tr_l])
            

        # line = f.readline().strip()

    conv_r = 10

    train_loss = np.asarray(train_loss)
    train_loss_ave = np.convolve(train_loss[:,1], np.ones(int(conv_r))/conv_r, 'same')
    
    loss_bbox = np.asarray(loss_bbox)
    loss_bbox_ave = np.convolve(loss_bbox[:,1], np.ones(int(conv_r))/conv_r, 'same')


    if not len(train_loss) > 2:
        quit()

    fig, ax1 = plt.subplots(figsize=(12,9))
    ax1.set_xlim([-1, max(train_loss[:, 0])])
    ax1.set_ylim([0.0, 15.0])
    ax1.set_xlabel('iterator')
    ax1.set_ylabel('loss')

    ax1.plot(train_loss[:, 0], train_loss_ave,
                label='train loss', c='b')
    ax1.plot(loss_bbox[:, 0], loss_bbox_ave,
                label='mbox loss', c='g')
   
    ax1.axhline(y=5, color="black", ls="dotted")
    ax1.axhline(y=3, color="black", ls="dotted")
    ax1.axhline(y=1, color="black", ls="dotted")

    ax1.legend(bbox_to_anchor=(0.25, -0.1), loc=9)
    #ax2.legend(bbox_to_anchor=(0.75, -0.1), loc=9)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.title("loss of SSD: \n%s" %  os.path.basename(argv[1]))
    plt.savefig(outfile, bbox_inches='tight')
    print "Figure saved: %s" % outfile
    plt.pause(30)
    plt.close()

    f.close()
