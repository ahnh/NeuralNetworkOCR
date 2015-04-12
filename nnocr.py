# Reference: http://g.sweyla.com/blog/2012/mnist-numpy/

#!/usr/bin/env python
import os
import mnist
from pylab import *
from numpy import *
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader


def showImage(img):
    imshow(img, cmap=cm.gray)
    show()

if __name__ == "__main__":
    file = "3notanhx1.xml"
    print file
    if  os.path.isfile(file):
        print "reading network"
        net = NetworkReader.readFrom(file)
    else:
        train_ds = SupervisedDataSet(28 * 28, 1)
        train_images, train_labels = mnist.load_mnist(dataset='training')
        for image, label in zip(train_images, train_labels):
            train_ds.addSample(ravel(image), label)

        #net = buildNetwork(train_ds.indim, 98, 98, 49, train_ds.outdim, bias=True, hiddenclass=TanhLayer)
        net = buildNetwork(train_ds.indim, 98, train_ds.outdim, bias=True)
        trainer = BackpropTrainer(net, train_ds)

        print "start training"
        for i in range(1):
            trainer.trainEpochs(1)
            print "epoch: %4d" % trainer.totalepochs

    for i in range(0, 10):
        num = i
        test_ds = SupervisedDataSet(28 * 28, 1)
        test_images, test_labels = mnist.load_mnist(dataset='testing', digits=[num])
        for image, label in zip(test_images, test_labels):
            test_ds.addSample(ravel(image), label)

        p = net.activateOnDataset(test_ds)
        res = np.round( p )
        total = len(res)
        count = sum(res == num)
        print num,
        print " ", count/float(total)*100
    NetworkWriter.writeToFile(net, file)