'''
Data sampling/augmentation class that will produce a batch for VAE
training from a list of image and points of interest
'''
import numpy as np
import argparse
import os
import cv2
import random
import time
import threading

global lock

class rgb_crop_batcher:
    def __init__(self,list_path,batch_size,crop_size):
        self.imgs = []
        self.pois = []
        self.batch_size = batch_size
        self.crop_size = crop_size
        #self.batch = np.zeros(shape=(batch_size,crop_size,crop_size,3),dtype='f')
        self.batches = []
        self.idx = 0
        self.running = True
        self.lock = threading.Lock()

        with open(list_path) as f:
            img_paths = f.read().splitlines()[:]
            for count, ip in enumerate(img_paths):
                print('loading img {}/{}'.format(count+1,len(img_paths)),end='\r')
                # Load images
                bgr = cv2.imread(ip)
                #rgb = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
                self.imgs.append(bgr)

                # Load annotations
                poi_path = ip[:-3] + "csv"
                poi = []
                with open(poi_path) as poi_file:
                    points = []
                    for poi in poi_file.read().splitlines():
                        x, y = poi.split(",")
                        points.append([int(x),int(y)])
                    self.pois.append(points)
        self.num_imgs = len(self.imgs)

        print('done loading {} imgs'.format(len(img_paths)))


    def show_example_with_box(self,idx):
        tmp = self.imgs[idx].copy()
        p1, p2 = self.roi_from_poi(self.pois[idx][0])
        cv2.rectangle(tmp, p1,p2,(0,255,0),1)
        cv2.imshow("example",tmp)

    def show_example_from_batch(self,idx):
        tmp = self.imgs[idx].copy()
        p1, p2 = self.roi_from_poi(self.pois[idx][0])
        cv2.rectangle(tmp, p1,p2,(0,255,0),1)
        cv2.imshow("example",tmp)

    def roi_from_poi(self,p):
        return (p[0] - self.crop_size/2, p[1] - self.crop_size/2), (p[0] + self.crop_size/2, p[1] + self.crop_size/2)

    def crop_sample(self,img,cen):
        x = int(cen[0])
        y = int(cen[1])

        width, height, _ = img.shape

        x_min = x-int(self.crop_size/2)
        x_max = x_min+self.crop_size
        y_min = y-int(self.crop_size/2)
        y_max = y_min+self.crop_size
        #print("x_min: {}, x_max {}, y_min {}, y_max {}".format(x_min,x_max,x_min,y_max))
        if(x_min >= 0 and x_max < width and y_min >= 0 and y_max < height):
            crop = img[y_min:y_max,x_min:x_max,:]
            w, h, _ = crop.shape
            if w != self.crop_size or h != self.crop_size:
                crop = cv2.resize(crop, (self.crop_size, self.crop_size))
            return crop.astype(np.float)/255.0
        return []

    def augment_sample(self,img,cen):
        # scaling augmentation
        scale_factor = random.uniform(0.7, 1.3)
        x = int(cen[0] * scale_factor)
        y = int(cen[1] * scale_factor)
        img = cv2.resize(img, dsize=None, fx=scale_factor, fy=scale_factor)

        # TODO: rot, skew, noise

        # cropping, randomize box jitter
        width, height, _ = img.shape
        x = x + random.randint(-15, 15)
        y = y + random.randint(-15, 15)
        #print("x: {}, y {}, width {}, height {}".format(x,y,width,height))
        x_min = x-int(self.crop_size/2)
        x_max = x_min+self.crop_size
        y_min = y-int(self.crop_size/2)
        y_max = y_min+self.crop_size
        #print("x_min: {}, x_max {}, y_min {}, y_max {}".format(x_min,x_max,x_min,y_max))
        if(x_min >= 0 and x_max < width and y_min >= 0 and y_max < height):
            crop = img[y_min:y_max,x_min:x_max,:]
            if random.randint(0,1):
                crop = cv2.flip(crop, 1) # horizontal flip, imitation left and right sides
            w, h, _ = crop.shape
            if w != self.crop_size or h != self.crop_size:
                crop = cv2.resize(crop, (self.crop_size, self.crop_size))
            return crop.astype(np.float)/255.0
        return []

    def make_batch(self):
        #self.batch = []
        batch_idx = 0
        batch = np.zeros(shape=(self.batch_size,self.crop_size,self.crop_size,3),dtype='f')

        while batch_idx < self.batch_size:
            img_idx = random.randint(0,self.num_imgs-1)
            if len(self.pois[img_idx]) > 0:
                crop = self.augment_sample(self.imgs[img_idx].copy(), self.pois[img_idx][random.randint(0,len(self.pois[img_idx])-1)])
            #else:
                #crop = self.augment_sample(self.imgs[idx].copy(), (random.randint(100,400),random.randint(100,400)))
                if len(crop) > 0:
                    batch[batch_idx] = crop
                    batch_idx = batch_idx + 1
        return batch


    def make_batches(self):
        while(self.running):
            num = len(self.batches)
            #print(num)
            if num < 50:
                batch = self.make_batch()
                with self.lock:
                    self.batches.append(batch)
            else:
                time.sleep(1)
            #time.sleep(0.1)

if __name__ == "__main__":
    """
    Main function for executing the .py script.
    Command:
        -p path/<filename>.npy
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--list", type=str,
                    help="list of img files")
    args = vars(ap.parse_args())

    rcb = rgb_crop_batcher(args["list"], 250, 64)


    # store crops
    selection_idx = 0
    num_selection = 50
    selection = np.zeros(shape=(num_selection,rcb.crop_size,rcb.crop_size,3),dtype='f')

    while selection_idx < num_selection:
        img_idx = random.randint(0,rcb.num_imgs-1)
        if len(rcb.pois[img_idx]) > 0:
            crop = rcb.crop_sample(rcb.imgs[img_idx].copy(), rcb.pois[img_idx][random.randint(0,len(rcb.pois[img_idx])-1)])
            if len(crop) > 0:
                selection[selection_idx] = crop
                selection_idx = selection_idx + 1

    print(selection.shape)
    cv2.imshow("rgb",selection[0])
    cv2.waitKey(0)

    '''
    cv2.namedWindow('example', cv2.WINDOW_NORMAL)

    threads = [threading.Thread(target=rcb.make_batches, daemon=True) for x in range(1)]
    for t in threads:
        t.start()

    while(True):
        #bc.show_example_with_box(random.randint(0,len(self.imgs)-1))
        #t0 = time.time()
        #batches.append(rcb.make_batch())
        #print("time: {}".format(time.time()-t0))
        print("number of batches in list: {}".format(len(rcb.batches)))
        if len(rcb.batches):
            test_samples = rcb.batches.pop(-1)
            cv2.imshow("example",test_samples[random.randint(0,rcb.batch_size-1)]) # show random example from batch

        #np.save("../data/np_batch_{}.npy".format(bc.crop_size), np.array(bc.batch))

        #np.save("../../data/example.npy", np.array(rcb.batch))

        k = cv2.waitKey(300) & 0xFF
        if k == 27:
            rcb.running = False
            break
    '''
