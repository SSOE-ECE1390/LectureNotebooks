# Modified from https://medium.com/@smilesajid14/simple-tracking-307aff2f80f3

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

# local module
import video
from video import presets


class App(object):
    def __init__(self, video_src):
        self.cam = video.create_capture(video_src, presets['cube'])
        _ret, self.frame = self.cam.read()
        cv.namedWindow('tracking')
        cv.setMouseCallback('tracking', self.onmouse)

        self.selection = None
        self.drag_start = None
        self.track_window = None

    def settracker(self,trackIdx):
        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        tracker_type = tracker_types[trackIdx]
        self.trackmethod=tracker_type
        if tracker_type == 'BOOSTING':
            self.tracker = cv.legacy.TrackerBoosting_create()
        if tracker_type == 'MIL':
            self.tracker = cv.legacy.TrackerMIL_create()
        if tracker_type == 'KCF':
            self.tracker = cv.legacy.TrackerKCF_create()
        if tracker_type == 'TLD':
            self.tracker = cv.legacy.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            self.tracker = cv.legacy.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            self.tracker = cv.legacy.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            self.tracker = cv.legacy.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            self.tracker = cv.legacy.TrackerCSRT_create()

        ok = self.tracker.init(self.frame,self.track_window)    

    def onmouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)
            
        if event == cv.EVENT_LBUTTONUP:
            self.drag_start = None
            self.track_window = (xmin, ymin, xmax - xmin, ymax - ymin)
            self.settracker(2)
   
    def run(self):
        while True:
            _ret, self.frame = self.cam.read()
            vis = self.frame.copy()
            

            if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
                self.selection = None
               
                ok, self.track_window = self.tracker.update(self.frame)
                
                cv.putText(self.frame, self.trackmethod, (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                # Draw bounding box
                if ok:
                    
                # Tracking success
                    p1 = (int(self.track_window[0]), int(self.track_window[1]))
                    p2 = (int(self.track_window[0] + self.track_window[2]), int(self.track_window[1] + self.track_window[3]))
                    cv.rectangle(self.frame, p1, p2, (255,0,0), 2, 1)
                else :
                # Tracking failure
                    cv.putText(self.frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 

            cv.imshow('tracking',self.frame)

            ch = cv.waitKey(5)
            if ch == 27:
                break
            if (ch >=48 and ch<= 55):
                self.settracker(ch-48)

           
        cv.destroyAllWindows()


if __name__ == '__main__':
    print(__doc__)
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    print('0: BOOSTING')
    print('1: MIL')
    print('2: KCF')
    print('3: TLD')
    print('4: MEDIANFLOW')
    print('5: GOTURN')
    print('6: MOSSE')
    print('7: CSRT')
    
    App(video_src).run()
