#!/home/avalocal/anaconda3/bin/python

import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import pandas as pd
from collections import Counter


FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # yolov5 strongsort root directory
# print(type(FILE.parents[0]))
ROOT = Path("/home/unreal/projects/Elijah/StrongSort/catkin_workspace/src/yolov7_strongsort_ros/src/Yolov7_StrongSORT_OSNet")
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print(ROOT)

from yolov5.models.common import DetectMultiBackend
try:
    from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
except:
    import sys
    sys.path.append('yolov5/utils')
    from dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from PIL import Image as IM

import torchvision.transforms as transforms
import rospy
import ros_numpy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from yolov5.utils.augmentations import letterbox

#VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes



class track():
    def __init__(self):
        self.source='0'
        self.yolo_weights=WEIGHTS / 'bsz_16_best.pt' # model.pt path(s),
        self.strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt' # model.pt path,
        self.config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml'
        self.imgsz=(2048, 2048) # inference size (height, width)
        self.conf_thres=0.4  # confidence threshold
        self.iou_thres=0.4  # NMS IOU threshold
        self.max_det=1000 # maximum detections per image
        self.device="0" # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.show_vid=True  # show results
        self.save_txt=True  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.save_vid=False  # save confidences in --save-txt labels
        self.nosave=False # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project=ROOT / 'runs/track'  # save results to project/name
        self.name='exp' # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False # hide confidences
        self.hide_class=False  # hide IDs
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        self.count=True,  # get counts of every object
        

        # Directories
        if not isinstance(self.yolo_weights, list):  # single yolo model
            print("----",self.imgsz,"------")
            exp_name =self.yolo_weights.stem
        elif type(self.yolo_weights) is list and len(self.yolo_weights) == 1:  # single models after --yolo_weights
            exp_name = Path(self.yolo_weights).stem
            self.yolo_weights = Path(self.yolo_weights)
        else:  # multiple models after --yolo_weights
            exp_name = 'ensemble'
        exp_name = self.name if self.name else exp_name + "_" + self.strong_sort_weights.stem
        self.save_dir = increment_path(Path(self.project) / exp_name, exist_ok=self.exist_ok)  # increment run
        (self.save_dir / 'tracks' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
       
        # Load model
        self.device = select_device(self.device)
        
        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.model = DetectMultiBackend(self.yolo_weights, device=self.device, dnn=self.dnn, data=None, fp16=self.half)
        
        self.names, = self.model.names,
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.stride = self.model.stride  # model stride
        self.imgsz = check_img_size(self.imgsz[0], s=self.stride)  # check image size
        cudnn.benchmark = True

        # Run inference - uncomment to check if model is runnin properly
        #if self.device.type != "cpu" :
        #    self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters())))  # run once
        #    print("interference is running")
        self.image_sub = rospy.Subscriber("/image_raw_tr0", Image, self.camera_callback, queue_size=1, buff_size=2**24) #"/camera_fr/image_raw"
        self.trackPublish = rospy.Publisher("trackResult", Image, queue_size=1)  

        # initialize StrongSORT
        self.cfg = get_config()
        self.cfg.merge_from_file(self.config_strongsort)

        # Create as many strong sort instances as there are video sources
        self.nr_sources = 1
        
        txt_path = [None] * self.nr_sources

        #self.strongsort_list = []
        #for i in range(self.nr_sources):
        self.strongsort_list= StrongSORT(
                self.strong_sort_weights,
                self.device,
                self.half,
                max_dist=self.cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=self.cfg.STRONGSORT.MAX_AGE,
                n_init=self.cfg.STRONGSORT.N_INIT,
                nn_budget=self.cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=self.cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=self.cfg.STRONGSORT.EMA_ALPHA,
            )
    

        self.strongsort_list.model.warmup()
        rate=rospy.Rate(50)
        rate.sleep()
        
        rospy.spin()

    def preProccess(self, img):
        device= torch.device('cuda:0')
        img=torch.from_numpy(img).to(device)
        img=img.half() if self.half else img.float()  # uint8 to fp16/32
        img=img/255.0
        if len(img.shape)==3:
            #img=img.unsqueeze(0)
            img=img[None]
        return img

    '''def getTopicName(self, topic, datatype, md5sum, msg_def, header):
        
        if(datatype=="sensor_msgs/Image"):
            cameraTopic=""
        else:
            print("camera output is not captured correctlly")'''

    #@torch.no_grad()
    def camera_callback(self, data):
        start_time_seg = rospy.Time.now().to_sec()

        self.img = ros_numpy.numpify(data)
       
        img0=self.img
        #img0=cv2.resize(img0, (640,640))
        self.img_size=(640,640)

        img=letterbox(img0, self.img_size, stride=self.stride)[0]
        #print(np.shape(img)) 
        img = img[:, :, ::-1].transpose(2, 0, 1)  #BGR to RGB
        img = np.ascontiguousarray(img)
        img=self.preProccess(img)
           
        
        outputs = [None] * self.nr_sources
        

        # Run tracking
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        curr_frames, prev_frames = [None] * self.nr_sources, [None] * self.nr_sources
       
        s = ''
        t1 = time_sync()
        
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        with torch.no_grad():
            pred = self.model(img)
            #print("pred after yolov7", pred[0])
            print(" \n")
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms)
            dt[2] += time_sync() - t3
            #print("---pred after NMS---",pred)

            #show_vid=False
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                im0=img0 #s.copy #[i].copy()
                
               
                curr_frames[i] = im0

                txt_path = str(self.save_dir / 'tracks' / 'im')  # im.txt

                s += '%gx%g ' % img.shape[2:]  # print string


                annotator = Annotator(im0, line_width=2, pil=not ascii)
                if self.cfg.STRONGSORT.ECC and prev_frames[i]!=None:  # camera motion compensation
                    self.strongsort_list.tracker.camera_update(prev_frames[i], curr_frames[i])
                    
                    
                if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                   
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    #Print results
                    print_result=False
                    if print_result:
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string


                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to strongsort
                    t4 = time_sync() ##############
                    outputs[i] = self.strongsort_list.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync() ##############
                    #print("output of deepsort",outputs[i])
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    if len(outputs[i]) > 0:
                        for j, (output, conf) in enumerate(zip(outputs[i], confs)):
        
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            if self.save_txt:
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 11 + '\n') % (start_time_seg, cls, id, bbox_left,  # MOT format
                                                                bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                            if self.show_vid:  # Add bbox to image
                                c = int(cls)  # integer class
                                id = int(id)  # integer id
                                label = None if self.hide_labels else (f'{id} {self.names[c]}' if self.hide_conf else \
                                    (f'{id} {conf:.2f}' if self.hide_class else f'{id} {self.names[c]} {conf:.2f}'))
                                annotator.box_label(bboxes, label, color=colors(c, True))
                     
                    print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

                else:
                    self.strongsort_list.increment_ages()
                    print('No detections')

                if self.count:
                    itemDict={}
                    ## NOTE: this works only if save-txt is true
                    try:
                        df = pd.read_csv(txt_path +'.txt' , header=None, delim_whitespace=True)
                        df = df.iloc[:,0:3]
                        df.columns=["frameid" ,"class","trackid"]
                        df = df[['class','trackid']]
                        df = (df.groupby('trackid')['class']
                                .apply(list)
                                .apply(lambda x:sorted(x))
                                ).reset_index()

                        df.columns = ["trackid","class"]
                        df['class']=df['class'].apply(lambda x: Counter(x).most_common(1)[0][0])
                        vc = df['class'].value_counts()
                        vc = dict(vc)

                        vc2 = {}
                        for key, val in enumerate(self.names):
                            vc2[key] = val
                        itemDict = dict((vc2[key], value) for (key, value) in vc.items())
                        itemDict  = dict(sorted(itemDict.items(), key=lambda item: item[0]))
                        # print(itemDict)

                    except:
                        pass

                    if self.save_txt:
                        ## overlay
                        display = im0.copy()
                        h, w = im0.shape[0], im0.shape[1]
                        x1 = 10
                        y1 = 10
                        x2 = 10
                        y2 = 70

                        txt_size = cv2.getTextSize(str(itemDict), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        cv2.rectangle(im0, (x1, y1 + 1), (txt_size[0] * 2, y2),(0, 0, 0),-1)
                        cv2.putText(im0, '{}'.format(itemDict), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX,0.7, (210, 210, 210), 2)
                        cv2.addWeighted(im0, 0.7, display, 1 - 0.7, 0, im0)

                # Stream results
                rviz=True
                if rviz:
                    image_out=im0[...,::-1]
                    img_out = IM.fromarray(image_out,'RGB')
                    msg=Image()
                    msg.header.stamp=rospy.Time.now()
                    msg.height=img_out.height
                    msg.width=img_out.width
                    msg.encoding="rgb8"
                    msg.is_bigendian=False
                    msg.step=3*img_out.width
                    msg.data=np.array(img_out).tobytes()
                    
                    self.trackPublish.publish(msg)

                prev_frames[i] = curr_frames[i]

            # Print results
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            #print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, self.img_size, self.img_size)}' % t)
            
            if self.update:
                strip_optimizer(self.yolo_weights)  # update model (to fix SourceChangeWarning)

            time_end_seg = rospy.Time.now().to_sec()
                            
            print("Total Processing time :: ", ( time_end_seg- start_time_seg ))        

    


if __name__ == "__main__":
    rospy.init_node('sortYoloRosNode')
    #check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    track()
    #run(**vars(opt))
    
