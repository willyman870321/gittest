 
#encoding:utf-8
#!/usr/bin/env python
# -*- coding: utf-8 -*- 


from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort
from flask import request
import json
import base64
import argparse
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet
from flask import Flask, render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
import time

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="all1026_best.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="all1026.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="obj.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


def check_batch_shape(images, batch_size):
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def prepare_batch(images, network, channels=3):
    width = darknet.network_width(network)
    height = darknet.network_height(network)

    darknet_images = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        custom_image = image_resized.transpose(2, 0, 1)
        darknet_images.append(custom_image)

    batch_array = np.concatenate(darknet_images, axis=0)
    batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
    darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
    return darknet.IMAGE(width, height, channels, darknet_images)


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def batch_detection(network, images, class_names, class_colors,
                    thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
    image_height, image_width, _ = check_batch_shape(images, batch_size)
    darknet_images = prepare_batch(images, network)
    batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                     image_height, thresh, hier_thresh, None, 0, 0)
    batch_predictions = []
    for idx in range(batch_size):
        num = batch_detections[idx].num
        detections = batch_detections[idx].dets
        if nms:
            darknet.do_nms_obj(detections, num, len(class_names), nms)
        predictions = darknet.remove_negatives(detections, class_names, num)
        images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
        batch_predictions.append(predictions)
    darknet.free_batch_detections(batch_detections, batch_size)
    return images, batch_predictions


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = name.split(".")[:-1][0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


def batch_detection_example():
    args = parser()
    check_arguments_errors(args)
    batch_size = 3
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=batch_size
    )
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)



app = Flask(__name__)
@app.route("/yanwei")
def yanwei():

        #if args.input:
           # if index >= len(images):
               # break
           # image_name = images[index]
        #else:
        index = 0
        image_name = input("Enter Image Path: ")
        prev_time = time.time()
        image, detections = image_detection(
            image_name, network, class_names, class_colors, args.thresh
            )
        if args.save_labels:
            save_annotations(image_name, image, detections, class_names)
        darknet.print_detections(detections, args.ext_output)
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        #if not args.dont_show:
            #cv2.imshow('Inference', image)
            #if cv2.waitKey() & 0xFF == ord('q'):
               # break
        index += 1
        print("FPS: {}".format(fps))

	
        return str(detections)
@app.route('/123', methods=['GET'])
def index():
    return render_template("index2.html")
@app.route('/123', methods=['POST'])
def submit():
        f = request.files['file']
        basepath = os.path.dirname(__file__) 
        upload_path = os.path.join(basepath, 'data',secure_filename(f.filename)) 
        index = 0
        image_name = upload_path
        prev_time = time.time()
        image, detections = image_detection(
            image_name, network, class_names, class_colors, args.thresh
            )
        if args.save_labels:
            save_annotations(image_name, image, detections, class_names)
        darknet.print_detections(detections, args.ext_output)
        fps = int(1/(time.time() - prev_time))
        print("FPS: {}".format(fps))
        #if not args.dont_show:
            #cv2.imshow('Inference', image)
            #if cv2.waitKey() & 0xFF == ord('q'):
               # break
        index += 1
        print("FPS: {}".format(fps))

	
        return str(detections)    
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF','heic'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload')
def upload_test():
    return render_template('up.html')


# 上傳文件
@app.route('/up_photo', methods=['POST'], strict_slashes=False)
def api_upload():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files['photo']
    if f :
        fname = secure_filename(f.filename)
        #print fname
        ext = fname.rsplit('.', 1)[1]
        new_filename =time.strftime('%Y%m%d-%H%M%S-')+f.filename #Pic_str().create_uuid() + '.' + ext
        f.save(os.path.join(file_dir, new_filename))
    basepath = os.path.dirname(__file__)  
    upload_path = os.path.join(basepath,'upload',new_filename) 
    image_name = upload_path
    prev_time = time.time()
    image, detections = image_detection(
            image_name, network, class_names, class_colors, args.thresh
    )
    if args.save_labels:
         save_annotations(image_name, image, detections, class_names)
    darknet.print_detections(detections, args.ext_output)
    fps = int(1/(time.time() - prev_time))
    print("FPS: {}".format(fps))
        #if not args.dont_show:
            #cv2.imshow('Inference', image)
            #if cv2.waitKey() & 0xFF == ord('q'):
               # break」
    taisamax=0
    shexingmax=0
    shesemax=0
    taizhimax=0
    taizhi2max=0
    cc=0
    classname={}
    for itemm in detections:
       th=float(itemm[1])
       if itemm[0] == 'white' or itemm[0] == 'yellow'or itemm[0] == 'black':         
          if th>taisamax:
             taisamax=th
             taisa=itemm[0]           
       if itemm[0] == 'pang' or itemm[0] == 'shou'or itemm[0] == 'zhengchang':
          if th>shexingmax:
             shexingmax=th
             shexing=itemm[0]
       if itemm[0] =='lao'or itemm[0] == 'nen'or itemm[0] == 'dianci'or itemm[0] == 'yuban'or itemm[0] == 'yudian'or itemm[0] == 'liewen'or itemm[0] == 'chihen':
             classname[cc]=itemm[0]
             cc=cc+1
       if itemm[0] == 'danbaishe' or itemm[0] == 'danredshe'or itemm[0] == 'redshe'or itemm[0] == 'crimsonshe'or itemm[0] == 'zishe'or itemm[0] == 'anshe':
          if th>shesemax:
             shesemax=th
             shese=itemm[0]
       if itemm[0] == 'botai' or itemm[0] == 'houtai'or itemm[0] == 'nitai'or itemm[0] == 'futai'or itemm[0] == 'boluotai'or itemm[0] == 'wutai':
          if th>taizhimax:
             taizhimax=th
             taizhi=itemm[0]
       if itemm[0] == 'huatai'or itemm[0] == 'runtai'or itemm[0] == 'zaotai':
          if th>taizhi2max:
             taizhi2max=th
             taizhi2=itemm[0]
    classname[cc]=taisa
    classname[cc+1]=shexing
    classname[cc+2]=shese
    classname[cc+3]=taizhi
    classname[cc+4]=taizhi2
    for it in classname:
        print(classname[it])
   
    a=0
    a1=0
    b=0
    b1=0
    c=0
    c1=0
    c2=0
    e=0
    d=0
    d1=0
    d2=0
    e-0
    f=0
    f1=0
    Qideficiency=0
    Peaceful=0
    Bloodstasis=0
    Yangdeficiency=0
    Blooddeficiency=0
    Dampheat=0
    Phlegm=0
    Yindeficiency=0
    Depression=0
    sensitive=0
    for item in classname:
      #氣虛
      if classname[item] == 'danredshe'or classname[item] == 'liewen'or classname[item] == 'pang'or classname[item]== 'white'or classname[item] == 'botai'or classname[item] == 'boluotai':
          if classname[item] == 'liewen'or classname[item] == 'pang':
              if a==0:
                Qideficiency=Qideficiency+1
              a=a+1
          elif classname[item] == 'botai'or classname[item] == 'boluotai':
              if a1==0:
                Qideficiency=Qideficiency+1
              a1=a1+1           
          else:
            Qideficiency=Qideficiency+1
    #平和
      if classname[item] == 'danredshe'or classname[item] == 'zhengchang'or classname[item]== 'white'or classname[item] == 'botai' :
         Peaceful=Peaceful+1
    #血瘀
      if classname[item] == 'zishe' or classname[item] == 'anshe'or classname[item] == 'lao'or classname[item] == 'yuban'or classname[item] == 'yudian'or classname[item] == 'zaotai'  :
         if classname[item] == 'zishe' or classname[item] == 'anshe':
             if b==0:
                Bloodstasis=Bloodstasis+1 
             b=b+1
             
         elif classname[item] == 'lao'or classname[item] == 'yuban'or classname[item] == 'yudian':
             if b1==0:
                 Bloodstasis=Bloodstasis+1
             b1=b1+1
             
         else:
             Bloodstasis=Bloodstasis+1
    #陽虛
      if classname[item]== 'danbaishe'or classname[item] == 'chihen'or classname[item] == 'white'or classname[item] == 'huatai':
          Yangdeficiency=Yangdeficiency+1
    #血虛
      if classname[item]== 'danbaishe'or classname[item] == 'danbaishe'or classname[item] == 'nen'or classname[item] == 'dianci'or classname[item]== 'white'or classname[item] == 'botai'or classname[item] == 'zaotai':
         if classname[item] == 'danbaishe'or classname[item]== 'danbaishe':
             if c==0:
                Blooddeficiency=Blooddeficiency+1 
             c=c+1
            
         elif classname[item] == 'nen'or classname[item] == 'dianci' :
             if c1==0:
                 Blooddeficiency=Blooddeficiency+1
             c1=c1+1
            
         elif classname[item] == 'botai'or classname[item] == 'zaotai':
            if c2==0:
                Blooddeficiency=Blooddeficiency+1
            c2=c2+1
            
         else:
            Blooddeficiency=Blooddeficiency+1
    #濕熱
      if classname[item] == 'redshe'or classname[item]== 'zishe'or classname[item] == 'pang'or classname[item] == 'liewen'or classname[item] == 'yellow'or classname[item] == 'nitai'or classname[item] == 'futai':
         if  classname[item] == 'redshe'or classname[item] == 'zishe':
             if d==0:
                Dampheat=Dampheat+1
             d=d+1
         elif classname[item] == 'pang'or classname[item] == 'liewen' :
             if d1==0:
                Dampheat=Dampheat+1
             d1=d1+1
         elif classname[item] == 'nitai'or classname[item] == 'futai' :
             if d2==0:
                Dampheat=Dampheat+1
             d2=d2+1
         else:
             Dampheat=Dampheat+1
    #痰濕
      if classname[item] == 'redshe' or classname[item] == 'pang'or classname[item] == 'chihen'or classname[item] == 'white'or classname[item] == 'nitai' :
         if  classname[item] == 'pang'or classname[item] == 'chihen':
             if e==0:
                Phlegm=Phlegm+1
             e=e+1
             
         else:
             Phlegm=Phlegm+1
    #陰虛
      if classname[item] == 'zishe'or classname[item]== 'crimsonshe'or classname[item] == 'shou'or classname[item] == 'white' or classname[item] == 'zaotai'or classname[item] == 'wutai':
         if classname[item]== 'zishe'or classname[item]== 'crimsonshe' :
             if f==0:
                Yindeficiency=Yindeficiency+1
             f=f+1
         elif classname[item] == 'zaotai'or classname[item] == 'wutai':
             if f1==0:
                 Yindeficiency=Yindeficiency+1
             f1=f1+1
             
         else:
             Yindeficiency=Yindeficiency+1
    #氣郁
      if classname[item] == 'redshe'or classname[item] == 'white':
         Depression=Depression+1
    #敏感
      if classname[item] == 'redshe'or classname[item] == 'dianci'or classname[item]== 'boluotai':
         sensitive=sensitive+1
             
             
    Qideficiency=(Qideficiency/4)*100
    Peaceful=(Peaceful/4)*100
    Bloodstasis=(Bloodstasis/3)*100
    Yangdeficiency=(Yangdeficiency/4)*100
    Blooddeficiency=(Blooddeficiency/4)*100
    Dampheat=(Dampheat/4)*100
    Phlegm=(Phlegm/4)*100
    Yindeficiency=(Yindeficiency/4)*100
    Depression=(Depression/2)*100
    sensitive=(sensitive/3)*100

    Qideficiency=str(Qideficiency)
    Peaceful=str(Peaceful)
    Bloodstasis=str(Bloodstasis)
    Yangdeficiency=str(Yangdeficiency)
    Blooddeficiency=str(Blooddeficiency)
    Dampheat=str(Dampheat)
    Phlegm=str(Phlegm)
    Yindeficiency=str(Yindeficiency)
    Depression=str(Depression)
    sensitive=str(sensitive)
    
    print("氣虛:"+Qideficiency+"%")
    print("平和:"+Peaceful+"%")
    print("血瘀:"+Bloodstasis+"%")
    print("陽虛:"+Yangdeficiency+"%")
    print("血虛:"+Blooddeficiency+"%")
    print("濕熱:"+Dampheat+"%")
    print("痰濕:"+Phlegm+"%")
    print("陰虛:"+Yindeficiency+"%")
    print("氣郁:"+Depression+"%")
    print("敏感:"+sensitive+"%")
    
    app.config['JSON_AS_ASCII'] = False
    return jsonify(detections)#str(detections)
        #return jsonify({"success": 0, "msg": "上傳成功"})
    #else:
        #return jsonify({"error": 1001, "msg": "上傳失敗"})

if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    o=0
    if o==0:
        args = parser()
        check_arguments_errors(args)

        random.seed(3)  # deterministic bbox colors
        network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
        )

        images = load_images(args.input)

    o += 1
    app.run(host="127.0.0.1",port=5000,debug=True)
