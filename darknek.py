 
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


    app.run(host="127.0.0.1",port=5000,debug=True)
