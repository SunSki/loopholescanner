from __future__ import print_function
from google.cloud import vision
import cv2
import numpy as np
from math import sqrt, sin, cos, atan2, radians, pi
import Levenshtein



def google_vision_response(vision_content):
    client = vision.ImageAnnotatorClient()
    vision_image = vision.Image(content=vision_content) #bytes data is available
    response = client.text_detection(image=vision_image) #send request and get response
    return response


def imread4warp(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

#--------------------------------
# <input>
# \   |
#  \ θ|
#   \ |
#    \|
#-----pos(x1,y1)--pos(x2,y2)
#     |           |
#     |           |
#     pos(x4,y4)--pos(x3,y3)
#--------------------------------
# <output>
#       /|--
#      / |   -- 
#     /  |      --
#    /   |         --
#   /    |         
#  /     |   
# /th_rad|
#
def textPos2recPos(pos, text_height, defaultParam, round_decimal=1):
    th = defaultParam['th']
    rate_textHeight2topLeft = defaultParam['rate_topLeft']
    rate_W = defaultParam['rate_W']
    rate_H = defaultParam['rate_H']
    width = rate_W * text_height
    height = rate_H * text_height
    
    # x1,y1: top left
    # x4,y4: buttom right
    x1 = pos[0]['x']
    y1 = pos[0]['y']
    x4 = pos[3]['x']
    y4 = pos[3]['y']
    
    th_rad = -(atan2((y1-y4), (x4-x1)) + 1/2*pi)
    th_c_rad = radians(th) - th_rad
    
    xc1 = x1 - rate_textHeight2topLeft * text_height * sin(th_c_rad)
    yc1 = y1 - rate_textHeight2topLeft * text_height * cos(th_c_rad)

    xc2 = xc1 + width * cos(th_rad)
    yc2 = yc1 + width * sin(th_rad)

    xc3 = xc1 + width * cos(th_rad) - height * sin(th_rad)
    yc3 = yc1 + width * sin(th_rad) + height * cos(th_rad)

    xc4 = xc1 - height * sin(th_rad)
    yc4 = yc1 + height * cos(th_rad)
    
    rec_pos = [
        {'x': xc1, 'y': yc1},
        {'x': xc2, 'y': yc2},
        {'x': xc3, 'y': yc3},
        {'x': xc4, 'y': yc4}
    ]
    rec_pos = [{'x':round(i['x'],round_decimal), 'y':round(i['y'],round_decimal)}for i in rec_pos]
    
    return rec_pos, th_rad, width, height


def div3x3Pos(board_pos):
    x1, x2, x3, x4 = board_pos[0]['x'], board_pos[1]['x'], board_pos[2]['x'], board_pos[3]['x']
    y1, y2, y3, y4 = board_pos[0]['y'], board_pos[1]['y'], board_pos[2]['y'], board_pos[3]['y']
    
    width = x2 - x1
    height = y3 - y2
    
    x_dif = x4 - x1
    y_dif = y1 - y2
    
    divpos = []
    for y_id in range(3):
        divpos_x = []
        for x_id in range(3):
            if x_id == 0:
                _x1 = x1 + x_id/3*width + y_id/3*x_dif
                _y1 = y1 - x_id/3*y_dif + y_id/3*height
            else:
                _x1, _y1 = _x2, _y2
                
            _x2 = _x1 + 1/3*width
            _x3 = _x2 + 1/3*x_dif
            _x4 = _x1 + 1/3*x_dif
            
            _y2 = _y1 - 1/3*y_dif
            _y3 = _y2 + 1/3*height
            _y4 = _y1 + 1/3*height
            
            divpos_x.append([{'x':_x1, 'y':_y1},
                             {'x':_x2, 'y':_y2},
                             {'x':_x3, 'y':_y3},
                             {'x':_x4, 'y':_y4}])
        divpos.append(divpos_x)
        
    return divpos


def divA2IClassPos(div3x3pos):
    divclass = [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]
    divA2Idic = {}
    for i,px in enumerate(div3x3pos):
        for j,p in enumerate(px):
            divA2Idic[divclass[i][j]] = p

    return divA2Idic


# Judge if center of card is in board
def checkCardInBoard(board_pos, card_pos):
    _b_x1 = board_pos[0]['x'] #b_x1=0として考える
    b_x2, b_x3, b_x4 = board_pos[1]['x']-_b_x1, board_pos[2]['x']-_b_x1, board_pos[3]['x']-_b_x1
    _b_y1 = board_pos[0]['y'] #b_y1=0として考える
    b_y2, b_y3, b_y4 = board_pos[1]['y']-_b_y1, board_pos[2]['y']-_b_y1, board_pos[3]['y']-_b_y1
    
    b_rad = -atan2(b_y2, b_x2)
    
    b_width = sqrt(b_x2**2 + b_y2**2)
    b_height = sqrt(b_x4**2 + b_y4**2)
    
    
    c_x1, c_x3 = card_pos[0]['x']-_b_x1, card_pos[2]['x']-_b_x1
    c_y1, c_y3 = card_pos[0]['y']-_b_y1, card_pos[2]['y']-_b_y1
    
    c_center_x = (c_x1+c_x3)/2
    c_center_y = (c_y1+c_y3)/2
    
    c_rotate_x = c_center_x*cos(b_rad) - c_center_y*sin(b_rad)
    c_rotate_y = c_center_y*cos(b_rad) + c_center_x*sin(b_rad)
    
    if 0 <= c_rotate_x <= b_width:
        if 0 <= c_rotate_y <= b_height:
            return True

    return False


def crop_img(img_np, rec_pos, width, height):
    # 4 positons of the corners
    pts1 = np.float32([[i['x'], i['y']] for i in rec_pos])
    # 4 positions converted
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_cropped_np = cv2.warpPerspective(img_np, M, (int(width), int(height)))
    
    return img_cropped_np


def cropTarget(target_pos, height_text, img_np, defaultParam):
    rec_pos, th_rad, card_width, card_height = textPos2recPos(target_pos, height_text, defaultParam)
    img_cropped_np = crop_img(img_np, rec_pos, card_width, card_height)
    return img_cropped_np


def normalized_distance(text_base, text_compare):
    dist = Levenshtein.distance(text_base, text_compare)
    max_len = max(len(text_base), len(text_compare))
    return dist / max_len


def overlay_np_img(x_offset, y_offset, background, front):
    _background = np.copy(background)
    _background[y_offset : y_offset+front.shape[0], x_offset : x_offset+front.shape[1]] = front
    return _background



if __name__ == '__main__':
    print("utils for loopholeScanner")