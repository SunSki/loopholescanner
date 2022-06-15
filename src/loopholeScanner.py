from _utils import * 

import cv2
import numpy as np
from math import sqrt
import statistics
import pandas as pd
import Levenshtein
import glob
import os
from logzero import logger



# GoogleAPI Key
GOOGLE_KEY = '../data/key/loopholes-352211-15c7b0c80782.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_KEY


##########
# #title #
#        #
#        #
#        #
#        #
##########
# #: x=10, y=12, w=12, h=15 <- Average of vision API results
# Card Size: W=175.748pt, H=249.449pt <- W=62mm, H=82mm
# rate_W: W/h, rate_H: H/h
## Captured Card ##
RECOG_SYMBOL = '#'
K_CARD_PARAMS_TOPLEFT = 0.8
K_CARD_PARAMS_WH = 1.
DEFAULT_CARD_PARAMS = {
    'th':39.8, 
    'rate_topLeft':1. * K_CARD_PARAMS_TOPLEFT, 
    'rate_W':11.7 * K_CARD_PARAMS_WH,
    'rate_H':16.6 * K_CARD_PARAMS_WH
}
## Digital card image info ##
# CSV Columns: HeadTitle, Title, Description, Box
DIGITAL_CARD_WIDTH = 175
DIGITAL_CARD_HEIGHT = 248
DIGITAL_CARDS_CSV_PATH = '../data/csv/triggerCards.csv' 
DIGITAL_CARD_FOLDER = '../data/img/cards'
DIGITAL_CARD_FILE_EXTENTION = '.jpg'
digital_card_paths = glob.glob(
    f"{DIGITAL_CARD_FOLDER}/*{DIGITAL_CARD_FILE_EXTENTION}")


##################### 
#               #   #
#               #   #
#  in the next  #   #
#               #   #
#               #   #
##################  #
#                   #
#####################
# HoleBoard; A1: 594 x 841 mm =>  1683.78 x 2383.94 px
# Board Space：inner length
# in the next: x=983, y=715, h=24
# W=2184pt, H=1474pt
# rate_W: W/h, rate_H: H/h
## Captured Board ##
BOARD_KEYWORD = 'in the next'
DEFAULT_CARD_BOARD_PARAMS = {
    'th':54.,
    'rate_topLeft':50.65,
    'rate_W':91,
    'rate_H':61.5
}
## Digital Board ##
BOARD_IMG_PATH = '../data/img/boards/gameBoard.jpg'
CARD_BOARD_HEIGHT = 1474



# get all words and its position
def extract_wordPos(google_response):
    extracted_textAnnotations = google_response.text_annotations[1:]
    extracted_words = [i.description for i in extracted_textAnnotations]
    extracted_pos = [[{'x':p.x, 'y':p.y} for p in i.bounding_poly.vertices] for i in extracted_textAnnotations]
    extracted_wordPos = [{'word':w, 'pos':p} for w,p in zip(extracted_words, extracted_pos)]
    return extracted_wordPos


# get board params; {'pos', 'width', 'height'}
def extract_captured_board(extracted_wordPos):
    BOARD_KEYWORDLen = len(BOARD_KEYWORD.split(' '))

    boardInitialKeyPos = None
    boardInitialKeyHeight = None

    boardInitialKeyPosList = []
    for i in range(len(extracted_wordPos)-BOARD_KEYWORDLen):
        extracted_words = [i['word'] for i in extracted_wordPos]
        extracted_pos = [i['pos'] for i in extracted_wordPos]
        serial_words = ' '.join(extracted_words[i:i+BOARD_KEYWORDLen])
        if serial_words == BOARD_KEYWORD:
            boardInitialKeyPosList.append(extracted_pos[i])
    
    if len(boardInitialKeyPosList) == 1:
        boardInitialKeyPos = boardInitialKeyPosList[0]
        boardInitialKeyHeight = sqrt(\
                                (boardInitialKeyPos[0]['x'] - boardInitialKeyPos[-1]['x'])**2\
                                + (boardInitialKeyPos[0]['y'] - boardInitialKeyPos[-1]['y'])**2)
        boardPos, _, boardWidth, boardHeight = textPos2recPos(
                                                boardInitialKeyPos,
                                                boardInitialKeyHeight,
                                                DEFAULT_CARD_BOARD_PARAMS)
        CapturedBoard = {'pos':boardPos, 'width':boardWidth, 'height':boardHeight}
    else:
        logger.error(f'Board Capture Error: {len(boardInitialKeyPosList)}')
        CapturedBoard = None
    
    return CapturedBoard


class CardSymbols:
    
    def __init__(self, google_response):
        self.array = []

        g_CardSymbols = []
        for block in google_response.full_text_annotation.pages[0].blocks:
            for para in block.paragraphs:
                for word in para.words:
                    for symbol in word.symbols:
                        if symbol.text == RECOG_SYMBOL:
                            g_CardSymbols.append(symbol)

        for s in g_CardSymbols:
            cornerPos = [{'x':i.x, 'y':i.y} for i in s.bounding_box.vertices]

            pos1 = s.bounding_box.vertices[0]
            pos4 = s.bounding_box.vertices[-1]
            text_height = sqrt((pos1.x - pos4.x)**2 + (pos1.y - pos4.y)**2)

            self.array.append({'pos':cornerPos, 'textHeight':text_height})

        logger.debug(f'The number of cardSymbols: {len(self.array)}')

    
    def add_rec_pos(self, caputuredCardTextHeight):
        for i,cs in enumerate(self.array):
            rec_pos,_,_,_ = textPos2recPos(
                                cs['pos'],
                                caputuredCardTextHeight,
                                DEFAULT_CARD_PARAMS)
            cs['rec_pos'] = rec_pos
            cs['rec_center_pos'] = {
                'x':(rec_pos[0]['x'] + rec_pos[2]['x'])/2, 
                'y':(rec_pos[0]['y'] + rec_pos[2]['y'])/2}
    
    
    def add_box(self, divDic):
        logger.debug("Start: add_box")
        for cs in self.array:
            boxIn = False
            for boxType,boxPos in divDic.items():
                if checkCardInBoard(boxPos, cs['pos']):
                    cs['box'] = boxType
                    boxIn = True
                    continue
            if not boxIn:
                cs['box'] = 'X' #どのboxにも入らなければXに

            logger.debug(f"box: {cs['box']}")
    

    def add_rec_img(self, caputuredCardTextHeight, captured_img_np):
        for cs in self.array:
            img_cropped = cropTarget(
                            cs['rec_pos'], 
                            caputuredCardTextHeight,
                            captured_img_np,
                            DEFAULT_CARD_PARAMS)
            cs['rec_img'] = img_cropped


    def add_word(self, extracted_wordPos):
        logger.debug("Start: add_word")
        for cs in self.array:
            add_words = []
            for ewp in extracted_wordPos:
                if checkCardInBoard(cs['rec_pos'], ewp['pos']):
                    add_words.append(ewp)

            if len(add_words) > 3:
                _log = ' '.join([i['word'] for i in add_words[:3]])
                logger.debug(f"initial 3 words: {_log}")
            else:
                _log = ' '.join([i['word'] for i in add_words])
                logger.debug(f"words: {_log}")

            cs['words'] = add_words
    

    # ['card']: {title, box}
    def add_card(self):
        logger.debug("Start: add_card")
        for cs in self.array:
            df_triggerCards = pd.read_csv(DIGITAL_CARDS_CSV_PATH)

            cs_head_title = cs['words'][0]['word'][1:] # [0]#を含む最初の単語を取得, [1:]先頭の#を除いた文字列
            matched_df = df_triggerCards[df_triggerCards['HeadTitle']==cs_head_title]

            if len(matched_df) == 1:
                cs_card = {'title':matched_df.iloc[0]['Title'], 'box':matched_df.iloc[0]['Box']}
            elif len(matched_df) > 1:
                cs_description = ' '.join([i['word'] for i in cs['words'][1:]]) # [1:]先頭の#を含むタイトルヘッドを除く単語列
                df_description = [matched_df.iloc[i]['Description'] for i in range(len(matched_df))]
                
                # レーベルシュタイン距離が最小のものを選ぶ
                min_dis_id = min(
                    enumerate([normalized_distance(cs_description, dd) for dd in df_description]),
                    key=(lambda x:x[1]))[0] # (id, dis)なので[0]でid取得
                cs_card = {'title':matched_df.iloc[min_dis_id]['Title'], 'box':matched_df.iloc[min_dis_id]['Box']}
            else:
                # matched_df is None
                cs_card = None

            logger.debug(f"head title: {cs_head_title}, card: {cs_card}")

            cs['card'] = cs_card


class DigitalCards:
    def __init__(self, CapturedBoard, cardSymbols):
        self.array = []

        board_scale = CARD_BOARD_HEIGHT/CapturedBoard['height']
        cap_board_pos1 = CapturedBoard['pos'][0]
        
        for cs in cardSymbols.array:
            cap_card_pos1 = cs['rec_pos'][0]    
            card_pos_left_top = {i: round((cap_card_pos1[i] - cap_board_pos1[i]) * board_scale) for i in ['x','y']}

            card_img = cv2.resize(cs['rec_img'], (DIGITAL_CARD_WIDTH, DIGITAL_CARD_HEIGHT)) # resize
            self.array.append({'posLeftTop':card_pos_left_top, 'capturedImg':card_img, 'box':cs['box'], 'card':cs['card']})

        logger.debug(f'The number of DigitalCards: {len(self.array)}')
        logger.debug(f"Cards Info")
        logger.debug([i['card'] for i in self.array])

    def add_card_path(self):
        logger.debug("Start: add_card_path")
        # dc['card_path']
        for dc in self.array:
            if dc['card']:
                dc_name = f"{dc['card']['box']}_{dc['card']['title'].replace(' ', '')}"
                dc_card_path = f"{DIGITAL_CARD_FOLDER}/{dc_name}{DIGITAL_CARD_FILE_EXTENTION}"
                
                logger.debug(f"card path: {dc_card_path}")
                if dc_card_path in digital_card_paths:
                    dc['card_path'] = dc_card_path
                else:
                    logger.error(f'Digital Card Found Error; {dc_card_path}')
                    dc['card_path'] = None
            else:
                logger.error(f'Digital Card Found Error; {dc_card_path}')
                dc['card_path'] = None


class DigitalBords:
    
    def __init__(self):
        self.dic = {}
        self.board_img = imread4warp(BOARD_IMG_PATH) 

    def generate_captured_scattered_img(self, DigitalCards):
        logger.debug('Start: generate captured scattered img')
        img = self.board_img #初期値はプリセットのボード画像

        for dc in DigitalCards.array:
            x_offset = max(0, dc['posLeftTop']['x']) #0以上にする
            y_offset = max(0, dc['posLeftTop']['y'])

            if x_offset + DIGITAL_CARD_WIDTH > self.board_img.shape[1]:
                x_offset = self.board_img.shape[1] - DIGITAL_CARD_WIDTH
            if y_offset + DIGITAL_CARD_HEIGHT > self.board_img.shape[0]:
                y_offset = self.board_img.shape[0] - DIGITAL_CARD_HEIGHT

            img = overlay_np_img(
                                x_offset, 
                                y_offset, 
                                img, 
                                cv2.cvtColor(dc['capturedImg'], cv2.COLOR_BGR2RGB))

        self.dic['captured_scattered'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    def generate_digital_scattered_img(self, DigitalCards):
        logger.debug('Start: generate digital scattered img')
        img = self.board_img #初期値はプリセットのボード画像
        for dc in DigitalCards.array:  
            x_offset = max(0, dc['posLeftTop']['x'])
            y_offset = max(0, dc['posLeftTop']['y'])

            if x_offset + DIGITAL_CARD_WIDTH > self.board_img.shape[1]:
                x_offset = self.board_img.shape[1] - DIGITAL_CARD_WIDTH
            if y_offset + DIGITAL_CARD_HEIGHT > self.board_img.shape[0]:
                y_offset = self.board_img.shape[0] - DIGITAL_CARD_HEIGHT

            if dc['card_path']:
                digital_card_img = imread4warp(dc['card_path'])
                img = overlay_np_img(
                                    x_offset, 
                                    y_offset, 
                                    img, 
                                    digital_card_img)
            else:
                img = overlay_np_img(
                            x_offset, 
                            y_offset, 
                            img, 
                            cv2.cvtColor(dc['capturedImg'], cv2.COLOR_BGR2RGB))

        self.dic['digital_scattered'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



def generate(captured_byte_img: bytes, captured_np_img: np.ndarray):
    
    logger.debug('1. get google api request...')
    response = google_vision_response(captured_byte_img)

    logger.debug('2. extract info from google api result...')
    extracted_wordPos = extract_wordPos(response) # 'word', 'pos'

    CapturedBoard = extract_captured_board(extracted_wordPos) # 'pos', 'width', 'height' or None
    if not CapturedBoard:
        raise ValueError("Board not found")

    cardSymbols = CardSymbols(response)

    # カードの高さの規定値を決める
    # #の高さの中央値と最大値を取る
    cardMedianHeight = statistics.median([i['textHeight'] for i in cardSymbols.array])
    cardThirdQuartileHeight = sorted([i['textHeight'] for i in cardSymbols.array])[round(len(cardSymbols.array) * .75)]
    cardMaxHeight = max([i['textHeight'] for i in cardSymbols.array])
    # TODO; Decide which index is better
    caputuredCardTextHeight = cardThirdQuartileHeight


    # 取得したボードのボックス部分ををA-Iまで9つに分ける
    #[[1,2,3],[4,5,6],[7,8,9]] => [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']]
    # => {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'}
    board3x3DivPos = div3x3Pos(CapturedBoard['pos']) 
    divA2IboxDic = divA2IClassPos(board3x3DivPos)


    #####################################
    ### Add Parameters to CardSymbols ###
    #####################################
    logger.debug('3. add info to captured cards...')
    cardSymbols.add_rec_pos(caputuredCardTextHeight)
    cardSymbols.add_box(divA2IboxDic)
    cardSymbols.add_rec_img(caputuredCardTextHeight, captured_np_img)
    cardSymbols.add_word(extracted_wordPos)
    cardSymbols.add_card()


    ##############################
    ### Generate Digital Board ###
    ##############################
    logger.debug('4. get digital cards...')
    # 'posLeftTop', 'capturedImg', 'box', 'card'
    digitalCards = DigitalCards(CapturedBoard, cardSymbols)
    digitalCards.add_card_path()

    logger.debug('5. generate digital boards...')
    digitalBoards = DigitalBords()
    digitalBoards.generate_captured_scattered_img(digitalCards)
    digitalBoards.generate_digital_scattered_img(digitalCards)

    
    return digitalBoards.dic
