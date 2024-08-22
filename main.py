import cv2
import numpy as np
import json
import spacy


def find_countours(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # preprocess image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #filter contours to find tables
    table_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  #for input image 2448x3168
            table_contours.append(contour)
    #draw the countors to debug
    cv2.drawContours(img, table_contours, -1, (0, 255, 0), 2)

    #save file with contours for debugging
    table = []
    img1 = np.zeros((2448, 3168, 3), dtype=np.uint8)  # new black img
    img1.fill(255)  # white
    for i, contour in enumerate(table_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 0), 1)
        table.append({'x': x, 'y': y, 'w': w, 'h': h})
    #print(table)
    cv2.imwrite(f'table.png', img1)
    return table, img1

def load_ocr_file(ocr_path):
    with open(ocr_path) as f:
        ocr_data = json.load(f)
    return ocr_data

def check_ocr_file_with_countors(ocr_data, img1): # visualize content from ocr + countours on the new images
    # with open(ocr_path) as f:
    #     ocr_data = json.load(f)
    #draw rectangles and content on the image
    for item in ocr_data:
        x, y, w, h = item['x'], item['y'], item['w'], item['h']
        content = item.get('content', '')
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img1, content, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (36, 255, 12), 2)
        cv2.imwrite(f'table_with_content.png', img1)

def do_overlap(rect1, rect2): #to check if two rectangles overlap
    return not (rect1['x'] + rect1['w'] < rect2['x'] or
                rect1['x'] > rect2['x'] + rect2['w'] or
                rect1['y'] + rect1['h'] < rect2['y'] or
                rect1['y'] > rect2['y'] + rect2['h'])

def match_content_with_countours(table, ocr_data): # matching content with coordinates
    matches = []
    for coord in table:
        matched_contents = []
        for content in ocr_data:
            if do_overlap(coord, content):
                matched_contents.append(content['content'])
        if matched_contents:
            matches.append({
                'coord': coord,
                'contents': matched_contents
            })
    #sort matches by 'x' and then by 'y'
    matches.sort(key=lambda match: (match['coord']['y'], match['coord']['x']))
    reference_coord = {'w': 1800} #to avoid big countour
    reference_contents = ['Quote', 'Line', 'No.'] #content where we should start analize the table
    start_index = next((index for (index, cell) in enumerate(matches)
                        if
                        cell['coord']['w'] < reference_coord['w'] and
                        all(item in reference_contents for item in cell['contents'])), None)
    #remove all elements before the reference cell
    if start_index is not None:
        matches = matches[start_index:]
    #the mapping of x to column_id and y to row_id
    x_to_column_id = {}
    y_to_row_id = {}
    #start from 0
    column_id = 0
    row_id = 0
    for cell in sorted(matches, key=lambda c: (c['coord']['y'], c['coord']['x'])):
        if cell['coord']['x'] not in x_to_column_id:
            x_to_column_id[cell['coord']['x']] = column_id
            column_id += 1
        if cell['coord']['y'] not in y_to_row_id:
            y_to_row_id[cell['coord']['y']] = row_id
            row_id += 1
        cell['column_id'] = x_to_column_id[cell['coord']['x']]
        cell['row_id'] = y_to_row_id[cell['coord']['y']]
        cell['is_header'] = (cell['row_id'] == 0)

    # for debugging, print the matches
    #for match in matches:
        #print(f"Coordinate: {match['coord']} matches with Contents: {match['contents']}, row_id: {match['row_id']}, column_id: {match['column_id']}, is_header: {match['is_header']}")
    return matches

def exchage_the_words(matches):
    #load the trained model from disk
    nlp = spacy.load("./model")
    final = []
    for match in matches:
        #check that it's not Price column and header Item Describtion:
        if match['is_header'] is True or match['column_id'] == 3:
            if 'Price' not in match['contents'] and 'Item' not in match['contents']:
                doc= nlp(' '.join(match['contents']))
                for ent in doc.ents:
                    #print(f"{ent.label_}: {ent.text}") to debug model
                    if ent.label_ != 'Vendor' and ent.label_ != 'Model':
                        final.append({'x':match['coord']['x'], 'y':match['coord']['y'], 'w':match['coord']['w'], 'h':match['coord']['h'], 'content': ent.label_,
                                      'colunm_id': match['column_id'], 'row_id': match['row_id'], 'is_header': match['is_header']})

                    elif ent.label_ == 'Vendor':
                        final.append({'x':match['coord']['x'], 'y':match['coord']['y'], 'w':match['coord']['w'], 'h':match['coord']['h'], 'content':ent.text,
                                      'colunm_id': match['column_id'], 'row_id': match['row_id'], 'is_header': match['is_header']})
                    elif ent.label_ == 'Model':
                        final.append({'x':match['coord']['x'], 'y':match['coord']['y'], 'w':match['coord']['w'], 'h':match['coord']['h'], 'content':ent.text,
                                      'colunm_id': match['column_id'], 'row_id': match['row_id'], 'is_header': match['is_header']})
            elif 'Price' in match['contents']:
                final.append({'x':match['coord']['x'], 'y':match['coord']['y'], 'w':match['coord']['w'], 'h':match['coord']['h'], 'content':'Unit price',
                              'colunm_id': match['column_id'], 'row_id': match['row_id'], 'is_header': match['is_header']})
            elif 'Item' in match['contents']:
                for j in ('Vendor name', 'Model name'):
                    final.append({'x':match['coord']['x'], 'y':match['coord']['y'], 'w':match['coord']['w'], 'h':match['coord']['h'],'content':j,
                                  'colunm_id': match['column_id'], 'row_id': match['row_id'], 'is_header': match['is_header']})
        elif match['column_id'] == 4:
            final.append({'x':match['coord']['x'], 'y':match['coord']['y'], 'w':match['coord']['w'], 'h':match['coord']['h'], 'content':match['contents'][2],
                          'colunm_id': match['column_id'], 'row_id': match['row_id'], 'is_header': match['is_header']})
        else:
            final.append({'x':match['coord']['x'], 'y':match['coord']['y'], 'w':match['coord']['w'], 'h':match['coord']['h'], 'content':match['contents'][0],
                          'colunm_id': match['column_id'], 'row_id': match['row_id'], 'is_header': match['is_header']})
    print(final) # for debug
    with open('final.json', 'w') as f: #save the final json
        json.dump(final, f, indent=4)
    return matches

img_path = 'doc_test.png' # initial image
ocr_path = 'ocr_test.json' # initial ocr
table, img1 = find_countours(img_path)

#load ocr file
ocr_data = load_ocr_file(ocr_path)

#print ocr text in the finding countours for debugging
#check_ocr_file_with_countors(ocr_data, img1)

#put content from ocr to the countours
matches = match_content_with_countours(table,ocr_data)

#exchange the words
exchage_the_words(matches)