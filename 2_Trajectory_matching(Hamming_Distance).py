import cv2
import time
import random
import PIL
import glob
from PIL import Image
import pandas as pd
import numpy as np

#hamming distance를 이용한 이미지 매칭

image_index = range(1,10000)
n_search = 1
# 1회 서칭시 나오는결과를 저장할 Dataframe 선언
searching_result = pd.DataFrame([{'n_search': 0, 'searching_time': 0, 'image_loc':0, 'similarity' : 0}])

#image 경로 
search_folder = "C:/Users/IDeAOcean LT1/Documents/Python_database/image_post_processed_224"


#이미지를 16x16 average hash로 변환하는 함수
def img2hash(img):
    resized_image = cv2.resize(img, (16, 16))
    avg = resized_image.mean()
    bi = 1 * (resized_image > avg)
    return bi

#hamming distance 측정 함수
def hamming_distance(a, b):
    a = a.reshape(1,-1)
    b = b.reshape(1,-1)
    # 같은 자리의 값이 서로 다른 것들의 합
    distance = (a !=b).sum()
    return distance

#이미지 폴더 내의 모든 이미지 경로 불러오기
image_files = glob.glob(search_folder + "/*.jpg")

#무작위 위치를 탐색하기 위해 Shuffle
shuffled_num = list(range(1,9752))
random.shuffle(shuffled_num)

for num in shuffled_num:
    search_image = image_files[num]
    #탐색대상 image를 hash로 변환
    search_hash = img2hash(cv2.imread(search_image, cv2.IMREAD_GRAYSCALE))
    flag = 0

    start = time.time()
    n_search = 0

    for temp_image in image_files:
        #비교대상 image를 hash로 변환
        temp_hash = img2hash(cv2.imread(temp_image, cv2.IMREAD_GRAYSCALE))
        #matching processing timer 시작
       
        hamming_dist = hamming_distance(search_hash, temp_hash)

        if(flag == 1):
            next
        else:
            n_search += 1

        #hamming distance가 10% 이내의 image만 출력
        #hashing 후 grid가 16 x 16임으로 256으로 나눔
        if hamming_dist/256 < 0.001 and flag == 0: 
            print("search : " + search_image[74:] + " || temp : " + temp_image[74:])
            flag = 1
            end = time.time()
            #탐색결과 저장
            #matching processing timer 종료
            current_searching_result = pd.DataFrame([{"n_search": n_search, "searching_time": end-start, "image_loc":temp_image[74:], 'hamming dist' : hamming_dist}])
            searching_result = searching_result._append(current_searching_result, ignore_index=True)
            print("image_loc :" + str(search_image[74:]) + " | n_search :"+str(n_search)+" | processing time :" + str(end-start) + ' | hamming dist :' + str(hamming_dist))
        
searching_result.to_csv("C:/Users/IDeAOcean LT1/Documents/Python_database/searching_result.csv")