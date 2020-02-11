import json
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
import imutils
import pytesseract
import re
import json

class Plate:
    def __init__(self):
        self.to_angka = [["O","0"],["!","1"],["I","1"],["|","1"],["S","5"],["E","8"]]
        self.to_huruf = [["e","O"],["0","O"],["!","I"],["1","I"],["|","I"],["5","S"],["8","O"],["7","L"]]
        self.kode_plat = ["BL","BB","BK","BA","BM","BP","BG","BN","BE","BD","BH","A",\
                            "B","D","E","F","T","Z","G","H","K","R","AA","AB","AD","L","M","N","P",\
                            "S","W","AE","AG","DK","DR","EA","DH","EB","ED","KB","DA","KH","KT",\
                            "DB","DL","DM","DN","DT","DD","DC","DE","DG","DS"]
        self.data_huruf = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P",\
                            "Q","R","S","T","U","V","W","X","Y","Z"]
        self.kode_plat =set (self.kode_plat)
        
        #KNN
        x=[]
        y=[]
        with open('data_plate.txt') as josn_file:
            data = json.load(josn_file)
            for p in data['data']:
                x.append([p['b'],p['w']])
                y.append(p['treshold'])
        self.knn=KNeighborsClassifier(n_neighbors=3) #define K=3
        self.knn.fit(x,y)

    def ocr_core(self,img):
        return pytesseract.image_to_string(img,lang='eng')
        
    def getTresh(self,b,w):
        a=np.array([[b,w]])
        result = self.knn.predict(a)
        return result[0]
    
    def toRgb(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def removeSymbol(self,val):
        return re.sub("[^0-9a-zA-Z]","",val)

    def toBin(self,img,tre):
        _,img=cv2.threshold(img, tre, 255, cv2.THRESH_BINARY_INV)
        return img

    def gaussianBlur(self,img):
        return cv2.GaussianBlur(img,(5,5),0)

    def toGray(self,img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def rotateImg(self,img,angle):
        return imutils.rotate_bound(img, angle)

    def cropImg(self,img):
        h, w= img.shape
        return img[5:0+h-10, 6:0+w]
        

    def replaceToAngka(self,val):
        for i in self.to_angka:
            if val==i[0]:
                return i[1]
        return ""

    def replaceToHuruf(self,val):
        for i in self.to_huruf:
            if val==i[0]:
                return i[1]
        return ""


    def getPlat(self,plat):
        
        pl0 = ""
        pl1 = ""
        pl2 = ""
        if len(plat) < 3:
            return ""
        #get plat kota
        for i in range(0,2):
            try:
                if(type(int(plat[i])))==int:
                    if plat[i] in self.kode_plat:
                        pl0+=self.replaceToHuruf(plat[i])
            except ValueError:
                val = pl0+plat[i]
                if val in self.kode_plat:
                    pl0+=plat[i]
        #get plat angka
        l=0
        for i in range(len(pl0),len(plat)):
            
            try:
                if(type(int(plat[i])))==int:
                    pl1+=plat[i]
                    l+=1
            except ValueError:
                pl1+=self.replaceToAngka(plat[i])
                l+=1
            if l>3:
                break
        #get plat huruf belakang
        l=0
        for i in range(len(pl0)+len(pl1),len(plat)):
            try:
                if(type(int(plat[i])))==int:
                    if plat[i] in self.data_huruf:
                        pl2+=self.replaceToHuruf(plat[i])
                        l+=1
            except ValueError:
                if plat[i] in self.data_huruf:
                    pl2+=plat[i]
                    l+=1
                else:
                    pl2+=self.replaceToHuruf(plat[i])
                    l+=1
            if l>2:
                break
        if (pl0 or pl1 or pl2 ) == "":
            return ""
        return pl0+"-"+pl1+"-"+pl2
    
    def getText(self,img):
        imgProc = self.toGray(img)
        # imgProc = self.cropImg(imgProc)

        w,h = imgProc.shape
        total = w*h
        bl = np.sum(imgProc >= 127) / total *100
        wh = np.sum(imgProc < 127) / total *100
        tre = self.getTresh(bl,wh)

        imgProc = self.toBin(imgProc,tre)
        imgProc = self.gaussianBlur(imgProc)
        extracted_text = self.ocr_core(imgProc)
        if len(extracted_text)> 4:
            extracted_text = self.removeSymbol(extracted_text)
            extracted_text  = self.getPlat(extracted_text)
        
        return extracted_text
