import numpy as np
import cv2
import pywt
import glob
from PIL import Image
import copy
import random

def mse(I1, I2):
    #[H, W] = np.shape(I1)
    ##NH=H*W
    #total=0
    #for i in range(0,H):
    #    for j in range(0,W):
    #        p1=I1[i][j]
    #        p2=I2[i][j]
    #        diff=p1-p2
    #        diff_sqd=diff*diff
    #        total=total+diff_sqd
    #print('total='+str(total))
    #err=total/NH
    err=np.square(np.subtract(np.double(I1), np.double(I2))).mean()
    return err


def halt():
    k = 0
    while k != 48:
       k = cv2.waitKey(100)
    cv2.destroyAllWindows()

def DFT(coverImage, watermarkImage):
    H,W=np.shape(coverImage)# find size of cover image
    cv2.imshow('Cover Image: '+FName,coverImage)
    coverImage = coverImage / 255# convert range to 0 to 1
    coverImageFFT = np.fft.fftshift(np.fft.fft2(coverImage))# apply FFT

    watermarkImage = cv2.resize(watermarkImage,(H,W))# resize watermark image to size of cover image
    cv2.imshow('Watermark Image',watermarkImage)
    watermarkImage = watermarkImage / 255  # convert range to 0 to 1
    watermarkImageFFT = np.fft.fftshift(np.fft.fft2(watermarkImage))# apply fft

    alpha=0.05
    wattermarkedFFT=coverImageFFT+ (alpha*watermarkImageFFT)# embed the watermark

    watermarkedImage = np.fft.ifft2(np.fft.ifftshift(wattermarkedFFT))#apply inverse FFT

    ## calculate MSE
    watermarkedImage = np.uint8(255 * watermarkedImage)  # convert to 0,255 range
    coverImage2=copy.deepcopy(coverImage)
    coverImage2=np.uint8(255*coverImage)
    MSE=mse(coverImage2,watermarkedImage)
    print('MSE (coverimage, watermarked)='+str(MSE))

    #save watermarked image
    fn='DFT_Watermarked_'+FName#set file name
    cv2.imshow('DFT_Watermarked_' + FName, watermarkedImage)

    img = Image.fromarray(watermarkedImage)
    Floc='./result/DFT/'+fn#set file path+file name
    img.save(Floc)#save image at location
    print('Watermarked Image saved at:'+Floc)


    ## Extraction process
    watermarkedImageR=watermarkedImage/255#convert to 0,1 range
    watermarkedImageRFFT = np.fft.fftshift(np.fft.fft2(watermarkedImageR))# apply FFT
    watermarkRFFT=(watermarkedImageRFFT-coverImageFFT)/alpha# extract watermark FFT
    watermarkR=np.fft.ifft2(np.fft.ifftshift(watermarkRFFT))# apply inverse FFT
    watermarkR=np.uint8(255*watermarkR)# convert to 0,255 range
    cv2.imshow('Extracted watermark',watermarkR)

    ## calculate MSE
    watermarkImage2=copy.deepcopy(watermarkImage)
    watermarkImage2=np.uint8(255*watermarkImage2)
    MSE=mse(watermarkImage2,watermarkR)
    #print('MSE (watermark, extracted watermark)='+str(MSE))

    #save extracted watermark image
    fn='DFT_Extracted_Watermark_'+FName#set file name
    cv2.imshow('DFT_Extracted_Watermark: ' + FName, watermarkR)

    img = Image.fromarray(watermarkR)
    Floc='./result/DFT/'+fn#set file path+file name
    img.save(Floc)#save image at location
    #print('Extracted Watermark Image saved at: '+Floc)

def DWT(coverImage, watermarkImage):
    coverImage = cv2.resize(coverImage,(300,300))
    cv2.imshow('Cover Image: '+FName,coverImage)
    watermarkImage = cv2.resize(watermarkImage,(150,150))
    cv2.imshow('Watermark Image',watermarkImage)

    #DWT on cover image
    coverImage =  np.float32(coverImage)
    coverImage /= 255
    coeffC = pywt.dwt2(coverImage, 'haar')
    cA, (cH, cV, cD) = coeffC
    print("DWT of coverimage done")
    watermarkImage = np.float32(watermarkImage)
    watermarkImage /= 255

    #Embedding
    alpha=0.1
    coeffW = (cA + alpha*watermarkImage, (cH, cV, cD))
    watermarkedImage = pywt.idwt2(coeffW, 'haar')
    watermarkedImage=np.uint8(255*watermarkedImage)
    coverImage=np.uint8(255*coverImage)
    print("Embedding Watermark done..")
    cv2.imshow('Watermarked Image: '+FName, watermarkedImage)

    ## calculate MSE
    coverImage2=copy.deepcopy(coverImage)
    watermarkedImage2=copy.deepcopy(watermarkedImage)

    coverImage2=np.uint8(255*coverImage2)
    watermarkedImage2=np.uint8(255*watermarkedImage2)
    MSE=mse(coverImage2,watermarkedImage2)
    print('MSE (coverimage, watermarked)='+str(MSE))

    # save watermarked image
    fn = 'DWT_Watermarked_' + FName  # set file name
    cv2.imshow('DWT_Watermarked_' + FName, watermarkedImage)

    img = Image.fromarray(watermarkedImage)
    Floc = './result/DWT/' + fn  # set file path+file name
    img.save(Floc)  # save image at location
    print('Watermarked Image displayed and saved at:' + Floc)

    #Extraction
    watermarkedImage=watermarkedImage/255
    print("Extraction watermarking start...")
    coeffWM = pywt.dwt2(watermarkedImage, 'haar')
    hA, (hH, hV, hD) = coeffWM

    extracted = (hA-cA)/alpha
    ## calculate MSE
    watermarkImage2=copy.deepcopy(watermarkImage)
    extracted2=copy.deepcopy(extracted)
    watermarkImage2=np.uint8(255*watermarkImage)
    extracted2=np.uint8(255*extracted)
    MSE=mse(watermarkImage2,extracted2)
    #print('MSE (watermark, extracted watermark)='+str(MSE))

    extracted *= 255
    extracted = np.uint8(extracted)
    print("Watermark Extracted...")
    #cv2.imshow('Extracted', extracted)
    watermarkedImage=255*watermarkedImage
    watermarkedImage = np.uint8(watermarkedImage)

    #save extracted watermark image
    fn='DWT_Extracted_Watermark_'+FName#set file name
    cv2.imshow('DWT_Extracted_Watermark_' + FName, extracted)

    img = Image.fromarray(extracted)
    Floc='./result/DWT/' +fn#set file path+file name
    img.save( Floc)#save image at location
    print('Extracted Watermark Image displayed and saved at:'+Floc)



    return watermarkedImage

def DCT(I, Imask):
    [Hw,Ww]=np.shape(Imask)
    NWp=Hw*Ww
    print('Watermark Image Size='+str(Hw)+'x'+str(Ww)+'='+str(NWp))

    #I = cv2.imread('D:\PROJECT\python\code\DCTTest\coverImage1.jpg', 0)
    cv2.imshow('Cover Image: '+FName, I)
    [H,W]=np.shape(I)
    print('Cover Image='+str(H)+'x'+str(W))

    ## Lower DCT Coefficient locations of 8x8 block
    RC=[[0,0],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],
       [0,1],[1,1],[2,1],[3,1],[4,1],[5,1],
       [0,2],[1,2],[2,2],[3,2],[4,2],
       [0,3],[1,3],[2,3],[3,3],
       [0,4],[1,4],[2,4],
       [0,5],[1,5],
       [0,6]]
    NALoc=RC.__len__()
    print('Low Coef N='+str(NALoc))

    ## Total pixels tobe processed
    Bh=np.uint8(H/8)
    Bw=np.uint8(W/8)
    TP=np.double(Bh)*np.double(Bw)*np.double(NALoc)
    print('Total Pixels can be processed=',str(TP))

    ## Reduce size of watermark image
    while (TP<NWp):#if watermark image is too large, reduce its size
            Nsize=np.uint16(np.round(0.7*np.double(np.shape(Imask))))
            #Imask=np.resize(Imask,Nsize)
            Imask = cv2.resize(Imask, (Nsize[0],Nsize[1]), interpolation=cv2.INTER_AREA)
            [Hw, Ww] = np.shape(Imask)
            NWp = Hw * Ww
            print('Watermark Pixels='+str(Hw)+'x'+str(Ww)+'='+str(NWp))

    Imask = cv2.threshold(Imask, 127, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow('Resized Watermark Image: ', Imask)
    Imask = Imask / 255
    cv2.imshow('BW Resized Watermark Image: ', Imask)

    ## Reshape watermark image to an single array
    ImaskA=np.reshape(Imask,(NWp,1))
    R=NWp % NALoc
    Zp=NALoc-R
    ZAp=np.zeros((Zp,1))
    ImaskA=np.concatenate((ImaskA,ZAp ))#combine in row direction
    NWp=ImaskA.size

    ##process blocks
    Wc = 0
    Iw =copy.deepcopy(I)

    alpha = 2
    DctOrg = np.zeros((NWp, 1))
    DctVal = np.zeros((NALoc,1))
    for i in range(0,H-8,8):
            for j in range (0,W-8,8):

                    Iblock = I[i:i + 8, j: j + 8] # crop image block 8x8
                    DctBlock = cv2.dct(np.double(Iblock)) # convert to dct

                    for k in range (0,NALoc):
                            r=RC[k][0]
                            c=RC[k][1]
                            val=DctBlock[r,c]
                            DctVal[k] =val  # find lower dc component

                    Wc2 = Wc + NALoc  # end location
                    #if(i==0):
                    #       print('i=' + str(i) + ', ' + 'j=' + str(j)+', Wc='+str(Wc)+', Wc2='+str(Wc2))

                    DctOrg[Wc: Wc2]=DctVal # copy original data

                    MaskVal = ImaskA[Wc:Wc2] # copy watermark pixel
                    DctVal2 = DctVal + alpha * MaskVal # Embed watermark

                    for k in range(0,NALoc):
                            r=RC[k][0]
                            c=RC[k][1]
                            val=DctVal2[k]
                            DctBlock[r,c] =val # find lower dc component end

                    Block2 = cv2.idct(DctBlock)# apply inverse dct
                    Iw[i: i + 8, j: j + 8] = Block2  # copy back to image
                    if (Wc > (NWp - NALoc - 1)):
                        break
                    else:
                        Wc = Wc + NALoc
            if (Wc > (NWp - NALoc)):
                    break
    # Display watermarked image

    Iw=np.round(Iw)
    cv2.imshow('Watermarked Image: '+FName, Iw)

    #MSE = np.square(np.subtract(np.double(I),np.double(Iw))).mean()
    MSE=mse(I,Iw)
    print('MSE (coverimage, watermarked cover image)='+str(MSE))

    # save watermarked image
    img = Image.fromarray(Iw)
    fn = 'DCT_Watermarked_' + FName  # set file name
    Floc='./result/DCT/' +fn#set file path+file name
    img.save(Floc)#save image at location
    print('Watermarked Image saved at: '+Floc)

    #Extraction process
    Iw=np.uint8(Iw)
    Wc=0
    #Im=I
    Imr=np.zeros((NWp,1))
    DctVal = np.zeros((NALoc,1))
    for i in range(0,H-8,8):
            for j in range(0,W-8,8):
                    Block = Iw[i:i + 8, j: j + 8] # crop image block 8 x8
                    DctBlock = cv2.dct(np.double(Block)) # convert to dct

                    for k in range(0,NALoc):
                            r = RC[k][0]
                            c = RC[k][1]
                            DctVal[k] = DctBlock[r,c]# find lower dc component end

                    Wc2 = Wc + NALoc  # end location
                    #print('i=' + str(i) + ', ' + 'j=' + str(j) + ', Wc=' + str(Wc) + ', Wc2=' + str(Wc2))

                    #DctOrg = np.zeros((NWp, 1))
                    DctOrgt = DctOrg[Wc:Wc2]# copy original data
                    DCTRec = abs(DctVal - DctOrgt) / alpha
                    Imr[Wc: Wc2]=DCTRec
                    if (Wc > (NWp - NALoc - 1)):
                            break
                    else:
                            Wc = Wc + NALoc
            if (Wc > (NWp - NALoc - 1)):
                    break
    ## reshape and display
    L=Imr.__len__()
    Imru=Imr[0:L-Zp]
    Imru=np.resize(Imru,(Hw,Ww))
    Imru=np.double(255*Imru)
    Imru = cv2.threshold(Imru, 127, 255, cv2.THRESH_BINARY)[1]

    Imru=Imru/255
    cv2.imshow('Extracted Watermark: '+FName,Imru)

    #calculate MSE
    Imask2=copy.deepcopy(Imask)
    Imru2=copy.deepcopy(Imru)
    Imask2=np.uint8(255*Imask)
    Imru2=np.uint8(255*Imru)
    MSE=mse(Imask2,Imru2)
    #print('MSE (watermark , extracted watermarked)='+str(MSE))

    Imru=np.uint8(255*Imru)

    #save Extracted watermarked image
    FileName=s[0:len(s)-4]
    fn='DCT_Extracted_Watermark_'+FileName+'.png'#set file name
    img = Image.fromarray(Imru)
    Floc='./result/DCT/' +fn#set file path+file name
    img.save( Floc, 'png')#save image at location
    print('Watermark Image displayed and saved at: '+Floc)

    return

def SVD(coverImage, watermarkImage):
    print("SVD Watermarking Started")
    cv2.imshow('Cover Image: '+FName,coverImage)
    [m,n]=np.shape(coverImage)
    coverImage=np.double(coverImage)

    watermarkImage=cv2.resize(watermarkImage,(m,n))#resize to size of cover image
    watermarkImageBW = cv2.threshold(watermarkImage, 127, 255, cv2.THRESH_BINARY)[1]
    watermarkImageBW=watermarkImageBW/255# convert to 0,1 range
    cv2.imshow('Watermark Image (Resized)', watermarkImageBW)

    #SVD of cover image
    Uimg,Simg,Vimg=np.linalg.svd(coverImage,full_matrices=1,compute_uv=1)
    Simg=np.diag(Simg)#convert to diagonal matrix
    Simg_temp=copy.deepcopy(Simg)# copy to new variable

    # Apply watermarking
    alpha=10
    Simg = Simg + (alpha*watermarkImageBW)# update simg

    # SVD of watermarked Simg
    U_SHL_w,S_SHL_w,V_SHL_w=np.linalg.svd(Simg,full_matrices=1,compute_uv=1)

    ## Applying inverse SVD
    watermarkedImage=np.uint8(np.dot((Uimg*S_SHL_w), Vimg))

    err=mse(coverImage,watermarkedImage)# calculate mse
    print('MSE (cover, watermarked)='+str(err))

    #save watermarked image
    watermarkedImage=np.uint8(watermarkedImage)
    fn='SVD_Watermarked_'+FName#set file name
    cv2.imshow('SVD_Watermarked_' + FName, watermarkedImage)

    img = Image.fromarray(watermarkedImage)
    Floc='./result/SVD/' +fn#set file path+file name
    img.save( Floc)#save image at location
    print('Watermarked Image saved at: '+Floc)

    ## EXTRACTION PROCESS
    # Appyly SVD on watermarked image
    Wimg, SWimg, VWimg = np.linalg.svd(watermarkedImage, full_matrices=1, compute_uv=1)

    # performing inverse SVD
    D_1 = np.dot((U_SHL_w * SWimg), V_SHL_w)
    # extracting watermark
    Watermark = np.abs(D_1 - Simg_temp) / alpha

    Watermark2=255*Watermark
    Watermark2 = cv2.threshold(Watermark2, 127, 255, cv2.THRESH_BINARY)[1]
    Watermark2 = np.uint8(Watermark2)
    cv2.imshow('Extracted Watermark :'+FName,Watermark2)

    img = Image.fromarray(Watermark2)
    fn = 'SVD_Extracted_Watermarked_' + FName  # set file name
    Floc='./result/SVD/' +fn#set file path+file name
    img.save( Floc)#save image at location
    print('Extracted Watermark Image saved at:'+Floc)

    err=mse(watermarkImage,Watermark2)
    #print('MSE (watermark, extracted watermark )= '+str(err))

    return

#option 5
def DWT_SVD(coverImage, watermarkImage):
    print("SVD Watermarking Started")
    cv2.imshow('Cover Image: '+FName,coverImage)
    coverImage=np.double(coverImage)


    # Apply DWT
    coeffC = pywt.dwt2(coverImage, 'haar')
    A, (H, V, D) = coeffC
    [m, n] = np.shape(A)

    watermarkImage=cv2.resize(watermarkImage,(m,n))#resize to size of cover image
    watermarkImageBW = cv2.threshold(watermarkImage, 127, 255, cv2.THRESH_BINARY)[1]
    watermarkImageBW=watermarkImageBW/255# convert to 0,1 range
    cv2.imshow('Watermark Image (Resized)', watermarkImageBW)


    # Apply SVD on A Band
    Uimg,Simg,Vimg=np.linalg.svd(A,full_matrices=1,compute_uv=1)
    Simg=np.diag(Simg)#convert to diagonal matrix
    Simg_temp=copy.deepcopy(Simg)# copy to new variable

    # Apply watermarking
    alpha=10
    Simg = Simg + (alpha*watermarkImageBW)# update simg

    # SVD of watermarked Simg
    U_SHL_w,S_SHL_w,V_SHL_w=np.linalg.svd(Simg,full_matrices=1,compute_uv=1)

    ## Applying inverse SVD
    A2=np.dot((Uimg*S_SHL_w), Vimg)
    #Perform inverse dwt and generate watermarked image
    coeffW= A2,(H, V, D)
    watermarkedImage = pywt.idwt2(coeffW, 'haar')
    watermarkedImage=np.round_(watermarkedImage)
    watermarkedImage=np.uint8(watermarkedImage)
    cv2.imshow('Watermarked Image: '+FName,watermarkedImage)

    #CALCULATE MSE
    err=mse(coverImage,watermarkedImage)
    print('MSE (cover image, watermarked image)=',err)
   
    # save watermarked image
    img = Image.fromarray(watermarkedImage)
    fn = 'DWTSVD_Watermarked_' + FName  # set file name
    Floc='./result/DWT_SVD/' +fn#set file path+file name
    img.save( Floc)#save image at location
    print('Extracted Watermark Image saved at:'+Floc)
    
    ## EXTRACTION PROCESS
    # Apply DWT
    coeffC = pywt.dwt2(watermarkedImage, 'haar')
    A, (H, V, D) = coeffC
    # Apply SVD on A Band
    Wimg, SWimg, VWimg = np.linalg.svd(A, full_matrices=1, compute_uv=1)
    # performing inverse SVD
    D_1 = np.dot((U_SHL_w * SWimg), V_SHL_w)
    # extracting watermark
    Watermark = np.abs(D_1 - Simg_temp) / alpha

    Watermark2=255*Watermark
    Watermark2 = cv2.threshold(Watermark2, 127, 255, cv2.THRESH_BINARY)[1]
    Watermark2 = np.uint8(Watermark2)
    cv2.imshow('Extracted Watermark: '+FName,Watermark2)
    err=mse(watermarkImage,Watermark2)
    print('MSE (cover image, watermarked image)=',err)

    # save extracted watermark image
    img = Image.fromarray(Watermark2)
    fn = 'DWTSVD_Extracted_Watermarked_' + FName  # set file name
    Floc='./result/DWT_SVD/' +fn#set file path+file name
    img.save( Floc)#save image at location
    print('Extracted Watermark Image saved at:'+Floc)
    
    return watermarkedImage

#Option 6
def DWT_DCT_SVD(coverImage, watermarkImage):
    coverImage = cv2.imread(st,0)#read cover image
    coverImage = cv2.resize(coverImage,(512,512))
    cv2.imshow('Cover Image : '+FName,coverImage)

    # Read watermark image
    watermarkImage= cv2.imread('watermarkImage4.PNG',0)
    watermarkImage = cv2.resize(watermarkImage,(32,32))
    #cv2.namedWindow("Watermark Image", cv2.WINDOW_NORMAL)
    cv2.imshow('Watermark Image',watermarkImage)

    print("Read watermark image")

    # Apply DWT
    coeffC = pywt.dwt2(coverImage, 'haar')
    A, (H, V, D) = coeffC
    print("DWT of coverimage done");
    [Ha,Wa]=np.shape(A)

    #Generate DCT Block
    Bh=np.uint8(Ha/8)#total block can be processed in horizontal direction
    Bw=np.uint8(Wa/8)#total blocks can be processed in vertical direction

    DMat=np.zeros((Bh, Bw),  np.float32)
    dblock=7
    ih=0
    iw=0
    for i in range(0, Ha-1, 8):
        iw=0
        for j in range(0, Wa-1, 8):
            i2=i+8;j2=j+8;
            Block=A[i:i2,  j:j2]# crop the 8x8 image block
            DctBlock=cv2.dct(Block)#convert to dct
            #DMat[ih, iw]=DctBlock[dblock, dblock]
            Dval=DctBlock[dblock, dblock]#read dct value
            DMat[ih, iw]=Dval#store it to 32x32 matrix

            iw=iw+1
            #print('['+str(i)+','+str(j)+']')
        ih=ih+1

    print('DCT Matrix of A band generated')

    ## Apply svd on Dmat and embed watermark image
    alpha=0.1
    #[DMat2,U_SHL_w,V_SHL_w,Simg_temp]=WSVD(DMat,watermarkImage,alpha)
    Uimg,Simg,Vimg=np.linalg.svd(DMat,full_matrices=1,compute_uv=1)
    Simg=np.diag(Simg)#convert to diagonal matrix
    Simg_temp=Simg# save a copy of original Simg

    [x, y]=np.shape(watermarkImage)
    Simg =Simg + alpha * watermarkImage
    # SVD of watermarked S
    U_SHL_w,S_SHL_w,V_SHL_w=np.linalg.svd(Simg,full_matrices=1,compute_uv=1);
    #S_SHL_w=np.diag(S_SHL_w)

    ## Applying inverse SVD
    #DMat2 =Uimg* S_SHL_w * Vimg
    DMat2=np.dot((Uimg*S_SHL_w), Vimg)

    print('SVD Applied on A band')
    A2=A
    ih=0;iw=0;
    for i in range(0, Ha-1, 8):
        iw=0
        for j in range(0, Wa-1, 8):
            Block=A2[i:i+8,j:j+8];
            DctBlock=cv2.dct(Block);
            DctBlock[dblock,dblock]=DMat2[ih,iw];
            Block2=cv2.idct(DctBlock);
            A2[i:i+8,j:j+8]=Block2;
            iw=iw+1;
        ih=ih+1
    print('Inverse SVD Applied updated A band generated')

    #Perform inverse dwt and generate watermarked image
    coeffW=A2, (H, V, D)
    watermarkedImage = pywt.idwt2(coeffW, 'haar')
    print('Inverse DWT Applied')
    watermarkedImage=np.round_(watermarkedImage)
    watermarkedImage=np.uint8(watermarkedImage)

    #save watermarked image
    fn='DWTDCTSVD_Watermarked_'+FName#set file name
    cv2.imshow('DWTDCTSVD_Watermarked_' + FName, watermarkedImage)

    # CALCULATE MSE
    MSE=mse(coverImage,watermarkedImage)
    print('MSE (coverimage, watermarked)='+str(MSE))

    img = Image.fromarray(watermarkedImage)
    Floc='./result/DWTDCTSVD/' +fn#set file path+file name
    img.save( Floc)#save image at location
    print('Watermark Image displayed and saved at:'+Floc)

    ############## Apply Inverse Extraction process
    print('****Watermark Extraction******')
    coeffC = pywt.dwt2(watermarkedImage, 'haar')
    Da, (Dh,Dv,Dd) = coeffC
    print('dwt applied on image')

    # Extract DCT
    DMatD=np.zeros((Bh, Bw),  np.float32)

    ih=0;iw=0;
    for i in range(0, Ha-1, 8):
        iw=0;
        for j in range(0, Wa-1, 8):
            Block=Da[i:i+8,j:j+8];
            DctBlock=cv2.dct(Block);
            DMatD[ih,iw]=DctBlock[dblock,dblock];
            iw=iw+1;
        ih=ih+1;
    print('DCT block generated')

    #Simg_temp=np.diag(Simg_temp);
    #Watermark=ISWD(DMatD,U_SHL_w,V_SHL_w,Simg_temp,alpha)
    [x, y]=np.shape(DMatD)
    Wimg,SWimg,VWimg=np.linalg.svd(DMatD,full_matrices=1,compute_uv=1)

    # apply inverse SVD, use U,V component of 2nd SVD
    #SWimg=np.diag(SWimg)
    #D_1=U_SHL_w * SWimg * V_SHL_w
    D_1=np.dot((U_SHL_w*SWimg), V_SHL_w )
    #D_1=np.matmul(U_SHL_w,np.matmul(SWimg, V_SHL_w) )


    WatermarkR= (D_1-Simg_temp)/alpha
    #Watermark=np.round(255*Watermark);
    WatermarkR=(255*WatermarkR);
    (thresh, WatermarkR) = cv2.threshold(WatermarkR, 127, 255, cv2.THRESH_BINARY)

    print('Inverse SVD Performed, watermark image extracted')
    WatermarkR=np.uint8(WatermarkR)
    #cv2.namedWindow("Extracted Watermark Image: "+FName, cv2.WINDOW_NORMAL)
    cv2.imshow('DWTDCTSVD_Extracted_Watermarked_'+FName,WatermarkR)

    # CALCULATE MSE
    MSE=mse(watermarkImage,WatermarkR)
    #print('MSE (watermark, extracted watermark)='+str(MSE))

    #save Extracted watermarked image
    FileName=s[0:len(s)-4]
    fn='DWTDCTSVD_Extracted_Watermark_'+FileName+'.png'#set file name
    img = Image.fromarray(WatermarkR)
    Floc='./result/DWTDCTSVD/' +fn#set file path+file name
    img.save( Floc, 'png')#save image at location
    print('Watermark Image displayed and saved at:'+Floc)

    return

def DWT_DFT_SVD(coverImage, watermarkImage):
    coverImage = cv2.resize(coverImage,(512,512))
    cv2.imshow('Cover Image',coverImage)
    watermarkImage = cv2.resize(watermarkImage,(256,256))
    cv2.imshow('Watermark Image',watermarkImage)
    coverImage = np.float32(coverImage)

    coverImage /= 255
    coeff = pywt.dwt2(coverImage, 'haar')
    cA, (cH, cV, cD) = coeff

    watermarkImage = np.float32(watermarkImage)
    watermarkImage_dct = cv2.dct(watermarkImage)

    cA_dft = cv2.dft(cA)

    UcA_dct,ScA_dct,VcA_dct=np.linalg.svd(cA_dft,full_matrices=1,compute_uv=1)
    uw,sw,vw=np.linalg.svd(watermarkImage,full_matrices=1,compute_uv=1)

    #Embedding
    alpha=0.1
    sA=np.zeros((256,256),np.uint8)
    sA[:256,:256]=np.diag(ScA_dct)
    sW=np.zeros((256,256),np.uint8)
    sW[:256,:256]=np.diag(sW)
    W=sA+alpha*sW

    u1,w1,v1=np.linalg.svd(W,full_matrices=1,compute_uv=1)
    ww=np.zeros((256,256),np.uint8)
    ww[:256,:256]=np.diag(w1)
    Wmodi=np.matmul(UcA_dct,np.matmul(ww,VcA_dct))

    widct= cv2.idct(Wmodi)
    watermarkedImage=pywt.idwt2((widct,(cH,cV,cD)),'haar')
    cv2.imshow('watermarkedImage',watermarkedImage)
    #return watermarkedImage

if __name__ == "__main__":

    cv2.destroyAllWindows()#close all previos window
    watermarkImage = cv2.imread('watermarkImage.JPG',0)
    #cv2.imshow('Original Watermark Image', watermarkImage)
    FuncName=['DWT', 'DCT', 'DFT', 'SVD', 'DWT_SVD', 'DWT_DCT_SVD', 'DWT_DFT_SVD']
    options = {
        1: DWT,
        2: DCT,
        3: DFT,
        4: SVD,
        5: DWT_SVD,
        6: DWT_DCT_SVD,
        7: DWT_DFT_SVD
    }

    val = input('What type of embedding you want to perform? \
                \n1.DWT\
                \n2.DCT\
                \n3.DFT\
                \n4.SVD\
                \n5.DWT-SVD\
                \n6.SVD-DCT-DWT\
                \n7.SVD-DFT-DWT\
                \nEnter your option: ')

    watermarking_function = options.get(int(val), None)


    if watermarking_function:

        f=glob.glob('.\coverimages\*.jpg')#search files with extension .jpg in same folder
        NFiles=len(f)#find total files found
        print('Files In folder')
        print(f)#print all file names

        for Fc in range(0, NFiles):#run till all fies processed
        #Fc=1;
            st=f[Fc]#select the file name
            s=st[14:len(st)]#remove unnecessary heading of file extension
            FName=st[14:len(st)]

            coverImage = cv2.imread(st,0)
            print('\n**----------------------------------------')
            print('Processing File:'+st)#print file name
            watermarking_function(coverImage,watermarkImage)



    else:
        print("Invalid Option")
        exit(1)

    k = 0
    while k != 48:
       k = cv2.waitKey(100)
    cv2.destroyAllWindows()
