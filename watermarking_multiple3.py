import numpy as np
import cv2
import pywt
import glob
from PIL import Image

def applyWatermarkDFT(imageMatrix, watermarkMatrix, alpha):
   shiftedDFT = np.fft.fftshift(np.fft.fft2(imageMatrix))
   watermarkedDFT = shiftedDFT + alpha * watermarkMatrix
   watermarkedImage = np.fft.ifft2(np.fft.ifftshift(watermarkedDFT))
   print("DFT Watermarking done")
   return watermarkedImage

def DFT(coverImage, watermarkImage):
    coverImage = cv2.resize(coverImage,(300,300))
    cv2.imshow('Cover Image: '+FName,coverImage)
    watermarkImage = cv2.resize(watermarkImage,(300,300))
    cv2.imshow('Watermark Image',watermarkImage)

    watermarkedImage = applyWatermarkDFT(coverImage,watermarkImage,10)
    watermarkedImage = np.uint8(watermarkedImage)
    
    #save watermarked image
    fn='DFT_Watermarked_'+FName#set file name
    cv2.imshow('DFT_Watermarked_' + FName, watermarkedImage)

    img = Image.fromarray(watermarkedImage)
    Floc='./result/DFT/' +fn#set file path+file name
    img.save( Floc)#save image at location
    print('Watermark Image displayed and saved at:'+Floc)

    #return watermarkedImage
    

def DWT(coverImage, watermarkImage):
    coverImage = cv2.resize(coverImage,(300,300))
    cv2.imshow('Cover Image: '+FName,coverImage)
    watermarkImage = cv2.resize(watermarkImage,(150,150))
    cv2.imshow('Watermark Image',watermarkImage)

    #DWT on cover image
    coverImage =  np.float32(coverImage)
    coverImage /= 255;
    coeffC = pywt.dwt2(coverImage, 'haar')
    cA, (cH, cV, cD) = coeffC
    print("DWT of coverimage done");
    watermarkImage = np.float32(watermarkImage)
    watermarkImage /= 255;

    #Embedding
    alpha=0.1
    coeffW = (cA + alpha*watermarkImage, (cH, cV, cD))
    watermarkedImage = pywt.idwt2(coeffW, 'haar')
    print("Embedding Watermark done..")
    #cv2.imshow('Watermarked Image', watermarkedImage)
    
    #Extraction
    print("Extraction watermarking start...")
    coeffWM = pywt.dwt2(watermarkedImage, 'haar')
    hA, (hH, hV, hD) = coeffWM

    extracted = (hA-cA)/alpha
    extracted *= 255
    extracted = np.uint8(extracted)
    print("Watermark Extracted...")
    #cv2.imshow('Extracted', extracted)
    watermarkedImage=255*watermarkedImage
    watermarkedImage = np.uint8(watermarkedImage)

    #save watermarked image
    fn='DWT_Watermarked_'+FName#set file name
    cv2.imshow('DWT_Watermarked_' + FName, watermarkedImage)

    img = Image.fromarray(watermarkedImage)
    Floc='./result/DWT/'+fn#set file path+file name
    img.save(Floc)#save image at location
    print('Watermarked Image displayed and saved at:'+Floc)
    
    #save extracted watermark image
    fn='DWT_Extracted_Watermark_'+FName#set file name
    cv2.imshow('DWT_Extracted_Watermark_' + FName, extracted)

    img = Image.fromarray(extracted)
    Floc='./result/DWT/' +fn#set file path+file name
    img.save( Floc)#save image at location
    print('Extracted Watermark Image displayed and saved at:'+Floc)

    
    return watermarkedImage

def DCT(coverImage, watermarkImage):
    coverImage = cv2.resize(coverImage,(512,512))
    cv2.imshow('Cover Image: '+FName,coverImage)
    watermarkImage = cv2.resize(watermarkImage,(64,64))
    cv2.imshow('Watermark Image: ',watermarkImage)

    coverImage =  np.float32(coverImage)
    watermarkImage = np.float32(watermarkImage)
    watermarkImage /= 255

    blockSize = 8
    c1 = np.size(coverImage, 0)
    c2 = np.size(coverImage, 1)
    max_message = (c1*c2)//(blockSize*blockSize)

    w1 = np.size(watermarkImage, 0)
    w2 = np.size(watermarkImage, 1)
    print("DCT Row blocks=", w1);
    print("DCT column blocks=", w2);
    
    watermarkImage = np.round(np.reshape(watermarkImage,(w1*w2, 1)),0)

    if w1*w2 > max_message:
        print ("Message too large to fit")

    message_pad = np.ones((max_message,1), np.float32)
    message_pad[0:w1*w2] = watermarkImage

    watermarkedImage = np.ones((c1,c2), np.float32)

    k=50
    a=0
    b=0

    for kk in range(max_message):
        dct_block = cv2.dct(coverImage[b:b+blockSize, a:a+blockSize])
        if message_pad[kk] == 0:
            if dct_block[4,1]<dct_block[3,2]:
                temp=dct_block[3,2]
                dct_block[3,2]=dct_block[4,1]
                dct_block[4,1]=temp
        else:
            if dct_block[4,1]>=dct_block[3,2]:
                temp=dct_block[3,2]
                dct_block[3,2]=dct_block[4,1]
                dct_block[4,1]=temp

        if dct_block[4,1]>dct_block[3,2]:
            if dct_block[4,1] - dct_block[3,2] <k:
                dct_block[4,1] = dct_block[4,1]+k/2
                dct_block[3,2] = dct_block[3,2]-k/2
        else:
            if dct_block[3,2] - dct_block[4,1]<k:
                dct_block[3,2] = dct_block[3,2]+k/2
                dct_block[4,1] = dct_block[4,1]-k/2

        watermarkedImage[b:b+blockSize, a:a+blockSize]=cv2.idct(dct_block)
        if a+blockSize>=c1-1:
            a=0
            b=b+blockSize
        else:
            a=a+blockSize

    watermarkedImage_8 = np.uint8(watermarkedImage)
    
    #save watermarked image
    fn='DCT_Watermarked_'+FName#set file name
    cv2.imshow('DCT_Watermarked_' + FName, watermarkedImage_8)

    img = Image.fromarray(watermarkedImage_8)
    Floc='./result/DCT/'+fn#set file path+file name
    img.save(Floc)#save image at location
    print('Watermarked Image displayed and saved at:'+Floc)
    
    return 
    
def SVD(coverImage, watermarkImage):
    watermarkImageBW = cv2.threshold(watermarkImage, 127, 255, cv2.THRESH_BINARY)[1]
    watermarkImageBW=watermarkImageBW/255

    print("SVD Started")
    cv2.imshow('Cover Image: '+FName,coverImage)
    [m,n]=np.shape(coverImage)
    coverImage=np.double(coverImage)
    cv2.imshow('Watermark Image',watermarkImage)
    watermarkImage = np.double(watermarkImage)
    watermarkImageBW=cv2.resize(watermarkImageBW,(m,n))

    #SVD of cover image
    print("SVD Elements Generation..")
    ucvr,wcvr,vtcvr=np.linalg.svd(coverImage,full_matrices=1,compute_uv=1)


    WcvrDiag=np.diag(wcvr)

    [x,y] = np.shape(watermarkImageBW)

    #modifying diagonal component
    print("Updating U matrix...")
    for i in range(0,x):
      for j in range(0,y):
            ucvr[i,j]=(ucvr[i,j]+0.01*watermarkImageBW[i,j])

    print("Applying Inverse SVD...")
    wimg=np.matmul(ucvr,np.matmul(WcvrDiag,vtcvr))
    wimg=np.double(wimg)

    print("Preprocessing watermarked Image")
    watermarkedImage = np.zeros(wimg.shape,np.double)
    normalized=cv2.normalize(wimg,watermarkedImage,1.0,0.0,cv2.NORM_MINMAX)
    print("Watermarked Image generated...")
    watermarkedImage=255*watermarkedImage;
    #save watermarked image
    fn='SVD_Watermarked_'+FName#set file name
    watermarkedImage = np.uint8(watermarkedImage)
    cv2.imshow('SVD_Watermarked_' + FName, watermarkedImage)

    img = Image.fromarray(watermarkedImage)
    Floc='./result/SVD/' +fn#set file path+file name
    img.save( Floc)#save image at location
    print('Watermark Image displayed and saved at:'+Floc)
    return 

#option 5
def DWT_SVD(coverImage, watermarkImage):
    cv2.imshow('Cover Image: '+FName,coverImage)
    [m,n]=np.shape(coverImage)
    [Wh,Ww]=np.shape(watermarkImage);
    coverImage=np.double(coverImage)
    cv2.imshow('Watermark Image',watermarkImage)
    watermarkImage = np.double(watermarkImage)
    #watermarkImageBW=cv2.resize(watermarkImageBW,(m,n))

    # Applying DWT on cover image and getting four sub-bands
    coverImage =  np.float32(coverImage)
    coverImage /= 255;
    coeffC = pywt.dwt2(coverImage, 'haar')
    cA, (cH, cV, cD) = coeffC
    print("DWT applied on CoverImage")
   
    #SVD on cA
    uA,wA,vA=np.linalg.svd(cA,full_matrices=1,compute_uv=1)
    [a1,a2]=np.shape(cA)
    WA=np.zeros((a1,a2),np.uint8)
    WA[:a1,:a2]=np.diag(wA)
    print("SVD applied on cA...")
    
    #SVD on cH
    uH,wH,vH=np.linalg.svd(cH,full_matrices=1,compute_uv=1)
    [h1,h2]=np.shape(cH)
    WH=np.zeros((h1,h2),np.uint8)
    WH[:h1,:h2]=np.diag(wH)
    print("SVD applied on cH...")

    #SVD on cV
    uV,wV,vV=np.linalg.svd(cV,full_matrices=1,compute_uv=1)
    [v1,v2]=np.shape(cV)
    WV=np.zeros((v1,v2),np.uint8)
    WV[:v1,:v2]=np.diag(wV)
    print("SVD applied on cV...")
    
    #SVD on cD
    uD,wD,vD=np.linalg.svd(cD,full_matrices=1,compute_uv=1)
    [d1,d2]=np.shape(cV)
    WD=np.zeros((d1,d2),np.uint8)
    WD[:d1,:d2]=np.diag(wD)
    print("SVD applied on cD...")

    #SVD on watermarke image
#    watermarkImage2=cv2.resize(watermarkImage,(round(Wh/2),round(Ww/2)),interpolation = cv2.INTER_AREA)
    watermarkImage2 = cv2.resize(watermarkImage, (a1, a2), interpolation=cv2.INTER_AREA)

    uw,ww,vw=np.linalg.svd(watermarkImage2,full_matrices=1,compute_uv=1)
    [x,y] = np.shape(watermarkImage2)
    WW=np.zeros((x,y),np.uint8)
    WW[:x,:y]=np.diag(ww)
    print("SVD applied on watermark image...")
    
    #Embedding Process
    for i in range (0,x):
        for j in range (0,y):
            WA[i,j]=WA[i,j]+0.01*WW[i,j]
   
    for i in range (0,x):
        for j in range (0,y):
            WV[i,j]=WV[i,j]+0.01*WW[i,j]

    for i in range (0,x):
        for j in range (0,y):
            WH[i,j]=WH[i,j]+0.01*WW[i,j]

    for i in range (0,x):
        for j in range (0,y):
            WD[i,j]=WD[i,j]+0.01*WW[i,j]
    print("Watermark image embedded in SVD of each band..")
    
    #Inverse of SVD
    cAnew=np.dot(uA,(np.dot(WA,vA)))
    cHnew=np.dot(uH,(np.dot(WH,vH)))
    cVnew=np.dot(uV,(np.dot(WV,vA)))
    cDnew=np.dot(uD,(np.dot(WD,vD)))
    coeff=cAnew,(cHnew,cVnew,cDnew)
    print("Applying inverse SVD of each band and dwt compoenents generated...")

    #Inverse DWT to get watermarked image
    watermarkedImage = pywt.idwt2(coeff,'haar')
    print("Inverse DWT Applied to get Watermarked Image...")
    
    #save watermarked image
    watermarkedImage=255*watermarkedImage
    watermarkedImage=np.uint8(watermarkedImage)
    fn='DWT_SVD_Watermarked_'+FName#set file name
    cv2.imshow('DWT_SVD_Watermarked_' + FName, watermarkedImage)
    watermarkedImage = np.uint8(watermarkedImage)
    img = Image.fromarray(watermarkedImage)
    Floc='./result/DWT_SVD/' +fn#set file path+file name
    img.save( Floc)#save image at location
    print('Watermark Image displayed and saved at:'+Floc)    
    
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

    img = Image.fromarray(watermarkedImage)
    Floc='./result/DWTDCTSVD/' +fn#set file path+file name
    img.save( Floc)#save image at location
    print('Watermark Image displayed and saved at:'+Floc)

    print('****Watermark Extraction******')



    ############## Apply Inverse Extraction process

    coeffC = pywt.dwt2(watermarkedImage, 'haar')
    Da, (Dh,Dv,Dd) = coeffC
    print('dwt aaplied on image')

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


    Watermark= (D_1-Simg_temp)/alpha
    #Watermark=np.round(255*Watermark);
    Watermark=(255*Watermark);
    (thresh, Watermark) = cv2.threshold(Watermark, 127, 255, cv2.THRESH_BINARY)

    print('Inverse SVD Performed, watermark image extracted')
    Watermark=np.uint8(Watermark)
    #cv2.namedWindow("Extracted Watermark Image: "+FName, cv2.WINDOW_NORMAL)
    cv2.imshow('DWTDCTSVD_Extracted_Watermarked_'+FName,Watermark)    

    #save Extracted watermarked image
    FileName=s[0:len(s)-4]
    fn='DWTDCTSVD_Extracted_Watermark_'+FileName+'.png'#set file name
    img = Image.fromarray(Watermark)
    Floc='./result/DWTDCTSVD/' +fn#set file path+file name
    img.save( Floc, 'png')#save image at location
    print('Watermark Image displayed and saved at:'+Floc)

    return 
 
if __name__ == "__main__":

    cv2.destroyAllWindows()#close all previos window
    watermarkImage = cv2.imread('watermarkImage.JPG',0)

    FuncName=['DWT', 'DCT', 'DFT', 'SVD', 'DWT_SVD', 'DWT_DCT_SVD', 'DWT_DFT_SVD']
    options = {
        1: DWT,
        2: DCT,
        3: DFT,
        4: SVD,
        5: DWT_SVD,
        6: DWT_DCT_SVD,
    }

    val = input(' please select the watermarking system to perform \
                \n1.DWT\
                \n2.DCT\
                \n3.DFT\
                \n4.SVD\
                \n5.DWT-SVD\
                \n6.SVD-DCT-DWT\
                \nEnter your option: ')
    
    watermarking_function = options.get(int(val), None)
    

    if watermarking_function:
        
        f=glob.glob('.\coverimages\*.jpg')#search files with extension .jpg in same folder
        NFiles=len(f)#find total files found
        print('Files In folder')
        print(f)#print all file names
        
        for Fc in range(0, NFiles):#run till all fies processed
        #Fc=0;
            st=f[Fc]#select the file name
            s=st[14:len(st)]#remove unnecessary heading of file extension
            FName=st[14:len(st)]

            coverImage = cv2.imread(st,0)
            print('\n\n**-------------------')
            print('Processing File:'+st+'\n')#print file name
            watermarking_function(coverImage,watermarkImage)

    
        
    else:
        print("Invalid Option")
        exit(1)

    k = 0
    while k != 48:
       k = cv2.waitKey(100)
    cv2.destroyAllWindows()
