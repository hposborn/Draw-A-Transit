'''

THIS IS THE PYTHON ADAPTION OF JOE LLAMA'S IDL TRANSIT CODE
# PURPOSE:
#  This routine takes in a transit lightcurve and, by creating shapes, attempts to find a simple way to simulate the eclipse
#
# CATEGORY:
#  SCIENCE
#
# CALLING SEQUENCE:
#  transit
#
# INPUTS:
# Lightcurve
# 
# OUTPUT:
#    Estimation of potential shape of occulting body, 
 
# MODIFICATION HISTORY:
#  Sept 2014: Created by Joe Llama (Lowell) for Hugh Osborn (Warwick).
#   Adapted to Python by Hugh Osborn
#
#
# To do:
# Change 3D array to for-loop to trade Memory errors for speed.
'''

import numpy as np
import matplotlib as plt
import me.planetlib as pl
import pygame
from os import sys
'''
def FindTransit(lc):
    #Uses deepest point and iterates until it finds a background
    deepest=np.where(lc[:, 1]==np.min(lc[:, 1]))
    for ts in range(0, np.max(np.hstack((len(lc[deepest:, 0]), len(lc[:deepest, 0]))) )):
        in0, out0 = deepest-ts, deepest+ts
        chisq=
        zmodel = 
        outof=range(len(lc[:, 0]))[:in0]
        times,flux,sigma = data[:,0],data[:,1],data[:,2]
        ChiSq  = ((((np.tile(1.0, len(outof)) - lc[outof, 1])/(lc[outof, 2]))**2.).ravel().sum())/(len(lc[outof, 0]))
    #Do this later
'''
    
def DrawShape(oldshape=[]):
    #This draws using circles the star, and using polygons an occulter shape using a PyGamewindow
    PGPoints=[]
    screen = pygame.display.set_mode((800, 800))
    if oldshape!=[]:
        pygame.draw.polygon(screen,(255,0,255), oldshape)
    pr=True
    part=1;hold=0
    star=np.zeros((2, 2))
    while pr==True:        
        for event in pygame.event.get():
            if part==2:
                #If sun already drawn, drawing polygon
                if hold==1:
                    #if mouse ehld down, recording mouse position
                    PGPoints+=[pygame.mouse.get_pos()]
                    if len(PGPoints)>2:
                        #when longer than 2 starts drawing polygons
                        screen.fill( (0,0,0 ))
                        pygame.draw.circle(screen,(255,204,0), star[0], star[1])
                        pygame.draw.polygon(screen,(0,0,255), np.array(PGPoints))
                        pygame.display.update()
                if event.type == pygame.MOUSEBUTTONUP:
                    #End of hold. 
                    hold=0
                    PGPoints+=[pygame.mouse.get_pos()]
                if event.type == pygame.MOUSEBUTTONDOWN:
                    #Start of hold
                    hold=1
                if event.type == pygame.KEYDOWN:
                    #At the end, Enter/escape exits window
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                        pygame.quit(); pr=False#sys.exit()
            if part==1:
            #Drawing star
                if hold==1:
                    screen.fill( (0,0,0 ))
                    pygame.draw.circle(screen,(255,204,0), (0.5*(star[0][0]+np.array(pygame.mouse.get_pos())[0]),0.5*(star[0][1]+np.array(pygame.mouse.get_pos())[1])), (((pygame.mouse.get_pos()-star[0])**2).sum()**0.5)/2)
                    pygame.display.update()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    #Finds initial point of star radius and records that mouse buttong is held down
                    star[0]=np.array(pygame.mouse.get_pos())
                    hold=1
                if event.type == pygame.MOUSEBUTTONUP:
                    star[1]=np.array(pygame.mouse.get_pos())
                    #calculating centre and radius from two edge points
                    star=[np.average(star, axis=0), (((star[1]-star[0])**2).sum()**0.5)/2]
                    pygame.draw.circle(screen,(255,204,0), star[0], star[1])
                    pygame.display.update()
                    part=2;hold=0
            if event.type == pygame.QUIT:
                pygame.quit(); pr=False; #sys.exit();
    return PGPoints,  star

def Transit(lc, PGpoints, intrans, starpos):
    #Takes in shape. Draws it as array. Moves it across the Star. Draws lightcurve. Compares with known ligthcurve
    movie=0
    #Drawing star
    star = DrawStar(starpos)
    #Normalising total flux to 1.0
    star/=np.sum(star)
    #Turning the polygon into an array
    from PIL import Image, ImageDraw
    img = Image.new('L', (800, 800), 0)
    ImageDraw.Draw(img).polygon(PGpoints, outline=1, fill=1)
    objbox=np.array(img)
    objbox=1-objbox
    #Setting animation to step 4 pixels at a time
    step=4.0
    lc=lc[lc[:, 0].argsort(), :]
    timearr=np.linspace(lc[0,0],lc[-1,0],1600)
    print timearr
    print lc[:, 0]
    lightcurvemodel=[]
    #Drawing initial plots
    fig=plt.figure(figsize=(6,  12))
    ax1=fig.add_subplot(211)
    ax1.imshow(star[:, :800],cmap='hot');
    ax2=fig.add_subplot(212)
    ax2.plot(lc[:,0],lc[:,1],'--k')
    ax2.plot(timearr, np.tile(1.0, 1600),'--', color='#CCCCCC')
    plt.pause(0.002)
    #Sweep across the star, keeping the object fixed
    for pix in range(400,2000, step):
        #Draw an 800x800 box from the star
        starbox=star[:,(pix-400):(pix+400)]
        total=objbox*starbox
        #if np.sum(starbox)!=0:#skipping to where there is some of the star on the screen
        ax1.cla()
        ax1.imshow(total,cmap='hot');
        #plotting lightcurve
        rest=np.sum(star[:,:(pix-400)])+np.sum(star[:, (pix+400):])
        lightcurvemodel+=[np.sum(total)+np.sum(rest)]
        ax2.plot(timearr[pix-400],np.array(lightcurvemodel)[-1],'.r')
        plt.pause(0.002)
        movie=1 #Setting to record pngs as movie
        if movie==1:
            plt.savefig('TransitMovie/Movie'+str(pix)+'.png')
    plt.pause(1)
    np.savetxt('OutLightcurve.txt',np.column_stack((timearr[0:len(lightcurvemodel)],np.array(lightcurvemodel))))

def DrawStar(stars):
    star=dist_circle([800,2400], stars[0][0],  stars[0][1]+800)#Drawing array where
    rstar=stars[1]
    star = star/rstar
    # Generate the outline of the star - ie. 1s inside stellar disk, 0s outside
    outline=np.where(star>1)
    star[outline]=0  #remove values outside stellar disk
    # limb-darken the star 
    ss = star**2.
    limb_array = [0.0188, 1.5337, -0.9613, 0.2356]
    star = 1 -limb_array[0]*(1.-(1.-ss)**(0.25)) - limb_array[1]*(1.-(1.-ss)**(0.5))  -  limb_array[2]*(1.-(1.-ss)**(0.75)) -  limb_array[3]*(1.-(1.-ss))
    star[outline]=0
    return star

def dist_circle(n, xcen=0 ,ycen=0):
    if type(n)==int or type(n)==float or type(n)==np.float64 or type(n)==np.int64:
        #Rectangle
        ny = n
        nx = n
    else:
        #Square
        nx = n[0]
        ny = n[1] 

    if xcen==0 and ycen==0:
        #Putting it in the middle
        xcen = (nx)/2.
        ycen = (ny)/2.
    
    x_2 = (np.arange(0, nx, 1.0) - xcen)**2     #X distances (squared)
    y_2 = (np.arange(0, ny, 1.0)  - ycen)**2     #Y distances (squared)  
    im = np.zeros((nx, ny))      #Make uninitialized output array
    for i in range(nx):              #Row loop
        for j in range(ny):
            im[i,j] = np.sqrt(x_2[i] + y_2[j])     #Euclidian distance
    return im

def Run():
    lc=np.genfromtxt('KIC08462852_examplelc.txt')
    lccut=lc[abs(lc[:, 0]-1519.6)<6, :]
    shape, starpos=DrawShape([])
    Transit(lccut, shape,abs(lc[:, 0]-1519.6)<3, starpos)

if __name__ == '__main__':
    Run()
