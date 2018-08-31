import cv2
import numpy as np
import time

file  = "/home/tomas/Desktop/video.mp4"
video = cv2.VideoCapture(file, 0) #si uso webcam, poner VideoCapture(0) o (1), probar con cual anda

def multiply_tuple(tu, factor):
    newtuple = tuple()
    for i in range(len(tu)):
        if type(tu[i]) == tuple:
            for m in range(len(tu[i])):
                if m == 0:
                    auxtuple = tuple([tu[i][m] * factor])
                else:
                    auxtuple += tuple([tu[i][m] * factor])
            newtuple += tuple([auxtuple])
        else:
            # print(tu[i])
            newtuple += tuple([tu[i] * 3, ])
    return newtuple

green = np.uint8([[[0, 255, 0]]])
yellow = np.uint8([[[0, 255, 255]]])
red = np.uint8([[[0, 0, 255]]])
blue = np.uint8([[[255, 0, 0]]])

green_hsv = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
yellow_hsv = cv2.cvtColor(yellow, cv2.COLOR_BGR2HSV)
red_hsv = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
blue_hsv = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)

green_hsv = green_hsv[0, 0, 0]
yellow_hsv = yellow_hsv[0, 0, 0]
red_hsv = red_hsv[0, 0, 0]
blue_hsv = blue_hsv[0, 0, 0]

print(green_hsv)
print(yellow_hsv)
print(red_hsv)
print(blue_hsv)

green_lower =   np.array([green_hsv - 20, 50, 30])
green_upper =   np.array([green_hsv + 20, 255, 255])
yellow_lower =  np.array([yellow_hsv - 10, 175, 175])
yellow_upper =  np.array([yellow_hsv + 30, 255, 255])
red_lower_1 =   np.array([180 - 10, 120, 120])
red_upper_1 =   np.array([180, 255, 255])
red_lower_2 =   np.array([red_hsv, 120, 120])
red_upper_2 =   np.array([red_hsv + 10, 255, 255])
blue_lower =    np.array([blue_hsv - 15, 100, 70])
blue_upper =    np.array([blue_hsv + 15, 255, 255])
white_lower =   np.array([0, 0, 150])
white_upper =   np.array([179, 50, 255])

kernel = np.ones((9, 9), np.uint8)
kernel2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) #sharpening kernel

frame_height = int(video.get(4))
frame_width = int(video.get(3))

#out = cv2.VideoWriter('/home/tomas/Desktop/outpy.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height), True)
scaleFactor = 0.2

time.sleep(0.5)
rate = video.get(cv2.CAP_PROP_FPS)
while True:
    ok, imagecolor_orig = video.read()   #si uso video
    if imagecolor_orig is None:
        print("Fin de video")
        break
    imagecolor_orig = cv2.resize(imagecolor_orig, None, fx = 0.8, fy = 0.8)
    imagecolor = imagecolor_orig.copy()
    imagecolor = cv2.resize(imagecolor ,None,fx=scaleFactor, fy=scaleFactor, interpolation = cv2.INTER_LINEAR)
    imagecolor_hsv = cv2.cvtColor(imagecolor, cv2.COLOR_BGR2HSV)
    imagecolor_bilateralFilter = cv2.bilateralFilter(imagecolor_hsv, 6, 50, 50)

    green_mask = cv2.inRange(imagecolor_bilateralFilter, green_lower, green_upper)
    yellow_mask = cv2.inRange(imagecolor_bilateralFilter, yellow_lower, yellow_upper)
    red_mask_1 = cv2.inRange(imagecolor_bilateralFilter, red_lower_1, red_upper_1)
    red_mask_2 = cv2.inRange(imagecolor_bilateralFilter, red_lower_2, red_upper_2)
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    blue_mask = cv2.inRange(imagecolor_bilateralFilter, blue_lower, blue_upper)
    white_mask = cv2.inRange(imagecolor_bilateralFilter, white_lower, white_upper)

    #green_image = cv2.bitwise_and(imagecolor, imagecolor, mask=green_mask)
    #yellow_image = cv2.bitwise_and(imagecolor, imagecolor, mask=yellow_mask)
    red_image = cv2.bitwise_and(imagecolor, imagecolor, mask=red_mask)

    green_closing =     cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel) #dilate and erode
    yellow_closing =    cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel) #dilate and erode
    red_closing  =      cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)  # dilate and erode
    blue_closing =      cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)  # dilate and erode
    white_closing =     cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)  # dilate and erode

    #yellow_closing = cv2.erode(yellow_mask, (1, 1), iterations = 1) #necesitaria usarlo pero borra demasiado

    _, green_caps, green_hierarchy =    cv2.findContours(green_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, yellow_caps, yellow_hierarchy =  cv2.findContours(yellow_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, red_caps, red_hierarchy =        cv2.findContours(red_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, blue_caps, blue_hierarchy =      cv2.findContours(blue_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, white_caps, white_hierarchy =    cv2.findContours(white_closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(green_caps) != 0:
        for i in range(len(green_caps)):
            if len(green_caps[i]) >= 5:
                area = cv2.contourArea(green_caps[i])
                if area < 300:
                    break
                green_ellipse = cv2.fitEllipse(green_caps[i])
                cv2.ellipse(imagecolor_orig, multiply_tuple(green_ellipse, 1/scaleFactor), (0, 0, 255), 2) #multiplico la tupla para mostrar los contornos en el video con tamaño original, pero habiendolos calculados en una imagen de tamaño reducido para procesar mas rapido
                #print(cv2.contourArea(green_caps[i]))
                M = cv2.moments(green_caps[i])
                if M['m00'] == 0:
                    break
                cx = int(M['m10'] / M['m00'])   #posicion X del centroide del contorno
                cy = int(M['m01'] / M['m00'])   #posicion Y del centroide del contorno
                #solo para marcar los centros:
                cv2.circle(imagecolor_orig, (np.uint32(cx/scaleFactor), np.uint32(cy/scaleFactor)), 3, (0, 0, 255), 2)
                cv2.putText(imagecolor_orig, '%s' % (area), (np.uint32(cx/scaleFactor), np.uint32(cy/scaleFactor)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)

    if len(yellow_caps) != 0:
        for i in range(len(yellow_caps)):
            if len(yellow_caps[i]) >= 5:
                area = cv2.contourArea(yellow_caps[i])
                if area < 300:
                    break
                yellow_ellipse = cv2.fitEllipse(yellow_caps[i])
                cv2.ellipse(imagecolor_orig, multiply_tuple(yellow_ellipse, 1/scaleFactor), (255, 0, 0), 2)
                M = cv2.moments(yellow_caps[i])
                if M['m00'] == 0:
                    break
                cx = int(M['m10'] / M['m00'])  # posicion X del centroide del contorno
                cy = int(M['m01'] / M['m00'])  # posicion Y del centroide del contorno
                # solo para marcar los centros:
                cv2.circle(imagecolor_orig, (np.uint32(cx / scaleFactor), np.uint32(cy / scaleFactor)), 3, (255, 0, 0), 2)

    if len(red_caps) != 0:
        for i in range(len(red_caps)):
            if len(red_caps[i]) >= 5:
                area = cv2.contourArea(red_caps[i])
                if area < 300:
                    break
                red_ellipse = cv2.fitEllipse(red_caps[i])
                cv2.ellipse(imagecolor_orig, multiply_tuple(red_ellipse, 1/scaleFactor), (0, 255, 0), 2)
                M = cv2.moments(red_caps[i])
                if M['m00'] == 0:
                    break
                cx = int(M['m10'] / M['m00'])  # posicion X del centroide del contorno
                cy = int(M['m01'] / M['m00'])  # posicion Y del centroide del contorno
                # solo para marcar los centros:
                cv2.circle(imagecolor_orig, (np.uint32(cx / scaleFactor), np.uint32(cy / scaleFactor)), 3, (0, 255, 0), 2)


    if len(blue_caps) != 0:
        for i in range(len(blue_caps)):
            if len(blue_caps[i]) >= 5:
                area = cv2.contourArea(blue_caps[i])
                if area < 300:
                    break
                blue_ellipse = cv2.fitEllipse(blue_caps[i])
                cv2.ellipse(imagecolor_orig, multiply_tuple(blue_ellipse, 1/scaleFactor), (255, 255, 0), 2)
                M = cv2.moments(blue_caps[i])
                if M['m00'] == 0:
                    break
                cx = int(M['m10'] / M['m00'])  # posicion X del centroide del contorno
                cy = int(M['m01'] / M['m00'])  # posicion Y del centroide del contorno
                # solo para marcar los centros:
                cv2.circle(imagecolor_orig, (np.uint32(cx / scaleFactor), np.uint32(cy / scaleFactor)), 3, (255, 255, 0), 2)

    if len(white_caps) != 0:
        for i in range(len(white_caps)):
            if len(white_caps[i]) >= 5:
                area = cv2.contourArea(white_caps[i])
                if area < 300:
                    break
                white_ellipse = cv2.fitEllipse(white_caps[i])
                cv2.ellipse(imagecolor_orig, multiply_tuple(white_ellipse, 1/scaleFactor), (120, 120, 120), 2) #multiplico la tupla para mostrar los contornos en el video con tamaño original
                #print(cv2.contourArea(green_caps[i]))
                M = cv2.moments(white_caps[i])
                if M['m00'] == 0:
                    break
                cx = int(M['m10'] / M['m00'])   #posicion X del centroide del contorno
                cy = int(M['m01'] / M['m00'])   #posicion Y del centroide del contorno
                #solo para marcar los centros:
                cv2.circle(imagecolor_orig, (np.uint32(cx/scaleFactor), np.uint32(cy/scaleFactor)), 3, (200, 200, 255), 2)
    #
    # #cv2.circle(imagecolor, (np.uint32(green_ellipse[0])[0], np.uint32(green_ellipse[0])[1]), (10), (255, 255, 0))
    #
    # #cv2.ellipse(imagecolor, yellow_ellipse, (0, 255, 0), 2)
    # #cv2.circle(imagecolor, (275,649), (10), (255, 0, 0))
    # #print(cv2.selectROI(imagecolor))
    #
    # green_and_yellow_mask = cv2.bitwise_or(green_mask, yellow_mask)
    # green_and_yellow_mask = cv2.resize(green_and_yellow_mask, (imagecolor_orig.shape[1], imagecolor_orig.shape[0]), interpolation=cv2.INTER_LINEAR)
    # imagecolor_orig = cv2.bitwise_and(imagecolor_orig, imagecolor_orig, dst=None, mask=green_and_yellow_mask)
    # imagecolor_orig = cv2.resize(imagecolor_orig, None, fx = 0.8, fy = 0.8, interpolation=cv2.INTER_LINEAR)
    #out.write(imagecolor_orig)
    cv2.imshow("Original", imagecolor_orig)
    #cv2.imshow("Bilateral", cv2.cvtColor(imagecolor_bilateralFilter, cv2.COLOR_HSV2BGR))
    #cv2.imshow("Green image", green_image)
    #cv2.imshow("Yellow image", yellow_image)
    # cv2.imshow("Red image", red_image)
    cv2.imshow("Green closing", green_closing)
    #cv2.imshow("Yellow closing", yellow_closing)
    # cv2.imshow("Red closing", red_closing)
    # cv2.imshow("White closing", white_closing)
    #cv2.imshow("Contours", image_contours)
    #cv2.imshow("ellipse", ellipse)
    #cv2.imshow("Green mask", green_mask)
    #cv2.imshow("Yellow mask", yellow_mask)
    #cv2.imshow("Red mask", red_mask)
    #cv2.imshow("Blue mask", blue_mask)

    imagecolor_orig_hsv = cv2.cvtColor(imagecolor_orig, cv2.COLOR_BGR2HSV)
    cv2.imshow("H", imagecolor_orig_hsv[:,:,0])
    cv2.imshow("S", imagecolor_orig_hsv[:,:,1])
    cv2.imshow("V", imagecolor_orig_hsv[:,:,2])

    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):
        break
    if k == ord('p'):
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == ord('p'):
                break
video.release()
#out.release()
print("Ready")
cv2.destroyAllWindows()



