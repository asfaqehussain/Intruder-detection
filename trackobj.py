from __future__ import print_function 
import cv2 
import numpy as np 
 
 
def main():
    # starting main 
 
    
    cap = cv2.VideoCapture(0)
 
    # background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2(history=700, 
        varThreshold=25, detectShadows=True)
 
    kernel = np.ones((20,20),np.uint8)
 
    while(True):
 
        ret, frame = cap.read()
 
        fg_mask = back_sub.apply(frame)
 
        # dark gaps in foreground 
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
 
        # median filter
        fg_mask = cv2.medianBlur(fg_mask, 5) 
         
        # B&W
        _, fg_mask = cv2.threshold(fg_mask,127,255,cv2.THRESH_BINARY)
 
        
        fg_mask_bb = fg_mask
        contours, hierarchy = cv2.findContours(fg_mask_bb,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        areas = [cv2.contourArea(c) for c in contours]
 
        if len(areas) < 1:
 
            cv2.imshow('frame',frame)
 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
            continue
 
        else:
            max_index = np.argmax(areas)
 
        # Draw box
        cnt = contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
 
        # Draw circle 
        x2 = x + int(w/2)
        y2 = y + int(h/2)
        cv2.circle(frame,(x2,y2),4,(0,255,0),-1)
 
        # Print the centroid coordinates
        text = "x: " + str(x2) + ", y: " + str(y2)
        cv2.putText(frame, text, (x2 - 10, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
         
        # Display the resulting frame
        cv2.imshow('frame',frame)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
 
    # Close
    cap.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    print(__doc__)
    main()
