#!/opt/anaconda3/envs/ECE1390/bin/python3

# Note: This code uses open3d (needs to be pip installed).  This code lets you
# read in any STL 3D mesh 
# This allows me to use any 3D objects I create in Blender or I downloaded a file 
# from SketchFab.com (tons of free user-contributed objects)
# I downloaded this one (https://sketchfab.com/3d-models/skull-planes-2363b34481234eab996d6a5c3f7ee3fa)
# and then downsampled and exported as an STL file in Blender

import numpy as np
import cv2
import mediapipe as mp
import open3d as o3d
import os
from pathlib import Path


# MediaPipe has some preset codes for drawing (used when overlaying the hands on the image)
# import them here, so we can use them later
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# This allows me to find the folder containing this file and take the stl file from the 
# same location
script_path = Path(__file__).absolute()
stlfile=str(script_path.parent)+'/skull.stl'
#stlfile=str(script_path.parent)+'/Pumpkin.OBJ'


# For now (I might change this later), let's use a 7-degree of freedom rotation.
# This solves for the 3 translation, 3 euler angles, and a single scaling term.  
# The standard 9-dof affine includes two stretch/skew terms that I don't want here because 
# I want the object to be scaled to the hand orientation, but still have the same relative dimensions

# I got this code from Google-AI and modified it for here
def find_helmert_transform(src_points, dst_points):
    """
    Finds the 7-parameter Helmert (similarity) transformation between two
    sets of corresponding 3D points using a least-squares approach.
    """
    
    # Translate both sets of points to their centroids
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)
    
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid
    
    # Compute the 3x3 covariance matrix H
    H = src_centered.T @ dst_centered
    
    # Compute the singular value decomposition (SVD) of H
    U, S, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix R
    R_matrix = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R_matrix) < 0:
        Vt[2, :] *= -1
        R_matrix = Vt.T @ U.T
    
  
    # Compute the scale factor (s)
    # The scale is the ratio of the norms of the centered point sets
    src_norm_sq = np.sum(src_centered**2)
    dst_norm_sq = np.sum(dst_centered**2)
    scale = np.sqrt(dst_norm_sq / src_norm_sq)
    R_matrix=R_matrix*scale
    
    # Compute the translation vector (T)
    # T = dst_centroid - s * R * src_centroid
    translation = dst_centroid - (R_matrix@src_centroid)
    translation=np.expand_dims(translation,axis=1)
   
    return R_matrix, translation



# This is the main class (video window) that will collect and process the images

class App(object):
    def __init__(self, video_src=0):
        # Standard CV2 webcam access.  
        self.cam = cv2.VideoCapture(video_src)

        # These flags will be used to control the visibility of the hands, cube, and object
        self.display_hand_landmarks=True
        self.display_cube=True
        self.display_obj=True

        # Create mediapipe hands detector
        self.mp_hands = mp_hands.Hands(static_image_mode=False,
                                        max_num_hands=2,
                                        min_detection_confidence=0.15)
        
        # Read in the STL file as a mesh
        o3dmesh = o3d.io.read_triangle_mesh(stlfile)
        # The mesh consists of triangles and vertices
        self.obj_tris=np.asarray(o3dmesh.triangles)

        # When I made the STL file, the X-Y axis are swapped and Z is inverted.
        # I could apply a rotation or go back and resave the STL file, but it 
        # is easy enough to just fix the corrdinate system here.  If you use a different
        # STL file, this part may need to be adjusted 
        verts2=np.asarray(o3dmesh.vertices)
        verts=verts2.copy()
        verts[:,0]= -verts2[:,1]
        verts[:,1]= verts2[:,0]
        verts[:,2]= -verts2[:,2]


        # make sure the object fits inside our 1x1x1 box
        verts[:,0]=verts[:,0]-np.mean(verts,axis=0)[0]
        verts[:,1]=verts[:,1]-np.mean(verts,axis=0)[1]
        verts[:,2]=verts[:,2]-np.mean(verts,axis=0)[2]

        verts=verts/np.max(np.abs(verts))
        verts[:,0]=verts[:,0]+.5 # move to the center of the box
        verts[:,1]=verts[:,1]+.5 
        verts[:,2]=verts[:,2]+.5 
        self.obj_vertices=verts


        # For now, just a cube
        self.cube_vertices = np.array([
            [0,0,1], [1,0,1], [1,1,1], [0,1,1],  # Front face
            [0,0,0], [1,0,0], [1,1,0], [0,1,0]   # Back face
            ], dtype=np.int32)
        self.cube_reg_idx=np.array([0,1,2,3,5,6,7])

        # Define edges to draw (connecting the projected vertices)
        self.cube_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Front face edges
            (4, 5), (5, 6), (6, 7), (7, 4),  # Back face edges
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
            ]

        # Launch a window to draw into and keep it open
        cv2.namedWindow('AR Demo')
    

    def calc_rotation(self,frame,hand_points):

        obj_points=self.cube_vertices[self.cube_reg_idx]


        # Compute the rotation to put the cube in the hand 
        rotation, translation = find_helmert_transform(obj_points,hand_points)
        
        # Then apply the rotation to the cube and the STL object
        cube_rot_points=(rotation@self.cube_vertices.T + translation@np.ones((1,self.cube_vertices.shape[0]))).T
        obj_rot_points= (rotation@self.obj_vertices.T  + translation@np.ones((1,self.obj_vertices.shape[0]))).T
     
     

        h=frame.shape[0]
        w=frame.shape[1]

        # Draw the cube in the hand
        if(self.display_cube):
            for edge in self.cube_edges:
                pt1 = tuple((cube_rot_points[edge[0]][0]*w,cube_rot_points[edge[0]][1]*h))
                pt2 = tuple((cube_rot_points[edge[1]][0]*w,cube_rot_points[edge[1]][1]*h))
                cv2.line(frame, np.uint64(pt1), np.uint64(pt2), (0, 255, 0), 2) # Green lines, thickness 2

        # Draw the STL object inside the cube
        if(self.display_obj):
            # This sorts the triangles from furthest to closest to the camera (z distance)
            idx=np.argsort(np.mean(obj_rot_points[self.obj_tris,2],axis=1))

            # Then when I draw, I draw furthest to closest so the closer are visible on top
            # I could also compute if a face was ocluded and not draw it, which would save graphics
            # but be more computational compared to my simple sort command
            for idx,tri in enumerate(self.obj_tris[idx,:]):
                pts=np.zeros((3,2),dtype=np.int32)
                pts[:,0]=obj_rot_points[tri,0]*w
                pts[:,1]=obj_rot_points[tri,1]*h

                # Draw a white triangle for the mesh faces
                cv2.fillPoly(frame, [pts], (255, 255, 255))
                
                # Outline the triangles with black lines
                color=(0.1, 0.1, 0.1)
                cv2.line(frame, np.uint64(pts[0,:]), np.uint64(pts[1,:]), color, 1) # color lines, thickness 1
                cv2.line(frame, np.uint64(pts[1,:]), np.uint64(pts[2,:]), color, 1) 
                cv2.line(frame, np.uint64(pts[2,:]), np.uint64(pts[0,:]), color, 1) 


    def run(self):
        while True:
            _, frame = self.cam.read()
            frame = cv2.flip(frame, 1)
            results = self.mp_hands.process(frame)
            
            #lighting = cv2.cvtColorTwoPlane(frame,cv2.COLOR_BGR2HSV)
            #[h,s,v]=cv2.split(lighting)

            if results.multi_hand_landmarks:
                for idx,hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if(self.display_hand_landmarks):
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        
                    handedness = results.multi_handedness[idx]
                    
                    #print(f"Detected hand is: {handedness}")
                    wrist=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    pinky_mcp=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                    index_mcp=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    thumb_mcp=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                    pinky_tip=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    index_tip=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    

                    hand_points=np.array([
                        [wrist.x,wrist.y,wrist.z],
                        [pinky_mcp.x,pinky_mcp.y,pinky_mcp.z],
                        [index_mcp.x,index_mcp.y,index_mcp.z],
                        [thumb_mcp.x,thumb_mcp.y,thumb_mcp.z],
                        [pinky_tip.x,pinky_tip.y,pinky_tip.z],
                        [index_tip.x,index_tip.y,index_tip.z],
                        [thumb_tip.x,thumb_tip.y,thumb_tip.z]
                    ])
                    self.calc_rotation(frame,hand_points)

            cv2.imshow('AR Demo', frame)

            ch = cv2.waitKey(5)
            if ch == 27:  # escape key code
                break
            elif ch == ord('c'):
                self.display_cube = not self.display_cube
            elif ch == ord('h'):
                self.display_hand_landmarks = not self.display_hand_landmarks
            elif ch == ord('o'):
                self.display_obj = not self.display_obj
            
        cv2.destroyAllWindows()



if __name__ == '__main__':
    print(__doc__)
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
    App(video_src).run()