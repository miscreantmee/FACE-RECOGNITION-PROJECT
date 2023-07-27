import os 
import numpy as np
import cv2 
import time 
from retinaface import RetinaFace 
from common import retrieve_data_from_dict_retinaface, \
    retrieve_data_from_dict_retinaface_tracker, create_tracker_by_name, plot_one_box,\
          bb_intersection_over_union, save_the_dict, open_the_file_with_dict
from common import gather_embeddings_for_database, compare_results, merge_folders, merge_timelog, timelog_from_dict, send_to_database_with_tracker








def main(videoname:str, x_frame_to_check:int,
    database_folder = 'Database', tracker_type = 'CSRT', 
    path_to_the_video:str = 'Dataset',output_folder:str = 'Output', frame_of_early_stop:int = 1000, frame_of_start = 0) -> None:
    """
    Main function that forms pipeline of video processing.
    
    Arguments:
    ---------------
    videoname : str
        Name of the video that you want to process.
    x_frame_to_check : int
        Each x frame that will be processed and saved to database.
    database_folder : str
        Name of the database folder.
    tracker_type : str
        Name of the tracker that will be used in this module.Possible options: ['KCF','MEDIANFLOW','MOSSE','CSRT']
    path_to_the_video : str
        Name of the folder with video/videos.
    output_folder : str
        Name of the folder where processed video will be saved.
    frame_of_early_stop : int
        Frame number when you want to stop processing. None to process whole video.
      

    Returns:
    ---------------
    dict
        dict_with_objects_frames: dict with frames of each detected object, key = id of folder, value = list with frame numbers
    int
        fps : number of frames per second in current video
    """
    out  = None
    cap = None
    current_folder = os.getcwd()
    print(current_folder)
    path_to_database_folder =  os.path.join(current_folder, database_folder)
    print(os.path.join(current_folder,path_to_the_video,videoname))
    cap = cv2.VideoCapture(os.path.join(current_folder,path_to_the_video,videoname))
    shape = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))] 

    if frame_of_start!=0:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_of_start)


    detector = RetinaFace
    model_name_of_detection = 'retinaface'
    number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 
    # print(number_of_frames)

    if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_folder+f'{model_name_of_detection}_' +videoname.split(".")[0] + ".avi", fourcc,fps,(shape[1], shape[0]),True)
    
    counter_for_track_id = 0
    
    dict_with_objects_from_tracker = {}
    dict_with_embeddings = {}
    dict_with_objects_frames = {}
    dict_with_objects_from_tracker = {}
    counter_for_tracker_frames = 0
    for frame_id in range(frame_of_start, number_of_frames):

                        
        if frame_of_early_stop != None and frame_id == frame_of_early_stop:
            print(dict_with_objects_frames)
            break
        
        ret, frame = cap.read()
        if frame_id % x_frame_to_check == 0:
            
            # multi object tracker initialization
            multi_tracker = cv2.legacy.MultiTracker_create()
            dict_with_objects = {}
            print(f'Frame number {frame_id}/{number_of_frames}')
            #frame reading
            counter_for_tracker_frames = 0
        
            img_to_draw = frame.copy()
            
            
            cv2.putText(img_to_draw,f'{frame_id}', [100,100], cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0,255,0], thickness=1)
            #inference
            detection = detector.detect_faces(frame, threshold = 0.97)
            #convertion of output of the model
            converted_detections = retrieve_data_from_dict_retinaface(detection)
            converted_detections_for_tracker, conf, cls_ids = retrieve_data_from_dict_retinaface_tracker(detection)
            # initialization of trackers
            for bbox in converted_detections_for_tracker:
                multi_tracker.add(create_tracker_by_name(tracker_type), frame, bbox)
            
            #saving images into database    
            time_start = time.time()        
            for i, (*xyxy, conf) in enumerate(converted_detections):
                print(f'Confidence: {conf}')
                xyxy = list(map(int,xyxy))
                if xyxy[2] - xyxy[0] < 35 or xyxy[3] - xyxy[1]<35:
                    continue
                if conf < 0.98:
                    continue
                # print(f'COUNTER FOR TRACK ID : {counter_for_track_id}')
                counter_for_track_id +=1
            
                
                dict_with_objects[counter_for_track_id] = xyxy

            
            if dict_with_objects_from_tracker == {}:
                garbage, dict_with_embeddings = send_to_database_with_tracker(dict_with_detections= dict_with_objects,path_to_database=database_folder, video_name=video_name, frame=frame, frame_id=frame_id, fps=fps, dict_with_embeddings=dict_with_embeddings)
                # print("EMPTY OUTPUT FROM TRACKER")
            elif dict_with_objects_from_tracker != {}:
                updated_dict = {}
                # print(f'DETECTED OBJECTS: {dict_with_objects}')
                for id_of_obj, bbox_detect in dict_with_objects.items():
                    dict_with_iou = {}
                    for id_of_obj_from_tracker, bbox_from_track  in dict_with_objects_from_tracker.items():
                        # print(f'RESULT_OF_IOU FOR OBJECT WITH ID : {id_of_obj} {id_of_obj_from_tracker} - {bb_intersection_over_union(bbox_detect, bbox_from_track)}')
                        if bb_intersection_over_union(bbox_detect, bbox_from_track) >= 0.3:
                            # updated_dict[id_of_obj_from_tracker] = bbox_detect
                            dict_with_iou[(bb_intersection_over_union(bbox_detect, bbox_from_track))] = [id_of_obj_from_tracker,bbox_detect ]
                            # counter_for_track_id -=1
                            # print(f'COUNTER FOR TRACK ID : {counter_for_track_id}')
                    if len(list(dict_with_iou.keys())) == 0:
                        # print("NEW ENTITY CREATED")
                        updated_dict[id_of_obj] = bbox_detect
                    elif len(list(dict_with_iou.keys())) > 0:
                        max_item = max(list(dict_with_iou.keys()))
                        dict_item_of_max_item = dict_with_iou[max_item]
                        updated_dict[dict_item_of_max_item[0]] = dict_item_of_max_item[1]
                dict_with_objects = updated_dict
                dict_with_objects_from_tracker = {}
                # print(f'DETECTED OBJECTS: {dict_with_objects}')
                dict_with_objects, dict_with_embeddings = send_to_database_with_tracker(dict_with_detections= dict_with_objects,path_to_database=database_folder, video_name=video_name, frame=frame, frame_id=frame_id, fps=fps, dict_with_embeddings=dict_with_embeddings)
                for key in list(dict_with_objects.keys()):
                    if key not in dict_with_objects_frames.keys():
                        dict_with_objects_frames[key] = [frame_id]
                    elif key in dict_with_objects_frames.keys():
                        dict_with_objects_frames[key].append(frame_id)
            # print(f'DETECTED OBJECTS: {dict_with_objects}')        
            for i, (*xyxy, conf) in enumerate(converted_detections):
                xyxy = list(map(int, xyxy))
                if xyxy[2] - xyxy[0] < 35 or xyxy[3] - xyxy[1]<35:
                    continue
                conf = float(conf)
                plot_one_box(xyxy, img=img_to_draw, color=[0,255,0], label=str(conf)[:4], line_thickness=1)

                
                
            
        else:
            counter_for_tracker_frames+=1
            success, boxes = multi_tracker.update(frame)
            img_to_draw = frame.copy()
            
            if dict_with_objects_from_tracker == {}:
                dict_with_objects_from_tracker = dict_with_objects
            
            updated_dict_t = {}

            for id_of_obj, bbox_detect in dict_with_objects_from_tracker.items():
                for  bbox_from_track  in boxes:
                    bbox_from_track = [bbox_from_track[0], bbox_from_track[1], bbox_from_track[0]+bbox_from_track[2], bbox_from_track[1]+bbox_from_track[3]]
                    # print(f'RESULT_OF_IOU FOR OBJECT WITH ID : {id_of_obj} - {bb_intersection_over_union(bbox_detect, bbox_from_track)}')

                    if bb_intersection_over_union(bbox_detect, bbox_from_track) >= 0.3:
                        updated_dict_t[id_of_obj] = bbox_from_track
                        break
                else:
                    # print('NEW ENTITY CREATED')
                    updated_dict_t[max(dict_with_objects_from_tracker.keys())+1] = bbox_from_track
            dict_with_objects_from_tracker = updated_dict_t
            for key in list(dict_with_objects_from_tracker.keys()):
                if key not in dict_with_objects_frames.keys():
                    dict_with_objects_frames[key] = [frame_id]
                elif key in dict_with_objects_frames.keys():
                    dict_with_objects_frames[key].append(frame_id)
            
        
                
            for key, value in dict_with_objects_from_tracker.items():
                xyxy = list(map(int,value))
                plot_one_box(xyxy, img_to_draw, color = [255,0,0], label = f'track_id: {key}', line_thickness=2)


        out.write(img_to_draw)
          
 
    print(dict_with_objects_frames)
    cap.release()
    return dict_with_objects_frames, fps

if __name__ == '__main__':
    
    video_name = 'Toronto.mp4'
    frame_to_check = 10
    database_folder = 'Database'
    tracker_type = 'CSRT'
    # if you run it from сmd please write full absolute path to folders
    path_to_the_video = 'Dataset'
    output_folder = '/Output/'
    stop_frame = 2000
    frame_of_start = 0
   
    
    dict_with_frames, fps = main(videoname=video_name, x_frame_to_check=frame_to_check, database_folder=database_folder, tracker_type=tracker_type,path_to_the_video=path_to_the_video, output_folder=output_folder, frame_of_early_stop=stop_frame, frame_of_start=frame_of_start )
    
    save_the_dict(dict_with_frames, output_path='./Database/', output_file_name='dict_with_frames.txt')
    
    dict_with_embeddings = gather_embeddings_for_database(path_to_database='./Database/', filename=f'{video_name}')

    dict_with_results = compare_results(dict_with_embeddings)
    print('MERGING IN PROCESS')
    dict_with_merged_entites = merge_folders(path_to_database='./Database', file_name=f'{video_name}', dict_with_results=dict_with_results)
    result_dict = merge_timelog(dict_with_frames, dict_with_merged_entites)
    timelog_from_dict(result_dict, filename='MergedFaces_DB' + video_name, garbage_folder_id='', fps=fps)